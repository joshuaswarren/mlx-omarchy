import payloadSchemaJson from "../schema/payload-v1.schema.json";
import {
  MAX_CHUNK_BYTES,
  MAX_PAYLOAD_BYTES,
  MIN_POW_BITS,
  expectedChunkLength,
} from "./caps";
import { sha256Hex } from "./hash";
import { scanPii } from "./pii";
import { verifyPow } from "./pow";
import { SchemaNode, validateSchemaRoot } from "./schema";
import * as store from "./store";
import { checkInitiate, isSha256 } from "./validate";

const payloadSchema = payloadSchemaJson as SchemaNode;

const CACHEABLE = "public, max-age=60";
const IMMUTABLE = "public, max-age=31536000, immutable";

function jsonResponse(status: number, data: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      ...headers,
    },
  });
}

function errorResponse(status: number, code: string, detail?: unknown): Response {
  return jsonResponse(status, { error: code, detail: detail ?? null });
}

function notAllowed(): Response {
  return errorResponse(405, "method_not_allowed");
}

function receiptUrl(origin: string, sha: string): string {
  return `${origin}/v1/results/${sha}`;
}

type PayloadFields = {
  arch?: string | null;
  model?: string | null;
  chip?: string | null;
  kernel?: string | null;
  mesa_driver?: string | null;
  mesa_device?: string | null;
  mlx_version?: string | null;
  mlx_device?: string | null;
};

function textColumn(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

async function handleInitiate(request: Request, env: Env): Promise<Response> {
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return errorResponse(415, "unsupported_media_type");
  }
  let body: unknown;
  try {
    body = JSON.parse(await request.text());
  } catch {
    return errorResponse(400, "bad_json");
  }
  const checked = checkInitiate(body);
  if (!checked.ok) {
    // Cap violations are 413 by name; every other shape problem is 400.
    const cap = checked.code === "archive_too_large" ||
      checked.code === "chunk_too_large" || checked.code === "too_many_chunks";
    return errorResponse(cap ? 413 : 400, checked.code);
  }
  const init = checked.value;

  const summaryText = JSON.stringify(init.payload);
  if (summaryText.length > MAX_PAYLOAD_BYTES) {
    return errorResponse(413, "payload_too_large", { limit: MAX_PAYLOAD_BYTES });
  }
  const schemaErrors = validateSchemaRoot(init.payload, payloadSchema);
  if (schemaErrors.length > 0) {
    return errorResponse(422, "schema_invalid", { errors: schemaErrors.slice(0, 20) });
  }

  const pow = await verifyPow(init.content_sha256, (body as Record<string, unknown>).pow, MIN_POW_BITS);
  if (!pow.ok) {
    const status = pow.code === "pow_missing" ? 400 : 403;
    return errorResponse(status, pow.code, { min_difficulty: MIN_POW_BITS });
  }

  const pii = scanPii(summaryText);
  if (pii !== null) {
    return errorResponse(422, "pii_detected", { kinds: pii });
  }

  const now = Math.floor(Date.now() / 1000);
  const fields: PayloadFields = init.payload as PayloadFields;
  const existing = await store.getSubmission(env.DB, init.content_sha256);

  if (existing !== null && existing.published === 1) {
    return jsonResponse(200, {
      status: "duplicate",
      content_sha256: init.content_sha256,
      missing_chunks: [],
      receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
    });
  }

  if (existing !== null) {
    const storedShape = {
      total_bytes: existing.archive_total_bytes,
      chunk_bytes: existing.archive_chunk_bytes,
      chunk_count: existing.archive_chunk_count,
    };
    const newShape = init.archive
      ? {
          total_bytes: init.archive.total_bytes,
          chunk_bytes: init.archive.chunk_bytes,
          chunk_count: init.archive.chunk_count,
        }
      : { total_bytes: null, chunk_bytes: null, chunk_count: null };
    if (JSON.stringify(storedShape) !== JSON.stringify(newShape)) {
      return errorResponse(409, "archive_conflict");
    }
    await store.touchIncomplete(env.DB, init.content_sha256, now);
    const missing = await store.missingChunks(
      env.DB,
      init.content_sha256,
      init.archive?.chunk_count ?? 0,
    );
    return jsonResponse(200, {
      status: "awaiting_chunks",
      content_sha256: init.content_sha256,
      missing_chunks: missing,
      receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
    });
  }

  const row = await store.initiateSubmission(env.DB, {
    contentSha: init.content_sha256,
    now,
    kind: init.kind,
    arch: textColumn(fields.arch),
    model: textColumn(fields.model),
    chip: textColumn(fields.chip),
    kernel: textColumn(fields.kernel),
    mesaDriver: textColumn(fields.mesa_driver),
    mesaDevice: textColumn(fields.mesa_device),
    mlxVersion: textColumn(fields.mlx_version),
    mlxDevice: textColumn(fields.mlx_device),
    summary: summaryText,
    archive: init.archive,
    powDifficulty: pow.difficulty,
  });
  const missing = row.published === 1
    ? []
    : await store.missingChunks(env.DB, init.content_sha256, init.archive?.chunk_count ?? 0);
  return jsonResponse(200, {
    status: row.published === 1 ? "stored" : "awaiting_chunks",
    content_sha256: init.content_sha256,
    missing_chunks: missing,
    receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
  });
}

async function handleChunk(
  sha: string,
  idxText: string,
  request: Request,
  env: Env,
): Promise<Response> {
  const idx = Number(idxText);
  if (!Number.isInteger(idx) || idx < 0) {
    return errorResponse(400, "chunk_index_invalid");
  }
  const row = await store.getSubmission(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  if (row.published === 1) return errorResponse(409, "already_published");
  if (row.archive_chunk_count === null || row.archive_chunk_bytes === null) {
    return errorResponse(409, "no_archive");
  }
  if (idx >= row.archive_chunk_count) {
    return errorResponse(400, "chunk_index_invalid");
  }
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.length > MAX_CHUNK_BYTES) {
    return errorResponse(413, "chunk_too_large", { limit: MAX_CHUNK_BYTES });
  }
  const expected = expectedChunkLength(
    row.archive_total_bytes ?? 0,
    row.archive_chunk_bytes,
    idx,
  );
  if (bytes.length !== expected) {
    return errorResponse(400, "chunk_size_invalid", { expected, got: bytes.length });
  }
  const chunkHash = await sha256Hex(bytes);
  const declared: string[] = JSON.parse(row.archive_chunk_sha256 ?? "[]");
  if (declared[idx] !== undefined && declared[idx] !== chunkHash) {
    return errorResponse(422, "chunk_hash_mismatch", { idx });
  }
  const outcome = await store.putChunk(env.DB, sha, idx, chunkHash, bytes);
  if (!outcome.ok) {
    return errorResponse(409, "chunk_hash_conflict", { idx });
  }
  const missing = await store.missingChunks(env.DB, sha, row.archive_chunk_count);
  return jsonResponse(200, { status: outcome.status, idx, missing_chunks: missing });
}

async function handleComplete(sha: string, request: Request, env: Env): Promise<Response> {
  const outcome = await store.completeSubmission(
    env.DB,
    sha,
    Math.floor(Date.now() / 1000),
  );
  if (!outcome.ok) {
    if (outcome.code === "not_found") return errorResponse(404, "not_found");
    if (outcome.code === "incomplete") {
      return errorResponse(409, "incomplete", { missing_chunks: outcome.missing });
    }
    return errorResponse(422, "chunk_hash_mismatch", { idx: outcome.idx });
  }
  return jsonResponse(200, {
    status: outcome.status,
    content_sha256: sha,
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
  });
}

async function handleProbe(sha: string, request: Request, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  return jsonResponse(200, {
    status: "duplicate",
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
  });
}

async function handleResultOne(sha: string, request: Request, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  return jsonResponse(200, {
    content_sha256: row.content_sha256,
    kind: row.kind,
    schema_version: row.schema_version,
    arch: row.arch,
    model: row.model,
    chip: row.chip,
    kernel: row.kernel,
    mesa_driver: row.mesa_driver,
    mesa_device: row.mesa_device,
    mlx_version: row.mlx_version,
    mlx_device: row.mlx_device,
    summary: JSON.parse(row.summary),
    archive:
      row.archive_chunk_count === null
        ? null
        : {
            total_bytes: row.archive_total_bytes,
            chunk_bytes: row.archive_chunk_bytes,
            chunk_count: row.archive_chunk_count,
          },
    received_at: new Date(row.received_at * 1000).toISOString(),
    published_at:
      row.published_at === null
        ? null
        : new Date(row.published_at * 1000).toISOString(),
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
  });
}

async function handleArchive(sha: string, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null || row.archive_chunk_count === null) {
    return errorResponse(404, "not_found");
  }
  const chunks = await store.archiveChunks(env.DB, sha);
  if (chunks.length !== row.archive_chunk_count) {
    return errorResponse(500, "archive_unavailable");
  }
  const total = row.archive_total_bytes ?? 0;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(new Uint8Array(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    headers: {
      "content-type": "application/octet-stream",
      "content-length": String(total),
      "cache-control": IMMUTABLE,
      "access-control-allow-origin": "*",
    },
  });
}

async function serveCache(env: Env, key: string, contentType: string): Promise<Response> {
  let cache = await store.getCache(env.DB, key);
  if (cache === null) {
    // Self-heal before the first cron fires: build once on demand.
    await store.rebuildCaches(env.DB, Math.floor(Date.now() / 1000));
    cache = await store.getCache(env.DB, key);
  }
  if (cache === null) return errorResponse(500, "cache_unavailable");
  return new Response(cache.text, {
    headers: {
      "content-type": contentType,
      "cache-control": CACHEABLE,
      "access-control-allow-origin": "*",
      "x-generated-at": cache.generated_at,
    },
  });
}

export async function handleFetch(request: Request, env: Env): Promise<Response> {
  const { pathname } = new URL(request.url);

  if (pathname === "/v1/submit") {
    if (request.method !== "POST") return notAllowed();
    return handleInitiate(request, env);
  }

  let match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})\/chunk\/(\d+)$/);
  if (match) {
    if (request.method !== "POST") return notAllowed();
    return handleChunk(match[1], match[2], request, env);
  }

  match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})\/complete$/);
  if (match) {
    if (request.method !== "POST") return notAllowed();
    return handleComplete(match[1], request, env);
  }

  match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleProbe(match[1], request, env);
  }

  match = pathname.match(/^\/v1\/results\/([0-9a-f]{64})\/archive$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleArchive(match[1], env);
  }

  match = pathname.match(/^\/v1\/results\/([0-9a-f]{64})$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleResultOne(match[1], request, env);
  }

  if (pathname === "/v1/results") {
    if (request.method !== "GET") return notAllowed();
    return serveCache(env, "results", "application/json");
  }

  if (pathname === "/v1/dataset/latest.jsonl") {
    if (request.method !== "GET") return notAllowed();
    return serveCache(env, "dataset", "application/x-ndjson");
  }

  return errorResponse(404, "not_found");
}

