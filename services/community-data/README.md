# mlx-omarchy community data

A self-contained Cloudflare Worker that accepts contributor hardware and
benchmark submissions from the `scripts/collect_*` collectors and serves
them as a public, agent-readable dataset. Zero dollars: Workers Free and
D1 Free only. No R2 (not enabled on the account), no KV, no paid
add-ons. Large archives are stored as chunks across multiple D1 rows.

## Layout

```
migrations/0001_init.sql   versioned D1 schema
schema/payload-v1.schema.json  canonical strict JSON Schema for summaries
src/                       worker (no framework: fetch + scheduled)
public/index.html          static human-readable index (no build step)
test/unit/                 pure unit tests (bun test)
test/smoke.ts              local end-to-end smoke over wrangler dev
```

## Wire contract

All requests and responses are JSON unless stated otherwise. Errors are
always `{"error": <machine code>, "detail": ...}`.

| Route | Behavior |
|---|---|
| `POST /v1/submit` | Initiate. Body: `{schema_version, kind, content_sha256, payload, archive, pow}`. Verifies proof of work, validates the summary against the pinned schema, scans it for PII (refuses on any hit), enforces caps. Returns `{status: "awaiting_chunks"\|"stored"\|"duplicate", content_sha256, missing_chunks, receipt_url}`. |
| `POST /v1/submit/<sha>/chunk/<idx>` | One raw octet-stream chunk. Idempotent; verifies the received bytes against the hash declared at initiate. Returns `{status, idx, missing_chunks}`. |
| `POST /v1/submit/<sha>/complete` | Publishes when every chunk is present and every stored hash matches the declared hash. `409 {error:"incomplete", missing_chunks}` otherwise. |
| `GET /v1/submit/<sha>` | Dedup probe: `200 {status:"duplicate", receipt_url}` or `404`. |
| `GET /v1/results` | Cached index of published summaries. Each entry carries an added `content_sha256` field so consumers can address records. |
| `GET /v1/results/<sha>` | One full published record (live, not cached). |
| `GET /v1/results/<sha>/archive` | The reassembled archive bytes, streamed in chunk order. Immutable, content-addressed. |
| `GET /v1/dataset/latest.jsonl` | One JSON object per line, for bulk agent consumption. Cached. |

Unpublished (incomplete) submissions never appear on any read route.

## Caps and limits

| Cap | Value | Error code |
|---|---|---|
| Archive total | 8 MB (11 chunks max) | `archive_too_large` / `too_many_chunks` |
| Single chunk | 768 KB | `chunk_too_large` |
| JSON summary payload | 256 KB | `payload_too_large` |
| Proof of work | >= 18 leading zero bits of sha256(`<sha>:<nonce>`) | `pow_missing` / `pow_invalid` |

The server never decompresses, parses, or executes uploaded archive
bytes; chunks are hashed and stored opaquely, which keeps every request
well inside the 10 ms CPU budget of the free plan. The PII scan runs on
the JSON summary only: MAC addresses, IPv4/IPv6, UUIDs, serial numbers,
home paths, credential shapes, and mDNS-style hostnames are refused
(`422 pii_detected`), never stored. Archive defense rests on the
collector's local redaction (see `scripts/collect_common.py`).

Incomplete submissions are garbage-collected by the hourly cron
(`17 * * * *`) after **7 days** without a completed upload; the same
cron rebuilds the cached index/dataset responses. Bulk routes self-heal
by building the cache on first request if the cron has not run yet.

## One-time deploy

The account is shared with unrelated projects, so every name is
namespaced: worker `mlx-omarchy-community-data`, D1 database
`mlx-omarchy-community`. `wrangler.jsonc` contains no `account_id`
(wrangler reads `CLOUDFLARE_ACCOUNT_ID` from the environment) and no
secrets. The ONE value to fill in by hand is `database_id`, printed by
`wrangler d1 create` and pasted into `wrangler.jsonc`; it is not a
secret.

```sh
cd services/community-data
npm install

# Load the canonical account token (never commit or print it).
set -a; . ~/.config/cloudflare/thewarrens-co.env; set +a
export CLOUDFLARE_API_TOKEN=$CF_API_TOKEN CLOUDFLARE_ACCOUNT_ID=$CF_ACCOUNT_ID

# 1. Create the database, then paste the printed database_id into
#    wrangler.jsonc (replacing the OWNER_FILL_ME placeholder).
npx wrangler d1 create mlx-omarchy-community

# 2. Apply migrations to the remote database.
npx wrangler d1 migrations apply mlx-omarchy-community --remote

# 3. Deploy the worker.
npx wrangler deploy
```

workers.dev fronting 403s default non-browser user agents, which is why
the collector sends `User-Agent: mlx-omarchy-collector/1`.

## Local development and tests (no credentials needed)

```sh
npm install
npm test          # pure unit tests: validation, PII, PoW, hashing, caps
npm run smoke     # end-to-end over wrangler dev with local D1
```

The smoke run covers: full multi-chunk submission, resumed upload that
sends only the missing chunks, replay dedupe, chunk hash mismatch,
oversize archive, oversize chunk, weak and absent proof of work, PII
refusal, completing with a missing chunk, byte-identical archive
round-trip, cron cache rebuild, and GC of stale incomplete submissions.

## Repository mirror

The GitHub side consumes ONLY the public read routes above
(`/v1/results`, `/v1/results/<sha>`, `/v1/results/<sha>/archive`,
`/v1/dataset/latest.jsonl`). Submitters never open issues, PRs, or
discussions; their data reaches the repo through the mirror.
