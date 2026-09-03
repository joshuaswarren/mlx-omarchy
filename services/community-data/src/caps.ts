// Wire caps. These mirror the documented contract; the server rejects
// anything larger BY NAME (the error code names the cap).
export const MAX_PAYLOAD_BYTES = 256 * 1024;
export const MAX_ARCHIVE_BYTES = 8 * 1024 * 1024;
export const MAX_CHUNK_BYTES = 768 * 1024;
export const MAX_CHUNKS = 11;

// Proof of work: minimum leading zero bits of sha256("<sha>:<nonce>").
export const MIN_POW_BITS = 18;

// Incomplete submissions older than this are deleted by the cron handler.
export const GC_AFTER_SECONDS = 7 * 24 * 3600;

// Payload JSON Schema version this deployment accepts.
export const SCHEMA_VERSION = 1;

// Cache parts are split so no D1 row approaches the 2 MB row limit.
export const CACHE_PART_CHARS = 500_000;

export function expectedChunkCount(totalBytes: number, chunkBytes: number): number {
  return Math.ceil(totalBytes / chunkBytes);
}

export function expectedChunkLength(
  totalBytes: number,
  chunkBytes: number,
  idx: number,
): number {
  const remaining = totalBytes - idx * chunkBytes;
  return Math.max(0, Math.min(chunkBytes, remaining));
}
