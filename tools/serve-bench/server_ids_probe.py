#!/usr/bin/env python3
"""Bench-only instrumented launcher for mlx_lm.server.

Runs the REAL mlx_lm.server (same wheel, same venv) with runtime
monkeypatches that record, per request:
  - the exact prompt token ids and their sha16 (prompt-ID proof),
  - which cache-fetch branch executed (exact / longer+trim / shorter /
    miss) and the cache list composition (class names + trimmability),
  - the raw generated token ids per request and their sha16,
  - per-phase timings (prefill vs decode).
Nothing on disk is modified: the wheel and all pinned artifacts are
untouched; patches live only in this process. Output: one JSON line per
request on stderr prefixed PROBE: (parse with the harness).

Usage: python server_ids_probe.py <normal mlx_lm.server args...>
"""
import json
import sys
import time

import mlx.core as mx

probes = []


def sha16(ids):
    import hashlib
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16]


def main():
    import mlx_lm.server as srv

    # --- 1. cache fetch branch + composition probe -------------------
    cache_mod = __import__("mlx_lm.models.cache", fromlist=["PromptCache"])
    orig_fetch = cache_mod.LRUPromptCache.fetch_nearest_cache

    def fetch_probe(self, model, tokens):
        result = self._trie.search(model, tokens)
        branch = "miss"
        detail = {}
        if result.exact is not None:
            branch = "exact"
        elif result.longer is not None and result.common_prefix > (
            len(result.shorter) if result.shorter is not None else 0):
            entry = self._trie.get(model, result.longer)
            trimmable = [type(c).__name__ + (
                ":trimmable" if c.is_trimmable() else ":NOT-trimmable")
                for c in entry.prompt_cache]
            branch = "longer+trim" if all(c.is_trimmable() for c in entry.prompt_cache) \
                else "longer+NOT-trimmable"
            # mlx-lm 0.31.3 fetch arithmetic: prefix = min(len-1, common_prefix);
            # num_to_trim = len(longer) - prefix; kept = cache_offset - num_to_trim
            # where cache_offset == len(longer) - 1 (last key token's KV is
            # unwritten). kept + len(rest) == prefix — one query token short of
            # the canonical context whenever the stored key is longer.
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(result.longer) - prefix
            detail = {"longer_len": len(result.longer),
                      "common_prefix": result.common_prefix,
                      "prefix": prefix, "num_to_trim": num_to_trim,
                      "kept": len(result.longer) - 1 - num_to_trim,
                      "rest_tokens": len(tokens) - prefix,
                      "context_total": len(result.longer) - 1 - num_to_trim
                                       + (len(tokens) - prefix),
                      "query_tokens": len(tokens),
                      "cache": trimmable}
        elif result.shorter is not None:
            branch = "shorter"
            detail = {"shorter_len": len(result.shorter)}
        probes.append({"event": "fetch", "query_tokens": len(tokens),
                       "branch": branch, **detail})
        return orig_fetch(self, model, tokens)

    cache_mod.LRUPromptCache.fetch_nearest_cache = fetch_probe

    # also surface the real class mix of a live cache at insert time
    orig_insert = cache_mod.LRUPromptCache.insert_cache

    def insert_probe(self, model, tokens, prompt_cache, *a, **k):
        probes.append({"event": "insert", "key_tokens": len(tokens),
                       "cache": [type(c).__name__ for c in prompt_cache],
                       "trimmable": [c.is_trimmable() for c in prompt_cache]})
        return orig_insert(self, model, tokens, prompt_cache, *a, **k)

    cache_mod.LRUPromptCache.insert_cache = insert_probe

    # --- 2. prompt + generated token ids -----------------------------
    tok_utils = __import__("mlx_lm.tokenizer_utils",
                           fromlist=["TokenizerWrapper"])
    orig_apply = tok_utils.TokenizerWrapper.apply_chat_template

    def apply_probe(self, *a, **k):
        out = orig_apply(self, *a, **k)
        try:
            ids = out if isinstance(out, list) else None
            if ids is None and hasattr(out, "input_ids"):
                ids = out["input_ids"]
            if ids is not None:
                probes.append({"event": "template", "n": len(ids),
                               "prompt_ids_sha16": sha16(ids)})
        except Exception:
            pass
        return out

    tok_utils.TokenizerWrapper.apply_chat_template = apply_probe

    # generated ids: wrap stream_generate used by the server module
    orig_stream = srv.stream_generate

    def stream_probe(*a, **k):
        ids, t0, t_first = [], time.perf_counter(), None
        for r in orig_stream(*a, **k):
            if t_first is None:
                t_first = time.perf_counter()
            ids.append(r.token)
            yield r
        probes.append({"event": "generation", "n": len(ids),
                       "wall_s": round(time.perf_counter() - t0, 3),
                       "ttft_s": round(t_first - t0, 4) if t_first else None,
                       "ids_sha16": sha16(ids),
                       "ids": ids})

    srv.stream_generate = stream_probe

    # --- 3. run the real server --------------------------------------
    sys.argv = ["mlx_lm.server"] + sys.argv[1:]
    from mlx_lm.server import main as server_main
    try:
        server_main()
    finally:
        for p in probes:
            print("PROBE:", json.dumps(p), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
