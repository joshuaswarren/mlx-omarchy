#!/usr/bin/env python3
"""Bench-only instrumented launcher for mlx_lm.server.

Runs the REAL mlx_lm.server (same wheel, same venv) with runtime
monkeypatches that record, per request:
  - the exact prompt token ids and their sha16 (prompt-ID proof),
  - which cache-fetch branch executed (exact / longer+trim /
    longer+NOT-trimmable / shorter / miss), with the trim arithmetic
    (prefix, num_to_trim, kept, rest, context_total) and the cache class
    mix with per-class trimmability,
  - every raw sampled token id at the detokenizer boundary (BEFORE any
    text/reasoning parsing), emitted per request via the detokenizer's
    reset() boundary,
  - per-generation timing (wall, ttft) and generated-id sha16.
Nothing on disk is modified: the wheel and all pinned artifacts are
untouched; patches live only in this process. Output: one JSON line per
event on stderr prefixed PROBE:.

Usage: python server_ids_probe.py <normal mlx_lm.server args...>
"""
import json
import sys
import time

import mlx.core as mx


def emit(event):
    # Print IMMEDIATELY: the window wrapper SIGTERMs the server at window
    # end and buffered events would be lost with the process.
    print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)


def sha16(ids):
    import hashlib
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16]


detok_state = {"ids": []}


def install_detok_probe():
    """Wrap the StreamingDetokenizer classes so add_token records every RAW
    sampled id (pre-reasoning-parser) and reset() emits the completed
    request's id list. Only touches class objects in this process."""
    from mlx_lm.tokenizer_utils import (
        NaiveStreamingDetokenizer,
        BPEStreamingDetokenizer,
        SPMStreamingDetokenizer,
        StreamingDetokenizer,
    )

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        orig_add = getattr(cls, "add_token", None)
        if orig_add is None:
            continue

        def add_token_probe(self, token, _orig=orig_add):
            detok_state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        if not hasattr(cls, "reset"):
            continue
        orig_reset = cls.reset

        def reset_probe(self, _o=orig_reset):
            if detok_state["ids"]:
                emit({"event": "detok_ids", "n": len(detok_state["ids"]),
                      "ids_sha16": sha16(detok_state["ids"]),
                      "ids": detok_state["ids"]})
                detok_state["ids"] = []
            return _o(self)

        cls.reset = reset_probe


def make_stream_probe(orig_stream, log):
    """Wrap a stream_generate generator so exactly ONE event is emitted per
    generation, flushed AT the finish token (before handing it to the
    caller) — not deferred to generator close/GC/next reset. A caller
    that breaks early without a finish token flushes via finally with
    finish_reason=None. An `emitted` guard makes double emission
    (finish + finally) impossible."""
    state = {"emitted": False}

    def stream_probe(*a, **k):
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(600, exit=True)
        ids = []
        t0 = time.perf_counter()
        t_first = None
        finish = None
        emitted = False

        def emit_now():
            nonlocal emitted
            if emitted:
                return
            emitted = True
            faulthandler.cancel_dump_traceback_later()
            wall = time.perf_counter() - t0
            log({"event": "generation", "n": len(ids),
                 "wall_s": round(wall, 3),
                 "ttft_s": round(t_first - t0, 4) if t_first else None,
                 "ids_sha16": sha16(ids), "ids": list(ids),
                 "finish_reason": finish})

        try:
            for r in orig_stream(*a, **k):
                if t_first is None:
                    t_first = time.perf_counter()
                ids.append(r.token)
                finish = getattr(r, "finish_reason", None)
                if finish is not None:
                    # finish token (length cap or stop): flush NOW, before
                    # handing it to the caller — one event per request,
                    # independent of caller break/close/GC and of any
                    # next request's reset.
                    emit_now()
                yield r
                if finish is not None:
                    return
        finally:
            # early caller abort without a finish token: flush the partial
            # generation once
            emit_now()

    return stream_probe


def make_batch_probe(orig_next, log):
    """Capture the batch path at finish, including the final HTTP request."""
    def next_probe(self, *args, **kwargs):
        result = orig_next(self, *args, **kwargs)
        if not hasattr(self, "_probe_token_ids"):
            self._probe_token_ids = {}
        for response in result[1]:
            ids = self._probe_token_ids.setdefault(response.uid, [])
            ids.append(int(response.token))
            if response.finish_reason is not None:
                log({"event": "generation", "path": "batch",
                     "uid": response.uid, "n": len(ids), "ids": list(ids),
                     "ids_sha16": sha16(ids),
                     "finish_reason": response.finish_reason})
                del self._probe_token_ids[response.uid]
        return result
    return next_probe


def main():
    import faulthandler
    import mlx_lm.server as srv

    # --- 1. cache fetch branch + arithmetic probe ---------------------
    cache_mod = __import__("mlx_lm.models.cache", fromlist=["LRUPromptCache"])
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
        emit({"event": "fetch", "query_tokens": len(tokens),
              "branch": branch, **detail})
        return orig_fetch(self, model, tokens)

    cache_mod.LRUPromptCache.fetch_nearest_cache = fetch_probe

    # also surface the real class mix of a live cache at insert time
    orig_insert = cache_mod.LRUPromptCache.insert_cache

    def insert_probe(self, model, tokens, prompt_cache, *a, **k):
        emit({"event": "insert", "key_tokens": len(tokens),
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
                emit({"event": "template", "n": len(ids),
                      "prompt_ids_sha16": sha16(ids)})
        except Exception:
            pass
        return out

    tok_utils.TokenizerWrapper.apply_chat_template = apply_probe

    # --- 2b. raw sampled token IDs at the detokenizer boundary ---------
    # add_token receives the RAW sampled id before any text reasoning
    # parsing; reset() marks a request boundary. Emitted per request.
    install_detok_probe()

    srv.BatchGenerator.next = make_batch_probe(srv.BatchGenerator.next, emit)

    srv.stream_generate = make_stream_probe(srv.stream_generate, emit)

    # --- 3. run the real server --------------------------------------
    sys.argv = ["mlx_lm.server"] + sys.argv[1:]
    from mlx_lm.server import main as server_main
    server_main()


if __name__ == "__main__":
    main()
