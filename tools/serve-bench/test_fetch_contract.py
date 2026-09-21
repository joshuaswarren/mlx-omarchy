#!/usr/bin/env python3
"""CPU unit test of mlx-lm 0.31.3 PromptCache.fetch_nearest_cache key/offset
contract — the exact arithmetic behind the warm-round context state.

Runs the REAL fetch_nearest_cache/insert_cache/trie logic from the pinned
mlx-lm 0.31.3 wheel with a stubbed mlx.core (cache bookkeeping is pure
Python) and mock cache entries that record trim calls. No GPU, no model.

Scenario proven (or refuted) per case:
  1. warm-hit trim arithmetic: stored key = prompt(60)+gen(16) = 76 tokens,
     cache offset = 75 (the last key token's KV is never written), query =
     the 60-token prompt. Does the server's steady round attend all 60
     prompt tokens, or does it drop one?
  2. non-trimmable cache (the ArraysCache case): which branch runs and
     what state the request gets.
  3. exact-key case: untrimmed deepcopy, full length.
"""
import hashlib
import json
import sys
import types

# ---- stub mlx.core / mlx so models.cache imports on a CPU-only box ----
mx_stub = types.ModuleType("mlx.core")
mx_stub.array = lambda *a, **k: None
mx_stub.sort = lambda *a, **k: None
mx_stub.logsumexp = lambda *a, **k: None
mx_stub.argmax = lambda *a, **k: None
mx_stub.clear_cache = lambda: None
mx_stub.set_wired_limit = lambda *a, **k: None
mx_stub.get_peak_memory = lambda: 0
mx_stub.eval = lambda *a, **k: None
nn_stub = types.ModuleType("mlx.nn")
nn_stub.Module = object
utils_stub = types.ModuleType("mlx.utils")


def _no_tree(fn_name):
    def _f(tree, *a, **k):
        return tree
    return _f


for _name in ("tree_flatten", "tree_map", "tree_unflatten", "tree_map_with_path", "tree_reduce"):
    setattr(utils_stub, _name, _no_tree(_name))
mlx_stub = types.ModuleType("mlx")
mlx_stub.core = mx_stub
mlx_stub.nn = nn_stub
mlx_stub.utils = utils_stub
mlx_stub.__path__ = []  # mark as package so 'mlx_lm' import resolves cleanly
sys.modules.setdefault("mlx", mlx_stub)
sys.modules.setdefault("mlx.core", mx_stub)
sys.modules.setdefault("mlx.nn", nn_stub)
sys.modules.setdefault("mlx.utils", utils_stub)

# pinned mlx-lm 0.31.3 wheel extracted at /tmp/mlxlm-src/x
WHEEL_MLX_LM = "/tmp/mlxlm-src/x/mlx_lm"
CACHE_SRC = WHEEL_MLX_LM + "/models/cache.py"

import importlib.util  # noqa: E402
import importlib.machinery  # noqa: E402

# Register a minimal 'mlx_lm' package (its real __init__ pulls transformers)
# so models/cache.py's relative imports resolve, then load cache.py inside it.
mlx_lm_pkg = importlib.util.module_from_spec(
    importlib.machinery.ModuleSpec("mlx_lm", None, is_package=True))
mlx_lm_pkg.__path__ = [WHEEL_MLX_LM]
sys.modules["mlx_lm"] = mlx_lm_pkg
models_pkg = importlib.util.module_from_spec(
    importlib.machinery.ModuleSpec("mlx_lm.models", None, is_package=True))
models_pkg.__path__ = [WHEEL_MLX_LM + "/models"]
sys.modules["mlx_lm.models"] = models_pkg
mlx_lm_pkg.models = models_pkg

_spec = importlib.util.spec_from_file_location("mlx_lm.models.cache", CACHE_SRC)
_cache_mod = importlib.util.module_from_spec(_spec)
sys.modules["mlx_lm.models.cache"] = _cache_mod
_spec.loader.exec_module(_cache_mod)
PromptCache = _cache_mod.LRUPromptCache


class MockCache:
    """Mimics the KVCache offset/trim contract with plain ints."""

    def __init__(self, offset, trimmable=True):
        self.offset = offset
        self._trimmable = trimmable
        self.trims = []

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self.trims.append(n)
        return n

    @property
    def nbytes(self):
        return 1_000_000

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        pass


def make_entry_cache(offset, layers=4, trimmable=True):
    return [MockCache(offset, trimmable) for _ in range(layers)]


def prompt_tokens(n):
    return list(range(1000, 1000 + n))


def main():
    results = {}
    # ---- case 1: warm hit, trimmable, stored key LONGER than query ----
    pc = PromptCache(max_size=10)
    warm_key = prompt_tokens(60) + list(range(2000, 2016))  # 60 + 16 gen = 76
    entry_cache = make_entry_cache(offset=75)  # 59 prefill + 16 decode writes
    pc.insert_cache("m", warm_key, entry_cache)
    query = prompt_tokens(60)
    cache, rest = pc.fetch_nearest_cache("m", query)
    kept = cache[0].offset if cache is not None else None
    trims = [c.trims for c in (cache or entry_cache)]
    results["case1_warm_hit_trimmable"] = {
        "branch": "longer+trim" if cache is not None else "miss/None",
        "kept_offset": kept,
        "rest_len": len(rest),
        "context_total": (kept + len(rest)) if cache is not None else None,
        "canonical_context": len(query),
        "token_dropped": (kept + len(rest)) < len(query) if cache is not None else None,
        "per_layer_trims": trims if cache is None else [c.trims for c in cache],
    }

    # ---- case 2: non-trimmable cache (ArraysCache case) ----
    pc2 = PromptCache(max_size=10)
    pc2.insert_cache("m", warm_key, make_entry_cache(offset=75, trimmable=False))
    cache2, rest2 = pc2.fetch_nearest_cache("m", query)
    results["case2_non_trimmable"] = {
        "branch": "longer+trim" if cache2 is not None else "miss/None",
        "cache_returned": cache2 is not None,
        "rest_len": len(rest2),
        "context_total": (len(query)) if cache2 is None else len(query),
    }

    # ---- case 3: exact-key hit ----
    pc3 = PromptCache(max_size=10)
    exact_key = prompt_tokens(60)
    pc3.insert_cache("m", exact_key, make_entry_cache(offset=59))
    cache3, rest3 = pc3.fetch_nearest_cache("m", exact_key)
    results["case3_exact_key"] = {
        "branch": "exact" if cache3 is not None else "miss/None",
        "kept_offset": cache3[0].offset if cache3 else None,
        "rest_len": len(rest3),
    }

    print(json.dumps(results, indent=1))

    # Assertions for the documented contract:
    c1 = results["case1_warm_hit_trimmable"]
    assert c1["kept_offset"] == 58, c1
    assert c1["rest_len"] == 1, c1
    assert c1["context_total"] == 59 and c1["canonical_context"] == 60, c1
    assert c1["token_dropped"] is True, c1
    assert results["case2_non_trimmable"]["cache_returned"] is False, "can_trim False must skip the longer+trim branch"
    assert results["case3_exact_key"]["kept_offset"] == 59, results["case3_exact_key"]

    print("FETCH_CONTRACT_OK: longer-key trim keeps prefix-1 entries; "
          "one prompt token dropped per warm hit in mlx-lm 0.31.3")
    return 0


sys.exit(main())
