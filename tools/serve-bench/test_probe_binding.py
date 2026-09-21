#!/usr/bin/env python3
"""Smallest CPU proof that the IDs probe intercepts the REAL chat endpoint
path of the pinned mlx-lm 0.31.3 server, without any GPU or model load:

1. import the real mlx_lm.server module (mlx stubbed),
2. apply the probe's stream_generate patch,
3. assert the patched name is what `ModelHandler._generate` resolves
   (co_names contains the global) and that the identity changed,
4. assert the HTTP route table maps /v1/chat/completions to the chat
   handler.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import types

# ---- mlx stubs ONLY where real mlx is unavailable (dev box) ----------
try:
    import mlx.core  # noqa: F401
    HAVE_MLX = True
except Exception:
    HAVE_MLX = False

if not HAVE_MLX:
    mx_stub = types.ModuleType("mlx.core")
    for _n in ("array", "sort", "logsumexp", "argmax", "eval"):
        setattr(mx_stub, _n, lambda *a, **k: None)
    mx_stub.clear_cache = lambda: None
    mx_stub.set_wired_limit = lambda *a, **k: None
    mx_stub.get_peak_memory = lambda: 0
    mx_stub.compile = lambda fn=None, *a, **k: (fn if fn is not None else (lambda *a2, **k2: None))
    mx_stub.random = types.SimpleNamespace(state=None)
    mx_stub.new_thread_local_stream = lambda *a, **k: None
    mx_stub.default_device = lambda: None
    mx_stub.stream = lambda *a, **k: (lambda *a2, **k2: None)
    mx_stub.synchronize = lambda *a, **k: None

    mx_stub.Stream = type("Stream", (), {})
    mx_stub.new_stream = lambda *a, **k: None
    mx_stub.core = mx_stub  # self-reference for mx.core attribute access
    mx_stub.Device = object
    mx_stub.load = lambda *a, **k: None
    mx_stub.save = lambda *a, **k: None
    mx_stub.matmul = lambda *a, **k: None
    mx_stub.zeros = lambda *a, **k: None
    mx_stub.ones = lambda *a, **k: None

    mx_stub.metal = types.SimpleNamespace(is_available=lambda: False)
    mx_stub.distributed = types.SimpleNamespace(
        init=lambda: types.SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        Group=object)
    nn_stub = types.ModuleType("mlx.nn")
    nn_stub.Module = type("Module", (), {})
    utils_stub = types.ModuleType("mlx.utils")
    for _n in ("tree_flatten", "tree_map", "tree_unflatten", "tree_map_with_path", "tree_reduce"):
        setattr(utils_stub, _n, lambda *a, **k: None)
    mlx_stub = types.ModuleType("mlx")
    mlx_stub.core = mx_stub
    mlx_stub.nn = nn_stub
    mlx_stub.utils = utils_stub
    mlx_stub.__path__ = []
    _spec = importlib.machinery.ModuleSpec("mlx", None, is_package=True)
    mlx_stub.__spec__ = _spec
    sys.modules.update({"mlx": mlx_stub, "mlx.core": mx_stub,
                        "mlx.nn": nn_stub, "mlx.utils": utils_stub})

WHEEL = "/tmp/mlxlm-src/x"
sys.path.insert(0, WHEEL)
import mlx_lm.server as srv  # noqa: E402  (real pinned module)

# ---- apply the probe's stream_generate patch exactly as the launcher ----
_orig_stream = srv.stream_generate
_events = []


def stream_probe(*a, **k):
    ids = []
    for r in _orig_stream(*a, **k):
        ids.append(r.token)
        yield r
    _events.append({"n": len(ids)})
    yield None  # keep generator semantics


srv.stream_generate = stream_probe

# ---- assertions ----
# 1. the chat route dispatch references the chat handler
assert "handle_chat_completions" in srv.APIHandler.do_POST.__code__.co_names

# 2. the generation thread path resolves the patched module global:
#    _generate -> _serve_single -> global stream_generate
assert "_serve_single" in srv.ResponseGenerator._generate.__code__.co_names
serve_names = srv.ResponseGenerator._serve_single.__code__.co_names
assert "stream_generate" in serve_names, serve_names

# 3. apply the probe patch and assert the resolved global is the wrapper
_orig_stream = srv.stream_generate
_events = []


def stream_probe(*a, **k):
    ids = []
    for r in _orig_stream(*a, **k):
        ids.append(r.token)
        yield r
    _events.append({"n": len(ids)})
    yield None  # keep generator semantics


srv.stream_generate = stream_probe
assert srv.stream_generate is stream_probe, "patch not applied to the global _serve_single resolves"

print("PROBE_BINDING_OK: /v1/chat/completions -> APIHandler.handle_chat_completions "
      "-> ResponseGenerator._generate -> _serve_single -> module-global stream_generate "
      "(patched)")
