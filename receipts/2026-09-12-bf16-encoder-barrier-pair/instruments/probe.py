# Engagement probe for MLX_OMARCHY_DEFERRED_POST_BARRIERS (ctypes, C ABI).
# Reads BOTH snapshot symbols: the published 8-field
# mlx_omarchy_trace_snapshot (dispatch structure) and the separate
# mlx_omarchy_trace_barriers (barrier decisions). Base wheel expects
# post_barriers_deferred == 0 and barriers_emitted ~= 2 per dispatch; the
# deferred wheel expects deferred ~= 1 per dispatch and emitted ~= 1 per
# dispatch (flushes included).
import ctypes
import glob
import importlib.util
import json
import os

import mlx.core as mx  # noqa: F401  (loads libmlx into this process)


def find_lib():
    spec = importlib.util.find_spec("mlx")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("mlx package not found")
    root = list(spec.submodule_search_locations)[0]
    hits = sorted(glob.glob(os.path.join(root, "**", "libmlx*.so"),
                            recursive=True))
    if not hits:
        raise SystemExit("libmlx.so not found under " + root)
    return hits[0]


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "gpu_primitive_dispatches",
        "vk_submissions",
        "vk_buffer_copies",
        "vk_buffer_fills",
        "vk_compute_dispatches",
        "omarchy_finalize_calls",
        "commit_calls_with_work",
        "commit_calls_noop",
    )]


class Barriers(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "barriers_emitted",
        "barriers_skipped",
        "post_barriers_deferred",
    )]


lib = ctypes.CDLL(find_lib())
lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snap)]
lib.mlx_omarchy_trace_barriers.argtypes = [ctypes.POINTER(Barriers)]


def delta(a, b):
    return {n: getattr(b, n) - getattr(a, n)
            for n, _ in a._fields_}


a = mx.random.normal((128, 128)).astype(mx.bfloat16)
mx.eval(a)
for _ in range(3):
    mx.eval(a @ a + a)  # warmup: pipelines, allocator, pools

s0, b0 = Snap(), Barriers()
lib.mlx_omarchy_trace_snapshot(ctypes.byref(s0))
lib.mlx_omarchy_trace_barriers(ctypes.byref(b0))
for _ in range(100):
    c = a @ a + a
    mx.eval(c)
mx.eval(c)
s1, b1 = Snap(), Barriers()
lib.mlx_omarchy_trace_snapshot(ctypes.byref(s1))
lib.mlx_omarchy_trace_barriers(ctypes.byref(b1))

print("PROBE " + json.dumps(
    {"dispatches": delta(s0, s1), "barriers": delta(b0, b1)},
    sort_keys=True))
