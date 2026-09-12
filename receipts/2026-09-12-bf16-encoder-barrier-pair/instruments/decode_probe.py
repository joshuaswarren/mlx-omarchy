# Decode-loop structure probe: per-token dispatch/submission/barrier
# counters around the real mlx_lm greedy decode loop (same stream_generate
# the canonical bench uses). Gate-on candidate wheel reports the barrier
# symbols too; the base wheel predates mlx_omarchy_trace_barriers and
# reports the published 8-field snapshot only.
import ctypes
import glob
import importlib.util
import json
import os
import sys

import mlx.core as mx  # noqa: F401


def find_lib():
    spec = importlib.util.find_spec("mlx")
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
snap = Snap()
lib.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
have_barriers = hasattr(lib, "mlx_omarchy_trace_barriers")
bar = Barriers()
if have_barriers:
    lib.mlx_omarchy_trace_barriers.argtypes = [ctypes.POINTER(Barriers)]
    lib.mlx_omarchy_trace_barriers(ctypes.byref(bar))

from mlx_lm.utils import load  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

model_path = sys.argv[1]
N_WARM = 4
N_MEAS = 12
model, tokenizer = load(model_path)
prompt_ids = tokenizer.encode("Hi")
sampler = make_sampler(temp=0.0)

for _ in stream_generate(model, tokenizer, prompt_ids,
                         max_tokens=N_WARM, sampler=sampler):
    pass

s0, b0 = Snap(), Barriers()
lib.mlx_omarchy_trace_snapshot(ctypes.byref(s0))
if have_barriers:
    lib.mlx_omarchy_trace_barriers(ctypes.byref(b0))
n = 0
for _ in stream_generate(model, tokenizer, prompt_ids,
                         max_tokens=N_MEAS, sampler=sampler):
    n += 1
s1, b1 = Snap(), Barriers()
lib.mlx_omarchy_trace_snapshot(ctypes.byref(s1))
if have_barriers:
    lib.mlx_omarchy_trace_barriers(ctypes.byref(b1))


def per_token(a, b):
    return {n_: (getattr(b, n_) - getattr(a, n_)) / n
            for n_, _ in a._fields_}


out = {"tokens": n, "per_token": per_token(s0, s1),
       "barriers_available": have_barriers}
if have_barriers:
    out["barriers_per_token"] = per_token(b0, b1)
print("DECODEPROBE " + json.dumps(out, sort_keys=True))
