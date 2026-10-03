"""27B per-layer state dump: mask None vs all-True, per-layer divergence.

Loads the 27B once per route (chosen via cache left_padding), runs the SAME
template ids, and prints per-layer cache state hashes. Diffing the two runs'
LAYER lines gives the first layer where the routes diverge and whether the
values are NaN.
"""
import hashlib, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

model_path = sys.argv[1]
mode = sys.argv[2]  # maskless | allvalid
print("LAYERDUMP", mode, flush=True)
model, tok = load(model_path)
seed_file = sys.argv[3] if len(sys.argv) > 3 else "serveprof/prompt600.txt"
seed_ids = tok.encode(open(seed_file).read())[:300]
templ = tok.apply_chat_template([{"role": "user", "content": tok.decode(seed_ids)}],
                                add_generation_prompt=True, tokenize=False)
ids = tok.encode(templ)
cache = model.make_cache()
if mode == "allvalid":
    for c in cache:
        if hasattr(c, "left_padding"):
            c.left_padding = mx.array([0])
h = model(mx.array([ids]), cache=cache)
def walk(x, out):
    if x is None:
        return
    if isinstance(x, (list, tuple)):
        for i in x:
            walk(i, out)
    else:
        out.append(x)
mx.eval([h] + walk(cache, []) if False else [h])
mx.eval(h)
for li, c in enumerate(cache):
    arrs = []
    walk(c.state, arrs)
    mx.eval(arrs)
    blob = b"".join(np.array(s.astype(mx.float32)).tobytes() for s in arrs)
    nan = int(sum(int(np.isnan(np.array(s.astype(mx.float32))).sum()) for s in arrs))
    print(f"LAYER {li} sha16={hashlib.sha256(blob).hexdigest()[:16]} nan={nan}", flush=True)
hn = np.array(h.astype(mx.float32))
print(f"LAST_H norm={float(np.linalg.norm(hn[0, -1])):.4f} nan={int(np.isnan(hn).sum())}", flush=True)
