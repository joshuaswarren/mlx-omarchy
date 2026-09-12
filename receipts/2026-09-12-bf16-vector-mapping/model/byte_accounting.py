"""Deterministic byte accounting for the BF16 decode gemv class.

Qwen2.5-0.5B bf16 (bench_matrix pin 56d07e76): 24 layers, hidden 896,
14 heads x 64, 2 kv heads, intermediate 4864, vocab 151936.

Per generated token the decode graph streams every projection weight once:
  MatmulVecBF16 (this receipt's kernel): wq, wo, gate, up, lm_head
  MatmulBF16 16x16 tile:                wk, wv (bias), down
Dispatch counts from receipts/2026-09-11-bf16-decode-attribution census:
97 vec + 72 tile per token.
"""
SHAPES = {
    # name: (out_n, k, route, count_per_token)
    "wq":     (896,    896, "vec",  24),
    "wo":     (896,    896, "vec",  24),
    "gate":   (4864,   896, "vec",  24),
    "up":     (4864,   896, "vec",  24),
    "lm_head": (151936, 896, "vec",  1),
    "wk":     (128,    896, "tile", 24),
    "wv":     (128,    896, "tile", 24),
    "down":   (896,   4864, "tile", 24),
}

vec = sum(n * k * 2 * c for name, (n, k, r, c) in SHAPES.items() if r == "vec")
tile = sum(n * k * 2 * c for name, (n, k, r, c) in SHAPES.items() if r == "tile")
total = vec + tile
lm_bytes = 151936 * 896 * 2

print(f"vec-class bytes/token : {vec:,}")
print(f"tile-class bytes/token: {tile:,}")
print(f"total bytes/token     : {total:,}")
print(f"lm_head bytes/token   : {lm_bytes:,}")
