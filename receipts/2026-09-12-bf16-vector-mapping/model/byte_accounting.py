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
CLASS_MARGINAL_MS = 18.31     # attribution receipt, short leg, wall-anchored
COPY_ROOF_GBS = 58.5          # same-instrument copy roof (attribution probe)
PART_GBS = 68.25              # M1 LPDDR4X part

vec = sum(n * k * 2 * c for name, (n, k, r, c) in SHAPES.items() if r == "vec")
tile = sum(n * k * 2 * c for name, (n, k, r, c) in SHAPES.items() if r == "tile")
total = vec + tile
ideal_ms = total / (COPY_ROOF_GBS * 1e9) * 1e3
lm_bytes = 151936 * 896 * 2
disp = 97 + 72

print(f"vec-class bytes/token : {vec:,}")
print(f"tile-class bytes/token: {tile:,}")
print(f"total bytes/token     : {total:,}")
print(f"ideal @ {COPY_ROOF_GBS} GB/s copy roof : {ideal_ms:.2f} ms")
print(f"measured class marginal: {CLASS_MARGINAL_MS} ms "
      f"-> {total / (CLASS_MARGINAL_MS * 1e-3) / 1e9:.1f} GB/s "
      f"= {total / (CLASS_MARGINAL_MS * 1e-3) / (COPY_ROOF_GBS * 1e9) * 100:.1f}% of copy roof, "
      f"{total / (CLASS_MARGINAL_MS * 1e-3) / (PART_GBS * 1e9) * 100:.1f}% of part")
print(f"above-roof bound (ALL kernel-recoverable payload time in class): "
      f"{CLASS_MARGINAL_MS - ideal_ms:.2f} ms/token")
print(f"  spread over {disp} gemv dispatches = "
      f"{(CLASS_MARGINAL_MS - ideal_ms) * 1e3 / disp:.1f} us/dispatch "
      f"(matches the ~9 us/dispatch skeleton scale, i.e. dispatch ramp, not kernel work)")
print(f"lm_head alone: {lm_bytes:,} B; at 4.687 ms isolated (span screen) = "
      f"{lm_bytes / 4.687e-3 / 1e9:.2f} GB/s = "
      f"{lm_bytes / 4.687e-3 / (COPY_ROOF_GBS * 1e9) * 100:.1f}% of copy roof; "
      f"max byte-preserving kernel gain {4.687 - lm_bytes / (COPY_ROOF_GBS * 1e9) * 1e3:.3f} ms")
