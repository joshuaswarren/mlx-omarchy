#!/usr/bin/env python3
"""Offline staging-index check; not a substitute for GPU numerical gates."""
from pathlib import Path

shader = (Path(__file__).resolve().parents[2] / "mlx/backend/omarchy/shaders/qmm_coopmat.comp").read_text()
for fragment in (
    "const uint STEP_K = 32u;",
    "#define X_LOADS (TILE_ROWS * STEP_K / 128u)",
    "uint local_row = i / (STEP_K / 2u);",
    "uint staged_part = w_part + 2u * j;",
    "k_base / 8u + staged_part",
    "8u * staged_part * TILE_N",
):
    assert fragment in shader, fragment

step = 32
for rows in (16, 32):
    x_addresses = []
    weight_addresses = []
    for lane in range(64):
        for j in range(rows * step // 128):
            i = lane + 64 * j
            row = i // (step // 2)
            col = 2 * (i % (step // 2))
            x_addresses.extend((row * step + col, row * step + col + 1))
        for j in range(step // 16):
            part = lane // 32 + 2 * j
            base = 8 * part * 32 + lane % 32
            weight_addresses.extend(base + nibble * 32 for nibble in range(8))
    assert sorted(x_addresses) == list(range(rows * step))
    assert sorted(weight_addresses) == list(range(step * 32))
    assert (rows * step + step * 32) * 4 == (8192 if rows == 32 else 6144)
    # Four output 8x8 blocks reuse the first 256 floats of x_s.
    assert 256 <= rows * step
for k in (64, 896, 4864):
    actual = [chunk * 64 + stage * step + fragment * 8
              for chunk in range(k // 64)
              for stage in range(64 // step)
              for fragment in range(step // 8)]
    assert actual == list(range(0, k, 8))
print("PASS 16/32-row input and weight staging coverage, shared bounds, ascending 8-wide K order")
