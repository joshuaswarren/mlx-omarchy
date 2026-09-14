"""Emit a one-16-byte-load-per-lane staging arm of qmm_coopmat.

Measured on the installed fork driver this session: the AGX scheduler
puts a scoreboard wait within one or two instructions of every global
load, in guarded and unguarded code alike (dump-base vs dump-clamp,
23 waits each, load->wait gaps 1-2). The staging phase therefore exposes
one full memory latency per load instruction, and the shipped kernel
issues five per 16-k step: four 32-bit x words plus one packed weight
word.

This arm cuts the four x-word loads to a single 16-byte load. Lane l
takes the four consecutive words (eight halves) at k half 8*(l&1) of row
l/2, so the same 32x16 tile is staged with the same values into the same
shared slots - bit-identical output - through one quarter of the global
load instructions. The generator also tightens the coopmat x-alignment
gate in primitives.cpp from an even f16 element offset to a 16-byte one,
reusing the route's existing staging copy for views that do not qualify;
without it a uvec4 read of an unaligned view would be misaligned.
"""
from pathlib import Path
import sys

ROOT = Path(
    "/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/overlay/mlx/"
    "backend/omarchy"
)

SHADER_EDITS = (
    (
        """layout(set = 0, binding = 0, std430) readonly buffer InputX {
  uint values[];
} input_x;
""",
        """// x is read as 16-byte quads: one load per lane per staged step.
layout(set = 0, binding = 0, std430) readonly buffer InputX {
  uvec4 values[];
} input_x;
""",
    ),
    (
        """          // Four half2 loads per lane cover the 32x16 x tile.
          uint x_words[4];
          for (uint j = 0u; j < 4u; ++j) {
            uint i = lid + 64u * j;
            uint local_row = i / 8u;
            uint row = row_base + local_row;
            x_words[j] = row < params.matrix_m
                ? input_x.values[
                      x_slice + row * x_words_per_row + k_base / 2u +
                      (i % 8u)]
                : 0u;
          }
          for (uint j = 0u; j < 4u; ++j) {
            uint i = lid + 64u * j;
            uint local_row = i / 8u;
            vec2 pair = unpackHalf2x16(x_words[j]);
            x_s[local_row * STEP_K + 2u * (i % 8u)] = pair.x;
            x_s[local_row * STEP_K + 2u * (i % 8u) + 1u] = pair.y;
          }
""",
        """          // One 16-byte load per lane covers the 32x16 x tile: lane
          // l stages the four consecutive words (eight halves) at k
          // half 8*(l&1) of tile row l/2. The word index is a multiple
          // of four because the route now requires a 16-byte-aligned x
          // view, k is a multiple of 64, and k_base a multiple of 16.
          uint x_row_local = lid / 2u;
          uint x_half = 8u * (lid % 2u);
          uint x_row = row_base + x_row_local;
          uvec4 x_quad = x_row < params.matrix_m
              ? input_x.values[
                    (x_slice + x_row * x_words_per_row + k_base / 2u +
                     x_half / 2u) / 4u]
              : uvec4(0u);
          for (uint t = 0u; t < 4u; ++t) {
            vec2 pair = unpackHalf2x16(x_quad[t]);
            x_s[x_row_local * STEP_K + x_half + 2u * t] = pair.x;
            x_s[x_row_local * STEP_K + x_half + 2u * t + 1u] = pair.y;
          }
""",
    ),
)

PRIM_EDITS = (
    (
        """  // The f16 Q4 shader reads eight halves as one uvec4. Materialize only the
  // rare row-contiguous view whose element offset is not 16-byte aligned;
  // the coopmat word-pair reader extends the same rule to a 2-byte
  // alignment (an even f16 element offset).""",
        """  // The f16 Q4 shader reads eight halves as one uvec4. Materialize only the
  // rare row-contiguous view whose element offset is not 16-byte aligned;
  // the coopmat staging reader takes its x tile as 16-byte quads too, so
  // it carries the same rule.""",
    ),
    (
        """      (!coopmat_reachable || x.offset() % (2 * x.itemsize()) == 0) &&""",
        """      (!coopmat_reachable || x.offset() % (8 * x.itemsize()) == 0) &&""",
    ),
)


def apply(text, edits, label):
    for old, new in edits:
        assert text.count(old) == 1, f"{label}: {old[:70]!r}"
        text = text.replace(old, new)
    return text


def main() -> int:
    shader_out, prim_out = Path(sys.argv[1]), Path(sys.argv[2])
    shader = (ROOT / "shaders" / "qmm_coopmat.comp").read_text()
    shader_out.write_text(apply(shader, SHADER_EDITS, "shader"))
    prim = (ROOT / "primitives.cpp").read_text()
    prim_out.write_text(apply(prim, PRIM_EDITS, "primitives"))
    print("wrote", shader_out, "and", prim_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
