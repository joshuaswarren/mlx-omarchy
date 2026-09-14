"""Emit a clamp-and-mask staging arm of the shipped qmm_coopmat kernel.

The 2026-09-12 agx-qmm-codegen receipt attributed the primary stall to
five serialized global-load waits per 16-k step: each staged load sits
in its own exec-mask guard, and the driver drains pending scoreboard
messages at every block exit. The 8f6de34c arm removed those guards but
also hoisted per-lane addresses, masks and shared store offsets into
loop-indexed arrays, and measured -13 to -26 percent; that receipt left
the two effects confounded and named the ISA dump of the new shader as
the open next step.

This arm separates them at source level: it removes exactly the four
x-word guards and the packed-weight guard by clamping the row and column
index and masking the loaded word, and changes nothing else. Addressing
stays inline and recomputed per step, as in the shipped kernel, so no
per-lane array is introduced. Staged values, accumulation order and
outputs stay bit-identical: an out-of-range row stages zero as before,
and an out-of-range column keeps the shipped kernel's zero scale and
bias, so its dequantized weight tile stays zero whatever word was read.
"""
from pathlib import Path
import sys

SRC = Path(
    "/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/overlay/mlx/"
    "backend/omarchy/shaders/qmm_coopmat.comp"
)

EDITS = (
    (
        """      uint w_row = w_slice + column * words_per_row;
""",
        """      // Clamped weight row for the unguarded staging load below: an
      // out-of-range column reads a live address and relies on its zero
      // scale and bias to stage zeros, instead of branching.
      uint w_row = w_slice + min(column, params.matrix_n - 1u) *
          words_per_row;
""",
    ),
    (
        """          uint x_words[4];
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
""",
        """          uint x_words[4];
          for (uint j = 0u; j < 4u; ++j) {
            uint i = lid + 64u * j;
            uint local_row = i / 8u;
            uint row = row_base + local_row;
            uint row_safe = min(row, params.matrix_m - 1u);
            uint row_mask = row < params.matrix_m ? 0xffffffffu : 0u;
            x_words[j] = input_x.values[
                x_slice + row_safe * x_words_per_row + k_base / 2u +
                (i % 8u)] & row_mask;
          }
""",
    ),
    (
        """          uint packed = column_ok
              ? input_w.values[w_row + k_base / 8u + w_part] : 0u;
""",
        """          uint packed = input_w.values[w_row + k_base / 8u + w_part];
""",
    ),
)


def main() -> int:
    out = Path(sys.argv[1])
    text = SRC.read_text()
    for old, new in EDITS:
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    out.write_text(text)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
