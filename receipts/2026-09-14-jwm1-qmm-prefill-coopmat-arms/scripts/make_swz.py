"""Emit a swizzled-tile-order arm of the shipped qmm_coopmat kernel.

Only the tile-to-workgroup mapping changes. The staging, cooperative
multiply and drain phases are copied byte for byte from the shipped
shader, so the hot 16-k loop presents the driver with identical code and
the arm isolates scheduling locality: consecutive workgroup ids walk
GROUP row tiles of one column tile, so the 14-78 KiB dequantized weight
strip is read by GROUP neighbours instead of once per workgroup.
"""
import sys
from pathlib import Path

SRC = Path(
    "/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/overlay/mlx/"
    "backend/omarchy/shaders/qmm_coopmat.comp"
)
OLD = """  for (uint tile_row = gl_WorkGroupID.y; tile_row < m_tiles;
       tile_row += gl_NumWorkGroups.y) {
    uint row_base = tile_row * TILE_M;
    for (uint tile_col = gl_WorkGroupID.x; tile_col < n_tiles;
         tile_col += gl_NumWorkGroups.x) {
"""
NEW = """  // Swizzled tile order: GROUP consecutive workgroup ids share one
  // column tile and differ in row tile, so the weight strip they all
  // dequantize is read once into cache instead of once per workgroup.
  // The padded index space covers every tile exactly once; ids past the
  // last row tile skip. Both the loop bound and the guard stay
  // workgroup-uniform, so the staging barriers below remain in uniform
  // control flow.
  uint swz_block = GROUP * n_tiles;
  uint swz_total = ((m_tiles + GROUP - 1u) / GROUP) * GROUP * n_tiles;
  uint swz_stride = gl_NumWorkGroups.x * gl_NumWorkGroups.y;
  uint swz_base = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
  for (uint swz = swz_base; swz < swz_total; swz += swz_stride) {
    uint tile_row = (swz / swz_block) * GROUP + (swz % GROUP);
    uint tile_col = (swz % swz_block) / GROUP;
    if (tile_row < m_tiles) {
      uint row_base = tile_row * TILE_M;
"""


def main() -> int:
    group = sys.argv[1]
    out = Path(sys.argv[2])
    text = SRC.read_text()
    assert text.count(OLD) == 1, "tile loop header not found verbatim"
    text = text.replace(OLD, NEW)
    text = text.replace(
        "const uint MAT = 8u;",
        f"const uint MAT = 8u;\nconst uint GROUP = {group}u;",
    )
    out.write_text(text)
    print("wrote", out, "GROUP =", group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
