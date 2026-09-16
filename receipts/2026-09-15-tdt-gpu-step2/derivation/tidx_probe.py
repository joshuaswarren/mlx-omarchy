#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Which thread indices actually execute? Write t (and gid) per thread for
several (grid, threadgroup) shapes and inspect the coverage."""
import sys

import numpy as np

SRC = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    out[g * 1024u + t] = float(t) + float(g) * 1000.0f;
"""


def main() -> int:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    for grid_tg, tsize in ((1, 640), (1, 256), (2, 512), (4, 256), (1, 1024)):
        k = mx.fast.metal_kernel(
            name=f"probe_tidx_{grid_tg}_{tsize}",
            input_names=["dummy"],
            output_names=["out"],
            header="",
            source=SRC,
            compile_options={"math_mode": "safe"},
        )
        dummy = mx.array(np.zeros(16, dtype=np.float16))
        mx.eval(dummy)
        r = k(
            inputs=[dummy],
            output_shapes=[(grid_tg * 1024,)],
            output_dtypes=[mx.float32],
            grid=(grid_tg, 1, 1),
            threadgroup=(tsize, 1, 1),
        )
        out = np.asarray(r[0])
        written = np.flatnonzero(out)  # 0.0 only from (g0,t0); inspect pattern
        tg0 = out[:1024]
        tid_max = -1
        nz = np.flatnonzero(tg0)
        if nz.size:
            vals = tg0[nz]
            tid_max = int(vals.max())
        print(f"grid={grid_tg} tsize={tsize}: written_in_tg0={nz.size} "
              f"max_t_written={tid_max} first_vals={tg0[nz[:5]].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
