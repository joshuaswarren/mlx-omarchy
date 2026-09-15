# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host proof: the shipped fused unary tables + mixed-precision cell
reproduce the capture golden 640/640 on the reduction-free lanes.

Uses the same fused_sigmoid / fused_tanh functions and the same cell algebra
as overlay/tools/coreml/vulkan_decoder.py::_lstm (the mx wrappers are data
plumbing only). Gate preacts equal the layer-0 fp16 bias bits at transition 0
(token 8192 embedding row is zero, entry state is zero). The forget term
f*f(c0) vanishes identically because the entry cell state is zero, and the
forget-gate biases are not part of the unary fixture; the remaining unaries
are exactly the measured lanes.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAC = REPO / "receipts" / "2026-09-14-lstm-unary-mac"
CAPTURE = (
    Path.home()
    / ".cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
    "20260913T105550Z-librispeech-tdt-tensors/ane"
)
F16 = np.dtype("<f2")

TOOLS = REPO / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.vulkan_decoder import fused_sigmoid, fused_tanh  # noqa: E402


def r16(values):
    return np.asarray(values, np.float64).astype(F16)


def main() -> int:
    sa = np.load(MAC / "sigma_args.npz")
    te = np.load(MAC / "tanh_extracted.npz")
    b_i = sa["args"][:640].astype(np.float64)   # input-gate preacts (bias bits)
    b_o = sa["args"][640:].astype(np.float64)   # output-gate preacts
    b_g = te["args"][:640].astype(np.float64)   # cell-gate preacts
    gold_c = np.load(CAPTURE / "tdt_trace_0000_decoder_next_cell.npy")[0].ravel().astype(np.float64)
    gold_h = np.load(CAPTURE / "tdt_trace_0000_decoder_next_hidden.npy")[0].ravel().astype(np.float64)

    c0 = np.zeros(640, dtype=np.float64)        # entry cell state is zero
    cell_int = fused_sigmoid(b_i) * fused_tanh(b_g) + fused_sigmoid(b_i) * 0.0 * c0
    next_cell = r16(cell_int)
    next_hidden = r16(fused_sigmoid(b_o) * fused_tanh(cell_int))

    cell_ok = int((next_cell.astype(np.float64) == gold_c).sum())
    hid_ok = int((next_hidden.astype(np.float64) == gold_h).sum())
    result = {
        "schema": "mlx-omarchy.lstm-fused-host-proof/1",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tables_sha256": hashlib.sha256(
            (REPO / "overlay/tools/coreml/unary_tables.npz").read_bytes()
        ).hexdigest(),
        "golden": {"next_cell": f"{cell_ok}/640", "next_hidden": f"{hid_ok}/640"},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    (Path(__file__).resolve().parent / "host_proof.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0 if (cell_ok, hid_ok) == (640, 640) else 1


if __name__ == "__main__":
    raise SystemExit(main())
