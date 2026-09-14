#!/usr/bin/env python3
"""Stage the three encoder islands' dense input buffers and host references.

Inputs are the real layer-0 tensors derived from the authenticated golden
capture (see milrun.py / validate_full.py). Every buffer is plain row-major
dense bytes in the manifest `nchw` axis order; the runtime does tile placement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/var/tmp/IslandsExecJwm1")
STAGE = ROOT / "stage"
NINF_BITS = 0xFC00


def fp16_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """fp32 accumulation, one fp16 rounding at the end."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def emit(name: str, array: np.ndarray, expect_bytes: int) -> dict:
    raw = np.ascontiguousarray(array).tobytes()
    if len(raw) != expect_bytes:
        raise SystemExit(f"{name}: staged {len(raw)} bytes, manifest wants {expect_bytes}")
    path = STAGE / f"{name}.bin"
    path.write_bytes(raw)
    return {
        "file": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    layer0 = np.load(ROOT / "tensors" / "layer0.npz")

    q_v = layer0["query_states_with_bias_v_1_cast_fp16"]
    pos_kT = layer0["var_355_to_fp16"]
    q_scaled = layer0["mul_0_cast_fp16"]
    k_heads = layer0["hidden_states_23_cast_fp16"]
    k_headsT = np.ascontiguousarray(np.transpose(k_heads, (0, 1, 3, 2)))
    matrix_bd_5 = layer0["matrix_bd_5_cast_fp16"]
    cond_1head = layer0["var_373"]
    probs = layer0["softmax_0_cast_fp16"]
    v_heads = layer0["hidden_states_25_cast_fp16"]

    # Vulkan materializes the [1,1,375,375] -> [1,8,375,375] broadcast; the
    # island binds 1125000 bool elements, one byte each (header channel 7).
    cond = np.ascontiguousarray(
        np.broadcast_to(cond_1head, (1, 8, 375, 375))
    ).astype(np.uint8)
    ninf = np.full((1, 8, 375, 375), np.frombuffer(
        NINF_BITS.to_bytes(2, "little"), dtype=np.float16
    )[0], dtype=np.float16)

    records: dict[str, dict] = {}
    records["A/q_v"] = emit("A_q_v", q_v, 768000)
    records["A/pos_kT"] = emit("A_pos_kT", pos_kT, 1533952)
    records["A/q_scaled"] = emit("A_q_scaled", q_scaled, 768000)
    records["A/k_headsT"] = emit("A_k_headsT", k_headsT, 768000)
    records["B/ninf_rt"] = emit("B_ninf_rt", ninf, 2250000)
    records["B/matrix_bd_5"] = emit("B_matrix_bd_5", matrix_bd_5, 2250000)
    records["B/cond"] = emit("B_cond", cond, 1125000)
    records["C/probs"] = emit("C_probs", probs, 2250000)
    records["C/v_heads"] = emit("C_v_heads", v_heads, 768000)

    # References computed from exactly the staged bytes.
    ref_scores = fp16_matmul(q_v, pos_kT)
    ref_matmul0 = fp16_matmul(q_scaled, k_headsT)
    ref_select = np.where(cond.astype(bool), ninf, matrix_bd_5).astype(np.float16)
    ref_pv = fp16_matmul(probs, v_heads)

    records["A/ref attention_scores_1"] = emit("A_ref_attention_scores_1", ref_scores, 4494000)
    records["A/ref matmul_0"] = emit("A_ref_matmul_0", ref_matmul0, 2250000)
    records["B/ref attention_mask_9"] = emit("B_ref_attention_mask_9", ref_select, 2250000)
    records["C/ref attn_output_1"] = emit("C_ref_attn_output_1", ref_pv, 768000)

    # Cross-check: the reference must agree with the same tensor as the full MIL
    # graph produced it. Any disagreement means the staging permuted something.
    checks = {
        "attention_scores_1": (ref_scores, layer0["attention_scores_1_cast_fp16"]),
        "matmul_0": (ref_matmul0, layer0["matmul_0_cast_fp16"]),
        "attention_mask_9": (ref_select, layer0["attention_mask_9_cast_fp16"]),
        "attn_output_1": (ref_pv, layer0["attn_output_1_cast_fp16"]),
    }
    graph_agreement = {}
    for name, (mine, graph) in checks.items():
        equal = int(
            np.count_nonzero(
                mine.view(np.uint16).ravel() == np.asarray(graph).view(np.uint16).ravel()
            )
        )
        graph_agreement[name] = {
            "elements": int(mine.size),
            "bitwise_equal": equal,
            "max_abs_err": float(
                np.abs(mine.astype(np.float32) - np.asarray(graph).astype(np.float32))[
                    np.isfinite(mine.astype(np.float32))
                ].max(initial=0.0)
            ),
        }

    summary = {
        "cond_true_elements": int(cond.sum()),
        "cond_total_elements": int(cond.size),
        "ninf_fp16_bits": f"0x{NINF_BITS:04x}",
        "staged": records,
        "reference_vs_graph": graph_agreement,
    }
    (ROOT / "stage-manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
