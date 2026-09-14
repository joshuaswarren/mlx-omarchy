# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Stage the real layer-0 encoder island tensors for hardware submission.

Evaluates the pinned Parakeet encoder MIL on the authenticated capture
inputs (host, numpy) and writes the dense row-major fp16/bool buffers the
three encoder ANE islands bind, plus the host reference outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mil_numpy import Program  # noqa: E402

MIL = Path("/tmp/coreml-text-adapter-pinned-source-v2/model.mil")
MODEL_ROOT = Path("/tmp/coreml-text-adapter-pinned-source-v2/model-root")
CAPTURE = Path("/tmp/parakeet-ane-capture.GKqLhJ/capture")
OUT = Path("/tmp/jw16-encoder-islands-stage")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(name: str, array: np.ndarray, dtype) -> dict:
    array = np.ascontiguousarray(array, dtype=dtype)
    path = OUT / f"{name}.bin"
    array.tofile(path)
    record = {
        "name": name,
        "dtype": str(np.dtype(dtype)),
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if dtype == np.float16:
        flat = array.astype(np.float32).ravel()
        record.update(
            finite=int(np.isfinite(flat).sum()),
            nan=int(np.isnan(flat).sum()),
            neg_inf=int(np.isneginf(flat).sum()),
            min=None if not np.isfinite(flat).any() else float(flat[np.isfinite(flat)].min()),
            max=None if not np.isfinite(flat).any() else float(flat[np.isfinite(flat)].max()),
            mean_abs=float(np.abs(flat[np.isfinite(flat)]).mean()) if np.isfinite(flat).any() else None,
        )
    else:
        record.update(true_count=int(array.astype(np.uint8).sum()))
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    program = Program(MIL, MODEL_ROOT)
    program.feed(
        "input_features", np.load(CAPTURE / "encoder_input_features.npy")
    )
    program.feed(
        "attention_mask", np.load(CAPTURE / "encoder_input_mask.npy")
    )

    get = program.get
    q_v = get("query_states_with_bias_v_1_cast_fp16")
    q_scaled = get("mul_0_cast_fp16")
    pos_kT = get("var_355_to_fp16")
    k_heads = get("hidden_states_23_cast_fp16")  # [1,8,375,128]
    k_headsT = np.ascontiguousarray(np.swapaxes(k_heads, -1, -2))
    matrix_bd_5 = get("matrix_bd_5_cast_fp16")
    cond_1head = get("var_373")  # [1,1,375,375] bool
    cond = np.ascontiguousarray(np.broadcast_to(cond_1head, (1, 8, 375, 375)))
    ninf_scalar = np.asarray(get("var_8_to_fp16"), dtype=np.float16)
    ninf_rt = np.full((1, 8, 375, 375), ninf_scalar, dtype=np.float16)
    probs = get("softmax_0_cast_fp16")
    v_heads = get("hidden_states_25_cast_fp16")

    # Host references, fp32 accumulation rounded to fp16 (CoreML fp16 model).
    def mm(x, y):
        return np.matmul(
            x.astype(np.float32), y.astype(np.float32)
        ).astype(np.float16)

    attention_scores_1 = mm(q_v, pos_kT)
    matmul_0 = mm(q_scaled, k_headsT)
    attention_mask_9 = np.where(cond, ninf_rt, matrix_bd_5).astype(np.float16)
    attn_output_1 = mm(probs, v_heads)

    graph_scores = np.asarray(get("attention_scores_1_cast_fp16"))
    graph_matmul_0 = np.asarray(get("matmul_0_cast_fp16"))
    graph_mask_9 = np.asarray(get("attention_mask_9_cast_fp16"))
    graph_attn_out = np.asarray(get("attn_output_1_cast_fp16"))
    for label, mine, graph in (
        ("attention_scores_1", attention_scores_1, graph_scores),
        ("matmul_0", matmul_0, graph_matmul_0),
        ("attention_mask_9", attention_mask_9, graph_mask_9),
        ("attn_output_1", attn_output_1, graph_attn_out),
    ):
        if not np.array_equal(mine, graph):
            raise SystemExit(
                f"island reference for {label} disagrees with the graph value"
            )

    manifest = {
        "schema": "mlx-omarchy.encoder-island-stage.v1",
        "mil": str(MIL),
        "model_root": str(MODEL_ROOT),
        "capture": str(CAPTURE),
        "ninf_scalar_u16": f"0x{ninf_scalar.view(np.uint16):04x}",
        "tensors": {},
    }
    staged = [
        ("q_v", q_v, np.float16),
        ("pos_kT", pos_kT, np.float16),
        ("q_scaled", q_scaled, np.float16),
        ("k_headsT", k_headsT, np.float16),
        ("attention_scores_1", attention_scores_1, np.float16),
        ("matmul_0", matmul_0, np.float16),
        ("ninf_rt", ninf_rt, np.float16),
        ("matrix_bd_5", matrix_bd_5, np.float16),
        ("cond", cond, np.bool_),
        ("attention_mask_9", attention_mask_9, np.float16),
        ("probs", probs, np.float16),
        ("v_heads", v_heads, np.float16),
        ("attn_output_1", attn_output_1, np.float16),
    ]
    for name, array, dtype in staged:
        manifest["tensors"][name] = write(name, array, dtype)

    (OUT / "stage-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
