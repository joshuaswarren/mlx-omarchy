#!/usr/bin/env python3
"""Prove the layer-0 derivation is real: run the whole pinned encoder MIL in the
same evaluator and compare its two function outputs against the golden capture.

If encoder_hidden matches the authenticated ANE capture inside the reference
lock's frozen tolerances, the layer-0 tensors that the islands bind are the real
encoder tensors, not a plausible reconstruction.
"""

from pathlib import Path

import numpy as np

from milrun import Program

SOURCE = Path("/var/tmp/IslandsExecJwm1/encoder-source")
CAPTURE = Path(
    "~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
    "20260912T154759Z-librispeech/ane"
).expanduser()

features = np.load(CAPTURE / "encoder_input_features.npy")
mask = np.load(CAPTURE / "encoder_input_mask.npy")
golden_hidden = np.load(CAPTURE / "encoder_hidden.npy")
golden_mask = np.load(CAPTURE / "encoder_mask.npy")

program = Program(SOURCE / "model.mil", SOURCE / "model-root")
keep, executed = program.run(
    stop_after="encoder_mask",
    wanted={"encoder_hidden", "encoder_mask"},
    inputs={
        "input_features": features.astype(np.float32).reshape(1, 3000, 128),
        "attention_mask": mask.astype(np.int32).reshape(1, 3000),
    },
)
print("ops executed", executed)

hidden = np.asarray(keep["encoder_hidden"]).astype(np.float32)
ref = golden_hidden.astype(np.float32)
print("encoder_hidden", hidden.shape, "golden", ref.shape)
diff = np.abs(hidden - ref)
rel_l2 = float(np.linalg.norm(hidden - ref) / np.linalg.norm(ref))
print(f"max_abs_err   {diff.max():.6f}   (lock bound 0.3)")
print(f"mean_abs_err  {diff.mean():.6f}   (lock bound 0.02)")
print(f"rel_l2_err    {rel_l2:.6f}   (lock bound 0.1)")
print("nan/inf", int(np.isnan(hidden).sum()), int(np.isinf(hidden).sum()))

got_mask = np.asarray(keep["encoder_mask"]).astype(np.int32).reshape(golden_mask.shape)
print("encoder_mask exact:", bool(np.array_equal(got_mask, golden_mask.astype(np.int32))))
print("encoder_mask sum", int(got_mask.sum()), "golden sum", int(golden_mask.sum()))


import json

Path("/var/tmp/IslandsExecJwm1/validate-full.json").write_text(
    json.dumps(
        {
            "ops_executed": int(executed),
            "encoder_hidden_shape": list(hidden.shape),
            "max_abs_err": float(diff.max()),
            "mean_abs_err": float(diff.mean()),
            "rel_l2_err": rel_l2,
            "lock_bounds": {
                "encoder_max_abs_err": 0.3,
                "encoder_mean_abs_err": 0.02,
                "encoder_rel_l2_err": 0.1,
            },
            "within_lock_bounds": bool(
                diff.max() <= 0.3 and diff.mean() <= 0.02 and rel_l2 <= 0.1
            ),
            "nan": int(np.isnan(hidden).sum()),
            "inf": int(np.isinf(hidden).sum()),
            "encoder_mask_exact": bool(
                np.array_equal(got_mask, golden_mask.astype(np.int32))
            ),
            "encoder_mask_sum": int(got_mask.sum()),
        },
        indent=2,
    )
)
print("wrote validate-full.json")