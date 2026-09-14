#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Validate the numpy MIL evaluator against the authenticated capture.

Runs the whole pinned encoder on the capture's input features and mask and
compares the function output with the captured macOS/ANE golden, against
the reference lock's frozen encoder contract. This is what makes the
staged layer-0 island tensors real encoder tensors rather than fixtures.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mil_numpy import Program  # noqa: E402

MIL = Path("/tmp/coreml-text-adapter-pinned-source-v2/model.mil")
MODEL_ROOT = Path("/tmp/coreml-text-adapter-pinned-source-v2/model-root")
CAPTURE = Path("/tmp/parakeet-ane-capture.GKqLhJ/capture")
REPO = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "overlay").is_dir()
)
LOCK = REPO / "overlay/tools/coreml/parakeet-reference.lock"
OUT = Path("/tmp/jw16-islands-out/interpreter-validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text())
    contract = lock["numerical_contract"]
    golden_shas = lock["macos_reference_paths"]

    inputs = {
        "encoder_input_features.npy": None,
        "encoder_input_mask.npy": None,
        "encoder_hidden.npy": None,
    }
    for file_name in inputs:
        inputs[file_name] = sha256(CAPTURE / file_name)

    program = Program(MIL, MODEL_ROOT)
    program.feed(
        "input_features", np.load(CAPTURE / "encoder_input_features.npy")
    )
    program.feed("attention_mask", np.load(CAPTURE / "encoder_input_mask.npy"))
    started = time.time()
    hidden = np.asarray(program.get("encoder_hidden"), dtype=np.float32)
    mask = np.asarray(program.get("encoder_mask"))
    elapsed = time.time() - started

    golden = np.load(CAPTURE / "encoder_hidden.npy").astype(np.float32)
    golden_mask = np.load(CAPTURE / "encoder_mask.npy")
    diff = np.abs(hidden - golden)
    report = {
        "schema": "mlx-omarchy.mil-numpy-validation.v1",
        "mil": str(MIL),
        "mil_sha256": sha256(MIL),
        "weight_bin_sha256": sha256(MODEL_ROOT / "weights/weight.bin"),
        "weight_bin_matches_reference_lock": (
            sha256(MODEL_ROOT / "weights/weight.bin")
            == next(
                entry["sha256"]
                for entry in lock["files"]
                if entry["path"].endswith("encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
            )
        ),
        "normalized_bin_sha256": sha256(MODEL_ROOT / "weights/normalized.bin"),
        "normalized_bin_note": (
            "depalettized companion blob emitted by the CoreML text adapter "
            "from the pinned mlpackage; the palettized linear weights resolve "
            "through it, and the end-to-end golden comparison below is what "
            "vouches for it"
        ),
        "capture": str(CAPTURE),
        "capture_sha256": inputs,
        "capture_matches_reference_lock": {
            name: inputs[name] == golden_shas.get(name)
            for name in ("encoder_input_features.npy", "encoder_input_mask.npy")
        },
        "golden_encoder_hidden_matches_capture_manifest": (
            inputs["encoder_hidden.npy"] == golden_shas.get("encoder_hidden.npy")
        ),
        "elapsed_s": round(elapsed, 1),
        "encoder_mask_equal": bool(np.array_equal(np.asarray(mask).ravel(), golden_mask.ravel())),
        "encoder_max_abs_err": float(diff.max()),
        "encoder_mean_abs_err": float(diff.mean()),
        "encoder_rel_l2_err": float(
            np.linalg.norm(hidden - golden) / np.linalg.norm(golden)
        ),
        "frozen_contract": {
            "encoder_max_abs_err": contract["encoder_max_abs_err"],
            "encoder_mean_abs_err": contract["encoder_mean_abs_err"],
            "encoder_rel_l2_err": contract["encoder_rel_l2_err"],
        },
        "all_finite": bool(np.isfinite(hidden).all()),
    }
    report["within_frozen_contract"] = (
        report["encoder_max_abs_err"] <= contract["encoder_max_abs_err"]
        and report["encoder_mean_abs_err"] <= contract["encoder_mean_abs_err"]
        and report["encoder_rel_l2_err"] <= contract["encoder_rel_l2_err"]
    )
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
