#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Compare an encoder run against the macOS golden and the frozen bounds.

    python3 compare_encoder.py RUN_DIR CAPTURE_DIR OUT_JSON

Prints and stores max/mean absolute error, relative L2, NaN and Inf
counts, the mask sum, and the SHA-256 of the produced tensors, so a
performance change can be shown not to have moved the numbers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BOUNDS = {
    "encoder_max_abs_err": 0.3,
    "encoder_mean_abs_err": 0.02,
    "encoder_rel_l2_err": 0.1,
    "nan_count_allowed": 0,
    "inf_count_allowed": 0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    run = Path(argv[1])
    capture = Path(argv[2])
    out = Path(argv[3])

    got = np.load(run / "encoder_hidden.npy").astype(np.float32)
    want = np.load(capture / "encoder_hidden.npy").astype(np.float32)
    if got.shape != want.shape:
        raise SystemExit(f"shape {got.shape} != golden {want.shape}")
    difference = got - want
    denominator = float(np.linalg.norm(want.reshape(-1)))
    measured = {
        "max_abs_err": float(np.max(np.abs(difference))),
        "mean_abs_err": float(np.mean(np.abs(difference))),
        "rel_l2_err": float(np.linalg.norm(difference.reshape(-1)) / denominator),
        "nan": int(np.count_nonzero(np.isnan(got))),
        "inf": int(np.count_nonzero(np.isinf(got))),
        "bit_exact_vs_golden": bool(np.array_equal(got, want)),
    }
    report = {
        "run": str(run),
        "bounds": BOUNDS,
        "measured": measured,
        "per_bound": {
            "encoder_max_abs_err": {
                "measured": measured["max_abs_err"],
                "bound": BOUNDS["encoder_max_abs_err"],
                "pass": measured["max_abs_err"] <= BOUNDS["encoder_max_abs_err"],
            },
            "encoder_mean_abs_err": {
                "measured": measured["mean_abs_err"],
                "bound": BOUNDS["encoder_mean_abs_err"],
                "pass": measured["mean_abs_err"] <= BOUNDS["encoder_mean_abs_err"],
            },
            "encoder_rel_l2_err": {
                "measured": measured["rel_l2_err"],
                "bound": BOUNDS["encoder_rel_l2_err"],
                "pass": measured["rel_l2_err"] <= BOUNDS["encoder_rel_l2_err"],
            },
            "nan_count": {"measured": measured["nan"], "bound": 0,
                          "pass": measured["nan"] == 0},
            "inf_count": {"measured": measured["inf"], "bound": 0,
                          "pass": measured["inf"] == 0},
        },
        "encoder_hidden_sha256": sha256(run / "encoder_hidden.npy"),
        "encoder_hidden_shape": list(got.shape),
    }
    mask_path = run / "encoder_mask.npy"
    if mask_path.is_file():
        mask = np.load(mask_path)
        golden_mask = np.load(capture / "encoder_mask.npy")
        report["encoder_mask_sum"] = int(np.sum(mask))
        report["golden_encoder_mask_sum"] = int(np.sum(golden_mask))
        report["encoder_mask_exact"] = bool(
            np.array_equal(mask.reshape(-1), golden_mask.reshape(-1))
        )
        report["encoder_mask_sha256"] = sha256(mask_path)
    report["all_bounds_pass"] = all(
        entry["pass"] for entry in report["per_bound"].values()
    )
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_bounds_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
