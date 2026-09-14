#!/usr/bin/env python3
"""Byte-equality matrix and deterministic metrics for the three arms.

Why this exists. `compare_golden.py` computes its reductions in float32, the
dtype of the tensors, so `np.linalg.norm` and `mean` accumulate in float32 and
their result depends on the summation order the BLAS picks. Two invocations on
byte-identical inputs can therefore disagree in the seventh significant digit
-- and they do: the 2026-09-14 two-island receipt records rel_l2
0.024916470050811768 for an `encoder_hidden.npy` whose sha256 is the very same
b3d60c81... this run reproduced, which the same script scored 0.024916632100939751.

That makes a rel_l2 difference smaller than about 1e-9 meaningless as evidence
about placement, and it means arm-to-arm deltas must be read off the bytes and
off a deterministic reduction, not off two float32 comparator runs. This script
provides both: exact byte equality, and float64 metrics that are reproducible
across invocations. The contract verdicts stay with `compare_golden.py`, which
scores the frozen bounds; nothing here relaxes or restates them.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(run: np.ndarray, golden: np.ndarray) -> dict:
    """float64 throughout, so the numbers are reproducible."""
    a = run.astype(np.float64)
    b = golden.astype(np.float64)
    diff = np.abs(a - b)
    return {
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_l2_err": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "nan": int(np.isnan(run).sum()),
        "inf": int(np.isinf(run).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    arms = {"ane3": "out-ane3", "ane2": "out-ane2", "vulkan": "out-vulkan"}
    golden = np.load(args.dir / "golden_encoder_hidden.npy")
    golden_mask = np.load(args.dir / "golden_encoder_mask.npy")

    hidden, result = {}, {}
    for arm, sub in arms.items():
        path = args.dir / sub / "encoder_hidden.npy"
        mask_path = args.dir / sub / "encoder_mask.npy"
        hidden[arm] = np.load(path)
        mask = np.load(mask_path).reshape(golden_mask.shape)
        result[arm] = {
            "encoder_hidden_sha256": sha256(path),
            "encoder_mask_sha256": sha256(mask_path),
            "float64_metrics": metrics(hidden[arm], golden),
            "encoder_mask_exact": bool(
                np.array_equal(mask, golden_mask.astype(mask.dtype))
            ),
        }

    pairs = {}
    for left, right in itertools.combinations(arms, 2):
        identical = bool(np.array_equal(hidden[left], hidden[right]))
        diff = np.abs(
            hidden[left].astype(np.float64) - hidden[right].astype(np.float64)
        )
        pairs[f"{left}_vs_{right}"] = {
            "bitwise_identical": identical,
            "differing_elements": int((hidden[left] != hidden[right]).sum()),
            "max_abs_diff": float(diff.max()),
            "rel_l2_delta": (
                result[right]["float64_metrics"]["rel_l2_err"]
                - result[left]["float64_metrics"]["rel_l2_err"]
            ),
        }

    prior = {
        "receipt": "receipts/2026-09-14-encoder-parity-ane.json",
        "ane_split_encoder_hidden_sha256": (
            "b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e"
        ),
        "vulkan_only_encoder_hidden_sha256": (
            "02092bac3054e1dd31afc96658446b9ce35e3bed72424955f57e87ca6ad850b7"
        ),
        "ane_split_rel_l2_as_recorded": 0.024916470050811768,
        "vulkan_only_rel_l2_as_recorded": 0.024595974013209343,
        "ane2_reproduces_prior_bytes": (
            result["ane2"]["encoder_hidden_sha256"]
            == "b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e"
        ),
        "vulkan_reproduces_prior_bytes": (
            result["vulkan"]["encoder_hidden_sha256"]
            == "02092bac3054e1dd31afc96658446b9ce35e3bed72424955f57e87ca6ad850b7"
        ),
    }

    report = {
        "golden_encoder_hidden_sha256": sha256(
            args.dir / "golden_encoder_hidden.npy"
        ),
        "arms": result,
        "pairwise": pairs,
        "prior_run": prior,
        "float32_nondeterminism": (
            "compare_golden.py reduces in float32, so its rel_l2 and mean are "
            "reproducible only to about 1e-9 relative across invocations even "
            "on byte-identical inputs. The float64_metrics above are the "
            "deterministic numbers; a rel_l2 difference below that floor is "
            "comparator noise, not a placement effect. max_abs_err is a max "
            "rather than a sum and is exact in both dtypes."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2))

    for arm in arms:
        m = result[arm]["float64_metrics"]
        print(
            f"{arm:7s} rel_l2={m['rel_l2_err']:.18f} max={m['max_abs_err']:.9f} "
            f"mean={m['mean_abs_err']:.12f} nan={m['nan']} inf={m['inf']} "
            f"sha={result[arm]['encoder_hidden_sha256'][:16]}"
        )
    for name, pair in pairs.items():
        print(
            f"{name:18s} identical={pair['bitwise_identical']} "
            f"differing={pair['differing_elements']} "
            f"max_abs_diff={pair['max_abs_diff']:.9f} "
            f"rel_l2_delta={pair['rel_l2_delta']:+.18f}"
        )
    print(
        "prior bytes reproduced: ane2 "
        f"{prior['ane2_reproduces_prior_bytes']}, vulkan "
        f"{prior['vulkan_reproduces_prior_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
