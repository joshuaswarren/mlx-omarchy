#!/usr/bin/env python3
"""Compare an encoder run's outputs against the authenticated macOS Core ML
golden capture, under the frozen numerical contract in parakeet-reference.lock.

Bounds are read from the lock and never relaxed. The comparison is analysis,
not part of the parity execution path, so it runs in numpy off the device.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def contract(lock: dict) -> dict:
    c = lock["numerical_contract"]
    return {
        "encoder_max_abs_err": c["encoder_max_abs_err"],
        "encoder_mean_abs_err": c["encoder_mean_abs_err"],
        "encoder_rel_l2_err": c["encoder_rel_l2_err"],
        "nan_count_allowed": c["nan_count_allowed"],
        "inf_count_allowed": c["inf_count_allowed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text())
    bounds = contract(lock)

    hidden = np.load(args.run / "encoder_hidden.npy").astype(np.float32)
    got_mask = np.load(args.run / "encoder_mask.npy").astype(np.int32)
    golden = np.load(args.capture / "encoder_hidden.npy").astype(np.float32)
    golden_mask = np.load(args.capture / "encoder_mask.npy")

    if hidden.shape != golden.shape:
        raise SystemExit(f"shape {hidden.shape} != golden {golden.shape}")

    diff = np.abs(hidden - golden)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel_l2 = float(np.linalg.norm(hidden - golden) / np.linalg.norm(golden))
    nan = int(np.isnan(hidden).sum())
    inf = int(np.isinf(hidden).sum())
    mask_flat = got_mask.reshape(golden_mask.shape)
    mask_exact = bool(np.array_equal(mask_flat, golden_mask.astype(np.int32)))

    checks = {
        "encoder_max_abs_err": (max_abs, bounds["encoder_max_abs_err"], max_abs <= bounds["encoder_max_abs_err"]),
        "encoder_mean_abs_err": (mean_abs, bounds["encoder_mean_abs_err"], mean_abs <= bounds["encoder_mean_abs_err"]),
        "encoder_rel_l2_err": (rel_l2, bounds["encoder_rel_l2_err"], rel_l2 <= bounds["encoder_rel_l2_err"]),
        "nan_count": (nan, bounds["nan_count_allowed"], nan == bounds["nan_count_allowed"]),
        "inf_count": (inf, bounds["inf_count_allowed"], inf == bounds["inf_count_allowed"]),
    }
    result = {
        "label": args.label,
        "encoder_hidden_shape": list(hidden.shape),
        "encoder_hidden_sha256": hashlib.sha256(
            (args.run / "encoder_hidden.npy").read_bytes()
        ).hexdigest(),
        "golden_encoder_hidden_sha256": hashlib.sha256(
            (args.capture / "encoder_hidden.npy").read_bytes()
        ).hexdigest(),
        "bounds": bounds,
        "measured": {
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
            "rel_l2_err": rel_l2,
            "nan": nan,
            "inf": inf,
        },
        "per_bound": {
            name: {"measured": m, "bound": b, "pass": bool(p)}
            for name, (m, b, p) in checks.items()
        },
        "encoder_mask_exact": mask_exact,
        "encoder_mask_sum": int(mask_flat.sum()),
        "golden_encoder_mask_sum": int(golden_mask.sum()),
        "all_bounds_pass": bool(all(p for _, _, p in checks.values()) and mask_exact),
    }
    args.out.write_text(json.dumps(result, indent=2))

    width = max(len(n) for n in checks)
    print(f"--- {args.label}")
    for name, (m, b, p) in checks.items():
        print(f"  {'PASS' if p else 'FAIL'} {name:<{width}}  {m:.6f}  bound {b}")
    print(f"  {'PASS' if mask_exact else 'FAIL'} encoder_mask bit-exact "
          f"(sum {int(mask_flat.sum())} vs {int(golden_mask.sum())})")
    print(f"  verdict: {'WITHIN CONTRACT' if result['all_bounds_pass'] else 'OUT OF CONTRACT'}")
    return 0 if result["all_bounds_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
