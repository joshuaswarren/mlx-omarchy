#!/usr/bin/env python3
"""Characterize the island differences: ULP distance for the matmul islands,
positional structure for the select island."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/var/tmp/IslandsExecJwm1")
DEV = ROOT / "device-out"
REF = ROOT / "stage"


def ordered(bits: np.ndarray) -> np.ndarray:
    """Monotone integer key for fp16 bit patterns, so |a-b| is an ULP distance."""
    signed = bits.astype(np.int32)
    negative = (signed & 0x8000) != 0
    return np.where(negative, 0x8000 - (signed & 0x7FFF), signed + 0x8000)


def ulp_report(dev_file: str, ref_file: str, shape) -> dict:
    got = np.frombuffer((DEV / dev_file).read_bytes(), dtype=np.float16).reshape(shape)
    want = np.frombuffer((REF / ref_file).read_bytes(), dtype=np.float16).reshape(shape)
    distance = np.abs(ordered(got.view(np.uint16).ravel()) - ordered(want.view(np.uint16).ravel()))
    counts = np.bincount(np.minimum(distance, 8))
    return {
        "elements": int(got.size),
        "max_ulp": int(distance.max()),
        "ulp_histogram": {str(i): int(n) for i, n in enumerate(counts) if n},
        "all_within_1_ulp": bool(distance.max() <= 1),
    }


def select_report() -> dict:
    shape = (1, 8, 375, 375)
    got = np.frombuffer((DEV / "B_attention_mask_9.out.bin").read_bytes(), dtype=np.float16).reshape(shape)
    want = np.frombuffer((REF / "B_ref_attention_mask_9.bin").read_bytes(), dtype=np.float16).reshape(shape)
    bad = np.asarray(got.view(np.uint16) != want.view(np.uint16))[0]
    per_head = bad.sum(axis=(1, 2))
    rows = np.flatnonzero(bad[0].any(axis=1))
    per_row = bad[0].sum(axis=1)
    cols = np.flatnonzero(bad[0].any(axis=0))
    heads_identical = bool(all(np.array_equal(bad[0], bad[h]) for h in range(8)))
    # Does the device ever write the -inf fill (cond is all false here)?
    inf_count = int(np.isinf(got.astype(np.float32)).sum())
    return {
        "cond_true_elements": 0,
        "mismatch_total": int(bad.sum()),
        "per_head": [int(n) for n in per_head],
        "heads_identical_mask": heads_identical,
        "rows_touched": [int(r) for r in rows],
        "mismatches_per_touched_row": {int(r): int(per_row[r]) for r in rows},
        "first_column": int(cols[0]) if cols.size else None,
        "last_column": int(cols[-1]) if cols.size else None,
        "device_inf_written": inf_count,
        "device_min": float(got.astype(np.float32).min()),
        "device_max": float(got.astype(np.float32).max()),
    }


report = {
    "A:attention_scores_1": ulp_report(
        "A_attention_scores_1.out.bin", "A_ref_attention_scores_1.bin", (1, 8, 375, 749)
    ),
    "A:matmul_0": ulp_report("A_matmul_0.out.bin", "A_ref_matmul_0.bin", (1, 8, 375, 375)),
    "C:attn_output_1": ulp_report(
        "C_attn_output_1.out.bin", "C_ref_attn_output_1.bin", (1, 8, 375, 128)
    ),
    "B:attention_mask_9": select_report(),
}
(ROOT / "analyze.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
