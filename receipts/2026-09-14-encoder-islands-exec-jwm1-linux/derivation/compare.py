#!/usr/bin/env python3
"""Compare each island's device output against the host reference computed from
the exact same staged input bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/var/tmp/IslandsExecJwm1")
DEV = ROOT / "device-out"
REF = ROOT / "stage"

CASES = [
    ("A", "attention_scores_1", "A_attention_scores_1.out.bin", "A_ref_attention_scores_1.bin", (1, 8, 375, 749)),
    ("A", "matmul_0", "A_matmul_0.out.bin", "A_ref_matmul_0.bin", (1, 8, 375, 375)),
    ("B", "attention_mask_9", "B_attention_mask_9.out.bin", "B_ref_attention_mask_9.bin", (1, 8, 375, 375)),
    ("C", "attn_output_1", "C_attn_output_1.out.bin", "C_ref_attn_output_1.bin", (1, 8, 375, 128)),
]


def load(path: Path, shape) -> tuple[np.ndarray, str]:
    raw = path.read_bytes()
    array = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    return array, hashlib.sha256(raw).hexdigest()


def main() -> int:
    report = {}
    for island, name, dev_file, ref_file, shape in CASES:
        got, got_sha = load(DEV / dev_file, shape)
        want, want_sha = load(REF / ref_file, shape)
        gbits = got.view(np.uint16).ravel()
        wbits = want.view(np.uint16).ravel()
        bit_equal = gbits == wbits
        # ±0 equivalence, matching the worker's fp16 value comparator.
        zero_pair = ((gbits & 0x7FFF) == 0) & ((wbits & 0x7FFF) == 0)
        value_equal = bit_equal | zero_pair

        gf = got.astype(np.float32).ravel()
        wf = want.astype(np.float32).ravel()
        finite = np.isfinite(gf) & np.isfinite(wf)
        diff = np.abs(gf[finite] - wf[finite])
        denom = float(np.linalg.norm(wf[finite]))
        entry = {
            "island": island,
            "output": name,
            "elements": int(got.size),
            "device_sha256": got_sha,
            "reference_sha256": want_sha,
            "value_equal": int(value_equal.sum()),
            "mismatch": int((~value_equal).sum()),
            "exact_fp16": bool(value_equal.all()),
            "device_nan": int(np.isnan(gf).sum()),
            "device_inf": int(np.isinf(gf).sum()),
            "reference_inf": int(np.isinf(wf).sum()),
            "device_unique_values": int(np.unique(gbits).size),
            "max_abs_err": float(diff.max()) if diff.size else None,
            "mean_abs_err": float(diff.mean()) if diff.size else None,
            "rel_l2_err": float(np.linalg.norm(gf[finite] - wf[finite]) / denom)
            if denom
            else None,
            "compared_finite_elements": int(finite.sum()),
        }
        bad = np.flatnonzero(~value_equal)
        if bad.size:
            first = int(bad[0])
            entry["first_mismatch"] = {
                "index": first,
                "device_u16": f"0x{int(gbits[first]):04x}",
                "device_fp16": float(gf[first]),
                "reference_u16": f"0x{int(wbits[first]):04x}",
                "reference_fp16": float(wf[first]),
            }
        report[f"{island}:{name}"] = entry

    (ROOT / "compare.json").write_text(json.dumps(report, indent=2))
    for key, entry in report.items():
        print(
            f"{key:28s} exact={entry['exact_fp16']!s:5s} "
            f"equal={entry['value_equal']}/{entry['elements']} "
            f"max_abs={entry['max_abs_err']} rel_l2={entry['rel_l2_err']} "
            f"uniq={entry['device_unique_values']} nan={entry['device_nan']} inf={entry['device_inf']}"
        )
        if "first_mismatch" in entry:
            print("   first:", entry["first_mismatch"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
