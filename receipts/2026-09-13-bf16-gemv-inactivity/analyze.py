#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_ROOT = HERE.parent / "2026-09-13-bf16-longctx-gpu-profile" / "window-final"
PROFILE = PROFILE_ROOT / "profile-longctx.jsonl"
MARKERS = PROFILE_ROOT / "markers-longctx.jsonl"
COMPUTE = PROFILE_ROOT / "compute.h"
GPU_SUMMARY = PROFILE_ROOT / "gpu-summary.json"
SOURCE_VERDICT = PROFILE_ROOT.parent / "verdict.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("gpu_profile_receipt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze():
    parser = load_module(PROFILE_ROOT / "gpu_profile.py")
    names = parser.kernel_names(COMPUTE)
    matvec_id = names.index("MatmulVecBF16")
    multi_id = names.index("MatmulVecMultiBF16")
    rows = [json.loads(line) for line in PROFILE.read_text().splitlines() if line]
    submissions = {row["s"]: row for row in rows if row.get("k") == "s"}
    token_times = [
        row["t"]
        for row in map(json.loads, MARKERS.read_text().splitlines())
        if row.get("p") == "tok"
    ]
    intervals = len(token_times) - 1
    selected = [
        row
        for row in rows
        if row.get("k") == "d"
        and row["e"] == matvec_id
        and token_times[0] <= submissions[row["s"]]["t"] < token_times[-1]
    ]

    interval_counts = []
    pairs = []
    singles = []
    for start, end in zip(token_times, token_times[1:]):
        dispatches = [
            row for row in selected if start <= submissions[row["s"]]["t"] < end
        ]
        interval_counts.append(len(dispatches))
        for _, group in itertools.groupby(dispatches, key=lambda row: tuple(row["b"][0])):
            run = list(group)
            if len(run) == 2:
                pairs.append(run)
            else:
                singles.extend(run)

    def ms_per_token(total_ns):
        return total_ns / intervals / 1e6

    summary = json.loads(GPU_SUMMARY.read_text())
    decode = summary["profiles"]["longctx"]["decode"]
    kernels = decode["kernels"]
    ranked = sorted(
        (
            value["gpu_busy_ms_per_token"],
            name,
            value["dispatches_per_token"],
        )
        for name, value in kernels.items()
    )
    source = json.loads(SOURCE_VERDICT.read_text())
    result = {
        "schema": "bf16-gemv-inactivity/1",
        "execution_model": "openai-codex/gpt-5.6-sol",
        "fallback_model": None,
        "source_commit": source["identity"]["source_commit"],
        "input_provenance": {
            "profile_sha256": sha256(PROFILE),
            "markers_sha256": sha256(MARKERS),
            "compute_h_sha256": sha256(COMPUTE),
            "gpu_summary_sha256": sha256(GPU_SUMMARY),
            "decode_intervals": intervals,
            "dropped_profile_records": summary["profiles"]["longctx"]["records"]["dropped"],
        },
        "measured_bottleneck": {
            "kernel": ranked[-1][1],
            "gpu_busy_ms_per_token": ranked[-1][0],
            "dispatches_per_token": ranked[-1][2],
            "share_of_decode_gpu_busy_percent":
                ranked[-1][0] / decode["gpu_busy_ms_per_token"] * 100.0,
            "next_two_gpu_busy_ms_per_token": [
                {"kernel": name, "value": busy}
                for busy, name, _ in reversed(ranked[-3:-1])
            ],
        },
        "grouped_route_observation": {
            "matmul_vec_dispatches_each_interval": interval_counts,
            "matmul_vec_multi_dispatches_whole_profile": sum(
                row.get("k") == "d" and row.get("e") == multi_id for row in rows
            ),
            "adjacent_same_input_binding_pairs": len(pairs),
            "pairs_per_token": len(pairs) / intervals,
            "pairs_in_same_submission": sum(a["s"] == b["s"] for a, b in pairs),
            "paired_gpu_busy_ms_per_token": ms_per_token(
                sum(row["t1"] - row["t0"] for pair in pairs for row in pair)
            ),
            "singleton_gpu_busy_ms_per_token": ms_per_token(
                sum(row["t1"] - row["t0"] for row in singles)
            ),
            "pair_internal_positive_gap_ms_per_token": ms_per_token(
                sum(max(0, b["t0"] - a["t1"]) for a, b in pairs)
            ),
            "paired_dispatch_record_host_ms_per_token": ms_per_token(
                sum(row["h"] for pair in pairs for row in pair)
            ),
        },
        "interpretation": {
            "finding": "MatmulVecBF16 is the largest measured decode GPU-time category, while the grouped MatmulVecMultiBF16 route emitted zero dispatches.",
            "boundary": "A repeated Vulkan binding proves physical input-buffer reuse, not the array-id equality required by the source matcher. Instrument the planner and record fusion environment gates before changing source.",
            "performance_claim": "No speedup is claimed. The GPU profile is instrumented, and grouped execution was not measured in this workload.",
            "next_measurement": "With hardware ownership restored, record MLX_OMARCHY_FUSED_CHAIN and MLX_OMARCHY_FUSED_GEMV, emit planner candidate/rejection counts, then run an alternating release-control matrix only if the real model produces grouped dispatches.",
        },
    }
    return result


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--verify", type=Path)
    args = arguments.parse_args()
    result = analyze()
    if args.verify:
        expected = json.loads(args.verify.read_text())
        if result != expected:
            raise SystemExit("GEMV_INACTIVITY_VERDICT_MISMATCH")
        print("GEMV_INACTIVITY_VERDICT_VALID")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
