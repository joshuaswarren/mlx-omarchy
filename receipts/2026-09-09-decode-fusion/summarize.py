#!/usr/bin/env python3
"""Assemble verdict.json for receipts/2026-09-09-decode-fusion from the
raw files the M1 window and the local llvmpipe runs left behind.

usage: summarize.py [--decision TEXT]
"""
import argparse
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def suite_summary(path):
    text = path.read_text(errors="replace") if path.exists() else ""
    m_cases = re.search(r"test cases:\s+(\d+)\s+\|\s+(\d+) passed \|\s+(\d+) failed", text)
    m_asserts = re.search(r"assertions:\s+(\d+)\s+\|\s+(\d+) passed \|\s+(\d+) failed", text)
    status = "SUCCESS" if "Status: SUCCESS!" in text else ("FAILURE" if text else "missing")
    return {
        "cases": int(m_cases.group(1)) if m_cases else None,
        "assertions": int(m_asserts.group(1)) if m_asserts else None,
        "failed": int(m_asserts.group(3)) if m_asserts else None,
        "status": status,
    }


def table(side):
    t = json.loads((HERE / f"profile-{side}" / "token10-table.json").read_text())
    per_layer = {}
    for row in t["rows"]:
        if row["layer"] == 1:
            per_layer.setdefault(row["kernel"], 0)
            per_layer[row["kernel"]] += 1
    analysis = (HERE / f"profile-{side}" / "4bit-analysis.txt").read_text()
    grab = lambda key: re.search(key + r": ([0-9.]+)", analysis).group(1)
    log = (HERE / f"profile-{side}" / "4bit.log").read_text()
    inter = re.search(r"median_inter_token_ms=([0-9.]+)", log).group(1)
    return {
        "dispatches_per_token": t["dispatches"],
        "submissions_per_token": len(t["submissions"]),
        "token_interval_profiled": t["token_interval"],
        "gpu_busy_us": t["gpu_busy_us"],
        "gap_us": t["gap_us"],
        "span_us": t["span_us"],
        "by_kernel": t["by_kernel"],
        "layer_1_dispatches": per_layer,
        "layer_1_dispatch_count": sum(per_layer.values()),
        "profile_dispatches_per_decode_interval": float(grab(r"dispatches/decode-interval")),
        "profile_submissions_per_decode_interval": float(grab(r"submissions/decode-interval")),
        "profile_gpu_busy_fraction": analysis.split("GPU busy fraction: ")[1].split("%")[0] + "%",
        "instrumented_median_inter_token_ms": float(inter),
        "version": (HERE / f"profile-{side}" / "version.txt").read_text().strip(),
        "wheel_sha256": (HERE / f"profile-{side}" / "wheel.sha256").read_text().split()[0],
        "table": f"profile-{side}/token10-table.txt",
    }


def legs():
    summary = json.loads((HERE / "legs" / "paired-summary.json").read_text())
    out = {"repetitions": summary["repetitions"], "sides": summary["sides"], "legs": {}}
    all_identical = True
    for row in summary["results"]:
        sides = {}
        for name, entry in row["sides"].items():
            sides[name] = {
                "decode_tok_s_samples": entry["decode_tok_s"]["samples"],
                "decode_tok_s_median": entry["decode_tok_s"]["median"],
                "prefill_tok_s_median": entry["prefill_tok_s"]["median"],
                "digests": entry["digests"],
            }
            for key in entry:
                if key.startswith("decode_over_"):
                    sides[name][key] = round(entry[key], 4)
        digests = {tuple(e["digests"]) for e in row["sides"].values()}
        identical = len(digests) == 1 and all(len(e["digests"]) == 1 for e in row["sides"].values())
        all_identical = all_identical and identical
        out["legs"][row["leg_id"]] = {"sides": sides, "ids_identical": identical,
                                      "native_digest": row["native_digest"]}
    out["ids_identical_all_legs"] = all_identical
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default="")
    args = ap.parse_args()
    fixed = json.loads((HERE / "fixed-projection-results.json").read_text())
    verdict = {
        "topic": "Q4 decode dispatch count: fused q/k/v and gate/up GEMV groups with bias and residual Add epilogues folded into the GEMV store (wave/DecodeFusion)",
        "source_commit": (HERE / "source-commit.txt").read_text().strip(),
        "base_commit": "6ca33e0c (origin/main)",
        "host": "jwm1-linux, Apple M1 (G13G B1), Honeykrisp, linux-asahi 7.1.6, glslc 2026.3; dev box llvmpipe (LLVM 22.1.8) for the local suites",
        "wheels": json.loads((HERE / "wheels.json").read_text()),
        "profile": {
            "method": "m1-model-profile.sh: diagnostics wheel, MLX_OMARCHY_GPU_PROFILE, 32 tokens of the pinned Qwen2.5-0.5B-Instruct-4bit from one prompt, MLX_DISABLE_COMPILE=1; dispatch-table.py lists every dispatch of inter-token interval 10 with its GPU time and the gap before it, grouped by layer (FastRmsNorm boundaries)",
            "before": table("base"),
            "after": table("cand"),
        },
        "per_dispatch_floor": json.loads((HERE / "floor.json").read_text()),
        "fixed_projection_call35": {
            "per_projection_decode_mismatches": {r["projection"]: r["different"] for r in fixed if r["call"] == 35 and "group" not in r},
            "grouped_decode_mismatches": {r["group"] + ":" + r["projection"]: r["different"] for r in fixed if "group" in r},
            "prefill_mismatches_total": sum(r["different"] for r in fixed if r["call"] == 0),
            "criterion": "receipts/2026-09-09-q4-gemv-order/verdict.json: 0 call35 decode mismatches on all 7 projections; the prefill count is the known out-of-scope general-Qmm value",
            "log": "fixed-projection.log",
        },
        "paired_legs": legs(),
        "tests_m1": {suite: suite_summary(HERE / "m1-tests" / f"{suite}.log") for suite in
                     ("omarchy_fused_chain_tests", "omarchy_matmul_family_tests", "omarchy_fast_ops_tests", "omarchy_runtime_tests")},
        "tests_llvmpipe": {suite: suite_summary(HERE / "local-llvmpipe" / f"{suite}.log") for suite in
                           ("omarchy_fused_chain_tests", "omarchy_matmul_family_tests", "omarchy_runtime_tests")},
        "decision": args.decision,
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: verdict[k] for k in ("source_commit", "decision")}, indent=2))
    for leg, row in verdict["paired_legs"]["legs"].items():
        print(leg, {n: (e["decode_tok_s_median"], e.get("decode_over_base")) for n, e in row["sides"].items()}, "ids_identical" if row["ids_identical"] else "IDS DIFFER")


if __name__ == "__main__":
    main()
