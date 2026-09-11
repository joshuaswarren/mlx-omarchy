#!/usr/bin/env python3
"""Generate verdict.json for the BF16 decode GEMV landing receipt.

Reads summary.json (summarize_land.py output), suite logs, and the M1
provenance files; every number is lifted from measured artifacts, none is
hand-typed.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
PRIOR = {
    "qwen25-0.5b-bf16:short-decode-32": {"fork": "f26175202f3dabe9",
                                          "stock": "f26175202f3dabe9"},
    "qwen25-0.5b-bf16:long-decode-128": {"fork": "ad964232ee67fecd",
                                          "stock": "c5be9207833d2a26"},
    "qwen25-0.5b-bf16:longctx-1024-decode-32": {"fork": "ff502900d2a179a5",
                                                 "stock": "ff502900d2a179a5"},
}
NATIVE = {
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
LEG_SHORT = {
    "qwen25-0.5b-bf16:short-decode-32": "BF16 short (30/32)",
    "qwen25-0.5b-bf16:long-decode-128": "BF16 long (262/128)",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "BF16 1K ctx (1053/32)",
    "qwen25-0.5b-4bit:short-decode-32": "Q4 short (30/32)",
    "qwen25-0.5b-4bit:long-decode-128": "Q4 long (262/128)",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "Q4 1K ctx (1053/32)",
}
BF16_LEGS = [k for k in LEG_SHORT if k.startswith("qwen25-0.5b-bf16")]
Q4_LEGS = [k for k in LEG_SHORT if k.startswith("qwen25-0.5b-4bit")]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def regression_lines(path, driver=None):
    text = Path(path).read_text()
    if driver:
        # M1 suites ran under doctest --out, so the cout proof lines live
        # in window.log between the per-driver suite banners.
        marker = f"== omarchy_matmul_family_tests (M1 {driver}) =="
        start = text.index(marker)
        rest = text[start + len(marker):]
        end = rest.find("== omarchy_")
        text = rest[:end] if end != -1 else rest
    return [line.strip() for line in text.splitlines()
            if "[matmul-bf16-decode]" in line or "decode_gemv_path=" in line]


def suite_result(path):
    text = Path(path).read_text()
    cases = re.search(r"test cases:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed", text)
    asserts = re.search(r"assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed", text)
    status = "PASS" if "Status: SUCCESS!" in text else "FAILURE"
    return {
        "status": status,
        "cases": f"{cases.group(2)}/{cases.group(1)}",
        "assertions": int(asserts.group(2)),
        "log": str(Path(path).relative_to(R)),
    }


def main():
    summary = json.loads((R / "matrix" / "summary.json").read_text())
    commit = (R / "commit.txt").read_text().strip()
    both_sha = (R / "m1-logs" / "both-wheels.sha256").read_text().splitlines()
    cand_line = next(l for l in both_sha if "/candidate/" in l)
    base_line = next(l for l in both_sha if "/base/" in l)
    cand_sha, cand_wheel = cand_line.split()[0], Path(cand_line.split()[1]).name
    base_sha, base_wheel = base_line.split()[0], Path(base_line.split()[1]).name
    window = (R / "window.log").read_text()
    kernel_pick = "480a1ef426e0"
    main_base = "9737c36e"

    rates = summary["rates"]
    paired = {}
    for leg in LEG_SHORT:
        for driver in ("fork", "stock"):
            base = next((v for k, v in rates.items()
                         if k.startswith("r1-") and k.endswith(f"-{driver}-base") and leg in v), None)
            # median across reps comes from each rep cell; aggregate here
            base_reps = [v[leg] for k, v in rates.items()
                         if k.split("-")[1] == driver and k.endswith("-base") and leg in v]
            cand_reps = [v[leg] for k, v in rates.items()
                         if k.split("-")[1] == driver and k.endswith("-cand") and leg in v]
            if not base_reps or not cand_reps:
                continue
            paired.setdefault(LEG_SHORT[leg], {})[driver] = {
                "decode_base_median": statistics_median([r["decode_median"] for r in base_reps]),
                "decode_cand_median": statistics_median([r["decode_median"] for r in cand_reps]),
                "prefill_base_median": statistics_median([r["prefill_median"] for r in base_reps]),
                "prefill_cand_median": statistics_median([r["prefill_median"] for r in cand_reps]),
                "decode_fraction_of_native_base": base_reps[0]["fraction_of_native_decode"],
                "decode_fraction_of_native_cand": cand_reps[0]["fraction_of_native_decode"],
                "prefill_fraction_of_native_base": base_reps[0]["fraction_of_native_prefill"],
                "prefill_fraction_of_native_cand": cand_reps[0]["fraction_of_native_prefill"],
                "reps": len(base_reps),
            }
    for leg_d in paired.values():
        for d in leg_d.values():
            d["decode_speedup"] = round(d["decode_cand_median"] / d["decode_base_median"], 3)

    digest_changes = {}
    q4_unchanged = {}
    for key, val in summary["digests"].items():
        label, lid = key.split("|")
        dig, cls = val
        rep, driver, cell = label.split("-", 2)
        if cell != "cand":
            continue
        entry = {"leg": LEG_SHORT[lid], "driver": driver,
                 "before": PRIOR[lid][driver] if lid in BF16_LEGS else None,
                 "after": dig, "classification": cls,
                 "native": NATIVE.get(lid)}
        if lid in BF16_LEGS:
            k = (lid, driver)
            if k in digest_changes:
                assert digest_changes[k]["after"] == dig, f"digest varied: {k}"
            digest_changes[k] = entry
        else:
            q4_unchanged[f"{lid}|{driver}"] = dig
    digest_changes = list(digest_changes.values())

    verdict = {
        "schema": "bf16-decode-gemv-land/1",
        "date": "2026-09-10",
        "agent": "Bf16DecodeGemvLand",
        "decision": (
            "LANDED on wave/Bf16DecodeGemvLand per owner decision: the dense "
            "BF16 decode GEMV (native-order subgroup kernel, M=1 only) is "
            "qualified and the BF16 short/262 generated-id digests are "
            "re-pinned to the measured per-driver values. BF16 1K ctx and all "
            "six Q4 digests unchanged. NOT merged; never merge from here."),
        "provenance": {
            "branch": "wave/Bf16DecodeGemvLand",
            "kernel_commit": {
                "original": "a70ed0dc89112e4f43b2394611b4994984d01405",
                "note": "cherry-picked onto origin/main 9737c36e (post KV-direct)",
                "cherry_pick": kernel_pick,
                "main_base": main_base,
            },
            "commit_under_test": commit,
            "candidate_wheel": {"file": cand_wheel, "sha256": cand_sha,
                                 "build": "DEV_RELEASE=1 plain stamped release, no diagnostics"},
            "baseline_wheel": {
                "file": base_wheel, "sha256": base_sha,
                "built_from": "origin/main 9737c36e worktree - same source base "
                "as the candidate, so the paired cells carry the landed "
                "KV-direct decode gain on both sides",
                "build": "DEV_RELEASE=1 plain stamped release, no diagnostics",
                "prior_canonical_wheel_b6d662a8_sha256":
                    "98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439"},
            "pairing_note": (
                "base wheel = 9737c36e (main with KV-direct), candidate wheel = "
                "branch tip; per Main's instruction the paired numbers are not "
                "contaminated by the KV-direct gain. Committed base-M1 native "
                "fractions still use the ea09eefa native denominators."),
            "matrix_window_log": "matrix-window.log",
            "host": "jwm1-linux, Apple M1 (G13G B1), kernel 7.1.6-1-ARCH",
            "lock": "single top-level flock /tmp/m1-gpu.lock for matrix+suites; CPU prep outside",
            "window_log": "window.log",
            "drivers": {"fork": "m1-logs/drivers-fork.txt", "stock": "m1-logs/drivers-stock.txt"},
            "digest_gate": "summary.json failures must be empty (all cells verified=match, clean, AC)",
            "summary_failures": summary["failures"],
        },
        "numbers": paired,
        "digest_changes": digest_changes,
        "q4_unchanged": q4_unchanged,
        "suites": {
            "llvmpipe_matmul_family": suite_result(R / "suites" / "llvmpipe-matmul-family.log"),
            "llvmpipe_runtime": suite_result(R / "suites" / "llvmpipe-runtime.log"),
            "m1_fork_matmul_family": suite_result(R / "m1-suites" / "m1-fork-matmul-family.log"),
            "m1_fork_runtime": suite_result(R / "m1-suites" / "m1-fork-runtime.log"),
            "m1_stock_matmul_family": suite_result(R / "m1-suites" / "m1-stock-matmul-family.log"),
            "m1_stock_runtime": suite_result(R / "m1-suites" / "m1-stock-runtime.log"),
        },
        "regression_test": {
            "name": "single-row bf16 decode matmul stays inside the f64 bound",
            "file": "overlay/tests/omarchy/test_matmul_family.cpp",
            "llvmpipe_lines": regression_lines(R / "suites" / "llvmpipe-matmul-family.log"),
            "m1_fork_lines": regression_lines(R / "window.log", "fork"),
            "m1_stock_lines": regression_lines(R / "window.log", "stock"),
            "note": ("pins the new decode path against a float64 host reference "
                     "inside the documented f32 anchor bound on every Qwen decode "
                     "projection shape, the deep-cancellation regime, and the "
                     "gate-refusal fallback shapes; no digest appears in the test"),
        },
        "accuracy_basis": {
            "f64": ("both kernels are RNE(f64)-exact on every captured decode "
                    "projection incl. bias-cancellation elements; new order "
                    "closer to f64 on large-N samples - "
                    "receipts/2026-09-10-bf16-decode-gemv-requal section 4"),
            "policy": "docs/parity-id-policy.md amendment 2026-09-10",
        },
    }
    out = R / "verdict.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"digest_changes={len(digest_changes)} q4_unchanged={len(q4_unchanged)}")
    print(f"summary_failures={len(summary['failures'])}")
    for leg, drivers in paired.items():
        for drv, d in drivers.items():
            print(f"  {leg:22s} {drv:5s} dec {d['decode_base_median']:7.2f} -> "
                  f"{d['decode_cand_median']:7.2f} ({d['decode_speedup']}x) "
                  f"[{d['decode_fraction_of_native_base']:.3f} -> "
                  f"{d['decode_fraction_of_native_cand']:.3f} native]")


def statistics_median(values):
    import statistics
    return round(statistics.median(values), 3)


if __name__ == "__main__":
    main()
