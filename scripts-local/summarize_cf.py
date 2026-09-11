#!/usr/bin/env python3
"""Summarize window B matrix cells: medians, digest gate, verdict JSON.

Usage: summarize_cf.py <benchq>/qmm-coop-bench
Reads matrix/<label>/matrix.json for warmup + r{1,2,3}-{fork,stock}-
{base,cand}, extracts the Q4 workload prefill rates and generated-id
digests, checks the six canonical Q4 digest cells (3 legs x 2 drivers,
candidate wheel), and writes matrix-verdict.json with paired medians.
"""
import json
import statistics
import sys
from pathlib import Path

CANONICAL_Q4 = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
}


def legs(path):
    out = {}
    data = json.load(open(path))
    for leg in data.get("legs", []):
        lid = leg.get("leg_id", "")
        if lid not in CANONICAL_Q4:
            continue
        m = leg.get("metrics", {})
        prefill_s = m.get("prefill_s")
        prompt_tokens = leg.get("prompt_tokens")
        out[lid] = {
            "status": leg.get("status"),
            "prefill_tok_s": (prompt_tokens / prefill_s)
            if prefill_s and prompt_tokens else None,
            "digest": m.get("generated_ids_sha256_16"),
        }
    return out


def main(root):
    root = Path(root)
    labels = {}
    for d in sorted((root / "matrix").iterdir()):
        f = d / "matrix.json"
        if f.exists():
            labels[d.name] = legs(f)
    reps = {}
    for name, lg in labels.items():
        if not (name.startswith("r") and "-" in name):
            continue
        rep, rest = name[1:].split("-", 1)
        driver, cell = rest.split("-")
        for key, v in lg.items():
            reps.setdefault((driver, cell, key), []).append(v)
    summary = {}
    failures = []
    for (driver, cell, key), rows in sorted(reps.items()):
        pre = [r["prefill_tok_s"] for r in rows if r["prefill_tok_s"]]
        digs = {r["digest"] for r in rows}
        entry = {
            "reps": len(pre),
            "prefill_median_tok_s": round(statistics.median(pre), 3) if pre else None,
            "digests": sorted(d for d in digs if d),
        }
        if cell == "cand":
            expected = CANONICAL_Q4[key]
            ok = len(rows) == 3 and digs == {expected}
            entry["digest_gate"] = "PASS" if ok else f"FAIL expected {expected}"
            if not ok:
                failures.append(f"{driver} {key}: {sorted(digs)} != [{expected}]")
        summary[f"{driver}/{cell}/{key}"] = entry
    verdict = {
        "schema": "mlx-omarchy/qmm-coop-datapath-matrix/1",
        "cells": summary,
        "digest_gate": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    json.dump(verdict, open(root / "matrix-verdict.json", "w"), indent=1)
    print("MATRIX-GATE", verdict["digest_gate"])
    for k, v in summary.items():
        print(k, v.get("prefill_median_tok_s"), v.get("digest_gate", ""), sep="  ")


if __name__ == "__main__":
    main(sys.argv[1])
