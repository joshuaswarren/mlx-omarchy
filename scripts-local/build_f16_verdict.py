#!/usr/bin/env python3
"""Build receipts/2026-09-11-qmm-f16-operand/verdict.json from the
measured data files. Run on jwm1 in ~/src/mlx-QmmF16Operand."""
import json
from pathlib import Path

R = Path("receipts/2026-09-11-qmm-f16-operand")

shapes_c = {f"{r['m']}x{r['k']}x{r['n']}": r
            for r in json.load(open("/tmp/f16-shapes-cand-coopmat.json"))}
shapes_b = {f"{r['m']}x{r['k']}x{r['n']}": r
            for r in json.load(open("/tmp/f16-shapes-base-coopmat.json"))}
shapes_t = {f"{r['m']}x{r['k']}x{r['n']}": r
            for r in json.load(open("/tmp/f16-shapes-cand-tile.json"))}
perf = {}
for key in shapes_c:
    c, b, t = shapes_c[key], shapes_b[key], shapes_t[key]
    perf[key] = {
        "f16_staged_gflop_s": round(c["gflop_s"], 1),
        "f32_staged_gflop_s": round(b["gflop_s"], 1),
        "f16_over_f32": round(c["gflop_s"] / b["gflop_s"], 3),
        "tile_kernel_gflop_s": round(t["gflop_s"], 1),
        "f16_digest": c["digest"], "f32_digest": b["digest"]}

acc_c = json.load(open("/tmp/f16-accuracy-cand.json"))
acc_b = json.load(open("/tmp/f16-accuracy-base.json"))
acc = {}
for a, bl in zip(acc_c, acc_b):
    acc[a["layer"]] = {
        "f16_staged": {
            "max_abs_err_vs_f64": a["max_abs_err_vs_ref"],
            "mean_abs_err_vs_f64": a["mean_abs_err_vs_ref"],
            "exact_frac_vs_w16_metal_model": a["exact_frac_vs_metal_model"],
            "f16_flips_vs_rne16_f64_ref": a["f16_flips_vs_rne16_ref"],
            "outputs": a["outputs"]},
        "f32_staged": {
            "max_abs_err_vs_f64": bl["max_abs_err_vs_ref"],
            "mean_abs_err_vs_f64": bl["mean_abs_err_vs_ref"],
            "exact_frac_vs_w16_metal_model": bl["exact_frac_vs_metal_model"],
            "f16_flips_vs_rne16_f64_ref": bl["f16_flips_vs_rne16_ref"],
            "outputs": bl["outputs"]}}

legs = json.load(open("/tmp/f16-legs/paired-summary.json"))
native_tok_s = {"short": 294.1, "long": 1213.0, "longctx": 1840.9}
native_digest = {"short": "7fd25a869ff21678",
                 "long": "254d73fd93164b98",
                 "longctx": "7da83f06ec9f001d"}
linux_old = {"short": "7fd25a869ff21678",
             "long": "4cc08910089477fd",
             "longctx": "7da83f06ec9f001d"}
leg_map = {"short": "qwen25-0.5b-4bit:short-decode-32",
           "long": "qwen25-0.5b-4bit:long-decode-128",
           "longctx": "qwen25-0.5b-4bit:longctx-1024-decode-32"}
digests = {}
for short, lid in leg_map.items():
    row = next(r for r in legs["results"] if r["leg_id"] == lid)
    e_b, e_c = row["sides"]["base"], row["sides"]["cand"]
    cd = e_c["digests"][0]
    cls = ("native" if cd == native_digest[short]
           else "current_linux" if cd == linux_old[short]
           else "third_value")
    nb = e_b["prefill_tok_s"]["median"]
    nc = e_c["prefill_tok_s"]["median"]
    digests[short] = {
        "leg_id": lid,
        "native_digest": native_digest[short],
        "base_digest": e_b["digests"][0],
        "cand_digest": cd,
        "cand_digest_class": cls,
        "base_prefill_median_tok_s": round(nb, 1),
        "cand_prefill_median_tok_s": round(nc, 1),
        "cand_over_base": round(nc / nb, 4),
        "base_frac_of_native": round(nb / native_tok_s[short], 4),
        "cand_frac_of_native": round(nc / native_tok_s[short], 4),
        "base_decode_median_tok_s": round(e_b["decode_tok_s"]["median"], 1),
        "cand_decode_median_tok_s": round(e_c["decode_tok_s"]["median"], 1),
        "digest_stable_across_reps":
            len(e_c["digests"]) == 1 and len(e_b["digests"]) == 1}

verdict = {
    "schema": "mlx-omarchy/qmm-f16-operand/1",
    "date": "2026-09-11",
    "agent": "QmmF16Operand",
    "decision": (
        "DO NOT LAND on this evidence: the f16-operand staging path moves "
        "the Q4 generated-id digests AWAY from native macOS (the 1K leg "
        "regressed from native-matching to a third value; the long leg "
        "moved from the current Linux value to another third value; short "
        "stays native) while buying only +3.5-4.3% paired prefill and "
        "1.00-1.07x kernel GFLOP/s. Branch wave/QmmF16Operand carries the "
        "kernel, harnesses, and this receipt only. Never merged; the pin "
        "decision stays with main."),
    "question": (
        "Does staging both Q4 coopmat prefill operands as f16 with a "
        "padded pitch - dequantizing packed weight words straight into "
        "the f16 tile the way upstream Metal does - recover native "
        "performance, and do the generated-id digests move toward native "
        "macOS?"),
    "base": {
        "branch": "wave/QmmF16Operand off origin/main 01cbac23",
        "kernel_commit": "2cd86ff5",
        "harness_commit": "05711668",
        "cand_wheel": ("mlx_omarchy-0.32.2.dev202609111013+2cd86ff-cp314-"
                       "cp314-linux_aarch64.whl"),
        "cand_wheel_sha256": ("6ca093d12e3ffc04ba19a10357cbcdaf4114f126"
                              "fe179598fcce46bde64d8142"),
        "base_wheel": ("mlx_omarchy-0.32.2.dev202609111016+01cbac2-cp314-"
                       "cp314-linux_aarch64.whl"),
        "base_wheel_sha256": ("f50348decd55c71d89f5da6d40aaa0c6bac8379c28"
                              "ffdadf08177ca2603cf92e"),
        "provenance": ("mlx_provenance.py verified=match for both wheels; "
                       "stamps asserted different by paired-legs.py"),
    },
    "kernel_change": {
        "what": (
            "shaders/qmm_coopmat.comp now stages x as bit-exact f16 and "
            "dequantizes packed w words straight into an f16 tile with "
            "upstream Metal's f32 s*q+bias arithmetic and a single f16 "
            "rounding at the store (quantized.h dequantize()); the "
            "cooperative multiply takes f16 A/B with f32 accumulate "
            "(driver-advertised 8x8x8 FLOAT16/FLOAT16/FLOAT32 combo); "
            "per-output ascending-k accumulation unchanged"),
        "staging": (
            "x_s 32x24 f16 (16+8 pad), w_s 16x40 f16 (32+8, upstream's "
            "pad), acc_s 4x8x8 f32 drain; 2.75 KiB staging + 1 KiB drain "
            "= 3840 B, below the shipped 4 KiB cadence and far below the "
            ">=8 KiB occupancy cliff measured on the double-buffer and "
            "chunk arms"),
        "host_gate": (
            "kQmmCoopmatSharedBytes updated; dispatch gate otherwise "
            "unchanged (cooperative_matrix_f32_8, subgroup 32, operand "
            "alignment, !MLX_OMARCHY_NO_COOPMAT)"),
    },
    "driver": {
        "installed": ("mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 "
                      "(pacman -Q verified before digest legs)"),
        "cmshape_probes": (
            "8x8x8 f16f16->f32 and f16f16->f16 PASS bit-exact (0 "
            "mismatches) on the installed driver; it advertises 8 and 16 "
            "node shapes x {f32, f16->f32, f16} combos"),
        "hardware": (
            "jwm1-linux, Apple M1 (G13G B1); every GPU access under one "
            "top-level flock /tmp/m1-gpu.lock (windows A/A2/A3/B announced "
            "in hub); wall-anchored timing only"),
    },
    "performance_kernel_shapes": {
        "note": (
            "GFLOP/s, median of 30 wall-anchored evals after 4 warmups, "
            "fresh expression per eval, f16 scales/biases, seeded N(0,1) "
            "f16 x; tile column is the NO_COOPMAT cross-check proving the "
            "coopmat route carried the fast numbers"),
        "cells": perf,
    },
    "performance_paired_legs": {
        "method": (
            "paired-legs.py, 3 repetitions, sides alternate order each "
            "rep, per-side single-rep run-baseline (6 measured legs incl. "
            "bf16 canary), provenance and contention asserts green on "
            "every rep"),
        "legs": digests,
        "bf16_canary": (
            "all three bf16 legs: cand digests identical to base digests "
            "on every rep (the f16 kernel does not touch bf16 paths)"),
    },
    "accuracy": {
        "method": (
            "f64 oracle on the real layer-5 checkpoint tensors (gate_up "
            "synthesized by concatenating gate_proj+up_proj exactly as "
            "the runtime fusion does), m=1053 seeded f16 activations; "
            "refw16 = the Metal-arithmetic model: f32 s*q+b rounded once "
            "to f16, then an exact-product f64 matmul"),
        "per_projection": acc,
        "reading": (
            "the f16-staged kernel reproduces the Metal-arithmetic model "
            "on 99.5-99.8% of outputs per projection (residual is "
            "f32-accumulation-order noise against the f64 oracle); the "
            "f32-staged kernel only coincides with it on ~71% since its "
            "weights are unrounded. Cost vs the pure-f64 oracle: max abs "
            "err grows 6-7% (q 0.00194 -> 0.00204), mean abs err roughly "
            "unchanged; both stay far below an f16 ULP at these "
            "magnitudes"),
    },
    "digest_verdict": (
        "AWAY from native. short(30): cand == native == current Linux "
        "(7fd25a869ff21678). long(262): cand b3692313ef82b28a is a third "
        "value - not native 254d73fd93164b98, not the current Linux "
        "4cc08910089477fd; 35/128 generated tokens flip vs base, first "
        "at token 93. 1K(1053): cand 31267e7ed4c6d0dc is a third value - "
        "this leg REGRESSED from native-matching (7da83f06ec9f001d) to "
        "non-matching; 4/32 tokens flip, first at token 28. All digests "
        "bit-stable across 3 reps on both sides. Combined with the "
        "99.5-99.8% kernel-level match to the w16-rounded oracle, this "
        "REFUTES the hypothesis that native Metal's long-prompt digest "
        "advantage is explained by f16 weight rounding alone: a kernel "
        "that faithfully implements that arithmetic moves the 1K leg off "
        "native, so native's remaining difference lives elsewhere "
        "(dequant contraction, accumulation schedule, or another "
        "data-path detail)."),
    "policy": (
        "No landing. wave/QmmF16Operand only (kernel commit 2cd86ff5, "
        "harness 05711668, receipt in receipts/2026-09-11-qmm-f16-operand/"
        "). Never merge; never re-pin digests from this branch. llvmpipe "
        "and stock Mesa never reach the route (gate unchanged)."),
    "logs": (
        "logs/ (windowA/A2/A3B logs, wheel builds), data/ (shapes + "
        "accuracy JSONs, cmshape probes), legs/ (paired-summary, per-rep "
        "side summaries + ids, token-divergence.json)"),
}
(R / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print("VERDICT_WRITTEN")
print(json.dumps(digests, indent=1))
