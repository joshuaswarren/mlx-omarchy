#!/usr/bin/env python3
"""Assemble the jwm1 encoder-islands execution receipt from the real artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/var/tmp/IslandsExecJwm1")
DEV = ROOT / "device-out"
WT = Path.home() / ".config/superpowers/worktrees/mlx-omarchy/EncoderSplitPlan"
CAPTURE = Path(
    "~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
    "20260912T154759Z-librispeech/ane"
).expanduser()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(tag: str) -> dict:
    out = {}
    for line in (DEV / f"state-{tag}.txt").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def island(tag: str) -> dict:
    stdout = (DEV / f"{tag}.stdout").read_text().splitlines()
    stderr = [line for line in (DEV / f"{tag}.stderr").read_text().splitlines() if line]
    fields = {}
    for line in stdout:
        if line.startswith("worker status="):
            for part in line.split():
                key, _, value = part.partition("=")
                fields[key] = value
    return {
        "exit": int((DEV / f"{tag}.exit").read_text().strip()),
        "started_at": (DEV / f"{tag}.started_at").read_text().strip(),
        "finished_at": (DEV / f"{tag}.finished_at").read_text().strip(),
        "worker_status": int(fields.get("status", -1)),
        "iterations_completed": int(fields.get("iterations", -1)),
        "released_programs": int(fields.get("released", -1)),
        "elapsed_ms": int(fields.get("elapsed_ms", -1)),
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "errno_110": False,
        "retried": False,
    }


compare = json.loads((ROOT / "compare.json").read_text())
analyze = json.loads((ROOT / "analyze.json").read_text())
staged = json.loads((ROOT / "stage-manifest.json").read_text())
validate = json.loads((ROOT / "validate-full.json").read_text())

head = subprocess.run(
    ["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()

bundles = {}
for name, tds, graph in (
    ("island-attn-a-kt", 416, "85325f3059a345dee7c62c53a0072b2e869c0c8116385c992e8963587714ff88"),
    ("island-select-8head", 5, "fd9e6c5b64f8b2c58fa2067799589ba0d32dfc24361b6612c9ef94d9e1a6422e"),
    ("island-pv", 208, "ed03818974f7e2728daff0c839fcd4486a87282515347e68db7d2219605905c3"),
):
    directory = ROOT / "ship" / "bundles" / name
    bundles[name] = {
        "task_descriptors": tds,
        "graph_hash": graph,
        "manifest_sha256": sha(directory / "manifest.json"),
        "anec_sha256": {
            path.name: sha(path) for path in sorted(directory.glob("*.anec"))
        },
    }

receipt = {
    "schema": "mlx-omarchy.encoder-islands-exec.v1",
    "date": "2026-09-14",
    "host": "jwm1-linux",
    "result": "ALL_THREE_ISLANDS_EXECUTED_ON_HARDWARE",
    "summary": {
        "islands_submitted": 3,
        "programs_dispatched": 4,
        "task_descriptors_dispatched": 629,
        "worker_completions": 3,
        "timeouts": 0,
        "errno_110": 0,
        "retries": 0,
        "device_live_at_end": True,
        "islands_numerically_accepted": ["A", "C"],
        "islands_numerically_rejected": ["B"],
    },
    "resolved_model": {
        "assigned_route": "openai (mlx-openai-deep agent type)",
        "actual": "anthropic/claude-opus-5 (session-reported identity)",
        "fallback": True,
        "note": (
            "The assignment named an OpenAI route; this session reports an Anthropic "
            "model. The routing discrepancy is reported to the parent and recorded here "
            "rather than hidden behind the usual receipt line, matching the disclosure "
            "in receipts/2026-09-14-encoder-split-plan.md."
        ),
    },
    "pins": {
        "worktree": f"mlx-omarchy receipts/2026-09-14-encoder-split-plan {head}",
        "plan": "receipts/2026-09-14-encoder-split-plan.md",
        "worker_binary": "/var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker",
        "worker_sha256": "762dd1de868b52e4d6ffa30e18d842d9341db463b06aef2355f18fb1eb248929",
        "worker_overlay_commit": "15ecc733ee91327dbf7887eac21ef14bebe824ce",
        "worker_source_identity": (
            "the five ane/worker sources this binary was built from hash identically to "
            "this worktree, so no rebuild was needed"
        ),
        "overlay_source_sha256": {
            "manifest.cpp": "fee93eae167c052a7f75ea130e03b36265d23fa34df1109e1765a76e1057abc2",
            "bundle.cpp": "7eb7130f9053e8345c1e44a309a10c527957589cea80b12dc35b7f4eb57a7264",
            "worker_libane.cpp": "710ded4edcb89149603d1d75e954219841925d7066b4da1bb24b98f98e795da0",
            "tile_layout.h": "c88b4d7a0c844a084a5a8ea251403daae5a6c783d2d5fa2574275f278b493f22",
            "mlx-omarchy-ane-worker/main.cpp": "40183118dc032c18f1ec86d17384f211536ac092431dcb1e763ec00444a3fa04",
        },
        "libane": "/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so",
        "libane_sha256": "1ab9d95debcc8b5fee3b6653dfce0b50412bc7efef43c2d2167dc83ce270ca49",
        "libane_source_commit": "f261a6cb537aca62f267ad3d01beda0d6877544c",
        "encoder": "mweinbach1/parakeet-tdt-0.6b-v3-coreml revision b650695c2322ee5281dff48d7345b2f3a58ff018",
        "kernel": "7.1.6-1-1-ARCH",
        "run_directory": "/var/tmp/jwm1-encoder-islands",
    },
    "bundles": bundles,
    "tensor_provenance": {
        "claim": (
            "The staged island inputs are the real layer-0 encoder tensors, derived from "
            "the authenticated golden capture. They are not synthetic fills."
        ),
        "why_derivation_was_needed": (
            "The golden capture stores only the encoder function inputs and outputs, not "
            "per-layer intermediates, so q_v / q_scaled / k_headsT / matrix_bd_5 / cond / "
            "probs / v_heads had to be recomputed from the capture inputs through the "
            "pinned encoder graph."
        ),
        "capture": str(CAPTURE),
        "capture_sha256": {
            name: sha(CAPTURE / name)
            for name in (
                "encoder_input_features.npy",
                "encoder_input_mask.npy",
                "encoder_hidden.npy",
                "encoder_mask.npy",
            )
        },
        "encoder_mil": {
            "path": "/var/tmp/IslandsExecJwm1/encoder-source/model.mil",
            "sha256": sha(ROOT / "encoder-source" / "model.mil"),
            "produced_by": "overlay/tools/coreml/mil_adapter.py emit (pinned reference lock)",
        },
        "evaluator": {
            "milrun.py": sha(ROOT / "milrun.py"),
            "validate_full.py": sha(ROOT / "validate_full.py"),
            "stage.py": sha(ROOT / "stage.py"),
            "compare.py": sha(ROOT / "compare.py"),
            "analyze.py": sha(ROOT / "analyze.py"),
            "run_islands.sh": sha(ROOT / "run_islands.sh"),
        },
        "full_encoder_validation": validate,
        "validation_verdict": (
            "The same evaluator run end to end over all 3351 encoder ops reproduces the "
            "authenticated ANE capture inside every frozen tolerance in the reference "
            "lock, and encoder_mask is bit-exact. That is the proof the layer-0 tensors "
            "the islands bind are the real encoder tensors."
        ),
        "staged": staged,
    },
    "device_state": {
        "pre": state("pre"),
        "post_B": state("post-B"),
        "post_C": state("post-C"),
        "post": state("post"),
        "boot_id_unchanged": True,
        "module_never_unloaded": True,
        "quarantine_bytes_after": 0,
        "workers_after": 0,
        "dmesg_tm_execution_failed": 0,
        "dmesg_errno_110": 0,
        "dmesg_tail": "/var/tmp/jwm1-encoder-islands/out/dmesg-tail.txt",
    },
    "islands": {
        "A": {
            "bundle": "parakeet-encoder-island-attn-a-kt",
            "programs": 2,
            "task_descriptors": 416,
            "submit": island("A"),
            "outputs": {
                "attention_scores_1": compare["A:attention_scores_1"],
                "matmul_0": compare["A:matmul_0"],
            },
            "difference_structure": {
                "attention_scores_1": analyze["A:attention_scores_1"],
                "matmul_0": analyze["A:matmul_0"],
            },
            "verdict": "ACCEPTED_RELATIVE_L2",
            "verdict_detail": (
                "Both 208-TD batched matmuls completed and are numerically correct. "
                "attention_scores_1: 2239419/2247000 bit-exact, rel_l2 1.09e-05, max_abs "
                "0.25 on a tensor spanning +/-2000. matmul_0: 960933/1125000 bit-exact, "
                "rel_l2 1.13e-04, max_abs 0.0039. On every element with |reference| > 1 "
                "the worst relative error is 2^-10, exactly one fp16 ULP; the larger ULP "
                "distances all sit at near-zero magnitudes. That is an accumulation-order "
                "difference against an fp32-accumulate host reference, which is why the "
                "parity harness classifies matmul as relative_l2 and not fp16_value_exact. "
                "Both are three to four orders of magnitude inside the lock bound of 0.1."
            ),
        },
        "B": {
            "bundle": "parakeet-encoder-island-select-8head",
            "programs": 1,
            "task_descriptors": 5,
            "submit": island("B"),
            "outputs": {"attention_mask_9": compare["B:attention_mask_9"]},
            "difference_structure": {"attention_mask_9": analyze["B:attention_mask_9"]},
            "verdict": "REJECTED_VALUE_MISMATCH",
            "verdict_detail": (
                "The 5-TD select completed in 53 ms and the worker's own --expect "
                "comparator refused it: 'output attention_mask_9 does not match the "
                "expected fp16 values'. 1114272/1125000 lanes are bit-exact; 10728 are "
                "wrong. The wrong lanes are exactly 1341 per head, the per-head mismatch "
                "mask is identical across all 8 heads, and every one of them lives in H "
                "rows 0-8 with no mismatch at H >= 9. That reproduces the already-named "
                "first-8-row L2 bool-tile signature from "
                "receipts/2026-09-14-select-tile-edge.md (its per-row leftover was 167, "
                "167, 168, 155, 143, 139, 139, 139, 136; this run's is 167, 167, 168, "
                "147, 139, 139, 139, 139, 136) on the recompiled d40ec023 anec with real "
                "encoder tensors instead of synthetic fills."
            ),
            "new_facts_from_real_tensors": [
                "The real layer-0 cond is all false: this capture's encoder_mask has all "
                "375 frames valid, so var_373 = logical_not(attention_mask_7) has zero "
                "true elements. The correct output is therefore matrix_bd_5 verbatim.",
                "The device wrote no -inf at all (device_inf_written 0) and its value "
                "range is exactly matrix_bd_5's range (-50.71875 to 40.84375), so the "
                "-inf fill channel was never selected and the defect is entirely in the "
                "cond/b read, not in the fill path.",
                "The defect is independent of cond content: it appears with an all-false "
                "cond, where a correct select never has to choose 'a' at any lane.",
                "The wrong values are not a column shift of b in the same row: no offset "
                "within +/-96 columns explains any of the 1341 head-0 mismatching lanes, "
                "and only 1170 of 1341 device values appear anywhere in the full 8-head b "
                "value set. The wrong lanes are therefore not characterized here beyond "
                "the positional signature above.",
            ],
        },
        "C": {
            "bundle": "parakeet-encoder-island-pv",
            "programs": 1,
            "task_descriptors": 208,
            "submit": island("C"),
            "outputs": {"attn_output_1": compare["C:attn_output_1"]},
            "difference_structure": {"attn_output_1": analyze["C:attn_output_1"]},
            "verdict": "ACCEPTED_RELATIVE_L2",
            "verdict_detail": (
                "The 208-TD PV matmul completed in 15 ms. 275572/384000 bit-exact, "
                "rel_l2 2.22e-04, max_abs 0.0078, no NaN and no Inf. At |reference| > 1 "
                "the worst relative error is 0.0033, about three fp16 ULP, consistent "
                "with a 375-term dot product accumulated in a different order than the "
                "host fp32 reference. Three orders of magnitude inside the lock bound."
            ),
        },
    },
    "prohibited_actions_observed": {
        "retry_after_failure": False,
        "retry_after_110": False,
        "more_than_one_submit_per_island": False,
        "reboot": False,
        "module_unload": False,
        "gpu_lock_taken": False,
        "executed_1x896": False,
        "wrote_0xf_on_jw16": "n/a (jwm1 only; jw16 never touched)",
        "ane_linux_experiments_edited_or_executed": False,
        "formatter_or_linter_run": False,
    },
    "submit_order": {
        "order": ["B", "C", "A"],
        "reason": (
            "B is 5 TDs, C is 208, A is 416. Running the cheap islands first banks their "
            "results in case the largest program wedges the task manager. Nothing was "
            "re-run in either case."
        ),
    },
    "what_this_establishes": [
        "Hardware execution of all three encoder ANE islands on a physical t8103 ANE. "
        "The 2026-09-14 plan listed this as not established.",
        "416 task descriptors in one package with dispatchPlan [0, 1] dispatch and "
        "release cleanly: two ordered logical results, no intermediate, 66 ms.",
        "The boundary contract in the plan is executable as written: dense row-major "
        "bytes in the manifest nchw order, runtime-side tile placement, no caller-side "
        "surface construction. Every staged buffer was accepted at its declared "
        "logical_bytes with no refusal.",
        "Islands A and C are numerically accepted against a host reference computed from "
        "the same staged bytes, inside the reference lock's relative_l2 class.",
        "Island B's select defect is confirmed on the recompiled d40ec023 anec with real "
        "encoder tensors and an all-false cond, and is confined to H rows 0-8.",
    ],
    "not_established": [
        "Numerical agreement of island B. It is a named, reproduced device-side defect, "
        "not closed here.",
        "The -inf fill path of island B. This capture's real cond is all false, so no "
        "lane in this run required the fill; a capture with padded frames is needed to "
        "exercise it.",
        "Whether the ANE split is faster than Vulkan alone. This run is three isolated "
        "single-iteration submits, not the alternating ANE-on / ANE-off measurement.",
        "Any other layer than layer 0, and any multi-layer or full-encoder pass.",
        "The AneRegion graph-level partitioner. These submits went through the bounded "
        "worker CLI, not through eval.",
    ],
    "artifacts": {
        "in_repo": {
            "receipt": "receipts/2026-09-14-encoder-islands-exec-jwm1-linux.json",
            "derivation": "receipts/2026-09-14-encoder-islands-exec-jwm1-linux/derivation/ "
            "(milrun.py evaluator, validate_full.py, stage.py, compare.py, analyze.py, "
            "run_islands.sh, build_receipt.py)",
            "device_evidence": "receipts/2026-09-14-encoder-islands-exec-jwm1-linux/device-out/ "
            "(per-island stdout, stderr, exit, timestamps, pre/post device state)",
            "measurements": "receipts/2026-09-14-encoder-islands-exec-jwm1-linux/"
            "{stage-manifest,validate-full,compare,analyze}.json",
        },
        "not_in_repo": {
            "staged_tensors": "/var/tmp/IslandsExecJwm1/stage (15 MB of fp16/bool buffers; "
            "sha256 of every one is recorded under tensor_provenance.staged)",
            "device_outputs": "/var/tmp/IslandsExecJwm1/device-out/*.out.bin (sha256 recorded "
            "per output under islands.*.outputs.*.device_sha256)",
            "encoder_model_root": "/var/tmp/IslandsExecJwm1/encoder-source (1 GB depalettized "
            "weights; regenerate with mil_adapter.py emit against the pinned revision)",
            "on_jwm1": "/var/tmp/jwm1-encoder-islands (bundles, staged inputs, out/)",
        },
    },
}

out = WT / "receipts" / "2026-09-14-encoder-islands-exec-jwm1-linux.json"
out.write_text(json.dumps(receipt, indent=2) + "\n")
print("wrote", out)
print(json.dumps(receipt["summary"], indent=2))
