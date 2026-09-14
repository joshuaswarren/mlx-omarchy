#!/usr/bin/env python3
"""Fill the section-42 counter report and evaluate the section-43 invariants
using the repo's own contract code (overlay/tools/coreml/encoder_parity.py),
not a local reimplementation of it.

The ANE byte totals are cross-checked against an analytic expectation derived
from the bundle manifests of the islands the run actually placed, so the
measured numbers are falsifiable rather than merely reported.

Worker liveness comes from the canonical helper, tools/ane_worker_liveness.py
on origin/main, imported rather than re-described. The two pgrep forms that
read this wrong in opposite directions are documented there.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Island tensor byte sizes, from the bundle manifests (dense row-major).
A_INPUTS = {
    "k_headsT": 1 * 8 * 128 * 375 * 2,
    "pos_kT": 1 * 8 * 128 * 749 * 2,
    "q_scaled": 1 * 8 * 375 * 128 * 2,
    "q_v": 1 * 8 * 375 * 128 * 2,
}
A_OUTPUTS = {
    "attention_scores_1": 1 * 8 * 375 * 749 * 2,
    "matmul_0": 1 * 8 * 375 * 375 * 2,
}
# Island B: two fp16 operands and a bool cond at one byte per element.
B_INPUTS = {
    "ninf_rt": 1 * 8 * 375 * 375 * 2,
    "matrix_bd_5": 1 * 8 * 375 * 375 * 2,
    "cond": 1 * 8 * 375 * 375 * 1,
}
B_OUTPUTS = {"attention_mask_9": 1 * 8 * 375 * 375 * 2}
C_INPUTS = {
    "probs": 1 * 8 * 375 * 375 * 2,
    "v_heads": 1 * 8 * 375 * 128 * 2,
}
C_OUTPUTS = {"attn_output_1": 1 * 8 * 375 * 128 * 2}

ISLAND_BYTES = {
    "A": (A_INPUTS, A_OUTPUTS),
    "B": (B_INPUTS, B_OUTPUTS),
    "C": (C_INPUTS, C_OUTPUTS),
}
# Submits and compiled programs each island costs per layer.
ISLAND_SUBMITS = {"A": 1, "B": 1, "C": 1}
ISLAND_PROGRAMS = {"A": 2, "B": 1, "C": 1}
ISLAND_TASK_DESCRIPTORS = {"A": 416, "B": 5, "C": 208}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--liveness", type=Path, required=True,
        help="path to tools/ane_worker_liveness.py (the canonical helper)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.package))
    from coreml.encoder_parity import (  # noqa: E402
        ANE_COUNTER_FIELDS,
        empty_counter_report,
        evaluate_invariants,
    )

    liveness_spec = importlib.util.spec_from_file_location(
        "ane_worker_liveness", args.liveness
    )
    liveness = importlib.util.module_from_spec(liveness_spec)
    liveness_spec.loader.exec_module(liveness)
    liveness_sha = hashlib.sha256(args.liveness.read_bytes()).hexdigest()
    live = liveness.snapshot()
    report_json = json.loads(args.run_report.read_text())
    ane = report_json["ane"]
    layers = report_json["layers"]
    gpu = report_json["gpu_counters"]

    placed = report_json["islands_placed"]
    if not placed:
        raise SystemExit("run report placed no islands; nothing to account")
    unknown = set(placed) - ISLAND_BYTES.keys()
    if unknown:
        raise SystemExit(f"run report placed unknown island(s) {sorted(unknown)}")
    expect_in = layers * sum(sum(ISLAND_BYTES[i][0].values()) for i in placed)
    expect_out = layers * sum(sum(ISLAND_BYTES[i][1].values()) for i in placed)
    expect_submits = layers * sum(ISLAND_SUBMITS[i] for i in placed)
    expect_programs = layers * sum(ISLAND_PROGRAMS[i] for i in placed)
    expect_tds = layers * sum(ISLAND_TASK_DESCRIPTORS[i] for i in placed)

    counters = empty_counter_report()
    # Measured at the worker process boundary: this run has no in-framework ANE
    # submission path, so the MLX backend's ane_* counters cannot see the work.
    counters.update(
        {
            "ane_models_loaded": ane["worker_starts"],
            "ane_packages_compiled": 0,
            "ane_package_cache_hits": 0,
            "ane_worker_starts": ane["worker_starts"],
            "ane_submissions": ane["submissions"],
            "ane_timeouts": ane["timeouts"],
            "ane_input_bytes": ane["input_bytes"],
            "ane_output_bytes": ane["output_bytes"],
            "ane_exec_ns": ane["exec_ns"],
            "gpu_primitive_dispatches": gpu["gpu_primitive_dispatches"],
            "vk_compute_dispatches": gpu["vk_compute_dispatches"],
            "cpu_tensor_events": report_json["cpu_tensor_events"],
            "accounted_input_bytes": ane["input_bytes"],
            "accounted_output_bytes": ane["output_bytes"],
            "expected_input_bytes": expect_in,
            "expected_output_bytes": expect_out,
        }
    )
    violations = evaluate_invariants(counters)

    completions = counters["ane_submissions"] - counters["ane_timeouts"]
    result = {
        "counter_report": counters,
        "ane_counter_fields": list(ANE_COUNTER_FIELDS),
        "evaluated_by": (
            "overlay/tools/coreml/encoder_parity.py::evaluate_invariants at "
            "receipts/2026-09-14-encoder-split-plan"
        ),
        "violations": violations,
        "section_43_clean": violations == [],
        "islands_placed": placed,
        "dispatch_accounting": {
            "measured_submissions": ane["submissions"],
            "expected_submissions": expect_submits,
            "submissions_match": ane["submissions"] == expect_submits,
            "programs_total": expect_programs,
            "task_descriptors_total": expect_tds,
            "derivation": (
                f"{layers} layers x islands {placed}: "
                + ", ".join(
                    f"{i} {ISLAND_SUBMITS[i]} submit / "
                    f"{ISLAND_PROGRAMS[i]} program(s) / "
                    f"{ISLAND_TASK_DESCRIPTORS[i]} TDs"
                    for i in placed
                )
                + ". Programs and task descriptors are the manifests' counts, "
                "not measured at the worker; submissions are measured."
            ),
        },
        "byte_accounting": {
            "measured_input_bytes": ane["input_bytes"],
            "analytic_input_bytes": expect_in,
            "input_match": ane["input_bytes"] == expect_in,
            "measured_output_bytes": ane["output_bytes"],
            "analytic_output_bytes": expect_out,
            "output_match": ane["output_bytes"] == expect_out,
            "derivation": (
                f"{layers} layers x islands {placed}, bytes in "
                + " + ".join(f"{i} {sum(ISLAND_BYTES[i][0].values())}" for i in placed)
                + "; bytes out "
                + " + ".join(f"{i} {sum(ISLAND_BYTES[i][1].values())}" for i in placed)
                + ". Sizes are the manifests' declared logical byte sizes: fp16 "
                "operands at two bytes an element, island B's bool cond at one."
            ),
            "scope_note": (
                "Section 43's byte-accounting clause is written for a single "
                "in-framework run where the ANE and Vulkan path counters both "
                "report. Here the compared checkpoints (encoder_hidden, "
                "encoder_mask) are produced on the GPU, and the ANE traffic is "
                "island tensors measured at the worker boundary. What is "
                "accounted is therefore every byte that crossed to and from the "
                "ANE, checked against the manifests rather than against the "
                "checkpoint totals. The deviation from the clause as literally "
                "written is named here rather than hidden by picking numbers "
                "that make the equality hold."
            ),
        },
        "worker_completions": completions,
        "completions_equal_submissions": completions == counters["ane_submissions"],
        "lingering_worker_processes": {
            "value": live["worker_count"],
            "device_holders": live["device_holders"],
            "workers": live["workers"],
            "measured_by": {
                "tool": "tools/ane_worker_liveness.py::snapshot",
                "landed": "origin/main 4ded332c",
                "tool_sha256": liveness_sha,
                "doc": "docs/ane-worker-liveness.md",
                "method": live["method"],
            },
            "why_not_pgrep": (
                "Neither pgrep form answers this. `pgrep -x "
                "mlx-omarchy-ane-worker` compares a 22-character name against "
                "/proc/<pid>/comm, which Linux truncates at 15 bytes, so it "
                "reports 0 with a worker alive; `pgrep -f` matches the calling "
                "shell and reports workers that do not exist. The helper above "
                "matches basename(argv[0]) and separates a real worker from a "
                "look-alike by an open descriptor on the accel device. The "
                "2026-09-14 two-island receipt carried this reasoning inline "
                "because the helper did not exist on origin/main yet; it does "
                "now, so this run imports it instead of restating it."
            ),
        },
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"section 43 clean: {result['section_43_clean']}")
    for name in ANE_COUNTER_FIELDS:
        print(f"  {name:24s} {counters[name]}")
    print(f"  {'cpu_tensor_events':24s} {counters['cpu_tensor_events']}")
    print(
        f"  byte accounting: in {ane['input_bytes']} == {expect_in} "
        f"{result['byte_accounting']['input_match']}, out "
        f"{ane['output_bytes']} == {expect_out} "
        f"{result['byte_accounting']['output_match']}"
    )
    print(
        f"  dispatch: islands {placed}, submissions {ane['submissions']} == "
        f"{expect_submits} {result['dispatch_accounting']['submissions_match']}, "
        f"{expect_programs} programs, {expect_tds} task descriptors"
    )
    print(
        f"  workers alive after run: {live['worker_count']}, "
        f"accel device holders: {live['device_holders']}"
    )
    if violations:
        print("VIOLATIONS:")
        for item in violations:
            print(f"  - {item}")
    clean = (
        result["section_43_clean"]
        and result["byte_accounting"]["input_match"]
        and result["byte_accounting"]["output_match"]
        and result["dispatch_accounting"]["submissions_match"]
        and live["worker_count"] == 0
        and live["device_holders"] == []
    )
    return 0 if clean else 3


if __name__ == "__main__":
    raise SystemExit(main())
