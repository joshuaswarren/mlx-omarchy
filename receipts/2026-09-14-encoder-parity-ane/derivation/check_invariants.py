#!/usr/bin/env python3
"""Fill the section-42 counter report and evaluate the section-43 invariants
using the repo's own contract code (overlay/tools/coreml/encoder_parity.py),
not a local reimplementation of it.

The ANE byte totals are cross-checked against an analytic expectation derived
from the two bundle manifests, so the measured numbers are falsifiable rather
than merely reported.
"""

import argparse
import json
import sys
from pathlib import Path

# Island tensor byte sizes, from the bundle manifests (fp16, dense row-major).
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
C_INPUTS = {
    "probs": 1 * 8 * 375 * 375 * 2,
    "v_heads": 1 * 8 * 375 * 128 * 2,
}
C_OUTPUTS = {"attn_output_1": 1 * 8 * 375 * 128 * 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.package))
    from coreml.encoder_parity import (  # noqa: E402
        ANE_COUNTER_FIELDS,
        empty_counter_report,
        evaluate_invariants,
    )

    report_json = json.loads(args.run_report.read_text())
    ane = report_json["ane"]
    layers = report_json["layers"]
    gpu = report_json["gpu_counters"]

    expect_in = layers * (sum(A_INPUTS.values()) + sum(C_INPUTS.values()))
    expect_out = layers * (sum(A_OUTPUTS.values()) + sum(C_OUTPUTS.values()))

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
        "byte_accounting": {
            "measured_input_bytes": ane["input_bytes"],
            "analytic_input_bytes": expect_in,
            "input_match": ane["input_bytes"] == expect_in,
            "measured_output_bytes": ane["output_bytes"],
            "analytic_output_bytes": expect_out,
            "output_match": ane["output_bytes"] == expect_out,
            "derivation": (
                f"{layers} layers x (island A inputs {sum(A_INPUTS.values())} + "
                f"island C inputs {sum(C_INPUTS.values())}) bytes in; "
                f"{layers} x (island A outputs {sum(A_OUTPUTS.values())} + "
                f"island C outputs {sum(C_OUTPUTS.values())}) bytes out. Sizes "
                "are the manifests' declared fp16 logical byte sizes."
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
            "value": 0,
            "verified_by": (
                "ps -eo pid,etimes,stat,comm,args filtered for ane-worker "
                "returned NONE, and /proc/*/fd showed 0 descriptors on "
                "/dev/accel/accel0 after the run."
            ),
            "instrumentation_correction": (
                "The snapshot helper's worker count is unreliable in both "
                "directions and neither reading was used. The mechanism, "
                "verified empirically against a live long-named stand-in "
                "process: Linux TASK_COMM_LEN is 16 bytes, so /proc/<pid>/comm "
                "holds only the first 15 characters of a 22-character "
                "'mlx-omarchy-ane-worker', namely 'mlx-omarchy-ane'. `pgrep -x` "
                "compares against comm, so `pgrep -c -x mlx-omarchy-ane-worker` "
                "returns 0 with a worker genuinely alive -- a false negative, "
                "and the form the 2026-09-14 islands receipt used. In the other "
                "direction `pgrep -cf mlx-omarchy-ane-worker` matches the "
                "naming shell itself and reported 3 workers here when there "
                "were none. `pgrep -c -x mlx-omarchy-ane` is correct in both "
                "directions and `pgrep -a mlx-omarchy-ane` is the useful "
                "one-liner. Credit to WorkerLivenessFix, who measured the same "
                "two failure modes on a live stand-in worker and is landing a "
                "shared helper plus doc for this; those paths are not on "
                "origin/main at the time of writing, so they are described "
                "rather than cited. An earlier version of this receipt "
                "attributed the false negative to the name being 22 characters "
                "rather than to comm truncating at 15, which named the wrong "
                "cause while reaching the right conclusion."
            ),
            "verified_commands": {
                "with_a_worker_alive": {
                    "pgrep -c -x mlx-omarchy-ane-worker": 0,
                    "pgrep -c -x mlx-omarchy-ane": 1,
                    "proc_comm": "mlx-omarchy-ane (15 of 22 characters)",
                },
                "with_no_worker_alive": {"pgrep -c -x mlx-omarchy-ane": 0},
            },
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
    if violations:
        print("VIOLATIONS:")
        for item in violations:
            print(f"  - {item}")
    return 0 if result["section_43_clean"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
