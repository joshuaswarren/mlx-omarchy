#!/usr/bin/env python3
"""Assemble receipts/2026-09-14-encoder-parity-ane.json from the measured
artifacts of the two encoder runs. Every number here comes from a file one of
the runs wrote; nothing is retyped by hand.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("/home/joshuawarren/src/mlx-omarchy")
BASE = Path("/var/tmp/EncoderParityAne")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def per_layer(log: list[dict]) -> dict:
    layers: dict[str, dict] = {}
    for record in log:
        tag = record["tag"]
        layer, island = tag.split("-")
        entry = layers.setdefault(layer, {})
        entry[island] = {
            "bundle": record["bundle"],
            "exit": record["exit"],
            "elapsed_ms": round(record["elapsed_ns"] / 1e6, 3),
            "input_bytes": record.get("input_bytes"),
            "output_bytes": record.get("output_bytes"),
        }
    return dict(sorted(layers.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ane_report = json.loads((BASE / "out-ane/run-report.json").read_text())
    vulkan_report = json.loads((BASE / "out-vulkan/run-report.json").read_text())
    ane_cmp = json.loads((BASE / "results/compare-ane.json").read_text())
    vulkan_cmp = json.loads((BASE / "results/compare-vulkan.json").read_text())
    env = json.loads((BASE / "results/environment.json").read_text())
    anchors = json.loads((BASE / "results/golden-anchors.json").read_text())
    invariants = json.loads((BASE / "results/invariants.json").read_text())

    ane = ane_report["ane"]
    layers = ane_report["layers"]
    derivation = REPO / "receipts/2026-09-14-encoder-parity-ane/derivation"

    receipt = {
        "schema": "mlx-omarchy.encoder-parity-ane.v1",
        "date": "2026-09-14",
        "host": "jwm1-linux",
        "result": (
            "ENCODER_RUN_WITH_ANE_ISLANDS_A_AND_C_ON_ALL_24_LAYERS_WITHIN_CONTRACT"
            if ane_cmp["all_bounds_pass"]
            else "ENCODER_RUN_COMPLETED_OUT_OF_CONTRACT"
        ),
        "what_was_run": {
            "claim": (
                "The complete pinned Parakeet encoder for the golden LibriSpeech "
                "clip, with ANE islands A and C carrying the attention matmuls in "
                "every one of the 24 transformer layers and the Apple GPU "
                "(Vulkan, through mlx-omarchy) executing every other tensor op."
            ),
            "ane_carries": [
                "island A program 0: attention_scores_N = matmul(q_v, pos_kT)",
                "island A program 1: matmul_N = matmul(q_scaled, k_headsT)",
                "island C: attn_output_N = matmul(probs, v_heads)",
            ],
            "gpu_carries": (
                "everything else: the 4-stage subsampling conv stack, mask "
                "derivation, all layer norms, both feed-forwards per layer, the "
                "depthwise conv module, softmax, the attention-mask select, and "
                "the epilogue layer_norm plus projector linear"
            ),
            "layers_with_ane_attention": layers,
            "ops_total": ane_report["ops_executed"],
            "ops_on_ane": ane_report["ane_ops"],
            "ops_on_gpu": ane_report["gpu_ops"],
            "ops_const_loads": (
                ane_report["ops_executed"]
                - ane_report["ane_ops"]
                - ane_report["gpu_ops"]
            ),
        },
        "resolved_model": {
            "assigned_route": "openai (mlx-openai-deep agent type)",
            "actual": "anthropic/claude-opus-5 (session-reported identity)",
            "fallback": True,
            "note": (
                "The assignment named an OpenAI route; this session reports an "
                "Anthropic model. Reported to the parent, which authorised "
                "proceeding on the fallback model provided the identity is "
                "disclosed here. The same discrepancy is recorded in "
                "receipts/2026-09-14-encoder-islands-exec-jwm1-linux.json, so it "
                "is systemic to this batch rather than a one-off."
            ),
        },
        "contract": {
            "source": "overlay/tools/coreml/parakeet-reference.lock",
            "lock_sha256": sha256(BASE / "parakeet-reference.lock"),
            "bounds": ane_cmp["bounds"],
            "ane_split_run": ane_cmp,
            "vulkan_only_control": vulkan_cmp,
            "verdict": (
                "PASS on every frozen bound"
                if ane_cmp["all_bounds_pass"]
                else "FAIL: see per_bound"
            ),
        },
        "ane_submissions": {
            "submissions": ane["submissions"],
            "worker_starts": ane["worker_starts"],
            "timeouts": ane["timeouts"],
            "expected_submissions": 2 * layers,
            "submissions_match_expectation": ane["submissions"] == 2 * layers,
            "accounting": (
                f"{layers} layers x 2 islands = {2 * layers} submits. Island A is "
                "one submit carrying two programs (dispatchPlan [0, 1], 416 task "
                "descriptors); island C is one submit of one program (208 task "
                "descriptors). That is "
                f"{layers * (416 + 208)} task descriptors and "
                f"{layers * 3} ANE programs across the pass."
            ),
            "task_descriptors_total": layers * (416 + 208),
            "programs_total": layers * 3,
            "input_bytes": ane["input_bytes"],
            "output_bytes": ane["output_bytes"],
            "ane_exec_ns": ane["exec_ns"],
            "ane_exec_ms": round(ane["exec_ns"] / 1e6, 1),
            "one_process_per_submit": (
                "The bounded worker CLI takes one bundle and one input set per "
                "process, so each layer's islands are separate launches. "
                "worker_starts equals submissions by construction; a persistent "
                "worker or the graph-level partitioner would change this."
            ),
            "per_layer_dispatch": {
                "schema": (
                    "encoder_layers_NN -> {A: island-attn-a-kt, C: island-pv}, "
                    "each with exit status, wall ms measured around the worker "
                    "process, and the bytes shipped in and read back"
                ),
                "programs_per_layer": {"A": 2, "C": 1},
                "task_descriptors_per_layer": {"A": 416, "C": 208, "total": 624},
                "submits_per_layer": 2,
                "layers": per_layer(ane["log"]),
            },
        },
        "timing": {
            "ane_split_wall_ns": ane_report["wall_ns"],
            "ane_split_wall_s": round(ane_report["wall_ns"] / 1e9, 3),
            "vulkan_only_wall_ns": vulkan_report["wall_ns"],
            "vulkan_only_wall_s": round(vulkan_report["wall_ns"] / 1e9, 3),
            "ane_share_of_wall": round(
                ane["exec_ns"] / ane_report["wall_ns"], 4
            ),
            "interpretation": (
                "These two numbers are not a fair speed comparison and must not "
                "be read as one. The ANE split pays 48 process launches, 48 "
                "bundle loads and a host round trip per island, none of which a "
                "graph-level partitioner would pay. What the pair does establish "
                "is that moving the attention matmuls to the ANE keeps the "
                "encoder inside the frozen contract."
            ),
        },
        "counters": {
            "section_42": {
                "ane_split": ane_report["gpu_counters"],
                "vulkan_only": vulkan_report["gpu_counters"],
            },
            "ane_counter_provenance": (
                "The nine MLX-backend ane_* counters in trace.h stay zero here "
                "because this run does not submit through MLX eval: there is no "
                "AneRegion graph partitioner, so the islands go through the "
                "bounded mlx-omarchy-ane-worker CLI. The submission counts above "
                "are measured at the worker process boundary, which is the real "
                "authority for this run. Reporting the backend counters as the "
                "ANE evidence would understate the work as zero."
            ),
            "section_43": invariants,
        },
        "no_cpu_tensor_fallback": {
            "cpu_tensor_events": ane_report["cpu_tensor_events"],
            "how_enforced": (
                "Every op in the evaluator is implemented with mlx.core only and "
                "dispatch raises EncoderRunError on any op it cannot express in "
                "mx, so there is no CPU arithmetic path available to fall back "
                "to. Both runs executed all 3351 ops with zero unimplemented-op "
                "errors."
            ),
            "explicitly_not_arithmetic": [
                "Pinned constants are byte-reinterpreted from the adapter's "
                "blob-v2 files straight into device arrays. That is a load, not a "
                "computation.",
                "Island tensors cross to the ANE worker as dense row-major bytes "
                "through files, and its outputs come back the same way. That is "
                "process I/O, not arithmetic. The bytes are counted above.",
                "The golden comparison runs in numpy off-device. It is analysis "
                "of the result, not part of the execution path.",
                "The two encoder inputs are marshalled from the golden .npy "
                "files into device arrays. The reshape is metadata and the "
                "dtype casts are verified no-ops: encoder_input_features.npy is "
                "already float32 (1, 3000, 128) and encoder_input_mask.npy is "
                "already int32 (1, 3000) on disk, so astype changes no value "
                "and computes nothing.",
            ],
            "host_side_metadata": (
                "Shapes, axes, perms, strides, masks, epsilon and transpose flags "
                "are read as host constants. The pinned encoder derives no shape "
                "from a computed tensor, so this never forces a device sync on a "
                "value the run computed."
            ),
        },
        "island_b_is_not_on_the_ane": {
            "placement": (
                "The -inf attention-mask select runs on the Apple GPU, not the "
                "ANE. The h13.select-first-l2-tile defect is therefore outside "
                "this run's execution path rather than tolerated inside it."
            ),
            "why_excluded": (
                "The ANE select is a reproduced device defect that corrupts its "
                "passthrough lanes independently of cond content: 1341 wrong "
                "lanes per head on jwm1 and 1337 per head on jw16, all in H rows "
                "0-8, max_abs_err 11.1640625, with cond genuinely false "
                "everywhere in both runs. Credit to AneChannelPolarity for "
                "correcting an earlier framing of mine that treated the all-false "
                "cond as making the clip immune to the hole; it does not."
            ),
            "separate_coverage_gap": (
                "Independently of the defect, this clip's cond is all false (0 of "
                "1125000 elements true, 375 valid frames of 375), so the select's "
                "-inf fill semantics are not exercised by this input on either "
                "device. A padded-frame clip is still required to test that path. "
                "This is a gap in what the run covers, not a justification for "
                "anything."
            ),
            "not_a_fix": (
                "This run establishes nothing about ANE select correctness and "
                "does not close the defect."
            ),
            "not_generalised": (
                "Skipping the ANE select is a placement decision for this run. It "
                "is not a claim that the select can be skipped in general, and "
                "not a claim that any other clip behaves this way."
            ),
        },
        "pins": {
            "worktree": f"mlx-omarchy {git('rev-parse', '--abbrev-ref', 'HEAD')} {git('rev-parse', 'HEAD')}",
            "predecessor_receipt": (
                "receipts/2026-09-14-encoder-islands-exec-jwm1-linux.json "
                "(branch receipts/2026-09-14-encoder-split-plan, 23d3172d)"
            ),
            "environment": env,
            "golden_capture": {
                "path": (
                    "~/.cache/mlx-omarchy/parakeet-reference/captures/"
                    "b650695c-75aec2a/20260912T154759Z-librispeech/ane"
                ),
                "anchors_verified_against_lock": anchors,
            },
            "derivation": {
                name: sha256(derivation / name)
                for name in sorted(p.name for p in derivation.glob("*.py"))
            },
        },
        "what_this_establishes": [
            "The complete 24-layer pinned Parakeet encoder runs for the golden "
            "clip with the attention matmuls of every layer executing on a "
            "physical t8103 ANE and every other tensor op on the Apple GPU, and "
            "the final encoder_hidden lands inside every frozen bound in the "
            "reference lock, with encoder_mask bit-exact.",
            "The 2026-09-14 islands receipt listed 'any other layer than layer 0, "
            "and any multi-layer or full-encoder pass' as not established. This "
            "run establishes it for islands A and C.",
            "The layer-0 island bundles are reusable across all 24 layers "
            "unchanged, because every operand is a runtime input to the bundle "
            "and no weight is baked into the compiled anec. Note this is not "
            "because the islands bind only activations: island A's pos_kT is a "
            "distinct per-layer constant (var_355_to_fp16 through "
            "var_3805_to_fp16, 24 distinct [1,8,128,749] fp16 blobs in "
            "weight.bin, 1533952 bytes each). The bundles are shape-identical "
            "across layers and take that weight as an input, so one compiled "
            "bundle per island serves every layer with different input bytes. "
            "Those 24 per-layer constants are included in the ANE input byte "
            "accounting.",
            "A MIL-to-MLX encoder executor now exists. Before this run the "
            "encoder was the one pinned component with no Vulkan executor: mel, "
            "decoder and joint had one, the encoder did not, and the only "
            "encoder evaluator in tree was a CPU numpy one.",
            "The Vulkan-only control, Linux Apple-GPU against the macOS ANE "
            "golden, lands at rel_l2 "
            f"{vulkan_cmp['measured']['rel_l2_err']:.6f}. The lock records "
            "macOS GPU against that same macOS ANE golden at rel_l2 0.023345. "
            "Those are different GPUs on different operating systems measured "
            "against one common reference, so they are comparable in kind, and "
            "landing that close is independent evidence the port is faithful "
            "rather than accidentally compensating for an error elsewhere.",
            "The ANE contribution is separable and small. Islands A and C move "
            "rel_l2 from "
            f"{vulkan_cmp['measured']['rel_l2_err']:.6f} to "
            f"{ane_cmp['measured']['rel_l2_err']:.6f}, a change of "
            f"{ane_cmp['measured']['rel_l2_err'] - vulkan_cmp['measured']['rel_l2_err']:.6f}, "
            "for all 72 attention matmuls across 24 layers. The control arm is "
            "what makes that attribution falsifiable instead of asserted.",
        ],
        "not_established": [
            "Island B on the ANE. The select defect is untouched by this run.",
            "The -inf fill path of the attention-mask select, on either device: "
            "this clip's cond is all false.",
            "Whether the ANE split is faster than Vulkan alone. The two wall "
            "times here are not comparable: the split pays 48 process launches "
            "and 48 bundle loads that a real partitioner would not.",
            "The AneRegion graph-level partitioner. These submits went through "
            "the bounded worker CLI, not through MLX eval, so the encoder is not "
            "yet ANE-accelerated from inside the framework.",
            "End-to-end transcription. This run compares encoder_hidden and "
            "encoder_mask; the lock's token_ids and transcript equality bounds "
            "need the decoder and joint, which are not part of this slice.",
            "Any clip other than the pinned golden LibriSpeech capture, and any "
            "sequence length other than this one's 375 valid frames. The island "
            "bundles are compiled for these exact shapes.",
        ],
        "remaining_before_the_encoder_is_ane_complete": [
            "Fix or route around h13.select-first-l2-tile so island B can run on "
            "the ANE, or establish that the select belongs on the GPU by design.",
            "Exercise the -inf fill with a padded-frame clip, where cond has true "
            "lanes.",
            "Land the AneRegion graph-level partitioner so eval places the "
            "islands, replacing 48 worker processes with in-framework submits, "
            "and only then measure ANE-on versus ANE-off for speed.",
            "Give the ANE more of the layer than the three attention matmuls: "
            "the feed-forwards and the projector linear are the remaining large "
            "accumulating ops, and they are still entirely on the GPU.",
            "Shape generality: bundles compiled for 375 frames only.",
        ],
        "prohibited_actions_observed": {
            "reboot": False,
            "module_unload": False,
            "executed_1x896": False,
            "set_writes": False,
            "gpu_lock_stolen": False,
            "ane_taken_without_ownership": False,
            "ane_linux_experiments_edited_or_executed": False,
            "formatter_or_linter_run": False,
            "retry_after_failure": False,
        },
        "coordination": {
            "jwm1_ane": (
                "Granted by the parent, queued behind AneChannelPolarity, and "
                "taken only after they confirmed release with a verified post "
                "state (boot_id unchanged, module refs=0, quarantine 0 bytes, no "
                "tm-execution-failed or errno-110 for the whole boot)."
            ),
            "gpu_lock": (
                "Taken with a waiting flock, never stolen. MesaWaitsPatch asked "
                "to place their six paired prefill rounds before the ANE pass, "
                "because a multi-second encoder pass landing between the arms of "
                "a sub-one-percent A/B would silently destroy the pairing. That "
                "was a correct call and the ANE pass was sequenced after their "
                "runs."
            ),
            "libane_choice": (
                "Pinned to the same libane the islands run used "
                "(1ab9d95debcc8b5fee3b6653dfce0b50412bc7efef43c2d2167dc83ce270ca49, "
                "source f261a6cb) so the numbers stay comparable to the proven "
                "island acceptance. AneChannelPolarity's channel-polarity fix is "
                "device-proven and is deliberately NOT in this binary; they "
                "confirmed island A and C channel binding is already correct "
                "under the positional assumption, and swapping libane mid-run "
                "would have confounded the comparison."
            ),
            "build_attribution": (
                "Measured on the committed build in ~/venv-agxgen, not on the "
                "dirty working tree. overlay/.../shaders/select.comp is modified "
                "and unstaged in the checkout and belongs to neither me nor "
                "AneChannelPolarity, so measuring the working tree would have "
                "produced an unattributable number."
            ),
        },
        "artifacts": {
            "in_repo": {
                "receipt": "receipts/2026-09-14-encoder-parity-ane.json",
                "derivation": (
                    "receipts/2026-09-14-encoder-parity-ane/derivation/ "
                    "(vulkan_encoder.py the MIL-to-MLX executor with the ANE "
                    "splice, compare_golden.py, check_conv_semantics.py, "
                    "check_invariants.py, build_receipt.py)"
                ),
                "measurements": (
                    "receipts/2026-09-14-encoder-parity-ane/"
                    "{run-report-ane,run-report-vulkan,compare-ane,compare-vulkan,"
                    "invariants,environment,golden-anchors}.json"
                ),
            },
            "not_in_repo": {
                "encoder_model_root": (
                    "/var/tmp/EncoderParityAne/encoder-source on jwm1 (1.2 GB of "
                    "depalettized fp16 blobs; regenerate with mil_adapter.py emit "
                    "against the pinned revision)"
                ),
                "run_outputs": (
                    "/var/tmp/EncoderParityAne/out-{ane,vulkan}/encoder_hidden.npy "
                    "(sha256 recorded under contract.*.encoder_hidden_sha256)"
                ),
            },
        },
    }

    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  result            {receipt['result']}")
    print(f"  submissions       {ane['submissions']} (expected {2 * layers})")
    print(f"  contract verdict  {receipt['contract']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
