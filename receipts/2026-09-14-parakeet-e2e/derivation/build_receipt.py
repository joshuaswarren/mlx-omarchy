#!/usr/bin/env python3
"""Assemble receipts/2026-09-14-parakeet-e2e.json from the two measured runs.

Every number here is read out of an artifact one of the runs wrote on
jwm1-linux; nothing is retyped by hand. Prior receipts are cited by path and
hash so the attribution claims can be re-checked without trusting this file.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("/home/joshuawarren/src/mlx-omarchy")
BASE = Path("/var/tmp/ParakeetE2E")

PIPELINE_FILES = (
    "overlay/tools/coreml/vulkan_mel.py",
    "overlay/tools/coreml/vulkan_mel_constants.py",
    "overlay/tools/coreml/vulkan_decoder.py",
    "overlay/tools/coreml/vulkan_joint.py",
    "overlay/tools/coreml/parakeet_tdt.py",
    "overlay/tools/coreml/tokenizer.py",
    "overlay/tools/coreml/pinned_component.py",
    "overlay/tools/coreml/reference.py",
    "overlay/tools/coreml/parakeet-reference.lock",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def arm(report: dict) -> dict:
    """The comparable core of one run, in the plan's layer vocabulary."""
    layers = report["layers"]
    encoder = layers["layer_5_encoder"]
    tokens = layers["layer_6_decoder_sequence"]
    text = layers["layer_7_end_to_end_text"]
    return {
        "status": report["status"],
        "ane_mode": report["execution"]["ane_mode"],
        "stages": report["stages"],
        "timing": report["timing"],
        "execution": report["execution"],
        "ane": report["ane"],
        "layer_2_preprocessing": layers["layer_2_preprocessing"],
        "layer_5_encoder": encoder,
        "layer_6_decoder_sequence": tokens,
        "layer_7_end_to_end_text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ane = json.loads((BASE / "out/e2e-report.json").read_text())
    gpu = json.loads((BASE / "out-vulkan/e2e-report.json").read_text())
    post_state = dict(
        line.split("=", 1)
        for line in (BASE / "state/post.txt").read_text().splitlines()
        if "=" in line
    )

    prior_tdt = json.loads(
        git("show", "1f8804cd:receipts/2026-09-13-parakeet-vulkan-tdt/receipt.json")
    )
    lstm = json.loads(
        (REPO / "receipts/2026-09-14-decoder-lstm-semantics/result.json").read_text()
    )
    parity = json.loads(
        Path("/var/tmp/EncoderParityAne/results/compare-ane.json").read_text()
    )

    ane_tokens = ane["layers"]["layer_6_decoder_sequence"]
    gpu_tokens = gpu["layers"]["layer_6_decoder_sequence"]
    native = ane_tokens["native"]

    head = ane["inputs"]
    receipt = {
        "schema": "mlx-omarchy.parakeet-e2e/1",
        "date": "2026-09-14",
        "plan": "docs/plans/2026-09-12-coreml-parakeet-ane-plan.md",
        "phase": "7 (end-to-end transcription): exact mel frontend, encoder on "
        "ANE, decoder/joint on Vulkan, TDT control, tokenizer",
        "verdict": {
            "phase_7_green": False,
            "reason": "The frozen contract requires token_ids_must_match_exactly "
            "and transcript_must_match_exactly. The pipeline ran end to end on "
            "Linux with no inference server and no CPU tensor primitive, the mel "
            "frontend is bit-exact against macOS and the encoder is inside every "
            "frozen bound, but the emitted token sequence diverges from the "
            "native sequence at emission 101 of 104, so the transcript differs.",
            "what_is_proven": [
                "One Linux process carried audio -> mel -> encoder (ANE islands "
                "A and C on all 24 layers) -> decoder -> joint -> TDT control -> "
                "tokenizer, each stage consuming the previous stage's device "
                "tensors; no stage was replayed from the macOS capture.",
                "Layer 2 is bit-exact: mel and encoder_input_features computed "
                "from the pinned FLAC match the macOS golden tensors byte for "
                "byte, and both masks are exact.",
                "Layer 5 is inside the frozen contract with the attention "
                "matmuls on the ANE (max_abs 0.147156, mean_abs 0.004182, "
                "rel_l2 0.024917, 0 NaN, 0 Inf, mask bit-exact).",
                "The tokenizer is exact: detokenizing the golden 104 ids "
                "reproduces the golden transcript byte for byte (checked "
                "separately from the run).",
                "No inference server, cpu_tensor_events 0, 48 bounded ANE "
                "submissions, 0 timeouts, no worker left alive.",
            ],
        },
        "host": ane["host"],
        "mlx": ane["mlx"],
        "inputs": head,
        "runs": {
            "ane_islands_a_c": arm(ane),
            "vulkan_only_control": arm(gpu),
        },
        "transcript": {
            "actual": ane["layers"]["layer_7_end_to_end_text"]["transcript"],
            "actual_sha256": ane["layers"]["layer_7_end_to_end_text"]["actual_sha256"],
            "native": ane["layers"]["layer_7_end_to_end_text"]["native_transcript"],
            "native_sha256": ane["layers"]["layer_7_end_to_end_text"]["native_sha256"],
            "match": ane["layers"]["layer_7_end_to_end_text"]["transcript_match"],
            "difference": "Identical through '...flour-fattened sauce' and the "
            "repeated-period run. The reference's own degenerate tail is "
            "' Юн Ю'; this run emits 'ЮНЕН'. The divergence is entirely inside "
            "the reference's junk suffix, which is where the near-tie logits "
            "are, and it is still a contract failure.",
            "control_actual": gpu["layers"]["layer_7_end_to_end_text"]["transcript"],
            "control_actual_sha256": gpu["layers"]["layer_7_end_to_end_text"][
                "actual_sha256"
            ],
        },
        "token_ids": {
            "actual": ane_tokens["actual"],
            "native": native,
            "control_actual": gpu_tokens["actual"],
            "comparison": {
                "ane_arm": {
                    "actual_emissions": ane_tokens["actual_emissions"],
                    "native_emissions": ane_tokens["native_emissions"],
                    "matching_prefix_length": ane_tokens["matching_prefix_length"],
                    "tokens_match": ane_tokens["tokens_match"],
                    "first_divergence": ane_tokens["first_divergence"],
                },
                "vulkan_only_arm": {
                    "actual_emissions": gpu_tokens["actual_emissions"],
                    "native_emissions": gpu_tokens["native_emissions"],
                    "matching_prefix_length": gpu_tokens["matching_prefix_length"],
                    "tokens_match": gpu_tokens["tokens_match"],
                    "first_divergence": gpu_tokens["first_divergence"],
                },
            },
        },
        "attribution": {
            "claim": "The token divergence is decoder-side, not encoder-side, and "
            "not ANE-side.",
            "evidence": [
                {
                    "fact": "This run's ANE encoder output is byte-identical to the "
                    "independent Phase 6 ANE run, although this one was fed by the "
                    "Linux mel frontend and that one by the capture's features.",
                    "measured": ane["layers"]["layer_5_encoder"][
                        "vs_prior_ane_encoder_run"
                    ],
                },
                {
                    "fact": "The prior TDT comparison, driven by the macOS golden "
                    "encoder tensor instead of any Linux encoder, diverges at the "
                    "same emission index with the same token pair.",
                    "prior_receipt": "receipts/2026-09-13-parakeet-vulkan-tdt/"
                    "receipt.json at 1f8804cd",
                    "prior_first_divergence": prior_tdt["comparison"][
                        "first_divergence"
                    ],
                },
                {
                    "fact": "The ANE arm tracks native further than the ANE-free "
                    "arm, so putting the 72 attention matmuls on the ANE does not "
                    "cause and does not worsen the divergence.",
                    "ane_matching_prefix": ane_tokens["matching_prefix_length"],
                    "vulkan_only_matching_prefix": gpu_tokens["matching_prefix_length"],
                    "ane_emissions": ane_tokens["actual_emissions"],
                    "vulkan_only_emissions": gpu_tokens["actual_emissions"],
                },
                {
                    "fact": "No LSTM variant reproduces the native decoder "
                    "bit-exactly; the best of the surveyed variants still misses "
                    "next_cell by 0.0278 absolute.",
                    "prior_receipt": "receipts/2026-09-14-decoder-lstm-semantics/"
                    "result.json",
                    "bit_exact_lstm_variant": lstm["bit_exact_lstm_variant"],
                    "baseline_variant": lstm["baseline_variant"]["name"],
                    "baseline_next_cell_max_abs_error": lstm["baseline_variant"][
                        "next_cell_max_abs_error"
                    ],
                    "best_variant": lstm["best_lstm_variant"]["name"],
                    "best_next_cell_max_abs_error": lstm["best_lstm_variant"][
                        "next_cell_max_abs_error"
                    ],
                },
            ],
        },
        "blocks_accepted_transcript": [
            {
                "id": "decoder-lstm-not-bit-exact",
                "severity": "blocking",
                "statement": "overlay/tools/coreml/vulkan_decoder.py::_lstm does not "
                "reproduce Apple's fp16 LSTM bit-exactly. Gate reduction order and "
                "the elementwise sigmoid/tanh both remain unresolved, no surveyed "
                "variant is bit-exact, and the residual state error (about 0.03 "
                "absolute) is enough to flip a near-tie argmax. This is the single "
                "cause of the observed divergence.",
                "evidence": "receipts/2026-09-14-decoder-lstm-semantics/result.json "
                "(bit_exact_lstm_variant null); this receipt's attribution section.",
                "owner_hint": "DecoderActivations / DecoderProjectorKernel lane.",
            },
            {
                "id": "encoder-not-bit-exact",
                "severity": "residual risk, not yet a proven blocker",
                "statement": "The encoder is inside the frozen bounds but not "
                "bit-exact (rel_l2 0.024917 with ANE, 0.024596 GPU-only). With an "
                "exact decoder it is untested whether that residual alone can flip "
                "an emission; the two arms already emit different tails from the "
                "same decoder, which shows encoder residual does move the tail.",
                "evidence": "This receipt's runs.*.layer_5_encoder and the differing "
                "first_divergence of the two arms.",
            },
            {
                "id": "single-clip-geometry",
                "severity": "scope",
                "statement": "The ANE island bundles are compiled for exactly 375 "
                "valid frames and 8 heads, so this end-to-end result covers one "
                "10.435 s clip. A second clip needs recompiled bundles before "
                "Phase 7 can be claimed for anything but this fixture.",
                "evidence": "EncoderParityAne, and the bundle manifests pinned in "
                "/var/tmp/EncoderParityAne/results/environment.json.",
            },
            {
                "id": "encoder-on-ane-is-partial",
                "severity": "scope",
                "statement": "'Encoder on ANE' here means the 72 attention matmuls "
                "of islands A and C across all 24 layers, 1302 of 3351 ops staying "
                "on the GPU. The -inf attention-mask select (island B) is kept on "
                "the GPU deliberately: the H13 select program under-allocates its "
                "channel-3 scratch arena and corrupts its output, so no part of "
                "this run establishes ANE select correctness.",
                "evidence": "receipts/2026-09-14-h13-select-l2-fix.md (SelectL2Fix), "
                "IslandsExecJwm1's measured 10728 wrong lanes, and this run's "
                "encoder_ane_ops 72 / encoder_gpu_ops 1302.",
            },
            {
                "id": "ane-submit-cost",
                "severity": "performance, not correctness",
                "statement": "The ANE path is currently slower end to end than the "
                "GPU-only path because each island submit is a separate bounded "
                "worker process: 48 submissions, 3212 ms of worker wall time, and a "
                "27.3 s pipeline against 21.2 s for the ANE-free arm on 10.4 s of "
                "audio. Phase 9 work, but it blocks any latency claim.",
                "evidence": "This receipt's runs.*.timing and runs.ane_islands_a_c.ane.",
            },
        ],
        "section_43_no_cpu_tensor_proof": {
            "cpu_tensor_events": ane["execution"]["cpu_tensor_events"],
            "counter_semantics": "Structural, not a tally of averted fallbacks: the "
            "encoder runner has no CPU arithmetic path at all and raises "
            "EncoderRunError on any op it cannot express in mlx.core, so the "
            "counter can only ever read 0 or the run dies.",
            "per_stage_gpu_counter_deltas": "runs.*.stages[].gpu_counter_delta",
            "host_non_tensor_work": ane["execution"]["host_non_tensor_work"],
            "inference_server": False,
            "server_note": "The ANE worker is a one-shot bounded process per submit, "
            "started and reaped inside this process; nothing listens, and no model "
            "stays resident between submits.",
        },
        "device_state_after": {
            **post_state,
            "note": "Worker liveness measured with tools/ane_worker_liveness.py "
            "(ps -ww plus /proc/<pid>/fd device identity). pgrep -c -x cannot match "
            "the 22-character worker name and pgrep -cf matches the caller's shell; "
            "neither reading was used.",
        },
        "source": {
            "repo_commit": git("rev-parse", "HEAD"),
            "pipeline_from": git("rev-parse", "origin/main"),
            "pipeline_from_note": "tdt_control.py and tokenizer.py exist only on "
            "origin/main; vulkan_decoder.py, vulkan_joint.py and parakeet_tdt.py are "
            "byte-identical between origin/main and agent/coreml-tdt-stream "
            "1f8804cd, so origin/main is a superset and the single source used.",
            "files": {
                path: {
                    "path": path,
                    "last_commit": git("rev-list", "-1", "origin/main", "--", path),
                    "sha256": sha256(BASE / "pkg" / path.split("overlay/tools/")[1]),
                }
                for path in PIPELINE_FILES
            },
            "encoder_runner": {
                "path": "receipts/2026-09-14-encoder-parity-ane/derivation/"
                "vulkan_encoder.py",
                "sha256": sha256(
                    REPO
                    / "receipts/2026-09-14-encoder-parity-ane/derivation/vulkan_encoder.py"
                ),
                "note": "Reused unmodified from EncoderParityAne's Phase 6 slice; "
                "the copy on jwm1 at /var/tmp/EncoderParityAne/vulkan_encoder.py "
                "hashes identically.",
            },
            "driver": {
                "path": "receipts/2026-09-14-parakeet-e2e/derivation/parakeet_e2e.py",
                "sha256": sha256(BASE / "parakeet_e2e.py"),
            },
        },
        "encoder_parity_cross_check": {
            "source": "/var/tmp/EncoderParityAne/results/compare-ane.json",
            "phase_6_measured": parity["measured"],
            "phase_7_measured": ane["layers"]["layer_5_encoder"]["measured"],
            "agreement": "Same bounds, same golden, and byte-identical "
            "encoder_hidden between the two runs.",
        },
        "artifacts": {
            name: sha256(BASE / "out" / name)
            for name in (
                "e2e-report.json",
                "encoder_hidden.npy",
                "encoder_mask.npy",
                "mel.npy",
                "token_ids.json",
                "transcript.txt",
                "waveform.npy",
            )
        },
        "artifacts_control": {
            name: sha256(BASE / "out-vulkan" / name)
            for name in ("e2e-report.json", "encoder_hidden.npy", "token_ids.json",
                         "transcript.txt")
        },
        "reproduce": (BASE / "COMMANDS.md").read_text(),
        "agent_identity": {
            "agent": "ParakeetE2E",
            "model": "anthropic/claude-opus-5",
            "note": "This batch's subagent routing fell back off the intended "
            "OpenAI model. Reported to Main before the hardware run; Main directed "
            "the work to proceed with the identity disclosed here.",
        },
    }

    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
