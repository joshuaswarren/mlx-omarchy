#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Phase 7: the complete Parakeet TDT pipeline in one Linux process.

docs/plans/2026-09-12-coreml-parakeet-ane-plan.md section 61 phase 7 and
section 40 layers 2, 5, 6 and 7. One process carries every stage:

  1. audio      pinned FLAC -> int16 PCM (host codec) -> fp32 on the GPU
  2. mel        overlay/tools/coreml/vulkan_mel.py, mx.fast.metal_kernel only
  3. encoder    the MIL encoder with attention islands A and C on the ANE
                (receipts/2026-09-14-encoder-parity-ane/derivation/vulkan_encoder.py)
  4. decoder    overlay/tools/coreml/vulkan_decoder.py on mx.gpu
     joint      overlay/tools/coreml/vulkan_joint.py on mx.gpu
     control    overlay/tools/coreml/parakeet_tdt.py::tdt_decode (GPU-resident
                loop by default, host control behind --tdt-host / env)
  5. tokenizer  overlay/tools/coreml/tokenizer.py, hash-pinned detokenization

Stage 3 consumes stage 2's device tensors and stage 4 consumes stage 3's, so
nothing here is replayed from the macOS capture: the capture is only ever read
as the comparison golden and as the authenticated identity of the inputs.

Section 43: every tensor stage issues its work inside ``mx.stream(mx.gpu)`` or
on the ANE, and the per-stage Vulkan counter deltas are recorded. The two
host-side, non-tensor costs are named rather than blurred into the tensor
accounting: the FLAC codec, and the ANE worker's file-based tensor transport.

There is no inference server: the ANE worker is a bounded one-shot process per
submit, and every other stage is in-process MLX.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class E2eError(RuntimeError):
    """The run cannot continue; the reason is named."""


# --------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise E2eError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stages:
    """Wall time and Vulkan counter deltas per named stage, in order."""

    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.records: list[dict] = []

    def run(self, name: str, work):
        before = self._snapshot()
        started = time.monotonic_ns()
        result = work()
        elapsed = time.monotonic_ns() - started
        after = self._snapshot()
        self.records.append(
            {
                "stage": name,
                "wall_ns": elapsed,
                "wall_ms": round(elapsed / 1e6, 3),
                "gpu_counter_delta": {k: after[k] - before[k] for k in after},
            }
        )
        return result

    def total_ns(self) -> int:
        return sum(record["wall_ns"] for record in self.records)


def tensor_stats(actual: np.ndarray, golden: np.ndarray) -> dict:
    if actual.shape != golden.shape:
        raise E2eError(f"shape {actual.shape} != golden {golden.shape}")
    a = actual.astype(np.float32)
    g = golden.astype(np.float32)
    diff = np.abs(a - g)
    denom = float(np.linalg.norm(g))
    return {
        "shape": list(a.shape),
        "bit_exact": bool(np.array_equal(actual, golden)),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_l2_err": float(np.linalg.norm(a - g) / denom) if denom else None,
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }


def decode_flac(path: Path) -> tuple[np.ndarray, int, str]:
    """Return int16 PCM, sample rate, and the decoder that produced them."""
    try:
        import soundfile
    except ImportError:
        pass
    else:
        pcm, rate = soundfile.read(str(path), dtype="int16")
        return np.ascontiguousarray(pcm), int(rate), f"soundfile {soundfile.__version__}"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["channels"]) != 1:
        raise E2eError("pinned fixture must be mono")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec",
         "pcm_s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    version = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    return np.frombuffer(raw, dtype="<i2"), int(stream["sample_rate"]), version


def authenticated(capture: Path) -> dict[str, str]:
    """Parse the capture's own manifest and verify every file it names."""
    entries: dict[str, str] = {}
    for line in (capture / "manifest.sha256").read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or Path(relative).name != relative:
            raise E2eError(f"capture manifest entry is invalid: {line!r}")
        entries[relative] = digest
    for relative, digest in entries.items():
        if sha256_file(capture / relative) != digest:
            raise E2eError(f"capture file {relative} does not match its manifest hash")
    return entries


# ------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True, help="macOS capture dir")
    parser.add_argument("--model", type=Path, required=True, help="pinned model revision")
    parser.add_argument("--pkg", type=Path, required=True, help="dir holding coreml/")
    parser.add_argument("--encoder-runner", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="depalettized root")
    parser.add_argument("--bundles", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--libane", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--ane-reference", type=Path, help="prior ANE encoder_hidden.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=int, default=20000)
    parser.add_argument("--no-ane", action="store_true")
    parser.add_argument("--tdt-host", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pkg.resolve()))
    sys.path.insert(0, str((args.pkg / "coreml").resolve()))

    import mlx.core as mx
    from importlib.metadata import distribution

    from coreml.parakeet_tdt import DecoderStep, JointDecision, tdt_decode
    from coreml.reference import ReferenceLock
    from coreml.tokenizer import ParakeetTokenizer
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import pack_step_weights, run_step
    from coreml.vulkan_joint import run_joint
    from coreml.vulkan_mel import extract_chunk_features, trace_snapshot

    encoder_module = load_module(args.encoder_runner.resolve(), "phase7_encoder")

    mx.set_default_device(mx.gpu)
    device_info = mx.device_info()
    device_name = str(device_info.get("device_name", ""))
    if "Apple" not in device_name:
        raise E2eError(f"phase 7 needs the Apple GPU, got {device_name!r}")

    lock = ReferenceLock.load(args.pkg / "coreml" / "parakeet-reference.lock")
    contract = lock.numerical_contract
    if contract is None:
        raise E2eError("the reference lock has no frozen numerical contract")
    tokenizer_sha = next(
        item.sha256 for item in lock.files if item.path == "tokenizer.json"
    )

    manifest = authenticated(args.golden.resolve())
    audio_sha = sha256_file(args.audio)
    if audio_sha != lock.audio.sha256:
        raise E2eError(
            f"audio fixture sha256 {audio_sha} is not the pinned {lock.audio.sha256}"
        )

    golden = {
        name: np.load(args.golden / name, allow_pickle=False)
        for name in (
            "waveform.npy", "mel.npy", "mel_mask.npy",
            "encoder_input_features.npy", "encoder_input_mask.npy",
            "encoder_hidden.npy", "encoder_mask.npy",
        )
    }
    golden_tokens = json.loads((args.golden / "token_ids.json").read_text())
    golden_transcript = (args.golden / "transcript.txt").read_text()

    stages = Stages(trace_snapshot)

    # ------------------------------------------------------------- 1. audio
    def stage_audio():
        pcm, rate, decoder_name = decode_flac(args.audio)
        if rate != lock.audio.sample_rate:
            raise E2eError(f"fixture sample rate {rate} != {lock.audio.sample_rate}")
        with mx.stream(mx.gpu):
            waveform = mx.array(pcm).astype(mx.float32) / 32768.0
        mx.eval(waveform)
        return waveform, pcm.size, decoder_name

    waveform, sample_count, decoder_name = stages.run("audio_load", stage_audio)

    # --------------------------------------------------------------- 2. mel
    def stage_mel():
        result = extract_chunk_features(waveform)
        mx.eval(result.mel, result.mask, result.encoder_features, result.encoder_mask)
        return result

    mel_result = stages.run("mel_frontend", stage_mel)

    # ----------------------------------------------------------- 3. encoder
    island = None
    if not args.no_ane:
        for name in ("bundles", "worker", "libane", "scratch"):
            if getattr(args, name) is None:
                raise E2eError(f"--{name} is required unless --no-ane is given")
        island = encoder_module.AneIsland(
            args.worker, args.libane, args.bundles, args.scratch, args.deadline_ms
        )
    runner = encoder_module.EncoderRunner(
        args.source / "model.mil", args.source / "model-root", island
    )

    def stage_encoder():
        return runner.run(
            inputs={
                "input_features": mel_result.encoder_features,
                "attention_mask": mel_result.encoder_mask,
            },
            wanted={"encoder_hidden", "encoder_mask"},
            stop_after="encoder_mask",
        )

    encoded = stages.run("encoder_ane" if island else "encoder_gpu", stage_encoder)
    encoder_hidden = encoded["encoder_hidden"].astype(mx.float32)
    encoder_mask = encoded["encoder_mask"].astype(mx.int32)
    mx.eval(encoder_hidden, encoder_mask)

    # ------------------------------------------- 4. decoder, joint, control
    decoder = stages.run(
        "decoder_load", lambda: load_decoder(args.model / "decoder.mlpackage")
    )
    fused_packed = pack_step_weights(decoder, args.model / "joint.mlpackage")
    counts = {"decoder_calls": 0, "joint_calls": 0}
    decoder_ns = 0
    joint_ns = 0
    frame_holder = [0]
    fused = {"frame": None, "state": None, "tok": None, "dur": None}

    def decoder_callback(token_id, current_hidden, current_cell):
        nonlocal decoder_ns
        counts["decoder_calls"] += 1
        started = time.monotonic_ns()
        state_out, tok, dur, _, _ = run_step(
            fused_packed, current_hidden, current_cell, token_id,
            encoder_hidden, frame_holder[0],
        )
        mx.eval(state_out)
        decoder_ns += time.monotonic_ns() - started
        dec_state = state_out[0:640].reshape(1, 640)
        fused["frame"] = frame_holder[0]
        fused["state"] = dec_state
        fused["tok"] = tok
        fused["dur"] = dur
        return DecoderStep(
            dec_state,
            state_out[640:1920].reshape(2, 1, 640),
            state_out[1920:3200].reshape(2, 1, 640),
        )

    def joint_callback(frame_index, decoder_state):
        nonlocal joint_ns
        counts["joint_calls"] += 1
        frame_holder[0] = frame_index
        started = time.monotonic_ns()
        if fused["frame"] == frame_index and fused["state"] is decoder_state:
            joint_ns += time.monotonic_ns() - started
            return JointDecision(fused["tok"], fused["dur"])
        state_out, tok, dur, _, _ = run_step(
            fused_packed, None, None, 0, encoder_hidden, frame_index,
            skip_lstm=True, dec_in=decoder_state,
        )
        mx.eval(state_out)
        joint_ns += time.monotonic_ns() - started
        return JointDecision(tok, dur)

    def stage_tdt():
        with mx.stream(mx.gpu):
            hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
            cell = mx.zeros((2, 1, 640), dtype=mx.float32)
        mx.eval(hidden, cell)
        return tdt_decode(
            packed=fused_packed,
            encoder=encoder_hidden,
            valid_frames=int(encoder_hidden.shape[1]),
            config=lock.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
            force_host=args.tdt_host,
        )

    tdt = stages.run("tdt_decode", stage_tdt)

    # ----------------------------------------------------------- 5. tokenizer
    def stage_tokenizer():
        tokenizer = ParakeetTokenizer.load(
            args.model / "tokenizer.json", expected_sha256=tokenizer_sha
        )
        return tokenizer, tokenizer.decode(tdt.token_ids)

    tokenizer, transcript = stages.run("detokenize", stage_tokenizer)

    # ------------------------------------------------------------ comparison
    mel_host = np.asarray(mel_result.mel)
    mel_mask_host = np.asarray(mel_result.mask).astype(np.int32)
    features_host = np.asarray(mel_result.encoder_features)
    features_mask_host = np.asarray(mel_result.encoder_mask).astype(np.int32)
    hidden_host = np.asarray(encoder_hidden).astype(np.float32)
    mask_host = np.asarray(encoder_mask).astype(np.int32)

    np.save(args.out / "waveform.npy", np.asarray(waveform))
    np.save(args.out / "mel.npy", mel_host)
    np.save(args.out / "encoder_hidden.npy", hidden_host)
    np.save(args.out / "encoder_mask.npy", mask_host)
    (args.out / "transcript.txt").write_text(transcript)
    (args.out / "token_ids.json").write_text(
        json.dumps(
            {
                "token_ids": list(tdt.token_ids),
                "frame_indices": list(tdt.frame_indices),
                "durations": list(tdt.durations),
            },
            indent=2,
        )
        + "\n"
    )

    encoder_checks = {
        "encoder_max_abs_err": contract.encoder_max_abs_err,
        "encoder_mean_abs_err": contract.encoder_mean_abs_err,
        "encoder_rel_l2_err": contract.encoder_rel_l2_err,
    }
    encoder_stats = tensor_stats(hidden_host, golden["encoder_hidden.npy"])
    encoder_bounds = {
        name: {
            "measured": encoder_stats[name.removeprefix("encoder_")],
            "bound": bound,
            "pass": bool(encoder_stats[name.removeprefix("encoder_")] <= bound),
        }
        for name, bound in encoder_checks.items()
    }
    encoder_bounds["nan_count"] = {
        "measured": encoder_stats["nan"],
        "bound": contract.nan_count_allowed,
        "pass": encoder_stats["nan"] == contract.nan_count_allowed,
    }
    encoder_bounds["inf_count"] = {
        "measured": encoder_stats["inf"],
        "bound": contract.inf_count_allowed,
        "pass": encoder_stats["inf"] == contract.inf_count_allowed,
    }

    actual_tokens = list(tdt.token_ids)
    native_tokens = list(golden_tokens["token_ids"])
    token_divergence = None
    for index in range(max(len(actual_tokens), len(native_tokens))):
        mine = actual_tokens[index] if index < len(actual_tokens) else None
        theirs = native_tokens[index] if index < len(native_tokens) else None
        if mine != theirs:
            token_divergence = {
                "emission_index": index,
                "actual": {
                    "token_id": mine,
                    "piece": tokenizer.id_to_piece[mine] if mine is not None else None,
                    "frame_index": tdt.frame_indices[index] if index < len(actual_tokens) else None,
                    "duration": tdt.durations[index] if index < len(actual_tokens) else None,
                },
                "native": {
                    "token_id": theirs,
                    "piece": tokenizer.id_to_piece[theirs] if theirs is not None else None,
                    "frame_index": golden_tokens["frame_indices"][index] if theirs is not None else None,
                    "duration": golden_tokens["durations"][index] if theirs is not None else None,
                },
            }
            break

    prefix = 0
    for mine, theirs in zip(actual_tokens, native_tokens):
        if mine != theirs:
            break
        prefix += 1

    tokens_match = actual_tokens == native_tokens
    transcript_match = transcript == golden_transcript
    ane_reference = None
    if args.ane_reference is not None and args.ane_reference.exists():
        ane_reference = {
            "path": str(args.ane_reference),
            "sha256": sha256_file(args.ane_reference),
            **tensor_stats(hidden_host, np.load(args.ane_reference)),
        }

    libmlx = Path(distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so"))
    report = {
        "schema": "mlx-omarchy.parakeet-e2e/1",
        "phase": "7",
        "status": "match" if (tokens_match and transcript_match) else "diverged",
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "mlx": {
            "version": mx.__version__,
            "device": device_name,
            "architecture": device_info.get("architecture"),
            "libmlx_path": str(libmlx),
            "libmlx_sha256": sha256_file(libmlx),
            "core_path": mx.__file__,
            "core_sha256": sha256_file(Path(mx.__file__)),
            "default_device": str(mx.default_device()),
        },
        "inputs": {
            "audio": {
                "path": str(args.audio),
                "sha256": audio_sha,
                "matches_lock": True,
                "url": lock.audio.url,
                "samples": int(sample_count),
                "sample_rate": lock.audio.sample_rate,
                "decoder": decoder_name,
            },
            "model": {
                "repo": lock.model_repo,
                "revision": lock.model_revision,
                "quantization": lock.model_quantization,
                "decoder_mlpackage_sha256": sha256_file(
                    args.model / "decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"
                ),
                "joint_mlpackage_sha256": sha256_file(
                    args.model / "joint.mlpackage/Data/com.apple.CoreML/model.mlmodel"
                ),
                "tokenizer_sha256": tokenizer.sha256,
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
            "encoder_source": {
                "mil_sha256": sha256_file(args.source / "model.mil"),
                "root": str(args.source),
            },
            "golden_capture": {
                "path": str(args.golden),
                "manifest_sha256": sha256_file(args.golden / "manifest.sha256"),
                "authenticated_files": len(manifest),
                "token_ids_sha256": sha256_file(args.golden / "token_ids.json"),
                "transcript_sha256": sha256_file(args.golden / "transcript.txt"),
                "encoder_hidden_sha256": sha256_file(args.golden / "encoder_hidden.npy"),
            },
        },
        "stages": stages.records,
        "timing": {
            "total_pipeline_ms": round(stages.total_ns() / 1e6, 3),
            "decoder_total_ms": round(decoder_ns / 1e6, 3),
            "joint_total_ms": round(joint_ns / 1e6, 3),
            "decoder_mean_ms": round(decoder_ns / 1e6 / max(counts["decoder_calls"], 1), 3),
            "joint_mean_ms": round(joint_ns / 1e6 / max(counts["joint_calls"], 1), 3),
        },
        "execution": {
            "inference_server": False,
            "ane_mode": island is not None,
            "encoder_ops_executed": runner.executed,
            "encoder_gpu_ops": runner.gpu_ops,
            "encoder_ane_ops": runner.ane_ops,
            "encoder_layers": runner.layers,
            "cpu_tensor_events": runner.cpu_tensor_events,
            "decoder_calls": counts["decoder_calls"],
            "joint_calls": counts["joint_calls"],
            "valid_encoder_frames": int(encoder_hidden.shape[1]),
            "control": tdt.decode_path,
            "tdt_fallback_reason": tdt.fallback_reason,
            "recorded_decisions_injected": False,
            "host_non_tensor_work": [
                f"FLAC codec: {decoder_name}, {int(sample_count)} int16 samples; "
                "the fp32 scaling and every later tensor op run on the GPU",
                "ANE island transport: input and output tensors cross to the "
                "bounded worker as files; process I/O, not arithmetic",
                "TDT control: scalar token, duration and frame decisions only",
                "detokenization: string concatenation over emitted ids",
                "comparison and reporting after the pipeline, in numpy",
            ],
        },
        "ane": (
            {
                "submissions": island.submissions,
                "worker_starts": island.worker_starts,
                "timeouts": island.timeouts,
                "input_bytes": island.input_bytes,
                "output_bytes": island.output_bytes,
                "exec_ns": island.exec_ns,
                "exec_ms": round(island.exec_ns / 1e6, 3),
                "worker": str(args.worker),
                "worker_sha256": sha256_file(args.worker),
                "libane": str(args.libane),
                "libane_sha256": sha256_file(args.libane),
                "bundles": sorted(
                    {record["bundle"] for record in island.log}
                ),
                "log": island.log,
            }
            if island is not None
            else None
        ),
        "layers": {
            "layer_2_preprocessing": {
                "mel": tensor_stats(mel_host, golden["mel.npy"]),
                "mel_mask_exact": bool(
                    np.array_equal(mel_mask_host, golden["mel_mask.npy"].astype(np.int32))
                ),
                "encoder_input_features": tensor_stats(
                    features_host, golden["encoder_input_features.npy"]
                ),
                "encoder_input_mask_exact": bool(
                    np.array_equal(
                        features_mask_host,
                        golden["encoder_input_mask.npy"].astype(np.int32),
                    )
                ),
            },
            "layer_5_encoder": {
                "measured": encoder_stats,
                "per_bound": encoder_bounds,
                "encoder_mask_exact": bool(
                    np.array_equal(mask_host, golden["encoder_mask.npy"].astype(np.int32))
                ),
                "encoder_mask_sum": int(mask_host.sum()),
                "all_bounds_pass": bool(
                    all(item["pass"] for item in encoder_bounds.values())
                ),
                "vs_prior_ane_encoder_run": ane_reference,
            },
            "layer_6_decoder_sequence": {
                "requirement": "token_ids_must_match_exactly",
                "required": contract.token_ids_must_match_exactly,
                "actual_emissions": len(actual_tokens),
                "native_emissions": len(native_tokens),
                "matching_prefix_length": prefix,
                "tokens_match": tokens_match,
                "durations_match": list(tdt.durations) == list(golden_tokens["durations"]),
                "frame_indices_match": list(tdt.frame_indices)
                == list(golden_tokens["frame_indices"]),
                "first_divergence": token_divergence,
                "actual": {
                    "token_ids": actual_tokens,
                    "frame_indices": list(tdt.frame_indices),
                    "durations": list(tdt.durations),
                },
                "native": golden_tokens,
            },
            "layer_7_end_to_end_text": {
                "requirement": "transcript_must_match_exactly",
                "required": contract.transcript_must_match_exactly,
                "transcript": transcript,
                "native_transcript": golden_transcript,
                "transcript_match": transcript_match,
                "actual_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "native_sha256": hashlib.sha256(golden_transcript.encode()).hexdigest(),
            },
        },
    }

    (args.out / "e2e-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "status": report["status"],
            "tokens_match": tokens_match,
            "transcript_match": transcript_match,
            "matching_prefix_length": prefix,
            "actual_emissions": len(actual_tokens),
            "native_emissions": len(native_tokens),
            "encoder_bounds_pass": report["layers"]["layer_5_encoder"]["all_bounds_pass"],
            "mel_bit_exact": report["layers"]["layer_2_preprocessing"]["mel"]["bit_exact"],
            "ane_submissions": island.submissions if island else 0,
            "cpu_tensor_events": runner.cpu_tensor_events,
            "total_pipeline_ms": report["timing"]["total_pipeline_ms"],
            "transcript": transcript,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
