#!/usr/bin/env python3
"""Apple-GPU localization of TDT decision 142 (8029 vs 7892)."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

from coreml.parakeet_tdt import DecoderStep, JointDecision, greedy_tdt_decode
from coreml.pinned_component import load_pinned_component
from coreml.reference import ReferenceLock
from coreml.vulkan_decoder import load_decoder
from coreml.vulkan_joint import run_joint
from coreml.vulkan_decoder_step import pack_step_weights, run_step


class _TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("gpu_primitive_dispatches", ctypes.c_uint64),
        ("vk_submissions", ctypes.c_uint64),
        ("vk_buffer_copies", ctypes.c_uint64),
        ("vk_buffer_fills", ctypes.c_uint64),
        ("vk_compute_dispatches", ctypes.c_uint64),
        ("omarchy_finalize_calls", ctypes.c_uint64),
        ("commit_calls_with_work", ctypes.c_uint64),
        ("commit_calls_noop", ctypes.c_uint64),
    ]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trace_snapshot() -> dict[str, int]:
    snapshot = _TraceSnapshot()
    library = ctypes.CDLL(
        str(importlib.metadata.distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so"))
    )
    function = library.mlx_omarchy_trace_snapshot
    function.argtypes = [ctypes.POINTER(_TraceSnapshot)]
    function.restype = None
    function(ctypes.byref(snapshot))
    return {name: int(getattr(snapshot, name)) for name, _ in snapshot._fields_}


def _metrics(actual: np.ndarray, golden: np.ndarray) -> dict:
    delta = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    return {
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "bit_exact": bool(actual.tobytes() == golden.tobytes()),
    }


def _require_apple_gpu() -> dict:
    info = mx.device_info()
    name = str(info.get("device_name", ""))
    if "Apple M1" not in name:
        raise SystemExit(f"expected Apple M1 Vulkan device, got {name!r}")
    if "llvmpipe" in name.lower():
        raise SystemExit(f"refusing software Vulkan device {name!r}")
    return {"device_name": name, "architecture": info.get("architecture")}


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _argmax_pair(token_logits, duration_logits):
    with mx.stream(mx.gpu):
        token_id = mx.argmax(token_logits)
        duration_index = mx.argmax(duration_logits)
    mx.eval(token_id, duration_index)
    return int(token_id.item()), int(duration_index.item())


def _joint_variants(hidden_fp16, weight, bias):
    variants = {}
    with mx.stream(mx.gpu):
        variants["baseline_fp16_matmul_then_fp16_bias"] = mx.add(
            mx.matmul(hidden_fp16, weight.T), bias
        )
        variants["mx_addmm_fp16"] = mx.addmm(bias, hidden_fp16, weight.T)
        hidden32 = hidden_fp16.astype(mx.float32)
        weight32 = weight.astype(mx.float32)
        bias32 = bias.astype(mx.float32)
        variants["explicit_fp32_dot_then_fp16_bias"] = (
            mx.matmul(hidden32, weight32.T).astype(mx.float16) + bias
        ).astype(mx.float16)
        variants["fused_fp32_dot_and_bias_then_fp16"] = (
            mx.matmul(hidden32, weight32.T) + bias32
        ).astype(mx.float16)
    for name, value in list(variants.items()):
        mx.eval(value)
        token = np.asarray(value[:, :8193], dtype=np.float32)
        duration = np.asarray(value[:, 8193:8198], dtype=np.float32)
        variants[name] = {
            "token_id": int(token.argmax()),
            "native_7892_logit": float(token[0, 7892]),
            "winner_logit": float(token[0, int(token.argmax())]),
            "winner_minus_native": float(
                token[0, int(token.argmax())] - token[0, 7892]
            ),
            "duration_index": int(duration.argmax()),
        }
    return variants


def main() -> int:
    capture = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/capture/ane")
    model = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/model")
    started = time.monotonic()
    before = _trace_snapshot()
    device = _require_apple_gpu()
    lock = ReferenceLock.load()
    decoder = load_decoder(model / "decoder.mlpackage")
    joint_package = model / "joint.mlpackage"
    packed = pack_step_weights(decoder, joint_package)
    component = load_pinned_component(model / "decoder.mlpackage", "decoder")
    projector = mx.array(component.constant("projector_weight_to_fp16"))
    projector_bias = mx.array(component.constant("projector_bias_to_fp16"))
    mx.eval(projector, projector_bias)

    encoder_host = _load_npy(capture / "encoder_hidden.npy")
    decisions = json.loads((capture / "tdt_decisions.json").read_text())["decisions"]
    traces = json.loads((capture / "tdt_tensors.json").read_text())["traces"]
    with mx.stream(mx.gpu):
        encoder = mx.array(encoder_host)
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(encoder, hidden, cell)

    decoder_traces = []
    joint_native_traces = []
    for trace in traces:
        paths = trace["tensor_paths"]
        native_joint_state = _load_npy(capture / paths["joint_decoder_state"])
        native_encoder = _load_npy(capture / paths["joint_encoder_frame"])
        joint_out = run_joint(
            mx.array(native_encoder),
            mx.array(native_joint_state),
            package_path=joint_package,
        )
        token_id, duration_index = _argmax_pair(
            joint_out.token_logits, joint_out.duration_logits
        )
        joint_native_traces.append(
            {
                "index": trace["index"],
                "token_match": token_id == trace["token_id"],
                "duration_match": duration_index == trace["duration_index"],
                "actual_token_id": token_id,
                "native_token_id": trace["token_id"],
            }
        )
        if not trace.get("ran_decoder"):
            continue
        native_ids = _load_npy(capture / paths["decoder_input_ids"])
        native_hidden = _load_npy(capture / paths["decoder_hidden"])
        native_cell = _load_npy(capture / paths["decoder_cell"])
        native_next_hidden = _load_npy(capture / paths["decoder_next_hidden"])
        native_next_cell = _load_npy(capture / paths["decoder_next_cell"])
        native_output = _load_npy(capture / paths["decoder_output_hidden"])
        result = decoder(
            mx.array(native_ids), mx.array(native_hidden), mx.array(native_cell)
        )
        mx.eval(result.decoder_hidden, result.next_hidden, result.next_cell)
        actual_hidden = np.asarray(result.next_hidden)
        actual_cell = np.asarray(result.next_cell)
        actual_output = np.asarray(result.decoder_hidden)
        with mx.stream(mx.gpu):
            native_layer1 = mx.array(native_next_hidden[1:2])
            projector_on_native = (
                native_layer1.astype(mx.float16) @ projector.T + projector_bias
            ).astype(mx.float32)
        mx.eval(projector_on_native)
        decoder_traces.append(
            {
                "index": trace["index"],
                "next_hidden": _metrics(actual_hidden, native_next_hidden),
                "next_cell": _metrics(actual_cell, native_next_cell),
                "decoder_hidden": _metrics(actual_output, native_output),
                "projector_on_native_lstm": _metrics(
                    np.asarray(projector_on_native), native_output
                ),
            }
        )

    # Native-injected recurrent path through decision 142.
    native_hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    native_cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(native_hidden, native_cell)
    target = None
    decoder_state = None
    input_token = lock.tdt.blank_token_id
    decoder_state_valid = False
    for index, decision in enumerate(decisions):
        if not decoder_state_valid or input_token != lock.tdt.blank_token_id:
            state_out, _, _, _, _ = run_step(
                packed, native_hidden, native_cell, input_token,
                encoder, int(decision["frame_before"]),
            )
            decoder_state = state_out[0:640].reshape(1, 640)
            native_hidden = state_out[640:1920].reshape(2, 1, 640)
            native_cell = state_out[1920:3200].reshape(2, 1, 640)
            decoder_state_valid = True
        frame = int(decision["frame_before"])
        joint_out = run_joint(
            encoder[:, frame, :],
            decoder_state,
            package_path=joint_package,
        )
        actual_token, actual_duration = _argmax_pair(
            joint_out.token_logits, joint_out.duration_logits
        )
        if index == 142:
            token = np.asarray(joint_out.token_logits)
            duration = np.asarray(joint_out.duration_logits)
            hidden_fp16 = decoder_state.astype(mx.float16)
            joint_component = load_pinned_component(joint_package, "joint")
            weight = mx.array(joint_component.constant("head_weight_to_fp16"))
            bias = mx.array(joint_component.constant("head_bias_to_fp16"))
            mx.eval(hidden_fp16, weight, bias)
            target = {
                "decision_index": index,
                "decoder_input_token": input_token,
                "actual_token_id": actual_token,
                "native_token_id": int(decision["token_id"]),
                "actual_logit": float(token[0, actual_token]),
                "native_token_logit": float(token[0, int(decision["token_id"])]),
                "actual_minus_native_logit": float(
                    token[0, actual_token] - token[0, int(decision["token_id"])]
                ),
                "duration_index_actual": actual_duration,
                "duration_index_native": int(decision["duration_index"]),
                "duration_logits": [float(x) for x in duration[0].tolist()],
                "decoder_state_sha256": _sha256(np.asarray(decoder_state).tobytes()),
                "encoder_frame_sha256": _sha256(
                    np.asarray(encoder[:, frame, :]).tobytes()
                ),
                "joint_variants": _joint_variants(hidden_fp16, weight, bias),
            }
            break
        if int(decision["token_id"]) != lock.tdt.blank_token_id:
            input_token = int(decision["token_id"])

    # Free decode through first divergence.
    with mx.stream(mx.gpu):
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)
    decoder_calls = 0
    joint_calls = 0
    frame_holder = [0]
    fused = {"frame": None, "state": None, "tok": None, "dur": None}

    def decoder_callback(token_id, current_hidden, current_cell):
        nonlocal decoder_calls
        decoder_calls += 1
        state_out, tok, dur, _, _ = run_step(
            packed, current_hidden, current_cell, token_id,
            encoder, frame_holder[0],
        )
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

    def joint_callback(frame_index, state):
        nonlocal joint_calls
        joint_calls += 1
        frame_holder[0] = frame_index
        if fused["frame"] == frame_index and fused["state"] is state:
            return JointDecision(fused["tok"], fused["dur"])
        result = run_joint(
            encoder[:, frame_index, :],
            state,
            package_path=joint_package,
        )
        token_id, duration_index = _argmax_pair(
            result.token_logits, result.duration_logits
        )
        return JointDecision(token_id, duration_index)

    if "--tdt-loop" in sys.argv:
        from coreml.vulkan_tdt_loop import run_tdt_loop

        output = run_tdt_loop(
            packed,
            encoder,
            valid_frames=int(encoder.shape[1]),
            config=lock.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
        )
    else:
        output = greedy_tdt_decode(
            valid_frames=int(encoder.shape[1]),
            config=lock.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
        )
    native_emissions = [
        {
            "token_id": item["token_id"],
            "frame_index": item["frame_before"],
            "duration": item["duration"],
        }
        for item in decisions
        if item["token_id"] != lock.tdt.blank_token_id
    ]
    actual_emissions = [
        {"token_id": token, "frame_index": frame, "duration": duration}
        for token, frame, duration in zip(
            output.token_ids, output.frame_indices, output.durations, strict=True
        )
    ]
    divergence = None
    for index, (actual, native) in enumerate(zip(actual_emissions, native_emissions)):
        if actual != native:
            divergence = {"emission_index": index, "actual": actual, "native": native}
            break
    after = _trace_snapshot()
    receipt = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "mlx_version": mx.__version__,
        "device": device,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_trace_delta": {k: after[k] - before[k] for k in before},
        "free_decode": {
            "decoder_calls": decoder_calls,
            "joint_calls": joint_calls,
            "first_divergence": divergence,
        },
        "joint_on_native_decoder_state": {
            "traces": len(joint_native_traces),
            "token_matches": sum(item["token_match"] for item in joint_native_traces),
            "duration_matches": sum(
                item["duration_match"] for item in joint_native_traces
            ),
        },
        "decoder_on_native_inputs": decoder_traces,
        "decision_142": target,
        "source_sha256": {
            "vulkan_decoder.py": _sha256(
                Path(load_decoder.__code__.co_filename).read_bytes()
            ),
            "vulkan_joint.py": _sha256(Path(run_joint.__code__.co_filename).read_bytes()),
        },
    }
    Path("/tmp/vulkan-tdt-142/result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
