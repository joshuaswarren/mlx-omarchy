# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host control for the pinned Parakeet greedy TDT decoder.

The state machine follows ``GreedyTDTDecoder.swift`` from
``mweinbach/parakeet-coreml-swift`` at
``75aec2a1c991319657ff4dec5f602c12da6c5012`` (Apache-2.0). Tensor work stays
behind the decoder and joint callbacks; this module only threads opaque state
handles and applies scalar token, duration, and frame decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference import ReferenceLock, TdtConfig


class TdtControlError(ValueError):
    """A named Parakeet host-control contract failure."""


@dataclass(frozen=True)
class DecoderStep:
    """Opaque outputs returned by one GPU decoder invocation."""

    decoder_state: Any
    hidden: Any
    cell: Any


@dataclass(frozen=True)
class JointDecision:
    """Scalar argmax results returned by one GPU joint invocation."""

    token_id: int
    duration_index: int


@dataclass(frozen=True)
class TdtOutput:
    """Emissions and recurrent state produced by the host control loop."""

    token_ids: list[int]
    frame_indices: list[int]
    durations: list[int]
    hidden: Any
    cell: Any


def _fail(message: str) -> TdtControlError:
    return TdtControlError(f"[omarchy-parakeet] {message}")


def _validate(valid_frames: int, config: TdtConfig) -> None:
    if not isinstance(valid_frames, int) or valid_frames < 0:
        raise _fail(f"valid frame count must be a nonnegative integer, got {valid_frames!r}")
    if config.vocab_size <= 0:
        raise _fail(f"vocabulary size must be positive, got {config.vocab_size}")
    if not 0 <= config.blank_token_id < config.vocab_size:
        raise _fail(
            f"blank token id {config.blank_token_id} is outside vocabulary size "
            f"{config.vocab_size}"
        )
    if not config.durations or any(duration < 0 for duration in config.durations):
        raise _fail("duration classes must be a nonempty list of nonnegative integers")
    if config.max_symbols_per_step <= 0:
        raise _fail(
            "max symbols per step must be positive, got "
            f"{config.max_symbols_per_step}"
        )


def greedy_tdt_decode(
    *,
    valid_frames: int,
    config: TdtConfig,
    initial_hidden: Any,
    initial_cell: Any,
    run_decoder: Callable[[int, Any, Any], DecoderStep],
    run_joint: Callable[[int, Any], JointDecision],
) -> TdtOutput:
    """Apply the pinned greedy TDT control loop to GPU-backed callbacks.

    ``run_decoder`` receives the last emitted token and the current hidden/cell
    handles. ``run_joint`` receives the encoder frame index and the opaque
    decoder-state handle. The callbacks own all tensor access and execution.
    """

    _validate(valid_frames, config)
    blank = config.blank_token_id
    input_token = blank
    hidden = initial_hidden
    cell = initial_cell
    decoder_state = None
    decoder_state_valid = False
    token_ids: list[int] = []
    frame_indices: list[int] = []
    emitted_durations: list[int] = []
    frame = 0

    while frame < valid_frames:
        symbols = 0
        advanced = False
        while symbols < config.max_symbols_per_step:
            if not decoder_state_valid or input_token != blank:
                step = run_decoder(input_token, hidden, cell)
                if not isinstance(step, DecoderStep):
                    raise _fail("decoder callback returned an invalid result")
                decoder_state = step.decoder_state
                hidden = step.hidden
                cell = step.cell
                decoder_state_valid = True

            decision = run_joint(frame, decoder_state)
            if not isinstance(decision, JointDecision):
                raise _fail("joint callback returned an invalid result")
            if not 0 <= decision.token_id < config.vocab_size:
                raise _fail(
                    f"token id {decision.token_id} is outside vocabulary size "
                    f"{config.vocab_size}"
                )
            if not 0 <= decision.duration_index < len(config.durations):
                raise _fail(
                    f"duration index {decision.duration_index} is outside "
                    f"{len(config.durations)} classes"
                )
            duration = config.durations[decision.duration_index]

            if decision.token_id == blank:
                frame += max(duration, 1)
                advanced = True
                break

            token_ids.append(decision.token_id)
            frame_indices.append(frame)
            emitted_durations.append(duration)
            input_token = decision.token_id
            symbols += 1
            if duration > 0:
                frame += duration
                advanced = True
                break

        if not advanced:
            frame += 1

    return TdtOutput(token_ids, frame_indices, emitted_durations, hidden, cell)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authenticated_capture_file(capture_dir: Path, name: str) -> bytes:
    manifest_bytes = (capture_dir / "manifest.sha256").read_bytes()
    entries = {}
    for line in manifest_bytes.decode().splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or Path(relative).name != relative or relative in entries:
            raise _fail("capture manifest has an invalid entry")
        entries[relative] = digest
    if name not in entries:
        raise _fail(f"capture manifest does not authenticate {name}")
    data = (capture_dir / name).read_bytes()
    if _sha256(data) != entries[name]:
        raise _fail(f"capture hash differs for {name}")
    return data


def _first_divergence(actual: list[dict], native: list[dict]) -> dict | None:
    for index in range(max(len(actual), len(native))):
        actual_value = actual[index] if index < len(actual) else None
        native_value = native[index] if index < len(native) else None
        if actual_value != native_value:
            return {"emission_index": index, "actual": actual_value, "native": native_value}
    return None


def compare_vulkan_capture(
    capture_dir: Path,
    model_dir: Path,
    *,
    source_commit: str,
    runtime_source_commit: str,
    allow_non_apple: bool = False,
) -> dict[str, Any]:
    """Run decoder and joint on a captured encoder tensor and compare emissions."""
    from importlib.metadata import distribution

    import mlx.core as mx

    from .vulkan_decoder import load_decoder
    from .vulkan_joint import run_joint as run_vulkan_joint
    from .vulkan_mel import trace_snapshot

    capture_dir = capture_dir.resolve()
    model_dir = model_dir.resolve()
    lock = ReferenceLock.load(Path(__file__).with_name("parakeet-reference.lock"))
    parent_receipt_bytes = (capture_dir.parent / "receipt.json").read_bytes()
    parent_receipt = json.loads(parent_receipt_bytes)
    manifest_bytes = (capture_dir / "manifest.sha256").read_bytes()
    if parent_receipt.get("capture_sha256", {}).get("manifest.sha256") != _sha256(manifest_bytes):
        raise _fail("parent receipt does not authenticate the capture manifest")
    if parent_receipt.get("model_revision") != lock.model_revision:
        raise _fail("capture model revision differs from the pinned reference")

    encoder_bytes = _authenticated_capture_file(capture_dir, "encoder_hidden.npy")
    encoder_host = np.load(io.BytesIO(encoder_bytes), allow_pickle=False)
    if encoder_host.shape != (1, 375, 640) or encoder_host.dtype != np.dtype("<f4"):
        raise _fail(
            "captured encoder hidden must be float32(1, 375, 640), got "
            f"{encoder_host.dtype}{encoder_host.shape}"
        )

    device_info = mx.device_info()
    device_name = str(device_info.get("device_name", ""))
    apple_gpu = "Apple M1" in device_name
    if not apple_gpu and not allow_non_apple:
        raise _fail(f"expected an Apple M1 Vulkan device, got {device_name!r}")
    backend = "apple-gpu-vulkan" if apple_gpu else "software-vulkan"
    if runtime_source_commit[:7] not in mx.__version__:
        raise _fail(
            f"runtime version {mx.__version__!r} does not identify source "
            f"{runtime_source_commit}"
        )
    libmlx_path = Path(distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so"))
    core_path = Path(mx.__file__)

    before = trace_snapshot()
    started = time.monotonic()
    with mx.stream(mx.gpu):
        encoder = mx.array(encoder_host)
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(encoder, hidden, cell)
    decoder = load_decoder(model_dir / "decoder.mlpackage")
    decoder_calls = 0
    joint_calls = 0

    def decoder_callback(token_id, current_hidden, current_cell):
        nonlocal decoder_calls
        decoder_calls += 1
        with mx.stream(mx.gpu):
            result = decoder(
                mx.array([[token_id]], dtype=mx.int32),
                current_hidden,
                current_cell,
            )
        mx.eval(result.decoder_hidden, result.next_hidden, result.next_cell)
        return DecoderStep(result.decoder_hidden[0], result.next_hidden, result.next_cell)

    def joint_callback(frame_index, decoder_state):
        nonlocal joint_calls
        joint_calls += 1
        result = run_vulkan_joint(
            encoder[:, frame_index, :],
            decoder_state,
            package_path=model_dir / "joint.mlpackage",
        )
        with mx.stream(mx.gpu):
            token_id = mx.argmax(result.token_logits)
            duration_index = mx.argmax(result.duration_logits)
        mx.eval(token_id, duration_index)
        return JointDecision(int(token_id.item()), int(duration_index.item()))

    output = greedy_tdt_decode(
        valid_frames=encoder.shape[1],
        config=lock.tdt,
        initial_hidden=hidden,
        initial_cell=cell,
        run_decoder=decoder_callback,
        run_joint=joint_callback,
    )
    elapsed_seconds = time.monotonic() - started
    after = trace_snapshot()
    trace_delta = {name: after[name] - before[name] for name in before}
    if trace_delta["gpu_primitive_dispatches"] <= 0 or trace_delta["vk_compute_dispatches"] <= 0:
        raise _fail("TDT decode produced no Vulkan compute dispatches")

    decisions = json.loads(_authenticated_capture_file(capture_dir, "tdt_decisions.json"))
    native_decisions = decisions.get("decisions")
    if not isinstance(native_decisions, list) or len(native_decisions) != 146:
        raise _fail("native capture must contain 146 decisions")
    native = [
        {
            "token_id": decision["token_id"],
            "frame_index": decision["frame_before"],
            "duration": decision["duration"],
        }
        for decision in native_decisions
        if decision["token_id"] != lock.tdt.blank_token_id
    ]
    if len(native) != 104:
        raise _fail(f"native capture must contain 104 emissions, got {len(native)}")
    actual = [
        {"token_id": token, "frame_index": frame, "duration": duration}
        for token, frame, duration in zip(
            output.token_ids, output.frame_indices, output.durations, strict=True
        )
    ]
    divergence = _first_divergence(actual, native)
    sources = {}
    for name in ("parakeet_tdt.py", "vulkan_decoder.py", "vulkan_joint.py"):
        path = Path(__file__).with_name(name)
        sources[name] = {
            "path": f"overlay/tools/coreml/{name}",
            "sha256": _sha256(path.read_bytes()),
        }
    sources["parakeet_tdt.py"]["origin_commit"] = "7d893bdeb9a1426745599754e238250575ce2220"
    sources["vulkan_decoder.py"]["origin_commit"] = "2af4f282a2b480c9854fe51a9688dc562719a411"
    sources["vulkan_joint.py"]["origin_commit"] = "81447d7a957e94ee520f03bdebed8721794e173b"

    return {
        "schema": "parakeet-vulkan-tdt-comparison/1",
        "status": "match" if divergence is None else "diverged",
        "backend": backend,
        "host": {"hostname": platform.node(), "kernel": platform.release()},
        "mlx": {
            "version": mx.__version__,
            "device": device_name,
            "architecture": device_info.get("architecture"),
        },
        "model": {
            "revision": lock.model_revision,
            "quantization": lock.model_quantization,
            "capture_parent_receipt_sha256": _sha256(parent_receipt_bytes),
            "encoder_hidden_npy_sha256": _sha256(encoder_bytes),
        },
        "source": {
            "branch_commit": source_commit,
            "base_commit": "c4768af65f1fd263c2ee55b6767291896ee18381",
            "excluded_decoder_commit": "8289b1bc",
            "runtime": {
                "source_commit": runtime_source_commit,
                "libmlx_path": str(libmlx_path),
                "libmlx_sha256": _sha256(libmlx_path.read_bytes()),
                "core_path": str(core_path),
                "core_sha256": _sha256(core_path.read_bytes()),
            },
            "files": sources,
        },
        "execution": {
            "control": "greedy_tdt_decode",
            "decoder_calls": decoder_calls,
            "joint_calls": joint_calls,
            "valid_encoder_frames": encoder.shape[1],
            "elapsed_seconds": elapsed_seconds,
            "recorded_decisions_injected": False,
            "gpu_trace_before": before,
            "gpu_trace_after": after,
            "gpu_trace_delta": trace_delta,
        },
        "comparison": {
            "native_emissions": len(native),
            "actual_emissions": len(actual),
            "emissions_compared": min(len(actual), len(native)),
            "exact_match": divergence is None,
            "first_divergence": divergence,
            "native": native,
            "actual": actual,
        },
    }


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-non-apple", action="store_true")
    args = parser.parse_args(arguments)
    receipt = compare_vulkan_capture(
        args.capture,
        args.model,
        source_commit=args.source_commit,
        runtime_source_commit=args.runtime_source_commit,
        allow_non_apple=args.allow_non_apple,
    )
    receipt["execution"]["command"] = [
        sys.executable,
        "-m",
        "coreml.parakeet_tdt",
        *arguments,
    ]
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
