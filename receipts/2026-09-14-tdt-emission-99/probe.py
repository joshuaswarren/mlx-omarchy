#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Snapshot decoder hidden/cell and joint logits at TDT emissions 98 and 99.

Runs greedy TDT twice (ANE encoder hidden vs native encoder hidden), then
re-runs joint at those states against the other encoder's frame 373.
Native capture has no hidden/cell/joint tensors at emission 99.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

NATIVE_99 = 7863
LINUX_99 = 7883
FRAME = 373
WANT = (98, 99)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def summary(arr: np.ndarray) -> dict:
    a = np.ascontiguousarray(arr)
    wide = a.astype(np.float64, copy=False)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": sha256_arr(a),
        "finite": bool(np.isfinite(a).all()),
        "min": float(np.min(wide)),
        "max": float(np.max(wide)),
        "mean_abs": float(np.mean(np.abs(wide))),
    }


def host(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value)


def logit_view(token_logits, duration_logits) -> dict:
    tok = host(token_logits).reshape(-1)
    dur = host(duration_logits).reshape(-1)
    return {
        "argmax": int(tok.argmax()),
        "logit_7883": float(tok[LINUX_99]),
        "logit_7863": float(tok[NATIVE_99]),
        "gap_7883_minus_7863": float(tok[LINUX_99] - tok[NATIVE_99]),
        "duration_argmax": int(dur.argmax()),
        "duration_logits": [float(x) for x in dur],
    }


def err_vs(a: np.ndarray, b: np.ndarray) -> dict:
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    diff = da - db
    denom = np.linalg.norm(db)
    return {
        "bit_exact": bool(np.array_equal(np.asarray(a), np.asarray(b))),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / denom) if denom else None,
    }


def run_arm(name, encoder, decoder, model_dir, config, mx, run_vulkan_joint, greedy, DecoderStep, JointDecision, save_dir):
    snapshots = {}
    decoder_calls = 0
    joint_calls = 0
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

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
        n_emitted = len(joint_callback.emitted)
        if n_emitted in WANT:
            hid = host(joint_callback.hidden)
            cel = host(joint_callback.cell)
            state = host(decoder_state)
            rec = {
                "emission_index": n_emitted,
                "frame_index": int(frame_index),
                "decoder_input_token": int(joint_callback.input_token),
                "hidden": summary(hid),
                "cell": summary(cel),
                "decoder_state": summary(state),
                "logits": logit_view(result.token_logits, result.duration_logits),
            }
            tag = f"{name}_e{n_emitted}"
            np.save(save_dir / f"{tag}_hidden.npy", hid)
            np.save(save_dir / f"{tag}_cell.npy", cel)
            np.save(save_dir / f"{tag}_decoder_state.npy", state)
            np.save(save_dir / f"{tag}_token_logits.npy", host(result.token_logits))
            snapshots[n_emitted] = rec
            snapshots[f"{n_emitted}_state"] = decoder_state
        return JointDecision(int(token_id.item()), int(duration_index.item()))

    joint_callback.emitted = []
    joint_callback.hidden = hidden
    joint_callback.cell = cell
    joint_callback.input_token = config.blank_token_id

    orig_decoder = decoder_callback

    def wrapped_decoder(token_id, current_hidden, current_cell):
        step = orig_decoder(token_id, current_hidden, current_cell)
        joint_callback.hidden = step.hidden
        joint_callback.cell = step.cell
        joint_callback.input_token = token_id
        return step

    def wrapped_joint(frame_index, decoder_state):
        decision = joint_callback(frame_index, decoder_state)
        if decision.token_id != config.blank_token_id:
            joint_callback.emitted.append(decision.token_id)
        return decision

    started = time.monotonic()
    output = greedy(
        valid_frames=int(encoder.shape[1]),
        config=config,
        initial_hidden=hidden,
        initial_cell=cell,
        run_decoder=wrapped_decoder,
        run_joint=wrapped_joint,
    )
    elapsed = time.monotonic() - started
    actual = [
        {"token_id": int(t), "frame_index": int(f), "duration": int(d)}
        for t, f, d in zip(output.token_ids, output.frame_indices, output.durations, strict=True)
    ]
    return {
        "name": name,
        "elapsed_s": elapsed,
        "decoder_calls": decoder_calls,
        "joint_calls": joint_calls,
        "emissions": len(actual),
        "actual": actual,
        "snapshots": {str(k): v for k, v in snapshots.items() if not str(k).endswith("_state")},
        "_states": {k: v for k, v in snapshots.items() if str(k).endswith("_state") or isinstance(k, str) and k.endswith("_state")},
        "_raw_states": {k: snapshots[k] for k in snapshots if isinstance(k, str) and k.endswith("_state")},
    }, snapshots


def first_div(actual, native):
    for i in range(max(len(actual), len(native))):
        a = actual[i] if i < len(actual) else None
        n = native[i] if i < len(native) else None
        if a != n:
            return {"emission_index": i, "actual": a, "native": n}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-tools", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ane-encoder", type=Path, required=True)
    parser.add_argument("--native-encoder", type=Path, required=True)
    parser.add_argument("--native-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.overlay_tools.resolve()))
    from coreml.parakeet_tdt import DecoderStep, JointDecision, greedy_tdt_decode
    from coreml.reference import ReferenceLock
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_joint import run_joint as run_vulkan_joint
    import mlx.core as mx

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = ReferenceLock.load()
    native_tokens = json.loads(args.native_tokens.read_text())
    native = [
        {
            "token_id": int(t),
            "frame_index": int(f),
            "duration": int(d),
        }
        for t, f, d in zip(
            native_tokens["token_ids"],
            native_tokens["frame_indices"],
            native_tokens["durations"],
            strict=True,
        )
    ]
    ane_np = np.load(args.ane_encoder)
    nat_np = np.load(args.native_encoder)
    if ane_np.shape != (1, 375, 640) or nat_np.shape != (1, 375, 640):
        raise SystemExit(f"encoder shape {ane_np.shape} {nat_np.shape}")

    frame_cmp = {
        "ane_sha256": sha256_arr(ane_np),
        "native_sha256": sha256_arr(nat_np),
        "all_frames": err_vs(ane_np, nat_np),
        "frame_373": err_vs(ane_np[0, FRAME], nat_np[0, FRAME]),
        "ane_frame_373": summary(ane_np[0, FRAME]),
        "native_frame_373": summary(nat_np[0, FRAME]),
    }

    decoder = load_decoder(args.model / "decoder.mlpackage")
    with mx.stream(mx.gpu):
        ane_enc = mx.array(ane_np)
        nat_enc = mx.array(nat_np)
    mx.eval(ane_enc, nat_enc)

    ane_arm, ane_snaps = run_arm(
        "ane",
        ane_enc,
        decoder,
        args.model,
        lock.tdt,
        mx,
        run_vulkan_joint,
        greedy_tdt_decode,
        DecoderStep,
        JointDecision,
        out,
    )
    nat_arm, nat_snaps = run_arm(
        "native_enc",
        nat_enc,
        decoder,
        args.model,
        lock.tdt,
        mx,
        run_vulkan_joint,
        greedy_tdt_decode,
        DecoderStep,
        JointDecision,
        out,
    )

    swaps = {}
    for arm_name, snaps, other_enc, other_label in (
        ("ane", ane_snaps, nat_enc, "native_frame"),
        ("native_enc", nat_snaps, ane_enc, "ane_frame"),
        ("ane", ane_snaps, ane_enc, "ane_frame"),
        ("native_enc", nat_snaps, nat_enc, "native_frame"),
    ):
        state = snaps.get("99_state")
        if state is None:
            continue
        result = run_vulkan_joint(
            other_enc[:, FRAME, :],
            state,
            package_path=args.model / "joint.mlpackage",
        )
        swaps[f"{arm_name}_state99_x_{other_label}"] = logit_view(
            result.token_logits, result.duration_logits
        )

    def arm_public(arm):
        div = first_div(arm["actual"], native)
        prefix = 0
        for a, n in zip(arm["actual"], native):
            if a["token_id"] != n["token_id"]:
                break
            prefix += 1
        return {
            "name": arm["name"],
            "elapsed_s": arm["elapsed_s"],
            "decoder_calls": arm["decoder_calls"],
            "joint_calls": arm["joint_calls"],
            "emissions": arm["emissions"],
            "matching_prefix": prefix,
            "first_divergence": div,
            "tokens_96_103": arm["actual"][96:104],
            "snapshots": arm["snapshots"],
        }

    ane_pub = arm_public(ane_arm)
    nat_pub = arm_public(nat_arm)
    ane_pick = swaps.get("ane_state99_x_ane_frame", {}).get("argmax")
    nat_frame_pick = swaps.get("ane_state99_x_native_frame", {}).get("argmax")
    native_enc_pick = None
    if "99" in nat_arm["snapshots"]:
        native_enc_pick = nat_arm["snapshots"]["99"]["logits"]["argmax"]

    if nat_frame_pick == NATIVE_99 and ane_pick == LINUX_99:
        cause = "a"
        named = "parakeet.tdt.encoder-frame-373.residual-into-joint"
        rule = (
            "Linux decoder_state after emission 98 plus native encoder frame 373 "
            "selects native token 7863; the same state plus ANE frame 373 selects 7883."
        )
    elif (nat_frame_pick == LINUX_99) or (native_enc_pick == LINUX_99):
        cause = "b"
        named = "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml"
        rule = (
            "After the matching 0-98 prefix, Linux LSTM state at the emission-99 joint "
            "selects 7883 even when the joint is given native encoder frame 373. "
            "The projector and joint operators are not the argmax path."
        )
    else:
        cause = "c"
        named = "parakeet.tdt.joint-or-projector"
        rule = (
            "Swap results did not isolate encoder frame 373 or LSTM state; "
            "joint/projector remains the named hole."
        )

    receipt = {
        "schema": "mlx-omarchy.tdt-emission-99/1",
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
        },
        "mlx": {
            "version": mx.__version__,
            "device": str(mx.default_device()),
            "device_name": str(mx.device_info().get("device_name")),
        },
        "source": {
            "vulkan_decoder_py_sha256": sha256_file(
                args.overlay_tools / "coreml" / "vulkan_decoder.py"
            ),
            "vulkan_joint_py_sha256": sha256_file(
                args.overlay_tools / "coreml" / "vulkan_joint.py"
            ),
            "parakeet_tdt_py_sha256": sha256_file(
                args.overlay_tools / "coreml" / "parakeet_tdt.py"
            ),
            "probe_sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            "ane_encoder": str(args.ane_encoder),
            "ane_encoder_sha256": sha256_file(args.ane_encoder),
            "native_encoder": str(args.native_encoder),
            "native_encoder_sha256": sha256_file(args.native_encoder),
            "native_tokens": str(args.native_tokens),
            "model": str(args.model),
        },
        "native_capture_tensors_at_99": False,
        "encoder_frame_373": frame_cmp,
        "ane_encoder_arm": ane_pub,
        "native_encoder_arm": nat_pub,
        "swaps_at_emission_99": swaps,
        "cause": {
            "letter": cause,
            "named": named,
            "rule": rule,
            "native_token_99": NATIVE_99,
            "linux_token_99": LINUX_99,
        },
    }
    (out / "result.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cause": cause, "named": named, "swaps": swaps, "ane_div": ane_pub["first_divergence"], "nat_div": nat_pub["first_divergence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
