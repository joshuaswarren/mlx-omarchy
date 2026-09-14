#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Live native-encoder decode with fitted decoder-LSTM unary candidates.

Runs greedy TDT decodes on the E2E overlay bytes (decoder bea0e2e6, joint
bf31537d) with the macOS capture's own encoder hidden, swapping the decoder
LSTM unaries per arm.  Decides whether any unary pair from the host unary-fit
search flips emission 101 to the native token 7892.

Runs on jwm1 under flock /tmp/m1-gpu.lock, one acquisition, never stolen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

WANT = range(95, 104)
NATIVE_101 = 7892


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host(value) -> np.ndarray:
    import mlx.core as mx

    return np.asarray(value)


def summary(arr: np.ndarray) -> dict:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return {"shape": list(a.shape), "sum": float(a.sum()), "max_abs": float(np.abs(a).max())}


def logit_view(token_logits, duration_logits, native_token) -> dict:
    tok = host(token_logits).reshape(-1)
    dur = host(duration_logits).reshape(-1)
    order = np.argsort(tok)[::-1]
    return {
        "argmax": int(order[0]),
        "logit_argmax": float(tok[order[0]]),
        f"logit_{native_token}": float(tok[native_token]),
        "gap_to_native": float(tok[native_token]) - float(tok[order[0]]),
        "duration_argmax": int(np.argmax(dur)),
    }


# ---- candidate unaries (mirror probe_unary_fit.py contracts) ---------------


def sigmoid_cr(x):
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        y = 1.0 / (1.0 + np.exp(-x64))
    return np.asarray(y, dtype=np.float16)


def sigmoid_chain1111(x):
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        e = np.exp((-x64).astype(np.float16).astype(np.float64))
        s = (1.0 + e.astype(np.float16).astype(np.float64)).astype(np.float16)
        y = (1.0 / s.astype(np.float64)).astype(np.float16)
    return y


def tanh_cr(x):
    return np.asarray(np.tanh(np.asarray(x, dtype=np.float64)), dtype=np.float16)


def tanh_fit(k: float):
    def fn(x):
        return np.asarray(
            np.tanh(np.asarray(x, dtype=np.float64)) * (1.0 + k), dtype=np.float16
        )

    return fn


# Capture pins over correctly-rounded baselines: the activation-stages
# receipt's pinned native values override CR at exactly those arguments,
# everything else CR (the piecewise candidate class).
_SIG_PINS = [
    (-0.288086, 0.428466797), (+0.090027, 0.522460938), (+0.154419, 0.538574219),
    (+0.154663, 0.538574219), (+0.261230, 0.564941406), (+0.314453, 0.577636719),
    (+0.438965, 0.607910156), (+0.529785, 0.629394531), (+0.566406, 0.638183594),
    (+0.566895, 0.638183594), (+0.567871, 0.638183594),
]
_TAN_PINS = [
    (-1.099609, -0.800292969), (-0.681641, -0.592773438), (-0.560547, -0.508300781),
    (-0.113953, -0.113464355), (+0.282715, +0.275390625), (+0.319092, +0.308837891),
    (+0.458984, +0.429199219), (+0.622070, +0.552734375),
]


def _pinned(base, pins):
    table = [(np.float16(a), np.float16(v)) for a, v in pins]

    def fn(x):
        x16 = np.asarray(x, dtype=np.float16)
        out = np.asarray(base(x), dtype=np.float64)
        for a, v in table:
            out = np.where(x16 == a, np.float64(v), out)
        return out.astype(np.float16)

    return fn


sigmoid_cr_pins = _pinned(sigmoid_cr, _SIG_PINS)
tanh_cr_pins = _pinned(tanh_cr, _TAN_PINS)


ARMS = [
    ("shipped", sigmoid_cr, tanh_cr),
    ("tanh-fit-1e4", sigmoid_cr, tanh_fit(1.0e-4)),
    ("pair-fit-1e4", sigmoid_chain1111, tanh_fit(1.0e-4)),
    ("pair-fit-15e4", sigmoid_chain1111, tanh_fit(1.5e-4)),
    ("cr+pins", sigmoid_cr_pins, tanh_cr_pins),
]


def run_arm(name, sigmoid_fn, tanh_fn, vd, decoder, encoder, model_dir, mx,
            run_vulkan_joint, greedy, DecoderStep, JointDecision, save_dir,
            native, lock_tdt):
    import mlx.core as mx_inner

    shipped_sigmoid = vd.fp16_sigmoid
    shipped_tanh = vd.fp16_tanh
    vd.fp16_sigmoid = sigmoid_fn
    vd.fp16_tanh = tanh_fn
    snapshots: dict = {}
    hidden0 = mx_inner.zeros((2, 1, 640), dtype=mx_inner.float32)
    cell0 = mx_inner.zeros((2, 1, 640), dtype=mx_inner.float32)
    mx_inner.eval(hidden0, cell0)
    calls = {"n": 0}
    joints = {"n": 0}

    def decoder_callback(token_id, current_hidden, current_cell):
        call_idx = calls["n"]
        calls["n"] += 1
        with mx_inner.stream(mx_inner.gpu):
            result = decoder(
                mx_inner.array([[token_id]], dtype=mx_inner.int32),
                current_hidden,
                current_cell,
            )
        mx_inner.eval(result.decoder_hidden, result.next_hidden, result.next_cell)
        return DecoderStep(result.decoder_hidden[0], result.next_hidden, result.next_cell)

    def joint_callback(frame_index, decoder_state):
        joints["n"] += 1
        result = run_vulkan_joint(
            encoder[:, frame_index, :],
            decoder_state,
            package_path=model_dir / "joint.mlpackage",
        )
        with mx_inner.stream(mx_inner.gpu):
            token_id = mx_inner.argmax(result.token_logits)
            duration_index = mx_inner.argmax(result.duration_logits)
        mx_inner.eval(token_id, duration_index)
        n_emitted = len(joint_callback.emitted)
        if n_emitted in WANT:
            st = host(decoder_state)
            rec = {
                "emission_index": n_emitted,
                "frame_index": int(frame_index),
                "decoder_input_token": int(joint_callback.input_token),
                "decoder_state": summary(st),
                "logits": logit_view(result.token_logits, result.duration_logits,
                                     int(native[n_emitted]["token_id"])),
            }
            tag = f"{name}_e{n_emitted}"
            np.save(save_dir / f"{tag}_decoder_state.npy", st)
            np.save(save_dir / f"{tag}_token_logits.npy", host(result.token_logits))
            snapshots[n_emitted] = rec
        token = int(token_id.item())
        if token != lock_tdt.blank_token_id:
            joint_callback.emitted.append(token)
        return JointDecision(token, int(duration_index.item()))

    joint_callback.emitted = []
    joint_callback.hidden = hidden0
    joint_callback.cell = cell0
    joint_callback.input_token = lock_tdt.blank_token_id

    def wrapped_decoder(token_id, current_hidden, current_cell):
        step = decoder_callback(token_id, current_hidden, current_cell)
        joint_callback.hidden = step.hidden
        joint_callback.cell = step.cell
        joint_callback.input_token = token_id
        return step

    started = time.monotonic()
    output = greedy(
        valid_frames=int(encoder.shape[1]),
        config=lock_tdt,
        initial_hidden=hidden0,
        initial_cell=cell0,
        run_decoder=wrapped_decoder,
        run_joint=joint_callback,
    )
    elapsed = time.monotonic() - started
    vd.fp16_sigmoid = shipped_sigmoid
    vd.fp16_tanh = shipped_tanh

    actual = [
        {"token_id": int(t), "frame_index": int(f), "duration": int(d)}
        for t, f, d in zip(output.token_ids, output.frame_indices, output.durations, strict=True)
    ]
    first_div = None
    for i in range(max(len(actual), len(native))):
        a = actual[i] if i < len(actual) else None
        n = native[i] if i < len(native) else None
        if a != n:
            first_div = {"emission_index": i, "actual": a, "native": n}
            break
    return {
        "name": name,
        "elapsed_s": elapsed,
        "decoder_calls": calls["n"],
        "joint_calls": joints["n"],
        "emissions": len(actual),
        "first_divergence": first_div,
        "matching_prefix": first_div["emission_index"] if first_div else len(actual),
        "flipped_e101": actual[101]["token_id"] == NATIVE_101 if len(actual) > 101 else False,
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-tools", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-encoder", type=Path, required=True)
    parser.add_argument("--native-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.overlay_tools.resolve()))
    from coreml import vulkan_decoder as vd
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
        {"token_id": int(t), "frame_index": int(f), "duration": int(d)}
        for t, f, d in zip(
            native_tokens["token_ids"],
            native_tokens["frame_indices"],
            native_tokens["durations"],
            strict=True,
        )
    ]
    encoder = mx.array(np.load(args.native_encoder))
    mx.eval(encoder)

    arms = {}
    for name, sfn, tfn in ARMS:
        decoder = load_decoder(args.model / "decoder.mlpackage")
        arms[name] = run_arm(
            name, sfn, tfn, vd, decoder, encoder, args.model, mx,
            run_vulkan_joint, greedy_tdt_decode, DecoderStep, JointDecision,
            out, native, lock.tdt,
        )

    result = {
        "schema": "mlx-omarchy.decoder-lstm-unary-fit-live/1",
        "named_hole": "parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml",
        "python": sys.version.split()[0],
        "source": {
            "vulkan_decoder_py_sha256": sha256_file(Path(vd.__file__)),
            "overlay_tools": str(args.overlay_tools),
            "native_encoder": {"path": str(args.native_encoder), "sha256": sha256_file(args.native_encoder)},
            "native_tokens": {"path": str(args.native_tokens), "sha256": sha256_file(args.native_tokens)},
            "model": str(args.model),
        },
        "native_emitted_96_103": native[96:104],
        "arms": arms,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    for name, arm in arms.items():
        print(name, "prefix", arm["matching_prefix"], "e101_flipped", arm["flipped_e101"])
        for e in (99, 100, 101, 102):
            s = arm["snapshots"].get(str(e))
            if s:
                print("  e", e, json.dumps(s["logits"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
