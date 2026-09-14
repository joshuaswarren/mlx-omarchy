#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Name the recurrent tensor/step behind the native-encoder emission-101 miss.

The native-encoder arm (macOS capture encoder_hidden + this Linux decoder)
first misses at emission 101 (8029 vs native 7892). Emissions 96..103 all sit
on frame 373 with duration 0, so there the joint argmax is a pure function of
the decoder state chain.

Three arms under one lock, one process:

  shipped        the pinned decoder, recorded per call: gate args (mx fp16
                 matmuls exactly as ``_lstm`` computes them), next_cell,
                 next_hidden, and snapshots at emissions 95..103
  shipped-repeat determinism control, must match ``shipped`` decision for
                 decision
  chain-sigmoid  identical except ``fp16_sigmoid`` is the fp16 chain
                 ``1/(1+exp(-x))`` (rounding after negate, exp, add, divide),
                 the stages receipt's best-known native sigmoid; tanh stays
                 correctly rounded

Host-side post-pass, no device: recompute every recorded step from its
recorded gate args under the host model of the shipped ``_lstm`` (host-rounded
correctly-rounded unaries, fp16 products and adds) and count bit-exact lanes
against the recorded GPU tensors; then the chain-sigmoid per-step tensor delta
on the same gate args.
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


def logit_view(token_logits, duration_logits, emitted_token, native_token) -> dict:
    tok = host(token_logits).reshape(-1)
    dur = host(duration_logits).reshape(-1)
    argmax = int(tok.argmax())
    return {
        "argmax": argmax,
        "logit_argmax": float(tok[argmax]),
        "logit_emitted": float(tok[emitted_token]),
        "logit_native": float(tok[native_token]),
        "gap_native_minus_argmax": float(tok[native_token] - tok[argmax]),
        "duration_argmax": int(dur.argmax()),
    }


def err_vs(a, b) -> dict:
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


def sigmoid_chain16(x):
    """fp16 chain 1/(1+exp(-x)): fp16 rounding after negate, exp, add, divide."""
    v = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        e = np.exp(-v).astype(np.float16).astype(np.float64)
        d = (1.0 + e).astype(np.float16).astype(np.float64)
        y = (1.0 / d).astype(np.float16)
    return y


def sig_chain64(x):
    with np.errstate(over="ignore"):
        e = np.exp(-x).astype(np.float16).astype(np.float64)
        d = (1.0 + e).astype(np.float16).astype(np.float64)
        return (1.0 / d).astype(np.float16).astype(np.float64)


def sig16(x):
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float16).astype(np.float64)


def tan16(x):
    return np.tanh(x).astype(np.float16).astype(np.float64)


def run_arm(name, decoder, encoder, model_dir, config, mx, run_vulkan_joint, greedy,
            DecoderStep, JointDecision, save_dir, native, spy):
    snapshots: dict = {}
    step_records: list = []
    hidden0 = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell0 = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden0, cell0)
    calls = {"n": 0}
    joints = {"n": 0}

    def decoder_callback(token_id, current_hidden, current_cell):
        call_idx = calls["n"]
        calls["n"] += 1
        with mx.stream(mx.gpu):
            result = decoder(
                mx.array([[token_id]], dtype=mx.int32),
                current_hidden,
                current_cell,
            )
        mx.eval(result.decoder_hidden, result.next_hidden, result.next_cell)
        step_records.append({
            "call": call_idx,
            "token": int(token_id),
            "hidden_in": np.asarray(current_hidden, dtype=np.float32).copy(),
            "cell_in": np.asarray(current_cell, dtype=np.float32).copy(),
            "next_hidden": np.asarray(result.next_hidden, dtype=np.float32).copy(),
            "next_cell": np.asarray(result.next_cell, dtype=np.float32).copy(),
            "decoder_state": np.asarray(result.decoder_hidden, dtype=np.float32).copy(),
        })
        return DecoderStep(result.decoder_hidden[0], result.next_hidden, result.next_cell)

    def joint_callback(frame_index, decoder_state):
        joints["n"] += 1
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
        if step_records and "emission" not in step_records[-1]:
            step_records[-1]["emission"] = n_emitted
        if n_emitted in WANT:
            hid = host(joint_callback.hidden)
            cel = host(joint_callback.cell)
            st = host(decoder_state)
            rec = {
                "emission_index": n_emitted,
                "frame_index": int(frame_index),
                "decoder_input_token": int(joint_callback.input_token),
                "hidden": summary(hid),
                "cell": summary(cel),
                "decoder_state": summary(st),
                "logits": logit_view(
                    result.token_logits, result.duration_logits,
                    int(joint_callback.input_token), int(native[n_emitted]["token_id"]),
                ),
            }
            tag = f"{name}_e{n_emitted}"
            np.save(save_dir / f"{tag}_hidden.npy", hid)
            np.save(save_dir / f"{tag}_cell.npy", cel)
            np.save(save_dir / f"{tag}_decoder_state.npy", st)
            np.save(save_dir / f"{tag}_token_logits.npy", host(result.token_logits))
            np.save(save_dir / f"{tag}_duration_logits.npy", host(result.duration_logits))
            snapshots[n_emitted] = rec
        token = int(token_id.item())
        if token != config.blank_token_id:
            joint_callback.emitted.append(token)
        return JointDecision(token, int(duration_index.item()))

    joint_callback.emitted = []
    joint_callback.hidden = hidden0
    joint_callback.cell = cell0
    joint_callback.input_token = config.blank_token_id

    def wrapped_decoder(token_id, current_hidden, current_cell):
        step = decoder_callback(token_id, current_hidden, current_cell)
        joint_callback.hidden = step.hidden
        joint_callback.cell = step.cell
        joint_callback.input_token = token_id
        return step

    started = time.monotonic()
    output = greedy(
        valid_frames=int(encoder.shape[1]),
        config=config,
        initial_hidden=hidden0,
        initial_cell=cell0,
        run_decoder=wrapped_decoder,
        run_joint=joint_callback,
    )
    elapsed = time.monotonic() - started
    actual = [
        {"token_id": int(t), "frame_index": int(f), "duration": int(d)}
        for t, f, d in zip(output.token_ids, output.frame_indices, output.durations, strict=True)
    ]

    if spy:
        np.savez_compressed(
            save_dir / f"{name}_steps.npz",
            tokens=np.array([r["token"] for r in step_records], dtype=np.int32),
            emissions=np.array([r.get("emission", -1) for r in step_records], dtype=np.int32),
            hidden_in=np.stack([r["hidden_in"] for r in step_records]),
            cell_in=np.stack([r["cell_in"] for r in step_records]),
            next_hidden=np.stack([r["next_hidden"] for r in step_records]),
            next_cell=np.stack([r["next_cell"] for r in step_records]),
            decoder_state=np.stack([r["decoder_state"] for r in step_records]),
        )

    return {
        "name": name,
        "elapsed_s": elapsed,
        "decoder_calls": calls["n"],
        "joint_calls": joints["n"],
        "emissions": len(actual),
        "actual": actual,
        "snapshots": {str(k): v for k, v in snapshots.items()},
    }, step_records


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

    shipped_sigmoid = vd.fp16_sigmoid
    arms: dict = {}

    # ---- arm 1: shipped with gate spy ------------------------------------
    decoder = load_decoder(args.model / "decoder.mlpackage")
    store: dict = {"l0": [], "l1": []}
    original_lstm = vd._lstm

    def spied_lstm(sequence, hidden, cell, weight_ih, weight_hh, bias, mx):
        gates = sequence[0] @ weight_ih.T + hidden @ weight_hh.T + bias
        gates_host = host(gates).reshape(-1).astype(np.float16)
        key = "l0" if len(store["l0"]) == len(store["l1"]) else "l1"
        store[key].append(gates_host)
        return original_lstm(sequence, hidden, cell, weight_ih, weight_hh, bias, mx)

    vd._lstm = spied_lstm
    arms["shipped"], _ = run_arm(
        "shipped", decoder, encoder, args.model, lock.tdt, mx, run_vulkan_joint,
        greedy_tdt_decode, DecoderStep, JointDecision, out, native, spy=True,
    )
    vd._lstm = original_lstm
    np.savez_compressed(out / "shipped_gates.npz", l0=np.stack(store["l0"]), l1=np.stack(store["l1"]))

    # ---- arm 2: shipped repeat (determinism control) ----------------------
    decoder2 = load_decoder(args.model / "decoder.mlpackage")
    arms["shipped-repeat"], _ = run_arm(
        "shipped-repeat", decoder2, encoder, args.model, lock.tdt, mx, run_vulkan_joint,
        greedy_tdt_decode, DecoderStep, JointDecision, out, native, spy=False,
    )
    repeat_ok = arms["shipped"]["actual"] == arms["shipped-repeat"]["actual"]

    # ---- arm 3: chain sigmoid ---------------------------------------------
    vd.fp16_sigmoid = sigmoid_chain16
    decoder3 = load_decoder(args.model / "decoder.mlpackage")
    arms["chain-sigmoid"], _ = run_arm(
        "chain-sigmoid", decoder3, encoder, args.model, lock.tdt, mx, run_vulkan_joint,
        greedy_tdt_decode, DecoderStep, JointDecision, out, native, spy=False,
    )
    vd.fp16_sigmoid = shipped_sigmoid

    for arm in arms.values():
        arm["first_divergence"] = first_div(arm["actual"], native)
        arm["matching_prefix"] = (
            arm["first_divergence"]["emission_index"] if arm["first_divergence"] else arm["emissions"]
        )

    for e in WANT:
        a = out / f"chain-sigmoid_e{e}_decoder_state.npy"
        b = out / f"shipped_e{e}_decoder_state.npy"
        if a.exists() and b.exists():
            arms["chain-sigmoid"]["snapshots"][str(e)]["state_err_vs_shipped"] = err_vs(
                np.load(a), np.load(b)
            )

    # ---- host post-pass: step model validation + chain delta --------------
    gates = np.load(out / "shipped_gates.npz")
    steps = np.load(out / "shipped_steps.npz")
    n_calls = steps["next_cell"].shape[0]
    exact_cell = 0
    exact_hidden = 0
    lanes = 0
    chain_cell_changed = []
    chain_hidden_changed = []
    for i in range(n_calls):
        for key in ("l0", "l1"):
            layer = 0 if key == "l0" else 1
            gates_row = gates[key][i].astype(np.float64)
            hidden_in = steps["hidden_in"][i][layer].astype(np.float16).astype(np.float64)
            cell_in = steps["cell_in"][i][layer].astype(np.float16).astype(np.float64)
            g_i, g_f, g_o, g_g = np.split(gates_row, 4)
            s_f, s_i, s_o = sig16(g_f), sig16(g_i), sig16(g_o)
            t_g = tan16(g_g)
            cell = (np.float16(s_f * cell_in) + np.float16(s_i * t_g)).astype(np.float16).astype(np.float64)
            hidden = np.float16(s_o * np.float16(np.tanh(cell))).astype(np.float64)
            rec_cell = steps["next_cell"][i][layer].astype(np.float16).astype(np.float64)
            rec_hidden = steps["next_hidden"][i][layer].astype(np.float16).astype(np.float64)
            exact_cell += int(np.count_nonzero(cell == rec_cell))
            exact_hidden += int(np.count_nonzero(hidden == rec_hidden))
            lanes += 640
            c_f, c_i, c_o = sig_chain64(g_f), sig_chain64(g_i), sig_chain64(g_o)
            cell_c = (np.float16(c_f * cell_in) + np.float16(c_i * t_g)).astype(np.float16).astype(np.float64)
            hidden_c = np.float16(c_o * np.float16(np.tanh(cell_c))).astype(np.float64)
            chain_cell_changed.append(int(np.count_nonzero(cell_c != cell)))
            chain_hidden_changed.append(int(np.count_nonzero(hidden_c != hidden)))

    post = {
        "host_step_model_vs_recorded_gpu": {
            "calls": int(n_calls),
            "cell_bit_exact_lanes": exact_cell,
            "hidden_bit_exact_lanes": exact_hidden,
            "lanes": lanes,
        },
        "chain_sigmoid_delta_vs_shipped_same_gates": {
            "cell_lanes_changed_per_call_layer_mean": float(np.mean(chain_cell_changed)),
            "hidden_lanes_changed_per_call_layer_mean": float(np.mean(chain_hidden_changed)),
            "cell_lanes_changed_max": int(np.max(chain_cell_changed)),
            "hidden_lanes_changed_max": int(np.max(chain_hidden_changed)),
        },
    }

    result = {
        "schema": "mlx-omarchy.decoder-lstm-e101-live/1",
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
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "snapshots"} for k, v in arms.items()},
        "snapshots": {k: v["snapshots"] for k, v in arms.items() if k != "shipped-repeat"},
        "determinism_repeat_match": bool(repeat_ok),
        "post": post,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print("repeat_match", repeat_ok)
    print(json.dumps(post, indent=1))
    for name in ("shipped", "chain-sigmoid"):
        arm = arms[name]
        print(name, "prefix", arm["matching_prefix"], "first_div", json.dumps(arm["first_divergence"]))
        for e in (99, 100, 101, 102):
            s = arm["snapshots"].get(str(e))
            if s:
                print("  e", e, json.dumps(s["logits"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
