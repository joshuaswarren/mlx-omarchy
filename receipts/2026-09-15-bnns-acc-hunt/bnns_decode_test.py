#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Free decode with the closed BNNS fused-LSTM contract (macstudio-RE'd).

Gate preacts: ONE fp16 GEMV over concat[x; h0] with repacked [wi | wh],
k-blocks B=128, per-term fp16-FMA chain from +0.0 (single rounding, ascending
k), fold y = rnd16(y + blocksum) per block (first block stores), then one
fp16 bias add. Unaries: dense sigma16/tanh16 bit-pattern LUTs. Cell:
c1 = rnd16(rnd16(sig_f*c0) + sig_i*tanh16(g)) (FMUL + FMA). Hidden:
h = rnd16(sig_o * tanh16(c1)).

Everything else (embedding, projector, joint) runs exactly like the pinned
overlay path. Reports the free-decode first divergence versus the
authenticated native emission trace.

Run on jwm1:
  flock -x -w 120 /tmp/m1-gpu.lock && \\
  PYTHONPATH=/tmp/vulkan-tdt-142 control-venv/bin/python bnns_decode_test.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

from coreml.parakeet_tdt import DecoderStep, JointDecision, greedy_tdt_decode
from coreml.pinned_component import load_pinned_component
from coreml.reference import ReferenceLock
from coreml.vulkan_joint import run_joint

_F16 = np.dtype("<f2")
_HIDDEN = 640
_VOCAB = 8193
_BLOCK = 128  # closed contract k-block
_CAPTURE = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/capture/ane")
_MODEL = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/model")
_LUTS = Path("/tmp/vulkan-tdt-142/bnns_unary_luts.npz")


def _r16(v):
    return np.asarray(v, np.float64).astype(_F16)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BnnsDecoder:
    """Pinned decoder with the BNNS GEMV contract in place of the mx matmuls."""

    _SPECS = (
        "embedding_weight_to_fp16",
        "concat_1_to_fp16", "concat_2_to_fp16", "concat_0_to_fp16",
        "concat_4_to_fp16", "concat_5_to_fp16", "concat_3_to_fp16",
        "projector_weight_to_fp16", "projector_bias_to_fp16",
    )

    def __init__(self, package_path: Path):
        component = load_pinned_component(package_path, "decoder")
        consts = {n: component.constant(n) for n in self._SPECS}
        self.embedding = consts["embedding_weight_to_fp16"]
        self.projector = consts["projector_weight_to_fp16"]
        self.projector_bias = consts["projector_bias_to_fp16"]
        self.layers = []
        for wih_n, whh_n, b_n in (
            ("concat_1_to_fp16", "concat_2_to_fp16", "concat_0_to_fp16"),
            ("concat_4_to_fp16", "concat_5_to_fp16", "concat_3_to_fp16"),
        ):
            wih = np.asarray(consts[wih_n], _F16).astype(np.float64)
            whh = np.asarray(consts[whh_n], _F16).astype(np.float64)
            bias = np.asarray(consts[b_n], _F16).astype(np.float64)
            wcat = np.concatenate([wih, whh], axis=1)  # repacked [wi | wh]
            self.layers.append(
                (wcat.reshape(4 * _HIDDEN, -1, _BLOCK), bias, (wih, whh, bias))
            )
        with np.load(_LUTS) as z:
            self.sig = z["sigma_lut"]
            self.tan = z["tanh_lut"]
        with mx.stream(mx.gpu):
            self.mx_embedding = mx.array(self.embedding)
            self.mx_projector = mx.array(self.projector)
            self.mx_projector_bias = mx.array(self.projector_bias)
        mx.eval(self.mx_embedding, self.mx_projector, self.mx_projector_bias)

    def _gates(self, layer_idx, x16, h16):
        wcat3, bias64, _ = self.layers[layer_idx]
        xcat = np.concatenate([x16.ravel(), h16.ravel()]).astype(np.float64)
        p3 = wcat3 * xcat.reshape(-1, _BLOCK)[None, :, :]
        y = np.zeros(4 * _HIDDEN, np.float64)
        for j in range(p3.shape[1]):
            col = p3[:, j, :]
            acc = _r16(col[:, 0])
            for t in range(1, _BLOCK):
                acc = _r16(acc + col[:, t])
            y = _r16(y + acc)
        return _r16(y + bias64).view(np.uint16)

    def _lstm(self, sequence, hidden, cell, layer_idx):
        x16 = np.asarray(sequence, dtype=np.float32).astype(_F16)
        h16 = np.asarray(hidden, dtype=np.float32).astype(_F16)
        z = self._gates(layer_idx, x16, h16)
        sig_i = self.sig[z[0:_HIDDEN]].astype(np.float64)
        sig_f = self.sig[z[_HIDDEN:2 * _HIDDEN]].astype(np.float64)
        sig_o = self.sig[z[2 * _HIDDEN:3 * _HIDDEN]].astype(np.float64)
        tan_g = self.tan[z[3 * _HIDDEN:]].astype(np.float64)
        c0 = np.asarray(cell, dtype=np.float32).astype(_F16).ravel().astype(np.float64)
        c1 = _r16(_r16(sig_f * c0) + sig_i * tan_g)
        next_cell = _r16(c1)
        tan_c1 = self.tan[next_cell.view(np.uint16)].astype(np.float64)
        next_hidden = _r16(sig_o * tan_c1)
        mx_next_hidden = mx.array(next_hidden.reshape(1, _HIDDEN))
        mx_next_cell = mx.array(next_cell.reshape(1, _HIDDEN))
        return mx.expand_dims(mx_next_hidden, axis=0), mx_next_hidden, mx_next_cell

    def __call__(self, input_ids, hidden, cell):
        indices = input_ids.astype(mx.int16).astype(mx.int32)
        indices = mx.where(indices >= 0, indices, indices + _VOCAB)
        indices = indices.astype(mx.int16)
        with mx.stream(mx.gpu):
            embedded = mx.take(self.mx_embedding, indices, axis=0)
            sequence = mx.transpose(embedded, (1, 0, 2))
            hidden_fp16 = hidden.astype(mx.float16)
            cell_fp16 = cell.astype(mx.float16)
        seq0 = np.asarray(sequence[:, 0, :], dtype=np.float32)
        hidden_0, next_hidden_0, next_cell_0 = self._lstm(
            seq0, np.asarray(hidden_fp16[0]), np.asarray(cell_fp16[0]), 0
        )
        hidden_1, next_hidden_1, next_cell_1 = self._lstm(
            np.asarray(hidden_0, dtype=np.float32),
            np.asarray(hidden_fp16[1]),
            np.asarray(cell_fp16[1]),
            1,
        )
        with mx.stream(mx.gpu):
            seq = mx.array(np.asarray(hidden_1, dtype=np.float32))
            decoder_hidden = (
                mx.transpose(seq, (1, 0, 2)) @ self.mx_projector.T
                + self.mx_projector_bias
            ).astype(mx.float32)
            next_hidden = mx.stack((next_hidden_0, next_hidden_1), axis=0).astype(mx.float32)
            next_cell = mx.stack((next_cell_0, next_cell_1), axis=0).astype(mx.float32)
            mx.eval(decoder_hidden, next_hidden, next_cell)
        return decoder_hidden, next_hidden, next_cell


def main() -> int:
    started = time.monotonic()
    lock = ReferenceLock.load()
    info = mx.device_info()
    device = {"device_name": info.get("device_name"), "architecture": info.get("architecture")}
    assert "Apple" in str(device["device_name"]), device
    decoder = BnnsDecoder(_MODEL / "decoder.mlpackage")
    joint_package = _MODEL / "joint.mlpackage"

    encoder_host = np.load(_CAPTURE / "encoder_hidden.npy")
    decisions = json.loads((_CAPTURE / "tdt_decisions.json").read_text())["decisions"]
    with mx.stream(mx.gpu):
        encoder = mx.array(encoder_host)
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(encoder, hidden, cell)

    def decoder_callback(token_id, current_hidden, current_cell):
        with mx.stream(mx.gpu):
            dh, nh, nc = decoder(
                mx.array([[token_id]], dtype=mx.int32),
                current_hidden,
                current_cell,
            )
        mx.eval(dh, nh, nc)
        return DecoderStep(dh[0], nh, nc)

    def joint_callback(frame_index, state):
        result = run_joint(
            encoder[:, frame_index, :],
            state,
            package_path=joint_package,
        )
        with mx.stream(mx.gpu):
            token_logits = mx.argmax(result.token_logits, axis=-1)
            duration_index = mx.argmax(result.duration_logits, axis=-1)
        mx.eval(token_logits, duration_index)
        return JointDecision(int(token_logits.item()), int(duration_index.item()))

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
    matches = 0
    for index, (actual, native) in enumerate(zip(actual_emissions, native_emissions)):
        if actual == native:
            matches += 1
        elif divergence is None:
            divergence = {"emission_index": index, "actual": actual, "native": native}
    receipt = {
        "schema": "mlx-omarchy.bnns-contract-free-decode/1",
        "hostname": platform.node(),
        "kernel": platform.release(),
        "mlx_version": mx.__version__,
        "device": device,
        "elapsed_seconds": time.monotonic() - started,
        "decoder_calls": int(getattr(output, "decoder_calls", -1)) if hasattr(output, "decoder_calls") else None,
        "emissions_compared": min(len(actual_emissions), len(native_emissions)),
        "emission_matches": matches,
        "first_divergence": divergence,
        "bnns_contract": {
            "block": _BLOCK,
            "gemv": "one fp16 GEMV over concat[x;h], per-term rnd16 FMA, per-block rnd16 fold",
            "bias": "one fp16 add after fold",
            "unaries": "dense sigma16/tanh16 bit-pattern LUTs",
            "cell": "rnd16(rnd16(sig_f*c0) + sig_i*tanh16(g))",
            "hidden": "rnd16(sig_o*tanh16(c1))",
            "luts_sha256": _sha256(_LUTS.read_bytes()),
        },
    }
    Path("/tmp/vulkan-tdt-142/bnns_decode_result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
