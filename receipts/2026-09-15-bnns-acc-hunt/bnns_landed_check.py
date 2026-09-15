#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Validate the LANDED vulkan_decoder (BNNS contract) on jwm1.

Free decode over the authenticated capture with the landed module, plus
bit-exact next_cell/next_hidden checks on every ran_decoder capture trace.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

from coreml.parakeet_tdt import DecoderStep, JointDecision, greedy_tdt_decode
from coreml.reference import ReferenceLock
from coreml.vulkan_decoder import load_decoder
from coreml.vulkan_joint import run_joint

CAPTURE = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/capture/ane")
MODEL = Path("/home/joshuawarren/tmp/coreml-tdt-0cf2d148/model")


def main() -> int:
    started = time.monotonic()
    lock = ReferenceLock.load()
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    joint_package = MODEL / "joint.mlpackage"

    encoder_host = np.load(CAPTURE / "encoder_hidden.npy")
    decisions = json.loads((CAPTURE / "tdt_decisions.json").read_text())["decisions"]
    traces = json.loads((CAPTURE / "tdt_tensors.json").read_text())["traces"]

    # Bit-exact single transitions on native inputs.
    trace_report = []
    for trace in traces:
        if not trace.get("ran_decoder"):
            continue
        paths = trace["tensor_paths"]
        ids = np.load(CAPTURE / paths["decoder_input_ids"])
        hid = np.load(CAPTURE / paths["decoder_hidden"])
        cel = np.load(CAPTURE / paths["decoder_cell"])
        nat_h = np.load(CAPTURE / paths["decoder_next_hidden"])
        nat_c = np.load(CAPTURE / paths["decoder_next_cell"])
        with mx.stream(mx.gpu):
            dh, nh, nc = decoder(mx.array(ids), mx.array(hid), mx.array(cel))
        mx.eval(dh, nh, nc)
        nh_a = np.asarray(nh)
        nc_a = np.asarray(nc)
        trace_report.append({
            "index": trace["index"],
            "next_hidden_bitexact": int((nh_a.astype(np.float16).view(np.uint16)
                                         == nat_h.astype(np.float16).view(np.uint16)).sum()),
            "next_cell_bitexact": int((nc_a.astype(np.float16).view(np.uint16)
                                       == nat_c.astype(np.float16).view(np.uint16)).sum()),
            "lanes": nat_h.size,
        })
    exact_h = sum(r["next_hidden_bitexact"] for r in trace_report)
    exact_c = sum(r["next_cell_bitexact"] for r in trace_report)
    lanes = sum(r["lanes"] for r in trace_report)

    # Free decode.
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
        result = run_joint(encoder[:, frame_index, :], state, package_path=joint_package)
        with mx.stream(mx.gpu):
            tok = mx.argmax(result.token_logits, axis=-1)
            dur = mx.argmax(result.duration_logits, axis=-1)
        mx.eval(tok, dur)
        return JointDecision(int(tok.item()), int(dur.item()))

    output = greedy_tdt_decode(
        valid_frames=int(encoder.shape[1]),
        config=lock.tdt,
        initial_hidden=hidden,
        initial_cell=cell,
        run_decoder=decoder_callback,
        run_joint=joint_callback,
    )
    native_emissions = [
        {"token_id": i["token_id"], "frame_index": i["frame_before"], "duration": i["duration"]}
        for i in decisions if i["token_id"] != lock.tdt.blank_token_id
    ]
    actual_emissions = [
        {"token_id": t, "frame_index": f, "duration": d}
        for t, f, d in zip(output.token_ids, output.frame_indices, output.durations, strict=True)
    ]
    divergence = None
    matches = 0
    for index, (actual, native) in enumerate(zip(actual_emissions, native_emissions)):
        if actual == native:
            matches += 1
        elif divergence is None:
            divergence = {"emission_index": index, "actual": actual, "native": native}
    receipt = {
        "schema": "mlx-omarchy.bnns-landed-decoder-check/1",
        "hostname": platform.node(),
        "mlx_version": mx.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "trace_bitexact": {
            "next_hidden": f"{exact_h}/{lanes}",
            "next_cell": f"{exact_c}/{lanes}",
        },
        "emissions_compared": min(len(actual_emissions), len(native_emissions)),
        "emission_matches": matches,
        "first_divergence": divergence,
    }
    Path("/tmp/vulkan-tdt-142/bnns_landed_check.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
