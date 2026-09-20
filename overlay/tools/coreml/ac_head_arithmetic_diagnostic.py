#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-program encoder arithmetic diagnostic for the AC head ladder.

Ladder over the four ANEC islands that compose layer-0's AC head on h13:

  island-select-rta   a_fill/bd/cond → masked
  island-scores-mm    q/k           → scores
  island-add-head     scores/masked → add
  island-softmax-head add           → smax

Wire contract (verified against worker.cpp / worker_libane.cpp /
tile_layout.h on t6001-test-host):

  - inputs travel as raw bytes; the worker checks
    ``payload.size() < logical_bytes`` (worker.cpp:118-122) and the
    LibaneDevice::send packs dense -> tile internally
    (worker_libane.cpp:104-114, ane_pack_rows).
  - outputs come back as raw bytes whose length is ``logical_bytes``
    (= manifest outputs[*].byte_size); the values are the DENSE
    logical row-major array (worker_libane.cpp:125-138,
    ane_unpack_rows writes row-major from tile). The caller reshapes
    to the manifest's ``outputs[*].shape``; no caller-side stride
    decode.
  - outputs[*].stride is the ANEC tile allocator stride, not the
    wire format; ignore it on the wire path.

Each stage computes its GPU reference from the exact arrays
serialized for that stage (same-input arithmetic isolation):
select-rta's masked ref uses the device-input a_fill / bd / cond
that the test actually stages; add-head's add ref uses the device-
output scores + device-output masked that the test actually stages.
A pristine reference set (from the original captured inputs) is
kept separately for the chaining delta only.

CPU/host prep only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


BD_SCALE = float.fromhex("0x1p-4")
CAPTURED_INPUTS = ("q", "k", "cond", "relpos", "a_fill")
LADDER = (
    "island-select-rta",
    "island-scores-mm",
    "island-add-head",
    "island-softmax-head",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_capture_input(directory: Path, name: str) -> np.ndarray:
    """Load one captured tensor (.bin + .json) by name from the F dump."""
    meta = json.loads((directory / f"{name}.json").read_text())
    raw = (directory / f"{name}.bin").read_bytes()
    return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])


def _output_meta(manifest: dict, tensor: str) -> tuple[tuple[int, ...], np.dtype]:
    for out in manifest.get("outputs", []):
        if out["name"] == tensor:
            return tuple(out["shape"]), np.dtype(out["dtype"])
    raise ValueError(f"no output {tensor!r} in manifest")


def _to_bytes(value: np.ndarray) -> bytes:
    """Dense contiguous raw bytes; the worker recv reads ``payload.size()``
    and rejects anything shorter than ``logical_bytes``.
    """
    return np.ascontiguousarray(value).tobytes()


def _mx(mx, x):
    return x if hasattr(x, "astype") else mx.array(np.asarray(x))


def compute_bd_ref(mx, relpos: np.ndarray, scale: float = BD_SCALE) -> np.ndarray:
    """Certified bd reshape chain on the captured relpos.

    pad dim3 +1 -> reshape (1,8,750,375) -> slice [1:] -> reshape
    (1,8,375,749) -> slice [:,:,:,:375] -> scale 1/16.
    """
    relpos_t = mx.array(relpos)
    padded = mx.pad(relpos_t, [(0, 0), (0, 0), (0, 0), (1, 0)])
    r1 = mx.reshape(padded, (1, 8, 750, 375))
    r2 = r1[:, :, 1:, :]
    r3 = mx.reshape(r2, (1, 8, 375, 749))
    bd = np.asarray(mx.contiguous(r3[:, :, :, :375])).astype(np.float16)
    return (bd.astype(np.float32) * np.float32(scale)).astype(np.float16)


def compare_record(dev: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Bit/ULP/inf/nan comparison record (signed-ULP semantics).

    -0/+0 -> bit_equal=False, signed-ULP distance 0.
    Sign-crossing pair of one-denorms -> bit_equal=False, signed-ULP
    distance 2 (crosses the zero).
    """
    d16 = np.ascontiguousarray(dev).astype(np.float16)
    r16 = np.ascontiguousarray(ref).astype(np.float16)
    d32 = d16.astype(np.float32)
    r32 = r16.astype(np.float32)
    diff = np.abs(d32 - r32)
    finite = np.isfinite(diff)
    d16v, r16v = d16.view(np.uint16), r16.view(np.uint16)
    ai = d16v.astype(np.int32)
    bi = r16v.astype(np.int32)
    sa = np.where(ai & 0x8000, -(ai & 0x7FFF), ai & 0x7FFF)
    sb = np.where(bi & 0x8000, -(bi & 0x7FFF), bi & 0x7FFF)
    ulp = np.abs(sa - sb)
    finite_mask = (
        np.isfinite(d32) & np.isfinite(r32)
        & ~np.isnan(d32) & ~np.isnan(r32)
    )
    return {
        "bit_equal": bool(np.array_equal(d16v, r16v)),
        "mismatch_count": int((d16v != r16v).sum()),
        "total": int(d16v.size),
        "max_abs_finite": (
            float(diff[finite].max()) if finite.any() else None
        ),
        "dev_neginf": int(np.isneginf(d32).sum()),
        "ref_neginf": int(np.isneginf(r32).sum()),
        "dev_posinf": int(np.isposinf(d32).sum()),
        "ref_posinf": int(np.isposinf(r32).sum()),
        "dev_nan": int(np.isnan(d32).sum()),
        "ref_nan": int(np.isnan(r32).sum()),
        "inf_mask_agree": bool(np.array_equal(np.isinf(d32), np.isinf(r32))),
        "nan_mask_agree": bool(np.array_equal(np.isnan(d32), np.isnan(r32))),
        "ulp_max_finite": (
            int(ulp[finite_mask].max()) if finite_mask.any() else None
        ),
    }


def _decode_dense(raw: bytes, manifest: dict, tensor: str) -> np.ndarray:
    """Decode wire output bytes to the manifest's output shape.

    The worker emits dense logical bytes (worker_libane.cpp:125-138,
    ane_unpack_rows row-major from tile); reshape is the only
    transformation needed.
    """
    shape, dt = _output_meta(manifest, tensor)
    arr = np.frombuffer(raw, dtype=dt)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"wire bytes for {tensor!r}: got {arr.size} elements, "
            f"manifest shape {shape} needs {expected}"
        )
    return arr.reshape(shape)


# Per-stage input normalization + the SAME-INPUT GPU ref each stage uses.
# Each ref is built from the exact array values serialized for that
# stage's submit, so the per-stage comparison isolates that program's
# arithmetic from any bias the producer's element mismatch might add.
def _normalize(name: str, value: np.ndarray) -> np.ndarray:
    """Per-input wire normalization.

    cond is captured as (1,1,375,375) byte-bool; the device reads
    (1,8,375,375) so we broadcast on the client. a_fill is captured
    as a scalar fp16; the device reads (1,8,375,375).
    """
    if name == "cond":
        return np.broadcast_to(value.astype(np.bool_), (1, 8, 375, 375))
    if name == "a_fill":
        return np.broadcast_to(value.astype(np.float16), (1, 8, 375, 375)).astype(np.float16)
    return value


def _stage_ref(mx, stage: str, serialized_inputs: dict[str, np.ndarray]) -> np.ndarray:
    """GPU reference computed from the SAME arrays the device receives.

    select-rta: where(cond, a_fill, bd) -> masked
    scores-mm:  matmul(q, transpose(k, (0,1,3,2))) -> scores
    add-head:   scores + masked -> add
    softmax-head: softmax(add, axis=-1) -> smax
    """
    if stage == "island-select-rta":
        fill = np.full((1, 8, 375, 375), float("-inf"), np.float16)
        return np.asarray(
            mx.where(
                mx.array(serialized_inputs["cond"]),
                mx.array(serialized_inputs["a_fill"]),
                mx.array(serialized_inputs["bd"]),
            )
        ).astype(np.float16)
    if stage == "island-scores-mm":
        q = mx.array(serialized_inputs["q"])
        k = mx.array(serialized_inputs["k"])
        return np.asarray(
            mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
        ).astype(np.float16)
    if stage == "island-add-head":
        return (
            serialized_inputs["scores"] + serialized_inputs["masked"]
        ).astype(np.float16)
    if stage == "island-softmax-head":
        return np.asarray(
            mx.softmax(mx.array(serialized_inputs["add"]), axis=-1)
        ).astype(np.float16)
    raise ValueError(f"unknown stage {stage!r}")


STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    "island-select-rta": ("a_fill", "bd", "cond"),
    "island-scores-mm": ("q", "k"),
    "island-add-head": ("scores", "masked"),
    "island-softmax-head": ("add",),
}

STAGE_OUTPUT: dict[str, str] = {
    "island-select-rta": "masked",
    "island-scores-mm": "scores",
    "island-add-head": "add",
    "island-softmax-head": "smax",
}


def _run_stage(
    stage: str,
    *,
    submit: Callable[..., dict[str, bytes]],
    manifests: dict[str, dict],
    cap: Mapping[str, np.ndarray],
    chained: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Submit one island; compare device output against a ref built
    from the SAME arrays the device just received.
    """
    bundle = manifests[stage]
    serialized: dict[str, np.ndarray] = {}
    inputs: dict[str, bytes] = {}
    for input_name in STAGE_INPUTS[stage]:
        if input_name in cap:
            value = _normalize(input_name, cap[input_name])
        elif input_name in chained:
            value = chained[input_name]
        else:
            raise ValueError(
                f"stage {stage} input {input_name!r} not in capture nor "
                "in chained"
            )
        serialized[input_name] = value
        inputs[input_name] = _to_bytes(value)
    output = STAGE_OUTPUT[stage]
    got = submit(stage, stage, inputs, [output])
    decoded = _decode_dense(bytes(got[output]), bundle, output)
    return decoded, {"comparisons": compare_record(decoded, _stage_ref(globals().get("mx", sys.modules["mlx.core"]), stage, serialized)), "decoded": decoded}


def diagnose_ladder(
    *,
    cap: Mapping[str, np.ndarray],
    submit: Callable[..., dict[str, bytes]],
    manifests: Mapping[str, dict],
    out: Path | None = None,
) -> dict[str, Any]:
    """Run the four-stage ladder; per-stage refs are SAME-INPUT.

    bd is derived from the captured ``relpos`` via the certified
    pad-reshape-slice-scale chain (slice-head is excluded — strided
    [1,8,375,749] binding crashes on this backend).
    """
    mx = sys.modules.get("mlx.core")
    if mx is None:
        import mlx.core as mx  # type: ignore[no-redef]  # noqa: WPS433
    cap_with_bd = dict(cap)
    cap_with_bd["bd"] = compute_bd_ref(mx, cap["relpos"])
    report: dict[str, Any] = {
        "arithmetic_isolation": {},
        "chaining": {},
        "bd_scale": BD_SCALE,
        "ladder": list(LADDER),
    }
    chained: dict[str, np.ndarray] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for stage in LADDER:
        # Same-input ref: ref must be built from the same values
        # _run_stage just serialized; expose mx for the per-stage ref.
        sys.modules["mlx.core"] = mx  # ensure _stage_ref uses it
        bundle = manifests[stage]
        serialized: dict[str, np.ndarray] = {}
        inputs: dict[str, bytes] = {}
        for input_name in STAGE_INPUTS[stage]:
            if input_name in cap_with_bd:
                value = _normalize(input_name, cap_with_bd[input_name])
            elif input_name in chained:
                value = chained[input_name]
            else:
                raise ValueError(
                    f"stage {stage} input {input_name!r} not in capture "
                    "nor in chained"
                )
            serialized[input_name] = value
            inputs[input_name] = _to_bytes(value)
        output = STAGE_OUTPUT[stage]
        got = submit(stage, stage, inputs, [output])
        decoded = _decode_dense(bytes(got[output]), bundle, output)
        ref = _stage_ref(mx, stage, serialized)
        rec = compare_record(decoded, ref)
        rec["stage"] = stage
        rec["output"] = output
        report["arithmetic_isolation"][stage] = {output: rec}
        chained[output] = decoded
        if out is not None:
            np.save(out / f"device-{output}.npy", decoded)
            np.save(out / f"ref-{output}.npy", ref)
        report["chaining"][stage] = {
            output: {
                "shape": list(decoded.shape),
                "sha256": _sha256(decoded.tobytes()),
                "bytes": int(decoded.nbytes),
            }
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdump", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--libane", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=int, default=20000)
    parser.add_argument("--relay-bypass", action="store_true")
    args = parser.parse_args()

    cap = {name: _load_capture_input(args.fdump, name) for name in CAPTURED_INPUTS}
    manifests = {
        name: json.loads((args.bundles / name / "manifest.json").read_text())
        for name in LADDER
    }
    sys.path.insert(0, str((Path(__file__).resolve().parent)))
    from ane_resident import ResidentAneWorker
    session = ResidentAneWorker(
        worker=args.worker,
        libane=args.libane,
        bundles={name: args.bundles / name for name in LADDER},
        scratch=args.out / "scratch",
        deadline_ms=args.deadline_ms,
        relay_bypass=True if args.relay_bypass else None,
    )
    session.start()
    try:
        report = diagnose_ladder(
            cap=cap,
            submit=session.submit,
            manifests=manifests,
            out=args.out,
        )
    finally:
        session.close()
    (args.out / "arithmetic-diagnostic-report.json").write_text(
        json.dumps(report, indent=2)
    )
    for stage, comps in report["arithmetic_isolation"].items():
        for tensor, rec in comps.items():
            print(
                f"{stage} {tensor}: mismatch {rec['mismatch_count']}/"
                f"{rec['total']} bit_equal={rec['bit_equal']} "
                f"max_abs_finite={rec['max_abs_finite']} "
                f"ulp_max_finite={rec['ulp_max_finite']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())