#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-program encoder arithmetic diagnostic for the AC head ladder.

Ladder over the four ANEC islands that compose layer-0's AC head on h13:

  island-select-rta   a_fill/bd/cond → masked
  island-scores-mm    q/k           → scores
  island-add-head     scores/masked → add
  island-softmax-head add           → smax

Each stage:
  - reads its declared captured input from the F dump (the C++ worker
    actually pack/unpacks dense tensors by tensor name per program; the
    client-side wrapper passes raw bytes, the worker slot-routes by
    program-input name, not by some cross-program channel index),
  - runs the ANE program on those exact bytes (transport is
    ``.tobytes()``; the wire counts are byte counts, never
    ``memoryview`` / ``ndarray`` / element counts),
  - computes the GPU reference from the same values via ``mx`` on the
    same device, applying the certified bd reshape chain
    ``pad dim3 +1 → reshape (1,8,750,375) → slice [1:] → reshape
    (1,8,375,749) → slice [:,:,:,:375] → scale 1/16``,
  - decodes the device output with the manifest's per-output stride
    (``outputs[*].stride`` for the produced tensor — never a single
    hardcoded 2310144),
  - reports per-stage ``arithmetic_isolation`` with bit equality,
    signed ULP, finite max-abs, and inf/nan mask agreement.

CPU/host prep only. The actual execution needs the ANE device; on a
flock-held host run

  python3 ac_head_arithmetic_diagnostic.py \\
      --fdump DIR --bundles DIR --out DIR \\
      --worker PATH --libane PATH

Tests ``tests/coreml/test_ac_head_arithmetic_diagnostic.py`` inject a
fake session and a numpy-only stub for ``mlx.core`` so the contract is
exercisable without an ANE device, a GPU, or even mlx installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


# The certified bd reshape chain needs the var_371 scale (1/16). Ladder
# v3's prior driver omitted it; the input here is post-deinterleave
# relpos, the device program also post-multiplies by 1/16, so the GPU
# reference MUST apply it too or the masked stage picks up the wrong
# values and the divergence signature shifts.
BD_SCALE = float.fromhex("0x1p-4")  # 1/16

# Captured tensor names emitted by the F dump (q/k/cond/relpos/a_fill).
# Order is irrelevant for the device (each island is dispatched
# independently and the C++ worker pack/unpacks dense tensors by tensor
# name per program); preserved here for the GPU reference ordering.
CAPTURED_INPUTS = ("q", "k", "cond", "relpos", "a_fill")

# The four islands in pipeline order.
LADDER = (
    "island-select-rta",
    "island-scores-mm",
    "island-add-head",
    "island-softmax-head",
)


# ---------------------------------------------------------------- helpers
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_capture_input(directory: Path, name: str) -> np.ndarray:
    """Load one captured tensor (.bin + .json) by name from the F dump.

    The dump's metadata drives both dtype and shape, so the same loader
    handles the fp16 q/k/relpos, the bool cond, and the scalar a_fill
    without name dispatch. ``a_fill`` is a scalar (broadcast to the
    full (1,8,375,375) at submit time by the device program; the
    captured value is the fp16 scalar bits the runtime sent).
    """
    meta = json.loads((directory / f"{name}.json").read_text())
    raw = (directory / f"{name}.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))
    return arr.reshape(meta["shape"])


def _fill_input(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Per-input broadcast/normalization for the device transport.

    ``a_fill`` is captured as a scalar; the device program reads it as
    a full (1,8,375,375) fp16 surface, so we broadcast here, on the
    client, and ship the full surface as bytes. Same for ``cond``
    (captured as a packed bit plane somewhere — kept as a 1-byte bool
    here in the test fixtures; the device consumes the same byte order
    on this backend).
    """
    if name == "a_fill":
        return np.broadcast_to(value.astype(np.float16), shape).astype(np.float16)
    if name == "cond":
        return np.broadcast_to(value.astype(np.bool_), shape)
    return value


def _output_stride(manifest: dict, tensor: str) -> int:
    """Per-output stride from the bundle manifest.

    The four islands all produce (1,8,375,375) fp16 with stride 2310144
    on this compile, but we look up the produced tensor's stride by
    name — a different compile (different allocator / different
    row-pad strategy) can change it, and the prior driver broke the
    minute that assumption did not hold.
    """
    for out in manifest.get("outputs", []):
        if out["name"] == tensor:
            return int(out["stride"])
    raise ValueError(
        f"bundle {manifest.get('name', '?')} has no output {tensor!r}; "
        f"available: {[o['name'] for o in manifest.get('outputs', [])]}"
    )


def _output_shape(manifest: dict, tensor: str) -> tuple[int, ...]:
    for out in manifest.get("outputs", []):
        if out["name"] == tensor:
            return tuple(out["shape"])
    raise ValueError(f"no output {tensor!r} in manifest")


def _output_dtype(manifest: dict, tensor: str) -> np.dtype:
    for out in manifest.get("outputs", []):
        if out["name"] == tensor:
            return np.dtype(out["dtype"])
    raise ValueError(f"no output {tensor!r} in manifest")


def _to_bytes(value: np.ndarray) -> bytes:
    """Dense contiguous raw bytes for the worker transport.

    The worker's wire protocol counts bytes, not elements; ``memoryview``
    / ``ndarray.tobytes()`` both travel, but ``ndarray.tobytes()`` is
    the only call guaranteed to match the manifest's ``byte_size`` on
    every backend (some islands allocate their input surface larger
    than the logical tensor and the C++ loader pins ``byte_size`` from
    the manifest, not from the supplied array's element count).
    """
    return np.ascontiguousarray(value).tobytes()


# ------------------------------------------------------- per-stage ref chain
def compute_bd_ref(mx, relpos: np.ndarray, scale: float = BD_SCALE) -> np.ndarray:
    """Certified bd reshape chain on the captured relpos.

    pad dim3 +1 → reshape (1,8,750,375) → slice [1:] → reshape
    (1,8,375,749) → slice [:,:,:,:375] → scale 1/16

    The scale comes from the var_371 constant in the head; the
    deinterleave/reshape/slice is the ANEC's "shift-gather" packing
    that the dispatch chain collapses. Without the trailing scale,
    the masked stage diverges by exactly that factor.
    """
    relpos_t = mx.array(relpos)
    padded = mx.pad(relpos_t, [(0, 0), (0, 0), (0, 0), (1, 0)])
    r1 = mx.reshape(padded, (1, 8, 750, 375))
    r2 = r1[:, :, 1:, :]
    r3 = mx.reshape(r2, (1, 8, 375, 749))
    bd = np.asarray(mx.contiguous(r3[:, :, :, :375])).astype(np.float16)
    return (bd.astype(np.float32) * np.float32(scale)).astype(np.float16)


def compute_stage_refs(mx, cap: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """All four GPU references from the captured inputs, same values.

    Every reference is computed from the SAME captured tensor that
    drives the device program for that stage. The chain (select-rta →
    add-head → softmax-head) reads the prior stage's device output for
    the chained comparison, but the GPU references themselves are
    derived from the original captured inputs so the per-stage
    comparison isolates each program from its peers.
    """
    q_t = mx.array(cap["q"])
    k_t = mx.array(cap["k"])
    cond_t = mx.array(cap["cond"])
    bd_ref = compute_bd_ref(mx, cap["relpos"])
    fill_t = mx.array(np.full((1, 8, 375, 375), float("-inf"), np.float16))
    masked_ref = np.asarray(
        mx.where(cond_t, fill_t, mx.array(bd_ref))
    ).astype(np.float16)
    scores_ref = np.asarray(
        mx.matmul(q_t, mx.transpose(k_t, (0, 1, 3, 2)))
    ).astype(np.float16)
    add_ref = (scores_ref + masked_ref).astype(np.float16)
    smax_ref = np.asarray(mx.softmax(mx.array(add_ref), axis=-1)).astype(np.float16)
    return {
        "bd": bd_ref,
        "masked": masked_ref,
        "scores": scores_ref,
        "add": add_ref,
        "smax": smax_ref,
    }


# ------------------------------------------------------- per-stage compare
def compare_record(dev: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Bit/ULP/inf/nan comparison record (single source of truth).

    Reuses the same body as the F-handler's per-program comparison so
    a failing bit/ULP here matches the F handler's; the F handler
    reused this contract, this is the original. -0 / +0 distance 0;
    a sign-crossing pair (-1denorm, +1denorm) distance 2.
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


# ----------------------------------------------------------- per-stage run
# Logical tensor name per stage → (bundle, set of input names).
STAGE_PLAN: dict[str, dict[str, Any]] = {
    "island-select-rta": {
        "bundle": "island-select-rta",
        "inputs": ("a_fill", "bd", "cond"),
        "outputs": ("masked",),
        "ref_name": "masked",
    },
    "island-scores-mm": {
        "bundle": "island-scores-mm",
        "inputs": ("q", "k"),
        "outputs": ("scores",),
        "ref_name": "scores",
    },
    "island-add-head": {
        "bundle": "island-add-head",
        "inputs": ("scores", "masked"),
        "outputs": ("add",),
        "ref_name": "add",
    },
    "island-softmax-head": {
        "bundle": "island-softmax-head",
        "inputs": ("add",),
        "outputs": ("smax",),
        "ref_name": "smax",
    },
}


def _run_stage(
    stage: str,
    *,
    submit: Callable[..., dict[str, bytes]],
    manifests: dict[str, dict],
    cap: Mapping[str, np.ndarray],
    refs: Mapping[str, np.ndarray],
    chained_outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run one island and compare device output to GPU reference.

    The device transport contract:

      - inputs are raw bytes (the worker's wire count is byte count)
      - the device returns raw bytes (the produced surface is read with
        the manifest's per-output ``stride``)
      - cross-program values (scores/masked/add) come from the prior
        stage's decoded device output, not from the GPU reference, so
        the chaining isolates the consumer stage's arithmetic from any
        bias the producer's element mismatch might introduce.
    """
    plan = STAGE_PLAN[stage]
    bundle = plan["bundle"]
    manifest = manifests[bundle]
    # Build the input bytes — per program, by tensor name. The C++ side
    # pack/unpacks dense tensors by name per program; per-program
    # handles map input positions, cross-program by tensor name.
    inputs: dict[str, bytes] = {}
    for input_name in plan["inputs"]:
        if input_name in cap:
            value = _fill_input(input_name, cap[input_name], (1, 8, 375, 375))
        elif input_name in chained_outputs:
            value = chained_outputs[input_name]
        else:
            raise ValueError(
                f"stage {stage} input {input_name!r} not in capture nor "
                "in chained_outputs"
            )
        inputs[input_name] = _to_bytes(value)
    got = submit(bundle, stage, inputs, list(plan["outputs"]))
    produced: dict[str, np.ndarray] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for output_name in plan["outputs"]:
        raw = bytes(got[output_name])
        shape = _output_shape(manifest, output_name)
        dtype = _output_dtype(manifest, output_name)
        stride = _output_stride(manifest, output_name)
        decoded = _decode_dense(raw, shape, dtype, stride)
        produced[output_name] = decoded
        rec = compare_record(decoded, refs[plan["ref_name"]])
        rec["stage"] = stage
        rec["output"] = output_name
        rec["ref"] = plan["ref_name"]
        comparisons[output_name] = rec
    return {"inputs": inputs, "produced": produced, "comparisons": comparisons}


def _decode_dense(raw: bytes, shape, dtype, stride: int) -> np.ndarray:
    """Decode a strided ANEC output surface into the dense logical array.

    Mirrors the F-handler decode (per-batch-stride slice); the
    arithmetic diagnostic needs the same dense form for
    ``compare_record``.
    """
    dt = np.dtype(dtype)
    rows = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    row_elems = shape[-2] * shape[-1]
    row_bytes = row_elems * dt.itemsize
    if stride <= 0:
        # Refuse to silently treat 0 as contiguous. A real ANEC
        # allocator always reports a non-zero stride on this backend;
        # a 0 here means we forgot to read ``manifest.outputs[*]``.
        raise ValueError(
            f"stride missing/0 for {np.dtype(dtype).name}{shape}; "
            "manifest.outputs[*].stride was not looked up"
        )
    if stride < row_bytes:
        raise ValueError(f"stride {stride} < row_bytes {row_bytes}")
    out = np.empty(rows * row_elems, dtype=dt)
    for r in range(rows):
        chunk = np.frombuffer(
            raw, dtype=dt, count=row_elems, offset=r * stride
        )
        out[r * row_elems: (r + 1) * row_elems] = chunk
    return out.reshape(shape)


# ----------------------------------------------------------- driver entry
def diagnose_ladder(
    *,
    cap: Mapping[str, np.ndarray],
    submit: Callable[..., dict[str, bytes]],
    manifests: Mapping[str, dict],
    out: Path | None = None,
) -> dict[str, Any]:
    """Run the four-stage ladder and report per-stage arithmetic isolation.

    ``cap`` is the captured input dict (q/k/cond/relpos/a_fill); the
    GPU references are computed via the lazy ``mx`` import below.
    ``submit`` is the resident worker's ``submit(bundle, tag, inputs,
    outputs)``. ``manifests`` is a ``{bundle_name: manifest_dict}`` map
    read from the bundle root (the worker reads it itself; we just need
    it here for output stride/dtype/shape lookup).

    bd is computed from the captured ``relpos`` via the certified
    pad-reshape-slice-scale chain (slice-head is excluded from this
    ladder — its device program is known to crash with the strided
    ``[1,8,375,749]`` binding on this backend, so the same chain on
    the same device produces bd in the host before submit). The
    diagnostic then ships bd as a bytes input to ``island-select-rta``
    — same value, same device, no cross-program channel routing.

    Returns a dict with keys ``arithmetic_isolation`` (one entry per
    stage with the comparison record) and ``chaining`` (the per-stage
    device outputs that fed the next stage's chained run).
    """
    # Lazy mx so tests can stub ``sys.modules['mlx.core']`` without
    # installing mlx. The lazy import is the production path; the stub
    # path is the test seam.
    mx = sys.modules.get("mlx.core")
    if mx is None:
        import mlx.core as mx  # type: ignore[no-redef]  # noqa: WPS433
    bd = compute_bd_ref(mx, cap["relpos"])
    # Inject bd into the captured set so the per-stage input builder
    # finds it the same way it finds ``a_fill`` and ``cond``.
    cap_with_bd = dict(cap)
    cap_with_bd["bd"] = bd
    refs = compute_stage_refs(mx, cap_with_bd)
    report: dict[str, Any] = {
        "arithmetic_isolation": {},
        "chaining": {},
        "bd_scale": BD_SCALE,
        "ladder": list(LADDER),
    }
    chained_outputs: dict[str, np.ndarray] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for nm, ref in refs.items():
            np.save(out / f"gpu-ref-{nm}.npy", ref)
    for stage in LADDER:
        result = _run_stage(
            stage,
            submit=submit,
            manifests=dict(manifests),
            cap=cap_with_bd,
            refs=refs,
            chained_outputs=chained_outputs,
        )
        report["arithmetic_isolation"][stage] = result["comparisons"]
        for output_name, decoded in result["produced"].items():
            chained_outputs[output_name] = decoded
            if out is not None:
                np.save(out / f"device-{output_name}.npy", decoded)
        report["chaining"][stage] = {
            output_name: {
                "shape": list(arr.shape),
                "sha256": _sha256(arr.tobytes()),
                "bytes": int(arr.nbytes),
            }
            for output_name, arr in result["produced"].items()
        }
    return report


def _fake_submit_factory(bundle_root: Path, manifests: dict):
    """A no-ANE CPU-only stub for the resident worker ``submit``.

    Each island's "device output" is the GPU reference itself (so a
    passing test asserts the per-stage comparison is bit-equal to the
    reference). This is intentionally only used by the in-test driver
    that monkeypatches ``ane_resident``'s session — the real diagnostic
    run on a flock-held host calls the actual ``ResidentAneWorker``.
    """
    def submit(bundle, tag, inputs, outputs):
        if bundle not in manifests:
            raise RuntimeError(f"fake submit: unknown bundle {bundle}")
        return {
            outputs[0]: refs_for_bundle(bundle, manifests, inputs)
        }
    return submit


def refs_for_bundle(bundle, manifests, inputs):
    """Return the GPU reference bytes for ``bundle`` (test seam only).

    The test seam computes the same chain on a numpy stub of ``mx`` and
    packs the per-output dense bytes. ``inputs`` is accepted to match
    the call signature; its content is irrelevant for the fake (we use
    the chained GPU refs, not the device's reprocessing).
    """
    raise NotImplementedError("wired by tests via monkeypatch")


# ----------------------------------------------------------- CLI entry
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdump", type=Path, required=True,
                        help="directory holding captured .bin + .json")
    parser.add_argument("--bundles", type=Path, required=True,
                        help="bundle root with the four island bundles")
    parser.add_argument("--worker", type=Path, required=True,
                        help="mlx-omarchy-ane-worker binary")
    parser.add_argument("--libane", type=Path, required=True,
                        help="libane-shared.so path")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for refs + report")
    parser.add_argument("--deadline-ms", type=int, default=20000)
    parser.add_argument("--relay-bypass", action="store_true",
                        help="use --relay-bypass mode (default off)")
    args = parser.parse_args()

    cap = {
        name: _load_capture_input(args.fdump, name) for name in CAPTURED_INPUTS
    }

    manifests = {
        name: json.loads((args.bundles / name / "manifest.json").read_text())
        for name in LADDER
    }

    # The worker / resident is loaded here. The diagnostic does not
    # start a session if --bundles doesn't contain every ladder island.
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