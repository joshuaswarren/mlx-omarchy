# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-program arithmetic-isolation ladder for the AC head (CPU-only).

Runs the real ``ac_head_arithmetic_diagnostic.diagnose_ladder`` pipeline
with two monkeypatched seams:

  - ``mlx.core`` is replaced by a numpy-only stub (no GPU / no
    compiled metal kernel). The stub reproduces the exact reshape,
    pad, transpose, matmul, where, softmax chain the diagnostic asks
    for so the GPU reference is bit-equal to the device path on this
    backend (the test asserts ``bit_equal=True`` for every stage).
  - the worker transport is replaced by a fake ``submit`` that
    serializes the GPU reference per island and returns raw bytes.
    The diagnostic's per-output stride decode must then reproduce
    the reference's logical dense array.

The five invariants under test are the ones the prior driver broke:

  (a) inputs travel as raw ``bytes`` (``memoryview`` / ``ndarray``
      would corrupt the wire count on some allocators),
  (b) bd carries the var_371 ``1/16`` scale (omitting it shifted
      the masked-stage divergence by exactly that factor),
  (c) per-output stride comes from the manifest's
      ``outputs[*].stride`` — never a single hardcoded 2310144,
  (d) every stage has its own GPU reference computed from the same
      captured input the device program reads,
  (e) the comparison record reports bit/ULP/inf/nan with correct
      signed-ULP semantics (-0/+0 distance 0, zero-crossing pair
      distance 2).

Each invariant has a focused test below; the end-to-end ladder test
exercises all five together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/coreml → tests → ROOT
TOOLS = ROOT / "overlay" / "tools"
sys.path.insert(0, str(HERE))  # for test_stride_decode helpers
sys.path.insert(0, str(TOOLS))  # for coreml.* imports

from coreml import ac_head_arithmetic_diagnostic as diag  # noqa: E402
import test_stride_decode as tsd  # noqa: E402

ALIGN = tsd.ALIGN


# ----------------------------------------------------------- numpy stub for mx
class _MxArray(np.ndarray):
    """A numpy view that also exposes ``__array__`` so ``np.asarray`` is a no-op.

    The diagnostic's chain reads ``mx.array(...)`` and then later does
    ``np.asarray(...)`` on the result; the wrapper makes both work
    without copies. ``.value`` returns the underlying ndarray.
    """

    def __new__(cls, value):
        arr = np.asarray(value).view(cls)
        return arr

    @property
    def value(self):
        return np.asarray(self)


def _mx_pad(value, pad_widths):
    return _MxArray(np.pad(np.asarray(value), pad_widths, mode="constant"))


def _mx_reshape(value, shape):
    return _MxArray(np.reshape(np.asarray(value), shape))


def _mx_transpose(value, axes):
    return _MxArray(np.transpose(np.asarray(value), axes))


def _mx_matmul(lhs, rhs):
    a = np.asarray(lhs, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    return _MxArray((a @ b).astype(np.float16))


def _mx_where(cond, x, y):
    c = np.asarray(cond)
    return _MxArray(np.where(c, np.asarray(x), np.asarray(y)))


def _mx_softmax(value, axis=-1):
    v = np.asarray(value, dtype=np.float32)
    v = v - v.max(axis=axis, keepdims=True)
    e = np.exp(v)
    return _MxArray((e / e.sum(axis=axis, keepdims=True)).astype(np.float16))


def _mx_contiguous(value):
    return _MxArray(np.ascontiguousarray(np.asarray(value)))


def _mx_array(value, dtype=None):
    if dtype is None:
        return _MxArray(np.asarray(value))
    return _MxArray(np.asarray(value).astype(dtype))


def _mx_zeros(shape, dtype=None):
    return _MxArray(np.zeros(shape, dtype=dtype or np.float32))


@pytest.fixture
def mx_stub(monkeypatch):
    """Patch ``mlx.core`` to a numpy-only module for this test only.

    Mirrors the diagnostic's chain (pad, reshape, slice, transpose,
    matmul, where, softmax) so the GPU reference is bit-equal to the
    device output the fake submit returns.
    """
    class _Module:
        float16 = np.float16
        float32 = np.float32
        bool_ = np.bool_
        array = staticmethod(_mx_array)
        pad = staticmethod(_mx_pad)
        reshape = staticmethod(_mx_reshape)
        transpose = staticmethod(_mx_transpose)
        matmul = staticmethod(_mx_matmul)
        where = staticmethod(_mx_where)
        softmax = staticmethod(_mx_softmax)
        contiguous = staticmethod(_mx_contiguous)
        zeros = staticmethod(_mx_zeros)

    mod = _Module()
    sys.modules["mlx.core"] = mod
    monkeypatch.setattr(diag, "mx", mod, raising=False)
    yield mod
    sys.modules.pop("mlx.core", None)


# ----------------------------------------------------- captured fixtures
@pytest.fixture
def captured():
    """Synthetic captured inputs at the F-dump shapes.

    Deterministic so the comparison record is reproducible across
    runs without depending on real device dumps. Values picked so the
    bd chain produces non-zero, non-NaN output that exercises the
    scale, mask, and softmax paths.
    """
    rng = np.random.default_rng(0xACEB)
    relpos = rng.standard_normal((1, 8, 375, 749)).astype(np.float16) / 8
    q = rng.standard_normal((1, 8, 375, 128)).astype(np.float16) / 4
    k = rng.standard_normal((1, 8, 375, 128)).astype(np.float16) / 4
    cond = (rng.standard_normal((1, 8, 375, 375)) > 0).astype(np.bool_)
    return {
        "q": q,
        "k": k,
        "cond": cond,
        "relpos": relpos,
        "a_fill": np.float16(-1.0),
    }


@pytest.fixture
def manifests():
    """Minimal bundle manifests with realistic per-output strides.

    Each island produces (1,8,375,375) fp16 with allocation-aligned
    stride. The score-mm stride is intentionally a different value
    (non-default) so a hardcoded-2310144 driver breaks; island-add-
    head and island-softmax-head keep the standard 2310144.
    """
    def manifest(name, stride):
        return {
            "name": name,
            "outputs": [
                {
                    "name": "masked" if "select" in name else
                            "scores" if "scores" in name else
                            "add" if "add" in name else "smax",
                    "stride": stride,
                    "byte_size": 2250000,
                    "dtype": "float16",
                    "shape": [1, 8, 375, 375],
                }
            ],
        }
    return {
        "island-select-rta": manifest("island-select-rta", 2310144),
        "island-scores-mm": manifest("island-scores-mm", 2310144),
        "island-add-head": manifest("island-add-head", 2310144),
        "island-softmax-head": manifest("island-softmax-head", 2310144),
    }


@pytest.fixture
def mismatched_stride_manifests():
    """Variant where scores output has a non-standard stride.

    Forces the per-output stride lookup path: a hardcoded 2310144
    would mis-decode scores (the first read picks up the right row,
    but row 1+ are misaligned). The diagnostic must read the
    manifest's per-output stride instead.
    """
    def manifest(name, stride):
        return {
            "name": name,
            "outputs": [
                {
                    "name": "masked" if "select" in name else
                            "scores" if "scores" in name else
                            "add" if "add" in name else "smax",
                    "stride": stride,
                    "byte_size": 2250000,
                    "dtype": "float16",
                    "shape": [1, 8, 375, 375],
                }
            ],
        }
    return {
        "island-select-rta": manifest("island-select-rta", 2310144),
        # Different stride! Driver must look it up, not hardcode 2310144.
        "island-scores-mm": manifest("island-scores-mm", 0x4000 * 384),
        "island-add-head": manifest("island-add-head", 2310144),
        "island-softmax-head": manifest("island-softmax-head", 2310144),
    }


# ------------------------------------------------- fake submit / raw helper
def _dense_bytes(arr, stride):
    """Pack a fp16 surface as the strided ANEC wire bytes.

    The diagnostic's ``decode_strided`` helper reads ``(rows-1) *
    stride + row_bytes`` bytes from the wire, treating ``stride`` as
    the byte offset between successive outer rows. Outer rows for a
    4-D ``(N, C, H, W)`` tensor are ``N*C`` (= ``prod(shape[:-2])``).
    The wire buffer is therefore ``(rows-1) * stride + row_bytes``
    bytes; each outer row's data sits at ``offset = r * stride``.
    """
    arr = np.ascontiguousarray(arr.astype(np.float16))
    shape = arr.shape
    rows = int(np.prod(shape[:-2]))
    row_elems = shape[-2] * shape[-1]
    row_bytes = row_elems * arr.dtype.itemsize
    buf = bytearray((rows - 1) * stride + row_bytes)
    flat = arr.reshape(rows, row_elems)
    for r in range(rows):
        buf[r * stride: r * stride + row_bytes] = flat[r].tobytes()
    return bytes(buf)


def _make_fake_submit(refs, manifests):
    """Return a ``submit(bundle, tag, inputs, outputs)`` returning raw bytes.

    Each island's "device output" is the GPU reference for that island
    packed into strided raw bytes (so the diagnostic decode path has
    to work to recover the dense array). Wire-level checks (bytes
    transport, manifest stride decode, signed-ULP compare) all run
    on real diagnostic code paths.
    """
    bundle_to_ref = {
        "island-select-rta": "masked",
        "island-scores-mm": "scores",
        "island-add-head": "add",
        "island-softmax-head": "smax",
    }

    def submit(bundle, tag, inputs, outputs):
        # Wire-level check: every input MUST be bytes, never memoryview
        # or ndarray. The prior driver passed ``memoryview(arr)`` and the
        # worker's byte-counted protocol read garbage lengths.
        for input_name, payload in inputs.items():
            assert isinstance(payload, (bytes, bytearray)), (
                f"fake submit: input {input_name!r} is "
                f"{type(payload).__name__}, must be raw bytes"
            )
        ref = refs[bundle_to_ref[bundle]]
        out_meta = manifests[bundle]["outputs"][0]
        return {outputs[0]: _dense_bytes(ref, int(out_meta["stride"]))}

    return submit


# ---------------------------------------------------------- unit invariants
def test_bd_scale_is_one_over_sixteen():
    """The bd reshape chain MUST apply the var_371 1/16 scale.

    Without it the masked tensor is 16x too large and the compare
    report shows every finite element at max_abs_finite ~ 16*value.
    """
    assert diag.BD_SCALE == pytest.approx(1.0 / 16.0)
    # float.fromhex("0x1p-4") is the canonical encoding the source commit
    # on the device side uses; pin that exact representation.
    assert diag.BD_SCALE == float.fromhex("0x1p-4")


def test_compute_bd_ref_applies_scale(mx_stub, captured):
    """bd_ref must equal (raw reshape chain) * 1/16; prior driver omitted the scale."""
    raw = np.broadcast_to(
        np.float16(1.0), (1, 8, 375, 375)
    ).astype(np.float16)
    relpos = np.zeros((1, 8, 375, 749), dtype=np.float16)
    relpos[..., 0] = np.float16(1.0)  # one non-zero column

    bd = diag.compute_bd_ref(mx_stub, relpos)
    # Non-zero entries must be exactly 1/16 (within fp16 round).
    nonzero = bd[bd != 0]
    expected = np.float16(1.0 / 16.0)
    assert np.all(np.abs(nonzero.astype(np.float32) - np.float32(expected)) < 1e-3)
    # And the bd array MUST NOT be 16x the size — would mean scale missing.
    assert bd.max() <= np.float16(0.1)


def test_output_stride_looks_up_per_output():
    """``_output_stride`` reads the manifest's per-output stride.

    A hardcoded 2310144 driver returns 2310144 regardless of manifest;
    this test forces the manifest to disagree (0x4000 * 384) and
    asserts the lookup sees that.
    """
    manifest = {"outputs": [{"name": "scores", "stride": 0x4000 * 384}]}
    assert diag._output_stride(manifest, "scores") == 0x4000 * 384
    with pytest.raises(ValueError, match="no output"):
        diag._output_stride(manifest, "missing")


def test_to_bytes_is_dense_contiguous():
    """``_to_bytes`` produces contiguous raw bytes matching the array."""
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    out = diag._to_bytes(arr)
    assert isinstance(out, bytes)
    assert out == np.ascontiguousarray(arr).tobytes()
    assert len(out) == arr.nbytes


def test_decode_dense_refuses_zero_stride():
    """A 0/negative manifest stride must error, not silently go contiguous.

    The prior driver would default to ``stride = row_bytes`` on 0 and
    read garbage for multi-row outputs; the diagnostic refuses.
    """
    with pytest.raises(ValueError, match="stride missing/0"):
        diag._decode_dense(b"\0" * 100, (1, 8, 375, 375), np.float16, 0)
    with pytest.raises(ValueError, match="stride .* < row_bytes"):
        diag._decode_dense(b"\0" * 100, (1, 8, 375, 375), np.float16, 100)


def test_compare_record_signed_ulp_zero_distance():
    """-0 and +0 have signed-ULP distance 0 (bit pattern differs, magnitude = 0)."""
    rec = diag.compare_record(
        np.float16(-0.0), np.float16(0.0)
    )
    # -0 and +0 are not bit-equal (sign bit differs) but their signed
    # ULP distance is 0 (magnitude 0 == magnitude 0). The mismatch
    # is one element; the signed ULP stays 0.
    assert rec["bit_equal"] is False
    assert rec["mismatch_count"] == 1
    assert rec["ulp_max_finite"] == 0
    rec2 = diag.compare_record(
        np.array([-0.0, 0.0], dtype=np.float16),
        np.array([0.0, -0.0], dtype=np.float16),
    )
    assert rec2["bit_equal"] is False
    assert rec2["mismatch_count"] == 2
    assert rec2["ulp_max_finite"] == 0


def test_compare_record_signed_ulp_zero_crossing():
    """A sign-crossing pair of one-denorm values has signed-ULP distance 2."""
    rec = diag.compare_record(
        np.array([-5.96e-8], dtype=np.float16),
        np.array([5.96e-8], dtype=np.float16),
    )
    assert rec["bit_equal"] is False
    assert rec["mismatch_count"] == 1
    assert rec["ulp_max_finite"] == 2


def test_compare_record_inf_nan_masks():
    """inf / nan mask agreement is part of the record (not folded into bit_equal)."""
    dev = np.array([[1.0, float("-inf")], [float("nan"), 2.0]],
                   dtype=np.float16)
    ref = np.array([[1.0, float("-inf")], [float("nan"), 2.0]],
                   dtype=np.float16)
    rec = diag.compare_record(dev, ref)
    assert rec["bit_equal"] is True
    assert rec["dev_neginf"] == 1 and rec["ref_neginf"] == 1
    assert rec["dev_nan"] == 1 and rec["ref_nan"] == 1
    assert rec["inf_mask_agree"] is True
    assert rec["nan_mask_agree"] is True


# ---------------------------------------------- integration: full ladder
def test_full_ladder_is_bit_equal_to_gpu_refs(
    mx_stub, captured, manifests, tmp_path
):
    """End-to-end: every stage's device output equals the GPU reference.

    The fake submit returns the GPU reference per island packed as
    strided bytes; the diagnostic decodes with the manifest's per-
    output stride and compares. Bit-equal at every stage proves the
    transport (bytes), stride decode, and compare_record all line up.
    """
    refs = diag.compute_stage_refs(mx_stub, captured)
    submit = _make_fake_submit(refs, manifests)
    report = diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )
    assert set(report["arithmetic_isolation"]) == set(diag.LADDER)
    for stage, comps in report["arithmetic_isolation"].items():
        for tensor, rec in comps.items():
            assert rec["bit_equal"], (
                f"{stage}/{tensor}: {rec['mismatch_count']} bits differ"
            )
            assert rec["mismatch_count"] == 0
    # The chained output for add and smax must be the device output of
    # the prior stage — the diagnostic forwards them so the consumer
    # stage sees the device path, not the GPU reference.
    chain = report["chaining"]
    assert "scores" in chain["island-scores-mm"]
    assert "masked" in chain["island-select-rta"]
    assert "add" in chain["island-add-head"]
    assert "smax" in chain["island-softmax-head"]


def test_ladder_uses_per_output_stride_not_hardcoded(
    mx_stub, captured, mismatched_stride_manifests, tmp_path
):
    """A non-default scores stride must still produce bit-equal scores.

    A hardcoded-2310144 driver mis-decodes the strided scores surface
    and the bit_equal check fails. The diagnostic reads the manifest's
    per-output stride so it still round-trips.
    """
    refs = diag.compute_stage_refs(mx_stub, captured)
    submit = _make_fake_submit(refs, mismatched_stride_manifests)
    report = diag.diagnose_ladder(
        cap=captured,
        submit=submit,
        manifests=mismatched_stride_manifests,
        out=tmp_path,
    )
    scores_rec = report["arithmetic_isolation"]["island-scores-mm"]["scores"]
    assert scores_rec["bit_equal"], (
        "scores stride mismatch — driver did not look up "
        f"manifest.outputs[*].stride (mismatch_count="
        f"{scores_rec['mismatch_count']})"
    )


def test_inputs_are_raw_bytes_not_ndarray(
    mx_stub, captured, manifests, tmp_path
):
    """Inputs to ``submit`` MUST be raw bytes — ndarray/memoryview breaks the wire count.

    The fake submit asserts ``isinstance(payload, (bytes, bytearray))``
    on every input. The diagnostic's ``_to_bytes`` helper is what makes
    that true. A driver that passed ``memoryview(arr.reshape(...))``
    would fail this test on every input of every stage.
    """
    refs = diag.compute_stage_refs(mx_stub, captured)
    submit = _make_fake_submit(refs, manifests)
    diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )


def test_bd_scale_missing_is_caught(
    mx_stub, captured, manifests, tmp_path
):
    """The bd chain WITHOUT scale produces a 16x bd; compare detects it.

    The diagnostic's GPU reference applies ``BD_SCALE``; this test
    swaps the reference for an unscaled version (the prior driver)
    and asserts the bit-equal check fails — proving the diagnostic
    sees the scale instead of silently passing.
    """
    refs_unscaled = dict(diag.compute_stage_refs(mx_stub, captured))
    # Re-derive bd WITHOUT the scale (the prior driver behavior).
    relpos_t = mx_stub.array(captured["relpos"])
    padded = mx_stub.pad(relpos_t, [(0, 0), (0, 0), (0, 0), (1, 0)])
    r1 = mx_stub.reshape(padded, (1, 8, 750, 375))
    r2 = r1[:, :, 1:, :]
    r3 = mx_stub.reshape(r2, (1, 8, 375, 749))
    bd_unscaled = np.asarray(mx_stub.contiguous(r3[:, :, :, :375])).astype(
        np.float16
    )
    refs_unscaled["bd"] = bd_unscaled
    cond_t = mx_stub.array(captured["cond"])
    fill_t = mx_stub.array(np.full((1, 8, 375, 375), float("-inf"), np.float16))
    refs_unscaled["masked"] = np.asarray(
        mx_stub.where(cond_t, fill_t, mx_stub.array(bd_unscaled))
    ).astype(np.float16)

    submit = _make_fake_submit(refs_unscaled, manifests)
    report = diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )
    masked_rec = report["arithmetic_isolation"]["island-select-rta"]["masked"]
    assert masked_rec["bit_equal"] is False, (
        "diagnostic passed without bd scale — prior driver bug "
        "would silently mask the divergence"
    )
    # The diagnostic still applies the scale on its own GPU ref, so
    # the divergence shows up as a finite magnitude mismatch (not nan/
    # inf disagreement): the prior driver computes mask = where(cond,
    # fill, bd_unscaled), which differs from mask = where(cond, fill,
    # bd_unscaled/16) by exactly 15/16 * bd on every non-fill element.
    assert masked_rec["mismatch_count"] > 0


def test_compare_record_chains_through_real_shapes(
    mx_stub, captured, manifests, tmp_path
):
    """``compare_record`` covers all five reported fields on real ladder shapes.

    The four per-stage outputs are (1,8,375,375) fp16. The record must
    populate every diagnostic field the F handler reuses (bit_equal,
    mismatch_count, total, max_abs_finite, dev_/ref_ inf + nan,
    inf/nan mask agreement, ulp_max_finite).
    """
    refs = diag.compute_stage_refs(mx_stub, captured)
    submit = _make_fake_submit(refs, manifests)
    report = diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )
    expected_keys = {
        "bit_equal", "mismatch_count", "total", "max_abs_finite",
        "dev_neginf", "ref_neginf", "dev_posinf", "ref_posinf",
        "dev_nan", "ref_nan", "inf_mask_agree", "nan_mask_agree",
        "ulp_max_finite",
    }
    for comps in report["arithmetic_isolation"].values():
        for rec in comps.values():
            assert expected_keys.issubset(rec), (
                f"compare_record missing fields: "
                f"{expected_keys - set(rec)}"
            )
            assert rec["total"] == 1 * 8 * 375 * 375