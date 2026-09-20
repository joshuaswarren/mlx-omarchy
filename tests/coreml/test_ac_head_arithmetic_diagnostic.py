# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-program arithmetic-isolation ladder for the AC head (CPU-only).

Runs ``ac_head_arithmetic_diagnostic.diagnose_ladder`` end-to-end with
two monkeypatched seams:

  - ``mlx.core`` is replaced by a numpy-only stub (no GPU / no
    compiled metal kernel). The stub reproduces the exact reshape,
    pad, transpose, matmul, where, softmax chain the diagnostic asks
    for so the GPU reference is bit-equal to the device path on this
    backend (the test asserts ``bit_equal=True`` for every stage).
  - the worker transport is replaced by a fake ``submit`` that
    serializes the GPU reference per island as dense logical bytes
    and decodes by reshape alone.

The invariants under test are the ones the prior driver broke:

  (a) inputs travel as raw ``bytes`` — the worker recv rejects
      anything shorter than ``logical_bytes``;
  (b) bd carries the var_371 ``1/16`` scale — omitting it shifted
      the masked-stage divergence by exactly that factor;
  (c) the output wire bytes are dense logical row-major — no
      caller-side stride decode; the manifest ``stride`` is the ANEC
      tile allocator stride and irrelevant to the wire;
  (d) cond captured as (1,1,375,375) byte-bool is broadcast to
      (1,8,375,375) on the client, never ``unpackbits``;
  (e) every stage has its own GPU reference computed from the SAME
      serialized arrays the device receives (not pristine refs);
  (f) the comparison record reports bit/ULP/inf/nan with correct
      signed-ULP semantics (-0/+0 distance 0, zero-crossing pair
      distance 2).

Each invariant has a focused test below; the end-to-end ladder test
exercises them all together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT / "overlay" / "tools"
sys.path.insert(0, str(HERE))  # for test_stride_decode helpers
sys.path.insert(0, str(TOOLS))  # for coreml.* imports

from coreml import ac_head_arithmetic_diagnostic as diag  # noqa: E402
import test_stride_decode as tsd  # noqa: E402

ALIGN = tsd.ALIGN


# ----------------------------------------------------------- numpy stub for mx
class _MxArray(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)


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
    return _MxArray(np.where(np.asarray(cond), np.asarray(x), np.asarray(y)))


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


@pytest.fixture
def mx_stub(monkeypatch):
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

    mod = _Module()
    sys.modules["mlx.core"] = mod
    monkeypatch.setattr(diag, "mx", mod, raising=False)
    yield mod
    sys.modules.pop("mlx.core", None)


# ----------------------------------------------------- captured fixtures
@pytest.fixture
def captured():
    """Captured shapes from the F-dump directory on t6001-test-host."""
    rng = np.random.default_rng(0xACEB)
    return {
        "q": rng.standard_normal((1, 8, 375, 128)).astype(np.float16) / 4,
        "k": rng.standard_normal((1, 8, 375, 128)).astype(np.float16) / 4,
        "cond": (rng.standard_normal((1, 1, 375, 375)) > 0).astype(np.bool_),
        "relpos": rng.standard_normal((1, 8, 375, 749)).astype(np.float16) / 8,
        "a_fill": np.float16(-1.0),
    }


@pytest.fixture
def manifests():
    """Minimal bundle manifests with realistic wire metadata."""
    def manifest(name):
        return {
            "name": name,
            "outputs": [{
                "name": ("masked" if "select" in name else
                         "scores" if "scores" in name else
                         "add" if "add" in name else "smax"),
                "stride": 0x4000 * 384,  # ANEC tile; irrelevant on wire
                "byte_size": 2250000,
                "dtype": "float16",
                "shape": [1, 8, 375, 375],
            }],
        }
    return {name: manifest(name) for name in diag.LADDER}


def _make_fake_submit(mx, bundle_to_ref_fn):
    """Return a ``submit(bundle, tag, inputs, outputs)`` returning raw bytes.

    The fake decodes the serialized inputs back to ndarrays, computes
    the per-stage GPU ref via the same ``mx`` chain the diagnostic
    uses, and returns that ref packed as dense logical bytes. This
    matches the real wire contract: device output is the GPU math
    applied to the SAME inputs the device received. Both the fake
    submit's ref and the diagnostic's per-stage ref are derived from
    the same input bytes, so ``bit_equal=True`` is the only honest
    outcome on a passing device.
    """
    bundle_to_inputs_dtype = {
        "island-select-rta": {
            "a_fill": (np.float16, (1, 8, 375, 375)),
            "bd": (np.float16, (1, 8, 375, 375)),
            "cond": (np.bool_, (1, 8, 375, 375)),
        },
        "island-scores-mm": {
            "q": (np.float16, (1, 8, 375, 128)),
            "k": (np.float16, (1, 8, 375, 128)),
        },
        "island-add-head": {
            "scores": (np.float16, (1, 8, 375, 375)),
            "masked": (np.float16, (1, 8, 375, 375)),
        },
        "island-softmax-head": {
            "add": (np.float16, (1, 8, 375, 375)),
        },
    }
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)

    def submit(bundle, tag, inputs, outputs):
        for input_name, payload in inputs.items():
            assert isinstance(payload, (bytes, bytearray)), (
                f"fake submit: input {input_name!r} is "
                f"{type(payload).__name__}, must be raw bytes"
            )
        dtypes = bundle_to_inputs_dtype[bundle]
        decoded_inputs = {
            name: np.frombuffer(payload, dtype=dt).reshape(shape)
            for name, (dt, shape) in dtypes.items()
            for payload in [inputs[name]]
        }
        ref = bundle_to_ref_fn(mx, bundle, decoded_inputs, fill_neg_inf)
        return {outputs[0]: np.ascontiguousarray(ref).tobytes()}

    return submit


def _gpu_ref_from_inputs(mx, bundle, decoded_inputs, fill_neg_inf):
    """Per-stage GPU ref from the inputs the device just received.

    Mirrors ``diag._stage_ref`` — same chain on the same device. The
    fake uses this so its returned bytes equal the diagnostic's
    per-stage ref on a passing device.
    """
    if bundle == "island-select-rta":
        return np.asarray(
            mx.where(
                mx.array(decoded_inputs["cond"]),
                mx.array(decoded_inputs["a_fill"]),
                mx.array(decoded_inputs["bd"]),
            )
        ).astype(np.float16)
    if bundle == "island-scores-mm":
        q = mx.array(decoded_inputs["q"])
        k = mx.array(decoded_inputs["k"])
        return np.asarray(
            mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
        ).astype(np.float16)
    if bundle == "island-add-head":
        return (decoded_inputs["scores"] + decoded_inputs["masked"]).astype(np.float16)
    if bundle == "island-softmax-head":
        return np.asarray(
            mx.softmax(mx.array(decoded_inputs["add"]), axis=-1)
        ).astype(np.float16)
    raise ValueError(f"unknown bundle {bundle!r}")


# ---------------------------------------------------------- unit invariants
def test_bd_scale_is_one_over_sixteen():
    assert diag.BD_SCALE == pytest.approx(1.0 / 16.0)
    assert diag.BD_SCALE == float.fromhex("0x1p-4")


def test_compute_bd_ref_applies_scale(mx_stub):
    """bd_ref must equal (raw reshape chain) * 1/16."""
    relpos = np.zeros((1, 8, 375, 749), dtype=np.float16)
    relpos[..., 0] = np.float16(1.0)
    bd = diag.compute_bd_ref(mx_stub, relpos)
    nonzero = bd[bd != 0]
    expected = np.float16(1.0 / 16.0)
    assert np.all(np.abs(nonzero.astype(np.float32) - np.float32(expected)) < 1e-3)
    assert bd.max() <= np.float16(0.1)


def test_decode_dense_ignores_stride():
    """The wire bytes are dense; the manifest stride is the ANEC tile stride.

    A caller-side decode that consulted ``outputs[*].stride`` would
    silently mis-decode this surface (the wire is already dense;
    ``stride`` is irrelevant on the wire path).
    """
    arr = np.arange(1 * 8 * 375 * 375, dtype=np.float16).reshape(1, 8, 375, 375)
    manifest = {
        "outputs": [{
            "name": "masked",
            "stride": 0xDEAD,
            "byte_size": arr.nbytes,
            "dtype": "float16",
            "shape": [1, 8, 375, 375],
        }]
    }
    decoded = diag._decode_dense(arr.tobytes(), manifest, "masked")
    assert np.array_equal(decoded, arr)


def test_decode_dense_rejects_wrong_byte_count():
    manifest = {
        "outputs": [{
            "name": "masked", "stride": 2310144, "byte_size": 2250000,
            "dtype": "float16", "shape": [1, 8, 375, 375],
        }]
    }
    with pytest.raises(ValueError, match="wire bytes"):
        diag._decode_dense(b"\0" * 100, manifest, "masked")


def test_to_bytes_is_dense_contiguous():
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    out = diag._to_bytes(arr)
    assert isinstance(out, bytes)
    assert out == np.ascontiguousarray(arr).tobytes()
    assert len(out) == arr.nbytes


def test_normalize_cond_broadcasts_to_8_heads():
    """cond captured (1,1,375,375) byte-bool must broadcast to (1,8,375,375)."""
    cond_captured = (np.random.default_rng(1).standard_normal((1, 1, 375, 375)) > 0).astype(np.bool_)
    normalized = diag._normalize("cond", cond_captured)
    assert normalized.shape == (1, 8, 375, 375)
    assert normalized.dtype == np.bool_
    # Broadcast copy: every channel is identical.
    for c in range(1, 8):
        assert np.array_equal(normalized[0, 0], normalized[0, c])


def test_normalize_a_fill_broadcasts_to_8_heads():
    normalized = diag._normalize("a_fill", np.float16(-1.0))
    assert normalized.shape == (1, 8, 375, 375)
    assert normalized.dtype == np.float16
    assert np.all(normalized == np.float16(-1.0))


def test_compare_record_signed_ulp_zero_distance():
    rec = diag.compare_record(np.float16(-0.0), np.float16(0.0))
    assert rec["bit_equal"] is False
    assert rec["mismatch_count"] == 1
    assert rec["ulp_max_finite"] == 0


def test_compare_record_signed_ulp_zero_crossing():
    rec = diag.compare_record(
        np.array([-5.96e-8], dtype=np.float16),
        np.array([5.96e-8], dtype=np.float16),
    )
    assert rec["bit_equal"] is False
    assert rec["ulp_max_finite"] == 2


def test_compare_record_inf_nan_masks():
    dev = np.array([[1.0, float("-inf")], [float("nan"), 2.0]], dtype=np.float16)
    ref = np.array([[1.0, float("-inf")], [float("nan"), 2.0]], dtype=np.float16)
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
    """End-to-end: every stage's device output equals its SAME-INPUT ref.

    The fake submit decodes the inputs the diagnostic just serialized
    and applies the same ``mx`` chain to produce the output bytes.
    The diagnostic decodes by reshape and compares to the per-stage
    ref built from the same serialized input bytes — both sides
    agree by construction.
    """
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)
    submit = _make_fake_submit(mx_stub, _gpu_ref_from_inputs)
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
    chain = report["chaining"]
    assert "scores" in chain["island-scores-mm"]
    assert "masked" in chain["island-select-rta"]
    assert "add" in chain["island-add-head"]
    assert "smax" in chain["island-softmax-head"]


def test_inputs_are_raw_bytes_not_ndarray(
    mx_stub, captured, manifests, tmp_path
):
    """Inputs to ``submit`` MUST be raw bytes."""
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)
    submit = _make_fake_submit(mx_stub, _gpu_ref_from_inputs)
    diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )


def test_bd_scale_missing_is_caught(
    mx_stub, captured, manifests, tmp_path
):
    """The bd chain WITHOUT scale produces a 16x bd; compare detects it.

    The fake submit returns the GPU ref WITHOUT the 1/16 scale for
    masked; the diagnostic's per-stage ref carries the scale; the
    bit_equal check then fails, exposing the missing scale.
    """
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)

    def unscaled_ref_fn(mx, bundle, decoded_inputs, fill):
        if bundle == "island-select-rta":
            relpos_t = mx.array(captured["relpos"])
            padded = mx.pad(relpos_t, [(0, 0), (0, 0), (0, 0), (1, 0)])
            r1 = mx.reshape(padded, (1, 8, 750, 375))
            r2 = r1[:, :, 1:, :]
            r3 = mx.reshape(r2, (1, 8, 375, 749))
            bd_unscaled = np.asarray(
                mx.contiguous(r3[:, :, :, :375])
            ).astype(np.float16)
            return np.asarray(
                mx.where(
                    mx.array(decoded_inputs["cond"]),
                    mx.array(decoded_inputs["a_fill"]),
                    mx.array(bd_unscaled),
                )
            ).astype(np.float16)
        return _gpu_ref_from_inputs(mx, bundle, decoded_inputs, fill_neg_inf)

    submit = _make_fake_submit(mx_stub, unscaled_ref_fn)
    report = diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )
    masked_rec = report["arithmetic_isolation"]["island-select-rta"]["masked"]
    assert masked_rec["bit_equal"] is False
    assert masked_rec["mismatch_count"] > 0


def test_compare_record_chains_through_real_shapes(
    mx_stub, captured, manifests, tmp_path
):
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)
    submit = _make_fake_submit(mx_stub, _gpu_ref_from_inputs)
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
            assert expected_keys.issubset(rec)
            assert rec["total"] == 1 * 8 * 375 * 375


def test_per_stage_ref_uses_serialized_arrays_not_pristine_capture(
    mx_stub, captured, manifests, tmp_path
):
    """The per-stage ref must come from the SAME values serialized to submit.

    The fake submit decodes inputs and produces the same per-stage
    GPU ref the diagnostic computes. Both are derived from the
    IDENTICAL serialized input bytes, so ``bit_equal=True`` is the
    only honest outcome — a pristine ref built once from
    ``captured`` would mismatch as soon as any chained input diverges
    from its captured pristine value (e.g. add-head's masked comes
    from device-select-rta output, not pristine masked).
    """
    fill_neg_inf = np.full((1, 8, 375, 375), float("-inf"), np.float16)
    submit = _make_fake_submit(mx_stub, _gpu_ref_from_inputs)
    report = diag.diagnose_ladder(
        cap=captured, submit=submit, manifests=manifests, out=tmp_path,
    )
    for stage, comps in report["arithmetic_isolation"].items():
        for rec in comps.values():
            assert rec["bit_equal"], (
                f"{stage} not bit-equal: per-stage ref diverged from "
                "the serialized input — pristine-ref contamination"
            )