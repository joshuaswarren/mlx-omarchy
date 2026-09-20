"""CPU regression: the F placement's bdscale head submit sends RAW relpos
and the certified fallback is unchanged.

Requires mlx.core (CPU backend is enough; no ANE device). Runs the real
EncoderRunner._run_island_ac for layer 0 twice against a mocked
AneIsland.submit that captures submitted payloads:

1. bdscale arm  - bundle dir island-attn-ac-head-L00-bdscale exists:
   the submit goes to that bundle with relpos = RAW unscaled
   [1,8,375,749] bytes (4,494,000) and NO 'bd' key.
2. fallback arm - no bdscale bundle dir: the legacy head submit receives
   bd = the host-scaled contiguous slice [1,8,375,375] (2,250,000),
   byte-identical to the pre-bdscale behavior for the same inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
try:
    import pytest
except ModuleNotFoundError:  # main() mode needs no pytest
    pytest = None
try:
    import mlx.core as mx
except ModuleNotFoundError:
    raise SystemExit("mlx.core unavailable: run on a venv that has it")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import vulkan_encoder as ve  # noqa: E402

L0_RAW_RELPOS_BYTES = 4494000
L0_BD_BYTES = 2250000


def _mk_runner(tmp_path: Path, with_bdscale: bool):
    island = ve.AneIsland.__new__(ve.AneIsland)
    island.bundles = tmp_path / "bundles"
    island.bundles.mkdir(parents=True, exist_ok=True)
    island.resident_bundles = set(ve.RESIDENT_BUNDLES)
    for name in ("island-attn-ac-L00", "island-pv"):
        (island.bundles / name).mkdir(parents=True, exist_ok=True)
    if with_bdscale:
        bds = island.bundles / "island-attn-ac-head-L00-bdscale"
        bds.mkdir(parents=True, exist_ok=True)
        (bds / "manifest.json").write_text(json.dumps({"name": "x"}))

    r = ve.EncoderRunner.__new__(ve.EncoderRunner)
    r.gpu_ops = 0
    r.ane_ops = 0
    r.executed = 0
    r.placed = frozenset({"F"})
    mx.set_default_device(mx.cpu)
    r.island = island
    r.statements = []
    r.placed = frozenset({"F"})
    idx = 0

    def stmt(name, op, shape, **kw):
        nonlocal idx
        s = ve.Statement(idx, [name], op, kw, "", "fp16", shape)
        r.statements.append(s)
        idx += 1
        return s

    s = stmt("attention_scores_1_cast_fp16", "matmul", (1, 8, 375, 749),
             transpose_x="false", transpose_y="false", x="pos", y="q")
    pad = stmt("var_pad0", "pad", (1, 8, 375, 750), x=s.names[0])
    mul = stmt("matrix_bd_5_0", "mul", (1, 8, 375, 375),
               x=pad.names[0], y="var_371")
    const_stmt = stmt("var_371", "const", None)
    r.meta = {"var_371": 0.0625}  # scalar() reads meta (MIL fp16 literal)
    sel = stmt("attention_mask_1_cast_fp16", "select", (1, 8, 375, 375),
               a="ninf", cond="var_373", b=mul.names[0])
    content = stmt("matmul_1_cast_fp16", "matmul", (1, 8, 375, 375),
                   transpose_x="false", transpose_y="true", x="qs", y="kt")
    out = stmt("attn_output_1_cast_fp16", "matmul", (1, 8, 375, 128),
               transpose_x="false", transpose_y="false", x="probs", y="vh",
               x2=content.names[0])

    r.island_a = {s.index: (0, s, content)}
    r.island_a_partner = {}
    r.island_b = {sel.index: (0, sel)}
    r.island_c = {out.index: (0, out)}
    r.island_oproj = {}
    r.producer = {mul.names[0]: mul, sel.names[0]: sel,
                  "var_371": const_stmt}
    # values the handler resolves by name
    rng = np.random.default_rng(7)
    raw_relpos = rng.standard_normal((1, 8, 375, 749)).astype(np.float16)
    raw_relpos = mx.array(raw_relpos)
    r.values = {
        "pos": mx.array(rng.standard_normal((1, 8, 375, 128)).astype(np.float16)),
        "q": mx.array(rng.standard_normal((1, 8, 128, 749)).astype(np.float16)),
        "kt": mx.array(rng.standard_normal((1, 8, 375, 128)).astype(np.float16)),
        "qs": mx.array(rng.standard_normal((1, 8, 375, 128)).astype(np.float16)),
        "vh": mx.array(rng.standard_normal((1, 8, 375, 128)).astype(np.float16)),
        "ninf": mx.array(-np.inf, mx.float16),
        "var_373": mx.zeros((1, 1, 375, 375), mx.bool_),
        s.names[0]: raw_relpos,
    }
    return r, s, sel, content, out, raw_relpos


def _capture(monkeypatch):
    calls = []

    def fake_submit(self, bundle, tag, inputs, outputs, **kw):
        record = {"bundle": bundle, "tag": tag,
                  "inputs": {k: bytes(v) for k, v in inputs.items()}}
        calls.append(record)
        out = {}
        for name in outputs:
            shape = (1, 8, 375, 375) if name == "smax" else (1, 8, 375, 128)
            out[name] = mx.zeros(shape, mx.float16)  # mx arrays, like the real island
        return out

    monkeypatch.setattr(ve.AneIsland, "submit",
                        lambda self, bundle, tag, inputs, outputs, **kw:
                        fake_submit(self, bundle, tag, inputs, outputs, **kw))
    return calls


def _run(r, s):
    # scalar()/tensor() reads; the scale const lookup needs producer plumbing
    r.scalar_const = {  # not used by handler; scale const resolves via producer
    }
    r._run_island_ac(r.statements[s.index])


def test_bdscale_head_submits_raw_relpos(tmp_path, monkeypatch) -> None:
    r, s, _sel, _content, _out, raw_relpos = _mk_runner(tmp_path, True)
    calls = _capture(monkeypatch)
    _run(r, s)
    head = next(c for c in calls if c["bundle"].endswith("-bdscale"))
    assert set(head["inputs"]) == {"a_fill", "cond", "q", "k", "relpos"}
    rel = head["inputs"]["relpos"]
    assert len(rel) == L0_RAW_RELPOS_BYTES, (
        f"relpos submitted {len(rel)} bytes; the raw unscaled tensor is "
        f"{L0_RAW_RELPOS_BYTES}")
    # The handler computes relpos = pos @ q on device (CPU backend here);
    # the submitted bytes must equal that RAW result, not a scaled slice.
    expected_rel = np.asarray(mx.matmul(r.values["pos"], r.values["q"])
                              ).astype(np.float16).tobytes()
    assert rel == expected_rel, (
        "relpos payload is not the raw unscaled GPU matmul output")
    scaled = np.asarray(mx.contiguous(
        mx.matmul(r.values["pos"], r.values["q"])[:, :, :, :375]
        * mx.array(float.fromhex("0x1p-4"), mx.float16))
    ).astype(np.float16).tobytes()
    assert rel != scaled, "relpos payload is a scaled slice, not raw"
    # and no legacy bd key anywhere in the bdscale submit
    assert "bd" not in head["inputs"]
    # shared bookkeeping: EXACTLY one head + one PV, exact counters
    assert [c["bundle"] for c in calls] == [
        "island-attn-ac-head-L00-bdscale", "island-pv"], calls
    assert r.executed == 3 and r.ane_ops == 2 and r.gpu_ops == 1, (
        r.executed, r.ane_ops, r.gpu_ops)
    assert s.done and _out.done and _content.done
    stored = r.values[_out.names[0]]
    assert isinstance(stored, mx.array) or isinstance(stored, object)
    assert np.asarray(stored).size == 8 * 375 * 128


def test_fallback_head_unchanged_without_bdscale(tmp_path, monkeypatch) -> None:
    r, s, _sel, _content, _out, raw_relpos = _mk_runner(tmp_path, False)
    calls = _capture(monkeypatch)
    _run(r, s)
    head = next(c for c in calls if "ac-head" in c["bundle"])
    assert set(head["inputs"]) == {"a_fill", "bd", "cond", "q", "k"}
    bd = head["inputs"]["bd"]
    assert len(bd) == L0_BD_BYTES
    # relpos inside the handler is pos @ q (recomputed on device/CPU);
    # with the var_371 const producer wired, the legacy bd MUST be the
    # 1/16-scaled contiguous slice (and differ from the raw slice).
    rel = np.asarray(mx.matmul(r.values["pos"], r.values["q"])
                     ).astype(np.float16)
    scaled = np.ascontiguousarray(
        (rel[:, :, :, :375] * np.float16(0.0625)).astype(np.float16)
    ).tobytes()
    raw = np.ascontiguousarray(rel[:, :, :, :375]).tobytes()
    assert bd == scaled, "fallback bd is not the 1/16-scaled slice"
    assert bd != raw, "scaled slice equals raw slice on nonzero fixture"
    assert [c["bundle"] for c in calls] == [
        "island-attn-ac-head-L00", "island-pv"], calls
    assert r.executed == 3 and r.ane_ops == 2 and r.gpu_ops == 1, (
        r.executed, r.ane_ops, r.gpu_ops)
    assert s.done and _out.done and _content.done

if __name__ == "__main__":
    import tempfile, traceback
    checks = 0
    for fn in (test_bdscale_head_submits_raw_relpos,
               test_fallback_head_unchanged_without_bdscale):
        with tempfile.TemporaryDirectory(prefix="bdscale-submit-") as d:
            class MP:  # minimal monkeypatch
                def setattr(self, obj, name, value):
                    setattr(obj, name, value)
            try:
                fn(Path(d), MP())
            except Exception:
                traceback.print_exc()
                print(f"{fn.__name__}: FAIL")
                sys.exit(1)
        print(f"{fn.__name__}: PASS")
        checks += 1
    print(f"{checks}/2 SUBMIT CHECKS PASS")
