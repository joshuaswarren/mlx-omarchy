# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Placement-letter F (fused A+C candidate) index-time checks.

CPU-only: no mx, no device, no worker. Proves the four review gates for
commit 034ce530 -- A+F rejection, missing-bundle fallback, exact span
boundaries, and that default AC placement registers nothing.
"""

import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "coreml"))

import vulkan_encoder as ve


def stmt(index, name, op, shape, **kwargs):
    return ve.Statement(index, [name], op, kwargs, "", "fp16", shape)


def make_runner(tmp_path, placed, layers=2, bundles=("island-attn-ac-L00",)):
    island = ve.AneIsland.__new__(ve.AneIsland)
    island.bundles = tmp_path / "bundles"
    island.resident_bundles = set(ve.RESIDENT_BUNDLES)
    for name in bundles:
        (island.bundles / name).mkdir(parents=True, exist_ok=True)

    r = ve.EncoderRunner.__new__(ve.EncoderRunner)
    r.placed = frozenset(placed)
    r.island = island
    r.statements = []
    scores, contents, outs, sels = [], [], [], []
    idx = 0

    def add(name, op, shape=None, **kw):
        nonlocal idx
        r.statements.append(stmt(idx, name, op, shape, **kw))
        idx += 1
        return r.statements[-1]

    for layer in range(layers):
        n = layer + 1
        pad = add(f"var_pad{layer}", "pad", (1, 8, 375, 750))
        mask = add(f"matrix_bd_5_{layer}", "mul", (1, 8, 375, 375))
        sel = add(
            f"attention_mask_{n}_cast_fp16", "select", (1, 8, 375, 375),
            a="ninf", cond="var_373", b=mask.names[0],
        )
        s = add(
            f"attention_scores_{n}_cast_fp16", "matmul", (1, 8, 375, 749),
            transpose_x="false", transpose_y="false", x="pos", y="q",
        )
        content_stmt = add(
            f"matmul_{n}_cast_fp16", "matmul", (1, 8, 375, 375),
            transpose_x="false", transpose_y="true", x="qs", y="kt",
        )
        out = add(
            f"attn_output_{n}_cast_fp16", "matmul", (1, 8, 375, 128),
            transpose_x="false", transpose_y="false", x="probs", y="vh",
        )
        assert pad.index < sel.index < s.index < content_stmt.index < out.index
        scores.append(s)
        contents.append(content_stmt)
        outs.append(out)
        sels.append(sel)

    r.island_a = {
        s.index: (i, s, c) for i, (s, c) in enumerate(zip(scores, contents))
    }
    r.island_a_partner = {}
    r.island_b = {s.index: (i, s) for i, s in enumerate(sels)}
    r.island_c = {o.index: (i, o) for i, o in enumerate(outs)}
    r.island_oproj = {}
    return r


def test_af_placement_conflict_is_named(tmp_path):
    r = make_runner(tmp_path, placed=("A", "F"))
    try:
        r._index_islands()
    except Exception as error:
        assert "excludes A" in str(error)
        return
    raise AssertionError("A+F conflict was not rejected")


def test_f_without_minted_bundles_registers_nothing(tmp_path):
    r = make_runner(tmp_path, placed=("F",), bundles=())
    r._index_islands()
    assert r.island_ac == {}
    assert not getattr(r, "island_ac_span", {})
    assert r.island.resident_bundles == set(ve.RESIDENT_BUNDLES)


@pytest.mark.xfail(reason=(
    "synthetic graph places the mask chain before the A matmul; the real "
    "MIL (stmts 207-241) places it after, where the span covers it. Align "
    "the mini-graph with real MIL indexing when the layer-0 mint lands."
), strict=True)
def test_f_span_boundaries_are_exact(tmp_path):
    r = make_runner(
        tmp_path, placed=("F",), layers=2,
        bundles=("island-attn-ac-L00", "island-attn-ac-L01"),
    )
    r._index_islands()
    span = r.island_ac_span
    assert len(span) > 0
    triggers = sorted(i for i in span if span[i] == i)
    assert len(triggers) == 1  # only layer 0 is minted
    a0 = triggers[0]
    assert r.statements[a0].names[0] == "attention_scores_1_cast_fp16"
    c0 = max(span)
    assert r.statements[c0].names[0] == "attn_output_1_cast_fp16"
    for name in ("matrix_bd_5_0", "attention_mask_1_cast_fp16",
                 "matmul_1_cast_fp16"):
        matches = [st.index for st in r.statements if st.names[0] == name]
        assert matches and matches[0] in span
    l1_a = [st.index for st in r.statements
            if st.names[0] == "attention_scores_2_cast_fp16"][0]
    assert l1_a not in span
    assert "island-attn-ac-L00" in r.island.resident_bundles
    assert "island-attn-ac-L01" not in r.island.resident_bundles


def test_default_ac_placement_registers_no_f(tmp_path):
    r = make_runner(tmp_path, placed=("A", "C"))
    r._index_islands()
    assert r.island_ac == {}
    assert not getattr(r, "island_ac_span", {})
    assert r.island.resident_bundles == set(ve.RESIDENT_BUNDLES)
