#!/usr/bin/env python3
"""Failing-first CPU regression: ac_head_differential wire framing.

Invokes the REAL ac_head_differential.main() end to end with a mocked
ResidentAneWorker that captures exactly what the driver submits. A stub
bundle/arm tree with small tensors keeps this CPU-only.

The framing contract under test (ane_resident.submit wire protocol):
the per-input frame header is f"in {name} {len(payload)}" and the
worker reads exactly len(payload) bytes, so len(payload) MUST equal the
tensor's byte count.

The pre-fix driver passed memoryview(reshaped arr); for a 4-D array
len(view) is the FIRST AXIS LENGTH (1), not the byte count, so the
frame promised 1 byte and the worker stalled/parsed garbage. This test
asserts the byte-count contract and fails against that old behavior.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import ac_head_differential as diff  # noqa: E402


class FakeSubmitSession:
    """Stands in for ResidentAneWorker; records every submitted payload."""

    instances: list["FakeSubmitSession"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submits: list[tuple[str, str, dict[str, object]]] = []
        self.started = False
        self.closed = False
        FakeSubmitSession.instances.append(self)

    def __enter__(self):
        self.started = True
        return self

    def __exit__(self, kind, value, traceback):
        self.closed = True
        return False

    def submit(self, bundle, tag, inputs, outputs):
        self.submits.append((bundle, tag, dict(inputs)))
        return {name: np.zeros(4, dtype=np.float16).tobytes()
                for name in outputs}


@pytest.fixture()
def small_fixture(tmp_path: Path):
    """Tiny arm dir + stub bundle dir the driver can walk CPU-only."""
    arms = tmp_path / "arms" / "tiny"
    arms.mkdir(parents=True)
    specs = {
        "a_fill": ([1, 2, 3, 4], "float16"),   # 24 elements
        "q": ([1, 2, 3, 4], "float16"),
        "k": ([1, 2, 3, 4], "float16"),
        "cond": ([1, 2, 3, 4], "bool"),        # 24 bytes
        "relpos": ([1, 2, 6, 4], "float16"),
    }
    for name, (shape, dtype) in specs.items():
        dt = np.dtype(dtype)
        raw = (np.arange(int(np.prod(shape)), dtype=np.uint16)
               .astype(dt).tobytes())
        (arms / f"{name}.bin").write_bytes(raw)
        (arms / f"{name}.json").write_text(
            json.dumps({"dtype": dtype, "shape": shape}))

    bundle = tmp_path / "bundles" / "b0"
    bundle.mkdir(parents=True)
    manifest = {"name": "b0", "graph_hash": "0" * 64, "programs": [{}]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, arms, bundle, specs


def _run_driver(tmp_path: Path, arms: Path, bundles: Path, monkeypatch,
                bundle_name: str = "b0") -> FakeSubmitSession:
    """Run diff.main() with ResidentAneWorker swapped for the fake."""
    FakeSubmitSession.instances.clear()
    monkeypatch.setattr(diff, "ResidentAneWorker", FakeSubmitSession)
    out = tmp_path / "out"
    argv = [
        "ac_head_differential.py",
        "--bundles", str(bundles),
        "--bundle-name", bundle_name,
        "--worker", "fake-worker",
        "--libane", "fake-libane",
        "--arms", str(arms.parent),
        "--out", str(out),
        "--deadline-ms", "1000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = diff.main()
    assert rc == 0
    assert len(FakeSubmitSession.instances) == 1
    session = FakeSubmitSession.instances[0]
    assert session.started and session.closed
    return session


def test_driver_submits_byte_counted_payloads(
        small_fixture, monkeypatch) -> None:
    """The load-bearing contract: every submitted payload's len() equals
    the tensor's byte count. The pre-fix multidim-memoryview payloads
    fail this (len == first axis length == 1)."""
    tmp_path, arms, bundle, specs = small_fixture
    session = _run_driver(tmp_path, arms, bundle, monkeypatch)
    assert len(session.submits) == 1
    _, tag, inputs = session.submits[0]
    assert tag == "tiny"
    for name, (shape, dtype) in specs.items():
        expected = int(np.prod(shape) * np.dtype(dtype).itemsize)
        payload = inputs[name]
        # len() is what the wire header prints: it MUST be byte count.
        assert len(payload) == expected, (
            f"{name}: len(payload)={len(payload)} != byte count={expected}")
        if isinstance(payload, memoryview):
            pytest.fail(
                f"{name}: submitted a memoryview (len={len(payload)}); "
                "driver must submit bytes so len() is the byte count")


def test_driver_refuses_manifest_name_mismatch(
        small_fixture, monkeypatch) -> None:
    """--bundle-name must match manifest.name before any submit."""
    tmp_path, arms, bundle, _ = small_fixture
    with pytest.raises(SystemExit, match="does not match"):
        _run_driver(tmp_path, arms, bundle, monkeypatch,
                    bundle_name="wrong-name")
    # And no session may have been constructed for the refused run.
    assert FakeSubmitSession.instances == []


def test_driver_cli_help_mentions_bundle_name() -> None:
    """CLI surface: --bundle-name exists and --help exits 0."""
    proc = pytest.importorskip("subprocess").run(
        [sys.executable, str(TOOLS / "ac_head_differential.py"), "--help"],
        capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--bundle-name" in proc.stdout


def test_len_semantics_documented_by_example() -> None:
    """len(multidim memoryview) is the FIRST AXIS LENGTH, not a
    dimension count and not bytes; len(bytes) is the byte count."""
    arr = np.zeros((7, 2, 3, 4), dtype=np.float16)
    view = memoryview(arr)
    assert len(view) == 7                      # first axis length
    assert view.nbytes == arr.nbytes           # byte count (fp16: 2 B/el)
    assert view.nbytes == 7 * 2 * 3 * 4 * 2
    assert len(arr.tobytes()) == view.nbytes   # bytes: byte count