# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Subprocess argv + pre-flight checks for ac_head_differential.py.

Runs the real driver as a subprocess against a recording fake worker and a
fixture bundle layout (manifest.json + program files), CPU-only:

- the worker argv binds the bundle under its session name to the JOINED
  directory (never the --bundles parent),
- a nested bundle copy (parent/bundle/bundle) is refused by name,
- a missing manifest is refused by name,
- stray subdirectories inside the bundle dir are refused by name.

No ANE device, no GPU, no lock.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parents[1] / "overlay" / "tools" / "coreml" / "ac_head_differential.py"

FAKE_WORKER = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["RECORD"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:]}) + chr(10))
print("resident bundle=island-attn-ac-head-L00 index=0 name=t programs=1 driver_abi=1 graph=deadbeef")
print("relay-bypass ready pid=1 deadline_ms=1 detail=loaded 1")
while True:
    if not sys.stdin.buffer.readline():
        break
"""


def make_layout(root: Path, stray: bool = False):
    """Bundles parent at root/bundles; arms under root/arms (outside)."""
    bundles = root / "bundles"
    bundle = bundles / "island-attn-ac-head-L00"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "graph_hash": "4bdf3b33", "dispatch_plan": [0],
        "logical_results": [], "programs": [], "tensors": {},
    }))
    (bundle / "program-0.anec").write_bytes(b"\0" * 16)
    if stray:
        inner = bundle / "island-attn-ac-head-L00"
        inner.mkdir()
        (inner / "manifest.json").write_text("{}")
    arms = root / "arms" / "a0"
    arms.mkdir(parents=True)
    for name in ("a_fill", "q", "k", "cond", "relpos"):
        meta = {"shape": [1], "dtype": "float16"}
        (arms / f"{name}.json").write_text(json.dumps(meta))
        (arms / f"{name}.bin").write_bytes(b"\0\0")
    return bundles


def run_driver(bundles: Path, record: Path, arms: Path, out: Path,
               worker: str = "/bin/true"):
    env = dict(os.environ, RECORD=str(record))
    return subprocess.run(
        [sys.executable, str(DRIVER), "--bundles", str(bundles),
         "--worker", worker, "--libane", "/dev/null",
         "--arms", str(arms), "--out", str(out),
         "--deadline-ms", "2000"],
        capture_output=True, text=True, env=env,
    )


def test_worker_argv_binds_joined_bundle_dir(tmp_path):
    worker = tmp_path / "recording-worker.py"
    worker.write_text(FAKE_WORKER)
    worker.chmod(0o755)
    bundles = make_layout(tmp_path, stray=False)
    record = tmp_path / "argv.jsonl"
    result = run_driver(bundles, record, arms=tmp_path / "arms",
                        out=tmp_path / "out", worker=str(worker))
    # The recording fake cannot serve real submits; what this test pins is
    # the ARGV: the bundle binds under its session name to the JOINED dir.
    assert result.returncode != 0, result.stderr[-400:]
    lines = [json.loads(l) for l in record.read_text().splitlines()]
    argv = lines[0]["argv"]
    i = argv.index("--bundle")
    assert argv[i + 1] == f"island-attn-ac-head-L00={bundles / 'island-attn-ac-head-L00'}"


def test_nested_bundle_copy_is_refused_by_name(tmp_path):
    bundles = make_layout(tmp_path, stray=True)
    result = run_driver(bundles, tmp_path / "argv.jsonl",
                        arms=tmp_path / "arms", out=tmp_path / "out")
    assert "unexpected subdirectories" in result.stderr
    assert "island-attn-ac-head-L00" in result.stderr


def test_missing_manifest_is_refused_by_name(tmp_path):
    result = run_driver(tmp_path / "empty", tmp_path / "argv.jsonl",
                        arms=tmp_path / "arms", out=tmp_path / "out")
    assert "cannot resolve" in result.stderr
