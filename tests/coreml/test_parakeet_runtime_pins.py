# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Tests for the shipped Parakeet runtime assets and the product CLI.

The pin manifest is the installed product's integrity contract: every
hash it records must match the bytes actually shipped in
``overlay/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/``,
and its end-to-end expectations must be internally consistent. The CLI
tests cover the explicit-refusal contract on hosts without the runtime.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHARE = (
    REPO / "overlay" / "tools" / "mlx-omarchy-parakeet"
    / "share" / "mlx-omarchy" / "parakeet-1"
)
CLI = (
    REPO / "overlay" / "tools" / "mlx-omarchy-parakeet"
    / "mlx_omarchy_parakeet.py"
)


@pytest.fixture(scope="module")
def pin() -> dict:
    return json.loads((SHARE / "parakeet-runtime-pin.json").read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pin_schema(pin: dict) -> None:
    assert pin["schema"] == "mlx-omarchy.parakeet-runtime-pin.v1"


def test_bundle_hashes_match_shipped_bytes(pin: dict) -> None:
    for name, files in pin["assets"]["bundles"].items():
        for relative, expected in sorted(files.items()):
            path = SHARE / "bundles" / name / relative
            assert path.is_file(), f"pinned bundle file missing: {path}"
            assert sha256_file(path) == expected, path


def test_libane_hash_matches_shipped_bytes(pin: dict) -> None:
    (expected,) = pin["assets"]["libane"].values()
    path = SHARE / "libane" / "libane-strict.so"
    assert path.is_file()
    assert sha256_file(path) == expected
    # The shipped loader is the strict-bind build of the pinned
    # omarchy-ane checkout: it must be an aarch64 shared object.
    assert path.read_bytes()[:4] == b"\x7fELF"


def test_resident_bundles_are_pinned(pin: dict) -> None:
    """Every bundle the runner submits must have a shipped, pinned copy.

    Read through ast so the check runs on hosts without mlx installed
    (importing the runner pulls in mlx.core at module level).
    """
    import ast

    tree = ast.parse(
        (REPO / "overlay" / "tools" / "coreml" / "vulkan_encoder.py")
        .read_text())
    names = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "RESIDENT_BUNDLES"
                for target in node.targets
            )
        ):
            names = ast.literal_eval(node.value)
    assert names is not None, "RESIDENT_BUNDLES assignment not found"
    assert set(names) == set(pin["assets"]["bundles"])


def test_e2e_pins_are_internally_consistent(pin: dict) -> None:
    e2e = pin["e2e"]
    count = e2e["emissions"]
    assert count == 104
    assert len(e2e["token_ids"]) == count
    assert len(e2e["frame_indices"]) == count
    assert len(e2e["durations"]) == count
    transcript_sha = hashlib.sha256(
        e2e["transcript"].encode()
    ).hexdigest()
    assert transcript_sha == e2e["transcript_sha256"]


def test_encoder_source_pin_is_recorded(pin: dict) -> None:
    assert len(pin["encoder_source"]["mil_sha256"]) == 64


def _run_cli(*argv: str, cache_dir: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ, MLX_OMARCHY_CACHE_DIR=str(cache_dir))
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True, text=True, env=env,
    )


def test_transcribe_refuses_without_runtime_assets(tmp_path: Path) -> None:
    """A wheel (or checkout) without the aarch64 assets refuses loudly."""
    done = _run_cli("transcribe", cache_dir=tmp_path)
    assert done.returncode == 1
    assert "runtime assets are not installed" in done.stderr


def test_verify_fails_cleanly_on_empty_cache(tmp_path: Path) -> None:
    done = _run_cli("verify", cache_dir=tmp_path)
    assert done.returncode == 1
    assert "MISMATCH" in done.stderr
    assert "not cached yet" in done.stderr
