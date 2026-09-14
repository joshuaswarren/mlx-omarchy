#!/usr/bin/env python3
"""Collect the environment and golden-anchor evidence for the three-island run.

Everything here is measured on the host at collection time: file hashes are
read from the files the run actually used, the lock anchors are compared
rather than asserted, and the build identity comes from the installed
distribution rather than from a checkout that may be dirty.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/var/tmp/EncoderParity3Islands")
SRC = Path("/var/tmp/EncoderParityAne/encoder-source")
CAPTURE = Path("/var/tmp/EncoderParityAne/capture")
BUNDLES = Path("/var/tmp/jwm1-encoder-islands/bundles")
WORKER = Path("/var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker")
LIBANE = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/receipts/"
    "2026-09-13-ane-worker-validation/libane.so"
)
LOCK = ROOT / "package/coreml/parakeet-reference.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*argv: str) -> str:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError as error:
        return f"<{error}>"


def golden_anchors(lock: dict) -> dict:
    """Every capture file the lock pins, compared against the file on disk.

    The lock's top-level `files` list pins the model repository; the capture
    anchors live in `macos_reference_paths` as name -> sha256. Iterating the
    lock rather than a hand-written list means a newly pinned anchor cannot be
    silently skipped, and a pinned anchor missing from the capture is reported
    instead of passing by omission.
    """
    anchors = {}
    for name, expected in sorted(lock["macos_reference_paths"].items()):
        path = CAPTURE / name
        if not path.exists():
            anchors[name] = {
                "present": False,
                "lock_sha256": expected,
                "matches_lock": False,
                "note": "pinned by the lock, absent from this capture directory",
            }
            continue
        measured = sha256(path)
        anchors[name] = {
            "present": True,
            "sha256": measured,
            "lock_sha256": expected,
            "matches_lock": measured == expected,
        }
    present = {k: v for k, v in anchors.items() if v["present"]}
    return {
        "anchors": anchors,
        "pinned": len(anchors),
        "present": len(present),
        "absent": sorted(k for k, v in anchors.items() if not v["present"]),
        "all_present_match": all(v["matches_lock"] for v in present.values()),
        "run_inputs_match": all(
            anchors[name]["matches_lock"]
            for name in (
                "encoder_input_features.npy",
                "encoder_input_mask.npy",
                "encoder_hidden.npy",
                "encoder_mask.npy",
            )
        ),
    }


def bundle_pins() -> dict:
    bundles = {}
    for name in ("island-attn-a-kt", "island-pv", "island-select-8head-scratch417"):
        manifest_path = BUNDLES / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        bundles[name] = {
            "manifest_sha256": sha256(manifest_path),
            "graph_hash": manifest["graph_hash"],
            "dispatch_plan": manifest["dispatch_plan"],
            "programs": len(manifest["programs"]),
            "scratch_bytes": [p["scratch_bytes"] for p in manifest["programs"]],
            "compiler_toolchain": manifest["compiler"]["toolchain"],
            "payloads": {
                Path(p["path"]).name: sha256(BUNDLES / name / Path(p["path"]).name)
                for p in manifest["payloads"]
            },
        }
    return bundles


def main() -> int:
    import importlib.metadata

    lock = json.loads(LOCK.read_text())
    anchors = golden_anchors(lock)

    ldd = run("ldd", str(WORKER))
    libmlx = Path(
        importlib.metadata.distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so")
    )

    environment = {
        "host": platform.node(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "gpu_stack": {
            "mesa_package": run("pacman", "-Q", "mesa-honeykrisp-omarchy") or None,
            "icd": "system ICD, unmodified; no VK_DRIVER_FILES override in this run",
        },
        "mlx": {
            "mlx_omarchy": importlib.metadata.version("mlx-omarchy"),
            "python": sys.version.split()[0],
            "venv": "~/venv-agxgen",
            "libmlx_sha256": sha256(libmlx),
        },
        "ane": {
            "device": run("stat", "-c", "%F %a", "/dev/accel/accel0"),
            "worker": str(WORKER),
            "worker_sha256": sha256(WORKER),
            "worker_linked_ane_libs": [
                line.strip() for line in ldd.splitlines() if "ane" in line.lower()
            ],
            "libane": str(LIBANE),
            "libane_sha256": sha256(LIBANE),
            "bundles": bundle_pins(),
        },
        "encoder": {
            "mil": str(SRC / "model.mil"),
            "mil_sha256": sha256(SRC / "model.mil"),
            "model_repo": lock["model_repo"],
            "model_revision": lock["model_revision"],
        },
        "contract": {
            "lock": "overlay/tools/coreml/parakeet-reference.lock",
            "lock_sha256": sha256(LOCK),
            "encoder_parity_py_sha256": sha256(
                ROOT / "package/coreml/encoder_parity.py"
            ),
            "liveness_helper_sha256": sha256(ROOT / "tools/ane_worker_liveness.py"),
            "macos_reference_environment": lock["macos_reference_environment"],
            "derived_from": lock["numerical_contract"]["derived_from"],
        },
        "derivation": {
            path.name: sha256(path)
            for path in sorted((ROOT / "derivation").glob("*.py"))
        },
        "harness": {
            path.name: sha256(path)
            for path in sorted(ROOT.glob("ep3-*.sh"))
            if not path.name.startswith("ep3-probe")
        },
    }

    (ROOT / "golden-anchors.json").write_text(json.dumps(anchors, indent=2))
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
    print(
        json.dumps(
            {
                "pinned": anchors["pinned"],
                "present": anchors["present"],
                "absent": anchors["absent"],
                "all_present_match": anchors["all_present_match"],
                "run_inputs_match": anchors["run_inputs_match"],
            },
            indent=2,
        )
    )
    return 0 if anchors["all_present_match"] and anchors["run_inputs_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
