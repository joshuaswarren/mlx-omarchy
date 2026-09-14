#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Extra schema-4 fixtures. Never 1x896. Never take the GPU lock."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

OUT = Path("/var/tmp/jwm1-ane-util-20260913")
SMOKE = Path(
    "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/.work/build-ane-runtime/"
    "tests/omarchy/mlx-omarchy-ane-smoke"
)
WORKER = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/.work/mlx/build-ane-device/"
    "tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker"
)
LIBANE = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/receipts/"
    "2026-09-13-ane-worker-validation/libane.so"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device() -> dict[str, str]:
    leftover = subprocess.run(
        ["pgrep", "-a", "mlx-omarchy-ane"], capture_output=True, text=True
    )
    return {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "nproc": subprocess.check_output(["nproc"], text=True).strip(),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "accel": subprocess.check_output(
            ["stat", "-c", "%F %a", "/dev/accel/accel0"], text=True
        ).strip(),
        "module": Path("/sys/module/ane/version").read_text().strip(),
        "service": subprocess.check_output(
            ["systemctl", "is-active", "jwm1-ane.service"], text=True
        ).strip(),
        "workers": leftover.stdout.strip() or "none",
    }


def main() -> int:
    rows: list[dict] = []

    bundle = Path("/var/tmp/Jwm1AneAccelSmoke2-a9f14124/schema4-add-mul")
    diag = OUT / "schema4-add-mul.diagnostic"
    cmd = [str(SMOKE), str(bundle), "add-mul", "5000", str(diag)]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    rows.append(
        {
            "fixture": "schema4-add-mul",
            "path": str(bundle),
            "elements": 64,
            "mode": "add-mul",
            "tool": str(SMOKE),
            "exit": proc.returncode,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "result": (
                "PASS"
                if proc.returncode == 0 and "exact_fp16=PASS" in proc.stdout
                else "FAIL"
            ),
            "diagnostic": diag.read_text() if diag.exists() else "",
            "submitted": True,
        }
    )

    rows.append(
        {
            "fixture": "schema4-add-mul-worker",
            "path": str(
                Path(
                    "/var/tmp/AneWorkerValidation-c05ba1df/receipts/"
                    "2026-09-13-ane-worker-validation/bundle"
                )
            ),
            "elements": 64,
            "mode": "add-mul",
            "tool": "mlx-omarchy-ane-worker 100 x --iterations 1",
            "result": "PASS",
            "submitted": True,
            "note": "covered by 100-run report; exact y, released=2 each run",
        }
    )

    candidates = [
        (
            "mil-oneop-bundle",
            Path(
                "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/receipts/fixtures/"
                "mil-oneop-bundle"
            ),
            512,
            "add",
        ),
        (
            "ane-add-fp16-1x512",
            Path(
                "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/receipts/fixtures/"
                "exported/ane-add-fp16-1x512"
            ),
            512,
            "add",
        ),
        (
            "ane-mul-fp16-1x512",
            Path(
                "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/receipts/fixtures/"
                "exported/ane-mul-fp16-1x512"
            ),
            512,
            "mul",
        ),
        (
            "ane-add-fp16-1x896",
            Path(
                "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/receipts/fixtures/"
                "exported/ane-add-fp16-1x896"
            ),
            896,
            "add",
        ),
    ]
    for name, path, elems, mode in candidates:
        man = json.loads((path / "manifest.json").read_text())
        forbidden = elems == 896
        rows.append(
            {
                "fixture": name,
                "path": str(path),
                "elements": elems,
                "mode": mode,
                "manifest_version": man.get("manifest_version"),
                "graph_hash": man.get("graph_hash"),
                "submitted": False,
                "result": "SKIP",
                "reason": (
                    "forbidden 1x896 (-110)"
                    if forbidden
                    else (
                        "not 64-element; smoke CLI binds 64-element a/b; "
                        "1x512 is unproven on device and 1x896 hangs, "
                        "so not submitted"
                    )
                ),
            }
        )

    h13 = Path(
        "/var/tmp/Jwm1AneAccelSmoke2-a9f14124/receipts/fixtures/"
        "h13-explicit-chain-add-mul"
    )
    h13m = json.loads((h13 / "manifest.json").read_text())
    rows.append(
        {
            "fixture": "h13-explicit-chain-add-mul",
            "path": str(h13),
            "submitted": False,
            "result": "SKIP",
            "reason": (
                "not schema-4 ("
                + str(h13m.get("schema"))
                + "); adapted form is schema4-add-mul"
            ),
        }
    )

    report = {
        "schema": "mlx-omarchy.jwm1-ane-utilization-fixtures.v1",
        "device_after": device(),
        "smoke_sha256": sha256(SMOKE),
        "worker_sha256": sha256(WORKER),
        "libane_sha256": sha256(LIBANE),
        "gpu_lock": "not acquired",
        "prohibited": {
            "reboot": False,
            "ane_unloaded": False,
            "executed_1x896": False,
            "took_gpu_lock": False,
        },
        "fixtures": rows,
    }
    (OUT / "fixture-table.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "device_after": report["device_after"],
                "results": [
                    {
                        "fixture": row.get("fixture"),
                        "elements": row.get("elements"),
                        "result": row.get("result"),
                        "submitted": row.get("submitted"),
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )
    smoke_row = rows[0]
    print("--- smoke stdout ---")
    print(smoke_row.get("stdout", ""))
    print("--- smoke stderr ---")
    print(smoke_row.get("stderr", ""))
    return 0 if smoke_row.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
