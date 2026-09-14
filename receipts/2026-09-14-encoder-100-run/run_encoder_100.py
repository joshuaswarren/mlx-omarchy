#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""100 consecutive three-island ANE encoder passes on jw16.

Submit stability only: 100/100, identical encoder_hidden/mask to run 0,
zero timeouts, no errno 110. Not token-exact E2E.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/var/tmp/Encoder100Run")
SRC = ROOT / "encoder-source"
CAPTURE = ROOT / "capture"
BUNDLES = ROOT / "bundles"
WORKER = Path("/var/tmp/Encoder100Run/mlx-omarchy-ane-worker")
LIBANE = Path("/var/tmp/jw16-ane-first-exec/libane.so")
ENCODER = ROOT / "derivation" / "vulkan_encoder.py"
STOP_RE = re.compile(r"(ETIMEDOUT|timed out|\b-110\b|errno 110)", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        parts = rest.split()
        if parts:
            out[key] = int(parts[0])
    return {
        "mem_available_kb": out.get("MemAvailable", 0),
        "mem_free_kb": out.get("MemFree", 0),
        "anon_pages_kb": out.get("AnonPages", 0),
    }


def device_state() -> dict[str, str]:
    accel = subprocess.check_output(["stat", "-c", "%F %a", "/dev/accel/accel0"], text=True).strip()
    version_path = Path("/sys/module/ane/version")
    leftover = subprocess.run(["pgrep", "-a", "mlx-omarchy-ane"], capture_output=True, text=True)
    return {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "accel": accel,
        "module_version": version_path.read_text().strip() if version_path.exists() else "",
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "nproc": subprocess.check_output(["nproc"], text=True).strip(),
        "workers": leftover.stdout.strip() or "none",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel": subprocess.check_output(["uname", "-r"], text=True).strip(),
    }


def dmesg_110() -> list[str]:
    proc = subprocess.run(
        ["sudo", "-n", "dmesg", "-T"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return [f"dmesg_unavailable rc={proc.returncode} {(proc.stderr or '')[:200]}"]
    return [line for line in proc.stdout.splitlines() if "tm execution failed w/ -110" in line][-5:]


def main() -> int:
    python = sys.executable
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    baseline_hidden: str | None = None
    baseline_mask: str | None = None
    per_run: list[dict] = []
    abort: str | None = None
    timeouts = 0
    errno_110 = 0
    submissions = 0
    before = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    dmesg_pre = dmesg_110()
    started = time.time()

    for iteration in range(100):
        out = runs_dir / f"{iteration:03d}"
        scratch = ROOT / "scratch" / f"{iteration:03d}"
        if out.exists():
            for child in out.iterdir():
                child.unlink()
        out.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = [
            python,
            str(ENCODER),
            "--source", str(SRC),
            "--capture", str(CAPTURE),
            "--bundles", str(BUNDLES),
            "--worker", str(WORKER),
            "--libane", str(LIBANE),
            "--scratch", str(scratch),
            "--out", str(out),
            "--deadline-ms", "20000",
            "--islands", "ABC",
        ]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            timeouts += 1
            abort = "subprocess.TimeoutExpired"
            per_run.append(
                {
                    "iteration": iteration,
                    "status": "timeout",
                    "stdout": (exc.stdout or "")[-2000:],
                    "stderr": (exc.stderr or "")[-2000:],
                }
            )
            break
        wall_s = time.monotonic() - t0
        text = (proc.stdout or "") + (proc.stderr or "")
        hit_110 = bool(STOP_RE.search(text)) or proc.returncode == 110
        report_path = out / "run-report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        ane = report.get("ane") or {}
        submissions += int(ane.get("submissions") or 0)
        ane_timeouts = int(ane.get("timeouts") or 0)
        hidden = out / "encoder_hidden.npy"
        mask = out / "encoder_mask.npy"
        hidden_sha = sha256(hidden) if hidden.exists() else None
        mask_sha = sha256(mask) if mask.exists() else None
        row = {
            "iteration": iteration,
            "exit": proc.returncode,
            "wall_s": round(wall_s, 3),
            "submissions": ane.get("submissions"),
            "ane_timeouts": ane_timeouts,
            "encoder_hidden_sha256": hidden_sha,
            "encoder_mask_sha256": mask_sha,
        }
        if hit_110 or ane_timeouts:
            errno_110 += 1
            timeouts += ane_timeouts
            abort = "errno_110"
            row["status"] = "errno_110"
            row["stdout_tail"] = (proc.stdout or "")[-1500:]
            row["stderr_tail"] = (proc.stderr or "")[-1500:]
            per_run.append(row)
            break
        if proc.returncode != 0 or hidden_sha is None or mask_sha is None:
            abort = "encoder_failed"
            row["status"] = "failed"
            row["stdout_tail"] = (proc.stdout or "")[-1500:]
            row["stderr_tail"] = (proc.stderr or "")[-1500:]
            per_run.append(row)
            break
        if baseline_hidden is None:
            baseline_hidden = hidden_sha
            baseline_mask = mask_sha
        elif hidden_sha != baseline_hidden or mask_sha != baseline_mask:
            abort = "output_mismatch"
            row["status"] = "mismatch"
            per_run.append(row)
            break
        row["status"] = "ok"
        per_run.append(row)
        # Keep run 0 outputs; drop later npy to bound disk.
        if iteration > 0:
            hidden.unlink(missing_ok=True)
            mask.unlink(missing_ok=True)
            if report_path.exists() and iteration not in {1, 99}:
                report_path.unlink()

    after = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    successful = sum(1 for row in per_run if row.get("status") == "ok")
    document = {
        "schema": "mlx-omarchy.encoder-100-run.v1",
        "date": "2026-09-14",
        "host": before["device"]["hostname"],
        "result": "PASS" if successful == 100 and abort is None else "FAIL",
        "successful_runs": successful,
        "iterations_requested": 100,
        "timeouts": timeouts,
        "errno_110": errno_110,
        "submissions": submissions,
        "expected_submissions": 72 * successful,
        "all_matched_baseline": successful == 100
        and abort is None
        and baseline_hidden is not None,
        "baseline_encoder_hidden_sha256": baseline_hidden,
        "baseline_encoder_mask_sha256": baseline_mask,
        "module_version": before["device"]["module_version"],
        "module_version_post": after["device"]["module_version"],
        "abort_reason": abort,
        "wall_s": round(time.time() - started, 3),
        "gpu_lock": "/tmp/m1-gpu.lock held by caller, never stolen, never unlinked",
        "pre": before,
        "post": after,
        "dmesg_110_pre": dmesg_pre,
        "dmesg_110_post": dmesg_110(),
        "worker_sha256": sha256(WORKER),
        "libane_sha256": sha256(LIBANE),
        "pins": {
            "island_a_program_0": sha256(BUNDLES / "island-attn-a-kt" / "program-0.anec"),
            "island_a_program_1": sha256(BUNDLES / "island-attn-a-kt" / "program-1.anec"),
            "island_b_program_0": sha256(
                BUNDLES / "island-select-8head-scratch417" / "program-0.anec"
            ),
            "island_c_program_0": sha256(BUNDLES / "island-pv" / "program-0.anec"),
            "encoder_input_features": sha256(CAPTURE / "encoder_input_features.npy"),
            "encoder_input_mask": sha256(CAPTURE / "encoder_input_mask.npy"),
            "model_mil": sha256(SRC / "model.mil") if (SRC / "model.mil").exists() else None,
        },
        "per_run": per_run,
    }
    (ROOT / "100-run-report.json").write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": document["result"],
                "successful_runs": successful,
                "all_matched_baseline": document["all_matched_baseline"],
                "timeouts": timeouts,
                "errno_110": errno_110,
                "submissions": submissions,
                "module_version": document["module_version"],
                "baseline_encoder_hidden_sha256": baseline_hidden,
                "abort_reason": abort,
                "wall_s": document["wall_s"],
            }
        )
    )
    return 0 if document["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
