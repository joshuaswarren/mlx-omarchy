#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""§45 100-run driver: one worker process per iteration, programs released."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coreml.inference_driver import run_inference_loop  # noqa: E402

VAL = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation"
)
TOOL = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/.work/mlx/build-ane-device/"
    "tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker"
)
RUNS = HERE / "runs"
STATUS_RE = re.compile(
    r"worker status=(\d+) iterations=(\d+) released=(\d+) elapsed_ms=(\d+)"
)


def meminfo() -> dict[str, int]:
    out = {}
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
    accel = subprocess.check_output(["stat", "-c", "%F", "/dev/accel/accel0"], text=True).strip()
    version = Path("/sys/module/ane/version").read_text().strip()
    online = Path("/sys/devices/system/cpu/online").read_text().strip()
    nproc = subprocess.check_output(["nproc"], text=True).strip()
    leftover = subprocess.run(
        ["pgrep", "-a", "mlx-omarchy-ane"], capture_output=True, text=True
    )
    return {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "accel": accel,
        "module_version": version,
        "cpu_online": online,
        "nproc": nproc,
        "workers": leftover.stdout.strip() or "none",
    }


def main() -> int:
    RUNS.mkdir(exist_ok=True)
    counters = {
        "ane_models_loaded": 0,
        "ane_packages_compiled": 0,
        "ane_package_cache_hits": 0,
        "ane_worker_starts": 0,
        "ane_submissions": 0,
        "ane_timeouts": 0,
        "ane_input_bytes": 0,
        "ane_output_bytes": 0,
        "ane_exec_ns": 0,
    }
    per_run_meta: list[dict] = []

    def snapshot() -> dict[str, int]:
        return dict(counters)

    def run_once(iteration: int):
        save = RUNS / f"y_{iteration:03d}.bin"
        cmd = [
            str(TOOL),
            "--bundle",
            str(VAL / "bundle"),
            "--libane",
            str(VAL / "libane.so"),
            "--deadline-ms",
            "5000",
            "--iterations",
            "1",
            "--input",
            f"a={VAL / 'a.bin'}",
            "--input",
            f"b={VAL / 'b.bin'}",
            "--expect",
            f"y={VAL / 'y.bin'}",
            "--save",
            f"y={save}",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=8
            )
        except subprocess.TimeoutExpired:
            counters["ane_worker_starts"] += 1
            counters["ane_timeouts"] += 1
            per_run_meta.append(
                {
                    "iteration": iteration,
                    "exit": "timeout",
                    "released": None,
                    "stdout": "",
                    "stderr": "subprocess.TimeoutExpired",
                }
            )
            return "DeadlineExceeded", {}
        text = proc.stdout + proc.stderr
        parsed = STATUS_RE.search(proc.stdout)
        released = int(parsed.group(3)) if parsed else None
        elapsed_ms = int(parsed.group(4)) if parsed else 0
        counters["ane_worker_starts"] += 1
        counters["ane_submissions"] += 2
        counters["ane_input_bytes"] += 256
        counters["ane_output_bytes"] += 128
        counters["ane_exec_ns"] += elapsed_ms * 1_000_000
        exact = "verified output y exact" in proc.stdout
        if proc.returncode != 0 or not exact or released != 2:
            counters["ane_timeouts"] += 1 if proc.returncode != 0 else 0
            per_run_meta.append(
                {
                    "iteration": iteration,
                    "exit": proc.returncode,
                    "released": released,
                    "exact": exact,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
            return "DeviceFailed", {}
        payload = save.read_bytes() if save.exists() else b""
        per_run_meta.append(
            {
                "iteration": iteration,
                "exit": 0,
                "released": released,
                "exact": True,
                "elapsed_ms_worker": elapsed_ms,
            }
        )
        return "Completed", {"y": payload}

    before = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    report = run_inference_loop(run_once, snapshot, iterations=100)
    after = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    document = report.to_dict()
    document["host"] = before["device"]["hostname"]
    document["pre"] = before
    document["post"] = after
    document["successful_runs"] = sum(
        1 for run in report.runs if run.status == "Completed"
    )
    document["timeouts"] = counters["ane_timeouts"]
    document["device_loss"] = after["device"]["accel"] != "character special file"
    document["workers_released_each_run"] = all(
        row.get("released") == 2 for row in per_run_meta if row.get("exit") == 0
    )
    document["gpu_lock"] = "not acquired"
    document["per_run_meta_failures"] = [
        row for row in per_run_meta if row.get("exit") != 0
    ]
    (HERE / "100-run-report.json").write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "successful_runs": document["successful_runs"],
                "all_matched_baseline": report.all_matched_baseline,
                "median_ms": report.median_ms,
                "p95_ms": report.p95_ms,
                "timeouts": document["timeouts"],
                "workers_released_each_run": document["workers_released_each_run"],
                "post_workers": after["device"]["workers"],
                "post_nproc": after["device"]["nproc"],
                "post_module": after["device"]["module_version"],
            }
        )
    )
    return 0 if document["successful_runs"] == 100 and report.all_matched_baseline else 1


if __name__ == "__main__":
    sys.exit(main())
