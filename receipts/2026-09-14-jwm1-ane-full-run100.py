#!/usr/bin/env python3
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import median

DIR = Path("/var/tmp/jwm1-ane-full-20260914")
RUNS = DIR / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
VAL = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation"
)
TOOL = Path(
    "/var/tmp/AneWorkerValidation-c05ba1df/.work/mlx/build-ane-device/"
    "tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker"
)
EXPECT_Y = "bade941d7d8f1e1097b9ce0298ff05617d85a9928de0e8c7eead5743f3f24352"
STATUS_RE = re.compile(
    r"worker status=(\d+) iterations=(\d+) released=(\d+) elapsed_ms=(\d+)"
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
    accel = subprocess.check_output(
        ["stat", "-c", "%F", "/dev/accel/accel0"], text=True
    ).strip()
    leftover = subprocess.run(
        ["pgrep", "-a", "mlx-omarchy-ane"], capture_output=True, text=True
    )
    svc = subprocess.check_output(
        ["systemctl", "is-active", "jwm1-ane.service"], text=True
    ).strip()
    return {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "accel": accel,
        "module_version": Path("/sys/module/ane/version").read_text().strip(),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "nproc": subprocess.check_output(["nproc"], text=True).strip(),
        "service": svc,
        "workers": leftover.stdout.strip() or "none",
        "uptime_s": subprocess.check_output(["uptime", "-s"], text=True).strip(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def main() -> int:
    pre = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    worker_sha = sha(TOOL)
    libane_sha = sha(VAL / "libane.so")
    print("WORKER", worker_sha)
    print("LIBANE", libane_sha)
    started = subprocess.check_output(
        ["date", "--iso-8601=seconds"], text=True
    ).strip()
    a_path = VAL / "a.bin"
    b_path = VAL / "b.bin"
    y_expect = VAL / "y.bin"
    bundle = VAL / "bundle"
    libane = VAL / "libane.so"
    rows = []
    timeouts = 0
    success = 0
    stop_reason = None
    for i in range(100):
        save = RUNS / ("y_%03d.bin" % i)
        cmd = [
            str(TOOL),
            "--bundle",
            str(bundle),
            "--libane",
            str(libane),
            "--deadline-ms",
            "5000",
            "--iterations",
            "1",
            "--input",
            "a=" + str(a_path),
            "--input",
            "b=" + str(b_path),
            "--expect",
            "y=" + str(y_expect),
            "--save",
            "y=" + str(save),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except subprocess.TimeoutExpired as e:
            timeouts += 1
            rows.append(
                {
                    "iteration": i,
                    "exit": "timeout",
                    "stdout": (e.stdout or "")[-500:],
                    "stderr": "TimeoutExpired",
                }
            )
            stop_reason = "timeout"
            break
        parsed = STATUS_RE.search(proc.stdout or "")
        released = int(parsed.group(3)) if parsed else None
        elapsed_ms = int(parsed.group(4)) if parsed else None
        exact = "verified output y exact" in (proc.stdout or "")
        y_sha = sha(save) if save.exists() else None
        minus110 = "-110" in (proc.stdout or "") or "-110" in (proc.stderr or "")
        row = {
            "iteration": i,
            "exit": proc.returncode,
            "released": released,
            "elapsed_ms_worker": elapsed_ms,
            "exact": exact,
            "y_sha256": y_sha,
            "minus110": minus110,
        }
        rows.append(row)
        if minus110:
            stop_reason = "-110"
            row["stdout"] = proc.stdout
            row["stderr"] = proc.stderr
            break
        if (
            proc.returncode != 0
            or not exact
            or released != 2
            or y_sha != EXPECT_Y
        ):
            stop_reason = "mismatch"
            row["stdout"] = proc.stdout
            row["stderr"] = proc.stderr
            break
        success += 1

    finished = subprocess.check_output(
        ["date", "--iso-8601=seconds"], text=True
    ).strip()
    post = {"device": device_state(), "memory": meminfo(), "t": time.time()}
    ok_ms = [
        r["elapsed_ms_worker"]
        for r in rows
        if r.get("elapsed_ms_worker") is not None and r.get("exit") == 0
    ]
    ok_ms_sorted = sorted(ok_ms)
    p95 = (
        ok_ms_sorted[int(round(0.95 * (len(ok_ms_sorted) - 1)))]
        if ok_ms_sorted
        else None
    )
    y_hashes = {}
    for p in sorted(RUNS.glob("y_*.bin")):
        h = sha(p)
        y_hashes[h] = y_hashes.get(h, 0) + 1
    report = {
        "started_at": started,
        "finished_at": finished,
        "worker_sha256": worker_sha,
        "libane_sha256": libane_sha,
        "successful_runs": success,
        "timeouts": timeouts,
        "stop_reason": stop_reason,
        "device_loss": post["device"]["accel"] != "character special file",
        "workers_released_each_run": all(
            r.get("released") == 2 for r in rows if r.get("exit") == 0
        ),
        "all_matched_baseline": success == 100 and y_hashes == {EXPECT_Y: 100},
        "y_files": sum(y_hashes.values()),
        "y_hashes": y_hashes,
        "median_ms": median(ok_ms) if ok_ms else None,
        "p95_ms": p95,
        "ane_exec_ns_sum": sum((m or 0) * 1_000_000 for m in ok_ms),
        "ane_worker_starts": len(rows),
        "ane_submissions": success * 2,
        "pre": pre,
        "post": post,
        "gpu_lock": "not acquired",
        "failures": [
            r for r in rows if r.get("exit") != 0 or r.get("exact") is False
        ],
    }
    (DIR / "100-run-report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary_keys = [
        "started_at",
        "finished_at",
        "successful_runs",
        "timeouts",
        "stop_reason",
        "device_loss",
        "workers_released_each_run",
        "all_matched_baseline",
        "y_files",
        "y_hashes",
        "median_ms",
        "p95_ms",
        "ane_exec_ns_sum",
        "ane_worker_starts",
        "ane_submissions",
        "gpu_lock",
    ]
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    print("POST_ACCEL", post["device"]["accel"])
    print("POST_SERVICE", post["device"]["service"])
    print("POST_WORKERS", post["device"]["workers"])
    print("POST_MODULE", post["device"]["module_version"])
    print("POST_NPROC", post["device"]["nproc"])
    print("POST_CPU", post["device"]["cpu_online"])
    print("MEM_BEFORE", pre["memory"]["mem_available_kb"])
    print("MEM_AFTER", post["memory"]["mem_available_kb"])
    ok = (
        success == 100
        and report["all_matched_baseline"]
        and not report["device_loss"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
