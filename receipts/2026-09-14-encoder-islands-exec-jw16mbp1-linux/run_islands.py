#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""One guarded ANE submit per encoder island on jw16mbp1-linux.

No pre-open of /dev/accel/accel0: the worker's own open resumes the device
and its SET domains. No SET write, no unload, no reboot, no GPU lock, and
no retry after a failure — a -110 ends that island and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = Path("/var/tmp/jw16-tiny-select/mlx-omarchy-ane-worker")
LIBANE = Path("/var/tmp/jw16-ane-first-exec/libane.so")
SET_READ = Path("/var/tmp/jw16-ane-set-resume-20260914/set_read.py")
STAGE = HERE / "tensors"
DEADLINE_MS = 10000

ISLANDS = {
    "B": {
        "bundle": "island-select-8head",
        "inputs": ["cond", "matrix_bd_5", "ninf_rt"],
        "outputs": ["attention_mask_9"],
        "task_descriptors": 5,
    },
    "C": {
        "bundle": "island-pv",
        "inputs": ["probs", "v_heads"],
        "outputs": ["attn_output_1"],
        "task_descriptors": 208,
    },
    "A": {
        "bundle": "island-attn-a-kt",
        "inputs": ["k_headsT", "pos_kT", "q_scaled", "q_v"],
        "outputs": ["attention_scores_1", "matmul_0"],
        "task_descriptors": 416,
    },
}


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unreadable: {exc}"


def set_read() -> object:
    proc = sh(["sudo", "-n", "python3", str(SET_READ)])
    if proc.returncode != 0:
        return {"error": (proc.stderr or "").strip()}
    return json.loads(proc.stdout)


def health() -> dict:
    return {
        "hostname": sh(["hostname"]).stdout.strip(),
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "accel": sh(
            ["stat", "-c", "%A %U:%G %t,%T %n", "/dev/accel/accel0"]
        ).stdout.strip(),
        "module_initstate": read("/sys/module/ane/initstate"),
        "refcnt": read("/sys/module/ane/refcnt"),
        "runtime_status": read(
            "/sys/class/accel/accel0/device/power/runtime_status"
        ),
        "workers": sh(["pgrep", "-a", "mlx-omarchy-ane"]).stdout.strip() or "none",
        "nproc": sh(["nproc"]).stdout.strip(),
        "loadavg": read("/proc/loadavg"),
    }


def kmsg(text: str) -> None:
    subprocess.run(
        ["sudo", "-n", "tee", "/dev/kmsg"], input=text + "\n", text=True,
        capture_output=True,
    )


def run(island: str) -> int:
    spec = ISLANDS[island]
    out_dir = HERE / f"island-{island}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = HERE / "bundles" / spec["bundle"]
    saved = {name: out_dir / f"{name}.out.bin" for name in spec["outputs"]}
    for path in saved.values():
        path.unlink(missing_ok=True)

    command = [
        str(WORKER),
        "--bundle", str(bundle),
        "--libane", str(LIBANE),
        "--deadline-ms", str(DEADLINE_MS),
        "--iterations", "1",
    ]
    for name in spec["inputs"]:
        command += ["--input", f"{name}={STAGE / (name + '.bin')}"]
    for name in spec["outputs"]:
        command += ["--expect", f"{name}={STAGE / (name + '.bin')}"]
        command += ["--save", f"{name}={saved[name]}"]

    health_pre = health()
    set_pre = set_read()
    marker = f"JW16_ENCODER_ISLAND_{island}"
    kmsg(f"{marker}_PRE {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    timed_out_locally = False
    try:
        proc = subprocess.run(
            command, text=True, capture_output=True, timeout=DEADLINE_MS / 1000 + 20
        )
        stdout, stderr, exit_code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out_locally = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        exit_code = None
    wall_s = time.time() - t0
    kmsg(f"{marker}_POST rc={exit_code} wall_ms={int(wall_s * 1000)}")
    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    set_post = set_read()
    health_post = health()
    text = stdout + stderr

    record = {
        "island": island,
        "bundle": spec["bundle"],
        "task_descriptors": spec["task_descriptors"],
        "command": command,
        "deadline_ms": DEADLINE_MS,
        "iterations_requested": 1,
        "started_at": started,
        "finished_at": finished,
        "wall_s": round(wall_s, 3),
        "exit": exit_code,
        "stdout": stdout.splitlines(),
        "stderr": stderr.splitlines(),
        "errno_110": any(
            token in text
            for token in ("ETIMEDOUT", "timed out", "-110", "errno 110")
        ),
        "local_timeout": timed_out_locally,
        "worker_sha256": sha256(WORKER),
        "libane_sha256": sha256(LIBANE),
        "bundle_payload_sha256": {
            payload.name: sha256(payload)
            for payload in sorted(bundle.glob("*.anec"))
        },
        "saved": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256(path) if path.exists() else None,
            }
            for name, path in saved.items()
        },
        "set_pre": set_pre,
        "set_post": set_post,
        "health_pre": health_pre,
        "health_post": health_post,
    }
    for line in stdout.splitlines():
        if line.startswith("worker status="):
            for field in line.split():
                key, _, value = field.partition("=")
                if key in ("status", "iterations", "released", "elapsed_ms"):
                    record[f"worker_{key}"] = int(value)
    (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in (
        "island", "exit", "wall_s", "errno_110", "stdout", "stderr",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ISLANDS:
        raise SystemExit("usage: run_islands.py A|B|C")
    raise SystemExit(run(sys.argv[1]))
