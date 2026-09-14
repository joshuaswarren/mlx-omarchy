#!/usr/bin/env python3
"""jw16 SET-resume one-shot, then 100-run if exact. No accel0 pre-open."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = Path("/var/tmp/jw16-ane-first-exec")
TOOL = VAL / "mlx-omarchy-ane-worker"
LIBANE = VAL / "libane.so"
BUNDLE = VAL / "bundle"
PHYS = "0x28e08c000"


def sh(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, **kw)


def set_read() -> dict:
    proc = sh(["sudo", "-n", "python3", str(HERE / "set_read.py")], check=True)
    return json.loads(proc.stdout)


def sysfs() -> dict:
    leftover = sh(["pgrep", "-a", "mlx-omarchy-ane"])
    return {
        "hostname": sh(["hostname"], check=True).stdout.strip(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "nproc": sh(["nproc"], check=True).stdout.strip(),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "accel": sh(
            ["stat", "-c", "%A %U:%G %t,%T %n", "/dev/accel/accel0"], check=True
        ).stdout.strip(),
        "bound": Path("/sys/class/accel/accel0/device").resolve().name,
        "module_initstate": Path("/sys/module/ane/initstate").read_text().strip(),
        "refcnt": Path("/sys/module/ane/refcnt").read_text().strip(),
        "runtime_status": Path(
            "/sys/class/accel/accel0/device/power/runtime_status"
        ).read_text().strip(),
        "workers": leftover.stdout.strip() or "none",
        "kernel": Path("/proc/sys/kernel/osrelease").read_text().strip()
        if Path("/proc/sys/kernel/osrelease").exists()
        else sh(["uname", "-r"], check=True).stdout.strip(),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kmsg(text: str) -> None:
    sh(["sudo", "-n", "tee", "/dev/kmsg"], input=text + "\n")


def start_sampler(out: Path, stop: Path) -> subprocess.Popen:
    if stop.exists():
        stop.unlink()
    return subprocess.Popen(
        ["sudo", "-n", "python3", str(HERE / "set_sample.py"), str(out), str(stop)]
    )


def stop_sampler(proc: subprocess.Popen, stop: Path) -> None:
    stop.write_text("1\n")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def main() -> int:
    y_out = HERE / "y.out.bin"
    if y_out.exists():
        y_out.unlink()
    set_pre = set_read()
    health_pre = sysfs()
    if health_pre["runtime_status"] != "suspended":
        print("ABORT runtime not suspended", json.dumps(health_pre), file=sys.stderr)
        return 2
    if set_pre["set0"]["ACTUAL"] != "0x0":
        print("ABORT SET0 not gated", json.dumps(set_pre), file=sys.stderr)
        return 2
    worker_hash = sha256(TOOL)
    libane_hash = sha256(LIBANE)
    kmsg(
        f"JW16_SET_RESUME_PRE {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
        f"boot={health_pre['boot_id']}"
    )
    sample_one = HERE / "set-during-oneshot.json"
    stop_one = HERE / "stop-oneshot"
    sampler = start_sampler(sample_one, stop_one)
    time.sleep(0.05)
    cmd = [
        str(TOOL),
        "--bundle",
        str(BUNDLE),
        "--libane",
        str(LIBANE),
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
        f"y={y_out}",
    ]
    t0 = time.time()
    try:
        worker = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired as exc:
        stop_sampler(sampler, stop_one)
        kmsg("JW16_SET_RESUME_POST timeout")
        print(json.dumps({"result": "TIMEOUT", "stdout": exc.stdout, "stderr": exc.stderr}))
        return 3
    elapsed_wall = time.time() - t0
    stop_sampler(sampler, stop_one)
    kmsg(f"JW16_SET_RESUME_POST rc={worker.returncode}")
    set_after_one = set_read()
    health_mid = sysfs()
    text = (worker.stdout or "") + (worker.stderr or "")
    hit_110 = any(tok in text for tok in ("ETIMEDOUT", "timed out", "-110", "errno 110"))
    exact = "verified output y exact" in (worker.stdout or "")
    oneshot = {
        "exit": worker.returncode,
        "stdout": (worker.stdout or "").splitlines(),
        "stderr": (worker.stderr or "").splitlines(),
        "exact": exact,
        "errno_110": hit_110,
        "wall_s": round(elapsed_wall, 3),
        "y_sha256": sha256(y_out) if y_out.exists() else None,
        "y_bytes": y_out.stat().st_size if y_out.exists() else 0,
        "set_pre": set_pre,
        "set_after": set_after_one,
        "set_during": json.loads(sample_one.read_text()) if sample_one.exists() else [],
        "health_pre": health_pre,
        "health_mid": health_mid,
        "worker_sha256": worker_hash,
        "libane_sha256": libane_hash,
        "command": cmd,
    }
    (HERE / "oneshot.json").write_text(json.dumps(oneshot, indent=2) + "\n")
    if hit_110 or worker.returncode != 0 or not exact:
        print(json.dumps({"phase": "oneshot", "result": "FAIL", "oneshot": oneshot}))
        return 1
    sample_100 = HERE / "set-during-100.json"
    stop_100 = HERE / "stop-100"
    sampler100 = start_sampler(sample_100, stop_100)
    soak = subprocess.run(
        [sys.executable, str(HERE / "run_100.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    stop_sampler(sampler100, stop_100)
    set_post = set_read()
    health_post = sysfs()
    dmesg = sh(["sudo", "-n", "dmesg", "-T"])
    lines = [ln for ln in (dmesg.stdout or "").splitlines() if "JW16_SET_RESUME" in ln or "ane 285c04000.ane" in ln]
    summary = {
        "phase": "complete",
        "result": "PASS" if soak.returncode == 0 else "SOAK_FAIL",
        "oneshot_exact": True,
        "soak_exit": soak.returncode,
        "soak_stdout": (soak.stdout or "").strip(),
        "soak_stderr": (soak.stderr or "").strip(),
        "set_post": set_post,
        "health_post": health_post,
        "set_during_100": json.loads(sample_100.read_text()) if sample_100.exists() else [],
        "dmesg_tail": lines[-80:],
        "worker_sha256": worker_hash,
        "libane_sha256": libane_hash,
        "set0_phys": PHYS,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("phase", "result", "oneshot_exact", "soak_exit", "soak_stdout", "set_post", "health_post")}))
    return soak.returncode


if __name__ == "__main__":
    raise SystemExit(main())
