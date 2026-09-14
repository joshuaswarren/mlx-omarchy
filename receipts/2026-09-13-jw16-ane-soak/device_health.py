#!/usr/bin/env python3
"""Snapshot jw16 ANE device health. No GPU lock, no unload."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def drm_version() -> dict:
    import ctypes
    import fcntl

    class DrmVersion(ctypes.Structure):
        _fields_ = [
            ("version_major", ctypes.c_int),
            ("version_minor", ctypes.c_int),
            ("version_patchlevel", ctypes.c_int),
            ("name_len", ctypes.c_size_t),
            ("name", ctypes.c_char_p),
            ("date_len", ctypes.c_size_t),
            ("date", ctypes.c_char_p),
            ("desc_len", ctypes.c_size_t),
            ("desc", ctypes.c_char_p),
        ]

    name = ctypes.create_string_buffer(64)
    date = ctypes.create_string_buffer(64)
    desc = ctypes.create_string_buffer(128)
    ver = DrmVersion(
        name_len=64,
        name=ctypes.cast(name, ctypes.c_char_p),
        date_len=64,
        date=ctypes.cast(date, ctypes.c_char_p),
        desc_len=128,
        desc=ctypes.cast(desc, ctypes.c_char_p),
    )
    DRM_IOCTL_VERSION = 0xC0406400
    fd = os.open("/dev/accel/accel0", os.O_RDWR)
    try:
        fcntl.ioctl(fd, DRM_IOCTL_VERSION, ver)
    finally:
        os.close(fd)
    return {
        "version": f"{ver.version_major}.{ver.version_minor}.{ver.version_patchlevel}",
        "name": name.value.decode(),
        "desc": desc.value.decode(),
    }


def main() -> int:
    leftover = subprocess.run(
        ["pgrep", "-a", "mlx-omarchy-ane"], capture_output=True, text=True
    )
    flock = subprocess.run(
        ["flock", "-n", "/tmp/m1-gpu.lock", "-c", "true"],
        capture_output=True,
        text=True,
    )
    bound = Path("/sys/class/accel/accel0/device").resolve().name
    initstate = Path("/sys/module/ane/initstate")
    refcnt = Path("/sys/module/ane/refcnt")
    out = {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "nproc": subprocess.check_output(["nproc"], text=True).strip(),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "accel": subprocess.check_output(
            ["stat", "-c", "%A %U:%G %t,%T %n", "/dev/accel/accel0"], text=True
        ).strip(),
        "bound": bound,
        "module_initstate": initstate.read_text().strip() if initstate.exists() else "",
        "refcnt": refcnt.read_text().strip() if refcnt.exists() else "",
        "drm": drm_version(),
        "workers": leftover.stdout.strip() or "none",
        "gpu_lock": "not acquired" if flock.returncode == 0 else "held-by-other",
        "flock_n_rc": flock.returncode,
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
