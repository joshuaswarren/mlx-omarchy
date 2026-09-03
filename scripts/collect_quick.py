#!/usr/bin/env python3
"""Collect a fast, privacy-safe capability report for mlx-omarchy.

The report answers the questions a maintainer asks first: which Apple
Silicon machine, which kernel, which Mesa/Honeykrisp Vulkan stack, is the
ANE visible, and which mlx-omarchy wheel is installed. It finishes in
seconds, downloads nothing, and never touches the network.

Personal data never reaches the output: every captured command output and
every free-text field passes through the shared Redactor, and the script
never reads serial numbers, machine IDs, or network configuration at all.

Usage:
  python3 scripts/collect_quick.py            # JSON on stdout
  python3 scripts/collect_quick.py --out f.json

Section shape (schema_version 1): every section carries "available"; a
missing tool or an import failure is recorded data, not an error. The
script exits 0 when the report was produced, 1 on an internal failure.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_common import SCHEMA_VERSION, Redactor, dump_json, json_bytes, run_tool

DT_BASE = "/sys/firmware/devicetree/base"


def probe_host(redactor):
    out = {"available": True}
    info = os.uname()
    out["arch"] = info.machine
    out["kernel_release"] = info.release
    try:
        out["cpu_online"] = len(os.sched_getaffinity(0))
    except AttributeError:
        out["cpu_online"] = os.cpu_count()
    try:
        out["page_size_bytes"] = os.sysconf("SC_PAGESIZE")
    except (ValueError, OSError):
        out["page_size_bytes"] = None
    mem = _mem_total()
    out["memory_total_mib"] = mem
    out["devicetree"] = _devicetree(redactor)
    return out


def _mem_total():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _devicetree(redactor):
    """Machine model and chip compatible strings; never the serial."""
    out = {"model": None, "compatible": None}
    model = _read_dt_file("model")
    if model:
        out["model"] = redactor.apply(model.strip("\x00\n"))
    compat = _read_dt_file("compatible")
    if compat:
        parts = [p for p in compat.split("\x00") if p]
        out["compatible"] = parts
    return out


def _read_dt_file(name):
    try:
        with open(os.path.join(DT_BASE, name), "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def _device_blocks(text):
    """Split `vulkaninfo --summary` into one dict per GPUn: block."""
    devices = []
    current = None
    for line in text.splitlines():
        if re.match(r"^GPU\d+:$", line.strip()):
            current = {}
            devices.append(current)
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and " " not in key:
            current[key] = value.strip()
    return devices


def _primary_device(devices):
    """The device that actually matters for this project.

    A host with Honeykrisp also enumerates llvmpipe, and llvmpipe is
    listed last. Reporting it would make every submission useless for
    driver work, so prefer the real GPU: Honeykrisp first, then any
    non-CPU device, and only then whatever came first.
    """
    if not devices:
        return {}
    for dev in devices:
        if "HONEYKRISP" in dev.get("driverID", "").upper():
            return dev
    for dev in devices:
        if dev.get("deviceType", "") != "PHYSICAL_DEVICE_TYPE_CPU":
            return dev
    return devices[0]


def probe_mesa(redactor):
    """Vulkan stack identity from `vulkaninfo --summary`."""
    out = {"available": False, "properties": {}, "summary": None}
    rec = run_tool(["vulkaninfo", "--summary"], redactor,
                   label="vulkaninfo --summary", timeout=30)
    out["vulkaninfo"] = rec
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    devices = _device_blocks(rec["stdout"])
    primary = _primary_device(devices)
    out["devices"] = devices
    out["device_count"] = len(devices)
    out["properties"] = primary
    known = ("deviceName", "driverName", "driverVersion", "apiVersion",
             "vendorID", "deviceID", "deviceType", "conformanceVersion",
             "driverID", "driverInfo")
    out["gpu"] = {k: primary[k] for k in known if k in primary}
    out["available"] = bool(out["gpu"])
    return out


def probe_mesa_package(redactor):
    """Mesa package version as a cross-check; distro-specific and optional."""
    return {
        "pacman": run_tool(["pacman", "-Q", "mesa"], redactor,
                           label="pacman -Q mesa", timeout=15),
        "dpkg": run_tool(["dpkg-query", "-W", "-f=${Package} ${Version}\\n",
                          "mesa"], redactor, label="dpkg-query mesa",
                         timeout=15),
    }


def probe_ane(redactor):
    """Apple Neural Engine visibility: device node and libane."""
    node = os.path.exists("/dev/ane")
    out = {"available": node, "device_node": node}
    lib = run_tool(["sh", "-c", "ldconfig -p 2>/dev/null | grep -i libane"],
                   redactor, label="ldconfig libane", timeout=15)
    out["libane"] = lib["stdout"].strip() if lib["exit_code"] == 0 else None
    out["libane_probe"] = lib
    pkg = run_tool(["pkg-config", "--modversion", "libane"], redactor,
                   label="pkg-config libane", timeout=15)
    out["libane_version"] = pkg["stdout"].strip() \
        if pkg["exit_code"] == 0 else None
    out["pkgconfig"] = pkg
    return out


MLX_PROBE_CODE = r"""
import json
out = {"distributions": {}, "import_ok": False, "import_error": None,
       "info_tool": None, "default_device": None, "mlx_version": None}
import importlib.metadata
for dist in ("mlx-omarchy", "mlx"):
    try:
        out["distributions"][dist] = importlib.metadata.version(dist)
    except Exception:
        pass
import pathlib
try:
    import mlx.core as mx
    out["import_ok"] = True
    out["default_device"] = str(mx.default_device())
    out["mlx_version"] = getattr(mx, "__version__", None)
    import mlx
    cand = pathlib.Path(mlx.__file__).resolve().parent / "bin" / "mlx-omarchy-info"
    if cand.exists():
        out["info_tool"] = str(cand)
except Exception as exc:
    out["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def probe_mlx(redactor):
    """Installed wheel identity plus the shipped capability tool."""
    rec = run_tool([sys.executable, "-c", MLX_PROBE_CODE], redactor,
                   label="mlx import probe", timeout=90)
    out = {"available": False, "probe": rec, "info": None,
           "capabilities": None}
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    try:
        found = json.loads(rec["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return out
    out["available"] = bool(found.get("import_ok"))
    for key in ("distributions", "import_error", "default_device",
                "mlx_version", "info_tool"):
        out[key] = found.get(key)
    tool = found.get("info_tool")
    if tool:
        cap = run_tool([tool, "--json"], redactor,
                       label="mlx-omarchy-info --json", timeout=60)
        out["info"] = cap
        if cap["exit_code"] == 0:
            try:
                out["capabilities"] = json.loads(cap["stdout"])
            except ValueError:
                out["capabilities"] = None
    return out


def _strip_volatile(value):
    """Drop run-to-run noise so equal machine state gives equal output."""
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items()
                if k != "duration_ms"}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


DEFAULT_PROBES = {
    "host": probe_host,
    "mesa": probe_mesa,
    "mesa_package": probe_mesa_package,
    "ane": probe_ane,
    "mlx": probe_mlx,
}


def collect(probes=None):
    """Build the report dict. `probes` overrides sections for tests."""
    active = dict(DEFAULT_PROBES)
    if probes:
        active.update(probes)
    redactor = Redactor()
    report = {
        "report": "mlx-omarchy-quick",
        "schema_version": SCHEMA_VERSION,
    }
    for name in sorted(active):
        try:
            report[name] = active[name](redactor)
        except Exception as exc:
            report[name] = {
                "available": False,
                "error": redactor.apply(f"{type(exc).__name__}: {exc}"),
            }
    return _strip_volatile(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    help="also write the JSON report to FILE")
    args = ap.parse_args()
    report = collect()
    text = dump_json(report)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(json_bytes(report))
        print(f"[receipt] wrote {args.out} ({len(json_bytes(report))} bytes)")
    print(text, end="" if text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
