#!/usr/bin/env python3
"""Collect a fast, privacy-safe capability report for mlx-omarchy.

The report answers the questions a maintainer asks first: which Apple
Silicon machine, which kernel, which Mesa/Honeykrisp Vulkan stack, is the
ANE visible, and which mlx-omarchy wheel is installed. It also records
the CPU topology (present cores versus online), the boot-chain firmware
identity from the devicetree /chosen node, the redacted kernel command
line, and whether the booted devicetree carries an ANE node. It finishes
in seconds, downloads nothing, and never touches the network.

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

from collect_common import (SCHEMA_VERSION, Redactor, build_payload,
                            dump_json, json_bytes, run_tool)

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
    out["cpu"] = _cpu_topology()
    out["boot"] = _boot_chain(redactor)
    out["cmdline"] = _kernel_cmdline(redactor)
    out["core_shortfall"] = _core_shortfall(out["cpu"],
                                            out["cmdline"] or "")
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


def _read_dt_file(name, base=DT_BASE):
    try:
        with open(os.path.join(base, name), "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None


CPU_SYSFS = "/sys/devices/system/cpu"


def _read_cpu_list(path):
    """(ids, raw) from a sysfs cpu list like "0-3,8"; (None, None) absent."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError:
        return None, None
    ids = []
    for part in raw.split(","):
        lo, sep, hi = part.partition("-")
        try:
            first = int(lo)
            last = int(hi) if sep else first
        except ValueError:
            return [], raw
        if last < first or last - first > 65535:
            return [], raw
        ids.extend(range(first, last + 1))
    return ids, raw


def _cpu_topology(base=CPU_SYSFS):
    """Core counts from sysfs: the machine's real size, not the taskset.

    host.cpu_online is scheduler affinity, so a contributor pinned to one
    core of eight looked identical to a one-core machine. `present` says
    how many cores the machine has. `hotplug_control` records whether
    cpuN/online switches exist; their absence is what spin-table bringup
    looks like on Apple Silicon.
    """
    out = {"present": None, "possible": None, "online": None,
           "offline": None, "present_list": None, "possible_list": None,
           "online_list": None, "offline_list": None,
           "hotplug_control": None}
    for key in ("present", "possible", "online", "offline"):
        ids, raw = _read_cpu_list(os.path.join(base, key))
        if raw is not None:
            out[f"{key}_list"] = raw or None
            out[key] = len(ids)
    try:
        entries = os.listdir(base)
    except OSError:
        return out
    out["hotplug_control"] = any(
        re.fullmatch(r"cpu([1-9]\d*)", name)
        and os.path.exists(os.path.join(base, name, "online"))
        for name in entries)
    return out


CHOSEN_PROPERTIES = (
    # (property under chosen, report key): boot firmware identity, never
    # personal. A bootloader devicetree override (GRUB's `devicetree`
    # command) REPLACES the m1n1-patched tree with a frozen snapshot, so
    # these strings describe whatever tree actually booted -- the
    # 2026-09-03 one-core incident came from exactly that override.
    ("asahi,m1n1-stage1-version", "m1n1_stage1"),
    ("asahi,m1n1-stage2-version", "m1n1_stage2"),
    ("asahi,iboot1-version", "iboot1"),
    ("asahi,iboot2-version", "iboot2"),
    ("asahi,system-fw-version", "system_fw"),
    ("asahi,os-fw-version", "os_fw"),
)


def _boot_chain(redactor, base=DT_BASE):
    """Boot firmware identity from the live /chosen node."""
    out = {}
    for prop, key in CHOSEN_PROPERTIES:
        raw = _read_dt_file(os.path.join("chosen", prop), base)
        out[key] = redactor.apply(raw.strip("\x00\n")) if raw else None
    return out


def _kernel_cmdline(redactor, path="/proc/cmdline"):
    """The live kernel command line, redacted.

    Carries root=UUID=... (the UUID rule replaces it) and reveals
    maxcpus/nosmp clamps or a custom-devicetree boot.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    return redactor.apply(raw.strip()) or None


def _core_shortfall(cpu, cmdline):
    """Plain fact: fewer cores running than present, unexplained.

    Recorded only when the command line carries no maxcpus=/nosmp clamp;
    a clamp is a deliberate choice, not a failure. Numbers, not alarms.
    """
    present = cpu.get("present")
    online = cpu.get("online")
    if not isinstance(present, int) or not isinstance(online, int):
        return None
    if present <= online:
        return None
    if cmdline and re.search(r"\bmaxcpus=\d+\b|\bnosmp\b", cmdline):
        return None
    return {"present": present, "online": online}


def _ane_devicetree(base=DT_BASE):
    """ANE node visibility in the booted devicetree, unlike /dev/ane.

    Packaged t8103 dtbs ship no ane node; a node appears only when the
    bootloader overrides the tree. This answers whether the running
    kernel was even offered an ANE by its boot chain.
    """
    out = {"node": False, "compatible": None}
    matches = []
    for dirpath, _dirs, _files in os.walk(base):
        named = bool(re.search(r"(?:^|/)ane(?:@[0-9a-f]+)?$", dirpath))
        tokens = [t for t in (_read_dt_file("compatible", dirpath) or "")
                  .split("\x00") if t]
        hit = [t for t in tokens if t == "apple,ane" or t.endswith("-ane")]
        if named or hit:
            matches.extend(hit or tokens or [os.path.basename(dirpath)])
    if matches:
        out["node"] = True
        out["compatible"] = sorted(set(matches))[:8]
    return out


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
    out["devicetree"] = _ane_devicetree()
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
    ap.add_argument("--submit", metavar="URL", default=None,
                    help="publish this report to the community endpoint "
                         "after printing it")
    args = ap.parse_args()
    report = collect()
    text = dump_json(report)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(json_bytes(report))
        print(f"[receipt] wrote {args.out} ({len(json_bytes(report))} bytes)")
    print(text, end="" if text.endswith("\n") else "\n")
    raise SystemExit(maybe_submit(args, report))


def maybe_submit(args, report):
    """Publish only after the report was shown and consent was given.

    The import is local so that a plain run of this script loads no
    network module at all.
    """
    import collect_submit

    endpoint = collect_submit.endpoint_from_args(args)
    if endpoint is None:
        return 0
    payload = build_payload("quick", report, {})
    digest = collect_submit.sha256_hex(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    if not args.submit and not collect_submit.confirm_interactive(
            endpoint, "quick report", digest):
        print("[submit] declined; nothing was uploaded")
        return 0
    try:
        receipt = collect_submit.submit_payload(
            endpoint, payload, token=collect_submit.token_from_env())
    except collect_submit.SubmitError as exc:
        print(f"[submit] FAILED: {exc}", file=sys.stderr)
        return 4
    print(f"[receipt] public URL: {receipt['url']} "
          f"(deduplicated={receipt['deduplicated']})")
    return 0


if __name__ == "__main__":
    main()
