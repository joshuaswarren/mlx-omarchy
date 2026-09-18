#!/usr/bin/env python3
"""Native macOS facts for the community collectors. No MLX install required."""

import json
import platform

import bench_matrix
from collect_common import run_tool, run_python_probe


def not_applicable():
    return {"available": False, "error": "not applicable to native macOS MLX"}


# Runs via run_python_probe on the Mac. Read-only: ioreg queries and one
# best-effort powermetrics sample. Structured so every string that
# reaches the report has passed the Redactor back in the parent.
ANE_PROBE_CODE = r"""
import json, plistlib, re, subprocess

out = {"available": False, "instances": [], "ane_nodes": [],
       "dart_nodes": [], "pmgr_nodes": [],
       "coreml": {"available": False, "compute_units": None, "error": None},
       "powermetrics": {"available": False, "power_mw": None,
                        "error": None},
       "driver": None, "platform": None, "compiler": None,
       "set_base_candidate": None,
       "truncated": []}

# Byte-order contract (oracle: t6021-test-host T6021 capture 2026-09-17,
# ane-linux-experiments receipts/2026-09-17-t6021-test-host-t6021-macos-capture/):
# ioreg OSData properties present multi-byte cells in HOST byte order
# (little-endian on arm64). AAPL,phandle <69010000> is 0x169; reg is
# u64-LE (address, size) pairs whose sizes match IODeviceMemory exactly.
# v0.6.4 read phandle big-endian and shipped swapped values in every
# macOS row (5 published rows carry garbage such as 0x69010000).

MAX_REG_BYTES = 4096   # v0.6.5: 64 could not carry a real pmgr reg
                       # (t6002 pmgr reg is 3856 bytes / 241 ranges)
MAX_RANGES = 256       # t602x pmgr reg = 73 ranges; t6002 = 241


def _text(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        value = value.split(b"\x00")[0].decode("utf-8", "replace")
    return str(value)[:128] if value is not None else None


def _ascii(value):
    # ASCII text out of an OSData blob (target-type, platform-name).
    if isinstance(value, bytes):
        value = value.split(b"\x00")[0]
        return value.decode("ascii", "replace") if value else None
    if isinstance(value, str):
        return value or None
    return None


def _run(argv, timeout=30):
    return subprocess.run(argv, capture_output=True, timeout=timeout)


def _plist_argv_output(argv, timeout=60):
    proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        return None
    return plistlib.loads(proc.stdout)


def _json_argv_output(argv, timeout=30):
    # plutil -convert json emits JSON, not a plist.
    proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout.decode("utf-8", "replace"))


def _compatible(node):
    raw = node.get("compatible")
    if isinstance(raw, bytes):
        texts = [t.decode("utf-8", "replace") for t in raw.split(b"\x00")
                 if t]
        return texts[:8]
    if isinstance(raw, list):
        texts = [_ascii(t) for t in raw if _ascii(t)]
        return texts[:8] or None
    return None


def _reg(node):
    raw = node.get("reg") or node.get("IODeviceMemory")
    if isinstance(raw, bytes):
        if len(raw) > MAX_REG_BYTES:
            out["truncated"].append("reg_bytes:%s" % _text(
                node.get("name")))
            raw = raw[:MAX_REG_BYTES]
        return raw.hex()
    return None


def _reg_ranges(node):
    # Decoded reg ranges as ["0xADDR/0xSIZE", ...], host-endian cells.
    # u64 (address, size) pairs are the norm (Apple ARM reg); u32 pairs
    # cover 4/8-byte stragglers (mapper-ane0 reg is <00000000>).
    # IODeviceMemory (plist dicts) is the fallback when the raw reg
    # OSData is absent; its values are already host-native ints.
    name = _text(node.get("name"))
    raw = node.get("reg")
    if isinstance(raw, bytes):
        cell = 8 if len(raw) % 16 == 0 else 4
        if len(raw) % (2 * cell):
            out["truncated"].append("reg_ranges:%s" % (name or "?"))
            return None
        vals = [int.from_bytes(raw[i:i + cell], "little")
                for i in range(0, len(raw), cell)]
        pairs = list(zip(vals[0::2], vals[1::2]))
    elif isinstance(node.get("IODeviceMemory"), list):
        pairs = []
        for entry in node["IODeviceMemory"]:
            if isinstance(entry, list) and entry and \
                    isinstance(entry[0], dict):
                pairs.append((int(entry[0].get("address") or 0),
                              int(entry[0].get("length") or 0)))
    else:
        return None
    total = len(pairs)
    ranges = ["0x%x/0x%x" % (base, size) for base, size in pairs]
    if len(ranges) > MAX_RANGES:
        out["truncated"].append(
            "reg_ranges:%s:%d" % (name or "?", total))
        ranges = ranges[:MAX_RANGES]
    return ranges or None


def _keep(node):
    # Keys the payload schema whitelists; AAPL,phandle is carried as
    # `phandle` below.
    keys = ("name", "compatible", "reg", "IODeviceMemory",
            "IOInterruptControllers", "IOInterruptSpecifiers", "IOClass")
    return {k: node[k] for k in keys
            if node.get(k) is not None and k != "IODeviceMemory"}


# --- ANE driver instances and candidate driver classes -------------------
try:
    raw = _plist_argv_output(
        ["ioreg", "-a", "-rc", "H11ANEIn", "-l"], timeout=30)
    for node in (raw or []):
        dp = node.get("DeviceProperties") or {}
        out["instances"].append({
            "name": _text(node.get("IONameMatched")),
            "matched": _text(node.get("IONameMatched")),
            "firmware_loaded": node.get("FirmwareLoaded") is True,
            "cores": dp.get("ANEDevicePropertyNumANECores"),
            "version": dp.get("ANEDevicePropertyANEVersion"),
            "minor_version": dp.get("ANEDevicePropertyANEMinorVersion"),
            "hw_board_type": dp.get("ANEDevicePropertyANEHWBoardType"),
            "arch": _text(dp.get(
                "ANEDevicePropertyTypeANEArchitectureTypeStr")),
        })
    out["available"] = bool(out["instances"])
    out["instances"] = out["instances"][:8]
except Exception as exc:
    out["truncated"].append("ioreg_instances:%s" % type(exc).__name__)

# --- Driver identity ------------------------------------------------------
try:
    driver = {}
    first = out["instances"][0] if out["instances"] else {}
    counts = {}
    for klass in ("H11ANEIn", "AppleH13ANEInterface",
                  "AppleH16ANEInterface"):
        try:
            # Zero matches: ioreg -a exits 0 and prints NOTHING (an empty
            # buffer, not a plist) - that is the load-bearing negative.
            proc = _run(["ioreg", "-a", "-rc", klass, "-l"], timeout=30)
            if proc.returncode != 0 or not proc.stdout.strip():
                counts[klass] = 0
            else:
                counts[klass] = len(plistlib.loads(proc.stdout))
        except Exception:
            counts[klass] = None
    # The negative result is load-bearing: a future generation must be
    # able to see that H13/H16 drivers were ABSENT, not merely unrecorded
    # (t6021 is an H11-driver / h14g-generation part).
    driver["classes_empty"] = [k for k in ("AppleH13ANEInterface",
                                           "AppleH16ANEInterface")
                               if counts.get(k) == 0]
    driver["matched_class"] = None
    driver["bundle_identifier"] = None
    try:
        raw = _plist_argv_output(
            ["ioreg", "-a", "-rc", "H11ANEIn", "-l"], timeout=30) or []
        if raw:
            driver["matched_class"] = _text(raw[0].get("IOClass")) \
                or "H11ANEIn"
            driver["bundle_identifier"] = _text(
                raw[0].get("CFBundleIdentifier"))
    except Exception:
        pass
    try:
        kext = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             "/System/Library/Extensions/AppleH11ANEInterface.kext/"
             "Contents/Info.plist"]) or {}
        driver["kext_version"] = _text(kext.get("CFBundleVersion"))
    except Exception as exc:
        driver["kext_version"] = None
        out["truncated"].append("driver_kext:%s" % type(exc).__name__)
    driver.update({
        "matched_compatible": first.get("matched"),
        "arch": first.get("arch"),
        "cores": first.get("cores"),
        "ane_version": first.get("version"),
        "ane_minor_version": first.get("minor_version"),
        "firmware_loaded": first.get("firmware_loaded"),
        "instance_count": len(out["instances"]),
    })
    out["driver"] = driver
except Exception as exc:
    out["truncated"].append("driver:%s" % type(exc).__name__)

# --- Platform identity (root platform-expert device) ---------------------
try:
    nodes = _plist_argv_output(
        ["ioreg", "-a", "-rc", "IOPlatformExpertDevice", "-l"],
        timeout=30) or []
    root = nodes[0] if nodes else {}
    out["platform"] = {
        "target_type": _ascii(root.get("target-type")),
        # platform-name is the raw SoC id ("t6021"); decoding it here
        # means no human ever hand-decodes the hex blob again.
        "soc_id": _ascii(root.get("platform-name")),
        "compatible": _compatible(root),
        "model": _text(root.get("model")),
    }
except Exception as exc:
    out["truncated"].append("platform:%s" % type(exc).__name__)

# --- ANE compiler provenance (disk facts; binaries are cache-resident) ---
try:
    compiler = {"daemons": [], "error": None}
    fw = "/System/Library/PrivateFrameworks/ANECompiler.framework"
    try:
        blob = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             fw + "/Resources/Info.plist"]) or {}
        compiler["framework_version"] = _text(
            blob.get("CFBundleShortVersionString"))
    except Exception:
        compiler["framework_version"] = None
    try:
        blob = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             "/System/Library/Extensions/AppleH11ANEInterface.kext/"
             "Contents/Info.plist"]) or {}
        compiler["kext_version"] = _text(blob.get("CFBundleVersion"))
    except Exception:
        compiler["kext_version"] = None
    for daemon in ("/usr/libexec/aned", "/usr/libexec/aneuserd"):
        if _run(["test", "-f", daemon]).returncode == 0:
            compiler["daemons"].append(daemon)
    svc = "/System/Library/PrivateFrameworks/ANECompilerService.framework"
    compiler["compiler_service_present"] = \
        _run(["test", "-d", svc]).returncode == 0
    # Framework bundles ship plist-only on this generation; the binary
    # lives in the dyld shared cache. A known condition, not a failure.
    compiler["binaries_cache_resident"] = \
        _run(["test", "-f", fw + "/Versions/A/ANECompiler"]
             ).returncode != 0
    compiler["daemons"] = compiler["daemons"][:8]
    out["compiler"] = compiler
except Exception as exc:
    out["truncated"].append("compiler:%s" % type(exc).__name__)

try:
    tree = _plist_argv_output(["ioreg", "-a", "-p", "IOService", "-l"],
                              timeout=120)
    if tree is None:
        raise RuntimeError("ioreg exit nonzero")
    ane_re = re.compile(r"^(ane\d*|dart-ane\d*|mapper-ane\d*)$")
    stack = list(tree) if isinstance(tree, list) else [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        name = _text(node.get("name"))
        if name and ane_re.match(name):
            entry = _keep(node)
            entry["name"] = name
            entry["compatible"] = _compatible(node)
            entry["reg"] = _reg(node)
            entry["reg_ranges"] = _reg_ranges(node)
            for key in ("IOInterruptControllers",
                        "IOInterruptSpecifiers", "IOClass"):
                if key in entry:
                    entry[key] = _text(entry[key])
            ph = node.get("AAPL,phandle")
            if isinstance(ph, bytes) and len(ph) == 4:
                # Host byte order: <69010000> is 0x169, not 0x69010000.
                ph = int.from_bytes(ph, "little")
            entry["phandle"] = ph
            if name.startswith(("dart-", "mapper-")):
                dart_id = node.get("dart-id")
                entry["dart_id"] = int.from_bytes(dart_id, "little") \
                    if isinstance(dart_id, bytes) and len(dart_id) == 4 \
                    else None
                # #iommu-cells is a devicetree-only property; IORegistry
                # never surfaces it. null here is a platform miss, not a
                # dropped value (marker added below).
                entry["iommu_cells"] = None
                out["dart_nodes"].append(entry)
            else:
                out["ane_nodes"].append(entry)
        elif name == "pmgr":
            # The IORegistry exposes the pmgr BLOCK base (reg first range
            # / IORegistryEntryLocation) but not the power-controller
            # children, so macOS can cross-check a derived SET base's
            # block but never supply the pwrstate offset itself.
            out["pmgr_nodes"].append({
                "name": name,
                "location": _text(node.get("IORegistryEntryLocation")),
                "reg": _reg(node),
                "reg_ranges": _reg_ranges(node),
            })
        children = node.get("IORegistryEntryChildren")
        if isinstance(children, list):
            stack.extend(children)
    # reg_ranges_total records the true range count even when the cap
    # dropped ranges (t602x pmgr reg: 73 ranges / 1168 bytes); null
    # means unknown (undecodable cells or capped list).
    for pmgr in out["pmgr_nodes"]:
        ranges = pmgr.get("reg_ranges")
        if ranges and len(ranges) < MAX_RANGES:
            pmgr["reg_ranges_total"] = len(ranges)
        else:
            pmgr["reg_ranges_total"] = None
    out["ane_nodes"] = out["ane_nodes"][:8]
    out["dart_nodes"] = out["dart_nodes"][:8]
    out["pmgr_nodes"] = out["pmgr_nodes"][:8]
    if len(out["ane_nodes"]) == 8 or len(out["dart_nodes"]) == 8 \
            or len(out["pmgr_nodes"]) == 8:
        out["truncated"].append("devicetree:node_cap")
    if out["dart_nodes"]:
        out["truncated"].append("iommu_cells:unavailable_on_macos")
except Exception as exc:
    out["truncated"].append("ioreg_tree:%s" % type(exc).__name__)

# --- SET-base candidate (the hypothesis label is the point) --------------
# On T8103/T6001 the ANE SET region is the ane_* pwrstate cluster at
# pmgr block base + 0xc000, and that sum equals the m1n1 ps_map
# constant. macOS cannot enumerate pwrstate offsets, but when Apple's
# own ane0 device is GRANTED a reg window at exactly base+0xc000 (as on
# t6021: ane0 reg range 3 = 0x8e08c000), that corroborates the
# hypothesis from Apple's own address assignment. A candidate, never a
# derived fact; the Linux/m1n1 side enumerates the cluster itself.
try:
    pmgr_base = None
    for pmgr in out["pmgr_nodes"]:
        ranges = pmgr.get("reg_ranges") or []
        if ranges:
            pmgr_base = int(ranges[0].split("/")[0], 16)
            break
    if pmgr_base is not None:
        candidate = pmgr_base + 0xc000
        confirms = False
        kinds = []
        for ane in out["ane_nodes"]:
            ane_kinds = []
            for rng in ane.get("reg_ranges") or []:
                base_s, size_s = rng.split("/")
                base, size = int(base_s, 16), int(size_s, 16)
                if base == candidate and size > 0:
                    confirms = True
                if base == candidate:
                    ane_kinds.append("pmgr_plus_c000")
                elif base == pmgr_base:
                    ane_kinds.append("pmgr_block")
                else:
                    ane_kinds.append("ane_mmio")
            if ane.get("reg_ranges"):
                ane["range_kinds"] = ane_kinds
        out["set_base_candidate"] = {
            "pmgr_block": "0x%x" % pmgr_base,
            "offset": "0xc000",
            "base": "0x%x" % candidate,
            "driver_window_confirms": confirms,
        }
except Exception as exc:
    out["truncated"].append("set_base:%s" % type(exc).__name__)

# CoreML compute-unit availability when pyobjc is present; otherwise a
# recorded miss, never a crash.
try:
    import CoreML  # type: ignore
    from CoreML import MLComputeUnits  # type: ignore
    out["coreml"] = {"available": True,
                     "compute_units": str(MLComputeUnits.all)[:64],
                     "error": None}
except Exception as exc:
    out["coreml"] = {"available": False, "compute_units": None,
                     "error": type(exc).__name__[:64]}

# ANE power/utilization needs root; record the miss cleanly without it.
try:
    proc = subprocess.run(
        ["powermetrics", "--samplers", "ane_power", "-n", "1", "-i", "100"],
        capture_output=True, timeout=30, text=True)
    if proc.returncode != 0:
        out["powermetrics"]["error"] = proc.stderr.strip()[:128] \
            or "exit %d" % proc.returncode
    else:
        watts = [float(m) for m in
                 re.findall(r"ANE Power: ([0-9.]+) mW", proc.stdout)]
        out["powermetrics"] = {"available": True,
                               "power_mw": watts[0] if watts else None,
                               "error": None}
except Exception as exc:
    out["powermetrics"]["error"] = type(exc).__name__[:64]

def _plain(value):
    if isinstance(value, bytes):
        return _text(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


print(json.dumps(_plain(out))[:120000])
"""


def probe_ane_port(redactor):
    """ANE porting facts from the working macOS installation.

    The macOS equivalents of the Linux devicetree capture: the driver
    instances (core count, hardware generation, firmware state), the
    DT-shaped ANE provider and ANE DART/mapper nodes with their MMIO
    ranges and interrupt specifiers, CoreML compute-unit availability,
    and a root-gated powermetrics sample. Everything is read-only; only
    powermetrics wants root, and its absence is recorded, not fatal.
    """
    rec = run_python_probe(ANE_PROBE_CODE, redactor,
                           label="ane-port macos", timeout=180)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        return {"available": False, "macos": None,
                "error": rec["error"] or rec["stderr"][:256] or
                "ane probe exit %s" % rec["exit_code"]}
    try:
        detail = json.loads(rec["stdout"])
    except ValueError:
        return {"available": False, "macos": None, "error": "bad-probe-json"}
    detail = {key: detail[key] for key in
              ("available", "instances", "ane_nodes", "dart_nodes",
               "pmgr_nodes", "coreml", "powermetrics", "truncated",
               "driver", "platform", "compiler", "set_base_candidate")
              if key in detail}
    return {"available": bool(detail.get("available")),
            "macos": redactor.apply_value(detail)}



def _text(record):
    return record["stdout"].strip() if record["exit_code"] == 0 else None


def _int(value):
    return int(value) if value and str(value).isdigit() else None


def probe_host(redactor):
    facts = bench_matrix.host_facts()
    model = run_tool(["sysctl", "-n", "hw.model"], redactor, timeout=10)
    active = run_tool(["sysctl", "-n", "hw.activecpu"], redactor, timeout=10)
    memory = facts.get("memsize_bytes")
    return {
        "available": True,
        "system": "Darwin",
        "arch": facts.get("machine"),
        "kernel_release": platform.release(),
        "os": facts.get("os"),
        "model": _text(model),
        "chip": facts.get("chip"),
        "cpu_online": _int(_text(active)),
        "cpu": {"present": _int(facts.get("cores")), "hotplug_control": None},
        "memory_total_mib": memory // (1024 * 1024) if memory else None,
        "gpu": facts.get("gpu"),
    }


def measurement_context():
    power = bench_matrix.power_state()
    processes = bench_matrix.clean_check()
    return {
        "power": {key: power.get(key) for key in ("source", "percent", "charging")}
        if power else None,
        "model_processes": {
            "status": processes["status"],
            "scanned": processes["scanned"],
            "matched_count": len(processes["matched"]),
        },
        "limits": "Process scan covers known model tools only. Other GPU activity "
                  "and temperature are not measured. Timings are observations, "
                  "not a controlled performance comparison.",
    }
