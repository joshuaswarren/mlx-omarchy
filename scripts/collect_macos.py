#!/usr/bin/env python3
"""Native macOS facts for the community collectors. No MLX install required."""

import platform

import bench_matrix
from collect_common import run_tool


def not_applicable():
    return {"available": False, "error": "not applicable to native macOS MLX"}


def probe_host(redactor):
    facts = bench_matrix.host_facts()
    model = run_tool(["sysctl", "-n", "hw.model"], redactor, timeout=10)
    active = run_tool(["sysctl", "-n", "hw.activecpu"], redactor, timeout=10)
    cores = facts.get("cores")
    present = int(cores) if cores and str(cores).isdigit() else None
    online = active["stdout"].strip() if active["exit_code"] == 0 else ""
    memory = facts.get("memsize_bytes")
    return {
        "available": True,
        "system": "Darwin",
        "arch": facts["machine"],
        "kernel_release": platform.release(),
        "os": facts.get("os"),
        "model": model["stdout"].strip() if model["exit_code"] == 0 else None,
        "chip": facts.get("chip"),
        "cpu_online": int(online) if online.isdigit() else None,
        "cpu": {"present": present, "hotplug_control": None},
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
