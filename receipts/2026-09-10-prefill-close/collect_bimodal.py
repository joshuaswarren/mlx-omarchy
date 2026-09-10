#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
out = root / "receipts/2026-09-10-prefill-close/bimodal-main-fork"
out.mkdir(parents=True, exist_ok=True)
base_manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
wheel = next((root / "receipts/2026-09-10-prefill-close/wheels/main").glob("*.whl"))
python = root / ".venv-base/bin/python"


def telemetry():
    values = {}
    patterns = (
        "/sys/class/thermal/thermal_zone*/type",
        "/sys/class/thermal/thermal_zone*/temp",
        "/sys/class/devfreq/*/name",
        "/sys/class/devfreq/*/cur_freq",
        "/sys/class/hwmon/hwmon*/name",
        "/sys/class/hwmon/hwmon*/temp*_input",
    )
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.removeprefix("/"))):
            try:
                values[str(path)] = path.read_text().strip()
            except OSError:
                pass
    return values


conditions = (
    ("normal", "idle30"),
    ("reverse", "warm"),
    ("reverse", "idle30"),
    ("normal", "warm"),
) * 3
records = []
for index, (order, residency) in enumerate(conditions, 1):
    if residency == "idle30":
        time.sleep(30)
    manifest = dict(base_manifest)
    manifest["workloads"] = list(base_manifest["workloads"])
    if order == "reverse":
        manifest["workloads"].reverse()
    manifest_path = out / f"manifest-{order}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result_path = out / f"sample-{index:02d}.json"
    log_path = out / f"sample-{index:02d}.log"
    before = telemetry()
    started_ns = time.monotonic_ns()
    command = [
        str(python), str(root / "scripts/bench_matrix.py"),
        "--manifest", str(manifest_path), "--mode", "run",
        "--python", str(python), "--host-label", "jwm1-linux",
        "--wheel", str(wheel), "--select", "short-decode-32",
        "--select", "long-decode-128", "--timeout", "600",
        "--out", str(result_path),
    ]
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=900)
    ended_ns = time.monotonic_ns()
    if completed.returncode != 0:
        raise SystemExit(f"sample {index} failed with {completed.returncode}")
    records.append({
        "sample": index,
        "order": order,
        "residency": residency,
        "fresh_scratch_cache": index == 1,
        "started_monotonic_ns": started_ns,
        "wall_seconds": (ended_ns - started_ns) / 1e9,
        "telemetry_before": before,
        "telemetry_after": telemetry(),
        "result": result_path.name,
    })
    (out / "conditions.json").write_text(json.dumps(records, indent=2) + "\n")
