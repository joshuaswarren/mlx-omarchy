#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
label = sys.argv[2]
records = []
for line in (root / f"perf-{label}.log").read_text().splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if value.get("instrument") == "canonical_decode_cpu_clock_v1":
        records.append(value)
if len(records) != 1:
    raise SystemExit(f"{label}: expected one CPU-clock record, got {len(records)}")
record = records[0]
start = record["decode_start_monotonic_ns"] / 1e9
end = record["decode_end_monotonic_ns"] / 1e9
span = f"{start:.9f},{end:.9f}"
with (root / f"perf-{label}-scoped.txt").open("w") as out:
    completed = subprocess.run(
        ["perf", "script", "--ns", "--time", span, "-i",
         str(root / f"perf-{label}.data")], stdout=out, stderr=subprocess.PIPE,
        text=True, timeout=120)
if completed.returncode:
    raise SystemExit(completed.stderr)
with (root / f"perf-{label}-flat.txt").open("w") as out:
    completed = subprocess.run(
        ["perf", "report", "--stdio", "--no-children", "--percent-limit",
         "0.1", "--time", span, "-i", str(root / f"perf-{label}.data")],
        stdout=out, stderr=subprocess.PIPE, text=True, timeout=120)
if completed.returncode:
    raise SystemExit(completed.stderr)
print(json.dumps({"label": label, "start_s": start, "end_s": end,
                  "scoped_stack_bytes": (root / f"perf-{label}-scoped.txt").stat().st_size,
                  "flat_report_bytes": (root / f"perf-{label}-flat.txt").stat().st_size},
                 sort_keys=True))
