#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
profile, markers, output = map(Path, sys.argv[1:4])
rows = [json.loads(line) for line in profile.read_text().splitlines()]
marks = [json.loads(line) for line in markers.read_text().splitlines()]
meta = next(row for row in rows if row.get("k") == "meta")
period = meta["period_ns"]
submits = {row["s"]: row for row in rows if row.get("k") == "s"}
sys.path.insert(0, str(root / "scripts"))
from profile_analyze import parse_kernel_names
names = parse_kernel_names(root / "overlay/mlx/backend/omarchy/compute.h")

start = next(m["t"] for m in marks if m["p"] == "prefill_start")
end = next(m["t"] for m in marks if m["p"] == "prefill_done")
def in_prefill(t):
    return start <= t < end

dispatches = sorted(
    (row for row in rows if row.get("k") == "d" and
     row["s"] in submits and in_prefill(submits[row["s"]]["t"])),
    key=lambda row: row["t0"])
durations = [(row["t1"] - row["t0"]) * period for row in dispatches]
busy = sum(durations)
span = (max(row["t1"] for row in dispatches) -
        min(row["t0"] for row in dispatches)) * period

def gap_summary(same_submission):
    values = sorted((b["t0"] - a["t1"]) * period
                    for a, b in zip(dispatches, dispatches[1:])
                    if (a["s"] == b["s"]) == same_submission)
    pick = lambda p: values[min(len(values) - 1, round(p * (len(values) - 1)))]
    return {"count": len(values), "total_ms": sum(values) / 1e6,
            "p50_us": pick(.5) / 1e3, "p90_us": pick(.9) / 1e3,
            "p99_us": pick(.99) / 1e3, "max_us": max(values) / 1e3}

by_kernel = {}
for row, duration in zip(dispatches, durations):
    name = names[row["e"]] if row["e"] < len(names) else f"kernel_{row['e']}"
    item = by_kernel.setdefault(name, {"dispatches": 0, "gpu_ms": 0.0})
    item["dispatches"] += 1
    item["gpu_ms"] += duration / 1e6
kernels = sorted(by_kernel.items(), key=lambda item: -item[1]["gpu_ms"])
for _, item in kernels:
    item["share_gpu_busy"] = item["gpu_ms"] / (busy / 1e6)

prefill_joins = [row for row in rows if row.get("k") == "j" and in_prefill(row["t"])]
prefill_submits = [row for row in rows if row.get("k") == "s" and in_prefill(row["t"])]
result = {
    "prompt": profile.parent.name,
    "host_prefill_ms": (end - start) / 1e6,
    "gpu": {"dispatches": len(dispatches), "submissions": len(prefill_submits),
            "busy_ms": busy / 1e6, "span_ms": span / 1e6,
            "busy_fraction": busy / span,
            "intra_submission_gaps": gap_summary(True),
            "inter_submission_gaps": gap_summary(False)},
    "host_instrumented": {
        "join_wait_ms": sum(row["wait"] for row in prefill_joins) / 1e6,
        "submit_ms": sum(row["dur"] for row in prefill_submits) / 1e6,
        "dispatch_record_ms": sum(row["h"] for row in dispatches) / 1e6},
    "kernels": [{"name": name, **item} for name, item in kernels],
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
