#!/usr/bin/env python3
import json
import sys
from pathlib import Path

profile, markers, output = map(Path, sys.argv[1:4])
rows = [json.loads(line) for line in profile.read_text().splitlines()]
marks = [json.loads(line) for line in markers.read_text().splitlines()]
meta = next(row for row in rows if row.get("k") == "meta")
period = meta["period_ns"]
submits = {row["s"]: row for row in rows if row.get("k") == "s"}
boundaries = {row["s"]: row for row in rows if row.get("k") == "q"}
joins = {row["s"]: row for row in rows if row.get("k") == "j"}
start = next(row["t"] for row in marks if row["p"] == "prefill_start")
end = next(row["t"] for row in marks if row["p"] == "prefill_done")
in_prefill = lambda t: start <= t < end
prefill_subs = {sub for sub, row in submits.items() if in_prefill(row["t"])}
dispatches = sorted(
    (row for row in rows if row.get("k") == "d" and row["s"] in prefill_subs),
    key=lambda row: row["t0"],
)
durations = [(row["t1"] - row["t0"]) * period for row in dispatches]
busy = sum(durations)
span = (dispatches[-1]["t1"] - dispatches[0]["t0"]) * period

def summarize(values):
    values = sorted(values)
    pick = lambda p: values[min(len(values) - 1, round(p * (len(values) - 1)))]
    return {
        "count": len(values),
        "total_ms": sum(values) / 1e6,
        "p50_us": pick(.5) / 1e3,
        "p90_us": pick(.9) / 1e3,
        "p99_us": pick(.99) / 1e3,
        "max_us": max(values) / 1e3,
    }

intra = []
inter = []
gap_rows = []
for left, right in zip(dispatches, dispatches[1:]):
    gap_ns = (right["t0"] - left["t1"]) * period
    if left["s"] == right["s"]:
        intra.append(gap_ns)
        continue
    inter.append(gap_ns)
    previous = boundaries[left["s"]]
    following = boundaries[right["s"]]
    join = joins.get(left["s"])
    gap_rows.append({
        "previous_submission": left["s"],
        "next_submission": right["s"],
        "gpu_gap_ms": gap_ns / 1e6,
        "previous_close_host_ns": previous["close"],
        "previous_queue_return_host_ns": previous["queue_t1"],
        "fence_observed_host_ns": None if join is None else join["t"] + join["wait"],
        "next_close_host_ns": following["close"],
        "next_queue_submit_host_ns": following["queue_t0"],
        "join_wait_ms": 0.0 if join is None else join["wait"] / 1e6,
        "cause": "no explicit host join" if join is None else join["reason"],
    })

result = {
    "prompt": profile.parent.name,
    "host_prefill_ms": (end - start) / 1e6,
    "dispatches": len(dispatches),
    "submissions": len(prefill_subs),
    "gpu_busy_ms": busy / 1e6,
    "gpu_span_ms": span / 1e6,
    "gpu_busy_fraction": busy / span,
    "intra_submission_gaps": summarize(intra),
    "inter_submission_gaps": summarize(inter),
    "join_reasons": {},
    "largest_inter_submission_gaps": sorted(gap_rows, key=lambda row: -row["gpu_gap_ms"])[:12],
}
for row in joins.values():
    if in_prefill(row["t"]):
        item = result["join_reasons"].setdefault(row["reason"], {"count": 0, "wait_ms": 0.0})
        item["count"] += 1
        item["wait_ms"] += row["wait"] / 1e6
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
