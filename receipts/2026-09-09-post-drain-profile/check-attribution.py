import json
import statistics
from itertools import pairwise
from pathlib import Path

root = Path(__file__).resolve().parent

def load(name):
    return json.loads((root / name).read_text())

def dispatches(name):
    return [r for line in (root / name).read_text().splitlines()
            if (r := json.loads(line))["k"] == "d"]

results = {}
for name in ("profile", "top"):
    wall = next(r["median_ms"] for r in load(f"dense-wall-{name}.json")["results"]
                if (r["n"], r["k"]) == (151936, 896))
    events = [r for r in dispatches(f"dense-wall-{name}.jsonl")
              if r["e"] == 26 and r["n"] == 151936]
    assert len(events) == 51, (name, len(events))
    durations = [(r["t1"] - r["t0"]) / 1e6 for r in events[1:]]
    assert all(d > 0 for d in durations)
    timestamp = statistics.median(durations)
    results[name] = {"head_wall_median_ms": wall,
                     "head_timestamp_median_ms": timestamp,
                     "wall_over_timestamp": wall / timestamp}
    assert wall / timestamp > 100, results[name]

full = dispatches("bf16.jsonl")
head_gaps = [(b["t0"] - a["t1"]) / 1e6 for a, b in pairwise(full)
             if a["s"] == b["s"] and b["e"] == 26 and b["n"] == 151936]
assert head_gaps
results["full_model_head_preceding_gap_median_ms"] = statistics.median(head_gaps)
results["verdict"] = "TOP_OF_PIPE candidate rejected: timestamp attribution discrepancy persists. Wall time includes submission and synchronization. Neither timestamp sums nor gaps establish GPU utilization or idle time. No runtime default changed."
results["scope"] = "Diagnostic attribution investigation, not native numerical or performance acceptance."
print(json.dumps(results, indent=2))
