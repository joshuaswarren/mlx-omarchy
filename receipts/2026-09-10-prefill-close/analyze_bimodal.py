#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
conditions = {x["sample"]: x for x in json.loads((root / "conditions.json").read_text())}
legs = defaultdict(list)
provenance = None
for path in sorted(root.glob("sample-*.json")):
    sample = int(path.stem.split("-")[1])
    run = json.loads(path.read_text())
    provenance = provenance or {
        "packages": run["packages"],
        "binary_provenance": run["binary_provenance"],
        "host": run["host"],
        "power": run["power"],
    }
    for position, leg in enumerate(
            (x for x in run["legs"] if x.get("measured")), 1):
        entry = dict(conditions[sample])
        entry.update({
            "leg_position": position,
            "prefill_tok_s": leg["metrics"]["prefill_tok_s"],
            "prefill_s": leg["metrics"]["prefill_s"],
            "decode_tok_s": leg["metrics"]["decode_tok_s"],
            "digest": leg["metrics"]["generated_ids_sha256_16"],
        })
        legs[leg["leg_id"]].append(entry)

summary = {"sample_count_per_leg": 12, "provenance": provenance, "legs": {}}
for leg_id, samples in sorted(legs.items()):
    values = sorted(x["prefill_tok_s"] for x in samples)
    gaps = [(values[i + 1] - values[i], i) for i in range(len(values) - 1)]
    gap, split = max(gaps)
    threshold = (values[split] + values[split + 1]) / 2
    for x in samples:
        x["cluster"] = "fast" if x["prefill_tok_s"] > threshold else "slow"
    groups = {}
    for field in ("residency", "order", "leg_position", "fresh_scratch_cache"):
        grouped = defaultdict(list)
        for x in samples:
            grouped[str(x[field])].append(x["prefill_tok_s"])
        groups[field] = {key: {"n": len(vals), "median": statistics.median(vals)}
                         for key, vals in sorted(grouped.items())}
    telemetry_keys = set().union(*(x["telemetry_before"] for x in samples))
    telemetry = {}
    for key in sorted(telemetry_keys):
        vals = [x["telemetry_before"].get(key) for x in samples]
        telemetry[key] = {"distinct": sorted(set(vals)), "fast": Counter(
            x["telemetry_before"].get(key) for x in samples if x["cluster"] == "fast"),
            "slow": Counter(x["telemetry_before"].get(key) for x in samples
                            if x["cluster"] == "slow")}
    summary["legs"][leg_id] = {
        "median_prefill_tok_s": statistics.median(values),
        "min_prefill_tok_s": min(values),
        "max_prefill_tok_s": max(values),
        "largest_gap_tok_s": gap,
        "cluster_threshold_tok_s": threshold,
        "cluster_counts": Counter(x["cluster"] for x in samples),
        "factor_medians": groups,
        "digests": sorted(set(x["digest"] for x in samples)),
        "telemetry": telemetry,
        "samples": samples,
    }
(root / "distribution-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({leg: {"median": x["median_prefill_tok_s"],
                        "range": [x["min_prefill_tok_s"], x["max_prefill_tok_s"]],
                        "clusters": x["cluster_counts"],
                        "factors": x["factor_medians"]}
                  for leg, x in summary["legs"].items()}, indent=2))
