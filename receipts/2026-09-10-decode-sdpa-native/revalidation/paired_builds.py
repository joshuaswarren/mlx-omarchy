#!/usr/bin/env python3
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

root, baseline_python, baseline_wheel, candidate_python, candidate_wheel, output = map(Path, sys.argv[1:7])
reps = int(sys.argv[7]) if len(sys.argv) > 7 else 3
source = root / "receipts/parity-baseline-20260908"
spec = importlib.util.spec_from_file_location("matrix_runner", source / "run-baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
output.mkdir(parents=True, exist_ok=True)
legs = {
    "baseline": (baseline_python, baseline_wheel, "df7d036"),
    "candidate": (candidate_python, candidate_wheel, "6c263b3"),
}
versions = {
    side: subprocess.check_output([str(python), "-c", "import mlx.core as mx; print(mx.__version__)"], text=True).strip()
    for side, (python, _, _) in legs.items()
}
assert versions["baseline"] != versions["candidate"], versions
for side, (_, _, commit) in legs.items():
    assert commit in versions[side], (side, versions[side], commit)

runs = []
for rep in range(1, reps + 1):
    pair = {}
    order = ("baseline", "candidate") if rep % 2 else ("candidate", "baseline")
    for side in order:
        folder = output / f"pair-{rep}-{side}"
        folder.mkdir()
        shutil.copyfile(source / "capture-ids.py", folder / "capture-ids.py")
        runner.HERE = folder
        runner.EXTRA_MLX_ENV = {}
        python, wheel, _ = legs[side]
        sys.argv = ["run-baseline.py", str(python), str(wheel), "1"]
        runner.main()
        summary = json.loads((folder / "baseline-summary.json").read_text())
        pair[side] = {row["leg_id"]: row for row in summary["legs"]}
    assert pair["baseline"].keys() == pair["candidate"].keys()
    runs.append(pair)
    print("PAIR_PASS", rep, flush=True)

rows = []
for leg_id in sorted(runs[0]["baseline"]):
    row = {"leg_id": leg_id, "sides": {}}
    for side in legs:
        entries = [pair[side][leg_id] for pair in runs]
        digests = sorted({digest for entry in entries for digest in entry["generated_ids_sha256_16"]})
        row["sides"][side] = {
            "decode_tok_s": {
                "samples": [entry["decode_tok_s_median"] for entry in entries],
                "median": statistics.median(entry["decode_tok_s_median"] for entry in entries),
            },
            "prefill_tok_s": {
                "samples": [entry["prefill_tok_s_median"] for entry in entries],
                "median": statistics.median(entry["prefill_tok_s_median"] for entry in entries),
            },
            "digests": digests,
        }
    assert row["sides"]["baseline"]["digests"] == row["sides"]["candidate"]["digests"], row
    row["candidate_over_baseline"] = row["sides"]["candidate"]["decode_tok_s"]["median"] / row["sides"]["baseline"]["decode_tok_s"]["median"]
    rows.append(row)

result = {
    "baseline_commit": "df7d036e5f5cdcd0e66a99e0047f488c1d431394",
    "candidate_commit": "6c263b325215ef2555bf525ebb83b1da22ca63de",
    "versions": versions,
    "paired_repetitions": reps,
    "measured_legs": reps * 12,
    "order": "alternating baseline/candidate; candidate/baseline",
    "all_digests_equal": True,
    "results": rows,
}
(output / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
