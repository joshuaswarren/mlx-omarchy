#!/usr/bin/env python3
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

root, baseline, baseline_python, baseline_wheel, candidate_python, candidate_wheel = map(Path, sys.argv[1:])
source = root / "receipts/parity-baseline-20260908"
spec = importlib.util.spec_from_file_location("matrix_runner", source / "run-baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
legs = {
    "baseline": (baseline_python, baseline_wheel, "0"),
    "candidate": (candidate_python, candidate_wheel, "1"),
}
output = root / "receipts/2026-09-09-decode-dispatch"
expected_ids = None
runs = []
for rep in range(1, 6):
    pair = {}
    for side in (("baseline", "candidate") if rep % 2 else ("candidate", "baseline")):
        folder = output / f"pair-{rep}-{side}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        shutil.copyfile(source / "capture-ids.py", folder / "capture-ids.py")
        runner.HERE = folder
        py, wheel, gate = legs[side]
        runner.EXTRA_MLX_ENV = {"MLX_OMARCHY_FUSED_CHAIN": gate}
        sys.argv = ["run-baseline.py", str(py), str(wheel), "1"]
        runner.main()
        ids = [json.loads(line)["ids"] for line in (folder / "rep1.ids.jsonl").read_text().splitlines()]
        if expected_ids is None:
            assert side == "baseline" and len(ids) == 6
            expected_ids = ids
        assert ids == expected_ids, (rep, side, "full generated-ID mismatch")
        summary = json.loads((folder / "baseline-summary.json").read_text())
        pair[side] = {row["leg_id"]: row for row in summary["legs"]}
    assert pair["baseline"].keys() == pair["candidate"].keys()
    runs.append(pair)
    print("PAIRED_FULL_ID_PASS", rep, flush=True)
rows = []
for key in sorted(runs[0]["baseline"]):
    row = {"leg_id": key}
    for metric in ("decode_tok_s_median", "prefill_tok_s_median"):
        values = {side: [pair[side][key][metric] for pair in runs] for side in legs}
        medians = {side: statistics.median(v) for side, v in values.items()}
        row[metric] = {
            "samples": values,
            "medians": medians,
            "candidate_over_baseline": medians["candidate"] / medians["baseline"],
        }
    row["generated_ids_sha256_16"] = runs[0]["baseline"][key]["generated_ids_sha256_16"]
    rows.append(row)
result = {
    "baseline_commit": subprocess.check_output(["git", "-C", baseline, "rev-parse", "HEAD"], text=True).strip(),
    "candidate_commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
    "paired_repetitions": 5,
    "measured_legs": 60,
    "all_full_generated_ids_equal": True,
    "results": rows,
}
(output / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
