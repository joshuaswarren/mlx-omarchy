#!/usr/bin/env python3
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

root, python, wheel = map(Path, sys.argv[1:])
source = root / "receipts/parity-baseline-20260908"
spec = importlib.util.spec_from_file_location("matrix_runner", source / "run-baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
modes = {"disabled": {"MLX_OMARCHY_FUSED_CHAIN": "0"}, "default": {}}
output = root / "receipts/2026-09-09-fused-chain-default"
expected_ids = None
runs = []
for rep in range(1, 4):
    pair = {}
    for mode in (("disabled", "default") if rep % 2 else ("default", "disabled")):
        folder = output / f"pair-{rep}-{mode}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        shutil.copyfile(source / "capture-ids.py", folder / "capture-ids.py")
        runner.HERE = folder
        runner.EXTRA_MLX_ENV = modes[mode]
        sys.argv = ["run-baseline.py", str(python), str(wheel), "1"]
        runner.main()
        ids = [json.loads(line)["ids"] for line in (folder / "rep1.ids.jsonl").read_text().splitlines()]
        if expected_ids is None:
            assert mode == "disabled" and len(ids) == 6
            expected_ids = ids
        assert ids == expected_ids, (rep, mode, "full generated-ID mismatch")
        summary = json.loads((folder / "baseline-summary.json").read_text())
        pair[mode] = {row["leg_id"]: row for row in summary["legs"]}
    runs.append(pair)
    print("PAIRED_FULL_ID_PASS", rep, flush=True)
rows = []
for key in sorted(runs[0]["disabled"]):
    row = {"leg_id": key}
    for metric in ("decode_tok_s_median", "prefill_tok_s_median"):
        values = {mode: [pair[mode][key][metric] for pair in runs] for mode in modes}
        medians = {mode: statistics.median(samples) for mode, samples in values.items()}
        row[metric] = {
            "samples": values,
            "medians": medians,
            "default_over_disabled": medians["default"] / medians["disabled"],
        }
    row["generated_ids_sha256_16"] = runs[0]["disabled"][key]["generated_ids_sha256_16"]
    rows.append(row)
result = {
    "source_commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
    "paired_repetitions": 3,
    "measured_legs": 36,
    "all_full_generated_ids_equal": True,
    "results": rows,
}
(output / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
