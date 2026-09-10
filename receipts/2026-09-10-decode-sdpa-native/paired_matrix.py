#!/usr/bin/env python3
import importlib.util
import json
import statistics
import sys
from pathlib import Path

root, python, wheel = map(Path, sys.argv[1:4])
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
source = root / "receipts/parity-baseline-20260908"
output = root / "receipts/2026-09-10-decode-sdpa-native/paired"
output.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("matrix_runner", source / "run-baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
old = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "ad964232ee67fecd",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
native = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "254d73fd93164b98",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
runs = []
for rep in range(1, reps + 1):
    pair = {}
    order = ("baseline", "candidate") if rep % 2 else ("candidate", "baseline")
    for side in order:
        folder = output / f"pair-{rep}-{side}"
        folder.mkdir()
        (folder / "capture-ids.py").write_text((source / "capture-ids.py").read_text())
        runner.HERE = folder
        runner.EXTRA_MLX_ENV = {
            "MLX_OMARCHY_SDPA_DECODE_NATIVE": "0" if side == "baseline" else "1"
        }
        sys.argv = ["run-baseline.py", str(python), str(wheel), "1"]
        runner.main()
        summary = json.loads((folder / "baseline-summary.json").read_text())
        pair[side] = {row["leg_id"]: row for row in summary["legs"]}
    runs.append(pair)
    print("PAIR_PASS", rep, flush=True)
rows = []
policy_ok = True
for leg_id in sorted(old):
    row = {"leg_id": leg_id, "old": old[leg_id], "native": native[leg_id], "sides": {}}
    for side in ("baseline", "candidate"):
        entries = [pair[side][leg_id] for pair in runs]
        digests = sorted({d for entry in entries for d in entry["generated_ids_sha256_16"]})
        samples = [entry["decode_tok_s_median"] for entry in entries]
        expected = {old[leg_id]} if side == "baseline" else {old[leg_id], native[leg_id]}
        valid = len(digests) == 1 and set(digests) <= expected
        policy_ok &= valid
        row["sides"][side] = {
            "decode_tok_s": {"samples": samples, "median": statistics.median(samples)},
            "prefill_tok_s": {
                "samples": [entry["prefill_tok_s_median"] for entry in entries],
                "median": statistics.median(entry["prefill_tok_s_median"] for entry in entries),
            },
            "digests": digests,
            "policy_ok": valid,
        }
    row["candidate_over_baseline"] = (
        row["sides"]["candidate"]["decode_tok_s"]["median"]
        / row["sides"]["baseline"]["decode_tok_s"]["median"]
    )
    rows.append(row)
result = {
    "source_commit": "da63329aaea2f373a79a3b338c224c99cbf19ecf",
    "paired_repetitions": reps,
    "measured_legs": reps * 12,
    "order": "alternating baseline/candidate; candidate/baseline",
    "gate": "MLX_OMARCHY_SDPA_DECODE_NATIVE=0/1 on the same release wheel",
    "policy_ok": policy_ok,
    "results": rows,
}
(output / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n")
assert policy_ok
print(json.dumps(result, indent=2), flush=True)
