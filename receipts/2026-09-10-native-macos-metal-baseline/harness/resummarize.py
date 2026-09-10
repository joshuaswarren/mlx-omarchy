#!/usr/bin/env python3
"""Re-summarize an existing native-baseline run tree with the final
contention classifier (clean_rows from run_native_matrix.py).

Used for the 2026-09-10 16m1mbp att7 matrix: its per-rep raw JSONs
(runs/rep*/*.json, verbatim bench_decode results + watch deltas) were
measured under the full protocol; the tree's original final JSON was
classified with an earlier, looser rule. This script re-runs ONLY the
summary step (identical clean_rows/stats code) over the untouched raw
reps and writes the final JSON. Measurements are not re-run or altered.

Usage: resummarize.py <rundir-with-runs-and-host-json> [reps]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_native_matrix import REFERENCE_NATIVE_DIGESTS, clean_rows, median

rundir = Path(sys.argv[1])
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 12

old = json.loads(next(rundir.glob("native-baseline-*.json")).read_text())
runs = {leg: [] for leg in old["legs"]}
for rep in range(1, reps + 1):
    for leg in runs:
        p = rundir / "runs" / f"rep{rep:02d}" / f"{leg}.json"
        runs[leg].append(json.loads(p.read_text()))

summary = {}
for leg, rows in runs.items():
    clean = clean_rows(rows)
    digests = {r["ids_sha256_16"] for r in rows}
    assert len(digests) == 1, (leg, digests)
    if len(clean) < 3:
        raise RuntimeError(f"{leg}: only {len(clean)} clean reps")
    def st(key):
        return {"median": round(median([r[key] for r in clean]), 3),
                "min": round(min(r[key] for r in clean), 3),
                "max": round(max(r[key] for r in clean), 3)}
    summary[leg] = {
        "model": rows[0]["model"],
        "prompt_id": rows[0]["prompt_id"],
        "prompt_tokens": rows[0]["prompt_tokens"],
        "generated_tokens": rows[0]["generated"],
        "reps": reps,
        "clean_reps": len(clean),
        "digest": digests.pop(),
        "digest_matches_reference_native":
            rows[0]["ids_sha256_16"] == REFERENCE_NATIVE_DIGESTS[leg],
        "decode_tok_s": st("decode_tps"),
        "prefill_tok_s": st("prefill_tps"),
        "all_rows": rows,
        "contended_reps_excluded":
            [i + 1 for i, r in enumerate(rows) if r not in clean],
    }

out = dict(old)
out["legs"] = summary
out["summary_method"] = ("resummarize.py over untouched runs/rep*/*.json "
                         "raw reps; clean = watch delta < 0.10s AND "
                         "decode_tps >= 70% of best-band median "
                         "(identical classifier to the committed "
                         "run_native_matrix.py)")
dest = rundir / ("native-baseline-"
                 + old["host"]["hostname"].split(".")[0] + ".json")
dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print("wrote", dest)
for leg, s in summary.items():
    print(leg, "clean", s["clean_reps"], "excl", s["contended_reps_excluded"],
          "decode", s["decode_tok_s"]["median"],
          "prefill", s["prefill_tok_s"]["median"])
