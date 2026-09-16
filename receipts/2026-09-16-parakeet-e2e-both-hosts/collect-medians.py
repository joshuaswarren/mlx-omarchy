#!/usr/bin/env python3
"""Collect per-run stage walls and medians from an E2E battery dir."""
import json, statistics, sys, glob, os

d = sys.argv[1]
runs = sorted((r for r in glob.glob(os.path.join(d, "out-*"))
               if os.path.exists(os.path.join(r, "e2e-report.json"))),
              key=lambda p: (not p[-1].isdigit(), p))
rows = []
for r in runs:
    rep = json.load(open(os.path.join(r, "e2e-report.json")))
    stages = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
    ex = rep.get("execution", {})
    row = {
        "run": os.path.basename(r),
        "audio_load": stages.get("audio_load"),
        "mel_frontend": stages.get("mel_frontend"),
        "encoder_ane": stages.get("encoder_ane") or stages.get("encoder_gpu"),
        "decoder_load": stages.get("decoder_load"),
        "tdt_decode": stages.get("tdt_decode"),
        "detokenize": stages.get("detokenize"),
        "ane_exec_ms": rep.get("ane", {}).get("exec_ms"),
        "total_pipeline_ms": (rep.get("timing") or {}).get("total_pipeline_ms"),
        "status": ex.get("status") or rep.get("status"),
        "emissions": ex.get("actual_emissions", rep.get("actual_emissions")),
    }
    if row["total_pipeline_ms"] is None:
        row["total_pipeline_ms"] = round(sum(v for k, v in stages.items() if k in
            ("audio_load", "mel_frontend", "encoder_ane", "encoder_gpu",
             "decoder_load", "tdt_decode", "detokenize")), 3)
    rows.append(row)

keys = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load",
        "tdt_decode", "detokenize", "ane_exec_ms", "total_pipeline_ms"]
print(json.dumps({"runs": rows}, indent=1))
print("\nmedians:")
for k in keys:
    vals = [r[k] for r in rows if r[k] is not None]
    if vals:
        print(f"  {k}: all={statistics.median(vals):.3f}"
              + (f" r2r6={statistics.median(vals[1:]):.3f}" if len(vals) > 5 else ""))
