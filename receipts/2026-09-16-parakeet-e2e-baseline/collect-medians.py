import json, statistics

base = "/var/tmp/ParakeetE2EBaseline7d82"
dirs = {1: "out-rr1"}
for i in range(2, 7):
    dirs[i] = f"out-r{i}"
rows = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load", "tdt_decode", "detokenize"]
data = {}
for i, o in dirs.items():
    d = json.load(open(f"{base}/{o}/e2e-report.json"))
    per = {s["stage"]: s["wall_ms"] for s in d["stages"]}
    per["total_pipeline_ms"] = d["timing"]["total_pipeline_ms"]
    per["ane_exec_ms"] = d["ane"]["exec_ms"]
    per["status"] = d["status"]
    per["submissions"] = d["ane"]["submissions"]
    per["control"] = d["execution"]["control"]
    per["tdt_fallback"] = d["execution"].get("tdt_fallback_reason")
    per["emissions"] = d["execution"].get("actual_emissions")
    per["worker_starts"] = d["ane"]["worker_starts"]
    per["timeouts"] = d["ane"]["timeouts"]
    data[i] = per

hdr = "run   " + "".join(f"{r[:10]:>12}" for r in rows) + f"{'total':>11}{'ane_exec':>10}"
print(hdr)
for i in sorted(data):
    d = data[i]
    print(f"r{i:<4}" + "".join(f"{d[r]:>12.3f}" for r in rows)
          + f"{d['total_pipeline_ms']:>11.3f}{d['ane_exec_ms']:>10.1f}")
print()
print("sanity: all runs status/control/emissions/submissions/timeouts/worker_starts")
for i in sorted(data):
    d = data[i]
    print(f"  r{i}: {d['status']} {d['control']} fallback={d['tdt_fallback']} "
          f"emissions={d['emissions']} submits={d['submissions']} "
          f"timeouts={d['timeouts']} starts={d['worker_starts']}")
print()
print("median over 5 repeats (r2-r6):")
for r in rows + ["total_pipeline_ms", "ane_exec_ms"]:
    v = statistics.median([data[i][r] for i in range(2, 7)])
    print(f"  {r:>18}: {v:.3f}")
print("median over all 6 runs (r1-r6):")
for r in rows + ["total_pipeline_ms", "ane_exec_ms"]:
    v = statistics.median([data[i][r] for i in range(1, 7)])
    print(f"  {r:>18}: {v:.3f}")
print()
print("mean over 5 repeats (r2-r6):")
for r in rows + ["total_pipeline_ms", "ane_exec_ms"]:
    v = statistics.mean([data[i][r] for i in range(2, 7)])
    print(f"  {r:>18}: {v:.3f}")
