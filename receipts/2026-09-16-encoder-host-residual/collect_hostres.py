"""Collect the EncoderHostResidual verification evidence."""
import hashlib
import json
import re
import statistics

BASE = "/var/tmp/enc-hostres"
dirs = [f"{BASE}/ver-out-r{i}" for i in range(1, 7)]
rows = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load", "tdt_decode", "detokenize"]
data = {}
for i, d in enumerate(dirs, 1):
    rep = json.load(open(f"{d}/e2e-report.json"))
    per = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
    per["total_pipeline_ms"] = rep["timing"]["total_pipeline_ms"]
    per["ane_exec_ms"] = rep["ane"]["exec_ms"]
    per["status"] = rep["status"]
    per["submissions"] = rep["ane"]["submissions"]
    per["control"] = rep["execution"]["control"]
    per["tdt_fallback"] = rep["execution"].get("tdt_fallback_reason")
    seq = rep["layers"]["layer_6_decoder_sequence"]
    per["emissions"] = seq.get("actual_emissions")
    per["native"] = seq.get("native_emissions")
    per["prefix"] = seq.get("matching_prefix_length")
    per["mel_bit_exact"] = rep["layers"]["layer_2_preprocessing"]["mel"]["bit_exact"]
    per["mel_mask_exact"] = rep["layers"]["layer_2_preprocessing"]["mel_mask_exact"]
    per["bounds_pass"] = rep["layers"]["layer_5_encoder"]["all_bounds_pass"]
    per["worker_starts"] = rep["ane"]["worker_starts"]
    per["timeouts"] = rep["ane"]["timeouts"]
    per["cpu_tensor_events"] = rep["execution"].get("cpu_tensor_events")
    data[i] = per

print("per-run stage walls (ms):")
hdr = "run   " + "".join(f"{r[:10]:>12}" for r in rows) + f"{'total':>11}{'ane_exec':>10}"
print(hdr)
for i in sorted(data):
    d = data[i]
    print(f"r{i:<4}" + "".join(f"{d[r]:>12.3f}" for r in rows)
          + f"{d['total_pipeline_ms']:>11.3f}{d['ane_exec_ms']:>10.1f}")
print()
print("sanity per run:")
for i in sorted(data):
    d = data[i]
    print(f"  r{i}: {d['status']} {d['control']} fallback={d['tdt_fallback']} "
          f"emissions={d['emissions']}/{d['native']} prefix={d['prefix']} "
          f"mel_exact={d['mel_bit_exact']}/{d['mel_mask_exact']} bounds={d['bounds_pass']} "
          f"submits={d['submissions']} timeouts={d['timeouts']} "
          f"starts={d['worker_starts']} cpu_events={d['cpu_tensor_events']}")
print()
for label, rng in (("median r2-r6", range(2, 7)), ("median r1-r6", range(1, 7))):
    print(f"{label}:")
    for r in rows + ["total_pipeline_ms", "ane_exec_ms"]:
        v = statistics.median([data[i][r] for i in rng])
        print(f"  {r:>18}: {v:.3f}")

print()
print("transcript + encoder_hidden shas per run:")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
for i in range(1, 7):
    t = sha(f"{BASE}/ver-out-r{i}/transcript.txt")
    e = sha(f"{BASE}/ver-out-r{i}/encoder_hidden.npy")
    print(f"  r{i}: transcript={t[:16]}… encoder_hidden={e[:16]}…")

print()
print("record-leg vk compute dispatches + GPU busy:")
counts = {}
gpu_ns = 0
with open(f"{BASE}/profile-ver-record.jsonl") as f:
    for line in f:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = ev.get("name", "?")
        counts[name] = counts.get(name, 0) + 1
        if "period_ns" in ev:
            gpu_ns += int(ev["end"] - ev["start"])
        elif ev.get("ph") == "X" and "dur" in ev and ev.get("args", {}).get("device_ticks") is not None:
            gpu_ns += int(ev["args"]["device_ticks"])
total = sum(counts.values())
print(f"  events: {total}")
for name, c in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    {c:>6}  {name}")
