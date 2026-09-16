import json, statistics
def battery(tag, base):
    rows = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load", "tdt_decode", "detokenize"]
    data = {}
    for i in range(1, 7):
        rep = json.load(open(f"{base}/{'ver' if tag=='cut' else 'base-e2e'}-out-r{i}/e2e-report.json"))
        per = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
        per["total"] = rep["timing"]["total_pipeline_ms"]
        per["ane_exec"] = rep["ane"]["exec_ms"]
        per["status"] = rep["status"]
        seq = rep["layers"]["layer_6_decoder_sequence"]
        per["emissions"] = seq.get("actual_emissions")
        per["prefix"] = seq.get("matching_prefix_length")
        per["bounds"] = rep["layers"]["layer_5_encoder"]["all_bounds_pass"]
        per["submits"] = rep["ane"]["submissions"]
        per["timeouts"] = rep["ane"]["timeouts"]
        data[i] = per
    enc = [data[i]["encoder_ane"] for i in sorted(data)]
    tot = [data[i]["total"] for i in sorted(data)]
    print(f"[{tag}] encoder_ane runs: " + " ".join(f"{e:.1f}" for e in enc))
    print(f"[{tag}] encoder_ane median r1-r6: {statistics.median(enc):.3f}  total median: {statistics.median(tot):.3f}")
    for i in sorted(data):
        d = data[i]
        assert d["status"] == "match" and d["emissions"] == 104 and d["prefix"] == 104 and d["bounds"] and d["submits"] == 1 and d["timeouts"] == 0, (tag, i, d)
    print(f"[{tag}] all 6 runs: match, 104/104, prefix 104, bounds PASS, 1 submission, 0 timeouts")
battery("base", "/var/tmp/enc-gpubusy")
battery("cut", "/var/tmp/enc-gpubusy")
