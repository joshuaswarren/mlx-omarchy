"""Collect battery E2E report fields: gates + stage walls + medians.
Usage: collect-fold.py <battery-out-root>
"""
import glob
import json
import statistics
import sys

root = sys.argv[1].rstrip("/")
stages = {}
disp = {}
rows = []
for r in range(1, 7):
    fs = glob.glob(f"{root}/out-{r}/e2e-report*.json")
    if not fs:
        print(f"r{r}: NO REPORT")
        continue
    d = json.load(open(fs[0]))
    gates = {}
    seq = d.get("layers", {}).get("layer_6_decoder_sequence", {})
    enc = d.get("layers", {}).get("layer_5_encoder", {})
    gates.update(
        status=d.get("status"),
        prefix=seq.get("matching_prefix_length"),
        emissions=seq.get("actual_emissions"),
        native=seq.get("native_emissions"),
        tokens_match=seq.get("tokens_match"),
        frames_match=seq.get("frame_indices_match"),
        bounds_pass=enc.get("all_bounds_pass"),
        control=d.get("execution", {}).get("control"),
        fallback=d.get("execution", {}).get("tdt_fallback_reason"),
        cpu_events=d.get("execution", {}).get("cpu_tensor_events"),
        ane_submissions=d.get("ane", {}).get("submissions"),
        ane_timeouts=d.get("ane", {}).get("timeouts"),
        ane_exec_ms=round(d.get("ane", {}).get("exec_ms") or 0, 1),
    )
    print(f"r{r}:", json.dumps(gates))
    for e in d.get("stages") or []:
        if isinstance(e, dict) and "stage" in e and "wall_ms" in e:
            stages.setdefault(e["stage"], []).append(e["wall_ms"])
        if isinstance(e, dict) and e.get("stage") == "encoder_ane":
            g = e.get("gpu_counter_delta") or {}
            for k in ("vk_compute_dispatches", "gpu_primitive_dispatches",
                      "vk_submissions"):
                disp.setdefault(k, []).append(g.get(k))
    t = d.get("timing", {}).get("total_pipeline_ms")
    if isinstance(t, (int, float)):
        rows.append(t)

for k, v in sorted(stages.items()):
    med = statistics.median(v[1:]) if len(v) > 1 else v[0]
    print(f"stage {k}: median_r2_r6={round(med, 1)} all={[round(x, 1) for x in v]}")
for k, v in sorted(disp.items()):
    print(f"encoder_ane {k}: {v}")
if len(rows) > 1:
    print(f"total_pipeline: median_r2_r6={round(statistics.median(rows[1:]), 1)} "
          f"all={[round(x, 1) for x in rows]}")
