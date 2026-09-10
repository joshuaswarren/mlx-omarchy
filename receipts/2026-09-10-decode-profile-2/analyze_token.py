#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def kernel_names(path):
    names = []
    inside = False
    pattern = re.compile(r"^\s{2}(\w+),\s*$")
    for line in path.read_text().splitlines():
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside and "};" in line:
            break
        if inside and (match := pattern.match(line)):
            names.append(match.group(1))
    return names


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("markers", type=Path)
    ap.add_argument("--compute-h", type=Path, required=True)
    ap.add_argument("--token-window", type=int, default=8,
                    help="zero-based inter-token window")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = load_jsonl(args.profile)
    markers = load_jsonl(args.markers)
    meta = next(r for r in records if r.get("k") == "meta")
    submits = {r["s"]: r for r in records if r.get("k") == "s"}
    all_dispatches = [r for r in records if r.get("k") == "d" and "t0" in r]
    toks = [r for r in markers if r.get("p") == "tok"]
    i = args.token_window
    if not 0 <= i < len(toks) - 1:
        raise SystemExit(f"token window {i} absent; have {len(toks) - 1}")
    left, right = toks[i]["t"], toks[i + 1]["t"]
    selected = [d for d in all_dispatches
                if d["s"] in submits and left <= submits[d["s"]]["t"] < right]
    selected.sort(key=lambda d: d["t0"])
    if not selected:
        raise SystemExit("selected token window has no dispatches")

    names = kernel_names(args.compute_h)
    period = float(meta["period_ns"])
    mask = (1 << int(meta["valid_bits"])) - 1 if meta["valid_bits"] < 64 else None

    def delta(a, b):
        raw = b - a
        if mask is not None:
            raw &= mask
        return raw * period

    globally_ordered = sorted(all_dispatches, key=lambda d: d["t0"])
    previous = {id(cur): prev for prev, cur in zip(globally_ordered, globally_ordered[1:])}
    rows = []
    busy_ns = 0.0
    intra_ns = 0.0
    inter_ns = 0.0
    by_kernel = defaultdict(lambda: {"dispatches": 0, "gpu_ns": 0.0})
    prior_selected = None
    for sequence, dispatch in enumerate(selected, 1):
        gpu_ns = delta(dispatch["t0"], dispatch["t1"])
        prev = previous.get(id(dispatch))
        gap_ns = None if prev is None else delta(prev["t1"], dispatch["t0"])
        gap_class = "first-profile-dispatch" if prev is None else (
            "intra-submission" if prev["s"] == dispatch["s"] else "inter-submission")
        if prior_selected is not None:
            selected_gap = max(0.0, delta(prior_selected["t1"], dispatch["t0"]))
            if prior_selected["s"] == dispatch["s"]:
                intra_ns += selected_gap
            else:
                inter_ns += selected_gap
        prior_selected = dispatch
        name = names[dispatch["e"]] if dispatch["e"] < len(names) else f"kernel_{dispatch['e']}"
        busy_ns += gpu_ns
        by_kernel[name]["dispatches"] += 1
        by_kernel[name]["gpu_ns"] += gpu_ns
        rows.append({
            "sequence": sequence,
            "submission": dispatch["s"],
            "kernel": name,
            "shape": {"count": dispatch["n"],
                      "grid": [dispatch["gx"], dispatch["gy"], dispatch["gz"]],
                      "binding_ranges_bytes": [binding[2] for binding in dispatch.get("b", [])]},
            "gpu_us": round(gpu_ns / 1e3, 3),
            "preceding_gap_us": None if gap_ns is None else round(gap_ns / 1e3, 3),
            "preceding_gap_class": gap_class,
            "operation": dispatch.get("op"),
            "tape": bool(dispatch.get("tp")),
        })

    wall_ns = right - left
    residual_ns = wall_ns - busy_ns - intra_ns - inter_ns
    if residual_ns < -1e3:
        raise SystemExit(f"negative residual host time: {residual_ns} ns")
    residual_ns = max(0.0, residual_ns)
    ranked_kernels = [
        {"item": name, "kind": "kernel", "dispatches": values["dispatches"],
         "ms": round(values["gpu_ns"] / 1e6, 6)}
        for name, values in by_kernel.items()
    ]
    ranked = ranked_kernels + [
        {"item": "inter_submission_gaps", "kind": "gap", "ms": round(inter_ns / 1e6, 6)},
        {"item": "intra_submission_gaps", "kind": "gap", "ms": round(intra_ns / 1e6, 6)},
        {"item": "non_gpu_host_time", "kind": "host", "ms": round(residual_ns / 1e6, 6)},
    ]
    ranked.sort(key=lambda row: -row["ms"])
    out = {
        "profile": args.profile.name,
        "device": meta["device"],
        "token_window": {"zero_based": i, "from_marker_ns": left,
                         "to_marker_ns": right, "wall_ms": round(wall_ns / 1e6, 6)},
        "dispatches_per_token": len(selected),
        "submissions_per_token": len({d["s"] for d in selected}),
        "gpu_busy_fraction_of_wall": round(busy_ns / wall_ns, 8),
        "gpu_busy_fraction_of_gpu_span": round(
            busy_ns / delta(selected[0]["t0"], selected[-1]["t1"]), 8),
        "wall_clock_split_ms": {
            "gpu_busy": round(busy_ns / 1e6, 6),
            "intra_submission_gaps": round(intra_ns / 1e6, 6),
            "inter_submission_gaps": round(inter_ns / 1e6, 6),
            "non_gpu_host_time": round(residual_ns / 1e6, 6),
        },
        "ranked_costs": ranked,
        "dispatch_table": rows,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"TOKEN_PROFILE_WRITTEN {args.out} dispatches={len(selected)} "
          f"submissions={out['submissions_per_token']} wall_ms={wall_ns / 1e6:.3f}")


if __name__ == "__main__":
    main()
