#!/usr/bin/env python3
import argparse
import json
import statistics
import re


def kernel_names(path):
    names = []
    inside = False
    for line in open(path, encoding="utf-8"):
        if "enum class ComputeKernel" in line:
            inside = True
        elif inside and line.strip() == "};":
            break
        elif inside:
            match = re.match(r"^  (\w+),$", line.rstrip())
            if match:
                names.append(match.group(1))
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--compute-h", required=True)
    parser.add_argument("--mode", choices=("qmm", "dense"), required=True)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()

    names = kernel_names(args.compute_h)
    events = [json.loads(line) for line in open(args.profile) if line.strip()]
    meta = next(event for event in events if event.get("k") == "meta")
    dispatches = [event for event in events if event.get("k") == "d" and "t0" in event]
    target = "QmmPrefillCoopmatF16" if args.mode == "qmm" else "MatmulF32Coopmat"
    slots_ns = []
    for current, following in zip(dispatches, dispatches[1:]):
        name = names[current["e"]] if current["e"] < len(names) else ""
        if name == target and current["s"] == following["s"]:
            slots_ns.append((following["t0"] - current["t0"]) * meta["period_ns"])
    if not slots_ns:
        raise SystemExit(f"no followed {target} dispatches in {args.profile}")

    slots_ns = slots_ns[args.warmups:]
    median_ms = statistics.median(slots_ns) / 1e6
    flops = 2 * args.m * args.k * args.n
    result = {
        "kernel": target,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "samples_ms": [value / 1e6 for value in slots_ns],
        "median_ms": median_ms,
        "tflops": flops / (median_ms * 1e9),
    }
    if args.mode == "qmm":
        # Unique packed Q4 weights plus one f16 scale and bias per group-64.
        weight_bytes = args.k * args.n * (0.5 + 4 / 64)
        requested_weight_bytes = weight_bytes * ((args.m + 31) // 32)
        result["unique_weight_mb"] = weight_bytes / 1e6
        result["requested_weight_mb"] = requested_weight_bytes / 1e6
        result["achieved_weight_gb_s"] = requested_weight_bytes / (median_ms * 1e6)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
