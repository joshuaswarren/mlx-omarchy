#!/usr/bin/env python3
import argparse
import json
import re
import statistics


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
    parser.add_argument("--target", required=True)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()

    names = kernel_names(args.compute_h)
    events = [json.loads(line) for line in open(args.profile) if line.strip()]
    meta = next(event for event in events if event.get("k") == "meta")
    dispatches = [event for event in events if event.get("k") == "d" and "t0" in event]
    slots_ns = []
    for current, following in zip(dispatches, dispatches[1:]):
        name = names[current["e"]] if current["e"] < len(names) else ""
        if name == args.target and current["s"] == following["s"]:
            slots_ns.append((following["t0"] - current["t0"]) * meta["period_ns"])
    if len(slots_ns) <= args.warmups:
        raise SystemExit(
            f"expected more than {args.warmups} followed {args.target} dispatches; "
            f"found {len(slots_ns)}")

    samples_ms = [value / 1e6 for value in slots_ns[args.warmups:]]
    median_ms = statistics.median(samples_ms)
    flops = 2 * args.m * args.k * args.n
    weight_bytes = args.k * args.n * (0.5 + 4 / 64)
    requested_weight_bytes = weight_bytes * ((args.m + 31) // 32)
    print(json.dumps({
        "kernel": args.target,
        "device": meta["device"],
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "tflops": flops / (median_ms * 1e9),
        "unique_weight_mb": weight_bytes / 1e6,
        "requested_weight_mb": requested_weight_bytes / 1e6,
        "achieved_weight_gb_s": requested_weight_bytes / (median_ms * 1e6),
    }, indent=2))


if __name__ == "__main__":
    main()
