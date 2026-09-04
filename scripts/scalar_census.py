#!/usr/bin/env python3
"""count==1 dispatch attribution over an MLX_OMARCHY_GPU_PROFILE stream.

Per decode token: total dispatches, dispatches with n==1 (single element),
bucketed by kernel enum and operation code. Also n==1 outside decode windows
(prefill/finalize) so the decode-only target is explicit.

Usage: sf_census.py PROFILE.jsonl --compute-h compute.h [--markers m.jsonl]
"""
import argparse, json, re, sys
from collections import Counter

# Operation-code maps, mirrored from overlay/mlx/backend/omarchy/primitives.cpp.
FLOAT_OPS = [
    "Add", "Multiply", "Divide", "Maximum", "Exp", "Sigmoid", "Square",
    "Sqrt", "Rsqrt", "Subtract", "Negative", "Abs", "Floor", "Ceil", "Log",
    "Log2", "Log1p", "Sign", "Round", "Tan", "Sin", "Cos", "Expm1", "Sinh",
    "Cosh", "Rsqrt?", "Minimum", "Atan", "Asin", "Acos", "Atanh", "Asinh",
    "Acosh", "LogAddExp", "Erf", "ErfInv", "Power", "Kron?", "?", "?", "?",
    "?", "?", "?", "?", "?",
]
CMP_OPS = ["Equal", "GreaterEqual", "Greater", "Less", "LessEqual", "NotEqual"]


def kernel_names(header_path):
    names, pat, inside = [], re.compile(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$"), False
    with open(header_path, encoding="utf-8") as fh:
        for line in fh:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                if "}" in line:
                    break
                m = pat.match(line)
                if m and m.group(1) != "Count":
                    names.append(m.group(1))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--compute-h", required=True)
    ap.add_argument("--markers")
    args = ap.parse_args()
    names = kernel_names(args.compute_h)

    sub_t = {}
    for line in open(args.profile, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("k") == "s":
            sub_t[ev["s"]] = ev["t"]

    toks = []
    if args.markers:
        for line in open(args.markers, encoding="utf-8"):
            ev = json.loads(line)
            if ev["p"] in ("decode_start", "tok", "decode_done"):
                toks.append(ev)
    starts = [e["t"] for e in toks if e["p"] == "decode_start"]
    tok_ts = [e["t"] for e in toks if e["p"] == "tok"]
    bounds = []
    if starts:
        bounds.append((0, starts[0], tok_ts[0] if tok_ts else None))
    for i, t in enumerate(tok_ts):
        hi = tok_ts[i + 1] if i + 1 < len(tok_ts) else None
        bounds.append((i + 1, t, hi))

    def token_of(ev):
        t = sub_t.get(ev["s"], 0)
        for idx, lo, hi in bounds:
            if t >= lo and (hi is None or t < hi):
                return idx
        return None

    per_tok_total, per_tok_scalar = Counter(), Counter()
    scalar_buckets = Counter()
    outside_scalar = Counter()
    scalar_seq = []
    for line in open(args.profile, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("k") != "d":
            continue
        kname = names[ev["e"]] if ev["e"] < len(names) else f"e{ev['e']}"
        if ev["n"] == 1:
            op = ev["op"]
            opname = op
            if kname.startswith("Elementwise") or kname in ("FillF32",):
                opname = FLOAT_OPS[op] if op < len(FLOAT_OPS) else op
            elif kname.startswith("Compare"):
                opname = CMP_OPS[op] if op < len(CMP_OPS) else op
            b = f"{kname}/op{op}({opname})"
            t = token_of(ev)
            if t is None:
                outside_scalar[b] += 1
            else:
                scalar_buckets[b] += 1
                per_tok_scalar[t] += 1
                scalar_seq.append((t, b, ev["s"]))
        t = token_of(ev)
        if t is not None:
            per_tok_total[t] += 1

    toks_seen = sorted(per_tok_total)
    if toks_seen:
        vals = sorted(per_tok_total[t] for t in toks_seen)
        med = vals[len(vals) // 2]
        svals = sorted(per_tok_scalar.get(t, 0) for t in toks_seen)
        print(f"decode tokens: {len(toks_seen)}")
        print(f"dispatches/token: median={med}")
        print(f"count==1 dispatches/token: median={svals[len(svals)//2]} "
              f"min={svals[0]} max={svals[-1]} sum={sum(svals)}")
    print("\n== count==1 inside decode, by kernel/op ==")
    for b, c in scalar_buckets.most_common():
        print(f"  {b:44s} {c}")
    print("\n== count==1 outside decode windows (prefill/finalize), by kernel/op ==")
    for b, c in outside_scalar.most_common():
        print(f"  {b:44s} {c}")
    if scalar_seq:
        print("\n== first count==1 events in decode (token, kernel/op) ==")
        for t, b, s in scalar_seq[:40]:
            print(f"  token {t}  {b}")


if __name__ == "__main__":
    main()
