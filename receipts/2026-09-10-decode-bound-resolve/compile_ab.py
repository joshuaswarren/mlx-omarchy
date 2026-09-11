#!/usr/bin/env python3
"""Paired compiled-vs-eager Q4 bench driver (part 2).

receipts/2026-09-10-decode-bound-resolve. The canonical protocol pins
MLX_DISABLE_COMPILE=1 (scripts/bench_matrix.json), so compiled-tape
decode speed has never been measured on this backend even though
compiled tapes are the default for Q4 on Apple GPUs. This driver runs
the canonical engine (scripts/bench_decode.py from the committed wheel
tree) per leg, alternating eager/compiled within each repetition, and
gates eager legs against the canonical digests. Compiled legs record
whether the generated-id digest holds.

One driver state per invocation (fork or stock Mesa); the caller
arranges the driver swap and the single top-level flock.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

CANONICAL = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}

DECODE_RE = re.compile(r"decode ([0-9.]+) tok/s over (\d+) tokens")
PREFILL_RE = re.compile(r"prefill ([0-9.]+)s")
IDS_RE = re.compile(r"generated_ids sha256:([0-9a-f]+) n=(\d+)")
PTOK_RE = re.compile(r"prompt_tokens (\d+)")
PROV_RE = re.compile(r"provenance: (.+)")


def prompt_text(manifest, prompt_id):
    """Verbatim copy of scripts/bench_matrix.py:prompt_text."""
    entry = manifest["prompts"][prompt_id]
    if "text" in entry:
        return entry["text"]
    if entry.get("template") == "numbered":
        parts = [entry["base"]] + [
            f"{entry['item']} Entry {i} of {entry['items']}."
            for i in range(1, entry["items"] + 1)]
        return " ".join(parts)
    raise KeyError(f"prompt {prompt_id}: unsupported prompt entry")


def quiet_gate(max_wait_s=900):
    t0 = time.monotonic()
    ok = 0
    samples = []
    while time.monotonic() - t0 < max_wait_s:
        la = os.getloadavg()[0]
        samples.append(round(la, 2))
        ok = ok + 1 if la < 1.0 else 0
        if ok >= 3:
            return samples
        time.sleep(4)
    raise SystemExit(f"machine never went quiet: {samples}")


def run_leg(python, bench, model, prompt, n_tok, wheel, compiled,
            leg_out, timeout=900):
    env = dict(os.environ)
    if compiled:
        env.pop("MLX_DISABLE_COMPILE", None)
    else:
        env["MLX_DISABLE_COMPILE"] = "1"
    proc = subprocess.run(
        [python, bench, "--model", model, "--prompt", prompt,
         "--tokens", str(n_tok), "--wheel", wheel],
        env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return {"rc": proc.returncode, "stderr_tail": proc.stderr[-500:]}
    out = proc.stdout
    leg = {"rc": 0}
    m = DECODE_RE.search(out)
    leg["decode_tok_s"] = float(m.group(1)) if m else None
    m = PREFILL_RE.search(out)
    leg["prefill_s"] = float(m.group(1)) if m else None
    m = IDS_RE.search(out)
    leg["digest"] = m.group(1) if m else None
    leg["n_generated"] = int(m.group(2)) if m else None
    m = PTOK_RE.search(out)
    leg["prompt_tokens"] = int(m.group(1)) if m else None
    m = PROV_RE.search(out)
    leg["provenance"] = m.group(1).strip() if m else None
    for line in out.splitlines():
        if line.startswith('{"engine"'):
            leg["result"] = json.loads(line)
    if leg.get("prefill_s") and leg.get("prompt_tokens"):
        leg["prefill_tps"] = round(
            leg["prompt_tokens"] / leg["prefill_s"], 3)
    Path(leg_out).write_text(out)
    return leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True,
                    help="committed wheel tree: holds scripts/bench_decode.py "
                         "and scripts/bench_matrix.json")
    ap.add_argument("--python", required=True)
    ap.add_argument("--wheel", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()
    bench = str(Path(args.bench_dir) / "scripts" / "bench_decode.py")
    manifest = json.loads(
        (Path(args.bench_dir) / "scripts" / "bench_matrix.json").read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    print(f"quiet gate: {quiet_gate() if not args.skip_gate else 'skipped'}",
          flush=True)
    legs_file = out / "legs.ndjson"
    gate_fail = []
    n = 0
    for wl in ["short-decode-32", "long-decode-128",
               "longctx-1024-decode-32"]:
        wentry = next(w for w in manifest["workloads"] if w["id"] == wl)
        prompt = prompt_text(manifest, wentry["prompt"])
        n_tok = int(wentry["tokens"])
        for rep in range(1, args.reps + 1):
            for compiled in (False, True):
                mode = "compiled" if compiled else "eager"
                la_pre = round(os.getloadavg()[0], 2)
                t0 = time.monotonic()
                leg = run_leg(args.python, bench, args.model, prompt,
                              n_tok, args.wheel, compiled,
                              str(raw / f"{wl}-{mode}-r{rep}.txt"))
                leg.update(workload=wl, mode=mode, rep=rep,
                           loadavg_pre=la_pre,
                           wall_s=round(time.monotonic() - t0, 1))
                if leg["rc"] != 0:
                    print(f"LEG FAIL {wl}/{mode}/r{rep}: "
                          f"{leg.get('stderr_tail','')}", flush=True)
                else:
                    expect = CANONICAL[wl]
                    leg["digest_matches_canonical"] = \
                        leg["digest"] == expect
                    if not compiled and not leg["digest_matches_canonical"]:
                        msg = (f"GATE {wl} eager r{rep} digest "
                               f"{leg['digest']} != {expect}")
                        print(msg, flush=True)
                        gate_fail.append(msg)
                with legs_file.open("a") as f:
                    f.write(json.dumps(leg) + "\n")
                n += 1
                print(f"leg {n:3d} {wl:24s} {mode:8s} r{rep} "
                      f"decode={leg.get('decode_tok_s')} "
                      f"prefill_tps={leg.get('prefill_tps')} "
                      f"digest={leg.get('digest')} "
                      f"canon={leg.get('digest_matches_canonical')} "
                      f"la={la_pre} {leg.get('wall_s')}s", flush=True)
    if gate_fail:
        print("EAGER DIGEST GATE FAILURES:")
        for m in gate_fail:
            print(" ", m)
        sys.exit(1)

    print("\nsummary (medians)")
    rows = [json.loads(l) for l in legs_file.read_text().splitlines() if l]
    for wl in CANONICAL:
        for mode in ("eager", "compiled"):
            sel = [r for r in rows
                   if r["workload"] == wl and r["mode"] == mode and r["rc"] == 0]
            if not sel:
                continue
            holds = sum(1 for r in sel if r["digest_matches_canonical"])
            print(f"{wl:24s} {mode:9s} n={len(sel)} "
                  f"decode={statistics.median(r['decode_tok_s'] for r in sel):.3f} "
                  f"tok/s prefill={statistics.median(r['prefill_tps'] for r in sel):.1f} "
                  f"tok/s digest_canonical={holds}/{len(sel)}")
    print(f"done: {n} legs")


if __name__ == "__main__":
    main()
