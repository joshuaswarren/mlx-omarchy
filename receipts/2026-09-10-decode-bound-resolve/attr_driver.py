#!/usr/bin/env python3
"""Arm-matrix driver for the decode-attribution discriminator.

Part 1 of receipts/2026-09-10-decode-bound-resolve. Runs attr_worker.py
under the release wheel (host-scale and GPU-scale knobs) and under the
ablation wheel (GPU-scale-down), interleaved, on a quiet machine.

Every leg runs EAGER (MLX_DISABLE_COMPILE=1) to match the canonical
baseline conditions. Release-wheel arms must reproduce the canonical
digest; ablation arms must flip it (that is their design). Prompts come
from the committed bench_matrix.json via its own template expansion, so
the bytes are exactly the canonical leg prompts.

No lock here: wrap the whole invocation in one top-level
`flock /tmp/m1-gpu.lock` per the project convention.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CANONICAL = {
    "short-decode-32": "7fd25a869ff21678",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}


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


ARMS = [
    ("base", "run", {}),
    ("spin1", "run", {"ATTR_SPIN_US": "1000"}),
    ("spin4", "run", {"ATTR_SPIN_US": "4000"}),
    ("gup2", "run", {"ATTR_GUP_K": "2"}),
    ("gup4", "run", {"ATTR_GUP_K": "4"}),
    ("gup6", "run", {"ATTR_GUP_K": "6"}),
    ("gup12", "run", {"ATTR_GUP_K": "12"}),
    ("ab_base", "ablate", {}),
    ("gemv", "ablate", {"MLX_OMARCHY_ABLATE": "gemv"}),
    ("ab_all", "ablate", {"MLX_OMARCHY_ABLATE": "all"}),
]

# workload id -> (default arm list, reps); prompt id and token count
# come from the manifest workload entry.
WORKLOADS = {
    "short-decode-32": (
        ["base", "spin1", "spin4", "gup2", "gup4", "gup6",
         "ab_base", "gemv", "ab_all"], 3),
    "longctx-1024-decode-32": (
        ["base", "spin4", "gup6", "gemv"], 2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--manifest", required=True,
                    help="committed scripts/bench_matrix.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--venv-run", required=True)
    ap.add_argument("--venv-ablate", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-wl", default=None,
                    help="comma list limiting workloads")
    ap.add_argument("--only-arms", default=None,
                    help="comma list limiting arms per workload")
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    only_wl = args.only_wl.split(",") if args.only_wl else None
    only_arms = args.only_arms.split(",") if args.only_arms else None

    print(f"quiet gate: {quiet_gate() if not args.skip_gate else 'skipped'}",
          flush=True)
    legs_file = out / "legs.ndjson"
    gate_fail = []
    n = 0
    for wl, (arms, reps) in WORKLOADS.items():
        if only_wl and wl not in only_wl:
            continue
        wentry = next(w for w in manifest["workloads"] if w["id"] == wl)
        prompt = prompt_text(manifest, wentry["prompt"])
        n_tok = int(wentry["tokens"])
        if only_arms:
            known = {a[0] for a in ARMS}
            arms = [a for a in arms if a in only_arms]
            arms += [a for a in only_arms if a in known and a not in arms]
        for rep in range(1, reps + 1):
            for arm in arms:
                _, venv, extra = next(a for a in ARMS if a[0] == arm)
                py = (args.venv_run if venv == "run" else args.venv_ablate)
                leg_out = out / f"{wl}-{arm}-r{rep}.json"
                env = dict(os.environ)
                env["MLX_DISABLE_COMPILE"] = "1"
                env.update(extra)
                la_pre = round(os.getloadavg()[0], 2)
                t0 = time.monotonic()
                proc = subprocess.run(
                    [py + "/bin/python", args.worker,
                     "--model", args.model, "--prompt", prompt,
                     "--tokens", str(n_tok), "--out", str(leg_out)],
                    env=env, capture_output=True, text=True, timeout=900)
                dur = round(time.monotonic() - t0, 1)
                la_post = round(os.getloadavg()[0], 2)
                if proc.returncode != 0:
                    print(f"LEG FAIL {wl}/{arm}/r{rep} rc={proc.returncode} "
                          f"{proc.stderr[-400:]}", flush=True)
                    continue
                res = json.loads(proc.stdout.strip().splitlines()[-1])
                res.update(arm=arm, rep=rep, workload=wl,
                           loadavg_pre=la_pre, loadavg_post=la_post,
                           wall_s=dur)
                expect = CANONICAL.get(wl)
                ablated = bool(extra.get("MLX_OMARCHY_ABLATE"))
                if expect:
                    if (not ablated and res["digest"] != expect) or \
                            (ablated and res["digest"] == expect):
                        msg = (f"GATE {wl}/{arm}/r{rep} digest "
                               f"{res['digest']} expected "
                               f"{'!=' if ablated else '=='} {expect}")
                        print(msg, flush=True)
                        gate_fail.append(msg)
                with legs_file.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                n += 1
                print(f"leg {n:3d} {wl} {arm:7s} r{rep} "
                      f"tok/s={res['decode_tok_s']:.1f} "
                      f"tok_ms={res['tok_ms']:.2f} "
                      f"tcpu={res['thread_cpu_ms_per_token']:.2f} "
                      f"pcpu={res['proc_cpu_ms_per_token']:.2f} "
                      f"gupw={res['gup_wall_ms_per_token']} "
                      f"digest={res['digest']} la={la_pre}->{la_post}",
                      flush=True)
    if gate_fail:
        print("DIGEST GATE FAILURES:")
        for m in gate_fail:
            print(" ", m)
        sys.exit(1)
    print(f"done: {n} legs, all digest gates passed")


if __name__ == "__main__":
    main()
