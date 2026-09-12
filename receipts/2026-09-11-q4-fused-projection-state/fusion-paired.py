#!/usr/bin/env python3
"""Paired fusion ON/OFF legs for the Q4 fused-projection state receipt.

One release wheel (current main), two sides differing ONLY by
MLX_OMARCHY_FUSED_GEMV: off = per-node QuantizedMatmul+Add path,
on = the fused q/k/v + gate/up + o + down GEMV groups (default since
wave/DecodeFusion). Rep 1 is a discarded warmup; reps 2..N are measured,
sides alternating order every rep. Every leg digest must equal its
canonical pin on BOTH sides - the off side matching the pin is the
bit-identity evidence for the fused path on this tree; a mismatch is
fatal (docs/parity-id-policy.md rule 2).

usage: fusion-paired.py REPS OUT_DIR
Run at the repo root (cwd = checkout) under the GPU lock.
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "fusion-paired")
HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
CANON = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
# receipts/native-baseline-2026-09-06/native-2026-09-06-nocompile-summary.json
NATIVE_DECODE_TOK_S = {
    "short-decode-32": 150.84,
    "long-decode-128": 146.57,
    "longctx-1024-decode-32": 140.25,
}
SIDES = {"off": "0", "on": "1"}

wheels = sorted((ROOT / "dist").glob("mlx_omarchy-*.whl"))
assert len(wheels) == 1, wheels
wheel = wheels[0]
py = ROOT / ".venv-accept/bin/python"
assert py.is_file(), py
OUT.mkdir(exist_ok=True)

summary = {
    "schema": "mlx-omarchy/fusion-paired/1",
    "wheel": str(wheel),
    "wheel_sha256": subprocess.run(
        ["sha256sum", str(wheel)], capture_output=True, text=True,
        check=True).stdout.split()[0],
    "commit": subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"], capture_output=True,
        text=True, check=True).stdout.strip(),
    "reps_requested": REPS, "warmup_rep": 1,
    "sides": SIDES, "pins": CANON, "native": NATIVE_DECODE_TOK_S,
    "runs": [], "legs": {},
}


def run_side(side: str, rep: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
    env.update(HF_HUB_OFFLINE="1", MLX_DISABLE_COMPILE="1",
               MLX_OMARCHY_FUSED_GEMV=SIDES[side],
               MESA_SHADER_CACHE_DISABLE="true")
    out_json = OUT / f"rep{rep}-{side}.json"
    log = OUT / f"rep{rep}-{side}.log"
    cmd = [str(py), "scripts/bench_matrix.py", "--mode", "run",
           "--manifest", str(HERE / "manifest-q4fp.json"),
           "--python", str(py), "--wheel", str(wheel),
           "--expect-pins", f"qwen25-0.5b-4bit={PIN}",
           "--host-label", "jwm1-q4fp-window",
           "--timeout", "300", "--out", str(out_json)]
    with open(log, "w") as lf:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                       stderr=subprocess.STDOUT, check=True, timeout=420)
    data = json.loads(out_json.read_text())
    assert data["clean_check"]["status"] == "clean", (rep, side, "contention")
    legs = {l["workload_id"]: l for l in data["legs"]
            if l["status"] == "measured"}
    assert set(legs) == set(CANON), (rep, side, sorted(legs))
    result = {}
    for wid, leg in legs.items():
        digest = leg["metrics"]["generated_ids_sha256_16"]
        assert digest == CANON[wid], (
            rep, side, wid, digest, CANON[wid],
            "DIGEST MOVED - fatal under docs/parity-id-policy.md rule 2")
        result[wid] = {
            "decode_tok_s": leg["metrics"]["decode_tok_s"],
            "prefill_tok_s": leg["metrics"].get("prefill_tok_s"),
            "prompt_tokens": leg["metrics"]["prompt_tokens"],
            "digest": digest,
        }
    return result


for rep in range(1, REPS + 1):
    order = list(SIDES)
    if rep % 2 == 0:
        order.reverse()
    for side in order:
        res = run_side(side, rep)
        summary["runs"].append(
            {"rep": rep, "side": side, "warmup": rep == 1, "legs": res})
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"rep {rep} {side}: "
              + " ".join(f"{w}={r['decode_tok_s']:.2f}" for w, r in
                         sorted(res.items())), flush=True)

for side in SIDES:
    for wid in CANON:
        runs = [r["legs"][wid]["decode_tok_s"]
                for r in summary["runs"]
                if r["side"] == side and not r["warmup"]]
        if not runs:
            continue
        summary["legs"].setdefault(wid, {})[side] = {
            "measured_reps": len(runs),
            "decode_tok_s_samples": runs,
            "decode_tok_s_median": statistics.median(runs),
            "fraction_of_native": statistics.median(runs)
            / NATIVE_DECODE_TOK_S[wid],
        }
for wid, sides in summary["legs"].items():
    if "off" in sides and "on" in sides:
        sides["on_over_off"] = (
            sides["on"]["decode_tok_s_median"]
            / sides["off"]["decode_tok_s_median"] - 1.0)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["legs"], indent=1))
