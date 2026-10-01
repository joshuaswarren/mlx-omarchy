#!/usr/bin/env python3
"""Jw16DecodeNorm: build the per-class census table from a W1 out dir.
usage: norm_census_table.py OUTDIR
Reads d{32,64,128}-reason.err ([barrier-reason] totals), d128-dag.err
([dag] line counts, [kv-plan] refusal lines), profile-analyze.txt
(per-kernel count/mean/median/share table), and prints:
  - barrier decisions/token from the d64->d128 slope (5 passes each)
  - dispatches/token from the dag dump (1 pass: lines / decode evals)
  - the ranked class table: count/token x (12.2 us boundary + kernel us)
    with the macOS chain-cost counterpart where one exists.
macOS chain costs (Jw16DecodeGap window 2, chain_costs_decode.py, quiet):
  dependent add 4.77 us, rms_norm 4.58, silu 6.81, q4 GEMV 2048^2 14.24,
  6144x2048 33.98, up+down pair 61.6, norm+GEMV 20.9.
"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])


def reason_totals(p):
    txt = p.read_text(errors="replace") if p.exists() else ""
    m = re.findall(
        r"\[barrier-reason\] calls=(\d+) none=(\d+) raw=(\d+) waw=(\d+) war=(\d+)", txt
    )
    if not m:
        return None
    c, n, r, waw, war = (sum(int(x[i]) for x in m) for i in range(5))
    return dict(calls=c, none=n, raw=r, waw=waw, war=war)


tot = {}
for n in (32, 64, 128):
    tot[n] = reason_totals(out / f"d{n}-reason.err")
    print(f"d{n}-reason:", tot[n])

if tot.get(64) and tot.get(128):
    d_calls = tot[128]["calls"] - tot[64]["calls"]
    d_raw = tot[128]["raw"] - tot[64]["raw"]
    d_none = tot[128]["none"] - tot[64]["none"]
    print(
        f"slope/token (64 tokens x 5 passes): calls={d_calls / 320:.1f}"
        f" raw={d_raw / 320:.1f} none={d_none / 320:.1f}"
    )

dag_err = out / "d128-dag.err"
if dag_err.exists():
    lines = dag_err.read_text(errors="replace").splitlines()
    dag = sum(1 for l in lines if "[dag]" in l)
    kvref = sum(1 for l in lines if "[kv-plan]" in l)
    print(f"d128-dag: dag_lines={dag} kv_plan_lines={kvref}")

# per-kernel table from profile-analyze.txt: rows like
# <name> count=.. total_us=.. mean_us=.. median_us=.. share=..% (best-effort parse)
pa = out / "profile-analyze.txt"
if not pa.exists():
    sys.exit("no profile-analyze.txt; run profile_analyze.py first")
txt = pa.read_text(errors="replace")
# Try JSON blocks first, else fall back to the text table.
try:
    start = txt.index("{")
    data = json.loads(txt[start: txt.rindex("}") + 1])
    kernels = data.get("kernels", data)
except Exception:
    kernels = None
rows = []
if isinstance(kernels, dict):
    it = kernels.items()
elif isinstance(kernels, list):
    it = [
        (
            k.get("name", "?"),
            k,
        )
        for k in kernels
    ]
else:
    it = []
for name, k in it:
    try:
        cnt = float(k.get("count", 0))
        if cnt <= 0:
            continue
        med = float(k.get("median_us", k.get("median", 0)))
        mean = float(k.get("mean_us", k.get("mean", 0)))
        share = k.get("share", "")
        rows.append((name, cnt, med, mean, share))
    except Exception:
        continue
if not rows:
    print("could not parse profile table; raw head:")
    print("\n".join(txt.splitlines()[:60]))
    sys.exit(0)
# tokens of the profiled run: 5 passes x 128 - 5 (inter-token intervals)
TOKS = 5 * 128
macos = {
    "rms": 4.58,
    "add": 4.77,
    "silu": 6.81,
    "gemv2048": 14.24,
    "gemv6144": 33.98,
}
print("\nclass table (profiled shares; boundary 12.2 us/dispatch):")
print(f"{'class':44s} {'cnt/tok':>8s} {'med_us':>8s} {'us/tok':>9s} {'bnd_us':>9s} {'total':>9s}")
table = []
for name, cnt, med, mean, share in rows:
    cpt = cnt / TOKS
    us_tok = mean * cnt / TOKS
    bnd = cpt * 12.2
    table.append((us_tok + bnd, name, cpt, med, us_tok, bnd))
table.sort(reverse=True)
for tot_us, name, cpt, med, us_tok, bnd in table:
    print(f"{name[:44]:44s} {cpt:8.2f} {med:8.1f} {us_tok:9.1f} {bnd:9.1f} {tot_us:9.1f}")
print(f"TOTAL us/token (boundary+kernel, profiled): {sum(t[0] for t in table):.1f}")
