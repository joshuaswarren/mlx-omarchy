#!/usr/bin/env python3
"""Build verdict.json for the Q4 GEMV bandwidth receipt from the screen
directories, the paired-legs summary, and the test logs in this folder."""
import json
import re
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = ["q4_896x896", "q4_896x128", "q4_896x4864", "q4_4864x896", "q4_896x151936"]
# Q4 decode GEMV dispatches per token for Qwen2.5-0.5B-Instruct-4bit
# (24 layers: q/o at 896x896, k/v at 896x128, gate/up at 896x4864, down
# at 4864x896, and the tied lm_head), from the 4bit profile of
# receipts/2026-09-09-timestamp-isolation.
PER_TOKEN = {"q4_896x896": 48, "q4_896x128": 48, "q4_896x4864": 48, "q4_4864x896": 24, "q4_896x151936": 1}


def load(side, variant="default"):
    r = json.loads((HERE / "screen" / side / "bandwidth.json").read_text())
    return r["version"], r["variants"][variant]


def table(base_side, cand_side, cand_variant="default"):
    bv, b = load(base_side)
    cv, c = load(cand_side, cand_variant)
    rows = {}
    for s in SHAPES:
        rows[s] = {
            "bytes": b[s]["bytes"], "workgroups": {"baseline": b[s]["workgroups"], "candidate": c[s]["workgroups"]},
            "dispatches_sampled": {"baseline": b[s]["dispatches"], "candidate": c[s]["dispatches"]},
            "p50_us": {"baseline": round(b[s]["p50_us"], 1), "candidate": round(c[s]["p50_us"], 1)},
            "GBps_p50": {"baseline": round(b[s]["gbps_p50"], 1), "candidate": round(c[s]["gbps_p50"], 1)},
            "candidate_over_baseline_GBps": round(c[s]["gbps_p50"] / b[s]["gbps_p50"], 3),
            "per_token_dispatches": PER_TOKEN[s],
        }
    per_token = {k: round(sum(PER_TOKEN[s] * v[s]["p50_us"] for s in SHAPES) / 1e3, 2) for k, v in (("baseline", b), ("candidate", c))}
    extras = {k: {"p50_us": round(v["p50_us"], 1), "GBps_p50": round(v["gbps_p50"], 1), "min_us": round(v["min_us"], 1)}
              for k, v in c.items() if k.startswith("ceiling") or k.startswith("floor")}
    return {"baseline_version": bv, "candidate_version": cv, "shapes": rows,
            "gemv_ms_per_token_from_p50": per_token, "device_reference": extras}


def variants(side):
    r = json.loads((HERE / "screen" / side / "bandwidth.json").read_text())
    out = {}
    for v, d in r["variants"].items():
        out[v] = {s[3:]: [round(d[s]["p50_us"], 1), round(d[s]["gbps_p50"], 1)] for s in SHAPES if s in d}
    return {"version": r["version"], "p50_us_and_GBps": out}


def identity():
    """Bit identity of every non-diagnostic variant against the main
    (de65554) diagnostics wheel on the same inputs. The worker changed
    its input generation at screen 6 (24 weight copies) so each
    generation compares against its own baseline run."""
    groups = {
        "screens_1_to_5": ("gb-screen-base", ["gb-screen-cand", "gb-screen-cand2", "gb-screen3-cand", "gb-screen4-cand", "gb-screen5-cand"]),
        "screen_6": ("gb-screen6-base", ["gb-screen6-cand"]),
        "curve": ("gb-curve-base", ["gb-curve-cand"]),
    }
    # Recorded before the screening-only variant outputs were deleted to
    # keep the receipt small (compare run on 2026-09-09, this host):
    out = {"before_trim": {
        "screens_1_to_5": {"compared_outputs": 36, "result": "ALL_IDENTICAL"},
        "screen_6": {"compared_outputs": 6, "result": "ALL_IDENTICAL"},
        "curve": {"compared_outputs": 1, "result": "ALL_IDENTICAL"}}}
    for name, (base, cands) in groups.items():
        others = sorted(p for c in cands for p in (HERE / "screen" / c).glob("*.npz") if "alu" not in p.name and "mem" not in p.name)
        res = subprocess.run(["python3", str(HERE / "kernel_bandwidth.py"), "compare", str(HERE / "screen" / base / "default.npz"), *map(str, others)],
                             capture_output=True, text=True)
        out[name] = {"reference": f"screen/{base}/default.npz", "compared_outputs": len(others),
                     "shapes_per_output": 9, "result": res.stdout.strip().splitlines()[-1]}
    return out


def tests(folder):
    out = {}
    for log in sorted(folder.glob("omarchy_*_tests.log")):
        text = log.read_text()
        m = re.search(r"assertions:\s+(\d+) \|\s+(\d+) passed \|\s+(\d+) failed", text)
        out[log.stem] = {"assertions": int(m.group(1)), "passed": int(m.group(2)), "failed": int(m.group(3)),
                         "status": "SUCCESS" if "Status: SUCCESS!" in text else "FAIL"} if m else {"status": "no summary"}
    commit = folder / "source-commit.txt"
    if commit.exists():
        out["source_commit"] = commit.read_text().strip()
    return out


def legs():
    p = HERE / "legs" / "paired-summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {"repetitions": s["repetitions"], "sides": s["sides"], "legs": {}}
    for row in s["results"]:
        leg = {}
        for name, e in row["sides"].items():
            leg[name] = {"decode_tok_s_samples": e["decode_tok_s"]["samples"],
                         "decode_tok_s_median": round(e["decode_tok_s"]["median"], 2),
                         "prefill_tok_s_median": round(e["prefill_tok_s"]["median"], 2),
                         "digests": e["digests"]}
            for k in e:
                if k.startswith("decode_over_") or k.startswith("prefill_over_"):
                    leg[name][k] = round(e[k], 4)
        leg["ids_identical_across_sides"] = len({tuple(e["digests"]) for e in row["sides"].values()}) == 1 and \
            all(len(e["digests"]) == 1 for e in row["sides"].values())
        out["legs"][row["leg_id"]] = leg
    out["ids_identical_all_legs"] = all(l["ids_identical_across_sides"] for l in out["legs"].values())
    return out


if __name__ == "__main__":
    verdict = {
        "topic": "Q4 decode GEMV (qmm_vec.comp QMM_VEC_Q4_WORD) bandwidth on the M1 under Honeykrisp",
        "branch": "wave/GemvBandwidth",
        "streaming_table": table("gb-screen6-base", "gb-screen6-cand"),
        "isolated_dispatch_table": table("gb-screen3-base", "gb-screen3-cand"),
        "screen_variants_isolated": variants("gb-screen3-cand"),
        "screen_variants_streaming": variants("gb-screen6-cand"),
        "diagnostic_variants_isolated": variants("gb-screen4-cand"),
        "bit_identity_micro": identity(),
        "tests": {"llvmpipe": tests(HERE / "local-llvmpipe"), "m1": tests(HERE / "m1-tests")},
        "paired_legs": legs(),
    }
    (HERE / "verdict-data.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict["streaming_table"]["gemv_ms_per_token_from_p50"]))
    if verdict["paired_legs"]:
        for leg, d in verdict["paired_legs"]["legs"].items():
            print(leg, {n: (v["decode_tok_s_median"], v.get("decode_over_base")) for n, v in d.items() if isinstance(v, dict)}, d["ids_identical_across_sides"])
