#!/usr/bin/env python3
"""Paired A/B of MLX_OMARCHY_GATED_BARRIERS on the pinned Q4 decode legs.

Holds /tmp/m1-gpu.lock via the outer flock. Never unlinks it.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PY = "/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python"
WHEEL = (
    "/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/dist/"
    "mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl"
)
MODEL = "/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx"
LOCK = "/tmp/m1-gpu.lock"
OUT = Path("/tmp/gated-barriers-ab/results.json")
REPS = 5
PIN_SHORT = "7fd25a869ff21678"
PIN_CTX = "7da83f06ec9f001d"


def prompt_text(manifest, prompt_id):
    entry = manifest["prompts"][prompt_id]
    if "text" in entry:
        return entry["text"]
    if entry.get("template") == "numbered":
        parts = [entry["base"]] + [
            f"{entry['item']} Entry {i} of {entry['items']}."
            for i in range(1, entry["items"] + 1)
        ]
        return " ".join(parts)
    raise KeyError(prompt_id)


def power_state():
    rows = {}
    for name in (
        "macsmc-ac/online",
        "tps6598x-source-psy-0-0038/online",
        "tps6598x-source-psy-0-003f/online",
        "macsmc-battery/status",
    ):
        p = Path("/sys/class/power_supply") / name
        rows[name] = p.read_text().strip() if p.exists() else None
    return rows


def lock_inode():
    return os.stat(LOCK)[stat.ST_INO]


def nested_flock_n():
    r = subprocess.run(
        ["flock", "-n", LOCK, "-c", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    return r.returncode


def parse_json_line(stdout: str):
    for line in stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("no JSON result line")


def run_leg(prompt: str, gate: str) -> dict:
    env = os.environ.copy()
    env["MLX_DISABLE_COMPILE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["MLX_OMARCHY_GATED_BARRIERS"] = gate
    t0 = time.monotonic()
    r = subprocess.run(
        [
            PY,
            str(SCRIPTS / "bench_decode.py"),
            "--model",
            MODEL,
            "--prompt",
            prompt,
            "--tokens",
            "32",
            "--temp",
            "0.0",
            "--seed",
            "0",
            "--warmup-tokens",
            "4",
            "--wheel",
            WHEEL,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(SCRIPTS),
    )
    wall = time.monotonic() - t0
    rec = {
        "gate": gate,
        "rc": r.returncode,
        "wall_s": round(wall, 3),
        "stdout": r.stdout,
        "stderr": r.stderr[-4000:] if r.stderr else "",
    }
    if r.returncode == 0:
        rec["result"] = parse_json_line(r.stdout)
    return rec


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    return ys[n // 2]


def main():
    started = time.time()
    manifest = json.loads((SCRIPTS / "bench_matrix.json").read_text())
    short = prompt_text(manifest, "short")
    ctx = prompt_text(manifest, "ctx1024")
    if len(short.encode()) != 2 or len(ctx.encode()) != 4759:
        raise SystemExit(
            f"prompt size mismatch short={len(short.encode())} ctx={len(ctx.encode())}"
        )
    record = {
        "hostname": os.uname().nodename,
        "started_unix": int(started),
        "lock_inode": lock_inode(),
        "nested_flock_n_rc_while_held": nested_flock_n(),
        "power_before": power_state(),
        "python": PY,
        "wheel": WHEEL,
        "model": MODEL,
        "reps": REPS,
        "prompt_bytes": {"short": len(short.encode()), "ctx1024": len(ctx.encode())},
        "legs": [],
    }
    order = [
        ("short", short, PIN_SHORT),
        ("ctx1053", ctx, PIN_CTX),
    ]
    for rep in range(1, REPS + 1):
        for name, prompt, pin in order:
            for gate in ("0", "1"):
                rec = run_leg(prompt, gate)
                rec["rep"] = rep
                rec["leg"] = name
                rec["pin"] = pin
                if rec.get("result"):
                    rec["digest_ok"] = rec["result"]["ids_sha256_16"] == pin
                record["legs"].append(rec)
                got = (rec.get("result") or {}).get("ids_sha256_16")
                tps = (rec.get("result") or {}).get("decode_tps")
                print(
                    f"rep={rep} {name} gate={gate} rc={rec['rc']} "
                    f"digest={got} pin_ok={rec.get('digest_ok')} "
                    f"decode_tps={tps}",
                    flush=True,
                )
                if rec["rc"] != 0:
                    record["power_after"] = power_state()
                    record["finished_unix"] = int(time.time())
                    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    print(r.stderr if False else rec["stderr"][-2000:], file=sys.stderr)
                    raise SystemExit(f"leg failed: rep={rep} {name} gate={gate}")
    summary = {}
    for name, pin in (("short", PIN_SHORT), ("ctx1053", PIN_CTX)):
        summary[name] = {}
        for gate in ("0", "1"):
            rows = [
                x["result"]
                for x in record["legs"]
                if x["leg"] == name and x["gate"] == gate and x.get("result")
            ]
            dec = [r["decode_tps"] for r in rows]
            pref = [r["prefill_tps"] for r in rows]
            digests = [r["ids_sha256_16"] for r in rows]
            pts = [r["prompt_tokens"] for r in rows]
            summary[name][gate] = {
                "n": len(rows),
                "decode_tps": dec,
                "decode_median": median(dec),
                "prefill_tps": pref,
                "prefill_median": median(pref),
                "digests": digests,
                "all_digest_pin": all(d == pin for d in digests),
                "prompt_tokens": pts,
            }
        off = summary[name]["0"]["decode_median"]
        on = summary[name]["1"]["decode_median"]
        summary[name]["decode_delta_tok_s"] = (
            None if off is None or on is None else round(on - off, 4)
        )
        summary[name]["decode_on_wins"] = bool(on is not None and off is not None and on > off)
    record["summary"] = summary
    record["all_digests_ok"] = all(
        summary[n][g]["all_digest_pin"] for n in ("short", "ctx1053") for g in ("0", "1")
    )
    record["gated_wins_both_decode"] = bool(
        summary["short"]["decode_on_wins"] and summary["ctx1053"]["decode_on_wins"]
    )
    record["power_after"] = power_state()
    record["finished_unix"] = int(time.time())
    record["nested_flock_n_rc_end_still_held"] = nested_flock_n()
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "all_digests_ok": record["all_digests_ok"],
                      "gated_wins_both_decode": record["gated_wins_both_decode"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
