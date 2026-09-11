#!/usr/bin/env python3
"""Native macOS Metal baseline runner for the canonical parity matrix.

Mirrors the protocol committed in
receipts/2026-09-10-main-parity-12-matrix (Linux): one full 6-leg warmup
matrix discarded, then measured repetitions per leg, each leg-run a FRESH
scripts/bench_decode.py process (pinned generation length, EOS suppressed,
greedy temp 0 seed 0, 4 warmup tokens inside bench_decode, prefill timed
separately, digest over the exact generated ids, MLX_DISABLE_COMPILE=1).

Contention gate: if WATCH_PID is set (e.g. the resident <resident-inference-service>
router leg), the watched process's cumulative CPU time is sampled around
every leg-run window; any delta > 0 marks that rep contended and it is
excluded from the medians (>= 3 clean reps required per leg).

Models are the pinned revisions from scripts/bench_matrix.json; the local
HF snapshot directory name must equal the pin or the run refuses.

Usage: WATCH_PID=<pid> run_native_matrix.py [reps] [outdir]
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DECODE = HERE / "bench_decode.py"
PROMPTS = json.loads((HERE / "prompts.json").read_text())

MODELS = {
    "q4": {"repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
           "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"},
    "bf16": {"repo": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
             "revision": "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"},
}
LEGS = {
    "q4_short": ("q4", "short", 32),
    "q4_long": ("q4", "long", 128),
    "q4_longctx": ("q4", "ctx1024", 32),
    "bf16_short": ("bf16", "short", 32),
    "bf16_long": ("bf16", "long", 128),
    "bf16_longctx": ("bf16", "ctx1024", 32),
}
EXPECTED_PROMPT_TOKENS = {"short": 30, "long": 262, "ctx1024": 1053}
# Native digests recorded in the committed Linux canonical verdict
# (receipts/2026-09-10-main-parity-12-matrix/verdict.md digest table).
REFERENCE_NATIVE_DIGESTS = {
    "q4_short": "7fd25a869ff21678",
    "q4_long": "254d73fd93164b98",
    "q4_longctx": "7da83f06ec9f001d",
    "bf16_short": "7fc0f968789b1882",
    "bf16_long": "407b7624ed1b3b29",
    "bf16_longctx": "ff502900d2a179a5",
}
REPS = None  # set in main() from argv; module must be importable
OUTDIR = None
WATCH_PID = os.environ.get("WATCH_PID", "").strip()
# Router heartbeat ticks (0.01 s CPU per window) are throughput-neutral:
# 16m1mbp att6 ticked reps 283.6-296.8 tok/s vs strictly-clean att2 band
# 282.5-303.9 tok/s on the same legs. Deltas >= 0.10 s are contended.
WATCH_THRESHOLD_S = 0.10


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def ps_cpu_seconds(pid):
    """Cumulative CPU time of pid in seconds (centisecond resolution)."""
    out = sh("ps", "-o", "time=", "-p", str(pid))
    if not out:
        return None
    parts = out.strip().split(":")
    try:
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    except ValueError:
        return None


def resolve_snapshot(entry):
    base = (Path.home() / ".cache/huggingface/hub"
            / ("models--" + entry["repo"].replace("/", "--"))
            / "snapshots" / entry["revision"])
    if not (base / "config.json").is_file():
        from huggingface_hub import snapshot_download
        base = Path(snapshot_download(entry["repo"],
                                      revision=entry["revision"]))
    if base.name != entry["revision"] or not (base / "config.json").is_file():
        sys.exit(f"REFUSING: {entry['repo']} snapshot is not pinned "
                 f"revision {entry['revision']} (got {base})")
    return str(base)


def gates():
    power = sh("pmset", "-g", "ps")
    if "AC Power" not in power:
        sys.exit(f"REFUSING: not on AC power: {power!r}")
    # Transient model-server probes come and go on these shared hosts
    # (hung `llama-server --version`, short python runs). Refuse only
    # when a standalone server persists across a ~1 min rescan loop.
    # Command-path match only: pgrep -f also matches environment text.
    refused = None
    for attempt in range(7):
        scan = subprocess.run(["ps", "-axo", "pid=,pcpu=,command="],
                              capture_output=True, text=True).stdout
        refused = [ln.strip()[:200] for ln in scan.splitlines()
                   if re.search(r"vllm|mlx_lm|ane-qwen|llama-server|"
                                r"llama-cli", ln[14:])]
        if not refused:
            break
        if attempt == 0:
            print("model-server matches seen; rescanning for persistence",
                  flush=True)
        time.sleep(10)
    if refused:
        sys.exit("REFUSING: standalone model servers persisted:\n"
                 + "\n".join(refused))
    # Count only. Command lines carry user paths, ports and service
    # inventories that must never land in a published receipt.
    resident = sum(1 for ln in scan.splitlines()
                   if re.search(r"ollama serve|[Oo]llama\.app|docling-serve",
                                ln[14:]))
    ops = sh("ollama", "ps")
    loaded = [ln for ln in ops.splitlines()[1:] if ln.strip()]
    if loaded:
        sys.exit("REFUSING: ollama holds loaded models "
                 "(ollama stop them first):\n" + "\n".join(loaded))
    return {"power": power,
            "gpu_holding_processes": loaded or "none",
            "standalone_servers": "none (persistence-scanned)",
            "resident_idle_service_count": resident,
            "router_watch_pid": WATCH_PID or None,
            "note": "resident services idle: no models loaded; any "
                    "watched resident service is CPU-watched per leg-run"}


def versions():
    code = (
        "import platform,json;"
        "import mlx.core as mx;"
        "import mlx_lm,transformers,numpy;"
        "d=mx.metal.device_info();"
        "d=dict(d) if hasattr(d,'items') else {'raw':str(d)};"
        "print(json.dumps({'python':platform.python_version(),"
        "'mlx':mx.__version__,'mlx_lm':mlx_lm.__version__,"
        "'transformers':transformers.__version__,'numpy':numpy.__version__,"
        "'metal':d}))"
    )
    return json.loads(sh(sys.executable, "-c", code))


def host_identity():
    gpu = sh("system_profiler", "SPDisplaysDataType")
    return {
        "hostname": sh("hostname"),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_cores": sh("sysctl", "-n", "hw.ncpu"),
        "perf_cores": sh("sysctl", "-n", "hw.perflevel0.logicalcpu"),
        "eff_cores": sh("sysctl", "-n", "hw.perflevel1.logicalcpu"),
        "mem_bytes": sh("sysctl", "-n", "hw.memsize"),
        "macos": sh("sw_vers", "-productVersion"),
        "macos_build": sh("sw_vers", "-buildVersion"),
        "arch": sh("uname", "-m"),
        "gpu_identity_lines": [ln.strip() for ln in gpu.splitlines()
                               if "Cores" in ln or "Chipset" in ln
                               or "Metal" in ln],
    }


def run_leg(model_path, prompt_text, tokens, log_path):
    watch_before = ps_cpu_seconds(WATCH_PID) if WATCH_PID else None
    env = dict(os.environ, MLX_DISABLE_COMPILE="1")
    p = subprocess.run(
        [sys.executable, str(BENCH_DECODE), "--model", model_path,
         "--prompt", prompt_text, "--tokens", str(tokens),
         "--temp", "0.0", "--seed", "0", "--warmup-tokens", "4"],
        capture_output=True, text=True, env=env)
    log_path.write_text(p.stdout + ("\n--stderr--\n" + p.stderr
                                    if p.stderr else ""))
    if p.returncode != 0:
        raise RuntimeError(f"bench_decode exited {p.returncode}; "
                           f"see {log_path}")
    result, prompt_tokens = None, None
    for ln in p.stdout.splitlines():
        if ln.startswith("{"):
            result = json.loads(ln)
        elif ln.startswith("prompt_tokens "):
            prompt_tokens = int(ln.split()[1])
    if result is None or prompt_tokens is None:
        raise RuntimeError(f"unparsable bench_decode output; see {log_path}")
    watch_after = ps_cpu_seconds(WATCH_PID) if WATCH_PID else None
    delta = (None if watch_before is None or watch_after is None
             else round(watch_after - watch_before, 2))
    if delta is not None and delta >= WATCH_THRESHOLD_S:
        print(f"CONTENDED leg-run: router pid {WATCH_PID} used "
              f"{delta}s CPU during the window", flush=True)
    return result, prompt_tokens, delta


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
def clean_rows(rows):
    """Reps fit for medians: watch delta < threshold AND decode within
    70% of the best-band (upper-half) median. Decode throughput on a
    quiet GPU is deterministic per leg (clean spread +/-3%), so a rep
    far below the best band marks contention even when the watched pid
    is idle - e.g. bf16_short rep12 hit 15.7 tok/s with a 0.04 s watch
    delta while an unwatched process held the GPU."""
    tps = sorted(r["decode_tps"] for r in rows)
    upper = tps[len(tps) // 2:]
    best = median(upper)
    return [r for r in rows
            if (r["watch_pid_cpu_delta_s"] or 0) < WATCH_THRESHOLD_S
            and r["decode_tps"] >= 0.70 * best]


def stats(rows, key):
    if len(rows) < 3:
        raise RuntimeError(f"only {len(rows)} clean reps; need >= 3")
    return {"median": round(median([r[key] for r in rows]), 3),
            "min": round(min(r[key] for r in rows), 3),
            "max": round(max(r[key] for r in rows), 3)}


def do_matrix(paths, tag, rundir):
    rundir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for leg, (mid, pid, gen) in LEGS.items():
        result, ptoks, watch_delta = run_leg(paths[mid], PROMPTS[pid], gen,
                                             rundir / f"{leg}.log")
        expect = EXPECTED_PROMPT_TOKENS[pid]
        if ptoks != expect:
            raise RuntimeError(
                f"{leg}: prompt_tokens {ptoks} != expected {expect}; "
                "prompt or template drift - refusing to record")
        if result["generated"] != gen:
            raise RuntimeError(f"{leg}: generated {result['generated']} "
                               f"!= pinned {gen}")
        rows[leg] = {"model": mid, "prompt_id": pid,
                     "watch_pid_cpu_delta_s": watch_delta, **result,
                     "prompt_tokens_asserted": expect}
        print(tag, leg, json.dumps(rows[leg], sort_keys=True), flush=True)
    return rows

def main():
    global REPS, OUTDIR
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    OUTDIR = (Path(sys.argv[2]) if len(sys.argv) > 2
              else HERE / "native-baseline")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    identity = host_identity()
    vers = versions()
    gate_state = gates()
    print("HOST", json.dumps(identity, sort_keys=True), flush=True)
    print("VERSIONS", json.dumps(vers, sort_keys=True), flush=True)
    print("GATES", json.dumps(gate_state), flush=True)
    paths = {mid: resolve_snapshot(e) for mid, e in MODELS.items()}
    print("SNAPSHOTS", json.dumps(paths, sort_keys=True), flush=True)

    warm_rows = do_matrix(paths, "WARMUP(discarded)", OUTDIR / "warmup")
    (OUTDIR / "warmup" / "warmup-matrix.json").write_text(
        json.dumps(warm_rows, indent=1, sort_keys=True) + "\n")

    legs = {leg: [] for leg in LEGS}
    for rep in range(1, REPS + 1):
        repdir = OUTDIR / "runs" / f"rep{rep:02d}"
        rows = do_matrix(paths, f"REP{rep}", repdir)
        for leg, row in rows.items():
            (repdir / f"{leg}.json").write_text(
                json.dumps(row, indent=1, sort_keys=True) + "\n")
            legs[leg].append(row)

    summary = {}
    for leg, rows in legs.items():
        clean = clean_rows(rows)
        digests = {r["ids_sha256_16"] for r in rows}
        if len(digests) != 1:
            raise RuntimeError(f"{leg}: digests diverged across reps: "
                               f"{digests}")
        summary[leg] = {
            "model": rows[0]["model"],
            "prompt_id": rows[0]["prompt_id"],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "generated_tokens": rows[0]["generated"],
            "reps": REPS,
            "clean_reps": len(clean),
            "digest": digests.pop(),
            "digest_matches_reference_native":
                rows[0]["ids_sha256_16"] == REFERENCE_NATIVE_DIGESTS[leg],
            "decode_tok_s": stats(clean, "decode_tps"),
            "prefill_tok_s": stats(clean, "prefill_tps"),
            "all_rows": rows,
            "contended_reps_excluded":
                [i + 1 for i, r in enumerate(rows) if r not in clean],
        }
    out = {"schema": "native-macos-metal-baseline/1",
           "protocol": {"warmup": "full 6-leg matrix discarded",
                        "reps": REPS,
                        "engine": "scripts/bench_decode.py (fresh process "
                                  "per leg-run)",
                        "generation": "greedy temp 0 seed 0, EOS suppressed, "
                                      "pinned generated counts 32/128/32, "
                                      "4 warmup tokens per leg",
                        "env": "MLX_DISABLE_COMPILE=1",
                        "contention": "per leg-run: WATCH_PID cumulative-"
                                      "CPU delta >= 0.10s, or decode_tps "
                                      "< 70% of the best-band (upper-half "
                                      "median) = contended, excluded "
                                      "from medians; >= 3 clean reps "
                                      "required; 0.01s heartbeat ticks "
                                      "measured throughput-neutral",
                        "gate": gate_state},
           "host": identity, "versions": vers,
           "models": MODELS, "snapshots": paths,
           "reference_native_digests": REFERENCE_NATIVE_DIGESTS,
           "legs": summary, "runs": legs}
    final = OUTDIR / ("native-baseline-"
                      + identity["hostname"].split(".")[0] + ".json")
    final.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("FINAL", final, flush=True)


if __name__ == "__main__":
    main()
