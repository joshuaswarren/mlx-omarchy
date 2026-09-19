#!/usr/bin/env python3
"""Serve-options bench on jw16: mlx_lm.server vs oMLX (ddalcu/mlx-serve is macOS-only, N/A here).

Single-stream, greedy, interleaved rounds on one exclusive GPU window.
Writes results.json + prints a summary. Runtime pins are fatal if mismatched.
"""
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error

PY = "/tmp/servebench/venv/bin/python"
import glob as _glob
MODEL_HF = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SNAPSHOT = _glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/*"))[0]
MODEL_DIR_B = SNAPSHOT
B_ID_HINT = "qwen2.5-7b"
PROMPT = "Explain why the sky is blue."
ROUNDS = int(os.environ.get("ROUNDS", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "128"))
EXPECT_LIBMLX_SHA = "df3d4e74c597956c"
EXPECT_VERSION = "0.32.3.dev202609190758+50eeb29"

procs = {}

LEGS = {
    "A_mlxlm": {
        "port": 8081,
        "model_name": MODEL_HF,
        "cmd": [PY, "-m", "mlx_lm.server", "--model", SNAPSHOT,
                "--host", "127.0.0.1", "--port", "8081"],
    },
    "B_omlx": {
        "port": 8082,
        "model_name": "ministral3-8b",
        "cmd": [PY, "-m", "omlx.server", "--model-dir", MODEL_DIR_B,
                "--host", "127.0.0.1", "--port", "8082"],
    },
}

def runtime_pins():
    import hashlib, pathlib
    import mlx
    ver = subprocess.run([PY, "-m", "pip", "show", "mlx-omarchy"],
                         capture_output=True, text=True).stdout
    version = next(l.split(": ", 1)[1] for l in ver.splitlines() if l.startswith("Version:"))
    libmlx = None
    for root in mlx.__path__:
        p = pathlib.Path(root) / "lib" / "libmlx.so"
        if p.exists():
            libmlx = p
    sha = hashlib.sha256(libmlx.read_bytes()).hexdigest()[:16] if libmlx else "MISSING"
    if sha != EXPECT_LIBMLX_SHA or version != EXPECT_VERSION:
        print(f"FATAL: runtime pin mismatch: {version=} {sha=}", file=sys.stderr)
        sys.exit(3)
    return {"mlx_omarchy_version": version, "libmlx_sha256_16": sha}

def wait_ready(port, name, deadline_s=420):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        for path in ("/v1/models", "/health"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                    if r.status == 200:
                        print(f"[ready] {name} via {path} in {time.time()-t0:.1f}s", flush=True)
                        return round(time.time() - t0, 1)
            except urllib.error.HTTPError:
                pass
            except Exception:
                pass
        time.sleep(2)
    print(f"FATAL: {name} not ready in {deadline_s}s", file=sys.stderr)
    sys.exit(4)

def http_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer mlx"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:500]
        except Exception:
            detail = "<unreadable>"
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def chat_once(port, max_tokens, model_name):
    body = {"model": model_name,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    t0 = time.time()
    payload, err = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    wall = round(time.time() - t0, 3)
    if err:
        return {"error": err, "wall_s": wall}
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    return {"wall_s": wall,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "finish": choice.get("finish_reason"),
            "text_head": ((choice.get("message") or {}).get("content") or "")[:80],
            "timings": payload.get("timings")}

def chat(port, max_tokens, name="", model_name="ministral3-8b"):
    """Chat with one retry on error (lazy-load transients); falls back to text
    completions for the A leg if chat keeps failing, recording which worked."""
    w = chat_once(port, max_tokens, model_name)
    if w.get("error"):
        print(f"[retry] {name} chat error: {w['error']}", flush=True)
        time.sleep(5)
        w = chat_once(port, max_tokens, model_name)
    if w.get("error") and name == "A_mlxlm":
        t0 = time.time()
        payload, err = http_json(f"http://127.0.0.1:{port}/v1/completions",
                                 {"model": model_name, "prompt": PROMPT,
                                  "max_tokens": max_tokens, "temperature": 0,
                                  "stream": False})
        wall = round(time.time() - t0, 3)
        if not err:
            usage = payload.get("usage") or {}
            choice = (payload.get("choices") or [{}])[0]
            w = {"wall_s": wall, "endpoint": "/v1/completions",
                 "completion_tokens": usage.get("completion_tokens"),
                 "prompt_tokens": usage.get("prompt_tokens"),
                 "finish": choice.get("finish_reason"),
                 "text_head": (choice.get("text") or "")[:80],
                 "timings": payload.get("timings")}
            print(f"[fallback] {name} served via /v1/completions", flush=True)
    return w

def free_port(port):
    """Kill whatever listens on the port (scoped: never matches our own
    shell/cmdline), then wait until the port binds."""
    subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        import socket
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return
        except OSError:
            s.close()
            time.sleep(2)
    print(f"FATAL: port {port} still busy after cleanup", file=sys.stderr)
    sys.exit(8)

def main():
    global procs
    pins = runtime_pins()
    print("pins:", pins, flush=True)
    procs = {}
    load_s = {}
    leg_failed = {}
    for leg in LEGS.values():
        free_port(leg["port"])
    try:
        for name, leg in LEGS.items():
            log = open(f"/tmp/servebench/server-{name}.log", "ab")
            log.write(f"\n==== run {time.strftime('%F %T')} ====\n".encode())
            log.flush()
            procs[name] = subprocess.Popen(leg["cmd"], stdout=log, stderr=log,
                                           start_new_session=True,
                                           env=dict(os.environ))
            load_s[name] = wait_ready(leg["port"], name)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{LEGS['B_omlx']['port']}/v1/models", timeout=10) as r:
                ids = [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
            match = [i for i in ids if B_ID_HINT in i.lower()]
            LEGS["B_omlx"]["model_name"] = match[0] if match else "default"
            print(f"[discover] B model_name={LEGS['B_omlx']['model_name']} "
                  f"from {ids[:6]}", flush=True)
        except Exception as e:
            LEGS["B_omlx"]["model_name"] = "default"
            print(f"[discover] B fallback default ({e})", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL server start: {e}", file=sys.stderr)
        sys.exit(5)

    results = {"pins": pins, "model_hf": MODEL_HF, "model_dir_b": MODEL_DIR_B,
               "prompt": PROMPT, "rounds": ROUNDS, "max_tokens": MAXTOK,
               "load_s": load_s, "warmup": {}, "rounds_data": {}}

    active = dict(LEGS)
    for name, leg in list(active.items()):
        w = chat(leg["port"], 16, name, leg["model_name"])
        results["warmup"][name] = w
        print(f"[warmup] {name}: {json.dumps(w)[:200]}", flush=True)
        if not w.get("completion_tokens"):
            leg_failed[name] = w.get("error", "no completion tokens")
            del active[name]
    if not active:
        print("FATAL: both legs failed warmup", file=sys.stderr)
        json.dump(results, open("/tmp/servebench/results.json", "w"), indent=2)
        sys.exit(6)

    for r in range(1, ROUNDS + 1):
        for name, leg in active.items():
            w = chat(leg["port"], MAXTOK, name, leg["model_name"])
            results["rounds_data"].setdefault(name, []).append(w)
            if w.get("completion_tokens") and w.get("wall_s"):
                tps = w["completion_tokens"] / w["wall_s"]
                print(f"[round {r}] {name}: {tps:.1f} tok/s "
                      f"({w['completion_tokens']} tok in {w['wall_s']}s) finish={w['finish']}",
                      flush=True)
            else:
                print(f"[round {r}] {name}: ERROR {w.get('error', '?')[:160]}", flush=True)

    summary = {}
    for name, rows in results["rounds_data"].items():
        good = [r for r in rows if r.get("completion_tokens") and r.get("wall_s")]
        if good:
            tps = [r["completion_tokens"] / r["wall_s"] for r in good]
            summary[name] = {
                "n": len(good), "n_errors": len(rows) - len(good),
                "median_tok_s": round(statistics.median(tps), 2),
                "min_tok_s": round(min(tps), 2), "max_tok_s": round(max(tps), 2),
                "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in good), 1),
                "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in good), 1),
                "endpoint": good[-1].get("endpoint", "/v1/chat/completions"),
                "all_finished_length": all(r["finish"] == "length" for r in good),
            }
    results["summary"] = summary
    results["leg_errors"] = leg_failed
    with open("/tmp/servebench/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== SUMMARY (single-stream, greedy, {MAXTOK} max_tokens, "
          f"{ROUNDS} interleaved rounds) ===", flush=True)
    for name, s in summary.items():
        print(f"{name}: median {s['median_tok_s']} tok/s (min {s['min_tok_s']}, "
              f"max {s['max_tok_s']}, n={s['n']}, errors={s['n_errors']}, "
              f"endpoint={s['endpoint']})", flush=True)
    for name, err in leg_failed.items():
        print(f"{name}: LEG FAILED — {str(err)[:200]}", flush=True)
    sys.exit(0 if summary else 7)

if __name__ == "__main__":
    rc = 1
    try:
        main()
    except SystemExit as e:
        rc = e.code or 0
        raise
    finally:
        for p in procs.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(3)
        for p in procs.values():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
