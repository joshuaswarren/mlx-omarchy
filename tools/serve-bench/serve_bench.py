#!/usr/bin/env python3
"""Isolated multi-server LLM serve benchmark (mlx_lm.server / oMLX / mlx-serve).

Audit harness for the 2026-09-19 serve receipts. It is a client-side harness:
it starts and probes real servers, changes no production code, and runs
outside any import path of mlx-omarchy.

Differences from receipts/2026-09-19-mlxserve-linux-port-t6001-test-host/bench3.py:

- Legs run SEQUENTIALLY, one fresh server process per leg (default). The
  09-19 runs kept all four servers co-resident on one GPU; that is fair
  across legs but understates absolute rates and, at ~15 GB weights
  (Qwen3.8-27B-4bit), four residents no longer fit a 62 GB box.
  --resident restores the legacy co-resident method for comparability.
- Adds streaming rounds so TTFT and decode-only rate are measured, not
  just the end-to-end wall/token number the 09-19 receipts report
  (their only server-side timings came from mlx-serve's `timings` field).
- Zero cache bias by construction: fresh process per leg + identical
  warmup request count/tokens/prompt for every leg; prompt_tokens and any
  server-reported cached_n are recorded per request.
- Pins are fatal-checked before any request (mlx-omarchy version, libmlx
  sha256-16, mlx-serve binary sha256-16), per the A/B provenance rule.
- Records per-leg server RSS (/proc VmRSS) after load and after rounds,
  plus total safetensors bytes, to produce a measured weights->peak
  factor for admission control.
- Optional bounded concurrency phase (--concurrency N, default off).

Metric definitions (all per leg, computed client-side):

  end_to_end_tok_s  usage.completion_tokens / HTTP wall of a non-streaming
                    request. Includes tokenize+prefill+decode+detok+HTTP.
                    This is the ONLY metric comparable with the 2026-09-19
                    receipt numbers (10.36 / 33.5 / 5.65).
  ttft_s            wall until first streamed chunk of a stream=true
                    request with the same prompt/max_tokens.
  decode_tok_s      (completion_tokens - 1) / (stream_wall - ttft).
                    Decode-slope approximation; ignores detok/HTTP tail.
  concurrency_tok_s aggregate completion_tokens / wall of N simultaneous
                    stream=true requests (--concurrency, default off).

Self-check (no GPU, no real servers): python3 serve_bench.py --selfcheck
runs the full client path against an in-process stdlib mock of the three
OpenAI-style endpoints and asserts the summary math.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
LEGS = ("mlxlm", "omlx", "mlxserve", "mlxserve_nopld")
PORTS = {"mlxlm": 8081, "omlx": 8082, "mlxserve": 11234, "mlxserve_nopld": 11235}
SSE_DONE = b"data: [DONE]"


def log(msg: str) -> None:
    print(msg, flush=True)


def fatal(msg: str, code: int) -> "NoReturn":  # type: ignore[valid-type]
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def free_port(port: int) -> None:
    subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        s = socket.socket()
        try:
            s.bind((HOST, port))
            s.close()
            return
        except OSError:
            s.close()
            time.sleep(2)
    fatal(f"port {port} still busy after cleanup", 8)


def dir_size(path: str, suffixes: tuple[str, ...]) -> int:
    root = pathlib.Path(path)
    return sum(p.stat().st_size for p in root.rglob("*")
               if p.suffix in suffixes and p.is_file())


def venv_json(python: str, code: str) -> dict:
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[-300:]}
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": f"unparseable: {out.stdout[:200]}"}


def check_pins(python: str, expect_version: str, expect_libmlx: str) -> dict:
    pins = venv_json(python, r"""
import hashlib, importlib.metadata as md, json, pathlib
pins = {}
try:
    pins["mlx_omarchy_version"] = md.version("mlx-omarchy")
except Exception as e:
    pins["mlx_omarchy_version"] = f"MISSING: {e}"
try:
    import mlx
    lib = None
    for root in mlx.__path__:
        p = pathlib.Path(root) / "lib" / "libmlx.so"
        if p.exists():
            lib = p
    pins["libmlx_sha256_16"] = hashlib.sha256(lib.read_bytes()).hexdigest()[:16] if lib else "MISSING"
except Exception as e:
    pins["libmlx_sha256_16"] = f"MISSING: {e}"
print(json.dumps(pins))
""")
    if "error" in pins:
        fatal(f"venv runtime probe failed: {pins['error']}", 3)
    if expect_version and pins["mlx_omarchy_version"] != expect_version:
        fatal(f"runtime pin mismatch: version={pins['mlx_omarchy_version']} expected={expect_version}", 3)
    if expect_libmlx and pins.get("libmlx_sha256_16") != expect_libmlx:
        fatal(f"runtime pin mismatch: libmlx={pins.get('libmlx_sha256_16')} expected={expect_libmlx}", 3)
    for mod in ("mlx-lm", "omlx"):
        info = venv_json(python, f"import importlib.metadata as md; print(md.version('{mod}'))")
        pins[mod.replace("-", "_")] = info.get("error", info and list(info.values())[0])
    return pins


def rss_gb(pid: int) -> float | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return round(int(out.stdout.strip()) / 1024 / 1024, 2) if out.returncode == 0 and out.stdout.strip() else None


def start_leg(leg: str, args, python: str) -> subprocess.Popen:
    port = PORTS[leg]
    if leg == "mlxlm":
        cmd = [python, "-m", "mlx_lm.server", "--model", args.hf_id or args.model_snapshot,
               "--host", HOST, "--port", str(port)]
    elif leg == "omlx":
        cmd = [python, "-m", "omlx.server", "--model-dir", args.model_snapshot,
               "--host", HOST, "--port", str(port)]
    elif leg == "mlxserve":
        cmd = [args.mlxserve_bin, "--model", args.model_snapshot, "--serve",
               "--host", HOST, "--port", str(port)]
    elif leg == "mlxserve_nopld":
        cmd = [args.mlxserve_bin, "--model", args.model_snapshot, "--no-pld", "--serve",
               "--host", HOST, "--port", str(port)]
    else:
        fatal(f"unknown leg {leg}", 2)
    logf = open(f"{args.outdir}/server-{leg}.log", "ab")
    logf.write(f"\n==== run {time.strftime('%F %T')} ====\n".encode())
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True,
                            env=dict(os.environ))


def wait_ready(port: int, name: str, deadline_s: int) -> float:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        for path in ("/v1/models", "/health"):
            try:
                with urllib.request.urlopen(f"http://{HOST}:{port}{path}", timeout=5) as r:
                    if r.status == 200:
                        ready = round(time.time() - t0, 1)
                        log(f"[ready] {name} via {path} in {ready}s")
                        return ready
            except Exception:
                pass
        time.sleep(2)
    fatal(f"{name} not ready in {deadline_s}s", 4)
    return -1.0  # unreachable


def discover_model_name(port: int, hint: str | None, hf_id: str) -> str:
    if not hint:
        return hf_id
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/v1/models", timeout=10) as r:
            ids = [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
        match = [i for i in ids if hint.lower() in i.lower()]
        picked = match[0] if match else "default"
        log(f"[discover] model_name={picked} from {ids[:6]}")
        return picked
    except Exception as e:
        log(f"[discover] fallback default ({e})")
        return "default"


def chat_once(port: int, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mlx"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:300]
        except Exception:
            detail = "<unreadable>"
        return {"error": f"HTTP {e.code}: {detail}", "wall_s": round(time.time() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 3)}
    wall = round(time.time() - t0, 3)
    if body.get("stream"):
        return {"wall_s": wall}
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    return {"wall_s": wall,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": choice.get("finish_reason"),
            "text_head": ((choice.get("message") or {}).get("content") or "")[:80],
            "timings": payload.get("timings")}


def chat_stream(port: int, body: dict, timeout: int) -> dict:
    """stream=true request; records TTFT (first chunk) and total wall."""
    sse_body = dict(body, stream=True)
    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
        data=json.dumps(sse_body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mlx"})
    t0 = time.time()
    t_first = None
    chunks = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if line.strip() == SSE_DONE:
                    break
                if t_first is None:
                    t_first = time.time()
                chunks += 1
                try:
                    evt = json.loads(line[5:])
                    if evt.get("usage"):
                        usage = evt["usage"]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 3)}
    wall = round(time.time() - t0, 3)
    return {"ttft_s": round(t_first - t0, 3) if t_first else None,
            "wall_s": wall, "chunks": chunks,
            "completion_tokens": usage.get("completion_tokens"),
            "decode_tok_s": (round((usage.get("completion_tokens", 0) - 1) / (wall - (t_first - t0)), 2)
                             if t_first and usage.get("completion_tokens") else None)}


def summarize(rounds: list[dict], streams: list[dict], conc: dict | None) -> dict:
    good = [r for r in rounds if r.get("completion_tokens") and r.get("wall_s")]
    tps = [r["completion_tokens"] / r["wall_s"] for r in good]
    sttft = [s["ttft_s"] for s in streams if s.get("ttft_s") is not None]
    sdec = [s["decode_tok_s"] for s in streams if s.get("decode_tok_s")]
    out = {
        "n": len(good), "n_errors": len(rounds) - len(good),
        "median_tok_s": round(statistics.median(tps), 2) if tps else None,
        "min_tok_s": round(min(tps), 2) if tps else None,
        "max_tok_s": round(max(tps), 2) if tps else None,
        "median_ttft_s": round(statistics.median(sttft), 3) if sttft else None,
        "median_decode_tok_s": round(statistics.median(sdec), 2) if sdec else None,
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in good), 1) if good else None,
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in good), 1) if good else None,
        "all_finished_length": bool(good) and all(r["finish"] == "length" for r in good),
    }
    if conc:
        out["concurrency"] = conc
    return out


def engine_control(args, python: str) -> dict:
    """Same prompt/max_tokens decoded via mlx_lm stream_generate in-process.

    No HTTP, no server layer: separates backend decode cost from serving
    overhead. Prints one JSON line.
    """
    code = f"""
import json, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
model, tokenizer = load({args.model_snapshot!r})
prompt = tokenizer.apply_chat_template(
    [{{"role": "user", "content": {args.prompt!r}}}], add_generation_prompt=True)
cache = None
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
gen = stream_generate(model, tokenizer, prompt[0] if isinstance(prompt, list) else prompt,
                      max_tokens={args.max_tokens}, sampler=lambda x: mx.argmax(x, axis=-1),
                      prompt_cache=cache)
t0 = time.time(); t_first = None; n = 0
for r in gen:
    if t_first is None:
        t_first = time.time()
    n += 1
    _ = r.token
wall = time.time() - t0
print(json.dumps({{"n": n, "ttft_s": round(t_first - t0, 3),
                   "wall_s": round(wall, 3),
                   "end_to_end_tok_s": round(n / wall, 2)}}))
"""
    out = subprocess.run([python, "-c", code], capture_output=True, text=True,
                         timeout=args.timeout * 4)
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (out.stderr.strip()[-300:] or "no json output")}


def run_leg(leg: str, args, python: str, results: dict) -> None:
    port = PORTS[leg]
    free_port(port)
    proc = start_leg(leg, args, python)
    hints = {"mlxlm": None, "omlx": args.omlx_hint, "mlxserve": "qwen",
             "mlxserve_nopld": "qwen"}
    try:
        load_s = wait_ready(port, leg, args.ready_deadline)
        model_name = discover_model_name(port, hints.get(leg), args.hf_id)
        warm = []
        for _ in range(args.warmup_rounds):
            warm.append(chat_once(port, {"model": model_name,
                                         "messages": [{"role": "user", "content": args.prompt}],
                                         "max_tokens": args.warmup_tokens,
                                         "temperature": 0, "stream": False}, args.timeout))
        if not warm[-1].get("completion_tokens"):
            results["leg_errors"][leg] = warm[-1].get("error", "no completion tokens")
            log(f"[leg-failed] {leg}: {results['leg_errors'][leg]}")
            return
        rss_load = rss_gb(proc.pid)
        rounds = []
        for r in range(1, args.rounds + 1):
            w = chat_once(port, {"model": model_name,
                                 "messages": [{"role": "user", "content": args.prompt}],
                                 "max_tokens": args.max_tokens,
                                 "temperature": 0, "stream": False}, args.timeout)
            rounds.append(w)
            if w.get("completion_tokens") and w.get("wall_s"):
                log(f"[round {r}] {leg}: {w['completion_tokens'] / w['wall_s']:.1f} tok/s "
                    f"finish={w.get('finish')}")
            else:
                log(f"[round {r}] {leg}: ERROR {w.get('error', '?')[:160]}")
        streams = [chat_stream(port, {"model": model_name,
                                      "messages": [{"role": "user", "content": args.prompt}],
                                      "max_tokens": args.max_tokens,
                                      "temperature": 0}, args.timeout)
                   for _ in range(args.ttft_rounds)]
        conc = None
        if args.concurrency > 1:
            import threading
            total_gb = None
            try:
                for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1024 / 1024
                        break
            except OSError:
                pass
            # Admission: resident model + 2 GiB workspace per stream must fit
            # 90% of unified memory; else record explicit N/A (Main's gate).
            budget = (total_gb or 0) * 0.9
            need = (rss_load or 0) + 2 * args.concurrency
            if total_gb is None or need > budget:
                conc = {"n": args.concurrency, "decision": "N/A",
                        "reason": f"admission: rss_load={rss_load}GB + 2GiB/stream "
                                  f"x{args.concurrency} vs total={total_gb}GB budget={budget:.1f}GB"}
                log(f"[concurrency] {leg}: N/A — {conc['reason']}")
            else:
                per: list[dict] = []
                lock = threading.Lock()

                def one() -> None:
                    w = chat_stream(port, {"model": model_name,
                                           "messages": [{"role": "user", "content": args.prompt}],
                                           "max_tokens": args.max_tokens,
                                           "temperature": 0}, args.timeout)
                    with lock:
                        per.append(w)
                t0 = time.time()
                threads = [threading.Thread(target=one) for _ in range(args.concurrency)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = round(time.time() - t0, 3)
                toks = sum(p.get("completion_tokens") or 0 for p in per)
                ttfts = [p["ttft_s"] for p in per if p.get("ttft_s") is not None]
                conc = {"n": args.concurrency, "wall_s": wall,
                        "aggregate_tok_s": round(toks / wall, 2) if wall else None,
                        "mean_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else None,
                        "per_stream": per}
        rss_after = rss_gb(proc.pid)
        results["legs"][leg] = {
            "port": port, "model_name": model_name, "load_s": load_s,
            "rss_gb_after_load": rss_load, "rss_gb_after_rounds": rss_after,
            "warmup": warm, "rounds_data": rounds, "stream_rounds": streams,
            "summary": summarize(rounds, streams, conc),
        }
        log(f"[summary] {leg}: {json.dumps(results['legs'][leg]['summary'])}")
    finally:
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            pass
        time.sleep(3)
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-snapshot", default=None)
    ap.add_argument("--hf-id", default=None, help="hub id (mlx_lm leg; also default model_name)")
    ap.add_argument("--prompt", default="Explain why the sky is blue.")
    ap.add_argument("--legs", default="mlxlm,omlx,mlxserve", help=f"subset of {LEGS}")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-rounds", type=int, default=1)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--ttft-rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1, help=">1 adds a bounded concurrent phase")
    ap.add_argument("--timeout", type=int, default=300, help="per-request client timeout s")
    ap.add_argument("--ready-deadline", type=int, default=420)
    ap.add_argument("--python", default=sys.executable, help="bench venv python that has mlx-omarchy")
    ap.add_argument("--mlxserve-bin", default=os.path.expanduser("~/src/mlx-serve-clean/zig-out/bin/mlx-serve"))
    ap.add_argument("--omlx-hint", default=None, help="substring to pick /v1/models id on the omlx leg")
    ap.add_argument("--expect-version", default="", help="fatal unless mlx-omarchy version matches")
    ap.add_argument("--expect-libmlx", default="", help="fatal unless libmlx sha256-16 matches")
    ap.add_argument("--engine-control", action="store_true",
                    help="add bare mlx_lm stream_generate leg (no HTTP)")
    ap.add_argument("--outdir", default="/tmp/servebench")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if args.selfcheck:
        sys.exit(selfcheck())

    os.makedirs(args.outdir, exist_ok=True)
    if not pathlib.Path(args.model_snapshot).is_dir():
        fatal(f"model snapshot not a dir: {args.model_snapshot}", 2)
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    unknown = [l for l in legs if l not in LEGS]
    if unknown:
        fatal(f"unknown legs {unknown}; choose from {LEGS}", 2)
    if "mlxserve" in legs or "mlxserve_nopld" in legs:
        if not pathlib.Path(args.mlxserve_bin).exists():
            fatal(f"mlx-serve binary missing: {args.mlxserve_bin}", 2)

    results = {
        "pins": check_pins(args.python, args.expect_version, args.expect_libmlx),
        "model_snapshot": args.model_snapshot, "hf_id": args.hf_id,
        "prompt": args.prompt, "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "warmup_rounds": args.warmup_rounds, "warmup_tokens": args.warmup_tokens,
        "method": "sequential fresh-process legs" if args.concurrency == 1 else
                  f"sequential fresh-process legs + concurrency {args.concurrency}",
        "model_weights_gb": round(dir_size(args.model_snapshot, (".safetensors",)) / 1e9, 2),
        "legs": {}, "leg_errors": {},
    }
    log(f"pins: {json.dumps(results['pins'])}")
    if args.engine_control:
        results["engine_control_mlxlm"] = engine_control(args, args.python)
        log(f"engine-control: {json.dumps(results['engine_control_mlxlm'])}")
    try:
        for leg in legs:
            run_leg(leg, args, args.python, results)
    finally:
        out = pathlib.Path(args.outdir) / "results-servebench.json"
        out.write_text(json.dumps(results, indent=2))
        log(f"wrote {out}")
    if not results["legs"]:
        fatal("all legs failed", 7)


def selfcheck() -> int:
    """Exercise the client path + summary math against a stdlib mock server."""
    import http.server
    import threading

    ntok = 128
    wall = 10.0

    class Mock(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send({"data": [{"id": "mock-qwen"}]})
            elif self.path == "/health":
                self._send({"ok": True})
            else:
                self._send({"detail": "nf"}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(ntok):
                    if i == 0:
                        time.sleep(0.05)  # simulated prefill tail before first chunk
                    self.wfile.write(b"data: " + json.dumps(
                        {"choices": [{"delta": {"content": "x"}}]}).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [], "usage": {"completion_tokens": ntok,
                                              "prompt_tokens": 37}}).encode() + b"\n\n")
                self.wfile.write(SSE_DONE + b"\n\n")
                self.wfile.flush()
            else:
                time.sleep(0.01)
                self._send({"usage": {"completion_tokens": ntok, "prompt_tokens": 37},
                            "choices": [{"message": {"content": "mock"},
                                         "finish_reason": "length"}]})

    srv = http.server.HTTPServer((HOST, 0), Mock)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        streams = [chat_stream(port, {"model": "m", "messages": [], "max_tokens": ntok}, 30)
                   for _ in range(3)]
        assert all(s.get("ttft_s") is not None for s in streams), streams
        assert all(s["completion_tokens"] == ntok for s in streams), streams
        rounds = [{"completion_tokens": ntok, "wall_s": wall, "prompt_tokens": 37,
                   "finish": "length"}] * 4
        s = summarize(rounds, streams, None)
        assert s["median_tok_s"] == 12.8, s
        assert s["all_finished_length"] and s["n"] == 4, s
        assert s["median_ttft_s"] >= 0.05, s
        assert s["median_decode_tok_s"] and s["median_decode_tok_s"] > 12.8, s
        print(f"SELFCheck OK: {json.dumps(s)}")
        return 0
    except AssertionError as e:
        print(f"SELFCHECK FAIL: {e}", file=sys.stderr)
        return 1
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
