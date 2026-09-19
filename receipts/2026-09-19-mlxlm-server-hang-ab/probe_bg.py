# Probe 2: isolate BatchGenerator vs thread as the stall trigger.
import os, threading, time, traceback
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache

MODEL = "/tmp/ab-models/qwen25-05b"
NTOK = 24
PROMPT = "Say hi in one word."

model, tok = load(MODEL)
main_stream = mx.default_stream(mx.default_device())
print(f"[setup] main stream = {main_stream}", flush=True)

def bg_run(label, stream=None):
    t0 = time.time()
    cache = make_prompt_cache(model)
    bg = BatchGenerator(
        model,
        prefill_step_size=512,
        stream=stream,
    )
    prompt = tok.encode(PROMPT)
    bg.insert_segments([[prompt]], [NTOK], [cache], [prompt])
    n = 0
    while n < NTOK:
        prompt_responses, gen_responses = bg.next()
        for r in gen_responses:
            n += 1
            if n == 1:
                print(f"[{label}] first-token {time.time()-t0:.2f}s", flush=True)
            if n >= NTOK:
                break
    dt = time.time() - t0
    print(f"[{label}] RESULT ok tokens={n} total={dt:.2f}s tok/s={n/dt:.2f}", flush=True)
    return "ok"

def probe(name, fn):
    out = {}
    def body():
        try:
            print(f"[{name}] thread stream = {mx.default_stream(mx.default_device())}", flush=True)
            out["r"] = fn(name)
        except Exception as e:
            out["r"] = f"error: {e!r}"
            print(f"[{name}] THREAD-ERROR {e!r}", flush=True)
            traceback.print_exc()
    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(120)
    if t.is_alive():
        print(f"[{name}] RESULT hang >120s", flush=True)
        return "hang"
    return out.get("r", "no-result")

verdict = {}
print("=== PROBE bg-main (BatchGenerator, main thread, default stream) ===", flush=True)
try:
    verdict["bg-main"] = bg_run("bg-main", None)
except Exception as e:
    verdict["bg-main"] = f"error: {e!r}"
    print(f"[bg-main] MAIN-ERROR {e!r}", flush=True)
    traceback.print_exc()

print("=== PROBE bg-thread (BatchGenerator, bare thread, no explicit stream) ===", flush=True)
verdict["bg-thread"] = probe("bg-thread", lambda n: bg_run(n, None))

print("=== PROBE bg-thread-stream (BatchGenerator, bare thread, thread default stream) ===", flush=True)
def bg_thread_stream(name):
    s = mx.default_stream(mx.default_device())
    print(f"[{name}] using explicit stream {s}", flush=True)
    return bg_run(name, s)
verdict["bg-thread-stream"] = probe("bg-thread-stream", bg_thread_stream)

print(f"VERDICT {verdict}", flush=True)
if any(v in ("hang",) or str(v).startswith("error") for v in verdict.values()):
    os._exit(7)
