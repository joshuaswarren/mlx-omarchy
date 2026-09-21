#!/usr/bin/env python3
"""CPU tests for the generation probe's emission lifecycle:

  A. length-stop: the consumer iterates until the finish token then
     breaks (exactly mlx_lm.server's loop) — exactly ONE event, n=128,
     finish_reason=length.
  B. early client close: the consumer breaks after 10 tokens (no finish)
     and closes the generator — exactly ONE partial event.
  C. double-emission guard: after the event, GC/close of the retained
     generator emits nothing further.
"""
import sys
import types

mx_stub = types.ModuleType("mlx.core")
mlx_stub = types.ModuleType("mlx")
mlx_stub.core = mx_stub
mlx_stub.__path__ = []
sys.modules.setdefault("mlx", mlx_stub)
sys.modules.setdefault("mlx.core", mx_stub)

import server_ids_probe as sip  # noqa: E402


class FakeResponse:
    def __init__(self, token, finish=None):
        self.token = token
        self.finish_reason = finish


def fake_stream(n, finish_at=None):
    def gen(*a, **k):
        for i in range(n):
            fr = "length" if (finish_at == "cap" and i == n - 1) else None
            yield FakeResponse(100 + i, fr)
    return gen


def run_case(server_break_at_finish, n, finish_at=None, early_break=None):
    events = []
    probe = sip.make_stream_probe(fake_stream(n, finish_at), events.append)
    yielded = 0
    for r in probe(model=None, tokenizer=None, prompt=[1], max_tokens=n):
        yielded += 1
        if early_break is not None and yielded >= early_break:
            break
        if server_break_at_finish and r.finish_reason is not None:
            break
    return yielded, len(events), (events[0] if events else None)


# Case A: length-stop, consumer breaks at finish (server behaviour)
y, ne, ev = run_case(True, 128, finish_at="cap")
assert y == 128 and ne == 1 and ev["n"] == 128 and ev["finish_reason"] == "length", (y, ne, ev)
print("CASE_A_OK one event at finish, n=128")

# Case B: early client close without finish
y, ne, ev = run_case(True, 128, finish_at=None, early_break=10)
assert y == 10 and ne == 1 and ev["n"] == 10, (y, ne, ev)
print("CASE_B_OK one partial event on early close, n=10")

# Case C: retained generator + GC emits nothing extra after the event
import gc
c_events = []
g = sip.make_stream_probe(fake_stream(5, finish_at=None), c_events.append)()
it = iter(g)
next(it)
g.close()
del g
gc.collect()
assert len(c_events) <= 1, c_events
print("CASE_C_OK no double emission on close/GC")

# Case D (Main gate): sequential generations — each flushes at ITS OWN
# finish; the last request's flush must not wait for any next reset.
ev2 = []
p2 = sip.make_stream_probe(fake_stream(128, finish_at="cap"), ev2.append)
g1 = p2(model=None)
for r in g1:
    if r.finish_reason is not None:
        break
g1.close()
assert len(ev2) == 1, ev2  # first generation flushed at its finish
g2 = p2(model=None)
for r in g2:
    if r.finish_reason is not None:
        break
g2.close()
assert len(ev2) == 2, ev2  # second generation flushed at its own finish
print("CASE_D_OK sequential generations emit one event each, no deferral")

print("EMISSION_LIFECYCLE_OK")
