# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Wire-level checks for the resident ANE worker client.

A scripted fake resident speaks both wire protocols (relay and
relay-bypass) over real pipes: no ANE hardware, no libane, no GPU lock.
The fakes validate the exact frame shape on the way in and return
deterministic transforms of the payload bytes on the way out, so the
tests prove payload bytes cross intact in both directions -- the
property the 104/104 pins depend on -- through the zero-copy staging
paths (streamed writes, socket-to-buffer reads).
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS / "coreml"))

from ane_resident import ResidentAneWorker, ResidentWorkerError

# Payload sizes straddle the client's 1 MiB read chunks and the 64 KiB
# splice chunks a real pump uses, so chunk-boundary and leftover-bytes
# paths are exercised, not the lucky single-chunk case.
SIZES = [0, 4096, (1 << 20) - 7, (1 << 20) + 3, 3 * (1 << 20) + 11]


FAKE_RESIDENT = r"""
import hashlib
import sys

mode = sys.argv[1]
stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_line():
    line = stdin.readline()
    if not line:
        raise SystemExit(3)
    return line.decode().rstrip("\n")


def read_exact(count):
    chunks = []
    got = 0
    while got < count:
        chunk = stdin.read(count - got)
        if not chunk:
            raise SystemExit(3)
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def emit_out(name, payload):
    stdout.write(f"out {name} {len(payload)}\n".encode())
    stdout.write(payload)
    stdout.flush()


def transform(name, payload):
    # Deterministic, size-preserving, position-sensitive: every byte of
    # the frame must have crossed the pipe to be reproduced.
    key = hashlib.sha256(name.encode()).digest()
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(payload))


# Startup banner: one bundle report, then the mode's readiness line.
print("resident bundle=island-test index=0 name=toy programs=1 "
      "driver_abi=1 graph=deadbeef", flush=True)
if mode == "relay":
    print("resident loaded pid=1 detail=\"loaded 1 programs\"", flush=True)
else:
    print("relay-bypass ready pid=1 deadline_ms=1 detail=loaded 1", flush=True)
if mode == "stall":
    # Takes requests, never answers: exercises the client's read guard.
    while stdin.read(1 << 16):
        pass
    import time
    time.sleep(600)
if mode != "relay":
    while True:
        line = read_line()
        if line == "close":
            print("released", flush=True)
            raise SystemExit(0)
        assert line.startswith("submit "), line
        inputs = []
        while True:
            frame = read_line()
            if frame == "run":
                break
            kind, name, length = frame.split(" ")
            assert kind == "in", frame
            inputs.append((name, read_exact(int(length))))
        for name, payload in inputs:
            # Echo every input back byte-transformed: the test proves
            # the exact request bytes crossed, not just the length.
            stdout.write(b"iter\n")
            emit_out(name, transform(name, payload))
        print("done", flush=True)
    raise SystemExit(0)

opened = False
rounds = 0
while True:
    line = read_line()
    if line == "quit":
        print("resident released programs=1", flush=True)
        raise SystemExit(0)
    if line.startswith("batch "):
        opened = True
        print(f"batch opened deadline={line.split()[1]}", flush=True)
        continue
    if line == "batch-end":
        opened = False
        print(f"batch closed rounds={rounds}", flush=True)
        continue
    tokens = line.split()
    assert tokens[0] == "submit", line
    emits = tokens[tokens.index("--emit") + 1::2]
    inline = {}
    payload_order = []
    for i, token in enumerate(tokens):
        if token == "--inline":
            name, length = tokens[i + 1].split("=")
            inline[name] = int(length)
            payload_order.append(name)
    for name in payload_order:
        read_exact(inline[name])
    rounds += 1
    for name in emits:
        # The relay path has no request bytes to echo proof from;
        # emit a fixed digest payload of the announced length.
        length = 64
        emit_out(name, hashlib.sha256(name.encode()).digest())
    print("job status=0 bundle=%s elapsed_ms=1 iterations=1 "
          "input_bytes=0 output_bytes=64 stage_ms=0 save_ms=0" % tokens[1],
          flush=True)
"""


def start_fake(mode: int, tmp_path: Path, bundles: dict) -> ResidentAneWorker:
    script = tmp_path / "fake_resident.py"
    script.write_text(FAKE_RESIDENT)
    session = ResidentAneWorker.__new__(ResidentAneWorker)
    session.worker = Path(sys.executable)
    session.libane = tmp_path / "libane.so"
    session.bundles = bundles
    session.scratch = tmp_path
    session.deadline_ms = 20000
    session.iterations = 1
    session.relay_bypass = mode == "bypass"
    session.submissions = 0
    session.batch_opens = 0
    session.batch_rounds = 0
    session.worker_starts = 0
    session.bundle_loads = 0
    session.device_program_loads = 0
    session.timeouts = 0
    session.input_bytes = 0
    session.output_bytes = 0
    session.exec_ns = 0
    session.start_ns = 0
    session.close_ns = 0
    session.log = []
    session._process = None
    session._stderr_path = tmp_path / "fake.stderr"
    session._stderr = session._stderr_path.open("wb")
    session._banner = []
    session._inbox = bytearray()
    session._batch_until = None
    session._bypass_batch_base = 0
    argv = [sys.executable, str(script), mode]
    session._process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=session._stderr,
    )
    session.worker_starts += 1
    for _ in bundles:
        line = session._readline("bundle report")
        assert line.startswith("resident bundle="), line
    if mode == "relay":
        line = session._readline("load report")
        assert line.startswith("resident loaded "), line
    else:
        line = session._readline("ready")
        assert line.startswith("relay-bypass ready"), line
    return session


def payload_for(size: int, seed: int) -> bytes:
    """Incompressible-ish, position-sensitive filler of exactly ``size``."""
    if size == 0:
        return b""
    block = bytes((seed + i * 31) % 256 for i in range(1 << 16))
    return (block * -(-size // len(block)))[:size]


@pytest.mark.parametrize("size", SIZES)
def test_bypass_round_trip_preserves_bytes(tmp_path, size):
    session = start_fake("bypass", tmp_path, {"island-test": tmp_path})
    try:
        sent = payload_for(size, 7)
        results = session.submit(
            "island-test",
            "t0",
            {"q_v": sent},
            ["q_v"],
        )
        key = hashlib.sha256(b"q_v").digest()
        want = bytes(b ^ key[i % len(key)] for i, b in enumerate(sent))
        assert bytes(results["q_v"]) == want
        assert session.submissions == 1
        assert session.input_bytes == size
        assert session.output_bytes == size
    finally:
        session.close()


def test_bypass_multi_frame_round_trip(tmp_path):
    session = start_fake("bypass", tmp_path, {"island-test": tmp_path})
    try:
        inputs = {f"in{i}": payload_for(size, i) for i, size in enumerate(SIZES)}
        results = session.submit(
            "island-test", "t1", inputs, [f"in{i}" for i in range(len(SIZES))]
        )
        for i, size in enumerate(SIZES):
            key = hashlib.sha256(f"in{i}".encode()).digest()
            want = bytes(
                b ^ key[j % len(key)] for j, b in enumerate(inputs[f"in{i}"])
            )
            assert bytes(results[f"in{i}"]) == want
        assert session.input_bytes == sum(SIZES)
    finally:
        session.close()


def test_bypass_batch_deadline_guards_reads(tmp_path):
    session = start_fake("bypass", tmp_path, {"island-test": tmp_path})
    try:
        session.begin_batch(120000)
        sent = payload_for((1 << 20) + 3, 3)
        results = session.submit("island-test", "t2", {"x": sent}, ["x"])
        assert len(results["x"]) == len(sent)
        assert session.end_batch() == 1
    finally:
        session.close()


def test_relay_mode_round_trip(tmp_path):
    session = start_fake("relay", tmp_path, {"island-test": tmp_path})
    try:
        session.begin_batch(120000)
        sent = payload_for((1 << 20) + 3, 5)
        results = session.submit("island-test", "t3", {"a": sent}, ["y"])
        want = hashlib.sha256(b"y").digest()
        assert bytes(results["y"]) == want
        assert session.end_batch() == 1
        assert session.submissions == 1
    finally:
        session.close()


def test_bypass_stall_is_bounded_by_client_guard(tmp_path):
    # Regression: a guard that re-arms its deadline per wait never fires
    # and the client hangs forever on a stalled worker.
    session = start_fake("stall", tmp_path, {"island-test": tmp_path})
    session.deadline_ms = 300
    with pytest.raises(ResidentWorkerError):
        session.submit("island-test", "t5", {"x": b"payload"}, ["x"])
    assert session.timeouts == 1
    assert session._process is None


def test_bypass_deadline_failure_ends_session(tmp_path):
    session = start_fake("bypass", tmp_path, {"island-test": tmp_path})
    # Inject a failure frame directly: the client must die with the
    # reason and count the timeout, never retry.
    session._inbox += b"failed: deadline exceeded\n"
    with pytest.raises(ResidentWorkerError, match="deadline"):
        session.submit("island-test", "t4", {"x": b"123"}, ["x"])
    assert session.timeouts == 1
    assert session._process is None
