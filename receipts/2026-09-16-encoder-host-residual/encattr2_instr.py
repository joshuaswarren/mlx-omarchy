"""Apply the ENCATTR2 bucket timers to a vulkan_encoder.py runner.

Same instrumentation EncoderAttribution2 used: additive monotonic clocks
around the island stage/pack loops, the statement loop, and the final drain,
plus one summary line. Timing only -- no op, order, or value changes.
"""
import sys
from pathlib import Path

SUBST = [
    # 1. island init counters
    (
        "        self.exec_ns = 0\n        self.timeouts = 0",
        "        self.exec_ns = 0\n"
        "        self.stage_ns = 0  # ENCATTR2: per-round input eval drain + byte pack\n"
        "        self.pack_ns = 0   # ENCATTR2: per-round output unpack + host->dev\n"
        "        self.timeouts = 0",
    ),
    # 2. stage timer start
    (
        "        payload = {}\n        in_bytes = 0\n        for name, value in inputs.items():",
        "        payload = {}\n        in_bytes = 0\n"
        "        stage_started = time.monotonic_ns()  # ENCATTR2\n"
        "        for name, value in inputs.items():",
    ),
    # 3. stage timer end
    (
        "            in_bytes += len(raw)\n        out_names = list(outputs)",
        "            in_bytes += len(raw)\n"
        "        self.stage_ns += time.monotonic_ns() - stage_started  # ENCATTR2\n"
        "        out_names = list(outputs)",
    ),
    # 4. pack timer start
    (
        '        record = {"tag": tag, "bundle": bundle, "elapsed_ns": elapsed, "round": True}\n'
        "        out_bytes = 0",
        '        record = {"tag": tag, "bundle": bundle, "elapsed_ns": elapsed, "round": True}\n'
        "        pack_started = time.monotonic_ns()  # ENCATTR2\n"
        "        out_bytes = 0",
    ),
    # 5. pack timer end
    (
        '            packed[name] = mx.array(host).reshape(shape)\n        record["input_bytes"] = in_bytes',
        '            packed[name] = mx.array(host).reshape(shape)\n'
        "        self.pack_ns += time.monotonic_ns() - pack_started  # ENCATTR2\n"
        '        record["input_bytes"] = in_bytes',
    ),
    # 6. runner init counters
    (
        "        self.text = mil_path.read_text()\n"
        "        self.blobs = Blobs(model_root)\n"
        "        self.island = island",
        "        self.text = mil_path.read_text()\n"
        "        self.blobs = Blobs(model_root)\n"
        "        self.graph_ns = 0  # ENCATTR2: statement build (host, lazy)\n"
        "        self.drain_ns = 0  # ENCATTR2: final wanted eval drain\n"
        "        self.run_ns = 0    # ENCATTR2: whole run() body\n"
        "        self.island = island",
    ),
    # 7. run() clock start
    (
        "        protected = set(wanted) | {stop_after}\n        for stmt in self.statements:",
        "        protected = set(wanted) | {stop_after}\n"
        "        run_started = time.monotonic_ns()  # ENCATTR2\n"
        "        for stmt in self.statements:",
    ),
    # 8. per-statement timer
    (
        "        for stmt in self.statements:\n"
        "            self.execute(stmt)\n"
        "            for name in stmt.names:",
        "        for stmt in self.statements:\n"
        "            stmt_started = time.monotonic_ns()  # ENCATTR2\n"
        "            self.execute(stmt)\n"
        "            self.graph_ns += time.monotonic_ns() - stmt_started  # ENCATTR2\n"
        "            for name in stmt.names:",
    ),
    # 9. drain timer + report
    (
        "        mx.eval(list(keep.values()))\n        return keep",
        "        drain_started = time.monotonic_ns()  # ENCATTR2\n"
        "        mx.eval(list(keep.values()))\n"
        "        self.drain_ns += time.monotonic_ns() - drain_started  # ENCATTR2\n"
        "        self.run_ns = time.monotonic_ns() - run_started  # ENCATTR2\n"
        "        island = self.island  # ENCATTR2 report\n"
        "        print(  # ENCATTR2\n"
        '            "ENCATTR2"\n'
        '            f" run_ms={self.run_ns / 1e6:.1f}"\n'
        '            f" graph_ms={self.graph_ns / 1e6:.1f}"\n'
        '            f" stage_ms={island.stage_ns / 1e6:.1f}"\n'
        '            f" submit_ms={island.exec_ns / 1e6:.1f}"\n'
        '            f" pack_ms={island.pack_ns / 1e6:.1f}"\n'
        '            f" drain_ms={self.drain_ns / 1e6:.1f}"\n'
        '            f" batch_open_ms={island.batch_open_ns / 1e6:.1f}"\n'
        '            f" rounds={island.rounds}"\n'
        '            f" worker_starts={island.worker_starts}",\n'
        "            flush=True,\n"
        "        )\n"
        "        return keep",
    ),
]

source = Path(sys.argv[1]).read_text()
for old, new in SUBST:
    if source.count(old) != 1:
        sys.exit(f"anchor not unique ({source.count(old)}x): {old[:60]!r}")
    source = source.replace(old, new)
markers = source.count("# ENCATTR2")
if markers != 17:
    sys.exit(f"expected 17 '# ENCATTR2' markers, found {markers}")
Path(sys.argv[2]).write_text(source)
print(f"OK {sys.argv[2]}: 17 markers")
