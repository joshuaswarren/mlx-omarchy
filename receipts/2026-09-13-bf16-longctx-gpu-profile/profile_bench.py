#!/usr/bin/env python3
import json
import pathlib
import sys
import time

scripts = pathlib.Path(sys.argv.pop(1)).resolve()
markers = pathlib.Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(scripts))
import bench_decode

marker_file = markers.open("w", encoding="utf-8")


def mark(phase):
    marker_file.write(json.dumps({"p": phase, "t": time.monotonic_ns()}) + "\n")
    marker_file.flush()


def instrument(iterator, ids=None, stats=None):
    mark("prefill_start")
    first = True

    def observed():
        nonlocal first
        for item in iterator:
            if first:
                mark("prefill_done")
                mark("decode_start")
                first = False
            mark("tok")
            yield item

    try:
        return original_generation(observed(), ids, stats)
    finally:
        mark("decode_done")


original_generation = bench_decode.run_generation
bench_decode.run_generation = instrument

if sys.argv[1:] == ["--self-test"]:
    got = []
    times, count = instrument(iter([7, 8, 9]), got)
    marker_file.close()
    phases = [json.loads(line)["p"] for line in markers.read_text().splitlines()]
    assert count == 3 and len(times) == 3 and got == [7, 8, 9]
    assert phases == ["prefill_start", "prefill_done", "decode_start",
                      "tok", "tok", "tok", "decode_done"], phases
    print("PROFILE_BENCH_SELF_TEST_OK")
else:
    mark("load_start")
    import mlx_lm.utils

    original_load = mlx_lm.utils.load

    def instrument_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        mark("load_done")
        return result

    mlx_lm.utils.load = instrument_load
    try:
        bench_decode.main()
    finally:
        marker_file.close()
