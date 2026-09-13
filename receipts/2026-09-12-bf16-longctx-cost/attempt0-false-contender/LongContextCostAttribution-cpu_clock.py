#!/usr/bin/env python3
import json
import pathlib
import resource
import sys
import time

scripts = pathlib.Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(scripts))
import bench_decode

original = bench_decode.run_generation


def measured(iterator, ids=None, stats=None):
    samples = []

    def observed():
        for item in iterator:
            samples.append((time.monotonic_ns(), time.process_time_ns(),
                            time.thread_time_ns(),
                            resource.getrusage(resource.RUSAGE_SELF)))
            yield item

    result = original(observed(), ids, stats)
    if len(samples) < 2:
        raise RuntimeError("No decode interval measured")
    first, last = samples[0], samples[-1]
    intervals = len(samples) - 1
    print(json.dumps({
        "decode_start_monotonic_ns": first[0],
        "decode_end_monotonic_ns": last[0],
        "instrument": "canonical_decode_cpu_clock_v1",
        "scope": "first through final yielded token; excludes prefill and load",
        "intervals": intervals,
        "wall_ms_per_token": (last[0] - first[0]) / intervals / 1e6,
        "process_cpu_ms_per_token": (last[1] - first[1]) / intervals / 1e6,
        "python_thread_cpu_ms_per_token": (last[2] - first[2]) / intervals / 1e6,
        "voluntary_context_switches": last[3].ru_nvcsw - first[3].ru_nvcsw,
        "involuntary_context_switches": last[3].ru_nivcsw - first[3].ru_nivcsw,
        "minor_faults": last[3].ru_minflt - first[3].ru_minflt,
        "major_faults": last[3].ru_majflt - first[3].ru_majflt,
        "ids_sha256_16": bench_decode.ids_digest(ids),
    }, sort_keys=True), flush=True)
    return result


bench_decode.run_generation = measured
bench_decode.main()
