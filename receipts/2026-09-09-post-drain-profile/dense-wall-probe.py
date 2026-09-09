import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

mx.random.seed(0)
results = []
for n, k in [(896,896),(128,896),(4864,896),(896,4864),(151936,896)]:
    x = mx.random.normal((1,k)).astype(mx.bfloat16)
    w = mx.random.normal((n,k)).astype(mx.bfloat16)
    mx.eval(x,w)
    y = x @ w.T
    mx.eval(y)
    assert bool(mx.all(mx.isfinite(y)).item())
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(10):
            y = x @ w.T
            mx.eval(y)
        samples.append((time.perf_counter_ns() - start) / 1e6 / 10)
    results.append({'n':n,'k':k,'samples_ms':samples,'median_ms':statistics.median(samples),'weight_bytes':n*k*2})
    print(results[-1],flush=True)
Path(sys.argv[1]).write_text(json.dumps({'mlx_file':mx.__file__,'results':results,'scope':'Synthetic BF16 matrix-vector wall time, no GPU timestamp instrumentation; not end-to-end performance acceptance.'},indent=2)+'\n')
