# Staging cost of the offset-alignment fix: quantized_matmul f16/q4/g64
# prefill shapes, aligned whole-buffer x vs odd-offset x view (the view arm
# pays the one-time dense staging copy post-fix; pre-fix it paid a full
# reroute to the 3x slower tile kernel). 8 warmup + 16 timed evals.
import time

import mlx.core as mx

SHAPES = [
    (262, 896, 896), (262, 896, 4864), (262, 4864, 896), (262, 896, 128),
    (1053, 896, 896), (1053, 896, 4864), (1053, 4864, 896), (1053, 896, 128),
]


def bench(fn, warmup=8, iters=16):
    for _ in range(warmup):
        mx.eval(fn())
    best = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        best.append((time.perf_counter() - t0) * 1e3)
    best.sort()
    return best[len(best) // 2]


def main():
    for m, k, n in SHAPES:
        w = mx.random.normal((n, k), key=mx.random.key(17))
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        x = mx.random.normal((m * k + 1,), key=mx.random.key(410)).astype(
            mx.float16)
        x_odd = x[1:].reshape(m, k)
        x_whole = mx.contiguous(x[1:]).reshape(m, k)

        def run(xx):
            return lambda: mx.quantized_matmul(
                xx, wq, scales, biases, transpose=True,
                group_size=64, bits=4)

        a = bench(run(x_whole))
        b = bench(run(x_odd))
        same = int((run(x_whole)() != run(x_odd)()).sum())
        print(f"m={m} k={k} n={n} aligned={a:.3f}ms odd_view={b:.3f}ms "
              f"staging_delta={b - a:+.3f}ms mismatches={same}")


if __name__ == "__main__":
    main()
