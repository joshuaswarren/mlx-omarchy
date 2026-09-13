# jwm1 and jw16 same-protocol MLX GPU comparison

Date: 2026-09-13

## Identity

- mlx-omarchy checkout receipt commit: `f7169fcb7c2df1713ee635ec5cdc57462292d90e`
- measured wheel source commit on both hosts: `b41e2b74c330f910b24cab0e7516e306527858f0`
- benchmark source SHA-256: `afc7f36a59fd3de6ee7a332bd5deae94fc41c6b720848249c2528aabd3c0a496`
- resolved agent model: `openai-codex/gpt-5.6-sol`; fallback: false
- model and quantization: not applicable; this is a synthetic fp16 matrix workload

The wheel build timestamps differ, but both local versions identify the same source commit:

| Host | Installed wheel |
| --- | --- |
| `jw16mbp1-linux` | `mlx-omarchy==0.32.2.dev202609122106+b41e2b74` |
| `jwm1-linux` | `mlx-omarchy==0.32.2.dev202609122355+b41e2b74` |

## Protocol

Each host ran the exact same benchmark source. The remote Python process held its host's `/tmp/m1-gpu.lock` through one `flock` for the complete run. The script:

1. selected `mx.gpu` as the default device;
2. evaluated a fixed 2x2 fp16 `matmul(a,b)+bias` and required exact equality with `[[20.0, 21.0], [43.5, 49.5]]`;
3. generated identical 2048x2048 fp16 inputs from seed `20260913` and evaluated them before timing;
4. ran three unrecorded warmups followed by 20 measured `matmul(a,b)+bias` evaluations;
5. timed graph construction through synchronized completion with `time.perf_counter_ns()` around `mx.matmul(a,b)+bias` and `mx.eval(result)`;
6. reported the sample median and computed TFLOPS from `(2*N^3+N^2) / elapsed_seconds`.

No thermal soak or temperature normalization was applied. The exclusive host-local lock, evaluated inputs, three warmups, and identical measured sequence make this a same-protocol warm comparison, not a thermally normalized peak specification.

Exact commands:

```sh
ssh -o ConnectTimeout=8 -o BatchMode=yes jw16mbp1-linux \
  'timeout -k 10s 180s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python -' \
  < /tmp/mlx-gpu-parity-bench.py

ssh -o ConnectTimeout=8 -o BatchMode=yes jwm1-linux \
  'timeout -k 10s 180s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python -' \
  < /tmp/mlx-gpu-parity-bench.py
```

The accepted benchmark source was:

```python
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import time

import mlx.core as mx

vulkan = subprocess.run(
    ["vulkaninfo", "--summary"],
    check=True,
    capture_output=True,
    text=True,
    timeout=15,
).stdout
device_name_match = re.search(r"^\s*deviceName\s*=\s*(.+)$", vulkan, re.MULTILINE)
driver_name_match = re.search(r"^\s*driverName\s*=\s*(.+)$", vulkan, re.MULTILINE)
driver_info_match = re.search(r"^\s*driverInfo\s*=\s*(.+)$", vulkan, re.MULTILINE)
if device_name_match is None:
    raise RuntimeError("vulkaninfo did not report deviceName")

mx.set_default_device(mx.gpu)
exact_a = mx.array([[1, 2], [3, 4]], dtype=mx.float16)
exact_b = mx.array([[5, 6], [7, 8]], dtype=mx.float16)
exact_bias = mx.array([[1, -1], [0.5, -0.5]], dtype=mx.float16)
exact_result = mx.matmul(exact_a, exact_b) + exact_bias
mx.eval(exact_result)
expected = [[20.0, 21.0], [43.5, 49.5]]
actual = exact_result.tolist()
if actual != expected:
    raise RuntimeError(f"exact matmul+add mismatch: expected={expected} actual={actual}")

size = 2048
mx.random.seed(20260913)
a = mx.random.uniform(low=-1.0, high=1.0, shape=(size, size)).astype(mx.float16)
b = mx.random.uniform(low=-1.0, high=1.0, shape=(size, size)).astype(mx.float16)
bias = mx.random.uniform(low=-1.0, high=1.0, shape=(size, size)).astype(mx.float16)
mx.eval(a, b, bias)


def run_once():
    start_ns = time.perf_counter_ns()
    result = mx.matmul(a, b) + bias
    mx.eval(result)
    return (time.perf_counter_ns() - start_ns) / 1_000_000


warmup_ms = [run_once() for _ in range(3)]
samples_ms = [run_once() for _ in range(20)]
median_ms = statistics.median(samples_ms)
flops = 2 * size**3 + size**2
print(
    json.dumps(
        {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "mlx_omarchy_version": importlib.metadata.version("mlx-omarchy"),
            "default_device": str(mx.default_device()),
            "device_name": device_name_match.group(1).strip(),
            "driver_name": driver_name_match.group(1).strip() if driver_name_match else None,
            "driver_info": driver_info_match.group(1).strip() if driver_info_match else None,
            "dtype": "float16",
            "operation": "matmul(a,b)+bias",
            "shape": [size, size],
            "seed": 20260913,
            "exact_expected": expected,
            "exact_actual": actual,
            "exact_pass": True,
            "warmup_count": 3,
            "measured_count": 20,
            "warmup_ms": warmup_ms,
            "samples_ms": samples_ms,
            "median_ms": median_ms,
            "flop_formula": "2*N^3+N^2",
            "tflops": flops / (median_ms / 1000) / 1e12,
        },
        indent=2,
        sort_keys=True,
    )
)
```

## Results

Both exact checks returned the required fp16 matrix exactly. Both runtimes reported `Device(gpu, 0)`.

| Host | Kernel | Vulkan device | Driver | Exact check | Median ms | TFLOPS |
| --- | --- | --- | --- | --- | ---: | ---: |
| `jw16mbp1-linux` | `7.1.6-1-1-ARCH` | Apple M1 Max (G13C C0) | Honeykrisp, Mesa 26.2.2-arch1.1 | pass | 8.712577 | 1.9723284498 |
| `jwm1-linux` | `7.1.6-1-1-ARCH` | Apple M1 (G13G B1) | Honeykrisp, Mesa 26.3.0-devel (git-6f6afc8968) | pass | 31.589904 | 0.5439732735 |

The measured M1 Max result is `3.6257819013x` the M1 result by both median latency and calculated TFLOPS. The M1 result is `27.5802579204%` of the M1 Max result.

### jw16 samples

Warmups, ms: `9.247559, 8.184136, 11.290740`

Measured, ms: `8.176928, 8.711432, 8.709973, 8.780223, 8.670722, 8.220053, 8.657598, 8.713722, 8.724931, 8.214553, 8.636222, 8.688723, 8.761015, 8.726056, 8.910641, 8.797973, 8.816432, 8.782682, 8.803515, 8.270012`

A preliminary jw16 run used the same workload but printed one long JSON line, which the command harness truncated. Its visible median was 8.6836605 ms. It is excluded; the fully captured repeat above is the accepted jw16 dataset. Only JSON indentation changed before the accepted repeat.

### jwm1 samples

Warmups, ms: `36.096902, 31.704757, 31.552259`

Measured, ms: `31.543176, 31.590467, 31.589341, 31.648049, 31.612841, 31.594008, 31.598008, 31.592092, 31.563217, 31.557966, 31.588508, 31.567049, 31.574591, 31.593133, 31.576133, 31.610508, 31.729381, 31.603383, 31.563050, 31.588217`

## Safety and lock release

The benchmark commands invoked only `timeout`, `flock`, the qualified Python interpreter, MLX GPU operations, and read-only Vulkan enumeration. They did not invoke `modprobe`, access an ANE device, change a device tree, reboot either host, or change an inference service.

Fresh nonblocking lock checks after both measurements, and again after the independent jwm1 ANE recovery reboot, returned:

```text
host=jw16mbp1-linux lock=/tmp/m1-gpu.lock state=free
host=jwm1-linux lock=/tmp/m1-gpu.lock state=free
```
