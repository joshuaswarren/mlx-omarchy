import os
os.environ.setdefault("MLX_OMARCHY_ALLOW_NON_APPLE", "1")
import mlx.core as mx
import numpy as np

print("version:", mx.__version__)
try:
    print("device_info:", mx.device_info())
except Exception as e:
    print("device_info failed:", e)

mx.random.seed(0)
a = mx.random.uniform(shape=(2,) * 10)
b = mx.random.uniform(shape=(2,) * 10)
a = a.T
mx.eval(a, b)
print("a shape:", a.shape)
print("b shape:", b.shape)

def fn(a, b):
    return mx.abs(a + b)

expected = fn(a, b)
mx.eval(expected)

for mode, env in [("fusion-default", None), ("fusion-off", "0"), ("fusion-on", "1")]:
    if env is None:
        os.environ.pop("MLX_OMARCHY_FUSED_CHAIN", None)
    else:
        os.environ["MLX_OMARCHY_FUSED_CHAIN"] = env
    out = mx.compile(fn)(a, b)
    mx.eval(out)
    ok = bool(mx.allclose(out, expected).item())
    diff = np.max(np.abs(np.array(out.tolist(), dtype=np.float64) - np.array(expected.tolist(), dtype=np.float64)))
    print(mode, "allclose:", ok, "maxabsdiff:", diff)
