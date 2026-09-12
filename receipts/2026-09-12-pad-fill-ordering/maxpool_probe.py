import sys
import mlx.core as mx
import numpy as np

# MaxPool1d k=2 s=2 p=1 churn probe, same shape as conv_probe3.py
# (receipts/2026-09-11-wrong-value-sweep): pooled padding routes through
# the same scalar fill the convolve boundary garbage came from.
fails = 0
bad_iters = []
for t in range(40):
    x_np = np.random.rand(24).astype(np.float32)
    x = mx.array(x_np)[None, None, :]
    got = np.array(mx.max_pool1d(x, kernel_size=2, stride=2, padding=1))[0, 0]
    xp = np.concatenate([np.array([-np.inf], dtype=np.float32), x_np,
                         np.array([-np.inf], dtype=np.float32)])
    want = np.array([max(xp[2 * i], xp[2 * i + 1]) for i in range(13)],
                    dtype=np.float32)
    bad = ~np.isclose(got, want, atol=1e-6, rtol=0)
    if bad.any():
        fails += 1
        bad_iters.append(t)
        if fails <= 3:
            print('t', t, 'bad', np.where(bad)[0].tolist(),
                  got[bad].tolist(), want[bad].tolist())
print('maxpool fails:', fails, '/40', 'bad_iters:', bad_iters[:10])
