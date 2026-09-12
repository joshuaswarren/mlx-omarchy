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
    x = mx.array(x_np)
    neg_inf = mx.array(np.array([-np.inf], dtype=np.float32))
    xp = mx.concatenate([neg_inf, x, neg_inf])
    windows = mx.as_strided(xp, (13, 2), (2, 1))
    got = np.array(mx.max(windows, axis=1))
    xp_np = np.concatenate([np.array([-np.inf], dtype=np.float32), x_np,
                            np.array([-np.inf], dtype=np.float32)])
    want = np.array([max(xp_np[2 * i], xp_np[2 * i + 1]) for i in range(13)],
                    dtype=np.float32)
    if ~np.isclose(got, want, atol=1e-6, rtol=0).all():
        fails += 1
        bad_iters.append(t)
        if fails <= 3:
            print('t', t, 'got', got.tolist(), 'want', want.tolist())
print('maxpool fails:', fails, '/40', 'bad_iters:', bad_iters[:10])
