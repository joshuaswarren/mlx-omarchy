import mlx.core as mx
import numpy as np

# flood and free so the allocator recycles non-zero blocks
garb = [mx.random.normal((1<<18,)) for _ in range(4)]
mx.eval(garb)
gvals = [np.array(g) for g in garb]  # hold host copies so eval really happened
del garb

fails = 0
for t in range(40):
    a_np = np.random.rand(24).astype(np.float32)
    v_np = np.random.rand(4).astype(np.float32)
    got = np.array(mx.convolve(mx.array(a_np), mx.array(v_np), mode='same')).astype(np.float32)
    want = np.convolve(a_np, v_np, mode='same')
    bad = ~np.isclose(got, want, atol=1e-5, rtol=0)
    if bad.any():
        fails += 1
        if fails <= 3:
            print('t', t, 'bad', np.where(bad)[0].tolist(), got[bad].tolist())
print('fails:', fails, '/40')
