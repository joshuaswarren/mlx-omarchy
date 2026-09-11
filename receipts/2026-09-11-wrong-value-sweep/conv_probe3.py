import sys
sys.path.insert(0, '.work/mlx/python/tests')
import test_conv, unittest
import mlx.core as mx
import numpy as np

suite = unittest.TestSuite()
suite.addTest(test_conv.TestConv('test_conv_general_flip_grad'))
unittest.TextTestRunner(verbosity=0).run(suite)

a_np = np.random.rand(24).astype(np.float32)
v_np = np.random.rand(4).astype(np.float32)
a_mx = mx.array(a_np); v_mx = mx.array(v_np)
got = np.array(mx.convolve(a_mx, v_mx, mode='same')).astype(np.float32)
want = np.convolve(a_np, v_np, mode='same')
print('bad idx', np.where(~np.isclose(got, want, atol=1e-5, rtol=0))[0].tolist())
for i in np.where(~np.isclose(got, want, atol=1e-5, rtol=0))[0]:
    print(f'idx {i}: got {got[i]:.6f} want {want[i]:.6f}')
# flip hypothesis: got == np.convolve(a, v[::-1], 'same')?
flipw = np.convolve(a_np, v_np[::-1].copy(), mode='same')
print('matches flipped-kernel conv:', np.allclose(got, flipw, atol=1e-5))
print('got ', got[:4].tolist(), '...', got[-2:].tolist())
print('want', want[:4].tolist(), '...', want[-2:].tolist())
# and: does a SECOND call give the same wrong values?
got2 = np.array(mx.convolve(a_mx, v_mx, mode='same')).astype(np.float32)
print('repeat identical:', np.array_equal(got, got2))
# CPU stream confirm
cpuc = np.array(mx.convolve(a_mx, v_mx, mode='same', stream=mx.cpu)).astype(np.float32)
print('cpu ok:', np.allclose(cpuc, want, atol=1e-5))
