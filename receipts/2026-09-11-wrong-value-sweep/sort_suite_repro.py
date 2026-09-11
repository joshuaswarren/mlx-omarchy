import mlx.core as mx
import numpy as np
from itertools import product

shape = (6, 4, 10)
for dtype, axis, strided in product(("int32", "float32", "complex64"), (None, 0, 1, 2), (True, False)):
    np.random.seed(0)
    np_dtype = getattr(np, dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        a_np = (np.random.uniform(0, 100, size=shape) + 1j*np.random.uniform(0, 100, size=shape)).astype(np_dtype)
    else:
        a_np = np.random.uniform(0, 100, size=shape).astype(np_dtype)
    a_mx = mx.array(a_np)
    if strided:
        a_mx = a_mx[::2, :, ::2]
        a_np = a_np[::2, :, ::2]
    b_np = np.sort(a_np, axis=axis)
    b_mx = mx.sort(a_mx, axis=axis)

np.random.seed(0)
a_np = np.random.normal(size=(32769,)).astype(np.float32)
a_mx = mx.array(a_np)
b_mx = mx.sort(a_mx)

a_np = np.array([1, 0, 2, 1, 3, 0, 4, 0])
print('0-strides input dtype:', a_np.dtype)
a_mx = mx.array(a_np)
b_np = np.broadcast_to(a_np, (16, 8))
b_mx = mx.broadcast_to(a_mx, (16, 8))
mx.eval(b_mx)
for axis in (0, 1):
    try:
        c_mx = mx.sort(b_mx, axis=axis)
        c_np_arr = np.array(c_mx)
        c_np = np.sort(b_np, axis=axis)
        print('axis', axis, 'dtype', c_mx.dtype, 'equal', np.array_equal(c_np, c_np_arr))
    except RuntimeError as e:
        print('axis', axis, 'REFUSED:', str(e)[:70])
