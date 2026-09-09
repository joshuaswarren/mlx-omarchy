import mlx.core as mx
import numpy as np
x = mx.array(np.load('/tmp/mlx-native-q4-long-operations/call0-norm1-input.npy')).astype(mx.float32)
y = mx.fast.rms_norm(x, None, 1e-6)
mx.eval(y)
np.save('/tmp/mlx-rms-nir-output.npy', np.array(y))
print('RMS_CAPTURE_COMPLETE', y.shape, mx.__version__)
