import mlx.core as mx
import numpy as np

x = mx.array(np.load('/tmp/mlx-attention-exp-inputs.npy'))
kernel = mx.fast.metal_kernel(name='attention_exp2_control', input_names=['x'], output_names=['y'], source='uint i = thread_position_in_grid.x; y[i] = metal::fast::exp2(x[i]);')
y, = kernel(inputs=[x], grid=(x.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[mx.float32])
mx.eval(y)
np.save('/tmp/mlx-native-attention-exp2.npy', np.array(y))
print(mx.__version__, y.shape)
