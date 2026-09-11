import mlx.core as mx
import numpy as np
f32 = mx.array(np.array([0x00000001], dtype=np.uint32)).view(mx.float32)
f16 = mx.array(np.array([0x0001], dtype=np.uint16)).view(mx.float16)
bf = mx.array(np.array([0x0001], dtype=np.uint16)).view(mx.bfloat16)
print("f32->bool", f32.astype(mx.bool_).item())
print("f16->bool", f16.astype(mx.bool_).item())
print("bf16->bool", bf.astype(mx.bool_).item())
