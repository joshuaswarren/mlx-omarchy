"""Introspect mx.fast.metal_kernel call contract + trivial build."""
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.gpu)
k = mx.fast.metal_kernel(
    name="probe_trivial",
    input_names=["src", "bias"],
    output_names=["dst"],
    source="""
        uint index = thread_position_in_grid.x;
        dst[index] = half(float(src[index]) + float(bias[index % bias_shape[0]]));
    """,
    compile_options={"math_mode": "safe"},
)
print("type:", type(k))
print("doc:", getattr(k.__call__, "__doc__", None))
x = mx.ones((16,), mx.float16)
b = mx.ones((16,), mx.float16)
try:
    out = k(inputs=[x, b], output_shapes=[(16,)], output_dtypes=[mx.float16],
            grid=(16, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]
    print("kwargs call OK:", np.asarray(out)[:4])
except Exception as e:
    print("kwargs call FAILED:", e)
try:
    out = k([x, b], [(16,)], [mx.float16], (16, 1, 1), (256, 1, 1), mx.gpu)[0]
    print("positional call OK:", np.asarray(out)[:4])
except Exception as e:
    print("positional FAILED:", e)
