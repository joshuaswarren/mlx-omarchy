#!/usr/bin/env python3
import json
import mlx.core as mx


def fixed_values(count, multiplier, bias):
    return [(((i * multiplier + bias) % 251) - 125) / 128.0 for i in range(count)]


def capture(keys):
    heads, kv_heads, width = 14, 2, 64
    q = mx.array(fixed_values(heads * width, 37, 11), dtype=mx.bfloat16).reshape(
        1, heads, 1, width
    )
    k = mx.array(
        fixed_values(kv_heads * keys * width, 53, 17), dtype=mx.bfloat16
    ).reshape(1, kv_heads, keys, width)
    v = mx.array(
        fixed_values(kv_heads * keys * width, 71, 23), dtype=mx.bfloat16
    ).reshape(1, kv_heads, keys, width)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    mx.eval(out)
    return [float(value) for value in out.flatten().tolist()]


print(
    json.dumps(
        {
            "device": str(mx.default_device()),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "outputs": {str(keys): capture(keys) for keys in (263, 1024, 1025)},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
