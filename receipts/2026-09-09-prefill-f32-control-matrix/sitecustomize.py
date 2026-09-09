import mlx.core as mx

_sdpa = mx.fast.scaled_dot_product_attention

def _prefill_f32(q, k, v, *args, **kwargs):
    if q.dtype == mx.float16 and q.shape[-2] > 1:
        return _sdpa(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), *args, **kwargs).astype(q.dtype)
    return _sdpa(q, k, v, *args, **kwargs)

mx.fast.scaled_dot_product_attention = _prefill_f32
