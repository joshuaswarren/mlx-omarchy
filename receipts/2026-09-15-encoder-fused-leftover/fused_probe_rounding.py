import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)
rng = np.random.default_rng(31)


def launch1(name, ins, src, inputs, n, dtype):
    k = mx.fast.metal_kernel(name=name, input_names=ins, output_names=["d"],
                             source=src, compile_options={"math_mode": "safe"})
    return k(inputs=inputs, output_shapes=[(n,)], output_dtypes=[dtype],
             grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]


def fd(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    d = np.flatnonzero(a != b)
    if d.size == 0:
        return "identical"
    i = int(d[0])
    return f"{d.size}/{a.size} first {i}: ref={a[i]!r} got={b[i]!r}"


# ---- GLU gate rounding: half-local vs packHalf local
b = mx.array(rng.standard_normal(4096).astype(np.float16))
a = mx.array(rng.standard_normal(4096).astype(np.float16))
s32_ref = 1.0 / (1.0 + mx.exp(-(b.astype(mx.float32))))
g16_ref = s32_ref.astype(mx.float16)
prod_ref = (a.astype(mx.float32) * g16_ref.astype(mx.float32)).astype(mx.float16)

got_g1 = launch1("p5_g1", ["lhs", "rhs"],
                 "uint i = thread_position_in_grid.x;"
                 " half gate = half(1.0 / (1.0 + exp(-float(rhs[i]))));"
                 " d[i] = half(float(lhs[i]) * float(gate));",
                 [a, b], 4096, mx.float16)
print("GLU half-local   :", fd(prod_ref, got_g1))

got_g2 = launch1("p5_g2", ["lhs", "rhs"],
                 "uint i = thread_position_in_grid.x;"
                 " float s = 1.0 / (1.0 + exp(-float(rhs[i])));"
                 " float gate = float(unpackHalf2x16(packHalf2x16(vec2(s, 0.0))).x);"
                 " d[i] = half(float(lhs[i]) * gate);",
                 [a, b], 4096, mx.float16)
print("GLU pack-local   :", fd(prod_ref, got_g2))

# ---- LN tail on coherent inputs
rows, width = 375, 1024
n = rows * width
xf = mx.array(rng.standard_normal(n).astype(np.float32))
xf2 = mx.reshape(xf, (rows, width))
mu = mx.mean(xf2, axis=-1, keepdims=True)
va = mx.mean(mx.square(xf2 - mu), axis=-1, keepdims=True)
eps = 1e-5
rstd = mx.rsqrt(va + eps)
ga = mx.array(rng.standard_normal(width).astype(np.float16))
be = mx.array(rng.standard_normal(width).astype(np.float16))
ref = (xf2 - mu) * rstd
ref = ref * ga.astype(mx.float32)
ref = (ref + be.astype(mx.float32)).astype(mx.float16)
ref = mx.reshape(ref, (n,))

# inversesqrt vs mx.rsqrt on the same fp32 values
got_inv = launch1("p5_inv", ["v", "e"],
                  "uint i = thread_position_in_grid.x; d[i] = inversesqrt(v[i] + e[0]);",
                  [mx.reshape(va, (rows,)), mx.array([eps], dtype=mx.float32)],
                  rows, mx.float32)
print("inversesqrt vs rsqrt:", fd(rstd, got_inv))

argl = [xf, mu, va, ga, be, mx.array([eps], dtype=mx.float32)]
head = ("uint i = thread_position_in_grid.x;"
        " uint r = i / 1024u; uint c = i % 1024u;"
        " float t = xf[i] - mu[r];"
        " float rstd = inversesqrt(va[r] + ep[0]);")
got_t1 = launch1("p5_t1", ["xf", "mu", "va", "ga", "be", "ep"],
                 head + " d[i] = half(t * rstd * float(ga[c]) + float(be[c]));",
                 argl, n, mx.float16)
print("tail half-final  :", fd(ref, got_t1))

got_t2 = launch1("p5_t2", ["xf", "mu", "va", "ga", "be", "ep"],
                 head + " float pr = t * rstd * float(ga[c]);"
                 " pr = pr + float(be[c]);"
                 " d[i] = float(unpackHalf2x16(packHalf2x16(vec2(pr, 0.0))).x);",
                 argl, n, mx.float16)
print("tail pack+split  :", fd(ref, got_t2))

got_t3 = launch1("p5_t3", ["xf", "mu", "va", "ga", "be", "ep"],
                 head + " float p1 = t * rstd;"
                 " float p2 = p1 * float(ga[c]);"
                 " float p3 = p2 + float(be[c]);"
                 " d[i] = float(unpackHalf2x16(packHalf2x16(vec2(p3, 0.0))).x);",
                 argl, n, mx.float16)
print("tail split-all   :", fd(ref, got_t3))
