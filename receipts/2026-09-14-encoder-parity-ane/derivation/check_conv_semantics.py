#!/usr/bin/env python3
"""Confirm mx.conv2d's grouped-conv weight layout matches what vulkan_encoder
assumes, using the encoder's real conv configurations. Runs on whichever
stream the caller selects; correctness of the layout is stream-independent.
"""

import sys

import mlx.core as mx
import numpy as np


def emulate(x_nhwc, w_ohwi, stride, groups):
    n, h, w, cin = x_nhwc.shape
    o, kh, kw, cing = w_ohwi.shape
    sh, sw = stride
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    out = np.zeros((n, oh, ow, o), dtype=np.float32)
    per = o // groups
    for g in range(groups):
        xg = x_nhwc[..., g * cing:(g + 1) * cing]
        wg = w_ohwi[g * per:(g + 1) * per]
        for i in range(oh):
            for j in range(ow):
                patch = xg[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                out[:, i, j, g * per:(g + 1) * per] = np.tensordot(
                    patch, wg, axes=([1, 2, 3], [1, 2, 3])
                )
    return out


CASES = [
    # name, x NHWC, weight OHWI, stride, groups   (encoder's real configurations)
    ("prologue g1 s2 k3x3", (1, 14, 12, 8), (6, 3, 3, 8), (2, 2), 1),
    ("prologue depthwise g=cin", (1, 14, 12, 16), (16, 3, 3, 1), (2, 2), 16),
    ("pointwise valid 1x1", (1, 6, 5, 8), (4, 1, 1, 8), (1, 1), 1),
    ("layer depthwise k1x9", (1, 1, 28, 32), (32, 1, 9, 1), (1, 1), 32),
]


def main() -> int:
    stream = mx.gpu if "--gpu" in sys.argv else mx.cpu
    mx.set_default_device(stream)
    rng = np.random.default_rng(11)
    ok = True
    for name, xs, ws, stride, groups in CASES:
        x = rng.standard_normal(xs).astype(np.float32)
        w = rng.standard_normal(ws).astype(np.float32)
        got = mx.conv2d(mx.array(x), mx.array(w), stride=stride, padding=0, groups=groups)
        mx.eval(got)
        got = np.asarray(got)
        exp = emulate(x, w, stride, groups)
        good = got.shape == exp.shape and np.allclose(got, exp, atol=1e-3, rtol=1e-3)
        ok &= good
        diff = np.abs(got - exp).max() if got.shape == exp.shape else float("nan")
        print(
            f"{'OK  ' if good else 'FAIL'} {name:26s} out {str(got.shape):20s} "
            f"want {str(exp.shape):20s} maxdiff {diff:.2e}"
        )
    print(f"\nstream={'gpu' if stream is mx.gpu else 'cpu'} grouped-conv layout agrees: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
