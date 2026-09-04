# Serialization roundtrips for strided views; constant writes require
# C-order (patches/mlx-export-dense-constants.patch).
# Expectations are NumPy arithmetic on the host, never a backend echo.
import os

os.environ.setdefault("MLX_OMARCHY_ALLOW_NON_APPLE", "1")

import sys
import tempfile

import mlx.core as mx
import numpy as np

SRC_NP = np.arange(48, dtype=np.float32).reshape(6, 8)


def views_np():
    return {
        "gapped": SRC_NP[:, 2:6],
        "transposed": SRC_NP.T,
        "strided": SRC_NP[1::3, ::2],
    }


def probe_np(view):
    return np.arange(view.size, dtype=np.float32).reshape(view.shape)


def expect_equal(name, got, want):
    got_np = np.asarray(got)
    if got_np.shape != want.shape or got_np.dtype != want.dtype:
        raise AssertionError(
            f"{name}: got {got_np.shape}/{got_np.dtype}, "
            f"want {want.shape}/{want.dtype}"
        )
    if not np.array_equal(got_np, want):
        raise AssertionError(f"{name}: values differ from NumPy expectation")


def check_safetensors(root):
    arrays = {name: mx.array(v) for name, v in views_np().items()}
    arrays["dense"] = mx.array(np.ascontiguousarray(SRC_NP[:, 2:6]))
    path = os.path.join(root, "st.safetensors")
    mx.save_safetensors(path, arrays)
    loaded = mx.load(path)
    for name, v in views_np().items():
        expect_equal(f"safetensors {name}", loaded[name], v)
    expect_equal("safetensors dense", loaded["dense"], SRC_NP[:, 2:6])
    print("PASS safetensors roundtrip (gapped, transposed, strided, dense)")


def check_npy(root):
    for name, v in views_np().items():
        path = os.path.join(root, f"{name}.npy")
        mx.save(path, mx.array(v))
        expect_equal(f"npy {name}", mx.load(path), v)
    print("PASS npy roundtrip (gapped, transposed, strided)")


def check_export(root):
    src = mx.array(SRC_NP)
    views = {name: mx.array(v) for name, v in views_np().items()}
    views["dense"] = mx.array(np.ascontiguousarray(SRC_NP[:, 2:6]))
    # Slice-born equivalents: with the eval densifier removed these stay
    # strided views, so the export constant path must normalize them.
    views["slice_gapped"] = src[:, 2:6]
    views["slice_transposed"] = src.transpose()
    views["slice_strided"] = src[1::3, ::2]
    mx.eval(*views.values())
    numpy_by_name = dict(views_np())
    numpy_by_name["dense"] = SRC_NP[:, 2:6]
    for name in ("slice_gapped", "slice_transposed", "slice_strided"):
        numpy_by_name[name] = numpy_by_name[name.split("slice_", 1)[1]]
    for name, v in views.items():
        want = probe_np(v) + numpy_by_name[name]

        def fn(x, _v=v):
            return mx.add(x, _v)

        path = os.path.join(root, f"export_{name}.mlxfn")
        mx.export_function(path, fn, mx.array(probe_np(v)))
        out = mx.import_function(path)(mx.array(probe_np(v)))
        if isinstance(out, (list, tuple)):
            out = out[0]
        expect_equal(f"export {name}", out, want)
    print("PASS export roundtrip (numpy-born and slice-born strided views)")


def main():
    with tempfile.TemporaryDirectory() as root:
        check_safetensors(root)
        check_npy(root)
        check_export(root)
    print("ALL STRIDED SERIALIZATION CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
