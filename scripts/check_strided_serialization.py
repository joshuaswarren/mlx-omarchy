# Roundtrip checks for the strided-view serialization boundaries
# (2026-09-04 strided-consumer audit, branch strided-serialization).
#
# Each check saves and reloads a gapped slice view and a C-order transposed
# view of the same source, alongside a dense control, and compares the
# loaded values against mx.contiguous of the view:
#   safetensors: upstream save_safetensors wraps every tensor in
#     contiguous()+eval before the packed write (mlx/io/safetensors.cpp);
#     the write must read logical C order.
#   npy: save_npy normalizes via contiguous(a, true) (mlx/io/load.cpp).
#   export_function: closed-over evaluated views become primitive-less
#     constants; patches/mlx-export-dense-constants.patch normalizes
#     non-row-contiguous constants before the flat tape write. A C-order
#     transpose reports flags().contiguous == true under the span-based
#     definition, so only row_contiguous is a valid gate.
#
# Normal valid roundtrips only; no hostile files, no partial saves.
import os
import sys
import tempfile

import mlx.core as mx


def make_views():
    src = mx.arange(48).reshape(6, 8)
    gapped = src[:, 2:6]
    transposed = src.transpose()
    dense = src[:, 2:6] + 0  # same values, freshly materialized
    return src, gapped, transposed, dense


def expect_equal(name, got, want):
    if got.shape != want.shape or got.dtype != want.dtype:
        raise AssertionError(
            f"{name}: got {got.shape}/{got.dtype}, "
            f"want {want.shape}/{want.dtype}"
        )
    if not mx.array_equal(got, want).item():
        raise AssertionError(f"{name}: values differ after roundtrip")


def check_safetensors(root, gapped, transposed, dense):
    path = os.path.join(root, "st.safetensors")
    mx.save_safetensors(
        path, {"gapped": gapped, "transposed": transposed, "dense": dense}
    )
    loaded = mx.load(path)
    expect_equal("safetensors gapped", loaded["gapped"], mx.contiguous(gapped))
    expect_equal(
        "safetensors transposed", loaded["transposed"], mx.contiguous(transposed)
    )
    expect_equal("safetensors dense", loaded["dense"], dense)
    print("PASS safetensors roundtrip (gapped, transposed, dense)")


def check_npy(root, gapped, transposed):
    for name, view in (("gapped", gapped), ("transposed", transposed)):
        path = os.path.join(root, f"{name}.npy")
        mx.save(path, view)
        expect_equal(f"npy {name}", mx.load(path), mx.contiguous(view))
    print("PASS npy roundtrip (gapped, transposed)")


def check_export(root, gapped, transposed):
    # Eval first: the Slice outputs detach into primitive-less views, so
    # export_function captures them as data constants (the B2 path).
    mx.eval(gapped, transposed)
    for name, view in (("gapped", gapped), ("transposed", transposed)):
        want = mx.contiguous(view)
        path = os.path.join(root, f"export_{name}.mlxfn")

        def fn(x, _view=view):
            return mx.add(x, _view)

        probe = mx.zeros(want.shape, mx.int32)
        mx.export_function(path, fn, probe)
        imported = mx.import_function(path)
        out = imported(probe)
        if isinstance(out, (list, tuple)):
            out = out[0]
        expect_equal(f"export {name} constant", out, mx.add(probe, want))
    print("PASS export roundtrip (closed-over gapped, transposed)")


def main():
    _, gapped, transposed, dense = make_views()
    with tempfile.TemporaryDirectory() as root:
        check_safetensors(root, gapped, transposed, dense)
        check_npy(root, gapped, transposed)
        check_export(root, gapped, transposed)
    print("ALL STRIDED SERIALIZATION CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
