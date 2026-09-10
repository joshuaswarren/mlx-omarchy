# Minimal deterministic reproduction of the qmm coopmat offset-parity
# defect (receipts/2026-09-10-qmm-splitk-parity "fragmentation_hazard_found").
#
# Forces the unaligned condition directly: the lhs activation is the same
# f16 values either as a fresh whole buffer (element offset 0) or as a
# row-contiguous view at f16 element offset 1 (byte offset 2) inside a
# parent buffer - the shape allocator pressure produces. No load, no
# fragmentation, no device_info call needed; the view offset parity is the
# whole trigger.
#
# Old gate: odd lhs_offset silently rerouted matrix_m>1 q4/g64 f16
# quantized_matmul from QmmPrefillCoopmatF16 to QmmTileRbF16. The tile
# kernel's accumulation order differs, so the two arms returned different
# bits - on the M1 fork the pinned long-decode-128 prefill digest flips
# 4cc08910 -> f873dc2b under exactly this condition.
# Fixed gate: the odd view is staged into an aligned buffer, so the coopmat
# route is allocation-independent and both arms are bit-identical.
#
# Run on a coopmat-capable driver (Honeykrisp fork, cooperative_matrix_f32_8=1):
#   python3 repro_offset_parity.py
# Old wheel: MISMATCH (tile vs coopmat bits).  Fixed wheel: IDENTICAL.
# On stock Mesa (coopmat_f32_8=0) both arms take the tile route and match
# on either wheel - the repro is only discriminating where the gate lives.

m, k, n = 64, 256, 192  # matrix_m > 1, k multiple of group_size 64

w = mx.random.normal((n, k), key=mx.random.key(17))
wq, scales, biases = mx.quantize(w, group_size=64, bits=4)

x = mx.random.normal((m * k + 1,), key=mx.random.key(410)).astype(mx.float16)
x_odd = x[1:].reshape(m, k)  # row-contiguous view, f16 element offset 1
assert x_odd.shape == (m, k)
x_whole = mx.contiguous(x[1:]).reshape(m, k)  # same values, fresh buffer

out_aligned = mx.quantized_matmul(
    x_whole, wq, scales, biases, transpose=True, group_size=64, bits=4
)
out_odd = mx.quantized_matmul(
    x_odd, wq, scales, biases, transpose=True, group_size=64, bits=4
)
mx.eval(out_aligned, out_odd)

mismatched = int((out_aligned != out_odd).sum())
print(f"mismatches={mismatched}/{out_aligned.size}")
if mismatched == 0:
    print("IDENTICAL - offset parity does not change output bits")
else:
    print("MISMATCH - odd-offset view silently rerouted kernels (old gate)")
