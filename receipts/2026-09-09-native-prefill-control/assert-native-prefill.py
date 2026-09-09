import sys

import numpy as np

actual = np.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx-native-prefill-fixed-attention/f16.npy")
expected = np.load('/tmp/mlx-native-q4-long-operations/call0-sdpa-output.npy')
assert actual.shape == expected.shape
assert np.isfinite(actual).all()
different = int(np.count_nonzero(actual != expected))
assert different == 0, f'Native prefill mismatch: {different} elements; max error {np.max(np.abs(actual - expected))}'
print('Native fixed-input prefill: exact match')
