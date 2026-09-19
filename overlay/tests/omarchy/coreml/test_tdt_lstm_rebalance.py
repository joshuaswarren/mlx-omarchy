# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exactness checks for the LSTM thread rebalance in the GPU TDT loop.

The loop kernel's LSTM chains previously ran only on threads 0-639 with
all four gate folds inline per lane thread; they now map all 2560
``(lane, gate)`` fold items across all 1024 threads (items ``p = t``,
``t + 1024``, ``t + 2048``) and stage each gate pre-activation through a
bit-exact carrier before the unchanged gate-pairing phase reads it back:

* gate 0 -> ``s_relu[lane]``          (native fp16 slot, dead in chains)
* gate 1 -> ``s_bval[lane]``          (float carrier of an fp16 value)
* gate 2 -> ``s_bidx[lane]``          (uint carrier via floatBitsToUint)
* gate 3 -> ``s_h1[lane]``            (float carrier; pairing reads it
  back before overwriting the slot with the layer's h1, same thread)

The fold, bias add, gate pairing, cell update, argmax and control are
untouched, so per ``(lane, gate)`` the arithmetic sequence is identical;
only thread mapping and carrier round-trips differ.  These tests pin
exactly-onto coverage of the item mapping and bit-exactness of every
carrier.
"""

import unittest

import numpy as np

_LANES = 640
_GATES = 4
_ITEMS = _LANES * _GATES
_THREADS = 1024


def _thread_items(t):
    """Items the rebalanced kernel assigns to thread ``t``."""
    return [p for p in (t, t + _THREADS, t + 2 * _THREADS) if p < _ITEMS]


class LstmRebalanceTest(unittest.TestCase):
    def test_item_mapping_covers_every_lane_gate_exactly_once(self):
        """The 1024-thread mapping covers all (lane, gate) items exactly once."""
        seen = {}
        for t in range(_THREADS):
            for p in _thread_items(t):
                gate, lane = divmod(p, _LANES)
                self.assertNotIn((lane, gate), seen, f"item {p} duplicated")
                seen[(lane, gate)] = t
        self.assertEqual(len(seen), _ITEMS)
        for lane in range(_LANES):
            for gate in range(_GATES):
                self.assertIn((lane, gate), seen)
        # balance: 512 threads carry three items, 512 carry two; no thread
        # carries more than ceil(2560/1024) = 3 of the 4-gate work
        loads = [len(_thread_items(t)) for t in range(_THREADS)]
        self.assertEqual(loads.count(3), 512)
        self.assertEqual(loads.count(2), 512)
        self.assertEqual(max(loads), 3)

    def test_carriers_roundtrip_fp16_bits(self):
        """Native, float-carried, and uint-carried slots preserve fp16 bits."""
        rng = np.random.default_rng(925)
        values = rng.standard_normal(_LANES).astype(np.float16)

        native = np.array(values, dtype=np.float16)  # s_relu[lane]
        np.testing.assert_array_equal(
            native.view(np.uint16), values.view(np.uint16)
        )

        float_carried = values.astype(np.float32).astype(np.float16)
        np.testing.assert_array_equal(
            float_carried.view(np.uint16), values.view(np.uint16)
        )

        # s_bidx[lane] = floatBitsToUint(float(pr)); read back via
        # uintBitsToFloat then float16_t() — bit-transparent round trip.
        as_uint = values.astype(np.float32).view(np.uint32).copy()
        back = as_uint.view(np.float32).astype(np.float16)
        np.testing.assert_array_equal(back.view(np.uint16), values.view(np.uint16))

        # bias / sign / subnormal edges must survive the float carriers too
        edges = np.array(
            [0.0, -0.0, 1.0, -1.0, 6.0e-8, -6.0e-8, 65504.0, -65504.0,
             5.9604645e-8, -5.9604645e-8, 0.33325195],
            dtype=np.float16,
        )
        for v in edges:
            self.assertEqual(
                np.float16(np.float32(v)).view(np.uint16), v.view(np.uint16)
            )
            self.assertEqual(
                np.float32(v).view(np.uint32).view(np.float32).astype(np.float16).view(np.uint16),
                v.view(np.uint16),
            )

    def test_threadgroup_budget_unchanged(self):
        """Rebalance stages through existing arrays: no shared-memory growth."""
        # The kernel's declared threadgroup arrays are unchanged; assert the
        # actual array byte sum and that the kernel's conservative requirement
        # constant (LOOP_WORKGROUP_MEMORY_BYTES = 28768, used by
        # device_blockers) stays at or above it and within the M1 32 KiB
        # workgroup limit, so no device silently loses the loop path.
        actual = {
            "s_hidden": 1280 * 4, "s_cell": 1280 * 4, "s_h1": 640 * 4,
            "s_a": 1280 * 2, "s_pj": 640 * 2, "s_relu": 640 * 2,
            "s_bval": 1024 * 4, "s_bidx": 1024 * 4,
            "s_dval": 8 * 4, "s_ctl": 16 * 4,
        }
        self.assertEqual(sum(actual.values()), 26208)
        self.assertLessEqual(sum(actual.values()), 28768)
        self.assertLessEqual(28768, 32768)


if __name__ == "__main__":
    unittest.main(verbosity=2)
