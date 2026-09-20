# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Code-linked guard for the phase-split instrumentation kernels.

The split chain (K_DEC + K_JOINT) must carry the landed fused kernel's
phase arithmetic BYTE-EXACT: each phase body embedded in a split source
must be a verbatim substring of the real rendered loop shader.  This is
the emitted-source link Main required for the direct-instrumentation
lane — the split exists only for timing, so any drift between the split
sources and the landed kernel must fail here before a window is spent.
"""

import re
import unittest

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.vulkan_tdt_loop import _loop_glsl
import sys
sys.path.insert(0, _TOOLS + "/../../scripts-local/tdt-phase-attribution")
from phase_split_kernels import build_all


class PhaseSplitSourceGuard(unittest.TestCase):
    def test_phase_bodies_byte_exact(self):
        dec, joint, ph = build_all()
        stock = ph["source"]
        in_dec = ("layer_block", "pjcopy", "pjpersist", "projector")
        in_joint = ("relu", "joint_chains", "publish", "reduce_control")
        for name in in_dec:
            self.assertIn(ph[name], dec, f"{name} drifted in K_DEC")
            self.assertIn(ph[name], stock, f"{name} not found in stock source")
        for name in in_joint:
            self.assertIn(ph[name], joint, f"{name} drifted in K_JOINT")
            self.assertIn(ph[name], stock, f"{name} not found in stock source")
        for marker in ("float bv = s_bval[0];", "emissions[c * 3 + 0] = tokd;",
                       "s_ctl[3] = c + 1;", "s_ctl[4] = op;"):
            self.assertIn(marker, ph["reduce_control"])

    def test_crossing_buffers_declared(self):
        dec, joint, _ = build_all()
        # K_DEC exports and K_JOINT consumes the pj bus; both declare it
        self.assertIn("pj_out", dec)
        self.assertIn("pj_bus", joint)
        self.assertIn("threadgroup float16_t s_pj[640];", joint)
        # K_DEC keeps the carrier arrays the rebalance needs (s_relu,
        # s_bval, s_bidx, s_h1) and K_JOINT keeps its own argmax arrays
        for arr in ("threadgroup float16_t s_relu[640];",
                    "threadgroup float s_bval[1024];",
                    "threadgroup uint s_bidx[1024];"):
            self.assertIn(arr, dec)
        self.assertIn("threadgroup float s_dval[8];", joint)
        # documented deviation: K_DEC reloads state from device buffers,
        # it does not seed pj from dec_in
        self.assertNotIn("float16_t(dec_in[i])", dec)
        self.assertIn("s_hidden[i] = hidden_in[i];", dec)

    def test_run_continuation_in_k_joint(self):
        _, joint, _ = build_all()
        self.assertIn("ctl_out[5] = (s_ctl[", joint)
        self.assertIn("cfg[0]", joint)
        self.assertIn("cfg[1]", joint)
        # no break in the split: the driver loops instead
        self.assertNotIn(" break;", joint)

    def test_validity_and_same_frame_symbol_count_cross_dispatches(self):
        _, joint, _ = build_all()
        self.assertIn("s_ctl[2] = 1;", joint)
        self.assertIn("s_ctl[9] = ctl_in[6];", joint)
        self.assertIn("ctl_out[6] = (s_ctl[4] == 0) ? s_ctl[9] : 0;", joint)


class SyntheticScopeGuard(unittest.TestCase):
    """Pins the zero-emission synthetic instrument's coverage limits.

    The synthetic sweep exercises blank decisions only.  Emission,
    recurrent-state and blank-skip/end coverage lives in the decisive
    mode (real captured inputs, 104 emissions, full-state compare).
    """

    def test_synthetic_scope_markers_present_in_rendered_source(self):
        from coreml.vulkan_tdt_loop import _loop_glsl
        src = _loop_glsl()
        # emission path exists in the kernel (control writes emissions)
        self.assertIn("emissions[c * 3 + 0]", src)
        # recurrent pj state export exists
        self.assertIn("state[i] = float(s_pj[i]);", src)
        # blank skip/end machinery exists (frame advance + run gate)
        self.assertIn("s_ctl[0] = s_ctl[0] + adv;", src)
        self.assertIn("if (s_ctl[8] == 0) { break; }", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
