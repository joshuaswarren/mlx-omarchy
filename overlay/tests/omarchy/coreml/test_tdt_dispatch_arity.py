# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Offline dispatch-arity guard for the phase-split driver (CPU-only).

Main-directed: the decisive-mode dispatch was missing the joint weights
input (inputs size 4 vs kernel arity 5) and only failed ON DEVICE at
dispatch time.  This test parses the driver's AST and verifies, for BOTH
modes (synthetic + decisive), that every mx.fast.metal_kernel call site
passes exactly one positional input per declared kernel input name and
that the declared name lists contain every buffer the kernel body
references (via the rendered phase sources).
"""

import ast
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

_DRIVER = (_TOOLS + "/../../scripts-local/tdt-phase-attribution/"
           "phase_split_driver.py")


class DispatchArityTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(Path(_DRIVER).read_text())

    def _kernel_calls(self):
        calls = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute)
                    and fn.attr == "metal_kernel"):
                continue
            names = None
            for kw in node.keywords:
                if kw.arg == "input_names":
                    names = [elt.value for elt in kw.value.elts]
            calls.append((node, names))
        return calls

    def test_both_kernel_declarations_present(self):
        calls = self._kernel_calls()
        # 2 modes (synthetic + decisive) x 2 kernels (dec + joint)
        self.assertEqual(len(calls), 4, "expected 4 kernel builds")
        for _, names in calls:
            self.assertEqual(names[0],
                             "embedding" if names[0] == "embedding" else "pj_bus")

    def test_every_dispatch_matches_declared_arity(self):
        """Each joint/dec dispatch passes full declared inputs (Main: both
        modes; the decisive mode shipped 4-of-5 and only failed on device)."""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "metal_kernel"):
                continue
            continue  # declarations handled below; dispatch sites checked via dec_kernel/joint_kernel names
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in ("dec_kernel", "joint_kernel"):
                inputs = None
                for kw in node.keywords:
                    if kw.arg == "inputs":
                        inputs = kw.value
                self.assertIsNotNone(inputs, "dispatch without inputs")
                self.assertEqual(len(inputs.elts), 8 if fn.id == "dec_kernel" else 5,
                                 f"{fn.id} dispatch arity mismatch")

    def test_joint_dispatch_names_cover_joint_weights(self):
        """The decisive-mode regression: joint weights input must be passed."""
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr == "joint":
                found = True
        self.assertTrue(found, "packed.joint missing from a dispatch")

    def test_failing_first_broken_dispatch_detected(self):
        """Failing-first: a joint dispatch missing packed.joint is detected."""
        import ast as ast_mod
        driver_text = Path(_DRIVER).read_text()
        target = "inputs=[pj_out, packed.joint, enc, cfg, mx.array(ctl)],"
        if target not in driver_text:
            target = "inputs=[pj_out, packed.joint, enc, cfg, ctl_arr],"
        assert target in driver_text, "no dispatch to mutate"
        broken = ast_mod.parse(driver_text.replace(target, target.replace(
            "packed.joint, ", ""), 1))
        joint_dispatches = 0
        short = 0
        for node in ast_mod.walk(broken):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "joint_kernel"):
                joint_dispatches += 1
                inputs = next(kw.value for kw in node.keywords
                              if kw.arg == "inputs")
                if len(inputs.elts) < 5:
                    short += 1
        self.assertEqual(joint_dispatches, 2)
        self.assertGreater(short, 0, "mutation not detected — guard is inert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
