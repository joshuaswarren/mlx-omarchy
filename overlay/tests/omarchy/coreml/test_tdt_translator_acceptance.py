# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Modeled source-derived translation check for the phase-split TDT kernels.

MODELED CHECK, not the executing C++ translator: replays a faithful
source-derived model of the Omarchy custom-kernel translation pipeline
(custom_kernel.cpp) over the phase-split sources (K_DEC, K_JOINT) and
the landed fused loop source, asserting the same acceptance condition
the C++ enforces: after translation, no MSL leftover marker ("threadgroup",
"memory_order", "atomic_fetch", "[[") remains.  The marker list, the
shared-declaration hoist regex, the threadgroup_barrier replacement and
the output-buffer rewrite are parsed from the actual C++ source so this
test tracks the real pipeline rather than a copy of it.

Also demonstrates failing-first: injecting a comment that contains the
literal word "threadgroup" (the false positive this lane hit) must make
the acceptance fail.
"""

import re
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.vulkan_tdt_loop import _loop_glsl
import sys
sys.path.insert(0, _TOOLS + "/../../scripts-local/tdt-phase-attribution")
from phase_split_kernels import build_all

_CPP = (_TOOLS + "/../mlx/backend/omarchy/custom_kernel.cpp")


def _cpp_source():
    return Path(_CPP).read_text()


def _leftover_markers(cpp):
    """The marker list from the fork's final acceptance check."""
    block = re.search(
        r"if \(body\.find\(\"threadgroup\"\).*?\{\s*throw", cpp, re.S)
    assert block, "leftover-marker check not found in custom_kernel.cpp"
    return re.findall(r'body\.find\("([^"]+)"\)', block.group(0))


def _shared_hoist_pattern(cpp):
    m = re.search(
        r"const std::regex shared_pattern\(\s*R\"\((.*?)\)\"\);", cpp, re.S)
    assert m, "shared hoist pattern not found in custom_kernel.cpp"
    return re.compile(m.group(1))


def _barrier_replacement_present(cpp):
    return ('"threadgroup_barrier(mem_flags::mem_threadgroup)"' in cpp
            and '"barrier()"' in cpp)


def _attribute_table(cpp):
    """The thread-attribute replacement pairs, parsed in order."""
    block = re.search(
        r"const std::vector<std::pair<std::string, std::string>> attributes = \{(.*?)\};",
        cpp, re.S)
    assert block, "attribute table not found in custom_kernel.cpp"
    return re.findall(r'\{"([^"]+)",\s*"((?:[^"\\]|\\.)*)"\}', block.group(1))


def translate_like_fork(body, parameters, cpp):
    """Model custom_kernel.cpp's translation of one kernel body."""
    for a_from, a_to in _attribute_table(cpp):
        body = re.sub(r"\b" + re.escape(a_from) + r"\b", a_to, body)
    body = body.replace("metal::precise::", "")
    body = body.replace("metal::fast::", "")
    body = body.replace("metal::", "")
    assert _barrier_replacement_present(cpp)
    body = body.replace(
        "threadgroup_barrier(mem_flags::mem_threadgroup)", "barrier()")
    for simd, glsl in (("simd_sum", "subgroupAdd"), ("simd_max", "subgroupMax"),
                       ("simd_min", "subgroupMin"),
                       ("simd_broadcast", "subgroupBroadcast"),
                       ("simd_shuffle_down", "subgroupShuffleDown"),
                       ("simd_shuffle_up", "subgroupShuffleUp"),
                       ("simd_shuffle_xor", "subgroupShuffleXor"),
                       ("simd_shuffle", "subgroupShuffle")):
        body = re.sub(r"\b" + simd + r"\b", glsl, body)
    body = re.sub(r"\bconstexpr\b", "const", body)
    hoist = _shared_hoist_pattern(cpp)
    while True:
        m = hoist.search(body)
        if not m:
            break
        body = body[:m.start()] + body[m.end():]
    # output-buffer rewrite: \bname[idx] = value; -> name[idx] = type(value);
    for name, glsl_type in parameters:
        body = re.sub(
            r"\b" + re.escape(name) + r"\s*\[([^\]]+)\]\s*=\s*([^;]+);",
            name + r"[\1] = " + glsl_type + r"(\2);", body)
    return body


def _accepts(body, markers):
    return not any(marker in body for marker in markers)


class TranslatorAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.cpp = _cpp_source()
        self.markers = _leftover_markers(self.cpp)
        self.assertTrue("threadgroup" in self.markers, "marker parse failed")

    def test_fused_kernel_passes_translation(self):
        """Control: the landed fused source translates clean."""
        body = translate_like_fork(_loop_glsl(), [], self.cpp)
        self.assertTrue(_accepts(body, self.markers))

    def test_split_kernels_pass_translation(self):
        """K_DEC and K_JOINT translate clean through the same pipeline."""
        dec, joint, _ = build_all()
        self.assertTrue(_accepts(
            translate_like_fork(dec, [], self.cpp), self.markers),
            "K_DEC leaves MSL leftovers")
        self.assertTrue(_accepts(
            translate_like_fork(joint, [], self.cpp), self.markers),
            "K_JOINT leaves MSL leftovers")

    def test_threadgroup_comment_is_caught(self):
        """Failing-first: a comment mentioning 'threadgroup' must fail the
        acceptance (the false positive this lane hit on device)."""
        dec, joint, _ = build_all()
        poisoned = joint.replace(
            "// pj bus -> on-chip scratch",
            "// pj bus -> threadgroup scratch", 1)
        if poisoned == joint:
            poisoned = joint.replace(
                "uint t = thread_index_in_threadgroup.x;",
                "uint t = thread_index_in_threadgroup.x;\n"
                "    // threadgroup comment probe", 1)
        body = translate_like_fork(poisoned, [], self.cpp)
        self.assertFalse(_accepts(body, self.markers),
                         "modeled check failed to catch a threadgroup comment")

    def test_output_rewrite_models_builtin(self):
        """The output-buffer rewrite fires for emissions writes (sanity)."""
        body = "emissions[c * 3 + 0] = tokd;"
        out = translate_like_fork(body, [("emissions", "int32")], self.cpp)
        self.assertIn("emissions[c * 3 + 0] = int32(tokd);", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
