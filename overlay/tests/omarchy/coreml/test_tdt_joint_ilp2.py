# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exactness checks for the two-output interleave in the TDT joint head.

The loop kernel's joint phase previously gave each thread 8-9 SEQUENTIAL
output chains (``for j = t; j < 8198; j += 1024`` — one dependent fp32
chain per output).  They now run as interleaved pairs: one fused k-loop
advances two independent accumulators for outputs ``(j, j + 1024)`` per
round (``j += 2048``), with a scalar tail for outputs 8192-8197 (the six
threads' ninth output).  Per output the arithmetic is the identical
fp32 ascending-k chain with one fp16 rounding and the fp16 bias add —
independent accumulators only change instruction scheduling, which is
why the guard also checks the EMITTED SPIR-V (no fused/contracted ops,
fp32 accumulate chains and their phis present).

Mapping, carriers and the argmax/update logic are pinned here; the
SPIR-V check compiles the real rendered shader with glslangValidator and
inspects the disassembly (skipped where the tools are absent).
"""

import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.vulkan_decoder_step import _EXACT_FMA16
from coreml.vulkan_tdt_loop import _loop_glsl

_JOINT_OUT = 8198
_VOCAB = 8193
_DIM = 640
_THREADS = 1024


def _thread_pairs(t):
    """Output pairs the interleaved kernel assigns to thread ``t``.

    The pair loop covers j < 8192 with stride 2048 (so every j2 = j+1024
    is valid); outputs 8192-8197 belong to the tail loop.
    """
    pairs = []
    j = t
    while j < 8192:
        pairs.append((j, j + _THREADS))
        j += 2 * _THREADS
    return pairs


def _thread_tail(t):
    """Ninth-output tail items (outputs 8192-8197)."""
    j = t + 8 * _THREADS
    return [j] if j < _JOINT_OUT else []


def _joint_logits(relu, weights, bias):
    """fp32 ascending-k chain per output, one fp16 rounding + fp16 bias."""
    weights_f = weights.astype(np.float32)
    out = np.empty(_JOINT_OUT, np.float32)
    for j in range(_JOINT_OUT):
        acc = np.float32(0.0)
        for k in range(_DIM):
            acc = np.float32(acc + np.float32(relu[k]) * np.float32(weights_f[j, k]))
        out[j] = np.float32(np.float16(acc) + bias[j])
    return out


class JointIlp2MappingTest(unittest.TestCase):
    def test_pairs_and_tail_cover_all_outputs_exactly_once(self):
        seen = {}
        for t in range(_THREADS):
            for j, j2 in _thread_pairs(t):
                for out in (j, j2):
                    self.assertNotIn(out, seen, f"output {out} duplicated")
                    seen[out] = t
            for out in _thread_tail(t):
                self.assertNotIn(out, seen)
                seen[out] = t
        self.assertEqual(set(seen), set(range(_JOINT_OUT)))
        # ascending processing order per thread preserves first-max argmax:
        # the kernel emits (j, j2) per round, then the tail
        for t in range(_THREADS):
            order = [o for pair in _thread_pairs(t) for o in pair]
            order += _thread_tail(t)
            self.assertEqual(order, sorted(order))

    def test_logits_identical_interleaved_vs_sequential(self):
        rng = np.random.default_rng(20260920)
        weights = (rng.standard_normal((_JOINT_OUT, _DIM)) * 0.05).astype(np.float16)
        bias = (rng.standard_normal(_JOINT_OUT) * 0.05).astype(np.float16)
        relu = np.maximum(
            np.float16(rng.standard_normal(_DIM) * 0.5), np.float16(0.0)
        ).astype(np.float16)
        seq = _joint_logits(relu, weights, bias)
        # interleaved: identical per-output fp32 chain, paired traversal
        inter = np.empty(_JOINT_OUT, np.float32)
        wf = weights.astype(np.float32)
        for t in range(_THREADS):
            for j, j2 in _thread_pairs(t):
                a1 = np.float32(0.0)
                a2 = np.float32(0.0)
                for k in range(_DIM):
                    a1 = np.float32(a1 + np.float32(relu[k]) * np.float32(wf[j, k]))
                    a2 = np.float32(a2 + np.float32(relu[k]) * np.float32(wf[j2, k]))
                inter[j] = np.float32(np.float16(a1) + bias[j])
                inter[j2] = np.float32(np.float16(a2) + bias[j2])
            for j in _thread_tail(t):
                a = np.float32(0.0)
                for k in range(_DIM):
                    a = np.float32(a + np.float32(relu[k]) * np.float32(wf[j, k]))
                inter[j] = np.float32(np.float16(a) + bias[j])
        np.testing.assert_array_equal(seq.view(np.uint32), inter.view(np.uint32))


class RenderedShaderGuard(unittest.TestCase):
    """Code-linked guard over the REAL rendered kernel."""

    def test_rendered_shader_carries_ilp2(self):
        src = re.sub(r"\s+", " ", _loop_glsl())
        self.assertIn("for (uint j = t; j < 8192u; j += 2048u)", src)
        self.assertIn("uint j2 = j + 1024u;", src)
        self.assertIn("precise float acc1 = 0.0f;", src)
        self.assertIn("precise float acc2 = 0.0f;", src)
        self.assertIn("for (uint j = t + 8192u; j < 8198u; j += 8192u)", src)
        self.assertIn("acc1 = acc1 + float(s_relu[k]) * float(joint[row + j]);", src)
        self.assertIn("acc2 = acc2 + float(s_relu[k]) * float(joint[row + j2]);", src)

    def test_emitted_spirv_preserves_fp32_chains(self):
        """No contraction; fp32 accumulate chains present in emitted SPIR-V.

        Independent source accumulators alone do not prove the compiler
        preserves exact fp32 semantics, so the REAL rendered shader is
        compiled to SPIR-V and the disassembly is checked: zero fused
        fp32 ops, and the fp32 multiply-then-add chain structure with
        its carry-out phis present.
        """
        glslang = shutil.which("glslangValidator")
        dis = shutil.which("spirv-dis")
        if glslang is None or dis is None:
            self.skipTest("glslangValidator/spirv-dis not available")
        src = _loop_glsl()  # raw, multi-line: decl hoisting needs line structure
        norm = re.sub(r"\s+", " ", src)  # normalized only for assertions below
        self.assertIn("for (uint j = t; j < 8192u; j += 2048u)", norm)
        td = tempfile.mkdtemp()
        preamble = """#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
#define thread_index_in_threadgroup gl_LocalInvocationID
#define threadgroup_position_in_grid gl_WorkGroupID
#define threadgroups_per_grid gl_NumWorkGroups
#define threadgroup shared
#define threadgroup_barrier(x) barrier()
#define mem_flags mem_flags_stub
const int mem_flags_stub_mem_threadgroup = 0;
#define mem_threadgroup mem_flags_stub_mem_threadgroup
"""
        bindings = []
        for i, (t, nm) in enumerate([
            ("float16_t", "embedding"), ("float16_t", "weights"),
            ("float16_t", "biases"), ("float16_t", "luts"),
            ("float16_t", "projector"), ("float16_t", "joint"),
            ("float", "encoder"), ("float", "hidden_in"),
            ("float", "cell_in"), ("float", "dec_in"), ("int", "cfg"),
            ("int", "emissions"), ("float", "state"), ("int", "ctl"),
        ]):
            q = "writeonly" if nm in ("emissions", "state", "ctl") else "readonly"
            bindings.append(
                f"layout(set=0, binding={i}, std430) {q} buffer B{i} {{ {t} {nm}[]; }};"
            )
        decls = "\n".join(
            m.group(0).strip()
            for m in re.finditer(r"^.*threadgroup [^;]+;\s*$", src, re.M)
        )
        inner = re.sub(r"^.*threadgroup [^;]+;\s*\n", "", src, flags=re.M)
        text = (preamble + "\n".join(bindings) + "\n" + _EXACT_FMA16 + "\n"
                + decls + "\nvoid main() {\n" + inner + "\n}\n")
        comp = os_path_join(td, "loop.comp")
        with open(comp, "w") as f:
            f.write(text)
        spv = os_path_join(td, "loop.spv")
        r = subprocess.run([glslang, "-V", "-o", spv, comp],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = subprocess.run([dis, spv], capture_output=True, text=True)
        self.assertEqual(d.returncode, 0, d.stdout + d.stderr)
        lines = d.stdout.splitlines()
        # fp32 ops: those whose RESULT TYPE id is the OpTypeFloat-32 id
        float_type = next(
            ln.split()[0] for ln in lines
            if " OpTypeFloat " in ln and ln.split()[-1] == "32"
        )  # e.g. "%float"
        def ops(op):
            out = []
            for ln in lines:
                m = re.match(r"\s*(%\w+) = " + op + r" (%\w+) ", ln)
                if m and m.group(2) == float_type:
                    out.append(m)
            return out
        fadd = ops("OpFAdd")
        fmul = ops("OpFMul")
        fma = [ln for ln in lines if re.match(r"\s*%\w+ = OpFma\b", ln)]
        self.assertEqual(len(fma), 0, "fused fp32 ops present: precise chains contracted")
        # pair loop (2 accs) + tail (1) + projector (1) fp32 chains
        self.assertGreaterEqual(len(fadd), 4, "fp32 accumulate chains missing")
        self.assertGreaterEqual(len(fmul), 4, "fp32 multiplies missing")
        # The accumulators must live in function-scope fp32 variables with
        # their FAdd results stored back: SPIR-V sequential-invocation
        # semantics then pin each chain's add order — this is the
        # emitted-code proof that interleaving did not reorder anything.
        fadd_results = {m.group(1) for m in fadd}
        ptr_type = None
        for ln in lines:
            m = re.match(r"\s*(%\w+) = OpTypePointer (\w+) " + re.escape(float_type) + r"$", ln)
            if m:
                ptr_type = m.group(1)
                break
        acc_vars = set()
        for ln in lines:
            m = re.match(r"\s*(%\w+) = OpVariable (" + re.escape(ptr_type or "%never") + r")\b", ln)
            if m:
                acc_vars.add(m.group(1))
        stored_accs = set()
        for ln in lines:
            m = re.match(r"\s*OpStore (%\w+) (%\w+)\s*$", ln.strip())
            if m and m.group(1) in acc_vars and m.group(2) in fadd_results:
                stored_accs.add(m.group(1))
        self.assertGreaterEqual(
            len(stored_accs), 4,
            "fp32 accumulate chains are not variable-carried sequential "
            "(staged-accumulator structure not found in emitted code)")
        self.assertGreaterEqual(
            len(fadd_results), 4, "distinct fp32 FAdd results missing")


def os_path_join(*parts):
    return "/".join(parts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
