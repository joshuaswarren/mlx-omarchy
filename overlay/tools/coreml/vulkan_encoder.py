#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Run the pinned Parakeet encoder with the attention matmuls and the
attention-mask select on the ANE, and every other tensor op on the Apple GPU
through mlx-omarchy (Vulkan).

Phase 6 of docs/plans/2026-09-12-coreml-parakeet-ane-plan.md. Per transformer
layer, three ANE islands carry the attention:

  island A (bundle parakeet-encoder-island-attn-a-kt, 2 programs, 416 TDs)
      attention_scores_N = matmul(q_v, pos_kT)
      matmul_N           = matmul(q_scaled, k_headsT)
  island B (bundle parakeet-encoder-island-select-8head, 1 program, 5 TDs)
      attention_mask_N   = select(a = -inf fill, b = matrix_bd_N, cond = mask)
  island C (bundle parakeet-encoder-island-pv, 1 program, 208 TDs)
      attn_output_N      = matmul(probs, v_heads)

Everything else -- the subsampling conv stack, mask derivation, all layer
norms, both feed-forwards, the depthwise conv module, softmax, the 24
all-masked-rows selects, and the epilogue projector -- executes on ``mx.gpu``.

Island B is here because mil-hwx-compiler feature/h13-concat 7ab3eb5 fixed the
defect that kept it on the GPU in the two-island run: the H13 select program
under-declared its channel-3 scratch arena by 3x, so the cond-true half was
written and read past the declared surface. The rebuilt bundle declares 417
tiles / 6832128 bytes and returned 0 wrong lanes on jwm1. ``--islands`` selects
which islands are placed, so the two-island arm is reproducible from this same
script and the delta is attributable to placement rather than to a script edit.

This run does NOT establish that the ANE select is correct in general: the
golden clip's cond is uniformly false, so the -inf fill is never selected on
either device. See the receipt's coverage-gap section.

Section 43 (no CPU tensor fallback): every op is implemented with mlx.core
only and the runner raises on any op it cannot express in mx, so there is no
CPU arithmetic path to fall back to. Pinned constants are byte-reinterpreted
from the adapter's blob files straight into device arrays -- a load, not a
computation. Moving island tensors to and from the bounded ANE worker is
process I/O, also not arithmetic; both are accounted explicitly.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import sys
import re
import struct
import subprocess
import time
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

BLOB_MAGIC = 0xDEADBEEF

STMT = re.compile(
    r"^\s*(?P<type>tensor<[^>]*>|string|int32|bool|fp16|fp32)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_@]*)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TUPLE_STMT = re.compile(
    r"^\s*\((?P<results>tensor<[^)]*)\)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TYPE = re.compile(r"^tensor<\s*(?P<dtype>\w+)\s*,\s*\[(?P<shape>[^\]]*)\]>$")
BLOBFILE = re.compile(
    r'BLOBFILE\(path = string\("(?P<path>[^"]+)"\), offset = uint64\((?P<offset>\d+)\)\)'
)

MX_DTYPES = {"fp16": mx.float16, "fp32": mx.float32, "int32": mx.int32, "bool": mx.bool_}
NP_DTYPES = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "bool": np.bool_}

# Identify the spliced islands by the MIL result names they produce.
RE_SCORES = re.compile(r"^attention_scores_\d+_cast_fp16$")
RE_CONTENT = re.compile(r"^matmul_\d+_cast_fp16$")
RE_ATTN_OUT = re.compile(r"^attn_output_\d+_cast_fp16$")
# Island B. Two traps here. The MIL also carries 24 all-masked-rows selects
# named input_N, so the name match excludes them and the shape assertion in
# _index_islands is the second gate. And the 24th mask select is
# attention_mask_cast_fp16, with no ordinal at all -- Core ML drops the suffix
# on the last of a repeated family -- so the ordinal is optional. A \d+-only
# pattern silently finds 23 of 24 and the balance check is what catches it.
RE_MASK_SELECT = re.compile(r"^attention_mask(_\d+)?_cast_fp16$")
ISLAND_B_SHAPE = (1, 8, 375, 375)



@cache
def _leftover_chain_kernel():
    """Reduces the batched matmul's fp32 block partials with the landed
    leftover-linear rounding: every 16-wide block's partial rounds to fp16,
    then accumulates in fp16 ascending. One thread per output element, so
    the chain inside the kernel is the same strictly sequential fp16 sum
    the 5688f8bd chunk loop performed dispatch by dispatch."""
    return mx.fast.metal_kernel(
        name="encoder_leftover_fp16_chain_f32",
        input_names=["partials"],
        output_names=["reduced"],
        source="""
            uint n = partials_shape[2];
            uint mn = partials_shape[1] * n;
            uint blocks = partials_shape[0];
            uint index = thread_position_in_grid.x;
            half acc = half(partials[index]);
            for (uint block = 1u; block < blocks; ++block) {
                acc = acc + half(partials[index + block * mn]);
            }
            reduced[index] = acc;
        """,
        compile_options={"math_mode": "safe"},
    )




@cache
def _silu_kernel():
    """silu in one dispatch. The fp32 math and the single fp16 rounding at
    the end replicate the two-dispatch chain exactly: mx.sigmoid lowers to
    `1.0 / (1.0 + exp(-x))` (elementwise.comp case 5) and the product stays
    fp32 until the op boundary cast. Buffer names never use `x` or `y`: the
    translator macros every name (`#define src _b0.data`) and would eat the
    `.x` swizzle of thread_position_in_grid."""
    return mx.fast.metal_kernel(
        name="encoder_silu_fused",
        input_names=["src"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            float v = float(src[index]);
            dst[index] = half(v * (1.0 / (1.0 + exp(-v))));
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _glu_kernel():
    """Conv-module GLU in one dispatch: `a * sigmoid(b)`.
    The gate rounds fp32->fp16 before the product, because the stored
    sigmoid result the mul consumes was itself an fp16 tensor. The rounding
    must go through packHalf2x16: an fp16_t local keeps fp32 precision in
    the following arithmetic (the fused_chain.comp M1-equality lesson)."""
    return mx.fast.metal_kernel(
        name="encoder_glu_fused",
        input_names=["lhs", "rhs"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            float s = 1.0 / (1.0 + exp(-float(rhs[index])));
            float gate = float(unpackHalf2x16(packHalf2x16(vec2(s, 0.0))).x);
            dst[index] = half(float(lhs[index]) * gate);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_cast_kernel():
    return mx.fast.metal_kernel(
        name="encoder_ln_cast",
        input_names=["src"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            dst[index] = float(src[index]);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_sq_kernel():
    """(xf - mean)^2 with the row mean broadcast per 1024-wide row. Same
    sub and square elementwise ops the separate dispatches ran, same fp32
    storage, so the following mx.mean reduce sees identical bits."""
    return mx.fast.metal_kernel(
        name="encoder_ln_centered_square",
        input_names=["xf", "mu"],
        output_names=["t2"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 1024u;
            float t = xf[index] - mu[row];
            t2[index] = t * t;
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_tail_kernel():
    """LayerNorm tail in one dispatch. mx.rsqrt lowers to inversesqrt
    (elementwise.comp case 8); the mul-mul-add order and the single fp16
    rounding at the end match the dispatch chain statement for statement."""
    return mx.fast.metal_kernel(
        name="encoder_ln_tail",
        input_names=["xf", "mu", "va", "ga", "be", "ep"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 1024u;
            uint col = index % 1024u;
            float t = xf[index] - mu[row];
            float rstd = inversesqrt(va[row] + ep[0]);
            precise float pr = t * rstd * float(ga[col]);
            pr = pr + float(be[col]);
            dst[index] = half(pr);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _sm_exp_kernel():
    """shifted exp in one dispatch. The row max runs as ReduceF16 over the
    fp16 input: max is order-insensitive and exact, so widening its fp16
    result is the same fp32 value the cast-then-ReduceF32 arm produced."""
    return mx.fast.metal_kernel(
        name="encoder_softmax_exp",
        input_names=["src", "rmax"],
        output_names=["expd"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 375u;
            expd[index] = exp(float(src[index]) - float(rmax[row]));
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _sm_div_kernel():
    return mx.fast.metal_kernel(
        name="encoder_softmax_div",
        input_names=["expd", "rsum"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 375u;
            dst[index] = half(expd[index] / rsum[row]);
        """,
        compile_options={"math_mode": "safe"},
    )



class EncoderRunError(RuntimeError):
    """The run cannot continue; the reason is named."""


class _TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("gpu_primitive_dispatches", ctypes.c_uint64),
        ("vk_submissions", ctypes.c_uint64),
        ("vk_buffer_copies", ctypes.c_uint64),
        ("vk_buffer_fills", ctypes.c_uint64),
        ("vk_compute_dispatches", ctypes.c_uint64),
        ("omarchy_finalize_calls", ctypes.c_uint64),
        ("commit_calls_with_work", ctypes.c_uint64),
        ("commit_calls_noop", ctypes.c_uint64),
    ]


@cache
def _trace_function():
    distribution = importlib.metadata.distribution("mlx-omarchy")
    library_path = distribution.locate_file("mlx/lib/libmlx.so")
    library = ctypes.CDLL(str(library_path))
    function = library.mlx_omarchy_trace_snapshot
    function.argtypes = [ctypes.POINTER(_TraceSnapshot)]
    function.restype = None
    return function


def trace_snapshot() -> dict[str, int]:
    snapshot = _TraceSnapshot()
    _trace_function()(ctypes.byref(snapshot))
    return {name: int(getattr(snapshot, name)) for name, _ in snapshot._fields_}


def split_top(text: str) -> list[str]:
    out, depth, start, quoted = [], 0, 0, False
    for i, ch in enumerate(text):
        if quoted:
            if ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [item.strip() for item in out]


def parse_kwargs(text: str) -> dict[str, str]:
    result = {}
    for item in split_top(text):
        if not item:
            continue
        key, _, value = item.partition("=")
        result[key.strip()] = value.strip()
    return result


def parse_type(text: str):
    match = TYPE.match(text)
    if match:
        name = match.group("dtype").lower()
        shape_text = match.group("shape").strip()
        shape = tuple(int(i) for i in shape_text.split(",")) if shape_text else ()
        return name, shape
    return text.strip().lower(), ()


class Blobs:
    """Reader for the blob-v2 files overlay/tools/coreml/mil_adapter.py emits."""

    def __init__(self, model_root: Path):
        self.root = model_root
        self._maps: dict[str, np.memmap] = {}

    def _map(self, path: str) -> np.memmap:
        name = path.replace("@model_path/", "")
        if name not in self._maps:
            self._maps[name] = np.memmap(self.root / name, dtype=np.uint8, mode="r")
        return self._maps[name]

    def read_bytes(self, path: str, offset: int, want: int) -> memoryview:
        raw = self._map(path)
        magic, _storage, length, payload = struct.unpack_from(
            "<IIQQ", raw[offset : offset + 24].tobytes(), 0
        )
        if magic != BLOB_MAGIC:
            raise EncoderRunError(f"blob magic {magic:#x} at {path}:{offset}")
        if want > length:
            raise EncoderRunError(
                f"blob at {path}:{offset} holds {length} bytes, want {want}"
            )
        return memoryview(raw[payload : payload + want].tobytes())


class Statement:
    __slots__ = ("index", "names", "op", "kwargs", "attrs", "dtype", "shape", "done")

    def __init__(self, index, names, op, kwargs, attrs, dtype, shape):
        self.index = index
        self.names = names
        self.op = op
        self.kwargs = kwargs
        self.attrs = attrs
        self.dtype = dtype
        self.shape = shape
        self.done = False


# The three island bundles the encoder handlers submit to. The resident
# session loads every one of them once, up front, exactly like the launch
# path loads its bundle per submit.
RESIDENT_BUNDLES = (
    "island-attn-a-kt",
    "island-select-8head-scratch417",
    "island-pv",
)


class AneIsland:
    """One bounded submit to the physical ANE through mlx-omarchy-ane-worker.

    ANE_ISLAND_MODE picks the path:

    launch (default) -- one process per submit: the worker CLI takes a
    single bundle and a single input set, so a 24-layer encoder needs one
    launch per island per layer.

    resident-batch -- one private ``--serve`` worker for the whole pass and
    ONE batch scope: every layer's island submit is a round inside a single
    deadline-bounded batch (ANE_ISLAND_BATCH_DEADLINE_MS, default 120000).
    One process, one bundle load, one bounded unit for all 72 submits; a
    failed round or a deadline miss ends the session and is reported, never
    retried.
    """

    def __init__(self, worker: Path, libane: Path, bundles: Path, scratch: Path,
                 deadline_ms: int = 20000):
        self.worker = worker
        self.libane = libane
        self.bundles = bundles
        self.scratch = scratch
        self.deadline_ms = deadline_ms
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.submissions = 0
        self.worker_starts = 0
        self.rounds = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.exec_ns = 0
        self.timeouts = 0
        self.batch_open_ns = 0
        self.log: list[dict] = []
        self._mode = os.environ.get("ANE_ISLAND_MODE", "launch")
        if self._mode not in ("launch", "resident-batch"):
            raise EncoderRunError(
                f"ANE_ISLAND_MODE {self._mode!r} is not launch or resident-batch"
            )
        self._batch_deadline_ms = int(
            os.environ.get("ANE_ISLAND_BATCH_DEADLINE_MS", "120000")
        )
        self._session = None

    def close(self) -> None:
        """Release the resident session; a no-op on the launch path."""
        if self._session is None:
            return
        session, self._session = self._session, None
        try:
            session.end_batch()
        except Exception as error:
            raise EncoderRunError(f"resident batch close failed: {error}") from error
        try:
            session.close()
        except Exception as error:
            raise EncoderRunError(f"resident session close failed: {error}") from error

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ane_resident import ResidentAneWorker
        started = time.monotonic_ns()
        session = ResidentAneWorker(
            worker=Path(self.worker),
            libane=Path(self.libane),
            bundles={name: Path(self.bundles) / name for name in RESIDENT_BUNDLES},
            scratch=Path(self.scratch),
            deadline_ms=self.deadline_ms,
        )
        session.start()
        session.begin_batch(self._batch_deadline_ms)
        self.batch_open_ns = time.monotonic_ns() - started
        self.worker_starts += 1
        self.submissions += 1
        self._session = session
        return session

    def _submit_resident(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        """One round inside the open batch: same bytes, no files, one session."""
        session = self._ensure_session()
        payload = {}
        in_bytes = 0
        for name, value in inputs.items():
            mx.eval(value)
            raw = np.ascontiguousarray(np.asarray(value)).tobytes()
            payload[name] = raw
            in_bytes += len(raw)
        out_names = list(outputs)
        started = time.monotonic_ns()
        try:
            results = session.submit(bundle, tag, payload, out_names)
        except Exception as error:
            raise EncoderRunError(
                f"ANE batch round {tag} ({bundle}) failed: {error}"
            ) from error
        elapsed = time.monotonic_ns() - started
        self.rounds += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        record = {"tag": tag, "bundle": bundle, "elapsed_ns": elapsed, "round": True}
        out_bytes = 0
        packed = {}
        for name, (shape, dtype_name) in outputs.items():
            raw = results[name]
            count = 1
            for dim in shape:
                count *= dim
            expect = count * np.dtype(NP_DTYPES[dtype_name]).itemsize
            if len(raw) != expect:
                raise EncoderRunError(
                    f"ANE output {name} for {tag} is {len(raw)} bytes, want {expect}"
                )
            out_bytes += len(raw)
            host = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name], count=count)
            packed[name] = mx.array(host).reshape(shape)
        record["input_bytes"] = in_bytes
        record["output_bytes"] = out_bytes
        self.output_bytes += out_bytes
        self.log.append(record)
        return packed

    def submit(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        """inputs: name -> mx array. outputs: name -> (shape, dtype name)."""
        if self._mode == "resident-batch":
            return self._submit_resident(bundle, tag, inputs, outputs)
        return self._submit_launch(bundle, tag, inputs, outputs)

    def _submit_launch(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        run_dir = self.scratch / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.worker),
            "--bundle", str(self.bundles / bundle),
            "--libane", str(self.libane),
            "--deadline-ms", str(self.deadline_ms),
            "--iterations", "1",
        ]
        in_bytes = 0
        for name, value in inputs.items():
            mx.eval(value)
            raw = np.ascontiguousarray(np.asarray(value))
            path = run_dir / f"in_{name}.bin"
            path.write_bytes(raw.tobytes())
            in_bytes += raw.nbytes
            argv += ["--input", f"{name}={path}"]
        saved = {}
        for name in outputs:
            path = run_dir / f"out_{name}.bin"
            saved[name] = path
            argv += ["--save", f"{name}={path}"]

        self.worker_starts += 1
        started = time.monotonic_ns()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=None)
        elapsed = time.monotonic_ns() - started
        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes

        record = {
            "tag": tag,
            "bundle": bundle,
            "exit": proc.returncode,
            "elapsed_ns": elapsed,
            "stdout": proc.stdout.strip().splitlines(),
            "stderr": proc.stderr.strip().splitlines(),
        }
        self.log.append(record)
        if proc.returncode != 0:
            if "errno" in proc.stderr and "110" in proc.stderr:
                self.timeouts += 1
            raise EncoderRunError(
                f"ANE submit {tag} ({bundle}) exited {proc.returncode}: "
                f"{proc.stderr.strip()[:400]}"
            )

        results = {}
        out_bytes = 0
        for name, (shape, dtype_name) in outputs.items():
            raw = saved[name].read_bytes()
            count = 1
            for dim in shape:
                count *= dim
            expect = count * np.dtype(NP_DTYPES[dtype_name]).itemsize
            if len(raw) != expect:
                raise EncoderRunError(
                    f"ANE output {name} for {tag} is {len(raw)} bytes, want {expect}"
                )
            out_bytes += len(raw)
            host = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name], count=count)
            results[name] = mx.array(host).reshape(shape)
            saved[name].unlink()
        self.output_bytes += out_bytes
        record["input_bytes"] = in_bytes
        record["output_bytes"] = out_bytes
        for path in run_dir.glob("in_*.bin"):
            path.unlink()
        return results


class EncoderRunner:
    def __init__(self, mil_path: Path, model_root: Path, island: AneIsland | None,
                 placed: frozenset[str] = frozenset("ABC")):
        self.text = mil_path.read_text()
        self.blobs = Blobs(model_root)
        self.island = island
        self.placed = placed if island is not None else frozenset()
        self.values: dict[str, mx.array] = {}
        self.meta: dict[str, object] = {}
        self.statements: list[Statement] = []
        self.producer: dict[str, Statement] = {}
        self.const_stmt: dict[str, Statement] = {}
        self.executed = 0
        self.gpu_ops = 0
        self.ane_ops = 0
        self.cpu_tensor_events = 0
        self.cond_census: dict | None = None
        self.glu_fusions: dict[int, tuple[str, str]] = {}
        self.glu_sigmoid_done: set[int] = set()
        self._parse()
        self._index_islands()
        self._index_fusions()
        self._last_use()

    # ---------------------------------------------------------------- parsing

    def _parse(self) -> None:
        index = 0
        for line in self.text.split("\n"):
            tup = TUPLE_STMT.match(line)
            if tup is not None:
                names = [
                    item.rsplit(" ", 1)[1] for item in split_top(tup.group("results"))
                ]
                stmt = Statement(
                    index, names, tup.group("op"),
                    parse_kwargs(tup.group("args")), tup.group("attrs") or "", None, None,
                )
            else:
                match = STMT.match(line)
                if match is None:
                    continue
                dtype, shape = parse_type(match.group("type"))
                stmt = Statement(
                    index, [match.group("name")], match.group("op"),
                    parse_kwargs(match.group("args")), match.group("attrs") or "",
                    dtype, shape,
                )
            self.statements.append(stmt)
            for name in stmt.names:
                self.producer[name] = stmt
                if stmt.op == "const":
                    self.const_stmt[name] = stmt
            index += 1

    def _index_islands(self) -> None:
        scores, content, attn_out, mask_select = [], [], [], []
        for stmt in self.statements:
            name = stmt.names[0]
            if stmt.op == "matmul":
                if RE_SCORES.match(name):
                    scores.append(stmt)
                elif RE_CONTENT.match(name):
                    content.append(stmt)
                elif RE_ATTN_OUT.match(name):
                    attn_out.append(stmt)
            elif stmt.op == "select" and RE_MASK_SELECT.match(name):
                if tuple(stmt.shape) != ISLAND_B_SHAPE:
                    raise EncoderRunError(
                        f"{name} is {tuple(stmt.shape)}, island B bundle is "
                        f"{ISLAND_B_SHAPE}"
                    )
                mask_select.append(stmt)
        if not (len(scores) == len(content) == len(attn_out) == len(mask_select)):
            raise EncoderRunError(
                "island sets unbalanced: "
                f"{len(scores)}/{len(content)}/{len(mask_select)}/{len(attn_out)}"
            )
        self.layers = len(scores)
        # island A pairs the i-th rel-pos matmul with the i-th content matmul.
        self.island_a = {s.index: (i, s, c) for i, (s, c) in enumerate(zip(scores, content))}
        self.island_a_partner = {c.index: s.index for s, c in zip(scores, content)}
        self.island_b = {s.index: (i, s) for i, s in enumerate(mask_select)}
        self.island_c = {o.index: (i, o) for i, o in enumerate(attn_out)}

    def _index_fusions(self) -> None:
        """Find the conv-module GLU: sigmoid(split_1) consumed by exactly one
        mul whose other operand is the sibling split half. That mul becomes
        one fused dispatch and the sigmoid statement never executes."""
        consumers: dict[str, list[Statement]] = {}
        for stmt in self.statements:
            for token in stmt.kwargs.values():
                for name in self._operand_names(token):
                    consumers.setdefault(name, []).append(stmt)
        for stmt in self.statements:
            if stmt.op != "sigmoid":
                continue
            users = consumers.get(stmt.names[0], [])
            if len(users) != 1 or users[0].op != "mul":
                continue
            mul = users[0]
            b_name = stmt.kwargs["x"].strip()
            a_names = [
                token.strip()
                for key, token in mul.kwargs.items() if key in ("x", "y")
                and token.strip() != stmt.names[0]
            ]
            if len(a_names) != 1:
                continue
            a_name = a_names[0]
            split = self.producer.get(a_name)
            if split is None or split is not self.producer.get(b_name):
                continue
            if split.op != "split":
                continue
            parent = self.producer.get(split.kwargs["x"].strip())
            if parent is None or parent.shape is None or len(parent.shape) != 3:
                continue
            # Only a channel-axis split of a contiguous 3D parent leaves both
            # halves contiguous, which the flat kernel indexing requires.
            axis = self.ints(split.kwargs["axis"])[0]
            if axis % len(parent.shape) != 1 or self.ints(
                split.kwargs["num_splits"]
            )[0] != 2:
                continue
            self.glu_fusions[mul.index] = (a_name, b_name)
            self.glu_sigmoid_done.add(stmt.index)

    def _last_use(self) -> None:
        """Index of the final statement that reads each name, so the runner can
        release device memory as it goes. The encoder's constants alone are
        1.2 GB of fp16; holding every intermediate would not fit."""
        self.last_use: dict[str, int] = {}
        for stmt in self.statements:
            for token in stmt.kwargs.values():
                for name in self._operand_names(token):
                    self.last_use[name] = max(self.last_use.get(name, -1), stmt.index)
        # The fused mul reads the split sibling after the skipped sigmoid
        # statement, so that operand must live until the mul, not the sigmoid.
        for mul_index, (_a, b_name) in self.glu_fusions.items():
            self.last_use[b_name] = max(
                self.last_use.get(b_name, -1), mul_index
            )

    def _operand_names(self, token: str) -> list[str]:
        token = token.strip()
        if token.startswith("("):
            return [t.strip() for t in split_top(token.strip("() ")) if t.strip() in self.producer]
        return [token] if token in self.producer else []

    # ------------------------------------------------------------- resolution

    def ensure(self, name: str) -> None:
        """Execute the statement producing ``name`` if it has not run yet.

        Island A's second program needs q_scaled and k_headsT, which the MIL
        schedules after the rel-pos matmul. They are pure functions of values
        already produced, so pulling them forward is semantically identical.
        """
        if name in self.values or name in self.meta:
            return
        stmt = self.producer.get(name)
        if stmt is None:
            raise EncoderRunError(f"unresolved operand {name!r}")
        if stmt.done:
            return
        for token in stmt.kwargs.values():
            for operand in self._operand_names(token):
                self.ensure(operand)
        self.execute(stmt)

    def tensor(self, token: str) -> mx.array:
        token = token.strip()
        self.ensure(token)
        if token in self.values:
            return self.values[token]
        if token in self.meta:
            value = self.meta[token]
            return mx.array(value)
        raise EncoderRunError(f"unresolved tensor {token!r}")

    def scalar(self, token: str):
        """Host-side metadata (shapes, axes, perms, masks, epsilon, flags).

        Only constants are read this way; the pinned encoder derives no shape
        from a computed tensor, so this never forces a device sync on a value
        the run computed.
        """
        token = token.strip()
        self.ensure(token)
        if token in self.meta:
            return self.meta[token]
        raise EncoderRunError(f"{token!r} is not a host-side constant")

    def ints(self, token: str) -> list[int]:
        value = self.scalar(token)
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(value)]

    def bools(self, token: str) -> list[bool]:
        value = self.scalar(token)
        if isinstance(value, (list, tuple)):
            return [bool(v) for v in value]
        return [bool(value)]

    # ------------------------------------------------------------------ const

    def eval_const(self, stmt: Statement) -> None:
        name = stmt.names[0]
        dtype_name, shape = stmt.dtype, stmt.shape
        kwargs = parse_kwargs(stmt.attrs)
        value_text = kwargs["val"]
        if dtype_name == "string":
            self.meta[name] = re.search(r'"([^"]*)"', value_text).group(1)
            return
        blob = BLOBFILE.search(value_text)
        if blob is None:
            inner = value_text[value_text.index("(") + 1 : value_text.rindex(")")]
            payload = inner.strip("[] ")
            if dtype_name == "bool":
                parsed = [item.strip() == "true" for item in payload.split(",")]
            elif dtype_name == "int32":
                parsed = [int(item) for item in payload.split(",")]
            else:
                parsed = [float(item) for item in payload.split(",")]
            self.meta[name] = parsed if shape else parsed[0]
            self.values[name] = mx.array(
                np.array(parsed, dtype=NP_DTYPES[dtype_name]).reshape(shape)
            )
            return
        count = 1
        for dim in shape:
            count *= dim
        np_dtype = NP_DTYPES[dtype_name]
        raw = self.blobs.read_bytes(
            blob.group("path"), int(blob.group("offset")),
            count * np.dtype(np_dtype).itemsize,
        )
        host = np.frombuffer(raw, dtype=np_dtype, count=count)
        self.values[name] = mx.array(host).reshape(shape) if shape else mx.array(host[0])
        if not shape:
            # Scalar weights are also pad values / multipliers read as metadata.
            self.meta[name] = float(host[0]) if np_dtype != np.int32 else int(host[0])

    # --------------------------------------------------------------- dispatch

    def execute(self, stmt: Statement) -> None:
        if stmt.done:
            return
        stmt.done = True
        if stmt.op == "const":
            self.eval_const(stmt)
            self.executed += 1
            return

        if "A" in self.placed and stmt.index in self.island_a:
            self._run_island_a(stmt)
            return
        if "B" in self.placed and stmt.index in self.island_b:
            self._run_island_b(stmt)
            return
        if "C" in self.placed and stmt.index in self.island_c:
            self._run_island_c(stmt)
            return
        if (
            "A" in self.placed
            and stmt.index in self.island_a_partner
            and stmt.names[0] not in self.values
        ):
            # The content matmul is produced by island A's second program; the
            # splice runs at the rel-pos statement, which comes first.
            self.ensure(self.statements[self.island_a_partner[stmt.index]].names[0])
            if stmt.names[0] in self.values:
                self.executed += 1
                return

        if stmt.op == "sigmoid" and stmt.index in self.glu_sigmoid_done:
            # Consumed only by the fused mul; its value is never read.
            self.executed += 1
            return
        if stmt.index in self.glu_fusions:
            a_name, b_name = self.glu_fusions[stmt.index]
            a = self.tensor(a_name)
            b = self.tensor(b_name)
            n = a.size
            out = _glu_kernel()(
                inputs=[a, b],
                output_shapes=[(n,)],
                output_dtypes=[mx.float16],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            self.values[stmt.names[0]] = mx.reshape(out, a.shape)
            self.executed += 1
            self.gpu_ops += 1
            return
        if stmt.op == "split":
            parts = self.apply_split(stmt)
            if len(parts) != len(stmt.names):
                raise EncoderRunError(
                    f"split produced {len(parts)} of {len(stmt.names)}"
                )
            self.values.update(zip(stmt.names, parts))
        else:
            self.values[stmt.names[0]] = self.apply(stmt)
        self.executed += 1
        self.gpu_ops += 1

    def apply_split(self, stmt: Statement) -> list[mx.array]:
        axis = self.ints(stmt.kwargs["axis"])[0]
        count = self.ints(stmt.kwargs["num_splits"])[0]
        return list(mx.split(self.tensor(stmt.kwargs["x"]), count, axis=axis))

    # ----------------------------------------------------------- ANE islands

    def _run_island_a(self, stmt: Statement) -> None:
        layer, scores_stmt, content_stmt = self.island_a[stmt.index]
        if self.bools(scores_stmt.kwargs["transpose_x"])[0] or self.bools(
            scores_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(f"layer {layer} rel-pos matmul is transposed")
        if self.bools(content_stmt.kwargs["transpose_x"])[0] or not self.bools(
            content_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(
                f"layer {layer} content matmul transpose flags are not (x=false, y=true)"
            )
        q_v = self.tensor(scores_stmt.kwargs["x"])
        pos_kT = self.tensor(scores_stmt.kwargs["y"])
        q_scaled = self.tensor(content_stmt.kwargs["x"])
        # The bundle binds the already-transposed key tensor.
        k_headsT = mx.swapaxes(self.tensor(content_stmt.kwargs["y"]), -1, -2)
        results = self.island.submit(
            "island-attn-a-kt", f"L{layer:02d}-A",
            {"q_v": q_v, "pos_kT": pos_kT, "q_scaled": q_scaled, "k_headsT": k_headsT},
            {
                "attention_scores_1": (scores_stmt.shape, "fp16"),
                "matmul_0": (content_stmt.shape, "fp16"),
            },
        )
        self.values[scores_stmt.names[0]] = results["attention_scores_1"]
        self.values[content_stmt.names[0]] = results["matmul_0"]
        content_stmt.done = True
        self.executed += 2
        self.ane_ops += 2

    def _run_island_b(self, stmt: Statement) -> None:
        """The -inf attention-mask select, one submit per layer.

        The MIL writes ``a`` as an fp16 scalar and ``cond`` as [1, 1, 375, 375],
        relying on broadcast. The compiled bundle binds all three operands at
        the full [1, 8, 375, 375], so both are expanded here. That expansion is
        marshalling, not arithmetic: it replicates bytes the run already holds,
        the same way island A hands the bundle an already-transposed key.
        """
        layer, sel_stmt = self.island_b[stmt.index]
        fill = self.tensor(sel_stmt.kwargs["a"])
        cond = self.tensor(sel_stmt.kwargs["cond"])
        matrix_bd = self.tensor(sel_stmt.kwargs["b"])
        if fill.shape != ():
            raise EncoderRunError(f"layer {layer} select fill is {fill.shape}, want scalar")
        if tuple(cond.shape) != (1, 1, 375, 375):
            raise EncoderRunError(f"layer {layer} select cond is {tuple(cond.shape)}")
        if tuple(matrix_bd.shape) != ISLAND_B_SHAPE:
            raise EncoderRunError(f"layer {layer} select b is {tuple(matrix_bd.shape)}")
        if cond.dtype != mx.bool_:
            raise EncoderRunError(f"layer {layer} select cond dtype {cond.dtype}")
        if self.cond_census is None:
            # Measured once, not inherited: whether any lane actually selects
            # the -inf fill decides what this run can claim about the select.
            # var_373 is one tensor shared by all 24 layers, so layer 0 is the
            # whole story, and this reads the host copy rather than adding a
            # device reduction that would perturb the GPU counters.
            host_cond = np.asarray(cond)
            self.cond_census = {
                "cond_tensor": sel_stmt.kwargs["cond"],
                "shape": list(host_cond.shape),
                "elements": int(host_cond.size),
                "true_lanes": int(host_cond.sum()),
                "broadcast_elements": int(host_cond.size) * ISLAND_B_SHAPE[1],
                "broadcast_true_lanes": int(host_cond.sum()) * ISLAND_B_SHAPE[1],
                # repr, not float: -inf is not valid JSON.
                "fill_value": repr(np.asarray(fill).astype(np.float16).item()),
                "fill_bits": "0x%04X" % int(
                    np.asarray(fill).astype(np.float16).view(np.uint16)
                ),
                "shared_across_layers": True,
            }
        results = self.island.submit(
            "island-select-8head-scratch417", f"L{layer:02d}-B",
            {
                "ninf_rt": mx.contiguous(mx.broadcast_to(fill, ISLAND_B_SHAPE)),
                "matrix_bd_5": matrix_bd,
                "cond": mx.contiguous(mx.broadcast_to(cond, ISLAND_B_SHAPE)),
            },
            {"attention_mask_9": (sel_stmt.shape, "fp16")},
        )
        self.values[sel_stmt.names[0]] = results["attention_mask_9"]
        self.executed += 1
        self.ane_ops += 1

    def _run_island_c(self, stmt: Statement) -> None:
        layer, out_stmt = self.island_c[stmt.index]
        if self.bools(out_stmt.kwargs["transpose_x"])[0] or self.bools(
            out_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(f"layer {layer} PV matmul is transposed")
        results = self.island.submit(
            "island-pv", f"L{layer:02d}-C",
            {
                "probs": self.tensor(out_stmt.kwargs["x"]),
                "v_heads": self.tensor(out_stmt.kwargs["y"]),
            },
            {"attn_output_1": (out_stmt.shape, "fp16")},
        )
        self.values[out_stmt.names[0]] = results["attn_output_1"]
        self.executed += 1
        self.ane_ops += 1

    # ------------------------------------------------------------------- ops

    def apply(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor

        if op == "cast":
            target = self.scalar(kwargs["dtype"])
            if target not in MX_DTYPES:
                raise EncoderRunError(f"cast dtype {target}")
            return tensor(kwargs["x"]).astype(MX_DTYPES[target])
        if op == "expand_dims":
            out = tensor(kwargs["x"])
            for axis in sorted(self.ints(kwargs["axes"])):
                out = mx.expand_dims(out, axis)
            return out
        if op == "squeeze":
            return mx.squeeze(tensor(kwargs["x"]), axis=tuple(self.ints(kwargs["axes"])))
        if op in ("reduce_sum", "reduce_min", "reduce_max"):
            axes = tuple(self.ints(kwargs["axes"]))
            keep = self.bools(kwargs["keep_dims"])[0]
            fn = {"reduce_sum": mx.sum, "reduce_min": mx.min, "reduce_max": mx.max}[op]
            return fn(tensor(kwargs["x"]), axis=axes, keepdims=keep)
        if op in ("add", "sub", "mul"):
            x, y = tensor(kwargs["x"]), tensor(kwargs["y"])
            if x.dtype == mx.bool_ and y.dtype == mx.bool_:
                if op != "mul":
                    raise EncoderRunError(f"bool {op}")
                return mx.logical_and(x, y)
            if x.dtype == mx.int32 and y.dtype == mx.int32:
                raw = {"add": x + y, "sub": x - y, "mul": x * y}[op]
                return raw.astype(mx.int32)
            fx, fy = x.astype(mx.float32), y.astype(mx.float32)
            return {"add": fx + fy, "sub": fx - fy, "mul": fx * fy}[op].astype(mx.float16)
        if op == "floor_div":
            x, y = tensor(kwargs["x"]), tensor(kwargs["y"])
            if x.dtype == mx.int32 and y.dtype == mx.int32:
                return mx.floor(x.astype(mx.float32) / y.astype(mx.float32)).astype(mx.int32)
            return mx.floor(
                x.astype(mx.float32) / y.astype(mx.float32)
            ).astype(mx.float16)
        if op == "floor":
            return mx.floor(tensor(kwargs["x"]).astype(mx.float32)).astype(mx.float16)
        if op == "less":
            return mx.less(tensor(kwargs["x"]), tensor(kwargs["y"]))
        if op == "logical_not":
            return mx.logical_not(tensor(kwargs["x"]))
        if op == "logical_and":
            return mx.logical_and(tensor(kwargs["x"]), tensor(kwargs["y"]))
        if op == "relu":
            return mx.maximum(tensor(kwargs["x"]).astype(mx.float32), 0.0).astype(mx.float16)
        if op == "sigmoid":
            return mx.sigmoid(tensor(kwargs["x"]).astype(mx.float32)).astype(mx.float16)
        if op == "silu":
            x = tensor(kwargs["x"])
            n = x.size
            out = _silu_kernel()(
                inputs=[x],
                output_shapes=[(n,)],
                output_dtypes=[mx.float16],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            return mx.reshape(out, x.shape)
        if op == "transpose":
            x = tensor(kwargs["x"])
            perm = [a % x.ndim for a in self.ints(kwargs["perm"])]
            return mx.contiguous(mx.transpose(x, perm))
        if op == "reshape":
            return mx.reshape(tensor(kwargs["x"]), self.ints(kwargs["shape"]))
        if op == "tile":
            return mx.tile(tensor(kwargs["x"]), self.ints(kwargs["reps"]))
        if op == "concat":
            axis = self.ints(kwargs["axis"])[0]
            if "values" in kwargs:
                names = split_top(kwargs["values"].strip("() "))
            else:
                names = [
                    kwargs[key]
                    for key in sorted(
                        (k for k in kwargs if re.fullmatch(r"x\d+", k)),
                        key=lambda k: int(k[1:]),
                    )
                ]
            return mx.concatenate([tensor(n) for n in names], axis=axis)
        if op == "linear":
            # fp16 leftover datapath, batched: one fp32 matmul over the
            # [K/16, 16]-blocked K axis yields every block's partial, and
            # _leftover_chain_kernel applies the landed rounding — each
            # block rounds to fp16, then accumulates in fp16 ascending —
            # in one dispatch. Byte-identical to the 5688f8bd chunk loop;
            # a stock single reduce cannot reproduce the ascending fp16
            # chain on the omarchy Vulkan backend (probed 2026-09-15:
            # mx.sum agrees on <=0.20 of elements in every layout, cumsum
            # has no Vulkan kernel), and the explicit chain serialized
            # ~21k dependent dispatches into the graph, costing more wall
            # than the chunk loop it replaced.
            x = tensor(kwargs["x"])
            weight = tensor(kwargs["weight"])
            k = int(x.shape[-1])
            if k % 16:
                raise EncoderRunError(f"linear K {k} not a multiple of 16")
            blocks = k // 16
            rows = 1
            for dim in x.shape[:-1]:
                rows *= dim
            xb = mx.transpose(mx.reshape(x, (rows, blocks, 16)), (1, 0, 2)).astype(
                mx.float32
            )  # [K/16, M, 16]
            wb = mx.transpose(
                mx.reshape(weight, (weight.shape[0], blocks, 16)), (1, 2, 0)
            ).astype(mx.float32)  # [K/16, 16, N]
            partials = xb @ wb  # [K/16, M, N] fp32
            out = _leftover_chain_kernel()(
                inputs=[partials],
                output_shapes=[(rows, weight.shape[0])],
                output_dtypes=[mx.float16],
                grid=(rows * weight.shape[0], 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            out = mx.reshape(out, tuple(x.shape[:-1]) + (weight.shape[0],))
            if "bias" in kwargs:
                out = out.astype(mx.float32) + tensor(kwargs["bias"]).astype(mx.float32)
            return out.astype(mx.float16)
        if op == "matmul":
            a = tensor(kwargs["x"]).astype(mx.float32)
            b = tensor(kwargs["y"]).astype(mx.float32)
            if self.bools(kwargs["transpose_x"])[0]:
                a = mx.swapaxes(a, -1, -2)
            if self.bools(kwargs["transpose_y"])[0]:
                b = mx.swapaxes(b, -1, -2)
            return (a @ b).astype(mx.float16)
        if op == "conv":
            return self.apply_conv(kwargs)
        if op == "layer_norm":
            axes = tuple(self.ints(kwargs["axes"]))
            x = tensor(kwargs["x"])
            if axes != (-1,) or x.ndim != 3 or x.shape[-1] != 1024:
                raise EncoderRunError(
                    f"layer_norm form {x.shape} axes {axes} is not the pinned "
                    "encoder envelope"
                )
            # Two mx.mean reductions keep the ReduceF32 dispatches and their
            # chunked order bit-identical; the standalone kernels only replace
            # the elementwise chain around them, in the same fp32 ops with the
            # same single fp16 rounding at the end.
            rows = x.size // 1024
            n = x.size
            xf = _ln_cast_kernel()(
                inputs=[x], output_shapes=[(n,)], output_dtypes=[mx.float32],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
            )[0]
            mean = mx.mean(mx.reshape(xf, (rows, 1024)), axis=-1, keepdims=True)
            t2 = _ln_sq_kernel()(
                inputs=[xf, mean], output_shapes=[(n,)], output_dtypes=[mx.float32],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
            )[0]
            var = mx.mean(mx.reshape(t2, (rows, 1024)), axis=-1, keepdims=True)
            gamma = tensor(kwargs["gamma"]) if "gamma" in kwargs else None
            beta = tensor(kwargs["beta"]) if "beta" in kwargs else None
            if gamma is None or beta is None:
                raise EncoderRunError("layer_norm without gamma/beta")
            eps_arr = mx.array([float(self.scalar(kwargs["epsilon"]))], dtype=mx.float32)
            y = _ln_tail_kernel()(
                inputs=[xf, mean, var, gamma, beta, eps_arr],
                output_shapes=[(n,)], output_dtypes=[mx.float16],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
            )[0]
            return mx.reshape(y, x.shape)
        if op == "softmax":
            axis = self.ints(kwargs["axis"])[0]
            x = tensor(kwargs["x"])
            if axis % x.ndim != x.ndim - 1 or x.ndim != 4 or x.shape[-1] != 375:
                raise EncoderRunError(
                    f"softmax form {x.shape} axis {axis} is not the pinned "
                    "encoder envelope"
                )
            # The exact ReduceF16 max and the fp32 ReduceF32 sum keep the
            # reduction dispatches; exp and the final divide fuse around them
            # with the same fp32 arithmetic and the same boundary rounding.
            rows = x.size // 375
            n = x.size
            rowmax = mx.max(x, axis=-1, keepdims=True)
            e = _sm_exp_kernel()(
                inputs=[x, rowmax], output_shapes=[(n,)], output_dtypes=[mx.float32],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
            )[0]
            rowsum = mx.sum(mx.reshape(e, (rows, 375)), axis=-1, keepdims=True)
            y = _sm_div_kernel()(
                inputs=[e, rowsum], output_shapes=[(n,)], output_dtypes=[mx.float16],
                grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
            )[0]
            return mx.reshape(y, x.shape)
        if op == "select":
            cond = tensor(kwargs["cond"])
            a = tensor(kwargs["a"]).astype(mx.float32)
            b = tensor(kwargs["b"]).astype(mx.float32)
            return mx.where(cond, a, b).astype(mx.float16)
        if op == "pad":
            mode = self.scalar(kwargs["mode"])
            if mode != "constant":
                raise EncoderRunError(f"pad mode {mode}")
            pad = self.ints(kwargs["pad"])
            value = float(self.scalar(kwargs["constant_val"]))
            x = tensor(kwargs["x"])
            pairs = [(pad[i], pad[i + 1]) for i in range(0, len(pad), 2)]
            pairs = pairs[-x.ndim :]
            widths = [(0, 0)] * (x.ndim - len(pairs)) + pairs
            return mx.pad(
                x.astype(mx.float32), widths, constant_values=value
            ).astype(mx.float16)
        if op == "slice_by_index":
            return self.apply_slice(kwargs)
        raise EncoderRunError(f"unimplemented op {op!r} ({stmt.dtype} {stmt.shape})")

    def apply_conv(self, kwargs: dict) -> mx.array:
        """MIL conv is NCHW; MLX conv is NHWC. Spatial rank 1 lifts to unit H."""
        x = self.tensor(kwargs["x"]).astype(mx.float32)
        weight = self.tensor(kwargs["weight"]).astype(mx.float32)
        bias = self.tensor(kwargs["bias"]).astype(mx.float32) if "bias" in kwargs else None
        strides = self.ints(kwargs["strides"])
        pad = self.ints(kwargs["pad"])
        pad_type = self.scalar(kwargs["pad_type"])
        groups = self.ints(kwargs["groups"])[0]
        dilations = self.ints(kwargs["dilations"])

        rank = x.ndim - 2
        if rank == 1:
            x = mx.expand_dims(x, 2)
            weight = mx.expand_dims(weight, 2)
            strides = [1, strides[0]]
            dilations = [1, dilations[0]]
            pad = [0, 0] + list(pad) if pad_type == "custom" else pad
        elif rank != 2:
            raise EncoderRunError(f"conv spatial rank {rank}")
        if pad_type == "valid":
            pad = [0, 0, 0, 0]
        elif pad_type != "custom":
            raise EncoderRunError(f"conv pad_type {pad_type}")
        if tuple(dilations) != (1, 1):
            raise EncoderRunError(f"conv dilations {dilations}")

        top, bottom, left, right = pad
        xp = mx.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)], constant_values=0.0)
        # NCHW -> NHWC, (cout, cin/g, kh, kw) -> (cout, kh, kw, cin/g)
        nhwc = mx.transpose(xp, [0, 2, 3, 1])
        ohwi = mx.transpose(weight, [0, 2, 3, 1])
        out = mx.conv2d(
            nhwc, ohwi, stride=(strides[0], strides[1]), padding=0, groups=groups
        )
        out = mx.transpose(out, [0, 3, 1, 2])
        if bias is not None:
            out = out + mx.reshape(bias, (1, -1, 1, 1))
        if rank == 1:
            out = mx.squeeze(out, axis=2)
        return out.astype(mx.float16)

    def apply_slice(self, kwargs: dict) -> mx.array:
        x = self.tensor(kwargs["x"])
        begin = self.ints(kwargs["begin"])
        end = self.ints(kwargs["end"])
        begin_mask = self.bools(kwargs["begin_mask"]) if "begin_mask" in kwargs else None
        end_mask = self.bools(kwargs["end_mask"]) if "end_mask" in kwargs else None
        strides = self.ints(kwargs["strides"]) if "strides" in kwargs else None
        squeeze_mask = (
            self.bools(kwargs["squeeze_mask"]) if "squeeze_mask" in kwargs else None
        )
        index = []
        for axis in range(x.ndim):
            start = None if (begin_mask and begin_mask[axis]) else int(begin[axis])
            stop = None if (end_mask and end_mask[axis]) else int(end[axis])
            step = 1 if strides is None else int(strides[axis])
            if squeeze_mask and squeeze_mask[axis]:
                index.append(int(begin[axis]))
            else:
                index.append(slice(start, stop, step))
        return x[tuple(index)]

    # ------------------------------------------------------------------- run

    def run(self, inputs: dict, wanted: set[str], stop_after: str) -> dict:
        for name, value in inputs.items():
            self.values[name] = value
        keep: dict[str, mx.array] = {}
        protected = set(wanted) | {stop_after}
        for stmt in self.statements:
            self.execute(stmt)
            for name in stmt.names:
                if name in wanted and name in self.values:
                    keep[name] = self.values[name]
            # Release anything whose final reader has run.
            for name, last in list(self.last_use.items()):
                if last <= stmt.index and name not in protected and name in self.values:
                    del self.values[name]
                    del self.last_use[name]
            if stop_after in keep:
                break
        else:
            raise EncoderRunError(f"never reached {stop_after}")
        missing = set(wanted) - keep.keys()
        if missing:
            raise EncoderRunError(f"never produced {sorted(missing)}")
        mx.eval(list(keep.values()))
        return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--libane", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=int, default=20000)
    parser.add_argument(
        "--no-ane", action="store_true",
        help="Vulkan-only control run: every op stays on the GPU.",
    )
    parser.add_argument(
        "--islands", default="ABC",
        help="which islands to place on the ANE, e.g. ABC or AC. Ignored with "
             "--no-ane. AC reproduces the two-island arm from this same script.",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    features = np.load(args.capture / "encoder_input_features.npy")
    mask = np.load(args.capture / "encoder_input_mask.npy")

    island = None
    if not args.no_ane:
        island = AneIsland(
            args.worker, args.libane, args.bundles, args.scratch, args.deadline_ms
        )

    placed = frozenset(args.islands.upper())
    if placed - frozenset("ABC"):
        raise SystemExit(f"--islands {args.islands!r}: unknown island(s)")
    runner = EncoderRunner(
        args.source / "model.mil", args.source / "model-root", island, placed
    )
    before = trace_snapshot()
    started = time.monotonic_ns()
    try:
        keep = runner.run(
            inputs={
                "input_features": mx.array(
                    features.astype(np.float32).reshape(1, 3000, 128)
                ),
                "attention_mask": mx.array(mask.astype(np.int32).reshape(1, 3000)),
            },
            wanted={"encoder_hidden", "encoder_mask"},
            stop_after="encoder_mask",
        )
    finally:
        if island is not None:
            island.close()
    wall_ns = time.monotonic_ns() - started
    after = trace_snapshot()

    hidden = np.asarray(keep["encoder_hidden"]).astype(np.float32)
    got_mask = np.asarray(keep["encoder_mask"]).astype(np.int32)
    np.save(args.out / "encoder_hidden.npy", hidden)
    np.save(args.out / "encoder_mask.npy", got_mask)

    gpu_delta = {k: after[k] - before[k] for k in after}
    report = {
        "layers": runner.layers,
        "ops_executed": runner.executed,
        "gpu_ops": runner.gpu_ops,
        "ane_ops": runner.ane_ops,
        "cpu_tensor_events": runner.cpu_tensor_events,
        "wall_ns": wall_ns,
        "ane_mode": not args.no_ane,
        "islands_placed": "".join(sorted(runner.placed)),
        "gpu_counters": gpu_delta,
        "island_b_cond_census": runner.cond_census,
        "encoder_hidden_shape": list(hidden.shape),
    }
    if island is not None:
        report["ane"] = {
            "mode": island._mode,
            "submissions": island.submissions,
            "rounds": island.rounds,
            "worker_starts": island.worker_starts,
            "timeouts": island.timeouts,
            "batch_open_ns": island.batch_open_ns,
            "input_bytes": island.input_bytes,
            "output_bytes": island.output_bytes,
            "exec_ns": island.exec_ns,
            "log": island.log,
        }
    (args.out / "run-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "ane"}, indent=2))
    if island is not None:
        print(
            f"ane mode={island._mode} submissions={island.submissions} "
            f"rounds={island.rounds} worker_starts={island.worker_starts} "
            f"timeouts={island.timeouts} exec_ms={island.exec_ns / 1e6:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
