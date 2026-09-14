# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Spell 1D k=1 valid conv as the GEMM encoder linears already use.

H13 rejects rank-3 encoder conv; ``convParityPlan`` never sees 1D.
The leftover 24 pointwise convs are ``[1, 1024, 375] × [2048, 1024, 1]``
valid g1. That is GEMM ``(M=375, K=1024, N=2048)``:
``x_nlc @ W^T`` with ``W = weight[:, :, 0]``. Encoder matmuls of other
shapes compile. This module probes that GEMM, not a new conv tile.

A reshape of ``[1, Cin, L]`` to ``[1, L, Cin]`` keeps the element
count and changes packing. ``numpy_naive_ncl_reshape`` is that
counterexample. The NCL→NLC transpose is the definition of the GEMM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class ConvAsMatmulError(RuntimeError):
    """A conv-as-matmul rewrite is unsound here; the reason is named."""


GEMM_M, GEMM_K, GEMM_N = 375, 1024, 2048

STANDALONE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [2048, 1024]> w = const()[name = string("w"), val = tensor<fp16, [2048, 1024]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 375, 2048]> y = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("y")];
  } -> (y);
}
"""

_DEF_LINE = re.compile(
    r"^(?P<indent>\s*)(?:tensor<(?P<dtype>[a-z0-9]+),\s*"
    r"(?P<shape>\[[^\]]*\])>|(?P<scalar>string|int32|bool))\s*"
    r"(?P<name>[A-Za-z_][\w]*) = "
    r"(?P<op>[A-Za-z_][\w]*)\("
)
_ARG = re.compile(r"(?P<key>[A-Za-z_][\w]*)\s*=\s*(?P<value>[A-Za-z_][\w]*)")
_PARAM = re.compile(
    r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*)"
)
_INT_VAL = re.compile(
    r"val = (?:tensor<int32, \[[^\]]*\]>\()?(?:int32\()?(?P<body>[^)]*)"
)
_STR_VAL = re.compile(
    r'val = (?:tensor<string, \[[^\]]*\]>\()?(?:string\()?"(?P<body>[^"]*)"'
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one rank-3 k=1 valid groups-1 conv."""

    index: int
    output_name: str
    disposition: str  # REWRITTEN | REFUSED
    reason: str


@dataclass
class LayoutReport:
    rewritten: list[str] = field(default_factory=list)
    refused: list[LayoutEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.conv-as-matmul.v1",
            "gemm": {"M": GEMM_M, "K": GEMM_K, "N": GEMM_N},
            "rewritten": list(self.rewritten),
            "refused": [
                {
                    "index": entry.index,
                    "output": entry.output_name,
                    "reason": entry.reason,
                }
                for entry in self.refused
            ],
        }


def numpy_conv1d_as_gemm(
    source: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """1D k=1 NCL conv vs ``transpose(x) @ W^T``, then transpose back.

    ``source`` is ``[N, Cin, L]``. ``weight`` is ``[Cout, Cin, 1]``.
    Same fp16 values. Callers compare the pair.
    """
    packed, kernel = _as_conv1d(source, weight)
    original = _fp16_conv1d(packed, kernel)
    nlc = np.transpose(packed, (0, 2, 1))
    acc = nlc.astype(np.float32) @ kernel[:, :, 0].astype(np.float32).T
    rewritten = np.transpose(acc.astype(np.float16), (0, 2, 1))
    return original, rewritten


def numpy_naive_ncl_reshape(
    source: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """1D k=1 NCL conv vs last-axis mix of reshape ``[N, L, Cin]``.

    Same element count, different packing. Callers compare the pair; a
    mismatch is a named counterexample, not a rounding tolerance.
    """
    packed, kernel = _as_conv1d(source, weight)
    original = _fp16_conv1d(packed, kernel)
    batch, channels, length = packed.shape
    naive = packed.reshape(batch, length, channels)
    acc = naive.astype(np.float32) @ kernel[:, :, 0].astype(np.float32).T
    rewritten = np.transpose(acc.astype(np.float16), (0, 2, 1))
    return original, rewritten


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``ConvAsMatmulError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise ConvAsMatmulError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise ConvAsMatmulError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_conv1d(source, weight):
    packed = np.asarray(source)
    kernel = np.asarray(weight)
    if packed.ndim != 3:
        raise ConvAsMatmulError(
            f"1D conv expects rank-3 [N, Cin, L], got {packed.shape}"
        )
    if kernel.ndim != 3 or kernel.shape[2] != 1:
        raise ConvAsMatmulError(
            f"k=1 weight expects [Cout, Cin, 1], got {kernel.shape}"
        )
    if kernel.shape[1] != packed.shape[1]:
        raise ConvAsMatmulError(
            f"weight Cin {kernel.shape[1]} != input C {packed.shape[1]}"
        )
    return packed, kernel


def _fp16_conv1d(source, weight):
    acc = np.einsum(
        "ncl,oc->nol",
        source.astype(np.float32),
        weight.astype(np.float32)[:, :, 0],
    )
    return acc.astype(np.float16)


def _parse_shape(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return ()
    try:
        return tuple(int(part.strip()) for part in inner.split(","))
    except ValueError:
        return None


def _code_of(line: str) -> str:
    cut = line.find("[name =")
    return line if cut < 0 else line[:cut]


def _parse_ints(line: str) -> list[int] | None:
    match = _INT_VAL.search(line)
    if match is None:
        return None
    body = match.group("body").strip()
    try:
        if body.startswith("[") and body.endswith("]"):
            inner = body[1:-1].strip()
            if not inner:
                return []
            return [int(part.strip()) for part in inner.split(",")]
        return [int(body)]
    except ValueError:
        return None


def _parse_string(line: str) -> str | None:
    match = _STR_VAL.search(line)
    if match is None:
        return None
    return match.group("body")


def _index_ops(lines: list[str]):
    table = {}
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match:
            continue
        args = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
        dtype = match.group("dtype") or match.group("scalar")
        shape = match.group("shape") or "[]"
        table[match.group("name")] = (
            index,
            match.group("op"),
            dtype,
            shape,
            args,
            match.group("indent"),
            match.group("scalar") is not None,
        )
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("func ") and "(" in stripped:
            params = stripped.split("(", 1)[1].rsplit(")", 1)[0]
            for match in _PARAM.finditer(params):
                table.setdefault(
                    match.group("name"),
                    (
                        index,
                        "param",
                        match.group("dtype"),
                        match.group("shape"),
                        {},
                        "",
                        False,
                    ),
                )
    return table


def _const_ints(table, lines, name: str) -> list[int] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_ints(lines[entry[0]])


def _const_string(table, lines, name: str) -> str | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_string(lines[entry[0]])


def _fresh(taken: set[str], prefix: str) -> str:
    if prefix not in taken:
        taken.add(prefix)
        return prefix
    counter = 1
    while f"{prefix}_{counter}" in taken:
        counter += 1
    name = f"{prefix}_{counter}"
    taken.add(name)
    return name


def _classify(table, lines, name: str) -> LayoutEntry:
    entry = table[name]
    index, op, dtype, shape_text, args = entry[:5]
    if op != "conv" or dtype != "fp16":
        return LayoutEntry(index, name, "REFUSED", f"op is {op}/{dtype}")
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 3 or shape[0] != 1:
        return LayoutEntry(
            index, name, "REFUSED", f"output shape {shape} is not [1, Cout, L]"
        )
    if "bias" in args:
        return LayoutEntry(index, name, "REFUSED", "bias is not this leftover class")
    weight = table.get(args.get("weight"))
    source = table.get(args.get("x"))
    if weight is None or source is None:
        return LayoutEntry(index, name, "REFUSED", "missing weight or x")
    weight_shape = _parse_shape(weight[3])
    source_shape = _parse_shape(source[3])
    if (
        weight_shape is None
        or len(weight_shape) != 3
        or weight_shape[2] != 1
        or source_shape is None
        or source_shape != (1, weight_shape[1], shape[2])
        or shape[1] != weight_shape[0]
    ):
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            f"weight {weight_shape} / x {source_shape} is not k=1 NCL",
        )
    pad_type = _const_string(table, lines, args.get("pad_type"))
    strides = _const_ints(table, lines, args.get("strides"))
    dilations = _const_ints(table, lines, args.get("dilations"))
    pad = _const_ints(table, lines, args.get("pad"))
    groups = _const_ints(table, lines, args.get("groups"))
    if pad_type != "valid":
        return LayoutEntry(index, name, "REFUSED", f"pad_type is {pad_type}")
    if strides != [1] or dilations != [1] or pad != [0, 0] or groups != [1]:
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            f"st{strides} dl{dilations} pad{pad} g{groups} is not unit valid g1",
        )
    if weight_shape != (GEMM_N, GEMM_K, 1) or source_shape != (1, GEMM_K, GEMM_M):
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            f"GEMM is ({source_shape[2]}, {weight_shape[1]}, {weight_shape[0]}), "
            f"not ({GEMM_M}, {GEMM_K}, {GEMM_N})",
        )
    return LayoutEntry(index, name, "REWRITTEN", "1D k=1 valid is GEMM (375,1024,2048)")


def plan_conv_as_matmul(mil_text: str) -> list[LayoutEntry]:
    """Classify every rank-3 conv; the leftover 24 are this GEMM."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    return [
        _classify(table, lines, name)
        for name, entry in table.items()
        if entry[1] == "conv"
    ]


def rewrite_conv_as_matmul(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace leftover 1D k=1 valid convs with the GEMM matmul.

    Weight ``[Cout, Cin, 1]`` becomes ``[Cout, Cin]`` (unit-K view).
    ``x`` stays ``[1, Cin, L]`` with ``transpose_x=true`` so last-two
    dims are ``(M, K)``. Result is NLC ``[1, L, Cout]``. Downstream
    NCL consumers are leftover transposes, not this class.
    """
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    report = LayoutReport()
    taken = set(table)
    replacements = {}
    const_rewrites = {}
    for name, entry in list(table.items()):
        if entry[1] != "conv":
            continue
        classified = _classify(table, lines, name)
        if classified.disposition != "REWRITTEN":
            report.refused.append(classified)
            continue
        index, _, _, _, args, indent, _ = entry
        weight_name = args["weight"]
        weight_entry = table[weight_name]
        tx = _fresh(taken, f"{name}_tx")
        ty = _fresh(taken, f"{name}_ty")
        replacements[index] = [
            f"{indent}bool {tx} = const()[name = string(\"{tx}\"), val = bool(true)];",
            f"{indent}bool {ty} = const()[name = string(\"{ty}\"), val = bool(true)];",
            (
                f"{indent}tensor<fp16, [1, {GEMM_M}, {GEMM_N}]> {name} = "
                f"matmul(transpose_x = {tx}, transpose_y = {ty}, "
                f"x = {args['x']}, y = {weight_name})[name = string(\"{name}\")];"
            ),
        ]
        weight_line = lines[weight_entry[0]]
        const_rewrites[weight_entry[0]] = weight_line.replace(
            f"[{GEMM_N}, {GEMM_K}, 1]", f"[{GEMM_N}, {GEMM_K}]"
        )
        report.rewritten.append(name)
    out = []
    for index, line in enumerate(lines):
        if index in const_rewrites:
            out.append(const_rewrites[index])
            continue
        if index in replacements:
            out.extend(replacements[index])
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if mil_text.endswith("\n") else ""), report


def _self_check() -> None:
    rng = np.random.default_rng(20260913)
    n, cin, length, cout = 1, 8, 6, 4
    cases = [
        rng.standard_normal((n, cin, length)).astype(np.float16),
        np.zeros((n, cin, length), np.float16),
        np.full((n, cin, length), np.float16("-0")),
    ]
    cases[0][0, 0, 0] = np.float16("-0")
    weight = rng.standard_normal((cout, cin, 1)).astype(np.float16)
    for source in cases:
        original, rewritten = numpy_conv1d_as_gemm(source, weight)
        require_exact(original, rewritten, "1D conv as GEMM")
    original, rewritten = numpy_naive_ncl_reshape(cases[0], weight)
    try:
        require_exact(original, rewritten, "naive ncl reshape")
    except ConvAsMatmulError as caught:
        if "inexact naive ncl reshape" not in str(caught):
            raise
    else:
        raise ConvAsMatmulError("naive ncl reshape was bit-exact")
    if "[1, 375, 1024]> x" not in STANDALONE_MIL:
        raise ConvAsMatmulError("standalone MIL is not GEMM (375,1024,2048)")
    if "val = bool(true)" not in STANDALONE_MIL or "matmul(" not in STANDALONE_MIL:
        raise ConvAsMatmulError("standalone MIL is not ty=true matmul")


if __name__ == "__main__":
    _self_check()
    print("conv-as-matmul numpy: PASS")
