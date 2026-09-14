# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Materialize encoder select fill ``a`` as a runtime input.

H13 refuses select with a MIL-const ``a``
(``h13.select-needs-decoded-encoder``). Runtime-a plus a bool cond at
CHW ``{8,375,375}`` compiles to one ``apple-parity-boolean`` program.
The leftover 24 encoder selects bind a scalar fp16 ``-inf`` const.
This rewrite promotes that const to a function input of the select
output shape; the host fills it. Do not emit the ninf packing. Do not
rewrite the ``+0.0``-fill family (``mask_lowering`` owns that mul).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class SelectRuntimeError(RuntimeError):
    """A select-runtime rewrite is unsound here; the reason is named."""


ENVELOPE_CHW = frozenset({(8, 375, 375), (64, 1, 1)})
NINF_F16 = np.float16("-inf")

STANDALONE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> a, tensor<fp16, [1, 8, 375, 375]> b, tensor<bool, [1, 8, 375, 375]> cond) {
    tensor<fp16, [1, 8, 375, 375]> y = select(a = a, b = b, cond = cond)[name = string("y")];
  } -> (y);
}
"""

_DEF_LINE = re.compile(
    r"^(?P<indent>\s*)tensor<(?P<dtype>[a-z0-9]+),\s*"
    r"(?P<shape>\[[^\]]*\])>\s*(?P<name>[A-Za-z_][\w]*) = "
    r"(?P<op>[A-Za-z_][\w]*)\("
)
_ARG = re.compile(r"(?P<key>[A-Za-z_][\w]*)\s*=\s*(?P<value>[A-Za-z_][\w]*)")
_PARAM = re.compile(
    r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*)"
)
_FOOTER = re.compile(r"^\s*\} -> \((?P<returns>[^)]*)\)\s*;\s*$")
_BLOB = re.compile(
    r'BLOBFILE\(path = string\("(?P<path>[^"]*)"\), '
    r"offset = uint64\((?P<offset>\d+)\)\)"
)
_FP16_VAL = re.compile(
    r"val = tensor<fp16, \[[^\]]*\]>\((?P<body>-inf|inf|-?0x[0-9A-Fa-f]+|-?[0-9.]+(?:e-?[0-9]+)?)\)"
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one const-a select."""

    index: int
    output_name: str
    disposition: str  # REWRITTEN | REFUSED
    reason: str


@dataclass
class LayoutReport:
    rewritten: list[str] = field(default_factory=list)
    refused: list[LayoutEntry] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.select-runtime-rewrite.v1",
            "rewritten": list(self.rewritten),
            "refused": [
                {
                    "index": entry.index,
                    "output": entry.output_name,
                    "reason": entry.reason,
                }
                for entry in self.refused
            ],
            "fills": list(self.fills),
        }


def materialize_fill(shape, value=NINF_F16) -> np.ndarray:
    """Host fill for a promoted select ``a``. Same bits in every lane."""
    return np.full(tuple(shape), np.float16(value), dtype=np.float16)


def numpy_select(cond, a, b) -> np.ndarray:
    """MIL ``select(a, b, cond)``: ``where(cond, a, b)`` in fp16."""
    cond = np.asarray(cond, dtype=bool)
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    return np.where(cond, a, b).astype(np.float16, copy=False)


def numpy_const_vs_runtime_fill(
    cond, x, fill=NINF_F16
) -> tuple[np.ndarray, np.ndarray]:
    """``select(cond, scalar fill, x)`` vs the same fill materialized.

    Callers compare the pair; a mismatch is a named counterexample.
    """
    packed = np.asarray(x, dtype=np.float16)
    scalar = np.float16(fill)
    original = numpy_select(cond, scalar, packed)
    rewritten = numpy_select(cond, materialize_fill(packed.shape, scalar), packed)
    return original, rewritten


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``SelectRuntimeError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise SelectRuntimeError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise SelectRuntimeError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


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


def _shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _code_of(line: str) -> str:
    cut = line.find("[name =")
    return line if cut < 0 else line[:cut]


def _index_ops(lines: list[str]):
    table = {}
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match:
            continue
        args = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
        table[match.group("name")] = (
            index,
            match.group("op"),
            match.group("dtype"),
            match.group("shape"),
            args,
            match.group("indent"),
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
                    ),
                )
    return table


def _consumers_of(lines: list[str], name: str) -> list[int]:
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    found = []
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match or match.group("name") == name:
            continue
        if pattern.search(_code_of(line)):
            found.append(index)
    return found


def _returns_of(lines: list[str]) -> set[str]:
    for line in lines:
        footer = _FOOTER.match(line.rstrip("\n"))
        if footer:
            return {
                name.strip()
                for name in footer.group("returns").split(",")
                if name.strip()
            }
    return set()


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


def _const_payload(line: str) -> dict:
    blob = _BLOB.search(line)
    if blob:
        return {
            "storage": "blob",
            "path": blob.group("path"),
            "offset": int(blob.group("offset")),
        }
    value = _FP16_VAL.search(line)
    if value is None:
        return {"storage": "const"}
    body = value.group("body")
    if body in ("-inf", "inf"):
        bits = 0xFC00 if body == "-inf" else 0x7C00
        return {"storage": "immediate", "value": body, "bits": bits}
    number = np.float16(int(body, 0) if body.lower().startswith("0x") else float(body))
    return {
        "storage": "immediate",
        "value": str(number),
        "bits": int(number.view("<u2")),
    }


def _func_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().startswith("func ") and "(" in line:
            return index
    return None


def _add_params(line: str, extras: list[str]) -> str:
    if not extras:
        return line
    cut = line.rfind(") {")
    open_at = line.find("(")
    if cut < 0 or open_at < 0:
        raise SelectRuntimeError("function header has no '(...) {'")
    existing = line[open_at + 1 : cut].strip()
    added = ", ".join(extras)
    inner = f"{existing}, {added}" if existing else added
    return f"{line[: open_at + 1]}{inner}{line[cut:]}"


def _classify(table, index, name) -> LayoutEntry | None:
    _index, op, dtype, shape_text, args, _indent = table[name]
    if op != "select" or dtype != "fp16":
        return None
    a_name = args.get("a")
    cond_name = args.get("cond")
    a = table.get(a_name) if a_name else None
    if a is None or a[1] != "const":
        return None
    out_shape = _parse_shape(shape_text)
    chw = (
        tuple(out_shape[1:])
        if out_shape is not None and len(out_shape) == 4 and out_shape[0] == 1
        else None
    )
    if chw not in ENVELOPE_CHW:
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            f"shape {shape_text} is outside the runtime-a envelope",
        )
    if a[2] != "fp16":
        return LayoutEntry(
            index, name, "REFUSED", f"const a '{a_name}' is {a[2]}, not fp16"
        )
    a_shape = _parse_shape(a[3])
    if a_shape not in ((), out_shape):
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            f"const a shape {a[3]} is neither scalar nor the select output",
        )
    cond = table.get(cond_name) if cond_name else None
    if cond is None or cond[2] != "bool":
        return LayoutEntry(
            index,
            name,
            "REFUSED",
            "cond is not bool; Apple rejected fp16-cond forms",
        )
    return LayoutEntry(
        index,
        name,
        "REWRITTEN",
        "promote const a to a runtime fill of the select output shape",
    )


def plan_select_runtime(mil_text: str) -> list[LayoutEntry]:
    """Classify every const-a select."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, _shape, _args, _indent) in table.items():
        if op != "select" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        entry = _classify(table, index, name)
        if entry is not None:
            entries.append(entry)
    return entries


def rewrite_select_runtime(mil_text: str) -> tuple[str, LayoutReport]:
    """Rewire each in-envelope const-a select to a runtime fill input.

    Leftover selects stay in the graph so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    returns = _returns_of(lines)
    plan = plan_select_runtime(mil_text)
    replacements: dict[int, str | None] = {}
    taken = set(table)
    fill_for: dict[tuple[str, tuple[int, ...]], str] = {}
    rewritten_at: dict[str, list[int]] = {}
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        name = entry.output_name
        index, _op, _dtype, shape_text, args, _indent = table[name]
        a_name = args["a"]
        out_shape = _parse_shape(shape_text)
        key = (a_name, out_shape)
        if key not in fill_for:
            fill_for[key] = _fresh(taken, f"{a_name}_rt")
        fill_name = fill_for[key]
        rewritten = re.sub(
            r"a\s*=\s*" + re.escape(a_name) + r"(?![\w])",
            f"a = {fill_name}",
            lines[index],
            count=1,
        )
        if rewritten == lines[index]:
            raise SelectRuntimeError(
                f"select '{name}' did not rewire a = {a_name}"
            )
        replacements[index] = rewritten
        rewritten_at.setdefault(a_name, []).append(index)
        report.rewritten.append(name)
    if not replacements:
        eol = "\n" if mil_text.endswith("\n") else ""
        return "\n".join(lines) + eol, report
    extras = []
    for (a_name, out_shape), fill_name in fill_for.items():
        extras.append(
            f"tensor<fp16, {_shape_text(out_shape)}> {fill_name}"
        )
        payload = _const_payload(lines[table[a_name][0]])
        report.fills.append(
            {
                "input": fill_name,
                "shape": list(out_shape),
                "const": a_name,
                **payload,
            }
        )
        others = [
            consumer
            for consumer in _consumers_of(lines, a_name)
            if consumer not in rewritten_at[a_name]
        ]
        if not others and a_name not in returns:
            replacements[table[a_name][0]] = None
    func_index = _func_index(lines)
    if func_index is None:
        raise SelectRuntimeError("no function header found")
    replacements[func_index] = _add_params(lines[func_index], extras)
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in replacements and replacements[index] is None:
            continue
        out.append(replacements.get(index, line))
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(out) + eol, report
