# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Decompose a noncontiguous ``slice_by_index`` into unit chunks + concat.

H13 rejects ``slice_by_index`` on fp16 ``[1, 8, 749, 375]`` as
``h13.noncontiguous-slice``: eight chunks of 280875 elements spaced
281250 apart. The producer is ``[1, 8, 750, 375]`` sliced as
``x[:, :, 1:, :]``. One binding cannot name those interleaved chunks.

A reshape of the same buffer does not help: ``[8, 750, 375][:, 1:, :]``
is the same eight-chunk layout. Dropping the last row instead of the
first (``[:, :, :749, :]``) has the same output shape and the wrong
values — ``numpy_relpos_drop_last`` returns that counterexample.
Cutting two dimensions in one op (``[:, h:h+1, 1:, :]``) is also
rejected: H13 allows at most one sliced dimension per binding.

The exact rewrite takes each head plane (``[:, h:h+1, :, :]``, one
dim) then drops the first row on that plane (``[:, :, 1:, :]``, one
dim). Concat on the heads axis is a permutation of the same bytes as
the original slice — not an approximation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class SliceLayoutError(RuntimeError):
    """A slice-layout rewrite is unsound here; the reason is named."""


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
_INT_VAL = re.compile(
    r"val = tensor<int32, \[[^\]]*\]>\((?P<body>[^)]*)\)"
)
_BOOL_VAL = re.compile(
    r"val = tensor<bool, \[[^\]]*\]>\((?P<body>[^)]*)\)"
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one noncontiguous slice_by_index."""

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
            "schema": "mlx-omarchy.slice-layout-rewrite.v1",
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


def numpy_relpos_drop_first(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original ``x[:, :, 1:, :]`` vs concat of per-head unit slices.

    ``source`` is ``[B, H, T, D]``. Returns ``(original, rewritten)``
    in the same dtype. The stack is the definition of that slice.
    """
    packed = _as_relpos(source)
    original = packed[:, :, 1:, :]
    rewritten = np.concatenate(
        [packed[:, head : head + 1, 1:, :] for head in range(packed.shape[1])],
        axis=1,
    )
    return original, rewritten


def numpy_relpos_drop_last(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x[:, :, 1:, :]`` vs ``x[:, :, :T-1, :]``.

    Same output shape, different packing. Callers compare the pair; a
    mismatch is a named counterexample, not a rounding tolerance.
    """
    packed = _as_relpos(source)
    return packed[:, :, 1:, :], packed[:, :, :-1, :]


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``SliceLayoutError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise SliceLayoutError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise SliceLayoutError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_relpos(source) -> np.ndarray:
    packed = np.asarray(source)
    if packed.ndim != 4:
        raise SliceLayoutError(
            f"rel-pos slice expects rank-4 [B, H, T, D], got {packed.shape}"
        )
    if packed.shape[2] < 2:
        raise SliceLayoutError(
            f"rel-pos T={packed.shape[2]} cannot drop the first row"
        )
    return packed


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


def _parse_bools(line: str) -> list[bool] | None:
    match = _BOOL_VAL.search(line)
    if match is None:
        return None
    body = match.group("body").strip()
    try:
        if body.startswith("[") and body.endswith("]"):
            inner = body[1:-1].strip()
            if not inner:
                return []
            return [_as_bool(part.strip()) for part in inner.split(",")]
        return [_as_bool(body)]
    except ValueError:
        return None


def _as_bool(text: str) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(text)


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


def _const_ints(table, lines, name: str) -> list[int] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_ints(lines[entry[0]])


def _const_bools(table, lines, name: str) -> list[bool] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_bools(lines[entry[0]])


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


def _resolve_index(value: int, dim: int) -> int | None:
    if value < 0:
        value += dim
    if value < 0 or value > dim:
        return None
    return value


def _resolved_ranges(source_shape, begin, end, begin_mask, end_mask, stride):
    rank = len(source_shape)
    if (
        len(begin) != rank
        or len(end) != rank
        or (begin_mask is not None and len(begin_mask) != rank)
        or (end_mask is not None and len(end_mask) != rank)
        or (stride is not None and len(stride) != rank)
    ):
        return None
    ranges = []
    for axis, dim in enumerate(source_shape):
        step = 1 if stride is None else stride[axis]
        if step != 1:
            return None
        if begin_mask is not None and begin_mask[axis]:
            start = 0
        else:
            start = _resolve_index(begin[axis], dim)
        if end_mask is not None and end_mask[axis]:
            stop = dim
        else:
            stop = _resolve_index(end[axis], dim)
        if start is None or stop is None or start > stop:
            return None
        out = stop - start
        full = start == 0 and stop == dim
        ranges.append((start, stop, out, full))
    return ranges


def _inner_partial(ranges) -> int | None:
    """Boundary dim of the innermost run, or None if the slice is full."""
    inner_full_from = len(ranges)
    for axis in range(len(ranges) - 1, -1, -1):
        if ranges[axis][3]:
            inner_full_from = axis
        else:
            break
    if inner_full_from == 0:
        return None
    return inner_full_from - 1


def _is_contiguous(source_shape, ranges) -> bool:
    partial = _inner_partial(ranges)
    if partial is None:
        return True
    n_chunks = 1
    for axis in range(partial):
        n_chunks *= ranges[axis][2]
    if n_chunks == 1:
        return True
    spacing = 1
    chunk_elems = 1
    for axis in range(partial, len(ranges)):
        spacing *= source_shape[axis]
        chunk_elems *= ranges[axis][2]
    return spacing == chunk_elems


def _chunk_axis(ranges) -> int | None:
    """The unique wrapping dim of a one-gap inner slice, or None."""
    partial = _inner_partial(ranges)
    if partial is None:
        return None
    wrapping = [axis for axis in range(partial) if ranges[axis][2] > 1]
    if len(wrapping) != 1:
        return None
    return wrapping[0]


def _classify(lines, table, index, name) -> LayoutEntry | None:
    _index, op, dtype, shape_text, args, _indent = table[name]
    _ = _index
    if op != "slice_by_index" or dtype != "fp16":
        raise SliceLayoutError(f"not a slice-layout candidate: {name}")
    if "squeeze_mask" in args:
        return LayoutEntry(
            index, name, "REFUSED",
            "squeeze_mask changes rank; this rewrite keeps the unit axis",
        )
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 4 or any(dim <= 0 for dim in shape):
        return LayoutEntry(
            index, name, "REFUSED",
            f"slice output shape {shape_text} is not static rank-4",
        )
    source_name = args.get("x")
    source = table.get(source_name) if source_name else None
    if source is None:
        return LayoutEntry(
            index, name, "REFUSED", "slice input has no defining line"
        )
    source_shape = _parse_shape(source[3])
    if source_shape is None or len(source_shape) != 4:
        return LayoutEntry(
            index, name, "REFUSED",
            f"slice input shape {source[3]} is not static rank-4",
        )
    begin_name = args.get("begin")
    end_name = args.get("end")
    begin = _const_ints(table, lines, begin_name) if begin_name else None
    end = _const_ints(table, lines, end_name) if end_name else None
    if begin is None or end is None:
        return LayoutEntry(
            index, name, "REFUSED", "begin/end are not const int32 vectors"
        )
    begin_mask = (
        _const_bools(table, lines, args["begin_mask"])
        if "begin_mask" in args
        else None
    )
    end_mask = (
        _const_bools(table, lines, args["end_mask"])
        if "end_mask" in args
        else None
    )
    if "begin_mask" in args and begin_mask is None:
        return LayoutEntry(
            index, name, "REFUSED", "begin_mask is not a const bool vector"
        )
    if "end_mask" in args and end_mask is None:
        return LayoutEntry(
            index, name, "REFUSED", "end_mask is not a const bool vector"
        )
    stride = (
        _const_ints(table, lines, args["stride"])
        if "stride" in args
        else None
    )
    if "stride" in args and stride is None:
        return LayoutEntry(
            index, name, "REFUSED", "stride is not a const int32 vector"
        )
    ranges = _resolved_ranges(
        source_shape, begin, end, begin_mask, end_mask, stride
    )
    if ranges is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "slice bounds are not unit-stride static ranges"
        )
    out_shape = tuple(item[2] for item in ranges)
    if out_shape != shape:
        return LayoutEntry(
            index, name, "REFUSED",
            f"resolved slice shape {list(out_shape)} != declared {shape_text}",
        )
    if _is_contiguous(source_shape, ranges):
        return None
    chunk_axis = _chunk_axis(ranges)
    if chunk_axis is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "noncontiguous slice is not one wrapping dim over a full "
            "suffix; a single concat cannot name the chunks",
        )
    n_chunks = ranges[chunk_axis][2]
    partial = _inner_partial(ranges)
    chunk_elems = 1
    spacing = 1
    for axis in range(partial, 4):
        chunk_elems *= ranges[axis][2]
        spacing *= source_shape[axis]
    return LayoutEntry(
        index, name, "REWRITTEN",
        f"per-chunk slice + concat axis={chunk_axis} emits {list(shape)}; "
        f"{n_chunks} chunks of {chunk_elems} spaced {spacing} apart",
    )


def plan_slice_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every noncontiguous fp16 rank-4 slice_by_index."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, shape_text, _args, _indent) in table.items():
        if op != "slice_by_index" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        shape = _parse_shape(shape_text)
        if shape is None or len(shape) != 4:
            continue
        entry = _classify(lines, table, index, name)
        if entry is not None:
            entries.append(entry)
    return entries


def _emit_rewrite(lines, table, taken, name) -> tuple[list[str], str]:
    index, _op, dtype, shape_text, args, indent = table[name]
    _ = index
    source_name = args["x"]
    source_shape = _parse_shape(table[source_name][3])
    begin = _const_ints(table, lines, args["begin"])
    end = _const_ints(table, lines, args["end"])
    begin_mask = (
        _const_bools(table, lines, args["begin_mask"])
        if "begin_mask" in args
        else None
    )
    end_mask = (
        _const_bools(table, lines, args["end_mask"])
        if "end_mask" in args
        else None
    )
    stride = (
        _const_ints(table, lines, args["stride"])
        if "stride" in args
        else None
    )
    ranges = _resolved_ranges(
        source_shape, begin, end, begin_mask, end_mask, stride
    )
    chunk_axis = _chunk_axis(ranges)
    n_chunks = ranges[chunk_axis][2]
    chunk_begin = ranges[chunk_axis][0]
    axis_name = _fresh(taken, f"{name}_slice_axis")
    inner_b = _fresh(taken, f"{name}_inner_begin")
    inner_e = _fresh(taken, f"{name}_inner_end")
    inner_begin = [
        0 if axis == chunk_axis else item[0]
        for axis, item in enumerate(ranges)
    ]
    inner_end = [
        1 if axis == chunk_axis else item[1]
        for axis, item in enumerate(ranges)
    ]
    inserted = [
        (
            f"{indent}tensor<int32, []> {axis_name} = const()"
            f"[name = tensor<string, []>(\"{axis_name}\"), "
            f"val = tensor<int32, []>({chunk_axis})];"
        ),
        (
            f"{indent}tensor<int32, [4]> {inner_b} = const()"
            f"[name = tensor<string, []>(\"{inner_b}\"), "
            f"val = tensor<int32, [4]>({_shape_text(tuple(inner_begin))})];"
        ),
        (
            f"{indent}tensor<int32, [4]> {inner_e} = const()"
            f"[name = tensor<string, []>(\"{inner_e}\"), "
            f"val = tensor<int32, [4]>({_shape_text(tuple(inner_end))})];"
        ),
    ]
    pieces = []
    plane_shape = list(source_shape)
    plane_shape[chunk_axis] = 1
    chunk_shape = list(item[2] for item in ranges)
    chunk_shape[chunk_axis] = 1
    for offset in range(n_chunks):
        head = chunk_begin + offset
        plane_begin = [
            head if axis == chunk_axis else 0
            for axis in range(4)
        ]
        plane_end = [
            head + 1 if axis == chunk_axis else source_shape[axis]
            for axis in range(4)
        ]
        b_name = _fresh(taken, f"{name}_h{offset}_begin")
        e_name = _fresh(taken, f"{name}_h{offset}_end")
        p_name = _fresh(taken, f"{name}_h{offset}_plane")
        s_name = _fresh(taken, f"{name}_h{offset}")
        inserted.append(
            f"{indent}tensor<int32, [4]> {b_name} = const()"
            f"[name = tensor<string, []>(\"{b_name}\"), "
            f"val = tensor<int32, [4]>({_shape_text(tuple(plane_begin))})];"
        )
        inserted.append(
            f"{indent}tensor<int32, [4]> {e_name} = const()"
            f"[name = tensor<string, []>(\"{e_name}\"), "
            f"val = tensor<int32, [4]>({_shape_text(tuple(plane_end))})];"
        )
        inserted.append(
            f"{indent}tensor<{dtype}, {_shape_text(tuple(plane_shape))}> "
            f"{p_name} = slice_by_index(begin = {b_name}, end = {e_name}, "
            f"x = {source_name})"
            f"[name = tensor<string, []>(\"{p_name}\")];"
        )
        inserted.append(
            f"{indent}tensor<{dtype}, {_shape_text(tuple(chunk_shape))}> "
            f"{s_name} = slice_by_index(begin = {inner_b}, end = {inner_e}, "
            f"x = {p_name})"
            f"[name = tensor<string, []>(\"{s_name}\")];"
        )
        pieces.append(s_name)
    args_text = ", ".join(
        [f"axis = {axis_name}"]
        + [f"x{offset} = {piece}" for offset, piece in enumerate(pieces)]
    )
    replacement = (
        f"{indent}tensor<{dtype}, {shape_text}> {name} = "
        f"concat({args_text})"
        f"[name = tensor<string, []>(\"{name}_slice_concat\")];"
    )
    return inserted, replacement


def rewrite_slice_layout(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace each sound noncontiguous slice with unit slices + concat.

    Leftover slices stay in the graph so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    plan = plan_slice_layout(mil_text)
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    taken = set(table)
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        inserted, replacement = _emit_rewrite(
            lines, table, taken, entry.output_name
        )
        slice_index = table[entry.output_name][0]
        insertions[slice_index] = inserted
        replacements[slice_index] = replacement
        report.rewritten.append(entry.output_name)
    if not replacements:
        eol = "\n" if mil_text.endswith("\n") else ""
        return "\n".join(lines) + eol, report
    out: list[str] = []
    for index, line in enumerate(lines):
        for text in insertions.get(index, []):
            out.append(text)
        out.append(replacements.get(index, line))
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(out) + eol, report
