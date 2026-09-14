# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Minimal numpy evaluator for a CoreML MIL text program.

Host-only. Evaluates the pinned Parakeet encoder MIL from its text form
plus its blob files so the real layer tensors that cross an ANE island
boundary can be staged from the authenticated capture inputs.

Elementwise ops evaluate in the dtype the MIL declares; matmul, conv,
linear and the reductions accumulate in fp32 and round to the declared
dtype, which is the CoreML fp16 execution model.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
    "int16": np.int16,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "bool": np.bool_,
}
_MAGIC = 0xDEADBEEF
_BLOB_DTYPE = {1: np.float16, 2: np.float32, 3: np.uint8, 4: np.int8, 14: np.int32}


class MilError(RuntimeError):
    pass


def _split_top(text: str, sep: str = ",") -> list[str]:
    depth = 0
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _balanced(text: str, start: int) -> int:
    """Index just past the ')' closing the '(' at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    raise MilError(f"unbalanced parentheses at {start}")


_LHS_DECL = re.compile(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s+([A-Za-z_]\w*)")
_HEAD = re.compile(r"^\s*(?P<lhs>.*?)\s*=\s*(?P<op>[A-Za-z_]\w*)\(")


class BlobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[Path, dict[int, tuple[int, int, int]]] = {}

    def _records(self, path: Path) -> dict[int, tuple[int, int, int]]:
        got = self._cache.get(path)
        if got is not None:
            return got
        size = path.stat().st_size
        records: dict[int, tuple[int, int, int]] = {}
        with path.open("rb") as stream:
            count, _version = struct.unpack("<II", stream.read(8))
            offset = 64
            for _ in range(count):
                stream.seek(offset)
                header = stream.read(64)
                magic, code, payload_size, payload_offset = struct.unpack(
                    "<IIQQ", header[:24]
                )
                if magic != _MAGIC:
                    raise MilError(f"bad blob record at {offset} in {path}")
                records[offset] = (code, payload_size, payload_offset)
                offset = (payload_offset + payload_size + 63) & ~63
        if offset > size + 63:
            raise MilError(f"blob record walk overran {path}")
        self._cache[path] = records
        return records

    def read(self, relative: str, offset: int, dtype, shape) -> np.ndarray:
        path = self.root / relative.removeprefix("@model_path/")
        code, payload_size, payload_offset = self._records(path)[offset]
        blob_dtype = _BLOB_DTYPE.get(code)
        if blob_dtype is None:
            raise MilError(f"unsupported blob dtype code {code}")
        count = int(np.prod(shape)) if shape else 1
        data = np.fromfile(
            path, dtype=blob_dtype, count=count, offset=payload_offset
        )
        if data.size != count:
            raise MilError(
                f"blob at {offset} in {path} yielded {data.size} of {count}"
            )
        if payload_size < count * blob_dtype().itemsize:
            raise MilError(f"blob record at {offset} is short for {shape}")
        return data.astype(dtype, copy=False).reshape(shape)


class Op:
    __slots__ = ("op", "args", "outs", "types", "shapes", "name")

    def __init__(self, op, args, outs, types, shapes, name):
        self.op = op
        self.args = args
        self.outs = outs
        self.types = types
        self.shapes = shapes
        self.name = name


class Program:
    """Lazily evaluates named tensors of one MIL ``main`` function."""

    def __init__(self, mil_path: Path, model_root: Path):
        self.blobs = BlobStore(model_root)
        self.producer: dict[str, Op] = {}
        self.values: dict[str, object] = {}
        self.inputs: dict[str, tuple] = {}
        self._parse(Path(mil_path).read_text())

    # ---------------------------------------------------------------- parse
    def _parse(self, text: str) -> None:
        for raw in text.splitlines():
            line = raw.strip()
            if not line.endswith(";"):
                if line.startswith("func main"):
                    for decl in _LHS_DECL.finditer(line):
                        dtype, shape, name = decl.groups()
                        self.inputs[name] = (
                            dtype,
                            tuple(int(x) for x in shape.split(",") if x.strip()),
                        )
                continue
            head = _HEAD.match(line)
            if head is None:
                continue
            open_paren = line.index("(", head.end() - 1)
            end_args = _balanced(line, open_paren)
            args_text = line[open_paren + 1 : end_args - 1]
            attrs_text = line[end_args:].strip()
            if not (attrs_text.startswith("[") and attrs_text.endswith("];")):
                raise MilError(f"unexpected attribute block: {attrs_text[:60]}")
            attrs = self._kv(attrs_text[1:-2])
            outs, types, shapes = [], [], []
            for dtype, shape, name in _LHS_DECL.findall(head.group("lhs")):
                outs.append(name)
                types.append(dtype)
                shapes.append(
                    tuple(int(x) for x in shape.split(",") if x.strip())
                )
            op = Op(
                head.group("op"),
                self._kv(args_text),
                outs,
                types,
                shapes,
                attrs.get("name"),
            )
            if op.op == "const":
                self.values[outs[0]] = self._literal(
                    attrs["val"], types[0], shapes[0]
                )
                continue
            for name in outs:
                self.producer[name] = op

    @staticmethod
    def _kv(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in _split_top(text):
            if not item:
                continue
            key, _, value = item.partition("=")
            out[key.strip()] = value.strip()
        return out

    def _literal(self, text: str, dtype: str, shape: tuple):
        m = re.match(r"^tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\((.*)\)$", text, re.S)
        if m is None:
            raise MilError(f"unparsed literal {text[:80]}")
        lit_dtype, lit_shape, payload = m.groups()
        shape = tuple(int(x) for x in lit_shape.split(",") if x.strip())
        payload = payload.strip()
        if lit_dtype == "string":
            items = [x.strip().strip('"') for x in _split_top(payload.strip("[]"))]
            return items[0] if not shape else items
        np_dtype = DTYPES[lit_dtype]
        if payload.startswith("BLOBFILE"):
            relative = re.search(r'path = string\("([^"]+)"\)', payload).group(1)
            offset = int(re.search(r"offset = uint64\((\d+)\)", payload).group(1))
            return self.blobs.read(relative, offset, np_dtype, shape)
        items = [
            self._scalar(x, np_dtype)
            for x in _split_top(payload.strip().strip("[]"))
        ]
        array = np.array(items, dtype=np_dtype)
        return array.reshape(shape) if shape else array.reshape(())

    @staticmethod
    def _scalar(text: str, np_dtype):
        text = text.strip()
        if text == "true":
            return True
        if text == "false":
            return False
        if np_dtype in (np.float16, np.float32):
            try:
                return float(text)
            except ValueError:
                return float.fromhex(text)
        return int(text, 0)

    # ----------------------------------------------------------------- eval
    def feed(self, name: str, array: np.ndarray) -> None:
        dtype, shape = self.inputs[name]
        array = np.asarray(array, dtype=DTYPES[dtype])
        if tuple(array.shape) != shape:
            raise MilError(f"input {name} is {array.shape}, expected {shape}")
        self.values[name] = array

    def get(self, name: str):
        if name in self.values:
            return self.values[name]
        op = self.producer.get(name)
        if op is None:
            raise MilError(f"no producer for {name!r}")
        results = self._apply(op)
        for out, dtype, result in zip(op.outs, op.types, results):
            self.values[out] = np.asarray(result, dtype=DTYPES[dtype])
        return self.values[name]

    def _arg(self, op: Op, key: str, default=None):
        text = op.args.get(key)
        if text is None:
            return default
        if text.startswith("tensor<"):
            return self._literal(text, "fp16", ())
        if text.startswith("("):
            return [self.get(x) for x in _split_top(text[1:-1])]
        return self.get(text)

    @staticmethod
    def _ints(value) -> list[int]:
        return [int(x) for x in np.atleast_1d(np.asarray(value)).ravel()]

    def _apply(self, op: Op) -> list:
        kind = op.op
        handler = getattr(self, f"_op_{kind}", None)
        if handler is None:
            raise MilError(f"unimplemented MIL op {kind!r} ({op.name})")
        result = handler(op)
        return result if isinstance(result, list) else [result]

    # --- elementwise -----------------------------------------------------
    def _op_add(self, op):
        return self._arg(op, "x") + self._arg(op, "y")

    def _op_sub(self, op):
        return self._arg(op, "x") - self._arg(op, "y")

    def _op_mul(self, op):
        return self._arg(op, "x") * self._arg(op, "y")

    def _op_floor_div(self, op):
        return np.floor_divide(self._arg(op, "x"), self._arg(op, "y"))

    def _op_floor(self, op):
        return np.floor(self._arg(op, "x"))

    def _op_less(self, op):
        return self._arg(op, "x") < self._arg(op, "y")

    def _op_logical_and(self, op):
        return np.logical_and(self._arg(op, "x"), self._arg(op, "y"))

    def _op_logical_not(self, op):
        return np.logical_not(self._arg(op, "x"))

    def _op_relu(self, op):
        return np.maximum(self._arg(op, "x"), 0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Split on the sign so exp() never overflows for large |x|.
        out = np.empty_like(x)
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    def _op_sigmoid(self, op):
        return self._sigmoid(self._arg(op, "x").astype(np.float32))

    def _op_silu(self, op):
        x = self._arg(op, "x").astype(np.float32)
        return x * self._sigmoid(x)

    def _op_cast(self, op):
        return self._arg(op, "x").astype(DTYPES[self._arg(op, "dtype")])

    def _op_identity(self, op):
        return self._arg(op, "x")

    def _op_select(self, op):
        return np.where(
            self._arg(op, "cond"), self._arg(op, "a"), self._arg(op, "b")
        )

    # --- shape -----------------------------------------------------------
    def _op_reshape(self, op):
        return self._arg(op, "x").reshape(self._ints(self._arg(op, "shape")))

    def _op_transpose(self, op):
        x = self._arg(op, "x")
        perm = [p % x.ndim for p in self._ints(self._arg(op, "perm"))]
        return np.transpose(x, perm)

    def _op_expand_dims(self, op):
        x = self._arg(op, "x")
        axes = sorted(
            a % (x.ndim + len(self._ints(self._arg(op, "axes"))))
            for a in self._ints(self._arg(op, "axes"))
        )
        for axis in axes:
            x = np.expand_dims(x, axis)
        return x

    def _op_squeeze(self, op):
        x = self._arg(op, "x")
        axes = self._arg(op, "axes")
        if axes is None:
            return np.squeeze(x)
        return np.squeeze(x, tuple(a % x.ndim for a in self._ints(axes)))

    def _op_tile(self, op):
        return np.tile(self._arg(op, "x"), self._ints(self._arg(op, "reps")))

    def _op_concat(self, op):
        values = self._arg(op, "values")
        axis = int(self._arg(op, "axis"))
        return np.concatenate(values, axis=axis)

    def _op_split(self, op):
        x = self._arg(op, "x")
        axis = int(self._arg(op, "axis"))
        sizes = self._arg(op, "split_sizes")
        if sizes is not None:
            bounds = np.cumsum(self._ints(sizes))[:-1]
            return list(np.split(x, bounds, axis=axis))
        num = self._arg(op, "num_splits")
        return list(np.split(x, int(num) if num is not None else len(op.outs), axis=axis))

    def _op_slice_by_index(self, op):
        x = self._arg(op, "x")
        begin = self._ints(self._arg(op, "begin"))
        end = self._ints(self._arg(op, "end"))
        strides = self._arg(op, "stride")
        strides = (
            self._ints(strides) if strides is not None else [1] * x.ndim
        )
        begin_mask = self._arg(op, "begin_mask")
        end_mask = self._arg(op, "end_mask")
        squeeze_mask = self._arg(op, "squeeze_mask")
        slices = []
        squeeze_axes = []
        for axis in range(x.ndim):
            if squeeze_mask is not None and bool(np.ravel(squeeze_mask)[axis]):
                slices.append(begin[axis])
                squeeze_axes.append(axis)
                continue
            lo = None if (begin_mask is not None and bool(np.ravel(begin_mask)[axis])) else begin[axis]
            hi = None if (end_mask is not None and bool(np.ravel(end_mask)[axis])) else end[axis]
            slices.append(slice(lo, hi, strides[axis]))
        return x[tuple(slices)]

    def _op_pad(self, op):
        x = self._arg(op, "x")
        pads = self._ints(self._arg(op, "pad"))
        value = self._arg(op, "constant_val")
        widths = [(0, 0)] * (x.ndim - len(pads) // 2)
        widths += [
            (pads[2 * i], pads[2 * i + 1]) for i in range(len(pads) // 2)
        ]
        mode = self._arg(op, "mode", "constant")
        if mode != "constant":
            raise MilError(f"pad mode {mode!r} is not implemented")
        return np.pad(
            x, widths, mode="constant",
            constant_values=float(value) if value is not None else 0.0,
        )

    # --- reductions and norms -------------------------------------------
    def _op_reduce_sum(self, op):
        axes = tuple(self._ints(self._arg(op, "axes")))
        keep = bool(self._arg(op, "keep_dims"))
        return np.sum(self._arg(op, "x").astype(np.float32) if self._arg(op, "x").dtype == np.float16 else self._arg(op, "x"), axis=axes, keepdims=keep)

    def _op_reduce_min(self, op):
        axes = tuple(self._ints(self._arg(op, "axes")))
        keep = bool(self._arg(op, "keep_dims"))
        return np.min(self._arg(op, "x"), axis=axes, keepdims=keep)

    def _op_softmax(self, op):
        x = self._arg(op, "x").astype(np.float32)
        axis = int(self._arg(op, "axis"))
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _op_layer_norm(self, op):
        x = self._arg(op, "x").astype(np.float32)
        axes = tuple(a % x.ndim for a in self._ints(self._arg(op, "axes")))
        gamma = self._arg(op, "gamma")
        beta = self._arg(op, "beta")
        eps = self._arg(op, "epsilon")
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
        out = (x - mean) / np.sqrt(var + (float(eps) if eps is not None else 1e-5))
        if gamma is not None:
            out = out * np.asarray(gamma, dtype=np.float32)
        if beta is not None:
            out = out + np.asarray(beta, dtype=np.float32)
        return out

    # --- linear algebra --------------------------------------------------
    def _op_matmul(self, op):
        x = self._arg(op, "x").astype(np.float32)
        y = self._arg(op, "y").astype(np.float32)
        if bool(self._arg(op, "transpose_x", False)):
            x = np.swapaxes(x, -1, -2)
        if bool(self._arg(op, "transpose_y", False)):
            y = np.swapaxes(y, -1, -2)
        return np.matmul(x, y)

    def _op_linear(self, op):
        x = self._arg(op, "x").astype(np.float32)
        weight = self._arg(op, "weight").astype(np.float32)
        bias = self._arg(op, "bias")
        out = x @ weight.T
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float32)
        return out

    def _op_conv(self, op):
        x = self._arg(op, "x").astype(np.float32)
        weight = self._arg(op, "weight").astype(np.float32)
        bias = self._arg(op, "bias")
        groups = int(self._arg(op, "groups", 1))
        spatial = x.ndim - 2
        strides = self._ints(self._arg(op, "strides", [1] * spatial))
        dilations = self._ints(self._arg(op, "dilations", [1] * spatial))
        if any(d != 1 for d in dilations):
            raise MilError("dilated conv is not implemented")
        pad_type = self._arg(op, "pad_type", "valid")
        pad = self._arg(op, "pad")
        kernel = weight.shape[2:]
        if pad_type == "custom":
            pads = self._ints(pad)
        elif pad_type == "valid":
            pads = [0] * (2 * spatial)
        elif pad_type == "same":
            pads = []
            for axis in range(spatial):
                total = max(
                    0,
                    (x.shape[2 + axis] - 1) * strides[axis]
                    + kernel[axis]
                    - x.shape[2 + axis],
                )
                pads += [total // 2, total - total // 2]
        else:
            raise MilError(f"pad_type {pad_type!r} is not implemented")
        widths = [(0, 0), (0, 0)] + [
            (pads[2 * i], pads[2 * i + 1]) for i in range(spatial)
        ]
        xp = np.pad(x, widths)
        out_spatial = [
            (xp.shape[2 + i] - kernel[i]) // strides[i] + 1
            for i in range(spatial)
        ]
        cout = weight.shape[0]
        cin_g = weight.shape[1]
        cout_g = cout // groups
        out = np.zeros((x.shape[0], cout, *out_spatial), dtype=np.float32)
        import itertools

        for tap in itertools.product(*[range(k) for k in kernel]):
            window = xp
            for axis in range(spatial):
                start = tap[axis]
                stop = start + (out_spatial[axis] - 1) * strides[axis] + 1
                window = np.take(
                    window,
                    np.arange(start, stop, strides[axis]),
                    axis=2 + axis,
                )
            taps = (slice(None), slice(None)) + tap
            wt = weight[taps]  # [cout, cin_g]
            if cin_g == 1 and cout_g == 1:
                shape = [1, cout] + [1] * spatial
                out += window * wt.reshape(shape)
                continue
            for group in range(groups):
                xs = window[:, group * cin_g : (group + 1) * cin_g]
                wg = wt[group * cout_g : (group + 1) * cout_g]
                contribution = np.tensordot(wg, xs, axes=([1], [1]))
                out[:, group * cout_g : (group + 1) * cout_g] += np.moveaxis(
                    contribution, 0, 1
                )
        if bias is not None:
            shape = [1, cout] + [1] * spatial
            out += np.asarray(bias, dtype=np.float32).reshape(shape)
        return out
