#!/usr/bin/env python3
"""Minimal numpy evaluator for the pinned Parakeet encoder MIL, prologue + layer 0.

Purpose: derive the REAL layer-0 tensors that the three ANE encoder islands bind,
starting from the authenticated golden capture inputs. Only the ops the encoder
prologue and layer 0 actually use are implemented; anything else is a hard error.

fp16 semantics: every op rounds its result to fp16 (the MIL is an fp16 program).
Accumulating ops (conv / linear / matmul / layer_norm / softmax) accumulate in
fp32 and round once at the end, which is the standard CoreML fp16 contract.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

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

DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
    "bool": np.bool_,
}


class MilError(RuntimeError):
    pass


def split_top(text: str) -> list[str]:
    """Split on commas that are not inside brackets, parens, or quotes."""
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


class Blobs:
    """Reader for the Core ML blob-v2 files the adapter emits."""

    def __init__(self, model_root: Path):
        self.root = model_root
        self._maps: dict[str, np.memmap] = {}

    def _map(self, path: str) -> np.memmap:
        name = path.replace("@model_path/", "")
        if name not in self._maps:
            self._maps[name] = np.memmap(self.root / name, dtype=np.uint8, mode="r")
        return self._maps[name]

    def read(self, path: str, offset: int, dtype, count: int) -> np.ndarray:
        raw = self._map(path)
        magic, storage, length, payload = struct.unpack_from(
            "<IIQQ", raw[offset : offset + 24].tobytes(), 0
        )
        if magic != BLOB_MAGIC:
            raise MilError(f"blob magic {magic:#x} at {path}:{offset}")
        itemsize = np.dtype(dtype).itemsize
        want = count * itemsize
        if want > length:
            raise MilError(f"blob at {path}:{offset} holds {length} bytes, want {want}")
        del storage
        return np.frombuffer(
            raw[payload : payload + want].tobytes(), dtype=dtype, count=count
        )


def literal(dtype, shape, payload: str) -> np.ndarray:
    if dtype is np.bool_:
        values = [item.strip() == "true" for item in payload.strip("[] ").split(",")]
    elif dtype is np.int32:
        values = [int(item) for item in payload.strip("[] ").split(",")]
    else:
        values = [float(item) for item in payload.strip("[] ").split(",")]
    return np.array(values, dtype=dtype).reshape(shape)


def to_fp16(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float16)


def broadcast_pair(x, y):
    return np.broadcast_arrays(x, y)


def conv2d(x, weight, bias, strides, pad, pad_type, groups, dilations):
    """MIL conv for spatial rank 1 and 2. Rank 1 is lifted to a unit H axis."""
    rank = x.ndim - 2
    if rank == 1:
        x = x[:, :, None, :]
        weight = weight[:, :, None, :]
        strides = [1, strides[0]]
        dilations = [1, dilations[0]]
        pad = [0, 0] + list(pad) if pad_type == "custom" else pad
    elif rank != 2:
        raise MilError(f"conv spatial rank {rank}")
    if pad_type == "valid":
        pad = (0, 0, 0, 0)
    elif pad_type != "custom":
        raise MilError(f"conv pad_type {pad_type}")
    if tuple(dilations) != (1, 1):
        raise MilError(f"conv dilations {dilations}")
    top, bottom, left, right = pad
    n, cin, h, w = x.shape
    cout, cin_g, kh, kw = weight.shape
    if cin_g * groups != cin:
        raise MilError(f"conv channel mismatch {x.shape} {weight.shape} g={groups}")
    sh, sw = strides
    xp = np.pad(
        x.astype(np.float32),
        ((0, 0), (0, 0), (top, bottom), (left, right)),
        mode="constant",
    )
    oh = (h + top + bottom - kh) // sh + 1
    ow = (w + left + right - kw) // sw + 1
    # im2col via stride tricks, one group at a time.
    out = np.empty((n, cout, oh, ow), dtype=np.float32)
    per = cout // groups
    wf = weight.astype(np.float32)
    for g in range(groups):
        xg = xp[:, g * cin_g : (g + 1) * cin_g]
        cols = np.lib.stride_tricks.as_strided(
            xg,
            shape=(n, cin_g, kh, kw, oh, ow),
            strides=(
                xg.strides[0],
                xg.strides[1],
                xg.strides[2],
                xg.strides[3],
                xg.strides[2] * sh,
                xg.strides[3] * sw,
            ),
            writeable=False,
        ).reshape(n, cin_g * kh * kw, oh * ow)
        wg = wf[g * per : (g + 1) * per].reshape(per, cin_g * kh * kw)
        out[:, g * per : (g + 1) * per] = (wg @ cols).reshape(n, per, oh, ow)
    if bias is not None:
        out += bias.astype(np.float32).reshape(1, cout, 1, 1)
    return to_fp16(out[:, :, 0, :] if rank == 1 else out)


def batched_matmul(x, y, transpose_x, transpose_y):
    a = x.astype(np.float32)
    b = y.astype(np.float32)
    if transpose_x:
        a = np.swapaxes(a, -1, -2)
    if transpose_y:
        b = np.swapaxes(b, -1, -2)
    return to_fp16(a @ b)


def slice_by_index(x, begin, end, begin_mask, end_mask, strides, squeeze_mask):
    rank = x.ndim
    index = []
    squeeze = []
    for axis in range(rank):
        start = None if (begin_mask is not None and begin_mask[axis]) else int(begin[axis])
        stop = None if (end_mask is not None and end_mask[axis]) else int(end[axis])
        step = 1 if strides is None else int(strides[axis])
        if squeeze_mask is not None and squeeze_mask[axis]:
            index.append(int(begin[axis]))
            squeeze.append(axis)
        else:
            index.append(slice(start, stop, step))
    return x[tuple(index)]


class Program:
    def __init__(self, mil_path: Path, model_root: Path):
        self.text = mil_path.read_text()
        self.blobs = Blobs(model_root)
        self.values: dict[str, object] = {}
        self.types: dict[str, tuple] = {}
        self.lines = self.text.split("\n")

    def resolve(self, token: str):
        token = token.strip()
        if token in self.values:
            return self.values[token]
        raise MilError(f"unresolved operand {token!r}")

    @staticmethod
    def parse_type(text: str):
        match = TYPE.match(text)
        if match:
            dtype = DTYPES[match.group("dtype").lower()] if match.group(
                "dtype"
            ).lower() in DTYPES else None
            shape_text = match.group("shape").strip()
            shape = (
                tuple(int(item) for item in shape_text.split(","))
                if shape_text
                else ()
            )
            return dtype, shape, match.group("dtype").lower()
        low = text.strip().lower()
        return DTYPES.get(low), (), low

    def eval_const(self, name: str, decl_type: str, attrs: str):
        dtype, shape, dtype_name = self.parse_type(decl_type)
        kwargs = parse_kwargs(attrs)
        value_text = kwargs["val"]
        blob = BLOBFILE.search(value_text)
        if dtype_name == "string":
            self.values[name] = re.search(r'"([^"]*)"', value_text).group(1)
            return
        if blob is None:
            inner = value_text[value_text.index("(") + 1 : value_text.rindex(")")]
            self.values[name] = literal(dtype, shape, inner)
            return
        count = int(np.prod(shape)) if shape else 1
        raw = self.blobs.read(blob.group("path"), int(blob.group("offset")), dtype, count)
        self.values[name] = raw.reshape(shape) if shape else raw.reshape(())

    def run(self, stop_after: str, wanted: set[str], inputs: dict[str, np.ndarray]):
        self.values.update(inputs)
        executed = 0
        keep: dict[str, np.ndarray] = {}
        for line in self.lines:
            tuple_match = TUPLE_STMT.match(line)
            if tuple_match is not None:
                names = [
                    item.rsplit(" ", 1)[1]
                    for item in split_top(tuple_match.group("results"))
                ]
                parts = self.apply_tuple(
                    tuple_match.group("op"), parse_kwargs(tuple_match.group("args"))
                )
                if len(parts) != len(names):
                    raise MilError(
                        f"{tuple_match.group('op')} produced {len(parts)} of {len(names)}"
                    )
                self.values.update(zip(names, parts))
                executed += 1
                continue
            match = STMT.match(line)
            if match is None:
                continue
            name = match.group("name")
            op = match.group("op")
            args = match.group("args")
            attrs = match.group("attrs") or ""
            if op == "const":
                self.eval_const(name, match.group("type"), attrs)
                executed += 1
                continue
            self.values[name] = self.apply(op, parse_kwargs(args), match.group("type"))
            executed += 1
            if name in wanted:
                keep[name] = np.array(self.values[name], copy=True)
            if name == stop_after:
                break
        else:
            raise MilError(f"never reached {stop_after}")
        missing = wanted - keep.keys()
        if missing:
            raise MilError(f"never produced {sorted(missing)}")
        return keep, executed

    def apply_tuple(self, op: str, kwargs: dict[str, str]) -> list[np.ndarray]:
        if op != "split":
            raise MilError(f"unimplemented multi-output op {op!r}")
        axis = int(np.asarray(self.resolve(kwargs["axis"])))
        count = int(np.asarray(self.resolve(kwargs["num_splits"])))
        return list(np.split(np.asarray(self.resolve(kwargs["x"])), count, axis=axis))

    def apply(self, op: str, kwargs: dict[str, str], decl_type: str):
        get = lambda key: self.resolve(kwargs[key])
        dtype, shape, dtype_name = self.parse_type(decl_type)

        if op == "cast":
            target = self.resolve(kwargs["dtype"])
            x = get("x")
            if target == "fp16":
                return np.asarray(x).astype(np.float16)
            if target == "fp32":
                return np.asarray(x).astype(np.float32)
            if target == "int32":
                return np.asarray(x).astype(np.int32)
            if target == "bool":
                return np.asarray(x).astype(np.bool_)
            raise MilError(f"cast dtype {target}")
        if op == "expand_dims":
            axes = np.atleast_1d(get("axes")).astype(int)
            out = np.asarray(get("x"))
            for axis in sorted(int(a) for a in axes):
                out = np.expand_dims(out, axis)
            return out
        if op == "squeeze":
            axes = tuple(int(a) for a in np.atleast_1d(get("axes")))
            return np.squeeze(np.asarray(get("x")), axis=axes)
        if op == "reduce_sum":
            axes = tuple(int(a) for a in np.atleast_1d(get("axes")))
            keep = bool(self.resolve(kwargs["keep_dims"]))
            return np.asarray(get("x")).sum(axis=axes, keepdims=keep)
        if op == "reduce_min":
            axes = tuple(int(a) for a in np.atleast_1d(get("axes")))
            keep = bool(self.resolve(kwargs["keep_dims"]))
            return np.asarray(get("x")).min(axis=axes, keepdims=keep)
        if op == "reduce_max":
            axes = tuple(int(a) for a in np.atleast_1d(get("axes")))
            keep = bool(self.resolve(kwargs["keep_dims"]))
            return np.asarray(get("x")).max(axis=axes, keepdims=keep)
        if op in ("add", "sub", "mul"):
            x, y = np.asarray(get("x")), np.asarray(get("y"))
            if x.dtype == np.bool_ and y.dtype == np.bool_:
                # The mask path multiplies two bool tensors: logical and.
                if op != "mul":
                    raise MilError(f"bool {op}")
                return np.logical_and(x, y)
            fx, fy = x.astype(np.float32), y.astype(np.float32)
            raw = {"add": fx + fy, "sub": fx - fy, "mul": fx * fy}[op]
            if x.dtype == np.int32 and y.dtype == np.int32:
                return raw.astype(np.int32)
            return to_fp16(raw)
        if op == "floor_div":
            x, y = np.asarray(get("x")), np.asarray(get("y"))
            if x.dtype == np.int32 and y.dtype == np.int32:
                return (x // y).astype(np.int32)
            return to_fp16(np.floor(x.astype(np.float32) / y.astype(np.float32)))
        if op == "floor":
            return to_fp16(np.floor(np.asarray(get("x")).astype(np.float32)))
        if op == "less":
            x, y = np.asarray(get("x")), np.asarray(get("y"))
            return np.less(*broadcast_pair(x, y))
        if op == "logical_not":
            return np.logical_not(np.asarray(get("x")))
        if op == "logical_and":
            return np.logical_and(*broadcast_pair(np.asarray(get("x")), np.asarray(get("y"))))
        if op == "relu":
            return to_fp16(np.maximum(np.asarray(get("x")).astype(np.float32), 0.0))
        if op == "sigmoid":
            return to_fp16(1.0 / (1.0 + np.exp(-np.asarray(get("x")).astype(np.float32))))
        if op == "silu":
            f = np.asarray(get("x")).astype(np.float32)
            return to_fp16(f / (1.0 + np.exp(-f)))
        if op == "transpose":
            perm = [int(a) for a in np.atleast_1d(get("perm"))]
            x = np.asarray(get("x"))
            perm = [a % x.ndim for a in perm]
            return np.ascontiguousarray(np.transpose(x, perm))
        if op == "reshape":
            shape_arg = [int(a) for a in np.atleast_1d(get("shape"))]
            return np.asarray(get("x")).reshape(shape_arg)
        if op == "tile":
            reps = [int(a) for a in np.atleast_1d(get("reps"))]
            return np.tile(np.asarray(get("x")), reps)
        if op == "concat":
            axis = int(self.resolve(kwargs["axis"]))
            if "values" in kwargs:
                names = split_top(kwargs["values"].strip("() "))
                parts = [self.resolve(item) for item in names]
            else:
                keys = sorted(
                    (key for key in kwargs if re.fullmatch(r"x\d+", key)),
                    key=lambda key: int(key[1:]),
                )
                parts = [self.resolve(kwargs[key]) for key in keys]
            return np.concatenate([np.asarray(part) for part in parts], axis=axis)
        if op == "linear":
            x = np.asarray(get("x")).astype(np.float32)
            weight = np.asarray(get("weight")).astype(np.float32)
            out = x @ weight.T
            if "bias" in kwargs:
                out = out + np.asarray(get("bias")).astype(np.float32)
            return to_fp16(out)
        if op == "matmul":
            return batched_matmul(
                np.asarray(get("x")),
                np.asarray(get("y")),
                bool(self.resolve(kwargs["transpose_x"])),
                bool(self.resolve(kwargs["transpose_y"])),
            )
        if op == "conv":
            return conv2d(
                np.asarray(get("x")),
                np.asarray(get("weight")),
                np.asarray(get("bias")) if "bias" in kwargs else None,
                [int(a) for a in np.atleast_1d(get("strides"))],
                [int(a) for a in np.atleast_1d(get("pad"))],
                self.resolve(kwargs["pad_type"]),
                int(np.atleast_1d(get("groups"))[0]),
                [int(a) for a in np.atleast_1d(get("dilations"))],
            )
        if op == "layer_norm":
            axes = tuple(int(a) for a in np.atleast_1d(get("axes")))
            x = np.asarray(get("x")).astype(np.float32)
            eps = float(np.asarray(get("epsilon")))
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            out = (x - mean) / np.sqrt(var + eps)
            if "gamma" in kwargs:
                out = out * np.asarray(get("gamma")).astype(np.float32)
            if "beta" in kwargs:
                out = out + np.asarray(get("beta")).astype(np.float32)
            return to_fp16(out)
        if op == "softmax":
            axis = int(np.asarray(self.resolve(kwargs["axis"])))
            x = np.asarray(get("x")).astype(np.float32)
            shifted = x - x.max(axis=axis, keepdims=True)
            exp = np.exp(shifted)
            return to_fp16(exp / exp.sum(axis=axis, keepdims=True))
        if op == "select":
            cond = np.asarray(get("cond"))
            a = np.asarray(get("a"))
            b = np.asarray(get("b"))
            return to_fp16(np.where(cond, a.astype(np.float32), b.astype(np.float32)))
        if op == "pad":
            pad = [int(a) for a in np.atleast_1d(get("pad"))]
            mode = self.resolve(kwargs["mode"])
            if mode != "constant":
                raise MilError(f"pad mode {mode}")
            value = float(np.asarray(get("constant_val")))
            x = np.asarray(get("x"))
            widths = [(0, 0)] * x.ndim
            pairs = [(pad[i], pad[i + 1]) for i in range(0, len(pad), 2)]
            for axis, pair in enumerate(pairs[-x.ndim :], start=x.ndim - len(pairs[-x.ndim :])):
                widths[axis] = pair
            return to_fp16(
                np.pad(x.astype(np.float32), widths, mode="constant", constant_values=value)
            )
        if op == "slice_by_index":
            return slice_by_index(
                np.asarray(get("x")),
                np.atleast_1d(get("begin")),
                np.atleast_1d(get("end")),
                np.atleast_1d(get("begin_mask")) if "begin_mask" in kwargs else None,
                np.atleast_1d(get("end_mask")) if "end_mask" in kwargs else None,
                np.atleast_1d(get("strides")) if "strides" in kwargs else None,
                np.atleast_1d(get("squeeze_mask")) if "squeeze_mask" in kwargs else None,
            )
        raise MilError(f"unimplemented op {op!r} (decl {dtype_name} {shape})")


def main() -> int:
    source = Path("/var/tmp/IslandsExecJwm1/encoder-source")
    capture = Path(
        "~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
        "20260912T154759Z-librispeech/ane"
    ).expanduser()
    out = Path("/var/tmp/IslandsExecJwm1/tensors")
    out.mkdir(parents=True, exist_ok=True)

    features = np.load(capture / "encoder_input_features.npy")
    mask = np.load(capture / "encoder_input_mask.npy")
    print("capture features", features.shape, features.dtype)
    print("capture mask", mask.shape, mask.dtype, "sum", int(mask.sum()))

    program = Program(source / "model.mil", source / "model-root")
    wanted = {
        "query_states_with_bias_v_1_cast_fp16",
        "var_355_to_fp16",
        "mul_0_cast_fp16",
        "hidden_states_23_cast_fp16",
        "matrix_bd_5_cast_fp16",
        "var_373",
        "var_8_to_fp16",
        "softmax_0_cast_fp16",
        "hidden_states_25_cast_fp16",
        "attention_scores_1_cast_fp16",
        "matmul_0_cast_fp16",
        "attention_mask_9_cast_fp16",
        "attn_output_1_cast_fp16",
    }
    # var_355/var_8 are consts; collect them straight from the value table after the run.
    dynamic = {name for name in wanted if not name.startswith(("var_355", "var_8"))}
    keep, executed = program.run(
        stop_after="attn_output_1_cast_fp16",
        wanted=dynamic,
        inputs={
            "input_features": features.astype(np.float32).reshape(1, 3000, 128),
            "attention_mask": mask.astype(np.int32).reshape(1, 3000),
        },
    )
    for name in ("var_355_to_fp16", "var_8_to_fp16"):
        keep[name] = np.asarray(program.values[name])
    print("ops executed", executed)
    for name in sorted(keep):
        value = np.asarray(keep[name])
        print(f"  {name:42s} {str(value.dtype):8s} {value.shape}")
    np.savez(out / "layer0.npz", **{name: np.asarray(v) for name, v in keep.items()})
    print("wrote", out / "layer0.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
