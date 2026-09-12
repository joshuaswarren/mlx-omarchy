# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Linux .mlpackage reader.

Given a directory laid out as a Core ML mlpackage, return the model
specification, MIL program structure, and weight references that
downstream compiler integration (Phase 3) needs.

The reader parses enough of the protobuf to answer every question on the
plan's inspector checklist (section 14 of
``docs/plans/2026-09-12-coreml-parakeet-ane-plan.md``):

* package format,
* model/spec type,
* functions,
* opset/deployment target,
* input names + dtypes + shapes,
* output names + dtypes + shapes,
* state tensors,
* external weight files,
* weight hashes,
* operation histogram,
* op versions,
* dynamic dimensions,
* control flow,
* compression / palettization metadata,
* compiler eligibility,
* unsupported constructs.

It does NOT open the ANE device. It does NOT depend on
``coremltools`` or any protobuf codegen. The hand-written wire decoder
in :mod:`proto` provides the structural extraction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import proto

# Core ML TensorType proto fields. Limited to the subset the inspector
# actually consumes.
#
#   message TensorType {
#     optional string name = 1;
#     optional DataType dataType = 2;     // enum
#     repeated int64 shape = 3;           // symbolic dims are negative
#     optional bool isShapeSymbolic = 4;   // not always present
#   }
# TensorType appears inside ModelDescription.input/output, NamedValueType
# (state), and Block.input/output/boundary.

_TENSOR_NAME_FIELD = 1
_TENSOR_DATATYPE_FIELD = 2
_TENSOR_SHAPE_FIELD = 3
_TENSOR_SRANK_FIELD = 7  # shape rank (alternate layout in some proto encodings)

# DataType enum (from specification.proto). We only need a few for the
# parakeet workloads; unknown values are reported as ``dataType=N``.
_DATATYPE_NAMES = {
    1: "float16", 2: "float32", 3: "double", 4: "int8", 5: "int16", 6: "int32",
    7: "string", 8: "bool", 9: "dictionary", 10: "uint8", 11: "uint16",
    12: "uint32", 13: "uint64", 14: "int64",
}

# MIL Tensor dtypes are different from the older model-spec DataType.
# In the MIL proto the dtype is its own message; for inspection we read
# the wire-level field and label the common shapes directly.
_MIL_DTYPE_FIELDS = {
    0x01: "float16",
    0x02: "float32",
    0x03: "double",
    0x04: "int8",
    0x05: "int16",
    0x06: "int32",
    0x07: "int64",
    0x08: "uint8",
    0x09: "uint16",
    0x0A: "uint32",
    0x0B: "uint64",
    0x0C: "bool",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: list[int]
    shape_symbolic: bool = False

    def is_static(self) -> bool:
        """A static shape has all non-negative dims and is not flagged symbolic."""
        if self.shape_symbolic:
            return False
        return all(d >= 0 for d in self.shape)


@dataclass
class OpSummary:
    type: str
    occurrences: int
    example_attrs: list[str]
    first_block_index: int
    first_op_index: int
    # Proven evidence from this op only. The inspector never assumes a
    # generic "last two dims are M x N" reading for matmul/linear: it
    # only annotates compiler eligibility when it has direct shape
    # evidence for the weight tensor and the transpose/batch-axis
    # permutation chain in front of it. Absence is the safe default;
    # the raw inventory always preserves the original dtypes and
    # shapes, even when the compiler cannot accept them today.
    observed_input_shapes: list[str] = field(default_factory=list)
    observed_weight_shapes: list[str] = field(default_factory=list)
    matvec_eligible: bool | None = None
    fp16: bool | None = None
    weight_binding: str | None = None
    envelope_evidence: list[str] = field(default_factory=list)


@dataclass
class FunctionSpec:
    name: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    inputs_raw: list[dict] = field(default_factory=list)
    outputs_raw: list[dict] = field(default_factory=list)
    op_histogram: list[OpSummary] = field(default_factory=list)
    op_total: int = 0
    has_control_flow: bool = False
    control_flow_ops: list[str] = field(default_factory=list)
    external_weight_refs: list[str] = field(default_factory=list)
    palettization_seen: bool = False
    palettization_metadata: list[dict] = field(default_factory=list)
    unsupported_constructs: list[str] = field(default_factory=list)
    spec_field_notes: list[str] = field(default_factory=list)
    output_count: int = 0
    multi_output: bool = False


@dataclass
class InspectionReport:
    package_path: str
    package_format: str
    file_format_version: str | None
    root_identifier: str | None
    specification_version: int | None
    model_description: str | None
    author: str | None
    license: str | None
    metadata: dict
    files: list[dict]
    weights: list[dict]
    functions: list[FunctionSpec]
    quantization: str
    minimum_deployment_target: str | None
    compute_units: str | None
    precision: str | None
    coremltools_version: str | None
    coremltools_source: str | None
    experiment: str | None
    palettization: list[dict]
    palettization_excluded: list[str]
    compiler_eligibility: dict
    unsupported_constructs: list[str]


# ---------------------------------------------------------------------------
# Manifest + model file discovery
# ---------------------------------------------------------------------------

class MlPackageError(RuntimeError):
    """Raised on malformed mlpackage contents."""


@dataclass
class MlPackage:
    path: Path
    manifest: dict
    item_info: dict  # raw Manifest.json itemInfoEntries
    model_pb: bytes
    model_pb_path: Path
    weights_path: Path | None
    files: list[Path]


def open_mlpackage(p: Path) -> MlPackage:
    """Open a .mlpackage directory and load the manifest + model proto.

    Refuses anything that does not look like an mlpackage directory
    (presence of Manifest.json + a Data/<author>/model.mlmodel file).
    """
    p = Path(p).resolve()
    if not p.is_dir():
        raise MlPackageError(f"{p}: not a directory")

    manifest_path = p / "Manifest.json"
    if not manifest_path.is_file():
        raise MlPackageError(f"{p}: missing Manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item_info = manifest.get("itemInfoEntries", {})
    if not item_info:
        raise MlPackageError(f"{manifest_path}: no itemInfoEntries")

    model_pb_path: Path | None = None
    weights_path: Path | None = None
    for info in item_info.values():
        rel = info.get("path")
        if not rel:
            continue
        target = p / "Data" / rel
        if info.get("name") == "model.mlmodel":
            model_pb_path = target
        elif info.get("name") == "weights":
            weights_path = target

    if model_pb_path is None:
        raise MlPackageError(f"{p}: Manifest.json does not name a model.mlmodel")
    if not model_pb_path.is_file():
        raise MlPackageError(f"{model_pb_path}: model file missing")

    files = sorted([f for f in p.rglob("*") if f.is_file()])
    return MlPackage(
        path=p,
        manifest=manifest,
        item_info=item_info,
        model_pb=model_pb_path.read_bytes(),
        model_pb_path=model_pb_path,
        weights_path=weights_path,
        files=files,
    )


# ---------------------------------------------------------------------------
# Model proto parsing
# ---------------------------------------------------------------------------

# Core ML spec message layout (from specification.proto, Model section):
#
#   message Model {
#     optional uint64 specificationVersion = 1;
#     optional ModelDescription description = 2;
#     oneof Type {
#       .. not used here, covered by description
#     }
#     optional MILSpec.Program program = 502;  // the ML Program block
#     .. metadata fields (100+)
#   }

_MODEL_SPEC_VERSION = 1
_MODEL_DESCRIPTION = 2
_MODEL_PROGRAM = 502
# ModelDescription fields (verified by walking an actual coremltools 9.0
# export of parakeet-tdt-0.6b-v3-coreml, sha b650695):
#   field 1:  inputs (NamedValueType repeated)
#   field 10: outputs (NamedValueType repeated)
#   field 100..: kv metadata
_MODEL_DESCRIPTION_NAME = 1  # unused; placeholder for compatibility
_MODEL_DESCRIPTION_INPUTS = 1
_MODEL_DESCRIPTION_OUTPUTS = 10
_MODEL_DESCRIPTION_METADATA = 100

# MIL Program (verified):
#   f1 = version (int)
#   f2 = functions[] (Function messages, packed)
#   f4 = build info
_PROGRAM_FUNCTIONS = 2
_PROGRAM_BUILD_INFO = 4

# Function message (verified):
#   f1 = name
#   f2 = FunctionBody wrapper (a synthetic wrapper we open to find
#        inputs[], opset, body Block)
_FUNCTION_NAME = 1
_FUNCTION_BODY_WRAPPER = 2

# FunctionBody wrapper (verified):
#   f1 = inputs[] (NamedValueType)
#   f2 = opset (string)
#   f3 = Block (the actual graph)
_FBODY_INPUTS = 1
_FBODY_OPSET = 2
_FBODY_BLOCK = 3

# Block (verified):
#   f1 = version ("CoreML8")
#   f2 = BlockBody wrapper (synthetic): f2=outputs, f3=operations[]
_BLOCK_VERSION = 1
_BLOCK_BODY = 2

# BlockBody (verified):
#   f2 = outputs (NamedValueType[])
#   f3 = operations[] (Operation messages)
_BBODY_OUTPUTS = 2
_BBODY_OPERATIONS = 3

# Operation message (verified):
#   f1 = type (string, e.g. "conv", "linear")
#   f2 = inputs[] (NamedValueType)
#   f3 = outputs[] (NamedValueType)
#   f4 = attributes[] (Attribute)
#   f5 = nested blocks[] (for control flow)
_OP_TYPE = 1
_OP_INPUTS = 2
_OP_OUTPUTS = 3
_OP_ATTRIBUTES = 4
_OP_BLOCKS = 5

# Operation:
#   message Operation {
#     optional string type = 1;            // e.g. "conv", "linear"
#     repeated NamedValueType inputs = 2;
#     repeated NamedValueType outputs = 3;
#     repeated Attribute attributes = 4;
#     repeated Block blocks = 5;            // nested control flow
#     ...
#   }
_OP_TYPE = 1
_OP_INPUTS = 2
_OP_OUTPUTS = 3
_OP_ATTRIBUTES = 4
_OP_BLOCKS = 5
_NVT_NAME = 1
_NVT_TYPE = 3

# ValueType oneof (verified on real coremltools 9.0 encoder):
#   oneof Type {
#     TensorType tensorType = 5;   # the wire-2 wrapper encloses the TensorType
#     StateTensor stateTensor = ..;
#     ListType listType = ..;
#   }
_VT_TENSOR_TYPE = 5

# TensorType (verified):
#   message TensorType {
#     repeated int64 shape = 1;          # packed varints
#     optional uint64 dataType = 2;       # large enum value (coremltools 9.0 emits MIL-style dtype numbers, e.g. 0x10060 = fp16)
#     ...
#   }
_TT_SHAPE = 1
_TT_DATATYPE = 2

# Coremltools 9.0 emits MIL DataType values that map to standard
# CoreML DataType enum 1..14 with an offset (e.g. 0x10060). Strip the
# 0x10000 high-bit prefix when present before consulting the table.
_MIL_DTYPE_OFFSET = 0x10000


def _read_tensor_type(blob: bytes) -> TensorSpec:
    name = ""
    dtype_name = "unknown"
    shape: list[int] = []
    for f in proto.iter_fields(blob):
        if f.number == _TT_SHAPE and f.wire == proto.WIRE_LENGTH_DELIMITED:
            sh = bytes(f.value)
            # Two encodings exist in coremltools 9.0:
            # 1) Packed repeated int64: a single blob containing all
            #    shape dims as packed varints.
            # 2) Repeated message-wrapped shape: each dim is a
            #    wire-2 wrapper carrying one int64.
            # Both forms begin with raw shape varints; the wrapper
            # case prepends `0a LL 08 NN` per dim. We strip any
            # `0a LL` header before reading varints, but the simpler
            # approach is: try packing first, and if that fails fall
            # back to walking nested wrapped dims.
            # Strategy: look for any inner length-delimited fields;
            # if found, they are the wrapped shape entries.
            inner = [ff for ff in proto.iter_fields(sh)
                     if ff.wire == proto.WIRE_LENGTH_DELIMITED]
            if inner:
                for ff in inner:
                    inner_blob = bytes(ff.value)
                    for iff in proto.iter_fields(inner_blob):
                        if iff.number == 1 and iff.wire == proto.WIRE_VARINT:
                            shape.append(int(iff.value))
                            break
            else:
                pos = 0
                while pos < len(sh):
                    try:
                        v, pos = proto.read_varint(sh, pos)
                    except proto.ProtoDecodeError:
                        break
                    shape.append(int(v))
        elif f.number == _TT_DATATYPE and f.wire == proto.WIRE_VARINT:
            raw = int(f.value)
            stripped = raw & 0xFFFF
            # Verified by walking the public Parakeet package (coremltools
            # 9.0.0, dialect TorchScript):
            #   0x10020 == MIL float16 (encoder_hidden, decoder_hidden,
            #                       token_logits, attention q/k/v proj, …)
            #   0x20020 == MIL int32   (attention_mask, encoder_mask)
            # The high nibble selects the kind (1=float, 2=int); the
            # low-byte 0x20 carries the same MIL dtype number. Earlier
            # drafts of this module guessed 0x10060; that was wrong.
            if stripped == raw:
                dtype_name = _DATATYPE_NAMES.get(stripped, f"dataType={raw}")
            elif stripped in (0x0020, 0x0060):
                kind = (raw >> 16) & 0xF
                if kind == 1:
                    dtype_name = "float16"
                elif kind == 2:
                    dtype_name = "int32"
                elif kind == 4:
                    dtype_name = "uint8"
                else:
                    dtype_name = f"mil_dtype=0x{raw:04x}"
            else:
                dtype_name = f"mil_dtype=0x{raw:04x}"

def _read_named_value(blob: bytes) -> tuple[str, TensorSpec | None, str | None]:
    name = ""
    tensor: TensorSpec | None = None
    kind: str | None = None
    for f in proto.iter_fields(blob):
        if f.number == _NVT_NAME and f.wire == proto.WIRE_LENGTH_DELIMITED:
            name = bytes(f.value).decode("utf-8", errors="replace")
        elif f.number == _NVT_TYPE and f.wire == proto.WIRE_LENGTH_DELIMITED:
            vt = bytes(f.value)
            # Two Type encodings exist in coremltools 9.0 output:
            #   * legacy oneof: VT.f5 wraps a TensorType message.
            #   * MIL-style:   VT.f1 wraps rank + dtype + shape directly.
            # Try the MIL-style first because that is what the encoder
            # export emits; fall back to the legacy oneof for older
            # packages.
            for vtf in proto.iter_fields(vt):
                if vtf.wire != proto.WIRE_LENGTH_DELIMITED:
                    continue
                sub = bytes(vtf.value)
                if vtf.number == 1:
                    # MIL-style: TT directly. Reuse the same TensorType
                    # walker we use for the legacy path by re-wrapping.
                    tensor = _read_tensor_type(sub)
                    if tensor.shape or tensor.dtype:
                        kind = "tensor"
                elif vtf.number == 5:
                    tensor = _read_tensor_type(sub)
                    kind = "tensor"
                elif vtf.number == 2:
                    tensor = _read_tensor_type(sub)
                    kind = "state"
                elif vtf.number == 3:
                    kind = "list"
    return name, tensor, kind


def _read_function(fn_blob: bytes) -> FunctionSpec:
    """Walk one Function message per the verified layout above.

    Real layout (coremltools 9.0):
      Function { f1: name, f2: FunctionBody{ f1: inputs[],
                                            f2: opset,
                                            f3: Block{ f1: "CoreML8",
                                                     f2: BlockBody{
                                                          f2: outputs[],
                                                          f3: operations[]
                                                      }}}}
    """
    spec = FunctionSpec(name="<unknown>", inputs=[], outputs=[])
    global op_counts, op_attr_examples, op_first_block, op_first_idx
    global op_input_shapes, op_weight_shapes, op_weight_binding, op_fp16, op_evidence_notes
    for f in proto.iter_fields(fn_blob):
        if f.number == _FUNCTION_NAME and f.wire == proto.WIRE_LENGTH_DELIMITED:
            spec.name = bytes(f.value).decode("utf-8", errors="replace")
        elif f.number == _FUNCTION_BODY_WRAPPER and f.wire == proto.WIRE_LENGTH_DELIMITED:
            _walk_function_body(bytes(f.value), spec)
    spec.output_count = len(spec.outputs)
    spec.multi_output = spec.output_count > 1
    spec.op_histogram = sorted(
        [
            OpSummary(
                type=op_type,
                occurrences=count,
                example_attrs=sorted({a for _, _, a in op_attr_examples[op_type]})[:6],
                first_block_index=op_first_block[op_type],
                first_op_index=op_first_idx[op_type],
                observed_input_shapes=sorted(set(op_input_shapes.get(op_type, [])))[:6],
                observed_weight_shapes=sorted(set(op_weight_shapes.get(op_type, [])))[:6],
                fp16=_consensus_bool(op_fp16.get(op_type)),
                weight_binding=_consensus_str(op_weight_binding.get(op_type)),
                envelope_evidence=sorted(set(op_evidence_notes.get(op_type, [])))[:6],
            )
            for op_type, count in op_counts.items()
        ],
        key=lambda s: (-s.occurrences, s.type),
    )
    return spec


def _walk_function_body(body: bytes, spec: FunctionSpec) -> None:
    """Walk a FunctionBody wrapper: inputs[], opset, Block."""
    for f in proto.iter_fields(body):
        if f.wire != proto.WIRE_LENGTH_DELIMITED:
            continue
        sub = bytes(f.value)
        if f.number == _FBODY_INPUTS:
            name, tensor, kind = _read_named_value(sub)
            spec.inputs_raw.append({"name": name, "kind": kind, "tensor": tensor})
            if tensor is not None and kind == "tensor":
                spec.inputs.append(tensor)
        elif f.number == _FBODY_OPSET:
            try:
                spec.spec_field_notes.append(f"opset={bytes(sub).decode('utf-8')}")
            except UnicodeDecodeError:
                pass
        elif f.number == _FBODY_BLOCK:
            _walk_block(sub, spec)


def _walk_block(block: bytes, spec: FunctionSpec) -> None:
    """Walk one Block: f1=version, f2=BlockBody."""
    for f in proto.iter_fields(block):
        if f.number != _BLOCK_BODY or f.wire != proto.WIRE_LENGTH_DELIMITED:
            continue
        _walk_block_body(bytes(f.value), spec)


def _walk_block_body(bb: bytes, spec: FunctionSpec) -> None:
    """Walk a BlockBody: f2=outputs[], f3=operations[]."""
    for f in proto.iter_fields(bb):
        if f.wire != proto.WIRE_LENGTH_DELIMITED:
            continue
        sub = bytes(f.value)
        if f.number == _BBODY_OUTPUTS:
            name, tensor, kind = _read_named_value(sub)
            spec.outputs_raw.append({"name": name, "kind": kind, "tensor": tensor})
            if tensor is not None and kind == "tensor":
                spec.outputs.append(tensor)
        elif f.number == _BBODY_OPERATIONS:
            _ingest_block(sub, spec)



def _consensus_bool(values: set[bool] | None) -> bool | None:
    """Reduce a set of observed booleans to a single confidence value.

    * ``None`` when no evidence.
    * ``True`` or ``False`` when all observed values agree.
    * ``"mixed"`` (a string, not a bool) when observation disagreed.
    """
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _consensus_str(values: set[str] | None) -> str | None:
    """Reduce a set of binding hints to one agreed string, or ``mixed``."""
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    if values == {"constant", "external"}:
        # Most Core ML constant weights live in @model_path/weights/*.bin
        # but are *constant*. The external token is the storage path,
        # not the binding semantics. Treat the storage constant.
        return "constant"
    return "mixed"
op_first_idx: dict[str, int]
# Per-op-type evidence (only populated when the inspector actually sees
# concrete shape/binding/dtype fields in the protobuf). Each list is
# capped to keep the report bounded.
op_input_shapes: dict[str, list[str]]
op_weight_shapes: dict[str, list[str]]
op_weight_binding: dict[str, set[str]]
op_fp16: dict[str, set[bool]]
op_evidence_notes: dict[str, list[str]]


def _ingest_operation(op_blob: bytes, fn: FunctionSpec, block_index: int = 0) -> None:
    """Walk one Operation message and update per-op-type evidence.

    Captures, for each op type, only the concrete evidence the protobuf
    actually exposes:

    * the names of input ports that point at external weights (those
      whose attribute value begins with ``weights/`` or
      ``@model_path/weights/``),
    * whether the weight is bound as a constant (``val``) vs runtime,
    * whether any input/weight dtype reads as ``float16``.

    No "last two dims are M x N" inference is performed here. The
    matvec/gemm/apple-parity-matmul classifier runs on the
    *func*-level evidence in :func:`_eligibility_hints` only.
    """
    global op_counts, op_attr_examples, op_first_block, op_first_idx
    global op_input_shapes, op_weight_shapes, op_weight_binding, op_fp16, op_evidence_notes
    op_type = ""
    op_attrs: list[str] = []
    op_inputs: list[tuple[str, str]] = []  # (port_name, value_string)
    op_attr_blob = b""
    op_blocks_blob: list[bytes] = []
    for of in proto.iter_fields(op_blob):
        if of.number == _OP_TYPE and of.wire == proto.WIRE_LENGTH_DELIMITED:
            op_type = bytes(of.value).decode("utf-8", errors="replace")
        elif of.number == _OP_INPUTS and of.wire == proto.WIRE_LENGTH_DELIMITED:
            inp = bytes(of.value)
            port_name = ""
            for nf in proto.iter_fields(inp):
                if nf.number == _NVT_NAME and nf.wire == proto.WIRE_LENGTH_DELIMITED:
                    port_name = bytes(nf.value).decode("utf-8", errors="replace")
                    break
            op_inputs.append((port_name, "(see const binding)"))
        elif of.number == _OP_ATTRIBUTES and of.wire == proto.WIRE_LENGTH_DELIMITED:
            op_attr_blob = bytes(of.value)
            for af in proto.iter_fields(bytes(of.value)):
                if af.number == 1 and af.wire == proto.WIRE_LENGTH_DELIMITED:
                    try:
                        op_attrs.append(bytes(af.value).decode("utf-8"))
                    except UnicodeDecodeError:
                        pass
                elif af.wire == proto.WIRE_LENGTH_DELIMITED:
                    b = bytes(af.value)
                    if 1 <= len(b) <= 80 and all(0x20 <= x <= 0x7E for x in b):
                        try:
                            op_attrs.append(b.decode("utf-8"))
                        except UnicodeDecodeError:
                            pass
        elif of.number == _OP_BLOCKS and of.wire == proto.WIRE_LENGTH_DELIMITED:
            op_blocks_blob.append(bytes(of.value))
    if op_type:
        op_counts[op_type] += 1
        op_attr_examples.setdefault(op_type, []).extend(
            (block_index, fn.op_total, a) for a in op_attrs[:3]
        )
        if op_type not in op_first_block:
            op_first_block[op_type] = block_index
            op_first_idx[op_type] = fn.op_total
        if op_type in ("while", "cond", "for_loop", "constexpr_block", "branch"):
            fn.has_control_flow = True
            fn.control_flow_ops.append(op_type)
        for af in proto.iter_fields(op_attr_blob):
            if af.wire == proto.WIRE_LENGTH_DELIMITED:
                b = bytes(af.value)
                if b.startswith(b"@model_path/weights/") or b.startswith(b"weights/"):
                    ref = b.decode("utf-8", errors="replace")
                    fn.external_weight_refs.append(ref)
                    op_weight_binding.setdefault(op_type, set()).add("external")
        for af in proto.iter_fields(op_attr_blob):
            if af.wire != proto.WIRE_LENGTH_DELIMITED:
                continue
            b = bytes(af.value)
            if b in (b"float16", b"fp16"):
                op_fp16.setdefault(op_type, set()).add(True)
            elif b in (b"float32", b"fp32"):
                op_fp16.setdefault(op_type, set()).add(False)
            elif b == b"val":
                op_weight_binding.setdefault(op_type, set()).add("constant")
            elif b == b"name":
                op_weight_binding.setdefault(op_type, set()).add("runtime")
        for nb in op_blocks_blob:
            # Nested blocks reuse the same walker; treat them as a fresh
            # block index for "first occurrence" reporting.
            for nbf in proto.iter_fields(nb):
                if nbf.number == _BBODY_OPERATIONS and nbf.wire == proto.WIRE_LENGTH_DELIMITED:
                    _ingest_operation(bytes(nbf.value), fn, block_index + 1)

# Backward-compat alias retained so any in-process callers still work.
_ingest_block = _ingest_operation

def reset_walk_state() -> None:
    """Clear the walker module globals between functions / packages."""
    global op_counts, op_attr_examples, op_first_block, op_first_idx
    global op_input_shapes, op_weight_shapes, op_weight_binding, op_fp16, op_evidence_notes
    op_counts = Counter()
    op_attr_examples = {}
    op_first_block = {}
    op_first_idx = {}
    op_input_shapes = {}
    op_weight_shapes = {}
    op_weight_binding = {}
    op_fp16 = {}
    op_evidence_notes = {}


# ---------------------------------------------------------------------------
# Top-level model spec metadata (Model field 100+ metadata, build info)
# ---------------------------------------------------------------------------

def _read_model_metadata(model_blob: bytes) -> tuple[dict, dict, dict]:
    """Return (metadata_dict, kv_metadata, build_info_dict)."""
    metadata: dict = {}
    kv_metadata: dict = {}
    build_info: dict = {}
    pos = 0
    # Top-level Model messages are read with iter_fields.
    specs = []
    for f in proto.iter_fields(model_blob):
        if f.number == _MODEL_SPEC_VERSION and f.wire == proto.WIRE_VARINT:
            specs.append(("spec_version", int(f.value)))
        elif f.number == _MODEL_DESCRIPTION and f.wire == proto.WIRE_LENGTH_DELIMITED:
            desc = bytes(f.value)
            for d in proto.iter_fields(desc):
                if d.number == _MODEL_DESCRIPTION_NAME and d.wire == proto.WIRE_LENGTH_DELIMITED:
                    metadata["name"] = bytes(d.value).decode("utf-8", errors="replace")
                elif d.number == _MODEL_DESCRIPTION_INPUTS and d.wire == proto.WIRE_LENGTH_DELIMITED:
                    metadata.setdefault("inputs_raw", []).append(bytes(d.value))
                elif d.number == _MODEL_DESCRIPTION_OUTPUTS and d.wire == proto.WIRE_LENGTH_DELIMITED:
                    metadata.setdefault("outputs_raw", []).append(bytes(d.value))
                elif d.number == _MODEL_DESCRIPTION_METADATA and d.wire == proto.WIRE_LENGTH_DELIMITED:
                    metadata.setdefault("kv", []).append(bytes(d.value))
        elif f.number == _MODEL_PROGRAM and f.wire == proto.WIRE_LENGTH_DELIMITED:
            metadata["program_size_bytes"] = len(bytes(f.value))
        elif f.number >= 100 and f.wire == proto.WIRE_LENGTH_DELIMITED:
            # Author / license / quantization / minimum deployment target etc.
            # Field layout for Model metadata is repeated name(string=1)/value(string=2) per
            # specification.proto. The blob may be one of those or a kv pair; we extract
            # strings only.
            b = bytes(f.value)
            strs: list[str] = []
            for sf in proto.iter_fields(b):
                if sf.wire == proto.WIRE_LENGTH_DELIMITED and 0 < len(bytes(sf.value)) <= 200:
                    sb = bytes(sf.value)
                    try:
                        strs.append(sb.decode("utf-8"))
                    except UnicodeDecodeError:
                        pass
            metadata.setdefault("model_fields", []).append((f.number, strs))
        elif f.number >= 200 and f.wire == proto.WIRE_LENGTH_DELIMITED:
            # Build-info message.
            for sf in proto.iter_fields(bytes(f.value)):
                if sf.wire == proto.WIRE_LENGTH_DELIMITED:
                    sb = bytes(sf.value)
                    try:
                        build_info[sf.number] = sb.decode("utf-8", errors="replace")
                    except UnicodeDecodeError:
                        build_info[sf.number] = None
    return metadata, kv_metadata, build_info


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def inspect(pkg_path: Path) -> InspectionReport:
    """Open ``pkg_path`` and return an :class:`InspectionReport`.

    Pure read-only operation. Does not touch the ANE device, does not
    require a Core ML runtime, and does not allocate output buffers.
    """
    pkg = open_mlpackage(pkg_path)
    metadata, kv_metadata, build_info = _read_model_metadata(pkg.model_pb)

    # MIL program functions.
    functions: list[FunctionSpec] = []
    # Find the program field in the model blob (field 502 wire 2).
    program_bytes: bytes | None = None
    for f in proto.iter_fields(pkg.model_pb):
        if f.number == _MODEL_PROGRAM and f.wire == proto.WIRE_LENGTH_DELIMITED:
            program_bytes = bytes(f.value)
            break

    spec_version = None
    for k, v in metadata.get("model_fields", []) or []:
        pass
    # Re-parse top-level for spec version if not captured yet.
    for f in proto.iter_fields(pkg.model_pb):
        if f.number == _MODEL_SPEC_VERSION and f.wire == proto.WIRE_VARINT:
            spec_version = int(f.value)
            break

    if program_bytes is not None:
        # Program message: field 2 = functions[]
        for fn_blob in proto.iter_nested(program_bytes, 2):
            reset_walk_state()
            functions.append(_read_function(fn_blob))

    # Weight references: collect from every function's external_weight_refs.
    seen_refs: list[str] = []
    seen_set: set[str] = set()
    for fn in functions:
        for ref in fn.external_weight_refs:
            if ref not in seen_set:
                seen_set.add(ref)
                seen_refs.append(ref)

    files_meta: list[dict] = []
    for f in pkg.files:
        try:
            rel = f.relative_to(pkg.path)
            size = f.stat().st_size
        except (OSError, ValueError):
            continue
        files_meta.append({
            "path": str(rel),
            "size": size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        })

    weights: list[dict] = []
    if pkg.weights_path is not None:
        for w in sorted(pkg.weights_path.iterdir()):
            if w.is_file():
                weights.append({
                    "path": str(w.relative_to(pkg.path)),
                    "size": w.stat().st_size,
                    "sha256": hashlib.sha256(w.read_bytes()).hexdigest(),
                })

    # Pull palettization + deployment metadata out of the model_fields blobs.
    palettization: list[dict] = []
    palettization_excluded: list[str] = []
    minimum_deployment_target = None
    compute_units = None
    precision = None
    coremltools_version = None
    coremltools_source = None
    experiment = None
    description = None
    author = None
    license_str = None
    for f in metadata.get("model_fields", []) or []:
        number, strs = f
        if not strs:
            continue
        key = strs[0]
        val = strs[1] if len(strs) > 1 else ""
        if key == "palettization":
            palettization.append({"raw": val})
            fn.palettization_seen = True if functions else False
            for fn in functions:
                fn.palettization_seen = True
                fn.palettization_metadata.append({"raw": val})
        elif key == "palettization_excluded":
            palettization_excluded.append(val)
        elif key == "minimum_deployment_target":
            minimum_deployment_target = val
        elif key == "compute_units":
            compute_units = val
        elif key == "precision":
            precision = val
        elif key == "coremltools-version":
            coremltools_version = val
        elif key == "coremltools-source":
            coremltools_source = val
        elif key == "experiment":
            experiment = val
        elif key == "description":
            description = val
        elif key == "author":
            author = val
        elif key == "license":
            license_str = val

    # Compiler eligibility: every op must map to a known ML primitive,
    # OR be an unsupported control-flow op that the frontend must reject.
    KNOWN_OPS: set[str] = set()
    for fn in functions:
        for op in fn.op_histogram:
            KNOWN_OPS.add(op.type)
    KNOWN_MIL_OPS = {
        "abs","add","add_param","arithmetic","average_pool","batch_norm","cast",
        "category_map","ceil","clip","concat","const","constexpr_block","constant",
        "conv","conv_transpose","cos","cumprod","cumsum","cummax","cummin","deconvolution",
        "depthwise_conv","dot","elementwise","elu","equal","erf","exp","expand_dims",
        "fake_quant","fill","flatten","floor","floor_div","gather","gather_nd","gelu",
        "greater","greater_equal","gru","identity","inner_product","leaky_relu","less",
        "less_equal","linear","log","logical_and","logical_not","logical_or","log_softmax",
        "lstm","matmul","max_pool","maximum","mean","minimum","mod","mul","multiply",
        "mvn","neg","non_max_suppression","non_zero","not_equal","one_hot","pad","permute",
        "permute_batch","pool","pow","random","range","real_div","reduce","reduce_axis1",
        "reduce_l1","reduce_l2","reduce_log_sum","reduce_log_sum_exp","reduce_max",
        "reduce_mean","reduce_min","reduce_prod","reduce_sum","reduce_sum_square",
        "register","relu","reshape","reverse","rnn","round","scale","select","shape",
        "sigmoid","sign","sin","sinh","slice","slice_by_index","slice_by_size",
        "slice_update","softmax","softplus","softsign","split","sqrt","squeeze","stack",
        "sub","subtract","tan","tanh","tile","topk","transpose","truncate_div",
        "truncate_mod","upsample_bilinear","upsample_nearest","where","while","cond",
        "branch","for_loop","loop","image_resize","constexpr_lut_to_dense","make_list",
        "list_gather","list_scatter","list_size","list_read","list_write","copy","quantize",
        "dequantize","embedding","matmul","band_part","scatter","scatter_nd","bitcast",
        "addmm","adaptive_avg_pool","adaptive_max_pool","instance_norm","group_norm",
        "layer_norm","roi_pool","pixel_unshuffle","pixel_shuffle","cumprod","log1p",
        "softsign","leaky_relu","erf","erfc","svd","batch_matmul","cumsum","input_debug",
        "constexpr_lut_to_dense","constexpr_sparse_to_dense","constexpr_cast",
        "constexpr_affine_dequantize","constexpr_lut","constexpr_divide",
        "constexpr_pow","constexpr_sub","constexpr_add","constexpr_mul",
        "constexpr_cast","constexpr","constexpr_const","reverse",
        # Fast variants present in coremltools 9.0 outputs:
        "fast_gelu","fast_sigmoid","fast_tanh",
    }
    unsupported_ops = sorted(KNOWN_OPS - KNOWN_MIL_OPS)

    # Aggregate unsupported constructs across functions.
    UNSUPPORTED_CONSTRUCTS: list[str] = []
    for fn in functions:
        for op in fn.op_histogram:
            if op.type not in KNOWN_MIL_OPS:
                fn.unsupported_constructs.append(op.type)
        if fn.has_control_flow:
            UNSUPPORTED_CONSTRUCTS.extend(f"{fn.name}: {o}" for o in fn.control_flow_ops)
    UNSUPPORTED_CONSTRUCTS = sorted(set(UNSUPPORTED_CONSTRUCTS) | set(unsupported_ops))

    quantization = precision or ("palettized" if palettization else "unknown")

    compiler_eligibility = {
        "spec_version": spec_version,
        "spec_supports_mlprogram": spec_version is not None and spec_version >= 6,
        "is_mlprogram": functions != [],
        "function_count": len(functions),
        "op_total": sum(fn.op_total for fn in functions),
        "unsupported_op_types": sorted(unsupported_ops),
        "has_palettization": bool(palettization),
        "control_flow_present": any(fn.has_control_flow for fn in functions),
    }

    return InspectionReport(
        package_path=str(pkg.path),
        package_format="Core ML ML Program (spec >= 6)" if functions else "Unknown",
        file_format_version=pkg.manifest.get("fileFormatVersion"),
        root_identifier=pkg.manifest.get("rootModelIdentifier"),
        specification_version=spec_version,
        model_description=description,
        author=author,
        license=license_str,
        metadata=metadata,
        files=files_meta,
        weights=weights,
        functions=functions,
        quantization=quantization,
        minimum_deployment_target=minimum_deployment_target,
        compute_units=compute_units,
        precision=precision,
        coremltools_version=coremltools_version,
        coremltools_source=coremltools_source,
        experiment=experiment,
        palettization=palettization,
        palettization_excluded=palettization_excluded,
        compiler_eligibility=compiler_eligibility,
        unsupported_constructs=UNSUPPORTED_CONSTRUCTS,
    )


def report_to_dict(report: InspectionReport) -> dict:
    """Serialise an :class:`InspectionReport` to a JSON-safe dictionary."""

    def _fn(fn: FunctionSpec) -> dict:
        return {
            "name": fn.name,
            "inputs": [dataclasses.asdict(t) for t in fn.inputs],
            "outputs": [dataclasses.asdict(t) for t in fn.outputs],
            "output_count": fn.output_count,
            "multi_output": fn.multi_output,
            "op_total": fn.op_total,
            "op_histogram": [dataclasses.asdict(o) for o in fn.op_histogram],
            "has_control_flow": fn.has_control_flow,
            "control_flow_ops": fn.control_flow_ops,
            "external_weight_refs": fn.external_weight_refs,
            "palettization_metadata": fn.palettization_metadata,
            "unsupported_constructs": fn.unsupported_constructs,
            # compiler_eligibility_hints: ONLY evidence the inspector
            # actually saw. Absence is the safe default.
            "compiler_eligibility_hints": _eligibility_hints(fn),
        }



def _eligibility_hints(fn: FunctionSpec) -> dict:
    """Build per-function eligibility annotations from PROVEN evidence only.

    The choices here are intentionally conservative. We do NOT:

    * assume a "last two dims are M x N" reading for matmul/linear;
    * infer matvec eligibility without seeing the actual weight shape
      and binding chain;
    * pretend the compiler can accept a dtype the model does not
      actually carry.

    The report therefore exposes only:

    * whether the function has multiple outputs (compiler envelope
      requires per-output functions);
    * whether the function statically declares every input/output shape
      (the compiler envelope requires positive static shapes);
    * whether the function uses palettization (decompression will need
      to happen before any H13 kernel sees the weight);
    * per-op observed dtypes and binding hints (for Phase 3).
    """
    static_inputs = [t for t in fn.inputs if t.is_static() and all(d > 0 for d in t.shape)]
    static_outputs = [t for t in fn.outputs if t.is_static() and all(d > 0 for d in t.shape)]
    dtype_inputs = sorted({t.dtype for t in fn.inputs})
    dtype_outputs = sorted({t.dtype for t in fn.outputs})
    return {
        "multi_output": fn.multi_output,
        "output_count": fn.output_count,
        "static_inputs": len(static_inputs),
        "static_outputs": len(static_outputs),
        "all_shapes_static": (
            len(static_inputs) == len(fn.inputs) and len(static_outputs) == len(fn.outputs)
        ),
        "input_dtypes": dtype_inputs,
        "output_dtypes": dtype_outputs,
        "palettization_present": fn.palettization_seen,
        "control_flow_ops": list(fn.control_flow_ops),
        # Per-op evidence is in op_histogram[*].fp16 / .weight_binding.
    }

    return {
        "package_path": report.package_path,
        "package_format": report.package_format,
        "file_format_version": report.file_format_version,
        "root_identifier": report.root_identifier,
        "specification_version": report.specification_version,
        "model_description": report.model_description,
        "author": report.author,
        "license": report.license,
        "metadata": {k: v for k, v in report.metadata.items() if k != "model_fields"},
        "files": report.files,
        "weights": report.weights,
        "functions": [_fn(f) for f in report.functions],
        "quantization": report.quantization,
        "minimum_deployment_target": report.minimum_deployment_target,
        "compute_units": report.compute_units,
        "precision": report.precision,
        "coremltools_version": report.coremltools_version,
        "coremltools_source": report.coremltools_source,
        "experiment": report.experiment,
        "palettization": report.palettization,
        "palettization_excluded": report.palettization_excluded,
        "compiler_eligibility": report.compiler_eligibility,
        "unsupported_constructs": report.unsupported_constructs,
    }


def render_human(report: InspectionReport) -> str:
    """Render the report for terminal output."""
    a("")
    a("compiler eligibility (from raw evidence only):")
    a(f"  spec_version={report.specification_version}  mlprogram={'yes' if ce['is_mlprogram'] else 'no'}  "
      f"functions={ce['function_count']}  ops={ce['op_total']}")
    a(f"  palettized={ce['has_palettization']}  control_flow={ce['control_flow_present']}")
    if ce["unsupported_op_types"]:
        a(f"  unsupported_op_types: {', '.join(ce['unsupported_op_types'])}")
    a("")
    a(f"weights ({len(report.weights)}):")
    for w in report.weights:
        a(f"  - {w['path']}  size={w['size']}  sha256={w['sha256'][:16]}…")
    a("")
    for fn in report.functions:
        a(f"function: {fn.name}")
        a(f"  inputs ({len(fn.inputs)}):")
        for t in fn.inputs:
            a(f"    - {t.name}  dtype={t.dtype}  shape={t.shape}  static={t.is_static()}")
        a(f"  outputs ({len(fn.outputs)}):")
        for t in fn.outputs:
            a(f"    - {t.name}  dtype={t.dtype}  shape={t.shape}  static={t.is_static()}")
        a(f"  output_count={fn.output_count}  multi_output={fn.multi_output}")
        if fn.has_control_flow:
            a(f"  control_flow: {', '.join(fn.control_flow_ops)}")
        a(f"  operations ({fn.op_total}):")
        for op in fn.op_histogram[:32]:
            attrs = ", ".join(op.example_attrs[:3]) if op.example_attrs else ""
            evidence = []
            if op.fp16 is not None:
                evidence.append(f"fp16={op.fp16}")
            if op.weight_binding is not None:
                evidence.append(f"weight={op.weight_binding}")
            ev_str = f"  [{', '.join(evidence)}]" if evidence else ""
            a(f"    - {op.type}: {op.occurrences}{('  attrs='+attrs) if attrs else ''}{ev_str}")
        if len(fn.op_histogram) > 32:
            a(f"    … +{len(fn.op_histogram) - 32} more op types")
        if fn.external_weight_refs:
            a(f"  external weight refs ({len(fn.external_weight_refs)}):")
            for ref in fn.external_weight_refs[:8]:
                a(f"    - {ref}")
        a("")
        a(f"experiment: {report.experiment}")
    if report.palettization:
        a("palettization metadata:")
        for p in report.palettization:
            a(f"  - {p['raw']}")
    if report.palettization_excluded:
        a(f"palettization excluded: {', '.join(report.palettization_excluded)}")
    if report.minimum_deployment_target:
        a(f"deploy target: {report.minimum_deployment_target}")
    if report.compute_units:
        a(f"compute units (compile-time): {report.compute_units}")
    if report.coremltools_version:
        a(f"coremltools: {report.coremltools_version} (source: {report.coremltools_source or '-'})")
    a("")
    ce = report.compiler_eligibility
    a(f"compiler eligibility: spec>={6}={'yes' if ce['spec_supports_mlprogram'] else 'no'}  "
      f"mlprogram={'yes' if ce['is_mlprogram'] else 'no'}  functions={ce['function_count']}  "
      f"ops={ce['op_total']}  palettized={ce['has_palettization']}  control_flow={ce['control_flow_present']}")
    if report.unsupported_constructs:
        a(f"unsupported constructs: {', '.join(report.unsupported_constructs) or '-'}")
    a("")
    a(f"weights ({len(report.weights)}):")
    for w in report.weights:
        a(f"  - {w['path']}  size={w['size']}  sha256={w['sha256'][:16]}…")
    a("")
    for fn in report.functions:
        a(f"function: {fn.name}")
        a(f"  inputs ({len(fn.inputs)}):")
        for t in fn.inputs:
            a(f"    - {t.name}  dtype={t.dtype}  shape={t.shape}  static={t.is_static()}")
        a(f"  outputs ({len(fn.outputs)}):")
        for t in fn.outputs:
            a(f"    - {t.name}  dtype={t.dtype}  shape={t.shape}  static={t.is_static()}")
        if fn.has_control_flow:
            a(f"  control_flow: {', '.join(fn.control_flow_ops)}")
        a(f"  operations ({fn.op_total}):")
        for op in fn.op_histogram[:32]:
            attrs = ", ".join(op.example_attrs[:3]) if op.example_attrs else ""
            a(f"    - {op.type}: {op.occurrences}{('  attrs='+attrs) if attrs else ''}")
        if len(fn.op_histogram) > 32:
            a(f"    … +{len(fn.op_histogram) - 32} more op types")
        if fn.external_weight_refs:
            a(f"  external weight refs ({len(fn.external_weight_refs)}):")
            for ref in fn.external_weight_refs[:8]:
                a(f"    - {ref}")
        a("")
    return "\n".join(out)
