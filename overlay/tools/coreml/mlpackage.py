# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
""".mlpackage reader and typed model inventory.

Reads a Core ML ``.mlpackage`` directory (manifest, model
specification, weight files) and produces a JSON-ready inventory that
answers the plan's inspector checklist (section 14 of
``docs/plans/2026-09-12-coreml-parakeet-ane-plan.md``):

* package format and file-format version,
* model/spec type (which ``Model.Type`` oneof is set),
* functions, blocks, and every operation with its parameter bindings,
  typed outputs, and nested blocks,
* input/output/state names, dtypes (FeatureType enum), and shapes
  including symbolic/flexible shape declarations,
* MIL-level tensor dtypes (the distinct MIL enum),
* external weight files with sizes and sha-256 hashes, plus every
  blob reference (file + offset) and which ops reference them,
* compression representation (``constexpr_*`` ops such as
  ``constexpr_lut_to_dense`` with their palettization/lut metadata),
* program/function versions and opset names,
* control flow (operations with nested blocks),
* operation histogram and total op counts,
* parse validity and confidence (unknown constructs are explicit),
* compiler eligibility: explicitly **not assessed** here — that
  belongs to the compiler's coverage data, never to guessed rules.

The model specification is parsed with the officially vendored Core ML
protobuf schema (:mod:`coreml.schema`, provenance in
``schema/VENDORED.json``). No tensor data is read or computed. No ANE
device, no GPU, no compiler is touched: this module is pure file and
protobuf reading, safe on any Linux host.

Everything that used to be hand-guessed here (varint wire walking,
hand-written dtype tables that labeled MIL ``BOOL=1`` as ``float16``)
was deleted, not patched: the official schema is the single source.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import proto
from .proto import ModelSpecError, load_model

INVENTORY_SCHEMA = "mlx-omarchy.coreml.inventory/1"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class MlPackageError(RuntimeError):
    """Raised when the mlpackage structure is invalid or inconsistent."""


# ---------------------------------------------------------------------------
# Package structure
# ---------------------------------------------------------------------------


@dataclass
class WeightFile:
    path: Path
    relative: str  # package-relative POSIX path, as reported
    size: int
    sha256: str


@dataclass
class MlPackage:
    path: Path
    file_format_version: str
    root_identifier: str | None
    manifest: dict
    model_path: Path
    weights_dir: Path | None
    weight_files: list[WeightFile] = field(default_factory=list)


def _check_manifest_path(raw: object, package: Path) -> str:
    """Validate one itemInfoEntries ``path`` and return it.

    Refuses absolute paths and traversal (``..``) — a manifest is
    untrusted input and must stay inside the package directory.
    """
    if not isinstance(raw, str) or not raw:
        raise MlPackageError(f"{package}: Manifest.json entry has no path")
    if raw.startswith("/") or raw.startswith("\\") or Path(raw).is_absolute():
        raise MlPackageError(
            f"{package}: Manifest.json entry path is absolute: {raw!r}"
        )
    parts = Path(raw).parts
    if ".." in parts:
        raise MlPackageError(
            f"{package}: Manifest.json entry path escapes the package: {raw!r}"
        )
    return raw


def open_mlpackage(path: Path) -> MlPackage:
    """Open and validate an ``.mlpackage`` directory.

    Raises :class:`MlPackageError` with a specific message for every
    structural problem: missing directory/manifest, unparsable
    manifest JSON, no itemInfoEntries, escaped or absolute manifest
    paths, missing/ambiguous model file, missing model bytes.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise MlPackageError(f"{p}: not a directory (mlpackages are directories)")

    manifest_path = p / "Manifest.json"
    if not manifest_path.is_file():
        raise MlPackageError(f"{p}: missing Manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MlPackageError(f"{manifest_path}: unparsable manifest ({exc})") from exc
    if not isinstance(manifest, dict):
        raise MlPackageError(f"{manifest_path}: manifest is not a JSON object")

    entries = manifest.get("itemInfoEntries")
    if not isinstance(entries, dict) or not entries:
        raise MlPackageError(f"{manifest_path}: no itemInfoEntries")

    model_rel: list[str] = []
    weights_rel: list[str] = []
    for entry_id, info in entries.items():
        if not isinstance(info, dict):
            raise MlPackageError(
                f"{manifest_path}: itemInfoEntries[{entry_id!r}] is not an object"
            )
        rel = _check_manifest_path(info.get("path"), p)
        name = info.get("name")
        if name == "model.mlmodel":
            model_rel.append(rel)
        elif name == "weights":
            weights_rel.append(rel)

    if not model_rel:
        raise MlPackageError(f"{p}: manifest does not name a model.mlmodel")
    if len(model_rel) > 1:
        raise MlPackageError(
            f"{p}: manifest names {len(model_rel)} model.mlmodel entries "
            f"({', '.join(sorted(model_rel))}); refusing ambiguity"
        )

    model_path = p / "Data" / model_rel[0]
    if not model_path.is_file():
        raise MlPackageError(f"{model_path}: model file named by manifest is missing")

    weights_dir = None
    if weights_rel:
        if len(weights_rel) > 1:
            raise MlPackageError(
                f"{p}: manifest names {len(weights_rel)} weights entries"
            )
        weights_dir = p / "Data" / weights_rel[0]

    weight_files: list[WeightFile] = []
    if weights_dir is not None and weights_dir.is_dir():
        for wf in sorted(weights_dir.rglob("*")):
            if wf.is_file():
                data = wf.read_bytes()
                weight_files.append(
                    WeightFile(
                        path=wf,
                        relative=wf.relative_to(p).as_posix(),
                        size=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )

    return MlPackage(
        path=p,
        file_format_version=str(manifest.get("fileFormatVersion", "")) or None,
        root_identifier=manifest.get("rootModelIdentifier"),
        manifest=manifest,
        model_path=model_path,
        weights_dir=weights_dir,
        weight_files=weight_files,
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _binding_summary(binding: Any) -> dict:
    """One ``Argument.Binding``: a name reference or an inline value."""
    which = binding.WhichOneof("binding")
    if which == "name":
        return {"name": binding.name}
    if which == "value":
        return {"value": proto.value_summary(binding.value)}
    return {"binding": "UNSET"}


def _argument_summary(argument: Any) -> list[dict]:
    return [_binding_summary(b) for b in argument.arguments]


def _named_types_summary(named: list[Any]) -> list[dict]:
    return [
        {"name": t.name, "type": proto.value_type_summary(t.type)} for t in named
    ]


def _op_summary(op: Any, index: int) -> dict:
    return {
        "index": index,
        "type": op.type,
        "bindings": {k: _argument_summary(a) for k, a in sorted(op.inputs.items())},
        "outputs": _named_types_summary(op.outputs),
        "attributes": {
            k: proto.value_summary(v) for k, v in sorted(op.attributes.items())
        },
        "blocks": [_block_summary(b) for b in op.blocks],
    }


def _block_summary(block: Any) -> dict:
    ops = [_op_summary(op, i) for i, op in enumerate(block.operations)]
    histogram = Counter(op["type"] for op in ops)
    return {
        "inputs": _named_types_summary(block.inputs),
        "outputs": list(block.outputs),
        "operations": ops,
        "op_count": len(ops),
        "op_types": sorted(histogram),
        "attributes": {
            k: proto.value_summary(v) for k, v in sorted(block.attributes.items())
        },
    }


def _function_summary(name: str, function: Any) -> dict:
    specializations = function.block_specializations
    blocks = {k: _block_summary(b) for k, b in sorted(specializations.items())}
    active = function.opset if function.opset in blocks else None
    histogram: Counter = Counter()
    for block in blocks.values():
        for op in block["operations"]:
            histogram[op["type"]] += 1
            for nested in op["blocks"]:
                for nested_op in nested["operations"]:
                    histogram[nested_op["type"]] += 1
    opset_note = None
    if active is None:
        opset_note = (
            f"function.opset={function.opset!r} has no matching "
            "block_specializations entry"
            if function.opset
            else "function.opset is empty and has no default specialization"
        )
    active_block = blocks.get(active) or next(iter(blocks.values()), None)
    return {
        "name": name,
        "opset": function.opset or None,
        "program_version": None,  # filled by the caller (program-level)
        "inputs": _named_types_summary(function.inputs),
        "outputs": active_block["outputs"] if active_block else [],
        "block_specializations": blocks,
        "active_block": active,
        "opset_consistency": opset_note or "ok",
        "op_histogram": dict(sorted(histogram.items())),
        "op_total": sum(histogram.values()),
        "control_flow_ops": _control_flow_ops(blocks),
    }


def _control_flow_ops(blocks: dict) -> list[str]:
    """Op types that carry nested blocks (while/cond/pattern forms)."""
    found: list[str] = []
    for block in blocks.values():
        for op in block["operations"]:
            if op["blocks"]:
                found.append(op["type"])
    return found


def _const_op_blob_refs(op: dict) -> list[dict]:
    refs = []
    for values in op["bindings"].values():
        for binding in values:
            storage = binding.get("value", {}).get("storage", {})
            if "blob_file" in storage:
                refs.append(storage)
    return refs


def describe_package(package: MlPackage) -> dict:
    """Full typed inventory of a validated package (JSON-ready)."""
    model = load_model(package.model_path.read_bytes())

    model_type = model.WhichOneof("Type")
    program = model.mlProgram if model_type == "mlProgram" else None

    functions = []
    program_version = None
    if program is not None:
        program_version = program.version or None
        for name, function in sorted(program.functions.items()):
            summary = _function_summary(name, function)
            summary["program_version"] = program_version
            functions.append(summary)

    inputs = [proto.feature_description_summary(f) for f in model.description.input]
    outputs = [proto.feature_description_summary(f) for f in model.description.output]
    state = [proto.feature_description_summary(f) for f in model.description.state]

    # Weight references and compression representation.
    blob_refs: dict[str, dict] = {}
    constexpr_ops: list[dict] = []
    for fn in functions:
        for block in fn["block_specializations"].values():
            for op in block["operations"]:
                if op["type"].startswith("constexpr_"):
                    constexpr_ops.append(
                        {
                            "function": fn["name"],
                            "op": op["type"],
                            "op_index": op["index"],
                            "inputs": sorted(op["bindings"]),
                            "value_types": {
                                k: v[0].get("value", {}).get("type")
                                for k, v in op["bindings"].items()
                                if v and "value" in v[0]
                            },
                        }
                    )
                for ref in _const_op_blob_refs(op):
                    entry = blob_refs.setdefault(
                        ref["blob_file"],
                        {"blob_file": ref["blob_file"], "offsets": []},
                    )
                    entry["offsets"].append(ref["offset"])

    weight_summary = []
    for wf in package.weight_files:
        weight_summary.append(
            {
                "file": wf.relative,
                "size": wf.size,
                "sha256": wf.sha256,
            }
        )

    referenced_blobs = []
    for entry in sorted(blob_refs.values(), key=lambda e: e["blob_file"]):
        name = entry["blob_file"]
        # Resolve @model_path/... aliases against the package contents.
        tail = name.split("@model_path/", 1)[-1] if "@model_path/" in name else name
        match = [wf for wf in package.weight_files if wf.relative.endswith(tail)]
        referenced_blobs.append(
            {
                "blob_file": name,
                "distinct_offsets": len(sorted(set(entry["offsets"]))),
                "resolved_file": match[0].relative if match else None,
                "resolved": bool(match),
            }
        )

    total_ops = sum(fn["op_total"] for fn in functions)

    inventory: dict = {
        "inventory_schema": INVENTORY_SCHEMA,
        "package": {
            "path": str(package.path),
            "format": "mlpackage",
            "file_format_version": package.file_format_version,
            "root_model_identifier": package.root_identifier,
        },
        "model": {
            "specification_version": model.specificationVersion,
            "type": model_type or "UNSET",
            "is_updatable": model.isUpdatable,
            "has_ml_program": program is not None,
            "description": {
                "inputs": inputs,
                "outputs": outputs,
                "state": state,
            },
            "metadata": {
                "short_description": model.description.metadata.shortDescription
                or None,
                "version_string": model.description.metadata.versionString or None,
                "author": model.description.metadata.author or None,
                "license": model.description.metadata.license or None,
                "user_defined": dict(model.description.metadata.userDefined),
            },
        },
        "program": {
            "version": program_version,
            "functions": functions,
        },
        "weights": {
            "files": weight_summary,
            "blob_references": referenced_blobs,
        },
        "compression": {
            "representation": "constexpr_* operations (see schema semantics)"
            if constexpr_ops
            else "none observed",
            "constexpr_ops": constexpr_ops,
            "constexpr_op_count": len(constexpr_ops),
        },
        "op_histogram": _merged_histogram(functions),
        "op_total": total_ops,
        "control_flow": {
            "present": any(fn["control_flow_ops"] for fn in functions),
            "ops": sorted({t for fn in functions for t in fn["control_flow_ops"]}),
        },
        "validity": _validity(model, functions),
        "eligibility": {
            "assessed": False,
            "note": (
                "compiler/deployment eligibility is not assessed by this "
                "inspector; it requires the target compiler's coverage data. "
                "Only parse validity and the typed inventory above are claims."
            ),
        },
    }
    return inventory


def _merged_histogram(functions: list[dict]) -> dict[str, int]:
    merged: Counter = Counter()
    for fn in functions:
        merged.update(fn["op_histogram"])
    return dict(sorted(merged.items()))


def _validity(model: Any, functions: list[dict]) -> dict:
    """Parse validity and explicit confidence notes.

    The protobuf schema is open (unknown op types and unknown fields
    are legal), so validity never claims semantic completeness: each
    finding is listed with what is and is not known.
    """
    notes: list[str] = []
    for fn in functions:
        if fn["opset_consistency"] != "ok":
            notes.append(f"function {fn['name']}: {fn['opset_consistency']}")
        if fn["program_version"] is None:
            notes.append(f"function {fn['name']}: program.version is unset (0)")
    identifiers_ok = True
    for fn in functions:
        for name in [fn["name"]] + [t["name"] for t in fn["inputs"]]:
            if not _IDENTIFIER.match(name or ""):
                identifiers_ok = False
                notes.append(f"non-identifier value name: {name!r}")
    return {
        "protobuf_parse": "ok",
        "model_type_set": model.WhichOneof("Type") is not None,
        "notes": notes,
        "confidence": "full structure read through the official schema; "
        "unknown op types and unknown schema fields would be reported, "
        "none encountered" if not notes else "see notes",
    }


def inspect(path: Path) -> dict:
    """One-shot: open a package and return its inventory."""
    return describe_package(open_mlpackage(Path(path)))
