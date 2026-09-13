# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Strict constant access for the pinned Parakeet decoder and joint packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mil_adapter import AdapterError, _Adapter
from .mlpackage import open_mlpackage
from .proto import count_unknown_fields, load_model, mil_dtype_name
from .reference import ReferenceLock, sha256_file

_COMPONENTS = {"decoder", "joint"}
_IMMEDIATE = {
    "BOOL": ("bools", np.bool_),
    "INT32": ("ints", np.int32),
    "FLOAT32": ("floats", np.float32),
    "STRING": ("strings", np.str_),
}
_BLOB = {"FLOAT16": (1, np.dtype("<f2"))}


class PinnedComponentError(ValueError):
    """The component differs from the pinned Parakeet package contract."""


@dataclass
class PinnedComponent:
    """One hash-verified MIL block with lazy named constant decoding."""

    package_path: Path
    block: Any
    _values: dict[str, Any] = field(repr=False)
    _blob_reader: Any = field(repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def constant(self, name: str) -> np.ndarray:
        """Return one named MIL ``const`` output as an immutable NumPy array."""
        if name in self._cache:
            return self._cache[name]
        try:
            value = self._values[name]
        except KeyError as exc:
            raise PinnedComponentError(f"pinned component has no constant {name!r}") from exc
        result = self._decode(value)
        result.setflags(write=False)
        self._cache[name] = result
        return result

    def _decode(self, value: Any) -> np.ndarray:
        tensor_type = value.type.tensorType
        shape = []
        for dimension in tensor_type.dimensions:
            if dimension.WhichOneof("dimension") != "constant":
                raise PinnedComponentError("pinned component constant has a dynamic shape")
            shape.append(int(dimension.constant.size))
        shape = tuple(shape)
        dtype = mil_dtype_name(tensor_type.dataType)
        storage = value.WhichOneof("value")
        if storage == "blobFileValue":
            try:
                record = self._blob_reader._blob_record(value)
            except AdapterError as exc:
                raise PinnedComponentError(str(exc)) from exc
            try:
                storage_code, numpy_dtype = _BLOB[dtype]
            except KeyError as exc:
                raise PinnedComponentError(
                    f"pinned component blob dtype {dtype} is unsupported"
                ) from exc
            expected_size = int(np.prod(shape, dtype=np.int64)) * numpy_dtype.itemsize
            if record.storage_code != storage_code or record.payload_size != expected_size:
                raise PinnedComponentError(
                    f"pinned component blob does not match {dtype}{shape}"
                )
            with record.path.open("rb") as stream:
                stream.seek(record.payload_offset)
                payload = stream.read(record.payload_size)
            if len(payload) != record.payload_size:
                raise PinnedComponentError("pinned component blob payload is truncated")
            return np.frombuffer(payload, dtype=numpy_dtype).reshape(shape)
        if storage != "immediateValue":
            raise PinnedComponentError(
                f"pinned component constant storage {storage!r} is unsupported"
            )
        tensor = value.immediateValue.tensor
        field_name = tensor.WhichOneof("value")
        if dtype == "FLOAT16":
            if field_name != "bytes":
                raise PinnedComponentError("pinned FLOAT16 immediate is not byte encoded")
            result = np.frombuffer(bytes(tensor.bytes.values), dtype=np.dtype("<f2"))
        else:
            try:
                expected_field, numpy_dtype = _IMMEDIATE[dtype]
            except KeyError as exc:
                raise PinnedComponentError(
                    f"pinned component immediate dtype {dtype} is unsupported"
                ) from exc
            if field_name != expected_field:
                raise PinnedComponentError(
                    f"pinned {dtype} immediate uses {field_name!r}, expected {expected_field!r}"
                )
            result = np.asarray(getattr(tensor, field_name).values, dtype=numpy_dtype)
        expected_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if result.size != expected_count:
            raise PinnedComponentError(
                f"pinned component constant has {result.size} values for shape {shape}"
            )
        return result.reshape(shape)


def _verify_package_files(package: Any, component: str, lock: ReferenceLock) -> None:
    prefix = f"{component}.mlpackage/"
    expected = {entry.path[len(prefix) :]: entry for entry in lock.files if entry.path.startswith(prefix)}
    actual = {
        "Manifest.json": package.path / "Manifest.json",
        package.model_path.relative_to(package.path).as_posix(): package.model_path,
        **{
            weight.path.relative_to(package.path).as_posix(): weight.path
            for weight in package.weight_files
        },
    }
    if set(actual) != set(expected):
        raise PinnedComponentError(
            f"{component}.mlpackage files differ from the pinned reference lock"
        )
    weight_hashes = {
        weight.path.relative_to(package.path).as_posix(): weight.sha256
        for weight in package.weight_files
    }
    for relative, path in actual.items():
        pin = expected[relative]
        digest = weight_hashes.get(relative) or sha256_file(path)
        if path.stat().st_size != pin.size or digest != pin.sha256:
            raise PinnedComponentError(
                f"{component}.mlpackage/{relative} differs from the pinned reference lock"
            )


def load_pinned_component(package_path: Path, component: str) -> PinnedComponent:
    """Load only the exact pinned decoder or joint package and its main MIL block."""
    if component not in _COMPONENTS:
        raise PinnedComponentError(f"unsupported pinned component {component!r}")
    package_path = Path(package_path).resolve()
    expected_name = f"{component}.mlpackage"
    if package_path.name != expected_name:
        raise PinnedComponentError(
            f"expected {expected_name}, got {package_path.name or str(package_path)!r}"
        )
    lock = ReferenceLock.load()
    package = open_mlpackage(package_path)
    _verify_package_files(package, component, lock)
    model = load_model(package.model_path.read_bytes())
    if model.specificationVersion != 9 or model.WhichOneof("Type") != "mlProgram":
        raise PinnedComponentError("pinned component is not the expected Core ML 9 MIL program")
    if count_unknown_fields(model):
        raise PinnedComponentError("pinned component contains unknown protobuf fields")
    if set(model.mlProgram.functions) != {"main"}:
        raise PinnedComponentError("pinned component must contain only the main function")
    function = model.mlProgram.functions["main"]
    if function.opset != "CoreML8" or set(function.block_specializations) != {"CoreML8"}:
        raise PinnedComponentError("pinned component must contain one CoreML8 block")
    block = function.block_specializations["CoreML8"]
    values = {}
    for operation in block.operations:
        if operation.type != "const":
            continue
        if len(operation.outputs) != 1 or "val" not in operation.attributes:
            raise PinnedComponentError("pinned component has an invalid const operation")
        output = operation.outputs[0]
        if output.name in values or output.type != operation.attributes["val"].type:
            raise PinnedComponentError("pinned component has an inconsistent const output")
        values[output.name] = operation.attributes["val"]
    return PinnedComponent(package_path, block, values, _Adapter(package, model))
