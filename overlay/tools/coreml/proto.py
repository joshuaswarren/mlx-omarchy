# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Minimal protobuf wire-format reader for Core ML model inspection.

The Core ML .mlpackage model specification is a protobuf serialization of
``specification.proto`` (Model message). The MIL program is the protobuf
serialization of ``MIL.proto`` (Program message). Both are standard
varint-tagged wire format.

This module is intentionally tiny: it exposes the raw field structure
needed by :mod:`mlx_omarchy.coreml.mlpackage` to walk to model
description, function, block, and operation boundaries. It is not a
general-purpose protobuf implementation; semantic interpretation lives
in :mod:`mlpackage`.

Why a hand-rolled decoder instead of ``protobuf`` / ``coremltools``:

* No runtime pip dependency. The host environment for the inspector is
  whatever Linux container the operator runs it in; we cannot assume
  ``coremltools`` is installed.
* Determinism. The decoder logic is pure Python with no codegen, so
  the inspector's behavior is reproducible from source.
* Licence footprint. Apple/coremltools protobuf schemas are BSD-3;
  vendoring them is a separate decision. This module reads the wire
  format, not the schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

# Wire types per https://protobuf.dev/programming-proto3/#wire
WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LENGTH_DELIMITED = 2
WIRE_START_GROUP = 3
WIRE_END_GROUP = 4
WIRE_FIXED32 = 5


class ProtoDecodeError(ValueError):
    """Raised when the wire bytes cannot be parsed."""


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a base-128 varint starting at ``pos``.

    Returns ``(value, new_pos)``. Raises :class:`ProtoDecodeError` on
    truncated input or a value wider than 64 bits.
    """
    if pos < 0 or pos > len(data):
        raise ProtoDecodeError(f"varint position {pos} out of range (len={len(data)})")
    shift = 0
    value = 0
    while True:
        if pos >= len(data):
            raise ProtoDecodeError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift >= 64:
            raise ProtoDecodeError("varint exceeds 64 bits")


@dataclass(frozen=True)
class Field:
    """One decoded protobuf field."""

    number: int
    wire: int
    value: int | bytes


def iter_fields(data: bytes) -> Iterator[Field]:
    """Yield :class:`Field` for each top-level field in ``data``.

    Unknown wire types terminate iteration with a :class:`ProtoDecodeError`
    only if a malformed tag is encountered; length-delimited payloads are
    returned as bytes for the caller to interpret.

    Truncated final fields at the end of the buffer are tolerated silently
    rather than raising: Core ML .mlmodel proto blobs may be padded or
    contain trailing bytes that the rest of the model spec does not need.
    """
    pos = 0
    while pos < len(data):
        tag_start = pos
        try:
            tag, pos = read_varint(data, pos)
        except ProtoDecodeError:
            return
        number = tag >> 3
        wire = tag & 0x7
        if wire == WIRE_VARINT:
            try:
                v, pos = read_varint(data, pos)
            except ProtoDecodeError:
                return
            yield Field(number, wire, v)
        elif wire == WIRE_FIXED64:
            if pos + 8 > len(data):
                return
            v = int.from_bytes(data[pos : pos + 8], "little", signed=False)
            yield Field(number, wire, v)
            pos += 8
        elif wire == WIRE_LENGTH_DELIMITED:
            try:
                length, pos = read_varint(data, pos)
            except ProtoDecodeError:
                return
            if pos + length > len(data):
                return
            payload = data[pos : pos + length]
            yield Field(number, wire, payload)
            pos += length
        elif wire == WIRE_FIXED32:
            if pos + 4 > len(data):
                return
            v = int.from_bytes(data[pos : pos + 4], "little", signed=False)
            yield Field(number, wire, v)
        elif wire in (WIRE_START_GROUP, WIRE_END_GROUP):
            # Deprecated; tolerantly skip the unknown group payload.
            # We do not understand the group body, so we stop emission
            # in this nested scope and let the caller continue at the
            # next parent message — the inspector treats group-encoded
            return
        else:
            # Unknown / experimental wire types: skip the rest of this
            # nested message. The inspector is robust to malformed
            # segments so that one bad sub-message does not poison
            # the entire report.
            return

def iter_nested(data: bytes, *fields: int) -> Iterator[bytes]:
    """Yield length-delimited payloads from the named wire-2 fields.

    Walks ``data`` once and yields the bytes for any field whose number
    matches one of ``fields``. Useful for descending into a known
    protobuf schema without writing nested loops every time.
    """
    wanted = set(fields)
    for f in iter_fields(data):
        if f.wire == WIRE_LENGTH_DELIMITED and f.number in wanted:
            assert isinstance(f.value, bytes)
            yield f.value


def collect_strings(data: bytes) -> list[str]:
    """Recursively collect UTF-8 strings from length-delimited payloads.

    Used by the operation-type histogram pass. A payload is treated as a
    string when it decodes as UTF-8 and contains only printable ASCII
    characters (0x20..0x7e). This excludes non-string bytes (weight
    blobs, dtype bytes) without missing genuine op names that can
    include underscores.
    """
    out: list[str] = []
    _collect_strings(data, out)
    return out


def _collect_strings(data: bytes, out: list[str]) -> None:
    for f in iter_fields(data):
        if f.wire != WIRE_LENGTH_DELIMITED:
            continue
        assert isinstance(f.value, bytes)
        blob = f.value
        if 1 <= len(blob) <= 200 and all(0x20 <= b <= 0x7E for b in blob):
            try:
                out.append(blob.decode("utf-8"))
            except UnicodeDecodeError:
                pass
        if 0 < len(blob) <= 4 * 1024 * 1024:
            _collect_strings(blob, out)
