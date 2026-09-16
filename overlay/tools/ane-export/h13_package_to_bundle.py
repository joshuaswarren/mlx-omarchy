#!/usr/bin/env python3
"""Convert an explicit mil-hwxc H13 ANEC package into a strict bundle."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import struct

from bundle_payload_identity import payload_collection_sha256

SCHEMA = "mil-hwxc.h13-anec-package.v2"
TILE = 0x4000
DRIVER_ABI_MAJOR = 1
DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "int32": 4,
    "uint8": 1, "bool": 1,
}
ANEC_DTYPES = {"float16", "bfloat16", "bool"}
ROLES = ("input", "output", "state", "intermediate")
TILE_SIZE_BYTES = 0x4000
ANEC_HEADER_SIZE = 0x6A8
ANEC_PAYLOAD_OFFSET = 0x1000
ANEC_TILE_COUNT = 0x20
BIND_FIRST_SURFACE = 4
BIND_DMA_DISABLED = 0x00008880
BIND_DST_REGISTER = 0x17800
BIND_SELECTOR_MASK = 0x1F
BIND_MIN_TASK_BYTES = 40
BIND_SELECTORS = ((0x13800, 0), (0x13804, 6), (BIND_DST_REGISTER, 12))


class AdapterError(ValueError):
    pass


def fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    raise AdapterError(message)


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} root must be an object")
    return value


def unsigned(value: object, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{where} must be a non-negative integer")
    if positive and value == 0:
        fail(f"{where} must be positive")
    return value


def shape(value: object, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        fail(f"{where} must be a non-empty array")
    parsed = [unsigned(dim, where, positive=True) for dim in value]
    return parsed


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# H13 role-to-channel derivation (mirrors omarchy-ane fb4dfa86
# `bind_task_dma` / `bind_walk` / `derive_role_channels`).
def _bind_word(buf: bytes, index: int) -> int:
    return (buf[index * 4]
            | (buf[index * 4 + 1] << 8)
            | (buf[index * 4 + 2] << 16)
            | (buf[index * 4 + 3] << 24))


def _bind_task_dma(task: bytes, nbytes: int) -> list[int] | None:
    words = nbytes // 4
    dma = [BIND_DMA_DISABLED, BIND_DMA_DISABLED, BIND_DMA_DISABLED]
    if nbytes < BIND_MIN_TASK_BYTES or nbytes % 4:
        return None
    index = 10 + (1 if (_bind_word(task, 9) & 3) == 3 else 0)
    while index < words:
        header = _bind_word(task, index)
        count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        if index + count >= words:
            return None
        for offset in range(count):
            for slot in range(3):
                if base + offset * 4 == BIND_SELECTORS[slot][0]:
                    dma[slot] = _bind_word(task, index + 1 + offset)
        index += 1 + count
    return dma


def _bind_walk(
        payload: bytes,
        task_descriptor_size: int,
        task_descriptor_count: int,
        tiles: list[int]) -> tuple[list[int], list[int]] | None:
    """Returns (is_src, is_dst) per kAnecTileCount, or None on walk failure."""
    if task_descriptor_count == 0:
        return None
    is_src = [0] * ANEC_TILE_COUNT
    is_dst = [0] * ANEC_TILE_COUNT
    offset = 0
    nbytes = task_descriptor_size
    for index in range(task_descriptor_count):
        if (nbytes < BIND_MIN_TASK_BYTES or offset > len(payload)
                or nbytes > len(payload) - offset):
            return None
        dma = _bind_task_dma(payload[offset:], nbytes)
        if dma is None:
            return None
        selectors = _bind_word(payload[offset:], 8)
        for slot in range(3):
            channel = (selectors >> BIND_SELECTORS[slot][1]) & BIND_SELECTOR_MASK
            if dma[slot] == BIND_DMA_DISABLED:
                continue
            if channel < BIND_FIRST_SURFACE or channel >= ANEC_TILE_COUNT:
                continue
            if tiles[channel] == 0:
                continue
            if BIND_SELECTORS[slot][0] == BIND_DST_REGISTER:
                is_dst[channel] = 1
            else:
                is_src[channel] = 1
        if index + 1 == task_descriptor_count:
            break
        next_offset = _bind_word(payload[offset:], 7)
        nbytes = (((_bind_word(payload[offset:], 1) >> 16) & 0x1FF) + 1) * 4
        if next_offset % 4 or next_offset > len(payload):
            return None
        offset = next_offset
    return is_src, is_dst


def derive_role_channels(anec_path: Path) -> tuple[list[int], list[int]] | None:
    """Derive (src_channels, dst_channels) ascending from the ANEC bytes.

    Returns None when the walk fails or the derived map does not account for
    every surface the ANEC header declares (mirrors `derive_role_channels`
    returning false in `validate_program_contract`). The caller is then
    expected to mirror libane's positional fallback, which the strict
    fb4dfa86 bundle gate refuses; the adapter does not fabricate a map here.
    """
    data = anec_path.read_bytes()
    if len(data) < ANEC_PAYLOAD_OFFSET + ANEC_HEADER_SIZE:
        # Older or smaller ANEC files may still be valid if the header fits.
        pass
    if len(data) < ANEC_HEADER_SIZE:
        raise AdapterError(
            f"ANEC file is smaller than the H13 header: {anec_path}")
    payload_size = int.from_bytes(data[0:8], "little")
    task_descriptor_size = int.from_bytes(data[8:12], "little")
    task_descriptor_count = int.from_bytes(data[12:16], "little")
    source_count = int.from_bytes(data[32:36], "little")
    destination_count = int.from_bytes(data[36:40], "little")
    tiles = list(struct.unpack(f"<{ANEC_TILE_COUNT}I", data[40:40 + 4 * ANEC_TILE_COUNT]))
    if source_count > ANEC_TILE_COUNT or destination_count > ANEC_TILE_COUNT:
        return None
    if len(data) < ANEC_PAYLOAD_OFFSET + payload_size:
        raise AdapterError(f"ANEC payload truncated: {anec_path}")
    payload = data[ANEC_PAYLOAD_OFFSET:ANEC_PAYLOAD_OFFSET + payload_size]
    walked = _bind_walk(payload, task_descriptor_size, task_descriptor_count, tiles)
    if walked is None:
        return None
    is_src, is_dst = walked
    derived_src: list[int] = []
    derived_dst: list[int] = []
    for channel in range(BIND_FIRST_SURFACE, ANEC_TILE_COUNT):
        if is_dst[channel]:
            if len(derived_dst) == ANEC_TILE_COUNT:
                return None
            derived_dst.append(channel)
        elif is_src[channel]:
            if len(derived_src) == ANEC_TILE_COUNT:
                return None
            derived_src.append(channel)
    if len(derived_src) != source_count or len(derived_dst) != destination_count:
        return None
    return derived_src, derived_dst



def require_fields(value: dict, required: set[str], allowed: set[str], where: str) -> None:
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        fail(f"{where} missing field '{sorted(missing)[0]}'")
    if unknown:
        fail(f"{where} unknown field '{sorted(unknown)[0]}'")


def convert_binding(binding: dict, where: str) -> tuple[dict, str]:
    allowed = {
        "allocationBytes", "dtype", "index", "logicalBytes",
        "name", "nchw", "role", "shape", "slice",
    }
    require_fields(
        binding,
        {"allocationBytes", "dtype", "index", "logicalBytes", "name", "nchw", "shape"},
        allowed,
        where,
    )
    # The per-binding dtype directly names the surface element type:
    # "bool" = 1-byte bool surfaces (compiler yield 2026-09-13; no
    # separate elementDtype field exists). logical_bytes stay bytes.
    dtype = binding["dtype"]
    if dtype not in DTYPE_BYTES:
        fail(f"{where}.dtype '{dtype}' is unsupported")
    if dtype not in ANEC_DTYPES:
        fail(f"{where}.dtype '{dtype}' is an unsupported ANEC hardware dtype")
    binding_shape = shape(binding["shape"], f"{where}.shape")
    logical_bytes = unsigned(binding["logicalBytes"], f"{where}.logicalBytes", positive=True)
    count = product(binding_shape)
    if logical_bytes != count * DTYPE_BYTES[dtype]:
        fail(f"{where}.logicalBytes does not match dtype geometry")
    allocation = unsigned(
        binding["allocationBytes"], f"{where}.allocationBytes", positive=True
    )
    if allocation < logical_bytes or allocation % TILE:
        fail(f"{where}.allocationBytes must cover logical bytes and be 0x4000-aligned")
    channel = unsigned(binding["index"], f"{where}.index")
    nchw = shape(binding["nchw"], f"{where}.nchw")
    if len(nchw) != 6:
        fail(f"{where}.nchw must contain exactly 6 integers")
    tensor = binding["name"]
    if not isinstance(tensor, str) or not tensor:
        fail(f"{where}.name must be a non-empty string")
    offset = 0
    physical = count
    if "slice" in binding:
        slice_value = binding["slice"]
        if not isinstance(slice_value, dict):
            fail(f"{where}.slice must be an object")
        require_fields(
            slice_value,
            {"tensor", "elementOffset", "elementCount", "physicalElements"},
            {"tensor", "elementOffset", "elementCount", "physicalElements"},
            f"{where}.slice",
        )
        tensor = slice_value["tensor"]
        if not isinstance(tensor, str) or not tensor:
            fail(f"{where}.slice.tensor must be a non-empty string")
        offset = unsigned(slice_value["elementOffset"], f"{where}.slice.elementOffset")
        sliced = unsigned(
            slice_value["elementCount"], f"{where}.slice.elementCount", positive=True
        )
        if sliced != count:
            fail(f"{where}.slice.elementCount does not match binding shape")
        physical = unsigned(
            slice_value["physicalElements"],
            f"{where}.slice.physicalElements",
            positive=True,
        )
    physical_from_nchw = product(nchw[:4])
    if physical != physical_from_nchw:
        fail(f"{where}.physicalElements does not match NCHW geometry")
    if count > physical:
        fail(f"{where}.slice.elementCount exceeds physicalElements")
    if nchw[4] % nchw[5] or nchw[5] % DTYPE_BYTES[dtype]:
        fail(f"{where}.nchw has invalid packed tile geometry")
    if nchw[4] // nchw[5] < nchw[2] or nchw[5] // DTYPE_BYTES[dtype] < nchw[3]:
        fail(f"{where}.nchw packed tile is smaller than logical shape")
    if nchw[0] * nchw[1] * nchw[4] > allocation:
        fail(f"{where}.allocationBytes is smaller than NCHW physical bytes")
    return ({
        "tensor": tensor,
        "channel": channel,
        "dtype": dtype,
        "shape": binding_shape,
        "nchw": nchw,
        "logical_bytes": logical_bytes,
        "allocation_bytes": allocation,
        "element_offset": offset,
        "element_count": count,
        "physical_elements": physical,
    }, tensor)


def adapt(package: Path, output: Path, identity: dict) -> dict:
    source = load_object(package / "manifest.json")
    if source.get("schema") != SCHEMA:
        fail(f"compiler manifest schema must be exactly '{SCHEMA}'")
    require_fields(
        source,
        {"artifactFormat", "dispatchPlan", "intermediates", "logicalResults",
         "physicalOutputs", "programs", "schema", "target", "tensors"},
        {"artifactFormat", "bytes", "constantBytes", "constantInputs", "constantOffset",
         "dispatchPlan", "encoder", "file", "inputs", "intermediates",
         "logicalResults", "operation", "outputs", "physicalOutputs", "programs",
         "schema", "scratchBytes", "target", "taskDescriptors", "tensors"},
        "compiler manifest",
    )
    if source["artifactFormat"] != "anec":
        fail("compiler manifest artifactFormat must be exactly 'anec'")
    if source["target"] != "H13":
        fail("compiler manifest target must be exactly 'H13'")
    programs = source["programs"]
    if not isinstance(programs, list) or not programs:
        fail("compiler manifest programs must be a non-empty array")
    dispatch = source["dispatchPlan"]
    if not isinstance(dispatch, list) or len(dispatch) != len(programs) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in dispatch) or             sorted(dispatch) != list(range(len(programs))):
        fail("compiler manifest dispatchPlan must be a permutation of program indices")
    intermediates = source["intermediates"]
    if not isinstance(intermediates, list) or any(
            not isinstance(name, str) or not name for name in intermediates) or             len(set(intermediates)) != len(intermediates):
        fail("compiler manifest intermediates must contain unique non-empty tensor names")
    tensors = source["tensors"]
    if not isinstance(tensors, dict) or not tensors:
        fail("compiler manifest tensors must be a non-empty object")
    single_program_fields = {
        "bytes", "constantBytes", "constantInputs", "constantOffset", "encoder", "file",
        "inputs", "operation", "outputs", "scratchBytes", "taskDescriptors",
    }
    present_single_program_fields = single_program_fields & source.keys()
    if present_single_program_fields and (
            present_single_program_fields != single_program_fields or len(programs) != 1 or
            any(source[field] != programs[0].get(field) for field in single_program_fields)):
        fail("compiler manifest single-program fields must exactly match programs[0]")

    converted_programs = []
    payloads = []
    payload_names: set[str] = set()
    inferred: dict[str, dict] = {}
    descriptors = 0
    for program_index, program in enumerate(programs):
        where = f"compiler manifest programs[{program_index}]"
        if not isinstance(program, dict):
            fail(f"{where} must be an object")
        allowed = {
            "bytes", "constantBytes", "constantInputs", "constantOffset", "encoder",
            "file", "inputs", "operation", "outputs", "scratchBytes", "taskDescriptors",
        }
        require_fields(program, allowed, allowed, where)
        if program["constantInputs"]:
            fail(f"{where}.constantInputs is not supported by bundle schema 4")
        filename = program["file"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            fail(f"{where}.file must be a plain filename")
        if filename in payload_names:
            fail(f"{where}.file duplicates another program payload")
        payload_names.add(filename)
        payload = package / filename
        if not payload.is_file():
            fail(f"{where}.file does not exist: {filename}")
        byte_size = unsigned(program["bytes"], f"{where}.bytes", positive=True)
        if payload.stat().st_size != byte_size:
            fail(f"{where}.bytes does not match {filename}")
        inputs = []
        outputs = []
        for direction, destination in (("inputs", inputs), ("outputs", outputs)):
            values = program[direction]
            if not isinstance(values, list) or (direction == "outputs" and not values):
                fail(f"{where}.{direction} must be {'a non-empty' if direction == 'outputs' else 'an'} array")
            for binding_index, binding in enumerate(values):
                if not isinstance(binding, dict):
                    fail(f"{where}.{direction}[{binding_index}] must be an object")
                converted, tensor_name = convert_binding(
                    binding, f"{where}.{direction}[{binding_index}]"
                )
                previous = inferred.get(tensor_name)
                facts = {"dtype": converted["dtype"], "stride": converted["allocation_bytes"]}
                if previous and previous["dtype"] != facts["dtype"]:
                    fail(f"tensor '{tensor_name}' has conflicting dtypes")
                inferred[tensor_name] = {
                    "dtype": facts["dtype"],
                    "stride": max(facts["stride"], previous["stride"] if previous else 0),
                }
                destination.append(converted)
        task_descriptors = unsigned(
            program["taskDescriptors"], f"{where}.taskDescriptors", positive=True
        )
        descriptors += task_descriptors
        scratch = unsigned(program["scratchBytes"], f"{where}.scratchBytes")
        for field in ("operation", "encoder"):
            if not isinstance(program[field], str) or not program[field]:
                fail(f"{where}.{field} must be a non-empty string")
        converted_programs.append({
            "payload": filename,
            "operation": program["operation"],
            "encoder": program["encoder"],
            "task_descriptors": task_descriptors,
            "scratch_bytes": scratch,
            "inputs": inputs,
            "outputs": outputs,
        })
        payloads.append({
            "role": "anec",
            "path": filename,
            "sha256": sha256(payload),
            "byte_size": byte_size,
        })

    physical_values = source["physicalOutputs"]
    if not isinstance(physical_values, list) or not physical_values:
        fail("compiler manifest physicalOutputs must be a non-empty array")
    physical_outputs = []
    physical_by_name: dict[str, dict] = {}
    for index, value in enumerate(physical_values):
        where = f"compiler manifest physicalOutputs[{index}]"
        if not isinstance(value, dict):
            fail(f"{where} must be an object")
        require_fields(value, {"tensor", "dtype", "shape", "logicalBytes"},
                       {"tensor", "dtype", "shape", "logicalBytes"}, where)
        name = value["tensor"]
        if not isinstance(name, str) or not name:
            fail(f"{where}.tensor must be a non-empty string")
        if name in physical_by_name:
            fail(f"{where}.tensor duplicates physical output '{name}'")
        dtype = value["dtype"]
        if dtype not in DTYPE_BYTES:
            fail(f"{where}.dtype '{dtype}' is unsupported")
        output_shape = shape(value["shape"], f"{where}.shape")
        byte_size = unsigned(value["logicalBytes"], f"{where}.logicalBytes", positive=True)
        elements = product(output_shape)
        if byte_size != elements * DTYPE_BYTES[dtype]:
            fail(f"{where}.logicalBytes does not match dtype geometry")
        tensor = tensors.get(name)
        if not isinstance(tensor, dict) or tensor.get("role") != "output" or                 tensor.get("shape") != output_shape or tensor.get("logicalBytes") != byte_size:
            fail(f"{where} does not match output tensor '{name}'")
        facts = inferred.get(name)
        if not facts or facts["dtype"] != dtype:
            fail(f"{where}.dtype does not match program output '{name}'")
        converted = {
            "name": name,
            "index": index,
            "dtype": dtype,
            "shape": output_shape,
            "byte_size": byte_size,
            "stride": max(facts["stride"], ((byte_size + TILE - 1) // TILE) * TILE),
        }
        physical_outputs.append(converted)
        physical_by_name[name] = {"dtype": dtype, "elements": elements}

    result_values = source["logicalResults"]
    if not isinstance(result_values, list) or not result_values:
        fail("compiler manifest logicalResults must be a non-empty array")
    logical_results = []
    referenced_physical_outputs: set[str] = set()
    for index, value in enumerate(result_values):
        where = f"compiler manifest logicalResults[{index}]"
        if not isinstance(value, dict):
            fail(f"{where} must be an object")
        require_fields(value, {"name", "dtype", "shape", "physical", "conversion"},
                       {"name", "dtype", "shape", "physical", "conversion"}, where)
        name = value["name"]
        if not isinstance(name, str) or not name:
            fail(f"{where}.name must be a non-empty string")
        dtype = value["dtype"]
        if dtype not in DTYPE_BYTES:
            fail(f"{where}.dtype '{dtype}' is unsupported")
        result_shape = shape(value["shape"], f"{where}.shape")
        if value["conversion"] != "identity":
            fail(f"{where}.conversion must be exactly 'identity'")
        physical = value["physical"]
        if not isinstance(physical, dict):
            fail(f"{where}.physical must be an object")
        require_fields(physical, {"tensor", "elementOffset", "elementCount"},
                       {"tensor", "elementOffset", "elementCount"}, f"{where}.physical")
        tensor_name = physical["tensor"]
        if not isinstance(tensor_name, str) or tensor_name not in physical_by_name:
            fail(f"{where}.physical.tensor does not name a physical output")
        offset = unsigned(physical["elementOffset"], f"{where}.physical.elementOffset")
        count = unsigned(physical["elementCount"], f"{where}.physical.elementCount",
                         positive=True)
        if count != product(result_shape):
            fail(f"{where}.physical.elementCount does not match logical shape")
        storage = physical_by_name[tensor_name]
        if dtype != storage["dtype"]:
            fail(f"{where}.dtype does not match physical output for identity conversion")
        if offset + count > storage["elements"]:
            fail(f"{where}.physical range exceeds physical output storage")
        logical_results.append({
            "name": name,
            "dtype": dtype,
            "shape": result_shape,
            "tensor": tensor_name,
            "element_offset": offset,
            "element_count": count,
            "conversion": "identity",
        })
        referenced_physical_outputs.add(tensor_name)
    if referenced_physical_outputs != set(physical_by_name):
        fail("compiler manifest logicalResults must reference every physical output")

    tensor_lists = {role: [] for role in ROLES}
    tensor_lists["output"] = physical_outputs
    expected_intermediates = set(intermediates)
    actual_intermediates = set()
    for name, tensor in tensors.items():
        where = f"compiler manifest tensors.{name}"
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            fail(f"{where} must be a named object")
        # The compiler spells `dtype` on a graph tensor only where the
        # surface is not the default fp16 (bool cond). It is redundant with
        # the program binding, so it is accepted and then checked against it.
        require_fields(tensor, {"logicalBytes", "role", "shape"},
                       {"accumulation", "aliasOf", "dtype", "logicalBytes",
                        "role", "shape"}, where)
        role = tensor["role"]
        if role not in ROLES:
            fail(f"{where}.role '{role}' is unsupported")
        tensor_shape = shape(tensor["shape"], f"{where}.shape")
        byte_size = unsigned(tensor["logicalBytes"], f"{where}.logicalBytes", positive=True)
        if role == "output" and name not in physical_by_name:
            if not isinstance(tensor.get("aliasOf"), str) or not tensor["aliasOf"]:
                fail(f"{where} is neither a physical output nor an explicit alias")
            continue
        facts = inferred.get(name)
        if not facts:
            fail(f"{where} has no program binding")
        if "dtype" in tensor and tensor["dtype"] != facts["dtype"]:
            fail(f"{where}.dtype does not match the program binding dtype")
        if byte_size != product(tensor_shape) * DTYPE_BYTES[facts["dtype"]]:
            fail(f"{where}.logicalBytes does not match inferred dtype geometry")
        if role == "intermediate":
            actual_intermediates.add(name)
        if role != "output":
            tensor_lists[role].append({
                "name": name,
                "index": len(tensor_lists[role]),
                "dtype": facts["dtype"],
                "shape": tensor_shape,
                "byte_size": byte_size,
                "stride": max(facts["stride"], ((byte_size + TILE - 1) // TILE) * TILE),
            })
    extras = sorted(set(inferred) - set(tensors))
    if extras:
        fail(f"program binding references unknown tensor '{extras[0]}'")
    declared_physical = {
        name for name, tensor in tensors.items()
        if isinstance(tensor, dict) and tensor.get("role") == "output" and "aliasOf" not in tensor
    }
    if declared_physical != set(physical_by_name):
        fail("compiler manifest physicalOutputs does not match physical output tensors")
    if actual_intermediates != expected_intermediates:
        fail("compiler manifest intermediates does not match tensor roles")
    if not tensor_lists["input"]:
        fail("compiler manifest requires at least one input tensor")
    tensor_roles = {name: value["role"] for name, value in tensors.items()}
    tensor_elements = {name: product(shape(value["shape"], f"compiler manifest tensors.{name}.shape"))
                       for name, value in tensors.items()}
    reads: set[str] = set()
    writes: set[str] = set()
    for program_index, program in enumerate(converted_programs):
        anec_path = package / program["payload"]
        derived = derive_role_channels(anec_path)
        if derived is None:
            fail(
                f"compiler manifest programs[{program_index}] {anec_path.name} "
                f"task stream does not name every surface; channel map is "
                f"positional. The strict bundle gate refuses this program. "
                f"Re-export from mil-hwxc so the ANEC descriptor records name "
                f"every surface, or accept the positional layout under a "
                f"pinned-bundle allowance by hand-editing the manifest with "
                f"channels {list(range(4, 4 + len(program['outputs'])))} "
                f"and {list(range(4 + len(program['outputs']), 4 + len(program['outputs']) + len(program['inputs'])))}.")
        derived_src, derived_dst = derived
        declared_outputs = [binding["channel"] for binding in program["outputs"]]
        declared_inputs = [binding["channel"] for binding in program["inputs"]]
        if declared_outputs != derived_dst or declared_inputs != derived_src:
            fail(
                f"compiler manifest programs[{program_index}] channel mapping "
                f"disagrees with the ANEC task stream: manifest outputs={declared_outputs} "
                f"inputs={declared_inputs}; derived outputs={derived_dst} "
                f"inputs={derived_src}. The strict bundle gate refuses mismatched "
                f"manifests. Re-export from mil-hwxc so the package index fields "
                f"match the derived channels.")
        for ordinal, binding in enumerate(program["outputs"]):
            binding["channel"] = derived_dst[ordinal]
        for ordinal, binding in enumerate(program["inputs"]):
            binding["channel"] = derived_src[ordinal]
        for direction, bindings in (("inputs", program["inputs"]),
                                    ("outputs", program["outputs"])):
            for binding in bindings:
                name = binding["tensor"]
                role = tensor_roles[name]
                if (direction == "inputs" and role == "output") or                         (direction == "outputs" and role == "input"):
                    fail(f"tensor '{name}' is bound in the wrong direction")
                if binding["element_offset"] + binding["element_count"] > tensor_elements[name]:
                    fail(f"tensor '{name}' binding range exceeds its logical shape")
                (reads if direction == "inputs" else writes).add(name)
    for name, role in tensor_roles.items():
        if role == "output" and name not in physical_by_name:
            continue
        if (role == "input" and name not in reads) or                 (role == "output" and name not in writes) or                 (role in ("state", "intermediate") and
                 (name not in reads or name not in writes)):
            fail(f"tensor '{name}' is not fully bound for role '{role}'")

    available = {
        name: [(0, tensor_elements[name])]
        for name, role in tensor_roles.items() if role in ("input", "state")
    }
    for program_index in dispatch:
        program = converted_programs[program_index]
        for binding in program["inputs"]:
            begin = binding["element_offset"]
            end = begin + binding["element_count"]
            if not any(start <= begin and stop >= end
                       for start, stop in available.get(binding["tensor"], [])):
                fail(f"dispatchPlan reads tensor '{binding['tensor']}' before its range is written")
        for binding in program["outputs"]:
            ranges = available.setdefault(binding["tensor"], [])
            ranges.append((binding["element_offset"],
                           binding["element_offset"] + binding["element_count"]))
            ranges.sort()
            merged: list[tuple[int, int]] = []
            for start, stop in ranges:
                if not merged or start > merged[-1][1]:
                    merged.append((start, stop))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            available[binding["tensor"]] = merged
    for name, role in tensor_roles.items():
        if role != "output" or name not in physical_by_name:
            continue
        complete = any(start == 0 and stop >= tensor_elements[name]
                       for start, stop in available.get(name, []))
        if not complete:
            fail(f"output tensor '{name}' is not fully written")

    required_identity = {
        "name", "graph_hash", "compiler_host_build", "compiler_toolchain",
        "source_repo", "source_commit", "exported_at", "model",
    }
    require_fields(identity, required_identity, required_identity, "identity")
    for field in required_identity:
        if not isinstance(identity[field], str) or not identity[field]:
            fail(f"identity.{field} must be a non-empty string")
    for field, length in (("graph_hash", 64), ("source_commit", 40)):
        value = identity[field]
        if len(value) != length or any(character not in "0123456789abcdef" for character in value):
            fail(f"identity.{field} must be {length} lowercase hex characters")
    manifest = {
        "manifest_version": 4,
        "name": identity["name"],
        "graph_hash": identity["graph_hash"],
        "task_descriptors": descriptors,
        "inputs": tensor_lists["input"],
        "outputs": tensor_lists["output"],
        "logical_results": logical_results,
        "state": tensor_lists["state"],
        "intermediates": tensor_lists["intermediate"],
        "programs": converted_programs,
        "dispatch_plan": dispatch,
        "payloads": payloads,
        "compiler": {
            "host_build": identity["compiler_host_build"],
            "toolchain": identity["compiler_toolchain"],
            "target": "h13",
        },
        "driver_abi_major": DRIVER_ABI_MAJOR,
        "provenance": {
            "source_repo": identity["source_repo"],
            "source_commit": identity["source_commit"],
            "exported_at": identity["exported_at"],
        },
        "release_asset": {
            "model": identity["model"],
            "model_sha256": payload_collection_sha256(payloads),
        },
    }

    if output.exists() and any(output.iterdir()):
        fail(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        shutil.copyfile(package / payload["path"], output / payload["path"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def git_output(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if result.returncode:
        fail(f"compiler source git check failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graph-source", type=Path, required=True)
    parser.add_argument(
        "--compiler-source",
        type=Path,
        required=True,
        help="repository witness used only to prove the generation commit exists",
    )
    parser.add_argument("--compiler-receipt", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exported-at", default=datetime.date.today().isoformat())
    args = parser.parse_args()
    try:
        receipt = load_object(args.compiler_receipt or args.package / "source.json")
        required = {
            "artifact_format", "compiler_binary_sha256",
            "compiler_executable_source_commit", "compiler_host_build",
            "compiler_manifest_sha256", "graph_source_sha256", "payloads",
            "schema", "target",
        }
        require_fields(receipt, required, set(receipt), "compiler receipt")
        if receipt["artifact_format"] != "anec" or receipt["target"] != "H13":
            fail("compiler receipt must identify an explicit H13 ANEC package")
        if receipt["schema"] != SCHEMA:
            fail(f"compiler receipt schema must be exactly '{SCHEMA}'")
        compiler_commit = receipt["compiler_executable_source_commit"]
        for field, length in (("compiler_executable_source_commit", 40),
                              ("compiler_binary_sha256", 64),
                              ("compiler_manifest_sha256", 64),
                              ("graph_source_sha256", 64)):
            value = receipt[field]
            if not isinstance(value, str) or len(value) != length or any(
                    character not in "0123456789abcdef" for character in value):
                fail(f"compiler receipt {field} must be {length} lowercase hex characters")
        if sha256(args.package / "manifest.json") != receipt["compiler_manifest_sha256"]:
            fail("compiler receipt compiler_manifest_sha256 does not match compiler manifest")
        if sha256(args.graph_source) != receipt["graph_source_sha256"]:
            fail("compiler receipt graph_source_sha256 does not match --graph-source")
        git_output(args.compiler_source, "cat-file", "-e", f"{compiler_commit}^{{commit}}")
        received_payloads = receipt["payloads"]
        source = load_object(args.package / "manifest.json")
        package_files = [program.get("file") for program in source.get("programs", [])
                         if isinstance(program, dict)]
        if not isinstance(received_payloads, dict) or set(received_payloads) != set(package_files):
            fail("compiler receipt payload set does not match compiler manifest programs")
        for filename, digest in received_payloads.items():
            if not isinstance(digest, str) or sha256(args.package / filename) != digest:
                fail(f"compiler receipt payload digest does not match {filename}")
        host_build = receipt["compiler_host_build"]
        if not isinstance(host_build, str) or not host_build:
            fail("compiler receipt compiler_host_build must be a non-empty string")
        graph_hash = receipt["graph_source_sha256"]
        identity = {
            "name": args.name,
            "graph_hash": graph_hash,
            "compiler_host_build": host_build,
            "compiler_toolchain": (
                f"mil-hwxc {compiler_commit} "
                f"sha256:{receipt['compiler_binary_sha256']}"
            ),
            "source_repo": args.source_repo,
            "source_commit": args.source_commit,
            "exported_at": args.exported_at,
            "model": args.model,
        }
        manifest = adapt(args.package, args.out_dir, identity)
    except (AdapterError, OSError, subprocess.TimeoutExpired) as error:
        print(f"h13_package_to_bundle: error: {error}", file=sys.stderr)
        return 1
    print(
        f"h13_package_to_bundle: PASS programs={len(manifest['programs'])} "
        f"payloads={len(manifest['payloads'])} output={args.out_dir}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
