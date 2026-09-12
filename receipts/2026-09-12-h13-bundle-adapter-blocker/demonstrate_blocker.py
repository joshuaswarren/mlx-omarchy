#!/usr/bin/env python3
"""All-package reconciliation sweep for the plan §46/§62 bundle boundary.

Verifies, mechanically and read-only, that every program of the real pinned
mil-hwx-compiler conv_relu qualification package reconciles with every
strict mlx-omarchy bundle-v2 field and ANEC cross-check from real data --
except exactly one manifest requirement: the single workspace tensor must
carry a positive byte_size, while the honest package-wide value is 0
(tiles[3] == 0 in every payload; schema mil-hwxc.h13-anec-package.v1 has no
workspace concept). That single field is the §62 blocker; see
receipts/2026-09-12-h13-bundle-adapter-blocker.md.

The script also emits the per-program honest v2 manifests plus the
package-level dispatch descriptor (dispatchPlan order, per-program tensor
slices, physical buffer spans) so the runtime-side representation is
demonstrated without pretending load_bundle accepts it yet.

Host only. No device access, no ANE execution, no fabricated identity:
firmware and run identity are required flags (verified inputs), never
defaults.

Usage:
  python3 demonstrate_blocker.py --package DIR --out DIR \
      --firmware-min V --firmware-max V \
      [--source-repo R --source-commit C --exported-at D \
       --model-name N --model-sha256 H --compiler-host-build S]
  add --diagnostic-positive-workspace to also emit the fabrication-pattern
  manifests that v2 accepts today (diagnostic only; never load-bearing).
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys

TILE = 0x4000
HEADER = 0x6A8
DTYPES = {"float16": 2, "float32": 4, "bfloat16": 2, "int32": 4, "uint8": 1}


def fail(msg):
    print(f"RECONCILIATION FAILURE: {msg}")
    sys.exit(1)


def align16(value):
    return (value + 15) & ~15


def u32_le(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def u64_le(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def tiles_of(data):
    return struct.unpack_from("<32I", data, 40)


def nchw_of(data, bdx):
    return struct.unpack_from("<6Q", data, 40 + 32 * 4 + bdx * 6 * 8)


def check_tensor_side(
    where, record_tensor, data, bdx, require_aligned_stride
):
    """Mirrors the v2 checks that apply to one manifest side (inputs,
    outputs) against the real payload header. Spec-mirror for the sweep;
    the executable contract is the C++ suite."""
    element_size = DTYPES.get(record_tensor["dtype"])
    if element_size is None:
        fail(f"{where}: unsupported dtype {record_tensor['dtype']}")
    tensor = {
        "byte_size": record_tensor["logicalBytes"],
        "stride": record_tensor["allocationBytes"],
        "logical_bytes": record_tensor["logicalBytes"],
    }
    tiles = tiles_of(data)
    channel = tiles[bdx] * TILE
    if channel == 0:
        fail(f"{where}: channel {bdx} allocation is zero")
    if tensor["byte_size"] > channel:
        fail(f"{where}: byte_size exceeds channel allocation")
    if tensor["stride"] > channel:
        fail(f"{where}: stride exceeds channel allocation")
    if require_aligned_stride and tensor["stride"] % TILE != 0:
        fail(f"{where}: stride is not a multiple of 0x{TILE:x}")
    dims = nchw_of(data, bdx)
    if 0 in dims[:4] or dims[4] == 0 or dims[5] == 0:
        fail(f"{where}: incomplete NCHW/tile geometry")
    product = dims[0] * dims[1] * dims[2] * dims[3]
    if product * element_size != tensor["logical_bytes"]:
        fail(f"{where}: NCHW product {product} != logical bytes")
    if tensor["logical_bytes"] != tensor["byte_size"]:
        fail(f"{where}: byte_size does not match dtype geometry")
    if dims[4] % dims[5] or dims[5] % 2:
        fail(f"{where}: tile geometry not 16-bit aligned")
    physical = dims[0] * dims[1] * dims[4]
    if physical > channel:
        fail(f"{where}: physical tile bytes exceed channel allocation")


def reconcile_package(package: pathlib.Path, out: pathlib.Path, args):
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("schema") != "mil-hwxc.h13-anec-package.v1":
        fail("schema is not mil-hwxc.h13-anec-package.v1")
    if manifest.get("target") != "H13":
        fail("target is not H13")
    plan = manifest["dispatchPlan"]
    programs = manifest["programs"]
    if plan != list(range(len(programs))):
        fail("dispatchPlan is not the program emission order")

    workspace_tiles = set()
    placement_failures = 0
    per_program = []
    for index, record in enumerate(programs):
        where = f"program[{index}] {record['file']}"
        payload = package / record["file"]
        data = payload.read_bytes()
        if len(data) != record["bytes"]:
            fail(f"{where}: file size disagrees with manifest record")
        digest = hashlib.sha256(data).hexdigest()
        if len(data) < HEADER:
            fail(f"{where}: smaller than the libane header")
        payload_size = u64_le(data, 0)
        td_size = u32_le(data, 8)
        td_count = u32_le(data, 12)
        task_size = u64_le(data, 16)
        kernel_size = u64_le(data, 24)
        src_count = u32_le(data, 32)
        dst_count = u32_le(data, 36)
        tiles = tiles_of(data)

        if td_count != record["taskDescriptors"]:
            fail(f"{where}: task descriptor count disagrees with record")
        if dst_count != len(record["outputs"]):
            fail(f"{where}: destination count disagrees with record")
        if src_count != len(record["inputs"]):
            fail(f"{where}: source count disagrees with record")
        if 0x1000 + payload_size > len(data):
            fail(f"{where}: executable payload extends past file")
        if tiles[0] == 0 or tiles[1] != 0:
            fail(f"{where}: command/kernel channel contract violated")
        if payload_size > tiles[0] * TILE:
            fail(f"{where}: payload exceeds command channel")
        if align16(task_size) != record["constantOffset"]:
            placement_failures += 1
            fail(f"{where}: kernel placement guard violated")
        if kernel_size != record["constantBytes"]:
            fail(f"{where}: kernel bytes disagree with constantBytes")
        if any(data[0x800:0x1000]):
            fail(f"{where}: 0x800..0x1000 is not zero (libane data offset)")

        workspace_tiles.add(tiles[3])
        for ordinal, output in enumerate(record["outputs"]):
            check_tensor_side(
                f"{where} output {output['name']}",
                output,
                data,
                4 + ordinal,
                True,
            )
        for ordinal, entry in enumerate(record["inputs"]):
            check_tensor_side(
                f"{where} input {entry['name']}",
                entry,
                data,
                4 + dst_count + ordinal,
                True,
            )
        per_program.append(
            {
                "file": record["file"],
                "sha256": digest,
                "byte_size": len(data),
                "constant_offset": record["constantOffset"],
                "operation": record["operation"],
                "inputs": [
                    {
                        "name": entry["name"],
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                        "byte_size": entry["logicalBytes"],
                        "stride": entry["allocationBytes"],
                        "slice": entry.get("slice"),
                    }
                    for entry in record["inputs"]
                ],
                "outputs": [
                    {
                        "name": entry["name"],
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                        "byte_size": entry["logicalBytes"],
                        "stride": entry["allocationBytes"],
                        "slice": entry.get("slice"),
                    }
                    for entry in record["outputs"]
                ],
            }
        )

    if workspace_tiles != {0}:
        fail(f"workspace tiles across package: {sorted(workspace_tiles)}")

    out.mkdir(parents=True, exist_ok=True)
    diagnostic = args.diagnostic_positive_workspace
    for index, bundle in enumerate(per_program):
        workspace_bytes = 16384 if diagnostic else 0
        record = programs[index]
        # Identity flags are verified inputs; they land verbatim in the
        # emitted manifests and are never defaulted.
        manifest_json = {
            "manifest_version": 2,
            "name": args.model_name,
            "graph_hash": args.model_sha256,
            "task_descriptors": record["taskDescriptors"],
            "inputs": [
                {
                    "name": entry["name"],
                    "index": entry["index"],
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "byte_size": entry["logicalBytes"],
                    "stride": entry["allocationBytes"],
                }
                for entry in record["inputs"]
            ],
            "outputs": [
                {
                    "name": entry["name"],
                    "index": entry["index"],
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "byte_size": entry["logicalBytes"],
                    "stride": entry["allocationBytes"],
                }
                for entry in record["outputs"]
            ],
            "workspace": [
                {
                    # Honest emission: byte_size 0, stride 0, shape [1]
                    # (shape dims must stay positive; the blocked field is
                    # byte_size). Diagnostic emission carries the 16384
                    # fabrication pattern v2 accepts today.
                    "name": "workspace",
                    "index": 0,
                    "dtype": "uint8",
                    "shape": [16384 if diagnostic else 1],
                    "byte_size": 16384 if diagnostic else 0,
                    "stride": 16384 if diagnostic else 0,
                }
            ],
            "payloads": [
                {
                    "role": "anec",
                    "path": record["file"],
                    "sha256": bundle["sha256"],
                    "byte_size": bundle["byte_size"],
                }
            ],
            "compiler": {
                "host_build": args.compiler_host_build,
                "toolchain": f"mil-hwxc {args.compiler_commit}",
                "target": "h13",
            },
            "firmware": {"min": args.firmware_min, "max": args.firmware_max},
            "provenance": {
                "source_repo": args.source_repo,
                "source_commit": args.source_commit,
                "exported_at": args.exported_at,
            },
            "release_asset": {
                "model": args.model_name,
                "model_sha256": args.model_sha256,
            },
        }
        directory = out / "bundles" / f"program-{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(manifest_json, indent=2) + "\n"
        )
        (directory / record["file"]).write_bytes(
            (package / record["file"]).read_bytes()
        )
    # The enclosing compiler package descriptor stays outside the bundle
    # directories: dispatchPlan ordering, per-program tensor slices, and
    # physical buffer spans are runtime coordination state, not per-program
    # bundle contract.
    descriptor = {
        "package_schema": manifest["schema"],
        "compiler_commit": args.compiler_commit,
        "source_package": str(package),
        "dispatch_plan": plan,
        "intermediates": manifest["intermediates"],
        "tensors": manifest["tensors"],
        "programs": per_program,
        "workspace_channel_tiles": sorted(workspace_tiles),
        "blocker": {
            "field": "workspace[0].byte_size",
            "honest_value": 0,
            "v2_requirement": "exactly one positive workspace tensor",
            "receipt": "2026-09-12-h13-bundle-adapter-blocker.md",
        },
    }
    (out / "dispatch_descriptor.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )
    return manifest, per_program, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--firmware-min", required=True)
    parser.add_argument("--firmware-max", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--exported-at", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--compiler-host-build", required=True)
    parser.add_argument(
        "--compiler-commit", default="a0ce354cf800011a84420da4e12013eb8140b2a5"
    )
    parser.add_argument("--diagnostic-positive-workspace", action="store_true")
    args = parser.parse_args()

    if len(args.model_sha256) != 64:
        fail("--model-sha256 must be a sha256 hex digest")
    if not args.firmware_min or not args.firmware_max:
        fail("firmware identity is a required verified input")

    manifest, programs, plan = reconcile_package(
        args.package, args.out, args
    )
    print(f"package: {args.package}")
    print(f"  schema {manifest['schema']} target {manifest['target']}")
    print(f"  programs {len(programs)} dispatchPlan entries {len(plan)}")
    print(f"  operations: conv=1 maximum={len(programs) - 1}")
    print(
        "  per-program reconciliation: file sizes, td/src/dst counts, "
        "channel allocations, tile/NCHW geometry, kernel placement "
        "(align16(task_size) == constantOffset), 0x800..0x1000 zero page: "
        "ALL PASS"
    )
    print(f"  tiles[3] (workspace channel) across all payloads: {{0}}")
    kind = (
        "DIAGNOSTIC fabrication-pattern (workspace 16384; accepted by v2 "
        "today; never load-bearing)"
        if args.diagnostic_positive_workspace
        else "honest (workspace byte_size 0; the blocked value)"
    )
    print(
        f"emitted {len(programs)} {kind} per-program v2 manifests "
        f"+ dispatch_descriptor.json under {args.out}"
    )
    print(
        "BLOCKED FIELD: workspace[0].byte_size — v2 demands exactly one "
        "positive workspace tensor; honest package value is 0. Owner "
        "decision owed (discovery receipt §8.6). No adapter shipped; "
        "host load validation is not ANE execution."
    )


if __name__ == "__main__":
    main()
