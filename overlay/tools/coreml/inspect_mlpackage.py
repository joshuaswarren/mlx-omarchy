#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML .mlpackage inspector (human or JSON output, device-free).

Usage:
    python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [--json] [--strict]
    python3 -m coreml.inspect_mlpackage inspect PATH [--json] [--strict]

Reports the plan's section-14 checklist for one ``.mlpackage``:
structure, model type, functions/ops (typed), weights and blob
references, compression representation, versions, control flow, and
parse validity. Compiler/deployment eligibility is explicitly **not**
assessed (see ``eligibility`` in the JSON output). No ANE device, no
GPU, and no compiler is opened; runs on any Linux host.

Exit codes: 0 = inspected, 1 = invalid package or usage error,
2 = ``--strict`` validation failure (missing weight files, unset
model type, opset inconsistencies).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # plain-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coreml.mlpackage import MlPackageError, inspect  # noqa: E402
from coreml.proto import ModelSpecError, schema_info  # noqa: E402


def _fmt_type(t: dict) -> str:
    if t.get("kind") == "multiArrayType":
        flex = t.get("shape_flexibility", {})
        suffix = ""
        if flex.get("kind") == "shapeRange":
            suffix = " range"
        elif flex.get("kind") == "enumeratedShapes":
            suffix = f" enumerated{len(flex.get('shapes', []))}"
        return f"{t.get('feature_dtype')} {t.get('shape')}{suffix}"
    return str(t.get("kind"))


def human_report(inv: dict) -> str:
    lines: list[str] = []
    pkg = inv["package"]
    model = inv["model"]
    lines.append(f"Package: {pkg['path']}")
    lines.append(
        f"Format: {pkg['format']} (fileFormatVersion {pkg['file_format_version']})"
    )
    lines.append(f"Model type: {model['type']}")
    lines.append(f"Specification version: {model['specification_version']}")
    for key, entries in (
        ("Inputs", model["description"]["inputs"]),
        ("Outputs", model["description"]["outputs"]),
        ("State", model["description"]["state"]),
    ):
        lines.append(f"{key}: {len(entries)}")
        for e in entries:
            lines.append(f"  {e['name']}: {_fmt_type(e)}")
    for fn in inv["program"]["functions"]:
        lines.append(
            f"Function {fn['name']}: opset={fn['opset']} "
            f"blocks={len(fn['block_specializations'])} ops={fn['op_total']}"
        )
        for name, dtype in sorted(fn["op_histogram"].items(), key=lambda kv: -kv[1])[:12]:
            lines.append(f"  {name}: {dtype}")
        if len(fn["op_histogram"]) > 12:
            lines.append(f"  ... {len(fn['op_histogram']) - 12} more op types")
    comp = inv["compression"]
    lines.append(
        f"Compression: {comp['representation']}"
        + (f" ({comp['constexpr_op_count']} constexpr ops)" if comp["constexpr_op_count"] else "")
    )
    lines.append("Weights:")
    for wf in inv["weights"]["files"]:
        lines.append(f"  {wf['file']}: {wf['size']} bytes sha256={wf['sha256']}")
    for ref in inv["weights"]["blob_references"]:
        status = "ok" if ref["resolved"] else "MISSING FILE"
        lines.append(
            f"  blob ref {ref['blob_file']}: {ref['distinct_offsets']} offsets [{status}]"
        )
    if inv["control_flow"]["present"]:
        lines.append(f"Control flow: {', '.join(inv['control_flow']['ops'])}")
    else:
        lines.append("Control flow: none")
    validity = inv["validity"]
    lines.append(f"Parse validity: {validity['protobuf_parse']}")
    for note in validity["notes"]:
        lines.append(f"  note: {note}")
    lines.append(f"Compiler eligibility: not assessed (see JSON eligibility.note)")
    lines.append(
        f"Schema: official Core ML protobuf, vendored from {schema_info()['upstream']} "
        f"tag {schema_info()['tag']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_mlpackage", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_inspect = sub.add_parser("inspect", help="inspect one .mlpackage")
    p_inspect.add_argument("path", type=Path, help="path to the .mlpackage directory")
    p_inspect.add_argument("--json", action="store_true", help="machine-readable output")
    p_inspect.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 if unresolved weight files, unset model type, or "
        "opset inconsistencies are found (parse validity is unaffected)",
    )
    args = parser.parse_args(argv)

    try:
        inv = inspect(args.path)
    except (MlPackageError, ModelSpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(inv, indent=2, sort_keys=False))
    else:
        print(human_report(inv))

    if args.strict:
        problems = list(inv["validity"]["notes"])
        problems += [
            f"unresolved blob reference {r['blob_file']}"
            for r in inv["weights"]["blob_references"]
            if not r["resolved"]
        ]
        if not inv["model"]["type"] or inv["model"]["type"] == "UNSET":
            problems.append("model Type oneof is unset")
        if problems:
            for problem in problems:
                print(f"strict: {problem}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
