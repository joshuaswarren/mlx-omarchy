# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML frontend tools for mlx-omarchy.

Phase 2 ships the public ``.mlpackage`` inspector under this package:

* :mod:`coreml.schema` — the officially vendored Core ML protobuf
  schema (Apple coremltools 9.0 generated bindings plus ``.proto``
  sources; provenance and hashes in ``schema/VENDORED.json``).
* :mod:`coreml.proto` — schema access: loads model specifications
  through the generated bindings and names dtypes/shapes from the
  official enums (FeatureType and MIL dtype enums stay distinct).
* :mod:`coreml.mlpackage` — .mlpackage reader and typed inventory:
  manifest validation, function/block/operation enumeration with
  bindings and typed outputs, weight files + blob references,
  compression representation, versions, control flow, parse validity.
  Compiler eligibility is explicitly not assessed here.
* :mod:`coreml.inspect_mlpackage` — CLI entry point::

      python3 overlay/tools/coreml/inspect_mlpackage.py inspect PATH [--json] [--strict]

* :mod:`coreml.reference` — parakeet-reference lock and cache
  verification (owned with the reference freeze; see docs/parakeet.md).

Inspection never opens the ANE device, never imports coremltools or
MLX, and never reads or computes tensor data. It runs on any Linux
host with ``protobuf`` installed.
"""
