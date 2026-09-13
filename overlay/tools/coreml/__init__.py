# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML frontend tools for mlx-omarchy.

The package currently provides:

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
* :mod:`coreml.compiled_cache` — content-addressed storage for Linux-compiled
  ANE bundles. Keys bind model artifact hashes, selected function and static
  shapes, compiler identity, operation set, bundle/driver ABI, firmware
  compatibility identity, and frontend version. Every hit re-hashes payloads;
  corrupt entries are invalidated before the producer runs again.
* :mod:`coreml.tdt_control` — the pinned greedy Parakeet TDT state machine.
  Tensor backends own decoder and joint execution; host control handles only
  scalar decisions, opaque recurrent-state handles, and frame/token progress.
* :mod:`coreml.tokenizer` — integrity-checked Parakeet BPE detokenization.
  The host maps scalar token IDs to text with the pinned Metaspace and special-
  token semantics; it performs no tensor work.
* :mod:`coreml.depalettize` — frontend pre-expansion of
  ``constexpr_lut_to_dense`` into dense fp16/fp32 ``const`` ops whose
  values are appended to MIL blob storage. The gather copies raw payload
  bytes, so materialization is byte-exact; vector palettization is
  rejected with a named error.

Inspection never opens the ANE device, never imports coremltools or
MLX, and never reads or computes tensor data. It runs on any Linux
host with ``protobuf`` installed.
"""
