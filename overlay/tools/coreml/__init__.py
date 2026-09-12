# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Core ML frontend tools for mlx-omarchy.

Phase 1 (reference freeze) and Phase 2 (Linux .mlpackage inspection)
ship under this package. Phase 3 (compiler integration) and later
phases belong to the canonical compiler/runtime repositories.

Layout:

* :mod:`proto` — minimal protobuf wire-format reader.
* :mod:`mlpackage` — .mlpackage directory reader: manifest, model
  specification, function/block/operation inventory, weight discovery.
* :mod:`reference` — parakeet-reference.lock schema and cache
  verification (hash-pinned files, predictable cache, reuse, mismatch
  refusal).
* :mod:`fetch_parakeet_reference` — CLI entry point:
  ``python -m mlx_omarchy.coreml.fetch_parakeet_reference download``.
* :mod:`inspect_mlpackage` — CLI entry point:
  ``python -m mlx_omarchy.coreml.inspect_mlpackage inspect PATH``.

Inspector output is both human-readable and ``--json`` machine-readable.
Inspection never opens the ANE device.
"""
