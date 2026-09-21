"""Bonsai-2 (prism_hadamard_qwen35) text serving for mlx-omarchy.

Serves prism-ml/Ternary-Bonsai-2-27B-mlx-2bit style packs: a Qwen3.5 hybrid
(GDN + attention) language model whose projections are stored as Hadamard-
folded ternary affine 2-bit group-128 quantized matrices. The pack's own
`runtime/` directory is remote code and is never imported or executed; the
semantics are re-implemented in this package so every line is auditable
here (`packed.py` is a reviewed port of the pinned pack loader).

Entry point (frozen serve contract, same as mlx_omarchy_laya):

    serve_main(["--model", <pack dir>, "--host", "127.0.0.1", "--port", "8080"])
"""

from .server import serve_main

__all__ = ["serve_main"]
