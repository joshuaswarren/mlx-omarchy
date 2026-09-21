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

from __future__ import annotations

from pathlib import Path

from .server import serve_main

__all__ = ["serve_main", "validate_artifact"]


def validate_artifact(model_dir) -> str | None:
    """Validate a Bonsai pack directory without loading any tensors.

    This is the contract the unified serve CLI (mlx_omarchy_serve.__main__)
    calls via `getattr(module, 'validate_artifact')` before admit-and-launch.
    Returns None on success, or a short error string on failure. The CLI
    refuses to launch when this returns non-None.

    Implementation reuses pack_footprint (config.json schema + safetensors
    header only -- ZERO tensor bytes read) and surface its _fail /
    _check_config diagnostics verbatim so the CLI message is actionable.
    No <pkg>.convert.checkpoint_state is needed: Bonsai packs ARE the
    upstream artifact and require no conversion step.

    Validates the real required snapshot files for a servable Bonsai
    pack: config.json (schema 2, prism_hadamard_qwen35) and
    model.safetensors (header must contain language_model.* tensors).
    The tokenizer files are required at SERVE time but are not strictly
    required for artifact validation -- missing tokenizer surfaces later
    from load_text_model's own _check_config path, which is the right
    place to fail loudly.
    """
    pack = Path(model_dir)
    if not pack.is_dir():
        return f"{pack}: not a directory"
    config_path = pack / "config.json"
    if not config_path.is_file():
        return f"{pack}: missing config.json"
    safetensors = pack / "model.safetensors"
    if not safetensors.is_file():
        return f"{pack}: missing model.safetensors"
    try:
        from .loader import pack_footprint
        facts = pack_footprint(pack)
    except Exception as exc:
        # pack_footprint raises on invalid config / missing tensors /
        # wrong schema; surface the exact message verbatim so CLI users
        # can act on it.
        return f"pack validation failed: {exc}"
    if not facts.get("live_weights_bytes"):
        return f"{pack}: no language_model.* tensors in checkpoint"
    return None
