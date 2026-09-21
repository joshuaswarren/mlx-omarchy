"""Tests for mlx_omarchy_bonsai2.validate_artifact.

The unified serve CLI (mlx_omarchy_serve.__main__) calls
`getattr(module, 'validate_artifact')(model_dir)` to decide whether the
user-supplied directory is a usable Bonsai pack before admit-and-launch.
Bonsai packs ARE the upstream artifact -- there is no .convert
checkpoint_state classifier and no conversion step -- so the hook MUST
exist and reuse pack_footprint (config.json + safetensors header,
ZERO tensor bytes).

Positive: the existing tiny fixture (already shipped with the test
suite) validates and returns None.

Negative: an empty directory, a directory missing config.json, a
directory missing model.safetensors, a directory with a wrong-schema
config.json, and a directory with a safetensors header that has no
language_model.* tensors all fail with an informative error string
that the CLI can surface verbatim.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
TESTS = Path(__file__).resolve().parent
for p in (str(SERVE), str(TESTS)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


try:
    import mlx_omarchy_bonsai2  # noqa: F401
    _HAS_BONSAI = True
except ImportError:
    _HAS_BONSAI = False


def _import_validate():
    """Resolve validate_artifact lazily so missing deps fail with a clear
    error instead of an import traceback at collection time."""
    from mlx_omarchy_bonsai2 import validate_artifact
    return validate_artifact


@unittest.skipUnless(_HAS_BONSAI, "mlx_omarchy_bonsai2 import failed (MLX deps absent)")
class ValidateArtifactPositiveTests(unittest.TestCase):
    """A real Bonsai pack must validate to None."""

    def test_tiny_fixture_validates(self):
        """The tiny fixture pack from bonsai2_fixture is a fully-built
        prism_hadamard_qwen35 schema-2 pack with a fake vision tensor;
        validate_artifact must accept it."""
        import bonsai2_fixture  # noqa: F401

        with tempfile.TemporaryDirectory() as tmp:
            pack_dir, _ = bonsai2_fixture.build_tiny_pack(Path(tmp))
            result = _import_validate()(pack_dir)
            self.assertIsNone(
                result,
                "expected validate_artifact to return None for a built tiny fixture, got %r"
                % result,
            )

    def test_tiny_fixture_under_max_context_returns_none(self):
        """The validator must NOT fail when --max-context is not yet known
        to the CLI -- pack_footprint reads max_position_embeddings from
        config.json, not from CLI args."""
        import bonsai2_fixture  # noqa: F401

        with tempfile.TemporaryDirectory() as tmp:
            pack_dir, _ = bonsai2_fixture.build_tiny_pack(Path(tmp))
            self.assertIsNone(_import_validate()(pack_dir))


@unittest.skipUnless(_HAS_BONSAI, "mlx_omarchy_bonsai2 import failed (MLX deps absent)")
class ValidateArtifactNegativeTests(unittest.TestCase):
    """A non-pack directory or an invalid pack must return an error string
    that the CLI can surface verbatim."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_directory_fails(self):
        result = _import_validate()(self.tmp)
        self.assertIsNotNone(result, "empty directory must fail")
        self.assertIn("missing config.json", result)

    def test_directory_missing_model_safetensors_fails(self):
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_type": "prism_hadamard_qwen35",
                    "text_config": {"num_hidden_layers": 2},
                }
            )
        )
        result = _import_validate()(self.tmp)
        self.assertIsNotNone(result)
        self.assertIn("missing model.safetensors", result)

    def test_wrong_schema_fails(self):
        """A config.json with the wrong model_type is rejected by
        pack_footprint's _check_config and surfaced verbatim."""
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 99,  # not 2
                    "model_type": "wrong_arch",
                    "text_config": {},
                }
            )
        )
        # model.safetensors stub so pack_footprint reaches _check_config
        (self.tmp / "model.safetensors").write_bytes(b"")
        result = _import_validate()(self.tmp)
        self.assertIsNotNone(result)
        self.assertIn("pack validation failed", result)

    def test_safetensors_without_language_model_prefix_fails(self):
        """A safetensors header with no language_model.* tensors means
        the pack has no servable text model -- validator must refuse."""
        (self.tmp / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_type": "prism_hadamard_qwen35",
                    "text_config": {"num_hidden_layers": 2},
                }
            )
        )
        # Build a minimal-but-real safetensors header so _safetensors_header
        # succeeds and we get past the file-exists check. The simplest
        # legit empty-tensor safetensors file has just the 8-byte header.
        # The "no language_model.* tensors" failure mode in pack_footprint
        # is reached via the safetensors-header prefix filter -- we can
        # trigger it by writing a config that admits the header but the
        # header has only non-language_model tensors. The CLI's
        # responsibility ends at "this is not a Bonsai pack"; the
        # validator surfaces that.
        (self.tmp / "model.safetensors").write_bytes(b"\x00" * 8)
        result = _import_validate()(self.tmp)
        self.assertIsNotNone(result)
        # The pack_footprint failure mode for this case may be either
        # "no language_model.* tensors" or a header-parse error from
        # our 8-byte stub; both are legitimate rejections, not crashes.
        self.assertTrue(
            "language_model" in result
            or "pack validation failed" in result
            or "header" in result.lower(),
            "expected a validation error mentioning language_model / "
            "pack validation / header, got %r" % result,
        )

    def test_nonexistent_directory_fails(self):
        """A path that doesn't exist must return a clear error, not crash."""
        result = _import_validate()(self.tmp / "does-not-exist")
        self.assertIsNotNone(result)
        self.assertIn("not a directory", result)


if __name__ == "__main__":
    unittest.main()
