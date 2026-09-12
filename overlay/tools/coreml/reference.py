# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Parakeet reference lock schema and cache verification.

Defines the :data:`ReferenceLock` dataclass, the cache directory layout,
and the verify / reuse logic for the pinned public Parakeet reference.

The lock file (``parakeet-reference.lock``) lives next to this module
under ``overlay/tools/coreml/``. It pins:

* the GitHub commit of ``mweinbach/parakeet-coreml-swift`` (the public
  reference implementation),
* the Hugging Face revision of ``mweinbach1/parakeet-tdt-0.6b-v3-coreml``
  (the public reference model),
* sha-256 + size of every file in that revision,
* the audio fixture (URL, sha-256, license, sample rate),
* the mel-feature configuration,
* the TDT-decoder configuration,
* the macOS Core ML reference environment used to produce accepted
  outputs (kernel, macOS release, coremltools version, compute unit).

The downloader writes the cache as::

    $MLX_OMARCHY_CACHE_DIR/parakeet-reference/
        <model-repo>/<hf-revision>/
            .gitattributes
            README.md
            encoder.mlpackage/...
            decoder.mlpackage/...
            joint.mlpackage/...
            tokenizer.json
            .sha256sums                 <- per-file hashes recorded on download
            .manifest-stamp             <- size of each file recorded on download

Default cache root (Linux) is ``~/.cache/mlx-omarchy/parakeet-reference``,
matching the ``~/.cache/mlx-omarchy/coreml`` cache root that later
compiler/runtime phases will use.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

# Default cache root: $MLX_OMARCHY_CACHE_DIR wins if set (allows test/CI
# isolation), else $XDG_CACHE_HOME/mlx-omarchy/parakeet-reference, else
# ~/.cache/mlx-omarchy/parakeet-reference.
def default_cache_root() -> Path:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    if base:
        return Path(base) / "parakeet-reference"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "mlx-omarchy" / "parakeet-reference"
    return Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference"


def model_cache_dir(cache_root: Path, model_repo: str, revision: str) -> Path:
    """Predictable cache directory for a Hugging Face model revision."""
    return cache_root / model_repo / revision


# Sentinel sub-file inside the cache. Recorded at download time so a
# mismatch on the next run fails closed without re-downloading.
CACHE_STAMP_NAME = ".manifest-stamp"
CACHE_HASHES_NAME = ".sha256sums"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIZE_RE = re.compile(r"^[0-9]+$")


class ReferenceError(RuntimeError):
    """Raised for any reference-lock or cache-verification problem."""


@dataclass
class LockedFile:
    """One pinned file inside the reference model revision."""

    path: str            # repo-relative path, forward slashes
    size: int            # exact byte size from the HF tree
    sha256: str          # sha-256, lowercase hex (HF LFS oid for LFS files)


@dataclass
class MelConfig:
    """Mel feature extractor configuration (must match the reference)."""

    sample_rate: int
    hop_length: int
    win_length: int
    n_fft: int
    n_mels: int
    preemphasis: float
    log_guard: float  # eps added before log


@dataclass
class TdtConfig:
    """TDT decoder configuration (must match the reference)."""

    blank_token_id: int
    durations: list[int]
    max_symbols_per_step: int
    vocab_size: int


@dataclass
class AudioFixture:
    """Pinned deterministic audio input for the encoder parity receipt."""

    url: str            # canonical source URL
    sha256: str         # lowercase hex
    size: int
    sample_rate: int    # expected after any decoding step
    duration_seconds: float
    license: str        # SPDX or short string ("public-domain", "CC-BY-4.0", ...)
    note: str = ""      # provenance / origin of the file


@dataclass
class MacosReferenceEnvironment:
    """Environment used to produce the macOS Core ML reference output."""

    chip: str           # e.g. "Apple M1 (T8103)"
    macos_release: str  # e.g. "macOS 15.5"
    coremltools_version: str
    swift_toolchain: str
    compute_units: str  # e.g. "cpuAndNeuralEngine" (the swift default)
    parakeet_coreml_swift_commit: str


@dataclass
class ReferenceLock:
    """Parsed :file:`parakeet-reference.lock`."""

    schema_version: int
    reference_repo: str            # "mweinbach/parakeet-coreml-swift"
    reference_commit: str          # full 40-char GitHub sha
    model_repo: str                # "mweinbach1/parakeet-tdt-0.6b-v3-coreml"
    model_revision: str            # full 40-char HF sha
    model_license: str             # SPDX, e.g. "CC-BY-4.0"
    model_quantization: str        # human description, e.g. "encoder 4-bit palettized, decoder/joint fp16"
    files: list[LockedFile]
    audio: AudioFixture
    mel: MelConfig
    tdt: TdtConfig
    macos_reference_environment: MacosReferenceEnvironment | None  # None until captured
    macos_reference_paths: dict[str, str] = field(default_factory=dict)
    # macos_reference_paths maps:
    #   "encoder_outputs_sha256"     ->  sha-256 of macOS encoder output tensor (.npy/.npz)
    #   "token_ids_sha256"           ->  sha-256 of expected token ids file
    #   "transcript_sha256"          ->  sha-256 of transcript text file
    #   "capture_log_sha256"         ->  sha-256 of the capture run log
    # Values are absent (key missing) until a Mac capture lands; the
    # downloader does not write them.

    # ---------- Serialisation ----------

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceLock":
        if not isinstance(d, dict):
            raise ReferenceError("reference lock must be a JSON object")
        try:
            return cls(
                schema_version=int(d["schema_version"]),
                reference_repo=str(d["reference_repo"]),
                reference_commit=str(d["reference_commit"]),
                model_repo=str(d["model_repo"]),
                model_revision=str(d["model_revision"]),
                model_license=str(d["model_license"]),
                model_quantization=str(d["model_quantization"]),
                files=[LockedFile(**f) for f in d["files"]],
                audio=AudioFixture(**d["audio"]),
                mel=MelConfig(**d["mel"]),
                tdt=TdtConfig(**d["tdt"]),
                macos_reference_environment=(
                    MacosReferenceEnvironment(**d["macos_reference_environment"])
                    if d.get("macos_reference_environment")
                    else None
                ),
                macos_reference_paths=dict(d.get("macos_reference_paths", {})),
            )
        except (KeyError, TypeError) as ex:
            raise ReferenceError(f"invalid reference lock: {ex}") from ex

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "reference_repo": self.reference_repo,
            "reference_commit": self.reference_commit,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "model_license": self.model_license,
            "model_quantization": self.model_quantization,
            "files": [asdict(f) for f in self.files],
            "audio": asdict(self.audio),
            "mel": asdict(self.mel),
            "tdt": asdict(self.tdt),
            "macos_reference_environment": (
                asdict(self.macos_reference_environment)
                if self.macos_reference_environment
                else None
            ),
            "macos_reference_paths": dict(self.macos_reference_paths),
        }

    @classmethod
    def load(cls, path: Path) -> "ReferenceLock":
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as ex:
            raise ReferenceError(f"reference lock not found: {path}") from ex
        try:
            data = json.loads(text)
        except json.JSONDecodeError as ex:
            raise ReferenceError(f"reference lock is not valid JSON: {ex}") from ex
        return cls.from_dict(data)


def write_lock(lock: ReferenceLock, path: Path) -> None:
    """Write the lock in sorted, stable JSON form for diff review."""
    payload = lock.to_dict()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_lock(lock: ReferenceLock) -> None:
    """Fail closed on a malformed lock. No I/O."""
    if lock.schema_version != 1:
        raise ReferenceError(f"unsupported reference lock schema_version={lock.schema_version}")
    if not _SHA256_RE.match(lock.reference_commit):
        raise ReferenceError(
            f"reference_commit must be 40-char hex, got {lock.reference_commit!r}"
        )
    if not _SHA256_RE.match(lock.model_revision):
        raise ReferenceError(
            f"model_revision must be 40-char hex, got {lock.model_revision!r}"
        )
    if not lock.files:
        raise ReferenceError("reference lock contains no files")
    seen: set[str] = set()
    for f in lock.files:
        if f.path in seen:
            raise ReferenceError(f"duplicate file entry: {f.path}")
        seen.add(f.path)
        if not _SHA256_RE.match(f.sha256):
            raise ReferenceError(f"{f.path}: sha256 must be 64-char hex")
        if f.size < 0:
            raise ReferenceError(f"{f.path}: negative size")
    if not _SHA256_RE.match(lock.audio.sha256):
        raise ReferenceError(f"audio fixture sha256 must be 64-char hex")
    if lock.tdt.blank_token_id >= lock.tdt.vocab_size:
        raise ReferenceError(
            f"tdt.blank_token_id={lock.tdt.blank_token_id} >= vocab_size={lock.tdt.vocab_size}"
        )
    if not lock.tdt.durations:
        raise ReferenceError("tdt.durations must not be empty")
    if lock.tdt.max_symbols_per_step <= 0:
        raise ReferenceError("tdt.max_symbols_per_step must be > 0")
    if lock.mel.n_fft <= 0 or lock.mel.hop_length <= 0 or lock.mel.win_length <= 0:
        raise ReferenceError("mel.n_fft/hop_length/win_length must be positive")


# ---------- Cache verification ----------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """FIPS 180-4 SHA-256 of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cache(
    cache_dir: Path,
    lock: ReferenceLock,
) -> tuple[bool, list[str]]:
    """Verify every pinned file is present with matching size + sha256.

    Returns ``(ok, mismatches)``. A single pass over all files; never
    raises on a single mismatch — the caller decides whether to redownload
    or to surface the failure.
    """
    mismatches: list[str] = []
    for f in lock.files:
        p = cache_dir / f.path
        if not p.is_file():
            mismatches.append(f"missing: {f.path}")
            continue
        try:
            actual_size = p.stat().st_size
        except OSError as ex:
            mismatches.append(f"stat-fail: {f.path}: {ex}")
            continue
        if actual_size != f.size:
            mismatches.append(
                f"size-mismatch: {f.path}: expected {f.size}, got {actual_size}"
            )
            continue
        actual_sha = sha256_file(p)
        if actual_sha != f.sha256:
            mismatches.append(
                f"sha256-mismatch: {f.path}: expected {f.sha256}, got {actual_sha}"
            )
    return (not mismatches), mismatches


def record_stamps(cache_dir: Path, lock: ReferenceLock) -> None:
    """Write the sidecar stamp files recording the verified state.

    These files let the next run skip the file-by-file verification when
    the cache_dir mtime, lock sha256, and stamp all match. They are not
    a substitute for verification — a re-run still rehashes on any
    divergence.
    """
    stamp = cache_dir / CACHE_STAMP_NAME
    hashes = cache_dir / CACHE_HASHES_NAME
    stamp_lines = []
    hash_lines = []
    for f in lock.files:
        p = cache_dir / f.path
        stamp_lines.append(f"{f.size}\t{f.path}\n")
        hash_lines.append(f"{f.sha256}  {f.path}\n")
    stamp.write_text("".join(stamp_lines), encoding="utf-8")
    hashes.write_text("".join(hash_lines), encoding="utf-8")
