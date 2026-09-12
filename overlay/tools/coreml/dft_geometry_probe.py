# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Generate and compare valid 512-point real-DFT impulse probes.

The impulse basis alternates a unit sample at ``2*c`` and ``2*c+1`` for
``c = 0..255``. Candidate 256-point complex cores feed the established
real-DFT untangle, then exact float32 and ULP signatures are compared with a
``mel-stage-capture --mode probe-dft`` dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_REAL = 512
N_COMPLEX = N_REAL // 2
N_BINS = N_REAL // 2 + 1


def f32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def c64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.complex64)


def twiddle(length: int, indices: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * indices / length).astype(np.complex64)


def digit_reverse_base4(length: int) -> np.ndarray:
    stages = length.bit_length() // 2
    if 4**stages != length:
        raise ValueError(f"radix-4 length must be a power of four, got {length}")
    result = np.zeros(length, dtype=np.int64)
    for source in range(length):
        value = source
        target = 0
        for _ in range(stages):
            target = 4 * target + value % 4
            value //= 4
        result[source] = target
    return result


def radix4_dit(values: np.ndarray) -> np.ndarray:
    z = np.take(c64(values), digit_reverse_base4(values.shape[1]), axis=1)
    length = z.shape[1]
    stage = 4
    while stage <= length:
        quarter = stage // 4
        groups = z.reshape(z.shape[0], length // stage, stage)
        indices = np.arange(quarter)
        a = groups[:, :, :quarter]
        b = c64(groups[:, :, quarter:2 * quarter] * twiddle(stage, indices))
        c = c64(groups[:, :, 2 * quarter:3 * quarter] * twiddle(stage, 2 * indices))
        d = c64(groups[:, :, 3 * quarter:] * twiddle(stage, 3 * indices))
        ac_sum = c64(a + c)
        ac_diff = c64(a - c)
        bd_sum = c64(b + d)
        bd_diff = c64(b - d)
        minus_i_bd = c64(bd_diff.imag - 1j * bd_diff.real)
        out = np.empty_like(groups)
        out[:, :, :quarter] = c64(ac_sum + bd_sum)
        out[:, :, quarter:2 * quarter] = c64(ac_diff + minus_i_bd)
        out[:, :, 2 * quarter:3 * quarter] = c64(ac_sum - bd_sum)
        out[:, :, 3 * quarter:] = c64(ac_diff - minus_i_bd)
        z = out.reshape(z.shape)
        stage *= 4
    return z


def radix4_dif(values: np.ndarray) -> np.ndarray:
    z = c64(values).copy()
    length = z.shape[1]
    stage = length
    while stage >= 4:
        quarter = stage // 4
        groups = z.reshape(z.shape[0], length // stage, stage)
        a = groups[:, :, :quarter]
        b = groups[:, :, quarter:2 * quarter]
        c = groups[:, :, 2 * quarter:3 * quarter]
        d = groups[:, :, 3 * quarter:]
        ac_sum = c64(a + c)
        ac_diff = c64(a - c)
        bd_sum = c64(b + d)
        bd_diff = c64(b - d)
        minus_i_bd = c64(bd_diff.imag - 1j * bd_diff.real)
        indices = np.arange(quarter)
        out = np.empty_like(groups)
        out[:, :, :quarter] = c64(ac_sum + bd_sum)
        out[:, :, quarter:2 * quarter] = c64(
            (ac_diff + minus_i_bd) * twiddle(stage, indices)
        )
        out[:, :, 2 * quarter:3 * quarter] = c64(
            (ac_sum - bd_sum) * twiddle(stage, 2 * indices)
        )
        out[:, :, 3 * quarter:] = c64(
            (ac_diff - minus_i_bd) * twiddle(stage, 3 * indices)
        )
        z = out.reshape(z.shape)
        stage //= 4
    return np.take(z, digit_reverse_base4(length), axis=1)


def split_radix_dit(values: np.ndarray) -> np.ndarray:
    z = c64(values)
    length = z.shape[1]
    if length == 1:
        return z.copy()
    if length == 2:
        return np.stack((c64(z[:, 0] + z[:, 1]), c64(z[:, 0] - z[:, 1])), axis=1)
    quarter = length // 4
    even = split_radix_dit(z[:, ::2])
    odd_one = split_radix_dit(z[:, 1::4])
    odd_three = split_radix_dit(z[:, 3::4])
    indices = np.arange(quarter)
    first = c64(odd_one * twiddle(length, indices))
    third = c64(odd_three * twiddle(length, 3 * indices))
    odd_sum = c64(first + third)
    odd_diff = c64(first - third)
    minus_i_diff = c64(odd_diff.imag - 1j * odd_diff.real)
    out = np.empty_like(z)
    out[:, :quarter] = c64(even[:, :quarter] + odd_sum)
    out[:, quarter:2 * quarter] = c64(even[:, quarter:] + minus_i_diff)
    out[:, 2 * quarter:3 * quarter] = c64(even[:, :quarter] - odd_sum)
    out[:, 3 * quarter:] = c64(even[:, quarter:] - minus_i_diff)
    return out


def untangle(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(1, N_COMPLEX)
    direct = core[:, indices]
    mirrored = np.conj(core[:, N_COMPLEX - indices])
    pair_sum = c64(direct + mirrored)
    pair_diff = c64(direct - mirrored)
    weighted = c64(c64(twiddle(N_REAL, indices) * np.complex64(0.5j)) * pair_diff)
    spectrum = c64(c64(np.float32(0.5) * pair_sum) - weighted)
    real = np.zeros((core.shape[0], N_BINS), dtype=np.float32)
    imag = np.zeros_like(real)
    real[:, 0] = f32(core[:, 0].real + core[:, 0].imag)
    real[:, -1] = f32(core[:, 0].real - core[:, 0].imag)
    real[:, 1:-1] = spectrum.real
    imag[:, 1:-1] = spectrum.imag
    return real, imag


def candidate_core(frames: np.ndarray, candidate: str) -> np.ndarray:
    packed = c64(frames[:, 0::2] + np.complex64(1j) * frames[:, 1::2])
    functions = {
        "radix4-dit": radix4_dit,
        "radix4-dif": radix4_dif,
        "split-radix-dit": split_radix_dit,
    }
    return functions[candidate](packed)


def candidate_spectrum(frames: np.ndarray, candidate: str) -> tuple[np.ndarray, np.ndarray]:
    return untangle(candidate_core(frames, candidate))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_probe(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = directory / f"{name}.npy"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest mismatch for {name}: expected {expected}, got {actual}")
    frames = np.load(directory / "probe_in.npy").reshape(-1, N_REAL)
    real = np.load(directory / "probe_dft_real.npy")
    imag = np.load(directory / "probe_dft_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("probe arrays must be float32")
    if real.shape != (frames.shape[0], N_BINS) or imag.shape != real.shape:
        raise ValueError(
            f"invalid probe shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("probe arrays contain non-finite values")
    return frames, real, imag


def load_core_probe(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = directory / f"{name}.npy"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest mismatch for {name}: expected {expected}, got {actual}")
    frames = np.load(directory / "probe_in.npy").reshape(-1, N_REAL)
    real = np.load(directory / "probe_core_real.npy")
    imag = np.load(directory / "probe_core_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("core probe arrays must be float32")
    if real.shape != (frames.shape[0], N_COMPLEX) or imag.shape != real.shape:
        raise ValueError(
            f"invalid core probe shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("core probe arrays contain non-finite values")
    return frames, real, imag


def load_stage(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("frames", "dft_real", "dft_imag"):
        expected = manifest[name]
        actual = sha256_file(directory / f"{name}.npy")
        if actual != expected:
            raise ValueError(
                f"manifest mismatch for {name}: expected {expected}, got {actual}"
            )
    frames = np.load(directory / "frames.npy")
    real = np.load(directory / "dft_real.npy")
    imag = np.load(directory / "dft_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("stage arrays must be float32")
    if frames.ndim != 2 or frames.shape[1] != N_REAL:
        raise ValueError(f"invalid stage frame shape: {frames.shape}")
    if real.shape != (frames.shape[0], N_BINS) or imag.shape != real.shape:
        raise ValueError(
            f"invalid stage shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("stage arrays contain non-finite values")
    return frames, real, imag


def ordered_float_bits(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    positive = bits ^ np.uint32(0x80000000)
    return np.where(bits & np.uint32(0x80000000), ~bits, positive).astype(np.int64)


def score(candidate_real: np.ndarray, candidate_imag: np.ndarray,
          reference_real: np.ndarray, reference_imag: np.ndarray) -> dict[str, object]:
    candidate = np.stack((candidate_real, candidate_imag), axis=-1)
    reference = np.stack((reference_real, reference_imag), axis=-1)
    bit_exact = candidate.view(np.uint32) == reference.view(np.uint32)
    numeric_exact = candidate == reference
    signed_ulp = ordered_float_bits(candidate) - ordered_float_bits(reference)
    mismatch_ulp = signed_ulp[~numeric_exact]
    histogram = {
        str(delta): int((mismatch_ulp == delta).sum())
        for delta in range(-8, 9)
        if (mismatch_ulp == delta).any()
    }
    histogram["<-8"] = int((mismatch_ulp < -8).sum())
    histogram[">8"] = int((mismatch_ulp > 8).sum())
    row_mismatches = (~numeric_exact).sum(axis=(1, 2)).astype(np.uint16)
    return {
        "bit_mismatches": int((~bit_exact).sum()),
        "numeric_mismatches": int((~numeric_exact).sum()),
        "signed_zero_mismatches": int((numeric_exact & ~bit_exact).sum()),
        "values": int(bit_exact.size),
        "exact_rows": int((row_mismatches == 0).sum()),
        "max_abs": float(np.max(np.abs(
            candidate.astype(np.float64) - reference.astype(np.float64)
        ))),
        "max_ulp_on_numeric_mismatch": int(np.max(np.abs(mismatch_ulp), initial=0)),
        "signed_ulp_histogram": histogram,
        "row_mismatch_signature_sha256": hashlib.sha256(row_mismatches.tobytes()).hexdigest(),
    }


def validate_impulse_basis(frames: np.ndarray) -> None:
    expected = np.eye(N_REAL, dtype=np.float32)
    if frames.shape != expected.shape or frames.tobytes() != expected.tobytes():
        raise ValueError("impulse probe input must be the exact 512x512 float32 identity basis")


def make_input(path: Path) -> None:
    basis = np.eye(N_REAL, dtype="<f4")
    path.write_bytes(basis.tobytes())
    print(json.dumps({
        "path": str(path),
        "shape": list(basis.shape),
        "dtype": str(basis.dtype),
        "sha256": sha256_file(path),
    }, sort_keys=True))


def compare(impulse_dump: Path, core_dump: Path | None, general_dump: Path | None,
            stage_dump: Path | None, json_out: Path | None) -> None:
    impulse_frames, impulse_real, impulse_imag = load_probe(impulse_dump)
    validate_impulse_basis(impulse_frames)
    data: dict[str, object] = {
        "impulse_manifest_sha256": sha256_file(impulse_dump / "manifest.json"),
        "candidates": {},
    }
    core_reference = None
    if core_dump is not None:
        core_frames, core_real, core_imag = load_core_probe(core_dump)
        validate_impulse_basis(core_frames)
        if core_frames.tobytes() != impulse_frames.tobytes():
            raise ValueError("real and complex impulse probes used different inputs")
        core_reference = core_real, core_imag
        data["core_manifest_sha256"] = sha256_file(core_dump / "manifest.json")
    datasets = {"impulse": (impulse_frames, impulse_real, impulse_imag)}
    if general_dump is not None:
        datasets["general"] = load_probe(general_dump)
    if stage_dump is not None:
        datasets["stage-3001"] = load_stage(stage_dump)
        data["stage_manifest_sha256"] = sha256_file(stage_dump / "manifest.json")
    for candidate in ("radix4-dit", "radix4-dif", "split-radix-dit"):
        candidate_results = {}
        if core_reference is not None:
            core = candidate_core(impulse_frames, candidate)
            candidate_results["complex-core"] = score(
                core.real, core.imag, core_reference[0], core_reference[1]
            )
        for dataset, (frames, reference_real, reference_imag) in datasets.items():
            candidate_real, candidate_imag = candidate_spectrum(frames, candidate)
            candidate_results[dataset] = score(
                candidate_real, candidate_imag, reference_real, reference_imag
            )
        data["candidates"][candidate] = candidate_results
    rendered = json.dumps(data, indent=2, sort_keys=True)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(rendered + "\n")
    print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make_parser = subparsers.add_parser("make-input")
    make_parser.add_argument("path", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--impulse-dump", type=Path, required=True)
    compare_parser.add_argument("--core-dump", type=Path)
    compare_parser.add_argument("--general-dump", type=Path)
    compare_parser.add_argument("--stage-dump", type=Path)
    compare_parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.command == "make-input":
        make_input(args.path)
    else:
        compare(args.impulse_dump, args.core_dump, args.general_dump,
                args.stage_dump, args.json_out)


if __name__ == "__main__":
    main()
