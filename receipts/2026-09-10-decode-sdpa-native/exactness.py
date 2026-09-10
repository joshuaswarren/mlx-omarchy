#!/usr/bin/env python3
"""Capture native decode SDPA or verify the Omarchy kernel bit-for-bit."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import mlx.core as mx
import numpy as np

FIXED_LENGTHS = (30, 62, 127, 263, 1053)
RANDOM_LENGTHS = (1, 30, 31, 32, 62, 127, 263, 1053)
HEADS = 14
KV_HEADS = 2
DIM = 64
SCALE = 1.0 / np.sqrt(DIM)


def inputs(seed, keys):
    rng = np.random.Generator(np.random.PCG64(seed))
    capacity = keys + 17
    q = (rng.standard_normal((1, HEADS, 1, DIM), dtype=np.float32) * 0.5).astype(np.float16)
    k = (rng.standard_normal((1, KV_HEADS, capacity, DIM), dtype=np.float32) * 0.5).astype(np.float16)
    v = (rng.standard_normal((1, KV_HEADS, capacity, DIM), dtype=np.float32) * 0.5).astype(np.float16)
    return q, k, v


def run_case(seed, keys, input_hash):
    q, k_cache, v_cache = inputs(seed, keys)
    for value in (q, k_cache, v_cache):
        input_hash.update(value.tobytes())
    q_mx = mx.array(q)
    k_mx = mx.array(k_cache)[:, :, :keys, :]
    v_mx = mx.array(v_cache)[:, :, :keys, :]
    out = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=SCALE)
    return np.asarray(out)


def cases():
    for keys in FIXED_LENGTHS:
        yield f"fixed_k{keys}", 9000 + keys, keys, True
    for keys in RANDOM_LENGTHS:
        for seed in range(100):
            yield f"random_k{keys}_s{seed}", seed, keys, False


def provenance(input_digest):
    info = mx.device_info()
    return {
        "mlx_version": mx.__version__,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": {key: str(value) for key, value in info.items()},
        "shapes": {"q_heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM},
        "fixed_key_lengths": list(FIXED_LENGTHS),
        "random_key_lengths": list(RANDOM_LENGTHS),
        "random_seeds_per_length": 100,
        "random_cases": len(RANDOM_LENGTHS) * 100,
        "input_sha256": input_digest,
    }


def capture(root):
    expected = {}
    input_hash = hashlib.sha256()
    for name, seed, keys, _ in cases():
        expected[name] = run_case(seed, keys, input_hash)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "native-captures.npz", **expected)
    record = provenance(input_hash.hexdigest())
    (root / "native-capture-provenance.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def verify(root):
    expected = np.load(root / "native-captures.npz")
    input_hash = hashlib.sha256()
    fixed = {}
    random_cases = 0
    random_elements = 0
    random_mismatches = 0
    first_failure = None
    for name, seed, keys, is_fixed in cases():
        actual = run_case(seed, keys, input_hash)
        reference = expected[name]
        mismatches = int(np.count_nonzero(
            actual.view(np.uint16) != reference.view(np.uint16)))
        max_error = float(np.max(np.abs(
            actual.astype(np.float32) - reference.astype(np.float32))))
        if is_fixed:
            fixed[str(keys)] = {
                "elements": int(actual.size),
                "mismatches": mismatches,
                "max_error": max_error,
            }
        else:
            random_cases += 1
            random_elements += int(actual.size)
            random_mismatches += mismatches
            if mismatches and first_failure is None:
                first_failure = {
                    "key_length": keys,
                    "seed": seed,
                    "mismatches": mismatches,
                    "max_error": max_error,
                }
    capture_provenance = json.loads(
        (root / "native-capture-provenance.json").read_text())
    digest = input_hash.hexdigest()
    result = {
        "candidate": provenance(digest),
        "capture_input_sha256": capture_provenance["input_sha256"],
        "inputs_identical": digest == capture_provenance["input_sha256"],
        "fixed": fixed,
        "fixed_mismatches": sum(row["mismatches"] for row in fixed.values()),
        "random_cases": random_cases,
        "random_element_comparisons": random_elements,
        "random_mismatches": random_mismatches,
        "first_failure": first_failure,
    }
    (root / "exactness-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["inputs_identical"] or result["fixed_mismatches"] or random_mismatches:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "verify"))
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.receipt)
    else:
        verify(args.receipt)


if __name__ == "__main__":
    main()
