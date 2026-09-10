#!/usr/bin/env python3
"""Shared deterministic input construction for the BF16 GEMV ULP capture.

Imported by m1_ulp_capture.py (GPU cell runs) and m1_ulp_truth.py (CPU f64
truth). Both must see byte-identical inputs; pattern construction is pure
arithmetic over recorded salts, no RNG.
"""
import json
import os
from pathlib import Path

import numpy as np

MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"
CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")


def patterned_bits(count, salt):
    """Same generator as receipts/2026-09-10-bf16-decode-gemv/fixed_bf16.py."""
    out = np.empty(count, dtype=np.uint16)
    chunk = 1 << 20
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        idx = np.arange(start, stop, dtype=np.uint32)
        mixed = idx + np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
        mixed ^= mixed >> np.uint32(16)
        mixed *= np.uint32(0x7FEB352D)
        mixed ^= mixed >> np.uint32(15)
        mixed *= np.uint32(0x846CA68B)
        mixed ^= mixed >> np.uint32(16)
        mantissa = (mixed >> np.uint32(9)) & np.uint32(0x7F)
        exponent = np.uint32(120) + ((mixed >> np.uint32(1)) & np.uint32(7))
        sign = (mixed & np.uint32(1)) << np.uint32(15)
        out[start:stop] = (sign | (exponent << np.uint32(7)) | mantissa).astype(np.uint16)
    return out


def to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def to_f64(bits):
    return to_f32(bits).astype(np.float64)


def rne_bf16(a):
    """float64/float32 array -> RNE bf16 bits (root-cause receipt convention)."""
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def capture_tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    return np.asarray(np.load(CAPTURE / meta["path"]), dtype=np.uint16)


def input_hash(*arrays):
    h = __import__("hashlib").sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def synthetic_cancel(k, n, salt, tail=128):
    """Patterned W whose f64 dot with the patterned x is deep-cancellation:
    partials O(0.1..40), exact sum ~1e-6 of the partial scale. Last `tail`
    columns of each row are set (in f64, then rounded to bf16) to cancel the
    leading partial; K stays a multiple of 128."""
    x = patterned_bits(k, 11).astype(np.float64)
    w = patterned_bits(k * n, salt).astype(np.float64).reshape(n, k)
    head = (k - tail) // 4 * 4
    partial = w[:, :head] @ x[:head]
    v = x[head:]
    v = v / (v @ v)
    w[:, head:] = np.outer(-partial * 1e-6, v)
    xb = np.asarray(patterned_bits(k, 11))
    wb = f64_to_bf16_bits(w)
    return xb, wb, partial


def f64_to_bf16_bits(a):
    return rne_bf16(a)
