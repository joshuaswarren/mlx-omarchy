#!/usr/bin/env python3
"""Capture one deterministic BF16 GEMV result as raw BF16 bits."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import mlx.core as mx
import numpy as np


def patterned_bits(count, salt):
    out = np.empty(count, dtype=np.uint16)
    chunk = 1 << 20
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        idx = np.arange(start, stop, dtype=np.uint32)
        mixed = idx + np.uint32((salt * 0x9e3779b9) & 0xffffffff)
        mixed ^= mixed >> np.uint32(16)
        mixed *= np.uint32(0x7feb352d)
        mixed ^= mixed >> np.uint32(15)
        mixed *= np.uint32(0x846ca68b)
        mixed ^= mixed >> np.uint32(16)
        mantissa = (mixed >> np.uint32(9)) & np.uint32(0x7f)
        exponent = np.uint32(120) + ((mixed >> np.uint32(1)) & np.uint32(7))
        sign = (mixed & np.uint32(1)) << np.uint32(15)
        out[start:stop] = (sign | (exponent << np.uint32(7)) | mantissa).astype(np.uint16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    x_bits = [patterned_bits(args.k, 11 + trial * 101).reshape(1, args.k)
              for trial in range(args.trials)]
    w_bits = patterned_bits(args.k * args.n, 23).reshape(args.n, args.k)
    x_bits[0].fill(np.uint16(0x3f80))
    w_bits[:4].fill(np.uint16(0))
    w_bits[:4, 0] = np.uint16(0x3f80)
    w_bits[:4, 4] = np.uint16(0x3380)
    w_bits[:4, 8] = np.uint16(0xbf80)
    digest = hashlib.sha256(w_bits.tobytes())
    for bits in x_bits:
        digest.update(bits.tobytes())
    input_hash = digest.hexdigest()
    w = mx.array(w_bits).view(mx.bfloat16)
    outputs = [mx.array(bits).view(mx.bfloat16) @ w.T for bits in x_bits]
    mx.eval(*outputs)
    result = np.stack([np.array(output.view(mx.uint16)) for output in outputs])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, result=result)
    meta = {
        "shape": {"trials": args.trials, "m": 1, "k": args.k, "n": args.n},
        "dtype": "bfloat16",
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "input_sha256": input_hash,
        "result_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "result_elements": int(result.size),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
