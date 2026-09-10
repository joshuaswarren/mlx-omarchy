#!/usr/bin/env python3
"""Check a bench_matrix run JSON against the canonical digest pins and
print prefill/decode medians. Usage: check_matrix.py <matrix.json> <fork|stock>
Exit 0 iff every measured leg matches the pin set for that driver."""

import json
import sys

FORK_PINS = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "ad964232ee67fecd",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
# Stock 26.1.7 differs only on bf16_long (range-reduction / division gaps,
# receipts/2026-09-10-main-parity-12-matrix).
STOCK_PINS = dict(FORK_PINS)
STOCK_PINS["qwen25-0.5b-bf16:long-decode-128"] = "c5be9207833d2a26"


def main() -> int:
    path, driver = sys.argv[1], sys.argv[2]
    pins = FORK_PINS if driver == "fork" else STOCK_PINS
    data = json.load(open(path))
    legs = data["legs"] if "legs" in data else data.get("results", [])
    ok = True
    for leg in legs:
        leg_id = leg.get("leg_id")
        digest = (leg.get("metrics") or {}).get(
            "generated_ids_sha256_16") or leg.get("generated_ids_sha256_16")
        status = leg.get("status")
        prefill = (leg.get("metrics") or {}).get(
            "prefill_tok_s") or leg.get("prefill_tok_s")
        decode = (leg.get("metrics") or {}).get(
            "decode_tok_s") or leg.get("decode_tok_s")
        if status != "measured":
            # Optional models (7B/14B) are legitimately skipped when absent
            # from the local HF cache; only the pinned 0.5B models gate.
            if "0.5b" in (leg_id or ""):
                print(f"{leg_id} status={status} MISMATCH(no digest)")
                ok = False
            else:
                print(f"{leg_id} status={status} (optional, not gated)")
            continue
        want = pins.get(leg_id)
        match = digest == want
        ok = ok and match
        print(f"{leg_id} digest={digest} pin={want} "
              f"{'OK' if match else 'MISMATCH'} "
              f"prefill={prefill} decode={decode}")
    print("DIGESTS_OK" if ok else "DIGESTS_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
