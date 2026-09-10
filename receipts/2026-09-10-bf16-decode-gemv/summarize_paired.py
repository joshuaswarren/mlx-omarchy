#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("driver", choices=("fork", "stock"))
    args = ap.parse_args()
    root = Path(args.root)
    samples = {"baseline": {}, "candidate": {}}
    packages = {}
    for pair in range(1, 4):
        for variant in samples:
            data = json.loads((root / f"pair-{pair}-{variant}" / "matrix.json").read_text())
            measured = [leg for leg in data["legs"] if leg["status"] == "measured"]
            if len(measured) != 6:
                raise SystemExit(f"pair {pair} {variant}: expected 6 measured legs, got {len(measured)}")
            packages.setdefault(variant, data["packages"])
            for leg in measured:
                samples[variant].setdefault(leg["leg_id"], []).append(leg["metrics"])

    expected_candidate = {
        "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
        "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
        "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
        "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
        "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    }
    legs = {}
    policy_ok = True
    for leg_id in sorted(samples["baseline"]):
        entry = {}
        for variant in samples:
            rows = samples[variant][leg_id]
            digests = sorted({row["generated_ids_sha256_16"] for row in rows})
            entry[variant] = {
                "decode_tok_s": statistics.median(row["decode_tok_s"] for row in rows),
                "prefill_tok_s": statistics.median(row["prefill_tok_s"] for row in rows),
                "decode_samples": [row["decode_tok_s"] for row in rows],
                "prefill_samples": [row["prefill_tok_s"] for row in rows],
                "digests": digests,
            }
        entry["decode_change_pct"] = 100 * (entry["candidate"]["decode_tok_s"] / entry["baseline"]["decode_tok_s"] - 1)
        entry["prefill_change_pct"] = 100 * (entry["candidate"]["prefill_tok_s"] / entry["baseline"]["prefill_tok_s"] - 1)
        entry["candidate_native_digest"] = entry["candidate"]["digests"] == [expected_candidate[leg_id]]
        policy_ok &= entry["candidate_native_digest"]
        legs[leg_id] = entry

    wheels = {}
    for variant in samples:
        paths = list((Path("receipt-bf16/wheels") / variant).glob("*.whl"))
        if len(paths) != 1:
            raise SystemExit(f"expected one {variant} wheel, got {len(paths)}")
        wheels[variant] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    out = {
        "driver": args.driver,
        "pairs": 3,
        "alternation": ["baseline", "candidate"] * 3,
        "packages": packages,
        "wheel_sha256": wheels,
        "legs": legs,
        "policy_ok": policy_ok,
    }
    (root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
