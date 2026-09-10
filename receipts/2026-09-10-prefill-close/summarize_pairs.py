#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = sorted(args.directory.glob("pair-*-*.json"))
    if not runs:
        raise SystemExit("no pair JSON files")

    legs = {}
    provenance = {}
    for path in runs:
        side = path.stem.rsplit("-", 1)[1]
        if side not in ("base", "cand"):
            raise SystemExit(f"unexpected side in {path.name}")
        data = json.loads(path.read_text())
        provenance.setdefault(side, {
            "packages": data.get("packages"),
            "wheel": data.get("wheel"),
            "host": data.get("host"),
        })
        measured = [leg for leg in data["legs"] if leg.get("measured")]
        if len(measured) != 6:
            raise SystemExit(f"{path.name}: expected 6 measured legs, got {len(measured)}")
        for leg in measured:
            metrics = leg["metrics"]
            entry = legs.setdefault(leg["leg_id"], {
                "base": [], "cand": [], "digests": {"base": set(), "cand": set()}})
            entry[side].append({
                "prefill": metrics["prefill_tok_s"],
                "decode": metrics["decode_tok_s"],
            })
            entry["digests"][side].add(metrics["generated_ids_sha256_16"])

    summary = {"provenance": provenance, "legs": {}, "policy_ok": True}
    for leg_id, entry in sorted(legs.items()):
        if len(entry["base"]) != 3 or len(entry["cand"]) != 3:
            raise SystemExit(f"{leg_id}: expected three samples per side")
        baseline_prefill = statistics.median(x["prefill"] for x in entry["base"])
        candidate_prefill = statistics.median(x["prefill"] for x in entry["cand"])
        baseline_decode = statistics.median(x["decode"] for x in entry["base"])
        candidate_decode = statistics.median(x["decode"] for x in entry["cand"])
        prefill_ratio = candidate_prefill / baseline_prefill
        decode_ratio = candidate_decode / baseline_decode
        base_digests = sorted(entry["digests"]["base"])
        cand_digests = sorted(entry["digests"]["cand"])
        ids_same = base_digests == cand_digests and len(base_digests) == 1
        leg_ok = prefill_ratio >= 0.99 and decode_ratio >= 0.99 and ids_same
        summary["policy_ok"] &= leg_ok
        summary["legs"][leg_id] = {
            "baseline_prefill": baseline_prefill,
            "candidate_prefill": candidate_prefill,
            "prefill_ratio": prefill_ratio,
            "baseline_decode": baseline_decode,
            "candidate_decode": candidate_decode,
            "decode_ratio": decode_ratio,
            "base_digests": base_digests,
            "candidate_digests": cand_digests,
            "ids_same": ids_same,
            "policy_ok": leg_ok,
            "samples": {"base": entry["base"], "cand": entry["cand"]},
        }

    if len(summary["legs"]) != 6:
        raise SystemExit(f"expected six legs, got {len(summary['legs'])}")
    output = args.directory / "paired-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "policy_ok": summary["policy_ok"],
        "legs": {
            key: {
                "prefill_ratio": value["prefill_ratio"],
                "decode_ratio": value["decode_ratio"],
                "ids_same": value["ids_same"],
            }
            for key, value in summary["legs"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
