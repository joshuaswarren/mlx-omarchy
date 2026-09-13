#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path


def analyze(profile, markers, decode_log):
    rows = [json.loads(line) for line in profile.read_text().splitlines() if line]
    marker_rows = [json.loads(line) for line in markers.read_text().splitlines() if line]
    decode_rows = [
        json.loads(line)
        for line in decode_log.read_text().splitlines()
        if line.startswith("{") and "ids_sha256_16" in line
    ]
    scopes = [row for row in rows if row.get("k") == "fg_scope"]
    summaries = [row for row in rows if row.get("k") == "fg_summary"]
    candidate_rejections = collections.Counter(
        row["result"]
        for row in rows
        if row.get("k") == "fg_candidate" and row["result"] != "accepted"
    )
    accepted = {}
    pending = collections.defaultdict(collections.deque)
    paired = []
    for row in rows:
        kind = row.get("k")
        if kind == "fg_candidate" and row.get("result") == "accepted":
            accepted[(row["sc"], row["node"])] = row
        elif kind == "fg_group":
            members = [
                candidate
                for (scope, _), candidate in accepted.items()
                if scope == row["sc"] and candidate["src"] == row["src"]
            ]
            if len(members) != row["members"]:
                raise RuntimeError(f"group member mismatch: {row}")
            pending[row["first"]].append((row, members))
        elif kind == "fg_dispatch":
            if not pending[row["first"]]:
                raise RuntimeError(f"dispatcher attempt lacks planned group: {row}")
            group, members = pending[row["first"]].popleft()
            paired.append((group, members, row))
    if any(pending.values()):
        raise RuntimeError("planned groups remain unmatched")

    effective_scope_gates = collections.Counter(
        (row["chain"], row["gemv"], row["chain_env"], row["gemv_env"])
        for row in scopes
    )
    summary_shapes = collections.Counter(
        (row["candidates"], row["groups"]) for row in summaries
    )
    group_sizes = collections.Counter(group["members"] for group, _, _ in paired)
    dispatch_results = collections.Counter(attempt["result"] for _, _, attempt in paired)
    input_contracts = collections.Counter(
        (
            member["xnd"], member["xm"], member["xk"], member["xsize"],
            member["xs0"], member["xs1"], member["xoff"], member["srcoff"],
        )
        for _, members, _ in paired
        for member in members
    )
    weight_contracts = collections.Counter(
        (
            member["wnd"], member["wk"], member["wn"],
            member["ws0"], member["ws1"], member["woff"],
        )
        for _, members, _ in paired
        for member in members
    )
    source_relations = collections.Counter(
        (
            attempt["x"] == group["src"],
            all(member["x"] != group["src"] for member in members),
            len({member["x"] for member in members}) == len(members),
        )
        for group, members, attempt in paired
    )
    token_digests = [row["ids_sha256_16"] for row in decode_rows]
    return {
        "schema": "bf16-gemv-hardware-root-cause/1",
        "execution_model": "openai-codex/gpt-5.6-sol",
        "fallback_model": None,
        "source_commit": "3e26a6a7f373303f6ae417936edc2f57ab372264",
        "device": next(row["device"] for row in rows if row.get("k") == "meta"),
        "effective_scope_gates": [
            {"chain": key[0], "gemv": key[1], "chain_env": key[2], "gemv_env": key[3], "count": count}
            for key, count in sorted(effective_scope_gates.items())
        ],
        "scope_summaries": [
            {"candidates": key[0], "groups": key[1], "count": count}
            for key, count in sorted(summary_shapes.items())
        ],
        "candidate_rejections": dict(sorted(candidate_rejections.items())),
        "planned_dispatch_pairs": len(paired),
        "group_sizes": {str(key): value for key, value in sorted(group_sizes.items())},
        "dispatcher_results": dict(sorted(dispatch_results.items())),
        "input_contracts": [
            {
                "ndim": key[0], "m": key[1], "k": key[2], "size": key[3],
                "stride0": key[4], "stride1": key[5], "offset": key[6],
                "source_offset": key[7], "count": count,
            }
            for key, count in sorted(input_contracts.items())
        ],
        "weight_contracts": [
            {
                "ndim": key[0], "k": key[1], "n": key[2],
                "stride0": key[3], "stride1": key[4], "offset": key[5],
                "count": count,
            }
            for key, count in sorted(weight_contracts.items())
        ],
        "source_relations": [
            {
                "dispatcher_uses_group_source": key[0],
                "member_ids_differ_from_source": key[1],
                "member_ids_are_distinct": key[2],
                "count": count,
            }
            for key, count in sorted(source_relations.items())
        ],
        "matmul_vec_multi_bf16_dispatches": 0,
        "generated_ids_sha256_16": token_digests,
        "token_markers": sum(row.get("p") == "tok" for row in marker_rows),
        "classification": "dispatcher_rejects_planned_flatten_views",
        "mechanism": "The planner groups distinct Flatten views by their shared pre-Flatten source and passes that source to the dispatcher. The dispatcher then rejects the unevaluated view ids because their materialized buffer identity is not yet available.",
        "speedup_claim": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--markers", required=True, type=Path)
    parser.add_argument("--decode-log", required=True, type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = analyze(args.profile, args.markers, args.decode_log)
    if args.verify:
        if result != json.loads(args.verify.read_text()):
            raise SystemExit("GEMV_HARDWARE_ROOT_CAUSE_MISMATCH")
        print("GEMV_HARDWARE_ROOT_CAUSE_VALID")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
