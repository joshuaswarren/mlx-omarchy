#!/usr/bin/env python3
import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED_PC = {
    "ls", "rs", "os", "lo", "ro", "oo", "m", "n", "k", "f",
    "dims", "shape", "is", "osv", "lg", "rg",
}


def kernel_names(path):
    text = path.read_text()
    body = text.split("enum class ComputeKernel", 1)[1].split("};", 1)[0]
    body = re.sub(r"//.*", "", body.split("{", 1)[1])
    return [part.split("=", 1)[0].strip() for part in body.split(",") if part.strip()]


def classify(scopes, candidates, groups, attempts, multi_count):
    if multi_count:
        return "grouped_kernel_emitted"
    if not any(row["chain"] and row["gemv"] for row in scopes):
        return "fusion_environment_gate"
    accepted = [row for row in candidates if row["result"] == "accepted"]
    if not accepted:
        return "planner_candidate_gate"
    if not groups:
        per_scope = collections.defaultdict(list)
        for row in accepted:
            per_scope[row["sc"]].append(row)
        if any(len(rows) >= 2 for rows in per_scope.values()):
            return "logical_source_identity"
        return "scheduling_lifetime"
    if not attempts:
        return "planned_group_not_dispatched"
    failed = [row for row in attempts if row["result"] != "success"]
    if failed:
        return f"dispatcher_{collections.Counter(row['result'] for row in failed).most_common(1)[0][0]}"
    return "dispatch_success_without_kernel_record"


def analyze(profile, compute_h, source_commit):
    rows = [json.loads(line) for line in profile.read_text().splitlines() if line]
    metas = [row for row in rows if row.get("k") == "meta"]
    if len(metas) != 1:
        raise RuntimeError(f"expected one meta event, got {len(metas)}")
    names = kernel_names(compute_h)
    multi_id = names.index("MatmulVecMultiBF16")
    dispatches = [row for row in rows if row.get("k") == "d"]
    missing_pc = [i for i, row in enumerate(dispatches) if set(row.get("pc", {})) != REQUIRED_PC]
    if missing_pc:
        raise RuntimeError(f"dispatch records missing exact ComputeParams schema: {missing_pc[:8]}")
    scopes = [row for row in rows if row.get("k") == "fg_scope"]
    candidates = [row for row in rows if row.get("k") == "fg_candidate"]
    groups = [row for row in rows if row.get("k") == "fg_group"]
    summaries = [row for row in rows if row.get("k") == "fg_summary"]
    attempts = [row for row in rows if row.get("k") == "fg_dispatch"]
    if not scopes or not summaries:
        raise RuntimeError("diagnostic planner events absent")
    if {row["sc"] for row in summaries} - {row["sc"] for row in scopes}:
        raise RuntimeError("summary references an unknown planner scope")
    accepted = [row for row in candidates if row["result"] == "accepted"]
    for row in accepted:
        required = {"node", "x", "src", "xoff", "srcoff", "xnd", "xsize", "xm", "xk", "xs0", "xs1", "weight", "woff", "wnd", "wk", "wn", "ws0", "ws1"}
        if not required <= row.keys():
            raise RuntimeError(f"accepted candidate lacks logical metadata: {row}")
    multi_count = sum(row.get("e") == multi_id for row in dispatches)
    return {
        "schema": "bf16-gemv-diagnostic/1",
        "execution_model": "openai-codex/gpt-5.6-sol",
        "fallback_model": None,
        "source_commit": source_commit,
        "device": metas[0].get("device"),
        "profile_label": metas[0].get("label"),
        "counts": {
            "scopes": len(scopes),
            "accepted_candidates": len(accepted),
            "candidate_rejections": dict(sorted(collections.Counter(
                row["result"] for row in candidates if row["result"] != "accepted"
            ).items())),
            "planned_groups": len(groups),
            "dispatcher_results": dict(sorted(collections.Counter(
                row["result"] for row in attempts
            ).items())),
            "matmul_vec_multi_bf16": multi_count,
        },
        "classification": classify(scopes, candidates, groups, attempts, multi_count),
        "speedup_claim": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--compute-h", required=True, type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    source_commit = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    result = analyze(args.profile, args.compute_h, source_commit)
    if args.verify:
        if result != json.loads(args.verify.read_text()):
            raise SystemExit("GEMV_DIAGNOSTIC_VERDICT_MISMATCH")
        print("GEMV_DIAGNOSTIC_VERDICT_VALID")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
