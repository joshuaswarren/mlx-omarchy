#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import itertools
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILE_ROOT = HERE.parent / "2026-09-13-bf16-longctx-gpu-profile" / "window-final"
PROFILE = PROFILE_ROOT / "profile-longctx.jsonl"
MARKERS = PROFILE_ROOT / "markers-longctx.jsonl"
COMPUTE = PROFILE_ROOT / "compute.h"
SOURCE_COMMIT = "6b1ac0296ba65a8e0075171ca9451e222ddff06b"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(path):
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=REPO
    )


def kernel_names():
    spec = importlib.util.spec_from_file_location(
        "gpu_profile_receipt", PROFILE_ROOT / "gpu_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel_names(COMPUTE)


def measured_pairs():
    names = kernel_names()
    matvec_id = names.index("MatmulVecBF16")
    multi_id = names.index("MatmulVecMultiBF16")
    rows = [json.loads(line) for line in PROFILE.read_text().splitlines() if line]
    submissions = {row["s"]: row for row in rows if row.get("k") == "s"}
    token_times = [
        row["t"]
        for row in map(json.loads, MARKERS.read_text().splitlines())
        if row.get("p") == "tok"
    ]
    intervals = len(token_times) - 1
    pairs = []
    counts = []
    for start, end in zip(token_times, token_times[1:]):
        dispatches = [
            row
            for row in rows
            if row.get("k") == "d"
            and row["e"] == matvec_id
            and start <= submissions[row["s"]]["t"] < end
        ]
        counts.append(len(dispatches))
        for _, group in itertools.groupby(
            dispatches, key=lambda row: tuple(row["b"][0])
        ):
            group = list(group)
            if len(group) == 2:
                pairs.append(group)
    return rows, submissions, intervals, counts, pairs, multi_id


def analyze():
    fused = git_blob("overlay/mlx/backend/omarchy/fused_chain.cpp")
    primitives = git_blob("overlay/mlx/backend/omarchy/primitives.cpp")
    profiler = git_blob("overlay/mlx/backend/omarchy/gpu_profiler.h")
    required_source = [
        b"dense_by_x.try_emplace(source.id())",
        b"x.shape(-2) != 1",
        b"x.size() != static_cast<size_t>(x.shape(-1))",
        b"weight.strides()[weight.ndim() - 2] != 1",
        b"weight.strides().back() != k",
        b"!input_ready(weight, stream)",
        b"(x_offset & 3u) != 0u",
        b"(weight_offset & 3u) != 0u",
    ]
    if any(item not in fused + primitives for item in required_source):
        raise RuntimeError("exact source no longer contains an expected eligibility gate")
    rows, submissions, intervals, counts, pairs, multi_id = measured_pairs()
    if intervals != 31 or counts != [97] * intervals or len(pairs) != 24 * intervals:
        raise RuntimeError("measured dispatch census changed")

    member_shapes = set()
    same_submission = 0
    exact_weight_ranges = 0
    for first, second in pairs:
        if first["s"] == second["s"]:
            same_submission += 1
        for row in (first, second):
            if len(row["b"]) != 4:
                raise RuntimeError("MatmulVecBF16 binding count changed")
            n = row["n"]
            inferred_k, remainder = divmod(row["b"][1][2], n * 2)
            member_shapes.add((n, row["gx"], row["gy"], row["gz"], inferred_k))
            exact_weight_ranges += remainder == 0

    expected_shape = {(4864, 1216, 1, 1, 896)}
    if member_shapes != expected_shape:
        raise RuntimeError(f"unexpected pair member shapes: {member_shapes}")

    return {
        "schema": "bf16-gemv-planner-eligibility/1",
        "execution_model": "openai-codex/gpt-5.6-sol",
        "fallback_model": None,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": {
            "fused_chain.cpp": sha256(fused),
            "primitives.cpp": sha256(primitives),
            "gpu_profiler.h": sha256(profiler),
        },
        "trace_sha256": {
            "profile-longctx.jsonl": sha256(PROFILE.read_bytes()),
            "markers-longctx.jsonl": sha256(MARKERS.read_bytes()),
            "compute.h": sha256(COMPUTE.read_bytes()),
        },
        "measured_pair_census": {
            "decode_intervals": intervals,
            "matmul_vec_dispatches_each_interval": counts,
            "adjacent_same_physical_input_binding_pairs": len(pairs),
            "pairs_per_token": len(pairs) / intervals,
            "pairs_in_same_submission": same_submission,
            "member_shape_records": [
                {
                    "output_elements": n,
                    "groups_x": gx,
                    "groups_y": gy,
                    "groups_z": gz,
                    "k_inferred_from_exact_weight_range": k,
                }
                for n, gx, gy, gz, k in sorted(member_shapes)
            ],
            "members_with_weight_range_exactly_n_times_k_times_two":
                exact_weight_ranges,
            "matmul_vec_multi_dispatches_whole_profile": sum(
                row.get("k") == "d" and row.get("e") == multi_id for row in rows
            ),
        },
        "eligibility_matrix": [
            {
                "fact": "dtype",
                "status": "proven",
                "value": "bfloat16 inputs and output",
                "basis": "The MatmulVecBF16 route is selected only for bfloat16 output after both inputs pass same-dtype validation.",
            },
            {
                "fact": "matrix_m_and_batch_count",
                "status": "proven",
                "value": "matrix_m=1 and one batch",
                "basis": "MatmulVecBF16 requires matrix_m=1; every pair member recorded groups_z=1.",
            },
            {
                "fact": "output_width",
                "status": "proven",
                "value": 4864,
                "basis": "Every member recorded count=4864 and groups_x=1216; the route uses ceil(matrix_n/4), and batch count is one.",
            },
            {
                "fact": "matrix_k",
                "status": "strongly_inferred",
                "value": 896,
                "basis": "Every weight descriptor range is exactly 4864*896*2 bytes. The trace does not record logical matrix_k directly.",
            },
            {
                "fact": "single_dispatch_layout_and_alignment",
                "status": "proven",
                "value": "B transposed, A not transposed, alpha=1, no C, K multiple 128, N multiple 4; lhs/rhs offsets, rhs gap, and batch strides four-element aligned",
                "basis": "These are necessary conditions of the exact MatmulVecBF16 source branch that emitted each record.",
            },
            {
                "fact": "group_weight_stride",
                "status": "partially_proven",
                "value": "stride[-2]=1 proven; stride[-1]==K unknown",
                "basis": "The single route's B-transposed classification proves stride[-2]=1 for K=896, but the trace omits rhs_gap and cannot prove the group route's exact stride[-1]==K gate.",
            },
            {
                "fact": "logical_offsets",
                "status": "partially_proven",
                "value": "four-element alignment proven; exact lhs and rhs offsets unknown",
                "basis": "The emitting branch checks alignment but the existing trace stores descriptor offsets, which are zero, not ComputeParams logical offsets.",
            },
            {
                "fact": "logical_input_identity",
                "status": "unknown",
                "value": None,
                "basis": "All pairs reuse one physical input binding, but the planner groups dense_gemv_source(x).id(); array ids are absent from the trace.",
            },
            {
                "fact": "scheduling_lifetime",
                "status": "unknown",
                "value": None,
                "basis": "All pairs are adjacent and in one Vulkan submission, but submissions can contain multiple eval tapes. The trace omits eager-scope boundaries and input_ready state at the first member.",
            },
            {
                "fact": "fusion_environment_gates",
                "status": "unknown",
                "value": None,
                "basis": "The measurement harness neither clears nor records MLX_OMARCHY_FUSED_CHAIN and MLX_OMARCHY_FUSED_GEMV.",
            },
        ],
        "verdict": {
            "planner_eligibility_established": False,
            "finding": "The measured pairs satisfy the single-kernel shape, dtype, layout-class, capability, and alignment gates. Existing evidence cannot prove exact rhs stride, logical offsets, array identity, same eager tape, readiness, or fusion environment gates.",
            "next_action": "Use the compile-time-only diagnostic patch to add ComputeParams and dense-group planner events to the existing GPU profile stream before making a source change.",
            "speedup_claim": None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = analyze()
    if args.verify:
        if result != json.loads(args.verify.read_text()):
            raise SystemExit("GEMV_ELIGIBILITY_VERDICT_MISMATCH")
        print("GEMV_ELIGIBILITY_VERDICT_VALID")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
