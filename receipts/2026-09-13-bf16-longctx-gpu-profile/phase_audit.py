#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
WINDOW = ROOT / "window-final"
LABELS = ("short", "longctx")
PHASES = {
    "load_start": "load",
    "prefill_start": "prefill",
    "decode_start": "decode",
    "tok": "decode",
}


def phase_at(markers, timestamp):
    current = None
    for marker in markers:
        if marker["t"] > timestamp:
            break
        current = PHASES.get(marker["p"])
    return current


def audit(label, expected):
    profile = WINDOW / f"profile-{label}.jsonl"
    marker_path = WINDOW / f"markers-{label}.jsonl"
    rows = [json.loads(line) for line in profile.read_text().splitlines()]
    markers = [json.loads(line) for line in marker_path.read_text().splitlines()]
    dispatches = [row for row in rows if row.get("k") == "d"]
    submits = {row["s"]: row for row in rows if row.get("k") == "s"}

    names = [marker["p"] for marker in markers]
    required = ("prefill_start", "prefill_done", "decode_start", "decode_done")
    assert all(names.count(name) == 1 for name in required)
    assert [names.index(name) for name in required] == sorted(
        names.index(name) for name in required)
    token_times = [marker["t"] for marker in markers if marker["p"] == "tok"]
    assert len(token_times) == 32 and token_times == sorted(token_times)

    decode_start = next(marker["t"] for marker in markers
                        if marker["p"] == "decode_start")
    decode_done = next(marker["t"] for marker in markers
                       if marker["p"] == "decode_done")
    first_token = token_times[0]
    assert decode_start <= first_token < decode_done

    for row in dispatches:
        assert row["s"] in submits
        assert row["t1"] >= row["t0"]

    selected = [row for row in dispatches
                if first_token <= submits[row["s"]]["t"] < decode_done]
    selected_by_existing_filter = [
        row for row in dispatches
        if phase_at(markers, submits[row["s"]]["t"]) == "decode"]
    assert selected and selected == selected_by_existing_filter
    assert len(selected) == expected["decode"]["dispatches"]

    selected_submit_ids = sorted({row["s"] for row in selected})
    assert selected_submit_ids == list(
        range(selected_submit_ids[0], selected_submit_ids[-1] + 1))
    assert selected_submit_ids[0] > 0
    before = [row for row in dispatches
              if submits[row["s"]]["t"] < first_token]
    after = [row for row in dispatches
             if submits[row["s"]]["t"] >= decode_done]
    assert before
    gpu_boundary_gap = min(row["t0"] for row in selected) - max(
        row["t1"] for row in before)
    assert gpu_boundary_gap >= 0

    return {
        "whole_process_dispatches": len(dispatches),
        "pre_decode_dispatches_excluded": len(before),
        "decode_dispatches_included": len(selected),
        "post_decode_dispatches_excluded": len(after),
        "decode_submissions_included": len(selected_submit_ids),
        "first_decode_submit_id": selected_submit_ids[0],
        "last_decode_submit_id": selected_submit_ids[-1],
        "decode_submit_ids_contiguous": True,
        "first_decode_submit_after_first_token_marker_ns": (
            submits[selected_submit_ids[0]]["t"] - first_token),
        "last_decode_submit_before_decode_done_ns": (
            decode_done - submits[selected_submit_ids[-1]]["t"]),
        "pre_decode_gpu_end_to_decode_gpu_start_ns": gpu_boundary_gap,
        "pre_decode_gpu_work_overlaps_selected_decode": False,
    }


def main():
    bench_path = ROOT.parent.parent / "scripts" / "bench_decode.py"
    bench_source = bench_path.read_text()
    window_source = (ROOT / "window.sh").read_text()
    warmup = "if args.warmup_tokens:\n        for _ in generate(args.warmup_tokens):"
    measured = "times, n = run_generation(generate(args.tokens)"
    assert "--warmup-tokens 4" in window_source
    assert bench_source.index(warmup) < bench_source.index("t0 = time.monotonic_ns()")
    assert bench_source.index("t0 = time.monotonic_ns()") < bench_source.index(measured)
    summary = json.loads((WINDOW / "gpu-summary.json").read_text())
    profiles = {
        label: audit(label, summary["profiles"][label]) for label in LABELS
    }
    short_dispatches = profiles["short"]["decode_dispatches_included"] / 31
    long_dispatches = profiles["longctx"]["decode_dispatches_included"] / 31
    assert long_dispatches - short_dispatches == -216
    result = {
        "schema": "bf16-longctx-gpu-phase-audit/1",
        "phase_scope": "decode_only",
        "capture_scope": "whole process captured; decode selected by markers",
        "reset_used": False,
        "bench_decode_sha256": hashlib.sha256(bench_source.encode()).hexdigest(),
        "warmup_contract": (
            "window.sh passes --warmup-tokens 4. bench_decode consumes that "
            "iterator before calling the measured run_generation wrapper, so all "
            "warmup submissions precede prefill_start and are excluded."
        ),
        "marker_contract": (
            "The wrapped measured iterator pulls the first generated item before "
            "emitting prefill_done, decode_start, and the first token marker. That "
            "item contains measured-run prefill and first-token work and is excluded. "
            "The selected window covers the following 31 token intervals."
        ),
        "dispatch_filter": (
            "A dispatch is included only when its linked submission monotonic "
            "timestamp is at or after the first token marker and before decode_done."
        ),
        "whole_process_totals_used_for_decode_attribution": False,
        "dispatch_comparison": {
            "short_decode_dispatches_per_token": short_dispatches,
            "longctx_decode_dispatches_per_token": long_dispatches,
            "longctx_minus_short_dispatches_per_token": -216,
            "meaning": (
                "The delta compares only phase-filtered dispatches over the same 31 "
                "steady-state decode intervals: 510 long-context minus 726 short."
            ),
        },
        "cost_categories": {
            "gpu_busy": "sum of selected decode dispatch GPU durations",
            "submit_host": "reported separately; not added to GPU busy",
            "begin_wait": "reported separately; not added to GPU busy",
            "join_wait": "reported separately; not added to GPU busy",
            "wall": "reported separately; not added to asynchronous components",
        },
        "profiles": profiles,
    }
    (ROOT / "phase-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    print("DECODE_PHASE_ISOLATION_VALID")


if __name__ == "__main__":
    main()
