#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).parent
WINDOW = ROOT / "window-final"
PINS = {"short": "f26175202f3dabe9", "longctx": "ff502900d2a179a5"}

summary = json.loads((WINDOW / "gpu-summary.json").read_text())
identity = summary["identity"]
profiles = summary["profiles"]
comparison = summary["comparison"]
phase_isolation = json.loads((ROOT / "phase-audit.json").read_text())
build_launcher = (ROOT / "build-launcher.log").read_text()
measurement_launcher = (ROOT / "measurement-launcher.log").read_text()
build_log = (ROOT / "build-attempt" / "build.log").read_text()
release_readback = (ROOT / "post-release-readback.txt").read_text()
source = (ROOT / "source-attribution.txt").read_text()
loads = [[float(value) for value in line.split()[1:]]
         for line in (WINDOW / "loadavg.txt").read_text().splitlines()]

assert identity["source_commit"] == "6b1ac0296ba65a8e0075171ca9451e222ddff06b"
assert identity["diagnostic_version"] == "0.32.2.dev202609130459+diag.6b1ac02"
assert identity["diagnostic_wheel_sha256"] == "ff89e8dab256f4b6b282c8bd083d7d1b6b59a3a07fe994498b966d016f85f530"
assert identity["release_wheel_sha256"] == "fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50"
assert identity["profiling_literal_present"] is True
assert identity["device_info"]["device_name"] == "Apple M1 (G13G B1)"
assert identity["model_snapshot"] == "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
for label, digest in PINS.items():
    controls = summary["control"][label]["records"]
    diagnostic = summary["diagnostic"][label]
    assert len(controls) == 3
    assert all(row["generated"] == 32 and row["ids_sha256_16"] == digest
               for row in controls)
    assert diagnostic["generated"] == 32
    assert diagnostic["ids_sha256_16"] == digest
    assert profiles[label]["records"]["dropped"] == 0
    assert profiles[label]["decode"]["intervals"] == 31
    assert profiles[label]["valid_bits"] == 64
    assert profiles[label]["period_ns"] > 0

assert phase_isolation["phase_scope"] == "decode_only"
assert phase_isolation["whole_process_totals_used_for_decode_attribution"] is False
for label in PINS:
    phase = phase_isolation["profiles"][label]
    assert phase["decode_dispatches_included"] == profiles[label]["decode"]["dispatches"]
    assert phase["decode_submit_ids_contiguous"] is True
    assert phase["first_decode_submit_after_first_token_marker_ns"] >= 0
    assert phase["last_decode_submit_before_decode_done_ns"] > 0
    assert phase["pre_decode_gpu_end_to_decode_gpu_start_ns"] >= 0
    assert phase["pre_decode_gpu_work_overlaps_selected_decode"] is False

assert "build_wait_rc=0" in build_launcher
assert "FATAL expected one exact diagnostic wheel, got 0" in build_launcher
assert "Successfully built mlx-omarchy" in build_log
assert "profiling harness: compiled IN" in build_log
assert "sha256: ff89e8dab256f4b6b282c8bd083d7d1b6b59a3a07fe994498b966d016f85f530" in build_log
assert "GPU_PROFILE_WINDOW_VALID" in measurement_launcher
assert "WINDOW_COMPLETE" in measurement_launcher
assert "CHILD_PIDS_CLEAR" in measurement_launcher
assert "lock_released=true" in measurement_launcher
assert json.loads((WINDOW / "device-reopen.json").read_text())["device_reopen"] is True
assert all(f"pid_{pid}=clear" in release_readback
           for pid in (3307403, 3307407, 3307636, 3311970, 3311973, 3311975))
assert release_readback.count("global_lock_free=true") == 2
assert "original_deadline_epoch=1789276458" in (ROOT / "measurement-launch.txt").read_text()
assert "deadline_epoch=1789276458" in (ROOT / "build-launch.txt").read_text()
assert "identical=True" in source
assert max(row[0] for row in loads) < 1.0
for script in ("build_diag.sh", "window.sh", "run_all.sh"):
    assert "pgrep" not in (ROOT / script).read_text()

short = profiles["short"]["decode"]
longctx = profiles["longctx"]["decode"]
release_delta = comparison["release_control_wall_delta_ms_per_token"]
diag_delta = comparison["diagnostic_deltas"]["wall_ms_per_token"]
gpu_busy_delta = comparison["diagnostic_deltas"]["gpu_busy_ms_per_token"]
gpu_gap_delta = comparison["diagnostic_deltas"]["positive_gpu_gap_ms_per_token"]
sdpa_delta = next(
    row["gpu_busy_delta_ms_per_token"]
    for row in comparison["kernel_gpu_busy_deltas"]
    if row["kernel"] == "SdpaDecodeNativeBF16")
other_kernel_delta = gpu_busy_delta - sdpa_delta
short_overhead = (
    short["wall_ms_per_token"]
    - summary["control"]["short"]["wall_ms_per_token"]["median"])
long_overhead = (
    longctx["wall_ms_per_token"]
    - summary["control"]["longctx"]["wall_ms_per_token"]["median"])

verdict = {
    "schema": "bf16-longctx-gpu-cost-attribution/1",
    "execution_model": "openai-codex/gpt-5.6-sol",
    "fallback_model": None,
    "identity": identity,
    "window": {
        "original_total_deadline_epoch": 1789276458,
        "original_total_deadline_utc": "2026-09-13T05:14:18Z",
        "deadline_reset": False,
        "build_lock": {
            "acquired_utc": "2026-09-13T04:59:18Z",
            "released_utc": "2026-09-13T05:04:34Z",
            "result": "diagnostic artifact built; receipt glob rejected its valid seven-character source stamp before measurement",
            "invalid_measurement": True,
        },
        "measurement_lock": {
            "acquired_utc": "2026-09-13T05:06:35Z",
            "released_utc": "2026-09-13T05:07:56Z",
            "quiet_load_1m": [0.25, 0.24, 0.17],
            "run_load_1m_max": max(row[0] for row in loads),
            "child_pids_clear": True,
            "launcher_pids_clear": True,
            "lock_free_post_readback": True,
            "device_reopen": True,
        },
        "build_jobs": 2,
        "persistent_build_tree": "~/src/mlx-bf16-grouped-profile-6b1ac029",
        "tmp_build_used": False,
        "process_detector_added": False,
    },
    "workload": {
        "execution": "eager BF16",
        "env": {"MLX_DISABLE_COMPILE": "1", "HF_HUB_OFFLINE": "1"},
        "tokens": 32,
        "decode_intervals": 31,
        "strict_digests": PINS,
        "control_repetitions": 3,
        "diagnostic_repetitions": 1,
        "provenance_match": True,
        "profiles_dropped_records": 0,
    },
    "phase_isolation": phase_isolation,
    "attribution": {
        "release_control_wall_delta_ms_per_token": release_delta,
        "diagnostic_wall_delta_ms_per_token": diag_delta,
        "instrumentation_differential_delta_ms_per_token": diag_delta - release_delta,
        "diagnostic_short_overhead_vs_control_ms_per_token": short_overhead,
        "diagnostic_longctx_overhead_vs_control_ms_per_token": long_overhead,
        "diagnostic_dispatch_delta_per_token": (
            longctx["dispatches_per_token"] - short["dispatches_per_token"]),
        "diagnostic_submission_delta_per_token": (
            longctx["submissions_per_token"] - short["submissions_per_token"]),
        "diagnostic_gpu_busy_delta_ms_per_token": gpu_busy_delta,
        "diagnostic_positive_gpu_gap_delta_ms_per_token": gpu_gap_delta,
        "sdpa_decode_native_bf16_gpu_busy_delta_ms_per_token": sdpa_delta,
        "all_other_kernels_net_gpu_busy_delta_ms_per_token": other_kernel_delta,
        "sdpa_decode_native_bf16_dispatches_per_token":
            longctx["kernels"]["SdpaDecodeNativeBF16"]["dispatches_per_token"],
        "components_are_not_additive": True,
        "diagnostic_wall_is_not_release_performance": True,
        "interpretation": "The 1024-context route adds 24 SdpaDecodeNativeBF16 dispatches and 11.0815 ms/token of instrumented GPU kernel time. Replaced kernels save 9.8202 ms/token, leaving a 1.2614 ms/token net GPU-busy increase. The diagnostic build reverses the wall delta because the short route executes 216 more instrumented dispatches per token; its timestamps and barriers add 30.3903 ms/token to short versus 18.3457 to long. GPU busy, gaps, submit cost, waits, and wall time overlap and are not summed.",
    },
    "source_attribution": {
        "target": "SdpaDecodeNativeBF16",
        "dispatch_site": "overlay/mlx/backend/omarchy/primitives.cpp:10635-10731",
        "shader": "overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp:99-272",
        "route_gate": "BF16 single-query decode with 256 <= k_len <= 2048",
        "exactness_constraint": "The BF16 arm preserves the decomposed f32 score, softmax, accumulation, and RNE store order.",
        "safe_removal_candidate": False,
        "reason": "The source records that this route already beats the exact decomposed composition at 1K. Disabling it would restore 216 dispatches/token and regress the qualified workload.",
    },
    "outcome": {
        "kind": "attributed_target_no_change",
        "candidate_selected": False,
        "implementation_commit": None,
        "reason": "The trace identifies the only new long-context GPU kernel, but the profiling perturbation is larger than the release wall penalty and the route already replaces a slower exact composition. No numerically safe source edit is supported by this evidence alone.",
        "next_source_investigation": "Evaluate a BF16-only 256-thread composition-exact arm: the current score phase uses 1024 threads, but the max, sum, normalize, and output phases use at most 256 while threads 256..1023 only cross barriers. A 256-thread score loop can retain each key's ascending 64-term f32 accumulation and every later reduction/store order without disabling the exactness route.",
        "next_evidence": "Build that composition-exact SdpaDecodeNativeBF16 shader candidate, prove word identity, then compare it against this release control without per-dispatch timestamp barriers. Retain strict ID digests and the existing k_len boundary.",
    },
    "invalid_attempts": [
        {
            "path": "build-attempt/",
            "reason": "The diagnostic build succeeded, but the receipt expected an eight-character stamp while build-wheel.sh intentionally uses git's seven-character short hash. No measurement ran in this attempt.",
        }
    ],
    "receipts": {
        "raw_profiles": ["window-final/profile-short.jsonl",
                         "window-final/profile-longctx.jsonl"],
        "canonical_profile_reports": [
            "window-final/profile-short.analysis.txt",
            "window-final/profile-longctx.analysis.txt"],
        "paired_summary": "window-final/gpu-summary.json",
        "phase_audit": "phase-audit.json",
        "artifact_reconciliation": "artifact-reconciliation.txt",
        "exact_source": "source-attribution.txt",
        "release_readback": "post-release-readback.txt",
    },
}
(ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps({
    "release_wall_delta_ms_per_token": release_delta,
    "diagnostic_wall_delta_ms_per_token": diag_delta,
    "instrumentation_differential_ms_per_token": diag_delta - release_delta,
    "gpu_busy_delta_ms_per_token": gpu_busy_delta,
    "sdpa_decode_native_bf16_delta_ms_per_token": sdpa_delta,
    "other_kernels_net_delta_ms_per_token": other_kernel_delta,
    "candidate_selected": False,
}, sort_keys=True))
print("GPU_COST_ATTRIBUTION_VALID")
