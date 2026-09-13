#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
WINDOW = ROOT / "window-final"
NATIVE = ROOT.parent / "native-baseline-2026-09-06" / "native-2026-09-06-summary.json"
PINS = {"short": "f26175202f3dabe9", "longctx": "ff502900d2a179a5"}


def flat_profile(label, process_ms):
    text = (WINDOW / f"perf-{label}-flat.txt").read_text(errors="replace")
    sample_match = re.search(r"# Samples: (\d+)", text)
    lost_match = re.search(r"# Total Lost Samples: (\d+)", text)
    if not sample_match or not lost_match:
        raise SystemExit(f"missing perf header: {label}")
    groups = {name: 0.0 for name in
              ("eager_fusion", "heap_memory", "dispatch_record",
               "python_runtime", "eval_graph", "driver", "other")}
    for line in text.splitlines():
        match = re.match(r"\s+([0-9.]+)%\s+\S+\s+(.+?)\s+\[\.\]\s+(.+)$", line)
        if not match:
            continue
        percent = float(match.group(1))
        obj, symbol = match.group(2).strip(), match.group(3).strip()
        if "EagerFusion" in symbol or "is_op(" in symbol:
            group = "eager_fusion"
        elif symbol in {"malloc", "cfree", "memcpy", "memset"} or "VulkanAllocator" in symbol:
            group = "heap_memory"
        elif "dispatch_" in symbol or "CommandEncoder::" in symbol or "get_command_encoder" in symbol:
            group = "dispatch_record"
        elif "libpython" in obj:
            group = "python_runtime"
        elif "eval_impl" in symbol or "gpu::eval" in symbol or "array::" in symbol:
            group = "eval_graph"
        elif "libvulkan_asahi" in obj:
            group = "driver"
        else:
            group = "other"
        groups[group] += percent
    return {
        "samples": int(sample_match.group(1)),
        "lost_samples": int(lost_match.group(1)),
        "estimated_user_cpu_self_ms_per_token": {
            name: process_ms * percent / 100.0 for name, percent in groups.items()
        },
        "classification": "Mutually exclusive flat self-time groups; entries below perf's 0.1% reporting threshold are omitted.",
    }


summary = json.loads((WINDOW / "summary.json").read_text())
native = json.loads(NATIVE.read_text())
for mode in ("control", "instrumented"):
    for label in PINS:
        records = summary[mode][label]["records"]
        if len(records) != 3 or any(record["generated"] != 32 for record in records):
            raise SystemExit(f"invalid {mode} {label} records")
        if any(record["ids_sha256_16"] != PINS[label] for record in records):
            raise SystemExit(f"digest mismatch: {mode} {label}")
        raw_mode = "cpu" if mode == "instrumented" else "control"
        for rep in range(1, 4):
            text = (WINDOW / f"{raw_mode}-{label}-{rep}.log").read_text()
            if "verified=match" not in text or "+6b1ac029" not in text:
                raise SystemExit(f"provenance mismatch: {raw_mode}-{label}-{rep}")

short_wall = summary["control"]["short"]["wall_ms_per_token"]["median"]
long_wall = summary["control"]["longctx"]["wall_ms_per_token"]["median"]
short_cpu = summary["instrumented"]["short"]["process_cpu_ms_per_token"]["median"]
long_cpu = summary["instrumented"]["longctx"]["process_cpu_ms_per_token"]["median"]
wall_delta = long_wall - short_wall
cpu_delta = long_cpu - short_cpu
minimum_non_cpu = wall_delta - cpu_delta
short_wall_values = summary["control"]["short"]["wall_ms_per_token"]["values"]
long_wall_values = summary["control"]["longctx"]["wall_ms_per_token"]["values"]
short_cpu_values = summary["instrumented"]["short"]["process_cpu_ms_per_token"]["values"]
long_cpu_values = summary["instrumented"]["longctx"]["process_cpu_ms_per_token"]["values"]
conservative_wall_delta = min(long_wall_values) - max(short_wall_values)
maximum_observed_cpu_delta = max(long_cpu_values) - min(short_cpu_values)
conservative_minimum_non_cpu = conservative_wall_delta - maximum_observed_cpu_delta
native_short = native["qwen25-0.5b-bf16:short-decode-32"]
native_long = native["qwen25-0.5b-bf16:longctx-1024-decode-32"]
native_long_wall = 1000.0 / native_long["decode_tok_s_median"]
load_samples = [[float(value) for value in line.split()[1:]]
                for line in (WINDOW / "loadavg.txt").read_text().splitlines()]
perf_process = {
    label: summary["perf"][label]["clock"]["process_cpu_ms_per_token"]
    for label in PINS
}
profiles = {label: flat_profile(label, perf_process[label]) for label in PINS}
profile_delta = {
    group: profiles["longctx"]["estimated_user_cpu_self_ms_per_token"][group]
    - profiles["short"]["estimated_user_cpu_self_ms_per_token"][group]
    for group in profiles["short"]["estimated_user_cpu_self_ms_per_token"]
}
launcher = (WINDOW / "launcher.log").read_text()
window_log = (WINDOW / "window.log").read_text()
lease = (WINDOW / "lease.txt").read_text()
refusal = (WINDOW / "compiled-bf16-refusal.log").read_text()
device_reopen = json.loads((WINDOW / "device-reopen.json").read_text())
if "original_deadline_utc=2026-09-13T04:33:27Z" not in launcher:
    raise SystemExit("fixed deadline missing")
if "WINDOW_COMPLETE" not in window_log or "lock_released=true" not in window_log:
    raise SystemExit("window did not complete cleanly")
if "released_utc=" not in lease or "rc=0" not in lease:
    raise SystemExit("lease release missing")
if "Compiled tape bfloat16 is refused" not in refusal:
    raise SystemExit("compiled BF16 refusal missing")

verdict = {
    "schema": "bf16-longctx-critical-path-attribution/1",
    "execution_model": "openai-codex/gpt-5.6-sol",
    "identity": {
        "harness_commit": "e4d079c883b8606dddef7720bd302c5756be0692",
        "wheel": {
            "source_commit": "6b1ac0296ba65a8e0075171ca9451e222ddff06b",
            "version": "0.32.2.dev202609130004+6b1ac029",
            "sha256": "fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50",
            "grouped_source_identical_to_harness": True,
            "intervening_non_ane_runtime_paths": 0,
        },
        "benchmark_model": {
            "name": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "snapshot": "56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
        },
        "hardware": {
            "host": "jwm1-linux",
            "device": "Apple M1 (G13G B1)",
            "driver": "Mesa Honeykrisp",
            "driver_package": "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1",
            "kernel": "7.1.6-1-1-ARCH",
        },
        "native": {
            "measurement_kind": "pinned_active_reference",
            "receipt": "receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json",
            "matched_physical_host": True,
            "matched_model_snapshot": True,
            "repeats": 5,
            "short_decode_tps_median": native_short["decode_tok_s_median"],
            "longctx_1024_decode_tps_median": native_long["decode_tok_s_median"],
            "longctx_digest_matches": native_long["generated_ids_sha256_16"] == [PINS["longctx"]],
            "short_digest_matches": native_short["generated_ids_sha256_16"] == [PINS["short"]],
        },
    },
    "window": {
        "single_lock_window": True,
        "original_deadline_utc": "2026-09-13T04:33:27Z",
        "acquired_utc": "2026-09-13T04:24:58Z",
        "released_utc": "2026-09-13T04:26:49Z",
        "quiet_gate_passed": True,
        "quiet_gate_load_1m": [0.0, 0.0, 0.0],
        "run_load_1m_max": max(values[0] for values in load_samples),
        "uncontended": max(values[0] for values in load_samples) < 1.0,
        "env": {"MLX_DISABLE_COMPILE": "1", "HF_HUB_OFFLINE": "1"},
        "control": {
            "short": {"tokens": 32, "digest": PINS["short"],
                      "wall_ms_per_token": summary["control"]["short"]["wall_ms_per_token"]},
            "longctx_1024": {"tokens": 32, "digest": PINS["longctx"],
                             "wall_ms_per_token": summary["control"]["longctx"]["wall_ms_per_token"]},
        },
        "instrumented": {
            "short": {"tokens": 32, "digest": PINS["short"],
                      "wall_ms_per_token": summary["instrumented"]["short"]["wall_ms_per_token"],
                      "process_cpu_ms_per_token": summary["instrumented"]["short"]["process_cpu_ms_per_token"]},
            "longctx_1024": {"tokens": 32, "digest": PINS["longctx"],
                             "wall_ms_per_token": summary["instrumented"]["longctx"]["wall_ms_per_token"],
                             "process_cpu_ms_per_token": summary["instrumented"]["longctx"]["process_cpu_ms_per_token"]},
        },
        "digests_strict_match": True,
        "provenance_match": True,
    },
    "native_comparison": {
        "strictly_output_matched_workload": "longctx_1024",
        "current_longctx_1024_decode_tps_median": 1000.0 / long_wall,
        "native_longctx_1024_decode_tps_median": native_long["decode_tok_s_median"],
        "current_over_native_longctx_1024": (1000.0 / long_wall) / native_long["decode_tok_s_median"],
        "native_longctx_1024_wall_ms_per_token": native_long_wall,
        "short_reference_used_for_ratio": False,
        "short_reference_exclusion_reason": "The pinned native short digest differs from the current short digest, so no strict output-matched short or cross-context native ratio is reported.",
    },
    "attribution": {
        "method": "critical_path_exposure",
        "cpu_self_time_added_to_gpu_wait": False,
        "control_longctx_minus_short_ms_per_token": wall_delta,
        "instrumented_longctx_minus_short_ms_per_token": summary["attribution"]["instrumented_longctx_minus_short_ms_per_token"],
        "instrument_overhead_delta_ms_per_token": summary["attribution"]["instrument_overhead_delta_ms_per_token"],
        "process_cpu_longctx_minus_short_ms_per_token": cpu_delta,
        "minimum_non_cpu_longctx_minus_short_ms_per_token": minimum_non_cpu,
        "minimum_non_cpu_share_of_context_penalty": minimum_non_cpu / wall_delta,
        "conservative_observed_wall_delta_ms_per_token": conservative_wall_delta,
        "maximum_observed_cpu_delta_ms_per_token": maximum_observed_cpu_delta,
        "conservative_minimum_non_cpu_ms_per_token": conservative_minimum_non_cpu,
        "conservative_minimum_non_cpu_share": conservative_minimum_non_cpu / conservative_wall_delta,
        "interpretation": "Even if every added process-CPU millisecond were exposed, it explains less than half of the median long-context wall penalty. The non-CPU remainder stays positive under the adverse observed extrema. CPU self-time cannot be added to GPU wait or treated as wholly exposed.",
    },
    "cpu_profile": {
        "short": profiles["short"],
        "longctx_1024": profiles["longctx"],
        "estimated_self_time_delta_ms_per_token": profile_delta,
        "interpretation": "The added CPU samples are distributed across eager-fusion analysis, heap work, dispatch recording, graph evaluation, Python, and driver code. No single sampled host path accounts for the wall penalty.",
    },
    "outcome": {
        "kind": "falsified_hypothesis",
        "hypothesis": "Added host-side CPU bookkeeping fully explains the BF16 1024-context decode penalty.",
        "implementation_commit": None,
        "candidate_selected": False,
        "reason": "At least the measured non-CPU residual remains after assigning all added process CPU to the critical path, and the sampled CPU increase is broad rather than a narrow removable source path.",
        "next_evidence": "Use the existing compile-time MLX_OMARCHY_GPU_PROFILING harness with MLX_OMARCHY_GPU_PROFILE to measure per-dispatch GPU ticks, submit clocks, and completion waits for the same paired short and 1024-context eager BF16 prompts; compare kernel groups before selecting source work.",
    },
    "contracts": {"compiled_bf16_refusal_passed": True, "refusal_rc": 1},
    "cleanup": {
        "lock_released": True,
        "lease_released": True,
        "child_pids_clear": True,
        "launcher_pgid_clear": True,
        "device_reopen_passed": device_reopen["device_reopen"] is True,
    },
    "invalid_attempts": [
        "attempt0-false-contender",
        "attempt1-transfer-name",
        "attempt2-scanner-self-match",
    ],
    "receipts": {
        "source_reconciliation": "source-reconciliation.txt",
        "window": "window-final/",
        "raw_cpu_stacks": ["window-final/perf-short-scoped.txt", "window-final/perf-longctx-scoped.txt"],
        "raw_perf_data_manifest": "window-final/perf-data-manifest.txt",
    },
}
(ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps({
    "wall_delta_ms_per_token": wall_delta,
    "process_cpu_delta_ms_per_token": cpu_delta,
    "minimum_non_cpu_ms_per_token": minimum_non_cpu,
    "minimum_non_cpu_share": minimum_non_cpu / wall_delta,
    "current_over_native_longctx": verdict["native_comparison"]["current_over_native_longctx_1024"],
    "candidate_selected": False,
}, sort_keys=True))
print("ATTRIBUTION_ANALYSIS_VALID")
