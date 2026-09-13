#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpu_profile import summarize

PINS = {"short": "f26175202f3dabe9", "longctx": "ff502900d2a179a5"}


def engine_record(path, version_tag):
    text = path.read_text()
    records = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("engine") == "bench_decode":
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"expected one bench_decode record in {path}")
    record = records[0]
    if "verified=match" not in text or version_tag not in text:
        raise ValueError(f"provenance mismatch in {path}")
    return record


def main():
    root = Path(sys.argv[1]).resolve()
    identity = json.loads((root / "identity.json").read_text())
    if identity["source_commit"] != "6b1ac0296ba65a8e0075171ca9451e222ddff06b":
        raise ValueError("diagnostic source commit mismatch")
    if identity["release_wheel_sha256"] != "fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50":
        raise ValueError("release wheel digest mismatch")
    if "+diag.6b1ac02" not in identity["diagnostic_version"]:
        raise ValueError("diagnostic version is not source-stamped")
    if not identity["profiling_literal_present"]:
        raise ValueError("diagnostic wheel lacks the profiling harness")

    controls = {}
    diagnostic = {}
    profiles = {}
    for label, digest in PINS.items():
        control_records = [
            engine_record(root / f"control-{label}-{repeat}.log", "+6b1ac029")
            for repeat in range(1, 4)
        ]
        diag_record = engine_record(root / f"diag-{label}.log",
                                    "+diag.6b1ac02")
        for record in [*control_records, diag_record]:
            if record.get("generated") != 32 or record.get("ids_sha256_16") != digest:
                raise ValueError(f"strict output mismatch for {label}")
        controls[label] = {
            "records": control_records,
            "wall_ms_per_token": {
                "values": [1000.0 / row["decode_tps"] for row in control_records],
                "median": statistics.median(
                    1000.0 / row["decode_tps"] for row in control_records),
            },
        }
        diagnostic[label] = diag_record
        profiles[label] = summarize(
            root / f"profile-{label}.jsonl",
            root / f"markers-{label}.jsonl",
            root / "compute.h")
        profile_wall = profiles[label]["decode"]["wall_ms_per_token"]
        reported_wall = 1000.0 / diag_record["decode_tps"]
        if abs(profile_wall - reported_wall) > 0.02:
            raise ValueError(f"marker and benchmark clocks disagree for {label}")

    short = profiles["short"]["decode"]
    longctx = profiles["longctx"]["decode"]
    metric_names = (
        "wall_ms_per_token",
        "gpu_busy_ms_per_token",
        "gpu_span_ms_per_token",
        "positive_gpu_gap_ms_per_token",
        "intra_submit_gap_ms_per_token",
        "inter_submit_gap_ms_per_token",
        "submit_host_ms_per_token",
        "dispatch_record_host_ms_per_token",
        "begin_wait_ms_per_token",
        "join_wait_ms_per_token",
        "dispatches_per_token",
        "submissions_per_token",
    )
    deltas = {name: longctx[name] - short[name] for name in metric_names}
    kernel_names = set(short["kernels"]) | set(longctx["kernels"])
    kernel_deltas = {
        name: longctx["kernels"].get(name, {}).get("gpu_busy_ms_per_token", 0.0)
        - short["kernels"].get(name, {}).get("gpu_busy_ms_per_token", 0.0)
        for name in kernel_names
    }
    ranked_kernels = [
        {"kernel": name, "gpu_busy_delta_ms_per_token": delta}
        for name, delta in sorted(kernel_deltas.items(),
                                  key=lambda item: -item[1])
    ]
    control_delta = (
        controls["longctx"]["wall_ms_per_token"]["median"]
        - controls["short"]["wall_ms_per_token"]["median"])
    output = {
        "schema": "bf16-longctx-gpu-profile-window/1",
        "identity": identity,
        "pins": PINS,
        "control": controls,
        "diagnostic": diagnostic,
        "profiles": profiles,
        "comparison": {
            "release_control_wall_delta_ms_per_token": control_delta,
            "diagnostic_deltas": deltas,
            "kernel_gpu_busy_deltas": ranked_kernels,
            "components_are_not_additive": True,
            "diagnostic_wall_is_not_release_performance": True,
            "reason": "GPU timestamps, host submit costs, waits, and wall time overlap on the critical path; the profiling build adds timestamps and barriers per dispatch.",
        },
    }
    (root / "gpu-summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "control_wall_delta_ms_per_token": control_delta,
        "diagnostic_wall_delta_ms_per_token": deltas["wall_ms_per_token"],
        "gpu_busy_delta_ms_per_token": deltas["gpu_busy_ms_per_token"],
        "gpu_gap_delta_ms_per_token": deltas["positive_gpu_gap_ms_per_token"],
        "top_kernel_deltas": ranked_kernels[:5],
    }, sort_keys=True))
    print("GPU_PROFILE_WINDOW_VALID")


if __name__ == "__main__":
    main()
