#!/usr/bin/env python3
import argparse
import json
import re
import tempfile
from pathlib import Path

PHASES = {
    "load_start": "load",
    "prefill_start": "prefill",
    "decode_start": "decode",
    "tok": "decode",
}


def kernel_names(path):
    names = []
    inside = False
    pattern = re.compile(r"^\s{2}(\w+),\s*$")
    for line in path.read_text().splitlines():
        if "enum class ComputeKernel" in line:
            inside = True
        elif inside and "};" in line:
            break
        elif inside and (match := pattern.match(line)):
            names.append(match.group(1))
    if not names:
        raise ValueError("ComputeKernel enum is empty")
    return names


def phase_at(markers, timestamp):
    current = None
    for marker in markers:
        if marker["t"] > timestamp:
            break
        current = PHASES.get(marker["p"])
    return current


def summarize(profile_path, markers_path, compute_path):
    records = [json.loads(line) for line in profile_path.read_text().splitlines()
               if line.strip()]
    markers = [json.loads(line) for line in markers_path.read_text().splitlines()
               if line.strip()]
    metas = [row for row in records if row.get("k") == "meta"]
    ends = [row for row in records if row.get("k") == "end"]
    if len(metas) != 1 or len(ends) != 1:
        raise ValueError("profile requires exactly one meta and one end record")
    meta, end = metas[0], ends[0]
    period = meta.get("period_ns")
    valid_bits = meta.get("valid_bits")
    if not isinstance(period, (int, float)) or period <= 0:
        raise ValueError("invalid GPU timestamp period")
    if valid_bits != 64:
        raise ValueError(f"expected 64 valid GPU timestamp bits, got {valid_bits!r}")

    dispatches = [row for row in records if row.get("k") == "d"]
    submits = [row for row in records if row.get("k") == "s"]
    joins = [row for row in records if row.get("k") == "j"]
    begins = [row for row in records if row.get("k") == "b"]
    if end.get("dropped") != 0:
        raise ValueError(f"profile dropped {end.get('dropped')} records")
    for key, actual in (("dispatches", len(dispatches)),
                        ("submissions", len(submits)),
                        ("joins", len(joins))):
        if end.get(key) != actual:
            raise ValueError(f"{key} count mismatch: end={end.get(key)} actual={actual}")

    phases = [marker["p"] for marker in markers]
    required = ["prefill_start", "prefill_done", "decode_start", "decode_done"]
    if any(phases.count(name) != 1 for name in required):
        raise ValueError("marker boundaries must occur exactly once")
    if phases.count("tok") != 32:
        raise ValueError(f"expected 32 token markers, got {phases.count('tok')}")
    boundary_positions = [phases.index(name) for name in required]
    if boundary_positions != sorted(boundary_positions):
        raise ValueError("marker boundaries are out of order")
    token_times = [marker["t"] for marker in markers if marker["p"] == "tok"]
    if token_times != sorted(token_times):
        raise ValueError("token markers are not monotonic")
    intervals = len(token_times) - 1

    submit_by_id = {row["s"]: row for row in submits}
    decode = []
    for row in dispatches:
        submit = submit_by_id.get(row.get("s"))
        if submit is None:
            raise ValueError(f"dispatch has no submit record: {row.get('s')}")
        if phase_at(markers, submit["t"]) == "decode":
            if not all(key in row for key in ("t0", "t1", "h", "e")):
                raise ValueError("decode dispatch lacks timestamp or host-cost fields")
            if row["t1"] < row["t0"]:
                raise ValueError("64-bit GPU timestamps moved backwards")
            decode.append(row)
    if not decode:
        raise ValueError("profile has no decode-window dispatches")

    ordered = sorted(decode, key=lambda row: row["t0"])
    durations = [(row["t1"] - row["t0"]) * period for row in ordered]
    positive_gaps = 0.0
    intra_gaps = 0.0
    inter_gaps = 0.0
    overlaps = 0
    for previous, current in zip(ordered, ordered[1:]):
        gap = (current["t0"] - previous["t1"]) * period
        if gap < 0:
            overlaps += 1
            continue
        positive_gaps += gap
        if previous["s"] == current["s"]:
            intra_gaps += gap
        else:
            inter_gaps += gap

    names = kernel_names(compute_path)
    kernels = {}
    for row, duration in zip(ordered, durations):
        name = names[row["e"]] if row["e"] < len(names) else f"kernel_{row['e']}"
        value = kernels.setdefault(name, {"dispatches": 0, "gpu_busy_ns": 0.0})
        value["dispatches"] += 1
        value["gpu_busy_ns"] += duration

    decode_submits = [row for row in submits
                      if phase_at(markers, row["t"]) == "decode"]
    decode_joins = [row for row in joins
                    if phase_at(markers, row["t"]) == "decode"]
    decode_begins = [row for row in begins
                     if phase_at(markers, row["t"]) == "decode"]

    def per_interval(ns):
        return ns / intervals / 1e6

    return {
        "device": meta.get("device"),
        "label": meta.get("label"),
        "period_ns": period,
        "valid_bits": valid_bits,
        "records": {
            "dispatches": len(dispatches),
            "submissions": len(submits),
            "joins": len(joins),
            "begins": len(begins),
            "dropped": end["dropped"],
        },
        "decode": {
            "intervals": intervals,
            "wall_ms_per_token": per_interval(token_times[-1] - token_times[0]),
            "dispatches": len(decode),
            "dispatches_per_token": len(decode) / intervals,
            "submissions": len(decode_submits),
            "submissions_per_token": len(decode_submits) / intervals,
            "gpu_busy_ms_per_token": per_interval(sum(durations)),
            "gpu_span_ms_per_token": per_interval(
                (max(row["t1"] for row in ordered)
                 - min(row["t0"] for row in ordered)) * period),
            "positive_gpu_gap_ms_per_token": per_interval(positive_gaps),
            "intra_submit_gap_ms_per_token": per_interval(intra_gaps),
            "inter_submit_gap_ms_per_token": per_interval(inter_gaps),
            "overlapping_dispatch_pairs": overlaps,
            "submit_host_ms_per_token": per_interval(
                sum(row["dur"] for row in decode_submits)),
            "dispatch_record_host_ms_per_token": per_interval(
                sum(row["h"] for row in decode)),
            "begin_wait_ms_per_token": per_interval(
                sum(row["dur"] for row in decode_begins)),
            "join_wait_ms_per_token": per_interval(
                sum(row["wait"] for row in decode_joins)),
            "kernels": {
                name: {
                    "dispatches": value["dispatches"],
                    "dispatches_per_token": value["dispatches"] / intervals,
                    "gpu_busy_ms_per_token": per_interval(value["gpu_busy_ns"]),
                }
                for name, value in sorted(
                    kernels.items(), key=lambda item: -item[1]["gpu_busy_ns"])
            },
        },
    }


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        compute = root / "compute.h"
        profile = root / "profile.jsonl"
        markers = root / "markers.jsonl"
        compute.write_text("enum class ComputeKernel {\n  Alpha,\n  Beta,\n};\n")
        rows = [
            {"k": "meta", "device": "probe", "period_ns": 1.0,
             "valid_bits": 64, "label": "self-test"},
            {"k": "b", "t": 110, "dur": 3},
            {"k": "s", "s": 1, "t": 110, "dur": 4},
            {"k": "d", "s": 1, "e": 0, "t0": 1000, "t1": 1010,
             "h": 2},
            {"k": "d", "s": 1, "e": 1, "t0": 1020, "t1": 1040,
             "h": 2},
            {"k": "j", "s": 1, "t": 250, "wait": 5, "inval": 0},
            {"k": "end", "dispatches": 2, "dropped": 0,
             "submissions": 1, "joins": 1},
        ]
        marks = [
            {"p": "prefill_start", "t": 0},
            {"p": "prefill_done", "t": 100},
            {"p": "decode_start", "t": 101},
            {"p": "tok", "t": 102},
            *({"p": "tok", "t": value} for value in range(103, 133)),
            {"p": "tok", "t": 202},
            {"p": "decode_done", "t": 203},
        ]
        profile.write_text("".join(json.dumps(row) + "\n" for row in rows))
        markers.write_text("".join(json.dumps(row) + "\n" for row in marks))
        result = summarize(profile, markers, compute)
        assert result["decode"]["intervals"] == 31
        assert result["decode"]["dispatches"] == 2
        assert result["decode"]["gpu_busy_ms_per_token"] == 30 / 31 / 1e6
        assert result["decode"]["positive_gpu_gap_ms_per_token"] == 10 / 31 / 1e6
        assert list(result["decode"]["kernels"]) == ["Beta", "Alpha"]
    print("GPU_PROFILE_SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", nargs="?")
    parser.add_argument("markers", nargs="?")
    parser.add_argument("compute_h", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.profile, args.markers, args.compute_h)):
        parser.error("profile, markers, and compute_h are required")
    print(json.dumps(summarize(Path(args.profile), Path(args.markers),
                               Path(args.compute_h)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
