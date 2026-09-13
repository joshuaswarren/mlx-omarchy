#!/usr/bin/env python3
import json
import pathlib
import statistics

root = pathlib.Path(__file__).parent
verdict = json.loads((root / "verdict.json").read_text())
analysis = json.loads((root / "valid-window/analysis.json").read_text())
assert verdict["execution_model"] == "openai-codex/gpt-5.6-sol"
assert verdict["fallback_model"] is None
assert verdict["outcome"]["verdict"] == "reject"
assert analysis["generated_ids_sha256_16"] == "ff502900d2a179a5"
assert analysis["candidate_faster_by_median"] is False

for side, stamp in (("control", "6b1ac029"), ("candidate", "ff815ad")):
    values = []
    for repeat in (1, 2, 3):
        text = (root / f"valid-window/{side}-{repeat}.log").read_text()
        assert f"+{stamp}" in text
        row = json.loads(text.strip().splitlines()[-1])
        assert row["ids_sha256_16"] == "ff502900d2a179a5"
        assert row["generated"] == 32
        assert row["prompt_tokens"] == 1053
        assert row["device"] == "Apple M1 (G13G B1)"
        values.append(row["decode_tps"])
    assert values == analysis["rows"][side]["decode_tps"]
    assert statistics.median(values) == analysis["rows"][side]["median_decode_tps"]
    assert values == verdict["performance"][f"{side}_decode_tps"]

for side, wheel_sha, stamp in (
    ("control", "fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50", "6b1ac029"),
    ("candidate", "2dc08eae980270b91799714dc81b379253d7c9f753da57bb997e956971606555", "ff815ad"),
):
    runtime = json.loads((root / f"valid-window/{side}-runtime.json").read_text())
    assert runtime["wheel_sha256"] == wheel_sha
    assert stamp in runtime["version"]
    assert runtime["pythonpath_cleared"]
    assert runtime["ld_library_path_cleared"]
    assert len(runtime["loaded_mappings"]) == 2
    assert sorted(runtime["loaded_mappings"].values()) == sorted(runtime["record_so_members"].values())
    assert runtime["device"]["device_name"] == "Apple M1 (G13G B1)"

window = (root / "valid-window/window.log").read_text()
for marker in (
    "runtime_provenance_complete utc=2026-09-13T06:19:35Z",
    "quiet_gate_complete utc=2026-09-13T06:20:15Z",
    "CHILD_PIDS_CLEAR",
    "lock_released=true utc=2026-09-13T06:20:53Z rc=0",
):
    assert marker in window
assert window.count("probe_start") == 6
assert window.count("probe_end") == 6
assert verdict["performance"]["control_median_decode_tps"] == 24.944
assert verdict["performance"]["candidate_median_decode_tps"] == 24.8461
assert verdict["performance"]["candidate_vs_control_percent"] == analysis["candidate_vs_control_percent"]

invalid = json.loads((root / "invalid-attempts.json").read_text())
assert invalid["performance_verdict"] is None
assert len(invalid["attempts"]) == 4
assert all(not attempt["candidate_numbers_valid"] for attempt in invalid["attempts"])
print("BF16_SDPA_256_RELEASE_VERDICT_VALID")
