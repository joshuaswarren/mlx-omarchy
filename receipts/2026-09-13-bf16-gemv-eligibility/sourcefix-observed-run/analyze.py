#!/usr/bin/env python3
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_DIGEST = "ff502900d2a179a5"
EXPECTED_RUNNER_SHA = "cfbda0d39788c222961edfc3b624445dfcc10ed02eda46082454cd090fbbc50c"
EXPECTED_PREFLIGHT_SHA = "172c68ba5a33721b3f5eccb372c602a2f7a53fc3651f74afd9a2fe46481c8845"
EXPECTED_CANDIDATE_WHEEL_SHA = "f05c64c9acb17d5280d5ea7a2f0644ca44f97d85ea01a51b4e291d7d07d9f656"


def sha256(name: str) -> str:
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def probe(name: str) -> dict:
    lines = (ROOT / name).read_text().splitlines()
    row = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
    assert row["device"] == "Apple M1 (G13G B1)", row
    assert row["engine"] == "bench_decode", row
    assert row["prompt_tokens"] == 1053, row
    assert row["generated"] == 32, row
    assert row["ids_sha256_16"] == EXPECTED_DIGEST, row
    return row


identity = dict(line.split("=", 1) for line in (ROOT / "identity.txt").read_text().splitlines())
assert identity["candidate_commit"] == "4f90815c056cbb29a2ff4e1a958856b85af6390a"
assert identity["control_commit"] == "6b1ac0296ba65a8e0075171ca9451e222ddff06b"
assert identity["candidate_sha256"] == EXPECTED_CANDIDATE_WHEEL_SHA
assert sha256("executed-runner.sh") == EXPECTED_RUNNER_SHA
assert sha256("executed-preflight.sh") == EXPECTED_PREFLIGHT_SHA

candidate_runtime = json.loads((ROOT / "candidate-runtime.json").read_text())
control_runtime = json.loads((ROOT / "control-runtime.json").read_text())
assert "+4f90815" in candidate_runtime["version"], candidate_runtime["version"]
assert "+6b1ac029" in control_runtime["version"], control_runtime["version"]
assert candidate_runtime["loaded_mappings"], candidate_runtime
assert control_runtime["loaded_mappings"], control_runtime

provenance = json.loads((ROOT / "runtime-provenance.json").read_text())
assert provenance["loaded_images_differ"] is True, provenance
assert provenance["candidate_loaded_mappings"] == candidate_runtime["loaded_mappings"]
assert provenance["control_loaded_mappings"] == control_runtime["loaded_mappings"]

separate = (ROOT / "separate-flatten-test.log").read_text()
assert "eager dense bf16 decode gemv groups separate flatten views" in separate
assert "Skipping:" not in separate
assert "[doctest] Status: SUCCESS!" in separate
separate_count = re.search(r"assertions:\s+(\d+)\s+\|\s+(\d+) passed", separate)
assert separate_count and separate_count.group(1) == separate_count.group(2)
assert int(separate_count.group(1)) == 259

with gzip.open(ROOT / "dense-gemv-suite.log.gz", "rt") as stream:
    dense = stream.read()
assert "Skipping:" not in dense
assert "[doctest] Status: SUCCESS!" in dense
dense_count = re.search(r"assertions:\s+(\d+)\s+\|\s+(\d+) passed", dense)
assert dense_count and dense_count.group(1) == dense_count.group(2)

fusion = json.loads((ROOT / "fusion-gate.json").read_text())
assert fusion == {
    "dispatch_count_and_output_assertions_passed": 259,
    "environment_overrides_absent": True,
    "separate_flatten_regression_executed": True,
    "source_bound_installed_library": True,
}

candidate = [probe(f"candidate-{repeat}.log") for repeat in (1, 2, 3)]
control = [probe(f"control-{repeat}.log") for repeat in (1, 2, 3)]
candidate_tps = [row["decode_tps"] for row in candidate]
control_tps = [row["decode_tps"] for row in control]
candidate_median = statistics.median(candidate_tps)
control_median = statistics.median(control_tps)
assert candidate_tps == [24.9319, 24.867, 24.6603]
assert control_tps == [25.035, 24.8525, 24.9587]
assert candidate_median == 24.867
assert control_median == 24.9587
assert candidate_median < control_median

analysis = json.loads((ROOT / "analysis.json").read_text())
assert analysis["candidate_faster_by_median"] is False
assert analysis["generated_ids_sha256_16"] == EXPECTED_DIGEST
assert analysis["rows"]["candidate"]["median_decode_tps"] == candidate_median
assert analysis["rows"]["control"]["median_decode_tps"] == control_median

with gzip.open(ROOT / "window.log.gz", "rt") as stream:
    window = stream.read()
expected_order = [
    "probe_start side=control repeat=1",
    "probe_start side=candidate repeat=1",
    "probe_start side=candidate repeat=2",
    "probe_start side=control repeat=2",
    "probe_start side=control repeat=3",
    "probe_start side=candidate repeat=3",
]
positions = [window.index(marker) for marker in expected_order]
assert positions == sorted(positions)
assert window.count("probe_end side=") == 6
assert window.count(" rc=0 epoch=") >= 6
assert "WINDOW_COMPLETE result_dir=" in window
assert "CHILD_PIDS_CLEAR" in window
assert "lock_released=true" in window and " rc=0 result_dir=" in window

clearance = json.loads((ROOT / "independent-clearance.json").read_text())
assert clearance["observed_result"] == {"residual_workers": [], "lock_free": True}
assert clearance["lifecycle_acceptance"] is False
assert clearance["owned_process_group_id"] is None

verdict = {
    "schema": "bf16-gemv-sourcefix-observed-run/1",
    "execution_model": "openai-codex/gpt-5.6-sol",
    "fallback_model": None,
    "benchmark_model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16@56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
    "host_device": "Apple M1 (G13G B1)",
    "source_commit": identity["candidate_commit"],
    "candidate_wheel_sha256": identity["candidate_sha256"],
    "runner_sha256": EXPECTED_RUNNER_SHA,
    "preflight_sha256": EXPECTED_PREFLIGHT_SHA,
    "runner_review_status": "unreviewed_stale_remote_stage",
    "lifecycle_acceptance": False,
    "owned_process_group_id": None,
    "regression": {
        "separate_flatten_assertions_passed": int(separate_count.group(1)),
        "dense_gemv_assertions_passed": int(dense_count.group(1)),
    },
    "matrix": {
        "prompt_tokens": 1053,
        "generated_tokens": 32,
        "output_digest": EXPECTED_DIGEST,
        "candidate_decode_tps": candidate_tps,
        "candidate_median_decode_tps": candidate_median,
        "control_decode_tps": control_tps,
        "control_median_decode_tps": control_median,
        "candidate_vs_control_percent": analysis["candidate_vs_control_percent"],
    },
    "optimization_claim_accepted": False,
    "observed_result": "The source-fix wheel passed the targeted regression and six-run fixed matrix, but its median decode rate was 0.3674% below control. The stale remote runner invalidates lifecycle acceptance and source promotion.",
    "post_release_clearance": clearance["observed_result"],
}
(ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
