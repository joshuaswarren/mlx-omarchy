#!/usr/bin/env python3
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(name: str, expected_hash: str) -> str:
    path = ROOT / name
    assert sha256(path) == expected_hash
    return path.read_text()


receipt = json.loads((ROOT / "verification.json").read_text())
assert receipt["execution_model"] == "openai-codex/gpt-5.6-sol"
assert receipt["fallback_model"] is None
assert receipt["qualified_base_commit"] == "e4d079c883b8606dddef7720bd302c5756be0692"
assert receipt["offline_qualification"] is True
assert receipt["hardware_grant"] is False
assert receipt["optimization_claim"] is None

for name, expected in receipt["source"].items():
    assert sha256(REPO / name) == expected

route = receipt["route_contract"]
assert route["primitive_source_changed"] is False
primitive = (REPO / "overlay/mlx/backend/omarchy/primitives.cpp").read_text()
assert re.search(
    rf"k_len >= uint32_t\{{{route['bf16_min_keys']}\}}\s*&&\s*"
    rf"k_len <= uint32_t\{{{route['bf16_max_keys']}\}}",
    primitive,
)

compiled = receipt["shader_compile"]
for key, name in (
    ("base_bf16_sha256", "base-bf16.spv"),
    ("candidate_bf16_sha256", "candidate-bf16.spv"),
    ("base_f16_sha256", "base-f16.spv"),
    ("candidate_f16_sha256", "candidate-f16.spv"),
):
    assert sha256(ROOT / name) == compiled[key]
assert (ROOT / "base-f16.spv").read_bytes() == (ROOT / "candidate-f16.spv").read_bytes()

base_asm = subprocess.check_output(["spirv-dis", ROOT / "base-bf16.spv"], text=True)
candidate_asm = subprocess.check_output(
    ["spirv-dis", ROOT / "candidate-bf16.spv"], text=True
)
assert len(re.findall(r"OpExtInst.* Exp", base_asm)) == compiled["base_bf16_exp_sites"] == 2
assert len(re.findall(r"OpExtInst.* Exp", candidate_asm)) == compiled["candidate_bf16_exp_sites"] == 1
for assembly, expected in (
    (base_asm, compiled["base_bf16_local_size"]),
    (candidate_asm, compiled["candidate_bf16_local_size"]),
):
    match = re.search(r"OpExecutionMode %main LocalSize (\d+) (\d+) (\d+)", assembly)
    assert match and [int(value) for value in match.groups()] == expected == [1024, 1, 1]

tests = receipt["tests"]
baseline = read_log(tests["baseline_exact"]["log"], tests["baseline_exact"]["sha256"])
candidate = read_log(tests["candidate_exact"]["log"], tests["candidate_exact"]["sha256"])
assert baseline == candidate
for log, record in ((baseline, tests["baseline_exact"]), (candidate, tests["candidate_exact"])):
    for keys in record["keys"]:
        assert f"keys {keys} dispatches 1" in log
    assert f"assertions: {record['assertions']} | {record['assertions']} passed | 0 failed" in log
    assert "Status: SUCCESS!" in log

full = read_log(tests["candidate_full"]["log"], tests["candidate_full"]["sha256"])
assert re.search(r"test cases:\s+3 \|\s+3 passed \| 0 failed \| 0 skipped", full)
assert re.search(r"assertions:\s+13467 \| 13467 passed \| 0 failed", full)
assert "Status: SUCCESS!" in full

negative = read_log(tests["fault_injection"]["log"], tests["fault_injection"]["sha256"])
assert "keys 256 dispatches 1" in negative
assert "bit mismatch at 0: got 1.19531 want 0.0275879" in negative
assert "assertions: 4493 | 4492 passed | 1 failed" in negative
assert "Status: FAILURE!" in negative

manifest = json.loads((ROOT / "manifest.json").read_text())["files"]
actual_names = sorted(path.name for path in ROOT.iterdir() if path.name != "manifest.json")
assert sorted(manifest) == actual_names
for name, expected in manifest.items():
    assert sha256(ROOT / name) == expected

print(
    json.dumps(
        {
            "schema": receipt["schema"],
            "base_exp_sites": 2,
            "candidate_exp_sites": 1,
            "exact_assertions": 9878,
            "full_assertions": 13467,
            "fault_injection_caught": True,
            "hardware_grant": False,
            "result": "PASS",
        },
        sort_keys=True,
    )
)
