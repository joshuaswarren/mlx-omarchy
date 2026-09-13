#!/usr/bin/env python3
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(name: str) -> str:
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


proposal = json.loads((ROOT / "proposal.json").read_text())
assert proposal["execution_model"] == "openai-codex/gpt-5.6-sol"
assert proposal["fallback_model"] is None
candidate = proposal["selected_candidate"]
assert candidate["status"] == "offline_exact_word_qualified_and_arm_wheels_built_pending_hardware_grant"
assert candidate["candidate_commit"] == "2278ca259618c51cd5ec99fa68c272ffe55b69a5"
assert proposal["future_hardware_acceptance"]["candidate_commit"] == candidate["candidate_commit"]
candidate_shader = subprocess.check_output(
    [
        "git",
        "show",
        candidate["candidate_commit"]
        + ":overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp",
    ],
    cwd=REPO,
)
assert hashlib.sha256(candidate_shader).hexdigest() == candidate["shader_source_sha256"]
candidate_spv = subprocess.check_output(
    [
        "git",
        "show",
        candidate["candidate_commit"]
        + ":receipts/2026-09-13-bf16-sdpa-exp-cache-local/candidate-bf16.spv",
    ],
    cwd=REPO,
)
assert hashlib.sha256(candidate_spv).hexdigest() == candidate["candidate_bf16_spv_sha256"]
assert proposal["offline_acceptance"]["hardware_allowed"] is False

protocol = proposal["protocol"]
assert sha256(protocol["launcher"]) == protocol["launcher_sha256"]
assert sha256(protocol["payload"]) == protocol["payload_sha256"]
assert protocol["cleanup_reserve_seconds"] == 60

source_preparation = proposal["source_preparation"]
build = json.loads((ROOT / source_preparation["receipt"]).read_text())
assert build["schema"] == "longctx-arm-wheel-build/1"
assert build["execution_model"] == proposal["execution_model"]
assert build["fallback_model"] is None
assert build["host"] == source_preparation["host"] == "jwm1-linux"
assert sha256(source_preparation["build_harness"]) == source_preparation["build_harness_sha256"] == build["build_harness_sha256"]
assert source_preparation["gpu_or_model_execution"] is False
for side, expected_commit, expected_wheel in (
    ("control", proposal["qualified_base_commit"], "6b308dfc58375c341c290057d236ee0e189b89d11fa55f4e09e5d9b4715f8ac2"),
    ("candidate", candidate["candidate_commit"], "ceaec9a6a8432466ab0e3034abb552455153d9c40c24dd5493033baea10cfbba"),
):
    artifact = build[side]
    assert artifact["source_commit"] == expected_commit
    assert artifact["tracked_status"] == ""
    assert artifact["build_rc"] == 0
    assert artifact["process_group_absent"] is True
    assert artifact["wheel_sha256"] == expected_wheel
    assert artifact["wheel_native_members"] == artifact["installed_native_files"]
    assert artifact["wheel_spv_members_present"] is False
assert build["compiled_shader_evidence"] == {
    "base_bf16_spv_sha256": "daa432ed514b05b2dbfa9adcab64a3b464e037f2f4271a6c962598c3ffef9b15",
    "base_f16_spv_sha256": "a301a801f73640c3c2fe11888bb85526618f1cc4ac839f4557d4143b0e3a6e60",
    "candidate_bf16_spv_sha256": "d8edeeb952599a2e280d9cfc9a3d54be5e8993a565236751b9489a4ffffde973",
    "candidate_f16_spv_sha256": "a301a801f73640c3c2fe11888bb85526618f1cc4ac839f4557d4143b0e3a6e60",
    "note": "The wheels contain no standalone .spv members; exact runtime wheel and native-image hashes bind the embedded build.",
    "source_commit": candidate["candidate_commit"],
}

base_shader = subprocess.check_output(
    [
        "git",
        "show",
        proposal["qualified_base_commit"]
        + ":overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp",
    ],
    cwd=REPO,
    text=True,
)
assert "layout(local_size_x = 1024" in base_shader
assert "local_sum += exp(s_stream[index] - row_max);" in base_shader
assert "s_stream[index] = exp(s_stream[index] - row_max) * normalizer;" in base_shader

profile = json.loads(
    (REPO / "receipts/2026-09-13-bf16-longctx-gpu-profile/verdict.json").read_text()
)
attribution = profile["attribution"]
evidence = proposal["profile_evidence"]
assert attribution["release_control_wall_delta_ms_per_token"] == evidence["release_control_longctx_wall_delta_ms_per_token"]
assert attribution["sdpa_decode_native_bf16_gpu_busy_delta_ms_per_token"] == evidence["sdpa_decode_native_bf16_gpu_busy_delta_ms_per_token"]
assert attribution["all_other_kernels_net_gpu_busy_delta_ms_per_token"] == evidence["all_other_kernels_net_gpu_busy_delta_ms_per_token"]
assert attribution["diagnostic_gpu_busy_delta_ms_per_token"] == evidence["net_instrumented_gpu_busy_delta_ms_per_token"]
assert attribution["components_are_not_additive"] is evidence["components_are_not_additive"] is True
assert profile["source_attribution"]["target"] == "SdpaDecodeNativeBF16"

sdpa_release = json.loads(
    (REPO / "receipts/2026-09-13-bf16-sdpa-256-release/verdict.json").read_text()
)
assert sdpa_release["candidate"]["commit"] == "ff815ada8a1730fdf9762f2cbd5c591d99a9c06b"
assert sdpa_release["performance"]["candidate_vs_control_percent"] == -0.39247915330339334
assert sdpa_release["outcome"]["verdict"] == "reject"

barriers = json.loads(
    (
        REPO
        / "receipts/2026-09-12-bf16-encoder-barrier-pair/window3/summary.json"
    ).read_text()
)["paired"]["qwen25-0.5b-bf16:longctx-1024-decode-32"]
assert barriers["base"] == [25.02, 24.98, 24.96]
assert barriers["cand"] == [24.95, 25.0, 24.93]
assert statistics.median(barriers["cand"]) < statistics.median(barriers["base"])

vector = json.loads(
    (REPO / "receipts/2026-09-12-bf16-vector-mapping/verdict.json").read_text()
)
assert vector["mechanism"]["landable"] is False

smoke = (ROOT / "smoke-receipt.txt").read_text()
guardian = re.search(
    r"guardian_identity=true guardian_pid=(\d+) comm=timeout pgid=(\d+)", smoke
)
workload = re.search(r"smoke_workload=true pid=(\d+) pgid=(\d+)", smoke)
assert guardian and workload
assert guardian.group(2) == workload.group(2)
for marker in (
    "fake_child_detected=true",
    "fake_child_left_for_cleanup=true",
    "fake_detached_child_detected=true",
    "fake_detached_child_left_for_cleanup=true",
    "workload_complete=true run_label=bounded-fake-child-smoke",
    "clearance=true outer_lock=free process_group_scan=pass process_group=empty descendant_cleanup=terminated_descendants=",
    "independent_clearance=true guardian=absent",
    "process_group=empty lease=absent lock=free",
    "run_rc=0",
):
    assert marker in smoke, marker
assert f"payload_sha256={protocol['payload_sha256']}" in smoke
failure_smoke = (ROOT / "failure-smoke-receipt.txt").read_text()
for marker in (
    "scanner_failure_case_rc=125 lease_retained=true lock=free",
    "clearance=false outer_lock=free process_group_scan=failed",
    "lease_present=true",
    "retained_lease_refusal_rc=78 lease_unchanged=true",
    "lease_acquired=false reason=existing_retained_lease",
    "exit_124_case_rc=124 lease_removed=true lock=free",
    "process_group_scan=pass process_group=empty descendant_cleanup=none lease=removed",
    "cleanup_deadline=within exit_rc=124",
):
    assert marker in failure_smoke, marker
launcher_smoke = (ROOT / "launcher-smoke-receipt.txt").read_text()
clearance_records = [
    line for line in launcher_smoke.splitlines() if line.startswith("clearance=")
]
assert len(clearance_records) == 2
for record in clearance_records:
    clearance_fields = dict(token.split("=", 1) for token in record.split())
    for key, expected in {
        "clearance": "true",
        "outer_lock": "free",
        "process_group_scan": "pass",
        "process_group": "empty",
        "lease": "removed",
        "cleanup_deadline": "within",
        "exit_rc": "0",
    }.items():
        assert clearance_fields.get(key) == expected
launch_records = [
    line for line in launcher_smoke.splitlines() if line.startswith("launch_complete=")
]
assert len(launch_records) == 1
launch_fields = dict(token.split("=", 1) for token in launch_records[0].split())
assert launch_fields["runner_sha256"] == protocol["payload_sha256"]
assert launch_fields["workload_sha256"] == sha256("smoke-workload.sh")
assert "launcher_contract_smoke=pass" in launcher_smoke
assert "independent_clearance_deadline_case_rc=124 launch_complete=false" in launcher_smoke

hardware = proposal["future_hardware_acceptance"]
assert sha256(hardware["workload_file"]) == hardware["workload_sha256"]
assert hardware["per_probe_deadline_seconds"] == 85
assert hardware["matrix_order"] == [
    "control-1",
    "candidate-1",
    "candidate-2",
    "control-2",
    "control-3",
    "candidate-3",
]
assert hardware["native_image_identity"]["control"] == build["control"]["wheel_native_members"]
assert hardware["native_image_identity"]["candidate"] == build["candidate"]["wheel_native_members"]
assert hardware["compiled_spv_identity"] == {
    "base_bf16": build["compiled_shader_evidence"]["base_bf16_spv_sha256"],
    "base_f16": build["compiled_shader_evidence"]["base_f16_spv_sha256"],
    "candidate_bf16": build["compiled_shader_evidence"]["candidate_bf16_spv_sha256"],
    "candidate_f16": build["compiled_shader_evidence"]["candidate_f16_spv_sha256"],
}
candidate_workload = (ROOT / hardware["workload_file"]).read_text()
for literal in (
    "EXPECTED_BASE=e4d079c883b8606dddef7720bd302c5756be0692",
    "EXPECTED_CANDIDATE=2278ca259618c51cd5ec99fa68c272ffe55b69a5",
    "EXPECTED_BASE_SHADER=e41b54a666e5284921593cd5c3fdac4fca26a4f475abf54cc5483967e88e8706",
    "EXPECTED_CANDIDATE_SHADER=b58a0b9c80cf01d0520ef66ee5fbbc2f16ddb22aefd2efec41c1f80557db5f5c",
    "EXPECTED_CONTROL_WHEEL_SHA=6b308dfc58375c341c290057d236ee0e189b89d11fa55f4e09e5d9b4715f8ac2",
    "EXPECTED_CANDIDATE_WHEEL_SHA=ceaec9a6a8432466ab0e3034abb552455153d9c40c24dd5493033baea10cfbba",
    "EXPECTED_CONTROL_CORE_SHA=4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5",
    "EXPECTED_CANDIDATE_CORE_SHA=4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5",
    "EXPECTED_CONTROL_LIBMLX_SHA=f604640d8c125eb2fb9082381691cd94b4f3aa42b169499561a6fd0d5eda542f",
    "EXPECTED_CANDIDATE_LIBMLX_SHA=155ba25b8e6fc8188654a56142358e63b28712bd9e3524b27194a7dbab6c2f5d",
    "EXPECTED_BASE_BF16_SPV_SHA=daa432ed514b05b2dbfa9adcab64a3b464e037f2f4271a6c962598c3ffef9b15",
    "EXPECTED_BASE_F16_SPV_SHA=a301a801f73640c3c2fe11888bb85526618f1cc4ac839f4557d4143b0e3a6e60",
    "EXPECTED_CANDIDATE_BF16_SPV_SHA=d8edeeb952599a2e280d9cfc9a3d54be5e8993a565236751b9489a4ffffde973",
    "EXPECTED_CANDIDATE_F16_SPV_SHA=a301a801f73640c3c2fe11888bb85526618f1cc4ac839f4557d4143b0e3a6e60",
    "assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected_wheel_sha",
    "assert digest == expected_members[member]",
    "snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
    "EXPECTED_IDS=ff502900d2a179a5",
    "mlx_omarchy-*+e4d079c-*.whl",
    "mlx_omarchy-*+2278ca2-*.whl",
    "timeout --foreground --kill-after=5s 85s",
    "mx.set_default_device(mx.gpu)",
    "for spec in control:1 candidate:1 candidate:2 control:2 control:3 candidate:3",
):
    assert literal in candidate_workload, literal
assert "+2278ca25-*.whl" not in candidate_workload
for line in candidate_workload.splitlines():
    if "timeout " in line:
        assert "timeout --foreground" in line
embedded_python = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\n|$)", candidate_workload, re.DOTALL)
assert len(embedded_python) == 5
for index, source in enumerate(embedded_python):
    compile(source, f"candidate-workload-heredoc-{index}.py", "exec")

launcher = (ROOT / "launch-exact-run.sh").read_text()
payload = (ROOT / "exact-run-payload.sh").read_text()
assert f"PAYLOAD_SHA={protocol['payload_sha256']}" in launcher
remote_launch = launcher.index('"set -e; ACQUIRE_CUTOFF_EPOCH=')
remote_check = launcher.index("$REMOTE_STAGE_CHECK", remote_launch)
remote_exec = launcher.index("exec setsid --wait timeout", remote_check)
assert remote_launch < remote_check < remote_exec
assert "remote_session_deadline=\$((remote_now + 900))" in launcher
assert "remote_limit=\$((remote_session_deadline - remote_now - 15))" in launcher
assert "CLEANUP_RESERVE_SECONDS=60" in payload
assert "WORK_DEADLINE_EPOCH=$((SESSION_DEADLINE_EPOCH - CLEANUP_RESERVE_SECONDS))" in payload
assert "timeout --foreground" in payload
assert "excluded = {script, guardian, me, scanner_parent}" in payload
assert "libc.prctl(36, 1, 0, 0, 0)" in payload
assert 'env["LONGCTX_SUBREAPER_ACTIVE"] = "1"' in payload
assert "if pid not in excluded and pid not in owned and (ppid == script or ppid in owned)" in payload
assert "set -o noclobber" in payload
assert "reason=existing_retained_lease" in payload
assert "reason=atomic_create_refused" in payload
assert 'timeout --foreground --signal=TERM --kill-after=5s "${remaining}s" "${SSH[@]}" python3' in launcher

verdict = {
    "schema": "bf16-longctx-next-candidate-protocol/1",
    "execution_model": proposal["execution_model"],
    "fallback_model": None,
    "candidate": proposal["selected_candidate"]["name"],
    "candidate_status": proposal["selected_candidate"]["status"],
    "hardware_grant": False,
    "protocol_ready": True,
    "payload_sha256": protocol["payload_sha256"],
    "launcher_sha256": protocol["launcher_sha256"],
    "fake_child_detected_and_cleared": True,
    "detached_child_detected_and_cleared": True,
    "child_subreaper_enabled": True,
    "retained_lease_refused_unchanged": True,
    "lease_creation_is_atomic": True,
    "independent_clearance_has_absolute_deadline": True,
    "source_preparation_receipt": source_preparation["receipt"],
    "arm_wheels_built": True,
    "control_wheel_sha256": build["control"]["wheel_sha256"],
    "candidate_wheel_sha256": build["candidate"]["wheel_sha256"],
    "prepared_native_images_are_fixed": True,
    "prepared_spirv_images_are_fixed": True,
    "scanner_failure_preserves_uncertainty": True,
    "exit_124_propagated": True,
    "guardian_and_workload_process_group_match": True,
    "cleanup_reserve_seconds": 60,
    "remote_hash_check_immediately_before_exec": True,
    "launcher_end_to_end_contract": True,
    "candidate_workload_sha256": hardware["workload_sha256"],
    "candidate_workload_static_contract": True,
    "planned_matrix": hardware["matrix_order"],
    "optimization_claim": None,
}
(ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
manifest = json.loads((ROOT / "manifest.json").read_text())["files"]
actual_names = sorted(path.name for path in ROOT.iterdir() if path.name != "manifest.json")
assert sorted(manifest) == actual_names
for name, expected in manifest.items():
    assert sha256(name) == expected
print(json.dumps(verdict, indent=2, sort_keys=True))
