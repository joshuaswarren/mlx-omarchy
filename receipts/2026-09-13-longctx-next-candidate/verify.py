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
assert proposal["selected_candidate"]["status"] == "source_proposal_not_implemented"
assert proposal["offline_acceptance"]["hardware_allowed"] is False

protocol = proposal["protocol"]
assert sha256(protocol["launcher"]) == protocol["launcher_sha256"]
assert sha256(protocol["payload"]) == protocol["payload_sha256"]
assert protocol["cleanup_reserve_seconds"] == 60

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
    "workload_complete=true run_label=bounded-fake-child-smoke",
    "clearance=true outer_lock=free process_group_scan=pass process_group=empty descendant_cleanup=terminated_descendants=",
    "independent_clearance=true guardian=absent",
    "process_group=empty lease=absent lock=free",
    "run_rc=0",
):
    assert marker in smoke, marker
assert f"payload_sha256={protocol['payload_sha256']}" in smoke

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
assert "pid not in (script, guardian, me, scanner_parent)" in payload

verdict = {
    "schema": "bf16-longctx-next-candidate-protocol/1",
    "execution_model": proposal["execution_model"],
    "fallback_model": None,
    "candidate": proposal["selected_candidate"]["name"],
    "candidate_status": "proposal_only_pending_exact_word_offline_qualification",
    "hardware_grant": False,
    "protocol_ready": True,
    "payload_sha256": protocol["payload_sha256"],
    "launcher_sha256": protocol["launcher_sha256"],
    "fake_child_detected_and_cleared": True,
    "guardian_and_workload_process_group_match": True,
    "cleanup_reserve_seconds": 60,
    "remote_hash_check_immediately_before_exec": True,
    "optimization_claim": None,
}
(ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
