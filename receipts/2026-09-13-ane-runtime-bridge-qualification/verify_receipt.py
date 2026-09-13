#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECEIPT = json.loads((ROOT / "receipt.json").read_text())

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name, expected in RECEIPT["artifacts"].items():
    actual = sha256(ROOT / name)
    if actual != expected:
        raise SystemExit(f"hash mismatch: {name}: expected {expected}, got {actual}")

log = (ROOT / "acceptance-success.log").read_text()
required = [
    "wheel_record_identity=pass installed_version=0.32.2.dev202609130952+57cc36a2e8ece78dd979b5344752d04c76b4d49d",
    "smoke=add status=pass",
    "smoke=add-mul status=pass",
    "gpu_smoke=pass default_device=Device(gpu, 0) numerical_match=true result=[[20.0, 23.0], [44.0, 51.0]]",
    "installed_57cc_mel_acceptance=pass comparisons=18 device=Device(gpu, 0)",
    "vulkan_device_reopen=true",
    "installed_combined_acceptance=pass source=57cc36a2e8ece78dd979b5344752d04c76b4d49d",
    "ane_smokes=2 ane_executions=4 gpu_postcheck=true mel_comparisons=18 vulkan_reopen=true",
    "clearance=true outer_lock=free process_group_scan=pass process_group=empty worker_scan=pass workers=absent quarantine=empty lease=removed",
    "cleanup_deadline=within exit_rc=0",
]
for marker in required:
    if marker not in log:
        raise SystemExit(f"missing success marker: {marker}")

terminal = (ROOT / "supervised-service-terminal.txt").read_text()
for marker in [
    "ane57cc-installed-combined: exited exit=0 uptime=29.7s restarts=0",
    "independent_clearance=pass guardian_absent=3390545 pgid_empty=3390545 workers_absent=true lease_absent=true quarantine_empty=true lock_free=true",
]:
    if marker not in terminal:
        raise SystemExit(f"missing supervised terminal marker: {marker}")

prior = (ROOT / "build-install-guard-124.log").read_text()
for marker in [
    "wheel=mlx_omarchy-0.32.2.dev202609130952+57cc36a2e8ece78dd979b5344752d04c76b4d49d-cp314-cp314-linux_aarch64.whl",
    "deadline_guard=failed phase=smoke_harness_compile remaining_seconds=258 required_seconds=300",
    "clearance=true outer_lock=free process_group_scan=pass process_group=empty worker_scan=pass workers=absent quarantine=empty lease=removed",
    "cleanup_deadline=within exit_rc=124",
]:
    if marker not in prior:
        raise SystemExit(f"missing prior guard marker: {marker}")

mel = json.loads((ROOT / "mel-report.json").read_text())
if not (mel["qualified"] and mel["all_bit_exact"] and mel["stage_set_exact"]):
    raise SystemExit("mel qualification flags are not all true")
if len(mel["comparisons"]) != 18 or not all(item["bit_exact"] for item in mel["comparisons"].values()):
    raise SystemExit("mel comparison set is not 18 bit-exact stages")
if mel["device"] != "Device(gpu, 0)":
    raise SystemExit(f"unexpected mel device: {mel['device']}")
expected_trace = RECEIPT["results"]["mel_frontend"]["trace_delta"]
if mel["trace_delta"] != expected_trace:
    raise SystemExit(f"trace mismatch: expected {expected_trace}, got {mel['trace_delta']}")

print("ane_runtime_bridge_receipt=verified artifacts=10 source=57cc36a2e8ece78dd979b5344752d04c76b4d49d ane_smokes=2 ane_executions=4 gpu_postcheck=true mel_comparisons=18 clearance=true independent_clearance=true prior_guard_124=verified")
