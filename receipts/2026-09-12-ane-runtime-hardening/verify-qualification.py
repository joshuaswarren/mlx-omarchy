#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "9ea7501fc748cc031f7cbdc58afbed984fc00d51"
VERSION = "0.32.2.dev202609130744+9ea7501fc748cc031f7cbdc58afbed984fc00d51"
WHEEL_SHA256 = "b672dbd592878335e5180fdadd2193e25bdc5ed1cc3c759c2805e590642d1d93"
LOGS = {
    "jwm1-3172-installed-startup-failure.log": "b7338d017e2294ecc5469ec048573187c53585c1bc45f1ce5dc19d6fecb80e3e",
    "jwm1-9ea7501f-acceptance.log": "c372f83fa643b99e274e7adbd2a69e89290876ed7f53218c72bfe1d4d5f708c4",
    "jwm1-9ea7501f-gpu-postcheck.log": "1051431e4089a8abe53948a9a08a11f195062f45e4c9afcf0e55e4e03049d670",
}


def require_text(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise SystemExit(f"missing receipt text in {path.name}: {needle}")


for name, expected in LOGS.items():
    path = ROOT / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"receipt hash mismatch for {name}: {actual}")

receipt = json.loads((ROOT / "corrected-source-qualification.json").read_text())
assert receipt["source"]["qualified_commit"] == SOURCE
assert receipt["source"]["hardware_executed_for_qualified_source"] is True
assert receipt["artifact"]["installed_version"] == VERSION
assert receipt["artifact"]["wheel_sha256"] == WHEEL_SHA256
assert receipt["model"] == {
    "requested_route": "mlx-openai-deep",
    "actual": "openai-codex/gpt-5.6-sol",
    "fallback": False,
}
assert "full-model ANE execution or output quality" in receipt["qualification_scope"]["not_qualified"]
assert "other hardware or hosts" in receipt["qualification_scope"]["not_qualified"]

smokes = {smoke["operation"]: smoke for smoke in receipt["hardware"]["smokes"]}
assert set(smokes) == {"add", "add-mul"}
for smoke in smokes.values():
    assert smoke["exact_fp16_iterations"] == 2
    assert smoke["process_released"] is True
assert smokes["add"]["released_programs"] == 1
assert smokes["add-mul"]["released_programs"] == 2

gpu = receipt["hardware"]["post_smoke"]["gpu_postcheck"]
assert gpu["source_commit"] == SOURCE
assert gpu["installed_version"] == VERSION
assert gpu["wheel_sha256"] == WHEEL_SHA256
assert gpu["default_device"] == "Device(gpu, 0)"
assert gpu["device_name"] == "Apple M1 (G13G B1)"
assert gpu["architecture"] == "honeykrisp"
assert gpu["numerical_result"] == [[20.0, 23.0], [44.0, 51.0]]
assert gpu["vulkan_reopen"] is True

require_text(
    ROOT / "jwm1-3172-installed-startup-failure.log",
    "libmlx.so: cannot open shared object file",
    "current_operation_submitted=false",
    "clearance=true outer_lock=free",
)
require_text(
    ROOT / "jwm1-9ea7501f-acceptance.log",
    "Library runpath: [$ORIGIN/../lib]",
    "smoke=add status=pass",
    "smoke=add-mul status=pass",
    "clearance=true outer_lock=free",
)
require_text(
    ROOT / "jwm1-9ea7501f-gpu-postcheck.log",
    f"source={SOURCE}",
    f"version={VERSION}",
    f"wheel_sha256={WHEEL_SHA256}",
    "default_device=Device(gpu, 0)",
    "numerical_match=true result=[[20.0, 23.0], [44.0, 51.0]]",
    "device_name': 'Apple M1 (G13G B1)'",
    "'architecture': 'honeykrisp'",
    "vulkan_reopen=pass",
    "clearance=true outer_lock=free workers=absent quarantine=empty lease=removed exit_rc=0",
)

print(
    "qualification_receipt=pass "
    f"source={SOURCE} version={VERSION} wheel_sha256={WHEEL_SHA256} "
    "scope=installed-jwm1-add-addmul-and-gpu-postcheck"
)
