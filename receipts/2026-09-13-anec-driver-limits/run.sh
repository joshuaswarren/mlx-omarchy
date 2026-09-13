#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${MLX_OMARCHY_BUILD_DIR:-$ROOT/.work/build}"

"$ROOT/scripts/prepare-mlx.sh" >/dev/null 2>&1
cmake -S "$ROOT/.work/mlx" -B "$BUILD" \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF >/dev/null 2>&1
cmake --build "$BUILD" --target omarchy_ane_bundle_tests mlx-omarchy-info -j4 >/dev/null 2>&1
printf '[receipt] source_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[receipt] driver_source_commit=f261a6cb537aca62f267ad3d01beda0d6877544c\n'
printf '[receipt] driver_contract=ane/src/ane_drv.c:313-315 td_count<=0xffff td_size<=0x40000\n'
"$BUILD/tests/omarchy/omarchy_ane_bundle_tests" \
  --test-case='ANEC task descriptor fields stay within driver submit limits'

ROOT="$ROOT" BUILD="$BUILD" python3 - <<'PY'
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

root = Path(os.environ["ROOT"])
cli = Path(os.environ["BUILD"]) / "tools/mlx-omarchy-info/mlx-omarchy-info"
source = root / "receipts/fixtures/mil-oneop-bundle"


def identity(payloads):
    records = [
        {key: payload[key] for key in ("role", "path", "byte_size", "sha256")}
        for payload in payloads
    ]
    records.sort(key=lambda payload: payload["path"])
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def make(target, *, count=None, size=None):
    shutil.copytree(source, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload_path = target / "model-512.anec"
    payload = bytearray(payload_path.read_bytes())
    if count is not None:
        struct.pack_into("<I", payload, 12, count)
        manifest["programs"][0]["task_descriptors"] = count
        manifest["task_descriptors"] = count
    if size is not None:
        payload.extend(bytes(0x1000 + size - len(payload)))
        struct.pack_into("<Q", payload, 0, size)
        struct.pack_into("<I", payload, 8, size)
        struct.pack_into("<I", payload, 40, (size + 0x3FFF) // 0x4000)
    payload_path.write_bytes(payload)
    record = next(p for p in manifest["payloads"] if p["path"] == payload_path.name)
    record["byte_size"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["release_asset"]["model_sha256"] = identity(manifest["payloads"])
    encoded_manifest = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(encoded_manifest)
    return {
        "anec_sha256": record["sha256"],
        "manifest_sha256": hashlib.sha256(encoded_manifest.encode()).hexdigest(),
        "model_sha256": manifest["release_asset"]["model_sha256"],
        "td_count": struct.unpack_from("<I", payload, 12)[0],
        "td_size": struct.unpack_from("<I", payload, 8)[0],
    }


cases = {
    "count": ({"count": 0x10000}, "ANEC task descriptor count exceeds driver limit 0xffff"),
    "size": ({"size": 0x40004}, "ANEC task descriptor size exceeds driver limit 0x40000"),
}
with tempfile.TemporaryDirectory(prefix="mlx-anec-driver-limits-") as temporary:
    for name, (mutation, expected) in cases.items():
        target = Path(temporary) / name
        hashes = make(target, **mutation)
        result = subprocess.run(
            [cli, "--check-bundle", target], text=True, capture_output=True, check=False
        )
        output = result.stdout + result.stderr
        print(f"[receipt] case={name} {json.dumps(hashes, sort_keys=True)}")
        print(f"[receipt] case={name} exit={result.returncode} output={output.strip()}")
        if result.returncode != 1 or expected not in output:
            raise SystemExit(f"{name} overflow was not refused as expected")
PY
