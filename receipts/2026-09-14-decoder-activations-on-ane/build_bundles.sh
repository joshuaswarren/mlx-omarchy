#!/usr/bin/env bash
set -euo pipefail
WORK=/tmp/jwm1-decoder-activations-build
COMPILER=/home/joshuawarren/src/mil-hwx-compiler
COMMIT=$(git -C "$COMPILER" rev-parse HEAD)
cp /var/tmp/Jwm1AneMoreFixtures-a9f14124/inspect/h13_package_to_bundle.py /var/tmp/Jwm1AneMoreFixtures-a9f14124/inspect/bundle_payload_identity.py "$WORK/"
sha256sum "$WORK/h13_package_to_bundle.py" "$WORK/bundle_payload_identity.py" "$COMPILER/build/mil-hwxc"
for OP in sigmoid tanh; do
  printf '%s\n' 'program(1.3)' '[buildInfo = dict<string, string>({})]' '{' '  func main<ios18>(tensor<fp16, [1, 512, 1, 1]> x) {' "    tensor<fp16, [1, 512, 1, 1]> y = $OP(x = x)[name = string(\"y\")];" '  } -> (y);' '}' > "$WORK/model_$OP.mil"
  "$COMPILER/build/mil-hwxc" --target H13 --format anec --mil "$WORK/model_$OP.mil" --model-root "$WORK" --output "$WORK/pkg-$OP"
  echo "COMPILE_RC_$OP=$?"
  python3 - "$OP" "$COMMIT" <<'PY'
import hashlib, json, os, platform, sys
from pathlib import Path
op, commit = sys.argv[1], sys.argv[2]
work = Path("/tmp/jwm1-decoder-activations-build"); pkg = work / f"pkg-{op}"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
m = json.loads((pkg / "manifest.json").read_text())
print(op, "programs", len(m["programs"]), "operation", m["operation"], "encoder", m["encoder"], "taskDescriptors", m["taskDescriptors"], "constantBytes", m["constantBytes"], "constantOffset", m["constantOffset"], "anec_sha", sha(pkg / m["file"]), "inputs", m["inputs"], "outputs", m["outputs"])
source = {
  "artifact_format": "anec",
  "compiler_binary_sha256": sha("/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc"),
  "compiler_manifest_sha256": sha(pkg / "manifest.json"),
  "compiler_executable_source_commit": commit,
  "compiler_repository_receipt_commit": commit,
  "generation_command": f"$COMPILER/build/mil-hwxc --target H13 --format anec --mil $WORK/model_{op}.mil --model-root $WORK --output $WORK/pkg-{op}",
  "graph_source_sha256": sha(work / f"model_{op}.mil"),
  "hardware_executed": False,
  "payloads": {m["file"]: sha(pkg / m["file"])},
  "qualification_scope": "Explicit --format anec package structure and loader adaptation only; no device or compiler-wide qualification.",
  "schema": "mil-hwxc.h13-anec-package.v2",
  "target": "H13",
  "compiler_host_build": f"{platform.system()} {os.uname().release} {platform.machine()}",
}
(pkg / "source.json").write_text(json.dumps(source, indent=2) + "\n")
PY
  python3 "$WORK/h13_package_to_bundle.py" "$WORK/pkg-$OP" --out-dir "$WORK/bundle-$OP" --graph-source "$WORK/model_$OP.mil" --compiler-source "$COMPILER" --name "schema4-$OP-512" --source-repo mil-hwxc --source-commit "$COMMIT" --model "$OP-chw-512"
  sha256sum "$WORK/bundle-$OP/manifest.json" "$WORK/bundle-$OP/program-0.anec"
done
