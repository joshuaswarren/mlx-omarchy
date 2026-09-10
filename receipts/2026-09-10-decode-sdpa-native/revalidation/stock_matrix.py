#!/usr/bin/env python3
import importlib.util
import json
import shutil
import sys
from pathlib import Path

root, python, wheel, output = map(Path, sys.argv[1:5])
source = root / "receipts/parity-baseline-20260908"
spec = importlib.util.spec_from_file_location("matrix_runner", source / "run-baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = root
output.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source / "capture-ids.py", output / "capture-ids.py")
runner.HERE = output
runner.EXTRA_MLX_ENV = {}
sys.argv = ["run-baseline.py", str(python), str(wheel), "1"]
runner.main()
summary = json.loads((output / "baseline-summary.json").read_text())
expected = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-decode-128": "c5be9207833d2a26",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
actual = {row["leg_id"]: row["generated_ids_sha256_16"] for row in summary["legs"]}
assert actual.keys() == expected.keys(), actual.keys()
for leg_id, digest in expected.items():
    assert actual[leg_id] == [digest], (leg_id, actual[leg_id], digest)
result = {
    "source_commit": "6c263b325215ef2555bf525ebb83b1da22ca63de",
    "driver": "/home/joshuawarren/stock-mesa/stock-icd.json",
    "expected": expected,
    "actual": actual,
    "all_match": True,
}
(output / "stock-digests.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
