#!/usr/bin/env bash
# install.sh <dist-dir> <venv-dir>
# Install the single wheel in <dist-dir> into <venv-dir>, borrowing the
# runtime dependencies already resolved in the grouped-base run venv.
set -euo pipefail
DIST=${1:?dist dir}
VENV=${2:?venv dir}
SRC=/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74
SP_RUN=$SRC/.work/venv-run/lib/python3.14/site-packages
shopt -s nullglob
wheels=("$DIST"/mlx_omarchy-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one wheel in $DIST, found ${#wheels[@]}" >&2
  exit 1
fi
WHEEL=${wheels[0]}
if [[ ! -x $VENV/bin/python ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --force-reinstall --no-deps "$WHEEL"
SP=$VENV/lib/python3.14/site-packages
python3 - "$SP_RUN" "$SP" <<'PY'
import os, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
allow = {
    "mlx_lm", "transformers", "tokenizers", "huggingface_hub", "safetensors",
    "regex", "tqdm", "yaml", "PyYAML", "packaging", "filelock", "requests",
    "numpy", "jinja2", "markupsafe", "MarkupSafe", "certifi",
    "charset_normalizer", "idna", "urllib3", "fsspec", "hf_xet", "typer",
    "click", "shellingham", "rich", "pygments", "markdown_it", "mdurl",
    "anyio", "sniffio", "httpx", "httpcore", "h11",
}
linked = []
for p in src.iterdir():
    stem = p.name.split("-")[0].split(".")[0]
    if p.name.startswith("mlx") and not p.name.startswith("mlx_lm"):
        continue
    if p.name in allow or stem in allow:
        dest = dst / p.name
        if not dest.exists():
            os.symlink(p, dest)
            linked.append(p.name)
print("linked", len(linked), "dependency paths")
PY
"$VENV/bin/python" - <<'PY'
import hashlib, pathlib
import mlx, mlx.core as mx
lib = pathlib.Path(mlx.__file__).resolve().parent / "lib" / "libmlx.so"
data = lib.read_bytes()
print("mx", mlx.__version__)
print("libmlx.so sha256", hashlib.sha256(data).hexdigest())
print("profiler literal", b"MLX_OMARCHY_GPU_PROFILE" in data)
if b"MLX_OMARCHY_GPU_PROFILE" not in data:
    raise SystemExit("installed libmlx.so lacks the profiler literal")
PY
sha256sum "$WHEEL"
