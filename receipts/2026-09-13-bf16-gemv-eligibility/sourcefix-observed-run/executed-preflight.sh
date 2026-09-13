#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR=${STAGE_DIR:?}
BASE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
CANDIDATE="$HOME/src/mlx-bf16-gemv-sourcefix-4f90815c"
EXPECTED_BASE=6b1ac0296ba65a8e0075171ca9451e222ddff06b
EXPECTED_CANDIDATE=4f90815c056cbb29a2ff4e1a958856b85af6390a
EXPECTED_CANDIDATE_SHA=f05c64c9acb17d5280d5ea7a2f0644ca44f97d85ea01a51b4e291d7d07d9f656
EXPECTED_IDENTITY_SHA=e5a5f70943ca7054a117389bc0bb256e8627930ec57eb4ebdab9c70119821eeb
PY="$BASE/.work/venv-run/bin/python"
CONTROL_SITE="$BASE/.work/venv-run/lib/python3.14/site-packages"
CANDIDATE_SITE="$CANDIDATE/.work-sourcefix/candidate-site"
IDENTITY="$STAGE_DIR/package-runtime-identity.py"
RUNNER="$STAGE_DIR/sourcefix-continuation.sh"

for command_name in bash cat cmake cut date df env flock git grep hostname mkdir nproc python3 rm seq sha256sum sleep stat tee timeout uname; do
  command -v "$command_name" >/dev/null || { echo "FATAL missing command $command_name"; exit 2; }
done
for path in "$BASE/.git" "$CANDIDATE/.git" "$BASE/scripts/bench_decode.py" \
  "$BASE/scripts/bench_matrix.py" "$BASE/scripts/bench_matrix.json" "$PY" \
  "$CANDIDATE/.work-sourcefix/mlx/CMakeLists.txt" \
  "$CANDIDATE/.work-sourcefix/mlx/tests/omarchy/test_fused_chain.cpp" \
  "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e/config.json" \
  "$CONTROL_SITE/mlx" "$CANDIDATE_SITE/mlx" "$IDENTITY" "$RUNNER"; do
  [ -e "$path" ] || { echo "FATAL missing path $path"; exit 2; }
done
bash -n "$RUNNER"
[ "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED_BASE" ] || { echo "FATAL control source mismatch"; exit 3; }
[ "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$EXPECTED_CANDIDATE" ] || { echo "FATAL candidate source mismatch"; exit 3; }
[ -z "$(git -C "$CANDIDATE" status --porcelain --untracked-files=no)" ] || { echo "FATAL candidate tracked source dirty"; exit 3; }
[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" = "$EXPECTED_IDENTITY_SHA" ] || { echo "FATAL identity helper mismatch"; exit 3; }
shopt -s nullglob
control_wheels=("$BASE"/dist/mlx_omarchy-*+6b1ac029-*.whl)
candidate_wheels=("$CANDIDATE"/dist/mlx_omarchy-*+4f90815-*.whl)
shopt -u nullglob
[ "${#control_wheels[@]}" -eq 1 ] || { echo "FATAL expected one control wheel"; exit 4; }
[ "${#candidate_wheels[@]}" -eq 1 ] || { echo "FATAL expected one candidate wheel"; exit 4; }
CONTROL=${control_wheels[0]}
CANDIDATE_WHEEL=${candidate_wheels[0]}
[ "$(sha256sum "$CANDIDATE_WHEEL" | cut -d' ' -f1)" = "$EXPECTED_CANDIDATE_SHA" ] || { echo "FATAL candidate wheel mismatch"; exit 3; }
env -u LD_LIBRARY_PATH PYTHONPATH="$CONTROL_SITE" "$PY" "$IDENTITY" --static \
  "$CONTROL" "$PY" "$CONTROL_SITE" "+6b1ac029" >/dev/null
env -u LD_LIBRARY_PATH PYTHONPATH="$CANDIDATE_SITE" "$PY" "$IDENTITY" --static \
  "$CANDIDATE_WHEEL" "$PY" "$CANDIDATE_SITE" "+4f90815" >/dev/null
env -u LD_LIBRARY_PATH PYTHONPATH="$CANDIDATE_SITE" "$PY" - "$BASE/scripts/bench_decode.py" <<'PY'
import ast
import importlib.util
import pathlib
import sys
assert importlib.util.find_spec("mlx") is not None
assert importlib.util.find_spec("mlx_lm") is not None
ast.parse(pathlib.Path(sys.argv[1]).read_text())
PY
printf 'NON_GPU_PREFLIGHT_OK source=%s candidate_wheel_sha=%s python=%s control_site=%s candidate_site=%s\n' \
  "$EXPECTED_CANDIDATE" "$EXPECTED_CANDIDATE_SHA" "$PY" "$CONTROL_SITE" "$CANDIDATE_SITE"
