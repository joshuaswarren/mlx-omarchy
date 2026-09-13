#!/usr/bin/env bash
# mlx-omarchy installer for Omarchy on Apple M1 (Asahi Linux, Honeykrisp Vulkan).
#
#   curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
#   bash install.sh --ane
#   bash install.sh --uninstall
#
# The default install writes only under $HOME, except for runtime packages
# installed through pacman. --ane also provisions host-global ANE ownership.
set -euo pipefail

REPO=joshuaswarren/mlx-omarchy
VERSION="${MLX_OMARCHY_VERSION:-v0.4.2}"
PREFIX="${MLX_OMARCHY_HOME:-$HOME/.local/share/mlx-omarchy}"
VENV="$PREFIX/venv"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
MLX_LM_VERSION=0.31.3
TRANSFORMERS_VERSION=5.16.1
ANE=0
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  --ane) ANE=1 ;;
  --uninstall)
    rm -rf "$PREFIX" "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-info" "$APPS/mlx-omarchy-demo.desktop"
    say "mlx-omarchy removed. Model downloads stay in ~/.cache/huggingface; delete them yourself if you want the space back."
    exit 0
    ;;
  "") ;;
  *) die "unknown option: $1 (supported: --ane, --uninstall)" ;;
esac

# 1. Hardware and interpreter checks. The release wheel is cp314 linux_aarch64
#    and is verified on the M1 (apple,t8103) only.
if (( ANE )); then
  [[ -c /dev/accel/accel0 ]] || die "ANE installation requires /dev/accel/accel0."
  command -v sudo >/dev/null || die "ANE installation requires sudo."
  command -v systemd-tmpfiles >/dev/null || die "ANE installation requires systemd-tmpfiles."
  getent group render >/dev/null || die "ANE installation requires the render group."
fi
[[ "$(uname -m)" == aarch64 ]] || die "mlx-omarchy runs on Apple Silicon (aarch64); this machine is $(uname -m)."
if [[ -r /proc/device-tree/compatible ]] && ! tr '\0' ' ' </proc/device-tree/compatible | grep -q 'apple,t8103'; then
  echo "warning: this is not an Apple M1 (t8103). Only the M1 is verified; later chips are untested." >&2
fi
command -v python3 >/dev/null || die "python3 is missing."
python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))' \
  || die "Python 3.14 is required (found $(python3 --version)); the wheel is built for cp314."

# 2. Runtime packages. omarchy-pkg-add is Omarchy's own helper; plain pacman
#    is the fallback on any other Asahi Arch install.
# openblas provides libopenblas.so.0, which the wheel's BLAS calls resolve
# against at import time. Without it `import mlx.core` fails on a fresh install.
say "Installing runtime packages (lapack, blas, openblas)"
if command -v omarchy-pkg-add >/dev/null; then
  omarchy-pkg-add lapack blas openblas
else
  sudo pacman -S --needed --noconfirm lapack blas openblas
fi

# 3. Download the release wheel and verify it against the SHA256SUMS asset.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
base="${MLX_OMARCHY_RELEASE_BASE:-https://github.com/$REPO/releases/download/$VERSION}"
say "Fetching $VERSION checksums"
curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS"
wheel="$(grep -o 'mlx_omarchy-[^ ]*cp314-cp314-linux_aarch64\.whl' "$tmp/SHA256SUMS" | head -n 1)"
[[ -n "$wheel" ]] || die "no aarch64 wheel listed in $base/SHA256SUMS"
say "Fetching $wheel"
curl -fsSL "$base/$wheel" -o "$tmp/$wheel"
(cd "$tmp" && sha256sum -c --ignore-missing --quiet SHA256SUMS) || die "checksum mismatch for $wheel"

if (( ANE )); then
  ane_conf="$tmp/mlx-omarchy-ane.conf"
  python3 - "$tmp/$wheel" "$ane_conf" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel, destination = sys.argv[1:]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    configs = [name for name in names if name.endswith("lib/tmpfiles.d/mlx-omarchy-ane.conf")]
    workers = [name for name in names if name.endswith("bin/mlx-omarchy-ane-worker")]
    if len(configs) != 1 or len(workers) != 1:
        raise SystemExit("wheel does not contain exactly one ANE worker and tmpfiles policy")
    Path(destination).write_bytes(archive.read(configs[0]))
PY
  say "Provisioning host-global ANE ownership"
  sudo install -D -m0644 "$ane_conf" /usr/lib/tmpfiles.d/mlx-omarchy-ane.conf
  sudo systemd-tmpfiles --create /usr/lib/tmpfiles.d/mlx-omarchy-ane.conf

  install_user="${SUDO_USER:-$(id -un)}"
  render_gid="$(getent group render | cut -d: -f3)"
  if [[ " $(id -G "$install_user") " != *" $render_gid "* ]]; then
    sudo usermod -aG render "$install_user"
    echo "note: $install_user was added to render; open a new login session before using ANE."
  fi

  sudo python3 - <<'PY'
import os
import stat

device = os.stat("/dev/accel/accel0", follow_symlinks=False)
if not stat.S_ISCHR(device.st_mode):
    raise SystemExit("/dev/accel/accel0 is not a character device")
expected = (
    ("/run/lock/mlx-omarchy-ane", stat.S_ISDIR, 0o750, False),
    ("/run/lock/mlx-omarchy-ane/device.lock", stat.S_ISREG, 0o660, True),
    ("/run/lock/mlx-omarchy-ane/quarantine", stat.S_ISREG, 0o660, True),
)
for path, type_check, mode, single_link in expected:
    status = os.stat(path, follow_symlinks=False)
    valid = (
        type_check(status.st_mode)
        and status.st_uid == 0
        and status.st_gid == device.st_gid
        and stat.S_IMODE(status.st_mode) == mode
        and (not single_link or status.st_nlink == 1)
    )
    if not valid:
        raise SystemExit(f"invalid ANE ownership path: {path}")
    print(f"  {status.st_uid}:{status.st_gid} {mode:o} {path}")
PY
fi

# 4. Private venv. Nothing is installed into the system Python.
say "Creating $VENV"
mkdir -p "$PREFIX" "$BIN" "$APPS"
python3 -m venv --clear "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$tmp/$wheel"
# mlx-lm declares a dependency on upstream mlx, which provides the same module
# and would conflict, so it is installed without dependencies and its real
# runtime dependencies are pinned explicitly.
"$VENV/bin/pip" install --quiet --no-deps "mlx-lm==$MLX_LM_VERSION"
"$VENV/bin/pip" install --quiet "transformers[sentencepiece]==$TRANSFORMERS_VERSION" numpy protobuf pyyaml jinja2 huggingface_hub

# 5. Demo and launchers.
say "Installing launchers into $BIN"
curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/demo/chat.py" -o "$PREFIX/chat.py"
cat >"$BIN/mlx-omarchy" <<EOF
#!/usr/bin/env bash
# Python interpreter with mlx-omarchy and mlx-lm installed.
exec "$VENV/bin/python" "\$@"
EOF
cat >"$BIN/mlx-omarchy-demo" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$PREFIX/chat.py" "\$@"
EOF
INFO=$("$VENV/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-info"), os.X_OK)))')
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$INFO" >"$BIN/mlx-omarchy-info"
chmod +x "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-info"
if command -v omarchy-launch-floating-terminal-with-presentation >/dev/null; then
  cat >"$APPS/mlx-omarchy-demo.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MLX Chat (Apple GPU)
Comment=Chat with a local LLM on the Apple GPU via mlx-omarchy
Exec=omarchy-launch-floating-terminal-with-presentation $BIN/mlx-omarchy-demo
Icon=utilities-terminal
Categories=Development;Utility;
EOF
  # The Omarchy shell scans desktop entries at startup; ask it to rescan so
  # the entry shows up in the launcher (Super+Space) without a re-login.
  omarchy-menu refresh >/dev/null 2>&1 || true
fi

# 6. Smoke test on the real GPU: import, device, one matmul.
say "Smoke test"
"$VENV/bin/python" - <<'EOF'
import mlx.core as mx
info = mx.device_info()
a = mx.random.normal((256, 256))
b = mx.random.normal((256, 256))
c = (a @ b).sum()
mx.eval(c)
assert mx.isfinite(c).item(), "matmul produced a non-finite result"
print(f"  device: {info.get('device_name', info)}")
print(f"  mlx-omarchy {mx.__version__}: matmul OK")
EOF

say "Done."
echo "  Run the demo:        mlx-omarchy-demo      (also in the Omarchy app launcher as 'MLX Chat')"
echo "  Use in your scripts: mlx-omarchy your_script.py   (import mlx.core as mx)"
echo "  Remove everything:   bash install.sh --uninstall"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "  note: $BIN is not on your PATH in this shell; open a new terminal or add it." ;; esac
