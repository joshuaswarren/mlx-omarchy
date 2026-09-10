#!/usr/bin/env bash
# mlx-omarchy installer for Omarchy on Apple M1 (Asahi Linux, Honeykrisp Vulkan).
#
#   curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
#   bash install.sh --uninstall
#
# Writes only under $HOME (venv, launchers, one .desktop entry) and installs the
# two runtime packages the wheel needs (lapack, blas) through pacman. It never
# replaces Mesa, touches Hyprland, or edits Omarchy files.
set -euo pipefail

REPO=joshuaswarren/mlx-omarchy
VERSION="${MLX_OMARCHY_VERSION:-v0.4.1}"
PREFIX="${MLX_OMARCHY_HOME:-$HOME/.local/share/mlx-omarchy}"
VENV="$PREFIX/venv"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
MLX_LM_VERSION=0.31.3
TRANSFORMERS_VERSION=5.16.1

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  rm -rf "$PREFIX" "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$APPS/mlx-omarchy-demo.desktop"
  say "mlx-omarchy removed. Model downloads stay in ~/.cache/huggingface; delete them yourself if you want the space back."
  exit 0
fi

# 1. Hardware and interpreter checks. The release wheel is cp314 linux_aarch64
#    and is verified on the M1 (apple,t8103) only.
[[ "$(uname -m)" == aarch64 ]] || die "mlx-omarchy runs on Apple Silicon (aarch64); this machine is $(uname -m)."
if [[ -r /proc/device-tree/compatible ]] && ! tr '\0' ' ' </proc/device-tree/compatible | grep -q 'apple,t8103'; then
  echo "warning: this is not an Apple M1 (t8103). Only the M1 is verified; later chips are untested." >&2
fi
command -v python3 >/dev/null || die "python3 is missing."
python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))' \
  || die "Python 3.14 is required (found $(python3 --version)); the wheel is built for cp314."

# 2. Runtime packages. omarchy-pkg-add is Omarchy's own helper; plain pacman
#    is the fallback on any other Asahi Arch install.
say "Installing runtime packages (lapack, blas)"
if command -v omarchy-pkg-add >/dev/null; then
  omarchy-pkg-add lapack blas
else
  sudo pacman -S --needed --noconfirm lapack blas
fi

# 3. Download the release wheel and verify it against the SHA256SUMS asset.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
base="https://github.com/$REPO/releases/download/$VERSION"
say "Fetching $VERSION checksums"
curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS"
wheel="$(grep -o 'mlx_omarchy-[^ ]*cp314-cp314-linux_aarch64\.whl' "$tmp/SHA256SUMS" | head -n 1)"
[[ -n "$wheel" ]] || die "no aarch64 wheel listed in $base/SHA256SUMS"
say "Fetching $wheel"
curl -fsSL "$base/$wheel" -o "$tmp/$wheel"
(cd "$tmp" && sha256sum -c --ignore-missing --quiet SHA256SUMS) || die "checksum mismatch for $wheel"

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
chmod +x "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo"
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
