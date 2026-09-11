#!/usr/bin/env bash
# PythonHostParity: build one CPython flavor on jwm1 (reniced, CPU-only, no GPU).
# usage: build_python.sh <version>-<flavor>   flavor: base | pgo
set -euo pipefail
tag="$1"
ver="${tag%%-*}"; flavor="${tag#*-}"
SRC="$HOME/opt/cpython-src"
PREFIX="$HOME/opt/$tag"
LOG="$HOME/opt/$tag.build.log"

renice -n 19 -p $$ >/dev/null || true
echo "[build_python] start $tag $(date -u +%FT%TZ) pid $$"

cd "$SRC"
[ -d "Python-$ver" ] || tar -xf "Python-$ver.tgz"
cd "Python-$ver"
make distclean >/dev/null 2>&1 || true

if [ "$flavor" = pgo ]; then
  ./configure --prefix="$PREFIX" --enable-optimizations --with-lto \
    --with-ensurepip=no >"$LOG" 2>&1
else
  ./configure --prefix="$PREFIX" --with-ensurepip=no >"$LOG" 2>&1
fi
make -j6 >>"$LOG" 2>&1
make install >>"$LOG" 2>&1

major_minor="$(echo "$ver" | cut -d. -f1-2)"
"$PREFIX/bin/python$major_minor" -VV
echo "[build_python] done $tag $(date -u +%FT%TZ)"
