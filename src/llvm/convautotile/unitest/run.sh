#!/usr/bin/env bash
# Build and run the conv2d auto-tiling model unit test (no LLVM needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
OUT="${TMPDIR:-/tmp}/convautotile_unitest_$$"
mkdir -p "$OUT"
g++ -std=c++17 -O1 -Wall -Wextra -pthread -DAIEHLC_SOURCE_DIR="\"$REPO\"" -o "$OUT/test_conv_autotile" \
    "$HERE/test_conv_autotile.cpp" "$HERE/../conv_autotile.cpp" "$HERE/../conv_desugar.cpp" "$HERE/../kernel_trace.cpp" "$HERE/../kernel_static.cpp" "$HERE/../kernel_front.cpp"
"$OUT/test_conv_autotile" "$REPO"
rc=$?
rm -rf "$OUT"
exit $rc
