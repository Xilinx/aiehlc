#!/usr/bin/env bash
# Build and run the conv2d auto-tiling model + trace->routing unit tests (no LLVM needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
OUT="${TMPDIR:-/tmp}/convautotile_unitest_$$"
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT
LIB=("$HERE/../conv_autotile.cpp" "$HERE/../conv_desugar.cpp" "$HERE/../kernel_trace.cpp" "$HERE/../kernel_static.cpp" "$HERE/../kernel_front.cpp"
     "$HERE/../trace_routing.cpp")
FLAGS=(-std=c++17 -O1 -Wall -Wextra -pthread -DAIEHLC_SOURCE_DIR="\"$REPO\"")
rc=0
for t in test_conv_autotile test_trace_routing; do
    g++ "${FLAGS[@]}" -o "$OUT/$t" "$HERE/$t.cpp" "${LIB[@]}"
    "$OUT/$t" "$REPO" || rc=1
done
exit $rc
