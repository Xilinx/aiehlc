#!/usr/bin/env bash
# Build and run the control-plane reservation table unit tests.
# The table is C; the harness is C++ with extern "C". Compile the table as C,
# then link.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rt="$(cd "$here/.." && pwd)"
out="${TMPDIR:-/tmp}/res_test"
gcc -c -I "$rt" -x c "$rt/aie_runtime_resource.c" -o "${TMPDIR:-/tmp}/res_resource.o"
g++ -I "$rt" "$here/test_ctrl_resource.cpp" "${TMPDIR:-/tmp}/res_resource.o" -o "$out"
"$out"
