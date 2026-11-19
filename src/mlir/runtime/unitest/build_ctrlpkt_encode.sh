#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rt="$(cd "$here/.." && pwd)"
out="${TMPDIR:-/tmp}/cpe_test"
gcc -c -I "$rt" -x c "$rt/aie_ctrlpkt_encode.c" -o "${TMPDIR:-/tmp}/cpe_encode.o"
g++ -I "$rt" "$here/test_ctrlpkt_encode.cpp" "${TMPDIR:-/tmp}/cpe_encode.o" -o "$out"
"$out"
