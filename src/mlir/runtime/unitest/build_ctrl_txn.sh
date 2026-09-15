#!/usr/bin/env bash
# Build and run the control-packet transaction translator host unit test.
#
# aie_runtime.c compiles host-native under -D__AIESIM__ with the cortexa78 BSP
# include path; we compile it with -ffunction-sections so the linker
# (--gc-sections) drops every function the test does not reference, leaving only
# the pure control-packet builders (no HW / driver symbols needed).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rt="$(cd "$here/.." && pwd)"
root="$(cd "$rt/../../.." && pwd)"
bsp="$root/thirdparty/arch/cortexa78_0/workspace/platform_baremetal/cortexa78_0/standalone_cortexa78_0/bsp/include"
out="${TMPDIR:-/tmp}/ctrl_txn_test"
obj="${TMPDIR:-/tmp}/ctrl_txn_aie_runtime.o"

g++ -std=c++17 -D__AIESIM__ -ffunction-sections -fdata-sections \
    -I "$bsp" -I "$root/include" -I "$rt" \
    -c "$rt/aie_runtime.c" -o "$obj"

g++ -std=c++17 -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -I "$bsp" -I "$rt" \
    "$here/test_ctrl_txn.cpp" "$obj" -o "$out"

"$out"
