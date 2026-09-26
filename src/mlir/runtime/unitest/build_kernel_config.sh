#!/usr/bin/env bash
# Build and run the KERNELCONFIGOFFLOAD BD/lock/channel MMIO encoder host test.
#
# The test validates a standalone C encoder (src/mlir/runtime/aie_kernel_runtime.h)
# that produces AIE2PS core-tile DMA buffer-descriptor / lock / channel-start
# register words, by diffing it against the GOLDEN words the real aie-rt driver
# (XAie_DmaWriteBd / XAie_LockSetValue / XAie_DmaChannelSetStartQueue) emits.
#
# The golden reference runs host-native: the whole aie-rt gen5 driver is
# compiled for x86 with the DEBUG IO backend (-D__AIEDEBUG__, no HW poke), and
# the writes are captured via XAie_StartTransaction(DISABLE_AUTO_FLUSH) +
# XAie_ExportSerializedTransaction (structured Write32 / BlockWrite32 ops).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rt="$(cd "$here/.." && pwd)"
root="$(cd "$rt/../../.." && pwd)"

drv="$root/thirdparty/alib/aie-rt/driver"
drvsrc="$drv/src"
inc="$root/include"

out="${TMPDIR:-/tmp}/kernel_config_test"
lib="${TMPDIR:-/tmp}/xaie_x86/libxaiengine_x86.a"
libdir="$(dirname "$lib")"

# Collect every driver include dir once.
# "$rt" is src/mlir/runtime: aie_kernel_runtime.h + the kernel_tm.h it includes.
incflags=(-I "$drv/include" -I "$drv/include/xaiengine" -I "$inc" -I "$rt")
while IFS= read -r d; do incflags+=(-I "$d"); done < <(find "$drvsrc" -type d)

# 1. Build the x86 driver static lib (debug backend) if missing / stale.
if [ ! -f "$lib" ] || [ "${FORCE_XAIE_REBUILD:-0}" = 1 ]; then
    echo "[build] compiling aie-rt gen5 driver for x86 (debug backend) ..."
    mkdir -p "$libdir/obj"
    rm -f "$libdir"/obj/*.o
    while IFS= read -r f; do
        case "$f" in */liburing/*) continue;; esac
        obj="$libdir/obj/$(echo "${f#$drvsrc/}" | tr '/' '_').o"
        gcc -std=c11 -D_POSIX_C_SOURCE=200809 -D_DEFAULT_SOURCE -D__AIEDEBUG__ \
            -ffunction-sections -fdata-sections -w "${incflags[@]}" \
            -c "$f" -o "$obj"
    done < <(find "$drvsrc" -name '*.c')
    ar rcs "$lib" "$libdir"/obj/*.o
    echo "[build] libxaiengine_x86.a: $(stat -c%s "$lib") bytes"
fi

# 2. Compile + link the test against the x86 driver lib.
g++ -std=c++17 -D__AIEDEBUG__ -w \
    "${incflags[@]}" \
    "$here/test_kernel_config.cpp" "$lib" -lpthread -o "$out"

# 3. Run.
"$out"
