#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
out="$1"
gen="${2:-5}"
src="$root/thirdparty/alib/aie-rt/driver/src"
inc="$root/thirdparty/alib/aie-rt/driver/internal"
rt="$root/src/mlir/runtime"
lib_dir="$root/build/aiert_host"
lib="$lib_dir/libxaiengine.a"

if [ ! -f "$lib" ]; then
    echo "[ctrlpkt] building x86 aie-rt -> $lib"
    mkdir -p "$lib_dir/obj"
    incs=$(find "$src" -maxdepth 2 -mindepth 1 -type d | grep -v 'liburing\|swig' | sed 's/^/-I/' | tr '\n' ' ')
    codegen_inc=""
    [ -n "${XILINX_VITIS:-}" ] && codegen_inc="-I${XILINX_VITIS}/aietools/include/drivers/aiengine/aie_codegen_inc"
    for f in $(find "$src" -mindepth 2 -maxdepth 3 -name '*.c' | grep -v '/liburing/\|/uc_driver/'); do
        o="$lib_dir/obj/$(echo "${f#$src/}" | tr '/' '_' | sed 's/\.c$/.o/')"
        # shellcheck disable=SC2086
        gcc -O1 -std=c11 -D_POSIX_C_SOURCE=200809 -D_DEFAULT_SOURCE -DXAIE_FEATURE_PRIVILEGED_ENABLE \
            -DXAIE_AIG_SOURCE -fPIC -I"$src" -I"$inc" $incs $codegen_inc -w -c "$f" -o "$o"
    done
    ar rcs "$lib" "$lib_dir"/obj/*.o
fi

gcc -O2 -std=gnu11 -Wall -DAIE_GEN="$gen" -I"$rt" -I"$root/include" -I"$inc" -I"$inc/xaiengine" \
    -o "$out" \
    "$root/src/tool/ctrlpkt/aiehlc_ctrlpkt.c" "$root/src/tool/ctrlpkt/aiehlc_ctrlpkt_sites.c" \
    "$root/src/tool/ctrlpkt/ctrlpkt_dbg.c" \
    "$rt/aie_ctrlpkt_encode.c" "$rt/aie_runtime_xaie_seq.c" "$rt/aie_runtime_control_plan.c" \
    "$rt/aie_runtime_resource.c" "$lib" -lm
