#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Build the netboot test app: test/build.sh [seconds] -> build/nbspin<seconds>.elf
# A plain BSP app (stock lscript.ld, app memory at 0x0): what netboot runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
BSP="${REPO}/thirdparty/arch/cortexa78_0/bsp"
CC="${CROSS:-aarch64-none-elf-}gcc"
S="${1:-20}"
OUT="${HERE}/../build/nbspin${S}.elf"
command -v "${CC}" >/dev/null || { echo "${CC} not on PATH (source Vitis settings64.sh)" >&2; exit 1; }
"${CC}" -O2 -mcpu=cortex-a78 -DSDT -DSPIN_S="${S}" -I"${BSP}/include" "${HERE}/nbspin.c" -o "${OUT}" \
    --specs=nosys.specs -Wl,--defsym,end=__bss_end__ -Wl,-T -Wl,"${REPO}/thirdparty/arch/cortexa78_0/lscript.ld" \
    -L"${BSP}/lib" -Wl,--start-group,-lxil,-lxilstandalone,-lxiltimer,-lgcc,-lc,--end-group
echo "built ${OUT}"
