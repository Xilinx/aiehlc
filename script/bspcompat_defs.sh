#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# bspcompat_defs <bsp include dir>
#
# Prints -D flags that paper over a standalone BSP whose xparameters.h uses the
# split-layout names (2026-11 regenerated A72 workspace BSP: only
# XPAR_CPU_TIMESTAMP_CLK_FREQ) while xtime_l.h, from the secondary
# psv_cortexa72_0/bsp tree, still expands COUNTS_PER_SECOND to the per-CPU
# XPAR_CPU_CORTEXA72_0_TIMESTAMP_CLK_FREQ. The value is the BSP's own number,
# not the other macro's name, so the secondary xparameters.h defining the same
# number is an identical (legal) redefinition. Keyed on the header's contents,
# not on the AIE generation -- but callers only apply it to the A72 (Gen1/2)
# BSP, since xparameters.h does not name its CPU. Sourced by aiehlc.sh (front end) and hostcompile.sh
# (host build); skill bspheadergen2 section 3.
bspcompat_defs() {
    local xp="$1/xparameters.h" val
    [ -f "$xp" ] || return 0
    grep -q "XPAR_CPU_CORTEXA72_0_TIMESTAMP_CLK_FREQ" "$xp" && return 0
    val=$(sed -n 's/^#define[[:space:]]\+XPAR_CPU_TIMESTAMP_CLK_FREQ[[:space:]]\+\([0-9]\+\).*/\1/p' "$xp" | head -1)
    [ -n "$val" ] && echo "-DXPAR_CPU_CORTEXA72_0_TIMESTAMP_CLK_FREQ=${val}"
    return 0
}
