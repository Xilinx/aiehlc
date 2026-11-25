#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
set -e

# Parse arguments:
#   'rebuild'/'-rebuild' -> force steps 2-3 (cmake/make/test)
#   'reboot'/'-reboot'   -> full reboot/power-cycle HW run (default is nonreboot/xsdb TCP connect)
REBUILD=0
REBOOT=0
for arg in "$@"; do
    if [[ "$arg" == "rebuild" || "$arg" == "-rebuild" ]]; then
        REBUILD=1
    elif [[ "$arg" == "reboot" || "$arg" == "-reboot" ]]; then
        REBOOT=1
    fi
done

echo "=== Step 1: Source environment ==="
pushd .
source ../../../../../../script/setup.sh --path-set-only
popd

pushd .
source ../../../../../../script/aiehlc.sh --aielib-only --aie-version 5
popd

if [ "${REBUILD}" -eq 1 ]; then
    echo "=== Step 2: Build unitest ==="
    if [ ! -d "./build" ]; then
        mkdir -p ./build
        pushd ./build
        cmake ..
    else
        pushd ./build
    fi
    make -j4
    popd

    echo "=== Step 3: Run unitest (generate host.cc + kernel.cc) ==="
    pushd ./build/
    ./test
    popd
else
    echo "=== Step 2+3: Skipped (pass 'rebuild' or '-rebuild' to regenerate host.cc + kernel.cc) ==="
fi

echo "=== Step 3b: Generate readable schedule view (schedule_view.json + host_schedule.html) ==="
if [ -f "$(pwd)/build/worklocal/dfscheduleprovenancemap.json" ]; then
    python3 ../../../../../../src/tool/debug/schedule_view.py "$(pwd)/build/worklocal" || \
        echo "warning: schedule_view.py failed (non-fatal)"
else
    echo "=== Step 3b: Skipped (no dfscheduleprovenancemap.json in build/worklocal) ==="
fi

echo "=== Step 4: Compile host + kernel (hostcompile.sh) ==="
REBUILD_ARG=""
[ "${REBUILD}" -eq 1 ] && REBUILD_ARG="rebuild"
WORKLOCAL_DIR="$(pwd)/build/worklocal" source ../../../../../../script/hostcompile.sh

source ~/palmtest/envlocal.sh
if [ "${REBOOT}" -eq 1 ]; then
    echo "=== Step 5: Run on HW (apppaltest, full reboot/power-cycle) ==="
    apppaltest.py ./build/worklocal/build/host
else
    echo "=== Step 5: Run on HW (apppaltest, xsdb TCP connect via -nonreboot; pass 'reboot' to power-cycle) ==="
    apppaltest.py -nonreboot ./build/worklocal/build/host
fi

echo "=== Pipeline completed successfully ==="