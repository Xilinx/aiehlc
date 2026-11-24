#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# aiehlcrebuild.sh — Rebuild host+kernel ELFs from aout/ without re-running
# the MLIR pipeline. Compiles kernel and host directly in aout/worklocal/
# (artifacts in aout/worklocal/build/).
#
# Usage: script/aiehlcrebuild.sh [--aout-dir <path>] [--aie-version 5] [--platform baremetal]
#
# Default aout dir: <repo>/aout
###############################################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AOUT_DIR="${AIEHLC_ROOT}/aout"

# Defaults
aie_version="5"
platform="baremetal"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --aout-dir)
            AOUT_DIR="$2"
            shift 2
            ;;
        --aie-version)
            aie_version="$2"
            shift 2
            ;;
        --platform)
            platform="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--aout-dir <path>] [--aie-version 5] [--platform baremetal]"
            echo ""
            echo "Rebuild host+kernel ELFs from generated .cc files in aout/worklocal/."
            echo ""
            echo "Options:"
            echo "  --aout-dir <path>      Path to aout directory (default: <repo>/aout)"
            echo "  --aie-version <ver>    AIE version: 2 or 5 (default: 5)"
            echo "  --platform <plat>      baremetal or linux (default: baremetal)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Make AOUT_DIR absolute: hostcompile.sh cd's into worklocal before it uses
# WORKLOCAL_DIR again and before kc.sh looks for ../kernel_build_flags.sh.
if [ -d "${AOUT_DIR}" ]; then
    AOUT_DIR="$(cd "${AOUT_DIR}" && pwd)"
fi

# Validate aout directory
AOUT_WORKLOCAL="${AOUT_DIR}/worklocal"
if [ ! -d "${AOUT_WORKLOCAL}" ]; then
    echo "Error: aout/worklocal/ not found at ${AOUT_WORKLOCAL}"
    echo "Run the pipeline first (e.g. ./test dfschedule or aiehlc.sh) to generate code."
    exit 1
fi

if [ ! -f "${AOUT_WORKLOCAL}/host.cc" ]; then
    echo "Error: host.cc not found in ${AOUT_WORKLOCAL}"
    exit 1
fi

if [ ! -f "${AOUT_WORKLOCAL}/kernel.cc" ]; then
    echo "Error: kernel.cc not found in ${AOUT_WORKLOCAL}"
    exit 1
fi

echo "============================================"
echo "aiehlcrebuild: Rebuild from aout"
echo "============================================"
echo "  aout dir:     ${AOUT_DIR}"
echo "  worklocal:    ${AOUT_WORKLOCAL}"
echo "  AIE version:  ${aie_version}"
echo "  platform:     ${platform}"
echo ""

# Step 1: Compile kernel + host via hostcompile.sh, building in aout/worklocal/.
echo "=== Step 1: Compile kernel + host ==="
WORKLOCAL_DIR="${AOUT_WORKLOCAL}" AIE_VERSION="${aie_version}" PLATFORM="${platform}" \
    bash "${AIEHLC_ROOT}/script/hostcompile.sh"

# Step 2: Copy the built ELF to aout/main.elf
BUILD_ELF="${AOUT_WORKLOCAL}/build/host"
if [ -f "${BUILD_ELF}" ]; then
    cp -f "${BUILD_ELF}" "${AOUT_DIR}/main.elf"
    echo ""
    echo "============================================"
    echo "Rebuild complete."
    echo "  ${AOUT_DIR}/main.elf"
    echo "============================================"
    ls -lh "${AOUT_DIR}/main.elf"
else
    echo "Error: build/host not found after compilation"
    exit 1
fi
