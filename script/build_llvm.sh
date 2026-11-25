#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Configure + build the local LLVM/Clang/MLIR that aiehlc builds against.
#
#   bash script/build_llvm.sh [--src DIR] [--build DIR] [--update] [--jobs N]
#
# Defaults: source $AIEHLC_LLVM_SRC (or /scratch/staff/$USER/llvm-project), build
# dir <src>/build. Same configuration as the shared build aiehlc used before
# (Release + assertions + RTTI, projects mlir;clang), so LLVM_INSTALL_DIR can point
# at the build dir directly. Then: source script/llvm_env.sh (bash) or
# source script/llvm_env.csh (tcsh).
set -euo pipefail

SRC="${AIEHLC_LLVM_SRC:-/scratch/staff/${USER}/llvm-project}"
BUILD=""
UPDATE=0
JOBS="$(nproc)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src) SRC="$2"; shift 2 ;;
        --build) BUILD="$2"; shift 2 ;;
        --update) UPDATE=1; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "usage: $0 [--src DIR] [--build DIR] [--update] [--jobs N]" >&2; exit 2 ;;
    esac
done
BUILD="${BUILD:-${SRC}/build}"

if [ ! -d "${SRC}/llvm" ]; then
    echo "[build_llvm] cloning llvm-project into ${SRC}"
    git clone --filter=blob:none https://github.com/llvm/llvm-project.git "${SRC}"
elif [ "${UPDATE}" -eq 1 ]; then
    git -C "${SRC}" pull --ff-only
fi
echo "[build_llvm] llvm-project $(git -C "${SRC}" log -1 --format='%h %cd')"

LAUNCHER=()
if command -v ccache >/dev/null 2>&1; then
    LAUNCHER=(-DLLVM_CCACHE_BUILD=ON)
fi

cmake -G Ninja -S "${SRC}/llvm" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="Native;AArch64" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_BUILD_TESTS=OFF -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DMLIR_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=8 \
    "${LAUNCHER[@]}"

ninja -C "${BUILD}" -j "${JOBS}"
echo "[build_llvm] done: ${BUILD}"
echo "[build_llvm] now: source script/llvm_env.sh   (tcsh: source script/llvm_env.csh)"
