#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Fetch and build the LLVM/MLIR/clang aiehlc links against (doc/build.md:
# LLVM 19.1.4, RTTI on, mlir+clang).
#   bash script/llvm/build.sh [LLVM_SRC_DIR]
# Env: LLVM_TAG (default llvmorg-19.1.4), LLVM_JOBS (default nproc),
#      LLVM_TARGETS (default "Native;NVPTX;AMDGPU").
# Result: <LLVM_SRC_DIR>/build -- pass it as LLVM_INSTALL_DIR
#      (cmake -DLLVM_INSTALL_DIR=... or export before `source script/setup.sh`).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLVM_TAG="${LLVM_TAG:-llvmorg-19.1.4}"
LLVM_SRC="${1:-$(dirname "$AIEHLC_DIR")/llvm-project}"
LLVM_JOBS="${LLVM_JOBS:-$(nproc)}"
LLVM_TARGETS="${LLVM_TARGETS:-Native;NVPTX;AMDGPU}"

# Report an unwritable target up front instead of a git/cmake permission error.
for want in "$LLVM_SRC" "$LLVM_SRC/build"; do
    probe="$want"
    while [ ! -e "$probe" ]; do probe="$(dirname "$probe")"; done
    if [ ! -w "$probe" ]; then
        echo "ERROR: cannot build LLVM into $LLVM_SRC: no write permission on $probe." >&2
        echo "       Pass a writable directory: bash script/llvm/build.sh <dir>" >&2
        echo "       (from cmake: -DAIEHLC_LLVM_SRC_DIR=<dir>)" >&2
        exit 1
    fi
done

if [ ! -d "$LLVM_SRC/llvm" ]; then
    git clone --depth 1 --branch "$LLVM_TAG" https://github.com/llvm/llvm-project.git "$LLVM_SRC"
fi

CCACHE_OPT=OFF
command -v ccache >/dev/null 2>&1 && CCACHE_OPT=ON
GEN=()
command -v ninja >/dev/null 2>&1 && GEN=(-G Ninja)

cmake "${GEN[@]}" -S "$LLVM_SRC/llvm" -B "$LLVM_SRC/build" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DMLIR_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF \
    -DLLVM_CCACHE_BUILD="$CCACHE_OPT"
cmake --build "$LLVM_SRC/build" -j"$LLVM_JOBS"
echo "LLVM ready: export LLVM_INSTALL_DIR=$LLVM_SRC/build"
