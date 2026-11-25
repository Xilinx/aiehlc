###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Source (bash/zsh): point aiehlc's builds at the local LLVM/Clang/MLIR build.
#   source script/llvm_env.sh            # $AIEHLC_LLVM_SRC or /scratch/staff/$USER/llvm-project
# Build it first with script/build_llvm.sh. tcsh: source script/llvm_env.csh
_llvm_src="${AIEHLC_LLVM_SRC:-/scratch/staff/${USER}/llvm-project}"
export AIEHLC_LLVM_BUILD="${AIEHLC_LLVM_BUILD:-${_llvm_src}/build}"
if [ ! -x "${AIEHLC_LLVM_BUILD}/bin/mlir-tblgen" ]; then
    echo "[llvm_env] WARNING: ${AIEHLC_LLVM_BUILD} has no bin/mlir-tblgen -- run script/build_llvm.sh"
fi
export LLVM_INSTALL_DIR="${AIEHLC_LLVM_BUILD}"
export LLVM_DIR="${AIEHLC_LLVM_BUILD}/lib/cmake/llvm"
export MLIR_DIR="${AIEHLC_LLVM_BUILD}/lib/cmake/mlir"
export Clang_DIR="${AIEHLC_LLVM_BUILD}/lib/cmake/clang"
case ":${PATH}:" in *":${AIEHLC_LLVM_BUILD}/bin:"*) ;; *) export PATH="${AIEHLC_LLVM_BUILD}/bin:${PATH}" ;; esac
unset _llvm_src
