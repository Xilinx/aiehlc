#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Regenerate every dialect's TableGen .inc files with the mlir-tblgen of the
# LLVM in $LLVM_INSTALL_DIR (source script/llvm_env.sh first). The .inc files
# are build artifacts that must match the LLVM the tree compiles against.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM="${LLVM_INSTALL_DIR:?source script/llvm_env.sh first}"
SRC="$(sed -n 's/^LLVM_MAIN_SRC_DIR:PATH=//p' "${LLVM}/CMakeCache.txt" 2>/dev/null)"
SRC="${SRC:-${LLVM}/../llvm}"
INC=(--mlir-include "${SRC}/../mlir/include" --mlir-include "${LLVM}/tools/mlir/include"
     --mlir-include "${SRC}/include" --mlir-include "${LLVM}/include")
rc=0
for g in src/mlir/mlirfront/gen.sh src/mlir/mlirfront/mlirlinkrelocate/gen.sh \
         src/mlir/mlirfront/tilinglinalg/routing/gen.sh src/mlir/mlirfront/tilinglinalg/routinghw/gen.sh \
         src/mlir/mlirfront/tilinglinalg/dataflowmap/dmap/gen.sh \
         src/mlir/mlirfront/tilinglinalg/dataflowmap/dmaphop/gen.sh \
         src/mlir/mlirfront/tilinglinalg/dataflowmap/dfscheblueprint/gen.sh \
         src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/gen.sh \
         src/mlir/mlirfront/frontend/aiegraph/gen.sh; do
    if ! out="$(bash "${ROOT}/${g}" "${INC[@]}" --llvm-bin "${LLVM}/bin/" 2>&1)"; then rc=1; fi
    errs="$(grep -E "error" <<<"${out}" | head -5)"
    if [ -n "${errs}" ]; then
        echo "[regen_tablegen] ${g}: FAILED"; echo "${errs}"; rc=1
    else
        echo "[regen_tablegen] ${g}: ok"
    fi
done
exit ${rc}
