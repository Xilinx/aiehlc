###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script is located in: $SCRIPT_DIR"
MLIR_INCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mlir-include)
      MLIR_INCLUDES+=(" -I$2 "); shift 2;;
    --llvm-bin)
      LLVM_BIN="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

pushd ${SCRIPT_DIR}
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs ./aielinkops.td -dialect=AieLink -I /usr/local/include/ ${MLIR_INCLUDES} > ./aielinkdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls ./aielinkops.td -dialect=AieLink -I /usr/local/include/ ${MLIR_INCLUDES} > ./aielinkdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-op-defs ./aielinkops.td -I /usr/local/include/ ${MLIR_INCLUDES} > ./aielinkop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls ./aielinkops.td -I /usr/local/include/ ${MLIR_INCLUDES} > ./aielinkop.h.inc

${LLVM_BIN}/mlir-tblgen -gen-typedef-defs ./aielinktypes.td --typedefs-dialect=AieLink -I /usr/local/include/ ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aielinktype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls ./aielinktypes.td --typedefs-dialect=AieLink -I /usr/local/include/ ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aielinktype.h.inc
popd
