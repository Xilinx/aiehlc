###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################
#!/usr/bin/env bash
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

GEN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script is located in: $GEN_SCRIPT_DIR"
pushd ${GEN_SCRIPT_DIR}
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs ./aieops.td -dialect=Aie -I /usr/local/include/ ${MLIR_INCLUDES} > ./aiedialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls ./aieops.td -dialect=Aie -I /usr/local/include/  ${MLIR_INCLUDES} > ./aiedialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-op-defs ./aieops.td -I /usr/local/include/  ${MLIR_INCLUDES} > ./aieop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls ./aieops.td -I /usr/local/include/  ${MLIR_INCLUDES} > ./aieop.h.inc

${LLVM_BIN}/mlir-tblgen -gen-typedef-defs ./aietypes.td --typedefs-dialect=Aie -I /usr/local/include/  ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aietype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls ./aietypes.td --typedefs-dialect=Aie -I /usr/local/include/  ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aietype.h.inc
popd
