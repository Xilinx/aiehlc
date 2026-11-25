###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
#!/usr/bin/env bash

GEN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC=$GEN_SCRIPT_DIR/inc
TD=$GEN_SCRIPT_DIR/td

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mlir-include)
      MLIR_INCLUDES+=(" -I$2 "); shift 2;;
    --llvm-bin)
      LLVM_BIN="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [ ${#MLIR_INCLUDES[@]} -eq 0 ]; then
  echo "No --mlir-include provided, using default."
  MLIR_INCLUDES=("-I/Users/llvm/llvm-project/mlir/include/")
fi
if [ ${#LLVM_BIN[@]} -eq 0 ]; then
  echo "No LLVM_BIN provided, using default."
  LLVM_BIN=("/Users/llvm/llvm-project/build/bin/")
fi

echo $MLIR_INCLUDES
echo "Script is located in: $GEN_SCRIPT_DIR"
mkdir -p "$INC"
pushd ${GEN_SCRIPT_DIR}
echo ${LLVM_BIN}
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/aiegraphtype.td -dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/aiegraphdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/aiegraphtype.td -dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/aiegraphtype.td -typedefs-dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphtype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/aiegraphtype.td -typedefs-dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphtype.h.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-defs $TD/aiegraphattr.td -attrdefs-dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphattr.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-decls $TD/aiegraphattr.td -attrdefs-dialect=aiegraph -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphattr.h.inc
${LLVM_BIN}/mlir-tblgen --gen-enum-decls $TD/aiegraphattr.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphenums.h.inc
${LLVM_BIN}/mlir-tblgen --gen-enum-defs $TD/aiegraphattr.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/aiegraphenums.cc.inc

${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/aiegraphop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/aiegraphop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/aiegraphop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/aiegraphop.h.inc
popd
