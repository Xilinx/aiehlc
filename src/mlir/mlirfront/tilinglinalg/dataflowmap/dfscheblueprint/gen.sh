###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
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
  #MLIR_INCLUDES=("-I/usr/local/include/mlir/")
  MLIR_INCLUDES=("-I/Users/llvm/llvm-project/mlir/include/")
fi
if [ ${#LLVM_BIN[@]} -eq 0 ]; then
  echo "No LLVM_BIN provided, using default."
  #LLVM_BIN=("/usr/local/bin/")
  LLVM_BIN=("/Users/llvm/llvm-project/build/bin/")
fi

echo $MLIR_INCLUDES
echo "Script is located in: $GEN_SCRIPT_DIR"
pushd ${GEN_SCRIPT_DIR}
echo ${LLVM_BIN}
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/dfscheblueprinttype.td -dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} > $INC/dfscheblueprintdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/dfscheblueprinttype.td -dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprintdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/dfscheblueprinttype.td -typedefs-dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprinttype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/dfscheblueprinttype.td -typedefs-dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprinttype.h.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-defs $TD/dfscheblueprintattr.td -attrdefs-dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprintattr.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-decls $TD/dfscheblueprintattr.td -attrdefs-dialect=dfscheblueprint -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprintattr.h.inc

${LLVM_BIN}/mlir-tblgen --gen-enum-decls $TD/dfscheblueprintattr.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprintenums.h.inc
${LLVM_BIN}/mlir-tblgen --gen-enum-defs $TD/dfscheblueprintattr.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheblueprintenums.cc.inc

${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/dfscheblueprintop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/dfscheblueprintop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/dfscheblueprintop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/dfscheblueprintop.h.inc
popd

