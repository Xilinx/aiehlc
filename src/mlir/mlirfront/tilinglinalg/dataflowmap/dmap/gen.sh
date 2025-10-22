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
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/dmaptype.td -dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} > $INC/dmapdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/dmaptype.td -dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dmapdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/dmaptype.td -typedefs-dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dmaptype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/dmaptype.td -typedefs-dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dmaptype.h.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-defs $TD/dmapattr.td -attrdefs-dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dmapattr.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-decls $TD/dmapattr.td -attrdefs-dialect=dmap -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dmapattr.h.inc

#${LLVM_BIN}/mlir-tblgen --gen-type-interface-defs $TD/dmamapterface.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/dmamapterface.cc.inc
#${LLVM_BIN}/mlir-tblgen --gen-type-interface-decls $TD/dmamapterface.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/dmamapterface.h.inc

${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/dmapop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/dmapop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/dmapop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/dmapop.h.inc
popd
