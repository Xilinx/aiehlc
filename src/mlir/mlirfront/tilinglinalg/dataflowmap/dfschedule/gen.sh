###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/dfscheduletype.td -dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/dfscheduledialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/dfscheduletype.td -dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduledialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/dfscheduletype.td -typedefs-dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduletype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/dfscheduletype.td -typedefs-dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduletype.h.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-defs $TD/dfscheduleattr.td -attrdefs-dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduleattr.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-decls $TD/dfscheduleattr.td -attrdefs-dialect=dfschedule -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduleattr.h.inc

${LLVM_BIN}/mlir-tblgen --gen-enum-decls $TD/dfscheduleattr.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduleenums.h.inc
${LLVM_BIN}/mlir-tblgen --gen-enum-defs $TD/dfscheduleattr.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} >  $INC/dfscheduleenums.cc.inc

#${LLVM_BIN}/mlir-tblgen --gen-type-interface-defs $TD/dmamapterface.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/dmamapterface.cc.inc
#${LLVM_BIN}/mlir-tblgen --gen-type-interface-decls $TD/dmamapterface.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/dmamapterface.h.inc

${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/dfscheduleop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/dfscheduleop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/dfscheduleop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > $INC/dfscheduleop.h.inc
popd
