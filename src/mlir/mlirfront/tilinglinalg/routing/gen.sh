###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################
#!/usr/bin/env bash

GEN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC=$GEN_SCRIPT_DIR/inc
TD=$GEN_SCRIPT_DIR/td

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

echo "Script is located in: $GEN_SCRIPT_DIR"
pushd ${GEN_SCRIPT_DIR}
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/routingtype.td -dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} > $INC/routingdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/routingtype.td -dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/routingdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/routingtype.td -typedefs-dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/routingtype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/routingtype.td -typedefs-dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/routingtype.h.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-defs $TD/routingattr.td -attrdefs-dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/routingattr.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-attrdef-decls $TD/routingattr.td -attrdefs-dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} >  $INC/routingattr.h.inc

${LLVM_BIN}/mlir-tblgen --gen-type-interface-defs $TD/routinginterface.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/routinginterface.cc.inc
${LLVM_BIN}/mlir-tblgen --gen-type-interface-decls $TD/routinginterface.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} -o  $INC/routinginterface.h.inc

${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/routingop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/routingop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/routingop.td -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include ${MLIR_INCLUDES} > $INC/routingop.h.inc
popd
