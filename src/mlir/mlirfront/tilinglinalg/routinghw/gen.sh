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

#mlir-tblgen --gen-attrdef-defs $TD/routingattr.td -attrdefs-dialect=routing -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include >  $INC/routingattr.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-attrdef-defs $TD/routinghwattr.td -attrdefs-dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwattr.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-defs $TD/routinghwtype.td -typedefs-dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwtype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-defs $TD/routinghwop.td -dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs $TD/routinghwattr.td -dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwdialect.cc.inc
#defines
${LLVM_BIN}/mlir-tblgen -gen-attrdef-decls $TD/routinghwattr.td -attrdefs-dialect=routinghw -I$TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwattr.h.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls $TD/routinghwtype.td -typedefs-dialect=routinghw -I$TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwtype.h.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls $TD/routinghwop.td -dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwop.h.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls $TD/routinghwattr.td -dialect=routinghw -I $TD -I $GEN_SCRIPT_DIR -I /usr/local/include/ ${MLIR_INCLUDES} -o $INC/routinghwdialect.h.inc

