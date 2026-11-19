###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

GEN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC=$GEN_SCRIPT_DIR/inc
TD=$GEN_SCRIPT_DIR/td
echo "Script is located in: $GEN_SCRIPT_DIR"
pushd ${GEN_SCRIPT_DIR}
mlir-tblgen -gen-dialect-defs $TD/testtype.td -dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include > $INC/testdialect.cc.inc
mlir-tblgen -gen-dialect-decls $TD/testtype.td -dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include >  $INC/testdialect.h.inc
mlir-tblgen -gen-typedef-defs $TD/testtype.td -typedefs-dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include >  $INC/testtype.cc.inc
mlir-tblgen -gen-typedef-decls $TD/testtype.td -typedefs-dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include >  $INC/testtype.h.inc
mlir-tblgen --gen-attrdef-defs $TD/testattr.td -attrdefs-dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include >  $INC/testattr.cc.inc
mlir-tblgen --gen-attrdef-decls $TD/testattr.td -attrdefs-dialect=test -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include >  $INC/testattr.h.inc

mlir-tblgen --gen-type-interface-defs $TD/testinterface.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include -o  $INC/testinterface.cc.inc
mlir-tblgen --gen-type-interface-decls $TD/testinterface.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include -o  $INC/testinterface.h.inc

mlir-tblgen -gen-op-defs $TD/testop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include > $INC/testop.cc.inc
mlir-tblgen -gen-op-decls $TD/testop.td -I $TD -I $GEN_SCRIPT_DIR ${MLIR_INCLUDES[@]} -I /usr/local/include > $INC/testop.h.inc
popd
