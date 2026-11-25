###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
${LLVM_BIN}/mlir-tblgen -gen-dialect-defs ./aielinkops.td -dialect=AieLink ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > ./aielinkdialect.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-dialect-decls ./aielinkops.td -dialect=AieLink ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > ./aielinkdialect.h.inc
${LLVM_BIN}/mlir-tblgen -gen-op-defs ./aielinkops.td ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > ./aielinkop.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-op-decls ./aielinkops.td ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} > ./aielinkop.h.inc

${LLVM_BIN}/mlir-tblgen -gen-typedef-defs ./aielinktypes.td --typedefs-dialect=AieLink ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aielinktype.cc.inc
${LLVM_BIN}/mlir-tblgen -gen-typedef-decls ./aielinktypes.td --typedefs-dialect=AieLink ${MLIR_INCLUDES[@]} -I /usr/local/include ${MLIR_INCLUDES} -asmformat-error-is-fatal=false > ./aielinktype.h.inc
# The generated .inc files are build output (gitignored); give each the license header anyway
# (script/ci/check_opensource.py), so a copy that does get committed stays compliant.
for f in ./*.inc; do
  if ! grep -q SPDX-License-Identifier "$f"; then
    { printf '%s\n\n' '/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/'; cat "$f"; } > "$f.tmp" && mv "$f.tmp" "$f"
  fi
done
popd
