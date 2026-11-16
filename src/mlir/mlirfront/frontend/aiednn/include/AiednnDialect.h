/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// The aiednn dialect (td/AiednnOps.td): a whole model as aiednn primitives.
#ifndef AIEDNN_DIALECT_H
#define AIEDNN_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "AiednnOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "AiednnOpsAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "AiednnOps.h.inc"

namespace aiednn {
// --aiednn-export-plan="out=<file>": the func's ops as JSON for aiegraph.dnn_mlir
std::unique_ptr<mlir::Pass> createExportPlanPass();
// --aiednn-fuse-epilogue: single-use eltwise / residual add -> GEMM-family post_ops
std::unique_ptr<mlir::Pass> createFuseEpiloguePass();
void registerFuseEpiloguePass();
// --aiednn-plan-memory: arena offsets (liveness reuse + in-place eltwise)
void registerPlanMemoryPass();
void registerAiednnPasses();
} // namespace aiednn

#endif // AIEDNN_DIALECT_H
