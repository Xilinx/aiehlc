/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __STARTIO_INTERLEAVE_PASS__
#define __STARTIO_INTERLEAVE_PASS__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

struct StartioInterleavePass : public PassWrapper<StartioInterleavePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StartioInterleavePass)
    StringRef getArgument() const final { return "startio-interleave"; }
    StringRef getDescription() const final { return "Issue each DMA channel right after its BD chain"; }
    void runOnOperation() override;
};

#endif
