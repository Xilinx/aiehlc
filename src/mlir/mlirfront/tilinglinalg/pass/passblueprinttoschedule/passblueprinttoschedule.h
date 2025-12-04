/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef __BLUEPRINT_TOSCHEDULE_PASS_H__
#define __BLUEPRINT_TOSCHEDULE_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"

namespace mlir {

class BlueprintToSchedulePass : public PassWrapper<BlueprintToSchedulePass, OperationPass<>> {
public:
    BlueprintToSchedulePass() = default;
    //MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlueprintToSchedulePass)

    StringRef getArgument() const final { return "lower-blueprint-to-schedule"; }
    StringRef getDescription() const final { return "Lower dfscheblueprint dialect to dfschedule dialect"; }
    
    void runOnOperation() override;
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfscheblueprint::dfscheblueprintdialect,
                        dfschedule::dfscheduledialect, 
                        func::FuncDialect, 
                        memref::MemRefDialect,
                        arith::ArithDialect,
                        scf::SCFDialect,
                        tensor::TensorDialect>();
    }
};

} // namespace mlir

#endif // __BLUEPRINT_TOSCHEDULE_PASS_H__
