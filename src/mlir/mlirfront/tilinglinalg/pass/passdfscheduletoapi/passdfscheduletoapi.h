/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef __DFSCHEDULE_TO_API_PASS_H__
#define __DFSCHEDULE_TO_API_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// Pass to convert dfschedule operations to API calls:
/// 1. Convert dfschedule.host @host_canonicalized to void hostruntime() function call
/// 2. Convert arith.constant dense to C array definitions using EmitC
/// 3. Convert dfscheblueprint.declare_data to XAie_MemAllocate with emitc.for data copy
/// 4. Erase all other dfschedule operations
class DfscheduleToApiPass : public PassWrapper<DfscheduleToApiPass, OperationPass<ModuleOp>> {
public:
    DfscheduleToApiPass() = default;

    StringRef getArgument() const final { return "dfschedule-to-api"; }
    StringRef getDescription() const final { 
        return "Convert dfschedule operations to API calls and EmitC"; 
    }
    
    void runOnOperation() override;
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, 
                        arith::ArithDialect,
                        emitc::EmitCDialect,
                        scf::SCFDialect>();
    }
};

} // namespace mlir

#endif // __DFSCHEDULE_TO_API_PASS_H__

