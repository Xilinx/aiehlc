/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef __SCHEDULE_CANONICALIZE_PASS_H__
#define __SCHEDULE_CANONICALIZE_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dfschedulemanager.h"

namespace mlir {

/// Pass to canonicalize dfschedule operations:
/// 1. Merge load_kernel_group calls - each tile should appear only once
/// 2. Deduplicate tile declarations with same (col, row)
/// 3. Merge packets per tile for multiple data streams
/// 4. Consolidate start_io operations
/// 5. Merge schedule.wait with all events
/// 6. Update kernel function to handle multiple packet inputs
class ScheduleCanonicalizePass : public PassWrapper<ScheduleCanonicalizePass, OperationPass<ModuleOp>> {
public:
    ScheduleCanonicalizePass() = default;

    StringRef getArgument() const final { return "canonicalize-schedule"; }
    StringRef getDescription() const final { 
        return "Canonicalize dfschedule dialect - merge kernel loads, deduplicate tiles, consolidate IOs"; 
    }
    
    void runOnOperation() override;
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect, 
                        func::FuncDialect, 
                        memref::MemRefDialect,
                        arith::ArithDialect,
                        scf::SCFDialect,
                        tensor::TensorDialect>();
    }
};

} // namespace mlir

#endif // __SCHEDULE_CANONICALIZE_PASS_H__

