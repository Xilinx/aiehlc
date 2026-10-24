/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __SCHEDULE_SEQUENTIAL_OP_PASS_H__
#define __SCHEDULE_SEQUENTIAL_OP_PASS_H__

#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Pass to reorder dfschedule operations within RoutingCreate blocks for
/// optimal execution sequencing:
///   1. Kernel load + launch at the beginning
///   2. DMA config + IO create after kernel
///   3. All start_io after DMA config; S2MM (sink) start_io before MM2S (source)
///   4. Wait ops after start_io group; for-loop waits before non-for-loop waits
class ScheduleSequentialOpPass : public PassWrapper<ScheduleSequentialOpPass, OperationPass<ModuleOp>> {
  public:
    ScheduleSequentialOpPass() = default;

    StringRef getArgument() const final { return "schedule-sequential-op"; }
    StringRef getDescription() const final {
        return "Reorder dfschedule ops for sequential execution: kernel first, then DMA config, then start_io, then "
               "wait";
    }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect, func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                        scf::SCFDialect>();
    }
};

} // namespace mlir

#endif // __SCHEDULE_SEQUENTIAL_OP_PASS_H__
