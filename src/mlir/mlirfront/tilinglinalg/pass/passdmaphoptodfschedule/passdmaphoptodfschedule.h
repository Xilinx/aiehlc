/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef __DMAPHOP_TODFSCHEDULE_PASS_H__
#define __DMAPHOP_TODFSCHEDULE_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "dmaphopmanager.h"
#include "dfschedulemanager.h"
#include "routingmanager.h"

namespace mlir {

class DmaphopTodfschedulePass : public PassWrapper<DmaphopTodfschedulePass, OperationPass<>> {
public:
    DmaphopTodfschedulePass() = default;
    //MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaphopTodfschedulePass)

    StringRef getArgument() const final { return "lower-dmaphop-to-dfschedule"; }
    StringRef getDescription() const final { return "Lower dmaphop dialect to dfschedule dialect"; }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dmaphop::dmaphopdialect, 
                        dfschedule::dfscheduledialect, 
                        func::FuncDialect, 
                        memref::MemRefDialect,
                        arith::ArithDialect,
                        scf::SCFDialect>();
    }
};

} // namespace mlir

#endif // __DMAPHOP_TODFSCHEDULE_PASS_H__
