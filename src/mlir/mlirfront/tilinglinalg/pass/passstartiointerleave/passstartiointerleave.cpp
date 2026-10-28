/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "passstartiointerleave.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static bool attrOn(ModuleOp mod, StringRef name) {
    auto a = mod->getAttrOfType<IntegerAttr>(name);
    return a && a.getInt() != 0;
}

static Operation *latestOperandDef(Operation *op, Block *block) {
    Operation *latest = nullptr;
    for (Value v : op->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != block)
            continue;
        if (!latest || latest->isBeforeInBlock(def))
            latest = def;
    }
    return latest;
}

void StartioInterleavePass::runOnOperation() {
    ModuleOp mod = getOperation();
    if (attrOn(mod, "routing.control_packet_mode") || attrOn(mod, "routing.control_plan_shim_bd_ctrl")) {
        llvm::errs() << "[TilingLinalg] StartioInterleavePass skipped (AOT / shim-BD control window)\n";
        return;
    }
    SmallVector<emitc::CallOpaqueOp> starts;
    mod.walk([&](emitc::CallOpaqueOp call) {
        if (call.getCallee() == "__Runtime_startio")
            starts.push_back(call);
    });
    unsigned moved = 0;
    for (emitc::CallOpaqueOp call : starts) {
        Operation *op = call.getOperation();
        Operation *after = latestOperandDef(op, op->getBlock());
        if (!after || after->getNextNode() == op)
            continue;
        op->moveAfter(after);
        moved++;
    }
    llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << moved << " of " << starts.size()
                 << " startio moved next to their createio\n";
}
