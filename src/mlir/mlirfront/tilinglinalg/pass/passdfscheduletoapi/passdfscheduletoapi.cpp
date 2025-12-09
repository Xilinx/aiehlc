/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passdfscheduletoapi.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace mlir;

namespace mlir {

void DfscheduleToApiPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    auto loc = builder.getUnknownLoc();
    
    // ===== PHASE 1: Generate EmitC for dense constants (just at module level) =====
    SmallVector<arith::ConstantOp> denseConstantOps;
    
    // Only collect top-level constants in func.func main
    if (auto mainFunc = moduleOp.lookupSymbol<func::FuncOp>("main")) {
        for (auto &op : mainFunc.getBody().front()) {
            if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
                if (isa<DenseElementsAttr>(constOp.getValue())) {
                    denseConstantOps.push_back(constOp);
                }
            }
        }
    }
    
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    int arrayIndex = 0;
    for (auto constOp : denseConstantOps) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!denseAttr) continue;
        
        auto tensorType = dyn_cast<RankedTensorType>(denseAttr.getType());
        if (!tensorType) continue;
        
        std::string arrayName = "g_data_array_" + std::to_string(arrayIndex++);
        
        std::string cTypeStr;
        Type elemType = tensorType.getElementType();
        if (elemType.isInteger(8)) cTypeStr = "int8_t";
        else if (elemType.isInteger(16)) cTypeStr = "int16_t";
        else if (elemType.isInteger(32)) cTypeStr = "int32_t";
        else if (elemType.isInteger(64)) cTypeStr = "int64_t";
        else if (elemType.isF32()) cTypeStr = "float";
        else if (elemType.isF64()) cTypeStr = "double";
        else cTypeStr = "uint8_t";
        
        std::string dimsStr;
        for (auto dim : tensorType.getShape()) {
            dimsStr += "[" + std::to_string(dim) + "]";
        }
        
        std::ostringstream initStream;
        initStream << "{";
        bool first = true;
        
        if (elemType.isIntOrIndex()) {
            for (auto val : denseAttr.getValues<llvm::APInt>()) {
                if (!first) initStream << ", ";
                first = false;
                initStream << val.getSExtValue();
            }
        } else if (elemType.isF32()) {
            for (auto val : denseAttr.getValues<llvm::APFloat>()) {
                if (!first) initStream << ", ";
                first = false;
                initStream << val.convertToFloat();
            }
        } else if (elemType.isF64()) {
            for (auto val : denseAttr.getValues<llvm::APFloat>()) {
                if (!first) initStream << ", ";
                first = false;
                initStream << val.convertToDouble();
            }
        }
        initStream << "}";
        
        std::string verbatimCode = "static const " + cTypeStr + " " + arrayName + dimsStr + " = " + initStream.str() + ";";
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(verbatimCode));
    }
    
    // ===== PHASE 2: Create hostruntime() function =====
    auto hostRuntimeFuncType = builder.getFunctionType({}, {});
    if (!moduleOp.lookupSymbol<func::FuncOp>("hostruntime")) {
        builder.setInsertionPointToStart(moduleOp.getBody());
        auto hostRuntimeFunc = builder.create<func::FuncOp>(loc, "hostruntime", hostRuntimeFuncType);
        hostRuntimeFunc.setPrivate();
    }
    
    // ===== PHASE 3: Replace launchhost with call to hostruntime =====
    // Collect first, then process
    SmallVector<Operation*> launchHostOps;
    moduleOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "dfschedule.launchhost") {
            launchHostOps.push_back(op);
        }
    });
    
    for (auto *op : launchHostOps) {
        builder.setInsertionPoint(op);
        if (auto hostRuntimeFunc = moduleOp.lookupSymbol<func::FuncOp>("hostruntime")) {
            builder.create<func::CallOp>(loc, hostRuntimeFunc, ValueRange{});
        }
        op->erase();
    }
    
    // ===== PHASE 4: Convert dfschedule.dskernel_receiver to EmitC kernel function =====
    SmallVector<Operation*> dskernelOps;
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if (op.getName().getStringRef() == "dfschedule.dskernel_receiver") {
            dskernelOps.push_back(&op);
        }
    }
    
    for (auto *op : dskernelOps) {
        // Get the symbol name from the operation
        std::string kernelName = "dskernel";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            kernelName = symNameAttr.getValue().str();
        }
        
        // Create __global__ void kernel function using emitc.verbatim
        builder.setInsertionPoint(op);
        std::string kernelCode = "__global__ void " + kernelName + "() {\n    // AIE kernel implementation\n}";
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(kernelCode));
        
        // Erase the original dskernel_receiver op
        op->erase();
    }
    
    // ===== PHASE 5: Erase dfschedule.host at module level =====
    // Collect first to avoid iterator invalidation
    SmallVector<Operation*> hostOps;
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if (op.getName().getStringRef() == "dfschedule.host") {
            hostOps.push_back(&op);
        }
    }
    for (auto *op : hostOps) {
        op->erase();
    }
    
    // ===== PHASE 6: Clean up func.func main - keep only return and call =====
    func::FuncOp mainFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc || mainFunc.getBody().empty()) return;
    
    Block &mainBlock = mainFunc.getBody().front();
    
    // Repeatedly find and erase ops until only return and call remain
    // This handles the def-use chain by erasing users before defs
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto &op : llvm::make_early_inc_range(mainBlock)) {
            // Keep return and call ops
            if (isa<func::ReturnOp>(&op) || isa<func::CallOp>(&op))
                continue;
            
            // Check if all users are already erased (op has no uses)
            if (op.use_empty()) {
                op.erase();
                changed = true;
                break;  // Restart iteration since we modified the block
            }
        }
    }
}

} // namespace mlir

