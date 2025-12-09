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

// Helper to get EmitC opaque type string from MLIR element type
static std::string getEmitCTypeString(Type elemType) {
    if (elemType.isInteger(8)) return "int8_t";
    if (elemType.isInteger(16)) return "int16_t";
    if (elemType.isInteger(32)) return "int32_t";
    if (elemType.isInteger(64)) return "int64_t";
    if (elemType.isF32()) return "float";
    if (elemType.isF64()) return "double";
    return "uint8_t";
}

// Helper to build array type string with dimensions
static std::string buildArrayTypeString(const std::string &baseType, ArrayRef<int64_t> shape) {
    std::string result = baseType;
    for (auto dim : shape) {
        result += "[" + std::to_string(dim) + "]";
    }
    return result;
}

// Helper to build initializer string from dense attribute
static std::string buildInitializerString(DenseElementsAttr denseAttr, Type elemType) {
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
    return initStream.str();
}

void DfscheduleToApiPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    auto loc = builder.getUnknownLoc();
    
    // ===== PHASE 1: Convert dense constants to emitc.global =====
    SmallVector<arith::ConstantOp> denseConstantOps;
    
    // Collect top-level constants in func.func main
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
        Type elemType = tensorType.getElementType();
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string arrayTypeStr = buildArrayTypeString(cTypeStr, tensorType.getShape());
        std::string initStr = buildInitializerString(denseAttr, elemType);
        
        // Create emitc.verbatim for the global array declaration
        std::string verbatimCode = "static const " + cTypeStr + " " + arrayName + 
            buildArrayTypeString("", tensorType.getShape()) + " = " + initStr + ";";
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(verbatimCode));
    }
    
    // ===== PHASE 2: Replace launchhost with emitc.call_opaque to hostruntime =====
    SmallVector<Operation*> launchHostOps;
    moduleOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "dfschedule.launchhost") {
            launchHostOps.push_back(op);
        }
    });
    
    for (auto *op : launchHostOps) {
        builder.setInsertionPoint(op);
        // Use emitc.call_opaque to call hostruntime()
        builder.create<emitc::CallOpaqueOp>(
            loc,
            /*resultTypes=*/TypeRange{},
            /*callee=*/builder.getStringAttr("hostruntime"),
            /*args=*/nullptr,
            /*templateArgs=*/nullptr,
            /*operands=*/ValueRange{}
        );
        op->erase();
    }
    
    // ===== PHASE 3: Convert dfschedule.dskernel_receiver to emitc.func =====
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
        
        builder.setInsertionPoint(op);
        
        // Create emitc.func for the kernel
        auto voidType = emitc::OpaqueType::get(builder.getContext(), "void");
        auto funcType = builder.getFunctionType({}, {});
        
        auto emitcFunc = builder.create<emitc::FuncOp>(
            loc,
            kernelName,
            funcType
        );
        
        // Add __global__ specifier as an attribute
        emitcFunc->setAttr("specifiers", builder.getStrArrayAttr({"__global__"}));
        
        // Create function body with a return
        Block *entryBlock = emitcFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        // Add a comment as emitc.call_opaque placeholder
        builder.create<emitc::CallOpaqueOp>(
            loc,
            TypeRange{},
            builder.getStringAttr("/* AIE kernel implementation */"),
            nullptr, nullptr, ValueRange{}
        );
        
        // emitc::ReturnOp takes optional single Value, not ValueRange
        builder.create<emitc::ReturnOp>(loc, Value{});
        
        // Erase the original dskernel_receiver op
        op->erase();
    }
    
    // ===== PHASE 4: Erase dfschedule.host at module level =====
    SmallVector<Operation*> hostOps;
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if (op.getName().getStringRef() == "dfschedule.host") {
            hostOps.push_back(&op);
        }
    }
    for (auto *op : hostOps) {
        op->erase();
    }
    
    // ===== PHASE 5: Clean up func.func main - keep return and emitc ops =====
    func::FuncOp mainFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc || mainFunc.getBody().empty()) return;
    
    Block &mainBlock = mainFunc.getBody().front();
    
    // Repeatedly find and erase ops until only return and emitc ops remain
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto &op : llvm::make_early_inc_range(mainBlock)) {
            // Keep return, call, and emitc ops
            if (isa<func::ReturnOp>(&op) || 
                isa<func::CallOp>(&op) ||
                op.getName().getStringRef().starts_with("emitc."))
                continue;
            
            // Check if all users are already erased (op has no uses)
            if (op.use_empty()) {
                op.erase();
                changed = true;
                break;
            }
        }
    }
}

} // namespace mlir

