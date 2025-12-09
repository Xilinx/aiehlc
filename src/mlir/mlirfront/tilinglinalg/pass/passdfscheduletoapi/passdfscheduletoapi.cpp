/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passdfscheduletoapi.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    
    // ===== PHASE 4: Convert dfscheblueprint.declare_data to XAie_MemAllocate and emitc.for copy =====
    SmallVector<Operation*> declareDataOps;
    moduleOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {
            declareDataOps.push_back(op);
        }
    });
    
    int memInstIndex = 0;
    for (auto *op : declareDataOps) {
        builder.setInsertionPoint(op);
        
        // Get tensor type from the result
        if (op->getNumResults() == 0) continue;
        auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!resultType) continue;
        
        // Calculate total size
        int64_t totalSize = 1;
        for (auto dim : resultType.getShape()) {
            totalSize *= dim;
        }
        
        Type elemType = resultType.getElementType();
        int64_t elemSize = 1;
        if (elemType.isInteger(8)) elemSize = 1;
        else if (elemType.isInteger(16)) elemSize = 2;
        else if (elemType.isInteger(32) || elemType.isF32()) elemSize = 4;
        else if (elemType.isInteger(64) || elemType.isF64()) elemSize = 8;
        
        int64_t byteSize = totalSize * elemSize;
        
        std::string memInstName = "MemInst_" + std::to_string(memInstIndex);
        std::string srcArrayName = "g_data_array_" + std::to_string(memInstIndex);
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string dstPtrName = "dst_" + std::to_string(memInstIndex);
        memInstIndex++;
        
        // Generate: XAie_MemInst* MemInst = XAie_MemAllocate(&DevInst, Size, XAIE_MEM_CACHEABLE);
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
            "XAie_MemInst* " + memInstName + " = XAie_MemAllocate(&DevInst, " + 
            std::to_string(byteSize) + ", XAIE_MEM_CACHEABLE);"
        ));
        
        // Generate: dst pointer = XAie_MemGetVAddr(MemInst)
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
            cTypeStr + "* " + dstPtrName + " = (" + cTypeStr + "*)XAie_MemGetVAddr(" + memInstName + ");"
        ));
        
        // Create emitc.for loop to copy data from source array to allocated memory
        // Loop bounds: 0 to totalSize, step 1
        auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto ub = builder.create<arith::ConstantIndexOp>(loc, totalSize);
        auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
        
        // Create emitc.for loop
        auto forOp = builder.create<emitc::ForOp>(loc, lb, ub, step);
        
        // Build the loop body - use the induction variable
        builder.setInsertionPointToStart(forOp.getBody());
        
        // Get induction variable and use emitc.subscript for array access
        Value iv = forOp.getInductionVar();
        
        // Use emitc.call_opaque to represent the assignment dst[i] = src[i]
        // Since we can't directly do pointer arithmetic with emitc types, use verbatim inside the loop
        // The induction variable 'v' will be emitted as the loop variable
        builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
            dstPtrName + "[v] = ((" + cTypeStr + "*)" + srcArrayName + ")[v];"
        ));
        
        // Reset insertion point after the for loop
        builder.setInsertionPointAfter(forOp);
    }
    
    // Don't erase declare_data ops - they are converted to EmitC
    
    // ===== PHASE 5: Erase dfschedule.host at module level =====
    SmallVector<Operation*> hostOps;
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if (op.getName().getStringRef() == "dfschedule.host") {
            hostOps.push_back(&op);
        }
    }
    for (auto *op : hostOps) {
        op->erase();
    }
    
    // ===== PHASE 6: Clean up func.func main - keep return, emitc, arith.constant, and dfscheblueprint ops =====
    func::FuncOp mainFunc = moduleOp.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc || mainFunc.getBody().empty()) return;
    
    Block &mainBlock = mainFunc.getBody().front();
    
    // Repeatedly find and erase ops until only allowed ops remain
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto &op : llvm::make_early_inc_range(mainBlock)) {
            // Keep return, call, emitc ops, arith.constant (for loop bounds), and dfscheblueprint.declare_data
            if (isa<func::ReturnOp>(&op) || 
                isa<func::CallOp>(&op) ||
                isa<arith::ConstantIndexOp>(&op) ||
                op.getName().getStringRef().starts_with("emitc.") ||
                op.getName().getStringRef() == "dfscheblueprint.declare_data")
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

