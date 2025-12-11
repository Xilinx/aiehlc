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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static std::string getEmitCTypeString(Type elemType) {
    if (elemType.isInteger(8)) return "int8_t";
    if (elemType.isInteger(16)) return "int16_t";
    if (elemType.isInteger(32)) return "int32_t";
    if (elemType.isInteger(64)) return "int64_t";
    if (elemType.isF32()) return "float";
    if (elemType.isF64()) return "double";
    return "uint8_t";
}

static std::string buildArrayDimString(ArrayRef<int64_t> shape) {
    std::string result;
    for (auto dim : shape) {
        result += "[" + std::to_string(dim) + "]";
    }
    return result;
}

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

// Get EmitC pointer type for element type
static Type getEmitCPtrType(MLIRContext *ctx, Type elemType) {
    std::string typeStr = getEmitCTypeString(elemType);
    return emitc::PointerType::get(emitc::OpaqueType::get(ctx, typeStr));
}

//===----------------------------------------------------------------------===//
// Pass Implementation - Using proper EmitC SSA values
//===----------------------------------------------------------------------===//

void DfscheduleToApiPass::runOnOperation() {
    llvm::errs() << "=== DfscheduleToApiPass START ===\n";
    
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();
    OpBuilder builder(ctx);
    
    int arrayIndex = 0;
    int partitionIndex = 0;
    
    // Map to track memory allocations: source tensor Value -> (dstPtr SSA Value, byteSize, numElements)
    DenseMap<Value, std::tuple<Value, int64_t, int64_t>> memAllocMap;
    
    // Map array name to global name for emitc.get_global
    DenseMap<Operation*, std::string> arrayNameMap;
    
    // Collect all operations to process
    SmallVector<Operation*> hostOps;
    SmallVector<Operation*> launchHostOps;
    SmallVector<Operation*> dsKernelReceiverOps;
    SmallVector<Operation*> allDfscheduleOps;
    SmallVector<Operation*> arithConstantDenseOps;
    
    moduleOp.walk([&](Operation *op) {
        StringRef opName = op->getName().getStringRef();
        
        if (opName == "dfschedule.host") {
            hostOps.push_back(op);
        } else if (opName == "dfschedule.launchhost") {
            launchHostOps.push_back(op);
        } else if (opName == "dfschedule.dskernel_receiver") {
            dsKernelReceiverOps.push_back(op);
        }
        
        // Collect all dfschedule and dfscheblueprint ops for later erasure
        if (opName.starts_with("dfschedule.") || opName.starts_with("dfscheblueprint.") ||
            opName.starts_with("routing.")) {
            allDfscheduleOps.push_back(op);
        }
        
        // Also collect arith.constant with dense attributes
        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (isa<DenseElementsAttr>(constOp.getValue())) {
                arithConstantDenseOps.push_back(op);
            }
        }
    });
    
    llvm::errs() << "[Pass] Found " << hostOps.size() << " host ops\n";
    llvm::errs() << "[Pass] Found " << allDfscheduleOps.size() << " total dfschedule/dfscheblueprint/routing ops\n";
    
    //==========================================================================
    // Phase 1: Generate EmitC code with proper SSA values
    //==========================================================================
    
    // 1a. Generate struct definition and extern declarations at module scope
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // PartitionTensor struct definition
    builder.create<emitc::VerbatimOp>(moduleOp.getLoc(), builder.getStringAttr(
        "/* PartitionTensor structure for memory management */\n"
        "typedef struct {\n"
        "    void* data;\n"
        "    size_t size;\n"
        "    size_t num_elements;\n"
        "    int splitnum;\n"
        "    int splitdim;\n"
        "} PartitionTensor;\n\n"
        "/* Helper function for PartitionTensor initialization */\n"
        "static inline PartitionTensor __emitc_init_PartitionTensor(\n"
        "    void* data, size_t size, size_t num_elements, int splitnum, int splitdim) {\n"
        "    PartitionTensor pt = {data, size, num_elements, splitnum, splitdim};\n"
        "    return pt;\n"
        "}"
    ));
    
    // Create extern global for DevInst
    auto devInstType = emitc::OpaqueType::get(ctx, "XAie_DevInst");
    builder.create<emitc::GlobalOp>(moduleOp.getLoc(),
        /*sym_name=*/"DevInst",
        /*type=*/devInstType,
        /*initial_value=*/Attribute{},
        /*extern_specifier=*/true,
        /*static_specifier=*/false,
        /*const_specifier=*/false);
    
    // 1b. Convert arith.constant dense to emitc.global arrays
    for (Operation *op : arithConstantDenseOps) {
        auto constOp = cast<arith::ConstantOp>(op);
        auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());
        auto tensorType = dyn_cast<RankedTensorType>(denseAttr.getType());
        if (!tensorType) continue;
        
        std::string arrayName = "g_data_array_" + std::to_string(arrayIndex++);
        Type elemType = tensorType.getElementType();
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string initStr = buildInitializerString(denseAttr, elemType);
        
        // Create verbatim for the array with initializer (emitc.global doesn't support array init well)
        std::string verbatimCode = "static const " + cTypeStr + " " + arrayName +
            buildArrayDimString(tensorType.getShape()) + " = " + initStr + ";";
        
        builder.setInsertionPointToStart(moduleOp.getBody());
        builder.create<emitc::VerbatimOp>(op->getLoc(), builder.getStringAttr(verbatimCode));
        
        // Store array name for later use
        arrayNameMap[op] = arrayName;
        
        llvm::errs() << "[Pass] Created global array: " << arrayName << "\n";
    }
    
    // 1c. Convert dfschedule.host to emitc.func with proper SSA
    for (Operation *op : hostOps) {
        std::string funcName = "hostruntime";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            funcName = symNameAttr.getValue().str();
        }
        
        builder.setInsertionPoint(op);
        auto funcType = builder.getFunctionType({}, {});
        auto emitcFunc = builder.create<emitc::FuncOp>(op->getLoc(), funcName, funcType);
        Block *entryBlock = emitcFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        
        auto loc = op->getLoc();
        
        // Create common types
        auto i32Type = builder.getI32Type();
        auto voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        auto memInstPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "XAie_MemInst"));
        
        // Get DevInst global reference
        auto devInstRef = builder.create<emitc::GetGlobalOp>(loc, devInstType, "DevInst");
        
        // Create XAIE_MEM_CACHEABLE constant (typically 0 or a specific value)
        auto cacheableConst = builder.create<emitc::ConstantOp>(loc, i32Type,
            emitc::OpaqueAttr::get(ctx, "XAIE_MEM_CACHEABLE"));
        
        // Process nested operations
        if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
            Block &srcBlock = op->getRegion(0).front();
            
            for (Operation &nestedOp : srcBlock.getOperations()) {
                StringRef nestedOpName = nestedOp.getName().getStringRef();
                auto nestedLoc = nestedOp.getLoc();
                
                // Handle dfscheblueprint.declare_data
                if (nestedOpName == "dfscheblueprint.declare_data") {
                    if (nestedOp.getNumResults() == 0) continue;
                    
                    auto resultType = dyn_cast<RankedTensorType>(nestedOp.getResult(0).getType());
                    if (!resultType) continue;
                    
                    // Get the array name from the input arith.constant
                    std::string arrayName = "g_data_array_0";
                    if (nestedOp.getNumOperands() > 0) {
                        Value initTensor = nestedOp.getOperand(0);
                        if (Operation *initOp = initTensor.getDefiningOp()) {
                            if (arrayNameMap.count(initOp)) {
                                arrayName = arrayNameMap[initOp];
                            }
                        }
                    }
                    
                    // Calculate sizes
                    int64_t totalElements = 1;
                    for (auto dim : resultType.getShape()) {
                        totalElements *= dim;
                    }
                    
                    Type elemType = resultType.getElementType();
                    std::string cTypeStr = getEmitCTypeString(elemType);
                    int64_t elemSize = 1;
                    if (elemType.isInteger(8)) elemSize = 1;
                    else if (elemType.isInteger(16)) elemSize = 2;
                    else if (elemType.isInteger(32) || elemType.isF32()) elemSize = 4;
                    else if (elemType.isInteger(64) || elemType.isF64()) elemSize = 8;
                    
                    int64_t byteSize = totalElements * elemSize;
                    
                    // Create size constant
                    auto sizeConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(byteSize));
                    
                    // XAie_MemAllocate(&DevInst, size, XAIE_MEM_CACHEABLE)
                    auto memInst = builder.create<emitc::CallOpaqueOp>(nestedLoc,
                        memInstPtrType,
                        "XAie_MemAllocate",
                        /*args=*/nullptr,
                        /*templateArgs=*/nullptr,
                        ValueRange{devInstRef.getResult(), sizeConst.getResult(), cacheableConst.getResult()});
                    
                    // XAie_MemGetVAddr(memInst)
                    auto vaddr = builder.create<emitc::CallOpaqueOp>(nestedLoc,
                        voidPtrType,
                        "XAie_MemGetVAddr",
                        /*args=*/nullptr,
                        /*templateArgs=*/nullptr,
                        ValueRange{memInst.getResult(0)});
                    
                    // Cast void* to element type pointer
                    auto elemPtrType = getEmitCPtrType(ctx, elemType);
                    auto dstPtr = builder.create<emitc::CastOp>(nestedLoc, elemPtrType, vaddr.getResult(0));
                    
                    // Get source array pointer via emitc.constant with opaque reference
                    auto srcPtr = builder.create<emitc::ConstantOp>(nestedLoc, elemPtrType,
                        emitc::OpaqueAttr::get(ctx, "(" + cTypeStr + "*)" + arrayName));
                    
                    // Create num elements constant for memcpy
                    auto numElemsConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(byteSize));
                    
                    // memcpy(dst, src, size)
                    builder.create<emitc::CallOpaqueOp>(nestedLoc,
                        voidPtrType,
                        "memcpy",
                        /*args=*/nullptr,
                        /*templateArgs=*/nullptr,
                        ValueRange{dstPtr.getResult(), srcPtr.getResult(), numElemsConst.getResult()});
                    
                    // Store SSA value for use by partitiontensor
                    memAllocMap[nestedOp.getResult(0)] = std::make_tuple(dstPtr.getResult(), byteSize, totalElements);
                    
                    llvm::errs() << "[Pass] Generated XAie_MemAllocate with SSA for " << arrayName << "\n";
                }
                
                // Handle routing.partitiontensor
                if (nestedOpName == "routing.partitiontensor") {
                    if (nestedOp.getNumResults() == 0 || nestedOp.getNumOperands() == 0) continue;
                    
                    auto resultType = dyn_cast<RankedTensorType>(nestedOp.getResult(0).getType());
                    if (!resultType) continue;
                    
                    Value inputTensor = nestedOp.getOperand(0);
                    
                    // Get splitnum and splitdim attributes
                    int splitnum = 1;
                    int splitdim = 0;
                    if (auto attr = nestedOp.getAttrOfType<IntegerAttr>("splitnum")) {
                        splitnum = attr.getInt();
                    }
                    if (auto attr = nestedOp.getAttrOfType<IntegerAttr>("splitdim")) {
                        splitdim = attr.getInt();
                    }
                    
                    // Calculate sizes for this partition
                    int64_t totalElements = 1;
                    for (auto dim : resultType.getShape()) {
                        totalElements *= dim;
                    }
                    
                    Type elemType = resultType.getElementType();
                    int64_t elemSize = 1;
                    if (elemType.isInteger(8)) elemSize = 1;
                    else if (elemType.isInteger(16)) elemSize = 2;
                    else if (elemType.isInteger(32) || elemType.isF32()) elemSize = 4;
                    else if (elemType.isInteger(64) || elemType.isF64()) elemSize = 8;
                    
                    int64_t partitionByteSize = totalElements * elemSize;
                    
                    // Get source data pointer from memAllocMap
                    Value srcDataPtr;
                    if (memAllocMap.count(inputTensor)) {
                        srcDataPtr = std::get<0>(memAllocMap[inputTensor]);
                    } else {
                        // Fallback: try to find any available memory
                        for (auto &entry : memAllocMap) {
                            srcDataPtr = std::get<0>(entry.second);
                            break;
                        }
                    }
                    
                    // Cast data pointer to void*
                    Value dataVoidPtr;
                    if (srcDataPtr) {
                        dataVoidPtr = builder.create<emitc::CastOp>(nestedLoc, voidPtrType, srcDataPtr).getResult();
                    } else {
                        // Create NULL pointer if no source found
                        dataVoidPtr = builder.create<emitc::ConstantOp>(nestedLoc, voidPtrType,
                            emitc::OpaqueAttr::get(ctx, "NULL")).getResult();
                    }
                    
                    // Create constants for struct initialization
                    auto sizeConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(partitionByteSize));
                    auto numElemsConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(totalElements));
                    auto splitnumConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(splitnum));
                    auto splitdimConst = builder.create<emitc::ConstantOp>(nestedLoc, i32Type,
                        builder.getI32IntegerAttr(splitdim));
                    
                    // Create PartitionTensor struct via helper function call
                    auto partitionType = emitc::OpaqueType::get(ctx, "PartitionTensor");
                    auto partition = builder.create<emitc::CallOpaqueOp>(nestedLoc,
                        partitionType,
                        "__emitc_init_PartitionTensor",
                        /*args=*/nullptr,
                        /*templateArgs=*/nullptr,
                        ValueRange{dataVoidPtr, sizeConst.getResult(), numElemsConst.getResult(),
                                   splitnumConst.getResult(), splitdimConst.getResult()});
                    
                    // Store for potential chained partitions (use the same data pointer)
                    if (srcDataPtr) {
                        memAllocMap[nestedOp.getResult(0)] = std::make_tuple(srcDataPtr, partitionByteSize, totalElements);
                    }
                    
                    llvm::errs() << "[Pass] Created PartitionTensor via SSA: partition_" << partitionIndex++ << "\n";
                }
            }
        }
        
        // Add emitc.return at the end
        builder.setInsertionPointToEnd(entryBlock);
        builder.create<emitc::ReturnOp>(op->getLoc(), Value{});
        
        llvm::errs() << "[Pass] Created emitc.func: " << funcName << "\n";
    }
    
    // 1d. Convert dfschedule.launchhost to emitc.call_opaque("hostruntime")
    for (Operation *op : launchHostOps) {
        builder.setInsertionPoint(op);
        builder.create<emitc::CallOpaqueOp>(
            op->getLoc(),
            TypeRange{},
            "hostruntime",
            nullptr, nullptr, ValueRange{}
        );
        llvm::errs() << "[Pass] Created hostruntime() call\n";
    }
    
    // 1e. Convert dfschedule.dskernel_receiver to emitc.func with __global__
    for (Operation *op : dsKernelReceiverOps) {
        std::string kernelName = "dskernel";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            kernelName = symNameAttr.getValue().str();
        }
        
        builder.setInsertionPoint(op);
        auto funcType = builder.getFunctionType({}, {});
        auto emitcFunc = builder.create<emitc::FuncOp>(op->getLoc(), kernelName, funcType);
        emitcFunc->setAttr("specifiers", builder.getStrArrayAttr({"__global__"}));
        
        Block *entryBlock = emitcFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        builder.create<emitc::VerbatimOp>(op->getLoc(), builder.getStringAttr("/* AIE kernel implementation */"));
        builder.create<emitc::ReturnOp>(op->getLoc(), Value{});
        
        llvm::errs() << "[Pass] Created __global__ func: " << kernelName << "\n";
    }
    
    //==========================================================================
    // Phase 2: Drop all uses and erase dfschedule/dfscheblueprint/routing ops
    //==========================================================================
    
    llvm::errs() << "[Pass] Phase 2: Erasing operations\n";
    
    // Collect ONLY top-level dfschedule/dfscheblueprint/routing ops
    SmallVector<Operation*> topLevelOpsToErase;
    
    for (Operation &op : *moduleOp.getBody()) {
        StringRef opName = op.getName().getStringRef();
        if (opName.starts_with("dfschedule.") || 
            opName.starts_with("dfscheblueprint.") ||
            opName.starts_with("routing.")) {
            topLevelOpsToErase.push_back(&op);
        }
        
        // Also check for arith.constant with dense attribute at module level
        if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
            if (isa<DenseElementsAttr>(constOp.getValue())) {
                topLevelOpsToErase.push_back(&op);
            }
        }
        
        // Check inside func.func operations
        if (auto funcOp = dyn_cast<func::FuncOp>(&op)) {
            funcOp.walk([&](Operation *nestedOp) {
                if (nestedOp == &op) return;
                
                StringRef nestedOpName = nestedOp->getName().getStringRef();
                
                if (nestedOp->getParentOp() == funcOp.getOperation()) {
                    if (nestedOpName.starts_with("dfschedule.") || 
                        nestedOpName.starts_with("dfscheblueprint.") ||
                        nestedOpName.starts_with("routing.")) {
                        topLevelOpsToErase.push_back(nestedOp);
                    }
                    
                    if (auto constOp = dyn_cast<arith::ConstantOp>(nestedOp)) {
                        if (isa<DenseElementsAttr>(constOp.getValue())) {
                            topLevelOpsToErase.push_back(nestedOp);
                        }
                    }
                }
            });
        }
    }
    
    llvm::errs() << "[Pass] Found " << topLevelOpsToErase.size() << " top-level ops to erase\n";
    
    // Drop all uses first
    for (Operation *op : topLevelOpsToErase) {
        for (Value result : op->getResults()) {
            result.dropAllUses();
        }
        op->walk([](Operation *nestedOp) {
            for (Value result : nestedOp->getResults()) {
                result.dropAllUses();
            }
        });
    }
    
    // Erase top-level ops
    for (Operation *op : topLevelOpsToErase) {
        llvm::errs() << "[Pass] Erasing: " << op->getName() << "\n";
        op->erase();
    }
    
    llvm::errs() << "=== DfscheduleToApiPass SUCCESS ===\n";
}

} // namespace mlir
