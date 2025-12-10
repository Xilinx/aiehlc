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

//===----------------------------------------------------------------------===//
// Pass Implementation - Two Phase Approach
//===----------------------------------------------------------------------===//

void DfscheduleToApiPass::runOnOperation() {
    llvm::errs() << "=== DfscheduleToApiPass START ===\n";
    
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    
    int arrayIndex = 0;
    int partitionIndex = 0;
    
    // Map to track memory allocations: tensor Value -> (memInstName, dstPtrName, byteSize)
    DenseMap<Value, std::tuple<std::string, std::string, int64_t>> memAllocMap;
    
    // Collect all operations to process
    SmallVector<Operation*> hostOps;
    SmallVector<Operation*> launchHostOps;
    SmallVector<Operation*> dsKernelReceiverOps;
    SmallVector<Operation*> declareDataOps;
    SmallVector<Operation*> partitionTensorOps;
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
        } else if (opName == "dfscheblueprint.declare_data") {
            declareDataOps.push_back(op);
        } else if (opName == "routing.partitiontensor") {
            partitionTensorOps.push_back(op);
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
    llvm::errs() << "[Pass] Found " << declareDataOps.size() << " declare_data ops\n";
    llvm::errs() << "[Pass] Found " << partitionTensorOps.size() << " partitiontensor ops\n";
    llvm::errs() << "[Pass] Found " << allDfscheduleOps.size() << " total dfschedule/dfscheblueprint/routing ops\n";
    
    //==========================================================================
    // Phase 1: Generate EmitC code
    //==========================================================================
    
    // 1a-0. Generate PartitionTensor struct definition at module scope
    builder.setInsertionPointToStart(moduleOp.getBody());
    builder.create<emitc::VerbatimOp>(moduleOp.getLoc(), builder.getStringAttr(
        "/* PartitionTensor structure for memory management */\n"
        "typedef struct {\n"
        "    void* data;           /* Memory pointer from XAie_MemGetVAddr */\n"
        "    size_t size;          /* Size in bytes */\n"
        "    size_t num_elements;  /* Total number of elements */\n"
        "    int splitnum;         /* Split number */\n"
        "    int splitdim;         /* Split dimension */\n"
        "} PartitionTensor;"
    ));
    
    // 1a. Convert arith.constant dense to emitc.verbatim global arrays
    for (Operation *op : arithConstantDenseOps) {
        auto constOp = cast<arith::ConstantOp>(op);
        auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());
        auto tensorType = dyn_cast<RankedTensorType>(denseAttr.getType());
        if (!tensorType) continue;
        
        std::string arrayName = "g_data_array_" + std::to_string(arrayIndex++);
        Type elemType = tensorType.getElementType();
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string initStr = buildInitializerString(denseAttr, elemType);
        
        std::string verbatimCode = "static const " + cTypeStr + " " + arrayName +
            buildArrayDimString(tensorType.getShape()) + " = " + initStr + ";";
        
        builder.setInsertionPointToStart(moduleOp.getBody());
        builder.create<emitc::VerbatimOp>(op->getLoc(), builder.getStringAttr(verbatimCode));
        
        // Store array name as attribute for later use
        op->setAttr("emitc_array_name", builder.getStringAttr(arrayName));
        
        llvm::errs() << "[Pass] Created global array: " << arrayName << "\n";
    }
    
    // 1b. Convert dfschedule.host to emitc.func
    for (Operation *op : hostOps) {
        std::string funcName = "hostruntime";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            funcName = symNameAttr.getValue().str();
        }
        
        builder.setInsertionPoint(op);
        auto funcType = builder.getFunctionType({}, {});
        auto emitcFunc = builder.create<emitc::FuncOp>(op->getLoc(), funcName, funcType);
        Block *entryBlock = emitcFunc.addEntryBlock();
        
        // Process nested operations - generate XAie_MemAllocate for declare_data
        if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
            Block &srcBlock = op->getRegion(0).front();
            builder.setInsertionPointToStart(entryBlock);
            
            for (Operation &nestedOp : srcBlock.getOperations()) {
                StringRef nestedOpName = nestedOp.getName().getStringRef();
                
                if (nestedOpName == "dfscheblueprint.declare_data") {
                    // Generate XAie_MemAllocate
                    if (nestedOp.getNumResults() == 0) continue;
                    
                    auto resultType = dyn_cast<RankedTensorType>(nestedOp.getResult(0).getType());
                    if (!resultType) continue;
                    
                    // Get the array name from the input arith.constant
                    std::string arrayName = "g_data_array_" + std::to_string(arrayIndex++);
                    if (nestedOp.getNumOperands() > 0) {
                        Value initTensor = nestedOp.getOperand(0);
                        if (Operation *initOp = initTensor.getDefiningOp()) {
                            if (auto nameAttr = initOp->getAttrOfType<StringAttr>("emitc_array_name")) {
                                arrayName = nameAttr.getValue().str();
                            }
                        }
                    }
                    
                    // Calculate sizes
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
                    std::string cTypeStr = getEmitCTypeString(elemType);
                    
                    // Extract index from array name
                    std::string idxStr = arrayName.substr(arrayName.find_last_of('_') + 1);
                    std::string memInstName = "MemInst_" + idxStr;
                    std::string dstPtrName = "dst_" + idxStr;
                    
                    auto loc = nestedOp.getLoc();
                    
                    // Generate: XAie_MemInst* MemInst = XAie_MemAllocate(&DevInst, Size, XAIE_MEM_CACHEABLE);
                    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
                        "XAie_MemInst* " + memInstName + " = XAie_MemAllocate(&DevInst, " +
                        std::to_string(byteSize) + ", XAIE_MEM_CACHEABLE);"
                    ));
                    
                    // Generate: dst pointer = XAie_MemGetVAddr(MemInst)
                    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
                        cTypeStr + "* " + dstPtrName + " = (" + cTypeStr + "*)XAie_MemGetVAddr(" + memInstName + ");"
                    ));
                    
                    // Generate: for loop to copy data
                    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(
                        "for (size_t i = 0; i < " + std::to_string(totalSize) + "; i++) { " +
                        dstPtrName + "[i] = ((" + cTypeStr + "*)" + arrayName + ")[i]; }"
                    ));
                    
                    // Store memory allocation info for later use by partitiontensor
                    memAllocMap[nestedOp.getResult(0)] = std::make_tuple(memInstName, dstPtrName, byteSize);
                    
                    llvm::errs() << "[Pass] Generated XAie_MemAllocate for " << arrayName << "\n";
                }
                
                // Handle routing.partitiontensor - create PartitionTensor struct instance
                if (nestedOpName == "routing.partitiontensor") {
                    if (nestedOp.getNumResults() == 0 || nestedOp.getNumOperands() == 0) continue;
                    
                    auto resultType = dyn_cast<RankedTensorType>(nestedOp.getResult(0).getType());
                    if (!resultType) continue;
                    
                    Value inputTensor = nestedOp.getOperand(0);
                    auto loc = nestedOp.getLoc();
                    
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
                    std::string cTypeStr = getEmitCTypeString(elemType);
                    
                    // Generate struct instance name
                    std::string partitionName = "partition_" + std::to_string(partitionIndex++);
                    
                    // Try to find the source memory from memAllocMap
                    std::string srcPtrName = "NULL";
                    if (memAllocMap.count(inputTensor)) {
                        srcPtrName = std::get<1>(memAllocMap[inputTensor]);
                    } else {
                        // Check if input is from another partitiontensor or declare_data
                        Operation *srcOp = inputTensor.getDefiningOp();
                        if (srcOp) {
                            // Walk up to find the original memory
                            for (auto &entry : memAllocMap) {
                                srcPtrName = std::get<1>(entry.second);
                                break; // Use first available for now
                            }
                        }
                    }
                    
                    // Generate: PartitionTensor partition_X = { .data = ptr, .size = ..., ... };
                    std::ostringstream structInit;
                    structInit << "PartitionTensor " << partitionName << " = {\n"
                               << "    .data = (void*)" << srcPtrName << ",\n"
                               << "    .size = " << partitionByteSize << ",\n"
                               << "    .num_elements = " << totalElements << ",\n"
                               << "    .splitnum = " << splitnum << ",\n"
                               << "    .splitdim = " << splitdim << "\n"
                               << "};";
                    
                    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(structInit.str()));
                    
                    // Store for potential chained partitions
                    memAllocMap[nestedOp.getResult(0)] = std::make_tuple(partitionName, partitionName + ".data", partitionByteSize);
                    
                    llvm::errs() << "[Pass] Created PartitionTensor struct: " << partitionName << "\n";
                }
            }
        }
        
        // Add emitc.return at the end
        builder.setInsertionPointToEnd(entryBlock);
        builder.create<emitc::ReturnOp>(op->getLoc(), Value{});
        
        llvm::errs() << "[Pass] Created emitc.func: " << funcName << "\n";
    }
    
    // 1c. Convert dfschedule.launchhost to emitc.call_opaque("hostruntime")
    for (Operation *op : launchHostOps) {
        builder.setInsertionPoint(op);
        builder.create<emitc::CallOpaqueOp>(
            op->getLoc(),
            TypeRange{},
            builder.getStringAttr("hostruntime"),
            nullptr, nullptr, ValueRange{}
        );
        llvm::errs() << "[Pass] Created hostruntime() call\n";
    }
    
    // 1d. Convert dfschedule.dskernel_receiver to emitc.func with __global__
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
    // (direct children of module or func)
    // Nested ops will be erased automatically when their parent is erased
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
                // Skip the func itself
                if (nestedOp == &op) return;
                
                StringRef nestedOpName = nestedOp->getName().getStringRef();
                
                // Only collect if direct child of func's body (not nested deeper in another op we'll erase)
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
        // Drop uses of this op's results
        for (Value result : op->getResults()) {
            result.dropAllUses();
        }
        // Also drop uses of any nested op results
        op->walk([](Operation *nestedOp) {
            for (Value result : nestedOp->getResults()) {
                result.dropAllUses();
            }
        });
    }
    
    // Erase top-level ops (nested ops are erased automatically)
    for (Operation *op : topLevelOpsToErase) {
        llvm::errs() << "[Pass] Erasing: " << op->getName() << "\n";
        op->erase();
    }
    
    llvm::errs() << "=== DfscheduleToApiPass SUCCESS ===\n";
}

} // namespace mlir
