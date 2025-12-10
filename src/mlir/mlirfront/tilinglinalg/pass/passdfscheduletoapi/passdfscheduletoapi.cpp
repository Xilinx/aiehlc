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
#include <atomic>

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
// Shared State for Index Tracking (thread-safe)
//===----------------------------------------------------------------------===//

struct ConversionState {
    std::atomic<int> arrayIndex{0};
    
    int nextArrayIndex() { return arrayIndex.fetch_add(1); }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert dfschedule.launchhost to emitc.call_opaque("hostruntime")
class LaunchHostToEmitCPattern : public ConversionPattern {
public:
    LaunchHostToEmitCPattern(MLIRContext *ctx)
        : ConversionPattern("dfschedule.launchhost", /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.create<emitc::CallOpaqueOp>(
            op->getLoc(),
            TypeRange{},
            rewriter.getStringAttr("hostruntime"),
            nullptr, nullptr, ValueRange{}
        );
        rewriter.eraseOp(op);
        return success();
    }
};

/// Pattern to convert dfschedule.dskernel_receiver to emitc.func with __global__
class DsKernelReceiverToEmitCPattern : public ConversionPattern {
public:
    DsKernelReceiverToEmitCPattern(MLIRContext *ctx)
        : ConversionPattern("dfschedule.dskernel_receiver", /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        std::string kernelName = "dskernel";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            kernelName = symNameAttr.getValue().str();
        }

        auto funcType = rewriter.getFunctionType({}, {});
        auto emitcFunc = rewriter.create<emitc::FuncOp>(
            op->getLoc(),
            kernelName,
            funcType
        );

        emitcFunc->setAttr("specifiers", rewriter.getStrArrayAttr({"__global__"}));

        Block *entryBlock = emitcFunc.addEntryBlock();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        rewriter.create<emitc::CallOpaqueOp>(
            op->getLoc(),
            TypeRange{},
            rewriter.getStringAttr("/* AIE kernel implementation */"),
            nullptr, nullptr, ValueRange{}
        );

        rewriter.create<emitc::ReturnOp>(op->getLoc(), Value{});
        rewriter.eraseOp(op);
        return success();
    }
};

/// Pattern to convert dfscheblueprint.declare_data to:
/// XAie_MemAllocate + emitc.for copy loop
/// Uses the array name from the emitc.constant operand (converted from arith.constant)
class DeclareDataToEmitCPattern : public ConversionPattern {
public:
    DeclareDataToEmitCPattern(MLIRContext *ctx, ConversionState &state)
        : ConversionPattern("dfscheblueprint.declare_data", /*benefit=*/1, ctx), state(state) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op->getNumResults() == 0)
            return failure();

        auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!resultType)
            return failure();

        if (op->getNumOperands() == 0)
            return failure();
        
        // Get the operand - could be emitc.constant (already converted) or arith.constant
        Value initTensor = operands[0];
        Operation *initOp = initTensor.getDefiningOp();
        
        std::string arrayName;
        
        // Find the module for inserting global arrays
        auto moduleOp = op->getParentOfType<ModuleOp>();
        if (!moduleOp)
            return failure();
        
        // Get unique index for this declare_data
        int idx = state.nextArrayIndex();
        
        // Check if operand is emitc.constant (already converted)
        if (auto emitcConstOp = dyn_cast_or_null<emitc::ConstantOp>(initOp)) {
            if (auto opaqueAttr = dyn_cast<emitc::OpaqueAttr>(emitcConstOp.getValue())) {
                arrayName = opaqueAttr.getValue().str();
            }
        }
        // Or if it's still arith.constant - handle it directly by creating global array
        else if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(initOp)) {
            auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
            if (denseAttr) {
                arrayName = "g_data_array_" + std::to_string(idx);
                
                auto tensorType = dyn_cast<RankedTensorType>(denseAttr.getType());
                if (tensorType) {
                    Type constElemType = tensorType.getElementType();
                    std::string constCTypeStr = getEmitCTypeString(constElemType);
                    std::string initStr = buildInitializerString(denseAttr, constElemType);
                    
                    // Create emitc.verbatim for the global array at module scope
                    std::string verbatimCode = "static const " + constCTypeStr + " " + arrayName +
                        buildArrayDimString(tensorType.getShape()) + " = " + initStr + ";";
                    
                    {
                        OpBuilder::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPointToStart(moduleOp.getBody());
                        rewriter.create<emitc::VerbatimOp>(op->getLoc(), rewriter.getStringAttr(verbatimCode));
                    }
                }
            }
        }
        
        // Fallback: if we still don't have an array name, generate one
        if (arrayName.empty()) {
            arrayName = "g_data_array_" + std::to_string(idx);
            
            // Create a zero-initialized array based on result type
            std::string cTypeStrFallback = getEmitCTypeString(resultType.getElementType());
            std::string verbatimCode = "static const " + cTypeStrFallback + " " + arrayName +
                buildArrayDimString(resultType.getShape()) + " = {0};";
            
            {
                OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(moduleOp.getBody());
                rewriter.create<emitc::VerbatimOp>(op->getLoc(), rewriter.getStringAttr(verbatimCode));
            }
        }

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
        
        // Extract the index from array name (g_data_array_X)
        std::string memInstName = "MemInst_" + arrayName.substr(arrayName.find_last_of('_') + 1);
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string dstPtrName = "dst_" + arrayName.substr(arrayName.find_last_of('_') + 1);

        auto loc = op->getLoc();

        // Generate: XAie_MemInst* MemInst = XAie_MemAllocate(&DevInst, Size, XAIE_MEM_CACHEABLE);
        rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(
            "XAie_MemInst* " + memInstName + " = XAie_MemAllocate(&DevInst, " +
            std::to_string(byteSize) + ", XAIE_MEM_CACHEABLE);"
        ));

        // Generate: dst pointer = XAie_MemGetVAddr(MemInst)
        rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(
            cTypeStr + "* " + dstPtrName + " = (" + cTypeStr + "*)XAie_MemGetVAddr(" + memInstName + ");"
        ));

        // Create emitc.for loop to copy data
        auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto ub = rewriter.create<arith::ConstantIndexOp>(loc, totalSize);
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto forOp = rewriter.create<emitc::ForOp>(loc, lb, ub, step);

        // Build the loop body
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(forOp.getBody());

            rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(
                dstPtrName + "[v] = ((" + cTypeStr + "*)" + arrayName + ")[v];"
            ));
        }

        // Erase the original declare_data op
        rewriter.eraseOp(op);
        return success();
    }

private:
    ConversionState &state;
};

/// Pattern to erase dfschedule.host operations
class HostOpErasePattern : public ConversionPattern {
public:
    HostOpErasePattern(MLIRContext *ctx)
        : ConversionPattern("dfschedule.host", /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

/// Pattern to convert arith.constant with DenseElementsAttr to emitc.verbatim global array
/// and emitc.constant that references the array
class DenseConstantToEmitCPattern : public OpConversionPattern<arith::ConstantOp> {
public:
    DenseConstantToEmitCPattern(MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<arith::ConstantOp>(ctx, /*benefit=*/10), state(state) {}

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!denseAttr)
            return failure();

        auto tensorType = dyn_cast<RankedTensorType>(denseAttr.getType());
        if (!tensorType)
            return failure();

        // Find the module to insert the global at the top
        auto moduleOp = op->getParentOfType<ModuleOp>();
        if (!moduleOp)
            return failure();

        int idx = state.nextArrayIndex();
        std::string arrayName = "g_data_array_" + std::to_string(idx);
        Type elemType = tensorType.getElementType();
        std::string cTypeStr = getEmitCTypeString(elemType);
        std::string initStr = buildInitializerString(denseAttr, elemType);

        // Create emitc.verbatim for the global array declaration at module scope
        std::string verbatimCode = "static const " + cTypeStr + " " + arrayName +
            buildArrayDimString(tensorType.getShape()) + " = " + initStr + ";";

        auto loc = op.getLoc();
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(moduleOp.getBody());
            rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(verbatimCode));
        }

        // Create emitc.constant to reference the global array (as a pointer)
        // This replaces the original arith.constant
        // Use emitc::PointerType wrapping an OpaqueType for the element type
        auto elemOpaqueType = emitc::OpaqueType::get(rewriter.getContext(), cTypeStr);
        auto ptrType = emitc::PointerType::get(elemOpaqueType);
        auto arrayRef = rewriter.create<emitc::ConstantOp>(
            loc,
            ptrType,
            emitc::OpaqueAttr::get(rewriter.getContext(), arrayName)
        );

        // Replace all uses of the original constant with the new emitc.constant
        rewriter.replaceOp(op, arrayRef.getResult());
        return success();
    }

private:
    ConversionState &state;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void DfscheduleToApiPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // Shared state for index tracking across patterns
    ConversionState state;

    // Setup conversion target
    ConversionTarget target(*ctx);

    // Mark EmitC dialect as legal
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalOp<arith::ConstantIndexOp>();
    
    // arith.constant with dense attr is illegal (needs conversion to emitc.constant)
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!denseAttr)
            return true; // Non-dense constants are legal
        // Dense constants are illegal - must be converted to emitc.constant
        return false;
    });

    // Mark dfschedule and dfscheblueprint operations as illegal
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
        StringRef opName = op->getName().getStringRef();
        if (opName == "dfschedule.launchhost" ||
            opName == "dfschedule.dskernel_receiver" ||
            opName == "dfschedule.host" ||
            opName == "dfscheblueprint.declare_data") {
            return false;
        }
        return true;
    });

    // Create rewrite patterns
    // DenseConstantToEmitCPattern has higher benefit to run first
    RewritePatternSet patterns(ctx);
    patterns.add<DenseConstantToEmitCPattern>(ctx, state);
    patterns.add<LaunchHostToEmitCPattern>(ctx);
    patterns.add<DsKernelReceiverToEmitCPattern>(ctx);
    patterns.add<DeclareDataToEmitCPattern>(ctx, state);
    patterns.add<HostOpErasePattern>(ctx);

    // Apply conversion
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        signalPassFailure();
        return;
    }
}

} // namespace mlir
