/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passblueprinttoschedule.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "dfscheblueprintmanager.h"
#include <sstream>
#include <vector>
#include <unordered_map>
#include <iostream>

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

namespace {

// Pattern to convert dfscheblueprint::ConfigOp to dfschedule module structure
struct ConfigOpConversion : public OpConversionPattern<dfscheblueprint::ConfigOp> {
    using OpConversionPattern<dfscheblueprint::ConfigOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dfscheblueprint::ConfigOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Create a module to contain the schedule
        rewriter.setInsertionPoint(op);
        auto moduleOp = rewriter.create<ModuleOp>(loc);
        Block *moduleBlock = &moduleOp.getBodyRegion().front();
        
        rewriter.setInsertionPointToEnd(moduleBlock);
        
        // Collect information from the blueprint
        SmallVector<dfscheblueprint::TileGroupOp> resourceGroups;
        SmallVector<dfscheblueprint::DeclareDataOp> declareDataOps;
        SmallVector<dfscheblueprint::DataSliceOp> dataSliceOps;
        SmallVector<dfscheblueprint::FlowConfigOp> bindOps;
        SmallVector<dfscheblueprint::FlowTransferOp> transferOps;
        SmallVector<dfscheblueprint::TransferManifestOp> manifestOps;
        
        // Walk the config body to collect operations
        for (Operation &innerOp : op.getBody().front().getOperations()) {
            if (auto resGroup = dyn_cast<dfscheblueprint::TileGroupOp>(&innerOp)) {
                resourceGroups.push_back(resGroup);
            } else if (auto declData = dyn_cast<dfscheblueprint::DeclareDataOp>(&innerOp)) {
                declareDataOps.push_back(declData);
            } else if (auto dataSlice = dyn_cast<dfscheblueprint::DataSliceOp>(&innerOp)) {
                dataSliceOps.push_back(dataSlice);
            } else if (auto bind = dyn_cast<dfscheblueprint::FlowConfigOp>(&innerOp)) {
                bindOps.push_back(bind);
            } else if (auto transfer = dyn_cast<dfscheblueprint::FlowTransferOp>(&innerOp)) {
                transferOps.push_back(transfer);
            } else if (auto manifest = dyn_cast<dfscheblueprint::TransferManifestOp>(&innerOp)) {
                manifestOps.push_back(manifest);
            }
        }
        
        // Create host function for data movement orchestration
        auto hostFuncType = rewriter.getFunctionType({}, {});
        auto hostFunc = rewriter.create<func::FuncOp>(loc, "schedule_host", hostFuncType);
        hostFunc.setPublic();
        
        Block *hostBody = hostFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(hostBody);
        
        // Process each resource group - get tile handles
        llvm::DenseMap<StringRef, SmallVector<Value>> tileHandles;
        for (auto resGroup : resourceGroups) {
            StringRef groupName = resGroup.getSymName();
            ArrayAttr tilesAttr = resGroup.getTiles();
            
            SmallVector<Value> handles;
            for (auto tileAttr : tilesAttr) {
                if (auto tileArray = dyn_cast<ArrayAttr>(tileAttr)) {
                    if (tileArray.size() >= 2) {
                        int64_t col = cast<IntegerAttr>(tileArray[0]).getInt();
                        int64_t row = cast<IntegerAttr>(tileArray[1]).getInt();
                        
                        auto tileHandle = rewriter.create<dfschedule::GetTileHandleOp>(
                            loc,
                            dfschedule::TileType::get(rewriter.getContext()),
                            rewriter.getI32IntegerAttr(col),
                            rewriter.getI32IntegerAttr(row));
                        handles.push_back(tileHandle.getResult());
                    }
                }
            }
            tileHandles[groupName] = handles;
        }
        
        // Allocate device memory for data declarations
        llvm::DenseMap<Operation*, Value> dataMemRefs;
        for (auto declData : declareDataOps) {
            Type dataType = declData.getDataType();
            if (auto tensorType = dyn_cast<RankedTensorType>(dataType)) {
                // Convert tensor type to memref type for device allocation
                auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
                auto allocOp = rewriter.create<dfschedule::AllocDeviceMemOp>(loc, memrefType);
                dataMemRefs[declData.getOperation()] = allocOp.getResult();
            }
        }
        
        // Process collective transfers
        SmallVector<Value> events;
        for (auto transfer : transferOps) {
            StringRef transferType = transfer.getType();
            int32_t packetId = transfer.getBasePacketId();
            
            // Get stream handle for this transfer
            auto streamHandle = rewriter.create<dfschedule::GetStreamHandleOp>(
                loc,
                dfschedule::StreamType::get(rewriter.getContext()),
                transfer.getFromAttr());
            
            // Create appropriate copy operations based on transfer type
            if (transferType == "one_to_many" || transferType == "broadcast") {
                // Host to Device scatter/broadcast
                // This would need actual host and device buffers
                // For now, create placeholder structure
            } else if (transferType == "many_to_one" || transferType == "gather") {
                // Device to Host gather
            }
        }
        
        // Add return
        rewriter.create<func::ReturnOp>(loc);
        
        // Create kernel functions for each compute tile
        for (auto &[groupName, handles] : tileHandles) {
            // Skip shim tiles (typically at row 0)
            if (groupName.contains("shim") || groupName.contains("gateway")) {
                continue;
            }
            
            rewriter.setInsertionPointToEnd(moduleBlock);
            
            // Create kernel function
            auto kernelFuncType = rewriter.getFunctionType(
                {rewriter.getI32Type()}, {});
            std::string kernelName = "kernel_" + groupName.str();
            auto kernelFunc = rewriter.create<func::FuncOp>(loc, kernelName, kernelFuncType);
            kernelFunc.setPrivate();
            
            Block *kernelBody = kernelFunc.addEntryBlock();
            rewriter.setInsertionPointToStart(kernelBody);
            
            Value tileIdx = kernelBody->getArgument(0);
            
            // Create ping-pong buffers
            auto memrefType = MemRefType::get({256}, rewriter.getF32Type());
            auto allocaPing = rewriter.create<memref::AllocaOp>(loc, memrefType);
            allocaPing->setAttr("buffer_type", rewriter.getStringAttr("ping"));
            
            auto allocaPong = rewriter.create<memref::AllocaOp>(loc, memrefType);
            allocaPong->setAttr("buffer_type", rewriter.getStringAttr("pong"));
            
            // Initialize locks
            auto lockType = dfschedule::LockType::get(rewriter.getContext());
            
            auto pingAcquireLock = rewriter.create<dfschedule::LockInitOp>(
                loc, lockType, rewriter.getI64IntegerAttr(0));
            auto pongAcquireLock = rewriter.create<dfschedule::LockInitOp>(
                loc, lockType, rewriter.getI64IntegerAttr(0));
            auto pingReleaseLock = rewriter.create<dfschedule::LockInitOp>(
                loc, lockType, rewriter.getI64IntegerAttr(1));
            auto pongReleaseLock = rewriter.create<dfschedule::LockInitOp>(
                loc, lockType, rewriter.getI64IntegerAttr(0));
            
            // Create DMA loop for receiving data
            auto dmaLoop = rewriter.create<dfschedule::LaunchDmaS2MLoopOp>(
                loc, 
                allocaPing.getResult(), 
                allocaPong.getResult(), 
                tileIdx,
                pingAcquireLock.getResult(), 
                pongAcquireLock.getResult(),
                pingReleaseLock.getResult(), 
                pongReleaseLock.getResult());
            
            Block *dmaBlock = rewriter.createBlock(&dmaLoop.getBody());
            rewriter.setInsertionPointToEnd(dmaBlock);
            
            // Set insertion point after DMA loop for compute logic
            rewriter.setInsertionPointAfter(dmaLoop);
            
            // Create compute loop with ping-pong logic
            auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            auto c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
            auto c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
            
            auto forOp = rewriter.create<scf::ForOp>(loc, c0, c4, c1);
            Block *forBody = forOp.getBody();
            rewriter.setInsertionPointToStart(forBody);
            
            Value loopIter = forOp.getInductionVar();
            
            // Check if iteration is even or odd
            auto remOp = rewriter.create<arith::RemUIOp>(loc, loopIter, c2.getResult());
            auto isEven = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, remOp.getResult(), c0.getResult());
            
            // Select buffer based on even/odd
            auto ifEvenBuf = rewriter.create<scf::IfOp>(loc, memrefType, isEven, true);
            
            rewriter.setInsertionPointToStart(&ifEvenBuf.getThenRegion().front());
            rewriter.create<scf::YieldOp>(loc, allocaPing.getResult());
            
            rewriter.setInsertionPointToStart(&ifEvenBuf.getElseRegion().front());
            rewriter.create<scf::YieldOp>(loc, allocaPong.getResult());
            
            rewriter.setInsertionPointAfter(ifEvenBuf);
            Value selectedBuffer = ifEvenBuf.getResult(0);
            
            // Select acquire lock
            auto ifEvenAcq = rewriter.create<scf::IfOp>(loc, lockType, isEven, true);
            
            rewriter.setInsertionPointToStart(&ifEvenAcq.getThenRegion().front());
            rewriter.create<scf::YieldOp>(loc, pingAcquireLock.getResult());
            
            rewriter.setInsertionPointToStart(&ifEvenAcq.getElseRegion().front());
            rewriter.create<scf::YieldOp>(loc, pongAcquireLock.getResult());
            
            rewriter.setInsertionPointAfter(ifEvenAcq);
            Value selectedAcquireLock = ifEvenAcq.getResult(0);
            
            // Select release lock
            auto ifEvenRel = rewriter.create<scf::IfOp>(loc, lockType, isEven, true);
            
            rewriter.setInsertionPointToStart(&ifEvenRel.getThenRegion().front());
            rewriter.create<scf::YieldOp>(loc, pingReleaseLock.getResult());
            
            rewriter.setInsertionPointToStart(&ifEvenRel.getElseRegion().front());
            rewriter.create<scf::YieldOp>(loc, pongReleaseLock.getResult());
            
            rewriter.setInsertionPointAfter(ifEvenRel);
            Value selectedReleaseLock = ifEvenRel.getResult(0);
            
            // Calculate lock value (iteration + 1)
            auto c1_i32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
            auto idxCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), loopIter);
            auto lockVal = rewriter.create<arith::AddIOp>(loc, idxCast.getResult(), c1_i32.getResult());
            
            // Acquire lock
            rewriter.create<dfschedule::AcquireLockOp>(loc, selectedAcquireLock, lockVal.getResult());
            
            // Compute on the selected buffer
            rewriter.create<dfschedule::ComputeOp>(loc, selectedBuffer);
            
            // Release lock
            rewriter.create<dfschedule::ReleaseLockOp>(loc, selectedReleaseLock, lockVal.getResult());
            
            // Add return to kernel
            rewriter.setInsertionPointToEnd(kernelBody);
            rewriter.create<func::ReturnOp>(loc);
        }
        
        // Erase the original config op
        rewriter.eraseOp(op);
        
        return success();
    }
};

// Unified template pattern to erase dfscheblueprint operations
template <typename OpTy>
struct EraseOpPattern : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// Special pattern for DataSliceOp - replaces with input tensor instead of erasing
struct DataSliceOpConversion : public OpConversionPattern<dfscheblueprint::DataSliceOp> {
    using OpConversionPattern<dfscheblueprint::DataSliceOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dfscheblueprint::DataSliceOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // DataSliceOp is used for symbol references, replace with the input tensor
        rewriter.replaceOp(op, adaptor.getTensorSlice());
        return success();
    }
};

} // namespace

namespace mlir {

void BlueprintToSchedulePass::runOnOperation() {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    
    // Mark target dialects as legal
    target.addLegalDialect<dfschedule::dfscheduledialect, 
                          func::FuncDialect,
                          memref::MemRefDialect,
                          arith::ArithDialect,
                          scf::SCFDialect,
                          tensor::TensorDialect,
                          BuiltinDialect>();
    
    // Mark dfscheblueprint operations as illegal to trigger conversion
    //target.addIllegalDialect<dfscheblueprint::dfscheblueprintdialect>();
    
    // Type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    
    // Convert tensor types to memref types where needed
    typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    });
    
    RewritePatternSet patterns(context);
    patterns.add<ConfigOpConversion>(context);
    patterns.add<DataSliceOpConversion>(context);
    // Use unified erase pattern for ops that just need to be removed
    /*
    patterns.add<EraseOpPattern<dfscheblueprint::TileGroupOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::DeclareDataOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::FlowConfigOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::FlowTransferOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::TransferManifestOp>>(context);
    */
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace mlir
