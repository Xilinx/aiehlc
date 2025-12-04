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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include <sstream>
#include <vector>
#include <unordered_map>
#include <iostream>

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

namespace {

// Helper function to look up TileGroupOp by symbol reference
static dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target) {
    // Walk up to find the ConfigOp that contains the TileGroupOp definitions
    Operation *configOp = rootOp->getParentOfType<dfscheblueprint::ConfigOp>();
    if (!configOp) {
        // Try to find ConfigOp among siblings
        if (auto parentOp = rootOp->getParentOp()) {
            for (Operation &op : parentOp->getRegion(0).front()) {
                if (auto config = dyn_cast<dfscheblueprint::ConfigOp>(&op)) {
                    configOp = config;
                    break;
                }
            }
        }
    }
    
    if (!configOp) return nullptr;
    
    // Search within the ConfigOp's body for matching TileGroupOp
    StringRef targetName = target.getRootReference().getValue();
    for (Operation &op : cast<dfscheblueprint::ConfigOp>(configOp).getBody().front()) {
        if (auto tileGroup = dyn_cast<dfscheblueprint::TileGroupOp>(&op)) {
            if (tileGroup.getSymName() == targetName) {
                return tileGroup;
            }
        }
    }
    return nullptr;
}

// Pattern to convert dfscheblueprint::FlowConfigOp to dfschedule operations
// Only handles FlowConfigOp with type="shim" - creates declaretile and config.dma_bd
struct FlowConfigConversion : public OpConversionPattern<dfscheblueprint::FlowConfigOp> {
    using OpConversionPattern<dfscheblueprint::FlowConfigOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dfscheblueprint::FlowConfigOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Only handle FlowConfigOp with type="shim"
        auto typeAttr = op.getType();
        if (!typeAttr || *typeAttr != "shim") {
            // Skip non-shim config operations - they will be handled by other patterns
            return failure();
        }
        
        // Get the target TileGroupOp by looking up the symbol reference
        SymbolRefAttr targetRef = op.getTarget();
        auto tileGroupOp = lookupTileGroup(op.getOperation(), targetRef);
        if (!tileGroupOp) {
            return op.emitError("failed to resolve target tile group: ") << targetRef;
        }
        
        // Extract tile coordinates from the TileGroupOp
        ArrayAttr tilesAttr = tileGroupOp.getTiles();
        if (tilesAttr.empty()) {
            return op.emitError("target tile group has no tiles");
        }
        
        // Get DMA configuration
        auto dmaAttr = op.getDma();
        auto dmaChannels = dmaAttr.getChannels();
        auto dmaDirection = dmaAttr.getDirection();
        
        // Get the view operand (tensor/memref containing the data)
        Value viewValue = adaptor.getView();
        Type viewType = viewValue.getType();
        
        // Convert tensor type to memref type if needed
        MemRefType memrefType;
        if (auto tensorType = dyn_cast<RankedTensorType>(viewType)) {
            memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            // Create a buffer cast from tensor to memref
            viewValue = rewriter.create<mlir::bufferization::ToMemrefOp>(loc, memrefType, viewValue);
        } else if (auto mrType = dyn_cast<MemRefType>(viewType)) {
            memrefType = mrType;
        } else {
            return op.emitError("view must be tensor or memref type");
        }
        
        // Create declaretile and config.dma_bd for each tile in the target group
        SmallVector<Value> bdHandles;
        int channelIdx = 0;
        
        for (auto tileAttr : tilesAttr) {
            auto tileArray = dyn_cast<ArrayAttr>(tileAttr);
            if (!tileArray || tileArray.size() < 2) {
                continue;
            }
            
            // Extract tile coordinates
            int64_t col = cast<IntegerAttr>(tileArray[0]).getInt();
            int64_t row = cast<IntegerAttr>(tileArray[1]).getInt();
            
            // Create dfschedule.declaretile to declare the physical tile
            auto declareTileOp = rewriter.create<dfschedule::DeclareTileOp>(
                loc,
                dfschedule::TileType::get(rewriter.getContext()),
                rewriter.getI32IntegerAttr(col),
                rewriter.getI32IntegerAttr(row));
            
            // Get DMA channel for this tile (cycle through available channels)
            int64_t dmaChannel = dmaChannels.empty() ? 0 : dmaChannels[channelIdx % dmaChannels.size()];
            channelIdx++;
            
            // Calculate buffer size from memref shape
            int64_t bufferLen = 1;
            for (int64_t dim : memrefType.getShape()) {
                bufferLen *= dim;
            }
            
            // Create dfschedule.config.dma_bd to configure DMA buffer descriptor
            // Parameters: buffer, tile, bd_id, offset, len, enable_packet, packet_id, next_bd
            auto configDmaBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
                loc,
                dfschedule::BdHandleType::get(rewriter.getContext()),
                viewValue,                                      // buffer
                declareTileOp.getTile(),                        // tile handle
                rewriter.getI32IntegerAttr(dmaChannel),         // bd_id (use channel as bd_id)
                rewriter.getI32IntegerAttr(0),                  // offset
                rewriter.getI32IntegerAttr(bufferLen),          // len
                rewriter.getBoolAttr(false),                    // enable_packet
                rewriter.getI32IntegerAttr(0),                  // packet_id
                rewriter.getI32IntegerAttr(0));                 // next_bd
            
            bdHandles.push_back(configDmaBdOp.getBdHandle());
        }
        
        // Erase the original FlowConfigOp
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
                          bufferization::BufferizationDialect,
                          BuiltinDialect>();
    
    // Mark shim-type FlowConfigOp as illegal to trigger conversion
    target.addDynamicallyLegalOp<dfscheblueprint::FlowConfigOp>([](dfscheblueprint::FlowConfigOp op) {
        // Only shim type should be converted, others remain legal
        auto typeAttr = op.getType();
        return !typeAttr || *typeAttr != "shim";
    });
    
    // Type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    
    // Convert tensor types to memref types where needed
    typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    });
    
    RewritePatternSet patterns(context);
    patterns.add<FlowConfigConversion>(context);
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
