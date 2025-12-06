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
// The TileGroupOp is typically a sibling operation in the same block as the FlowConfigOp
static dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target) {
    StringRef targetName = target.getRootReference().getValue();
    
    // First, search in the same block as the FlowConfigOp
    Block *parentBlock = rootOp->getBlock();
    if (parentBlock) {
        for (Operation &op : *parentBlock) {
            if (auto tileGroup = dyn_cast<dfscheblueprint::TileGroupOp>(&op)) {
                if (tileGroup.getSymName() == targetName) {
                    return tileGroup;
                }
            }
        }
    }
    
    // If not found, try searching in parent regions (for nested structures)
    Operation *parentOp = rootOp->getParentOp();
    while (parentOp) {
        for (Region &region : parentOp->getRegions()) {
            for (Block &block : region) {
                for (Operation &op : block) {
                    if (auto tileGroup = dyn_cast<dfscheblueprint::TileGroupOp>(&op)) {
                        if (tileGroup.getSymName() == targetName) {
                            return tileGroup;
                        }
                    }
                }
            }
        }
        parentOp = parentOp->getParentOp();
    }
    
    return nullptr;
}

// Unified template pattern to erase dfscheblueprint operations
// FlowConfigOp is just erased since FlowTransferConversion reads its attributes
// and generates all the DMA BD configuration logic
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

// Helper function to look up FlowConfigOp by symbol reference
static dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target) {
    StringRef targetName = target.getRootReference().getValue();
    
    // Search in the same block as the FlowTransferOp
    Block *parentBlock = rootOp->getBlock();
    if (parentBlock) {
        for (Operation &op : *parentBlock) {
            if (auto flowConfig = dyn_cast<dfscheblueprint::FlowConfigOp>(&op)) {
                if (flowConfig.getSymName() == targetName) {
                    return flowConfig;
                }
            }
        }
    }
    
    // Search in parent regions
    Operation *parentOp = rootOp->getParentOp();
    while (parentOp) {
        for (Region &region : parentOp->getRegions()) {
            for (Block &block : region) {
                for (Operation &op : block) {
                    if (auto flowConfig = dyn_cast<dfscheblueprint::FlowConfigOp>(&op)) {
                        if (flowConfig.getSymName() == targetName) {
                            return flowConfig;
                        }
                    }
                }
            }
        }
        parentOp = parentOp->getParentOp();
    }
    
    return nullptr;
}

// Pattern to convert dfscheblueprint::FlowTransferOp to dfschedule operations
// Generates: declaretensor, declaretile, config.dma_bd, config.create_io, packet, 
//            load_kernel_group, schedule.launch_kernel_group, schedule.getbdid, 
//            schedule.start_io, schedule.wait
struct FlowTransferConversion : public OpConversionPattern<dfscheblueprint::FlowTransferOp> {
    using OpConversionPattern<dfscheblueprint::FlowTransferOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dfscheblueprint::FlowTransferOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Look up "from" FlowConfigOp (source/sender)
        SymbolRefAttr fromRef = op.getFrom();
        auto fromFlowConfig = lookupFlowConfig(op.getOperation(), fromRef);
        if (!fromFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Look up "to" FlowConfigOp (destination/receiver)
        SymbolRefAttr toRef = op.getTo();
        auto toFlowConfig = lookupFlowConfig(op.getOperation(), toRef);
        if (!toFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get sender's tile group
        auto fromTileGroup = lookupTileGroup(fromFlowConfig.getOperation(), fromFlowConfig.getTarget());
        if (!fromTileGroup) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get receiver's tile group
        auto toTileGroup = lookupTileGroup(toFlowConfig.getOperation(), toFlowConfig.getTarget());
        if (!toTileGroup) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get base packet id for packet operations
        uint32_t basePacketId = op.getBasePacketId();
        
        // Get the view operand from the source FlowConfig
        Value viewValue = fromFlowConfig.getView();
        Type viewType = viewValue.getType();
        
        // Convert tensor type to memref type if needed
        MemRefType memrefType;
        Value memrefValue;
        if (auto tensorType = dyn_cast<RankedTensorType>(viewType)) {
            // Create 1D memref type with total size
            int64_t totalSize = 1;
            for (int64_t dim : tensorType.getShape()) {
                totalSize *= dim;
            }
            memrefType = MemRefType::get({totalSize}, tensorType.getElementType());
            
            // Create dfschedule.declaretensor
            memrefValue = rewriter.create<dfschedule::DeclareTensorOp>(
                loc, memrefType, viewValue);
        } else if (auto mrType = dyn_cast<MemRefType>(viewType)) {
            memrefType = mrType;
            memrefValue = viewValue;
        } else {
            rewriter.eraseOp(op);
            return success();
        }
        
        // --- SENDER SIDE (from FlowConfig) ---
        ArrayAttr fromTilesAttr = fromTileGroup.getTiles();
        if (fromTilesAttr.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get first sender tile coordinates
        auto firstFromTile = dyn_cast<ArrayAttr>(fromTilesAttr[0]);
        if (!firstFromTile || firstFromTile.size() < 2) {
            rewriter.eraseOp(op);
            return success();
        }
        int64_t fromCol = cast<IntegerAttr>(firstFromTile[0]).getInt();
        int64_t fromRow = cast<IntegerAttr>(firstFromTile[1]).getInt();
        
        // Create dfschedule.declaretile for sender
        auto senderTileOp = rewriter.create<dfschedule::DeclareTileOp>(
            loc,
            dfschedule::TileType::get(rewriter.getContext()),
            rewriter.getI32IntegerAttr(fromCol),
            rewriter.getI32IntegerAttr(fromRow));
        
        // Get DMA configuration from sender
        auto fromDmaAttr = fromFlowConfig.getDma();
        auto fromDmaChannels = fromDmaAttr.getChannels();
        int64_t senderChannel = fromDmaChannels.empty() ? 0 : fromDmaChannels[0];
        
        // Calculate buffer size
        int64_t bufferLen = 1;
        for (int64_t dim : memrefType.getShape()) {
            bufferLen *= dim;
        }
        
        // Create bd_id constant for config
        auto bdIdConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        
        // Create dfschedule.config.dma_bd for sender
        auto configDmaBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
            loc,
            dfschedule::BdHandleType::get(rewriter.getContext()),
            memrefValue,                                      // buffer
            senderTileOp.getTile(),                           // tile
            bdIdConst.getResult(),                            // bd_id
            rewriter.getI32IntegerAttr(0),                    // offset
            rewriter.getI32IntegerAttr(bufferLen),            // len
            rewriter.getBoolAttr(true),                       // enable_packet
            rewriter.getI32IntegerAttr(basePacketId),         // packet_id
            rewriter.getI32IntegerAttr(4294967295));          // next_bd (-1 as unsigned)
        
        // Create dfschedule.config.create_io for sender
        auto createIoOp = rewriter.create<dfschedule::ConfigCreateIoOp>(
            loc,
            dfschedule::IoHandleType::get(rewriter.getContext()),
            configDmaBdOp.getBdHandle(),                      // bd_config
            senderTileOp.getTile(),                           // tile
            rewriter.getI32IntegerAttr(senderChannel),        // channel
            rewriter.getStringAttr("MM2S"),                   // direction
            rewriter.getStringAttr("SEND"));                  // io_operation
        
        // --- RECEIVER SIDE (to FlowConfig) ---
        ArrayAttr toTilesAttr = toTileGroup.getTiles();
        SmallVector<Value> receiverTiles;
        SmallVector<SymbolRefAttr> packetSymbols;
        uint32_t packetIdx = 0;
        
        for (auto tileAttr : toTilesAttr) {
            auto tileArray = dyn_cast<ArrayAttr>(tileAttr);
            if (!tileArray || tileArray.size() < 2) {
                continue;
            }
            
            int64_t col = cast<IntegerAttr>(tileArray[0]).getInt();
            int64_t row = cast<IntegerAttr>(tileArray[1]).getInt();
            
            // Create dfschedule.declaretile for each receiver
            auto receiverTileOp = rewriter.create<dfschedule::DeclareTileOp>(
                loc,
                dfschedule::TileType::get(rewriter.getContext()),
                rewriter.getI32IntegerAttr(col),
                rewriter.getI32IntegerAttr(row));
            receiverTiles.push_back(receiverTileOp.getTile());
            
            // Create packet symbol name (packet0, packet1, ...)
            std::string packetName = "packet" + std::to_string(packetIdx);
            
            // Create dfschedule.packet for each receiver
            auto toDmaAttr = toFlowConfig.getDma();
            auto toDmaChannels = toDmaAttr.getChannels();
            int64_t recvChannel = toDmaChannels.empty() ? 0 : toDmaChannels[0];
            
            rewriter.create<dfschedule::PacketOp>(
                loc,
                dfschedule::PacketType::get(rewriter.getContext()),
                rewriter.getStringAttr(packetName),
                memrefValue,
                rewriter.getI32IntegerAttr(recvChannel));
            
            packetSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), packetName));
            packetIdx++;
        }
        
        if (receiverTiles.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Create callee symbol refs (dskernel_receiver for all)
        SmallVector<Attribute> calleeAttrs;
        calleeAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "dskernel_receiver"));
        
        // Create distributed_compute_kernel_args (compute0 for all)
        SmallVector<Attribute> computeKernelAttrs;
        for (size_t i = 0; i < receiverTiles.size(); ++i) {
            computeKernelAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "compute0"));
        }
        
        // Create distributed_args from packet symbols
        SmallVector<Attribute> distArgsAttrs(packetSymbols.begin(), packetSymbols.end());
        
        // Create dfschedule.config.load_kernel_group
        auto loadKernelGroupOp = rewriter.create<dfschedule::LoadKernelGroupOp>(
            loc,
            dfschedule::KernelGroupType::get(rewriter.getContext()),
            receiverTiles,
            rewriter.getArrayAttr(calleeAttrs),
            rewriter.getArrayAttr(computeKernelAttrs),
            rewriter.getArrayAttr(distArgsAttrs));
        
        // Create dfschedule.schedule.launch_kernel_group
        auto launchKernelGroupOp = rewriter.create<dfschedule::LaunchKernelGroupOp>(
            loc,
            dfschedule::EventType::get(rewriter.getContext()),
            loadKernelGroupOp.getKernelGroup());
        
        // Create dfschedule.schedule.getbdid
        auto getBdIdOp = rewriter.create<dfschedule::GetBdIdOp>(
            loc,
            rewriter.getI32Type(),
            senderTileOp.getTile());
        
        // Create dfschedule.schedule.start_io
        auto startIoOp = rewriter.create<dfschedule::StartIoOp>(
            loc,
            dfschedule::EventType::get(rewriter.getContext()),
            createIoOp.getIoHandle(),
            getBdIdOp.getBdId());
        
        // Create dfschedule.schedule.wait with both events
        SmallVector<Value> events;
        events.push_back(startIoOp.getEvent());
        events.push_back(launchKernelGroupOp.getEvent());
        rewriter.create<dfschedule::ScheduleWaitOp>(loc, events);
        
        // Erase the original FlowTransferOp
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
    
    // Mark all dfscheblueprint operations as illegal to trigger conversion/erasure
    target.addIllegalOp<dfscheblueprint::FlowConfigOp>();
    target.addIllegalOp<dfscheblueprint::TileGroupOp>();
    //target.addIllegalOp<dfscheblueprint::DeclareDataOp>();
    //target.addIllegalOp<dfscheblueprint::DataSliceOp>();
    target.addIllegalOp<dfscheblueprint::FlowTransferOp>();
    target.addIllegalOp<dfscheblueprint::TransferManifestOp>();
    
    // Type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    
    // Convert tensor types to memref types where needed
    typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    });
    
    RewritePatternSet patterns(context);
    // FlowTransferConversion converts flow_transfer to dfschedule operations
    // It reads from FlowConfigOps to get DMA configuration
    patterns.add<FlowTransferConversion>(context);
    // DataSliceOp replaces with input tensor
    patterns.add<DataSliceOpConversion>(context);
    // Use unified erase pattern for ops that just need to be removed
    // FlowConfigOp is erased since FlowTransferConversion reads its attributes directly
    patterns.add<EraseOpPattern<dfscheblueprint::FlowConfigOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::TileGroupOp>>(context);
    //patterns.add<EraseOpPattern<dfscheblueprint::DeclareDataOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::TransferManifestOp>>(context);
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace mlir
