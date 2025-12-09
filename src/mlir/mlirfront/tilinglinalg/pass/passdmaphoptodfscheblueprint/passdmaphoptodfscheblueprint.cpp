/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passdmaphoptodfscheblueprint.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <atomic>

using namespace mlir;
using namespace dmaphop;
using namespace dfscheblueprint;
using namespace routing;

namespace mlir {

// Global counter for unique sequential naming (avoids conflict)
static std::atomic<int> g_pullPushCounter{0};

// Helper to process pull (Gather)
void processPull(dmaphop::pull op, OpBuilder &builder, dfscheblueprint::ConfigOp configOp, Value viewHandle) {
    // 1. Get path and extract source/destination from producers/consumers attributes
    auto pathValue = op.getPath();
    auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pathValue.getDefiningOp());
    if (!pathOp) {
        llvm::errs() << "ERROR: processPull - failed to get create_path from pull op\n";
        return;
    }
    
    SmallVector<Attribute> sourceTiles;
    SmallVector<Attribute> destTiles;
    int64_t sourceChannel = -1;
    int64_t destChannel = -1;
    std::string srcTileType;
    std::string dstTileType;
    
    // Get producers (sources) from create_path attribute via symbol table lookup
    auto producersAttr = pathOp.getProducers();
    if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
        for (auto innerAttr : arrayAttr) {
            if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                for (auto symbolAttr : innerArray) {
                    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(symbolAttr);
                    if (!symbolRef) {
                        llvm::errs() << "ERROR: processPull - producer is not a FlatSymbolRefAttr\n";
                        continue;
                    }
                    auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPull - failed to find producer port symbol '" 
                                    << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
            sourceChannel = static_cast<int64_t>(port.getDirectionChannel().value());
            auto tileValue = port.getTile();
                    auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                    if (!tileOp) {
                        llvm::errs() << "ERROR: processPull - failed to get tile from producer port '" 
                                    << symbolRef.getValue() << "'\n";
                        continue;
                    }
                    srcTileType = tileOp.getTiletype().str();
                sourceTiles.push_back(builder.getArrayAttr({
                    builder.getI64IntegerAttr(tileOp.getCol()),
                    builder.getI64IntegerAttr(tileOp.getRow())
                }));
            }
        }
    }
    }
    
    if (sourceTiles.empty()) {
        llvm::errs() << "WARNING: processPull - no source tiles found from producers attribute\n";
    }
    
    // Get unique sequential ID for naming
    int opId = g_pullPushCounter.fetch_add(1);
    
    std::string srcGroupName = "group_src_" + std::to_string(opId);
    builder.create<dfscheblueprint::TileGroupOp>(
        op.getLoc(),
        srcGroupName,
        builder.getArrayAttr(sourceTiles)
    );
    
    // Get consumers (destinations) from create_path attribute via symbol table lookup
        auto consumersAttr = pathOp.getConsumers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
        for (auto innerAttr : arrayAttr) {
            if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                for (auto symbolAttr : innerArray) {
                    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(symbolAttr);
                    if (!symbolRef) {
                        llvm::errs() << "ERROR: processPull - consumer is not a FlatSymbolRefAttr\n";
                        continue;
                    }
                            auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPull - failed to find consumer port symbol '" 
                                    << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                                destChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                    auto tileValue = port.getTile();
                    auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                    if (!tileOp) {
                        llvm::errs() << "ERROR: processPull - failed to get tile from consumer port '" 
                                    << symbolRef.getValue() << "'\n";
                        continue;
                    }
                    dstTileType = tileOp.getTiletype().str();
                                destTiles.push_back(builder.getArrayAttr({
                                    builder.getI64IntegerAttr(tileOp.getCol()),
                                    builder.getI64IntegerAttr(tileOp.getRow())
                                }));
                            }
                        }
                    }
                }
    
    if (destTiles.empty()) {
        llvm::errs() << "WARNING: processPull - no destination tiles found from consumers attribute\n";
    }
    
    std::string dstGroupName = "group_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::TileGroupOp>(
        op.getLoc(),
        dstGroupName,
        builder.getArrayAttr(destTiles)
    );
    
    // 3. Create Binds
    std::string srcBindName = "flow_src_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(),
        srcBindName,
        FlatSymbolRefAttr::get(builder.getContext(), srcGroupName),
        viewHandle,
        builder.getStringAttr("linear"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({sourceChannel}), dfscheblueprint::bp_direction::MM2S),
        builder.getArrayAttr({}),
        builder.getStringAttr(srcTileType) 
    );
    
    std::string dstBindName = "flow_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(),
        dstBindName,
        FlatSymbolRefAttr::get(builder.getContext(), dstGroupName),
        viewHandle,
        builder.getStringAttr("root"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({destChannel}), dfscheblueprint::bp_direction::S2MM),
        nullptr, // slice_symbols
        builder.getStringAttr(dstTileType)
    );
    
    // 4. Collective Transfer
    builder.create<dfscheblueprint::FlowTransferOp>(
        op.getLoc(),
        "transfer_" + std::to_string(opId),
        "many_to_one",
        FlatSymbolRefAttr::get(builder.getContext(), srcBindName),
        FlatSymbolRefAttr::get(builder.getContext(), dstBindName),
        builder.getStringAttr("sequential"),
        0
    );
}

// Helper to process PushOp (Scatter)
void processPush(dmaphop::push op, OpBuilder &builder, dfscheblueprint::ConfigOp configOp, Value viewHandle) {
    // 1. Identify Source (One)
    // Push %data to %path
    // Source is implicit or we need to find where %data comes from?
    // Usually Push is from Shim/Host.
    // Let's look at the path producers.
    
    auto pathValue = op.getPath();
    int64_t srcChannel = -1;
    SmallVector<Attribute> srcTiles;
    SmallVector<Attribute> dstTiles;
    int64_t dstChannel = -1;
    std::string srcTileType;
    std::string dstTileType;
    
    auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pathValue.getDefiningOp());
    if (!pathOp) {
        llvm::errs() << "ERROR: processPush - failed to get create_path from push op\n";
        return;
    }
    
    // Producers (sources) via symbol table lookup
        auto producersAttr = pathOp.getProducers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
        for (auto innerAttr : arrayAttr) {
            if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                for (auto symbolAttr : innerArray) {
                    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(symbolAttr);
                    if (!symbolRef) {
                        llvm::errs() << "ERROR: processPush - producer is not a FlatSymbolRefAttr\n";
                        continue;
                    }
                            auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPush - failed to find producer port symbol '" 
                                    << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                                srcChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                    auto tileValue = port.getTile();
                    auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                    if (!tileOp) {
                        llvm::errs() << "ERROR: processPush - failed to get tile from producer port '" 
                                    << symbolRef.getValue() << "'\n";
                        continue;
                    }
                    srcTileType = tileOp.getTiletype().str();
                                srcTiles.push_back(builder.getArrayAttr({
                                    builder.getI64IntegerAttr(tileOp.getCol()),
                                    builder.getI64IntegerAttr(tileOp.getRow())
                                }));
                            }
                        }
                    }
                }
    
    if (srcTiles.empty()) {
        llvm::errs() << "WARNING: processPush - no source tiles found from producers attribute\n";
    }
    
    // Consumers (destinations) via symbol table lookup
        auto consumersAttr = pathOp.getConsumers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
        for (auto innerAttr : arrayAttr) {
            if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                for (auto symbolAttr : innerArray) {
                    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(symbolAttr);
                    if (!symbolRef) {
                        llvm::errs() << "ERROR: processPush - consumer is not a FlatSymbolRefAttr\n";
                        continue;
                    }
                            auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPush - failed to find consumer port symbol '" 
                                    << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                    dstChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                    auto tileValue = port.getTile();
                    auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                    if (!tileOp) {
                        llvm::errs() << "ERROR: processPush - failed to get tile from consumer port '" 
                                    << symbolRef.getValue() << "'\n";
                        continue;
                    }
                    dstTileType = tileOp.getTiletype().str();
                                dstTiles.push_back(builder.getArrayAttr({
                                    builder.getI64IntegerAttr(tileOp.getCol()),
                                    builder.getI64IntegerAttr(tileOp.getRow())
                                }));
                            }
                        }
                    }
                }
    
    if (dstTiles.empty()) {
        llvm::errs() << "WARNING: processPush - no destination tiles found from consumers attribute\n";
    }
    
    // Get unique sequential ID for naming
    int opId = g_pullPushCounter.fetch_add(1);
    
    std::string srcGroupName = "group_src_" + std::to_string(opId);
    builder.create<dfscheblueprint::TileGroupOp>(
        op.getLoc(),
        srcGroupName,
        builder.getArrayAttr(srcTiles)
    );
    
    std::string dstGroupName = "group_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::TileGroupOp>(
        op.getLoc(),
        dstGroupName,
        builder.getArrayAttr(dstTiles)
    );
    
    // Binds
    std::string srcBindName = "flow_src_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(),
        srcBindName,
        FlatSymbolRefAttr::get(builder.getContext(), srcGroupName),
        viewHandle,
        builder.getStringAttr("root"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({srcChannel}), dfscheblueprint::bp_direction::MM2S),
        nullptr, // slice_symbols
        builder.getStringAttr(srcTileType)
    );
    
    std::string dstBindName = "flow_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(),
        dstBindName,
        FlatSymbolRefAttr::get(builder.getContext(), dstGroupName),
        viewHandle,
        builder.getStringAttr("linear"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({dstChannel}), dfscheblueprint:: bp_direction::S2MM),
        builder.getArrayAttr({}),
        builder.getStringAttr(dstTileType)
    );
    
    builder.create<dfscheblueprint::FlowTransferOp>(
        op.getLoc(),
        "transfer_" + std::to_string(opId),
        "one_to_many",
        FlatSymbolRefAttr::get(builder.getContext(), srcBindName),
        FlatSymbolRefAttr::get(builder.getContext(), dstBindName),
        builder.getStringAttr(""),
        0
    );
}

// Generic lowering pattern to erase an op that is no longer needed.
template <typename Op_T>
struct EraseOpLowering : public OpConversionPattern<Op_T> {
    using OpConversionPattern<Op_T>::OpConversionPattern;
    LogicalResult matchAndRewrite(Op_T op, typename Op_T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

struct CreateScheduleTensorConversion : public OpConversionPattern<routing::createscheduletensor> {
    using OpConversionPattern<routing::createscheduletensor>::OpConversionPattern;
    LogicalResult matchAndRewrite(routing::createscheduletensor op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto shapeAttr = op.getShape();
        SmallVector<int64_t> shape;
        for (auto dim : shapeAttr) {
            shape.push_back(cast<IntegerAttr>(dim).getInt());
        }
        // Get element type from the original op's result type
        auto origResultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
        Type elementType = origResultType ? origResultType.getElementType() : rewriter.getF32Type();
        auto tensorType = RankedTensorType::get(shape, elementType);
        
        // Get the init_tensor from adaptor (converted value)
        Value initTensor = adaptor.getInitTensor();
        
        // Convert to dfscheblueprint::DeclareDataOp with init_tensor
        auto declareDataOp = rewriter.create<dfscheblueprint::DeclareDataOp>(
            op.getLoc(),
            tensorType,     // result type (AnyTensor)
            initTensor      // init_tensor input
        );
        rewriter.replaceOp(op, declareDataOp.getResult());
        return success();
    }
};

struct PartitionTensorConversion : public OpConversionPattern<routing::partitiontensor> {
    using OpConversionPattern<routing::partitiontensor>::OpConversionPattern;
    LogicalResult matchAndRewrite(routing::partitiontensor op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOp(op, adaptor.getTensor());
        return success();
    }
};

struct ExtractDataConversion : public OpConversionPattern<routing::extract_data> {
    using OpConversionPattern<routing::extract_data>::OpConversionPattern;
    LogicalResult matchAndRewrite(routing::extract_data op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Get the input tensor (from adaptor for converted value)
        Value inputTensor = adaptor.getTensor();
        auto inputType = dyn_cast<RankedTensorType>(inputTensor.getType());
        auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
        
        if (!inputType || !resultType) {
            return op.emitError("ExtractDataConversion: input and result must be ranked tensors");
        }
        
        auto inputShape = inputType.getShape();
        auto resultShape = resultType.getShape();
        
        if (inputShape.size() != 2 || resultShape.size() != 2) {
            return op.emitError("ExtractDataConversion: tensors must be 2D");
        }

        int64_t splitDim = 0;
        int64_t sliceSize = 0;
        bool partitionFound = false;

        // 1. Try to get partition info
        if (auto partitionOp = dyn_cast_or_null<routing::partitiontensor>(op.getTensor().getDefiningOp())) {
            splitDim = partitionOp.getSplitdim();
        int64_t splitNum = partitionOp.getSplitnum();
            if (splitNum > 0) {
                 if (splitDim == 0) sliceSize = inputShape[0] / splitNum;
                 else sliceSize = inputShape[1] / splitNum;
            }
            partitionFound = true;
        }

        // 2. Infer/Override from result shape (Crucial for validity)
        if (resultShape[0] != inputShape[0]) {
             splitDim = 0;
             sliceSize = resultShape[0];
        } else if (resultShape[1] != inputShape[1]) {
             splitDim = 1;
             sliceSize = resultShape[1];
        } else if (!partitionFound) {
             // Identity default
             splitDim = 0;
             sliceSize = inputShape[0];
        }

        // 3. Constant Index Analysis
        Value indexValRaw = op.getIdx();
        int64_t constIndex = -1;  // Default: unknown index
        bool isConstIndex = false;
        
        // 1. Try to get constant index from the original op's idx (direct constant)
        if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(indexValRaw.getDefiningOp())) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
                constIndex = intAttr.getInt();
                isConstIndex = true;
            }
        }
        
        // 2. Try to trace back to RoutingCreate block argument
        if (!isConstIndex) {
             if (auto barg = dyn_cast<BlockArgument>(indexValRaw)) {
                Operation *parentOp = barg.getOwner()->getParentOp();
                if (auto create = dyn_cast<routing::RoutingCreate>(parentOp)) {
                    unsigned idx = barg.getArgNumber();
                    Value incoming = create->getOperand(idx);
                    IntegerAttr intAttr;
                    if (matchPattern(incoming, m_Constant(&intAttr))) {
                        constIndex = intAttr.getInt();
                        isConstIndex = true;
                    }
                }
            }
        }
        
        // 4. Build ExtractSlice operands
        SmallVector<OpFoldResult> offsets(2);
        SmallVector<OpFoldResult> sizes(2);
        SmallVector<OpFoldResult> strides(2, rewriter.getIndexAttr(1));
        
        // Always use resultShape for sizes to ensure type match
        sizes[0] = rewriter.getIndexAttr(resultShape[0]);
        sizes[1] = rewriter.getIndexAttr(resultShape[1]);
        
        if (isConstIndex) {
             // Static offset calculation
             int64_t offset = constIndex * sliceSize;
             Attribute offsetAttr = rewriter.getIndexAttr(offset);
             
             if (splitDim == 0) {
                 offsets[0] = offsetAttr;
                 offsets[1] = rewriter.getIndexAttr(0);
             } else {
                 offsets[0] = rewriter.getIndexAttr(0);
                 offsets[1] = offsetAttr;
             }
        } else {
             // Dynamic calculation using arith ops
        Value indexVal = adaptor.getIdx();
        if (!indexVal.getType().isIndex()) {
            indexVal = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), indexVal);
        }

             Value sliceSizeVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), sliceSize);
             Value offsetVal = rewriter.create<arith::MulIOp>(op.getLoc(), indexVal, sliceSizeVal);
             
             if (splitDim == 0) {
                 offsets[0] = offsetVal;
                 offsets[1] = rewriter.getIndexAttr(0);
            } else {
                 offsets[0] = rewriter.getIndexAttr(0);
                 offsets[1] = offsetVal;
             }
        }
        
        // Create tensor.extract_slice with "partitionsliceN" tag
        auto extractSlice = rewriter.create<tensor::ExtractSliceOp>(
            op.getLoc(),
            resultType,
            inputTensor,
            offsets,
            sizes,
            strides
        );
        
        // Add "partitionsliceN" tag attribute (with index if known)
        std::string tagName = "partitionslice";
        if (constIndex >= 0) {
            tagName += std::to_string(constIndex);
        }
        extractSlice->setAttr("tag", rewriter.getStringAttr(tagName));
        
        rewriter.replaceOp(op, extractSlice.getResult());
        return success();
    }
};

struct PushOpConversion : public OpConversionPattern<dmaphop::push> {
    PushOpConversion(MLIRContext *context) 
        : OpConversionPattern<dmaphop::push>(context) {}

    LogicalResult matchAndRewrite(dmaphop::push op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // For Push (scatter/one-to-many):
        // - Producer from create_path is the source (one tile)
        // - Consumers from create_path are the destinations (many tiles)
        
        // Use adaptor to get converted consumer buffers (now DeclareDataOp results)
        auto consumerBuffers = adaptor.getConsumerBuffers();
        if (consumerBuffers.empty()) return failure();

        Value viewSplit = adaptor.getData();
        
        // Get unique sequential ID for naming (moved up to use in slice names)
        int opId = g_pullPushCounter.fetch_add(1);
        
        // Create dfscheblueprint.data_slice ops to wrap the converted consumer buffers
        SmallVector<Attribute> sliceSymbols;
        for (size_t i = 0; i < consumerBuffers.size(); ++i) {
            Value buffer = consumerBuffers[i];
            auto tensorType = dyn_cast<RankedTensorType>(buffer.getType());
            if (!tensorType) {
                // Try MemRef for backward compatibility
                if (auto memrefType = dyn_cast<MemRefType>(buffer.getType())) {
                    SmallVector<int64_t> shapeVec(memrefType.getShape().begin(), memrefType.getShape().end());
                    tensorType = RankedTensorType::get(shapeVec, memrefType.getElementType());
                }
            }
            if (!tensorType) continue;

            std::string sliceName = "consumer_slice_" + std::to_string(opId) + "_" + std::to_string(i);
            rewriter.create<dfscheblueprint::DataSliceOp>(
                op.getLoc(),
                tensorType, // Result type
                rewriter.getStringAttr(sliceName),
                buffer // Use adapted consumer buffer (DeclareDataOp result)
            );
            
            // Collect slice symbol reference for flow_group
            sliceSymbols.push_back(FlatSymbolRefAttr::get(getContext(), sliceName));
        }

        // 2. Get path and extract source/destination from producers/consumers attributes
        // For Push: producer is src (one), consumers are dst (many)
        auto pathValue = op.getPath();
        auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pathValue.getDefiningOp());
        if (!pathOp) {
            llvm::errs() << "ERROR: PushOpConversion - failed to get create_path from push op\n";
            return failure();
        }
        
        SmallVector<Attribute> sourceTiles;
        SmallVector<Attribute> destTiles;
        int64_t sourceChannel = -1;
        int64_t destChannel = -1;
        std::string srcTileType;
        std::string dstTileType;
        
        // Get producers (source - one tile for push/scatter) from create_path attribute via symbol table lookup
        auto producersAttr = pathOp.getProducers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
            for (auto innerAttr : arrayAttr) {
                auto symbolRef = dyn_cast<FlatSymbolRefAttr>(innerAttr);
                if (!symbolRef) {
                    llvm::errs() << "ERROR: PushOpConversion - producer is not a FlatSymbolRefAttr\n";
                    continue;
                }
                auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                if (!port) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to find producer port symbol '" 
                                << symbolRef.getValue() << "' in symbol table\n";
                    continue;
                }
                sourceChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                auto tileValue = port.getTile();
                auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                if (!tileOp) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to get tile from producer port '" 
                                << symbolRef.getValue() << "'\n";
                    continue;
                }
                srcTileType = tileOp.getTiletype().str();
                sourceTiles.push_back(rewriter.getArrayAttr({
                    rewriter.getI64IntegerAttr(tileOp.getCol()),
                    rewriter.getI64IntegerAttr(tileOp.getRow())
                }));
            }
        }
        
        if (sourceTiles.empty()) {
            llvm::errs() << "WARNING: PushOpConversion - no source tiles found from producers attribute\n";
        }
        
        // Get consumers (destinations - many tiles for push/scatter) from create_path attribute via symbol table lookup
        auto consumersAttr = pathOp.getConsumers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
            for (auto innerAttr : arrayAttr) {
                auto symbolRef = dyn_cast<FlatSymbolRefAttr>(innerAttr);
                if (!symbolRef) {
                    llvm::errs() << "ERROR: PushOpConversion - consumer is not a FlatSymbolRefAttr\n";
                    continue;
                }
                auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                if (!port) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to find consumer port symbol '" 
                                << symbolRef.getValue() << "' in symbol table\n";
                    continue;
                }
                destChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                auto tileValue = port.getTile();
                auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                if (!tileOp) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to get tile from consumer port '" 
                                << symbolRef.getValue() << "'\n";
                    continue;
                }
                dstTileType = tileOp.getTiletype().str();
                destTiles.push_back(rewriter.getArrayAttr({
                    rewriter.getI64IntegerAttr(tileOp.getCol()),
                    rewriter.getI64IntegerAttr(tileOp.getRow())
                }));
            }
        }
        
        if (destTiles.empty()) {
            llvm::errs() << "WARNING: PushOpConversion - no destination tiles found from consumers attribute\n";
        }
        
        std::string srcGroupName = "group_src_" + std::to_string(opId);
        std::string dstGroupName = "group_dst_" + std::to_string(opId);
        
        // 3. Create resource_group ops at the beginning of RoutingCreate block
        if (auto routingCreateOp = op->getParentOfType<routing::RoutingCreate>()) {
            Block &routingBlock = routingCreateOp.getRegion().front();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&routingBlock);
            
            // Source resource group (producer - one for push)
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(srcGroupName),
                rewriter.getArrayAttr(sourceTiles)
            );
            
            // Destination resource group (consumers - many for push)
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(dstGroupName),
                rewriter.getArrayAttr(destTiles)
            );
        } else {
            // Fallback: insert at current position if not inside RoutingCreate
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(srcGroupName),
                rewriter.getArrayAttr(sourceTiles)
            );
            
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(dstGroupName),
                rewriter.getArrayAttr(destTiles)
            );
        }
        
        // 4. Create Binds
        // For Push: src is one tile (FlowConfigOp with root), dst is many tiles (FlowConfigGroupOp with linear and slice_symbols)
        std::string srcBindName = "flow_src_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(),
            rewriter.getStringAttr(srcBindName),
            FlatSymbolRefAttr::get(getContext(), srcGroupName),
            viewSplit,
            rewriter.getStringAttr("root"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({sourceChannel}), dfscheblueprint::bp_direction::MM2S),
            nullptr, // slice_symbols - source is root, no slice
            rewriter.getStringAttr(srcTileType)
        );
        
        std::string dstBindName = "flow_dst_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(),
            rewriter.getStringAttr(dstBindName),
            FlatSymbolRefAttr::get(getContext(), dstGroupName),
            viewSplit,
            rewriter.getStringAttr("linear"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({destChannel}), dfscheblueprint::bp_direction::S2MM),
            rewriter.getArrayAttr(sliceSymbols),  // Associate with @consumer_slice_0 to @consumer_slice_N
            rewriter.getStringAttr(dstTileType)
        );
        
        // 5. Create Collective Transfer - one_to_many for push/scatter
        rewriter.create<dfscheblueprint::FlowTransferOp>(
            op.getLoc(),
            rewriter.getStringAttr("transfer_" + std::to_string(opId)),
            rewriter.getStringAttr("one_to_many"),
            FlatSymbolRefAttr::get(getContext(), srcBindName),
            FlatSymbolRefAttr::get(getContext(), dstBindName),
            rewriter.getStringAttr("sequential"),
            0
        );
        
        rewriter.eraseOp(op);
        return success();
    }
};

struct PullOpConversion : public OpConversionPattern<dmaphop::pull> {
    PullOpConversion(MLIRContext *context) 
        : OpConversionPattern<dmaphop::pull>(context) {}

    LogicalResult matchAndRewrite(dmaphop::pull op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Use adaptor to get converted producer buffers (now DeclareDataOp results)
        auto producerBuffers = adaptor.getProducerBuffers();
        if (producerBuffers.empty()) return failure();

        Value viewSplit = adaptor.getData();
        
        // Get unique sequential ID for naming (moved up to use in slice names)
        int opId = g_pullPushCounter.fetch_add(1);
        
        // Create dfscheblueprint.data_slice ops to wrap the converted producer buffers
        SmallVector<Attribute> sliceSymbols;
        for (size_t i = 0; i < producerBuffers.size(); ++i) {
            Value buffer = producerBuffers[i];
            auto tensorType = dyn_cast<RankedTensorType>(buffer.getType());
            if (!tensorType) {
                // Try MemRef for backward compatibility
                if (auto memrefType = dyn_cast<MemRefType>(buffer.getType())) {
                    SmallVector<int64_t> shapeVec(memrefType.getShape().begin(), memrefType.getShape().end());
                    tensorType = RankedTensorType::get(shapeVec, memrefType.getElementType());
                }
            }
            if (!tensorType) continue;

            std::string sliceName = "producer_slice_" + std::to_string(opId) + "_" + std::to_string(i);
            rewriter.create<dfscheblueprint::DataSliceOp>(
                op.getLoc(),
                tensorType, // Result type
                rewriter.getStringAttr(sliceName),
                buffer // Use adapted producer buffer (DeclareDataOp result)
            );

            // Collect slice symbol reference for flow_group
            sliceSymbols.push_back(FlatSymbolRefAttr::get(getContext(), sliceName));
        }

        // 2. Get path and extract source/destination from producers/consumers attributes
        // Path structure: create_path[hop1, hop2, ...] with producers/consumers attributes
        // producers/consumers are symbol references to port ops
        auto pathValue = op.getPath();
        auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pathValue.getDefiningOp());
        if (!pathOp) {
            llvm::errs() << "ERROR: PullOpConversion - failed to get create_path from pull op\n";
            return failure();
        }
        
        SmallVector<Attribute> sourceTiles;
        SmallVector<Attribute> destTiles;
        int64_t sourceChannel = -1;
        int64_t destChannel = -1;
        std::string srcTileType;
        std::string dstTileType;
        
        // Get producers (sources) from create_path attribute via symbol table lookup
        auto producersAttr = pathOp.getProducers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
            for (auto innerAttr : arrayAttr) {
                //if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                    //for (auto symbolAttr : innerArray) {
                        auto symbolRef = dyn_cast<FlatSymbolRefAttr>(innerAttr);
                        if (!symbolRef) {
                            llvm::errs() << "ERROR: PullOpConversion - producer is not a FlatSymbolRefAttr\n";
                            continue;
                        }
                        auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                        if (!port) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to find producer port symbol '" 
                                        << symbolRef.getValue() << "' in symbol table\n";
                            continue;
                        }
                        sourceChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                        auto tileValue = port.getTile();
                        auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                        if (!tileOp) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to get tile from producer port '" 
                                        << symbolRef.getValue() << "'\n";
                            continue;
                        }
                        srcTileType = tileOp.getTiletype().str();
                        sourceTiles.push_back(rewriter.getArrayAttr({
                            rewriter.getI64IntegerAttr(tileOp.getCol()),
                            rewriter.getI64IntegerAttr(tileOp.getRow())
                        }));
                    //}
                //}
            }
        }
        
        if (sourceTiles.empty()) {
            llvm::errs() << "WARNING: PullOpConversion - no source tiles found from producers attribute\n";
        }
        
        // Get consumers (destinations) from create_path attribute via symbol table lookup
        auto consumersAttr = pathOp.getConsumers();
        if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
            for (auto innerAttr : arrayAttr) {
                //if (auto innerArray = dyn_cast<ArrayAttr>(innerAttr)) {
                    //for (auto symbolAttr : innerArray) {
                        auto symbolRef = dyn_cast<FlatSymbolRefAttr>(innerAttr);
                        if (!symbolRef) {
                            llvm::errs() << "ERROR: PullOpConversion - consumer is not a FlatSymbolRefAttr\n";
                            continue;
                        }
                        auto port = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(op, symbolRef);
                        if (!port) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to find consumer port symbol '" 
                                        << symbolRef.getValue() << "' in symbol table\n";
                            continue;
                        }
                        destChannel = static_cast<int64_t>(port.getDirectionChannel().value());
                        auto tileValue = port.getTile();
                        auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
                        if (!tileOp) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to get tile from consumer port '" 
                                        << symbolRef.getValue() << "'\n";
                            continue;
                        }
                        dstTileType = tileOp.getTiletype().str();
                        destTiles.push_back(rewriter.getArrayAttr({
                            rewriter.getI64IntegerAttr(tileOp.getCol()),
                            rewriter.getI64IntegerAttr(tileOp.getRow())
                        }));
                    //}
                //}
            }
        }
        
        if (destTiles.empty()) {
            llvm::errs() << "WARNING: PullOpConversion - no destination tiles found from consumers attribute\n";
        }
        
        std::string srcGroupName = "group_src_" + std::to_string(opId);
        std::string dstGroupName = "group_dst_" + std::to_string(opId);
        
        // 3. Create resource_group ops at the beginning of RoutingCreate block
        if (auto routingCreateOp = op->getParentOfType<routing::RoutingCreate>()) {
            Block &routingBlock = routingCreateOp.getRegion().front();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&routingBlock);
            
            // Source resource group (producers)
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(srcGroupName),
                rewriter.getArrayAttr(sourceTiles)
            );
            
            // Destination resource group (consumers)
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(dstGroupName),
                rewriter.getArrayAttr(destTiles)
            );
        } else {
            // Fallback: insert at current position if not inside RoutingCreate
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(srcGroupName),
                rewriter.getArrayAttr(sourceTiles)
            );
            
            rewriter.create<dfscheblueprint::TileGroupOp>(
                op.getLoc(),
                rewriter.getStringAttr(dstGroupName),
                rewriter.getArrayAttr(destTiles)
            );
        }
        
        // 4. Create Binds
        std::string srcBindName = "flow_src_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(),
            rewriter.getStringAttr(srcBindName),
            FlatSymbolRefAttr::get(getContext(), srcGroupName),
            viewSplit,
            rewriter.getStringAttr("linear"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({sourceChannel}), dfscheblueprint::bp_direction::MM2S),
            rewriter.getArrayAttr(sliceSymbols),  // slice_symbols
            rewriter.getStringAttr(srcTileType)
        );
        
        std::string dstBindName = "flow_dst_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(),
            rewriter.getStringAttr(dstBindName),
            FlatSymbolRefAttr::get(getContext(), dstGroupName),
            viewSplit,
            rewriter.getStringAttr("root"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({destChannel}), dfscheblueprint::bp_direction::S2MM),
            nullptr, // slice_symbols
            rewriter.getStringAttr(dstTileType)
        );
        
        // 5. Create Collective Transfer
        rewriter.create<dfscheblueprint::FlowTransferOp>(
            op.getLoc(),
            rewriter.getStringAttr("transfer_" + std::to_string(opId)),
            rewriter.getStringAttr("many_to_one"),
            FlatSymbolRefAttr::get(getContext(), srcBindName),
            FlatSymbolRefAttr::get(getContext(), dstBindName),
            rewriter.getStringAttr("sequential"),
            0
        );
        
        rewriter.eraseOp(op);
        return success();
    }
};

void DmaphopTodfscheblueprintPass::runOnOperation() {
    auto module = getOperation();
    OpBuilder builder(module->getContext());

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<dfscheblueprint::dfscheblueprintdialect>();
    //target.addLegalDialect<routing::routingdialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<scf::ExecuteRegionOp>();
    
    //target.addIllegalDialect<dmaphop::dmaphopdialect>();
    //target.addIllegalOp<dmaphop::push>();
    //target.addIllegalOp<dmaphop::pull>();
    //target.addIllegalOp<routing::partitiontensor>();
    
    RewritePatternSet patterns(context);
    patterns.add<EraseOpLowering<dmaphop::tile>,
                 EraseOpLowering<dmaphop::port>,
                 EraseOpLowering<dmaphop::create_hop>,
                 EraseOpLowering<dmaphop::create_path>,
                 EraseOpLowering<dmaphop::alloc_buffer>,
                 EraseOpLowering<dmaphop::sync>,
                 EraseOpLowering<dmaphop::dealloc_buffer>>(context);
    
    patterns.add<CreateScheduleTensorConversion>(context);
    //patterns.add<PartitionTensorConversion>(context);
    patterns.add<ExtractDataConversion>(context);
    patterns.add<PushOpConversion>(context);
    patterns.add<PullOpConversion>(context);

    target.addIllegalOp<routing::createscheduletensor>();
    //target.addIllegalOp<routing::partitiontensor>();
    target.addIllegalOp<routing::extract_data>();
    target.addIllegalOp<dmaphop::push>();
    target.addIllegalOp<dmaphop::pull>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
    
    //SmallVector<scf::ExecuteRegionOp> execOps;
    //module->walk([&](scf::ExecuteRegionOp op) { execOps.push_back(op); });
    /*
    for (auto execOp : execOps) {
        bool hasDmapHop = false;
        execOp.walk([&](Operation *op) {
            if (op->getDialect()->getNamespace() == "dmaphop") {
                hasDmapHop = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        
        if (!hasDmapHop) continue;

        builder.setInsertionPoint(execOp);
        auto configOp = builder.create<dfscheblueprint::ConfigOp>(
            execOp.getLoc(), 
            builder.getStringAttr("broadcast_gather_blueprint")
        );
        
        Block *configBody = builder.createBlock(&configOp.getRegion());
        // No need to set insertion point here as patterns will set it

        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<dfscheblueprint::dfscheblueprintdialect>();
        target.addLegalDialect<routing::routingdialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<memref::MemRefDialect>();
        target.addLegalOp<scf::ExecuteRegionOp>();
        
        target.addIllegalDialect<dmaphop::dmaphopdialect>();
        target.addIllegalOp<dmaphop::push>();
        target.addIllegalOp<dmaphop::pull>();
        //target.addIllegalOp<routing::partitiontensor>();
        
        RewritePatternSet patterns(context);
        patterns.add<EraseOpLowering<dmaphop::tile>,
                     EraseOpLowering<dmaphop::port>,
                     EraseOpLowering<dmaphop::create_hop>,
                     EraseOpLowering<dmaphop::create_path>,
                     EraseOpLowering<dmaphop::alloc_buffer>,
                     EraseOpLowering<dmaphop::sync>,
                     EraseOpLowering<dmaphop::dealloc_buffer>>(context);
        
        patterns.add<PartitionTensorConversion>(context, configOp);
        patterns.add<PushOpConversion>(context, configOp);
        patterns.add<PullOpConversion>(context, configOp);

        if (failed(applyPartialConversion(execOp, target, std::move(patterns)))) {
            signalPassFailure();
        }

        execOp.erase();
    }
    */
}

std::unique_ptr<Pass> createDmaphopTodfscheblueprintPass() {
    return std::make_unique<DmaphopTodfscheblueprintPass>();
}

} // namespace mlir
