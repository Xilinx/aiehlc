/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdmaphoptodfscheblueprint.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <atomic>
#include <iostream>

using namespace mlir;
using namespace dmaphop;
using namespace dfscheblueprint;
using namespace routing;

namespace mlir {

// Global counter for unique sequential naming (avoids conflict)
static std::atomic<int> g_pullPushCounter{0};

// Map from root-tensor SSA Value (partitiontensor result) to data_id.
// All push sub-flows sharing the same root tensor get the same data_id,
// enabling ScheduleCanonicalizePass to merge their shim BDs into one.
static llvm::DenseMap<mlir::Value, int32_t> g_rootViewToDataId;
static int32_t g_dataIdCounter = 0;

// Helper: resolve a symbol from create_path's producers/consumers array.
// If the symbol refers to a dmaphop.consumer, follows from -> port.
// If the symbol refers to a dmaphop.producer, follows tp -> port.
// Otherwise assumes it's a direct port symbol.
static dmaphop::port resolvePortFromPathSymbol(Operation *contextOp, FlatSymbolRefAttr symRef) {
    // Try direct port lookup first (shim/mem ports)
    if (auto portOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, symRef))
        return portOp;
    // Try consumer indirection
    if (auto consumerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::consumer>(contextOp, symRef)) {
        auto fromRef = consumerOp.getFromAttr();
        return SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, fromRef);
    }
    // Try producer indirection
    if (auto producerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::producer>(contextOp, symRef)) {
        auto tpRef = producerOp.getTpAttr();
        return SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, tpRef);
    }
    return nullptr;
}

// Helper: resolve the canonical tensor index (= func-arg number) for a root
// view value. Per-tensor module attributes (tensor_N.halo, tensor_N.shim_dma,
// tensor_N.layout_transform) are keyed by func-arg order (arg0=tensor_0, ...),
// NOT by the encounter-order data_id. Walk the def-use chain from the root view
// back to a func.func BlockArgument and return its arg number. Returns -1 if the
// root view does not trace to a block argument (e.g. an intermediate buffer).
static int32_t resolveFuncArgIndexForRootView(Value rootView) {
    Value cur = rootView;
    // Bounded walk to avoid cycles.
    for (int guard = 0; guard < 64 && cur; ++guard) {
        if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
            // Only func.func entry-block arguments are canonical tensor indices.
            if (isa_and_nonnull<func::FuncOp>(blockArg.getOwner()->getParentOp()))
                return static_cast<int32_t>(blockArg.getArgNumber());
            return -1;
        }
        Operation *def = cur.getDefiningOp();
        if (!def)
            return -1;
        // Step through value-forwarding ops that preserve tensor identity.
        if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(def)) {
            cur = toTensor.getMemref();
            continue;
        }
        if (auto sched = dyn_cast<routing::createscheduletensor>(def)) {
            cur = sched.getInitTensor();
            continue;
        }
        if (auto part = dyn_cast<routing::partitiontensor>(def)) {
            cur = part.getTensor();
            continue;
        }
        if (auto extractData = dyn_cast<routing::extract_data>(def)) {
            cur = extractData.getTensor();
            continue;
        }
        if (auto extract = dyn_cast<tensor::ExtractSliceOp>(def)) {
            cur = extract.getSource();
            continue;
        }
        // Generic single-operand forwarding fallback (e.g. casts/materializations).
        if (def->getNumOperands() == 1) {
            cur = def->getOperand(0);
            continue;
        }
        return -1;
    }
    return -1;
}

// Helper: look up dma_port from a dmaphop.consumer or dmaphop.producer op
// matching the given symbol name (own sym_name or from/tp port ref).
// Returns the dma_port, or fallback if not found.
static int64_t getDmaPortForSym(Operation *contextOp, StringRef portSymName, int64_t fallback) {
    Operation *searchRoot = contextOp->getParentOfType<ModuleOp>();
    if (!searchRoot)
        searchRoot = contextOp->getParentOp();
    if (!searchRoot)
        return fallback;

    int64_t result = fallback;
    searchRoot->walk([&](Operation *op) {
        if (auto consumerOp = dyn_cast<dmaphop::consumer>(op)) {
            // Match by own symbol name (from create_path) or by from port ref
            if (consumerOp.getSymName() == portSymName || consumerOp.getFrom() == portSymName) {
                result = consumerOp.getDmaPort();
                return WalkResult::interrupt();
            }
        } else if (auto producerOp = dyn_cast<dmaphop::producer>(op)) {
            // Match by own symbol name (from create_path) or by tp port ref
            if (producerOp.getSymName() == portSymName || producerOp.getTp() == portSymName) {
                result = producerOp.getDmaPort();
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return result;
}

// Helper: look up dmapktid from the first producer port of a pull path.
// Returns 1 as fallback (pkt_id=0 is reserved).
static int32_t getBasePktIdFromProducers(dmaphop::create_path pathOp, Operation *contextOp) {
    for (auto attr : pathOp.getProducers()) {
        if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
            // Use resolvePortFromPathSymbol to handle producer symbol indirection
            auto portOp = resolvePortFromPathSymbol(contextOp, symRef);
            if (portOp) {
                if (auto pktIdOpt = portOp.getDmapktid())
                    return static_cast<int32_t>(*pktIdOpt);
            }
        }
    }
    return 1; // fallback: 1-based
}

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
                    auto port = resolvePortFromPathSymbol(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPull - failed to find producer port symbol '"
                                     << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                    sourceChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                     static_cast<int64_t>(port.getDirectionChannel().value()));
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

    // Assign data_id: same ID for all pull sub-flows sharing the same root tensor.
    // viewHandle is the root tensor passed in by the caller.
    int32_t dataId;
    {
        auto it = g_rootViewToDataId.find(viewHandle);
        if (it != g_rootViewToDataId.end()) {
            dataId = it->second;
        } else {
            dataId = g_dataIdCounter++;
            g_rootViewToDataId[viewHandle] = dataId;
        }
    }

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
                    auto port = resolvePortFromPathSymbol(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPull - failed to find consumer port symbol '"
                                     << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                    destChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                   static_cast<int64_t>(port.getDirectionChannel().value()));
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
        op.getLoc(), srcBindName, FlatSymbolRefAttr::get(builder.getContext(), srcGroupName), viewHandle,
        builder.getStringAttr("linear"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({sourceChannel}),
                                      dfscheblueprint::bp_direction::MM2S),
        builder.getArrayAttr({}), builder.getStringAttr(srcTileType),
        nullptr, // data_id
        nullptr, // shim_dim_strides
        nullptr, // shim_dim_wraps
        nullptr  // pp_depth
    );

    std::string dstBindName = "flow_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(), dstBindName, FlatSymbolRefAttr::get(builder.getContext(), dstGroupName), viewHandle,
        builder.getStringAttr("root"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({destChannel}),
                                      dfscheblueprint::bp_direction::S2MM),
        nullptr, // slice_symbols
        builder.getStringAttr(dstTileType),
        builder.getI32IntegerAttr(dataId), // data_id for shim BD grouping
        nullptr,                           // shim_dim_strides
        nullptr,                           // shim_dim_wraps
        nullptr                            // pp_depth
    );

    // 4. Collective Transfer
    int32_t basePktId = getBasePktIdFromProducers(pathOp, op);
    builder.create<dfscheblueprint::FlowTransferOp>(
        op.getLoc(), builder.getStringAttr("transfer_" + std::to_string(opId)), builder.getStringAttr("many_to_one"),
        FlatSymbolRefAttr::get(builder.getContext(), srcBindName),
        FlatSymbolRefAttr::get(builder.getContext(), dstBindName), builder.getStringAttr("sequential"),
        builder.getI32IntegerAttr(basePktId),
        builder.getI32IntegerAttr(opId) // flow_index
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
                    auto port = resolvePortFromPathSymbol(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPush - failed to find producer port symbol '"
                                     << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                    srcChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                  static_cast<int64_t>(port.getDirectionChannel().value()));
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
                    auto port = resolvePortFromPathSymbol(op, symbolRef);
                    if (!port) {
                        llvm::errs() << "ERROR: processPush - failed to find consumer port symbol '"
                                     << symbolRef.getValue() << "' in symbol table\n";
                        continue;
                    }
                    dstChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                  static_cast<int64_t>(port.getDirectionChannel().value()));
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
        op.getLoc(), srcBindName, FlatSymbolRefAttr::get(builder.getContext(), srcGroupName), viewHandle,
        builder.getStringAttr("root"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({srcChannel}),
                                      dfscheblueprint::bp_direction::MM2S),
        nullptr, // slice_symbols
        builder.getStringAttr(srcTileType),
        nullptr, // data_id
        nullptr, // shim_dim_strides
        nullptr, // shim_dim_wraps
        nullptr  // pp_depth
    );

    std::string dstBindName = "flow_dst_" + std::to_string(opId);
    builder.create<dfscheblueprint::FlowConfigOp>(
        op.getLoc(), dstBindName, FlatSymbolRefAttr::get(builder.getContext(), dstGroupName), viewHandle,
        builder.getStringAttr("linear"),
        dfscheblueprint::DMAAttr::get(builder.getContext(), ArrayRef<int64_t>({dstChannel}),
                                      dfscheblueprint::bp_direction::S2MM),
        builder.getArrayAttr({}), builder.getStringAttr(dstTileType),
        nullptr, // data_id
        nullptr, // shim_dim_strides
        nullptr, // shim_dim_wraps
        nullptr  // pp_depth
    );

    int32_t basePktId = 0; // Push flows are circuit-switched; no pkt_id
    builder.create<dfscheblueprint::FlowTransferOp>(
        op.getLoc(), builder.getStringAttr("transfer_" + std::to_string(opId)), builder.getStringAttr("one_to_many"),
        FlatSymbolRefAttr::get(builder.getContext(), srcBindName),
        FlatSymbolRefAttr::get(builder.getContext(), dstBindName), builder.getStringAttr("sequential"),
        builder.getI32IntegerAttr(basePktId),
        builder.getI32IntegerAttr(0) // flow_index
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

        // Support N-D tensors (>=2). The 3D conv OUTPUT <H x W x C> flows through
        // here as a genuine rank-3 tensor; matmul/2D paths stay rank-2. The split
        // happens along exactly ONE dim (the mesh-row split); all other dims take
        // their full extent.
        if (inputShape.size() != resultShape.size() || inputShape.size() < 2) {
            return op.emitError("ExtractDataConversion: input and result must have equal rank >= 2");
        }
        const int64_t rank = (int64_t)inputShape.size();

        int64_t splitDim = 0;
        int64_t sliceSize = 0;
        bool partitionFound = false;
        // Halo (spatial overlap) step: when set (>0), consecutive tile-rows advance
        // by `haloStep` rows instead of by `sliceSize`, producing overlapping slices.
        int64_t haloStep = 0;
        // Full tiling descriptor (incl. nested levels); kept to annotate the slice.
        routing::TilingAttr tilingAttr;

        // 1. Try to get partition info
        if (auto partitionOp = dyn_cast_or_null<routing::partitiontensor>(op.getTensor().getDefiningOp())) {
            splitDim = partitionOp.getPartition().getSplitdim();
            int64_t splitNum = partitionOp.getPartition().getSplitnum();
            if (splitNum > 0 && splitDim >= 0 && splitDim < rank) {
                sliceSize = inputShape[splitDim] / splitNum;
            }
            partitionFound = true;
            // Read halo step from the optional #routing.tiling attr propagated from
            // createroutingfuncBySplitModel (the split dim's outer level step).
            if (auto tiling = partitionOp.getTilingAttr()) {
                haloStep = tiling.getDims()[splitDim].getOuter().getStep();
                tilingAttr = tiling;
            }
        }

        // 2. Infer/Override from result shape (Crucial for validity).
        // The slice differs from the full tensor on exactly one dim (the mesh-row
        // split); find it. All other dims match input (full extent).
        bool diffFound = false;
        for (int64_t d = 0; d < rank; ++d) {
            if (resultShape[d] != inputShape[d]) {
                splitDim = d;
                sliceSize = resultShape[d];
                diffFound = true;
                break;
            }
        }
        if (!diffFound && !partitionFound) {
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

        // 4. Build ExtractSlice operands (N-D). Full extent on every dim except
        // splitDim; on splitDim the offset advances per tile index.
        SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes(rank);
        SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

        // Always use resultShape for sizes to ensure type match
        for (int64_t d = 0; d < rank; ++d)
            sizes[d] = rewriter.getIndexAttr(resultShape[d]);

        // Halo mode: offset advances by haloStep (overlap), size stays sliceSize.
        // Non-halo: offset advances by sliceSize (contiguous, non-overlapping).
        int64_t offsetStride = (haloStep > 0) ? haloStep : sliceSize;

        if (isConstIndex) {
             // Static offset calculation
             int64_t offset = constIndex * offsetStride;
             offsets[splitDim] = rewriter.getIndexAttr(offset);
        } else {
             // Dynamic calculation using arith ops
             Value indexVal = adaptor.getIdx();
             if (!indexVal.getType().isIndex()) {
                 indexVal = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), indexVal);
             }

             Value sliceSizeVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), offsetStride);
             Value offsetVal = rewriter.create<arith::MulIOp>(op.getLoc(), indexVal, sliceSizeVal);
             offsets[splitDim] = offsetVal;
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

        // Carry the nested-tiling descriptor onto the slice for traceability.
        // Source of truth for codegen remains the module-level tensor_N.halo dict;
        // this annotation is additive. Gated on a nested (L2) slice_tiling level
        // so non-L2 IR is unchanged.
        if (tilingAttr) {
            auto inner = tilingAttr.getDims()[splitDim].getOuter().getSliceTiling();
            if (inner && inner.getRounds() > 1)
                extractSlice->setAttr("tiling", tilingAttr);
        }

        rewriter.replaceOp(op, extractSlice.getResult());
        return success();
    }
};

struct PushOpConversion : public OpConversionPattern<dmaphop::push> {
    // Tiling scalars read once at pass entry from the #routing.tiling op (see
    // runOnOperation). The K-tiling shim-addressing block below consumes these
    // instead of re-reading routing.tile_*/effective_k/full_k module attrs.
    routing::GemmTilingScalars tiling;

    PushOpConversion(MLIRContext *context, routing::GemmTilingScalars tiling)
        : OpConversionPattern<dmaphop::push>(context), tiling(tiling) {}

    LogicalResult matchAndRewrite(dmaphop::push op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // For Push (scatter/one-to-many):
        // - Producer from create_path is the source (one tile)
        // - Consumers from create_path are the destinations (many tiles)
        
        // Use adaptor to get converted consumer buffers (now DeclareDataOp results)
        auto consumerBuffers = adaptor.getConsumerBuffers();
        if (consumerBuffers.empty()) return failure();

        Value viewSplit = adaptor.getData();

        // Find the root (pre-split) view: trace extract_slice -> partitiontensor.
        // The partitiontensor result is shared by all sub-flows for the same tensor,
        // so it serves as the stable identity key for data_id assignment.
        Value rootAdaptedView = viewSplit;
        if (auto extractSlice = viewSplit.getDefiningOp<tensor::ExtractSliceOp>()) {
            rootAdaptedView = extractSlice.getSource();
            // rootAdaptedView is now the routing.partitiontensor result (full tensor)
        }

        // Assign data_id: same ID for all sub-flows sharing the same root tensor
        int32_t dataId;
        {
            auto it = g_rootViewToDataId.find(rootAdaptedView);
            if (it != g_rootViewToDataId.end()) {
                dataId = it->second;
            } else {
                dataId = g_dataIdCounter++;
                g_rootViewToDataId[rootAdaptedView] = dataId;
            }
        }

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
                auto port = resolvePortFromPathSymbol(op, symbolRef);
                if (!port) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to find producer port symbol '"
                                 << symbolRef.getValue() << "' in symbol table\n";
                    continue;
                }
                sourceChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                 static_cast<int64_t>(port.getDirectionChannel().value()));
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
                auto port = resolvePortFromPathSymbol(op, symbolRef);
                if (!port) {
                    llvm::errs() << "ERROR: PushOpConversion - failed to find consumer port symbol '"
                                 << symbolRef.getValue() << "' in symbol table\n";
                    continue;
                }
                destChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                               static_cast<int64_t>(port.getDirectionChannel().value()));
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
        // For Push: src is one shim tile (FlowConfigOp with partition slice view),
        //           dst is many core tiles (FlowConfigOp with linear and slice_symbols)
        std::string srcBindName = "flow_src_" + std::to_string(opId);

        // Compute multi-dimensional DMA addressing for shim MM2S input when
        // effectiveK < fullK (K-dimension temporal tiling). In this case, the
        // routing IR uses tensor shapes with effectiveK, but the DDR data has
        // full-K row width. The shim DMA must use 2D addressing to extract the
        // correct K-chunk columns from each DDR row.
        ArrayAttr inputShimDimStrides = nullptr;
        ArrayAttr inputShimDimWraps = nullptr;

        if (srcTileType == "shim") {
            auto moduleOp = op->getParentOfType<ModuleOp>();
            // Per-tensor module attributes (tensor_N.halo, tensor_N.shim_dma,
            // tensor_N.layout_transform) are keyed by func-arg order (arg0=tensor_0,
            // arg1=tensor_1, ...). The encounter-order `dataId` assigned above does
            // NOT match that order (e.g. input B is visited first -> dataId=0 while
            // it is func arg1 -> tensor_1). Resolve the canonical func-arg index from
            // the root view and use it for all per-tensor attribute lookups; fall back
            // to dataId if the root view does not trace to a func argument.
            int32_t tensorIdx = resolveFuncArgIndexForRootView(rootAdaptedView);
            if (tensorIdx < 0)
                tensorIdx = dataId;
            // Spatial-halo mode (tensor_N.halo present): the A-tensor is sliced into
            // overlapping contiguous row-blocks. Each shim BD stays FLAT (no multi-dim
            // strides); overlap is realized purely through per-tile DDR base offsets
            // computed in ExtractDataConversion. Skip all multi-dim addressing here.
            bool isHaloTensor = false;
            if (moduleOp) {
                std::string haloKey = "tensor_" + std::to_string(tensorIdx) + ".halo";
                isHaloTensor = moduleOp->getAttrOfType<DictionaryAttr>(haloKey) != nullptr;
            }
            if (moduleOp && !isHaloTensor) {
                // Check layout_transform: if "core_shuffle", skip DmaTransform strides
                // (data goes flat to core tiles; the kernel handles reordering).
                std::string layoutTransformKey = "tensor_" + std::to_string(tensorIdx) + ".layout_transform";
                auto layoutTransformAttr = moduleOp->getAttrOfType<StringAttr>(layoutTransformKey);
                bool isCoreShuffle = layoutTransformAttr && layoutTransformAttr.getValue() == "core_shuffle";
                // Check for user-specified DmaTransform (tensor_N.shim_dma module attribute)
                // If present and not core_shuffle, use it directly instead of computing K-tiling strides.
                std::string shimDmaKey = "tensor_" + std::to_string(tensorIdx) + ".shim_dma";

                if (isCoreShuffle) {
                    llvm::errs() << "[ShimMultiDim] layout_transform=core_shuffle for tensor_" << tensorIdx
                                 << ": skipping DmaTransform strides (flat DMA to core)\n";
                    // Attach core_shuffle hint for the kernel path: if shimDmaAttr has
                    // strides/wraps, store them as tensor_N.core_shuffle so kernel code
                    // generation can use the transform description for reordering.
                    auto coreDmaAttr = moduleOp->getAttrOfType<DictionaryAttr>(shimDmaKey);
                    if (coreDmaAttr) {
                        moduleOp->setAttr("tensor_" + std::to_string(tensorIdx) + ".core_shuffle", coreDmaAttr);
                    }
                }
                auto shimDmaAttr = moduleOp->getAttrOfType<DictionaryAttr>(shimDmaKey);
                if (shimDmaAttr && !isCoreShuffle) {
                    auto userStrides = shimDmaAttr.getAs<ArrayAttr>("strides");
                    auto userWraps = shimDmaAttr.getAs<ArrayAttr>("wraps");
                    if (userStrides && userWraps) {
                        // Convert strides from element units to bytes (32-bit word aligned).
                        // The DmaTransform strides are in element counts; the AIEML DMA
                        // hardware operates in 32-bit word granularity. Each stride must be
                        // a multiple of 4 bytes. For sub-word elements (e.g., i8 = 1 byte),
                        // we scale by wordBytes (4) since the DMA moves whole words.
                        constexpr int32_t wordBytes = 4;
                        auto viewType = dyn_cast<RankedTensorType>(viewSplit.getType());
                        unsigned bitWidth = viewType ? viewType.getElementTypeBitWidth() : 8;
                        int32_t bytesPerElem = bitWidth / 8;
                        if (bytesPerElem < 1)
                            bytesPerElem = 1;
                        SmallVector<int32_t> byteStrides;
                        SmallVector<int32_t> adjWraps;
                        for (auto attr : userStrides) {
                            int32_t elemStride = cast<IntegerAttr>(attr).getInt();
                            int32_t byteStride = elemStride * bytesPerElem;
                            // Round up to 32-bit word alignment (AIEML DMA requirement)
                            if (byteStride > 0 && byteStride < wordBytes)
                                byteStride = wordBytes;
                            else if (byteStride % wordBytes != 0)
                                byteStride = ((byteStride + wordBytes - 1) / wordBytes) * wordBytes;
                            byteStrides.push_back(byteStride);
                        }
                        for (auto attr : userWraps) {
                            adjWraps.push_back(cast<IntegerAttr>(attr).getInt());
                        }
                        inputShimDimStrides = rewriter.getI32ArrayAttr(byteStrides);
                        inputShimDimWraps = rewriter.getI32ArrayAttr(adjWraps);

                        llvm::errs() << "[ShimMultiDim] Using user-specified DmaTransform for " << shimDmaKey
                                     << " (bytesPerElem=" << bytesPerElem << "): strides=[";
                        for (unsigned si = 0; si < byteStrides.size(); ++si) {
                            if (si > 0)
                                llvm::errs() << ",";
                            llvm::errs() << byteStrides[si];
                        }
                        llvm::errs() << "] wraps=[";
                        for (unsigned si = 0; si < userWraps.size(); ++si) {
                            if (si > 0)
                                llvm::errs() << ",";
                            llvm::errs() << cast<IntegerAttr>(userWraps[si]).getInt();
                        }
                        llvm::errs() << "]\n";
                    }
                }

                // K-tiling multi-dim addressing (only if no user DmaTransform was set).
                // Tiling scalars come from the entry-read #routing.tiling struct;
                // effectiveK==0 (conv / no GEMM tiling) skips this block exactly like
                // the old effective_k/full_k attr-presence check.
                if (!inputShimDimStrides) {
                    {
                        int64_t effectiveK = tiling.effectiveK;
                        int64_t fullK = tiling.fullK;
                        int64_t tileM = tiling.tileM;

                        if (effectiveK > 0 && effectiveK < fullK) {
                            // Get element bit width from the view tensor type
                            auto viewType = dyn_cast<RankedTensorType>(viewSplit.getType());
                            if (viewType && viewType.getRank() == 2) {
                                unsigned bitWidth = viewType.getElementTypeBitWidth();
                                int64_t elemsPerWord = 32 / bitWidth; // 4 for i8, 2 for i16
                                constexpr int64_t wordBytes = 4;

                                // Partition slice shape: {partRows, effectiveK}
                                int64_t partRows = viewType.getDimSize(0);
                                int64_t effK_w = effectiveK / elemsPerWord; // effectiveK in words
                                int64_t fullK_w = fullK / elemsPerWord;     // fullK in words

                                if (tileM > 0 && tileM < partRows) {
                                    // 3D addressing: D0=K-chunk, D1=sub-tile rows, D2=kRounds
                                    // Determine the sub-tile dimension based on flow type:
                                    //   Input B (dataId==0): D1 = tile_n (col sub-tiling)
                                    //   Input A (dataId==1): D1 = tile_m (row sub-tiling)
                                    int64_t subTileDim = tileM; // default: tile_m for input A
                                    if (dataId == 0 && tiling.tileN > 0) {
                                        subTileDim = tiling.tileN; // input B (dataId=0): use tile_n
                                    }

                                    int64_t kRounds = fullK / effectiveK;
                                    SmallVector<int32_t> strides = {
                                        static_cast<int32_t>(1 * wordBytes),       // D0: 4 bytes
                                        static_cast<int32_t>(fullK_w * wordBytes), // D1: fullK bytes
                                        static_cast<int32_t>(effK_w * wordBytes)   // D2: effectiveK bytes
                                        // NOTE: D2 is the k-round stride (advance one effectiveK chunk
                                        // along K per round). It must be byte-scaled like D0/D1 via
                                        // (words * wordBytes). Using raw `effectiveK` was only correct
                                        // for i8 (elemsPerWord==4 => effK_w*wordBytes == effectiveK);
                                        // for i16/i32 it under-advanced the DDR address and starved the
                                        // cores (DMA wait_io TIMEOUT). Mirrors output D2 (tileW_w*wordBytes).
                                    };
                                    SmallVector<int32_t> wraps = {
                                        static_cast<int32_t>(effK_w),     // D0: effectiveK/4
                                        static_cast<int32_t>(subTileDim), // D1: tile_m or tile_n
                                        static_cast<int32_t>(kRounds)     // D2: kRounds
                                    };
                                    inputShimDimStrides = rewriter.getI32ArrayAttr(strides);
                                    inputShimDimWraps = rewriter.getI32ArrayAttr(wraps);

                                    llvm::errs()
                                        << "[ShimMultiDim] Push input (3D K-tiling, sub-tile<partRows): "
                                        << "effectiveK=" << effectiveK << " fullK=" << fullK << " partRows=" << partRows
                                        << " tileM=" << tileM << " subTileDim=" << subTileDim << " dataId=" << dataId
                                        << " kRounds=" << kRounds << " elemsPerWord=" << elemsPerWord << " strides=["
                                        << strides[0] << "," << strides[1] << "," << strides[2] << "]"
                                        << " wraps=[" << wraps[0] << "," << wraps[1] << "," << wraps[2] << "]\n";
                                } else {
                                    // 2D addressing (existing): D0=K-chunk, D1=partRows
                                    // tile_m == partRows (or unset): no M sub-tiling needed.
                                    // D0+D1 covers one K-chunk across all rows, BD iteration
                                    // handles kRounds.
                                    SmallVector<int32_t> strides = {static_cast<int32_t>(1 * wordBytes),
                                                                    static_cast<int32_t>(fullK_w * wordBytes)};
                                    SmallVector<int32_t> wraps = {static_cast<int32_t>(effK_w),
                                                                  static_cast<int32_t>(partRows)};

                                    inputShimDimStrides = rewriter.getI32ArrayAttr(strides);
                                    inputShimDimWraps = rewriter.getI32ArrayAttr(wraps);

                                    llvm::errs() << "[ShimMultiDim] Push input (2D K-tiling): "
                                                 << "effectiveK=" << effectiveK << " fullK=" << fullK
                                                 << " partRows=" << partRows << " elemsPerWord=" << elemsPerWord
                                                 << " strides=[" << strides[0] << "," << strides[1] << "]"
                                                 << " wraps=[" << wraps[0] << "," << wraps[1] << "]\n";
                                }
                            }
                        }
                    }
                } // end if (!inputShimDimStrides) — K-tiling fallback
            }

            // 2D width-split halo BD: when the conv WIDTH is chunked into on-core
            // rounds (w_rounds > 1), the per-round slab is a NARROW block
            // [halo_slice rows, chunk_width cols] cut from the PADDED DDR buffer with
            // a wide row pitch. The flat 1D BD (which assumes a contiguous slab) would
            // read chunk_width*halo_slice contiguous bytes and walk off into the next
            // padded rows. Emit a 2D shim BD that gathers chunk_width contiguous
            // elements per row, stepping row_pitch between rows, for halo_slice rows.
            // Mirrors the K-tiling 2D convention (D0=contiguous run, D1=row pitch),
            // strides in BYTES, wraps in WORDS for D0 / row-count for D1.
            if (moduleOp && isHaloTensor && !inputShimDimStrides) {
                std::string haloKey = "tensor_" + std::to_string(tensorIdx) + ".halo";
                auto haloAttr = moduleOp->getAttrOfType<DictionaryAttr>(haloKey);
                auto wRoundsAttr = haloAttr ? haloAttr.getAs<IntegerAttr>("w_rounds") : nullptr;
                // Prefer a full 3D D0/D1/D2 BD (the K-accum branch below) whenever the
                // K (input-channel) contraction is split into k_rounds>1 width-bands.
                // Both this width-split branch and the K-accum branch are gated on
                // !inputShimDimStrides (first match wins); when w_rounds>1 AND k_rounds>1
                // this 2D branch would otherwise fire first and block the 3D K-accum BD,
                // pushing the round count into iteration+channel-repeat instead of D2.
                // Yield to the 3D K-accum branch by skipping here when k_rounds>1.
                auto kRoundsGuard = haloAttr ? haloAttr.getAs<IntegerAttr>("k_rounds") : nullptr;
                bool kAccumWins = kRoundsGuard && kRoundsGuard.getInt() > 1;
                if (wRoundsAttr && wRoundsAttr.getInt() > 1 && !kAccumWins) {
                    auto rowPitchAttr = haloAttr.getAs<IntegerAttr>("row_pitch");
                    auto sliceAttr = haloAttr.getAs<IntegerAttr>("slice");
                    auto bufSizeAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size");
                    auto viewType = dyn_cast<RankedTensorType>(viewSplit.getType());
                    if (rowPitchAttr && sliceAttr && bufSizeAttr && viewType) {
                        unsigned bitWidth = viewType.getElementTypeBitWidth();
                        int64_t elemsPerWord = 32 / bitWidth; // 4 for i8, 2 for i16
                        constexpr int64_t wordBytes = 4;
                        int64_t haloSlice = sliceAttr.getInt();        // rows per slab (19)
                        int64_t rowPitchElems = rowPitchAttr.getInt(); // padded row pitch (920)
                        // chunk_width = narrow slab cols*C: bufSize = halo_slice * chunk_width.
                        int64_t chunkWidthElems = haloSlice > 0 ? bufSizeAttr.getInt() / haloSlice : 0;
                        if (chunkWidthElems > 0 && elemsPerWord > 0) {
                            int64_t chunkW_w = chunkWidthElems / elemsPerWord; // 244/4 = 61
                            int64_t rowPitch_w = rowPitchElems / elemsPerWord; // 920/4 = 230
                            SmallVector<int32_t> strides = {
                                static_cast<int32_t>(1 * wordBytes),         // D0: 4 bytes (contiguous)
                                static_cast<int32_t>(rowPitch_w * wordBytes) // D1: row pitch bytes (920)
                            };
                            SmallVector<int32_t> wraps = {
                                static_cast<int32_t>(chunkW_w), // D0: chunk_width/4 words (61)
                                static_cast<int32_t>(haloSlice) // D1: halo_slice rows (19)
                            };
                            inputShimDimStrides = rewriter.getI32ArrayAttr(strides);
                            inputShimDimWraps = rewriter.getI32ArrayAttr(wraps);
                            llvm::errs() << "[ShimMultiDim] Push input (2D halo width-split): chunkWidthElems="
                                         << chunkWidthElems << " haloSlice=" << haloSlice
                                         << " rowPitchElems=" << rowPitchElems << " strides=[" << strides[0] << ","
                                         << strides[1] << "] wraps=[" << wraps[0] << "," << wraps[1] << "]\n";
                        }
                    }
                }
            }

            // 3D K-accum halo BD: when the conv K (input-channel) contraction is split
            // into k_rounds width-bands of k_slice cols, ONE multi-dim shim BD can cover
            // the whole [l2_slice rows, k_slice cols, k_rounds bands] slab (the BD
            // iteration dim, plumbed in passblueprinttoschedule, then walks the l2_rounds
            // L2 bands). The slab is cut from the PADDED DDR buffer with a wide row pitch:
            //   D0 = contiguous k_slice run (cols),   stride 1 word
            //   D1 = l2_slice rows,                   stride row_pitch
            //   D2 = k_rounds width-bands,            stride k_step
            // strides in BYTES, wraps in WORDS for D0 / row-count for D1 / band-count D2.
            // Mirrors the matmul 3D K-tiling convention. Gate strictly on k_rounds>1 so
            // the w_rounds / pure-L2 / matmul paths stay byte-identical.
            if (moduleOp && isHaloTensor && !inputShimDimStrides) {
                std::string haloKeyK = "tensor_" + std::to_string(tensorIdx) + ".halo";
                auto haloAttrK = moduleOp->getAttrOfType<DictionaryAttr>(haloKeyK);
                auto kRoundsAttr = haloAttrK ? haloAttrK.getAs<IntegerAttr>("k_rounds") : nullptr;
                if (kRoundsAttr && kRoundsAttr.getInt() > 1) {
                    auto kSliceAttr = haloAttrK.getAs<IntegerAttr>("k_slice");
                    auto kStepAttr = haloAttrK.getAs<IntegerAttr>("k_step");
                    auto l2SliceAttr = haloAttrK.getAs<IntegerAttr>("l2_slice");
                    auto rowPitchAttrK = haloAttrK.getAs<IntegerAttr>("row_pitch");
                    auto viewTypeK = dyn_cast<RankedTensorType>(viewSplit.getType());
                    if (kSliceAttr && kStepAttr && l2SliceAttr && rowPitchAttrK && viewTypeK) {
                        unsigned bitWidthK = viewTypeK.getElementTypeBitWidth();
                        int64_t elemsPerWordK = 32 / bitWidthK; // 4 for i8, 2 for i16
                        constexpr int64_t wordBytesK = 4;
                        int64_t kSliceElems = kSliceAttr.getInt();   // 244
                        int64_t kStepElems = kStepAttr.getInt();     // 224
                        int64_t l2SliceRows = l2SliceAttr.getInt();  // 19
                        int64_t rowPitchEl = rowPitchAttrK.getInt(); // 920
                        int64_t kRounds = kRoundsAttr.getInt();      // 4
                        if (elemsPerWordK > 0) {
                            int64_t kSlice_w = kSliceElems / elemsPerWordK;  // 244/4 = 61
                            int64_t rowPitch_w = rowPitchEl / elemsPerWordK; // 920/4 = 230
                            int64_t kStep_w = kStepElems / elemsPerWordK;    // 224/4 = 56
                            SmallVector<int32_t> strides = {
                                static_cast<int32_t>(1 * wordBytesK),          // D0: 4 bytes (contiguous)
                                static_cast<int32_t>(rowPitch_w * wordBytesK), // D1: row pitch bytes (920)
                                static_cast<int32_t>(kStep_w * wordBytesK)     // D2: k_step bytes (224)
                            };
                            SmallVector<int32_t> wraps = {
                                static_cast<int32_t>(kSlice_w),    // D0: k_slice/4 words (61)
                                static_cast<int32_t>(l2SliceRows), // D1: l2_slice rows (19)
                                static_cast<int32_t>(kRounds)      // D2: k_rounds bands (4)
                            };
                            inputShimDimStrides = rewriter.getI32ArrayAttr(strides);
                            inputShimDimWraps = rewriter.getI32ArrayAttr(wraps);
                            llvm::errs() << "[ShimMultiDim] Push input (3D halo K-accum): k_slice=" << kSliceElems
                                         << " k_step=" << kStepElems << " l2_slice=" << l2SliceRows
                                         << " row_pitch=" << rowPitchEl << " k_rounds=" << kRounds << " strides=["
                                         << strides[0] << "," << strides[1] << "," << strides[2] << "] wraps=["
                                         << wraps[0] << "," << wraps[1] << "," << wraps[2] << "]\n";
                        }
                    }
                }
            }
        }

        // Use viewSplit (partition slice) so the shim BD covers only the data
        // for this round; rootAdaptedView is kept only for data_id assignment.
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(), rewriter.getStringAttr(srcBindName), FlatSymbolRefAttr::get(getContext(), srcGroupName),
            viewSplit, // partition slice (tensor<8x16xi8> for each round)
            rewriter.getStringAttr("root"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({sourceChannel}),
                                          dfscheblueprint::bp_direction::MM2S),
            nullptr, // slice_symbols - source is root, no slice
            rewriter.getStringAttr(srcTileType),
            rewriter.getI32IntegerAttr(dataId), // data_id for shim BD grouping/merging
            inputShimDimStrides,                // shim_dim_strides (user DmaTransform or K-tiling)
            inputShimDimWraps,                  // shim_dim_wraps
            nullptr                             // pp_depth
        );

        std::string dstBindName = "flow_dst_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(), rewriter.getStringAttr(dstBindName), FlatSymbolRefAttr::get(getContext(), dstGroupName),
            viewSplit, rewriter.getStringAttr("linear"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({destChannel}),
                                          dfscheblueprint::bp_direction::S2MM),
            rewriter.getArrayAttr(sliceSymbols), // Associate with @consumer_slice_0 to @consumer_slice_N
            rewriter.getStringAttr(dstTileType),
            nullptr, // data_id
            nullptr, // shim_dim_strides
            nullptr, // shim_dim_wraps
            nullptr  // pp_depth
        );

        // 5. Create Collective Transfer - one_to_many for push/scatter
        int32_t basePktId = 0; // Push flows are circuit-switched; no pkt_id
        rewriter.create<dfscheblueprint::FlowTransferOp>(
            op.getLoc(), rewriter.getStringAttr("transfer_" + std::to_string(opId)),
            rewriter.getStringAttr("one_to_many"), FlatSymbolRefAttr::get(getContext(), srcBindName),
            FlatSymbolRefAttr::get(getContext(), dstBindName), rewriter.getStringAttr("sequential"),
            rewriter.getI32IntegerAttr(basePktId),
            rewriter.getI32IntegerAttr(opId) // flow_index
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

        // Find the root (pre-split) view for data_id keying.
        // The partition view (%extracted_slice_0, the ExtractDataConversion output)
        // carries the #routing.tiling descriptor (H slice_tiling rounds=4/step=7,
        // W rounds=4/step=28). Propagate it onto each producer data_slice so the
        // downstream shim S2MM BD builder can realise the on-core 4x4 H/W iteration.
        Value rootAdaptedView = viewSplit;
        Attribute producerTiling; // null unless the partition view is tiled
        if (auto extractSlice = viewSplit.getDefiningOp<tensor::ExtractSliceOp>()) {
            rootAdaptedView = extractSlice.getSource();
            producerTiling = extractSlice->getAttr("tiling");
        }

        // Assign data_id: same ID for all pull sub-flows sharing the same root tensor.
        int32_t dataId;
        {
            auto it = g_rootViewToDataId.find(rootAdaptedView);
            if (it != g_rootViewToDataId.end()) {
                dataId = it->second;
            } else {
                dataId = g_dataIdCounter++;
                g_rootViewToDataId[rootAdaptedView] = dataId;
            }
        }

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
            auto dataSlice = rewriter.create<dfscheblueprint::DataSliceOp>(
                op.getLoc(),
                tensorType, // Result type
                rewriter.getStringAttr(sliceName),
                buffer // Use adapted producer buffer (DeclareDataOp result)
            );

            // Carry the partition-view tiling descriptor onto the producer slice so
            // the shim S2MM reassembly BD can realise the on-core H/W iteration.
            if (producerTiling)
                dataSlice->setAttr("tiling", producerTiling);

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
                        auto port = resolvePortFromPathSymbol(op, symbolRef);
                        if (!port) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to find producer port symbol '"
                                         << symbolRef.getValue() << "' in symbol table\n";
                            continue;
                        }
                        sourceChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                         static_cast<int64_t>(port.getDirectionChannel().value()));
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
                        auto port = resolvePortFromPathSymbol(op, symbolRef);
                        if (!port) {
                            llvm::errs() << "ERROR: PullOpConversion - failed to find consumer port symbol '"
                                         << symbolRef.getValue() << "' in symbol table\n";
                            continue;
                        }
                        destChannel = getDmaPortForSym(op, symbolRef.getValue(),
                                                       static_cast<int64_t>(port.getDirectionChannel().value()));
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
        // Look up pp_depth from module attribute for output (Pull) flows.
        // The pp_depth_map module attribute maps tensor_N -> pp_depth.
        // For Pull flows (output Gather), we find the matching output tensor.
        IntegerAttr ppDepthAttr = nullptr;
        if (auto moduleOp = op->getParentOfType<ModuleOp>()) {
            if (auto ppDepthMap = moduleOp->getAttrOfType<DictionaryAttr>("routing.pp_depth_map")) {
                // Trace rootAdaptedView back through the op chain to find
                // the func arg index, then look up pp_depth from the map.
                // Chain: routingextract_data -> partitiontensor ->
                //        createscheduletensor -> to_tensor -> block arg
                Value traceVal = rootAdaptedView;
                for (int step = 0; step < 10 && traceVal; ++step) {
                    if (auto blockArg = dyn_cast<BlockArgument>(traceVal)) {
                        // Reached func argument — look up pp_depth
                        unsigned tensorIdx = blockArg.getArgNumber();
                        std::string key = "tensor_" + std::to_string(tensorIdx);
                        if (auto attr = ppDepthMap.getAs<IntegerAttr>(key)) {
                            ppDepthAttr = attr;
                            llvm::errs() << "[PullOp] Found pp_depth=" << attr.getInt() << " for tensor_" << tensorIdx
                                         << "\n";
                        }
                        break;
                    }
                    Operation *defOp = traceVal.getDefiningOp();
                    if (!defOp)
                        break;
                    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(defOp))
                        traceVal = extractSlice.getSource();
                    else if (auto extractData = dyn_cast<routing::extract_data>(defOp))
                        traceVal = extractData.getTensor();
                    else if (auto partTensor = dyn_cast<routing::partitiontensor>(defOp))
                        traceVal = partTensor.getTensor();
                    else if (auto schedTensor = dyn_cast<routing::createscheduletensor>(defOp))
                        traceVal = schedTensor.getInitTensor();
                    else if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp))
                        traceVal = toTensor.getMemref();
                    else
                        break; // unknown op
                }
            }
        }

        std::string srcBindName = "flow_src_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(), rewriter.getStringAttr(srcBindName), FlatSymbolRefAttr::get(getContext(), srcGroupName),
            viewSplit, rewriter.getStringAttr("linear"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({sourceChannel}),
                                          dfscheblueprint::bp_direction::MM2S),
            rewriter.getArrayAttr(sliceSymbols), // slice_symbols
            rewriter.getStringAttr(srcTileType),
            nullptr,    // data_id
            nullptr,    // shim_dim_strides
            nullptr,    // shim_dim_wraps
            ppDepthAttr // pp_depth (from module attribute, for output flows)
        );

        // Compute multi-dimensional DMA addressing for shim S2MM (output assembly).
        // Each shim S2MM channel receives a row-strip from multiple cores.
        // The data arrives tile-by-tile but must be written into the correct 2D DDR layout.
        // The 4D addressing pattern reassembles tiles within one row-strip
        // across ping-pong rounds:
        //   D0: contiguous elements within a tile row   (wrap=tileW,             stride=1)
        //   D1: next row within same core per round     (wrap=coreRowsPerRound,  stride=outW)
        //   D2: next tile column                        (wrap=numTileCols,        stride=tileW)
        //   D3: next ping-pong round                    (wrap=pingPongFactor,     stride=coreRowsPerRound*outW)
        ArrayAttr shimDimStrides = nullptr;
        ArrayAttr shimDimWraps = nullptr;

        if (dstTileType == "shim") {
            // viewSplit is the per-channel aggregate (e.g. tensor<4x16xi8> for one row-strip)
            auto viewType = dyn_cast<RankedTensorType>(viewSplit.getType());
            // rootAdaptedView is the full output tensor (e.g. tensor<16x16xi8>)
            auto rootType = dyn_cast<RankedTensorType>(rootAdaptedView.getType());

            if (viewType && rootType && viewType.getRank() == 2 && rootType.getRank() == 2) {
                int64_t outW = rootType.getDimSize(1);   // full output width
                int64_t stripH = viewType.getDimSize(0); // strip height = tile height

                // Number of source tiles = number of cores sending to this channel
                int64_t numTileCols = static_cast<int64_t>(sourceTiles.size());

                // Per-core tile width = strip width / number of source tiles
                int64_t tileW = outW / numTileCols; // e.g. 16/4 = 4

                if (numTileCols > 1) {
                    // Strides are in BYTE units in the IR.
                    // The runtime (__Runtime_dma_bd_config_multidim) converts
                    // to 32-bit word units (÷4) before the XAie driver call.
                    //
                    // The DMA operates at 32-bit (4-byte) word granularity,
                    // so strides are first computed in word units and then
                    // multiplied by 4 to express them in bytes.
                    unsigned bitWidth = rootType.getElementTypeBitWidth();
                    int64_t elemsPerWord = 32 / bitWidth; // e.g. 4 for i8, 2 for i16
                    constexpr int64_t wordBytes = 4;      // 32-bit DMA word = 4 bytes

                    int64_t tileW_w = tileW / elemsPerWord; // tile width in words (for D0 wrap)
                    int64_t outW_w = outW / elemsPerWord;   // full output width in words

                    // 3D addressing for shim S2MM output assembly:
                    //
                    // Data arrival order: each tile sends all its rows (all
                    // ping-pong rounds) contiguously before the next tile.
                    // So: tile0_allrows, tile1_allrows, tile2_allrows, ...
                    //
                    // D0: 1 word stride (contiguous within tile row)
                    // D1: full output row width (jump to next row in DDR)
                    // D2: tile width (jump to next tile column)
                    SmallVector<int32_t> strides = {static_cast<int32_t>(1 * wordBytes),
                                                    static_cast<int32_t>(outW_w * wordBytes),
                                                    static_cast<int32_t>(tileW_w * wordBytes)};
                    SmallVector<int32_t> wraps = {static_cast<int32_t>(tileW_w), static_cast<int32_t>(stripH),
                                                  static_cast<int32_t>(numTileCols)};
                    shimDimStrides = rewriter.getI32ArrayAttr(strides);
                    shimDimWraps = rewriter.getI32ArrayAttr(wraps);

                    llvm::errs() << "[ShimMultiDim] Pull per-channel row-strip (3D): "
                                 << "outW=" << outW << " stripH=" << stripH << " tileW=" << tileW
                                 << " numTileCols=" << numTileCols << " elemsPerWord=" << elemsPerWord << " strides=["
                                 << strides[0] << "," << strides[1] << "," << strides[2] << "]"
                                 << " wraps=[" << wraps[0] << "," << wraps[1] << "," << wraps[2] << "]\n";
                }
            }
        }

        std::string dstBindName = "flow_dst_" + std::to_string(opId);
        rewriter.create<dfscheblueprint::FlowConfigOp>(
            op.getLoc(), rewriter.getStringAttr(dstBindName), FlatSymbolRefAttr::get(getContext(), dstGroupName),
            viewSplit, rewriter.getStringAttr("root"),
            dfscheblueprint::DMAAttr::get(getContext(), ArrayRef<int64_t>({destChannel}),
                                          dfscheblueprint::bp_direction::S2MM),
            nullptr, // slice_symbols
            rewriter.getStringAttr(dstTileType),
            rewriter.getI32IntegerAttr(dataId), // data_id for shim BD grouping
            shimDimStrides,                     // multi-dim strides for output assembly
            shimDimWraps,                       // multi-dim wraps for output assembly
            ppDepthAttr                         // pp_depth (same as src for consistency)
        );

        // 5. Create Collective Transfer
        int32_t basePktId = getBasePktIdFromProducers(pathOp, op);
        rewriter.create<dfscheblueprint::FlowTransferOp>(
            op.getLoc(), rewriter.getStringAttr("transfer_" + std::to_string(opId)),
            rewriter.getStringAttr("many_to_one"), FlatSymbolRefAttr::get(getContext(), srcBindName),
            FlatSymbolRefAttr::get(getContext(), dstBindName), rewriter.getStringAttr("sequential"),
            rewriter.getI32IntegerAttr(basePktId),
            rewriter.getI32IntegerAttr(opId) // flow_index
        );

        rewriter.eraseOp(op);
        return success();
    }
};

void DmaphopTodfscheblueprintPass::runOnOperation() {
    // Reset data_id map each pass run so IDs are fresh per compilation unit
    g_rootViewToDataId.clear();
    g_dataIdCounter = 0;

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
    patterns.add<EraseOpLowering<dmaphop::tile>, EraseOpLowering<dmaphop::port>, EraseOpLowering<dmaphop::create_hop>,
                 EraseOpLowering<dmaphop::create_path>, EraseOpLowering<dmaphop::alloc_buffer>,
                 EraseOpLowering<dmaphop::sync>, EraseOpLowering<dmaphop::dealloc_buffer>,
                 EraseOpLowering<dmaphop::consumer>, EraseOpLowering<dmaphop::producer>>(context);

    // Read the #routing.tiling scalars once here, before applyPartialConversion,
    // and hand them to PushOpConversion. The partitiontensor op survives this pass
    // (its conversion is disabled above), but reading at entry keeps the K-tiling
    // block independent of conversion order and consistent with the other passes.
    routing::GemmTilingScalars gemmTiling;
    if (auto moduleOp = dyn_cast<ModuleOp>(module))
        gemmTiling = routing::readGemmTilingScalars(moduleOp);

    patterns.add<CreateScheduleTensorConversion>(context);
    //patterns.add<PartitionTensorConversion>(context);
    patterns.add<ExtractDataConversion>(context);
    patterns.add<PushOpConversion>(context, gemmTiling);
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
