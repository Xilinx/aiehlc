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

// ============================================================================
// Fake Resource Manager for Lock and BD ID allocation
// ============================================================================
class KernelResourceManager {
public:
    KernelResourceManager() : nextBdId(0), nextLockId(0) {}
    
    // Allocate next BD ID (0, 1 for ping-pong)
    int32_t allocateBdId() { return nextBdId++; }
    
    // Allocate next Lock ID
    int64_t allocateLockId() { return nextLockId++; }
    
    // Reset for new kernel
    void reset() {
        nextBdId = 0;
        nextLockId = 0;
    }
    
private:
    int32_t nextBdId;
    int64_t nextLockId;
};

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

// Helper function to check if dskernel_receiver already exists in the module
static bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName) {
    // Find the module-level operation
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }
    
    // Search for existing dskernel_receiver with the given name
    for (Region &region : moduleOp->getRegions()) {
        for (Block &block : region) {
            for (Operation &op : block) {
                if (auto receiver = dyn_cast<dfschedule::DSKernelReceiverOp>(&op)) {
                    if (receiver.getSymName() == kernelName) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Helper function to get the module-level insertion point
static Operation* getModuleOp(Operation *rootOp) {
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }
    return moduleOp;
}

// Generate dfschedule.dskernel_receiver function
// Parameters:
//   - kernelName: symbol name for the kernel (same as load_kernel_group callee)
//   - tensorType: the tensor type for data
//   - bufferLen: buffer length for DMA BD
//   - basePacketId: base packet ID from FlowTransferOp
//   - coreChannel: DMA channel from core FlowConfig
static void generateDSKernelReceiver(
    ConversionPatternRewriter &rewriter,
    Location loc,
    Operation *insertBeforeOp,
    StringRef kernelName,
    RankedTensorType tensorType,
    int64_t bufferLen,
    uint32_t basePacketId,
    int64_t coreChannel,
    KernelResourceManager &resourceMgr) {
    
    // Save current insertion point
    OpBuilder::InsertionGuard guard(rewriter);
    
    // Find module and insert at module level (at the end of the first block)
    Operation *moduleOp = getModuleOp(insertBeforeOp);
    Block &moduleBlock = moduleOp->getRegions().front().front();
    
    // Check if block has a terminator, insert before it; otherwise insert at end
    if (!moduleBlock.empty() && moduleBlock.back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(&moduleBlock.back());
    } else {
        rewriter.setInsertionPointToEnd(&moduleBlock);
    }
    
    // Create the dskernel_receiver operation
    auto receiverOp = rewriter.create<dfschedule::DSKernelReceiverOp>(
        loc,
        rewriter.getStringAttr(kernelName));
    
    // Create the body block with arguments:
    // %arg0: !dfschedule.packet
    // %arg1: !dfschedule.tile  
    // %arg2: !dfschedule.compute
    // %arg3: index (loop count)
    Block *body = &receiverOp.getBody().emplaceBlock();
    
    auto packetType = dfschedule::PacketType::get(rewriter.getContext());
    auto tileType = dfschedule::TileType::get(rewriter.getContext());
    auto computeType = dfschedule::ComputeType::get(rewriter.getContext());
    auto indexType = rewriter.getIndexType();
    
    auto arg0 = body->addArgument(packetType, loc);   // packet
    auto arg1 = body->addArgument(tileType, loc);     // tile
    auto arg2 = body->addArgument(computeType, loc);  // compute
    auto arg3 = body->addArgument(indexType, loc);    // loop_count
    
    // Set insertion point to the body
    rewriter.setInsertionPointToStart(body);
    
    // Reset resource manager for this kernel
    resourceMgr.reset();
    
    // %1 = dfschedule.gettensor(%arg0) : (!dfschedule.packet) -> tensor<...>
    auto getTensorOp = rewriter.create<dfschedule::GetTensorOp>(
        loc, tensorType, arg0);
    
    // %2 = dfschedule.getdmachannel(%arg0) : (!dfschedule.packet) -> !dfschedule.dma_channel
    auto getDmaChannelOp = rewriter.create<dfschedule::GetDmaChannelOp>(
        loc, dfschedule::DmaChannelType::get(rewriter.getContext()), arg0);
    
    // Create LOCAL memref type for ping-pong buffers
    auto localMemRefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType(),
        AffineMap(),
        rewriter.getStringAttr("LOCAL"));
    
    // %3 = dfschedule.kernel.memalloc(%1) : ping buffer
    auto pingBuffer = rewriter.create<dfschedule::KernelMemAllocOp>(
        loc, localMemRefType, getTensorOp.getTensor());
    
    // %4 = dfschedule.kernel.memalloc(%1) : pong buffer
    auto pongBuffer = rewriter.create<dfschedule::KernelMemAllocOp>(
        loc, localMemRefType, getTensorOp.getTensor());
    
    // Create bd_id constants for ping (0) and pong (1)
    int32_t pingBdId = resourceMgr.allocateBdId();  // 0
    int32_t pongBdId = resourceMgr.allocateBdId();  // 1
    
    auto pingBdIdConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(pingBdId));
    auto pongBdIdConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(pongBdId));
    
    // %5 = dfschedule.config.dma_bd for ping buffer
    // bd_id=0, next_bd=1, packet_id=basePacketId
    auto pingDmaBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc,
        dfschedule::BdHandleType::get(rewriter.getContext()),
        pingBuffer.getBuffer(),                       // buffer
        arg1,                                         // tile
        pingBdIdConst.getResult(),                    // bd_id
        rewriter.getI32IntegerAttr(0),                // offset
        rewriter.getI32IntegerAttr(bufferLen),        // len
        rewriter.getBoolAttr(true),                   // enable_packet
        rewriter.getI32IntegerAttr(basePacketId),     // packet_id
        rewriter.getI32IntegerAttr(pongBdId));        // next_bd -> points to pong
    
    // %6 = dfschedule.config.dma_bd for pong buffer
    // bd_id=1, next_bd=0, packet_id=basePacketId+1
    auto pongDmaBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc,
        dfschedule::BdHandleType::get(rewriter.getContext()),
        pongBuffer.getBuffer(),                       // buffer
        arg1,                                         // tile
        pongBdIdConst.getResult(),                    // bd_id
        rewriter.getI32IntegerAttr(0),                // offset
        rewriter.getI32IntegerAttr(bufferLen),        // len
        rewriter.getBoolAttr(true),                   // enable_packet
        rewriter.getI32IntegerAttr(basePacketId + 1), // packet_id (next)
        rewriter.getI32IntegerAttr(pingBdId));        // next_bd -> points back to ping
    
    // Initialize locks for ping-pong synchronization
    // Lock values: acquire locks start at 0, release locks: ping=1, pong=0
    
    // %7 = dfschedule.dskernel.lock_init(0, "ping_acquire_lock")
    auto pingAcqLock = rewriter.create<dfschedule::DSKernelLockInitOp>(
        loc,
        dfschedule::LockType::get(rewriter.getContext()),
        rewriter.getI64IntegerAttr(0),
        rewriter.getStringAttr("ping_acquire_lock"));
    
    // %8 = dfschedule.dskernel.lock_init(0, "pong_acquire_lock")
    auto pongAcqLock = rewriter.create<dfschedule::DSKernelLockInitOp>(
        loc,
        dfschedule::LockType::get(rewriter.getContext()),
        rewriter.getI64IntegerAttr(0),
        rewriter.getStringAttr("pong_acquire_lock"));
    
    // %9 = dfschedule.dskernel.lock_init(1, "ping_release_lock")
    auto pingRelLock = rewriter.create<dfschedule::DSKernelLockInitOp>(
        loc,
        dfschedule::LockType::get(rewriter.getContext()),
        rewriter.getI64IntegerAttr(1),
        rewriter.getStringAttr("ping_release_lock"));
    
    // %10 = dfschedule.dskernel.lock_init(0, "pong_release_lock")
    auto pongRelLock = rewriter.create<dfschedule::DSKernelLockInitOp>(
        loc,
        dfschedule::LockType::get(rewriter.getContext()),
        rewriter.getI64IntegerAttr(0),
        rewriter.getStringAttr("pong_release_lock"));
    
    // dfschedule.dskernel.launch_dma_s2m_loop
    rewriter.create<dfschedule::DSKernelLaunchDmaLoopOp>(
        loc,
        pingBuffer.getBuffer(),
        pongBuffer.getBuffer(),
        pingAcqLock.getLock(),
        pingRelLock.getLock(),
        pongAcqLock.getLock(),
        pongRelLock.getLock(),
        getDmaChannelOp.getChannel());
    
    // Create constants for loop
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    
    // scf.for %arg4 = %c0 to %arg3 step %c1
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0.getResult(), arg3, c1.getResult());
    
    // Build the loop body
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    
    // %11 = arith.remui %arg4, %c2 : index
    auto remOp = rewriter.create<arith::RemUIOp>(loc, iv, c2.getResult());
    
    // %12 = arith.cmpi eq, %11, %c0 : index
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, remOp.getResult(), c0.getResult());
    
    // %13 = scf.if %12 -> (memref<...>) { yield ping } else { yield pong }
    auto selectBufferOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{localMemRefType}, cmpOp.getResult(), /*withElseRegion=*/true);
    
    // Then block: yield ping
    rewriter.setInsertionPointToStart(&selectBufferOp.getThenRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pingBuffer.getBuffer()});
    
    // Else block: yield pong
    rewriter.setInsertionPointToStart(&selectBufferOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pongBuffer.getBuffer()});
    
    // Continue after if
    rewriter.setInsertionPointAfter(selectBufferOp);
    Value selectedBuffer = selectBufferOp.getResult(0);
    
    // %14 = scf.if %12 -> (!dfschedule.lock) { yield ping_acq } else { yield pong_acq }
    auto lockType = dfschedule::LockType::get(rewriter.getContext());
    auto selectAcqLockOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{lockType}, cmpOp.getResult(), /*withElseRegion=*/true);
    
    rewriter.setInsertionPointToStart(&selectAcqLockOp.getThenRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pingAcqLock.getLock()});
    
    rewriter.setInsertionPointToStart(&selectAcqLockOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pongAcqLock.getLock()});
    
    rewriter.setInsertionPointAfter(selectAcqLockOp);
    Value selectedAcqLock = selectAcqLockOp.getResult(0);
    
    // %15 = scf.if %12 -> (!dfschedule.lock) { yield ping_rel } else { yield pong_rel }
    auto selectRelLockOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{lockType}, cmpOp.getResult(), /*withElseRegion=*/true);
    
    rewriter.setInsertionPointToStart(&selectRelLockOp.getThenRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pingRelLock.getLock()});
    
    rewriter.setInsertionPointToStart(&selectRelLockOp.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{pongRelLock.getLock()});
    
    rewriter.setInsertionPointAfter(selectRelLockOp);
    Value selectedRelLock = selectRelLockOp.getResult(0);
    
    // dfschedule.dskernel.acquire_lock(%14, 1)
    rewriter.create<dfschedule::DSKernelAcquireLockOp>(
        loc, selectedAcqLock, rewriter.getI32IntegerAttr(1));
    
    // dfschedule.core.compute(%13, %arg2)
    rewriter.create<dfschedule::CoreComputeOp>(loc, selectedBuffer, arg2);
    
    // dfschedule.dskernel.release_lock(%15, 1)
    rewriter.create<dfschedule::DSKernelReleaseLockOp>(
        loc, selectedRelLock, rewriter.getI32IntegerAttr(1));
}

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
// Logic:
// 1. Find shim tile from the "from" and "to" FlowConfigOps by checking type="shim"
// 2. Get DMA configuration from the shim FlowConfig's DMA attribute
// 3. Only do DMA config (declaretensor, declaretile, config.dma_bd, config.create_io) for shim tile
// 4. Create packet ops for core tiles, load_kernel_group, launch, schedule ops
// 5. Generate dskernel_receiver function with kernel DMA BD config
struct FlowTransferConversion : public OpConversionPattern<dfscheblueprint::FlowTransferOp> {
    using OpConversionPattern<dfscheblueprint::FlowTransferOp>::OpConversionPattern;
    
    // Mutable resource manager for kernel resource allocation
    mutable KernelResourceManager resourceMgr;

    LogicalResult
    matchAndRewrite(dfscheblueprint::FlowTransferOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Look up "from" FlowConfigOp
        SymbolRefAttr fromRef = op.getFrom();
        auto fromFlowConfig = lookupFlowConfig(op.getOperation(), fromRef);
        if (!fromFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Look up "to" FlowConfigOp
        SymbolRefAttr toRef = op.getTo();
        auto toFlowConfig = lookupFlowConfig(op.getOperation(), toRef);
        if (!toFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Step 1: Find the shim tile by checking type field
        // Determine which FlowConfig is shim and which is core
        dfscheblueprint::FlowConfigOp shimFlowConfig = nullptr;
        dfscheblueprint::FlowConfigOp coreFlowConfig = nullptr;
        bool shimIsSender = false;  // true if shim is "from" (MM2S), false if shim is "to" (S2MM)
        
        auto fromType = fromFlowConfig.getType();
        auto toType = toFlowConfig.getType();
        
        if (fromType && *fromType == "shim") {
            shimFlowConfig = fromFlowConfig;
            coreFlowConfig = toFlowConfig;
            shimIsSender = true;
        } else if (toType && *toType == "shim") {
            shimFlowConfig = toFlowConfig;
            coreFlowConfig = fromFlowConfig;
            shimIsSender = false;
        } else {
            // No shim tile found, just erase
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get tile groups
        auto shimTileGroup = lookupTileGroup(shimFlowConfig.getOperation(), shimFlowConfig.getTarget());
        auto coreTileGroup = lookupTileGroup(coreFlowConfig.getOperation(), coreFlowConfig.getTarget());
        if (!shimTileGroup || !coreTileGroup) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get base packet id for packet operations
        uint32_t basePacketId = op.getBasePacketId();
        
        // Get the view operand from the shim FlowConfig (shim holds the data buffer)
        Value viewValue = shimFlowConfig.getView();
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
        
        // --- Step 2 & 3: DMA CONFIG FOR SHIM TILE ONLY ---
        ArrayAttr shimTilesAttr = shimTileGroup.getTiles();
        if (shimTilesAttr.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Get first shim tile coordinates
        auto firstShimTile = dyn_cast<ArrayAttr>(shimTilesAttr[0]);
        if (!firstShimTile || firstShimTile.size() < 2) {
            rewriter.eraseOp(op);
            return success();
        }
        int64_t shimCol = cast<IntegerAttr>(firstShimTile[0]).getInt();
        int64_t shimRow = cast<IntegerAttr>(firstShimTile[1]).getInt();
        
        // Create dfschedule.declaretile for shim
        auto shimTileOp = rewriter.create<dfschedule::DeclareTileOp>(
            loc,
            dfschedule::TileType::get(rewriter.getContext()),
            rewriter.getI32IntegerAttr(shimCol),
            rewriter.getI32IntegerAttr(shimRow));
        
        // Step 2: Get DMA configuration from shim FlowConfig's DMA attribute
        auto shimDmaAttr = shimFlowConfig.getDma();
        auto shimDmaChannels = shimDmaAttr.getChannels();
        int64_t shimChannel = shimDmaChannels.empty() ? 0 : shimDmaChannels[0];
        
        // Determine DMA direction based on whether shim is sender or receiver
        StringRef dmaDirection = shimIsSender ? "MM2S" : "S2MM";
        StringRef ioOperation = shimIsSender ? "SEND" : "RECV";
        
        // Calculate buffer size
        int64_t bufferLen = 1;
        for (int64_t dim : memrefType.getShape()) {
            bufferLen *= dim;
        }
        
        // Create bd_id constant for config
        auto bdIdConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        
        // Step 3: Create dfschedule.config.dma_bd for shim tile only
        auto configDmaBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
            loc,
            dfschedule::BdHandleType::get(rewriter.getContext()),
            memrefValue,                                      // buffer
            shimTileOp.getTile(),                             // tile
            bdIdConst.getResult(),                            // bd_id
            rewriter.getI32IntegerAttr(0),                    // offset
            rewriter.getI32IntegerAttr(bufferLen),            // len
            rewriter.getBoolAttr(true),                       // enable_packet
            rewriter.getI32IntegerAttr(basePacketId),         // packet_id
            rewriter.getI32IntegerAttr(4294967295));          // next_bd (-1 as unsigned)
        
        // Create dfschedule.config.create_io for shim tile
        auto createIoOp = rewriter.create<dfschedule::ConfigCreateIoOp>(
            loc,
            dfschedule::IoHandleType::get(rewriter.getContext()),
            configDmaBdOp.getBdHandle(),                      // bd_config
            shimTileOp.getTile(),                             // tile
            rewriter.getI32IntegerAttr(shimChannel),          // channel
            rewriter.getStringAttr(dmaDirection),             // direction (MM2S or S2MM)
            rewriter.getStringAttr(ioOperation));             // io_operation (SEND or RECV)
        
        // --- CORE TILES (no DMA config, just declaretile and packets) ---
        ArrayAttr coreTilesAttr = coreTileGroup.getTiles();
        SmallVector<Value> coreTiles;
        SmallVector<SymbolRefAttr> packetSymbols;
        uint32_t packetIdx = 0;
        
        // Get DMA channel from core FlowConfig for packet ops
        auto coreDmaAttr = coreFlowConfig.getDma();
        auto coreDmaChannels = coreDmaAttr.getChannels();
        int64_t coreChannel = coreDmaChannels.empty() ? 0 : coreDmaChannels[0];
        
        for (auto tileAttr : coreTilesAttr) {
            auto tileArray = dyn_cast<ArrayAttr>(tileAttr);
            if (!tileArray || tileArray.size() < 2) {
                continue;
            }
            
            int64_t col = cast<IntegerAttr>(tileArray[0]).getInt();
            int64_t row = cast<IntegerAttr>(tileArray[1]).getInt();
            
            // Create dfschedule.declaretile for each core tile
            auto coreTileOp = rewriter.create<dfschedule::DeclareTileOp>(
                loc,
                dfschedule::TileType::get(rewriter.getContext()),
                rewriter.getI32IntegerAttr(col),
                rewriter.getI32IntegerAttr(row));
            coreTiles.push_back(coreTileOp.getTile());
            
            // Create packet symbol name (packet0, packet1, ...)
            std::string packetName = "packet" + std::to_string(packetIdx);
            
            // Create dfschedule.packet for each core tile
            rewriter.create<dfschedule::PacketOp>(
                loc,
                dfschedule::PacketType::get(rewriter.getContext()),
                rewriter.getStringAttr(packetName),
                memrefValue,
                rewriter.getI32IntegerAttr(coreChannel));
            
            packetSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), packetName));
            packetIdx++;
        }
        
        if (coreTiles.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Create callee symbol refs (dskernel_receiver for all)
        SmallVector<Attribute> calleeAttrs;
        calleeAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "dskernel_receiver"));
        
        // Create distributed_compute_kernel_args (compute0 for all)
        SmallVector<Attribute> computeKernelAttrs;
        for (size_t i = 0; i < coreTiles.size(); ++i) {
            computeKernelAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "compute0"));
            }
            
        // Create distributed_args from packet symbols
        SmallVector<Attribute> distArgsAttrs(packetSymbols.begin(), packetSymbols.end());
        
        // Create dfschedule.config.load_kernel_group
        auto loadKernelGroupOp = rewriter.create<dfschedule::LoadKernelGroupOp>(
                loc,
            dfschedule::KernelGroupType::get(rewriter.getContext()),
            coreTiles,
            rewriter.getArrayAttr(calleeAttrs),
            rewriter.getArrayAttr(computeKernelAttrs),
            rewriter.getArrayAttr(distArgsAttrs));
            
        // Create dfschedule.schedule.launch_kernel_group
        auto launchKernelGroupOp = rewriter.create<dfschedule::LaunchKernelGroupOp>(
            loc,
            dfschedule::EventType::get(rewriter.getContext()),
            loadKernelGroupOp.getKernelGroup());
        
        // Create dfschedule.schedule.getbdid for shim tile
        auto getBdIdOp = rewriter.create<dfschedule::GetBdIdOp>(
            loc,
            rewriter.getI32Type(),
            shimTileOp.getTile());
        
        // Create dfschedule.schedule.start_io for shim
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
        
        // --- Step 5: Generate dskernel_receiver function ---
        // Use the same symbol name as load_kernel_group callee
        StringRef kernelName = "dskernel_receiver";
        
        // Only generate if it doesn't already exist
        if (!hasDSKernelReceiver(op.getOperation(), kernelName)) {
            // Get the tensor type from view for kernel generation
            RankedTensorType kernelTensorType;
            if (auto tensorType = dyn_cast<RankedTensorType>(viewType)) {
                kernelTensorType = tensorType;
            } else if (auto mrType = dyn_cast<MemRefType>(viewType)) {
                kernelTensorType = RankedTensorType::get(mrType.getShape(), mrType.getElementType());
            }
            
            if (kernelTensorType) {
                generateDSKernelReceiver(
                    rewriter,
                    loc,
                    op.getOperation(),
                    kernelName,
                    kernelTensorType,
                    bufferLen,
                    basePacketId,
                    coreChannel,
                    resourceMgr);
            }
        }
        
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
