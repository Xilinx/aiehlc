/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// HOST group of the split FlowTransferConversion — shim DMA + schedule.
//
// Pure mechanical extraction of the former single matchAndRewrite shim-DMA and
// schedule sections (orig lines 856-1544 and 2248-2943). Op-creation ordering is
// byte-identical to the original; all cross-boundary locals are carried in
// FlowLoweringCtx (`c`).

#include "flowtransfer_internal.h"

namespace blueprint_sched {

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

// ---------------------------------------------------------------------------
// emitShimTileAndParams — shim declaretile + DMA dir + buffer-size + halo detect
// (orig 856-957).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitShimTileAndParams(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    MemRefType memrefType = c.memrefType;

    // --- Step 2 & 3: DMA CONFIG FOR SHIM TILE ONLY ---
    ArrayAttr shimTilesAttr = c.shimTileGroup.getTiles();
    // (caller guarantees shimTilesAttr non-empty check already produced valid tile;
    //  keep same guards for byte-identical behavior)
    auto firstShimTile = dyn_cast<ArrayAttr>(shimTilesAttr[0]);
    c.shimCol = cast<IntegerAttr>(firstShimTile[0]).getInt();
    c.shimRow = cast<IntegerAttr>(firstShimTile[1]).getInt();

    // Create dfschedule.declaretile for shim
    c.shimTileOp = rewriter.create<dfschedule::DeclareTileOp>(loc, dfschedule::TileType::get(rewriter.getContext()),
                                                              rewriter.getI32IntegerAttr(c.shimCol),
                                                              rewriter.getI32IntegerAttr(c.shimRow));

    // Step 2: Get DMA configuration from shim FlowConfig's DMA attribute
    auto shimDmaAttr = c.shimFlowConfig.getDma();
    auto shimDmaChannels = shimDmaAttr.getChannels();
    c.shimChannel = shimDmaChannels.empty() ? 0 : shimDmaChannels[0];

    // Determine DMA direction based on whether shim is sender or receiver
    c.dmaDirection = c.shimIsSender ? "MM2S" : "S2MM";
    c.ioOperation = c.shimIsSender ? "SEND" : "RECV";

    // Calculate buffer size (logical partition view)
    c.bufferLen = 1;
    for (int64_t dim : memrefType.getShape()) {
        c.bufferLen *= dim;
    }

    // Shim BD len in bytes: runtime passes len directly to
    // XAie_DmaSetAddrLen, so compute the total byte count here.
    c.elementSizeBytesShim = 1;
    if (memrefType.getElementType().isIntOrFloat())
        c.elementSizeBytesShim = memrefType.getElementTypeBitWidth() / 8;
    if (c.elementSizeBytesShim == 0)
        c.elementSizeBytesShim = 1;
    c.transferType = op.getType();
    c.shimBdLen = c.bufferLen * c.elementSizeBytesShim;
    // Read data_id from shimFlowConfig (set by DmaphopTodfscheblueprintPass).
    auto dataIdOpt = c.shimFlowConfig.getDataId();
    c.dataId = dataIdOpt.has_value() ? static_cast<int32_t>(*dataIdOpt) : -1;

    // Spatial-halo IFM slab detection (function-scope, reused below).
    c.isHaloSlab = false;
    c.kAccumHaloSlab = false;
    c.kAccumL2Rounds = 0;
    c.kAccumL2Step = 0;
    c.kAccumRowPitchElems = 0;
    c.l2CorePerRoundLen = 0;
    {
        auto moduleOpHalo = op->getParentOfType<ModuleOp>();
        if (moduleOpHalo) {
            int haloArgIdx = traceFlowConfigToFuncArgIndex(c.shimFlowConfig);
            if (haloArgIdx >= 0) {
                std::string haloAttrName = "tensor_" + std::to_string(haloArgIdx) + ".halo";
                if (auto haloDict = moduleOpHalo->getAttrOfType<DictionaryAttr>(haloAttrName)) {
                    c.isHaloSlab = true;
                    auto kRoundsA = haloDict.getAs<IntegerAttr>("k_rounds");
                    if (kRoundsA && kRoundsA.getInt() > 1) {
                        c.kAccumHaloSlab = true;
                        if (auto a = haloDict.getAs<IntegerAttr>("l2_rounds"))
                            c.kAccumL2Rounds = a.getInt();
                        if (auto a = haloDict.getAs<IntegerAttr>("l2_step"))
                            c.kAccumL2Step = a.getInt();
                        if (auto a = haloDict.getAs<IntegerAttr>("row_pitch"))
                            c.kAccumRowPitchElems = a.getInt();
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// computeShimBdParams — shim BD length + OOO/iter param compute (orig 958-1174).
// ---------------------------------------------------------------------------
LogicalResult FlowTransferConversion::computeShimBdParams(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    MemRefType memrefType = c.memrefType;
    const int64_t elementSizeBytesShim = c.elementSizeBytesShim;
    const int64_t shimBdLen = c.shimBdLen;
    const int32_t dataId = c.dataId;

    // --- Step 3: Single shim BD with NO packet mode ---
    ArrayAttr coreTilesAttrForShim = c.coreTileGroup.getTiles();
    c.numCoreTiles = coreTilesAttrForShim.size();
    if (c.numCoreTiles <= 0)
        c.numCoreTiles = 1;
    int64_t numCoreTiles = c.numCoreTiles;

    // Shim BD len = partition size for BOTH one_to_many and many_to_one.
    c.perTileShimLen = shimBdLen;

    // Two-level (nested) halo: shrink the per-round shim BD length.
    if (c.isHaloSlab && c.shimIsSender) {
        auto moduleOpL2 = op->getParentOfType<ModuleOp>();
        int haloArgIdxL2 = traceFlowConfigToFuncArgIndex(c.shimFlowConfig);
        if (moduleOpL2 && haloArgIdxL2 >= 0) {
            std::string haloNameL2 = "tensor_" + std::to_string(haloArgIdxL2) + ".halo";
            if (auto haloDictL2 = moduleOpL2->getAttrOfType<DictionaryAttr>(haloNameL2)) {
                auto l2RoundsA = haloDictL2.getAs<IntegerAttr>("l2_rounds");
                auto l2SliceA = haloDictL2.getAs<IntegerAttr>("l2_slice");
                auto kSliceA = haloDictL2.getAs<IntegerAttr>("k_slice");
                auto kRoundsA = haloDictL2.getAs<IntegerAttr>("k_rounds");
                if (l2RoundsA && l2RoundsA.getInt() > 1 && l2SliceA && l2SliceA.getInt() > 0) {
                    // Row width = product of all dims after dim 0. Rank-generic: a 2D
                    // flattened input [H, W*C] gives shape[1]; a genuine 3D input
                    // [H, W, C] gives W*C — the same contiguous padded row pitch.
                    int64_t rawWcL2 = 1;
                    if (memrefType.getRank() >= 2)
                        for (int64_t di = 1; di < memrefType.getRank(); ++di)
                            rawWcL2 *= memrefType.getShape()[di];
                    else if (memrefType.getRank() == 1)
                        rawWcL2 = memrefType.getShape()[0];
                    if (kRoundsA && kRoundsA.getInt() > 1 && kSliceA && kSliceA.getInt() > 0) {
                        c.perTileShimLen = l2SliceA.getInt() * kSliceA.getInt() * kRoundsA.getInt() *
                                           elementSizeBytesShim; // 19*244*4 = 18544
                        c.l2CorePerRoundLen = l2SliceA.getInt() * kSliceA.getInt() * elementSizeBytesShim; // 4636
                    } else {
                        c.perTileShimLen = l2SliceA.getInt() * rawWcL2 * elementSizeBytesShim; // legacy 17480
                        c.l2CorePerRoundLen = c.perTileShimLen;
                    }
                    llvm::errs() << "[BlueprintToSchedule] L2 halo slab: l2_slice=" << l2SliceA.getInt()
                                 << " raw_wc=" << rawWcL2 << " k_slice=" << (kSliceA ? kSliceA.getInt() : 0)
                                 << " k_rounds=" << (kRoundsA ? kRoundsA.getInt() : 0)
                                 << " perTileShimLen=" << c.perTileShimLen << " (was shimBdLen=" << shimBdLen << ")\n";
                }
            }
        }
    }

    // K-round iteration: per-iteration transfer size for iter_wrap>1.
    // Tiling scalars come from passState (read once at pass entry from the
    // #routing.tiling op, flat-attr fallback). A missing attr previously read as
    // 0, matching the struct default, so the value-based guards are equivalent.
    {
        int64_t kRounds = passState->kRounds;
        int64_t tileM = passState->tileM;
        int64_t effectiveK = passState->effectiveK;
        int64_t tileRows = passState->tileRows;

        if (!c.isHaloSlab && kRounds > 1 && c.shimIsSender) {
            if (dataId == 0) {
                int64_t tileN = passState->tileN;
                int64_t tileCols = passState->tileCols;
                if (tileN > 0 && tileN < tileCols) {
                    c.perTileShimLen = tileN * effectiveK * kRounds;
                } else {
                    c.perTileShimLen = shimBdLen / kRounds;
                }
            } else if (tileM > 0 && tileM < tileRows) {
                c.perTileShimLen = tileM * effectiveK * kRounds;
            } else {
                c.perTileShimLen = shimBdLen / kRounds;
            }
        }
    }

    // Single shim BD: allocate BD ID from ResourceMgr.
    c.shimBdIdVal = -1;
    if (resourceMgr) {
        auto bdOpt = resourceMgr->allocateTileBd(c.shimRow, c.shimCol, /*ownerId=*/c.flowIndex);
        if (bdOpt)
            c.shimBdIdVal = *bdOpt;
    }
    if (c.shimBdIdVal < 0)
        c.shimBdIdVal = static_cast<int32_t>(c.shimChannel); // fallback

    c.shimBdIdConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(c.shimBdIdVal));

    // Read multi-dim addressing from FlowConfigOp.
    c.shimDimStrides = c.shimFlowConfig.getShimDimStridesAttr();
    c.shimDimWraps = c.shimFlowConfig.getShimDimWrapsAttr();

    // === Output gather (many_to_one) handling ===
    c.isManyToOne = (c.transferType == "many_to_one");
    c.useOOO = c.isManyToOne;

    c.shimPerTileBdIds.clear();
    c.lastShimBdHandle = Value();
    c.shimBdHandles.clear();

    // --- Compute per-round parameters for OOO iteration ---
    c.ooElementSizeBytes = 1;
    if (memrefType.getElementType().isIntOrFloat())
        c.ooElementSizeBytes = memrefType.getElementTypeBitWidth() / 8;
    if (c.ooElementSizeBytes == 0)
        c.ooElementSizeBytes = 1;

    c.ooFullPartitionElements = shimBdLen / c.ooElementSizeBytes;
    c.ooPerCoreElements = c.ooFullPartitionElements;
    // group2 conv OUTPUT (channel-split): the partition VIEW is ALREADY channel-
    // split (Part 1), so shimBdLen == ONE core's full output (one channel group).
    // Do NOT re-divide by numCoreTiles (kept in lockstep with flowtransfer_kernel.cpp).
    bool ooGroup2ConvOutput = false;
    if (c.isManyToOne && memrefType.getRank() == 3) {
        auto moduleOpG2 = op->getParentOfType<ModuleOp>();
        if (detectConvHalo(moduleOpG2).valid)
            ooGroup2ConvOutput = true;
    }
    if (c.isManyToOne && !ooGroup2ConvOutput)
        c.ooPerCoreElements = c.ooFullPartitionElements / numCoreTiles;

    c.ooPingPongSize = c.ooPerCoreElements;
    if (c.ooPingPongSize <= 0)
        c.ooPingPongSize = 1;
    // Spatial-halo conv output: the shim S2MM gathers one kernel output slab
    // ([oh_per_row*ow_t, tile_n]) per packet-switched round, so its per-round
    // size MUST equal the core MM2S per-slab size (perCoreElements/out_rounds),
    // NOT the largest arbitrary divisor. The generic divisor search below would
    // pick the mathematically-largest divisor ≤ maxElements (e.g. 3584 for
    // 50176) which does not fall on a kernel slab boundary and desyncs the
    // packet stream from the core sender. Honor the authoritative slab count
    // carried in "routing.spatial_out_rounds" so shim and core agree exactly.
    // Kept in lockstep with flowtransfer_kernel.cpp / passblueprinttoschedulekernel.cpp.
    bool ooSpatialOutSlab = false;
    if (c.isManyToOne) {
        auto moduleOpOut = op->getParentOfType<ModuleOp>();
        if (moduleOpOut) {
            if (auto outRoundsAttr = moduleOpOut->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds")) {
                int64_t outRounds = outRoundsAttr.getInt();
                if (outRounds > 1 && c.ooPerCoreElements % outRounds == 0) {
                    c.ooPingPongSize = c.ooPerCoreElements / outRounds;
                    ooSpatialOutSlab = true;
                    llvm::errs() << "[BlueprintToSchedule] Spatial-halo output shim: ooPingPongSize="
                                 << c.ooPingPongSize << " (perCoreElements=" << c.ooPerCoreElements
                                 << " / out_rounds=" << outRounds << ")\n";
                }
            }
        }
    }
    if (!ooSpatialOutSlab && maxPingPongBytes > 0 && c.ooElementSizeBytes > 0) {
        int64_t maxElements = maxPingPongBytes / c.ooElementSizeBytes;
        if (maxElements > 0 && c.ooPingPongSize > maxElements) {
            int64_t chosen = 1;
            for (int64_t cand = maxElements; cand >= 1; --cand) {
                if (c.ooPerCoreElements % cand == 0) {
                    chosen = cand;
                    break;
                }
            }
            c.ooPingPongSize = chosen;
        }
    }
    if (c.ooPerCoreElements % c.ooPingPongSize != 0) {
        op.emitError("OOO iteration: perCoreElements (")
            << c.ooPerCoreElements << ") not divisible by pingPongSize (" << c.ooPingPongSize << ")";
        return failure();
    }
    c.ooNumIterations = c.ooPerCoreElements / c.ooPingPongSize;
    c.perRoundBytes = c.ooPingPongSize * c.ooElementSizeBytes;

    // Shim BD iteration settings for K-round stepping.
    c.shimIterStepSize = 0;
    c.shimIterWrap = 0;

    // OOO m_rounds loop variables.
    c.usedMRounds3D = false;
    c.oooMRounds = 1;
    c.oooIterStepSize = 0;
    c.oooIterWrap = 0;
    c.perTileStrideFromDims = 0;
    c.perTileDimStrides = nullptr;
    c.perTileDimWraps = nullptr;

    c.outDesc = OutputTileDescriptor();
    return success();
}

// ---------------------------------------------------------------------------
// emitShimBdOoo — OOO path: N per-tile shim BDs (orig 1175-1358).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitShimBdOoo(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    const int64_t numCoreTiles = c.numCoreTiles;
    const int64_t shimBdLen = c.shimBdLen;
    const int32_t dataId = c.dataId;
    ArrayAttr shimDimStrides = c.shimDimStrides;
    ArrayAttr shimDimWraps = c.shimDimWraps;

    // Compute per-tile DDR byte size for offset calculation
    int64_t perTileDdrBytes = shimBdLen / numCoreTiles;

    // Extract per-tile DDR offset stride from shimDimStrides.
    c.perTileStrideFromDims = perTileDdrBytes; // fallback: flat linear
    if (shimDimStrides && shimDimStrides.size() >= 3) {
        c.perTileStrideFromDims = cast<IntegerAttr>(shimDimStrides[2]).getInt();
    }

    // OOO iteration parameters.
    int64_t ddrRowStride = 0;
    int64_t perTileD1Rows = 0;
    if (shimDimStrides && shimDimStrides.size() >= 2) {
        ddrRowStride = cast<IntegerAttr>(shimDimStrides[1]).getInt();
    }
    if (shimDimWraps && shimDimWraps.size() >= 2) {
        perTileD1Rows = cast<IntegerAttr>(shimDimWraps[1]).getInt();
    }
    int64_t rowsPerRound = (c.ooNumIterations > 1) ? perTileD1Rows / c.ooNumIterations : perTileD1Rows;

    c.oooIterWrap = (int32_t)c.ooNumIterations;
    c.oooIterStepSize = (c.oooIterWrap > 1) ? (int32_t)(rowsPerRound * ddrRowStride) : 0;

    // Build per-tile addressing from shimDimStrides.
    {
        int64_t tileM = passState->tileM;
        int64_t tileRowsVal = passState->tileRows;

        auto moduleOpGuard = op->getParentOfType<ModuleOp>();
        // The generic GEMM 3D path fires when tile_m sub-tiles the M dimension under
        // fullconnect_auto. The conv2d width-split (spatial-halo) path drops
        // tile_m/tile_rows and fullconnect_auto, so it is triggered separately by the
        // presence of a width-split halo on a receiving (S2MM gather) shim channel.
        ConvHaloGeom convHalo = detectConvHalo(moduleOpGuard);
        bool convWidthSplit = convHalo.valid && !c.shimIsSender;
        bool gemm3D = tileM > 0 && tileM < tileRowsVal && !c.shimIsSender && isFullConnectAuto(moduleOpGuard);

        if (gemm3D || convWidthSplit) {
            auto moduleOp = op->getParentOfType<ModuleOp>();
            // Thread the partition #routing.tiling descriptor (if present) so the
            // NCHW conv-output BD geometry is derived from the authoritative tiling
            // attr rather than the detectConvHalo heuristic.
            mlir::Attribute tilingAttr;
            if (c.partExtractSlice)
                tilingAttr = c.partExtractSlice->getAttr("tiling");
            c.outDesc = buildOutputTileDescriptor(*passState, c.memrefType, numCoreTiles, moduleOp,
                                                  c.ooElementSizeBytes, tilingAttr);

            SmallVector<Attribute> strideAttrs, wrapAttrs;
            for (const auto &d : c.outDesc.bdDims) {
                strideAttrs.push_back(rewriter.getI32IntegerAttr(static_cast<int32_t>(d.first)));
                wrapAttrs.push_back(rewriter.getI32IntegerAttr(static_cast<int32_t>(d.second)));
            }
            c.perTileDimStrides = rewriter.getArrayAttr(strideAttrs);
            c.perTileDimWraps = rewriter.getArrayAttr(wrapAttrs);
            c.oooIterStepSize = c.outDesc.iterStep;
            c.oooIterWrap = c.outDesc.iterWrap;
            c.oooMRounds = c.outDesc.totalRounds;
            c.usedMRounds3D = true;

            // 3D channel-split (LtoR conv) overrides the per-tile DDR offset: the
            // gathered cores are mesh COLS interleaved on C, not contiguous blocks.
            if (c.outDesc.perTileStrideBytes > 0)
                c.perTileStrideFromDims = c.outDesc.perTileStrideBytes;

            llvm::errs() << "[OOO ShimBD desc] tileM=" << tileM << " tileRows=" << tileRowsVal
                         << " bdDims=" << c.outDesc.bdDims.size() << " roundDims=" << c.outDesc.roundDims.size()
                         << " iter_step=" << c.oooIterStepSize << " iter_wrap=" << c.oooIterWrap
                         << " totalRounds=" << c.oooMRounds << " bdLen=" << c.outDesc.bdLenBytes << "\n";
        } else if (shimDimStrides && shimDimWraps && shimDimStrides.size() >= 2) {
            SmallVector<Attribute> strides2d, wraps2d;
            strides2d.push_back(shimDimStrides[0]);
            wraps2d.push_back(shimDimWraps[0]);
            strides2d.push_back(shimDimStrides[1]);
            wraps2d.push_back(rewriter.getI32IntegerAttr(rowsPerRound));
            c.perTileDimStrides = rewriter.getArrayAttr(strides2d);
            c.perTileDimWraps = rewriter.getArrayAttr(wraps2d);
        }
    }

    if (c.usedMRounds3D) {
        c.perRoundBytes = c.outDesc.bdLenBytes;
        llvm::errs() << "[OOO ShimBD desc] perRoundBytes=" << c.perRoundBytes
                     << " oooIterStepSize=" << c.oooIterStepSize << " oooIterWrap=" << c.oooIterWrap
                     << " oooMRounds=" << c.oooMRounds << "\n";
    }

    // Allocate N shim BD IDs
    SmallVector<int32_t> shimBdIds;
    for (int64_t t = 0; t < numCoreTiles; t++) {
        int32_t bid = -1;
        if (resourceMgr) {
            auto bdOpt = resourceMgr->allocateTileBd(c.shimRow, c.shimCol, /*ownerId=*/c.flowIndex);
            if (bdOpt)
                bid = *bdOpt;
        }
        if (bid < 0)
            bid = c.shimBdIdVal + (int32_t)t; // fallback
        shimBdIds.push_back(bid);
    }
    c.shimPerTileBdIds = shimBdIds;

    // Create N shim BDs in reverse order.
    c.shimBdHandles.resize(numCoreTiles);
    for (int64_t t = numCoreTiles - 1; t >= 0; t--) {
        int32_t nextBdId = -1;
        int32_t thisBdId = shimBdIds[t];
        int64_t ddrOffset = t * c.perTileStrideFromDims;

        Value linkedBd = (t < numCoreTiles - 1) ? c.shimBdHandles[t + 1] : Value();

        auto shimBdIdC =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(thisBdId));

        auto ddrOffsetConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(static_cast<int32_t>(ddrOffset)));
        auto shimBd = rewriter.create<dfschedule::ConfigDmaBdOp>(
            loc, dfschedule::BdHandleType::get(rewriter.getContext()),
            c.ddrBuffer,                                             // DDR buffer
            c.shimTileOp.getTile(),                                  // tile
            shimBdIdC.getResult(),                                   // bd_id
            ddrOffsetConst.getResult(),                              // offset
            rewriter.getI32IntegerAttr(c.perRoundBytes),             // len
            rewriter.getBoolAttr(false),                             // enable_packet = false
            rewriter.getI32IntegerAttr(c.basePacketId + (int32_t)t), // packet_id
            rewriter.getI32IntegerAttr(nextBdId),                    // next_bd = -1
            rewriter.getI32IntegerAttr(-1),                          // acquire_lock_id = -1
            rewriter.getI32IntegerAttr(0),                           // acquire_lock_val
            rewriter.getI32IntegerAttr(-1),                          // release_lock_id = -1
            rewriter.getI32IntegerAttr(0),                           // release_lock_val
            rewriter.getI32IntegerAttr(dataId),                      // data_id
            linkedBd,                                                // linked_bd
            rewriter.getI32IntegerAttr(-1),                          // out_of_order_bd_id
            /*dim_strides=*/c.perTileDimStrides, /*dim_wraps=*/c.perTileDimWraps,
            rewriter.getI32IntegerAttr(c.oooIterStepSize), // iter_step_size
            rewriter.getI32IntegerAttr(c.oooIterWrap));    // iter_wrap

        c.shimBdHandles[t] = shimBd.getBdHandle();

        llvm::errs() << "[OOO ShimBD] tile " << t << " bd_id=" << thisBdId << " pkt_id=" << (c.basePacketId + t)
                     << " ddr_offset=" << ddrOffset << " len=" << c.perRoundBytes << " next_bd=" << nextBdId
                     << " iter_step=" << c.oooIterStepSize << " iter_wrap=" << c.oooIterWrap
                     << " d1_wrap=" << rowsPerRound << "\n";
    }
    c.lastShimBdHandle = c.shimBdHandles[0];
}

// ---------------------------------------------------------------------------
// emitShimBdNonOoo — non-OOO path: single shim BD + shim create_io
// (orig 1359-1544).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitShimBdNonOoo(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    MemRefType memrefType = c.memrefType;
    const int32_t dataId = c.dataId;

    // K-accum halo slab iteration.
    if (c.shimIsSender && c.kAccumHaloSlab) {
        int64_t bitWidthK = memrefType.getElementType().isIntOrFloat() ? memrefType.getElementTypeBitWidth() : 8;
        int64_t elemsPerWordK = bitWidthK > 0 ? 32 / bitWidthK : 4;
        constexpr int64_t wordBytesK = 4;
        if (elemsPerWordK > 0 && c.kAccumL2Rounds > 1 && c.kAccumL2Step > 0 && c.kAccumRowPitchElems > 0) {
            int64_t rowPitch_w = c.kAccumRowPitchElems / elemsPerWordK;
            c.shimIterStepSize = static_cast<int32_t>(c.kAccumL2Step * rowPitch_w * wordBytesK);
            c.shimIterWrap = static_cast<int32_t>(c.kAccumL2Rounds);
            llvm::errs() << "[BlueprintToSchedule] K-accum halo iter: iter_step_size=" << c.shimIterStepSize
                         << " iter_wrap=" << c.shimIterWrap << " l2_step=" << c.kAccumL2Step
                         << " row_pitch=" << c.kAccumRowPitchElems << " l2_rounds=" << c.kAccumL2Rounds << "\n";
        }
    }
    if (c.shimIsSender && !c.isHaloSlab) {
        auto moduleOp = op->getParentOfType<ModuleOp>();
        if (moduleOp) {
            // Tiling scalars from passState (entry-read #routing.tiling op, flat-attr
            // fallback). GEMM emits all three K scalars; conv leaves them 0, so the
            // effectiveK>0 guard skips this block exactly like the old presence check.
            int64_t effectiveK = passState->effectiveK;
            int64_t fullK = passState->fullK;
            int64_t kRounds = passState->kRounds;
            {
                if (effectiveK > 0 && effectiveK < fullK && kRounds > 1) {
                    int64_t elemBytes = 1;
                    if (memrefType.getElementType().isIntOrFloat())
                        elemBytes = memrefType.getElementTypeBitWidth() / 8;
                    if (elemBytes == 0)
                        elemBytes = 1;

                    bool nOuterPolicy = isNOuterPolicy(moduleOp);

                    if (dataId == 0) {
                        int64_t tileN = passState->tileN;
                        int64_t tileCols = passState->tileCols;

                        if (tileN > 0 && tileN < tileCols) {
                            if (nOuterPolicy) {
                                int64_t mRoundsIter = 1;
                                int64_t tM = passState->tileM;
                                int64_t tR = passState->tileRows;
                                if (tM > 0 && tM < tR)
                                    mRoundsIter = tR / tM;
                                c.shimIterStepSize = 0;
                                c.shimIterWrap = 0;
                                llvm::errs() << "[BlueprintToSchedule] Input B iter (n_outer, mRounds repeat): "
                                             << "iter_step_size=" << c.shimIterStepSize
                                             << " iter_wrap=" << c.shimIterWrap << " tileN=" << tileN
                                             << " tileCols=" << tileCols << " mRounds=" << mRoundsIter << "\n";
                            } else {
                                int64_t nRounds = tileCols / tileN;
                                c.shimIterStepSize = static_cast<int32_t>(tileN * fullK * elemBytes);
                                c.shimIterWrap = static_cast<int32_t>(nRounds);
                                llvm::errs()
                                    << "[BlueprintToSchedule] Input B iter (m_outer, nRounds): "
                                    << "iter_step_size=" << c.shimIterStepSize << " iter_wrap=" << c.shimIterWrap
                                    << " tileN=" << tileN << " tileCols=" << tileCols << " nRounds=" << nRounds << "\n";
                            }
                        } else {
                            c.shimIterStepSize = static_cast<int32_t>(effectiveK * elemBytes);
                            c.shimIterWrap = static_cast<int32_t>(kRounds);
                            llvm::errs() << "[BlueprintToSchedule] Input B iter (no N sub-tiling, kRounds): "
                                         << "iter_step_size=" << c.shimIterStepSize << " iter_wrap=" << c.shimIterWrap
                                         << "\n";
                        }
                    } else {
                        int64_t tileM = passState->tileM;
                        int64_t tileRows = passState->tileRows;

                        if (tileM > 0 && tileM < tileRows) {
                            if (nOuterPolicy) {
                                int64_t mRoundsA = tileRows / tileM;
                                c.shimIterStepSize = static_cast<int32_t>(tileM * fullK * elemBytes);
                                c.shimIterWrap = static_cast<int32_t>(mRoundsA);
                                llvm::errs() << "[BlueprintToSchedule] Input A iter (n_outer, mRounds advance): "
                                             << "iter_step_size=" << c.shimIterStepSize
                                             << " iter_wrap=" << c.shimIterWrap << " tileM=" << tileM
                                             << " tileRows=" << tileRows << " mRounds=" << mRoundsA << "\n";
                            } else {
                                int64_t nRounds2 = 1;
                                int64_t tN = passState->tileN;
                                int64_t tC = passState->tileCols;
                                if (tN > 0 && tN < tC)
                                    nRounds2 = tC / tN;
                                c.shimIterStepSize = 0;
                                c.shimIterWrap = 0;
                                llvm::errs() << "[BlueprintToSchedule] Input A iter (m_outer, nRounds repeat): "
                                             << "iter_step_size=" << c.shimIterStepSize
                                             << " iter_wrap=" << c.shimIterWrap << " tileM=" << tileM
                                             << " tileRows=" << tileRows << " nRounds=" << nRounds2 << "\n";
                            }
                        } else {
                            c.shimIterStepSize = static_cast<int32_t>(effectiveK * elemBytes);
                            c.shimIterWrap = static_cast<int32_t>(kRounds);
                            llvm::errs() << "[BlueprintToSchedule] Input A iter (2D, kRounds): "
                                         << "iter_step_size=" << c.shimIterStepSize << " iter_wrap=" << c.shimIterWrap
                                         << "\n";
                        }
                    }
                }
            }
        }
    }

    auto shimOffsetConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto shimBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc, dfschedule::BdHandleType::get(rewriter.getContext()),
        c.ddrBuffer,                                  // DDR receive buffer
        c.shimTileOp.getTile(),                       // tile
        c.shimBdIdConst.getResult(),                  // bd_id
        shimOffsetConst.getResult(),                  // offset
        rewriter.getI32IntegerAttr(c.perTileShimLen), // len (per-tile portion)
        rewriter.getBoolAttr(false),                  // enable_packet = false
        rewriter.getI32IntegerAttr(0),                // packet_id (unused)
        rewriter.getI32IntegerAttr(4294967295),       // next_bd = none
        rewriter.getI32IntegerAttr(0),                // acquire_lock_id = 0 (shim lock; -1 starves at 4096)
        rewriter.getI32IntegerAttr(0),                // acquire_lock_val
        rewriter.getI32IntegerAttr(0),                // release_lock_id = 0
        rewriter.getI32IntegerAttr(0),                // release_lock_val
        rewriter.getI32IntegerAttr(dataId),           // data_id
        Value(),                                      // linked_bd = none
        rewriter.getI32IntegerAttr(-1),               // out_of_order_bd_id
        /*dim_strides=*/c.shimDimStrides, /*dim_wraps=*/c.shimDimWraps,
        rewriter.getI32IntegerAttr(c.shimIterStepSize), // iter_step_size
        rewriter.getI32IntegerAttr(c.shimIterWrap));    // iter_wrap

    c.lastShimBdHandle = shimBdOp.getBdHandle();
    if (c.isManyToOne) {
        c.shimPerTileBdIds.push_back(c.shimBdIdVal);
    }
}

// ---------------------------------------------------------------------------
// classifyScheduleMode — sets needsOuterLoop/fullConnect + getBdIdOp
// (orig 2248-2319).
// ---------------------------------------------------------------------------
void FlowTransferConversion::classifyScheduleMode(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;

    // === Schedule emission: classify tiling mode ===
    auto moduleOp = op->getParentOfType<ModuleOp>();
    // Tiling scalars were read once at pass entry from the #routing.tiling op into
    // passState->tilingScalars (empty for conv); do not re-walk partitiontensor here.
    c.classification = classifyTiling(passState->tilingScalars);

    // Create dfschedule.schedule.getbdid for shim tile
    c.getBdIdOp = rewriter.create<dfschedule::GetBdIdOp>(loc, rewriter.getI32Type(), c.shimTileOp.getTile());

    bool nOuterPolicy = isNOuterPolicy(moduleOp);
    c.needsOuterLoop = false;
    if (nOuterPolicy) {
        c.needsOuterLoop = (c.classification.nMode == TilingMode::Multiple);
    } else {
        c.needsOuterLoop = (c.classification.mMode == TilingMode::Multiple);
    }

    c.fullConnect = true;
    if (moduleOp) {
        if (auto fcAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.fullconnect_auto"))
            c.fullConnect = (fcAttr.getInt() != 0);
    }

    c.haloL2RoundsForLoop = 0;
    c.haloKRoundsForLoop = 0;
    if (moduleOp && c.shimIsSender) {
        int haloArgIdxL2 = traceFlowConfigToFuncArgIndex(c.shimFlowConfig);
        if (haloArgIdxL2 >= 0) {
            std::string haloNameL2 = "tensor_" + std::to_string(haloArgIdxL2) + ".halo";
            if (auto haloDictL2 = moduleOp->getAttrOfType<DictionaryAttr>(haloNameL2)) {
                if (auto a = haloDictL2.getAs<IntegerAttr>("l2_rounds"))
                    c.haloL2RoundsForLoop = a.getInt();
                if (auto a = haloDictL2.getAs<IntegerAttr>("k_rounds"))
                    c.haloKRoundsForLoop = a.getInt();
            }
        }
    }
    if (c.haloL2RoundsForLoop > 1 || c.haloKRoundsForLoop > 1)
        c.needsOuterLoop = true;
    if (!c.fullConnect && c.kAccumHaloSlab) {
        if (c.needsOuterLoop)
            llvm::errs() << "[BlueprintToSchedule] nofullconnectauto K-accum halo: "
                         << "suppressing host scf.for (needsOuterLoop -> false)\n";
        c.needsOuterLoop = false;
    }

    if (!c.fullConnect && !c.isHaloSlab && c.haloL2RoundsForLoop <= 1 && c.haloKRoundsForLoop <= 1) {
        if (c.needsOuterLoop)
            llvm::errs() << "[BlueprintToSchedule] nofullconnectauto: suppressing host "
                         << "M×N round loop (needsOuterLoop -> false)\n";
        c.needsOuterLoop = false;
    }
}

// ---------------------------------------------------------------------------
// emitScheduleMultipleInput — Multiple mode, input flow unified scf.for
// (orig 2321-2634).
// ---------------------------------------------------------------------------
// Pure parameter computation for emitScheduleMultipleInput (no op emission).
void FlowTransferConversion::computeMultipleInputOffsetParams(FlowLoweringCtx &c, ModuleOp moduleOp, bool nOuterPolicy,
                                                              MultipleInputOffsetParams &p) const {
    MemRefType memrefType = c.memrefType;
    const int32_t dataId = c.dataId;

    int64_t outerRounds = nOuterPolicy ? c.classification.nRounds : c.classification.mRounds;
    if (c.haloL2RoundsForLoop > 1) {
        if (outerRounds <= 1)
            outerRounds = c.haloL2RoundsForLoop;
        else
            outerRounds *= c.haloL2RoundsForLoop;
    }
    if (c.haloKRoundsForLoop > 1)
        outerRounds *= c.haloKRoundsForLoop;
    p.outerRounds = outerRounds;
    llvm::errs() << "[BlueprintToSchedule] Multiple mode input flow (policy=" << (nOuterPolicy ? "n_outer" : "m_outer")
                 << "): "
                 << "outerRounds=" << outerRounds << ", emitting unified SCF loop from 0\n";

    // Compute per-iteration repeat.
    int32_t perIterRepeat = 1;
    if (c.isHaloSlab) {
        perIterRepeat = 1;
    } else if (moduleOp) {
        // Tiling scalars from passState (entry-read #routing.tiling op, flat-attr
        // fallback). A missing attr read as 0, matching the struct default.
        int64_t kRounds = passState->kRounds;
        if (kRounds > 1) {
            if (nOuterPolicy) {
                int64_t tM = passState->tileM;
                int64_t tR = passState->tileRows;
                if (tM > 0 && tM < tR) {
                    perIterRepeat = static_cast<int32_t>(tR / tM); // mRounds
                } else {
                    perIterRepeat = static_cast<int32_t>(kRounds); // fallback
                }
            } else {
                int64_t tN = passState->tileN;
                int64_t tC = passState->tileCols;
                if (tN > 0 && tN < tC) {
                    // Always repeat nRounds regardless of shimIterWrap: relying on the hw
                    // iter (perIterRepeat=1) starves the B-input stream at 4096 and deadlocks.
                    perIterRepeat = static_cast<int32_t>(tC / tN); // nRounds
                } else {
                    perIterRepeat = static_cast<int32_t>(kRounds); // fallback
                }
            }
        }
    }

    // Compute the byte stride for offset computation.
    int64_t elemBytes = 1;
    if (memrefType.getElementType().isIntOrFloat())
        elemBytes = memrefType.getElementTypeBitWidth() / 8;
    if (elemBytes == 0)
        elemBytes = 1;
    // Tiling scalars from passState (entry-read #routing.tiling op, flat-attr
    // fallback; 0 when absent, matching the old getAttrOfType default).
    int64_t tileMVal = passState->tileM;
    int64_t tileNVal = passState->tileN;
    int64_t fullKVal = passState->fullK;
    int64_t subTileStride = 0; // default: no offset advancement
    bool halo2D = false;
    int64_t haloHStrideBytes = 0;
    int64_t haloWStrideBytes = 0;
    int64_t haloWRoundsVal = 0;
    if (c.isHaloSlab) {
        int64_t haloStep = 0, haloRowPitch = 0, haloWStep = 0, haloWRounds = 0, haloSlice = 0, haloWSlice = 0;
        int haloArgIdx = traceFlowConfigToFuncArgIndex(c.shimFlowConfig);
        if (haloArgIdx >= 0) {
            std::string haloAttrName = "tensor_" + std::to_string(haloArgIdx) + ".halo";
            if (auto haloDict = moduleOp->getAttrOfType<DictionaryAttr>(haloAttrName)) {
                if (auto stepAttr = haloDict.getAs<IntegerAttr>("step"))
                    haloStep = stepAttr.getInt();
                if (auto a = haloDict.getAs<IntegerAttr>("row_pitch"))
                    haloRowPitch = a.getInt();
                if (auto a = haloDict.getAs<IntegerAttr>("w_step"))
                    haloWStep = a.getInt();
                if (auto a = haloDict.getAs<IntegerAttr>("w_rounds"))
                    haloWRounds = a.getInt();
                if (auto a = haloDict.getAs<IntegerAttr>("slice"))
                    haloSlice = a.getInt();
                if (auto a = haloDict.getAs<IntegerAttr>("w_slice"))
                    haloWSlice = a.getInt();
            }
        }
        // Row width = product of all dims after dim 0. Rank-generic: a 2D
        // flattened input [H, W*C] gives shape[1]; a genuine 3D input [H, W, C]
        // gives W*C — the same contiguous padded row pitch (e.g. 920).
        int64_t rawWc = 1;
        if (memrefType.getRank() >= 2)
            for (int64_t di = 1; di < memrefType.getRank(); ++di)
                rawWc *= memrefType.getShape()[di];
        else if (memrefType.getRank() == 1)
            rawWc = memrefType.getShape()[0];
        subTileStride = haloStep * rawWc * elemBytes;
        if (haloWRounds > 1 && haloRowPitch > 0 && haloSlice > 0 && haloWSlice > 0) {
            int64_t bufSize = 0;
            if (auto b = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size"))
                bufSize = b.getInt();
            int64_t chunkWidthElems = bufSize / haloSlice;
            int64_t convC = chunkWidthElems / haloWSlice;
            haloHStrideBytes = haloStep * haloRowPitch * elemBytes;
            haloWStrideBytes = haloWStep * convC * elemBytes;
            haloWRoundsVal = haloWRounds;
            halo2D = true;
            llvm::errs() << "[BlueprintToSchedule] 2D halo offset: w_rounds=" << haloWRounds
                         << " hStrideBytes=" << haloHStrideBytes << " (halo_step=" << haloStep
                         << "*row_pitch=" << haloRowPitch << ") wStrideBytes=" << haloWStrideBytes
                         << " (w_step=" << haloWStep << "*C=" << convC << ")\n";
        }
        if (!halo2D) {
            int64_t kStepB = 0, kRoundsB = 0, kSliceB = 0, l2StepB = 0;
            if (auto hd = moduleOp->getAttrOfType<DictionaryAttr>("tensor_" + std::to_string(haloArgIdx) + ".halo")) {
                if (auto a = hd.getAs<IntegerAttr>("k_step"))
                    kStepB = a.getInt();
                if (auto a = hd.getAs<IntegerAttr>("k_rounds"))
                    kRoundsB = a.getInt();
                if (auto a = hd.getAs<IntegerAttr>("k_slice"))
                    kSliceB = a.getInt();
                if (auto a = hd.getAs<IntegerAttr>("l2_step"))
                    l2StepB = a.getInt();
            }
            if (kRoundsB > 1 && kSliceB > 0 && l2StepB > 0) {
                haloHStrideBytes = l2StepB * rawWc * elemBytes;
                haloWStrideBytes = kStepB * elemBytes;
                haloWRoundsVal = kRoundsB;
                halo2D = true;
                llvm::errs() << "[BlueprintToSchedule] K-split 2D halo offset: k_rounds=" << kRoundsB
                             << " hStrideBytes=" << haloHStrideBytes << " (l2_step=" << l2StepB << "*raw_wc=" << rawWc
                             << ") wStrideBytes=" << haloWStrideBytes << " (k_step=" << kStepB << ")\n";
            }
        }
        if (haloWRounds <= 1 && !halo2D) {
            int64_t haloL2Step = 0;
            if (auto haloDict =
                    moduleOp->getAttrOfType<DictionaryAttr>("tensor_" + std::to_string(haloArgIdx) + ".halo")) {
                if (auto a = haloDict.getAs<IntegerAttr>("l2_step"))
                    haloL2Step = a.getInt();
            }
            if (c.haloL2RoundsForLoop > 1 && haloL2Step > 0) {
                subTileStride = haloL2Step * rawWc * elemBytes;
                llvm::errs() << "[BlueprintToSchedule] L2 halo offset: l2_rounds=" << c.haloL2RoundsForLoop
                             << " l2_step=" << haloL2Step << " subTileStride=" << subTileStride
                             << " (l2_step*raw_wc=" << rawWc << ")\n";
            }
        }
    } else if (!nOuterPolicy && dataId == 1) {
        subTileStride = tileMVal * fullKVal * elemBytes;
    } else if (nOuterPolicy && dataId == 0) {
        subTileStride = tileNVal * fullKVal * elemBytes;
    }

    p.perIterRepeat = perIterRepeat;
    p.subTileStride = subTileStride;
    p.halo2D = halo2D;
    p.haloHStrideBytes = haloHStrideBytes;
    p.haloWStrideBytes = haloWStrideBytes;
    p.haloWRoundsVal = haloWRoundsVal;
}

LogicalResult FlowTransferConversion::emitScheduleMultipleInput(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    const int32_t dataId = c.dataId;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    bool nOuterPolicy = isNOuterPolicy(moduleOp);

    MultipleInputOffsetParams p;
    computeMultipleInputOffsetParams(c, moduleOp, nOuterPolicy, p);
    const int64_t outerRounds = p.outerRounds;
    const int32_t perIterRepeat = p.perIterRepeat;
    const int64_t subTileStride = p.subTileStride;
    const bool halo2D = p.halo2D;
    const int64_t haloHStrideBytes = p.haloHStrideBytes;
    const int64_t haloWStrideBytes = p.haloWStrideBytes;
    const int64_t haloWRoundsVal = p.haloWRoundsVal;

    // Erase the initial createIoOp and shimBdOp created above.
    {
        Operation *shimBdDefOp = c.lastShimBdHandle ? c.lastShimBdHandle.getDefiningOp() : nullptr;
        rewriter.eraseOp(c.createIoOp);
        if (shimBdDefOp)
            rewriter.eraseOp(shimBdDefOp);
    }

    // 1. load_kernel_group OUTSIDE the loop
    auto loadKernelGroupOp = rewriter.create<dfschedule::LoadKernelGroupOp>(
        loc, dfschedule::KernelGroupType::get(rewriter.getContext()), c.coreTiles, rewriter.getArrayAttr(c.calleeAttrs),
        rewriter.getArrayAttr(c.computeKernelAttrs), nullptr, rewriter.getArrayAttr(c.kernelConfigSymbols));

    // 2. launch_kernel_group OUTSIDE the loop
    auto launchKernelGroupOp = rewriter.create<dfschedule::LaunchKernelGroupOp>(
        loc, dfschedule::EventType::get(rewriter.getContext()), loadKernelGroupOp.getKernelGroup());

    // 3. Core start_io OUTSIDE the loop
    SmallVector<Value> coreStartIoEvents;
    for (auto &deferred : c.deferredCoreStartIos) {
        auto coreStartIo = rewriter.create<dfschedule::StartIoOp>(
            loc, dfschedule::EventType::get(rewriter.getContext()), deferred.ioHandle, deferred.bdId,
            rewriter.getI32IntegerAttr(deferred.flowIdx), rewriter.getI32IntegerAttr(deferred.repeatCount));
        coreStartIoEvents.push_back(coreStartIo.getEvent());
    }

    // 4. Single scf.for from 0 to outerRounds (ALL iterations uniform)
    {
        auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto ub = rewriter.create<arith::ConstantIndexOp>(loc, outerRounds);
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar(); // index type

        auto ivI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), iv);

        Value offset;
        if (halo2D) {
            auto wRoundsC =
                rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(haloWRoundsVal), rewriter.getI32Type());
            auto hc = rewriter.create<arith::DivSIOp>(loc, ivI32, wRoundsC);
            auto wc = rewriter.create<arith::RemSIOp>(loc, ivI32, wRoundsC);
            auto hStrideC = rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(haloHStrideBytes),
                                                                  rewriter.getI32Type());
            auto wStrideC = rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(haloWStrideBytes),
                                                                  rewriter.getI32Type());
            auto hOff = rewriter.create<arith::MulIOp>(loc, hc, hStrideC);
            auto wOff = rewriter.create<arith::MulIOp>(loc, wc, wStrideC);
            offset = rewriter.create<arith::AddIOp>(loc, hOff, wOff);
        } else {
            auto strideConst =
                rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(subTileStride), rewriter.getI32Type());
            offset = rewriter.create<arith::MulIOp>(loc, ivI32, strideConst);
        }

        auto loopBd = rewriter.create<dfschedule::ConfigDmaBdOp>(
            loc, dfschedule::BdHandleType::get(rewriter.getContext()),
            c.ddrBuffer,                                  // buffer
            c.shimTileOp.getTile(),                       // tile
            c.shimBdIdConst.getResult(),                  // bd_id
            offset,                                       // offset (dynamic)
            rewriter.getI32IntegerAttr(c.perTileShimLen), // len
            rewriter.getBoolAttr(false),                  // enable_packet
            rewriter.getI32IntegerAttr(0),                // packet_id
            rewriter.getI32IntegerAttr(4294967295),       // next_bd = none
            rewriter.getI32IntegerAttr(0),                // acquire_lock_id = 0 (shim lock; -1 starves at 4096)
            rewriter.getI32IntegerAttr(0),                // acquire_lock_val
            rewriter.getI32IntegerAttr(0),                // release_lock_id = 0
            rewriter.getI32IntegerAttr(0),                // release_lock_val
            rewriter.getI32IntegerAttr(dataId),           // data_id
            Value(),                                      // linked_bd = none
            rewriter.getI32IntegerAttr(-1),               // out_of_order_bd_id
            c.shimDimStrides, c.shimDimWraps,
            rewriter.getI32IntegerAttr(c.shimIterStepSize), // iter_step_size (K-round)
            rewriter.getI32IntegerAttr(c.shimIterWrap));    // iter_wrap (kRounds)

        auto loopCreateIo = rewriter.create<dfschedule::ConfigCreateIoOp>(
            loc, dfschedule::IoHandleType::get(rewriter.getContext()), loopBd.getBdHandle(), c.shimTileOp.getTile(),
            rewriter.getI32IntegerAttr(c.shimChannel), rewriter.getStringAttr(c.dmaDirection),
            rewriter.getStringAttr(c.ioOperation), rewriter.getBoolAttr(false));

        auto loopGetBdId = rewriter.create<dfschedule::GetBdIdOp>(loc, rewriter.getI32Type(), c.shimTileOp.getTile());
        auto loopStartIo = rewriter.create<dfschedule::StartIoOp>(
            loc, dfschedule::EventType::get(rewriter.getContext()), loopCreateIo.getIoHandle(), loopGetBdId.getBdId(),
            rewriter.getI32IntegerAttr(c.flowIndex), rewriter.getI32IntegerAttr(perIterRepeat));

        SmallVector<Value> loopWaitEvents;
        loopWaitEvents.push_back(loopStartIo.getEvent());
        rewriter.create<dfschedule::ScheduleWaitOp>(loc, loopWaitEvents);

        rewriter.setInsertionPointAfter(forOp);
    }

    // 5. After all iterations: wait for kernel launch event
    SmallVector<Value> finalEvents;
    finalEvents.push_back(launchKernelGroupOp.getEvent());
    rewriter.create<dfschedule::ScheduleWaitOp>(loc, finalEvents);
    return success();
}

// ---------------------------------------------------------------------------
// emitScheduleOooOutput — OOO output flow with outer_rounds>1 scf.for
// (orig 2636-2821).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitScheduleOooOutput(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    const int64_t numCoreTiles = c.numCoreTiles;
    const int32_t dataId = c.dataId;

    llvm::errs() << "[BlueprintToSchedule] OOO output outerRounds loop: "
                 << "outerRounds=" << c.oooMRounds << " roundDims=" << c.outDesc.roundDims.size()
                 << " numCoreTiles=" << numCoreTiles << " iter_wrap=" << c.oooIterWrap << "\n";

    // Erase the initial createIoOp and N shim BDs created above.
    {
        rewriter.eraseOp(c.createIoOp);
        for (int64_t t = 0; t < numCoreTiles; t++) {
            if (c.shimBdHandles[t]) {
                Operation *bdOp = c.shimBdHandles[t].getDefiningOp();
                if (bdOp) {
                    SmallVector<Operation *, 2> deadConsts;
                    for (Value operand : bdOp->getOperands()) {
                        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
                            if (constOp->hasOneUse())
                                deadConsts.push_back(constOp);
                        }
                    }
                    rewriter.eraseOp(bdOp);
                    for (auto *dc : deadConsts)
                        rewriter.eraseOp(dc);
                }
            }
        }
    }

    // 1. load_kernel_group OUTSIDE the loop
    auto loadKernelGroupOp = rewriter.create<dfschedule::LoadKernelGroupOp>(
        loc, dfschedule::KernelGroupType::get(rewriter.getContext()), c.coreTiles, rewriter.getArrayAttr(c.calleeAttrs),
        rewriter.getArrayAttr(c.computeKernelAttrs), nullptr, rewriter.getArrayAttr(c.kernelConfigSymbols));

    // 2. launch_kernel_group OUTSIDE the loop
    auto launchKernelGroupOp = rewriter.create<dfschedule::LaunchKernelGroupOp>(
        loc, dfschedule::EventType::get(rewriter.getContext()), loadKernelGroupOp.getKernelGroup());

    // 3. Core start_io OUTSIDE the loop
    SmallVector<Value> coreStartIoEvents;
    for (auto &deferred : c.deferredCoreStartIos) {
        auto coreStartIo = rewriter.create<dfschedule::StartIoOp>(
            loc, dfschedule::EventType::get(rewriter.getContext()), deferred.ioHandle, deferred.bdId,
            rewriter.getI32IntegerAttr(deferred.flowIdx), rewriter.getI32IntegerAttr(deferred.repeatCount));
        coreStartIoEvents.push_back(coreStartIo.getEvent());
    }

    // 4. scf.for from 0 to oooMRounds.
    int32_t perIterRepeat = static_cast<int32_t>(numCoreTiles * c.oooIterWrap);
    {
        auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto ub = rewriter.create<arith::ConstantIndexOp>(loc, c.oooMRounds);
        auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar(); // index type

        auto ivI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), iv);

        Value mBaseOffset;
        {
            const auto &rd = c.outDesc.roundDims;
            size_t nDims = rd.size();
            SmallVector<Value> comps(nDims);
            for (size_t i = 0; i < nDims; i++) {
                int64_t innerProd = 1;
                for (size_t j = i + 1; j < nDims; j++)
                    innerProd *= rd[j].first;
                Value comp = ivI32;
                if (innerProd > 1) {
                    auto divC = rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(innerProd),
                                                                      rewriter.getI32Type());
                    comp = rewriter.create<arith::DivSIOp>(loc, comp, divC);
                }
                if (i != 0) {
                    auto wrapC = rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(rd[i].first),
                                                                       rewriter.getI32Type());
                    comp = rewriter.create<arith::RemSIOp>(loc, comp, wrapC);
                }
                comps[i] = comp;
            }
            SmallVector<Value> offs(nDims);
            for (size_t i = 0; i < nDims; i++) {
                auto strideC = rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(rd[i].second),
                                                                     rewriter.getI32Type());
                offs[i] = rewriter.create<arith::MulIOp>(loc, comps[i], strideC);
            }
            mBaseOffset = offs[0];
            for (size_t i = 1; i < nDims; i++)
                mBaseOffset = rewriter.create<arith::AddIOp>(loc, mBaseOffset, offs[i]);
        }

        // Re-configure N OOO shim BDs with updated DDR offset
        SmallVector<Value> loopBdHandles(numCoreTiles);
        for (int64_t t = numCoreTiles - 1; t >= 0; t--) {
            int32_t thisBdId = c.shimPerTileBdIds[t];
            int64_t perTileOffset = t * c.perTileStrideFromDims;

            auto bdIdConst = rewriter.create<arith::ConstantIntOp>(loc, thisBdId, rewriter.getI32Type());
            auto perTileOffsetConst =
                rewriter.create<arith::ConstantIntOp>(loc, static_cast<int32_t>(perTileOffset), rewriter.getI32Type());
            auto totalOffset = rewriter.create<arith::AddIOp>(loc, mBaseOffset, perTileOffsetConst.getResult());

            Value linkedBd = (t < numCoreTiles - 1) ? loopBdHandles[t + 1] : Value();

            auto loopBd = rewriter.create<dfschedule::ConfigDmaBdOp>(
                loc, dfschedule::BdHandleType::get(rewriter.getContext()),
                c.ddrBuffer,                                             // DDR buffer
                c.shimTileOp.getTile(),                                  // tile
                bdIdConst.getResult(),                                   // bd_id (reused)
                totalOffset.getResult(),                                 // offset (dynamic)
                rewriter.getI32IntegerAttr(c.perRoundBytes),             // len (one d0×d1 block)
                rewriter.getBoolAttr(false),                             // enable_packet = false
                rewriter.getI32IntegerAttr(c.basePacketId + (int32_t)t), // packet_id (debug)
                rewriter.getI32IntegerAttr(-1),                          // next_bd = -1
                rewriter.getI32IntegerAttr(-1),                          // acquire_lock_id = -1
                rewriter.getI32IntegerAttr(0),                           // acquire_lock_val
                rewriter.getI32IntegerAttr(-1),                          // release_lock_id = -1
                rewriter.getI32IntegerAttr(0),                           // release_lock_val
                rewriter.getI32IntegerAttr(dataId),                      // data_id
                linkedBd,                                                // linked_bd
                rewriter.getI32IntegerAttr(-1),                          // out_of_order_bd_id
                /*dim_strides=*/c.perTileDimStrides, /*dim_wraps=*/c.perTileDimWraps,
                rewriter.getI32IntegerAttr(c.oooIterStepSize), // iter_step_size
                rewriter.getI32IntegerAttr(c.oooIterWrap));    // iter_wrap

            loopBdHandles[t] = loopBd.getBdHandle();
        }

        auto loopCreateIo = rewriter.create<dfschedule::ConfigCreateIoOp>(
            loc, dfschedule::IoHandleType::get(rewriter.getContext()), loopBdHandles[0], c.shimTileOp.getTile(),
            rewriter.getI32IntegerAttr(c.shimChannel), rewriter.getStringAttr(c.dmaDirection),
            rewriter.getStringAttr(c.ioOperation),
            rewriter.getBoolAttr(true)); // enable_out_of_order = true

        auto loopGetBdId = rewriter.create<dfschedule::GetBdIdOp>(loc, rewriter.getI32Type(), c.shimTileOp.getTile());
        auto loopStartIo = rewriter.create<dfschedule::StartIoOp>(
            loc, dfschedule::EventType::get(rewriter.getContext()), loopCreateIo.getIoHandle(), loopGetBdId.getBdId(),
            rewriter.getI32IntegerAttr(c.flowIndex), rewriter.getI32IntegerAttr(perIterRepeat));

        SmallVector<Value> loopWaitEvents;
        loopWaitEvents.push_back(loopStartIo.getEvent());
        rewriter.create<dfschedule::ScheduleWaitOp>(loc, loopWaitEvents);

        rewriter.setInsertionPointAfter(forOp);
    }

    // 5. After all iterations: wait for kernel launch event
    SmallVector<Value> finalEvents;
    finalEvents.push_back(launchKernelGroupOp.getEvent());
    rewriter.create<dfschedule::ScheduleWaitOp>(loc, finalEvents);
}

// ---------------------------------------------------------------------------
// emitScheduleStraightLine — Match mode / output straight-line schedule
// (orig 2823-2942).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitScheduleStraightLine(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    bool nOuterPolicy = isNOuterPolicy(moduleOp);
    const int64_t numCoreTiles = c.numCoreTiles;

    int32_t repeatCount = 1;
    if (c.kAccumHaloSlab) {
        repeatCount = (c.kAccumL2Rounds > 1) ? static_cast<int32_t>(c.kAccumL2Rounds) : 1;
    } else if (c.isHaloSlab) {
        repeatCount = 1;
    } else if (c.useOOO) {
        repeatCount = (int32_t)(numCoreTiles * c.ooNumIterations);
    } else if (c.shimIsSender) {
        if (moduleOp) {
            // Tiling scalars from passState (entry-read #routing.tiling op, flat-attr
            // fallback). A missing attr read as 0, matching the struct default.
            int64_t kRounds = passState->kRounds;
            if (kRounds > 1) {
                if (nOuterPolicy) {
                    int64_t tM = passState->tileM;
                    int64_t tR = passState->tileRows;
                    if (tM > 0 && tM < tR) {
                        repeatCount = static_cast<int32_t>(tR / tM); // mRounds
                    } else {
                        repeatCount = static_cast<int32_t>(kRounds);
                    }
                } else {
                    int64_t tN = passState->tileN;
                    int64_t tC = passState->tileCols;
                    if (tN > 0 && tN < tC) {
                        repeatCount = static_cast<int32_t>(tC / tN); // nRounds
                    } else {
                        repeatCount = static_cast<int32_t>(kRounds);
                    }
                }
            }
        }
    }
    if (!c.fullConnect && c.shimIsSender && !c.isHaloSlab && !c.kAccumHaloSlab) {
        if (repeatCount != 1)
            llvm::errs() << "[BlueprintToSchedule] nofullconnectauto: forcing shim BD "
                         << "repeatCount " << repeatCount << " -> 1\n";
        repeatCount = 1;
    }
    auto loadKernelGroupOp = rewriter.create<dfschedule::LoadKernelGroupOp>(
        loc, dfschedule::KernelGroupType::get(rewriter.getContext()), c.coreTiles, rewriter.getArrayAttr(c.calleeAttrs),
        rewriter.getArrayAttr(c.computeKernelAttrs), nullptr, rewriter.getArrayAttr(c.kernelConfigSymbols));

    auto launchKernelGroupOp = rewriter.create<dfschedule::LaunchKernelGroupOp>(
        loc, dfschedule::EventType::get(rewriter.getContext()), loadKernelGroupOp.getKernelGroup());

    // Core S2MM receivers must be armed before shim MM2S pushes data, else the shim
    // stalls on STALL_STREAM (downstream stream switch has no credit).
    llvm::errs() << "[emitScheduleStraightLine] flowIdx=" << c.flowIndex << " shimIsSender=" << c.shimIsSender
                 << " deferredCoreStartIos.size()=" << c.deferredCoreStartIos.size() << "\n";
    SmallVector<Value> coreStartIoEvents;
    for (auto &deferred : c.deferredCoreStartIos) {
        auto coreStartIo = rewriter.create<dfschedule::StartIoOp>(
            loc, dfschedule::EventType::get(rewriter.getContext()), deferred.ioHandle, deferred.bdId,
            rewriter.getI32IntegerAttr(deferred.flowIdx), rewriter.getI32IntegerAttr(deferred.repeatCount));
        coreStartIoEvents.push_back(coreStartIo.getEvent());
    }

    // Shim MM2S startIo AFTER core S2MM receivers are armed.
    auto startIoOp = rewriter.create<dfschedule::StartIoOp>(
        loc, dfschedule::EventType::get(rewriter.getContext()), c.createIoOp.getIoHandle(), c.getBdIdOp.getBdId(),
        rewriter.getI32IntegerAttr(c.flowIndex), rewriter.getI32IntegerAttr(repeatCount));

    // Determine whether to bypass input sending IO wait.
    bool bypassInputIoWait = false;
    if (c.shimIsSender) {
        if (c.funcArgIdx == 0 && c.classification.mMode == TilingMode::Match) {
            bypassInputIoWait = true;
            llvm::errs() << "[BlueprintToSchedule] Bypassing input IO wait for input A "
                         << "(funcArgIdx=0, M-dimension Match: tile_m == tile_rows)\n";
        } else if (c.funcArgIdx == 1 && c.classification.nMode == TilingMode::Match) {
            bypassInputIoWait = true;
            llvm::errs() << "[BlueprintToSchedule] Bypassing input IO wait for input B "
                         << "(funcArgIdx=1, N-dimension Match: tile_n == tile_cols)\n";
        }
    }

    // Wait for kernel launch + shim IO events
    SmallVector<Value> events;
    events.push_back(launchKernelGroupOp.getEvent());
    if (!bypassInputIoWait)
        events.push_back(startIoOp.getEvent());
    rewriter.create<dfschedule::ScheduleWaitOp>(loc, events);
}

} // namespace blueprint_sched
