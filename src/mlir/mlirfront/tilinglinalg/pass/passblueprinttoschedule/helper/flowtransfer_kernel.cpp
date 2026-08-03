/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// KERNEL group of the split FlowTransferConversion — core-tile config.
//
// Pure mechanical extraction of the former single matchAndRewrite core-tile
// section (orig lines 1546-2246). Op-creation ordering is byte-identical to the
// original; all cross-boundary locals are carried in FlowLoweringCtx (`c`) and
// per-core-tile locals in CoreTileCtx (`t`).

#include "flowtransfer_internal.h"

namespace blueprint_sched {

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

// ---------------------------------------------------------------------------
// emitCoreTileConfigs — core-tile loop scaffold + per-tile params/BD dispatch
// (orig 1546-2210) + coreTiles.empty() early handling is left to the caller.
// ---------------------------------------------------------------------------
LogicalResult FlowTransferConversion::emitCoreTileConfigs(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;

    // --- CORE TILES: declaretile, kernel_config, and ping-pong DMA config ---
    c.coreTilesAttr = c.coreTileGroup.getTiles();

    // Get DMA channel and direction from core FlowConfig
    auto coreDmaAttr = c.coreFlowConfig.getDma();
    auto coreDmaChannels = coreDmaAttr.getChannels();
    c.coreChannel = coreDmaChannels.empty() ? 0 : coreDmaChannels[0];
    c.coreDmaDir = coreDmaAttr.getDirection();
    c.coreDmaDirection = (c.coreDmaDir == dfscheblueprint::bp_direction::MM2S) ? "MM2S" : "S2MM";
    c.coreIoOperation = (c.coreDmaDir == dfscheblueprint::bp_direction::MM2S) ? "SEND" : "RECV";

    // Get per-tile data slices from core FlowConfig's slice_symbols
    c.sliceSymbolsOpt = c.coreFlowConfig.getSliceSymbols();

    // Pre-allocate buffer addresses once for this flow (shared kernel binary)
    // All tiles use the same buffer placement.
    c.flowPingL1Offset = 0;
    c.flowPongL1Offset = 0;
    c.flowAddrsValid = false;

    // Compute dirIdx (funcArgIndex-based) before the tile loop so it's
    // available for both lock allocation and buffer naming.
    // This ensures the host lock IDs match the kernel's window ordering.
    c.isInput = c.shimIsSender;
    c.funcArgIdx = traceFlowConfigToFuncArgIndex(c.shimFlowConfig);
    if (c.isInput) {
        auto it = dataIdToInputIdx.find(c.dataId);
        if (it != dataIdToInputIdx.end()) {
            c.dirIdx = it->second;
        } else {
            c.dirIdx = (c.funcArgIdx >= 0) ? c.funcArgIdx : nextInputIdx;
            dataIdToInputIdx[c.dataId] = c.dirIdx;
            nextInputIdx = std::max(nextInputIdx, c.dirIdx + 1);
        }
    } else {
        auto it = dataIdToOutputIdx.find(c.dataId);
        if (it != dataIdToOutputIdx.end()) {
            c.dirIdx = it->second;
        } else {
            c.dirIdx = nextOutputIdx++;
            dataIdToOutputIdx[c.dataId] = c.dirIdx;
        }
    }

    // Collect tile config dictionaries for kernel_config
    // Deferred core StartIoOp data: collect IO handles and BD IDs inside the
    // per-tile loop, then emit StartIoOp AFTER LoadKernelGroup/LaunchKernelGroup
    // so that ELF BSS initialization does not overwrite DMA data.
    c.tileIndex = 0;

    for (auto tileAttr : c.coreTilesAttr) {
        auto tileArray = dyn_cast<ArrayAttr>(tileAttr);
        if (!tileArray || tileArray.size() < 2) {
            continue;
        }

        CoreTileCtx t;
        t.col = cast<IntegerAttr>(tileArray[0]).getInt();
        t.row = cast<IntegerAttr>(tileArray[1]).getInt();

        // Create dfschedule.declaretile for each core tile
        t.coreTileOp = rewriter.create<dfschedule::DeclareTileOp>(loc, dfschedule::TileType::get(rewriter.getContext()),
                                                                  rewriter.getI32IntegerAttr(t.col),
                                                                  rewriter.getI32IntegerAttr(t.row));
        c.coreTiles.push_back(t.coreTileOp.getTile());

        if (failed(emitCoreTileParams(c, t)))
            return failure();

        if (failed(emitCoreBufferDma(c, t)))
            return failure();

        c.tileIndex++;
    }

    return success();
}

// ---------------------------------------------------------------------------
// emitCoreTileParams — per-tile size/round/lock computation + config dict
// (orig 1622-1866).
// ---------------------------------------------------------------------------
LogicalResult FlowTransferConversion::emitCoreTileParams(FlowLoweringCtx &c, CoreTileCtx &t) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    dfscheblueprint::FlowTransferOp op = c.op;

    // Calculate buffer size from the DDR memref type
    t.bufferSize = c.bufferLen;
    t.elementSizeBytes = 1;
    if (c.memrefType.getElementType().isIntOrFloat())
        t.elementSizeBytes = c.memrefType.getElementTypeBitWidth() / 8;
    if (t.elementSizeBytes == 0)
        t.elementSizeBytes = 1;
    t.bufferSize *= t.elementSizeBytes;

    // With circuit-switched broadcast every core tile receives the FULL
    // partition data from the shim stream.  The kernel selects its own
    // portion via buffer_offset.  DMA iterations must cover the full
    // partition so the stream is fully consumed.
    t.perTileSize = t.bufferSize / c.coreTilesAttr.size();
    t.bufferOffset = c.tileIndex * t.perTileSize;

    // Per-core data depends on transfer type:
    // many_to_one (gather/output): each core produces partition / numCoreTiles.
    // one_to_many (broadcast/input): each core receives the full partition.
    t.fullPartitionElements = t.bufferSize / t.elementSizeBytes;
    t.perCoreElements = t.fullPartitionElements;
    // group2 conv OUTPUT (channel-split): the partition VIEW is ALREADY channel-
    // split (Part 1, e.g. [28,112,16]), so bufferLen == ONE core's full output
    // (one channel group). Do NOT re-divide by numCoreTiles: the other cores' data
    // is a *different* channel group, not part of this view. The rank-3 conv-halo
    // many_to_one gather is the exclusive signal.
    bool group2ConvOutput = false;
    {
        auto moduleOpG2 = op->getParentOfType<ModuleOp>();
        if (c.transferType == "many_to_one" && c.memrefType.getRank() == 3 && detectConvHalo(moduleOpG2).valid)
            group2ConvOutput = true;
    }
    if (c.transferType == "many_to_one" && !group2ConvOutput)
        t.perCoreElements = t.fullPartitionElements / c.numCoreTiles;
    // K-round adjustment: for input flows with kRounds > 1, the kernel
    // operates on per-k-round data (tile_rows * effectiveK), not the
    // full partition (tile_rows * fullK). We must divide perCoreElements
    // by kRounds so that pingPongBufferSize matches the kernel's
    // buf_sz_a/b (= rowsPerRound * effectiveK). The numIterations
    // multiplication by kRounds (below) then correctly recovers the
    // total number of DMA rounds across all k-rounds.
    t.perCorePerKRound = t.perCoreElements;
    if (c.isInput) {
        int64_t kRounds = passState->kRounds;
        if (kRounds > 1) {
            t.perCorePerKRound = t.perCoreElements / kRounds;
            // When tile_m < tileRows, each k-round only needs
            // tile_m rows (not partRows). Divide by mRounds
            // so pingPongBufferSize = tile_m * effectiveK.
            int64_t tileM = passState->tileM;
            int64_t tileRows = passState->tileRows;
            if (tileM > 0 && tileM < tileRows) {
                int64_t mRounds = tileRows / tileM;
                t.perCorePerKRound = t.perCorePerKRound / mRounds;
            }
        }
    } else {
        // Output: the kernel produces one sub-tile (tile_m × tile_n_sub)
        // per release_output_window call. Divide perCoreElements by
        // mRounds * nRounds so BD len matches one kernel output window.
        auto moduleOp = op->getParentOfType<ModuleOp>();
        if (moduleOp) {
            int64_t tileM = passState->tileM;
            int64_t tileRows = passState->tileRows;
            int64_t tileN = passState->tileN;
            int64_t tileCols = passState->tileCols;
            int64_t mRounds = (tileM > 0 && tileM < tileRows) ? (tileRows / tileM) : 1;
            int64_t nRounds = (tileN > 0 && tileN < tileCols) ? (tileCols / tileN) : 1;
            int64_t outDivisor = mRounds * nRounds;
            // Spatial-halo conv: the tile_m/tile_rows/n-round attrs are DROPPED
            // when fullconnect_auto=0 (that policy only governs INPUT re-send), so
            // the divisor above collapses to 1 and the output BD would stay at the
            // full per-core partition and get clamped to maxPingPongBytes (=4096)
            // — wrong. The kernel still emits one [oh_per_row*ow_t, tile_n] slab
            // per on-core round, so honor the authoritative per-slab round count
            // carried in "routing.spatial_out_rounds" (= spatialMRounds*spatialNRounds).
            if (auto outRoundsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds")) {
                int64_t outRounds = outRoundsAttr.getInt();
                if (outRounds > outDivisor)
                    outDivisor = outRounds;
            }
            if (outDivisor > 1) {
                t.perCorePerKRound = t.perCoreElements / outDivisor;
            }
        }
    }

    // Read pp_depth from FlowConfigOp attribute (set by dmaphop→blueprint pass).
    // pp_depth controls physical ping-pong buffer count (for DMA/compute
    // overlap), NOT data splitting.  Buffer size = full per-k-round data,
    // clamped only by maxPingPongBytes when the data exceeds tile memory.
    t.ppDepth = static_cast<int>(1.0 / bufferRatio + 0.5); // e.g. bufferRatio=0.5 → ppDepth=2
    if (c.coreFlowConfig.getPpDepth())
        t.ppDepth = static_cast<int>(*c.coreFlowConfig.getPpDepth());
    if (t.ppDepth <= 0)
        t.ppDepth = 2;

    t.pingPongBufferSize = t.perCorePerKRound;
    if (t.pingPongBufferSize <= 0)
        t.pingPongBufferSize = 1;

    // Spatial-halo conv IFM override: the kernel needs the WHOLE contiguous
    // halo slab (halo_slice * raw_wc) in one window buffer for on-chip
    // im2col. The host-side buffer address/size allocation must match the
    // kernel-side window allocation (BUF_SZ_IN), so use the authoritative
    // slab size carried via "routing.spatial_halo_buf_size" and identify
    // the IFM port by the per-tensor "tensor_N.halo" attr (N = funcArgIdx).
    // This bypasses the maxPingPong clamp (slab must stay contiguous) — kept
    // in lockstep with passblueprinttoschedulekernel.cpp.
    t.hostSpatialHaloPort = false;
    if (c.isInput && c.funcArgIdx >= 0) {
        auto moduleOpHalo = op->getParentOfType<ModuleOp>();
        if (moduleOpHalo) {
            std::string haloAttrName = "tensor_" + std::to_string(c.funcArgIdx) + ".halo";
            auto haloDict = moduleOpHalo->getAttrOfType<DictionaryAttr>(haloAttrName);
            auto haloBufAttr = moduleOpHalo->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size");
            if (haloDict && haloBufAttr && haloBufAttr.getInt() > 0) {
                t.pingPongBufferSize = haloBufAttr.getInt() / t.elementSizeBytes;
                if (t.pingPongBufferSize <= 0)
                    t.pingPongBufferSize = 1;
                // Single contiguous slab transfer per kernel invocation:
                // one DMA round delivers the whole slab (numIterations==1).
                t.perCorePerKRound = t.pingPongBufferSize;
                t.hostSpatialHaloPort = true;
            }
        }
    }

    // K-contraction split slab: the core S2MM receives one per-round slab of
    // l2_slice*k_slice elements (= l2CorePerRoundLen bytes, set by the shim-send
    // block above). The ping-pong buffer must hold exactly one slab; pp_depth
    // stays >=2 so the BD chain re-arms across all l2_rounds*k_rounds rounds.
    // Gated on l2CorePerRoundLen>0 (driven by tensor_N.halo k_rounds), distinct
    // from the single-slab hostSpatialHaloPort path (spatial_halo_buf_size).
    t.kSplitSlabPort = false;
    if (c.isInput && c.l2CorePerRoundLen > 0) {
        int64_t slabElems = c.l2CorePerRoundLen / (t.elementSizeBytes > 0 ? t.elementSizeBytes : 1);
        if (slabElems > 0) {
            t.pingPongBufferSize = slabElems;
            t.perCorePerKRound = slabElems;
            t.kSplitSlabPort = true;
        }
    }

    if (failed(computeCoreIterations(c, t)))
        return failure();

    // Compute lock IDs from dirIdx to match the kernel's window ordering.
    // The kernel allocates locks sequentially per sorted window:
    //   input0 → lock 0/1, input1 → lock 2/3, output0 → lock 4/5, ...
    // (kernel adds LOCK_BASE=48 offset internally).
    // For outputs, offset by numInputs * 2 (we use nextInputIdx as
    // an estimate of numInputs since inputs are processed first).
    if (c.isInput) {
        t.acquireLockId = c.dirIdx * 2;
        t.releaseLockId = c.dirIdx * 2 + 1;
    } else {
        // Output locks start after all input locks.
        // nextInputIdx tracks the highest input dirIdx+1 seen so far.
        int outputLockBase = nextInputIdx * 2;
        t.acquireLockId = outputLockBase + c.dirIdx * 2;
        t.releaseLockId = outputLockBase + c.dirIdx * 2 + 1;
    }

    // Build config dictionary for this tile
    // buffer_mode: 0 = single buffer (pp_depth=1), 1 = ping-pong (pp_depth>=2)
    int bufferMode = (t.ppDepth == 1) ? 0 : 1;
    int numBuffers = (t.ppDepth == 1) ? 1 : 2;

    NamedAttrList configAttrs;
    configAttrs.append("tile_index", rewriter.getI32IntegerAttr(c.tileIndex));
    configAttrs.append("flow_index", rewriter.getI32IntegerAttr(c.flowIndex));
    configAttrs.append("packet_id", rewriter.getI32IntegerAttr(c.basePacketId + c.tileIndex));
    configAttrs.append("dma_channel", rewriter.getI32IntegerAttr(c.coreChannel));
    configAttrs.append("buffer_mode", rewriter.getI32IntegerAttr(bufferMode));
    configAttrs.append("num_buffers", rewriter.getI32IntegerAttr(numBuffers));
    configAttrs.append("buffer_size", rewriter.getI32IntegerAttr(t.pingPongBufferSize));
    configAttrs.append("num_iterations", rewriter.getI32IntegerAttr(t.numIterations));
    configAttrs.append("buffer_offset", rewriter.getI32IntegerAttr(t.bufferOffset));
    configAttrs.append("element_size", rewriter.getI32IntegerAttr(t.elementSizeBytes));
    configAttrs.append("acquire_lock_id", rewriter.getI32IntegerAttr(t.acquireLockId));
    configAttrs.append("release_lock_id", rewriter.getI32IntegerAttr(t.releaseLockId));

    c.tileConfigDicts.push_back(rewriter.getDictionaryAttr(configAttrs));

    return success();
}

// Sub-helper of emitCoreTileParams: buffer clamp + numIterations (halo /
// K-round / K-split) + pp_depth=1 validation. Pure computation into CoreTileCtx.
LogicalResult FlowTransferConversion::computeCoreIterations(FlowLoweringCtx &c, CoreTileCtx &t) const {
    dfscheblueprint::FlowTransferOp op = c.op;

    // Clamp to maxPingPongBytes to prevent exceeding core tile memory.
    // Skip the clamp for the spatial-halo IFM slab (must stay contiguous) and
    // for the K-split per-round slab (must match the shim per-round len exactly).
    if (!t.hostSpatialHaloPort && !t.kSplitSlabPort && maxPingPongBytes > 0 && t.elementSizeBytes > 0) {
        int64_t maxElements = maxPingPongBytes / t.elementSizeBytes;
        if (maxElements > 0 && t.pingPongBufferSize > maxElements)
            t.pingPongBufferSize = maxElements;
    }
    // numIterations: use perCorePerKRound (per-k-round data) so that
    // base iterations count one k-round. The kRounds multiplier below
    // then scales to the total across all k-rounds.
    t.numIterations = (t.perCorePerKRound + t.pingPongBufferSize - 1) / t.pingPongBufferSize;

    // Validate: pp_depth=1 requires numIterations==1.
    // With single-buffer mode (no BD chaining, no next_bd cycling),
    // the DMA fires exactly once. If the kernel needs multiple
    // rounds (numIterations > 1), the DMA cannot re-arm and the
    // kernel will deadlock on the second acquire_window call.
    if (t.ppDepth == 1 && t.numIterations > 1) {
        op.emitError("pp_depth=1 (single buffer) incompatible with "
                     "numIterations=")
            << t.numIterations << " (buffer_size=" << t.pingPongBufferSize
            << " < perCorePerKRound=" << t.perCorePerKRound
            << "). Single-buffer DMA cannot re-arm for multiple rounds. "
               "Set pp_depth>=2 to enable ping-pong BD chaining, or "
               "increase max_buffer_bytes to fit the full per-k-round data.";
        return failure();
    }

    // K-round multiplication: when effectiveK < K, the kernel runs
    // kRounds iterations, each consuming numIterations DMA rounds.
    // The host must send numIterations * kRounds total BD iterations
    // for input flows to match the kernel's acquire/release pattern.
    if (c.isInput) {
        int64_t kRounds = passState->kRounds;
        if (kRounds > 1) {
            llvm::errs() << "[BlueprintToSchedule] K-round: input numIterations " << t.numIterations << " * kRounds "
                         << kRounds << " = " << t.numIterations * kRounds << "\n";
            t.numIterations *= kRounds;
        }
    }

    // K-contraction split (tensor_N.halo): module-level routing.k_rounds is
    // absent in this config, so derive the total on-core round count directly
    // from the per-tensor halo dict. The core S2MM BD chain must re-arm
    // l2_rounds*k_rounds times (= 16) to match the shim's outerRounds (A2) and
    // the kernel's numRounds (C1). pp_depth>=2 supplies the BD chaining.
    if (c.isInput && c.l2CorePerRoundLen > 0) {
        if (auto hd = op->getParentOfType<ModuleOp>()->getAttrOfType<DictionaryAttr>(
                "tensor_" + std::to_string(c.funcArgIdx) + ".halo")) {
            auto l2A = hd.getAs<IntegerAttr>("l2_rounds");
            auto kA = hd.getAs<IntegerAttr>("k_rounds");
            int64_t l2r = l2A ? l2A.getInt() : 1, kr = kA ? kA.getInt() : 1;
            int64_t total = (l2r > 0 ? l2r : 1) * (kr > 0 ? kr : 1);
            if (total > 1) {
                llvm::errs() << "[BlueprintToSchedule] K-split: core numIterations = l2_rounds " << l2r
                             << " * k_rounds " << kr << " = " << total << "\n";
                t.numIterations = total; // 16
            }
        }
    }

    return success();
}

// ---------------------------------------------------------------------------
// emitCoreBufferDma — per-tile subview/mapping + BD dispatch (orig 1868-2205).
// ---------------------------------------------------------------------------
LogicalResult FlowTransferConversion::emitCoreBufferDma(FlowLoweringCtx &c, CoreTileCtx &t) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;
    dfscheblueprint::FlowTransferOp op = c.op;

    // --- Core tile ping-pong DMA BD configuration ---
    // Look up per-tile data slice from slice_symbols (maps 1:1 to tiles)
    llvm::errs() << "[DeferredStartIo] emitCoreBufferDma: flowIdx=" << c.flowIndex << " tileIdx=" << c.tileIndex
                 << " sliceSymbolsOpt=" << (c.sliceSymbolsOpt ? "yes" : "null")
                 << " sliceSize=" << (c.sliceSymbolsOpt ? (int)c.sliceSymbolsOpt->size() : -1)
                 << " flowRootMemref=" << (c.flowRootMemref ? "valid" : "null") << "\n";
    if (c.sliceSymbolsOpt && c.tileIndex < (int)c.sliceSymbolsOpt->size()) {
        auto sliceSymRef = cast<SymbolRefAttr>((*c.sliceSymbolsOpt)[c.tileIndex]);
        auto dataSliceOp = lookupDataSlice(op.getOperation(), sliceSymRef);
        llvm::errs() << "[DeferredStartIo] dataSliceOp=" << (dataSliceOp ? "found" : "null") << "\n";
        if (dataSliceOp) {
            Value perTileTensor = dataSliceOp.getTensorSlice();
            Type perTileType = perTileTensor.getType();
            (void)perTileType;

            // Trace back to tensor.extract_slice to get per-tile offsets/sizes
            auto tileExtractSlice = perTileTensor.getDefiningOp<tensor::ExtractSliceOp>();
            llvm::errs() << "[DeferredStartIo] tileExtractSlice=" << (tileExtractSlice ? "yes" : "null") << "\n";

            int64_t perTileTotalSize = t.perTileSize / t.elementSizeBytes;
            (void)perTileTotalSize;

            if (c.flowRootMemref && tileExtractSlice) {
                // Get slice info from tileExtractSlice
                auto sliceOffsets = tileExtractSlice.getStaticOffsets();
                auto sliceSizes = tileExtractSlice.getStaticSizes();
                auto sliceStrides = tileExtractSlice.getStaticStrides();

                Value partSubview;
                SmallVector<int64_t> perTileSizes;
                SmallVector<int64_t> perTileOffsets;
                SmallVector<int64_t> perTileStrides(sliceSizes.size(), 1);

                if (c.partExtractSlice) {
                    // Original path: tileExtractSlice gives per-tile offsets relative
                    // to partition subview (output flow has distinct extract_slices per tile)
                    perTileSizes.assign(sliceSizes.begin(), sliceSizes.end());
                    perTileOffsets.assign(sliceOffsets.begin(), sliceOffsets.end());
                    partSubview = c.partitionSubview;
                } else {
                    // Path 2: tileExtractSlice gives partition-level offsets from root.
                    // Create partition subview first, then compute per-tile split.
                    auto partSubviewOp = rewriter.create<memref::SubViewOp>(
                        loc, c.flowRootMemref, toOpFoldResult(sliceOffsets, rewriter),
                        toOpFoldResult(sliceSizes, rewriter), toOpFoldResult(sliceStrides, rewriter));
                    partSubview = partSubviewOp.getResult();

                    // Split first dimension evenly among core tiles
                    int64_t numCoreTiles = c.coreTilesAttr.size();
                    perTileSizes.assign(sliceSizes.begin(), sliceSizes.end());
                    perTileSizes[0] = sliceSizes[0] / numCoreTiles;
                    perTileOffsets.assign(sliceSizes.size(), 0);
                    perTileOffsets[0] = c.tileIndex * perTileSizes[0];
                }

                // Capture the #routing.tiling descriptor (if any) from the source
                // partition/tile extract_slice so it can be carried onto the core
                // subview + mapping. Copied generically as mlir::Attribute (no
                // routing::TilingAttr C++ type / include / regen needed). Gated below:
                // only set when the source actually has tiling → non-halo (matmul)
                // path stays byte-identical.
                mlir::Attribute tilingAttr;
                if (tileExtractSlice)
                    tilingAttr = tileExtractSlice->getAttr("tiling");
                if (!tilingAttr && c.partExtractSlice)
                    tilingAttr = c.partExtractSlice->getAttr("tiling");

                // Create per-tile subview
                auto tileSubviewOp = rewriter.create<memref::SubViewOp>(
                    loc, partSubview, toOpFoldResult(perTileOffsets, rewriter), toOpFoldResult(perTileSizes, rewriter),
                    toOpFoldResult(perTileStrides, rewriter));
                if (tilingAttr)
                    tileSubviewOp->setAttr("tiling", tilingAttr);

                // Annotate the per-tile subview with the DMA accumulation descriptor so the
                // subview shape (e.g. 64x256) self-explains the per-round dma_bd len
                // (tileM*effectiveK, e.g. 16x64). Additive/informational only; gated so
                // non-accumulating subviews stay byte-identical.
                int64_t accRoundRows = passState->tileM;
                int64_t accRoundCols = passState->effectiveK;
                dfschedule::AccumAttr accumAttr; // null unless this is an accumulating tile
                if (accRoundRows > 0 && accRoundCols > 0 && perTileSizes.size() == 2 &&
                    perTileSizes[0] % accRoundRows == 0 && perTileSizes[1] % accRoundCols == 0) {
                    int64_t d1 = perTileSizes[0] / accRoundRows;
                    int64_t d2 = perTileSizes[1] / accRoundCols;
                    if (d1 > 1 || d2 > 1) {
                        accumAttr =
                            dfschedule::AccumAttr::get(rewriter.getContext(), accRoundRows, accRoundCols, d1, d2);
                        tileSubviewOp->setAttr("accumulate", accumAttr);
                    }
                }

                // memref_mapping: strip strides, produce clean shaped type
                t.shapedPerTileType = MemRefType::get(perTileSizes, passState->elementType);
                perTileTotalSize = 1;
                for (int64_t d : perTileSizes)
                    perTileTotalSize *= d;

                auto coreMappingOp =
                    rewriter.create<dfschedule::MemRefMappingOp>(loc, t.shapedPerTileType, tileSubviewOp.getResult());
                // Propagate the accumulation descriptor onto the consumer mapping op so
                // the info survives even if the subview is later folded/elided.
                if (accumAttr)
                    coreMappingOp->setAttr("accumulate", accumAttr);
                // Carry the tiling descriptor onto the consumer mapping op too, so
                // the info survives even if the subview is later folded/elided.
                if (tilingAttr)
                    coreMappingOp->setAttr("tiling", tilingAttr);
                t.perTileToken = coreMappingOp.getMapped();

                // bind_core_buffer with shaped memref type
                // Use pre-allocated buffer addresses from CoreMemAllocator
                // (allocated once for first tile, reused for all tiles in this flow)
                emitCoreBufferAlloc(c, t);
                // Core BD len in bytes: runtime passes len directly to
                // XAie_DmaSetAddrLen, so compute the total byte count here.
                t.coreBdLen = t.pingPongBufferSize * t.elementSizeBytes;
                // L2-halo input flow: the shim ships one per-round L2 slab
                // (l2_slice*raw_wc bytes) per scf.for iteration, so the core
                // S2MM BD len must match that slab — not the clamped ping/pong
                // ceiling. Gated to L2-halo inputs only (0 otherwise → no change).
                if (c.l2CorePerRoundLen > 0 && c.isInput)
                    t.coreBdLen = c.l2CorePerRoundLen;
                if (t.coreBdLen <= 0)
                    t.coreBdLen = 1;

                // For output (MM2S on core), swap lock IDs in BD config:
                // Input (S2MM):  DMA acquires lock 0 (buffer free), releases lock 1 (data ready)
                // Output (MM2S): DMA acquires lock 1 (data produced by kernel), releases lock 0 (buffer free)
                t.isOutputFlow = (c.coreDmaDir == dfscheblueprint::bp_direction::MM2S);
                t.bdAcquireLockId = t.isOutputFlow ? t.releaseLockId : t.acquireLockId;
                t.bdReleaseLockId = t.isOutputFlow ? t.acquireLockId : t.releaseLockId;

                // Output (MM2S) uses packet-switched routing: core DMA must emit
                // packet headers so the packet switch can route data to the shim.
                // Input (S2MM) uses circuit-switched routing: no packet headers needed.
                t.coreBdEnablePacket = t.isOutputFlow;
                t.coreBdPacketId = t.isOutputFlow ? (int32_t)(c.basePacketId + c.tileIndex) : 0;

                // Compute out_of_order_bd_id for output (MM2S) core BDs.
                // This tells the shim S2MM DMA which BD to use for this tile's data.
                t.coreOooBdId = -1;
                if (t.isOutputFlow && !c.shimPerTileBdIds.empty()) {
                    size_t idx = static_cast<size_t>(c.tileIndex);
                    if (idx < c.shimPerTileBdIds.size())
                        t.coreOooBdId = c.shimPerTileBdIds[idx];
                }

                if (t.ppDepth == 1) {
                    emitCoreSingleBufferBd(c, t);
                } else {
                    emitCorePingPongBd(c, t);
                }

                // Create IO handle for core tile
                auto coreCreateIoOp = rewriter.create<dfschedule::ConfigCreateIoOp>(
                    loc, dfschedule::IoHandleType::get(rewriter.getContext()), t.firstCoreBdHandle,
                    t.coreTileOp.getTile(), rewriter.getI32IntegerAttr(c.coreChannel),
                    rewriter.getStringAttr(c.coreDmaDirection), rewriter.getStringAttr(c.coreIoOperation),
                    rewriter.getBoolAttr(false)); // enable_out_of_order=false for core tiles
                auto coreBdIdOp =
                    rewriter.create<dfschedule::GetBdIdOp>(loc, rewriter.getI32Type(), t.coreTileOp.getTile());
                // Defer core StartIoOp until after ELF is loaded (LoadKernelGroup)
                // to prevent BSS initialization from overwriting DMA data.
                // Core tiles use ping-pong BD chaining (next_bd links ping↔pong),
                // so the DMA hardware automatically re-arms via the chain.
                // repeat=1 is sufficient; the BD chain does the work.
                int32_t coreRepeat = 1;
                llvm::errs() << "[DeferredStartIo] PUSH deferredCoreStartIos flowIdx=" << c.flowIndex
                             << " tileIdx=" << c.tileIndex << " total=" << (c.deferredCoreStartIos.size() + 1) << "\n";
                c.deferredCoreStartIos.push_back(
                    {coreCreateIoOp.getIoHandle(), coreBdIdOp.getBdId(), c.flowIndex, coreRepeat});
            } // end if (passState && ...)
        }
    }

    return success();
}

// Sub-helper of emitCoreBufferDma: allocate ping/pong buffers via
// CoreMemAllocator on the first tile of a flow, then assign per-tile L1
// offsets. Pure bookkeeping into FlowLoweringCtx/CoreTileCtx (no op emission).
void FlowTransferConversion::emitCoreBufferAlloc(FlowLoweringCtx &c, CoreTileCtx &t) const {
    t.pingL1Offset = 0;
    // Pong offset must be int32-aligned for DMA transfers
    int64_t pingBufBytes = t.pingPongBufferSize * t.elementSizeBytes;
    t.pongL1Offset = (pingBufBytes + 3) & ~3;
    if (!c.flowAddrsValid) {
        // First tile: allocate addresses for this flow.
        // dirIdx, isInput, and funcArgIdx are computed before
        // the tile loop to ensure consistent ordering.
        llvm::errs() << "[HostParamMapping] data_id=" << c.dataId << " funcArgIdx=" << c.funcArgIdx
                     << " isInput=" << c.isInput << " dirIdx=" << c.dirIdx << "\n";
        std::string pingName, pongName;
        if (c.isInput) {
            pingName = "buf_in_ping_" + std::to_string(c.dirIdx);
            pongName = "buf_in_pong_" + std::to_string(c.dirIdx);
        } else {
            pingName = "buf_out_ping_" + std::to_string(c.dirIdx);
            pongName = "buf_out_pong_" + std::to_string(c.dirIdx);
        }
        // Ensure buffer size is int32-aligned for DMA.
        // Use perCorePerKRound (per-k-round data for one core) as the
        // allocation size.  The kernel uses a single global BUF_SZ for
        // ALL windows, determined by the largest flow (input one_to_many).
        // Output buffers (many_to_one) need the same allocation size
        // even though they transfer fewer elements per core.
        int64_t kernelBufElements = t.perCorePerKRound;
        // Clamp kernelBufElements to maxPingPongBytes (same as pingPongBufferSize clamping)
        if (maxPingPongBytes > 0 && t.elementSizeBytes > 0) {
            int64_t maxElements = maxPingPongBytes / t.elementSizeBytes;
            if (maxElements > 0 && kernelBufElements > maxElements)
                kernelBufElements = maxElements;
        }
        if (kernelBufElements < t.pingPongBufferSize)
            kernelBufElements = t.pingPongBufferSize;
        uint32_t bufSizeBytes = ((kernelBufElements * t.elementSizeBytes) + 3) & ~3;
        try {
            auto &allocator = ResourceMgr::instance()->coreMemAllocator();
            auto pingAddr = allocator.allocate(pingName, bufSizeBytes, /*alignment=*/32);
            if (t.ppDepth == 1) {
                // Single buffer mode: only allocate ping buffer, no pong
                if (pingAddr) {
                    c.flowPingL1Offset = static_cast<int64_t>(*pingAddr) - 0x70000;
                    c.flowPongL1Offset = c.flowPingL1Offset; // unused but set for safety
                    c.flowAddrsValid = true;
                }
            } else {
                auto pongAddr = allocator.allocate(pongName, bufSizeBytes, /*alignment=*/32);
                if (pingAddr && pongAddr) {
                    // Convert core processor view (0x78000+) to DMA view (0x08000+)
                    // Core DMA engine sees memory starting at 0x00000, not 0x70000
                    c.flowPingL1Offset = static_cast<int64_t>(*pingAddr) - 0x70000;
                    c.flowPongL1Offset = static_cast<int64_t>(*pongAddr) - 0x70000;
                    c.flowAddrsValid = true;
                }
            }
        } catch (...) {
            // ResourceMgr singleton not initialized; fall back to relative offsets
        }
    }
    if (c.flowAddrsValid) {
        t.pingL1Offset = c.flowPingL1Offset;
        t.pongL1Offset = c.flowPongL1Offset;
    }
}

// ---------------------------------------------------------------------------
// emitCoreSingleBufferBd — single-buffer path (orig 2073-2111).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitCoreSingleBufferBd(FlowLoweringCtx &c, CoreTileCtx &t) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;

    // === Single buffer mode (pp_depth=1) ===
    // One buffer, one BD, no next_bd chaining, no pong.
    auto singleL1 = rewriter.create<dfschedule::BindCoreBufferOp>(
        loc, t.shapedPerTileType, t.perTileToken, t.coreTileOp.getTile(), rewriter.getI64IntegerAttr(t.pingL1Offset));

    int32_t singleBdId = -1;
    if (resourceMgr) {
        auto bd0 = resourceMgr->allocateTileBd(t.row, t.col, /*ownerId=*/c.flowIndex);
        if (bd0)
            singleBdId = *bd0;
    }
    if (singleBdId < 0)
        singleBdId = 0;

    auto singleBdIdConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(singleBdId));
    auto singleOffsetConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto singleBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc, dfschedule::BdHandleType::get(rewriter.getContext()), singleL1.getBuffer(), t.coreTileOp.getTile(),
        singleBdIdConst.getResult(),
        singleOffsetConst.getResult(),                 // offset
        rewriter.getI32IntegerAttr(t.coreBdLen),       // len (bytes)
        rewriter.getBoolAttr(t.coreBdEnablePacket),    // enable_packet
        rewriter.getI32IntegerAttr(t.coreBdPacketId),  // packet_id
        rewriter.getI32IntegerAttr(-1),                // next_bd = -1 (no chaining)
        rewriter.getI32IntegerAttr(t.bdAcquireLockId), // acquire_lock_id
        rewriter.getI32IntegerAttr(-1),                // acquire_lock_val
        rewriter.getI32IntegerAttr(t.bdReleaseLockId), // release_lock_id
        rewriter.getI32IntegerAttr(1),                 // release_lock_val
        rewriter.getI32IntegerAttr(-1),                // data_id
        Value(),                                       // linked_bd = none
        rewriter.getI32IntegerAttr(t.coreOooBdId),     // out_of_order_bd_id
        /*dim_strides=*/nullptr, /*dim_wraps=*/nullptr,
        rewriter.getI32IntegerAttr(0),  // iter_step_size (no iteration)
        rewriter.getI32IntegerAttr(0)); // iter_wrap (no iteration)

    t.firstCoreBdHandle = singleBdOp.getBdHandle();
}

// ---------------------------------------------------------------------------
// emitCorePingPongBd — ping-pong path (orig 2113-2187).
// ---------------------------------------------------------------------------
void FlowTransferConversion::emitCorePingPongBd(FlowLoweringCtx &c, CoreTileCtx &t) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;

    // === Ping-pong mode (pp_depth>=2, existing behavior) ===
    auto pingL1 = rewriter.create<dfschedule::BindCoreBufferOp>(
        loc, t.shapedPerTileType, t.perTileToken, t.coreTileOp.getTile(), rewriter.getI64IntegerAttr(t.pingL1Offset));
    auto pongL1 = rewriter.create<dfschedule::BindCoreBufferOp>(
        loc, t.shapedPerTileType, t.perTileToken, t.coreTileOp.getTile(), rewriter.getI64IntegerAttr(t.pongL1Offset));

    // Allocate BD IDs from ResourceMgr per-tile pool
    int32_t pingBdId = -1, pongBdId = -1;
    if (resourceMgr) {
        auto bd0 = resourceMgr->allocateTileBd(t.row, t.col, /*ownerId=*/c.flowIndex);
        auto bd1 = resourceMgr->allocateTileBd(t.row, t.col, /*ownerId=*/c.flowIndex);
        if (bd0 && bd1) {
            pingBdId = *bd0;
            pongBdId = *bd1;
        }
    }
    if (pingBdId < 0 || pongBdId < 0) {
        llvm::errs() << "WARNING: BD allocation failed for tile (" << t.col << "," << t.row
                     << "), falling back to 0/1\n";
        pingBdId = 0;
        pongBdId = 1;
    }

    // Pong BD first (no linked_bd): next_bd -> ping
    auto pongBdIdConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(pongBdId));
    auto pongOffsetConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto pongBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc, dfschedule::BdHandleType::get(rewriter.getContext()), pongL1.getBuffer(), t.coreTileOp.getTile(),
        pongBdIdConst.getResult(),
        pongOffsetConst.getResult(),                   // offset
        rewriter.getI32IntegerAttr(t.coreBdLen),       // len (bytes)
        rewriter.getBoolAttr(t.coreBdEnablePacket),    // enable_packet
        rewriter.getI32IntegerAttr(t.coreBdPacketId),  // packet_id
        rewriter.getI32IntegerAttr(pingBdId),          // next_bd -> ping
        rewriter.getI32IntegerAttr(t.bdAcquireLockId), // acquire_lock_id
        rewriter.getI32IntegerAttr(-1),                // acquire_lock_val
        rewriter.getI32IntegerAttr(t.bdReleaseLockId), // release_lock_id
        rewriter.getI32IntegerAttr(1),                 // release_lock_val
        rewriter.getI32IntegerAttr(-1),                // data_id
        Value(),                                       // linked_bd = none
        rewriter.getI32IntegerAttr(t.coreOooBdId),     // out_of_order_bd_id
        /*dim_strides=*/nullptr, /*dim_wraps=*/nullptr,
        rewriter.getI32IntegerAttr(0),  // iter_step_size (no iteration)
        rewriter.getI32IntegerAttr(0)); // iter_wrap (no iteration)

    // Ping BD second (linked_bd = pong handle): next_bd -> pong
    auto pingBdIdConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(pingBdId));
    auto pingOffsetConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto pingBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
        loc, dfschedule::BdHandleType::get(rewriter.getContext()), pingL1.getBuffer(), t.coreTileOp.getTile(),
        pingBdIdConst.getResult(),
        pingOffsetConst.getResult(),                   // offset
        rewriter.getI32IntegerAttr(t.coreBdLen),       // len (bytes)
        rewriter.getBoolAttr(t.coreBdEnablePacket),    // enable_packet
        rewriter.getI32IntegerAttr(t.coreBdPacketId),  // packet_id
        rewriter.getI32IntegerAttr(pongBdId),          // next_bd -> pong
        rewriter.getI32IntegerAttr(t.bdAcquireLockId), // acquire_lock_id
        rewriter.getI32IntegerAttr(-1),                // acquire_lock_val
        rewriter.getI32IntegerAttr(t.bdReleaseLockId), // release_lock_id
        rewriter.getI32IntegerAttr(1),                 // release_lock_val
        rewriter.getI32IntegerAttr(-1),                // data_id
        pongBdOp.getBdHandle(),                        // linked_bd = pong BD
        rewriter.getI32IntegerAttr(t.coreOooBdId),     // out_of_order_bd_id
        /*dim_strides=*/nullptr, /*dim_wraps=*/nullptr,
        rewriter.getI32IntegerAttr(0),  // iter_step_size (no iteration)
        rewriter.getI32IntegerAttr(0)); // iter_wrap (no iteration)

    t.firstCoreBdHandle = pingBdOp.getBdHandle();
}

// ---------------------------------------------------------------------------
// finalizeKernelConfig — DeclareKernelConfigOp + callee/compute attrs
// (orig 2217-2246). Preserves the function-local static kernelConfigIdx.
// ---------------------------------------------------------------------------
void FlowTransferConversion::finalizeKernelConfig(FlowLoweringCtx &c) const {
    ConversionPatternRewriter &rewriter = c.rewriter;
    Location loc = c.loc;

    // Create individual kernel_config ops for each tile (e.g., @kernelconfig0, @kernelconfig1)
    // Use static counter to ensure unique names across multiple transfer manifests
    static int kernelConfigIdx = 0;
    for (size_t i = 0; i < c.tileConfigDicts.size(); ++i) {
        std::string configName = "kernelconfig" + std::to_string(kernelConfigIdx++);

        // Create a kernel_config op with a single tile's config
        SmallVector<Attribute> singleTileConfig;
        singleTileConfig.push_back(c.tileConfigDicts[i]);

        auto kernelConfigOp = rewriter.create<dfschedule::DeclareKernelConfigOp>(
            loc, dfschedule::KernelConfigType::get(rewriter.getContext()), rewriter.getStringAttr(configName),
            rewriter.getArrayAttr(singleTileConfig));
        (void)kernelConfigOp;

        // Store symbol reference
        c.kernelConfigSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), configName));
    }

    // Create callee symbol refs (dskernel_receiver for all)
    c.calleeAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "dskernel_receiver"));

    // Create distributed_compute_kernel_args (compute0 for all)
    for (size_t i = 0; i < c.coreTiles.size(); ++i) {
        c.computeKernelAttrs.push_back(SymbolRefAttr::get(rewriter.getContext(), "compute0"));
    }
}

} // namespace blueprint_sched
