/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// Shared utilities for the split FlowTransferConversion implementation.
//
// These were formerly file-local `static` helpers at the top of
// passblueprinttoschedule.cpp. They are promoted to external linkage (inside
// namespace blueprint_sched to avoid ODR clashes with the identically-named
// statics in sibling passes) so the host/kernel helper TUs can share them.
//
// Bodies are moved verbatim: only the `static` keyword is removed and the
// namespace wrapper added. No behavioral change.

#include "flowtransfer_internal.h"
#include "routingmanager.h"
#include <cassert>

namespace blueprint_sched {

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

// Trace a Value back through the SSA chain to find the originating function
// argument index.  This walks through bufferization.to_tensor,
// routing.routingcreatescheduletensor, routing.partitiontensor,
// routing.routingextract_data, dfscheblueprint.declare_data, and
// scf.execute_region captures until it reaches a BlockArgument of a func::FuncOp.
// Returns the argument index (0-based), or -1 if the chain cannot be resolved.
int traceToFuncArgIndex(Value v) {
    // Walk up the def chain, max 20 hops to avoid infinite loops
    for (int depth = 0; depth < 20; ++depth) {
        // If v is a block argument of a func op, we found it
        if (auto blockArg = dyn_cast<BlockArgument>(v)) {
            if (auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp()))
                return static_cast<int>(blockArg.getArgNumber());
            break; // can't follow further through a block argument
        }
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
            break;

        // routing ops that forward data through operand 0
        if (defOp->getName().getStringRef() == "routing.routingextract_data" ||
            defOp->getName().getStringRef() == "routing.routingcreatescheduletensor" ||
            defOp->getName().getStringRef() == "routing.partitiontensor") {
            v = defOp->getOperand(0);
            continue;
        }
        // bufferization.to_tensor %memref -> follow %memref (operand 0)
        if (defOp->getName().getStringRef() == "bufferization.to_tensor") {
            v = defOp->getOperand(0);
            continue;
        }
        // dfscheblueprint.declare_data -> follow init_tensor (operand 0)
        if (defOp->getName().getStringRef() == "dfscheblueprint.declare_data") {
            v = defOp->getOperand(0);
            continue;
        }
        // tensor.extract_slice -> follow source tensor (operand 0)
        if (defOp->getName().getStringRef() == "tensor.extract_slice") {
            v = defOp->getOperand(0);
            continue;
        }
        // Generic single-result ops that just forward operand 0
        if (defOp->getNumOperands() > 0) {
            v = defOp->getOperand(0);
            continue;
        }
        break;
    }
    return -1; // unable to resolve
}

// Trace from a FlowConfigOp's view value back through the SSA chain to find
// the originating function argument index.
int traceFlowConfigToFuncArgIndex(dfscheblueprint::FlowConfigOp flowConfig) {
    Value viewValue = flowConfig.getView();
    if (!viewValue)
        return -1;
    return traceToFuncArgIndex(viewValue);
}

// Helper function to look up TileGroupOp by symbol reference (wrapper for backward compatibility)
dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::TileGroupOp>(rootOp, target);
}

// Helper function to check if dskernel_receiver already exists in the module
bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName) {
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
Operation *getModuleOp(Operation *rootOp) {
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }
    return moduleOp;
}

// Generate a kernel symbol declaration only (dfschedule.dskernel_receiver with empty body).
// Details (kernel module, buffers, locks, etc.) are filled in by a separate pass.
// Parameters:
//   - kernelName: symbol name for the kernel (same as load_kernel_group callee)
//   - insertBeforeOp: used to find module and insertion point
void generateDSKernelReceiver(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                              StringRef kernelName, RankedTensorType tensorType, int64_t bufferLen,
                              uint32_t basePacketId, int64_t coreChannel, uint32_t flowIndex) {
    (void)tensorType;
    (void)bufferLen;
    (void)basePacketId;
    (void)coreChannel;
    (void)flowIndex;

    Operation *rootModuleOp = getModuleOp(insertBeforeOp);
    Block &moduleBlock = rootModuleOp->getRegions().front().front();
    if (!moduleBlock.empty() && moduleBlock.back().hasTrait<OpTrait::IsTerminator>())
        rewriter.setInsertionPoint(&moduleBlock.back());
    else
        rewriter.setInsertionPointToEnd(&moduleBlock);

    auto receiverOp = rewriter.create<dfschedule::DSKernelReceiverOp>(loc, kernelName);
    receiverOp.getBody().emplaceBlock();
}

// Helper function to look up FlowConfigOp by symbol reference (wrapper for backward compatibility)
dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::FlowConfigOp>(rootOp, target);
}

dfscheblueprint::DataSliceOp lookupDataSlice(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::DataSliceOp>(rootOp, target);
}

SmallVector<OpFoldResult> toOpFoldResult(ArrayRef<int64_t> values, OpBuilder &b) {
    SmallVector<OpFoldResult> result;
    for (int64_t v : values)
        result.push_back(b.getI64IntegerAttr(v));
    return result;
}

// Pre-processing: lower arith.constant dense tensors to memrefs via
// bufferization.to_memref, keeping everything inside @main with no module-level
// memref.global.  Processes ALL DeclareDataOps so each data tensor (input A,
// input B, output C, etc.) gets its own backing memref.
LogicalResult preprocessConstantToMemref(Operation *topLevel, std::shared_ptr<BlueprintPassState> state) {
    func::FuncOp mainFunc = nullptr;
    topLevel->walk([&](func::FuncOp f) {
        if (f.getName() == "main" || f.getName() == "host_canonicalized")
            mainFunc = f;
    });
    if (!mainFunc)
        return success();

    // Collect ALL DeclareDataOps
    SmallVector<dfscheblueprint::DeclareDataOp> declareDataOps;
    mainFunc.walk([&](dfscheblueprint::DeclareDataOp op) { declareDataOps.push_back(op); });
    if (declareDataOps.empty())
        return success();

    // Track which constants have already been lowered to avoid duplicates
    // (multiple DeclareDataOps may reference the same constant)
    llvm::DenseSet<Operation *> processedConstants;
    SmallVector<Value> allocatedMemrefs;

    for (auto declareDataOp : declareDataOps) {
        Value initTensor = declareDataOp.getInitTensor();

        // Case 1: arith.constant (original path — hardcoded data)
        if (auto constantOp = initTensor.getDefiningOp<arith::ConstantOp>()) {
            // Skip if this constant was already lowered
            if (processedConstants.count(constantOp.getOperation()))
                continue;
            processedConstants.insert(constantOp.getOperation());

            auto tensorType = dyn_cast<RankedTensorType>(constantOp.getType());
            if (!tensorType)
                continue;

            // Use the first constant's shape/type for the legacy rootShape/elementType fields
            if (!state->elementType) {
                state->rootShape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
                state->elementType = tensorType.getElementType();
            }

            MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

            // Lower the dense constant: get a read-only memref view via bufferization.to_memref,
            // then alloc a writable buffer and copy into it.
            OpBuilder builder(constantOp.getOperation());
            builder.setInsertionPointAfter(constantOp.getOperation());
            auto loc = constantOp.getLoc();
            auto toMemref = builder.create<bufferization::ToMemrefOp>(loc, memrefType, constantOp.getResult());
            auto allocOp = builder.create<memref::AllocOp>(loc, memrefType);
            builder.create<memref::CopyOp>(loc, toMemref.getResult(), allocOp.getResult());

            // Register this constant's memref in the map
            state->constantToMemref[constantOp.getResult()] = allocOp.getResult();
            allocatedMemrefs.push_back(allocOp.getResult());

            // Keep legacy rootMemref/rootMemrefType pointing to the first allocation
            if (!state->rootMemref) {
                state->rootMemref = allocOp.getResult();
                state->rootMemrefType = memrefType;
            }
        }
        // Case 2: bufferization.to_tensor of func arg (external DDR pointer)
        // The user's pointer IS the DDR buffer — no alloc, no copy needed.
        else if (auto toTensorOp = initTensor.getDefiningOp<bufferization::ToTensorOp>()) {
            Value srcMemref = toTensorOp.getMemref();

            auto tensorType = dyn_cast<RankedTensorType>(toTensorOp.getType());
            if (!tensorType)
                continue;

            // Skip if already processed
            if (processedConstants.count(toTensorOp.getOperation()))
                continue;
            processedConstants.insert(toTensorOp.getOperation());

            if (!state->elementType) {
                state->rootShape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
                state->elementType = tensorType.getElementType();
            }

            MemRefType memrefType = cast<MemRefType>(srcMemref.getType());

            // Use the func arg memref DIRECTLY — no alloc, no copy needed.
            state->constantToMemref[initTensor] = srcMemref;

            if (!state->rootMemref) {
                state->rootMemref = srcMemref;
                state->rootMemrefType = memrefType;
            }
        }
    }

    // Insert memref.dealloc for ALL allocated memrefs before the return op.
    if (!allocatedMemrefs.empty()) {
        Block &mainBlock = mainFunc.getBody().front();
        for (auto &op : mainBlock) {
            if (isa<func::ReturnOp>(op)) {
                OpBuilder builder(&op);
                for (auto memref : allocatedMemrefs) {
                    builder.create<memref::DeallocOp>(op.getLoc(), memref);
                }
                break;
            }
        }
    }

    return success();
}

// Trace a view value back through tensor.extract_slice → routing.partitiontensor
// → dfscheblueprint.declare_data → arith.constant to find the originating constant,
// then look up its memref in the constantToMemref map.
Value resolveMemrefForView(Value viewValue, const BlueprintPassState &state) {
    // Walk backwards through extract_slice chain
    Value current = viewValue;
    while (auto extractSlice = current.getDefiningOp<tensor::ExtractSliceOp>()) {
        current = extractSlice.getSource();
    }
    // Now current should be the partition tensor result.
    // Walk through routing.partitiontensor to find its source tensor.
    if (auto partitionOp = current.getDefiningOp<routing::partitiontensor>()) {
        current = partitionOp.getTensor();
    }
    // Walk through dfscheblueprint.declare_data to find the init_tensor.
    if (auto declareOp = current.getDefiningOp<dfscheblueprint::DeclareDataOp>()) {
        current = declareOp.getInitTensor();
    }
    // Now current should be the arith.constant result or bufferization.to_tensor
    // result — look it up in the map.
    auto it = state.constantToMemref.find(current);
    if (it != state.constantToMemref.end())
        return it->second;
    return Value();
}

// Reads routing.fullconnect_auto (default true = M×N cartesian repeat / M-N round
// auto-generation enabled). false = A/B streamed once, no M/N round split; conv2d
// (halo) output tiling is halo-driven, never derived from tile_m/tile_rows.
bool isFullConnectAuto(ModuleOp moduleOp) {
    if (!moduleOp)
        return true;
    if (auto a = moduleOp->getAttrOfType<IntegerAttr>("routing.fullconnect_auto"))
        return a.getInt() != 0;
    return true;
}

TilingClassification classifyTiling(const routing::GemmTilingScalars &t) {
    TilingClassification result;
    result.mMode = TilingMode::Match;
    result.mRounds = 1;
    result.nMode = TilingMode::Match;
    result.nRounds = 1;

    // The tiling scalars come from the partitiontensor #routing.tiling op (read
    // once at pass entry). When fullconnect_auto==0 (conv) the reader returns all
    // zeros, so m==0 / n==0 below correctly yields Match/1/1 (no M/N round split).

    // M-dimension classification
    int64_t m = t.tileM;
    int64_t rows = t.tileRows;

    if (m == 0 || m == rows) {
        result.mMode = TilingMode::Match;
        result.mRounds = 1;
    } else if (m < rows && rows % m == 0) {
        result.mMode = TilingMode::Multiple;
        result.mRounds = rows / m;
    } else {
        result.mMode = TilingMode::Invalid;
        result.mRounds = 0;
    }

    // N-dimension classification
    int64_t n = t.tileN;
    int64_t cols = t.tileCols;

    if (n == 0 || n == cols) {
        result.nMode = TilingMode::Match;
        result.nRounds = 1;
    } else if (n < cols && cols % n == 0) {
        result.nMode = TilingMode::Multiple;
        result.nRounds = cols / n;
    } else {
        result.nMode = TilingMode::Invalid;
        result.nRounds = 0;
    }

    return result;
}

// Returns true if policy is "n_outer_m_inner" (N is outer loop, M is inner/iter)
bool isNOuterPolicy(ModuleOp moduleOp) {
    if (!moduleOp)
        return false;
    auto attr = moduleOp->getAttrOfType<StringAttr>("routing.iter_policy");
    if (attr && attr.getValue() == "n_outer_m_inner")
        return true;
    return false; // default: m_outer_n_inner
}

// Scan moduleOp attrs for a conv width-split halo. Returns {valid=false} otherwise.
// The width-split halo is announced by any "tensor_N.halo" dict carrying w_rounds>1,
// gated on routing.spatial_halo_buf_size>0.
ConvHaloGeom detectConvHalo(ModuleOp moduleOp) {
    ConvHaloGeom g;
    if (!moduleOp)
        return g;
    int64_t convBufSize = 0;
    if (auto b = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size"))
        convBufSize = b.getInt();
    if (convBufSize <= 0)
        return g;
    for (const NamedAttribute &na : moduleOp->getAttrs()) {
        if (!na.getName().getValue().ends_with(".halo"))
            continue;
        auto hd = dyn_cast<DictionaryAttr>(na.getValue());
        if (!hd)
            continue;
        auto wr = hd.getAs<IntegerAttr>("w_rounds");
        auto ot = hd.getAs<IntegerAttr>("ow_t");
        if (wr && wr.getInt() > 1 && ot) {
            g.valid = true;
            g.wRounds = wr.getInt();
            g.owT = ot.getInt();
            if (auto l2 = hd.getAs<IntegerAttr>("l2_rounds"))
                g.l2Rounds = l2.getInt();
            if (g.l2Rounds < 1)
                g.l2Rounds = 1;
            break;
        }
    }
    return g;
}

// Build the output tile descriptor from the true output geometry. All conv-vs-gemm
// geometry knowledge lives here. The output memref is modeled as [M, N] (row-major),
// with N = output row width (= OC for conv) and M split into mRounds x tileM. The
// per-tile column split (numCoreTiles) handles the N dimension via per-tile offsets.
OutputTileDescriptor buildOutputTileDescriptor(const BlueprintPassState &passState, MemRefType memrefType,
                                               int64_t numCoreTiles, ModuleOp moduleOp, int64_t ooElementSizeBytes,
                                               mlir::Attribute tilingAttr) {
    OutputTileDescriptor desc;

    unsigned bitWidth = memrefType.getElementTypeBitWidth();
    int64_t elemsPerWord = 32 / bitWidth;
    constexpr int64_t wordBytes = 4;
    int64_t outW = memrefType.getDimSize(1);  // full output row width (elements)
    int64_t outW_w = outW / elemsPerWord;     // ... in 32-bit words
    int64_t tileN_full = outW / numCoreTiles; // per-tile column width (elements)

    int64_t elemBytes = bitWidth / 8;
    if (elemBytes == 0)
        elemBytes = 1;

    // Conv2D width-split halo takes precedence: the spatial-halo path drops
    // routing.tile_m/tile_rows, so ALL geometry (tileM, mRounds, iter) is derived
    // from the halo attrs. Each core produces an [oh_per_row, OW_T, OC_PER_G] tile
    // scattered into the full [OH, OW, OC] image, across wRounds L->R x l2Rounds T->B.
    ConvHaloGeom halo = detectConvHalo(moduleOp);
    if (halo.valid && memrefType.getRank() == 3) {
        // === 3D conv output, channel-split (LtoR_Merge) ===
        // Generic gather descriptor: honor the DECLARED dim order in the
        // partition #routing.tiling attr and lay the DDR tensor out row-major of
        // that order (innermost declared dim contiguous). Each dim carries an
        // `axis` role: 1=mesh_row (T->B scf.for round loop), 2=mesh_col (per-tile
        // base offset), 0=on-core (folded into the BD iteration). Because every
        // BD stride is a row-major stride of the declared shape, the SAME code
        // emits the correct descriptor for CHW, HWC, or any permutation with no
        // layout special-casing. Falls back to the legacy detectConvHalo + memref
        // heuristic only when no 3-dim tiling attr is present.
        auto tiling = tilingAttr ? dyn_cast_or_null<routing::TilingAttr>(tilingAttr) : routing::TilingAttr();
        if (tiling && tiling.getDims().size() == 3) {
            auto dims = tiling.getDims();
            constexpr int n = 3;
            routing::LevelAttr outer[n];
            int64_t F[n], blk[n], rounds[n], axis[n];
            for (int k = 0; k < n; ++k) {
                outer[k] = dims[k].getOuter();
                F[k] = outer[k].getBase();
                axis[k] = dims[k].getAxis();
                if (routing::LevelAttr st = outer[k].getSliceTiling()) {
                    blk[k] = st.getSlice();
                    rounds[k] = st.getRounds();
                } else {
                    blk[k] = outer[k].getSlice();
                    rounds[k] = outer[k].getRounds();
                }
            }
            // Row-major strides (elements) of the declared shape F[].
            int64_t rmStride[n];
            rmStride[n - 1] = 1;
            for (int k = n - 2; k >= 0; --k)
                rmStride[k] = rmStride[k + 1] * F[k + 1];

            // Resolve dim roles from axis; positional H,W,C fallback on ambiguity.
            int rowDim = -1, colDim = -1, iterDim = -1;
            for (int k = 0; k < n; ++k) {
                if (axis[k] == 1)
                    rowDim = k;
                else if (axis[k] == 2)
                    colDim = k;
                else
                    iterDim = k;
            }
            if (rowDim < 0 || colDim < 0 || iterDim < 0) {
                rowDim = 0;
                iterDim = 1;
                colDim = 2;
            }

            assert(blk[n - 1] % elemsPerWord == 0 && "innermost per-fire block must be word-aligned");

            int64_t prod = 1;
            for (int k = 0; k < n; ++k)
                prod *= blk[k];
            desc.bdLenBytes = prod * ooElementSizeBytes;

            // BD scatter dims innermost-first: contiguous innermost dim counted in
            // 32-bit words, the outer dims by their row-major element stride.
            for (int k = n - 1; k >= 0; --k) {
                if (k == n - 1)
                    desc.bdDims.push_back({wordBytes, blk[k] / elemsPerWord});
                else
                    desc.bdDims.push_back({rmStride[k] * elemBytes, blk[k]});
            }

            // On-core (iter) dim folds into the BD iteration; mesh-row dim drives
            // the scf.for round loop; mesh-col dim advances the per-tile base.
            desc.iterStep = static_cast<int32_t>(blk[iterDim] * rmStride[iterDim] * elemBytes);
            desc.iterWrap = static_cast<int32_t>(outer[iterDim].getRounds());
            desc.totalRounds = rounds[rowDim];
            desc.roundDims.push_back({rounds[rowDim], blk[rowDim] * rmStride[rowDim] * elemBytes});
            desc.perTileStrideBytes = blk[colDim] * rmStride[colDim] * elemBytes;

            // #pragma output_tile_major: lay each core's rounds out back to back
            // instead of scattering them into the image. The dense layout needs
            // a 2-D round walk per core (iter dim x row-dim rounds) and a BD has
            // ONE iteration dimension, so the row-dim rounds became an scf.for
            // that re-programs the shim S2MM BDs between rounds -- while fast
            // cores may already be emitting the next round into them. Tile-major
            // makes the walk 1-D: per core, rounds r = 0..R-1 land at
            // r*roundBytes, the per-tile (mesh col) base advances by
            // R*roundBytes, and the mesh-row (channel) base is unchanged (a
            // mesh row owns the same byte count either way). One BD per core,
            // configured once, totalRounds = 1 -> straight-line schedule.
            // DDR layout: [mesh row][mesh col][round][on-core tile, row-major].
            if (passState.outputTileMajor) {
                const int64_t roundBytes = desc.bdLenBytes;
                const int64_t outRounds = rounds[rowDim] * static_cast<int64_t>(desc.iterWrap);
                desc.bdDims.clear();
                // 2-D contiguous walk (64-byte rows) keeps each wrap small; a
                // round that is not a multiple of 64 bytes uses one word dim.
                if (roundBytes % 64 == 0) {
                    desc.bdDims.push_back({wordBytes, 64 / wordBytes});
                    desc.bdDims.push_back({64, roundBytes / 64});
                } else {
                    desc.bdDims.push_back({wordBytes, roundBytes / wordBytes});
                }
                desc.iterStep = static_cast<int32_t>(roundBytes);
                desc.iterWrap = static_cast<int32_t>(outRounds);
                desc.totalRounds = 1;
                desc.roundDims.clear();
                desc.roundDims.push_back({1, 0});
                desc.perTileStrideBytes = outRounds * roundBytes;
                llvm::errs() << "[OutputTileDescriptor] output_tile_major: roundBytes=" << roundBytes
                             << " rounds/core=" << outRounds << " perTileStride=" << desc.perTileStrideBytes << "\n";
            }
            return desc;
        }

        // --- Legacy attr-less fallback (NCHW derivation) ---
        int64_t OUTPUT_H = 0, OUTPUT_W = 0, fullC = 0;
        int64_t ohPerRow = 0, owT = 0, wRounds = 0, hChunks = 0, cPerCore = 0;

        // Fallback derivation (no tiling attr): reuse the detectConvHalo heuristic
        // and the (channel-split) memref view + numCoreTiles, matching the tiling
        // attr values on the working conv (OUTPUT_H = ohPerRow*hChunks*meshRows is
        // not directly known, so approximate via the view's dim0 * numCoreTiles is
        // NOT valid for H; the tiling attr is the authoritative source and is
        // always present on the conv output partition).
        if (!(ohPerRow > 0 && owT > 0 && wRounds > 0 && hChunks > 0 && cPerCore > 0 && OUTPUT_H > 0 && OUTPUT_W > 0 &&
              fullC > 0)) {
            owT = halo.owT;
            wRounds = halo.wRounds;
            hChunks = halo.l2Rounds;
            OUTPUT_W = memrefType.getDimSize(1);
            cPerCore = memrefType.getDimSize(2);
            fullC = (numCoreTiles > 0) ? cPerCore * numCoreTiles : cPerCore;
            int64_t perCoreElems = 1;
            for (int64_t d = 0; d < memrefType.getRank(); ++d)
                perCoreElems *= memrefType.getDimSize(d);
            int64_t outRounds = wRounds * hChunks;
            if (auto orAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds"))
                if (orAttr.getInt() > 0)
                    outRounds = orAttr.getInt();
            int64_t slabElems = (outRounds > 0) ? (perCoreElems / outRounds) : perCoreElems;
            int64_t colElems = owT * cPerCore;
            ohPerRow = (colElems > 0) ? (slabElems / colElems) : 1;
            if (ohPerRow <= 0)
                ohPerRow = 1;
            // OUTPUT_H = per-mesh-row H (dim0) * mesh rows; mesh rows = ohPerRow*hChunks
            // rows-per-shim-view / ohPerRow-per-fire is ambiguous without the attr, so
            // fall back to the full-image H = ohPerRow * hChunks (single mesh row).
            OUTPUT_H = ohPerRow * hChunks;
        }

        int64_t owT_w = owT / elemsPerWord; // OW_T columns in 32-bit words (contiguous)

        // One BD fire = one [cPerCore, ohPerRow, owT] NCHW block.
        desc.bdLenBytes = ohPerRow * owT * cPerCore * ooElementSizeBytes;

        // BD scatter dims, innermost-first (NCHW [C,H,W], W contiguous):
        desc.bdDims.push_back({wordBytes, owT_w});                          // D0: W words (contiguous run)
        desc.bdDims.push_back({OUTPUT_W * elemBytes, ohPerRow});            // D1: H rows, one image row apart
        desc.bdDims.push_back({OUTPUT_H * OUTPUT_W * elemBytes, cPerCore}); // D2: C planes, one plane apart

        // Width (L->R) rounds fold into the BD iteration; height (T->B) rounds
        // become the scf.for outer round loop.
        desc.iterStep = static_cast<int32_t>(owT * elemBytes); // advance OW_T columns (W)
        desc.iterWrap = static_cast<int32_t>(wRounds);
        desc.totalRounds = hChunks;

        // Per-round base-offset: only the height (T->B) chunks move the base, by
        // ohPerRow image rows.
        desc.roundDims.push_back({hChunks, ohPerRow * OUTPUT_W * elemBytes});

        // Per-mesh-col tile offset advances by the owned C-group = cPerCore planes,
        // each plane OUTPUT_H*OUTPUT_W bytes.
        desc.perTileStrideBytes = cPerCore * OUTPUT_H * OUTPUT_W * elemBytes;

        return desc;
    }
    if (halo.valid) {
        int64_t haloOwT = halo.owT;             // OW_T columns per width-round (e.g. 28)
        int64_t haloWRounds = halo.wRounds;     // L->R rounds (e.g. 4)
        int64_t hChunks = halo.l2Rounds;        // T->B rounds (e.g. 4)
        int64_t fullOW = haloOwT * haloWRounds; // full output width (e.g. 112)
        // OC per channel-group == full row width / numCoreTiles == tileN_full.
        int64_t tileN_sub = tileN_full;
        int64_t tileN_sub_w = tileN_sub / elemsPerWord;

        // ohPerRow (image rows produced per BD fire / per on-core slab) is derived
        // from the authoritative per-core slab count so it always matches the core
        // MM2S sender. Total fires per core = spatial_out_rounds (= wRounds*l2Rounds).
        // perCoreElems = product(memref dims) / numCoreTiles;  each slab is
        //   [ohPerRow, OW_T, tileN_sub] => ohPerRow = slabElems / (OW_T * tileN_sub).
        int64_t totalElems = 1;
        for (int64_t d = 0; d < memrefType.getRank(); ++d)
            totalElems *= memrefType.getDimSize(d);
        int64_t perCoreElems = (numCoreTiles > 0) ? (totalElems / numCoreTiles) : totalElems;
        int64_t outRounds = haloWRounds * hChunks; // = spatial_out_rounds
        if (auto orAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds"))
            if (orAttr.getInt() > 0)
                outRounds = orAttr.getInt();
        int64_t slabElems = (outRounds > 0) ? (perCoreElems / outRounds) : perCoreElems;
        int64_t colElems = haloOwT * tileN_sub; // OW_T columns * OC-per-group
        int64_t ohPerRow = (colElems > 0) ? (slabElems / colElems) : 1;
        if (ohPerRow <= 0)
            ohPerRow = 1;

        // One BD activation = one [ohPerRow, OW_T, tileN_sub] scattered block.
        desc.bdLenBytes = ohPerRow * haloOwT * tileN_sub * ooElementSizeBytes;

        // BD scatter dims, innermost-first.
        desc.bdDims.push_back({1 * wordBytes, tileN_sub_w});          // D0: channel-group words
        desc.bdDims.push_back({outW * elemBytes, haloOwT});           // D1: OW_T columns, one pixel (OC) apart
        desc.bdDims.push_back({fullOW * outW * elemBytes, ohPerRow}); // D2: oh_per_row rows, one image row apart

        // The width (L->R) rounds are folded into the BD iteration dim; the height
        // (T->B) rounds become the scf.for outer round loop.
        desc.iterStep = static_cast<int32_t>(haloOwT * outW * elemBytes); // advance OW_T columns
        desc.iterWrap = static_cast<int32_t>(haloWRounds);
        desc.totalRounds = hChunks;

        // Per-round base-offset dims, outermost-first: only the height (T->B) chunks
        // move the base; the width rounds are already folded into iter above.
        desc.roundDims.push_back({hChunks, ohPerRow * fullOW * outW * elemBytes}); // hc stride

        return desc;
    }

    // === Non-halo (GEMM) path ===
    int64_t tileM = passState.tileM;
    int64_t tileRows = passState.tileRows;
    int64_t mRounds = (tileM > 0) ? (tileRows / tileM) : 1;

    // Per-tile N sub-tiling (nRounds): when tile_n < tile_cols, the per-tile width is
    // further split into nRounds iterations of tileN_sub columns.
    int64_t nRounds = 1;
    int64_t tileN_sub = tileN_full;
    if (passState.tileN > 0 && passState.tileN < passState.tileCols) {
        nRounds = passState.tileCols / passState.tileN;
        tileN_sub = passState.tileN;
    }
    int64_t tileN_sub_w = tileN_sub / elemsPerWord;

    // One d0xd1(xd2) block per BD activation (single OOO packet).
    desc.bdLenBytes = tileM * tileN_sub * ooElementSizeBytes;

    // Policy-aware iteration assignment. The "outer" loop is the scf.for round
    // dimension; the "inner" dimension is folded into the BD's iter_step/iter_wrap.
    bool nOuterPolicy = moduleOp ? isNOuterPolicy(moduleOp) : false;
    int64_t outerRounds;
    int64_t outerStrideBytes;
    if (nOuterPolicy) {
        // n_outer_m_inner: scf.for over nRounds, iter over mRounds
        desc.iterStep = static_cast<int32_t>(tileM * outW_w * wordBytes);
        desc.iterWrap = static_cast<int32_t>(mRounds);
        outerRounds = nRounds;
        outerStrideBytes = tileN_sub_w * wordBytes;
    } else {
        // m_outer_n_inner (default): scf.for over mRounds, iter over nRounds
        desc.iterStep = static_cast<int32_t>(tileN_sub_w * wordBytes);
        desc.iterWrap = static_cast<int32_t>(nRounds);
        outerRounds = mRounds;
        outerStrideBytes = tileM * outW_w * wordBytes;
    }
    desc.totalRounds = outerRounds;

    // GEMM contiguous-M: each round writes tileM contiguous M-rows.
    desc.bdDims.push_back({1 * wordBytes, tileN_sub_w}); // D0: contiguous word step
    desc.bdDims.push_back({outW_w * wordBytes, tileM});  // D1: DDR row stride, tileM rows
    desc.roundDims.push_back({outerRounds, outerStrideBytes});

    return desc;
}

} // namespace blueprint_sched
