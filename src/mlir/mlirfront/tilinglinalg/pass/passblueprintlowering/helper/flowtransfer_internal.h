/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// Shared internal header for the split FlowTransferConversion implementation.
//
// Lives in passblueprintlowering/helper/ — one level ABOVE the two passes that
// use it, because the host path (passblueprinttoschedule) and the kernel path
// (passblueprinttoschedulekernel) lower the SAME core/kernel-tile DMA, lock and
// buffer configuration and must agree on it. Anything describing core-tile
// config that both paths need belongs here, not inside either pass.
//
//   passblueprintlowering/
//     helper/                        — this shared layer
//     passblueprinttoschedule/       — host path  → host.cc
//     passblueprinttoschedulekernel/ — kernel path → kernel.cc
//
// The single ~2200-line FlowTransferConversion::matchAndRewrite has been broken
// into a thin dispatcher (passblueprinttoschedule.cpp) plus host / kernel helper
// member methods defined across:
//   helper/flowtransfer_common.cpp  — shared utilities (traceToFuncArgIndex, ...)
//   helper/flowtransfer_host.cpp    — shim DMA + schedule orchestration
//   helper/flowtransfer_kernel.cpp  — core-tile config (ping-pong / single-buffer)
//
// All extracted helpers are member methods of FlowTransferConversion so they can
// naturally access the pattern's mutable buffer-index bookkeeping members. Locals
// that cross section boundaries are carried in FlowLoweringCtx; per-core-tile
// locals are carried in CoreTileCtx.
//
// Everything shared lives in namespace blueprint_sched to avoid ODR clashes with
// the identically-named static helpers in sibling passes.

#ifndef __FLOWTRANSFER_INTERNAL_H__
#define __FLOWTRANSFER_INTERNAL_H__

#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include "hw/ResourceManager.h"
#include "hw/hwresource.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "routingmanager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace blueprint_sched {

using namespace mlir;

// ---------------------------------------------------------------------------
// Shared state between pre-processing, FlowTransferConversion, and post-processing.
// ---------------------------------------------------------------------------
struct BlueprintPassState {
    Value rootMemref;
    MemRefType rootMemrefType;
    SmallVector<int64_t> rootShape;
    Type elementType;
    // Map from each arith.constant result to its allocated memref.
    // When multiple data tensors exist (e.g. input A, input B, output C),
    // each has its own backing memref.
    llvm::DenseMap<Value, Value> constantToMemref;
    // Routing module attributes, cached before conversion (module attrs may be
    // stripped during applyPartialConversion).
    int64_t tileM = 0;
    int64_t tileRows = 0;
    int64_t tileN = 0;
    int64_t tileCols = 0;
    int64_t effectiveK = 0;
    int64_t fullK = 0;
    int64_t kRounds = 0;
    // Raw #routing.tiling reader output (found=false for conv / fullconnect_auto=0),
    // read once at pass entry. classifyTiling consumes this so it never re-walks the
    // partitiontensor op during conversion. Empty for conv, mirroring the old
    // !isFullConnectAuto Match/1/1 behavior.
    routing::GemmTilingScalars tilingScalars;
    // KERNELCONFIGOFFLOAD (routing.kernel_config_offload): when set, the AIE core
    // self-programs its own incoming (S2MM) DMA from kernel.cc via raw MMIO, so the
    // host must NOT emit the S2MM core-tile DMA chain. emitCoreBufferDma reads this
    // to skip the S2MM BD + create_io + start_io for core tiles.
    bool kernelConfigOffload = false;
};

// KERNELCONFIGOFFLOAD: lowest tile row that is a compute core able to run
// kernel.cc and therefore self-program its own DMA. Rows below this are the
// shim (0) and memtiles (1), which have no core to execute the offloaded MMIO
// block, so their DMA config always stays host-side.
inline constexpr int64_t kOffloadCoreRowMin = 2;

// === Tiling Classification ===
// Determines whether the M/N-dimension tiling requires host-side SCF loops.
enum class TilingMode { Match, Multiple, Invalid };

struct TilingClassification {
    TilingMode mMode;
    int64_t mRounds; // 1 for Match, tileRows/tileM for Multiple
    TilingMode nMode;
    int64_t nRounds; // 1 for Match, tileCols/tileN for Multiple
};

// === Generic N-D output access-pattern descriptor ===
struct OutputTileDescriptor {
    // BD scatter dims, innermost-first: {strideBytes, wrap}.
    SmallVector<std::pair<int64_t, int64_t>> bdDims;
    // Per-round base-offset dims, OUTERMOST-first: {wrap(extent), strideBytes}.
    SmallVector<std::pair<int64_t, int64_t>> roundDims;
    int64_t perTileStrideBytes = 0; // per-tile channel-column step (offset += t*this)
    int64_t bdLenBytes = 0;         // -> perRoundBytes
    int64_t totalRounds = 1;        // -> scf.for ub (= product of roundDims wraps)
    int32_t iterStep = 0;           // -> oooIterStepSize
    int32_t iterWrap = 1;           // -> oooIterWrap
};

// Deferred core StartIoOp data: collected inside the per-tile loop and emitted
// AFTER LoadKernelGroup/LaunchKernelGroup so ELF BSS init does not overwrite DMA.
struct DeferredCoreStartIo {
    Value ioHandle;
    Value bdId;
    uint32_t flowIdx;
    int32_t repeatCount;
};

// Offset / repeat parameters computed by computeMultipleInputOffsetParams and
// consumed by the op-emission tail of emitScheduleMultipleInput.
struct MultipleInputOffsetParams {
    int64_t outerRounds = 1;
    int32_t perIterRepeat = 1;
    int64_t subTileStride = 0;
    bool halo2D = false;
    int64_t haloHStrideBytes = 0;
    int64_t haloWStrideBytes = 0;
    int64_t haloWRoundsVal = 0;
};

// ---------------------------------------------------------------------------
// Shared utilities (defined in flowtransfer_common.cpp). Promoted from the
// former file-local statics; kept in namespace blueprint_sched to avoid clashes.
// ---------------------------------------------------------------------------
int traceToFuncArgIndex(Value v);
int traceFlowConfigToFuncArgIndex(dfscheblueprint::FlowConfigOp flowConfig);

// lookupSymbolOp is a template so its definition must live in the header.
template <typename OpTy> OpTy lookupSymbolOp(Operation *rootOp, SymbolRefAttr target) {
    StringRef targetName = target.getRootReference().getValue();

    // First, search in the same block as the rootOp
    Block *parentBlock = rootOp->getBlock();
    if (parentBlock) {
        for (Operation &op : *parentBlock) {
            if (auto targetOp = dyn_cast<OpTy>(&op)) {
                if (targetOp.getSymName() == targetName) {
                    return targetOp;
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
                    if (auto targetOp = dyn_cast<OpTy>(&op)) {
                        if (targetOp.getSymName() == targetName) {
                            return targetOp;
                        }
                    }
                }
            }
        }
        parentOp = parentOp->getParentOp();
    }

    return nullptr;
}

dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target);
dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target);
dfscheblueprint::DataSliceOp lookupDataSlice(Operation *rootOp, SymbolRefAttr target);
bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName);
Operation *getModuleOp(Operation *rootOp);
void generateDSKernelReceiver(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                              StringRef kernelName, RankedTensorType tensorType, int64_t bufferLen,
                              uint32_t basePacketId, int64_t coreChannel, uint32_t flowIndex);
SmallVector<OpFoldResult> toOpFoldResult(ArrayRef<int64_t> values, OpBuilder &b);
LogicalResult preprocessConstantToMemref(Operation *topLevel, std::shared_ptr<BlueprintPassState> state);
Value resolveMemrefForView(Value viewValue, const BlueprintPassState &state);
bool isFullConnectAuto(ModuleOp moduleOp);
TilingClassification classifyTiling(const routing::GemmTilingScalars &t);
bool isNOuterPolicy(ModuleOp moduleOp);

// === Conv2D width-split halo geometry ===
// Extracted from the "tensor_N.halo" module dict (set by DmaphopTodfscheblueprint /
// tilinglinalg_pipeline). Present only when the output is a scattered conv image
// (spatial_halo_buf_size>0 and w_rounds>1). The spatial-halo path drops
// routing.tile_m / routing.tile_rows (they are 0), so the output-gather BD geometry
// must be derived entirely from these halo attrs rather than passState.tileM/tileRows.
struct ConvHaloGeom {
    bool valid = false;   // true if a width-split conv halo was found
    int64_t owT = 0;      // OW_T: output columns produced per width-round per core
    int64_t wRounds = 0;  // number of L->R width rounds (e.g. 4)
    int64_t l2Rounds = 1; // number of T->B height rounds (e.g. 4); 1 if absent
};
// Scan moduleOp attrs for a conv width-split halo. Returns {valid=false} otherwise.
ConvHaloGeom detectConvHalo(ModuleOp moduleOp);

OutputTileDescriptor buildOutputTileDescriptor(const BlueprintPassState &passState, MemRefType memrefType,
                                               int64_t numCoreTiles, ModuleOp moduleOp, int64_t ooElementSizeBytes,
                                               mlir::Attribute tilingAttr = nullptr);

// ---------------------------------------------------------------------------
// FlowLoweringCtx — trunk state shared across the extracted host/kernel helpers.
// Only variables that cross the major section boundaries live here; section-local
// temporaries stay inside their owning method.
// ---------------------------------------------------------------------------
struct FlowTransferConversion; // fwd

struct FlowLoweringCtx {
    // Non-owning references into matchAndRewrite.
    ConversionPatternRewriter &rewriter;
    Location loc;
    dfscheblueprint::FlowTransferOp op;

    FlowLoweringCtx(ConversionPatternRewriter &rw, Location l, dfscheblueprint::FlowTransferOp o)
        : rewriter(rw), loc(l), op(o) {}

    // --- prologue: roles + memref resolve ---
    dfscheblueprint::FlowConfigOp shimFlowConfig = nullptr;
    dfscheblueprint::FlowConfigOp coreFlowConfig = nullptr;
    bool shimIsSender = false;
    dfscheblueprint::TileGroupOp shimTileGroup = nullptr;
    dfscheblueprint::TileGroupOp coreTileGroup = nullptr;
    uint32_t basePacketId = 0;
    uint32_t flowIndex = 0;
    Value viewValue;
    Type viewType;
    tensor::ExtractSliceOp partExtractSlice;
    RankedTensorType shimTensorType;
    Value flowRootMemref;
    Value partitionSubview;
    MemRefType memrefType;
    Value ddrBuffer;

    // --- shim tile + DMA params ---
    int64_t shimCol = 0, shimRow = 0;
    dfschedule::DeclareTileOp shimTileOp;
    int64_t shimChannel = 0;
    StringRef dmaDirection;
    StringRef ioOperation;
    int64_t bufferLen = 1;
    int64_t elementSizeBytesShim = 1;
    StringRef transferType;
    int64_t shimBdLen = 0;
    int32_t dataId = -1;
    bool isHaloSlab = false;
    bool kAccumHaloSlab = false;
    int64_t kAccumL2Rounds = 0, kAccumL2Step = 0, kAccumRowPitchElems = 0;
    int64_t l2CorePerRoundLen = 0;
    int64_t numCoreTiles = 1;
    int64_t perTileShimLen = 0;
    int32_t shimBdIdVal = -1;
    arith::ConstantOp shimBdIdConst;
    ArrayAttr shimDimStrides;
    ArrayAttr shimDimWraps;

    // --- shim BD (OOO / non-OOO) ---
    bool isManyToOne = false;
    bool useOOO = false;
    SmallVector<int32_t> shimPerTileBdIds;
    Value lastShimBdHandle;
    SmallVector<Value> shimBdHandles;
    int64_t ooElementSizeBytes = 1;
    int64_t ooFullPartitionElements = 0;
    int64_t ooPerCoreElements = 0;
    int64_t ooPingPongSize = 1;
    int64_t ooNumIterations = 0;
    int64_t perRoundBytes = 0;
    int32_t shimIterStepSize = 0;
    int32_t shimIterWrap = 0;
    bool usedMRounds3D = false;
    int64_t oooMRounds = 1;
    int32_t oooIterStepSize = 0;
    int32_t oooIterWrap = 0;
    int64_t perTileStrideFromDims = 0;
    ArrayAttr perTileDimStrides = nullptr;
    ArrayAttr perTileDimWraps = nullptr;
    OutputTileDescriptor outDesc;
    dfschedule::ConfigCreateIoOp createIoOp;

    // --- core tiles ---
    ArrayAttr coreTilesAttr;
    SmallVector<Value> coreTiles;
    int64_t coreChannel = 0;
    dfscheblueprint::bp_direction coreDmaDir = dfscheblueprint::bp_direction::S2MM;
    StringRef coreDmaDirection;
    StringRef coreIoOperation;
    std::optional<ArrayAttr> sliceSymbolsOpt;
    int64_t flowPingL1Offset = 0;
    int64_t flowPongL1Offset = 0;
    bool flowAddrsValid = false;
    bool isInput = false;
    int funcArgIdx = -1;
    int dirIdx = 0;
    SmallVector<DeferredCoreStartIo> deferredCoreStartIos;
    int tileIndex = 0;
    // KERNELCONFIGOFFLOAD gate for this flow, decided once by the orchestrator
    // (BlueprintToSchedulePass::matchAndRewrite) from
    // BlueprintPassState::kernelConfigOffload. When true the host must not emit
    // this flow's core-tile DMA *configuration* (BD chain + create_io +
    // start_io); the core self-programs it from kernel.cc via raw MMIO.
    //
    // Scope note: BOTH directions are offloadable. S2MM config is identical on
    // every core tile and emits straight-line; MM2S carries per-tile packet_id /
    // ooo_bd_id and emits under a get_coreid() dispatch built from the plan this
    // host walk publishes (ResourceMgr::coreOffloadPlan).
    //
    // DeclareTileOp is never suppressed: it feeds `coreTiles` ->
    // load_kernel_group, i.e. it is what puts the kernel ELF on the core. The
    // offload depends on that ELF running, so dropping it would remove the very
    // code meant to replace the host configuration.
    bool offloadCoreDmaConfig = false;

    // --- kernel_config finalize + schedule ---
    SmallVector<Attribute> calleeAttrs;
    TilingClassification classification;
    bool needsOuterLoop = false;
    bool fullConnect = true;
    int64_t haloL2RoundsForLoop = 0;
    int64_t haloKRoundsForLoop = 0;
    dfschedule::GetBdIdOp getBdIdOp;
};

// Per-core-tile locals, carried across the extracted kernel sub-methods so the
// per-tile loop body stays byte-identical while each sub-method is < 200 lines.
struct CoreTileCtx {
    int64_t col = 0, row = 0;
    dfschedule::DeclareTileOp coreTileOp;
    int64_t bufferSize = 0;
    int64_t elementSizeBytes = 1;
    int64_t perTileSize = 0;
    int64_t bufferOffset = 0;
    int64_t fullPartitionElements = 0;
    int64_t perCoreElements = 0;
    int64_t perCorePerKRound = 0;
    int ppDepth = 2;
    int64_t pingPongBufferSize = 1;
    bool hostSpatialHaloPort = false;
    bool kSplitSlabPort = false;
    int64_t numIterations = 0;
    int acquireLockId = 0;
    int releaseLockId = 0;
    // per-tile subview / BD emission
    MemRefType shapedPerTileType;
    Value perTileToken;
    int64_t pingL1Offset = 0;
    int64_t pongL1Offset = 0;
    int64_t coreBdLen = 0;
    bool isOutputFlow = false;
    int bdAcquireLockId = 0;
    int bdReleaseLockId = 0;
    bool coreBdEnablePacket = false;
    int32_t coreBdPacketId = 0;
    int32_t coreOooBdId = -1;
    Value firstCoreBdHandle;
    // KERNELCONFIGOFFLOAD: BD ids reserved for the chain the CORE programs itself
    // (reserveOffloadedCoreBds). -1 when not offloaded or reservation failed.
    // Recorded so the provenance map can describe that chain to the debug UI.
    int32_t offloadPingBdId = -1;
    int32_t offloadPongBdId = -1;
};

// ---------------------------------------------------------------------------
// The conversion pattern. Extracted helper member methods are declared here and
// defined across flowtransfer_host.cpp / flowtransfer_kernel.cpp.
// ---------------------------------------------------------------------------
// Everything the core-tile emitters need that is NOT per-flow state.
//
// Exists so the core-tile lowering can be called from BOTH blueprint-lowering
// passes: the host path (BlueprintToSchedulePass) and, under
// #pragma KERNELCONFIGOFFLOAD, the kernel path (BlueprintToScheduleKernelPass),
// where the core programs its own DMA and the ops belong in the kernel module.
// Sharing the emitter — rather than each pass growing its own copy — is what
// makes the two descriptions agree by construction: one ping/pong chain shape,
// one set of L1 offsets, one lock convention.
//
// `emitCoreDma` is the who-emits switch. Under offload the HOST still reserves
// BD ids and publishes ResourceMgr::coreOffloadPlan (the provenance map runs on
// hostModule before the kernel pass, and the BD bank is shared hardware), but
// does not emit; the KERNEL emits. Without offload the host emits as before and
// the kernel emits nothing.
struct CoreTileEmitDeps {
    std::shared_ptr<ResourceMgr> resourceMgr;
    std::shared_ptr<BlueprintPassState> passState;
    double bufferRatio = 0.5;
    int64_t maxPingPongBytes = 0;
    // Buffer index mapping keyed by data_id, carried across flows within one pass.
    std::unordered_map<int32_t, int> *dataIdToInputIdx = nullptr;
    std::unordered_map<int32_t, int> *dataIdToOutputIdx = nullptr;
    int *nextInputIdx = nullptr;
    int *nextOutputIdx = nullptr;
    // true  -> this caller emits the core-tile DMA ops
    // false -> this caller only does accounting (reserve + publish)
    bool emitCoreDma = true;
};

// Core-tile lowering, shared by the host and kernel blueprint passes.
LogicalResult emitCoreTileConfigs(FlowLoweringCtx &c, const CoreTileEmitDeps &d);

struct FlowTransferConversion : public OpConversionPattern<dfscheblueprint::FlowTransferOp> {
    std::shared_ptr<ResourceMgr> resourceMgr;
    std::shared_ptr<BlueprintPassState> passState;
    double bufferRatio;
    int64_t maxPingPongBytes;
    // Buffer index mapping keyed by data_id (see original comment).
    mutable std::unordered_map<int32_t, int> dataIdToInputIdx;
    mutable std::unordered_map<int32_t, int> dataIdToOutputIdx;
    mutable int nextInputIdx = 0;
    mutable int nextOutputIdx = 0;

    // Bundle this pattern's members for the shared core-tile emitters.
    CoreTileEmitDeps coreDeps(bool emitCoreDma) const {
        CoreTileEmitDeps d;
        d.resourceMgr = resourceMgr;
        d.passState = passState;
        d.bufferRatio = bufferRatio;
        d.maxPingPongBytes = maxPingPongBytes;
        d.dataIdToInputIdx = &dataIdToInputIdx;
        d.dataIdToOutputIdx = &dataIdToOutputIdx;
        d.nextInputIdx = &nextInputIdx;
        d.nextOutputIdx = &nextOutputIdx;
        d.emitCoreDma = emitCoreDma;
        return d;
    }

    FlowTransferConversion(MLIRContext *ctx, std::shared_ptr<ResourceMgr> mgr,
                           std::shared_ptr<BlueprintPassState> state, double ratio, int64_t maxPPBytes)
        : OpConversionPattern<dfscheblueprint::FlowTransferOp>(ctx), resourceMgr(std::move(mgr)),
          passState(std::move(state)), bufferRatio(ratio), maxPingPongBytes(maxPPBytes) {}

    LogicalResult matchAndRewrite(dfscheblueprint::FlowTransferOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;

    // --- host helpers (flowtransfer_host.cpp) ---
    void emitShimTileAndParams(FlowLoweringCtx &c) const;
    LogicalResult computeShimBdParams(FlowLoweringCtx &c) const;
    void emitShimBdOoo(FlowLoweringCtx &c) const;
    void emitShimBdNonOoo(FlowLoweringCtx &c) const;
    void classifyScheduleMode(FlowLoweringCtx &c) const;
    LogicalResult emitScheduleMultipleInput(FlowLoweringCtx &c) const;
    // Sub-helper of emitScheduleMultipleInput: pure offset/repeat parameter
    // computation (no op emission). Fills MultipleInputOffsetParams.
    void computeMultipleInputOffsetParams(FlowLoweringCtx &c, ModuleOp moduleOp, bool nOuterPolicy,
                                          struct MultipleInputOffsetParams &p) const;
    void emitScheduleOooOutput(FlowLoweringCtx &c) const;
    void emitScheduleStraightLine(FlowLoweringCtx &c) const;

    // --- kernel helpers: see the free functions below ---
    void finalizeKernelConfig(FlowLoweringCtx &c) const;
};

// ---------------------------------------------------------------------------
// Core-tile lowering (flowtransfer_kernel.cpp)
// ---------------------------------------------------------------------------
// Free functions, not members, so BOTH blueprint passes can call them — see
// CoreTileEmitDeps above. They take their dependencies explicitly rather than
// reading a pattern's members.
LogicalResult emitCoreTileParams(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
// Sub-helper of emitCoreTileParams: clamp buffer + compute numIterations
// (halo / K-round / K-split) + validate pp_depth=1. No op emission.
LogicalResult computeCoreIterations(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
LogicalResult emitCoreBufferDma(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
// Sub-helper of emitCoreBufferDma: CoreMemAllocator ping/pong address
// allocation (first tile only) + per-tile ping/pong L1 offset assignment.
void emitCoreBufferAlloc(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
void emitCoreSingleBufferBd(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
void emitCorePingPongBd(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);
// KERNELCONFIGOFFLOAD counterpart of the two emitCore*Bd helpers: claims the
// same per-tile BD ids from the pool WITHOUT emitting any op, for the case
// where the core self-programs those BDs from kernel.cc. Keeps the host's
// allocator honest so a host-programmed flow on the same tile (e.g. MM2S
// output) is never handed a BD the core is already using.
void reserveOffloadedCoreBds(FlowLoweringCtx &c, CoreTileCtx &t, const CoreTileEmitDeps &d);

} // namespace blueprint_sched

#endif // __FLOWTRANSFER_INTERNAL_H__
