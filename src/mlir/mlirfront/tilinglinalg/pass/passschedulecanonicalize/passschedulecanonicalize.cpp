/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passschedulecanonicalize.h"
#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

using namespace mlir;
using namespace dfschedule;

namespace {

//===----------------------------------------------------------------------===//
// TileKey helpers
//===----------------------------------------------------------------------===//

using TileKey = std::pair<int64_t, int64_t>; // (col, row)

static TileKey getTileKey(dfschedule::DeclareTileOp op) {
    return {op.getCol(), op.getRow()};
}

//===----------------------------------------------------------------------===//
// Phase 1 — Greedy rewrite patterns (run inside RoutingCreate blocks)
//===----------------------------------------------------------------------===//

/// Replace a dfschedule.declaretile with an earlier sibling that declares the
/// same physical tile (same col + row attributes). Runs both within a single
/// RoutingCreate block (intra-block dedup) and inside dfschedule.host after
/// all RoutingCreate blocks are inlined (cross-iteration dedup).
struct FoldDuplicateDeclareTilePattern : public OpRewritePattern<dfschedule::DeclareTileOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(dfschedule::DeclareTileOp op, PatternRewriter &rewriter) const override {
        int64_t col = op.getCol();
        int64_t row = op.getRow();
        for (Operation &prior : *op->getBlock()) {
            if (&prior == op.getOperation())
                break;
            if (auto other = dyn_cast<dfschedule::DeclareTileOp>(&prior))
                if (other.getCol() == col && other.getRow() == row) {
                    rewriter.replaceOp(op, other.getTile());
                    return success();
                }
        }
        return failure();
    }
};

/// Replace a shim dfschedule.config.dma_bd that is identical (same tile,
/// data_id, DMA offset, bd_id, and len) to an earlier sibling with that
/// earlier BD handle.  bd_id must be included in the key because two distinct
/// DMA channels on the same shim tile configure different hardware BD
/// registers and must never be merged, even when data_id and offset coincide.
struct FoldDuplicateShimBdPattern : public OpRewritePattern<dfschedule::ConfigDmaBdOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(dfschedule::ConfigDmaBdOp op, PatternRewriter &rewriter) const override {
        auto tileDecl = op.getTile().getDefiningOp<dfschedule::DeclareTileOp>();
        if (!tileDecl || tileDecl.getRow() != 0)
            return failure();

        // Never merge packet-enabled shim BDs — OOO per-tile shim BDs have
        // distinct packet IDs and DDR offsets for out-of-order execution.
        // Also skip BDs with out_of_order_bd_id set (core MM2S OOO BDs).
        if (op.getEnablePacket())
            return failure();
        if (static_cast<int32_t>(op.getOutOfOrderBdId()) >= 0)
            return failure();

        int32_t dataId = static_cast<int32_t>(op.getDataId());
        int32_t len = static_cast<int32_t>(op.getLen());

        // bd_id and offset are SSA i32 values; extract the constants if available.
        APInt bdIdVal;
        bool hasBdId = matchPattern(op.getBdId(), m_ConstantInt(&bdIdVal));
        APInt offsetVal;
        bool hasOffset = matchPattern(op.getOffset(), m_ConstantInt(&offsetVal));

        for (Operation &prior : *op->getBlock()) {
            if (&prior == op.getOperation())
                break;
            if (auto other = dyn_cast<dfschedule::ConfigDmaBdOp>(&prior)) {
                auto otherTile = other.getTile().getDefiningOp<dfschedule::DeclareTileOp>();
                if (!otherTile)
                    continue;
                // Require matching bd_id constants; if either bd_id is non-constant
                // (dynamic), conservatively skip the fold.
                APInt otherBdIdVal;
                bool otherHasBdId = matchPattern(other.getBdId(), m_ConstantInt(&otherBdIdVal));
                if (!hasBdId || !otherHasBdId || bdIdVal != otherBdIdVal)
                    continue;
                APInt otherOffsetVal;
                bool otherHasOffset = matchPattern(other.getOffset(), m_ConstantInt(&otherOffsetVal));
                if (otherTile.getCol() == tileDecl.getCol() && otherTile.getRow() == tileDecl.getRow() &&
                    static_cast<int32_t>(other.getDataId()) == dataId && hasOffset && otherHasOffset &&
                    offsetVal == otherOffsetVal && static_cast<int32_t>(other.getLen()) == len) {
                    rewriter.replaceOp(op, other.getBdHandle());
                    return success();
                }
            }
        }
        return failure();
    }
};

//===----------------------------------------------------------------------===//
// Phase 2 — IRMapping-driven clone + inline into dfschedule.host
//===----------------------------------------------------------------------===//

/// If `allocVal` is a memref.alloc that is the dest of a memref.copy whose
/// source traces through bufferization.to_memref → arith.constant, return
/// that constant op. Returns nullptr if the chain is not found.
static arith::ConstantOp findInitConstantForAlloc(Value allocVal) {
    if (!allocVal.getDefiningOp<memref::AllocOp>())
        return nullptr;
    Block *parentBlock = allocVal.getDefiningOp()->getBlock();
    if (!parentBlock)
        return nullptr;
    for (Operation &op : *parentBlock) {
        auto copyOp = dyn_cast<memref::CopyOp>(&op);
        if (!copyOp || copyOp.getTarget() != allocVal)
            continue;
        auto toMr = copyOp.getSource().getDefiningOp<bufferization::ToMemrefOp>();
        if (!toMr)
            continue;
        if (auto constOp = toMr.getOperand().getDefiningOp<arith::ConstantOp>())
            return constOp;
    }
    return nullptr;
}

/// Deep-walk the entire RoutingCreate subtree and pre-populate `mapper` with
/// host-local replacements for every value referenced inside `rcOp` but defined
/// OUTSIDE it.  dfschedule.host has IsolatedFromAbove so external values must
/// be substituted with host-local equivalents:
/// - external memref → plain memref.alloc of same shape/element type (with init chain if present)
/// - external arith.constant → cloned constant inside host
static void preMapExternalValues(Operation *rcOp, OpBuilder &builder, Location loc, IRMapping &mapper,
                                 SmallVector<Value> &hostAllocs, Block *hostBody = nullptr) {
    // Phase 1: Collect all unique external block arguments and add them to the
    // host block sorted by their original argument number. This prevents the
    // walk-encounter order (which depends on IR layout, e.g. std::map key
    // ordering of "col" vs "row") from scrambling A/B/C argument positions.
    if (hostBody) {
        SmallVector<std::pair<unsigned, Value>> blockArgs; // (argNumber, value)
        rcOp->walk([&](Operation *innerOp) {
            for (Value operand : innerOp->getOperands()) {
                if (mapper.contains(operand))
                    continue;
                Operation *defOp = operand.getDefiningOp();
                if (!defOp) {
                    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                        if (isa<MemRefType>(operand.getType())) {
                            bool found = false;
                            for (auto &pair : blockArgs) {
                                if (pair.second == operand) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                                blockArgs.push_back({blockArg.getArgNumber(), operand});
                        }
                    }
                }
            }
        });
        // Sort by original argument number to preserve declaration order.
        llvm::sort(blockArgs, [](auto &a, auto &b) { return a.first < b.first; });
        // Add to host block in correct order.
        for (auto &[argNum, operand] : blockArgs) {
            auto memrefType = cast<MemRefType>(operand.getType());
            auto plainType = MemRefType::get(memrefType.getShape(), memrefType.getElementType());
            Value hostArg = hostBody->addArgument(plainType, loc);
            mapper.map(operand, hostArg);
        }
    }

    // Phase 2: Handle non-block-arg external values (memref allocs, constants).
    // Block arguments are already mapped above, so they are skipped here.
    rcOp->walk([&](Operation *innerOp) {
        for (Value operand : innerOp->getOperands()) {
            if (mapper.contains(operand))
                continue;
            Operation *defOp = operand.getDefiningOp();
            if (!defOp)
                continue; // block args already handled in Phase 1
            if (rcOp->isAncestor(defOp))
                continue;
            if (dyn_cast<MemRefType>(operand.getType())) {
                auto memrefType = cast<MemRefType>(operand.getType());
                auto plainType = MemRefType::get(memrefType.getShape(), memrefType.getElementType());
                Value newAlloc;
                if (auto constOp = findInitConstantForAlloc(operand)) {
                    // Reproduce the init chain inside the host body:
                    //   arith.constant → bufferization.to_memref → memref.alloc → memref.copy
                    Value clonedConst = builder.clone(*constOp.getOperation())->getResult(0);
                    auto tensorType = cast<RankedTensorType>(constOp.getType());
                    auto roType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
                    Value roView = builder.create<bufferization::ToMemrefOp>(loc, roType, clonedConst).getResult();
                    newAlloc = builder.create<memref::AllocOp>(loc, plainType).getResult();
                    builder.create<memref::CopyOp>(loc, roView, newAlloc);
                } else {
                    newAlloc = builder.create<memref::AllocOp>(loc, plainType).getResult();
                }
                mapper.map(operand, newAlloc);
                hostAllocs.push_back(newAlloc);
            } else if (isa<arith::ConstantOp>(defOp)) {
                mapper.map(operand, builder.clone(*defOp)->getResult(0));
            }
        }
    });
}

struct KernelCfgEntry {
    TileKey tileKey;
    ArrayAttr tileConfigs; // from DeclareKernelConfigOp.tile_configs
    Attribute computeArg;  // from LoadKernelGroupOp.distributed_compute_kernel_args[i]
};

/// Collect kernel config info from the cloned RoutingCreate block, then erase
/// the ops that must NOT be inlined (ScheduleWait, LaunchKernelGroup,
/// LoadKernelGroup, DeclareKernelConfig, FreeDeviceMem).
/// Non-launch events from ScheduleWaitOp are pushed into `allWaitEvents`.
static void collectAndErasePreInlineOps(Block &clonedBlock, Operation *clonedRc,
                                        SmallVector<KernelCfgEntry> &kernelCfgEntries, ArrayAttr &calleeAttrs,
                                        SmallVector<Value> &allWaitEvents) {
    // First pass: collect from LoadKernelGroupOp (tile→config association).
    for (Operation &op : clonedBlock) {
        auto loadOp = dyn_cast<dfschedule::LoadKernelGroupOp>(&op);
        if (!loadOp)
            continue;

        if (!calleeAttrs)
            calleeAttrs = loadOp.getCalleeAttr();

        auto distArgsOpt = loadOp.getDistributedArgs();
        auto computeKernelArgs = loadOp.getDistributedComputeKernelArgs();
        auto tiles = loadOp.getTiles();

        for (size_t i = 0; i < tiles.size(); ++i) {
            auto tileDecl = tiles[i].getDefiningOp<dfschedule::DeclareTileOp>();
            if (!tileDecl)
                continue;
            TileKey k = getTileKey(tileDecl);

            // Find the corresponding DeclareKernelConfigOp via symbol lookup.
            ArrayAttr tileConfigs;
            if (distArgsOpt) {
                ArrayAttr distArgs = *distArgsOpt;
                if (i < (size_t)distArgs.size()) {
                    if (auto symRef = dyn_cast<SymbolRefAttr>(distArgs[i])) {
                        if (auto cfgOp = SymbolTable::lookupSymbolIn(clonedRc, symRef.getRootReference()))
                            if (auto kCfg = dyn_cast<dfschedule::DeclareKernelConfigOp>(cfgOp))
                                tileConfigs = kCfg.getTileConfigs();
                    }
                }
            }

            Attribute computeArg = SymbolRefAttr::get(clonedRc->getContext(), "compute0");
            if (i < computeKernelArgs.size())
                computeArg = computeKernelArgs[i];

            kernelCfgEntries.push_back({k, tileConfigs, computeArg});
        }
    }

    // Second pass: collect non-launch events from ScheduleWaitOp.
    for (Operation &op : clonedBlock) {
        if (auto waitOp = dyn_cast<dfschedule::ScheduleWaitOp>(&op)) {
            for (Value ev : waitOp.getEvents()) {
                if (ev.getDefiningOp() && isa<dfschedule::LaunchKernelGroupOp>(ev.getDefiningOp()))
                    continue;
                allWaitEvents.push_back(ev);
            }
        }
    }

    // Third pass: collect ops to erase from cloned block (in program order).
    // We erase them in REVERSE dependency order so uses are cleared safely.
    SmallVector<Operation *> eraseList;
    for (Operation &op : clonedBlock) {
        if (isa<dfschedule::ScheduleWaitOp, dfschedule::LaunchKernelGroupOp, dfschedule::LoadKernelGroupOp,
                dfschedule::DeclareKernelConfigOp, dfschedule::FreeDeviceMemOp>(&op))
            eraseList.push_back(&op);
    }
    for (Operation *op : llvm::reverse(eraseList)) {
        op->dropAllUses();
        op->erase();
    }
}

static void buildHostBlockByCloning(func::FuncOp mainFunc, ModuleOp moduleOp) {
    MLIRContext *ctx = mainFunc.getContext();
    OpBuilder builder(ctx);
    Location loc = mainFunc.getLoc();

    // Create dfschedule.host at module scope after func.func @main.
    builder.setInsertionPointAfter(mainFunc);
    auto hostOp = builder.create<dfschedule::HostBlockOp>(loc, "host_canonicalized");
    Block *hostBody = &hostOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(hostBody);

    // Shared mapper and collection state across all RoutingCreate iterations.
    IRMapping mapper;
    SmallVector<KernelCfgEntry> kernelCfgEntries;
    ArrayAttr calleeAttrs;
    SmallVector<Value> allWaitEvents;
    SmallVector<Value> hostAllocs;

    // ── Pre-map ALL external block arguments BEFORE processing any
    //    RoutingCreate.  Collecting across ALL RoutingCreate ops and sorting
    //    globally by argNumber ensures host block arguments match the original
    //    @main signature order (arg0=A, arg1=B, arg2=C) regardless of which
    //    RoutingCreate happens to reference which argument first.
    {
        SmallVector<std::pair<unsigned, Value>> blockArgs;
        mainFunc.walk([&](Operation *rcOp) {
            if (rcOp->getName().getStringRef() != "routing.RoutingCreate")
                return;
            if (rcOp->getNumRegions() == 0 || rcOp->getRegion(0).empty())
                return;
            rcOp->walk([&](Operation *innerOp) {
                for (Value operand : innerOp->getOperands()) {
                    Operation *defOp = operand.getDefiningOp();
                    if (!defOp) {
                        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                            if (isa<MemRefType>(operand.getType())) {
                                bool found = false;
                                for (auto &pair : blockArgs) {
                                    if (pair.second == operand) {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
                                    blockArgs.push_back({blockArg.getArgNumber(), operand});
                            }
                        }
                    }
                }
            });
        });
        llvm::sort(blockArgs, [](auto &a, auto &b) { return a.first < b.first; });
        for (auto &[argNum, operand] : blockArgs) {
            auto memrefType = cast<MemRefType>(operand.getType());
            auto plainType = MemRefType::get(memrefType.getShape(), memrefType.getElementType());
            Value hostArg = hostBody->addArgument(plainType, loc);
            mapper.map(operand, hostArg);
        }
    }

    // Walk routing.RoutingCreate ops in mainFunc (in program order).
    mainFunc.walk([&](Operation *rcOp) {
        if (rcOp->getName().getStringRef() != "routing.RoutingCreate")
            return;
        if (rcOp->getNumRegions() == 0 || rcOp->getRegion(0).empty())
            return;

        // Always insert new content at the end of host body.
        builder.setInsertionPointToEnd(hostBody);

        // ── Pre-map external values to host-local ops ──────────────────────
        // Block arguments are already mapped globally above; this only handles
        // non-block-arg external values (memref allocs, constants).
        preMapExternalValues(rcOp, builder, loc, mapper, hostAllocs, nullptr);

        // ── Clone the entire RoutingCreate (with its body) into host ────────
        // builder.clone performs a full deep clone, remapping operands via mapper.
        Operation *clonedRc = builder.clone(*rcOp, mapper);

        // ── Collect kernel info + erase pre-inline-excluded ops ────────────
        Block &clonedBlock = clonedRc->getRegion(0).front();
        collectAndErasePreInlineOps(clonedBlock, clonedRc, kernelCfgEntries, calleeAttrs, allWaitEvents);

        // ── Inline cloned body into host block (before the cloned RC op) ───
        // Remove the block terminator if present (RoutingCreate may have one).
        if (Operation *term = clonedBlock.getTerminator()) {
            term->dropAllUses();
            term->erase();
        }
        // Move every remaining op from the cloned block to just before clonedRc.
        SmallVector<Operation *> opsToMove;
        for (Operation &op : clonedBlock)
            opsToMove.push_back(&op);
        for (Operation *op : opsToMove)
            op->moveBefore(clonedRc);

        // Erase the now-empty cloned RoutingCreate wrapper.
        clonedRc->erase();
    });

    // ── Phase 2b: Deduplicate tiles + shim BDs inside the flat host block ──
    {
        RewritePatternSet patterns(ctx);
        patterns.add<FoldDuplicateDeclareTilePattern>(ctx);
        patterns.add<FoldDuplicateShimBdPattern>(ctx);
        FrozenRewritePatternSet frozen(std::move(patterns));
        GreedyRewriteConfig cfg;
        cfg.maxIterations = 32;
        // Ignore failure — dedup is best-effort.
        (void)applyPatternsAndFoldGreedily(hostOp, frozen, cfg);
    }

    // ── Build canonicalTileMap from surviving declaretile ops in host ───────
    DenseMap<TileKey, Value> canonicalTileMap;
    hostOp.walk([&](dfschedule::DeclareTileOp tileOp) {
        TileKey k = getTileKey(tileOp);
        canonicalTileMap.try_emplace(k, tileOp.getTile());
    });

    // ── Emit merged kernel configs, load, launch at end of host ────────────
    builder.setInsertionPointToEnd(hostBody);

    // Deduplicate kernelCfgEntries by TileKey (first occurrence wins).
    SmallVector<KernelCfgEntry> uniqueEntries;
    DenseSet<TileKey> seenTiles;
    for (auto &entry : kernelCfgEntries) {
        if (seenTiles.insert(entry.tileKey).second)
            uniqueEntries.push_back(entry);
    }

    SmallVector<Value> allCoreTileVals;
    SmallVector<Attribute> allComputeArgs;
    SmallVector<Attribute> kernelConfigSyms;

    for (size_t i = 0; i < uniqueEntries.size(); ++i) {
        auto &entry = uniqueEntries[i];
        Value tileVal = canonicalTileMap.lookup(entry.tileKey);
        if (!tileVal)
            continue;

        allCoreTileVals.push_back(tileVal);
        allComputeArgs.push_back(entry.computeArg);

        std::string cfgName = "kernelconfig_merged" + std::to_string(i);
        ArrayAttr cfgAttr = entry.tileConfigs ? entry.tileConfigs : builder.getArrayAttr({});
        builder.create<dfschedule::DeclareKernelConfigOp>(loc, dfschedule::KernelConfigType::get(ctx),
                                                          builder.getStringAttr(cfgName), cfgAttr);
        kernelConfigSyms.push_back(SymbolRefAttr::get(ctx, cfgName));
    }

    Value launchEvent;
    if (!allCoreTileVals.empty() && calleeAttrs) {
        auto loadOp = builder.create<dfschedule::LoadKernelGroupOp>(
            loc, dfschedule::KernelGroupType::get(ctx), allCoreTileVals, calleeAttrs,
            builder.getArrayAttr(allComputeArgs),
            /*kernel_config=*/nullptr, builder.getArrayAttr(kernelConfigSyms));
        auto launchOp = builder.create<dfschedule::LaunchKernelGroupOp>(loc, dfschedule::EventType::get(ctx),
                                                                        loadOp.getKernelGroup());
        launchEvent = launchOp.getEvent();

        // Ordering rule for StartIoOp: core S2MM receivers BEFORE shim MM2S senders.
        //
        // Background: the AIE stream switch uses a credit-based flow-control protocol.
        // When shim MM2S (sender) fires before the downstream core S2MM (receiver) is
        // armed, the stream switch has no credits and the shim channel stalls permanently
        // (STALL_STREAM deadlock). The correct order is:
        //   1. load_kernel_group  — loads ELF, zeros BSS (protects core memory)
        //   2. launch_kernel_group
        //   3. core S2MM startios — arm receivers (input flows) FIRST
        //   4. shim MM2S startios — start senders only AFTER receivers are ready
        //   5. core MM2S + shim S2MM — output flow startios (safe at any point)
        //
        // Collect all StartIoOp ops and classify by tile type and DMA direction.
        // isShim: row==0 (shim tile), !isShim: row>0 (core tile).
        // isSender (MM2S): this is a DMA source pushing data into the stream.
        SmallVector<Operation *> coreStartIoOps;       // core tile startios (input S2MM + output MM2S)
        SmallVector<Operation *> shimSenderStartIoOps; // shim MM2S startios (shim sends to core)
        for (Operation &op : *hostBody) {
            if (auto startIo = dyn_cast<dfschedule::StartIoOp>(&op)) {
                // Trace: StartIoOp → io_handle (ConfigCreateIoOp) → tile + direction
                bool isShim = false;
                bool isMM2S = false;
                if (auto createIo = startIo.getIoHandle().getDefiningOp<dfschedule::ConfigCreateIoOp>()) {
                    if (auto declareTile = createIo.getTile().getDefiningOp<dfschedule::DeclareTileOp>()) {
                        isShim = (declareTile.getRow() == 0);
                    }
                    // direction attribute: "MM2S" = sender, "S2MM" = receiver
                    if (auto dirAttr = createIo->getAttrOfType<mlir::StringAttr>("direction"))
                        isMM2S = (dirAttr.getValue() == "MM2S");
                }
                if (!isShim) {
                    coreStartIoOps.push_back(&op);
                } else if (isMM2S) {
                    // Shim MM2S = shim is sender → must come AFTER core S2MM receivers
                    shimSenderStartIoOps.push_back(&op);
                }
                // Shim S2MM (receiver, output flow) stays in place — no deadlock risk.
            }
        }
        // Step 1: Move all core startios right after launch (ELF-load protection +
        // ensures core receivers are placed before shim senders in the block).
        Operation *insertAfter = launchOp;
        for (Operation *op : coreStartIoOps) {
            op->moveAfter(insertAfter);
            insertAfter = op;
        }
        // Step 2: Move shim MM2S (sender) startios to AFTER all core startios.
        // This guarantees core S2MM receivers are armed before shim pushes data.
        for (Operation *op : shimSenderStartIoOps) {
            op->moveAfter(insertAfter);
            insertAfter = op;
        }
        llvm::errs() << "[ScheduleCanonicalize] StartIo ordering: " << coreStartIoOps.size() << " core, "
                     << shimSenderStartIoOps.size() << " shim-MM2S moved after launch\n";
    }

    // Prepend launch event so wait covers it first.
    if (launchEvent)
        allWaitEvents.insert(allWaitEvents.begin(), launchEvent);

    if (!allWaitEvents.empty())
        builder.create<dfschedule::ScheduleWaitOp>(loc, allWaitEvents);

    // Emit dealloc for each host-local alloc created to substitute external memrefs.
    for (Value allocVal : hostAllocs)
        builder.create<memref::DeallocOp>(loc, allocVal);

    // ── Insert dfschedule.launchhost in func.func @main ─────────────────────
    Block &mainBlock = mainFunc.getBody().front();
    Operation *terminator = mainBlock.getTerminator();
    if (terminator)
        builder.setInsertionPoint(terminator);
    else
        builder.setInsertionPointToEnd(&mainBlock);
    builder.create<dfschedule::LaunchHostOp>(loc, SymbolRefAttr::get(ctx, "host_canonicalized"));
}

//===----------------------------------------------------------------------===//
// Phase 3 — Safe cleanup
//===----------------------------------------------------------------------===//

static void eraseOldOps(func::FuncOp mainFunc) {
    if (!mainFunc)
        return;

    // Collect scf.execute_region ops (they contain all the old dfschedule ops
    // including the original routing.RoutingCreate ops).
    SmallVector<Operation *> execRegions;
    mainFunc.walk([&](scf::ExecuteRegionOp op) {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            execRegions.push_back(op.getOperation());
    });

    // Erase execute_region ops. Drop all defined-value uses first (the cloned
    // copies in dfschedule.host have taken over).
    for (auto *op : llvm::reverse(execRegions)) {
        op->dropAllDefinedValueUses();
        op->erase();
    }

    // Erase dead init-chain allocs in @main (alloc + copy + dealloc) that were
    // migrated into dfschedule.host by preMapExternalValues.
    {
        SmallVector<Operation *> deadAllocs;
        mainFunc.walk([&](memref::AllocOp allocOp) {
            if (allocOp->getParentOp() != mainFunc.getOperation())
                return;
            bool onlyInitUsers = true;
            for (Operation *user : allocOp.getResult().getUsers()) {
                if (!isa<memref::CopyOp>(user) && !isa<memref::DeallocOp>(user)) {
                    onlyInitUsers = false;
                    break;
                }
            }
            if (onlyInitUsers)
                deadAllocs.push_back(allocOp.getOperation());
        });
        for (Operation *allocOp : llvm::reverse(deadAllocs)) {
            // Erase users (copy + dealloc) first, then the alloc itself.
            for (Operation *user : llvm::make_early_inc_range(cast<memref::AllocOp>(allocOp).getResult().getUsers()))
                user->erase();
            allocOp->erase();
        }
    }

    // Erase any remaining top-level dead arith.constants in @main.
    // Also erase dead bufferization.to_memref ops (and their arith.constant sources)
    // left over after preMapExternalValues substituted them with host-local allocs.
    SmallVector<Operation *> dead;
    mainFunc.walk([&](Operation *op) {
        if (op->getParentOp() != mainFunc.getOperation())
            return;
        if (!op->use_empty())
            return;
        if (isa<arith::ConstantOp>(op) || isa<bufferization::ToMemrefOp>(op))
            dead.push_back(op);
    });
    for (auto *op : llvm::reverse(dead))
        if (op->use_empty())
            op->erase();
    // Second pass: catch arith.constant whose only user was the now-erased to_memref.
    dead.clear();
    mainFunc.walk([&](arith::ConstantOp cOp) {
        if (cOp->getParentOp() == mainFunc.getOperation() && cOp.use_empty())
            dead.push_back(cOp.getOperation());
    });
    for (auto *op : llvm::reverse(dead))
        if (op->use_empty())
            op->erase();
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

namespace mlir {

void ScheduleCanonicalizePass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // ── Find func.func @main or @host_canonicalized ────────────────────────
    func::FuncOp mainFunc;
    moduleOp.walk([&](func::FuncOp f) {
        if (f.getName() == "main" || f.getName() == "host_canonicalized")
            mainFunc = f;
    });
    if (!mainFunc)
        return;

    // Early exit: already canonical if no routing.RoutingCreate ops present.
    bool hasRoutingCreate = false;
    mainFunc.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "routing.RoutingCreate")
            hasRoutingCreate = true;
    });
    if (!hasRoutingCreate)
        return;

    // ── Phase 1: Run dedup patterns inside RoutingCreate blocks ─────────────
    // These handle intra-block duplicates; cross-iteration dedup happens in
    // Phase 2b after all RoutingCreate bodies are inlined into dfschedule.host.
    {
        RewritePatternSet patterns(ctx);
        patterns.add<FoldDuplicateDeclareTilePattern>(ctx);
        patterns.add<FoldDuplicateShimBdPattern>(ctx);
        GreedyRewriteConfig cfg;
        cfg.maxIterations = 32;
        if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns), cfg)))
            return signalPassFailure();
    }

    // ── Phase 2: Clone + inline RoutingCreate blocks, build dfschedule.host ─
    buildHostBlockByCloning(mainFunc, moduleOp);

    // ── Phase 3: Erase old execute_region ops and dead constants ────────────
    eraseOldOps(mainFunc);
}

} // namespace mlir
