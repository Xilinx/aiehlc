/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passgroupregwrite.h"
#include "dfschedulemanager.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace dfschedule;

namespace {

// aie2ps core-tile memory-module LOCK0_VALUE tile-local byte offset and the
// per-lock register stride (see thirdparty/alib xaie2psgbl_params.h:
// XAIE2PSGBL_MEMORY_MODULE_LOCK0_VALUE = 0x1F000, stride 0x10). A control-packet
// write of the lock value to (0x1F000 + lock_id*0x10) reproduces the effect of
// XAie_LockSetValue(XAie_LockInit(lock_id, value)).
constexpr uint32_t kCoreLock0ValueOff = 0x1F000u;
constexpr uint32_t kLockValueStride = 0x10u;
// Core tiles begin at this row (rows 0=shim, 1=memtile on this partition). Only
// core-tile lock inits share the 0x1F000 offset, so folding is gated to them.
constexpr int kCoreRowMin = 2;

// One derived lock-init register write.
struct LockInit {
    int col;
    int row;
    int lockId;
    int value;
    uint32_t tileAddr;
};

// Mirror DfscheduleToApiPass: trace a config.dma_bd result through any BD chain
// to the config.create_io that consumes it; the flow is "output" iff that
// create_io direction is MM2S.
static bool bdIsOutput(dfschedule::ConfigDmaBdOp op) {
    SmallVector<Operation *, 4> worklist;
    for (auto *u : op.getResult().getUsers())
        worklist.push_back(u);
    while (!worklist.empty()) {
        auto *u = worklist.pop_back_val();
        if (auto io = dyn_cast<dfschedule::ConfigCreateIoOp>(u))
            return io.getDirection().str() == "MM2S";
        if (isa<dfschedule::ConfigDmaBdOp>(u))
            for (auto *uu : u->getResults().front().getUsers())
                worklist.push_back(uu);
    }
    return false;
}

} // namespace

void GroupRegWritePass::runOnOperation() {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    dfschedule::HostBlockOp hostOp;
    mod.walk([&](dfschedule::HostBlockOp h) { hostOp = h; });
    if (!hostOp)
        return;

    // --- 1. Collect deduplicated lock-init tuples (mirror DfscheduleToApi) ---
    // Dedup by (col,row,lockId), matching state.initializedLocks bookkeeping.
    std::set<std::tuple<int, int, int>> seen;
    std::vector<LockInit> inits;

    hostOp.walk([&](dfschedule::ConfigDmaBdOp op) {
        int acqId = static_cast<int32_t>(op.getAcquireLockId());
        int acqVal = static_cast<int32_t>(op.getAcquireLockVal());
        // Gate identical to DfscheduleToApi (excludes shim BDs).
        if (!(acqId >= 0 && acqVal != 0))
            return;
        auto td = op.getTile().getDefiningOp<dfschedule::DeclareTileOp>();
        if (!td)
            return;
        int col = td.getCol();
        int row = td.getRow();
        if (row < kCoreRowMin) // core tiles only (0x1F000 lock offset)
            return;
        // Outer dedup key = (col,row,acquireLockId), inserted before the
        // output/input split (matches DfscheduleToApi).
        auto outerKey = std::make_tuple(col, row, acqId);
        if (seen.count(outerKey))
            return;
        seen.insert(outerKey);

        int nextBd = static_cast<int32_t>(op.getNextBd());
        bool isSingleBuffer = (nextBd == -1) && !op.getLinkedBd();
        int value = isSingleBuffer ? 1 : 2;

        int lockId;
        if (bdIsOutput(op)) {
            // Output (MM2S): the emitted init targets the kernel-acquire lock
            // (= release lock id), with its own dedup key.
            lockId = static_cast<int32_t>(op.getReleaseLockId());
            auto k = std::make_tuple(col, row, lockId);
            if (seen.count(k))
                return;
            seen.insert(k);
        } else {
            // Input (S2MM): the emitted init targets the DMA-acquire lock.
            lockId = acqId;
        }

        LockInit li;
        li.col = col;
        li.row = row;
        li.lockId = lockId;
        li.value = value;
        li.tileAddr = kCoreLock0ValueOff + static_cast<uint32_t>(lockId) * kLockValueStride;
        inits.push_back(li);
    });

    if (inits.empty())
        return;

    // --- 2. Configured tile set + per-row column coverage ---
    std::set<std::pair<int, int>> configuredTiles; // (col,row)
    std::map<int, std::set<int>> rowCols;          // row -> {cols}
    int shimCol = INT32_MAX;
    for (const auto &li : inits) {
        configuredTiles.insert({li.col, li.row});
        rowCols[li.row].insert(li.col);
        shimCol = std::min(shimCol, li.col);
    }

    // --- 3. Cluster by (tileAddr,value) ---
    std::map<std::pair<uint32_t, int>, std::vector<const LockInit *>> clusters;
    for (const auto &li : inits)
        clusters[{li.tileAddr, li.value}].push_back(&li);

    // --- 4. Classify + emit. Insert before schedule.launch_kernel_group. ---
    dfschedule::LaunchKernelGroupOp launchOp;
    hostOp.walk([&](dfschedule::LaunchKernelGroupOp l) {
        if (!launchOp)
            launchOp = l;
    });

    OpBuilder builder(ctx);
    if (launchOp)
        builder.setInsertionPoint(launchOp);
    else
        builder.setInsertionPointToEnd(&hostOp.getBody().front());
    Location loc = hostOp.getLoc();

    // Folded (col,row,lockId) triples for DfscheduleToApi to skip.
    std::vector<Attribute> foldedTriples;
    // Group writes to emit (kind, row, tileAddr, value); built after fabric.
    struct Emit {
        std::string kind;
        int row;
        uint32_t tileAddr;
        int value;
    };
    std::vector<Emit> emits;

    for (auto &kv : clusters) {
        uint32_t tileAddr = kv.first.first;
        int value = kv.first.second;
        const auto &members = kv.second;

        // Cluster tile set + per-row cols.
        std::set<std::pair<int, int>> clusterTiles;
        std::map<int, std::set<int>> clusterRowCols;
        for (const auto *li : members) {
            clusterTiles.insert({li->col, li->row});
            clusterRowCols[li->row].insert(li->col);
        }

        std::string kind;
        int targetRow = -1;
        if (clusterTiles == configuredTiles) {
            kind = "broadcast";
        } else if (clusterRowCols.size() == 1) {
            int r = clusterRowCols.begin()->first;
            if (clusterRowCols[r] == rowCols[r]) { // all configured cols of row r
                kind = "row";
                targetRow = r;
            }
        }
        if (kind.empty())
            continue; // leave these writes individual

        emits.push_back({kind, targetRow, tileAddr, value});
        for (const auto *li : members)
            foldedTriples.push_back(builder.getI32ArrayAttr({li->col, li->row, li->lockId}));
    }

    if (emits.empty())
        return;

    // --- 5. Build ctrl_plan_init rows (bottom-up) from configured tiles ---
    std::vector<Attribute> rowsAttr;
    for (auto &rc : rowCols) {
        int row = rc.first;
        int colLo = *rc.second.begin();
        int colHi = *rc.second.rbegin();
        SmallVector<NamedAttribute, 3> fields;
        fields.push_back(builder.getNamedAttr("row", builder.getI32IntegerAttr(row)));
        fields.push_back(builder.getNamedAttr("col_lo", builder.getI32IntegerAttr(colLo)));
        fields.push_back(builder.getNamedAttr("col_hi", builder.getI32IntegerAttr(colHi)));
        rowsAttr.push_back(builder.getDictionaryAttr(fields));
    }

    auto fabricTy = dfschedule::CtrlFabricType::get(ctx);
    auto planInit = builder.create<dfschedule::CtrlPlanInitOp>(
        loc, fabricTy, static_cast<uint32_t>(shimCol), static_cast<uint32_t>(ctrlId), static_cast<uint32_t>(respS2mmCh),
        builder.getArrayAttr(rowsAttr));
    Value fabric = planInit.getResult();

    for (const auto &e : emits) {
        builder.create<dfschedule::GroupRegWriteOp>(loc, fabric, StringRef(e.kind),
                                                    static_cast<uint32_t>(e.row < 0 ? 0 : e.row), e.tileAddr,
                                                    builder.getI32ArrayAttr({e.value}), static_cast<uint32_t>(sendBdId),
                                                    static_cast<uint32_t>(sendMm2sCh), /*log=*/0u);
    }

    // --- 6. Record folded triples for DfscheduleToApi ---
    mod->setAttr("dfschedule.grouped_lock_inits", builder.getArrayAttr(foldedTriples));

    llvm::errs() << "[GroupRegWrite] folded " << foldedTriples.size() << " lock inits into " << emits.size()
                 << " control-packet group write(s); fabric shim_col=" << shimCol << " ctrl_id=" << ctrlId << "\n";
}
