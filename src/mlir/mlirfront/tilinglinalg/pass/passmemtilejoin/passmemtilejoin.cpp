/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "passmemtilejoin.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace routinghw;

int64_t mlir::memtileStaging(ModuleOp mod) {
    int64_t m = 0;
    if (auto a = mod->getAttrOfType<IntegerAttr>("routing.memtile_staging"))
        m = a.getInt();
    if (std::getenv("AIEHLC_MEMTILE_OUTPUT"))
        m |= kMemtileStageOutput;
    if (std::getenv("AIEHLC_MEMTILE_INPUT"))
        m |= kMemtileStageInput;
    if (std::getenv("AIEHLC_MEMTILE_C_BLOCKED"))
        m |= kMemtileStageCBlocked;
    return m;
}

bool mlir::isMemtileWeightsIo(ModuleOp mod, int64_t shimCol, int64_t channel) {
    auto list = mod->getAttrOfType<ArrayAttr>("routing.memtile_weights_io");
    if (!list)
        return false;
    for (Attribute a : list)
        if (auto i = dyn_cast<IntegerAttr>(a); i && i.getInt() == shimCol * 2 + channel)
            return true;
    return false;
}

static bool tileAt(Value v, int64_t col, int64_t row) {
    auto t = v.getDefiningOp<TileCreate>();
    if (!t)
        return false;
    auto c = t->getAttrOfType<IntegerAttr>("col"), r = t->getAttrOfType<IntegerAttr>("row");
    return c && r && c.getInt() == col && r.getInt() == row;
}

// Input half (input stage): an input flow climbs its shim column through the
// memtile at row 1 as a SOUTH -> NORTH circuit. Terminate it in memtile S2MM 2*ch+1 and
// re-send from MM2S 2*ch+1 on the same NORTH master, so the broadcast continues above.
static unsigned stageInputs(ModuleOp mod, bool weightsOnly) {
    SmallVector<EnableExtToAieShimPort> ins;
    mod.walk([&](EnableExtToAieShimPort op) { ins.push_back(op); });
    unsigned staged = 0;
    for (EnableExtToAieShimPort in : ins) {
        auto shim = in.getCurtile().getDefiningOp<IOShimTileCreate>();
        if (!shim)
            continue;
        auto colA = shim->getAttrOfType<IntegerAttr>("col"), chA = shim->getAttrOfType<IntegerAttr>("channelused");
        if (!colA || !chA)
            continue;
        const int64_t col = colA.getInt(), mch = 2 * chA.getInt() + 1;
        if (weightsOnly && !isMemtileWeightsIo(mod, col, chA.getInt()))
            continue;
        ConnectStreamSingleSwitchPort hop;
        for (Operation &op : *in->getBlock()) {
            auto c = dyn_cast<ConnectStreamSingleSwitchPort>(&op);
            if (c && tileAt(c.getCurtile(), col, 1) && c.getMasterportdirection() == "NORTH" &&
                c.getSlaveportdirection() == "SOUTH") {
                hop = c;
                break;
            }
        }
        if (!hop)
            continue;
        OpBuilder b(hop);
        const int32_t northIdx = (int32_t)hop.getMasterportidx();
        hop->setAttr("masterportdirection", b.getStringAttr("DMA"));
        hop->setAttr("masterportidx", b.getI32IntegerAttr((int32_t)mch));
        b.setInsertionPointAfter(hop);
        b.create<ConnectStreamSingleSwitchPort>(hop.getLoc(), hop.getResult().getType(), hop.getCurtile(),
                                                b.getStringAttr("DMA"), b.getI32IntegerAttr((int32_t)mch),
                                                b.getStringAttr("NORTH"), b.getI32IntegerAttr(northIdx));
        staged++;
    }
    return staged;
}

void MemtileOutputJoinRoutingPass::runOnOperation() {
    const int64_t stages = memtileStaging(getOperation());
    if (stages & (kMemtileStageInput | kMemtileStageWeights))
        llvm::errs() << "[TilingLinalg] MemtileOutputJoinRoutingPass: "
                     << stageInputs(getOperation(), !(stages & kMemtileStageInput))
                     << " input flow(s) staged through a memtile DMA\n";
    if (!(stages & kMemtileStageOutput))
        return;
    SmallVector<EnableAieToExtShimPort> outs;
    getOperation().walk([&](EnableAieToExtShimPort op) { outs.push_back(op); });
    unsigned joined = 0;
    for (EnableAieToExtShimPort out : outs) {
        auto shim = out.getCurtile().getDefiningOp<IOShimTileCreate>();
        if (!shim)
            continue;
        auto colA = shim->getAttrOfType<IntegerAttr>("col"), chA = shim->getAttrOfType<IntegerAttr>("channelused");
        if (!colA || !chA)
            continue;
        const int64_t col = colA.getInt(), mch = 2 * chA.getInt();
        // The flow's circuit hop through memtile (col, 1), in the same RoutingCreate region.
        ConnectStreamSingleSwitchPort hop;
        for (Operation &op : *out->getBlock()) {
            auto c = dyn_cast<ConnectStreamSingleSwitchPort>(&op);
            if (c && tileAt(c.getCurtile(), col, 1) && c.getMasterportdirection() == "SOUTH" &&
                c.getSlaveportdirection() == "NORTH") {
                hop = c;
                break;
            }
        }
        if (!hop)
            continue;
        OpBuilder b(hop);
        const int32_t southIdx = (int32_t)hop.getMasterportidx();
        hop->setAttr("masterportdirection", b.getStringAttr("DMA"));
        hop->setAttr("masterportidx", b.getI32IntegerAttr((int32_t)mch));
        b.setInsertionPointAfter(hop);
        auto emit = b.create<ConnectStreamSingleSwitchPort>(hop.getLoc(), hop.getResult().getType(), hop.getCurtile(),
                                                            b.getStringAttr("DMA"), b.getI32IntegerAttr((int32_t)mch),
                                                            b.getStringAttr("SOUTH"), b.getI32IntegerAttr(southIdx));
        (void)emit;
        joined++;
    }
    llvm::errs() << "[TilingLinalg] MemtileOutputJoinRoutingPass: " << joined
                 << " output flow(s) terminated in a memtile DMA\n";
}
