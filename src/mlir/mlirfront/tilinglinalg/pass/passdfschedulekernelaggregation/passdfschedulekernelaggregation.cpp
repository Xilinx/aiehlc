/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdfschedulekernelaggregation.h"
#include "dfschedulemanager.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;
using namespace dfschedule;

namespace {

// Attribute names written onto the aggregated dma_bd. Discardable (set via
// setAttr), so no .td change is needed -- the same approach lock_def uses for
// init_value and buffer_def for address.
constexpr llvm::StringLiteral kAggregatedAttr = "aggregated";
constexpr llvm::StringLiteral kTileCoordsAttr = "tile_coords";
constexpr llvm::StringLiteral kTilePacketIdsAttr = "tile_packet_ids";
constexpr llvm::StringLiteral kTileOooBdIdsAttr = "tile_ooo_bd_ids";

// packet_id and out_of_order_bd_id are OPERANDS, not attributes, so the operand
// comparison below must skip exactly these two positions. Everything else --
// attributes and the remaining operands alike -- is compared, so a newly added
// field defaults to being checked rather than silently ignored: a wrong merge
// programs real DMA registers and would still compile.
//
// Operand order (see td/dfscheduleop.td): buffer, tile, bd_id, offset,
// packet_id, out_of_order_bd_id, [linked_bd].
constexpr unsigned kPacketIdOperandIdx = 4;
constexpr unsigned kOooBdIdOperandIdx = 5;

bool isPerTileVaryingOperand(unsigned idx) { return idx == kPacketIdOperandIdx || idx == kOooBdIdOperandIdx; }

// One entry of the dfschedule.core_offload_plan module attribute.
struct PlanEntry {
    int32_t packetId = 0;
    int32_t oooBdId = -1;
};

// Mirror DfscheduleToApiPass / GroupRegWritePass: trace a config.dma_bd result
// through any BD chain to the config.create_io that consumes it; the flow is
// "output" iff that create_io direction is MM2S.
bool bdIsOutput(ConfigDmaBdOp op) {
    SmallVector<Operation *, 4> worklist;
    for (auto *u : op.getResult().getUsers())
        worklist.push_back(u);
    while (!worklist.empty()) {
        auto *u = worklist.pop_back_val();
        if (auto io = dyn_cast<ConfigCreateIoOp>(u))
            return io.getDirection().str() == "MM2S";
        if (isa<ConfigDmaBdOp>(u))
            for (auto *uu : u->getResults().front().getUsers())
                worklist.push_back(uu);
    }
    return false;
}

// One per-tile dma_bd occurrence, before clustering.
struct BdRecord {
    ConfigDmaBdOp op;
    int col = 0;
    int row = 0;
    bool isOutput = false;
    int32_t bdId = 0;      // the SSA bd_id constant, identifies the ping/pong slot
    int64_t bufOffset = 0; // bind_core_buffer offset, identifies the L1 buffer
};

// Cluster identity: which physical BD slot of which buffer this is. Two records
// with the same key describe the same BD on different tiles.
using ClusterKey = std::tuple<int32_t, int64_t, bool>;

// Read the bind_core_buffer offset feeding a dma_bd, so two BDs that differ only
// by which L1 buffer they point at never merge. Returns false if the operand
// chain is not the expected memref.alloc -> memref_mapping -> bind_core_buffer.
bool getBufferOffset(ConfigDmaBdOp op, int64_t &out) {
    auto bind = op.getBuffer().getDefiningOp<BindCoreBufferOp>();
    if (!bind)
        return false;
    out = bind.getOffset();
    return true;
}

// Constant value behind an i32 SSA operand (bd_id / offset are materialized as
// arith.constant by the blueprint pass).
bool getConstantI32(Value v, int32_t &out) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return false;
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!intAttr)
        return false;
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
}

} // namespace

void DfscheduleKernelAggregationPass::runOnOperation() {
    ModuleOp mod = getOperation();

    // The per-tile core DMA config only exists under KERNELCONFIGOFFLOAD. The
    // gate lives on the kernel module (set by BlueprintToScheduleKernelPass), not
    // on the top-level module, because that is the op this pass operates on.
    KernelModuleOp kernelModuleOp;
    mod.walk([&](KernelModuleOp k) {
        if (!kernelModuleOp)
            kernelModuleOp = k;
    });
    if (!kernelModuleOp)
        return;

    auto offloadAttr = kernelModuleOp->getAttrOfType<IntegerAttr>("dfschedule.kernel_config_offload");
    if (!offloadAttr || offloadAttr.getInt() == 0)
        return;

    // NOTE (WIP prototype): this block replaces the original implementation,
    // which is commented out below. It compiles now, but see the caveats at the
    // end of this comment before relying on it.
    //
    // Accessor shapes on ConfigDmaBdOp, which the first draft got wrong:
    //   OPERANDS (return Value -> use getConstantI32):
    //       getBdId(), getOffset(), getPacketId(), getOutOfOrderBdId()
    //   ATTRIBUTES (return uint32_t directly -> no extraction):
    //       getLen(), getNextBd(), getAcquireLockId(), getReleaseLockId()
    //   The tile is NOT on the BD: there is no getCol()/getRow(). Coordinates
    //   come from the DeclareTileOp behind getTile().
    struct ConfigBdKey {
        int32_t bdId, offset, oooBdId, packetId, len, nextBd, acquireLockId, releaseLockId;
        bool operator<(const ConfigBdKey &o) const {
            return std::tie(bdId, offset, oooBdId, packetId, len, nextBd, acquireLockId, releaseLockId) <
                   std::tie(o.bdId, o.offset, o.oooBdId, o.packetId, o.len, o.nextBd, o.acquireLockId, o.releaseLockId);
        }
    };
    // Key -> the tiles that carry an identical BD. std::set needs operator<
    // (the original draft supplied only operator== and a Hash, which std::set
    // does not use); the tile list must live OUTSIDE the key, because set
    // elements are const and cannot be mutated after insertion.
    std::map<ConfigBdKey, std::set<std::pair<int32_t, int32_t>>> configBds;

    SmallVector<ConfigDmaBdOp> duplicates;
    kernelModuleOp.walk([&](ConfigDmaBdOp op) {
        auto td = op.getTile().getDefiningOp<DeclareTileOp>();
        if (!td)
            return; // already a self-tile, or not a core-tile BD
        int32_t col = static_cast<int32_t>(td.getCol());
        int32_t row = static_cast<int32_t>(td.getRow());

        ConfigBdKey k;
        if (!getConstantI32(op.getBdId(), k.bdId) || !getConstantI32(op.getOffset(), k.offset) ||
            !getConstantI32(op.getOutOfOrderBdId(), k.oooBdId) || !getConstantI32(op.getPacketId(), k.packetId))
            return; // non-constant operand: cannot key on it, leave alone
        k.len = static_cast<int32_t>(op.getLen());
        k.nextBd = static_cast<int32_t>(op.getNextBd());
        k.acquireLockId = static_cast<int32_t>(op.getAcquireLockId());
        k.releaseLockId = static_cast<int32_t>(op.getReleaseLockId());

        auto it = configBds.find(k);
        if (it == configBds.end()) {
            configBds[k].insert({col, row});
            return;
        }
        it->second.insert({col, row});
        // Duplicate. Do NOT op.erase() here: we are inside a walk (erasing the
        // op being visited invalidates the iterator), and the BD still has
        // users -- its result feeds a chained BD / create_io, and
        // ConfigDmaBdOp's verifier rejects an unused bd_handle, so the whole
        // downstream group has to come with it. Collect and handle after.
        duplicates.push_back(op);
    });

    // Erase each duplicate BD together with everything that uses it.
    //
    // A bare op.erase() aborts: MLIR forbids destroying an op that still has
    // users, and a duplicate BD's bd_handle feeds a chained BD (as linked_bd),
    // that chain's config.create_io, and the io's schedule.start_io. Even if it
    // did not abort, ConfigDmaBdOp/ConfigCreateIoOp both verify that their
    // result is used, so a half-erased group fails verification.
    //
    // So: take the forward slice (the op plus every op transitively reachable
    // through its results), then erase in REVERSE discovery order -- users are
    // discovered after their defs, so reversing puts users first and every op
    // is use-empty by the time it is destroyed.
    //
    // Operand-side defs (declaretile, arith.constant, bind_core_buffer,
    // memref.alloc, getbdid) are offered afterwards and removed ONLY once
    // genuinely unused: a tile's ping and pong BDs share a declaretile, and on
    // a partially-duplicated tile the survivor still needs it.
    llvm::SmallPtrSet<Operation *, 32> visited;
    SmallVector<Operation *> order;       // forward slice, defs-before-users
    SmallVector<Operation *> operandDefs; // candidates for dead-def cleanup

    for (ConfigDmaBdOp dup : duplicates) {
        SmallVector<Operation *> worklist{dup.getOperation()};
        while (!worklist.empty()) {
            Operation *cur = worklist.pop_back_val();
            if (!visited.insert(cur).second)
                continue;
            order.push_back(cur);
            for (Value res : cur->getResults())
                for (Operation *user : res.getUsers())
                    worklist.push_back(user);
        }
        for (Value operand : dup->getOperands())
            if (Operation *def = operand.getDefiningOp())
                operandDefs.push_back(def);
    }

    size_t erased = 0;
    for (Operation *op : llvm::reverse(order)) {
        if (!op->use_empty()) {
            // Something outside the slice still refers to it -- a survivor's
            // operand, most likely. Leaving it is correct; erasing would abort.
            op->emitWarning("kernel aggregation: skipping erase, op still has uses outside the duplicate group");
            continue;
        }
        op->erase();
        ++erased;
    }

    // Fixpoint over the now-possibly-dead operand defs: erasing a declaretile
    // orphans nothing further, but erasing a bind_core_buffer can orphan its
    // memref_mapping and then its memref.alloc, so repeat until quiescent.
    //
    // Pop from the worklist rather than iterating it -- the loop appends the
    // erased op's own operand defs, and a range-for over a growing SmallVector
    // invalidates its iterators/references on reallocation.
    llvm::SmallPtrSet<Operation *, 32> destroyed;
    while (!operandDefs.empty()) {
        Operation *def = operandDefs.pop_back_val();
        // A def can be queued more than once (the ping and pong BDs of one tile
        // share a declaretile, and each pushes it). Without this guard the
        // second pop would erase freed memory.
        if (!def || destroyed.count(def) || visited.count(def) || !def->use_empty())
            continue;
        for (Value operand : def->getOperands())
            if (Operation *d2 = operand.getDefiningOp())
                operandDefs.push_back(d2);
        destroyed.insert(def);
        def->erase();
        ++erased;
    }

    llvm::errs() << "[KernelAggregation] prototype: " << configBds.size() << " distinct BD config(s), "
                 << duplicates.size() << " duplicate BD(s), erased " << erased << " op(s) total\n";

    /*
    // ---- Phase 1: parse the per-tile plan -----------------------------------
    // The plan is the host path's published view of each MM2S tile. It is a
    // SECOND, independent derivation of packet_id / ooo_bd_id (the dma_bd ops
    // carry their own); Phase 4b cross-checks the two and fails on a mismatch.
    std::map<std::tuple<int, int, bool>, PlanEntry> plan;
    if (auto planAttr = kernelModuleOp->getAttrOfType<ArrayAttr>("dfschedule.core_offload_plan")) {
        for (Attribute a : planAttr) {
            auto d = dyn_cast<DictionaryAttr>(a);
            if (!d)
                continue;
            auto getI = [&](StringRef n, int32_t dflt) -> int32_t {
                if (auto i = d.getAs<IntegerAttr>(n))
                    return static_cast<int32_t>(i.getInt());
                return dflt;
            };
            auto getB = [&](StringRef n) -> bool {
                if (auto b = d.getAs<BoolAttr>(n))
                    return b.getValue();
                return false;
            };
            PlanEntry e;
            e.packetId = getI("packet_id", 0);
            e.oooBdId = getI("ooo_bd_id", -1);
            // Keyed (col,row,is_output). The plan itself is keyed by
            // (col,row,is_output,flow) upstream; a tile in two flows on the same
            // direction describes the same physical channel, so first wins --
            // matching the dedup DfscheduleToKernelApiPass already does.
            auto key = std::make_tuple(getI("col", 0), getI("row", 0), getB("is_output"));
            if (!plan.count(key))
                plan[key] = e;
        }
    }

    // ---- Phase 2: collect per-tile dma_bd records ---------------------------
    std::vector<BdRecord> records;
    bool collectFailed = false;
    kernelModuleOp.walk([&](ConfigDmaBdOp op) {
        if (collectFailed)
            return;
        // Already aggregated (pass run twice) -- nothing to do.
        if (op->hasAttr(kAggregatedAttr))
            return;
        auto td = op.getTile().getDefiningOp<DeclareTileOp>();
        if (!td)
            return; // not a core-tile BD group; leave alone
        BdRecord r;
        r.op = op;
        r.col = td.getCol();
        r.row = td.getRow();
        if (r.col < 0 || r.row < 0)
            return; // already a sentinel
        r.isOutput = bdIsOutput(op);
        if (!getConstantI32(op.getBdId(), r.bdId)) {
            op.emitError("kernel aggregation: bd_id is not an arith.constant; cannot identify the BD slot "
                         "for tile (")
                << r.col << "," << r.row << ")";
            collectFailed = true;
            return;
        }
        if (!getBufferOffset(op, r.bufOffset)) {
            op.emitError("kernel aggregation: dma_bd buffer does not come from a bind_core_buffer; "
                         "cannot identify the L1 buffer for tile (")
                << r.col << "," << r.row << ")";
            collectFailed = true;
            return;
        }
        records.push_back(r);
    });
    if (collectFailed)
        return signalPassFailure();
    if (records.empty())
        return;

    // ---- Phase 3: cluster ---------------------------------------------------
    // Insertion-ordered, so tile_coords follows IR order -- which is the order
    // the emitter turns into if/else-if arms. Preserving it keeps kernel.cc
    // byte-identical.
    std::map<ClusterKey, std::vector<BdRecord>> clusters;
    std::vector<ClusterKey> clusterOrder;
    for (const auto &r : records) {
        ClusterKey key{r.bdId, r.bufOffset, r.isOutput};
        if (!clusters.count(key))
            clusterOrder.push_back(key);
        clusters[key].push_back(r);
    }

    // ---- Phase 4: strict verification + rewrite -----------------------------
    OpBuilder builder(mod.getContext());
    size_t aggregatedGroups = 0, erasedOps = 0;

    // One self-tile for the whole kernel module, created at the TOP of the body
    // so it dominates every aggregated group regardless of which cluster is
    // rewritten first. Created lazily: if no cluster ends up merging, we must
    // not leave an unused op behind (ops here are only erased when dead, and a
    // stray tile handle with no users is noise in the dump).
    Value selfTile;
    auto getSelfTile = [&]() -> Value {
        if (!selfTile) {
            OpBuilder b(kernelModuleOp.getContext());
            b.setInsertionPointToStart(&kernelModuleOp.getBody().front());
            selfTile = b.create<DeclareTileSelfOp>(kernelModuleOp.getLoc(), TileType::get(b.getContext())).getTile();
        }
        return selfTile;
    };
    // Redundant dma_bd ops, erased in one fixpoint sweep after every cluster has
    // been stamped (see Phase 5 for why this cannot be done per-cluster).
    SmallVector<Operation *> doomed;

    for (const ClusterKey &key : clusterOrder) {
        auto &members = clusters[key];
        if (members.size() < 2)
            continue; // nothing to merge

        const BdRecord &first = members.front();
        ConfigDmaBdOp firstOp = first.op;

        // 4a. Every non-varying attribute AND operand must match across the cluster.
        for (size_t i = 1; i < members.size(); ++i) {
            ConfigDmaBdOp other = members[i].op;

            // Operands first. packet_id / out_of_order_bd_id are the only two
            // allowed to differ; every other operand must be constant-equal.
            // Comparing Values directly would be wrong -- each tile has its own
            // arith.constant / buffer op -- so compare what they denote.
            if (other->getNumOperands() != firstOp->getNumOperands()) {
                other.emitError("kernel aggregation: cannot merge core DMA config -- tiles (")
                    << first.col << "," << first.row << ") and (" << members[i].col << "," << members[i].row
                    << ") have different operand counts (" << firstOp->getNumOperands() << " vs "
                    << other->getNumOperands() << "), e.g. one has a linked_bd and the other does not";
                collectFailed = true;
            } else {
                for (unsigned oi = 0; oi < firstOp->getNumOperands(); ++oi) {
                    if (isPerTileVaryingOperand(oi))
                        continue;
                    Value a = firstOp->getOperand(oi), b = other->getOperand(oi);
                    if (a == b)
                        continue;
                    // Same SSA value is trivially equal; otherwise both must be
                    // constants of the same value. The tile and buffer operands
                    // legitimately differ per tile and are already keyed on by
                    // the cluster key (bd_id slot + L1 offset + direction), so
                    // only compare the i32 scalars here.
                    int32_t av = 0, bv = 0;
                    bool aConst = getConstantI32(a, av), bConst = getConstantI32(b, bv);
                    if (aConst && bConst) {
                        if (av == bv)
                            continue;
                    } else if (!isa<IntegerType>(a.getType())) {
                        // buffer / tile / linked_bd: identity handled by the
                        // cluster key, not comparable across tiles.
                        continue;
                    }
                    other.emitError("kernel aggregation: cannot merge core DMA config -- tiles (")
                        << first.col << "," << first.row << ") and (" << members[i].col << "," << members[i].row
                        << ") differ in operand #" << oi
                        << ". One kernel ELF runs on every core tile, so this configuration cannot be "
                           "expressed as a single BD program.";
                    collectFailed = true;
                }
            }

            for (NamedAttribute na : firstOp->getAttrs()) {
                StringRef name = na.getName().strref();
                Attribute otherVal = other->getAttr(name);
                if (otherVal != na.getValue()) {
                    other.emitError("kernel aggregation: cannot merge core DMA config -- tiles (")
                        << first.col << "," << first.row << ") and (" << members[i].col << "," << members[i].row
                        << ") differ in '" << name
                        << "'. One kernel ELF runs on every core tile, so this configuration cannot be "
                           "expressed as a single BD program.";
                    collectFailed = true;
                }
            }
            // Guard the reverse direction too: an attribute present only on the
            // other op would otherwise slip through the loop above.
            for (NamedAttribute na : other->getAttrs()) {
                StringRef name = na.getName().strref();
                if (!firstOp->getAttr(name)) {
                    other.emitError("kernel aggregation: cannot merge core DMA config -- tile (")
                        << members[i].col << "," << members[i].row << ") has attribute '" << name
                        << "' which tile (" << first.col << "," << first.row << ") does not";
                    collectFailed = true;
                }
            }
        }
        if (collectFailed)
            return signalPassFailure();

        // 4b. Build the per-tile arrays, cross-checking against the plan.
        SmallVector<Attribute> coords, packetIds, oooBdIds;
        for (auto &m : members) {
            coords.push_back(builder.getI32ArrayAttr({m.col, m.row}));
            if (!m.isOutput)
                continue;
            auto it = plan.find(std::make_tuple(m.col, m.row, true));
            if (it == plan.end()) {
                m.op.emitError("kernel aggregation: MM2S tile (")
                    << m.col << "," << m.row
                    << ") has no entry in dfschedule.core_offload_plan; the host path "
                       "(helper/flowtransfer_kernel.cpp) must publish every MM2S tile before the core "
                       "can self-program it";
                collectFailed = true;
                continue;
            }
            // packet_id and ooo_bd_id exist in BOTH the op and the plan, and the
            // two are derived independently. Disagreement means one derivation
            // drifted, so stop rather than silently picking a winner -- the wrong
            // ooo id sends a tile's output to another tile's shim BD, which shows
            // up as a data mismatch on hardware, not a build failure.
            // packet_id / out_of_order_bd_id are operands now, and the kernel path
            // has no runtime for them: their values are string-formatted into
            // kernel.cc. A non-constant here cannot be lowered at all, so fail
            // loudly rather than substitute a default -- same contract as bd_id.
            int32_t irPacketId = 0;
            if (!getConstantI32(m.op.getPacketId(), irPacketId)) {
                m.op.emitError("kernel aggregation: packet_id is not an arith.constant for tile (")
                    << m.col << "," << m.row
                    << "); the kernel path bakes this value into kernel.cc and has no runtime path for it";
                collectFailed = true;
                continue;
            }
            if (irPacketId != it->second.packetId) {
                m.op.emitError("kernel aggregation: packet_id mismatch for tile (")
                    << m.col << "," << m.row << "): dma_bd says " << irPacketId << ", core_offload_plan says "
                    << it->second.packetId;
                collectFailed = true;
                continue;
            }
            int32_t irOooBdId = 0;
            if (!getConstantI32(m.op.getOutOfOrderBdId(), irOooBdId)) {
                m.op.emitError("kernel aggregation: out_of_order_bd_id is not an arith.constant for tile (")
                    << m.col << "," << m.row
                    << "); the kernel path bakes this value into kernel.cc and has no runtime path for it";
                collectFailed = true;
                continue;
            }
            if (irOooBdId != it->second.oooBdId) {
                m.op.emitError("kernel aggregation: out_of_order_bd_id mismatch for tile (")
                    << m.col << "," << m.row << "): dma_bd says " << irOooBdId << ", core_offload_plan says "
                    << it->second.oooBdId
                    << ". Both must name the same shim S2MM BD; see the ooo_bd_id provenance note in "
                       "helper/flowtransfer_kernel.cpp";
                collectFailed = true;
                continue;
            }
            packetIds.push_back(builder.getI32IntegerAttr(it->second.packetId));
            oooBdIds.push_back(builder.getI32IntegerAttr(it->second.oooBdId));
        }
        if (collectFailed)
            return signalPassFailure();

        // 4c. Stamp the aggregate onto the FIRST group and retarget it to the
        // sentinel tile. Rewriting in place (rather than creating a new group
        // and erasing all N) keeps the buffer/constant operand chain valid and
        // preserves this group's position in the body.
        firstOp->setAttr(kAggregatedAttr, builder.getBoolAttr(true));
        firstOp->setAttr(kTileCoordsAttr, builder.getArrayAttr(coords));
        if (first.isOutput) {
            firstOp->setAttr(kTilePacketIdsAttr, builder.getArrayAttr(packetIds));
            firstOp->setAttr(kTileOooBdIdsAttr, builder.getArrayAttr(oooBdIds));
        }

        // Retarget every user of this tile handle to the module-wide self-tile.
        // The surviving group is no longer bound to a location: one ELF runs on
        // every core tile and discovers its own (col,row) via get_coreid(), so
        // the covered tiles live in tile_coords instead.
        //
        // replaceAllUsesWith (rather than rewriting just this dma_bd) moves the
        // ping/pong sibling, the create_io and the getbdid in one step -- they
        // all share this tile handle, and a half-retargeted group would leave a
        // coordinate-bearing tile alive with no BDs on it.
        if (auto firstTile = firstOp.getTile().getDefiningOp<DeclareTileOp>()) {
            firstTile.getTile().replaceAllUsesWith(getSelfTile());
            // The surviving group's own declaretile is now unused. It is NOT
            // reachable from any doomed BD, so the Phase 5 sweep would never see
            // it -- offer it explicitly or a dead coordinate-bearing tile is left
            // in the dump, which is exactly the confusion this op removes.
            doomed.push_back(firstTile.getOperation());
        }

        // 4d. Mark the redundant tile groups for deletion. The actual erase is a
        // fixpoint pass after ALL clusters are stamped -- see below for why.
        for (size_t i = 1; i < members.size(); ++i)
            doomed.push_back(members[i].op.getOperation());

        ++aggregatedGroups;
    }

    // ---- Phase 5: fixpoint erase -------------------------------------------
    // Erasing cannot be done per-cluster, because the BDs of one tile form a
    // CHAIN: the pong dma_bd's result is the ping dma_bd's linked_bd operand,
    // and only the ping feeds create_io. Ping and pong live in different
    // clusters (different bd_id and buffer), so erasing cluster-by-cluster
    // deletes a tile's ping while its pong still exists -- leaving the pong with
    // no users, which ConfigDmaBdOp's verifier rejects.
    //
    // Instead: collect every doomed dma_bd plus everything downstream of it,
    // then erase repeatedly until no op can be removed. An op is removable once
    // all of its results are unused, so the chain unwinds in dependency order
    // regardless of the order the ops were collected in.
    SmallVector<Operation *> eraseSet;
    llvm::SmallPtrSet<Operation *, 32> inSet;
    auto addToEraseSet = [&](Operation *op) {
        if (op && inSet.insert(op).second)
            eraseSet.push_back(op);
    };
    for (Operation *d : doomed) {
        addToEraseSet(d);
        // Everything fed by this BD: the chained sibling BD, its create_io, and
        // that io's start_io.
        SmallVector<Operation *> worklist;
        for (auto *u : d->getResults().front().getUsers())
            worklist.push_back(u);
        while (!worklist.empty()) {
            Operation *u = worklist.pop_back_val();
            if (!isa<ConfigDmaBdOp, ConfigCreateIoOp, StartIoOp>(u))
                continue;
            addToEraseSet(u);
            for (Value res : u->getResults())
                for (auto *uu : res.getUsers())
                    worklist.push_back(uu);
        }
        // Operand-side defs (declaretile / bind_core_buffer / getbdid chain)
        // become dead once the BD goes; offer them to the fixpoint, which only
        // removes them if nothing else still refers to them. The surviving
        // aggregated group keeps its own operands alive this way.
        for (Value operand : d->getOperands())
            addToEraseSet(operand.getDefiningOp());
    }
    // getbdid ops are consumed by start_io; once those die the getbdid on a
    // doomed tile is dead too. Collect them via the tile they read.
    for (Operation *op : llvm::to_vector(eraseSet))
        if (auto bd = dyn_cast<ConfigDmaBdOp>(op))
            if (auto *tileDef = bd.getTile().getDefiningOp())
                for (Value res : tileDef->getResults())
                    for (auto *u : res.getUsers())
                        if (isa<GetBdIdOp>(u))
                            addToEraseSet(u);

    bool progress = true;
    while (progress) {
        progress = false;
        for (Operation *&op : eraseSet) {
            if (!op)
                continue;
            if (!op->use_empty())
                continue;
            op->erase();
            op = nullptr;
            ++erasedOps;
            progress = true;
        }
    }

    llvm::errs() << "[KernelAggregation] aggregated " << aggregatedGroups << " BD group(s), erased " << erasedOps
                 << " redundant op(s)\n";
    */
}
