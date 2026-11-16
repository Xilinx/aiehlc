/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "passstartiointerleave.h"
#include "passmemtilejoin/passmemtilejoin.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>

using namespace mlir;

static bool attrOn(ModuleOp mod, StringRef name) {
    auto a = mod->getAttrOfType<IntegerAttr>(name);
    return a && a.getInt() != 0;
}

// Latest op in `block` defining one of `op`'s operands, or null when every
// operand is a block argument / defined outside the block.
static Operation *latestOperandDef(Operation *op, Block *block) {
    Operation *latest = nullptr;
    for (Value v : op->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != block)
            continue;
        if (!latest || latest->isBeforeInBlock(def))
            latest = def;
    }
    return latest;
}

static bool isCall(Operation *op, StringRef callee) {
    auto c = dyn_cast<emitc::CallOpaqueOp>(op);
    return c && c.getCallee() == callee;
}

static bool isCallPrefix(Operation *op, StringRef prefix) {
    auto c = dyn_cast<emitc::CallOpaqueOp>(op);
    return c && c.getCallee().starts_with(prefix);
}

// createio(tile, desc) {args = [.., .., ch, bd, DMA_MM2S|DMA_S2MM]} on a row-0 tile.
static bool isShimS2mmCreateio(emitc::CallOpaqueOp c) {
    if (c.getCallee() != "__Runtime_dma_createio_4" || c.getNumOperands() < 1)
        return false;
    ArrayAttr args = c.getArgsAttr();
    auto dir = args && !args.empty() ? dyn_cast<emitc::OpaqueAttr>(args[args.size() - 1]) : emitc::OpaqueAttr();
    if (!dir || dir.getValue() != "DMA_S2MM")
        return false;
    auto loc = c.getOperand(0).getDefiningOp<emitc::CallOpaqueOp>();
    if (!loc || loc.getCallee() != "XAie_TileLoc" || !loc.getArgsAttr() || loc.getArgsAttr().size() != 2)
        return false;
    auto row = dyn_cast<IntegerAttr>(loc.getArgsAttr()[1]);
    return row && row.getInt() == 0;
}

// The contiguous run of ops ending at a shim S2MM createio that programs its
// channel: BD configs and OOO enables on the same tile value, the buffer-arg
// calls feeding them, then the createio and its startio.
static SmallVector<Operation *> outputGroup(emitc::CallOpaqueOp createio) {
    Value tile = createio.getOperand(0);
    SmallVector<Operation *> group;
    for (Operation *op = createio->getPrevNode(); op; op = op->getPrevNode()) {
        bool sameTile = llvm::is_contained(op->getOperands(), tile);
        bool mine = (isCallPrefix(op, "__Runtime_dma_bd_config") || isCall(op, "__Runtime_dma_channel_enable_ooo") ||
                     isCall(op, "__Runtime_memtile_join_shim_bd")) &&
                    sameTile;
        // Address operands: __runtime_buffer_arg and the pure emitc ops (constants,
        // offsets, casts) between it and the BD config, used only inside the group.
        bool pure = isCall(op, "__runtime_buffer_arg") || isCall(op, "__runtime_buffer_offset") ||
                    (op->getDialect() && op->getDialect()->getNamespace() == "emitc" && !isa<emitc::CallOpaqueOp>(op) &&
                     op->getNumResults() > 0);
        bool feeds = pure && !op->use_empty() &&
                     llvm::all_of(op->getUsers(), [&](Operation *u) { return llvm::is_contained(group, u); });
        if (!mine && !feeds)
            break;
        group.insert(group.begin(), op);
    }
    group.push_back(createio);
    for (Operation *u : createio->getUsers())
        if (isCall(u, "__Runtime_startio") && u->getBlock() == createio->getBlock())
            group.push_back(u);
    return group;
}

// Move every shim output channel's programming behind the last input start: the
// cores need their inputs first, and output BDs are not needed until a core has
// computed a tile, so programming them while the array computes takes them off
// the critical path. Safe when every use of a moved result lands after `before`.
static unsigned outputsLast(Block &block) {
    Operation *before = nullptr;
    for (Operation &op : block)
        if (isCall(&op, "__Runtime_wait")) {
            before = &op;
            break;
        }
    if (!before)
        return 0;
    SmallVector<SmallVector<Operation *>> groups;
    for (Operation &op : block) {
        if (&op == before)
            break;
        auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
        if (c && isShimS2mmCreateio(c))
            groups.push_back(outputGroup(c));
    }
    unsigned moved = 0;
    for (auto &g : groups) {
        bool ok = true;
        for (Operation *op : g)
            for (Operation *u : op->getUsers())
                if (!llvm::is_contained(g, u) && (u->getBlock() != &block || u->isBeforeInBlock(before)))
                    ok = false;
        if (!ok)
            continue;
        for (Operation *op : g)
            op->moveBefore(before);
        moved++;
    }
    return moved;
}

static IntegerAttr argInt(emitc::CallOpaqueOp c, unsigned i) {
    ArrayAttr a = c.getArgsAttr();
    return a && i < a.size() ? dyn_cast<IntegerAttr>(a[i]) : IntegerAttr();
}

// Memtile output join (doc/design/memtile_staging.md stage 1, #pragma memtile_staging(output)):
// a shim output channel fed by n packet-merged cores (n = its startio repeat count)
// loses its n out-of-order BDs and OOO enable; __Runtime_memtile_join_setup (hoisted
// before the kernel launch, i.e. setup) programs the memtile above it instead, and the
// run window keeps one linear shim BD (__Runtime_memtile_join_shim_bd) started once. The
// routing side (MemtileOutputJoin in tilinglinalg_pipeline.cpp) redirects the packets into
// the memtile DMA. Groups whose BDs are not exactly ids cbd .. cbd+n-1 are left alone.
static unsigned memtileOutputJoin(Block &block, bool cBlocked) {
    Operation *launch = nullptr;
    for (Operation &op : block)
        if (isCallPrefix(&op, "__Runtime_launch_kernel_group")) {
            launch = &op;
            break;
        }
    if (!launch)
        return 0;
    SmallVector<emitc::CallOpaqueOp> creates;
    for (Operation &op : block) {
        auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
        if (c && isShimS2mmCreateio(c) && launch->isBeforeInBlock(c))
            creates.push_back(c);
    }
    MLIRContext *ctx = block.getParentOp()->getContext();
    OpBuilder b(ctx);
    auto idx = [&](int64_t i) -> Attribute { return b.getIndexAttr(i); };
    auto i32 = [&](int64_t v) -> Attribute { return b.getI32IntegerAttr((int32_t)v); };
    unsigned joined = 0;
    for (emitc::CallOpaqueOp create : creates) {
        // createio / OOO enable / startio need not sit next to their BD chain, so find the
        // channel's out-of-order BDs directly: plain multidim BDs on the same shim tile with
        // ids cbd .. cbd+3 (the inputs use the _ooo variant).
        Value tile = create.getOperand(0);
        IntegerAttr ch = argInt(create, 2), cbd = argInt(create, 3);
        if (!ch || !cbd)
            continue;
        Operation *ooo = nullptr, *start = nullptr;
        for (Operation *u : create->getUsers())
            if (isCall(u, "__Runtime_startio"))
                start = u;
        if (!start)
            continue;
        auto startArgs = cast<emitc::CallOpaqueOp>(start).getArgsAttr();
        auto nAttr = startArgs ? dyn_cast<IntegerAttr>(startArgs[startArgs.size() - 1]) : IntegerAttr();
        // a literal, not an operand reference (those are index-typed)
        const int64_t n = nAttr && !nAttr.getType().isIndex() ? nAttr.getInt() : 0; // one out-of-order task per core
        if (n < 1 || n > 16)
            continue;
        SmallVector<emitc::CallOpaqueOp> bds;
        for (Operation &op : block) {
            auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
            if (!c || c.getNumOperands() < 2 || c.getOperand(1) != tile)
                continue;
            IntegerAttr id = argInt(c, 3), occh = argInt(c, 2);
            if (isCall(&op, "__Runtime_dma_bd_config_multidim") && id && id.getInt() >= cbd.getInt() &&
                id.getInt() < cbd.getInt() + n)
                bds.push_back(c);
            else if (isCall(&op, "__Runtime_dma_channel_enable_ooo") && occh && occh.getInt() == ch.getInt())
                ooo = &op;
        }
        if ((int64_t)bds.size() != n || !ooo)
            continue;
        IntegerAttr tileBytes = argInt(bds[0], 4);
        if (!tileBytes || tileBytes.getType().isIndex() || tileBytes.getInt() <= 0 ||
            (n * tileBytes.getInt()) % 1024 != 0)
            continue;
        Value dev = bds[0].getOperand(0);
        SmallVector<Value, 16> bufs;
        SmallVector<int64_t, 16> ids;
        for (emitc::CallOpaqueOp bd : bds) {
            if (bd.getNumOperands() < 3 || !argInt(bd, 4) || argInt(bd, 4).getInt() != tileBytes.getInt())
                break;
            ids.push_back(argInt(bd, 3).getInt());
            bufs.push_back(bd.getOperand(2));
        }
        if ((int64_t)ids.size() != n)
            continue;
        // Hoist each buffer's address computation (and the tile value) before the launch,
        // operands first, keeping program order.
        SmallVector<Operation *> hoist;
        std::function<void(Value)> collect = [&](Value v) {
            Operation *def = v.getDefiningOp();
            if (!def || def->getBlock() != &block || !launch->isBeforeInBlock(def) || llvm::is_contained(hoist, def))
                return;
            for (Value o : def->getOperands())
                collect(o);
            hoist.push_back(def);
        };
        collect(tile);
        for (Value v : bufs)
            collect(v);
        llvm::sort(hoist, [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
        for (Operation *op : hoist)
            op->moveBefore(launch);
        SmallVector<Value, 18> operands{dev, tile};
        operands.append(bufs.begin(), bufs.end());
        // setup(dev, shim, ch, n, tile_bytes, c_blocked, {bd id, DDR target} x n)
        SmallVector<Attribute> setupArgs{
            idx(0), idx(1), i32(ch.getInt()), i32(n), i32(tileBytes.getInt()), i32(cBlocked ? 1 : 0)};
        // shim_bd(dev, shim, bd, n, tile_bytes, DDR target x n)
        SmallVector<Attribute> shimArgs{idx(0), idx(1), i32(cbd.getInt()), i32(n), i32(tileBytes.getInt())};
        for (int64_t i = 0; i < n; i++) {
            setupArgs.push_back(i32(ids[i]));
            setupArgs.push_back(idx(2 + i));
            shimArgs.push_back(idx(2 + i));
        }
        b.setInsertionPoint(launch);
        b.create<emitc::CallOpaqueOp>(launch->getLoc(), TypeRange{}, "__Runtime_memtile_join_setup",
                                      b.getArrayAttr(setupArgs), ArrayAttr(), operands);
        b.setInsertionPoint(create);
        auto shimBd =
            b.create<emitc::CallOpaqueOp>(create.getLoc(), bds[0].getResultTypes(), "__Runtime_memtile_join_shim_bd",
                                          b.getArrayAttr(shimArgs), ArrayAttr(), operands);
        create->setOperand(1, shimBd.getResult(0));
        auto st = cast<emitc::CallOpaqueOp>(start);
        SmallVector<Attribute> sargs(st.getArgsAttr().begin(), st.getArgsAttr().end());
        sargs.back() = i32(1); // one row-block task instead of one per core
        st.setArgsAttr(b.getArrayAttr(sargs));
        ooo->erase();
        for (emitc::CallOpaqueOp bd : bds)
            bd->erase();
        joined++;
    }
    return joined;
}

// Memtile input staging (doc/design/memtile_staging.md stages 2-3, AIEHLC_MEMTILE_INPUT=1):
// each shim MM2S input channel gets __Runtime_memtile_stage_input_setup before the
// kernel launch; the run window is unchanged (same shim BDs and starts). The tensor
// kind comes from the BD's buffer: host_canonicalized(dev, A, B, C) -> argument 1 = A
// (kind 0), argument 2 = B^T (kind 1). The routing side is MemtileOutputJoinRoutingPass.
static unsigned memtileInputStage(Block &block) {
    Operation *launch = nullptr;
    for (Operation &op : block)
        if (isCallPrefix(&op, "__Runtime_launch_kernel_group")) {
            launch = &op;
            break;
        }
    if (!launch)
        return 0;
    MLIRContext *ctx = block.getParentOp()->getContext();
    OpBuilder b(ctx);
    unsigned staged = 0;
    SmallVector<emitc::CallOpaqueOp> creates;
    for (Operation &op : block) {
        auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
        if (!c || c.getCallee() != "__Runtime_dma_createio_4" || !launch->isBeforeInBlock(c))
            continue;
        ArrayAttr args = c.getArgsAttr();
        auto dir = args && !args.empty() ? dyn_cast<emitc::OpaqueAttr>(args[args.size() - 1]) : emitc::OpaqueAttr();
        auto loc = c.getOperand(0).getDefiningOp<emitc::CallOpaqueOp>();
        auto row = loc && loc.getArgsAttr() && loc.getArgsAttr().size() == 2
                       ? dyn_cast<IntegerAttr>(loc.getArgsAttr()[1])
                       : IntegerAttr();
        if (dir && dir.getValue() == "DMA_MM2S" && row && row.getInt() == 0)
            creates.push_back(c);
    }
    for (emitc::CallOpaqueOp create : creates) {
        Value tile = create.getOperand(0);
        IntegerAttr ch = argInt(create, 2), cbd = argInt(create, 3);
        if (!ch || !cbd)
            continue;
        // The channel's BD and the kernel argument its buffer comes from.
        int64_t kind = -1;
        Value dev;
        for (Operation &op : block) {
            auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
            if (!c || !c.getCallee().starts_with("__Runtime_dma_bd_config") || c.getNumOperands() < 3 ||
                c.getOperand(1) != tile || !argInt(c, 3) || argInt(c, 3).getInt() != cbd.getInt())
                continue;
            dev = c.getOperand(0);
            Value v = c.getOperand(2);
            for (int depth = 0; depth < 8 && v; depth++) {
                if (auto barg = dyn_cast<BlockArgument>(v)) {
                    kind = barg.getArgNumber() == 1 ? 0 : barg.getArgNumber() == 2 ? 1 : -1;
                    break;
                }
                Operation *def = v.getDefiningOp();
                v = def && def->getNumOperands() ? def->getOperand(0) : Value();
            }
        }
        if (kind < 0 || !dev)
            continue;
        if (Operation *def = tile.getDefiningOp())
            if (def->getBlock() == &block && launch->isBeforeInBlock(def))
                def->moveBefore(launch);
        b.setInsertionPoint(launch);
        b.create<emitc::CallOpaqueOp>(
            launch->getLoc(), TypeRange{}, "__Runtime_memtile_stage_input_setup",
            b.getArrayAttr({b.getIndexAttr(0), b.getIndexAttr(1), b.getI32IntegerAttr((int32_t)ch.getInt()),
                            b.getI32IntegerAttr((int32_t)kind)}),
            ArrayAttr(), ValueRange{dev, tile});
        staged++;
    }
    return staged;
}

// Memtile-resident weights (#pragma memtile_staging(weights)): every column-broadcast input
// channel (routing.memtile_weights_io; the BD reads kernel argument 2 = B) loses its shim BD,
// createio and startio in the run window; __Runtime_memtile_weights_start takes the same BD
// arguments plus channel and repeat, keeps the channel's chunk in the memtile above, and on a
// key hit (__Runtime_memtile_weights_key) moves only the per-call tail rows from DDR.
static bool isShimTile(Value tile);

static unsigned memtileWeightsStage(Block &block, ModuleOp mod) {
    unsigned staged = 0;
    SmallVector<emitc::CallOpaqueOp> starts;
    for (Operation &op : block)
        if (auto c = dyn_cast<emitc::CallOpaqueOp>(&op); c && c.getCallee() == "__Runtime_startio")
            starts.push_back(c);
    for (emitc::CallOpaqueOp start : starts) {
        auto create = start.getNumOperands() > 1 ? start.getOperand(1).getDefiningOp<emitc::CallOpaqueOp>()
                                                 : emitc::CallOpaqueOp();
        if (!create || create.getCallee() != "__Runtime_dma_createio_4" || create.getNumOperands() < 2)
            continue;
        ArrayAttr cargs = create.getArgsAttr();
        auto dir = cargs && !cargs.empty() ? dyn_cast<emitc::OpaqueAttr>(cargs[cargs.size() - 1]) : emitc::OpaqueAttr();
        if (!dir || dir.getValue() != "DMA_MM2S" || !isShimTile(create.getOperand(0)))
            continue;
        auto loc = create.getOperand(0).getDefiningOp<emitc::CallOpaqueOp>();
        IntegerAttr col = dyn_cast<IntegerAttr>(loc.getArgsAttr()[0]), ch = argInt(create, 2);
        if (!col || !ch || !isMemtileWeightsIo(mod, col.getInt(), ch.getInt()))
            continue;
        auto bd = create.getOperand(1).getDefiningOp<emitc::CallOpaqueOp>();
        if (!bd || bd.getCallee() != "__Runtime_dma_bd_config_multidim_ooo" || bd.getNumOperands() != 3 ||
            !bd->hasOneUse() || !create->hasOneUse())
            continue;
        ArrayAttr bargs = bd.getArgsAttr(), sargs = start.getArgsAttr();
        if (!bargs || bargs.size() != 22 || !sargs || sargs.empty())
            continue;
        OpBuilder b(start);
        SmallVector<Attribute> args(bargs.begin(), bargs.end()); // dev, tile, buf, bd .. iter_wrap
        args.push_back(b.getI32IntegerAttr((int32_t)ch.getInt()));
        args.push_back(sargs[sargs.size() - 1]); // repeat
        auto call =
            b.create<emitc::CallOpaqueOp>(start.getLoc(), start.getResultTypes(), "__Runtime_memtile_weights_start",
                                          b.getArrayAttr(args), ArrayAttr(), bd.getOperands());
        start->replaceAllUsesWith(call);
        start->erase();
        create->erase();
        bd->erase();
        staged++;
    }
    return staged;
}

// Synchronized input start: gather every input channel's startio (those ahead of the
// first shim output group) right before that group, so all inputs start within a few
// hundred ns of each other. Each B stream is broadcast to a column of cores and only
// advances when all of them have a free buffer, so a core row whose A starts late
// stalls the others (VEK280 KCYC: first-window waits staggered 8k..21k cycles by row).
static unsigned syncInputStarts(Block &block) {
    Operation *boundary = nullptr;
    for (Operation &op : block) {
        auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
        if (isCall(&op, "__Runtime_wait") || (c && isShimS2mmCreateio(c))) {
            boundary = &op;
            break;
        }
    }
    if (!boundary)
        return 0;
    // An output group starts with its BD configs: move the boundary up to the group's first op.
    if (!isCall(boundary, "__Runtime_wait"))
        boundary = outputGroup(cast<emitc::CallOpaqueOp>(boundary)).front();
    SmallVector<Operation *> starts;
    for (Operation &op : block) {
        if (&op == boundary)
            break;
        if (isCall(&op, "__Runtime_startio"))
            starts.push_back(&op);
    }
    for (Operation *op : starts)
        for (Operation *u : op->getUsers())
            if (u->getBlock() == &block && u->isBeforeInBlock(boundary))
                return 0;
    for (Operation *op : starts)
        op->moveBefore(boundary);
    return starts.size();
}

// Setup-time BD templates: the runtime builds one shim BD template per shape the first
// time it sees that shape, which put a full driver encode (+ cold code) into the first
// run (~5 us of 28 on the VEK280 GEMM). Clone the first shim BD config of each shape
// before the kernel launch between __Runtime_bd_template_prepare_begin/_end, where it
// only encodes the template (no register write); every run's BDs are then template
// hits. Shape key: callee + every argument except the BD id. AIEHLC_NO_BDT_PREPARE=1 off.
static bool isShimTile(Value tile) {
    auto loc = tile.getDefiningOp<emitc::CallOpaqueOp>();
    if (!loc || loc.getCallee() != "XAie_TileLoc" || !loc.getArgsAttr() || loc.getArgsAttr().size() != 2)
        return false;
    auto row = dyn_cast<IntegerAttr>(loc.getArgsAttr()[1]);
    return row && row.getInt() == 0;
}

static unsigned bdTemplatePrepare(Block &block) {
    Operation *launch = nullptr;
    for (Operation &op : block)
        if (isCallPrefix(&op, "__Runtime_launch_kernel_group")) {
            launch = &op;
            break;
        }
    if (!launch)
        return 0;
    SmallVector<Operation *> firsts;
    SmallVector<std::string> keys;
    for (Operation &op : block) {
        auto c = dyn_cast<emitc::CallOpaqueOp>(&op);
        if (!c || !launch->isBeforeInBlock(c) || c.getNumOperands() < 2 || !isShimTile(c.getOperand(1)))
            continue;
        StringRef callee = c.getCallee();
        unsigned bdArg = callee == "__Runtime_memtile_join_shim_bd" ? 2 : 3;
        if (callee != "__Runtime_dma_bd_config_multidim" && callee != "__Runtime_dma_bd_config_multidim_ooo" &&
            callee != "__Runtime_memtile_join_shim_bd")
            continue;
        std::string key = callee.str();
        llvm::raw_string_ostream os(key);
        if (ArrayAttr args = c.getArgsAttr())
            for (unsigned i = 0; i < args.size(); i++)
                if (i != bdArg)
                    os << "," << args[i];
        os.flush();
        if (llvm::is_contained(keys, key))
            continue;
        keys.push_back(key);
        firsts.push_back(c);
    }
    if (firsts.empty())
        return 0;
    // Operand address chains (tile locs, buffer args) move before the launch, as for the join.
    SmallVector<Operation *> hoist;
    std::function<void(Value)> collect = [&](Value v) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != &block || !launch->isBeforeInBlock(def) || llvm::is_contained(hoist, def))
            return;
        for (Value o : def->getOperands())
            collect(o);
        hoist.push_back(def);
    };
    for (Operation *op : firsts)
        for (Value v : op->getOperands())
            collect(v);
    llvm::sort(hoist, [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
    for (Operation *op : hoist)
        op->moveBefore(launch);
    OpBuilder b(launch);
    b.create<emitc::CallOpaqueOp>(launch->getLoc(), TypeRange{}, "__Runtime_bd_template_prepare_begin", ArrayAttr(),
                                  ArrayAttr(), ValueRange{});
    for (Operation *op : firsts)
        b.clone(*op);
    b.create<emitc::CallOpaqueOp>(launch->getLoc(), TypeRange{}, "__Runtime_bd_template_prepare_end", ArrayAttr(),
                                  ArrayAttr(), ValueRange{});
    return firsts.size();
}

void StartioInterleavePass::runOnOperation() {
    ModuleOp mod = getOperation();
    if (std::getenv("AIEHLC_NO_STARTIO_INTERLEAVE")) {
        llvm::errs() << "[TilingLinalg] StartioInterleavePass disabled (AIEHLC_NO_STARTIO_INTERLEAVE)\n";
        return;
    }
    if (attrOn(mod, "routing.control_packet_mode") || attrOn(mod, "routing.control_plan_shim_bd_ctrl")) {
        llvm::errs() << "[TilingLinalg] StartioInterleavePass skipped (AOT / shim-BD control window)\n";
        return;
    }
    SmallVector<emitc::CallOpaqueOp> starts;
    mod.walk([&](emitc::CallOpaqueOp call) {
        if (call.getCallee() == "__Runtime_startio")
            starts.push_back(call);
    });
    // Program order: each move lands after its own createio, so earlier starts
    // keep their relative order and none crosses a later channel's BD chain.
    unsigned moved = 0;
    for (emitc::CallOpaqueOp call : starts) {
        Operation *op = call.getOperation();
        Operation *after = latestOperandDef(op, op->getBlock());
        if (!after || after->getNextNode() == op)
            continue;
        op->moveAfter(after);
        moved++;
    }
    llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << moved << " of " << starts.size()
                 << " startio moved next to their createio\n";
    if (std::getenv("AIEHLC_NO_OUTPUTS_LAST"))
        return;
    SmallVector<Block *> blocks;
    for (emitc::CallOpaqueOp call : starts)
        if (!llvm::is_contained(blocks, call->getBlock()))
            blocks.push_back(call->getBlock());
    const int64_t stages = memtileStaging(mod);
    if (stages & kMemtileStageInput) {
        unsigned staged = 0;
        for (Block *blk : blocks)
            staged += memtileInputStage(*blk);
        llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << staged
                     << " shim input channel(s) staged through a memtile\n";
    }
    if ((stages & kMemtileStageWeights) && !(stages & kMemtileStageInput)) {
        unsigned staged = 0;
        for (Block *blk : blocks)
            staged += memtileWeightsStage(*blk, mod);
        llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << staged
                     << " weight input channel(s) resident in a memtile\n";
    }
    if (stages & kMemtileStageOutput) {
        unsigned joined = 0;
        for (Block *blk : blocks)
            joined += memtileOutputJoin(*blk, stages & kMemtileStageCBlocked);
        llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << joined
                     << " shim output channel(s) joined through a memtile\n";
    }
    // A shim tile has BDs 0-15; a larger id lands in the channel-control registers (skill
    // shimbdindexoverflow). A mesh row of more than ~6 cores per output channel overflows
    // the out-of-order BDs; #pragma memtile_staging(output) moves them to the memtile.
    bool bdOverflow = false;
    mod.walk([&](emitc::CallOpaqueOp c) {
        if (!c.getCallee().starts_with("__Runtime_dma_bd_config") || c.getNumOperands() < 2 ||
            !isShimTile(c.getOperand(1)))
            return;
        IntegerAttr id = argInt(c, 3);
        if (id && !id.getType().isIndex() && id.getInt() >= 16) {
            c.emitError() << "shim BD id " << id.getInt()
                          << " is past the shim's 16 BDs; too many out-of-order output BDs on one shim -- "
                             "add #pragma memtile_staging(output)";
            bdOverflow = true;
        }
    });
    if (bdOverflow)
        return signalPassFailure();
    unsigned groups = 0;
    for (Block *b : blocks)
        groups += outputsLast(*b);
    llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << groups
                 << " shim output channel(s) programmed after the inputs\n";
    if (!std::getenv("AIEHLC_NO_BDT_PREPARE")) {
        unsigned shapes = 0;
        for (Block *b : blocks)
            shapes += bdTemplatePrepare(*b);
        llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << shapes
                     << " shim BD template shape(s) built before the kernel launch\n";
    }
    // Off by default: on VEK280 holding the inputs back until every input BD is written
    // measured slower (40.6 / 33.2 us vs 36.7 / 29.6 us); host BD programming, not the
    // row stagger, bounds the run. AIEHLC_SYNC_INPUT_START=1 to try it.
    if (!std::getenv("AIEHLC_SYNC_INPUT_START"))
        return;
    unsigned synced = 0;
    for (Block *b : blocks)
        synced += syncInputStarts(*b);
    llvm::errs() << "[TilingLinalg] StartioInterleavePass: " << synced
                 << " input startio(s) issued together after the input BDs\n";
}
