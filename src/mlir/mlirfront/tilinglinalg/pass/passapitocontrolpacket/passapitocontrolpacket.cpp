/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passapitocontrolpacket.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <optional>

using namespace mlir;
using namespace dfschedule;

static constexpr int kCtrlPktBcastId = 0x10;

void APIToControlPacketPass::runOnOperation() {
    if (phase == Phase::Collect)
        runCollect(getOperation());
    else
        runEmit(getOperation());
}

static bool aotEnabled(ModuleOp mod) {
    auto modeAttr = mod->getAttrOfType<IntegerAttr>("routing.control_packet_mode");
    auto ctrlAttr = mod->getAttrOfType<IntegerAttr>("routing.control_plan_op_control_packet");
    return modeAttr && modeAttr.getInt() != 0 && ctrlAttr && ctrlAttr.getInt() != 0;
}

void APIToControlPacketPass::runCollect(ModuleOp mod) {
    OpBuilder b(mod.getContext());
    mod->setAttr("routing.control_packet_aot", b.getI64IntegerAttr(0));
    if (!aotEnabled(mod))
        return;

    LoadKernelGroupOp loadOp;
    mod.walk([&](LoadKernelGroupOp op) { loadOp = op; });
    if (!loadOp) {
        llvm::errs() << "[TilingLinalg] WARNING: control_packet_mode(aot) but no load_kernel_group op; using JIT.\n";
        return;
    }
    int nresp = static_cast<int>(loadOp.getTiles().size());
    if (nresp <= 0) {
        llvm::errs() << "[TilingLinalg] WARNING: load_kernel_group has no tiles; AOT control packets disabled.\n";
        return;
    }
    loadOp->setAttr("aot_ctrl_packet", b.getUnitAttr());
    mod->setAttr("routing.control_packet_aot", b.getI64IntegerAttr(1));
    mod->setAttr("routing.control_packet_nresp", b.getI64IntegerAttr(nresp));
    mod->setAttr("routing.control_packet_stream_id", b.getI64IntegerAttr(kCtrlPktBcastId));

    mod.walk([&](CtrlPlanInitOp op) {
        SmallVector<int64_t> v = {static_cast<int32_t>(op.getShimCol()), static_cast<int32_t>(op.getCtrlId()),
                                  static_cast<int32_t>(op.getRespS2mmCh()), static_cast<int64_t>(op.getRows().size())};
        for (Attribute r : op.getRows()) {
            auto d = cast<DictionaryAttr>(r);
            for (const char *k : {"row", "col_lo", "col_hi"})
                v.push_back(cast<IntegerAttr>(d.get(k)).getInt());
        }
        mod->setAttr("routing.ctrlpkt_plan", b.getI64ArrayAttr(v));
    });
    SmallVector<Attribute> writes;
    mod.walk([&](GroupRegWriteOp op) {
        SmallVector<int64_t> v = {op.getKind() == "broadcast" ? 1 : 2, static_cast<int32_t>(op.getRow()),
                                  static_cast<int64_t>(op.getTileAddr()), static_cast<int64_t>(op.getData().size())};
        for (Attribute d : op.getData())
            v.push_back(cast<IntegerAttr>(d).getInt());
        writes.push_back(b.getI64ArrayAttr(v));
    });
    mod->setAttr("routing.ctrlpkt_writes", b.getArrayAttr(writes));
    llvm::outs() << "[TilingLinalg] AOT control packets: nresp=" << nresp << " group writes=" << writes.size()
                 << "\n";
}

namespace {

struct Arg {
    enum Kind { Int, Opaque, Val } kind = Int;
    int64_t i = 0;
    std::string s;
    Value v;
};

std::optional<int64_t> constInt(Value v) {
    if (auto c = v.getDefiningOp<emitc::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
            return ia.getInt();
    return std::nullopt;
}

Arg fromValue(Value v) {
    Arg a;
    if (auto c = constInt(v)) {
        a.i = *c;
        return a;
    }
    a.kind = Arg::Val;
    a.v = v;
    return a;
}

SmallVector<Arg> callArgs(emitc::CallOpaqueOp op) {
    SmallVector<Arg> out;
    auto args = op.getArgs();
    if (!args) {
        for (Value v : op.getOperands())
            out.push_back(fromValue(v));
        return out;
    }
    for (Attribute at : *args) {
        Arg a;
        if (auto ia = dyn_cast<IntegerAttr>(at)) {
            if (ia.getType().isIndex()) {
                out.push_back(fromValue(op.getOperand(ia.getInt())));
                continue;
            }
            a.i = ia.getInt();
        } else if (auto oa = dyn_cast<emitc::OpaqueAttr>(at)) {
            a.kind = Arg::Opaque;
            a.s = oa.getValue().str();
        } else {
            a.kind = Arg::Opaque;
            a.s = "?";
        }
        out.push_back(a);
    }
    return out;
}

bool tileOf(Value v, int &col, int &row) {
    auto op = v.getDefiningOp<emitc::CallOpaqueOp>();
    if (!op || op.getCallee() != "XAie_TileLoc")
        return false;
    auto a = callArgs(op);
    if (a.size() < 2 || a[0].kind != Arg::Int || a[1].kind != Arg::Int)
        return false;
    col = static_cast<int>(a[0].i);
    row = static_cast<int>(a[1].i);
    return true;
}

void bufferOf(Value v, int &argIdx, int64_t &off) {
    argIdx = -1;
    off = 0;
    for (int guard = 0; guard < 16 && v; guard++) {
        if (auto ba = dyn_cast<BlockArgument>(v)) {
            argIdx = static_cast<int>(ba.getArgNumber());
            return;
        }
        auto op = v.getDefiningOp<emitc::CallOpaqueOp>();
        if (!op)
            return;
        auto a = callArgs(op);
        if (op.getCallee() == "__runtime_buffer_offset" && a.size() >= 2 && a[1].kind == Arg::Int)
            off += a[1].i;
        else if (op.getCallee() != "__runtime_buffer_arg")
            return;
        if (a.empty() || a[0].kind != Arg::Val)
            return;
        v = a[0].v;
    }
}

int dirOf(const Arg &a) { return (a.kind == Arg::Opaque && a.s == "DMA_MM2S") ? 1 : 0; }

struct IoInfo {
    int col = 0, row = 0, ch = 0, bd = 0, dir = 0;
};

struct WindowOps {
    Operation *begin = nullptr, *commit = nullptr;
    std::string fabric;
    SmallVector<Operation *> bds, chs, ios, starts;
    SmallVector<Value> bufs;
    Value dev;
    bool ok = true;
};

// Mirror of the runtime's rt_plain_shim_bd_is_1d(): a plain __Runtime_dma_bd_config
// on a shim tile with a word-aligned length of at most 1023 words is sent as the
// 1-D multidim BD, so it can be an AOT "bd" site. Keep the two tests identical.
bool plainShimBdAsSite(int row, const Arg &len) {
    return row == 0 && len.kind == Arg::Int && len.i > 0 && len.i % 4 == 0 && len.i / 4 <= 1023;
}

bool isWindowHelper(StringRef callee) {
    return callee == "XAie_TileLoc" || callee == "__runtime_buffer_arg" || callee == "__runtime_buffer_offset" ||
           callee.starts_with("__Runtime_core_trace_");
}

bool usersAll(Operation *op, const SmallVector<Operation *> &set) {
    for (Operation *u : op->getResult(0).getUsers())
        if (std::find(set.begin(), set.end(), u) == set.end())
            return false;
    return true;
}

void batchWindow(MLIRContext *ctx, WindowOps &w) {
    OpBuilder b(w.commit);
    Location loc = w.commit->getLoc();
    auto fabTy = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__Runtime_CtrlRowFabric"));
    Value fab = b.create<emitc::ConstantOp>(loc, fabTy, emitc::OpaqueAttr::get(ctx, w.fabric)).getResult();
    Value nbuf = b.create<emitc::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr((int32_t)w.bufs.size()))
                     .getResult();
    SmallVector<Value> operands = {w.dev, fab, nbuf};
    operands.append(w.bufs.begin(), w.bufs.end());
    b.create<emitc::VerbatimOp>(loc, "/* AOT control packets: this call sends the whole shim-BD window above (" +
                                         std::to_string(w.bds.size()) + " BD configs, " +
                                         std::to_string(w.chs.size()) + " out-of-order enables, " +
                                         std::to_string(w.starts.size()) +
                                         " startio) from host_ctrlpkt.h, listed call by call in "
                                         "host_ctrlpkt_win_calls; the startio became __Runtime_ioevent_make. */");
    b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_ctrl_aot_window", nullptr, nullptr,
                                  ValueRange(operands));
    auto dirTy = emitc::OpaqueType::get(ctx, "XAie_DmaDirection");
    for (Operation *st : w.starts) {
        auto io = callArgs(cast<emitc::CallOpaqueOp>(st))[1].v.getDefiningOp<emitc::CallOpaqueOp>();
        auto ia = callArgs(io);
        OpBuilder sb(st);
        Value ch = sb.create<emitc::ConstantOp>(st->getLoc(), sb.getI32Type(), sb.getI32IntegerAttr((int32_t)ia[2].i));
        Value bd = sb.create<emitc::ConstantOp>(st->getLoc(), sb.getI32Type(), sb.getI32IntegerAttr((int32_t)ia[3].i));
        Value dir = sb.create<emitc::ConstantOp>(st->getLoc(), dirTy,
                                                 emitc::OpaqueAttr::get(ctx, ia[4].kind == Arg::Opaque ? ia[4].s
                                                                                                      : "DMA_S2MM"))
                        .getResult();
        auto ev = sb.create<emitc::CallOpaqueOp>(st->getLoc(), st->getResult(0).getType(), "__Runtime_ioevent_make",
                                                 nullptr, nullptr, ValueRange{ia[0].v, ch, bd, dir});
        st->getResult(0).replaceAllUsesWith(ev.getResult(0));
        st->erase();
    }
    for (Operation *op : w.ios)
        op->erase();
    for (Operation *op : w.chs)
        op->erase();
    for (Operation *op : w.bds)
        op->erase();
    w.begin->erase();
    w.commit->erase();
}

}

void APIToControlPacketPass::runEmit(ModuleOp mod) {
    auto aot = mod->getAttrOfType<IntegerAttr>("routing.control_packet_aot");
    if (!aot || aot.getInt() == 0)
        return;
    std::string path = outputDir + "/ctrlpkt_sites.txt";
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "[TilingLinalg] WARNING: cannot write " << path << "; AOT sites stay JIT.\n";
        return;
    }
    os << "# aiehlc ctrlpkt sites v1: one runtime API call per line, in host.cc order\n";
    os << "gen " << aieGen << "\npartition " << partStartCol << " " << partNumCols << "\n";
    if (auto plan = mod->getAttrOfType<ArrayAttr>("routing.ctrlpkt_plan")) {
        os << "plan";
        for (Attribute a : plan)
            os << " " << cast<IntegerAttr>(a).getInt();
        os << "\n";
    }
    if (auto writes = mod->getAttrOfType<ArrayAttr>("routing.ctrlpkt_writes"))
        for (Attribute w : writes) {
            auto v = cast<ArrayAttr>(w);
            os << "write " << (cast<IntegerAttr>(v[0]).getInt() == 1 ? "bcast" : "row");
            for (unsigned i = 1; i < v.size(); i++)
                os << " " << cast<IntegerAttr>(v[i]).getInt();
            os << "\n";
        }

    int lo = 0x7FFF, hi = -1;
    if (auto plan = mod->getAttrOfType<ArrayAttr>("routing.ctrlpkt_plan"))
        for (unsigned r = 0; 4 + 3 * r + 2 < plan.size(); r++) {
            lo = std::min<int>(lo, cast<IntegerAttr>(plan[5 + 3 * r]).getInt());
            hi = std::max<int>(hi, cast<IntegerAttr>(plan[6 + 3 * r]).getInt());
        }
    std::map<Operation *, IoInfo> ios;
    WindowOps win;
    bool inWindow = false;
    unsigned ncalls = 0;
    auto inRange = [&](int col, int row) { return row == 0 && col >= lo && col <= hi; };
    mod.walk([&](Operation *op) {
        if (auto vb = dyn_cast<emitc::VerbatimOp>(op)) {
            StringRef t = vb.getValue();
            if (t.contains("__Runtime_ctrl_shim_bd_begin(")) {
                inWindow = true;
                win.begin = op;
                auto l = t.find('('), r = t.rfind(')');
                win.fabric = (l != StringRef::npos && r != StringRef::npos && r > l) ? t.slice(l + 1, r).str() : "";
                os << "shim_bd begin\n";
            } else if (t.contains("__Runtime_ctrl_shim_bd_commit(") && inWindow) {
                inWindow = false;
                win.commit = op;
                os << "shim_bd end\n";
            } else if (inWindow && !t.trim().starts_with("/*")) {
                win.ok = false;
            }
            return;
        }
        auto call = dyn_cast<emitc::CallOpaqueOp>(op);
        if (!call)
            return;
        StringRef callee = call.getCallee();
        auto a = callArgs(call);
        int col = 0, row = 0;
        if (callee == "__Runtime_launch_kernel_group_ctrl") {
            os << "core_enable\n";
        } else if (callee == "__Runtime_dma_createio_4" && a.size() >= 5 && a[0].kind == Arg::Val &&
                   tileOf(a[0].v, col, row)) {
            ios[op] = IoInfo{col, row, static_cast<int>(a[2].i), static_cast<int>(a[3].i), dirOf(a[4])};
            if (inWindow) {
                win.ios.push_back(op);
                win.ok &= inRange(col, row) && a[2].kind == Arg::Int && a[3].kind == Arg::Int;
            }
        } else if (!inWindow) {
            return;
        } else if (isWindowHelper(callee)) {
            return;
        } else if ((callee == "__Runtime_dma_bd_config_multidim" || callee == "__Runtime_dma_bd_config_multidim_ooo") &&
                   a.size() == 22 && a[1].kind == Arg::Val && tileOf(a[1].v, col, row)) {
            int argIdx = -1;
            int64_t off = 0;
            if (a[2].kind == Arg::Val)
                bufferOf(a[2].v, argIdx, off);
            os << "call " << (callee.ends_with("_ooo") ? "bd_ooo" : "bd") << " " << col << " " << row << " "
               << argIdx << " " << off;
            for (unsigned i = 3; i < 22; i++) {
                os << " " << (a[i].kind == Arg::Int ? a[i].i : -999999);
                win.ok &= a[i].kind == Arg::Int;
            }
            os << "\n";
            ncalls++;
            win.bds.push_back(op);
            win.bufs.push_back(a[2].v);
            win.ok &= inRange(col, row) && a[0].kind == Arg::Val && a[2].kind == Arg::Val;
            if (a[0].kind == Arg::Val)
                win.dev = a[0].v;
        } else if (callee == "__Runtime_dma_bd_config" && a.size() == 13 && a[1].kind == Arg::Val &&
                   tileOf(a[1].v, col, row) && plainShimBdAsSite(row, a[4])) {
            // A plain shim BD: the runtime sends it as the 1-D multidim BD
            // (aie_runtime.c, rt_plain_shim_bd_is_1d), so record it as that "bd"
            // site -- 10 plain args, then num_dims 1, stride 4, wrap len/4, zeros.
            // Without this the call fell into the else below, which unbatched the
            // whole window.
            int argIdx = -1;
            int64_t off = 0;
            if (a[2].kind == Arg::Val)
                bufferOf(a[2].v, argIdx, off);
            os << "call bd " << col << " " << row << " " << argIdx << " " << off;
            for (unsigned i = 3; i < 13; i++) {
                os << " " << (a[i].kind == Arg::Int ? a[i].i : -999999);
                win.ok &= a[i].kind == Arg::Int;
            }
            os << " 1 4 " << a[4].i / 4 << " 0 0 0 0 0 0\n";
            ncalls++;
            win.bds.push_back(op);
            win.bufs.push_back(a[2].v);
            win.ok &= inRange(col, row) && a[0].kind == Arg::Val && a[2].kind == Arg::Val;
            if (a[0].kind == Arg::Val)
                win.dev = a[0].v;
        } else if (callee == "__Runtime_dma_channel_enable_ooo" && a.size() >= 4 && a[1].kind == Arg::Val &&
                   tileOf(a[1].v, col, row)) {
            os << "call ch_ooo " << col << " " << row << " " << a[2].i << " " << dirOf(a[3]) << "\n";
            ncalls++;
            win.chs.push_back(op);
            win.ok &= inRange(col, row) && a[2].kind == Arg::Int;
        } else if (callee == "__Runtime_startio" && a.size() >= 4 && a[1].kind == Arg::Val) {
            auto it = ios.find(a[1].v.getDefiningOp());
            if (it == ios.end())
                return;
            const IoInfo &io = it->second;
            os << "call start " << io.col << " " << io.row << " " << io.ch << " " << io.dir << " " << io.bd << " "
               << a[3].i << "\n";
            ncalls++;
            win.starts.push_back(op);
            win.ok &= inRange(io.col, io.row) && a[3].kind == Arg::Int;
        } else {
            win.ok = false;
        }
    });
    bool batch = win.ok && win.begin && win.commit && !win.fabric.empty() && win.dev && !win.bds.empty();
    for (Operation *bd : win.bds)
        batch &= usersAll(bd, win.ios);
    for (Operation *io : win.ios)
        batch &= usersAll(io, win.starts);
    if (batch) {
        os << "shim_bd batch " << win.bufs.size() << "\n";
        os.flush();
        batchWindow(mod.getContext(), win);
        llvm::outs() << "[TilingLinalg] AOT shim-BD window batched into __Runtime_ctrl_aot_window (" << ncalls
                     << " calls, " << win.bufs.size() << " buffers)\n";
    }
    llvm::outs() << "[TilingLinalg] wrote AOT control-packet sites: " << path << " (" << ncalls
                 << " shim-BD window calls)\n";
}
