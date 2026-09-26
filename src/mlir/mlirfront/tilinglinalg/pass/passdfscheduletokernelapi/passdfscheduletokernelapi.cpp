/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdfscheduletokernelapi.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace dfschedule;

namespace {

// ---------------------------------------------------------------------------
// Single pattern for dfschedule.module: convert body line-by-line (one emitc op
// per dfschedule op) then erase module. Always matches so conversion does not
// depend on nested ops being converted first (driver may try parent first).
// ---------------------------------------------------------------------------

/// Window info from WindowDefOp for emitting correct buffer/lock names.
struct WindowInfo {
    std::string pingBuffer;
    std::string pongBuffer;
    std::string acquireLock;
    std::string releaseLock;
    int32_t bufferSize = 0;    // Per-window buffer size from window_def attribute
    int32_t numRounds = 0;     // Number of ping-pong rounds (0 = use bufferSize as fallback)
    std::string direction;     // "in" or "out"
    int32_t channel = 0;       // Core-tile DMA channel (S2MM in / MM2S out)
    bool singleBuffer = false; // Single-buffer (no pong) mode
    // KERNELCONFIGOFFLOAD resources resolved by BlueprintToScheduleKernelPass.
    // Present on both input and output windows under offload; -1 = not provided.
    int32_t pingBdId = -1;
    int32_t pongBdId = -1;
    int32_t acquireLockHwId = -1; // 0..15 hardware lock index for MMIO encoding
    int32_t releaseLockHwId = -1; // (acquireLock/releaseLock above stay the 48+N
                                  // intrinsic names used by kernel builtins)
};

/// KERNELCONFIGOFFLOAD per-tile entry, mirrored out of the kernel module's
/// `dfschedule.core_offload_plan` array attr (published by the host path via
/// ResourceMgr). One per (col,row,direction).
///
/// This exists because MM2S core config is NOT uniform across tiles: packet_id
/// and ooo_bd_id differ per tile, and there is only one kernel.cc/ELF broadcast
/// to all of them. The emitted code therefore branches on get_coreid().
struct CoreOffloadEntry {
    int32_t col = -1;
    int32_t row = -1;
    bool isOutput = false;
    int32_t channel = 0;
    int32_t packetId = 0;
    bool enablePacket = false;
    int32_t oooBdId = -1;
    int32_t bdLenBytes = 0;
    int32_t ppDepth = 2;
};

/// Read the per-tile MM2S fan-out off the AGGREGATED dma_bd ops that
/// DfscheduleKernelAggregationPass leaves in the kernel module body.
///
/// That pass collapses the 16 (or N) identical per-tile BD groups into one and
/// records what varied as discardable array attributes:
///
///     aggregated      = true
///     tile_coords     = [[col,row], ...]
///     tile_packet_ids = [id, ...]        // parallel to tile_coords
///     tile_ooo_bd_ids = [bd, ...]        // parallel to tile_coords
///
/// Only output (packet-enabled) groups carry the id arrays -- S2MM is uniform
/// across tiles and needs no dispatch. Leaves `out` empty when the aggregation
/// pass did not run, so the caller can fall back to core_offload_plan.
static void collectAggregatedOutTiles(KernelModuleOp kernelModuleOp, SmallVectorImpl<CoreOffloadEntry> &out) {
    for (Operation &inner : kernelModuleOp.getBody().front()) {
        auto bd = dyn_cast<ConfigDmaBdOp>(&inner);
        if (!bd)
            continue;
        auto agg = bd->getAttrOfType<BoolAttr>("aggregated");
        if (!agg || !agg.getValue())
            continue;
        auto coords = bd->getAttrOfType<ArrayAttr>("tile_coords");
        auto pkts = bd->getAttrOfType<ArrayAttr>("tile_packet_ids");
        auto ooos = bd->getAttrOfType<ArrayAttr>("tile_ooo_bd_ids");
        // Output groups only: an S2MM aggregate has no id arrays.
        if (!coords || !pkts || !ooos)
            continue;
        if (pkts.size() != coords.size() || ooos.size() != coords.size())
            continue;
        for (size_t i = 0; i < coords.size(); ++i) {
            auto pair = dyn_cast<ArrayAttr>(coords[i]);
            if (!pair || pair.size() != 2)
                continue;
            CoreOffloadEntry e;
            e.col = static_cast<int32_t>(cast<IntegerAttr>(pair[0]).getInt());
            e.row = static_cast<int32_t>(cast<IntegerAttr>(pair[1]).getInt());
            e.isOutput = true;
            e.enablePacket = bd.getEnablePacket();
            e.packetId = static_cast<int32_t>(cast<IntegerAttr>(pkts[i]).getInt());
            e.oooBdId = static_cast<int32_t>(cast<IntegerAttr>(ooos[i]).getInt());
            e.bdLenBytes = static_cast<int32_t>(bd.getLen());
            bool seen = false;
            for (const auto &o : out)
                if (o.col == e.col && o.row == e.row)
                    seen = true;
            if (!seen)
                out.push_back(e);
        }
        // One output window == one aggregated MM2S group; the first is enough.
        if (!out.empty())
            return;
    }
}

/// dfschedule.module -> convert entire body line-by-line then erase module.
/// Always matches and does full conversion so we do not depend on nested patterns
/// firing first (conversion may try the parent before descending into regions).
struct KernelModuleToEmitCPattern : public OpConversionPattern<KernelModuleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(KernelModuleOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Block &body = op.getBody().front();

        // KERNELCONFIGOFFLOAD gate: the core self-configures its incoming S2MM
        // DMA (BD chain + lock inits + channel-start) from kernel.cc via raw
        // MMIO instead of the host programming it over the config bus.
        // BlueprintToScheduleKernelPass owns that decision (and the BD/lock ids
        // it implies) and stamps it here, so this pass only formats the result.
        bool offloadOn = false;
        if (auto a = op->getAttrOfType<IntegerAttr>("dfschedule.kernel_config_offload"))
            offloadOn = a.getInt() != 0;

        // Build map: window symbol name -> (ping_buffer, pong_buffer, acquire_lock, release_lock)
        llvm::StringMap<WindowInfo> windowInfoMap;
        for (Operation &inner : body) {
            if (auto windowDefOp = dyn_cast<WindowDefOp>(&inner)) {
                DictionaryAttr winAttrs = windowDefOp.getWindowAttrs();
                WindowInfo info;
                if (auto a = winAttrs.getAs<SymbolRefAttr>("ping_buffer"))
                    info.pingBuffer = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("pong_buffer"))
                    info.pongBuffer = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("acquire_lock"))
                    info.acquireLock = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("release_lock"))
                    info.releaseLock = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<IntegerAttr>("buffer_size"))
                    info.bufferSize = a.getInt();
                if (auto a = winAttrs.getAs<IntegerAttr>("num_rounds"))
                    info.numRounds = a.getInt();
                if (auto a = winAttrs.getAs<StringAttr>("direction"))
                    info.direction = a.getValue().str();
                if (auto a = winAttrs.getAs<IntegerAttr>("dma_channel"))
                    info.channel = a.getInt();
                if (auto a = winAttrs.getAs<BoolAttr>("single_buffer"))
                    info.singleBuffer = a.getValue();
                if (auto a = winAttrs.getAs<IntegerAttr>("ping_bd_id"))
                    info.pingBdId = a.getInt();
                if (auto a = winAttrs.getAs<IntegerAttr>("pong_bd_id"))
                    info.pongBdId = a.getInt();
                if (auto a = winAttrs.getAs<IntegerAttr>("acquire_lock_hw_id"))
                    info.acquireLockHwId = a.getInt();
                if (auto a = winAttrs.getAs<IntegerAttr>("release_lock_hw_id"))
                    info.releaseLockHwId = a.getInt();
                windowInfoMap[windowDefOp.getSymName().str()] = info;
            }
        }

        // KERNELCONFIGOFFLOAD per-tile plan (see CoreOffloadEntry).
        SmallVector<CoreOffloadEntry> offloadPlan;
        if (auto planAttr = op->getAttrOfType<ArrayAttr>("dfschedule.core_offload_plan")) {
            for (Attribute a : planAttr) {
                auto d = dyn_cast<DictionaryAttr>(a);
                if (!d)
                    continue;
                CoreOffloadEntry e;
                auto geti = [&](StringRef k, int32_t dflt) -> int32_t {
                    if (auto v = d.getAs<IntegerAttr>(k))
                        return v.getInt();
                    return dflt;
                };
                auto getb = [&](StringRef k) -> bool {
                    if (auto v = d.getAs<BoolAttr>(k))
                        return v.getValue();
                    return false;
                };
                e.col = geti("col", -1);
                e.row = geti("row", -1);
                e.isOutput = getb("is_output");
                e.channel = geti("channel", 0);
                e.packetId = geti("packet_id", 0);
                e.enablePacket = getb("enable_packet");
                e.oooBdId = geti("ooo_bd_id", -1);
                e.bdLenBytes = geti("bd_len_bytes", 0);
                e.ppDepth = geti("pp_depth", 2);
                offloadPlan.push_back(e);
            }
        }

        // Config state (from kernel_config_def) for buffer_def emission
        int32_t bufferSize = 256;
        int32_t vectorWidth = 4;
        std::string elementType = "int32";
        std::string kernelFileName = "compute_kernel.cc";

        rewriter.setInsertionPoint(op);

        for (Operation &inner : body) {
            if (auto configOp = dyn_cast<KernelConfigDefOp>(&inner)) {
                DictionaryAttr attrs = configOp.getConfigAttrs();
                if (auto a = attrs.getAs<IntegerAttr>("buffer_size"))
                    bufferSize = a.getInt();
                if (auto a = attrs.getAs<IntegerAttr>("vector_width"))
                    vectorWidth = a.getInt();
                if (auto a = attrs.getAs<TypeAttr>("element_type")) {
                    Type t = a.getValue();
                    if (t.isInteger(32))
                        elementType = "int32";
                    else if (t.isInteger(16))
                        elementType = "int16";
                    else if (t.isInteger(8))
                        elementType = "int8";
                    else if (t.isF32())
                        elementType = "float";
                }
                if (auto a = attrs.getAs<StringAttr>("kernel_file"))
                    kernelFileName = a.getValue().str();
                rewriter.create<emitc::VerbatimOp>(loc, "#include <stdint.h>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <adf.h>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <aie_api/aie.hpp>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <aie_api/aie_adf.hpp>");
                if (offloadOn)
                    rewriter.create<emitc::VerbatimOp>(loc, "#include \"aie_kernel_runtime.h\"");
                rewriter.create<emitc::VerbatimOp>(loc, "#define FOR_READ  1");
                rewriter.create<emitc::VerbatimOp>(loc, "#define FOR_WRITE 0");
                // Emit per-window BUF_SZ defines (e.g. BUF_SZ_IN_0, BUF_SZ_OUT_0)
                // Collect window_def buffer_size attributes first
                {
                    int inIdx = 0, outIdx = 0;
                    int32_t maxBufSz = bufferSize;
                    for (Operation &winInner : body) {
                        if (auto wOp = dyn_cast<WindowDefOp>(&winInner)) {
                            DictionaryAttr wAttrs = wOp.getWindowAttrs();
                            int32_t wBufSz = bufferSize;
                            std::string wDir = "in";
                            if (auto a = wAttrs.getAs<IntegerAttr>("buffer_size"))
                                wBufSz = a.getInt();
                            if (auto a = wAttrs.getAs<StringAttr>("direction"))
                                wDir = a.getValue().str();
                            std::string macroName;
                            if (wDir == "out") {
                                macroName = "BUF_SZ_OUT_" + std::to_string(outIdx++);
                            } else {
                                macroName = "BUF_SZ_IN_" + std::to_string(inIdx++);
                            }
                            rewriter.create<emitc::VerbatimOp>(loc,
                                                               "#define " + macroName + " " + std::to_string(wBufSz));
                            if (wBufSz > maxBufSz)
                                maxBufSz = wBufSz;
                        }
                    }
                    // Backward-compat: BUF_SZ = max of all per-window sizes
                    rewriter.create<emitc::VerbatimOp>(loc, "#define BUF_SZ " + std::to_string(maxBufSz));
                }
                // ADF kernel header: window types and helpers (required for xchesscc)
                /*
                rewriter.create<emitc::VerbatimOp>(loc,
                    "typedef struct { void* ptr; int ping_acq; int ping_rel; int pong_acq; int pong_rel; int size; }
                window_internal;"); rewriter.create<emitc::VerbatimOp>(loc, "typedef void* output_window_int8;");
                rewriter.create<emitc::VerbatimOp>(loc,
                    "inline void window_init(window_internal* win, int count, void* ping, int ping_acq_lock, void* pong,
                int ping_rel_lock, int ping_size, int pong_size) { win->ptr = ping; win->ping_acq = ping_acq_lock;
                win->ping_rel = ping_rel_lock; win->size = ping_size; }"); rewriter.create<emitc::VerbatimOp>(loc,
                    "inline output_window_int8 get_output_async_window_int8(window_internal* win) { return win->ptr;
                }");
                */
                // AIEML Gen-2 lock protocol (acquire_greater_equal / release with value).
                // Reference: cardano/src/windowfunctions/window.h lines 1232-1286.
                // Input window: DMA S2MM writes data then releases lockids[1].
                //   Kernel acquire: acquire_greater_equal(lockids[1], 1)  -- wait for "data ready"
                //   Kernel release: release(lockids[0], 1)               -- signal "buffer free"
                // Output window: Kernel writes data then releases lockids[1].
                //   Kernel acquire: acquire_greater_equal(lockids[0], 1)  -- wait for "buffer free"
                //   Kernel release: release(lockids[1], 1)               -- signal "data ready"
                std::string cType = elementType + "_t"; // e.g. "int32_t", "int8_t"
                std::string winInType = "input_window_" + elementType;   // e.g. "input_window_int32"
                std::string winOutType = "output_window_" + elementType; // e.g. "output_window_int32"
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline " + cType + "* acquire_output_window(" + winOutType + "* win) {\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\n"
                         "  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\n"
                         "  acquire_greater_equal(w->lockids[0], 1);\n"
                         "  return (" + cType + "*)w->ptr;\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline void release_output_window(" + winOutType + "* win) {\n"
                         "  chess_memory_fence();\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  release(w->lockids[1], 1);\n"
                         "  w->heads[w->current_bufid] = w->head;\n"
                         "  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline " + cType + "* acquire_input_window(" + winInType + "* win) {\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\n"
                         "  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\n"
                         "  acquire_greater_equal(w->lockids[1], 1);\n"
                         "  return (" + cType + "*)w->ptr;\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline void release_input_window(" + winInType + "* win) {\n"
                         "  chess_memory_fence();\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  release(w->lockids[0], 1);\n"
                         "  w->heads[w->current_bufid] = w->head;\n"
                         "  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\n"
                         "}");

                continue;
            }
            if (auto lockOp = dyn_cast<LockDefOp>(&inner)) {
                std::string line = "#define " + lockOp.getSymName().str() + " " + std::to_string(lockOp.getId());
                rewriter.create<emitc::VerbatimOp>(loc, line);
                continue;
            }
            if (auto bufferOp = dyn_cast<BufferDefOp>(&inner)) {
                Type t = bufferOp.getBufferTypeAttr().getValue();
                std::string vecType = "v4int32";
                int64_t bufSize = bufferSize; // fallback to global
                if (auto memref = dyn_cast<MemRefType>(t)) {
                    // Use the memref dimension as the per-buffer size
                    if (!memref.getShape().empty())
                        bufSize = memref.getShape()[0];
                    Type elem = memref.getElementType();
                    if (auto vec = dyn_cast<VectorType>(elem)) {
                        int w = vec.getShape().empty() ? 4 : vec.getShape()[0];
                        if (vec.getElementType().isInteger(32))
                            vecType = "v" + std::to_string(w) + "int32";
                        else if (vec.getElementType().isInteger(16))
                            vecType = "v" + std::to_string(w) + "int16";
                        else if (vec.getElementType().isInteger(8))
                            vecType = "v" + std::to_string(w) + "int8";
                        else if (vec.getElementType().isF32())
                            vecType = "v" + std::to_string(w) + "float";
                    }
                }
                std::string line = vecType + " " + bufferOp.getSymName().str() + "[" + std::to_string(bufSize) + "];";
                rewriter.create<emitc::VerbatimOp>(loc, line);
                continue;
            }
            if (auto windowOp = dyn_cast<WindowDefOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// window_def " + windowOp.getSymName().str());
                continue;
            }
            if (auto declOp = dyn_cast<KernelDeclOp>(&inner)) {
                // Emit kernel_log.h include for klog() debug logging.
                // klog writes (tag, value) pairs to fixed DM address 0x7F800.
                // Host reads via __Runtime_read_kernel_log() at DM offset 0xF800.
                rewriter.create<emitc::VerbatimOp>(loc, "#include \"kernel_log.h\"");
                rewriter.create<emitc::VerbatimOp>(loc, "#include \"" + kernelFileName + "\"");
                rewriter.create<emitc::VerbatimOp>(loc, "// kernel_decl " + declOp.getSymName().str());
                continue;
            }
            if (auto mainOp = dyn_cast<KernelMainOp>(&inner)) {
                if (failed(
                        convertMainToEmitC(rewriter, mainOp, op, windowInfoMap, elementType, offloadOn, offloadPlan)))
                    return failure();
                continue;
            }
        }

        rewriter.eraseOp(op);
        return success();
    }

    // KERNELCONFIGOFFLOAD: emit the raw-MMIO block that self-configures every
    // incoming S2MM window (ping/pong BD chain + lock inits + channel-start)
    // via the aie_kernel_runtime.h encoder. BD ids and hardware lock indices are
    // resolved upstream by BlueprintToScheduleKernelPass and read off window_def;
    // this function only formats them. BD base address + length come from the
    // core's own C buffer symbols. Lock values mirror the host emitCorePingPongBd:
    // DMA acquires the window ACQ lock (init = ppdepth 2 / single 1) with val -1,
    // releases the window REL lock (init 0) with val 1.
    //
    // Locks are emitted as *hardware* indices (0..15), not the 48+N intrinsic ids
    // the kernel's acquire/release builtins use: aie_kc_encode_lock addresses
    // LOCK0_VALUE + id*0x10 (valid only through LOCK15), and the BD LOCK_ACQ_ID /
    // LOCK_REL_ID fields are 4 bits wide. Passing 48 there would write past the
    // lock array into the LOCKS_EVENT_SELECTION registers and silently truncate
    // the BD fields. window_init() still takes the intrinsic macro names.
    // Shared by both directions: validate the upstream-resolved ids on a window.
    // Missing/out-of-range values mean the two passes disagree, which would
    // otherwise surface as MMIO writes to unrelated registers on hardware — fail
    // the build instead.
    static LogicalResult checkWindowOffloadIds(WindowDefOp windowDefOp, const WindowInfo &w,
                                               ArrayRef<CoreOffloadEntry> offloadPlan) {
        if (w.pingBdId < 0 || (!w.singleBuffer && w.pongBdId < 0)) {
            windowDefOp.emitError() << "KERNELCONFIGOFFLOAD: window '" << windowDefOp.getSymName()
                                    << "' is missing ping_bd_id/pong_bd_id from BlueprintToScheduleKernelPass";
            return failure();
        }
        // Buffering-mode agreement between the two paths.
        //
        // The host allocates only a PING buffer when pp_depth==1 (emitCoreBufferAlloc
        // in helper/flowtransfer_kernel.cpp) — the pong symbol is never registered in
        // CoreMemAllocator. The kernel path, however, never sets `singleBuffer` today,
        // so it would emit a full ping<->pong BD chain whose pong buffer address
        // resolves to 0 via CoreMemAllocator::getAddress (which returns 0 on a miss,
        // silently). That is a DMA writing to a bogus L1 address, visible only on
        // hardware.
        //
        // The host publishes its real pp_depth per tile, so cross-check it rather than
        // trusting either side alone, and fail the build on disagreement.
        const bool isOut = (w.direction == "out");
        for (const auto &e : offloadPlan) {
            if (e.isOutput != isOut)
                continue;
            if (e.ppDepth == 1 && !w.singleBuffer) {
                windowDefOp.emitError()
                    << "KERNELCONFIGOFFLOAD: window '" << windowDefOp.getSymName() << "' is ping-pong on the kernel "
                    << "path but the host reports pp_depth=1 for tile (" << e.col << "," << e.row
                    << "); the host allocates no pong buffer in that mode, so the emitted pong BD would target L1 "
                       "address 0. Propagate single-buffer mode into KernelParamInfo::singleBuffer, or keep "
                       "pp_depth>=2 on offloaded tiles";
                return failure();
            }
        }
        if (w.acquireLockHwId < 0 || w.acquireLockHwId > 15 || w.releaseLockHwId < 0 || w.releaseLockHwId > 15) {
            windowDefOp.emitError()
                << "KERNELCONFIGOFFLOAD: window '" << windowDefOp.getSymName() << "' has hardware lock ids ("
                << w.acquireLockHwId << ", " << w.releaseLockHwId
                << ") outside the 0..15 memory-module lock range; register-level programming needs the "
                   "hardware index, not the 48+N kernel-intrinsic lock id";
            return failure();
        }
        return success();
    }

    // Body of one window's BD chain + lock init, shared by S2MM and MM2S.
    // `pktArgs` supplies the en_packet/pkt_id/pkt_type/ooo_bd_id tail of
    // aie_kc_encode_bd: constant "0, 0, 0, -1" for S2MM (circuit-switched), and
    // per-tile literals for MM2S. `indent` keeps the emitted C readable when this
    // sits inside a get_coreid() dispatch arm.
    static std::string emitWindowBdAndLocks(const WindowInfo &w, StringRef pktArgs, StringRef indent) {
        const std::string in = indent.str();
        // Register writes go through core_reg_write (aie_kernel_runtime.h), which
        // issues the TILE-LOCAL offset into the core's TM space via TM_W.
        //
        // This was the S2MM-not-starting bug: the emitted code wrote the bare
        // tile-local offset (`*(volatile uint32_t *)0x1DE04`), which from the core's
        // own address map is DATA memory (DM base 0x70000, 0x40000-0x7FFFF), not the
        // DMA register. The BD/lock/start writes silently landed in data memory and
        // the DMA was never programmed — so the channel raised no start event even
        // though the code ran.
        //
        // Adding the 0x80000 bias alone does NOT fix that: a numeric cast cannot
        // express the memory space, so it still assembles to a plain ST that never
        // leaves the core (and reads back fine on-core, hiding the failure). Only
        // TM_W lowers to ST.TM. See src/mlir/runtime/kernel_tm.h.
        //
        // Every write is also traced to klog under KERNELCONFIGOFFLOAD_TRACE. Tags
        // are 4 chars (klog's format); "OFF "/"VAL " pair per write, so a reader sees
        // the exact (tile-local register, value) stream the core issued.
        const std::string trace = in + "#ifdef KERNELCONFIGOFFLOAD_TRACE\n" + in +
                                  "  for (_k = 0; _k < _n; _k++) { klog(\"OFF \", (int32_t)_kc[_k].off);"
                                  " klog(\"VAL \", (int32_t)_kc[_k].val); }\n" +
                                  in + "#endif\n";
        const std::string traceOne = in + "#ifdef KERNELCONFIGOFFLOAD_TRACE\n" + in +
                                     "  klog(\"OFF \", (int32_t)_kc[0].off); klog(\"VAL \", (int32_t)_kc[0].val);\n" +
                                     in + "#endif\n";
        const std::string flush = in + "  core_reg_write_block(_kc, _n);\n" + trace;
        const std::string one = in + "  core_reg_write(_kc[0].off, _kc[0].val);\n" + traceOne;
        const std::string acq = std::to_string(w.acquireLockHwId);
        const std::string rel = std::to_string(w.releaseLockHwId);
        const std::string ping = w.pingBuffer;
        const std::string pong = w.pongBuffer.empty() ? w.pingBuffer : w.pongBuffer;
        const std::string pkt = pktArgs.str();

        // The BD's acquire/release lock ids are SWAPPED for output, mirroring the
        // host (helper/flowtransfer_kernel.cpp: `bdAcquireLockId = isOutputFlow ?
        // releaseLockId : acquireLockId`):
        //   S2MM (in):  DMA acquires the window ACQ lock, releases REL.
        //   MM2S (out): DMA acquires the window REL lock, releases ACQ.
        const bool isOut = (w.direction == "out");
        const std::string bdAcq = isOut ? rel : acq;
        const std::string bdRel = isOut ? acq : rel;
        const std::string ppInit = w.singleBuffer ? "1" : "2";

        std::string s;
        s += in + "  // hw lock ids " + acq + "/" + rel + " = " + w.acquireLock + "/" + w.releaseLock +
             " minus the 48 kernel-intrinsic lock base\n";
        // _k only exists for the trace loop; declaring it unconditionally would warn
        // as unused in the normal (untraced) build.
        s += in + "  AieKcReg _kc[8];\n" + in + "  int _n;\n" + in + "#ifdef KERNELCONFIGOFFLOAD_TRACE\n" + in +
             "  int _k;\n" + in + "#endif\n";
        if (w.singleBuffer) {
            // Single buffer: one BD, no next chaining.
            s += in + "  _n = aie_kc_encode_bd(_kc, " + std::to_string(w.pingBdId) + ", (uintptr_t)" + ping +
                 ", sizeof(" + ping + "), -1, " + bdAcq + ", -1, " + bdRel + ", 1, " + pkt + ");\n" + flush;
        } else {
            // Ping-pong: pong BD (next -> ping) then ping BD (next -> pong).
            s += in + "  _n = aie_kc_encode_bd(_kc, " + std::to_string(w.pongBdId) + ", (uintptr_t)" + pong +
                 ", sizeof(" + pong + "), " + std::to_string(w.pingBdId) + ", " + bdAcq + ", -1, " + bdRel + ", 1, " +
                 pkt + ");\n" + flush;
            s += in + "  _n = aie_kc_encode_bd(_kc, " + std::to_string(w.pingBdId) + ", (uintptr_t)" + ping +
                 ", sizeof(" + ping + "), " + std::to_string(w.pongBdId) + ", " + bdAcq + ", -1, " + bdRel + ", 1, " +
                 pkt + ");\n" + flush;
        }
        // Lock INITS are the same expression in both directions — window ACQ gets
        // ppdepth, window REL gets 0 — even though the two directions mean opposite
        // things by it, because the BD acq/rel swap above already encodes the
        // difference:
        //   S2MM: ACQ is the DMA's acquire (buffers free to receive into).
        //   MM2S: ACQ is the KERNEL's acquire (buffers free to produce into), while
        //         the DMA's acquire is REL at 0 so it waits for the kernel.
        // This mirrors the host (passdfscheduletoapi.cpp: output inits
        // `kernelAcquireLock = releaseLockId` to ppdepth and leaves the DMA's
        // acquire at the hardware default 0). Swapping these deadlocks.
        s += in + "  aie_kc_encode_lock(_kc, " + acq + ", " + ppInit + ");\n" + one;
        s += in + "  aie_kc_encode_lock(_kc, " + rel + ", 0);\n" + one;
        // Channel start. Traced with its own tag AND the channel number, because
        // this is the write most likely to be silently ineffective: with start_bd=0
        // and repeat=1 every field of the value is 0, so the register write is a
        // no-content write and the DMA raises no start event. "STCH"/"STOF"/"STVL"
        // give channel / offset / value so a zero value is visible as such rather
        // than looking like the block never executed.
        s += in + "  " + (isOut ? "aie_kc_encode_mm2s_start" : "aie_kc_encode_s2mm_start") + "(_kc, " +
             std::to_string(w.channel) + ", " + std::to_string(w.pingBdId) + ", 1, 0);\n";
        s += in + "#ifdef KERNELCONFIGOFFLOAD_TRACE\n" + in + "  klog(\"" + (isOut ? "STMM" : "STS2") + "\", " +
             std::to_string(w.channel) + ");\n" + in +
             "  klog(\"STOF\", (int32_t)_kc[0].off); klog(\"STVL\", (int32_t)_kc[0].val);\n" + in + "#endif\n";
        s += one;
        return s;
    }

    // KERNELCONFIGOFFLOAD: emit the raw-MMIO blocks that self-configure the core's
    // own DMA (BD chain + lock inits + channel-start) via the aie_kernel_runtime.h
    // encoder. BD ids and hardware lock indices are resolved upstream by
    // BlueprintToScheduleKernelPass and read off window_def; this only formats them.
    // BD base address + length come from the core's own C buffer symbols.
    //
    // Two shapes, because the two directions differ in whether config is uniform:
    //   S2MM (input)  — identical on every core tile (circuit-switched, no packet
    //                   id, no out-of-order bd). Emitted straight-line.
    //   MM2S (output) — packet_id and ooo_bd_id are PER TILE, but there is exactly
    //                   one kernel.cc/ELF broadcast to every tile. Emitted inside a
    //                   get_coreid() dispatch, one arm per tile, from the host's
    //                   published plan (dfschedule.core_offload_plan).
    //
    // Locks are emitted as *hardware* indices (0..15), not the 48+N intrinsic ids
    // the kernel's acquire/release builtins use: aie_kc_encode_lock addresses
    // LOCK0_VALUE + id*0x10 (valid only through LOCK15), and the BD LOCK_ACQ_ID /
    // LOCK_REL_ID fields are 4 bits wide. Passing 48 there would write past the
    // lock array into the LOCKS_EVENT_SELECTION registers and silently truncate
    // the BD fields. window_init() still takes the intrinsic macro names.
    LogicalResult emitCoreDmaConfigBlocks(ConversionPatternRewriter &rewriter, Location loc,
                                          KernelModuleOp kernelModuleOp,
                                          const llvm::StringMap<WindowInfo> &windowInfoMap,
                                          ArrayRef<CoreOffloadEntry> offloadPlan) const {
        for (Operation &inner : kernelModuleOp.getBody().front()) {
            auto windowDefOp = dyn_cast<WindowDefOp>(&inner);
            if (!windowDefOp)
                continue;
            auto it = windowInfoMap.find(windowDefOp.getSymName());
            if (it == windowInfoMap.end())
                continue;
            const WindowInfo &w = it->second;
            const bool isOut = (w.direction == "out");
            if (!isOut && w.direction != "in")
                continue;

            if (failed(checkWindowOffloadIds(windowDefOp, w, offloadPlan)))
                return failure();

            std::string block;
            if (!isOut) {
                block = "{ // KERNELCONFIGOFFLOAD S2MM " + windowDefOp.getSymName().str() + "\n";
                block += emitWindowBdAndLocks(w, /*pktArgs=*/"0, 0, 0, -1", /*indent=*/"");
                block += "}";
            } else {
                // Collect this direction's per-tile entries.
                // One arm per PHYSICAL TILE, not per plan entry. The plan is keyed
                // by (col,row,direction,flow) because a tile can belong to several
                // flows on the same direction, but the dispatch below branches on
                // (col,row) alone — a second entry for the same tile would emit an
                // unreachable `else if` and silently drop whichever arm lost.
                // First entry wins; they describe the same physical MM2S channel.
                //
                // Preferred source is the AGGREGATED dma_bd emitted by
                // DfscheduleKernelAggregationPass: it carries tile_coords /
                // tile_packet_ids / tile_ooo_bd_ids describing exactly this
                // fan-out, so the IR is what drives codegen. The
                // core_offload_plan walk below is the fallback for when that
                // pass did not run.
                SmallVector<CoreOffloadEntry> aggTiles;
                collectAggregatedOutTiles(kernelModuleOp, aggTiles);

                SmallVector<const CoreOffloadEntry *> outTiles;
                for (const auto &e : aggTiles)
                    outTiles.push_back(&e);
                if (outTiles.empty()) {
                    for (const auto &e : offloadPlan) {
                        if (!e.isOutput)
                            continue;
                        bool seen = false;
                        for (const auto *o : outTiles)
                            if (o->col == e.col && o->row == e.row)
                                seen = true;
                        if (!seen)
                            outTiles.push_back(&e);
                    }
                }
                if (outTiles.empty()) {
                    windowDefOp.emitError()
                        << "KERNELCONFIGOFFLOAD: output window '" << windowDefOp.getSymName()
                        << "' has no per-tile entries in dfschedule.core_offload_plan; the host path "
                           "(helper/flowtransfer_kernel.cpp) must publish MM2S tiles before the core can "
                           "self-program them";
                    return failure();
                }

                block = "{ // KERNELCONFIGOFFLOAD MM2S " + windowDefOp.getSymName().str() + "\n";
                block += "  // packet_id and ooo_bd_id differ per tile, but one ELF runs on all of\n";
                block += "  // them, so dispatch on the core's own id.\n";
                block += "  unsigned _cid = get_coreid();\n";
                block += "  int _col = (int)(_cid >> 16);\n";
                block += "  int _row = (int)(_cid & 0x1F);\n";
                for (size_t i = 0; i < outTiles.size(); ++i) {
                    const CoreOffloadEntry &e = *outTiles[i];
                    std::string pkt = std::string(e.enablePacket ? "1" : "0") + ", " + std::to_string(e.packetId) +
                                      ", 0, " + std::to_string(e.oooBdId);
                    block += std::string("  ") + (i == 0 ? "if" : "else if") + " (_col == " + std::to_string(e.col) +
                             " && _row == " + std::to_string(e.row) + ") {\n";
                    block += emitWindowBdAndLocks(w, pkt, "  ");
                    block += "  }\n";
                }
                // No else: a core tile outside the plan has no host-programmed MM2S
                // either, so arming nothing is the correct and safe outcome.
                block += "}";
            }
            rewriter.create<emitc::VerbatimOp>(loc, block);
        }
        return success();
    }

    LogicalResult convertMainToEmitC(ConversionPatternRewriter &rewriter, KernelMainOp mainOp,
                                     KernelModuleOp kernelModuleOp, const llvm::StringMap<WindowInfo> &windowInfoMap,
                                     const std::string &elementType, bool offloadOn,
                                     ArrayRef<CoreOffloadEntry> offloadPlan) const {
        Location loc = mainOp.getLoc();
        Block &mainBody = mainOp.getBody().front();

        // Map SSA values to C expression names (e.g. window_init result -> "window_win")
        llvm::DenseMap<Value, std::string> valueToCName;

        // Insert main func in parent block (before kernelModuleOp), not inside module body
        rewriter.setInsertionPoint(kernelModuleOp);
        auto funcType = rewriter.getFunctionType({}, rewriter.getI32Type());
        auto emitcMain = rewriter.create<emitc::FuncOp>(loc, "main", funcType);
        Block *entry = emitcMain.addEntryBlock();
        rewriter.setInsertionPointToStart(entry);

        // avoids sim crash
        rewriter.create<emitc::VerbatimOp>(loc, "#ifdef AIEHLC_KERNEL_SIM");
        rewriter.create<emitc::VerbatimOp>(loc, "volatile int sync_buffer[8];");
        rewriter.create<emitc::VerbatimOp>(loc, "#else");
        rewriter.create<emitc::VerbatimOp>(loc, "volatile static int sync_buffer[8] = {0, -1};");
        rewriter.create<emitc::VerbatimOp>(loc, "#endif");
        rewriter.create<emitc::VerbatimOp>(loc, "sync_buffer[0] = 0;");
        rewriter.create<emitc::VerbatimOp>(loc, "sync_buffer[1] = -1;");
        rewriter.create<emitc::VerbatimOp>(loc, "klog_init();");

        // KERNELCONFIGOFFLOAD: arm the core's own DMA (S2MM in, MM2S out) before
        // the first window_init so every descriptor is live at core entry.
        //
        // Ordering: the host has already armed the shim side and enabled this core
        // before main() runs, so the shim may already be streaming. Arming here —
        // as early as possible in main(), ahead of window_init and any compute —
        // is the tightest the offload can be. The incoming S2MM has buffering and
        // lock backpressure to absorb the gap; the outgoing MM2S cannot produce
        // until the kernel releases a buffer, which happens strictly later.
        if (offloadOn && failed(emitCoreDmaConfigBlocks(rewriter, loc, kernelModuleOp, windowInfoMap, offloadPlan)))
            return failure();

        for (Operation &inner : mainBody) {
            if (isa<AllocSyncBufferOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// alloc_sync_buffer");
                continue;
            }
            if (isa<SyncBufferWriteOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// sync_buffer_write");
                continue;
            }
            if (isa<LogOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// log(...)");
                continue;
            }
            if (auto winInit = dyn_cast<WindowInitOp>(&inner)) {
                std::string winSym = winInit.getWindowRefAttr().getRootReference().getValue().str();
                valueToCName[winInit.getResult()] = "window_" + winSym;
                rewriter.create<emitc::VerbatimOp>(loc, "window_internal window_" + winSym + "[1];");
                std::string pingBuf, pongBuf, acqLock, relLock;
                auto it = windowInfoMap.find(winSym);
                if (it != windowInfoMap.end()) {
                    pingBuf = it->second.pingBuffer;
                    pongBuf = it->second.pongBuffer;
                    acqLock = it->second.acquireLock;
                    relLock = it->second.releaseLock;
                } else {
                    pingBuf = winSym + "_ping";
                    pongBuf = winSym + "_pong";
                    acqLock = "LOCK_" + winSym + "_ACQ";
                    relLock = "LOCK_" + winSym + "_REL";
                }
                // Use per-window buffer size and num_rounds from window_def attribute
                std::string winBufSzStr = "BUF_SZ"; // fallback
                std::string winNumRoundsStr = winBufSzStr; // fallback: same as bufferSize
                if (it != windowInfoMap.end() && it->second.bufferSize > 0)
                    winBufSzStr = std::to_string(it->second.bufferSize);
                if (it != windowInfoMap.end() && it->second.numRounds > 0)
                    winNumRoundsStr = std::to_string(it->second.numRounds);
                else
                    winNumRoundsStr = winBufSzStr; // backward compat: numRounds = bufferSize
                rewriter.create<emitc::VerbatimOp>(loc, "window_init(window_" + winSym + ", 1, " + pingBuf + ", " +
                                                            acqLock + ", " + pongBuf + ", " + relLock + ", " +
                                                            winBufSzStr + ", " + winNumRoundsStr + ");");
                continue;
            }
            if (auto invokeOp = dyn_cast<KernelInvokeOp>(&inner)) {
                std::string callee = invokeOp.getKernelRefAttr().getRootReference().getValue().str();
                std::string argList;
                for (Value arg : invokeOp.getArgs()) {
                    auto it = valueToCName.find(arg);
                    std::string cArg = (it != valueToCName.end()) ? it->second : "/*unknown*/";
                    if (!argList.empty())
                        argList += ", ";
                    if (isa<InputWindowType>(arg.getType())) {
                        argList += "get_input_async_window_" + elementType + "(" + cArg + ")";
                    } else {
                        argList += "get_output_async_window_" + elementType + "(" + cArg + ")";
                    }
                }
                rewriter.create<emitc::VerbatimOp>(loc, "// kernel_invoke " + callee);
                rewriter.create<emitc::VerbatimOp>(loc, callee + "(" + argList + ");");
                continue;
            }
            if (isa<DoneOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "done();");
                continue;
            }
            if (isa<KernelReturnOp>(&inner)) {
                auto c0 = rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
                rewriter.create<emitc::ReturnOp>(loc, c0.getResult());
                break;
            }
        }

        if (!entry->back().hasTrait<OpTrait::IsTerminator>()) {
            auto c0 = rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
            rewriter.create<emitc::ReturnOp>(loc, c0.getResult());
        }
        return success();
    }
};

} // namespace

namespace mlir {

void DfscheduleToKernelApiPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect, memref::MemRefDialect>();
    target.addIllegalOp<KernelModuleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(ctx);
    patterns.add<KernelModuleToEmitCPattern>(typeConverter, ctx);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
        signalPassFailure();
}

} // namespace mlir
