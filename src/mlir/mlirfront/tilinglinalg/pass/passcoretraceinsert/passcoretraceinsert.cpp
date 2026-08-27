/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "passcoretraceinsert.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <iostream>

using namespace mlir;

// Emit one emitc.call_opaque
// "__Runtime_core_trace_begin_dma"(dev, col, row, dma_kind, dma_ch) per traced
// tile immediately before `beforeOp`. `dev` is the host function's first
// argument (XAie_DevInst*); col/row/dma_kind/dma_ch are emitc.constant i32
// values. Default-selection tiles carry S2MM(1)/ch0, identical to the previous
// __Runtime_core_trace_begin behaviour.
static void emitTraceBegin(OpBuilder &builder, Operation *beforeOp, Value dev,
                           const std::vector<TraceTileSpec> &tiles) {
    MLIRContext *ctx = builder.getContext();
    Type i32Type = IntegerType::get(ctx, 32);
    builder.setInsertionPoint(beforeOp);
    for (const auto &t : tiles) {
        Location loc = beforeOp->getLoc();
        auto colConst = builder.create<emitc::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(t.col));
        auto rowConst = builder.create<emitc::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(t.row));
        auto kindConst = builder.create<emitc::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(t.dmaKind));
        auto chConst = builder.create<emitc::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(t.dmaCh));
        builder.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "__Runtime_core_trace_begin_dma", nullptr, nullptr,
            ValueRange{dev, colConst.getResult(), rowConst.getResult(), kindConst.getResult(), chConst.getResult()});
    }
}

// Emit one emitc.call_opaque "__Runtime_core_trace_sync_begin"(dev) before
// `beforeOp`. Placed after the trace_begin calls and just before the launch so
// the runtime captures anchor0 (host clock + each armed tile's AIE timer) right
// before the cores run; the paired __Runtime_core_trace_end then adds anchor1
// and emits the full host<->AIE [TIMESYNC] block.
static void emitTraceSyncBegin(OpBuilder &builder, Operation *beforeOp, Value dev) {
    builder.setInsertionPoint(beforeOp);
    builder.create<emitc::CallOpaqueOp>(beforeOp->getLoc(), TypeRange{}, "__Runtime_core_trace_sync_begin", nullptr,
                                        nullptr, ValueRange{dev});
}

// Emit one emitc.call_opaque "__Runtime_core_trace_end"(dev) before `beforeOp`.
static void emitTraceEnd(OpBuilder &builder, Operation *beforeOp, Value dev) {
    builder.setInsertionPoint(beforeOp);
    builder.create<emitc::CallOpaqueOp>(beforeOp->getLoc(), TypeRange{}, "__Runtime_core_trace_end", nullptr, nullptr,
                                        ValueRange{dev});
}

// Create an emitc.constant of type `const char *` rendering the C string
// literal "phase" (quotes included). Same OpaqueAttr/ConstantOp idiom used for
// string literals across routinghwlower.cpp / passdfscheduletoapi.cpp.
static Value mkStr(OpBuilder &builder, Location loc, StringRef phase) {
    MLIRContext *ctx = builder.getContext();
    // `const char *` cannot be an emitc.opaque outer type (the verifier rejects a
    // trailing pointer), so build a proper emitc.ptr<opaque<"const char">> which
    // renders as `const char *`.
    Type strType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const char"));
    std::string literal = "\"" + phase.str() + "\"";
    return builder.create<emitc::ConstantOp>(loc, strType, emitc::OpaqueAttr::get(ctx, literal)).getResult();
}

// Create an emitc.constant i32 with value v.
static Value mkI32(OpBuilder &builder, Location loc, int32_t v) {
    Type i32Type = IntegerType::get(builder.getContext(), 32);
    return builder.create<emitc::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(v)).getResult();
}

// Emit one emitc.call_opaque
// "__Runtime_core_trace_event"(dev, iter, "phase") anchored at `at`. When
// `placeAfter` is set the call (and its string constant) land immediately after
// `at`, else immediately before it. The string constant is created right before
// the call so it dominates; `dev` (func arg) and `iterVal` (block arg / const)
// already dominate every in-body insertion point.
static void emitEvent(OpBuilder &builder, Operation *at, bool placeAfter, Value dev, Value iterVal, StringRef phase) {
    if (placeAfter)
        builder.setInsertionPointAfter(at);
    else
        builder.setInsertionPoint(at);
    Location loc = at->getLoc();
    Value strVal = mkStr(builder, loc, phase);
    builder.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_core_trace_event", nullptr, nullptr,
                                        ValueRange{dev, iterVal, strVal});
}

// Same as emitEvent but with a literal iteration index. The iter constant is
// created AFTER the insertion point is set (immediately before the call) so it
// dominates the use — unlike a Value made earlier at a stale insertion point.
static void emitEventC(OpBuilder &builder, Operation *at, bool placeAfter, Value dev, int32_t iter, StringRef phase) {
    if (placeAfter)
        builder.setInsertionPointAfter(at);
    else
        builder.setInsertionPoint(at);
    Location loc = at->getLoc();
    Value iterVal = mkI32(builder, loc, iter);
    Value strVal = mkStr(builder, loc, phase);
    builder.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_core_trace_event", nullptr, nullptr,
                                        ValueRange{dev, iterVal, strVal});
}

// Return true when `op` is an emitc.call_opaque whose callee is exactly `name`.
static bool isCall(Operation *op, StringRef name) {
    auto callOp = dyn_cast<emitc::CallOpaqueOp>(op);
    return callOp && callOp.getCallee() == name;
}

void CoreTraceInsertPass::runOnOperation() {
    if (traceTiles_.empty())
        return; // clean no-op

    ModuleOp module = getOperation();

    // Find the host dispatch function (emitc.func @host_canonicalized[_<suffix>]).
    // At this point in the pipeline the function has not been renamed yet, so it
    // is still exactly "host_canonicalized"; match by prefix for robustness.
    emitc::FuncOp hostFunc;
    for (auto func : module.getOps<emitc::FuncOp>()) {
        if (func.getName().starts_with("host_canonicalized")) {
            hostFunc = func;
            break;
        }
    }
    if (!hostFunc) {
        std::cout << "[CoreTraceInsert] no emitc.func host_canonicalized found; skipping" << std::endl;
        return;
    }
    if (hostFunc.getNumArguments() == 0) {
        std::cout << "[CoreTraceInsert] host function has no arguments; cannot resolve device handle" << std::endl;
        return;
    }
    Value dev = hostFunc.getArgument(0); // XAie_DevInst*

    // Locate the anchor call_opaque ops by callee name.
    emitc::CallOpaqueOp launchOp;
    emitc::CallOpaqueOp teardownOp;
    hostFunc.walk([&](emitc::CallOpaqueOp callOp) {
        StringRef callee = callOp.getCallee();
        if (callee == "__Runtime_launch_kernel_group" && !launchOp)
            launchOp = callOp;
        else if (callee == "__Runtime_device_teardown" && !teardownOp)
            teardownOp = callOp;
    });

    OpBuilder builder(&getContext());

    // trace_begin (then sync_begin) before the first launch; if no launch anchor
    // exists, place at the start of the entry block so setup precedes any kernel
    // activity. sync_begin is inserted at the SAME anchor as the begins, so it
    // lands immediately after them and just before the launch (anchor0 captured
    // right before the cores run).
    if (launchOp) {
        emitTraceBegin(builder, launchOp.getOperation(), dev, traceTiles_);
        emitTraceSyncBegin(builder, launchOp.getOperation(), dev);
    } else {
        Block &entry = hostFunc.getBlocks().front();
        emitTraceBegin(builder, &entry.front(), dev, traceTiles_);
        emitTraceSyncBegin(builder, &entry.front(), dev);
    }

    // trace_end before device teardown; else before the function terminator.
    if (teardownOp) {
        emitTraceEnd(builder, teardownOp.getOperation(), dev);
    } else {
        Block &entry = hostFunc.getBlocks().front();
        emitTraceEnd(builder, entry.getTerminator(), dev);
    }

    // ---------------------------------------------------------------------
    // Host phase events. sync_begin (just inserted before launch) arms the
    // correlated session, so __Runtime_core_trace_event fires from here on.
    // Once, outside the loop: "launch" right after the kernel-group launch.
    // ---------------------------------------------------------------------
    if (launchOp)
        emitEventC(builder, launchOp.getOperation(), /*placeAfter=*/true, dev, /*iter=*/-1, "launch");

    // Collect round loops: each emitc.for whose body DIRECTLY contains a
    // __Runtime_wait call (the per-round wait block). Do not descend into
    // nested loops when scanning for the round's own startio/wait calls.
    std::vector<emitc::ForOp> roundLoops;
    hostFunc.walk([&](emitc::ForOp forOp) {
        for (Operation &op : forOp.getBody()->getOperations()) {
            if (isCall(&op, "__Runtime_wait")) {
                roundLoops.push_back(forOp);
                break;
            }
        }
    });

    for (emitc::ForOp forOp : roundLoops) {
        Value iterVal = forOp.getInductionVar();
        Block *body = forOp.getBody();

        // iter_start: top of the loop body.
        emitEvent(builder, &body->front(), /*placeAfter=*/false, dev, iterVal, "iter_start");

        // Scan direct body children for the round's own startio/wait markers.
        Operation *firstStartio = nullptr;
        Operation *firstWait = nullptr;
        Operation *lastWait = nullptr;
        for (Operation &op : body->getOperations()) {
            if (!firstStartio && isCall(&op, "__Runtime_startio"))
                firstStartio = &op;
            if (isCall(&op, "__Runtime_wait")) {
                if (!firstWait)
                    firstWait = &op;
                lastWait = &op;
            }
        }

        // dma_start: before the first startio kick.
        if (firstStartio)
            emitEvent(builder, firstStartio, /*placeAfter=*/false, dev, iterVal, "dma_start");
        // wait_start: before the first wait.
        if (firstWait)
            emitEvent(builder, firstWait, /*placeAfter=*/false, dev, iterVal, "wait_start");
        // wait_done: after the last wait.
        if (lastWait)
            emitEvent(builder, lastWait, /*placeAfter=*/true, dev, iterVal, "wait_done");
    }

    // Fallback for a single-shot schedule (no round loop): emit a "wait_done"
    // before teardown so the host lane still gets a marker. "launch" already
    // fired above when a launch anchor exists.
    if (roundLoops.empty() && teardownOp)
        emitEventC(builder, teardownOp.getOperation(), /*placeAfter=*/false, dev, /*iter=*/0, "wait_done");

    std::cout << "[CoreTraceInsert] injected core trace for " << traceTiles_.size() << " tile(s) into "
              << hostFunc.getName().str() << "; round loops=" << roundLoops.size() << std::endl;
}
