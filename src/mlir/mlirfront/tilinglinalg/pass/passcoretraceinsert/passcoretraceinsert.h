/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __CORE_TRACE_INSERT_PASS__
#define __CORE_TRACE_INSERT_PASS__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

// Optional mem-module DMA/stream selection carried by `#pragma aie_trace`.
// A trace tile may additionally request which tile DMA the memory-module trace
// unit watches (see the pragma grammar in aiehlc.cc):
//   Default   : no second tuple -> runtime default (S2MM ch0).
//   Stream    : (STREAM, "s2mm"|"mm2s", ch) -> explicit dmaKind/dmaCh.
//   Parameter : (PARAMETER, "win_x")        -> named kernel window, resolved to
//               the physical (direction, channel) the tiling flow assigned on
//               the traced tile (input window => S2MM, output window => MM2S).
enum class TraceDmaSel { Default, Stream, Parameter };

// dmaKind mirrors the runtime enum ints in aie_runtime.h:
//   AIE_TRACE_DMA_NONE=0, AIE_TRACE_DMA_S2MM=1, AIE_TRACE_DMA_MM2S=2.
struct TraceTileSpec {
    int col = 0, row = 0;
    TraceDmaSel sel = TraceDmaSel::Default;
    // Stream form (and the resolved result of the Parameter form):
    int dmaKind = 1; // AIE_TRACE_DMA_S2MM
    int dmaCh = 0;
    // Parameter form (kernel window/port name to resolve, e.g. "win_a"):
    std::string paramName;
};

// CoreTraceInsertPass — host-path (emitc) pass that auto-injects declarative
// per-tile core trace calls requested via `#pragma aie_trace(col,row)`.
//
// For each traced (col,row) compute tile it emits, into the host dispatch
// function (emitc.func @host_canonicalized[_<suffix>]):
//   * `__Runtime_core_trace_begin(dev, col, row)` right BEFORE the
//     `__Runtime_launch_kernel_group` call, and
//   * `__Runtime_core_trace_end(dev)` right BEFORE `__Runtime_device_teardown`
//     (or at the function terminator if no teardown is present).
// The device handle is the host function's first argument (XAie_DevInst*).
// The pass is a clean no-op when the trace-tile list is empty.
class CoreTraceInsertPass : public PassWrapper<CoreTraceInsertPass, OperationPass<ModuleOp>> {
  public:
    explicit CoreTraceInsertPass(std::vector<TraceTileSpec> traceTiles = {}) : traceTiles_(std::move(traceTiles)) {}

  private:
    std::vector<TraceTileSpec> traceTiles_;
    void runOnOperation() override;
    mlir::StringRef getArgument() const final { return "core-trace-insert"; }
    mlir::StringRef getDescription() const final {
        return "Injects __Runtime_core_trace_begin/_end calls for #pragma aie_trace tiles.";
    }
};

#endif // __CORE_TRACE_INSERT_PASS__
