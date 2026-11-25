/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __MEMTILE_JOIN_ROUTING_PASS_H__
#define __MEMTILE_JOIN_ROUTING_PASS_H__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "routinghwmanager.h"

namespace mlir {

// MemtileOutputJoinRoutingPass -- routing half of the memtile output join
// (doc/design/memtile_staging.md stage 1; host half in StartioInterleavePass).
// An output flow descends its shim column through the memtile at row 1 as a
// NORTH -> SOUTH circuit carrying the cores' packets. The pass terminates it in
// the memtile instead (NORTH -> DMA S2MM 2*ch, ch = the shim S2MM channel) and
// adds the memtile's re-emit (DMA MM2S 2*ch -> SOUTH, same SOUTH master), so the
// shim receives one plain stream. Runs on the routinghw IR before
// RoutingHWLowerPass, with the output stage on; the input stage also stages the input
// flows (SOUTH -> DMA S2MM 2*ch+1, DMA MM2S 2*ch+1 -> NORTH).
// Memtile staging stages (doc/design/memtile_staging.md): #pragma memtile_staging(output,
// input, cblocked, weights), published by aiehlc as the bit mask routing.memtile_staging.
// The AIEHLC_MEMTILE_OUTPUT / _INPUT / _C_BLOCKED environment variables force a stage on.
enum : int64_t {
    kMemtileStageOutput = 1,
    kMemtileStageInput = 2,
    kMemtileStageCBlocked = 4,
    kMemtileStageWeights = 8, // column-broadcast inputs (GEMM B) kept resident in the memtile
};
int64_t memtileStaging(ModuleOp mod);
// The column-broadcast input channels the weights stage covers, as shim col * 2 + channel,
// published by DmapToDmaphopPass as routing.memtile_weights_io (both clones see it).
bool isMemtileWeightsIo(ModuleOp mod, int64_t shimCol, int64_t channel);

class MemtileOutputJoinRoutingPass : public PassWrapper<MemtileOutputJoinRoutingPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemtileOutputJoinRoutingPass)
    StringRef getArgument() const final { return "memtile-output-join-routing"; }
    StringRef getDescription() const final { return "Route output packet flows into a memtile DMA"; }
    void runOnOperation() override;
};

} // namespace mlir

#endif
