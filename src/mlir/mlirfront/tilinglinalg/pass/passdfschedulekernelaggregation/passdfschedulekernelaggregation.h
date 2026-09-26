/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __DFSCHEDULE_KERNEL_AGGREGATION_PASS_H__
#define __DFSCHEDULE_KERNEL_AGGREGATION_PASS_H__

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "dfschedulemanager.h"

namespace mlir {

/// DfscheduleKernelAggregationPass — collapse the per-core-tile core DMA config
/// emitted under KERNELCONFIGOFFLOAD into one op group per window.
///
/// There is exactly ONE kernel.cc, broadcast to every core tile, so emitting a
/// separate declaretile/dma_bd/create_io/start_io group per tile is redundant:
/// on a 4x4 mesh a 3-window GEMM produces 48 tile groups (96 dma_bd ops) that
/// describe at most two distinct configurations.
///
///   S2MM (input): identical on every tile. Circuit-switched, so no packet id
///                 and no out-of-order bd — nothing varies at all.
///   MM2S (output): identical EXCEPT packet_id (basePacketId + tileIndex) and
///                 ooo_bd_id (which shim S2MM BD this tile's data targets).
///
/// The pass rewrites each cluster into a single group anchored on
/// `dfschedule.declaretile.self` -- the tile executing this kernel, resolved at
/// runtime via get_coreid() -- carrying the per-tile variation as discardable
/// array attributes on the dma_bd:
///
///     aggregated      = true
///     tile_coords     = [[0,3],[1,3],...]   // covered tiles, in emission order
///     tile_packet_ids = [1,2,...]           // MM2S only
///     tile_ooo_bd_ids = [2,3,...]           // MM2S only
///
/// The self-tile is a distinct OP rather than a declaretile{col = -1, row = -1}
/// sentinel so that "has no coordinates" is a type-level fact: every existing
/// `getDefiningOp<DeclareTileOp>()` returns null for it and therefore fails
/// closed, instead of silently reading -1 and (on the host path) emitting
/// XAie_TileLoc(-1,-1). The sentinel was also easy to test wrongly, since
/// DeclareTileOp::getCol() returns uint32_t and `getCol() < 0` is always false.
///
/// ooo_bd_id provenance. out_of_order_bd_id is derived in
/// helper/flowtransfer_kernel.cpp from FlowLoweringCtx::shimPerTileBdIds, which
/// only the HOST clone populates (helper/flowtransfer_host.cpp). On the kernel
/// clone that vector is empty, so the emitter falls back to reading the value
/// the host published per (col,row,MM2S,flow) into the ResourceMgr singleton --
/// which is what keeps the kernel-clone dma_bd ops honest. (Before that fallback
/// existed every MM2S core BD in the stage-15 dump read -1 while the emitted
/// kernel.cc carried real ids, so the IR actively misdescribed the hardware.)
///
/// Both packet_id and ooo_bd_id therefore exist in the op AND the plan, derived
/// independently. The pass cross-checks them and FAILS on disagreement rather
/// than preferring a source: a wrong ooo id routes a tile's output into another
/// tile's shim BD, which surfaces as a silent data mismatch on hardware.
///
/// Mergeability is verified STRICTLY: every attribute other than the two known
/// varying fields must be identical across a cluster, and every tile must have a
/// plan entry. A mismatch fails the build with the offending tile coordinates and
/// attribute name. A silent wrong merge would program the wrong DMA registers on
/// real hardware and still compile, so this pass prefers to stop.
///
/// Runs on the kernel ModuleOp between BlueprintToScheduleKernelPass (which
/// creates the per-tile groups) and DfscheduleToKernelApiPass (which formats
/// them into kernel.cc). Gated on routing.kernel_config_offload — without that
/// pragma no per-tile core DMA config exists to aggregate.
class DfscheduleKernelAggregationPass : public PassWrapper<DfscheduleKernelAggregationPass, OperationPass<ModuleOp>> {
  public:
    DfscheduleKernelAggregationPass() = default;

    StringRef getArgument() const final { return "dfschedule-kernel-aggregation"; }
    StringRef getDescription() const final {
        return "Aggregate per-core-tile KERNELCONFIGOFFLOAD DMA config into one op group per window";
    }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect, func::FuncDialect, arith::ArithDialect>();
    }
};

} // namespace mlir

#endif // __DFSCHEDULE_KERNEL_AGGREGATION_PASS_H__
