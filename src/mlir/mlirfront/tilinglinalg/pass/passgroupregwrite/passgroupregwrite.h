/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __GROUP_REG_WRITE_PASS_H__
#define __GROUP_REG_WRITE_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dfschedulemanager.h"

namespace mlir {

/// GroupRegWritePass — coalesce identical per-tile register writes (currently
/// only the lock-init writes that DfscheduleToApiPass emits as individual
/// XAie_LockSetValue calls) into hardware control-packet group writes.
///
/// A control packet pushed once from the shim DMA fans out to N tiles through
/// the stream switch, replacing N host-side MMIO register writes. The pass runs
/// on the host ModuleOp after ScheduleCanonicalizePass and before
/// DfscheduleToApiPass.
///
/// Scope (per design): broadcast (whole configured array) and row-multicast
/// (all columns of one row) only. Per-tile-unique writes stay individual.
///
/// The pass:
///   1. Derives lock-init tuples (col,row,lock_id,value) from each
///      dfschedule.config.dma_bd, mirroring DfscheduleToApiPass's own logic.
///   2. Clusters tuples by (tile_addr,value) and classifies broadcast / row.
///   3. Emits one dfschedule.ctrl_plan_init + one dfschedule.group_reg_write per
///      foldable cluster, and records the folded (col,row,lock_id) triples in a
///      module attribute so DfscheduleToApiPass skips their individual writes.
class GroupRegWritePass : public PassWrapper<GroupRegWritePass, OperationPass<ModuleOp>> {
  public:
    GroupRegWritePass() = default;
    // Resource selection for the control fabric (must not collide with the
    // data-plane DMA channels/BDs used by the offloaded kernels).
    GroupRegWritePass(int ctrlId, int respS2mmCh, int sendBdId, int sendMm2sCh)
        : ctrlId(ctrlId), respS2mmCh(respS2mmCh), sendBdId(sendBdId), sendMm2sCh(sendMm2sCh) {}

    StringRef getArgument() const final { return "group-reg-write"; }
    StringRef getDescription() const final {
        return "Coalesce identical per-tile lock-init register writes into control-packet group writes";
    }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect, func::FuncDialect, arith::ArithDialect>();
    }

  private:
    int ctrlId = 1;
    int respS2mmCh = 1;
    // Blocking write-ack uses shim send BD @sendBdId (forward push) plus return
    // BDs sendBdId+1..sendBdId+ncols (per column). For a 4-column row that is 5
    // BDs; they must all be in-range [0,15] and free on the spine shim tile
    // (which uses only BD0/BD1 for GEMM input). sendBdId=2 -> BDs 2..6.
    int sendBdId = 2;
    int sendMm2sCh = 1;
};

} // namespace mlir

#endif // __GROUP_REG_WRITE_PASS_H__
