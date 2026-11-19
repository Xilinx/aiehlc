/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __STARTIO_INTERLEAVE_PASS__
#define __STARTIO_INTERLEAVE_PASS__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// StartioInterleavePass — host-path (emitc) pass that issues each DMA channel as
// soon as its BD chain is programmed.
//
// The dfschedule lowering emits every BD config first and every
// __Runtime_startio last, so the host programs all BDs (tens of µs of serial
// MMIO) before the first DMA moves a byte. mlir-aie's runtime sequences
// interleave instead: write a channel's BDs, push its start queue, move on, so
// the remaining BD writes overlap the transfers already running. This pass
// moves each startio up to just after the latest definition of its operands —
// the createio, which follows the channel's BD chain. Stream back-pressure makes
// the earlier start safe: a producer started before its consumer simply stalls.
//
// Second step, outputs last: each shim S2MM channel's programming (BD chain,
// OOO enable, createio, startio) moves behind the last input start, so input
// DMAs and compute begin while the host still programs the output BDs
// (AIEHLC_NO_OUTPUTS_LAST=1 turns just this step off).
//
// Third step, synchronized input start: all input startios move to just before
// the first output group, so every input channel starts within a few hundred ns
// of the others instead of staggered by the BD programming between them; a
// broadcast B stream stalls on its slowest consumer row
// Off by default (measured slower on VEK280); AIEHLC_SYNC_INPUT_START=1 enables it.
//
// Skipped under control_packet_mode(aot) and control_plan_shim_bd_ctrl: both
// batch the post-launch shim-BD window by call order. AIEHLC_NO_STARTIO_INTERLEAVE=1
// in the environment turns it off (A/B measurements).
struct StartioInterleavePass : public PassWrapper<StartioInterleavePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StartioInterleavePass)
    StringRef getArgument() const final { return "startio-interleave"; }
    StringRef getDescription() const final { return "Issue each DMA channel right after its BD chain"; }
    void runOnOperation() override;
};

#endif
