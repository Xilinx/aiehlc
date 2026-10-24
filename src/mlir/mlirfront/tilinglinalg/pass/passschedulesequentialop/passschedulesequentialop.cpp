/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passschedulesequentialop.h"
#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dfschedule;

namespace {

//===----------------------------------------------------------------------===//
// Operation classification
//===----------------------------------------------------------------------===//

/// Category enum for ordering operations within a RoutingCreate block.
/// Lower numeric value = earlier in the reordered sequence.
enum class OpCategory {
    /// Setup ops: arith.constant, declaretile, memref.subview, memref_mapping,
    /// bind_core_buffer, declare_kernel_config, schedule.getbdid
    Setup = 0,

    /// Kernel ops: config.load_kernel_group, schedule.launch_kernel_group
    Kernel = 1,

    /// DMA config ops: config.dma_bd, config.create_io (outside scf.for)
    DmaConfig = 2,

    /// Start IO ops that RECEIVE (S2MM create_io): the sinks. Armed before any
    /// source starts -- see startIoIsSink().
    StartIoSink = 3,

    /// Start IO ops that SEND (MM2S create_io, or direction unknown).
    StartIo = 4,

    /// SCF for loops (contain their own internal ordering)
    ScfFor = 5,

    /// Wait ops: schedule.wait (outside scf.for)
    Wait = 6,

    /// Cleanup ops: free_device_mem, routing.yield
    Cleanup = 7,
};

/// Arm sinks before sources. A start_io whose create_io is S2MM receives data;
/// starting it after the MM2S that (transitively) feeds it lets the array
/// produce into a channel that is not armed yet. With out-of-order shim S2MM
/// that is not a stall but a corruption: conv2dstem's 4x4 mesh, once its kernel
/// got fast enough to emit its first output tile before the host had armed the
/// output channels, landed whole tiles at the previous launch's addresses
/// (GOLDEN FAIL, timing-dependent). Starting every S2MM first costs nothing --
/// an armed sink with no data just waits -- and makes the order a guarantee
/// instead of a race the host usually wins.
static bool startIoIsSink(dfschedule::StartIoOp startIo) {
    auto io = startIo.getIoHandle().getDefiningOp<dfschedule::ConfigCreateIoOp>();
    return io && io.getDirection() == "S2MM";
}

/// Classify a single operation into its scheduling category.
static OpCategory classifyOp(Operation *op) {
    // Kernel ops — highest priority (go first after setup)
    if (isa<dfschedule::LoadKernelGroupOp>(op))
        return OpCategory::Kernel;
    if (isa<dfschedule::LaunchKernelGroupOp>(op))
        return OpCategory::Kernel;

    // DMA configuration ops
    if (isa<dfschedule::ConfigDmaBdOp>(op))
        return OpCategory::DmaConfig;
    if (isa<dfschedule::ConfigCreateIoOp>(op))
        return OpCategory::DmaConfig;

    // Start IO ops: sinks (S2MM) before sources (MM2S)
    if (auto startIo = dyn_cast<dfschedule::StartIoOp>(op))
        return startIoIsSink(startIo) ? OpCategory::StartIoSink : OpCategory::StartIo;

    // Wait ops
    if (isa<dfschedule::ScheduleWaitOp>(op))
        return OpCategory::Wait;

    // SCF for loops
    if (isa<scf::ForOp>(op))
        return OpCategory::ScfFor;

    // Cleanup ops
    if (isa<dfschedule::FreeDeviceMemOp>(op))
        return OpCategory::Cleanup;
    if (op->getName().getStringRef() == "routing.yield")
        return OpCategory::Cleanup;

    // Everything else is setup
    return OpCategory::Setup;
}

//===----------------------------------------------------------------------===//
// Topological reordering within a block
//===----------------------------------------------------------------------===//

/// Reorder operations within a single block according to the category ordering.
/// This is a stable sort that respects SSA def-use constraints:
/// ops are moved as early as possible within their category while ensuring
/// all operand-defining ops precede them.
static void reorderBlock(Block &block) {
    // Collect all ops (excluding the block terminator if present)
    SmallVector<Operation *> ops;
    for (Operation &op : block) {
        ops.push_back(&op);
    }

    if (ops.empty())
        return;

    // Build the desired order by sorting ops by category, preserving relative
    // order within each category (stable sort).
    SmallVector<Operation *> sorted(ops.begin(), ops.end());
    std::stable_sort(sorted.begin(), sorted.end(), [](Operation *a, Operation *b) {
        return static_cast<int>(classifyOp(a)) < static_cast<int>(classifyOp(b));
    });

    // Now we need to place ops in the sorted order while respecting SSA deps.
    // Strategy: iterate through sorted order. For each op, ensure all its
    // operand-defining ops (in the same block) have already been placed.
    // If not, defer the op and try again later.

    // We use a simple approach: place ops one by one. For each op, if its
    // deps are satisfied (all operand-defining ops already placed), place it.
    // Otherwise, skip and retry later. This handles cross-category dependencies.

    llvm::DenseSet<Operation *> placed;
    SmallVector<Operation *> finalOrder;
    finalOrder.reserve(sorted.size());

    // Multiple passes until all ops are placed
    SmallVector<Operation *> remaining(sorted.begin(), sorted.end());
    int maxIters = remaining.size() + 1;
    while (!remaining.empty() && maxIters-- > 0) {
        SmallVector<Operation *> deferred;
        for (Operation *op : remaining) {
            // Check if all operand-defining ops in this block are already placed
            bool depsReady = true;
            for (Value operand : op->getOperands()) {
                Operation *defOp = operand.getDefiningOp();
                if (defOp && defOp->getBlock() == &block && !placed.contains(defOp)) {
                    depsReady = false;
                    break;
                }
            }
            if (depsReady) {
                finalOrder.push_back(op);
                placed.insert(op);
            } else {
                deferred.push_back(op);
            }
        }
        if (deferred.size() == remaining.size()) {
            // No progress — break and append remaining in original order
            for (Operation *op : deferred)
                finalOrder.push_back(op);
            break;
        }
        remaining = std::move(deferred);
    }

    // Apply the final order: detach all ops, then re-insert in desired sequence.
    // We move ops one by one to the end of the block. After all moves,
    // the block contains ops in finalOrder sequence.
    //
    // Strategy: pick an anchor = block.end(), and moveBefore(block, block.end())
    // for each op in order. This appends ops to the block end in sequence.
    for (Operation *op : finalOrder) {
        op->moveBefore(&block, block.end());
    }
}

/// Reorder the body of an scf.for loop: within the loop, apply the same
/// ordering rules. The scf.for body is a single block with a yield terminator.
static void reorderScfForBody(scf::ForOp forOp) {
    Block &body = *forOp.getBody();

    // Collect non-terminator ops
    SmallVector<Operation *> ops;
    for (Operation &op : body) {
        if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
        ops.push_back(&op);
    }

    if (ops.empty())
        return;

    // Sort by category (stable)
    SmallVector<Operation *> sorted(ops.begin(), ops.end());
    std::stable_sort(sorted.begin(), sorted.end(), [](Operation *a, Operation *b) {
        return static_cast<int>(classifyOp(a)) < static_cast<int>(classifyOp(b));
    });

    // Place with dep resolution
    llvm::DenseSet<Operation *> placed;
    SmallVector<Operation *> finalOrder;
    finalOrder.reserve(sorted.size());

    SmallVector<Operation *> remaining(sorted.begin(), sorted.end());
    int maxIters = remaining.size() + 1;
    while (!remaining.empty() && maxIters-- > 0) {
        SmallVector<Operation *> deferred;
        for (Operation *op : remaining) {
            bool depsReady = true;
            for (Value operand : op->getOperands()) {
                Operation *defOp = operand.getDefiningOp();
                if (defOp && defOp->getBlock() == &body && !placed.contains(defOp)) {
                    depsReady = false;
                    break;
                }
            }
            if (depsReady) {
                finalOrder.push_back(op);
                placed.insert(op);
            } else {
                deferred.push_back(op);
            }
        }
        if (deferred.size() == remaining.size()) {
            for (Operation *op : deferred)
                finalOrder.push_back(op);
            break;
        }
        remaining = std::move(deferred);
    }

    // Move ops before the yield terminator
    Operation *yieldOp = body.getTerminator();
    for (Operation *op : finalOrder) {
        op->moveBefore(yieldOp);
    }
}

//===----------------------------------------------------------------------===//
// Reorder within a RoutingCreate block
//===----------------------------------------------------------------------===//

/// Process a single RoutingCreate region block: reorder its operations
/// according to the scheduling rules.
static void processRoutingCreateBlock(Block &block) {
    // First, recursively reorder scf.for bodies
    for (Operation &op : block) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            reorderScfForBody(forOp);
        }
    }

    // Now reorder the top-level block
    reorderBlock(block);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

void mlir::ScheduleSequentialOpPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();

    int blocksProcessed = 0;

    // After ScheduleCanonicalizePass, the IR has dfschedule.host blocks
    // instead of routing.RoutingCreate blocks. Walk HostBlockOp.
    moduleOp.walk([&](dfschedule::HostBlockOp hostOp) {
        Block &block = hostOp.getBody().front();
        processRoutingCreateBlock(block);
        blocksProcessed++;
    });

    // Also handle routing.RoutingCreate blocks (pre-canonicalize IR)
    if (blocksProcessed == 0) {
        moduleOp.walk([&](Operation *op) {
            if (op->getName().getStringRef() != "routing.RoutingCreate")
                return;
            for (Region &region : op->getRegions()) {
                for (Block &block : region) {
                    processRoutingCreateBlock(block);
                    blocksProcessed++;
                }
            }
        });
    }

    llvm::errs() << "[ScheduleSequentialOp] Processed " << blocksProcessed << " blocks\n";
}
