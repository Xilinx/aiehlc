/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passdmaphoptodfschedule.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "routingmanager.h"
#include <sstream>
#include <vector>
#include <unordered_map>
#include <iostream>

using namespace mlir;
using namespace dmaphop;
using namespace dfschedule;
using namespace routing;

namespace {

// Pattern to convert routing.RoutingCreate with dmaphop operations into dfschedule module
struct RoutingCreateTodfschedulePattern : public OpConversionPattern<routing::RoutingCreate> {
    using OpConversionPattern<routing::RoutingCreate>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(routing::RoutingCreate op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto &routingRegion = op.getRegion();
        
        if (routingRegion.empty()) {
            return failure();
        }

        // Get the block argument (scf_idx)
        Block &routingBlock = routingRegion.front();
        Value scfIdx = routingBlock.getArgument(0);

        // Extract dmaphop operations and routing data
        SmallVector<Operation*> coreTiles;
        SmallVector<Operation*> corePortsIn;
        SmallVector<Operation*> corePortsOut;
        Operation* shimTile = nullptr;
        Operation* shimPortIn = nullptr;
        Operation* shimPortOut = nullptr;
        Operation* pathOp = nullptr;
        SmallVector<Value> allocBuffers;
        Operation* pullOp = nullptr;

        // Walk through the routing block to extract operations
        for (Operation &innerOp : routingBlock.getOperations()) {
            if (innerOp.getName().getStringRef() == "dmaphop.tile") {
                auto tileTypeAttr = innerOp.getAttrOfType<StringAttr>("TILETYPE");
                if (tileTypeAttr) {
                    if (tileTypeAttr.getValue() == "core") {
                        coreTiles.push_back(&innerOp);
                    } else if (tileTypeAttr.getValue() == "shim") {
                        shimTile = &innerOp;
                    }
                }
            } else if (innerOp.getName().getStringRef() == "dmaphop.port") {
                auto dirAttr = innerOp.getAttrOfType<StringAttr>("direction");
                Value tileOperand = innerOp.getOperand(0);
                Operation* portTile = tileOperand.getDefiningOp();
                
                if (portTile) {
                    auto tileTypeAttr = portTile->getAttrOfType<StringAttr>("TILETYPE");
                    if (tileTypeAttr && tileTypeAttr.getValue() == "core") {
                        if (dirAttr && dirAttr.getValue() == "In") {
                            corePortsIn.push_back(&innerOp);
                        } else if (dirAttr && dirAttr.getValue() == "Out") {
                            corePortsOut.push_back(&innerOp);
                        }
                    } else if (tileTypeAttr && tileTypeAttr.getValue() == "shim") {
                        if (dirAttr && dirAttr.getValue() == "In") {
                            shimPortIn = &innerOp;
                        } else if (dirAttr && dirAttr.getValue() == "Out") {
                            shimPortOut = &innerOp;
                        }
                    }
                }
            } else if (innerOp.getName().getStringRef() == "dmaphop.create_path") {
                pathOp = &innerOp;
            } else if (innerOp.getName().getStringRef() == "dmaphop.alloc_buffer") {
                allocBuffers.push_back(innerOp.getResult(0));
            } else if (innerOp.getName().getStringRef() == "dmaphop.pull") {
                pullOp = &innerOp;
            }
        }

        // Create the dfschedule module structure
        // Insert the module at the parent level of routing.RoutingCreate
        rewriter.setInsertionPoint(op);
        auto moduleOp = rewriter.create<ModuleOp>(loc);
        Block *moduleBlock = &moduleOp.getBodyRegion().front();
        
        // Create dskernel function inside the module
        rewriter.setInsertionPointToEnd(moduleBlock);
        
        // Create function type: (i32) -> ()
        auto funcType = rewriter.getFunctionType({rewriter.getI32Type()}, {});
        auto dskernelFunc = rewriter.create<func::FuncOp>(
            loc, "dskernel_coretile_compute", funcType);
        dskernelFunc.setPrivate();
        
        // Create function body
        Block *funcBody = dskernelFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(funcBody);
        
        // Get function argument
        Value funcArg = funcBody->getArgument(0);
        
        // Allocate ping-pong buffers
        auto memrefType = MemRefType::get({256}, rewriter.getF32Type());
        auto allocaPing = rewriter.create<memref::AllocaOp>(loc, memrefType);
        allocaPing->setAttr("buffer_type", rewriter.getStringAttr("ping"));
        
        auto allocaPong = rewriter.create<memref::AllocaOp>(loc, memrefType);
        allocaPong->setAttr("buffer_type", rewriter.getStringAttr("pong"));
        
        // Create lock types and initialize locks
        auto lockType = dfschedule::LockType::get(rewriter.getContext());
        
        auto pingAcquireLock = rewriter.create<dfschedule::LockInitOp>(
            loc, lockType, rewriter.getI64IntegerAttr(0));
        pingAcquireLock->setAttr("sym_name", rewriter.getStringAttr("ping_aquire_lock"));
        
        auto pongAcquireLock = rewriter.create<dfschedule::LockInitOp>(
            loc, lockType, rewriter.getI64IntegerAttr(0));
        pongAcquireLock->setAttr("sym_name", rewriter.getStringAttr("pong_aquire_lock"));
        
        auto pingReleaseLock = rewriter.create<dfschedule::LockInitOp>(
            loc, lockType, rewriter.getI64IntegerAttr(1));
        pingReleaseLock->setAttr("sym_name", rewriter.getStringAttr("ping_release_lock"));
        
        auto pongReleaseLock = rewriter.create<dfschedule::LockInitOp>(
            loc, lockType, rewriter.getI64IntegerAttr(0));
        pongReleaseLock->setAttr("sym_name", rewriter.getStringAttr("pong_release_lock"));
        
        // Create DMA launch operation with region
        auto dmaLoop = rewriter.create<dfschedule::LaunchDmaS2MLoopOp>(
            loc, 
            allocaPing.getResult(), 
            allocaPong.getResult(), 
            funcArg,
            pingAcquireLock.getResult(), 
            pongAcquireLock.getResult(),
            pingReleaseLock.getResult(), 
            pongReleaseLock.getResult());
        
        // Create empty region for DMA loop (the region is named 'body')
        Block *dmaBlock = rewriter.createBlock(&dmaLoop.getBody());
        rewriter.setInsertionPointToEnd(dmaBlock);
        
        // Set insertion point after DMA loop for compute logic
        rewriter.setInsertionPointAfter(dmaLoop);
        
        // Create compute loop with ping-pong logic
        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        auto c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
        
        auto forOp = rewriter.create<scf::ForOp>(loc, c0, c4, c1);
        Block *forBody = forOp.getBody();
        rewriter.setInsertionPointToStart(forBody);
        
        Value loopIter = forOp.getInductionVar();
        
        // Calculate if iteration is even or odd
        auto remOp = rewriter.create<arith::RemUIOp>(loc, loopIter, c2.getResult());
        auto isEven = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, remOp.getResult(), c0.getResult());
        
        // Select buffer based on even/odd
        auto ifEvenBuf = rewriter.create<scf::IfOp>(
            loc, memrefType, isEven, true);
        
        rewriter.setInsertionPointToStart(&ifEvenBuf.getThenRegion().front());
        rewriter.create<scf::YieldOp>(loc, allocaPing.getResult());
        
        rewriter.setInsertionPointToStart(&ifEvenBuf.getElseRegion().front());
        rewriter.create<scf::YieldOp>(loc, allocaPong.getResult());
        
        rewriter.setInsertionPointAfter(ifEvenBuf);
        Value selectedBuffer = ifEvenBuf.getResult(0);
        
        // Select acquire lock
        auto ifEvenAcq = rewriter.create<scf::IfOp>(
            loc, lockType, isEven, true);
        
        rewriter.setInsertionPointToStart(&ifEvenAcq.getThenRegion().front());
        rewriter.create<scf::YieldOp>(loc, pingAcquireLock.getResult());
        
        rewriter.setInsertionPointToStart(&ifEvenAcq.getElseRegion().front());
        rewriter.create<scf::YieldOp>(loc, pongAcquireLock.getResult());
        
        rewriter.setInsertionPointAfter(ifEvenAcq);
        Value selectedAcquireLock = ifEvenAcq.getResult(0);
        
        // Select release lock
        auto ifEvenRel = rewriter.create<scf::IfOp>(
            loc, lockType, isEven, true);
        
        rewriter.setInsertionPointToStart(&ifEvenRel.getThenRegion().front());
        rewriter.create<scf::YieldOp>(loc, pingReleaseLock.getResult());
        
        rewriter.setInsertionPointToStart(&ifEvenRel.getElseRegion().front());
        rewriter.create<scf::YieldOp>(loc, pongReleaseLock.getResult());
        
        rewriter.setInsertionPointAfter(ifEvenRel);
        Value selectedReleaseLock = ifEvenRel.getResult(0);
        
        // Calculate lock value (iteration + 1)
        auto c1_i32 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(1));
        auto idxCast = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI32Type(), loopIter);
        auto lockVal = rewriter.create<arith::AddIOp>(
            loc, idxCast.getResult(), c1_i32.getResult());
        
        // Acquire lock
        rewriter.create<dfschedule::AcquireLockOp>(
            loc, selectedAcquireLock, lockVal.getResult());
        
        // Inner compute loop
        auto c0_inner = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto c1_inner = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto c10 = rewriter.create<arith::ConstantIndexOp>(loc, 10);
        
        auto innerForOp = rewriter.create<scf::ForOp>(
            loc, c0_inner, c10, c1_inner);
        Block *innerForBody = innerForOp.getBody();
        rewriter.setInsertionPointToStart(innerForBody);
        
        // Create compute operation using dfschedule::ComputeOp
        rewriter.create<dfschedule::ComputeOp>(loc, selectedBuffer);
        
        rewriter.setInsertionPointAfter(innerForOp);
        
        // Release lock
        rewriter.create<dfschedule::ReleaseLockOp>(
            loc, selectedReleaseLock, lockVal.getResult());
        
        // Add return to function
        rewriter.setInsertionPointToEnd(funcBody);
        rewriter.create<func::ReturnOp>(loc);
        
        // Replace the original routing.RoutingCreate with the module
        rewriter.replaceOp(op, moduleOp.getOperation()->getResult(0));
        
        return success();
    }
};

} // namespace

namespace mlir {

void DmaphopTodfschedulePass::runOnOperation() {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    
    // Mark target dialects as legal
    target.addLegalDialect<dfschedule::dfscheduledialect, 
                          func::FuncDialect,
                          memref::MemRefDialect,
                          arith::ArithDialect,
                          scf::SCFDialect,
                          BuiltinDialect>();
    
    // Mark routing operations as illegal to trigger conversion
    target.addIllegalOp<routing::RoutingCreate>();
    
    RewritePatternSet patterns(context);
    patterns.add<RoutingCreateTodfschedulePattern>(context);
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace mlir
