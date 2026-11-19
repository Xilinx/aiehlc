/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "routingunrolling.h"
RoutingUnrollingLowerPass::RoutingUnrollingLowerPass() {
}
void RoutingUnrollingLowerPass::runOnOperation() {
    auto moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    bool scfunrollall = true;
    if (auto mop = dyn_cast<ModuleOp>(*moduleOp)) {
        // 1. Define the list of headers.
        std::vector<std::string> headers = {"stdint.h", "stdio.h", "custom_lib.h"};

        // 2. Build the MLIR ArrayAttr from the C++ string vector.
        std::vector<mlir::Attribute> headerAttrs;
        for (const auto &header : headers) {
            headerAttrs.push_back(mlir::StringAttr::get(context, header));
        }
        mlir::ArrayAttr headersAttr = mlir::ArrayAttr::get(context, headerAttrs);

        // 3. Define the attribute name.
        llvm::StringRef attrName = "codegen.headers";

        // 4. Attach the attribute to the module.
        // Use a builder to make the modification transactional.
        mlir::OpBuilder builder(context);
        mop->setAttr(attrName, headersAttr);
        //urolling
        llvm::SmallVector<mlir::Operation*, 8> constantsToErase;
        mop->walk([&](mlir::scf::ForOp forOp) {
            auto lb = forOp.getLowerBound();
            auto ub = forOp.getUpperBound();
            auto step = forOp.getStep();
            int il=0 , iu=1, is=1;
            if (auto val = mlir::getConstantIntValue(lb)) {
                il =   static_cast<int>(*val);;
            }
            if (auto val = mlir::getConstantIntValue(ub)) {
                iu =   static_cast<int>(*val);;
            }else if (auto cast = ub.getDefiningOp<mlir::arith::IndexCastOp>()) {
                if (auto val2 = mlir::getConstantIntValue(cast.getIn())) {
                    iu = static_cast<int>(*val2);
                }
            }
            if (auto val = mlir::getConstantIntValue(step)) {
                is =   static_cast<int>(*val);;
            }
            const int unrollFactor = scfunrollall ? (iu - il)/is : 1;
            llvm::outs() << "Found an scf.for loop, attempting to unroll by factor " 
                        << unrollFactor << "...\n\n";
            if (failed(mlir::loopUnrollByFactor(forOp, unrollFactor))) {
                llvm::errs() << "Failed to unroll the loop.\n";
                // 'walk' continues, so we use return to stop processing this loop.
                return; 
            }
            /*
            mlir::Operation* lbDef = lb.getDefiningOp();
            mlir::Operation* ubDef = ub.getDefiningOp();
            mlir::Operation* stepDef = step.getDefiningOp();

            if (lbDef && lbDef->use_empty()) {
                constantsToErase.push_back(lbDef);
            }
            if (ubDef && ubDef->use_empty()) {
                constantsToErase.push_back(ubDef);
            }
            if (stepDef && stepDef->use_empty()) {
                constantsToErase.push_back(stepDef);
            }
            */
            //return mlir::WalkResult::interrupt();
        });
        for (mlir::Operation *op : constantsToErase) {
            op->erase();
        }
        /// the upper bound can not be remove, we need simplify the ir to remove it
        /// remove the dead code like constant ,after unrolling the scf.for
        mlir::RewritePatternSet patterns(context);
        for (auto* dialect : context->getLoadedDialects()) {
            dialect->getCanonicalizationPatterns(patterns);
        }
        for (mlir::RegisteredOperationName op : context->getRegisteredOperations()) {
           // op.getCanonicalizationPatterns(patterns, context);
        }

            // Apply the patterns greedily. This will simplify and clean up the IR.
        if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns)))) {
            // Handle error
            signalPassFailure();
        }
            ///
    }
   
}