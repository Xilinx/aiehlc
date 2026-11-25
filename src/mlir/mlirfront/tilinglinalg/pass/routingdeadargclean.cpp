/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "routingdeadargclean.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
void RoutingDeadArgPass::runOnOperation()  {
    
    auto mop = getOperation();
    if (auto module = dyn_cast<ModuleOp>(*mop)) {
        mlir::SymbolTable symbolTable(module);
        // We collect all functions to modify first, to avoid iterator invalidation issues.
        std::vector<mlir::func::FuncOp> functionsToProcess;
        for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
            functionsToProcess.push_back(funcOp);
        }

        for (auto oldFunc : functionsToProcess) {
        // We can only modify functions that are defined in this module.
        // We cannot change the signature of external library functions.
            if (oldFunc.isExternal() || oldFunc.getBody().empty()) {
                continue;
            }

            // Step 1: Analyze arguments to find which ones are dead.
            // use_empty() only checks direct uses; we must also check uses inside
            // nested regions (e.g. emitc.if bodies) by walking all ops in the function.
            llvm::DenseSet<unsigned> deadArgIndices;
            for (unsigned i = 0; i < oldFunc.getNumArguments(); ++i) {
                mlir::Value arg = oldFunc.getArgument(i);
                bool hasUses = !arg.use_empty();
                if (!hasUses) {
                    // Walk all nested ops to detect uses inside regions
                    oldFunc.walk([&](mlir::Operation *op) {
                        if (hasUses)
                            return;
                        for (auto operand : op->getOperands()) {
                            if (operand == arg) {
                                hasUses = true;
                                return;
                            }
                        }
                    });
                }
                if (!hasUses) {
                    deadArgIndices.insert(i);
                }
            }

        // If no arguments are dead, there's nothing to do for this function.
            if (deadArgIndices.empty()) {
                continue;
            }

        // Step 2a: Create the new function type without the dead arguments.
            llvm::SmallVector<mlir::Type> newArgTypes;
            for (unsigned i = 0; i < oldFunc.getNumArguments(); ++i) {
                if (deadArgIndices.find(i) == deadArgIndices.end()) { // If not dead
                    newArgTypes.push_back(oldFunc.getArgument(i).getType());
                }
            }
            auto newFuncType = mlir::FunctionType::get(
                &getContext(), newArgTypes, oldFunc.getFunctionType().getResults());

            ///
            std::string typeAsString;
            llvm::raw_string_ostream stream(typeAsString);
            stream << newFuncType;
            llvm::errs() << "The type stored in a string is: " << typeAsString << "\n";
            ///

        // Step 2b: Create the new function.
            mlir::OpBuilder builder(module.getBodyRegion());
            auto newFunc = builder.create<mlir::func::FuncOp>(
                oldFunc.getLoc(), oldFunc.getName(), newFuncType);
            {
                newFunc->print(llvm::errs());
                llvm::errs() << "\n";
            }
        // Copy all attributes (like linkage visibility) from the old function.
        // Copy attributes individually, skipping the ones handled by the builder ---
            //newFunc->setAttrs(oldFunc->getAttrs());
            for (const mlir::NamedAttribute &attr : oldFunc->getAttrs()) {
                // The builder already set sym_name and function_type. Don't overwrite them.
                if (attr.getName() == mlir::SymbolTable::getSymbolAttrName()||
                    attr.getName() == "function_type") {
                    continue;
                }
                //newFunc->setAttr(attr.getName(), attr.getValue());
            }

            // Step 2c: Clone the body and map the live arguments.
            // addEntryBlock() must be called before getArgument() so the new
            // function's body has block arguments to index into.
            mlir::Block *newEntryBlock = newFunc.addEntryBlock();
            mlir::Block *oldEntryBlock = &oldFunc.front();

            mlir::IRMapping mapping;
            unsigned newArgIdx = 0;
            for (unsigned oldArgIdx = 0; oldArgIdx < oldFunc.getNumArguments(); ++oldArgIdx) {
                if (deadArgIndices.find(oldArgIdx) == deadArgIndices.end()) { // If not dead
                    mapping.map(oldFunc.getArgument(oldArgIdx), newFunc.getArgument(newArgIdx++));
                }
            }

            newEntryBlock->getOperations().splice(
                newEntryBlock->end(),                // Point to insert at
                oldEntryBlock->getOperations(),      // Source list of ops
                oldEntryBlock->begin(),              // Start of range to move
                std::prev(oldEntryBlock->end()));    // End of range (all but terminator)

            // Remap operands of spliced ops: splice moves ops without
            // remapping SSA values, so any op using an old block arg
            // (e.g. the XAie_DevInst* dev param) still references the
            // old function's block arg. Walk ALL ops including those
            // inside nested regions (e.g. emitc.if bodies) and replace.
            newFunc.walk([&](mlir::Operation *op) {
                for (unsigned i = 0; i < op->getNumOperands(); i++) {
                    auto mapped = mapping.lookupOrNull(op->getOperand(i));
                    if (mapped)
                        op->setOperand(i, mapped);
                }
            });

            // Now, clone the terminator operation, remapping its operands.
            builder.setInsertionPointToEnd(newEntryBlock);
            builder.clone(*oldEntryBlock->getTerminator(), mapping);
            //dFunc.getBody().cloneInto(&newFunc.getBody(), mapping);
            {
                newFunc->print(llvm::errs());
                llvm::errs() << "\n";
            }
        // Step 2d: Update all call sites.
            auto uses = symbolTable.getSymbolUses(oldFunc, module);
            if (uses) {
                for (auto use : *uses) {
                    auto callOp = mlir::cast<mlir::func::CallOp>(use.getUser());
                    builder.setInsertionPoint(callOp);

                    llvm::SmallVector<mlir::Value> newOperands;
                    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
                        if (deadArgIndices.find(i) == deadArgIndices.end()) { // If not dead
                            newOperands.push_back(callOp.getOperand(i));
                        }
                    }
            
            // Create the new call to the new function.
                    auto newCall = builder.create<mlir::func::CallOp>(
                        callOp.getLoc(), newFunc, newOperands);
            
            // Replace the old call's results with the new one's.
                    callOp.replaceAllUsesWith(newCall.getResults());
            
            // Erase the old call.
                    callOp.erase();
                }
            }

        // Step 2e: Erase the old function now that it has no uses.
            oldFunc.erase();
        }
        //after remove the dead arg, the local variable may become dead code remove them
      
        auto &ctx = getContext();
        mlir::RewritePatternSet patternscde(&ctx);
        for (auto* dialect : ctx.getLoadedDialects()) {
            dialect->getCanonicalizationPatterns(patternscde);
        }
        for (mlir::RegisteredOperationName op : ctx.getRegisteredOperations()) {
            op.getCanonicalizationPatterns(patternscde, &ctx);
        }
        // Apply the patterns greedily. This will simplify and clean up the IR.
        if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patternscde)))) {
            // Handle error
            signalPassFailure();
        }
    }
}
