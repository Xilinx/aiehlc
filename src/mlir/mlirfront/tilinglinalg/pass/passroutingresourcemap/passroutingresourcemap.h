/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_RESOURCEMAP_PASS_H__
#define __ROUTING_RESOURCEMAP_PASS_H__

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "routinghwmanager.h"
#include "routingmanager.h"

#include <string>

namespace mlir {

class RoutingResourceMapPass : public PassWrapper<RoutingResourceMapPass, OperationPass<>> {
  public:
    RoutingResourceMapPass() = default;
    explicit RoutingResourceMapPass(const std::string &outputDir) : outputDir(outputDir) {}
    RoutingResourceMapPass(const std::string &outputDir, int startCol) : outputDir(outputDir), startCol(startCol) {}
    RoutingResourceMapPass(const std::string &outputDir, int startCol, const std::string &aieGen)
        : outputDir(outputDir), startCol(startCol), aieGen(aieGen) {}

    void runOnOperation() override;

    StringRef getArgument() const override { return "routing-resource-map"; }
    StringRef getDescription() const override {
        return "Generate physical routing resource map JSON (tile/port/pktid/mask) from routinghw IR";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<routinghw::RoutingHWDialect>();
        registry.insert<routing::routingdialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<scf::SCFDialect>();
    }

  private:
    std::string outputDir;
    int startCol = -1;
    std::string aieGen;
};

} // namespace mlir

#endif
