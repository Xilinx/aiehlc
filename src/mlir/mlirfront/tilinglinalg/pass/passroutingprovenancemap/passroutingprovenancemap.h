/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_PROVENANCEMAP_PASS_H__
#define __ROUTING_PROVENANCEMAP_PASS_H__

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "routinghwmanager.h"
#include "routingmanager.h"

#include <string>

namespace mlir {

class RoutingProvenanceMapPass : public PassWrapper<RoutingProvenanceMapPass, OperationPass<>> {
  public:
    RoutingProvenanceMapPass() = default;
    explicit RoutingProvenanceMapPass(const std::string &outputDir) : outputDir(outputDir) {}
    RoutingProvenanceMapPass(const std::string &outputDir, int startCol) : outputDir(outputDir), startCol(startCol) {}
    RoutingProvenanceMapPass(const std::string &outputDir, int startCol, const std::string &aieGen)
        : outputDir(outputDir), startCol(startCol), aieGen(aieGen) {}

    void runOnOperation() override;

    StringRef getArgument() const override { return "routing-provenance-map"; }
    StringRef getDescription() const override {
        return "Generate physical routing provenance map JSON from routinghw IR";
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
