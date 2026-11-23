/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// aiednn-opt: parse / verify / transform / print aiednn IR.
//   aiednn-opt model.aiednn.mlir                       # verify + print
//   aiednn-opt model.aiednn.mlir --aiednn-export-plan="out=plan.json"
#include "AiednnDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<aiednn::AiednnDialect, mlir::func::FuncDialect>();
    aiednn::registerAiednnPasses();
    mlir::registerCanonicalizerPass();
    mlir::registerCSEPass();
    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "aiednn optimizer\n", registry));
}
