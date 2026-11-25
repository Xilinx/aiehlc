/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __AIEGRAPH_MANAGER__
#define __AIEGRAPH_MANAGER__
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// fieldparser
#include "mlir/IR/DialectImplementation.h"

#include "aiegraphdialect.h.inc"

#include "aiegraphenums.h.inc"

using namespace mlir;
using namespace aiegraph;

#define GET_ATTRDEF_CLASSES
#include "aiegraphattr.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "aiegraphtype.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#define GET_OP_DECLS
#include "aiehlcsymbol.h"
#include "aiegraphop.h.inc"
#undef GET_OP_DECLS
#undef GET_OP_CLASSES

class aiegraphmanager {
  public:
    aiegraphmanager() {};
    // Build a small well-formed graph for round-trip testing.
    ModuleOp ops_test(MLIRContext *ctx);
    static void loaddialect(MLIRContext *ctx);
};
#endif //__AIEGRAPH_MANAGER__
