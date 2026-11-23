/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __DMAPHOP_MANAGER__
#define __DMAPHOP_MANAGER__
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

//fieldparser
#include "mlir/IR/DialectImplementation.h"

#include "dmaphopdialect.h.inc"

using namespace mlir;
using namespace dmaphop;

#define GET_ATTRDEF_CLASSES
#include "dmaphopattr.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "dmaphoptype.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#define GET_OP_DECLS
#include "aiehlcsymbol.h"
#include "dmaphopop.h.inc"
namespace dmaphop {
// LLVM 24 ODS emits Op::create(builder, loc, <result>, ...); a result named like its
// class (`port`, `tile`) shadowed the class inside it, so the classes are PortOp /
// TileOp and the old names stay as aliases.
using port = PortOp;
using tile = TileOp;
} // namespace dmaphop
#undef GET_OP_DECLS
#undef GET_OP_CLASSEST

//#include "../dmap/dmapmanager.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "routingmanager.h"

class dmaphopmanager{
public:
    dmaphopmanager(){};
    ModuleOp ops_test(MLIRContext* ctx,int totalN=2) ;
    void createdmaphopfuncByDim(OpBuilder& builder, MLIRContext* ctx,SymbolTable& symTable);
    static void loaddialect(MLIRContext* ctx);
};
#endif//__DMAPHOP_MANAGER__