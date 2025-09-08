/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#ifndef __ROUTING_MANAGER__
#define __ROUTING_MANAGER__
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

#include "routinginterface.h.inc"

#include "routingdialect.h.inc"

using namespace mlir;
using namespace routing;

#define GET_ATTRDEF_CLASSES
#include "routingattr.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "routingtype.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#include "routingop.h.inc"
#undef GET_OP_CLASSEST

class routingmanager{
public:
    routingmanager(){};
    void type_interface_test(MLIRContext* ctx) ;
    ModuleOp ops_test(MLIRContext* ctx,int totalN=2) ;
    ModuleOp ops_testNew(MLIRContext* ctx,int totalN=2) ;
    mlir::func::FuncOp createroutingfunc(MLIRContext* ctx, int totalN = 16,bool purefunc=false) ;
    mlir::func::FuncOp createroutingfuncByDim(MLIRContext* ctx,bool braodcastbyrow,bool purefunc=false) ;
    static void loaddialect(MLIRContext* ctx);
};
#endif//__ROUTING_MANAGER__