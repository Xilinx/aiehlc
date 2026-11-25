/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "routinghwmanager.h"
#include <iostream>

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#include "routinghwdialect.cc.inc"
#include "routinghwattr.cc.inc"
#include "routinghwtype.cc.inc"

#include "routinghwop.cc.inc"
#include <sstream>
#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

void RoutingHWDialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "routinghwop.cc.inc"
        >();
    // if there are Attrs / Types: addAttributes<...>(); addTypes<...>();
    addAttributes<
    #define GET_ATTRDEF_LIST
    #include "routinghwattr.cc.inc"
        >();

        // 3. Types
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "routinghwtype.cc.inc"
    >();
}

ModuleOp routinghwmanager::ops_test(MLIRContext* ctx) {
    auto func = createroutingfunc(ctx);
    OpBuilder builder(ctx);
    ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    m.push_back(func);
    {
        m->print(llvm::errs());
        llvm::errs() << "\n";
    }

    return m;
}
mlir::func::FuncOp routinghwmanager::createroutingfunc(MLIRContext* ctx) {
    func::FuncOp func;
    OpBuilder builder(ctx);
    auto functype = builder.getFunctionType({},{});
    func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "routinghw", functype);
    auto block = func.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    auto output = builder.getI32Type();
    auto routinghandle = builder.create<TileArrayHandleCreate>(builder.getUnknownLoc(), output, "col1");
    auto routingshimio = builder.create<IOShimTileCreate>(builder.getUnknownLoc(), output, 0, 2,1, "col2",0,0);
    SmallVector<mlir::Value, 4> tileArray;
    for (int i = 0; i < 8 ; i++) {
        std::ostringstream ostr;
        ostr << "tile" << i + 1;
        auto tile1 = builder.create<TileCreate>(builder.getUnknownLoc(), output, 3+i, 0, ostr.str());
        tileArray.push_back(tile1);
    }
    for (int i = 0; i < 8; i++) {
        auto tile = tileArray[i];
        auto result = builder.create<ConnectIOToTile>(builder.getUnknownLoc(), output, routingshimio.getResult(), tile);
    }
    //ConnecIOToTileArray
    builder.create<ConnecIOToTileArray>(builder.getUnknownLoc(), output, routingshimio.getResult(),routinghandle.getResult());
    Value shimport = builder.create<CreateShimStreamSwitchPort>(builder.getUnknownLoc(), output, routingshimio.getResult(), "SOUTH",0,0);
    builder.create<ConnectStreamSwitchPort>(builder.getUnknownLoc(), output,routingshimio.getResult(),shimport, tileArray[0], 
                                            "SOUTH", 0, "NORTH", 0,
                                            "SOUTH", 0, "NORTH", 0);

    
    builder.create<ConnectStreamSingleSwitchPort>(builder.getUnknownLoc(), output, tileArray[1], "SOUTH", 0, "NORTH", 0);
    builder.create<EnableExtToAieShimPort>(builder.getUnknownLoc(), output, tileArray[1], "SOUTH", 0);
    
    auto ret = builder.create<func::ReturnOp>(builder.getUnknownLoc());
    return func;
}
void routinghwmanager::loaddialect(MLIRContext* ctx){
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<routinghw::RoutingHWDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    //
    ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::emitc::EmitCDialect>();
    //ctx->getOrLoadDialect<mlir::DLTIDialect>();

};