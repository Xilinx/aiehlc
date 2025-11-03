/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include "dmaphopmanager.h"
#include <iostream>

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#define GET_OP_DEFS
//#define GET_OP_LIST
#include "dmaphopdialect.cc.inc"
#include "dmaphopattr.cc.inc"
#include "dmaphoptype.cc.inc"

#include "dmaphopop.cc.inc"
//#undef GET_OP_LIST
#undef GET_OP_DEFS

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


void dmaphopdialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "dmaphopop.cc.inc"
        >();
    addAttributes<
    #define GET_ATTRDEF_LIST
    #include "dmaphopattr.cc.inc"
    >();
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "dmaphoptype.cc.inc"
    >();
}

ModuleOp dmaphopmanager::ops_test(MLIRContext* ctx, int totalN) {
    const int hwrowused= 4, hwcolused=4;
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    //auto func = createdmapfuncByDim(ctx, true);
    //m.push_back(func);
    auto functype = builder.getFunctionType({},{});
    
    auto main = builder.create<dmaphop::FuncOp>(builder.getUnknownLoc(), "main", functype);
    m.push_back(main);
    //auto block = main.addEntryBlock();
    auto &block = main.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    SymbolTable symTable(main);

    createdmaphopfuncByDim(builder, ctx, symTable);
    auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    

    

  // ------------------------------------------------------------------
  // 4. Look up the symbol anywhere inside the module
  // ------------------------------------------------------------------
   Operation *found = symTable.lookup("receive1");
   if (!found) {
     llvm::errs() << "Symbol @receive1 not found!\n";
     
 
   } else {
     llvm::outs() << "receive1 found \n";
     llvm::outs() << "Found: " << found->getName() <<"\n";
   }
    llvm::errs() << m;
    return m;
}

void dmaphopmanager::loaddialect(MLIRContext* ctx) {
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<dmaphop::dmaphopdialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
}
/*
      %data = dataflowmap.create_data {type="i32", dim1=10, dim2 20}
      %coreenginegroup = dataflowmap.dmap_create_core_engine_group {1, 4, "row"}
      %ioengine = dataflowmap.dmap_create_io_engine {0, "shim"}
      %send_port = dataflowmap.configure_port %ioengine {peorioidx=0, dataacces=}  
      %portreceive1 = dataflowmap.configure_port on %{peorioidx = 0, dataacces=}  
      %portreceive2 = dataflowmap.configure_port on %{peorioidx = 1, dataacces=}   
      %receive_group = dataflowmap.create_port_group(%portreceive1, %portreceive2)    
      %broadcast_stream = dataflowmap.create_stream %send_port, %receive_group         
      dataflowmap.push %data, %broadcast_stream {cache_policy = "force_memtile" }
*/
void dmaphopmanager::createdmaphopfuncByDim(OpBuilder& builder, MLIRContext* ctx, SymbolTable& symTable) {
    auto location = builder.getUnknownLoc();
    
    // Create function type and operation
    //auto funcType = builder.getFunctionType({}, {});  // No inputs/outputs
    //auto func = builder.create<dmaphop::FuncOp>(
    //    location, 
    //    "cache_and_forward_broadcast", 
    //    funcType
    //);
    
    // Create entry block in the function
    //Block* entry = func.addEntryBlock();
    //builder.setInsertionPointToStart(entry);

    // --- 1. Define Physical Tile Instances ---
    auto shimTile = builder.create<dmaphop::tile>(location, 
        builder.getStringAttr("shim"),
        builder.getI64IntegerAttr(0),  // col
        builder.getI64IntegerAttr(0)   // row
    );

    auto tileA = builder.create<dmaphop::tile>(location,
        builder.getStringAttr("core"),
        builder.getI64IntegerAttr(1),  // col
        builder.getI64IntegerAttr(2)   // row
    );
    
    auto tileB = builder.create<dmaphop::tile>(location,
        builder.getStringAttr("core"),
        builder.getI64IntegerAttr(2),  // col
        builder.getI64IntegerAttr(2)   // row
    );
 
    // --- 2. Define Logical Ports ---
    auto directionSend =  "Out";
    auto directionReceive =  "In";
    
    // SHIM output port
    auto portShimOut = builder.create<dmaphop::port>(location,
        shimTile.getResult(),
        directionSend,
        "portShimOut",
        builder.getI64IntegerAttr(0)  // channel
    );
    symTable.insert(portShimOut);
    
    // Tile A ports
    auto portAIn = builder.create<dmaphop::port>(location,
        tileA.getResult(),
        directionReceive,
        "portAIn",
        builder.getI64IntegerAttr(0)
    );
    symTable.insert(portAIn);
    
    auto portAOut = builder.create<dmaphop::port>(location,
        tileA.getResult(),
        directionSend,
        "portAOut",
        builder.getI64IntegerAttr(0)
    );
    symTable.insert(portAOut);
    
    // Tile B input port
    auto portBIn = builder.create<dmaphop::port>(location,
        tileB.getResult(),
        directionReceive,
        "portBIn",
        builder.getI64IntegerAttr(0)
    );
    symTable.insert(portBIn);
 
    // --- 3. Define Hops ---
    auto hop1 = builder.create<dmaphop::create_hop>(location,
        portShimOut.getResult(),
        portAIn.getResult()
    );
    
    auto hop2 = builder.create<dmaphop::create_hop>(location,
        portAOut.getResult(),
        portBIn.getResult()
    );

     
    mlir::SymbolRefAttr symbolRefA = mlir::SymbolRefAttr::get(ctx,"portAIn");
    mlir::SymbolRefAttr symbolRefB = mlir::SymbolRefAttr::get(ctx,"portBIn");
        //auto useOp = builder.create<dmap::UseSymbolOp>(builder.getUnknownLoc(), symbolRef);
    // --- 4. Define Path with Consumers and Tee Points ---
    SmallVector<Attribute> consumers{symbolRefA, symbolRefB};

    SmallVector<Attribute> teePoints{symbolRefA};
    
    auto serialPath = builder.create<dmaphop::create_path>(location,
        ValueRange{hop1, hop2},
        builder.getArrayAttr(consumers),
        builder.getArrayAttr(teePoints)
    );

    // --- 5. Prepare Data and Buffers ---
    // Create memref type for the buffers
    auto memrefType = MemRefType::get({1024}, builder.getF32Type());

    auto memrefTypeOut = MemRefType::get({1024}, builder.getF32Type());

    // Allocate source data buffer
    auto data = builder.create<memref::AllocOp>(location, memrefType);
    
    // Allocate destination buffers on tiles
    auto bufferA = builder.create<dmaphop::alloc_buffer>(location,
        memrefTypeOut,
        tileA.getResult(),
        data.getResult()
    );
    
    auto bufferB = builder.create<dmaphop::alloc_buffer>(location,
        memrefTypeOut,
        tileB.getResult(),
        data.getResult()
    );

    // --- 6. Execute Transfer ---
    builder.create<dmaphop::push>(location,
        data.getResult(),
        serialPath.getResult(),
        ValueRange{bufferA, bufferB},
        ValueRange{portAIn, portBIn}
    );

    // --- 7. Synchronization ---
    builder.create<dmaphop::sync>(location, serialPath.getResult());
///*
    // --- 8. Cleanup ---
    builder.create<dmaphop::dealloc_buffer>(location, bufferA);
    builder.create<dmaphop::dealloc_buffer>(location, bufferB);
    builder.create<memref::DeallocOp>(location, data);

    // Add return
    builder.create<func::ReturnOp>(location);
   // */
}
