/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include "routingmanager.h"
#include <iostream>

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#define GET_OP_DEFS
//#define GET_OP_LIST
#include "routinginterface.cc.inc"
#include "routingdialect.cc.inc"
#include "routingattr.cc.inc"
#include "routingtype.cc.inc"

#include "routingop.cc.inc"
//#undef GET_OP_LIST
#undef GET_OP_DEFS

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

void routingdialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "routingop.cc.inc"
        >();
    // 如果有 Attr / Type：addAttributes<...>(); addTypes<...>();
     addAttributes<
    #define GET_ATTRDEF_LIST
    #include "routingattr.cc.inc"
    >();

        // 3. Types
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "routingtype.cc.inc"
    >();
}
// implment interface api
int64_t tilearrayType::getTileBase() const {
    return 100;
}

// implement creattilearrayop result assembly
LogicalResult createtilearrayOp::inferReturnTypes(mlir::MLIRContext* ctx, 
    std::optional<mlir::Location>, 
    mlir::ValueRange, 
    mlir::DictionaryAttr args, 
    mlir::OpaqueProperties, 
    mlir::RegionRange, 
    llvm::SmallVectorImpl<mlir::Type>& infer) {
        //args is arguments
        //auto items = args.get("items").dyn_cast_or_null<TileRangeArrayAttr>();
        //if (!items) return failure();
    TileRangeAttr ta = TileRangeAttr::get(ctx, -1,-1);
    llvm::SmallVector<TileRangeAttr,4> slist;
    slist.push_back(ta);
    infer.push_back(tilearrayType::get(ctx,slist));

    return success();

}

LogicalResult createdataio::inferReturnTypes(mlir::MLIRContext* ctx, 
    std::optional<mlir::Location>, 
    mlir::ValueRange, 
    mlir::DictionaryAttr args, 
    mlir::OpaqueProperties, 
    mlir::RegionRange, 
    llvm::SmallVectorImpl<mlir::Type>& infer) {
        //args is arguments
        auto items = args.get("iotype").dyn_cast_or_null<StringAttr>();
        auto items2 = args.get("direction").dyn_cast_or_null<StringAttr>();
        if (!items||!items) return failure();
        infer.push_back(dataioType::get(ctx,items, items2));

    return success();

}

// In RoutingOps.cpp

// This is the C++ implementation for the printer.
void routing::RoutingCreate::print(OpAsmPrinter &p) {
  // `p` is the printer object. The `<<` operator prints literal strings.
  //p << " on_row ";

  // Use printAttribute to print attributes. Angle brackets are just literals.
  p << "<";
  p << "Memo = \"";
  p << (getMemo());
  p << "\">";

  // Use printOperand for SSA values, and print its type.
  p << " ( scf_idx = ";
  p.printOperand(getScfIdx());
  p << " : ";
  p.printType(getScfIdx().getType());
  p << ")";

  // Use printOptionalAttrDict to print any attributes we haven't
  // explicitly printed. This is good practice for forward compatibility.
  // We need to tell it which attributes we already handled.
  //p.printOptionalAttrDict(this->getAttrs(), /*elidedAttrs=*/{"device_row"});

  // Print the result type.
  p << " -> ";
  p.printType(getResult().getType());
  
  // Use printRegion to print the region.
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true, 
                              /*printBlockTerminators=*/false);
}

// In RoutingOps.cpp

// This is the C++ implementation for the parser.
ParseResult RoutingCreate::parse(OpAsmParser &parser, OperationState &result) {
  // --- 1. Parse the components of the op ---

  // `result` is the blueprint we will populate.
  
  // Parse the keyword "on_row"
  if (parser.parseKeyword("on_row"))
    return failure();
  
  // Parse the device_row attribute, which is inside <>
  IntegerAttr deviceRowAttr;
  if (parser.parseLess() ||
      parser.parseAttribute(deviceRowAttr, "device_row", result.attributes) ||
      parser.parseGreater())
    return failure();

  // Parse the input operand and its type
  OpAsmParser::UnresolvedOperand inputOperand;
  Type inputType;
  if (parser.parseLParen() || 
      parser.parseOperand(inputOperand) || 
      parser.parseColon() ||
      parser.parseType(inputType) || 
      parser.parseRParen())
    return failure();

  // Parse any optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the arrow and result type
  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  
  // Parse the region
  Region *body = result.addRegion();
  // 1. Create a list to hold the region's expected arguments.
  llvm::SmallVector<OpAsmParser::Argument> regionArgs;

    // 2. Create an Argument struct for our input. We can leave the SSA name
    //    and location fields empty as the parser will fill them.
  OpAsmParser::Argument arg; 
  arg.type = inputType;
  regionArgs.push_back(arg);
  // 3. Pass the list of arguments to parseRegion.
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // --- 2. Populate the OperationState (the result) ---

  // Add the result type to the blueprint
  result.addTypes(resultType);
  
  // Resolve the parsed operand and add it to the blueprint
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return failure();

  return success();
}

//routing class
void routingmanager::type_interface_test(MLIRContext* ctx) {
        //ctx->getOrLoadDialect<routing::routingdialect>();
        TileRangeAttr tarrayattr = TileRangeAttr::get(ctx, 1, 1);
        llvm::SmallVector<TileRangeAttr,4> slist;
        slist.push_back(tarrayattr);
        tilearrayType ttype = tilearrayType::get(ctx, slist);
        if (auto tt = dyn_cast<TileInterface>(ttype)) {
            llvm::outs() << tt.getTileBase() << "\n";
        }
        auto len = ttype.getItems().size();
        llvm::outs() << len << " is the length\n"; 
    }
ModuleOp routingmanager::ops_test(MLIRContext* ctx, int totalN) {
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    auto func = createroutingfunc(ctx, totalN);
    m.push_back(func);
    auto functype = builder.getFunctionType({},{});
    mlir::func::FuncOp main = builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", functype);
    auto block = main.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    //Value cnum   = builder.create<arith::ConstantIndexOp>(location, 1);
    // cnum and rnum is not the ARRAY/hw shape, it means reserve how many row , how many col used to do broadcast
    Value cnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    Value rnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 8);
    //Value total = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),16);
    auto callop = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), func, ValueRange({cnum, rnum}));
    auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    m.push_back(main);
    llvm::errs() << m;
    return m;
}

ModuleOp routingmanager::ops_testNew(MLIRContext* ctx, int totalN) {
    const int hwrowused= 4, hwcolused=2;
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    //auto func = createroutingfuncByDim(ctx, true);
    //m.push_back(func);
    auto functype = builder.getFunctionType({},{});
    
    mlir::func::FuncOp main = builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", functype);
    
    auto block = main.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    auto mesh = builder.create<createhwmesh>(builder.getUnknownLoc(),  hwrowused, hwcolused);
    
    //dummy tensor
    SmallVector<Attribute> shape;
    for (int64_t v : {10, 20})
    shape.push_back(builder.getI64IntegerAttr(v));
    ArrayAttr vals = builder.getArrayAttr(shape);  // satisfies I64ArrayAttr
    IntegerAttr dimnum = builder.getI64IntegerAttr(2);
    auto tensor = builder.create<createdummytensor>(builder.getUnknownLoc(), vals, dimnum);
    //convert into index typ
    //auto mesh_index = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(),builder.getIndexType(), mesh);
    //auto tensor_index = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(),builder.getIndexType(), tensor);
    
    //Value cnum   = builder.create<arith::ConstantIndexOp>(location, 1);
   // Value cnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    //Value rnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 8);
    //Value total = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),16);
    /*
    auto callop = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), func, ValueRange({mesh, 
                                                                                                tensor,
                                                                                                splitnum,
                                                                                                axis,
                                                                                                partensor_splitdim,
                                                                                                partensor_axis_owner,
                                                                                                partensor_replicate_on,
                                                                                                partensor_singleowner,
                                                                                                io_direction}));
    //*/
    //createroutingfuncByDim(builder, ctx, false, mesh, tensor, hwrowused, "row");
    createroutingfuncByDim(builder, ctx, true, mesh, tensor, hwrowused, "row");
    //createroutingfuncByDim(builder, ctx, true, mesh, tensor, hwcolused, "col");
    auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    m.push_back(main);
    llvm::errs() << m;
    return m;
}

void routingmanager::loaddialect(MLIRContext* ctx) {
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<routing::routingdialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
}
mlir::func::FuncOp routingmanager::createroutingfunc(MLIRContext* ctx, int totalN, bool purefunc) {
        OpBuilder builder(ctx);
        auto location = builder.getUnknownLoc();
        //ModuleOp module= ModuleOp::create(builder.getUnknownLoc());
        IndexType itype= builder.getIndexType();
        mlir::FunctionType ftype = builder.getFunctionType({itype,itype},{});
        func::FuncOp func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "createroute", ftype);
        Block* eb = func.addEntryBlock();
        builder.setInsertionPointToStart(eb);

        auto getconstant = [&](Value value) -> int {
            if (auto definingOp = value.getDefiningOp()) {
                if (auto constantOp = dyn_cast<arith::ConstantIntOp>(definingOp)) {
                    if (auto intAttr = constantOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
                        return intAttr.getInt();
                    }
                }
            }
            return 0;
        };

        Value row = eb->getArgument(0);
        Value col = eb->getArgument(1);
        //Value total_col = eb->getArgument(2);

        Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
        Value ub = builder.create<arith::ConstantIndexOp>(location, totalN);
        Value step = builder.create<arith::ConstantIndexOp>(location,1);
  // ──────────────────────────────
  // 2. 创建 scf.forall
  //    OpBuilder 会回调一个 lambda，让你往 region 里塞指令
  // ──────────────────────────────
        auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
        {
            //use such format to fix the generic format print issue 
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(scf.getBody());
            Value cnum_i32, rnum_i32;
            //TileRangeArrayAttr tlist = TileRangeArrayAttr::get(ctx, slist);
            if (purefunc) {
                col   = builder.create<arith::ConstantIndexOp>(location, 1);
                row   = builder.create<arith::ConstantIndexOp>(location, 8);
            }
            cnum_i32 = builder.create<arith::IndexCastOp>(location,builder.getI32Type(), col);
            rnum_i32 = builder.create<arith::IndexCastOp>(location, builder.getI32Type(), row);

            /*
            //create mesh
            auto mesh = builder.create<createhwmesh>(builder.getUnknownLoc(),  rnum_i32, cnum_i32);
            
            auto patitionmesh = builder.create<partitionmesh>(builder.getUnknownLoc(),  mesh, cnum_i32, "row");
            //------------
            //fixme should use real subview 
            Value subview = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 32);
            SmallVector<Attribute> shape;
            for (int64_t v : {10, 20})
            shape.push_back(builder.getI64IntegerAttr(v));
            ArrayAttr vals = builder.getArrayAttr(shape);  // satisfies I64ArrayAttr
            // 3) I64Attr ($dim).
            IntegerAttr dimnum = builder.getI64IntegerAttr(2);
            auto tensor = builder.create<createdummytensor>(builder.getUnknownLoc(),  subview, vals, dimnum);
            //
            auto hw_row_number = rnum_i32;
            IntegerAttr splitdim = builder.getI64IntegerAttr(0);//dim 0 is 

            mlir::StringAttr hw_axis_owner=builder.getStringAttr("row");
		    mlir::StringAttr replicate_on=builder.getStringAttr("col");
		    mlir::StringAttr single_tile_owner=builder.getStringAttr("");
            auto rowtensor = builder.create<partitiontensor>(builder.getUnknownLoc(),  
                    tensor, hw_row_number, splitdim,hw_axis_owner,replicate_on, single_tile_owner);
            //extract data
            auto edata = builder.create<extract_data>(builder.getUnknownLoc(), rowtensor, cnum_i32);
            auto emeshtile = builder.create<extract_tiles>(builder.getUnknownLoc(), patitionmesh, cnum_i32);
            //extract tile
            */
            //
            
            auto tilearray = builder.create<createtilearrayOp>(builder.getUnknownLoc(), rnum_i32, cnum_i32);
            auto io = builder.create<createdataio>(builder.getUnknownLoc(), "mem", "input");
            
          
            auto output = builder.getI32Type();
            auto op3 = builder.create<creatbroadcast>(builder.getUnknownLoc(), io.getResult(), tilearray.getResult());
            auto result = tilearray.getResult().getType().dyn_cast<tilearrayType>();

            llvm::outs() << "result count is "<< result.getItems().size() <<" \n";
           
        }
         
         //without return the print will go into generic as some verify failed.
         auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

       
        return func;
}

void routingmanager::createroutingfuncByDim(OpBuilder& builder, MLIRContext* ctx, bool binput, Value mesh, Value tensor,
                                           uint32_t hwsplitnum, std::string splitAxis) {
        auto location = builder.getUnknownLoc();
        auto tensorhwaxisowner = splitAxis;
        // no region creatation
        //
        auto exec = builder.create<scf::ExecuteRegionOp>(builder.getUnknownLoc(), /*result types*/TypeRange{});
        exec->setAttr("routing_memo", builder.getStringAttr(splitAxis));
        //Block *body = builder.createBlock(&exec.getRegion());
        {
                    OpBuilder::InsertionGuard guard(builder);
                    //fix the upper scope return go inside this region issue
                    builder.setInsertionPointToStart(&exec.getRegion().emplaceBlock());
                ///*                
                    auto patitionmesh = builder.create<partitionmesh>(builder.getUnknownLoc(),  mesh, hwsplitnum, splitAxis);
                    IntegerAttr splitdim = builder.getI64IntegerAttr(0);//dim 0 is 
                    auto outTy = builder.getI32Type();
                    auto rowtensor = builder.create<partitiontensor>(builder.getUnknownLoc(), tensor, hwsplitnum, 0, tensorhwaxisowner,"col","");
                    Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
                    Value ub = builder.create<arith::ConstantIndexOp>(location, hwsplitnum);
                    Value step = builder.create<arith::ConstantIndexOp>(location,1);
                    // create RoutingCreate region op
                    // inside region
                    ///*
                    auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
                    { 
                        OpBuilder::InsertionGuard guard(builder);
                        builder.setInsertionPointToStart(scf.getBody());
                        auto memo = builder.getStringAttr(splitAxis);
                        mlir::Value scf_idx = scf.getInductionVar();
                        
                        Value idx = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(),builder.getI32Type(), scf_idx);
                        
                        auto routingcreateOp = builder.create<routing::RoutingCreate>(builder.getUnknownLoc(), idx, memo, [&](OpBuilder &builder1, Location bodyLoc,Value sidx) { 
                            //use such format to fix the generic format print issue 
                            ///*
                            auto slicetensor = builder1.create<extract_data>(builder1.getUnknownLoc(), rowtensor, sidx);
                            auto tilelist = builder1.create<extract_tiles>(builder1.getUnknownLoc(), patitionmesh, sidx);
                            
                            if (binput) {
                                auto hwio = builder1.create<createhwiowithtarget>(builder1.getUnknownLoc(), tilelist, "input", "mem2");
                                auto datamov = builder1.create<movedatabyio>(builder1.getUnknownLoc(), slicetensor, hwio);
                            } else {
                                auto gatherdata = builder1.create<routinggatherout>(builder1.getUnknownLoc(), tilelist, slicetensor);
                                auto hwio = builder1.create<createhwiowithtarget>(builder1.getUnknownLoc(), tilelist, "output", "mem2");
                                auto datamov = builder1.create<movedatabyio>(builder1.getUnknownLoc(), gatherdata, hwio);
                            }
                            //*/
                            //mlir::Type i32Type = builder1.getI32Type();
                            //mlir::IntegerAttr intAttr = builder1.getIntegerAttr(i32Type, 42);
                            //auto intConst = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), intAttr);
                            //builder1.create<routing::YieldOp>(builder1.getUnknownLoc(), intConst);
                            builder1.create<routing::YieldOp>(builder1.getUnknownLoc());
                            
                        });
                        
                        //extract tile
                    }
                    //*/

                    // use routing.yield to return value finish the yield
                 builder.create<scf::YieldOp>(builder.getUnknownLoc());
        };
       
        return ;//func;
}
/*
int main() {
    MLIRContext ctx;
    mlirtest mtest;
    mtest.type_interface_test(&ctx);
    mtest.ops_test(&ctx);
    std::cout << "main" <<std::endl;
    return 0;
}
    */