/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <iostream>
#include "routingmanager.h"
/*
void routingdialect::initialize()  {
    addOperations<
    #define GET_OP_LIST
    #include "routingop.cc.inc"
        >();
    // if there are Attrs / Types: addAttributes<...>(); addTypes<...>();
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

//test class
class mlirtest{
public:
    mlirtest(){};
    void type_interface_test(MLIRContext* ctx) {
        ctx->getOrLoadDialect<routing::routingdialect>();
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
    void ops_test(MLIRContext* ctx) {
        ctx->getOrLoadDialect<mlir::func::FuncDialect>();
        ctx->getOrLoadDialect<routing::routingdialect>();
        ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
        ctx->getOrLoadDialect<mlir::arith::ArithDialect>();


        OpBuilder builder(ctx);
        auto location = builder.getUnknownLoc();
        ModuleOp module= ModuleOp::create(builder.getUnknownLoc());
        mlir::FunctionType ftype = builder.getFunctionType({},{});
        func::FuncOp routing = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                         "routingcreate",
                                                         ftype);
        {
        OpBuilder::InsertionGuard guard(builder);
        Block* eb = routing.addEntryBlock();
        builder.setInsertionPointToStart(eb);

        Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
        Value ub = builder.create<arith::ConstantIndexOp>(location, 16);
        Value step = builder.create<arith::ConstantIndexOp>(location,1);

        SmallVector<OpFoldResult> lbs = {builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> ubs = {builder.getIndexAttr(16)};
        SmallVector<OpFoldResult> steps = {builder.getIndexAttr(1)};
  // ──────────────────────────────
  // 2. Create the scf.forall
  //    OpBuilder calls back a lambda that lets you put ops into the region
  // ──────────────────────────────
        auto scf = builder.create<mlir::scf::ForallOp>(
                    location,
                    lbs, ubs, steps,
                    ValueRange{},
                    std::nullopt,
                    [&](OpBuilder &sbuilder, Location bloc, ValueRange ivs) {

                        Value iv = ivs.front();
                        TileRangeAttr ta = TileRangeAttr::get(ctx, 1,2);
                        llvm::SmallVector<TileRangeAttr,4> slist;
                        slist.push_back(ta);

                        //TileRangeArrayAttr tlist = TileRangeArrayAttr::get(ctx, slist);
                        Value cnum   = builder.create<arith::ConstantIndexOp>(location, 1);
                        Value rnum   = builder.create<arith::ConstantIndexOp>(location, 8);
                        createtilearrayOp op = builder.create<createtilearrayOp>(builder.getUnknownLoc(), rnum, cnum);
                        auto op2 = builder.create<createdataio>(builder.getUnknownLoc(), "mem", "input");
                        auto output = builder.getI32Type();
                        auto op3 = builder.create<creatbroadcast>(builder.getUnknownLoc(), output, op.getResult(),
op2.getResult());
                        //test the result

                        auto result = op.getResult().getType().dyn_cast<tilearrayType>();

                        llvm::outs() << "result count is "<< result.getItems().size() <<" \n";
                    }
                );
        routing.dump();
        }
        mlir::FunctionType mftype = builder.getFunctionType({},{});
        func::FuncOp mfunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                          "main",
                                                          mftype);
        {
            OpBuilder::InsertionGuard gd(builder);
            auto block = mfunc.addEntryBlock();
            builder.setInsertionPointToEnd(block);
            builder.create<func::CallOp>(mfunc.getLoc(), routing, ValueRange{});
            builder.create<func::ReturnOp>(mfunc.getLoc());
        }
        module.push_back(routing);
        module.push_back(mfunc);
        module.dump();
    }
};
*/
int main(int argc, char* argv[]) {
    MLIRContext ctx;
    routingmanager mtest;
    mtest.loaddialect(&ctx);
    mtest.type_interface_test(&ctx);
    mtest.ops_testNew(&ctx);
    std::cout << "main" <<std::endl;
    return 0;
}