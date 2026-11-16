/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "tilinglinalg.h"

tilinglinalg::tilinglinalg() {
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::transform::TransformDialect>();

    mlir::DialectRegistry registry;
    mlir::linalg::registerTransformDialectExtension(registry);
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    
    //registry.addExtension<mlir::transform::LinalgTransformOps>();
    //mlir::MLIRContext context2(registry);

    //mlir::transform::registerTransformDialectExtension<mlir::linalg::LinalgTransformOpsExtension>(context);

    //mlir::OpBuilder builder(&context);
    //module = mlir::ModuleOp::create(builder.getUnknownLoc());
    context.appendDialectRegistry(registry);
}

bool  isSignlessInteger(mlir::Type * t)  {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(*t)) {
    llvm::outs() << "is IntegerType singless is " << intTy.isSignless() << "\n";
    return intTy.isSignless();
  } else {
    llvm::outs() << "is not IntegerType" << "\n";
  }
  return false;
}

void tilinglinalg::creatematmul(int m, int k, int n) {
    //test();
    //return;
    //linalg matmul
    routingmanager::loaddialect(&context);
    //
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto itype = builder.getIntegerType(8);
    auto i32type = builder.getIntegerType(32);
    //itype.isSignlessIntOrIndex()
    llvm::outs() << "i8type is singless ? = " << isSignlessInteger(&itype) << " i32 is " << i32type.isSignlessIntOrIndex() << "\n"; 
    auto memrefA = mlir::MemRefType::get({m,k},itype);
    auto memrefB = mlir::MemRefType::get({k,n},itype);
    auto memrefC = mlir::MemRefType::get({m,n},itype);
    // Define function type
    mlir::FunctionType funcType = builder.getFunctionType({memrefA, memrefB, memrefC}, {});
  
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    
    mlir::Block *entryBlock = func.addEntryBlock();
    // Set insertion point to the beginning of the block

    builder.setInsertionPointToStart(entryBlock);
  
    mlir::Value a = entryBlock->getArgument(0);
    mlir::Value b = entryBlock->getArgument(1);
    mlir::Value c = entryBlock->getArgument(2);

    builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(),
                                     mlir::ValueRange{}, mlir::ValueRange{a, b}, mlir::ValueRange{c});

  
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    auto sattr = mlir::SymbolRefAttr::get(&context, "do_tiling");
    module->setAttr("transform.with_named_sequence", sattr);
  
    module.push_back(func);
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }

    //tranform logic
    builder.setInsertionPointToEnd(module.getBody());
    auto anyOpType = mlir::transform::AnyOpType::get(&context);  
    auto seq = builder.create<mlir::transform::NamedSequenceOp>(builder.getUnknownLoc(),
                    /*sym_name=*/builder.getStringAttr("do_tiling"),
                   /*function_type=*/ mlir::TypeAttr::get(mlir::FunctionType::get(&context, mlir::TypeRange{anyOpType}, mlir::TypeRange{})),
                   /*sym_visibility*/ mlir::StringAttr::get(&context, "public"),
                   /*arg_attrs=*/mlir::ArrayAttr::get(&context, mlir::DictionaryAttr::get(&context, {})),
                   /*res_attrs=*/mlir::ArrayAttr::get(&context, mlir::ArrayRef<mlir::Attribute>()));
    auto seqb = seq.addEntryBlock();
    builder.setInsertionPointToStart(seqb);
    ///*
    //mlir::Value root = seqb->addArgument(mlir::transform::AnyOpType::get(&context),
    //                                builder.getUnknownLoc());
    mlir::Value root = seq.getArgument(0);

    mlir::ArrayRef<mlir::StringRef> initListRef = {"linalg.matmul"};
   
    auto mats = builder.create<mlir::transform::MatchOp>(builder.getUnknownLoc(),
                root,
                initListRef);

    SmallVector<int64_t> numThreads = {8, 16};
    mlir::Value target = mats.getResult();
    llvm::SmallVector<int64_t> staticTileSizes = {64, 64, 32};
    // 3. distribution mapping 
    /*
    mlir::ArrayAttr mapping = builder.getStrArrayAttr({
        builder.getStringAttr("cyclic"),
        builder.getStringAttr("cyclic"),
        builder.getStringAttr("cyclic")
      });
    */
    mlir::ArrayAttr mapping = mlir::ArrayAttr();
      //builder.getArrayAttr({"cyclic", "cyclic", "cyclic"});
     ///*
    auto outer = builder.create<mlir::transform::TileUsingForallOp>(
      builder.getUnknownLoc(),
      /*target=*/target,
      /*staticNumThreads=*/numThreads,        // 走“num_threads”那条重载
      /*spec      =*/mlir::transform::NumThreadsSpec(),    // 绝大多数场景直接 None
      /*mapping   =*/mapping);            // 可 later attach gpu.block 等

    Value tiledOpH   = outer.getResult(0);      // = %gridTile
    // Value forallH = outer.getResult(1);      // = %forallGrid（如果需要的话）

    builder.create<mlir::transform::TileUsingForallOp>(builder.getUnknownLoc(),
                                      tiledOpH/*target*/,
                                      staticTileSizes, 
                                      mlir::transform::TileSizesSpec(), 
                                      mapping);
    builder.create<mlir::transform::YieldOp>(builder.getUnknownLoc());
    module->setAttr("transform.with_named_sequence",
                    mlir::SymbolRefAttr::get(&context, "do_tiling"));

    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
    //run pass
    mlir::PassManager pm(&context);
    // tiling pass
    mlir::transform::InterpreterPassOptions opts;
    opts.entryPoint.assign("do_tiling");  
    pm.addPass(mlir::transform::createInterpreterPass(opts));
    //after tiling code gen
    pm.addPass(std::make_unique<TilingCodePass>());
    (void)pm.run(module);
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
    //*/
}
/*
MLIRContext ctx;
ctx.getOrLoadDialect<linalg::LinalgDialect>();
ctx.getOrLoadDialect<tensor::TensorDialect>();
ctx.getOrLoadDialect<func::FuncDialect>();
ctx.getOrLoadDialect<transform::TransformDialect>();
ctx.getOrLoadDialect<scf::SCFDialect>();

OpBuilder b(&ctx);
auto mod = ModuleOp::create(b.getUnknownLoc());

// ---  payload func  -------------------------------------------------
auto i8  = b.getIntegerType(8);
auto tA  = RankedTensorType::get({256,128}, i8);
auto tB  = RankedTensorType::get({128, 64}, i8);
auto tC  = RankedTensorType::get({256, 64}, i8);
auto fTy = b.getFunctionType({tA,tB,tC}, {});
auto fn  = b.create<func::FuncOp>(b.getUnknownLoc(), "main", fTy);

Block *body = fn.addEntryBlock();
b.setInsertionPointToStart(body);
b.create<linalg::MatmulOp>(b.getUnknownLoc(),  ValueRange{body->getArgument(2)},
                            ValueRange{body->getArgument(0), body->getArgument(1)});
b.create<func::ReturnOp>(b.getUnknownLoc());
mod.push_back(fn);

// ---  transform sequence  -------------------------------------------
b.setInsertionPointToEnd(mod.getBody());
auto seq = b.create<transform::NamedSequenceOp>(b.getUnknownLoc(),
              b.getStringAttr("do_tiling"),
               mlir::SymbolTable::Visibility::Public);

auto *seqBlock = seq.addEntryBlock();
b.setInsertionPointToStart(seqBlock);

Value root = seqBlock->addArgument(
               transform::AnyOpType::get(&ctx), b.getUnknownLoc());
auto mats  = b.create<transform::MatchOp>(b.getUnknownLoc(),
               root, b.getStringAttr("linalg.matmul"));
b.create<transform::TileOp>(b.getUnknownLoc(), mats,
            b.getI64ArrayAttr({64, 32, 32}),
             ArrayAttr(),
             transform::TileOp::DistributionMethod::Forall);
b.create<transform::YieldOp>(b.getUnknownLoc());
mod->setAttr("transform.with_named_sequence",
             SymbolRefAttr::get(&ctx, "do_tiling"));

// ---  run interpreter  ----------------------------------------------
PassManager pm(&ctx);
pm.addPass(transform::createTransformInterpreterPass());
(void)pm.run(mod);
mod.dump();   // shows tiled IR
*/