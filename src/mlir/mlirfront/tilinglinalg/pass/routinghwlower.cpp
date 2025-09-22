/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "routinghwlower.h"
struct EnableExtToAieShimPortpattern: public ConversionPattern {
    explicit EnableExtToAieShimPortpattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routinghw::EnableExtToAieShimPort::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        auto shimtileoprand = operands[0];
        auto shimtileop = shimtileoprand.getDefiningOp();
        //get the enable ext to aie port attribute
        int32_t portdirection=-1, portidx = -1;
        if (auto pd = op->getAttrOfType<IntegerAttr>("portdirection")) {
            portdirection = pd.getInt();
        }
        if (auto pi = op->getAttrOfType<IntegerAttr>("portidx")) {
            portidx = pi.getInt();
        }
        // get the tilecreate parameter
        // Access attributes by name
        int32_t rowValue=-1, colValue=-1;
        if (auto colAttr = shimtileop->getAttrOfType<IntegerAttr>("col")) {
            colValue = colAttr.getInt();
        } 
        if (auto rowAttr = shimtileop->getAttrOfType<IntegerAttr>("row")) {
            rowValue = rowAttr.getInt();
        }
        auto colConstOp = rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(colValue));
        auto rowConstOp = rewriter.create<emitc::ConstantOp>(op->getLoc(),rewriter.getI32Type(), rewriter.getI32IntegerAttr(rowValue));

        auto tileLocType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_LocType");

        auto tileLocOp = rewriter.create<emitc::CallOp>(
            op->getLoc(), "XAie_TileLoc", TypeRange{tileLocType}, 
            ValueRange{rowConstOp, colConstOp});

        //get the device instance
        auto devInstType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_DevInst");
        auto devInstPtrType = emitc::PointerType::get(devInstType);
        auto deviceInstOp = rewriter.create<emitc::CallOp>(
                op->getLoc(), "getOrCreateDeviceInstance", TypeRange{devInstPtrType}, ValueRange{});
        Value deviceInst = deviceInstOp.getResult(0);

        StringRef callee = "XAie_EnableShimDmaToAieStrmPort";
        Value arg0 = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(portidx));
        auto callOp = rewriter.create<mlir::emitc::CallOp>(op->getLoc(), TypeRange{rewriter.getI32Type()}, callee, 
            ValueRange{deviceInst, tileLocOp.getResult(0), arg0});

        rewriter.eraseOp(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

struct ConnectStreamSingleSwitchPortpattern: public ConversionPattern {
    explicit ConnectStreamSingleSwitchPortpattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routinghw::ConnectStreamSingleSwitchPort::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        auto tileoprand = operands[0];
        auto tileop = tileoprand.getDefiningOp();
        
        int32_t rowValue=-1, colValue=-1;
        if (auto colAttr = tileop->getAttrOfType<IntegerAttr>("col")) {
            colValue = colAttr.getInt();
        } 
        if (auto rowAttr = tileop->getAttrOfType<IntegerAttr>("row")) {
            rowValue = rowAttr.getInt();
        }

        int32_t masterportdirection=-1, masterportidx = -1,slaveportdirection=-1, slaveportidx = -1;
        std::string masterportdirectionstr="fixme",slaveportdirectionstr="fixme";
        if (auto pd = op->getAttrOfType<StringAttr>("masterportdirection")) {
            masterportdirectionstr = pd.getValue().str();
        }
        if (auto pi = op->getAttrOfType<IntegerAttr>("masterportidx")) {
            masterportidx = pi.getInt();
        }

        if (auto pd = op->getAttrOfType<StringAttr>("slaveportdirection")) {
            slaveportdirectionstr = pd.getValue().str();
        }
        if (auto pi = op->getAttrOfType<IntegerAttr>("slaveportidx")) {
            slaveportidx = pi.getInt();
        }

        //StringRef callee = "XAie_StrmConnCctEnable";
        //Value arg0 = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(42));
        ///auto callOp = rewriter.create<mlir::emitc::CallOp>(op->getLoc(), TypeRange{rewriter.getI32Type()}, callee, ValueRange{arg0});

        auto colConstOp = rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(colValue));
        auto rowConstOp = rewriter.create<emitc::ConstantOp>(op->getLoc(),rewriter.getI32Type(), rewriter.getI32IntegerAttr(rowValue));

        auto tileLocType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_LocType");

        auto tileLocOp = rewriter.create<emitc::CallOp>(
            op->getLoc(), "XAie_TileLoc", TypeRange{tileLocType}, 
            ValueRange{rowConstOp, colConstOp});

        auto devInstType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_DevInst");
        auto devInstPtrType = emitc::PointerType::get(devInstType);
        auto deviceInstOp = rewriter.create<emitc::CallOp>(
                op->getLoc(), "getOrCreateDeviceInstance", TypeRange{devInstPtrType}, ValueRange{});
        Value deviceInst = deviceInstOp.getResult(0);

        StringRef callee = "XAie_StrmConnCctEnable";
         //string type
        mlir::Type stringType = mlir::emitc::PointerType::get(rewriter.getI8Type());

        Value masterport = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), stringType,
                                        mlir::emitc::OpaqueAttr::get(rewriter.getContext(), masterportdirectionstr));
        Value masteridx = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(masterportidx));
        Value slaveport = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), stringType,
                                        mlir::emitc::OpaqueAttr::get(rewriter.getContext(), slaveportdirectionstr));
        Value slaveidx = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), rewriter.getI32Type(),rewriter.getI32IntegerAttr(slaveportidx));
        auto callOp = rewriter.create<mlir::emitc::CallOp>(op->getLoc(), TypeRange{rewriter.getI32Type()}, callee, 
            ValueRange{deviceInst, tileLocOp.getResult(0), masterport,masteridx,slaveport,slaveidx});

        rewriter.eraseOp(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

struct routingRoutingCreatePattern: public ConversionPattern {
    explicit routingRoutingCreatePattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routing::RoutingCreate::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
         // Preconditions for the simple lowering.
         auto getRoutingCreateOprandConst = [&] (Value operand) -> int {
            IntegerAttr intAttr;
            if (matchPattern(operand, m_Constant(&intAttr))) {
                auto concrete = intAttr.getInt();        // -> 0
                return concrete;
            }
            return 0;
        };
        auto loc = op->getLoc();
        
        // 1. Cast the generic 'Operation*' to your specific op type.
        auto routingOp = cast<routing::RoutingCreate>(op);
        auto memo = routingOp.getMemo();
        // 2. Get the single block from the op's region.
        mlir::Region &region = routingOp.getRegion();
        if (!llvm::hasSingleElement(region)) {
            return rewriter.notifyMatchFailure(op, "expected a single block in the region");
        }
        mlir::Block &bodyBlock = region.front();

        // 3. Find the terminator (yield) to get the results from the body.
        auto yieldOp = dyn_cast<routing::YieldOp>(bodyBlock.getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(op, "region must end with a routing::YieldOp");
        }
        // These are the values that will replace the results of the routingOp.
        ValueRange yieldedValues = yieldOp.getODSOperands(0);
        //rewriter.setInsertionPoint(op);
        auto idx = getRoutingCreateOprandConst(operands[0]);
        std::ostringstream ostr;
        if (1) {// use if block
            // First, erase the default terminator in the 'then' block.
            auto trueAttr = rewriter.getBoolAttr(true);
            mlir::Value trueVal = rewriter.create<mlir::arith::ConstantOp>(loc, trueAttr);
            auto ifOp = rewriter.create<mlir::emitc::IfOp>(loc, trueVal, /*withElseRegion=*/false);
            rewriter.setInsertionPoint(ifOp);
            ostr << "\n//round is " << idx << " hw split in : " << memo.str() <<" -----------";
            auto open = rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(ostr.str()));
            auto& tblock = ifOp.getThenRegion().front();
            //remove block parameter
            Value idx = operands[0];
            ///auto& body = op->getRegion(0).front();
            //BlockArgument barg = body.getArgument(0);
            //rewriter.replaceAllUsesWith(barg, idx);
            /*
            llvm::errs() << "--- DEBUG: Contents of the 'then' block: ---\n";
            llvm::errs() << tblock << "\n";
            llvm::errs() << "--- END DEBUG ---\n";
            */
            //rewriter.eraseOp(tblock.getTerminator());
            // Then, merge the scopeOp's block into the now-empty 'then' block.
            rewriter.mergeBlocks(&bodyBlock, &tblock, operands);
            //rewriter.setInsertionPoint(yieldOp);
            //rewriter.create<mlir::emitc::YieldOp>(loc);
            rewriter.eraseOp(op);  
        } else { // use {} block
            ostr << "\n{ //round is " << idx << " -----------";
            auto open = rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(ostr.str()));
            // 4. Inline the body, remapping the region arguments to the NEW operands.
            // This is the key step: we use the 'operands' array passed into this function.
            rewriter.inlineBlockBefore(&bodyBlock, op, operands);
            rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr("}\n"));
            // 5. Replace the original op with the values from the yield.
            // The conversion framework requires you to either erase or replace the original op.
            rewriter.replaceOp(op, yieldedValues);
            // 6. Clean up the now-obsolete yield op.
            rewriter.eraseOp(yieldOp);
  
        }
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

//scf::ExecuteRegionOp
struct ScfExecuteRegionOpPattern: public ConversionPattern {
    explicit ScfExecuteRegionOpPattern(MLIRContext* ctx, LLVMTypeConverter &converter) :
        ConversionPattern(scf::ExecuteRegionOp::getOperationName(), 1, ctx), typeconverter(converter) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        auto loc = op->getLoc();
    // 1. Cast the generic 'Operation*' to your specific op type.
        auto seop = cast<scf::ExecuteRegionOp>(op);
        std::string memo;
        if (auto memoAttr = seop->getAttrOfType<mlir::StringAttr>("routing_memo")) {
            memo= memoAttr.getValue().str();
            llvm::errs() << "Successfully found memo: " << memo << "\n";
        }
        // 2. Get the single block from the op's region.
        mlir::Region &region = seop.getRegion();
        if (!llvm::hasSingleElement(region)) {
            return rewriter.notifyMatchFailure(op, "expected a single block in the region");
        }
        mlir::Block &bodyBlock = region.front();
        // 3. Find the terminator (yield) to get the results from the body.
        auto yieldOp = dyn_cast<scf::YieldOp>(bodyBlock.getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(op, "region must end with a routing::YieldOp");
        }
        // These are the values that will replace the results of the routingOp.
        ValueRange yieldedValues = yieldOp.getODSOperands(0);
        //rewriter.setInsertionPoint(op);
        std::ostringstream ostr, ostrend;
        ostr << "\n{ //----routing creation in " << memo << " ----start-------";
        ostrend << "\n} //----routing creation in " << memo << " ----end-------\n";

        auto open = rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(ostr.str()));
        // 4. Inline the body, remapping the region arguments to the NEW operands.
        // This is the key step: we use the 'operands' array passed into this function.
        rewriter.inlineBlockBefore(&bodyBlock, op, operands);
        rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(ostrend.str()));
        // 5. Replace the original op with the values from the yield.
        // The conversion framework requires you to either erase or replace the original op.
        rewriter.replaceOp(op, yieldedValues);
        // 6. Clean up the now-obsolete yield op.
        rewriter.eraseOp(yieldOp);
  
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
};

struct IOShimTileCreatepattern: public ConversionPattern {
    explicit IOShimTileCreatepattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routinghw::IOShimTileCreate::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

struct RoutingYieldOp: public ConversionPattern {
    explicit RoutingYieldOp(MLIRContext* ctx, LLVMTypeConverter &converter) :
        ConversionPattern(routing::YieldOp::getOperationName(), 1, ctx), typeconverter(converter) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        //rewriter.eraseOp(op);
        auto constantOp = cast<routing::YieldOp>(op);
        rewriter.replaceOpWithNewOp<mlir::emitc::YieldOp>(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
};

struct TileArrayHandleCreatepattern: public ConversionPattern {
    explicit TileArrayHandleCreatepattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routinghw::TileArrayHandleCreate::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

struct TileCreatepattern: public ConversionPattern {
    explicit TileCreatepattern(MLIRContext* ctx, LLVMTypeConverter &converter, RoutingTopology & router) :
        ConversionPattern(routinghw::TileCreate::getOperationName(), 1, ctx), typeconverter(converter), router_(router) {

    }
    LogicalResult matchAndRewrite(Operation *op , ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override{
        rewriter.eraseOp(op);
        return success();
    }

private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

struct arithconstantconvert : public ConversionPattern {
    explicit arithconstantconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(arith::ConstantOp::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        //rewriter.eraseOp(op);
        //return success();
        auto constantOp = cast<mlir::arith::ConstantOp>(op);
        rewriter.replaceOpWithNewOp<mlir::emitc::ConstantOp>(op, constantOp.getType(), constantOp.getValue());
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

void declareAieTileFunction(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);

  builder.setInsertionPointToStart(module.getBody());
  builder.create<mlir::emitc::IncludeOp>(module.getLoc(), "xaiengine.h", true);
  // 1. Define the custom `xaie_loc` type using emitc.opaque.
  mlir::Type xaieLocType = mlir::emitc::OpaqueType::get(context, "XAie_LocType");
  auto devInstType = emitc::OpaqueType::get(context, "XAie_DevInst");
  auto devInstPtrType = emitc::PointerType::get(devInstType);
  mlir::Type i32Type = builder.getI32Type();
  //string type
  mlir::Type stringType = mlir::emitc::PointerType::get(builder.getI8Type());

  llvm::ArrayRef<mlir::Type> argTypes = {i32Type, i32Type};

  //auto funcType = mlir::FunctionType::get(context, argTypes, {xaieLocType});
  mlir::FunctionType funcType = builder.getFunctionType({i32Type, i32Type}, {xaieLocType});
  mlir::FunctionType getdevInstType = builder.getFunctionType({}, {devInstPtrType});
  mlir::FunctionType shimportenableType = builder.getFunctionType({devInstPtrType, xaieLocType, i32Type}, {i32Type});
  mlir::FunctionType tileconnectType = builder.getFunctionType({devInstPtrType, xaieLocType, stringType,i32Type,stringType,i32Type}, {i32Type});

  auto decl1 = builder.create<emitc::FuncOp>(module.getLoc(), "XAie_TileLoc", funcType);
  decl1.setVisibility(SymbolTable::Visibility::Private);

  auto decl2 = builder.create<emitc::FuncOp>(module.getLoc(), "getOrCreateDeviceInstance", getdevInstType);
  decl2.setVisibility(SymbolTable::Visibility::Private);

  auto decl3 = builder.create<emitc::FuncOp>(module.getLoc(), "XAie_EnableShimDmaToAieStrmPort", shimportenableType);
  decl3.setVisibility(SymbolTable::Visibility::Private);

  auto decl4 = builder.create<emitc::FuncOp>(module.getLoc(), "XAie_StrmConnCctEnable", tileconnectType);
  decl4.setVisibility(SymbolTable::Visibility::Private);
}

void RoutingHWLowerPass::runOnOperation() {
    auto module = getOperation();
    auto& ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    LLVMTypeConverter typeConverter(&ctx);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    //target.addIllegalOp<arith::ConstantOp>();
    //target.addIllegalOp<routing::RoutingCreate>();
    target.addIllegalOp<routinghw::EnableExtToAieShimPort>();
    target.addIllegalOp<routinghw::ConnectStreamSingleSwitchPort>();
    target.addIllegalOp<routinghw::TileCreate>();
    target.addIllegalOp<routinghw::IOShimTileCreate>();
    target.addIllegalOp<routinghw::TileArrayHandleCreate>();
    target.addLegalDialect<mlir::emitc::EmitCDialect>();

    //add header
    // The headers to add
    
    if (auto mop = dyn_cast<ModuleOp>(*module)) {
        declareAieTileFunction(mop);
    }
 ///*
    //target.addIllegalDialect<routinghw::RoutingHWDialect>();
    llvm::outs() << "RoutingHWLowerPass::runOnOperation\n";
    patterns.add<EnableExtToAieShimPortpattern>(&ctx,typeConverter,rtopology_);
    patterns.add<ConnectStreamSingleSwitchPortpattern>(&ctx,typeConverter,rtopology_);
    patterns.add<TileCreatepattern>(&ctx,typeConverter,rtopology_);
    patterns.add<IOShimTileCreatepattern>(&ctx,typeConverter,rtopology_);
    patterns.add<TileArrayHandleCreatepattern>(&ctx,typeConverter,rtopology_);
    patterns.add<routingRoutingCreatePattern>(&ctx,typeConverter,rtopology_);
    patterns.add<arithconstantconvert>(&ctx,typeConverter);
    patterns.add<RoutingYieldOp>(&ctx,typeConverter);
    patterns.add<ScfExecuteRegionOpPattern>(&ctx,typeConverter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        llvm::outs() << "applyPartialConversion failed\n";
    }

    std::string cppOutput;
    llvm::raw_string_ostream os(cppOutput);

    //remove the dead code
    
    mlir::RewritePatternSet patternscde(&ctx);
    for (auto* dialect : ctx.getLoadedDialects()) {
        dialect->getCanonicalizationPatterns(patternscde);
    }
   // /*
    for (mlir::RegisteredOperationName op : ctx.getRegisteredOperations()) {
        op.getCanonicalizationPatterns(patternscde, &ctx);
    }
    // Apply the patterns greedily. This will simplify and clean up the IR.
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patternscde)))) {
                // Handle error
        signalPassFailure();
    }
    

  // This is the core translation call.
  // It takes the module and an output stream.
    if (mlir::failed(mlir::emitc::translateToCpp(module, os))) {
        llvm::errs() << "Failed to translate MLIR to C++.\n";
        return;
    }
    //*/



  // 4. Print the resulting C++ code.
    std::cout << "--- Generated C++ Code ---\n" << os.str() << std::endl;
       // */
};