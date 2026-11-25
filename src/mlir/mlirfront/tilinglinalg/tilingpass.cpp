/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "tilingpass.h"

static std::string ivName(Value v) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return llvm::formatv("iv{0}", arg.getArgNumber()).str();
  return "v";
}

struct TileKey {
  Value source;    // 
  int64_t rowOff;  // m * tileHeight
  int64_t colOff;  // n * tileWidth
  bool operator==(const TileKey &o) const {
    return source == o.source && rowOff == o.rowOff && colOff == o.colOff;
  }
};

struct AffineInfo {
  Value iv;      // induction var
  int64_t mul;   // const multiplier
};

int getConstantInt(OpFoldResult bound) {
  int ret = -1;
  if (auto val = mlir::getConstantIntValue(bound)) {
    ret =   static_cast<int>(*val);;
  } else {
    llvm::outs() << "Not a constant integer.\n";
  }
  return ret;
}

void DataPatternAnalysis(scf::ForallOp& forall, OpBuilder& builder,
                         std::vector<std::vector<std::pair<int,int>>>& offsetpair,
                         int forallIdx) 
{
  SmallVector<Value> ivs = forall.getInductionVars();  // arg3, arg4, arg5
  auto lbs   = forall.getLowerBound(builder);  // SmallVector<Value>
  auto ubs   = forall.getUpperBound(builder);
  auto steps = forall.getStep(builder);
  auto indvs   = forall.getInductionVars();
  llvm::outs() << "-------DataPatternAnalysis--------" << "\n";
  if (forallIdx == 0) return;
  scf::ForallOp outer = forall->getParentOfType<scf::ForallOp>();
  SmallVector<Value> ivsOuter;
  SmallVector<mlir::Value> lbsOuter;
  SmallVector<mlir::Value> ubsOuter ;
  SmallVector<mlir::Value> stepsOuter;

  if (outer) {
    ivsOuter = outer.getInductionVars();
    lbsOuter   = outer.getLowerBound(builder);  // SmallVector<Value>
    ubsOuter   = outer.getUpperBound(builder);
    stepsOuter = outer.getStep(builder);
    // use ivsOuter
  } else {
    llvm::outs() << "No outer forall exists — this is top-level forall.\n";
  }

  forall.walk([&](memref::SubViewOp sv) {
    llvm::outs() << "-------memref::SubViewOp--------" << "\n";
    DenseSet<unsigned> usedDims,usedDimsOuter;
    auto parentsubview = sv.getSource();
    for (auto off : llvm::enumerate(sv.getMixedOffsets())) {
      llvm::outs() << "--****************------" << "\n";
      if (Value v = off.value().dyn_cast<Value>()) {
        if (auto apply = v.getDefiningOp<affine::AffineApplyOp>()) {
          Value v = apply.getOperand(0);
          if (BlockArgument bar = llvm::dyn_cast<BlockArgument>(v)) {
              unsigned idx = bar.getArgNumber();
              usedDims.insert(idx);
          }
        }
      }
    }

    memref::SubViewOp parentsvop;
    if (parentsvop = parentsubview.getDefiningOp<memref::SubViewOp>()) {
      for (auto off : llvm::enumerate(parentsvop.getMixedOffsets())) {
        llvm::outs() << "--*********--22222--*******------" << "\n";
        if (Value v = off.value().dyn_cast<Value>()) {
          if (auto apply = v.getDefiningOp<affine::AffineApplyOp>()) {
            Value v = apply.getOperand(0);
            if (BlockArgument bar = llvm::dyn_cast<BlockArgument>(v)) {
                unsigned idx = bar.getArgNumber();
                usedDimsOuter.insert(idx);
            }
          }
        }
      }
    }
    // decide whether each IV is absent
    std::vector<int> missingpos; 
    for (auto en : llvm::enumerate(ivs)) {
      unsigned pos = en.index();
      //llvm::outs() << "IV[" << en.index() << "] = ";
      en.value().print(llvm::outs());
      llvm::outs() << "\n";
      if (!usedDims.contains(pos)) {
            missingpos.push_back(pos);
        }
    }
    {
        sv->print(llvm::errs());
        llvm::errs() << "\n";
    }
    if (missingpos.size()) {
      for (auto x : missingpos) {
        std::ostringstream ostr;
        ostr << " should broadcast in DIM " << x << "\n";
        std::cout << ostr.str();  
        auto lbC = getConstantInt(lbs[x]);
        auto ubC = getConstantInt(ubs[x]);
        auto stC = getConstantInt(steps[x]);
        int64_t trip = (ubC - lbC + stC - 1) / stC; 
        //markBroadcast(sv, /*alongIV=*/en.value());
        llvm::outs() << "there is one need broadcast into " << trip << " destination " << " in Parameter DIM " << x << "\n" ;
      }
    } else {
      llvm::outs()<< "no need broadcast send to 1 tile\n";
    }

    std::vector<int> missingposouter; 
    for (auto en : llvm::enumerate(ivsOuter)) {
      unsigned pos = en.index();
      //llvm::outs() << "IV[" << en.index() << "] = ";
      en.value().print(llvm::outs());
      llvm::outs() << "\n";
      if (!usedDimsOuter.contains(pos)) {
            missingposouter.push_back(pos);
        }
    }
    {
        parentsvop->print(llvm::errs());
        llvm::errs() << "\n";
    }
    if (missingposouter.size()) {
      for (auto x : missingposouter) {
        std::ostringstream ostr;
        ostr << " 2  ---- outer should broadcast in DIM " << x << "\n";
        std::cout << ostr.str();  
        auto lbC = getConstantInt(lbsOuter[x]);
        auto ubC = getConstantInt(ubsOuter[x]);
        auto stC = getConstantInt(stepsOuter[x]);
        int64_t trip = (ubC - lbC + stC - 1) / stC; 
        //markBroadcast(sv, /*alongIV=*/en.value());
        llvm::outs() << "there is one need broadcast into " << trip << " destination " << " in Parameter DIM " << x << "\n" ;
      }
    } else {
      llvm::outs()<< "no need broadcast send to 1 tile\n";
    }
    
  });

  return;
}

void TilingCodePass::runOnOperation() {
    mlir::Operation* cop = getOperation();
    mlir::OpBuilder builder(&getContext());  
    llvm::outs() << "TilingCodePass::runOnOperation" << "\n";
    DenseMap<Value, AffineInfo> affineMap;

    SmallVector<int64_t> ublb;
    std::vector<std::vector<std::pair<int,int>>> offsetPair;

    //chec
    int forallIdx = 0;
    auto process = [&](auto &self, Block &block) -> void {
      ///*
      for (Operation &op : block) {
        ///*
        if (auto forallop = dyn_cast<scf::ForallOp>(op)) {
          //forallop analysis
          DataPatternAnalysis(forallop, builder,offsetPair,forallIdx++);
          //SmallVector<int64_t> ubs;
          //for(auto ub : forallop.getUpperBound(builder))
          //  ubs.push_back(cast<IntegerAttr>(ub).getInt());
          auto upper = forallop.getUpperBound(builder);
          int usize = upper.size();
          //if (usize > 0) {
            for(size_t i = 0; i< upper.size(); ++i) {
              auto var = ivName(forallop.getInductionVars()[i]);
              int intVal = 0;
              ///*
              // auto val = upper[i].get<mlir::Value>();
              if (auto val = llvm::dyn_cast<mlir::Value>(upper[i])) {
                  if (auto cstOp = val.getDefiningOp<mlir::arith::ConstantOp>()) {
                      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
                          intVal = intAttr.getInt();
                          // Use 'intVal' here
                      }
                  }
              }
              //*/
              ublb.push_back(intVal);
              llvm::outs() << "for (int " << var << "=0;" << var << " < " << intVal << ";" << var << "++) {\n";
            }
            ///*
            for(auto &b :forallop.getRegion()) self(self,b);
            for(auto x: upper) llvm::outs() << "}\n";
        //}
          //*/
        }//*/
        ///*
        if (auto apply = dyn_cast<mlir::affine::AffineApplyOp>(op)) {
          auto map = apply.getAffineMap();
          if (map.getNumDims() == 1 && llvm::isa<mlir::AffineBinaryOpExpr>(map.getResult(0))) {
              auto mul = llvm::cast<mlir::AffineBinaryOpExpr>(map.getResult(0));
              if (auto cst = llvm::dyn_cast<mlir::AffineConstantExpr>(mul.getRHS()))
                  affineMap[apply.getResult()] = {apply.getOperand(0), cst.getValue()};
          }
        }
      //*/
      ///*
        if (auto sub = dyn_cast<mlir::memref::SubViewOp>(op)) {
          auto off0 = sub.getMixedOffsets()[0];
          auto off1 = sub.getMixedOffsets()[1];
          auto size0= 0;
          if (auto szOpt = mlir::getConstantIntValue(sub.getMixedSizes()[0])) {
            size0 = *szOpt;
            llvm::outs() << "static constant size = " << size0 << "\n";
          } else {
              llvm::outs() << "dynamic constant size = " << size0 << "\n";
          }
          //sub.getMixedSizes()[0].get<mlir::Attribute>().cast<IntegerAttr>().getInt();
          auto size1=0;
          if (auto szOpt = mlir::getConstantIntValue(sub.getMixedSizes()[1])) {
            size1 = *szOpt;
            llvm::outs() << "static = " << size1 << "\n";
          }else {
            llvm::outs() << "Dynamic constant size = " << size0 << "\n";
          }
          //sub.getMixedSizes()[1].get<mlir::Attribute>().cast<IntegerAttr>().getInt();
          auto offExpr=[&](OpFoldResult o) {
          if(auto v=o.dyn_cast<Value>()){
            auto it=affineMap.find(v);
            if(it!=affineMap.end())
              //return std::make_pair(it->second.iv,it->second.mul);
              return llvm::formatv("{0}*{1}",ivName(it->second.iv),it->second.mul).str();
              //return ivName(v);
            }
            return std::to_string(llvm::cast<mlir::IntegerAttr>(llvm::dyn_cast<mlir::Attribute>(o)).getInt());
          };
          //llvm::outs() << "// tile view\nint off0="<<offExpr(off0)<<"; int off1="<<offExpr(off1)<<";\n";
          std::ostringstream ostr;
          //std::string code = "adf::tiling({.buffer_dimension={IFM_K_SIZE, IFM_K_SIZE, 3*8}, 
          //                                        .tiling_dimension={IFM_K_SIZE, IFM_K_SIZE, 3*8},  
          //                                        .offset={0,0,0}, 
          //                                        .tile_traversal = {{.dimension=0, .stride=IFM_K_SIZE, .wrap=1}, 
          //                                                          {.dimension=1, .stride=IFM_K_SIZE, .wrap=1}}});"
          //llvm::outs()<<"for(int i=0;i<"<<size0<<";++i)for(int j=0;j<"<<size1<<";++j){}\n";
          //auto iv0 = offExpr(off0);
          //auto iv1 = offExpr(off1);
          //auto v0 = llvm::formatv("{0}*{1}",ivName(iv0.first),iv0.second).str();
          //auto v1 = llvm::formatv("{0}*{1}",ivName(iv0.first),iv0.second).str();
          // ostr << "adf::tiling({.offset={" << v0 << "," << v1 << "},\n";
          ostr << "adf::tiling({.offset={" << offExpr(off0) << "," << offExpr(off1) << "},\n";
          ostr << "            {.tiling_dimention={" << size0 << "," << size1 << "},\n";
          ostr << "            {.tile_traversal={}})";
          llvm::outs() << ostr.str() << "\n";
        }
        //for (Region &r: op.getRegions()) for(Block &b:r) self(self,b);
      }
       // */
      // 
      //*/
    };
    //for(auto func:getOperation().getOps<func::FuncOp>())
    //  process(process, func.getBody().front());
    ModuleOp mod = dyn_cast<ModuleOp>(cop);
    if (!mod) return;
    for (func::FuncOp func : mod.getOps<func::FuncOp>()) { 
      process(process, func.getBody().front());
    };

    for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
      if (func.getName()=="main") {
        auto &ctx = getContext();
        OpBuilder builder(&ctx);
        auto& block = func.getBody().front();
        builder.setInsertionPointToStart(&block);
        routingmanager mtest;
        //mtest.loaddialect(&ctx);
        //mtest.type_interface_test(&ctx);
        //mtest.ops_test(&ctx);
        auto func = mtest.createroutingfunc(&ctx);
        mod.push_back(func);
        auto cnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
        auto rnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 8);
        auto total_col = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 16);
        builder.create<func::CallOp>(builder.getUnknownLoc(), func, ValueRange({cnum, rnum, total_col}));
      }
    }

    for(auto x:ublb) {
      llvm::outs()<< " upper bound is " << x << "\n";
    }
    //for (Region &r: cop->getRegions()) for(Block &b:r) process(process,b);
}
