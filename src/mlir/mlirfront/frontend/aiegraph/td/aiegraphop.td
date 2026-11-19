/******************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#ifndef aiegraph_OPS
#define aiegraph_OPS
include "aiegraphtype.td"
include "aiegraphattr.td"


#ifndef AIEHLC_PUBLIC_SYMBOL_TD
#define AIEHLC_PUBLIC_SYMBOL_TD
// every aiehlc symbol is public (tilinglinalg/aiehlcsymbol.h)
def AiehlcPublicSymbol : NativeOpTrait<"PublicSymbol"> { let cppNamespace = "::aiehlc"; }
#endif
/*
  A whole fused+quantized network is one aiegraph.func whose body is a straight
  line of per-op ops. Dataflow is SSA def-use over builtin tensor<...xi8> values
  (the informal Python `LayerOp.out`/`.ins` string wiring becomes verifiable
  edges). Weights/bias ride as a SymbolRefAttr into a module-level weights dict
  (deterministic param generators in model.py), keeping the IR lean.

    aiegraph.func @resnet {
      %in  = ... (block arg)
      %f1  = aiegraph.conv_bn_relu %in  {H,W,Cin,Cout,K,stride, quant, weights=@w0}
      %t1  = aiegraph.conv_bn_relu %f1  {...}
      %t2  = aiegraph.conv_bn      %t1  {...}
      %f2  = aiegraph.residual_add_relu %t2, %f1 {length}
      ...
      %lg  = aiegraph.avgpool_fc   %fN  {spatial_h,spatial_w,channels,num_classes, weights=@fc}
      aiegraph.yield
    }
*/

// int8 feature-map / param tensor used across the graph.
def AIEGraph_I8Tensor : TensorOf<[I8]>;

def aiegraph_FuncOp : aiegraphop<"func",[
    SymbolTable,
    Symbol, SymbolName, AiehlcPublicSymbol,
    SingleBlock,
    NoTerminator
  ]>
{
    let summary = "Container for one fused+quantized network graph.";
    let description = [{
      A symbol-named single-block region holding the per-op aiegraph ops in
      topological (dataflow) order. Mirrors dmap.func / routing container ops.
    }];

  let arguments = (ins
                    SymbolNameAttr:$sym_name,
                    TypeAttrOf<FunctionType>:$funcType
                  );
  let regions = (region SizedRegion<1>:$body);
  let builders = [
    OpBuilder<(ins
                   "Attribute":$Memo,
                   "TypeAttr":$type,
                   "function_ref<void(OpBuilder &, Location)>":$bodyBuilder),
    [{
      $_state.addAttribute("sym_name", Memo);
      $_state.addAttribute("funcType", type);
      auto *bodyRegion = $_state.addRegion();
      Block *bodyBlock = new Block();
      bodyRegion->push_back(bodyBlock);
      OpBuilder::InsertionGuard guard($_builder);
      $_builder.setInsertionPointToStart(bodyBlock);
      auto loc = mlir::UnknownLoc::get($_builder.getContext());
      bodyBuilder($_builder, loc);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

def aiegraph_YieldOp : aiegraphop<"yield", [Terminator]> {
  let summary = "Terminator for the aiegraph.func region.";
}

//===----------------------------------------------------------------------===//
// Fused+quantized tensor ops (one per LayerOp kind)
//===----------------------------------------------------------------------===//

def aiegraph_ConvBnReluOp : aiegraphop<"conv_bn_relu"> {
  let summary = "Fused Conv2D + Q7 BatchNorm + ReLU on int8 feature maps.";
  let description = [{
    Consumes an int8 feature map, produces an int8 feature map. Geometry
    (H,W,Cin,Cout,K,stride) and int8 quant params ride as attributes; the
    convolution weights + BN scale/bias live in a module-level weights table
    referenced by `weights`.
  }];
  let arguments = (ins
    AIEGraph_I8Tensor:$in,
    I64Attr:$H, I64Attr:$W, I64Attr:$Cin, I64Attr:$Cout,
    I64Attr:$K, I64Attr:$stride,
    quantparam:$quant,
    OptionalAttr<FlatSymbolRefAttr>:$weights
  );
  let results = (outs AIEGraph_I8Tensor:$out);
  let assemblyFormat = [{
    $in `:` type($in) attr-dict `->` type($out)
  }];
  let hasVerifier = 1;
}

def aiegraph_ConvBnOp : aiegraphop<"conv_bn"> {
  let summary = "Fused Conv2D + Q7 BatchNorm (no ReLU) on int8 feature maps.";
  let arguments = (ins
    AIEGraph_I8Tensor:$in,
    I64Attr:$H, I64Attr:$W, I64Attr:$Cin, I64Attr:$Cout,
    I64Attr:$K, I64Attr:$stride,
    quantparam:$quant,
    OptionalAttr<FlatSymbolRefAttr>:$weights
  );
  let results = (outs AIEGraph_I8Tensor:$out);
  let assemblyFormat = [{
    $in `:` type($in) attr-dict `->` type($out)
  }];
  let hasVerifier = 1;
}

def aiegraph_ResidualAddReluOp : aiegraphop<"residual_add_relu"> {
  let summary = "Element-wise int8 residual add + ReLU + saturate.";
  let description = [{
    Joins the main path (`main`) and skip path (`skip`); both carry `length`
    int8 elements. Mirrors the residual join recovered from `add + nn.relu`.
  }];
  let arguments = (ins
    AIEGraph_I8Tensor:$main,
    AIEGraph_I8Tensor:$skip,
    I64Attr:$length
  );
  let results = (outs AIEGraph_I8Tensor:$out);
  let assemblyFormat = [{
    $main `,` $skip `:` type($main) `,` type($skip) attr-dict `->` type($out)
  }];
  let hasVerifier = 1;
}

def aiegraph_AvgPoolFcOp : aiegraphop<"avgpool_fc"> {
  let summary = "Global average pool + fully-connected classifier (int8).";
  let description = [{
    Consumes a `channels`-deep `spatial_h`x`spatial_w` int8 feature map and
    produces `num_classes` int8 logits. FC weights/bias live in the module-level
    weights table referenced by `weights`.
  }];
  let arguments = (ins
    AIEGraph_I8Tensor:$in,
    I64Attr:$spatial_h, I64Attr:$spatial_w,
    I64Attr:$channels, I64Attr:$num_classes,
    OptionalAttr<FlatSymbolRefAttr>:$weights
  );
  let results = (outs AIEGraph_I8Tensor:$logits);
  let assemblyFormat = [{
    $in `:` type($in) attr-dict `->` type($logits)
  }];
  let hasVerifier = 1;
}

#endif // aiegraph_OPS
