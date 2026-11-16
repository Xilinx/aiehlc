/******************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
// The aiednn dialect: a whole model as aiednn primitives (include/aiednn.h),
// one op per primitive, SSA tensors, inside a func.func.
//
//   func.func @mobilenet_v2(%x: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
//     %w0 = aiednn.const {index = 0 : i64} : tensor<32x3x3x3xsi8>
//     %xq = aiednn.reorder %x {scale = 0.0187 : f32, zero_point = 0 : i64}
//             : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xsi8> loc("conv1_q")
//     %y  = aiednn.conv %xq, %w0 bias %b0 {strides = array<i64: 2, 2>, ...,
//             output_scales = array<f32: ...>, post_ops = [#aiednn.post_op<eltwise, "clip", 0.0, 6.0>]}
//             : (...) -> tensor<1x32x112x112xf32> loc("conv1")
//     ...
//     func.return %out : tensor<1x1000xf32>
//   }
//
// Constants carry an index into an external store (the weights stay out of the
// IR); the op location names the source node. Execution order is block order.
#ifndef AIEDNN_OPS
#define AIEDNN_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Aiednn_Dialect : Dialect {
  let name = "aiednn";
  let cppNamespace = "::aiednn";
  let summary = "deep-learning primitives over the AIE kernel library (include/aiednn.h)";
  let useDefaultAttributePrinterParser = 1;
}

class Aiednn_Attr<string name, string mnemonic_> : AttrDef<Aiednn_Dialect, name> {
  let mnemonic = mnemonic_;
}

class Aiednn_Op<string mnemonic, list<Trait> traits = []> : Op<Aiednn_Dialect, mnemonic, traits>;

// f32 activations, s8/u8 quantized tensors, s32 accumulators.
def Aiednn_Tensor : RankedTensorOf<[F32, SI8, UI8, SI32]>;

// One entry of a fused epilogue (aiednn_attr_t.post_ops): kind is "eltwise" or
// "binary"; a binary post-op takes the next operand of the op's post_srcs.
def Aiednn_PostOpAttr : Aiednn_Attr<"PostOp", "post_op"> {
  let summary = "fused epilogue step";
  let parameters = (ins "::mlir::StringAttr":$kind, "::mlir::StringAttr":$alg,
                        "::mlir::FloatAttr":$alpha, "::mlir::FloatAttr":$beta);
  let assemblyFormat = "`<` $kind `,` $alg `,` $alpha `,` $beta `>`";
}

def Aiednn_PostOpArray : TypedArrayAttrBase<Aiednn_PostOpAttr, "fused epilogue">;

//===----------------------------------------------------------------------===//
// Values without a primitive
//===----------------------------------------------------------------------===//

def Aiednn_ConstOp : Aiednn_Op<"const", [Pure]> {
  let summary = "constant tensor stored outside the IR (weights, bias, statistics)";
  let arguments = (ins I64Attr:$index);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "prop-dict attr-dict `:` type($result)";
}

def Aiednn_ViewOp : Aiednn_Op<"view", [Pure]> {
  let summary = "same bytes, another shape (reshape / flatten / squeeze): no data movement";
  let arguments = (ins Aiednn_Tensor:$src);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$src prop-dict attr-dict `:` type($src) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;            // view to the same type; view of a view
}

//===----------------------------------------------------------------------===//
// GEMM family (fused epilogue: output scales -> bias -> post-ops -> dst requant)
//===----------------------------------------------------------------------===//

class Aiednn_GemmOp<string mnemonic, dag extra> : Aiednn_Op<mnemonic, [Pure, AttrSizedOperandSegments]> {
  let arguments = !con((ins Aiednn_Tensor:$src, Aiednn_Tensor:$weights,
                            Optional<Aiednn_Tensor>:$bias, Variadic<Aiednn_Tensor>:$post_srcs),
                       extra,
                       (ins OptionalAttr<DenseF32ArrayAttr>:$output_scales,
                            OptionalAttr<Aiednn_PostOpArray>:$post_ops,
                            OptionalAttr<F32Attr>:$dst_scale, OptionalAttr<I64Attr>:$dst_zero_point));
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = [{
    $src `,` $weights (`bias` $bias^)? (`post` `(` $post_srcs^ `)`)? prop-dict attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

defvar ConvGeom = (ins DenseI64ArrayAttr:$strides, DenseI64ArrayAttr:$pad_l, DenseI64ArrayAttr:$pad_r,
                       DenseI64ArrayAttr:$dilations, I64Attr:$groups);

def Aiednn_ConvOp : Aiednn_GemmOp<"conv", ConvGeom> {
  let summary = "convolution: src [N,C,H,W], weights [O,C/groups,KH,KW] -> [N,O,Ho,Wo]; dilations 0-based";
}

def Aiednn_DeconvOp : Aiednn_GemmOp<"deconv", ConvGeom> {
  let summary = "transposed convolution";
}

def Aiednn_MatmulOp : Aiednn_GemmOp<"matmul", (ins BoolAttr:$weights_transposed)> {
  let summary = "matmul: src [...,M,K] x weights [...,K,N] (or [...,N,K] transposed)";
}

def Aiednn_InnerProductOp : Aiednn_GemmOp<"inner_product", (ins)> {
  let summary = "inner product: src [N,...] x weights [O,...] -> [N,O]";
}

def Aiednn_SdpaOp : Aiednn_Op<"sdpa", [Pure]> {
  let summary = "scaled dot-product attention softmax(q k^T * scale [+ mask]) v";
  let arguments = (ins Aiednn_Tensor:$q, Aiednn_Tensor:$k, Aiednn_Tensor:$v, Optional<Aiednn_Tensor>:$mask,
                       F32Attr:$scale, BoolAttr:$causal,
                       // --aiednn-fold-head-permute: bit 0 q/k/v, bit 1 the result are [B,S,H,D] in memory
                       OptionalAttr<I64Attr>:$layout,
                       // --aiednn-fold-norm-quant: the quantizing reorder after it (s8 / u8 result)
                       OptionalAttr<F32Attr>:$dst_scale, OptionalAttr<I64Attr>:$dst_zero_point);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$q `,` $k `,` $v (`mask` $mask^)? prop-dict attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Elementwise / spatial / normalization / data movement
//===----------------------------------------------------------------------===//

class Aiednn_UnaryOp<string mnemonic, dag attrs, list<Trait> traits = []> : Aiednn_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = !con((ins Aiednn_Tensor:$src), attrs);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$src prop-dict attr-dict `:` functional-type(operands, results)";
}

def Aiednn_EltwiseOp : Aiednn_UnaryOp<"eltwise", (ins StrAttr:$alg, F32Attr:$alpha, F32Attr:$beta)> {
  let summary = "y = f(x) (alg without the eltwise_ prefix: relu, clip, logistic, gelu_erf, ...)";
}
def Aiednn_ReductionOp : Aiednn_UnaryOp<"reduction", (ins StrAttr:$alg, F32Attr:$p, F32Attr:$eps)> {
  let summary = "reduce over the dims where dst has extent 1";
}
def Aiednn_PoolingOp : Aiednn_UnaryOp<"pooling", (ins StrAttr:$alg, DenseI64ArrayAttr:$kernel,
                                                     DenseI64ArrayAttr:$strides, DenseI64ArrayAttr:$pad_l,
                                                     DenseI64ArrayAttr:$pad_r, DenseI64ArrayAttr:$dilations)> {
  let summary = "max / avg pooling";
}
def Aiednn_ResamplingOp : Aiednn_UnaryOp<"resampling", (ins StrAttr:$alg)> {
  let summary = "nearest / linear resampling to the dst spatial size";
}
def Aiednn_LrnOp : Aiednn_UnaryOp<"lrn", (ins StrAttr:$alg, I64Attr:$local_size, F32Attr:$alpha,
                                             F32Attr:$beta, F32Attr:$k)> {
  let summary = "local response normalization";
}
def Aiednn_SoftmaxOp : Aiednn_UnaryOp<"softmax", (ins StrAttr:$alg, I64Attr:$axis)> {
  let summary = "softmax / logsoftmax along axis";
}
def Aiednn_SliceOp : Aiednn_UnaryOp<"slice", (ins DenseI64ArrayAttr:$offsets, DenseI64ArrayAttr:$steps)> {
  let summary = "dst[i] = src[offsets + i * steps] per axis (aiednn_slice_create; dst dims are the sizes)";
}
def Aiednn_RopeOp : Aiednn_Op<"rope", [Pure]> {
  let summary = "rotary position embedding: x * cos + rotate(x) * sin, cos / sin broadcast to src (aiednn_rope_create)";
  let arguments = (ins Aiednn_Tensor:$src, Aiednn_Tensor:$cos, Aiednn_Tensor:$sin, I64Attr:$interleaved);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$src `,` $cos `,` $sin prop-dict attr-dict `:` functional-type(operands, results)";
}
def Aiednn_ShuffleOp : Aiednn_UnaryOp<"shuffle", (ins I64Attr:$axis, I64Attr:$group_size)> {
  let summary = "channel shuffle";
}
def Aiednn_ReorderOp : Aiednn_UnaryOp<"reorder", (ins OptionalAttr<DenseI32ArrayAttr>:$perm, F32Attr:$scale,
                                                     I64Attr:$zero_point)> {
  let summary = "permute and/or convert type (f32 -> s8/u8 quantize, s8/u8 -> f32 dequantize)";
  let hasVerifier = 1;
  let hasFolder = 1;            // identity: same type, no permutation
  let hasCanonicalizer = 1;     // quantize(dequantize(q)) with the same scale / zero point -> q
}

def Aiednn_BinaryOp : Aiednn_Op<"binary", [Pure]> {
  let summary = "y = a op b with numpy broadcast (alg without the binary_ prefix)";
  let arguments = (ins Aiednn_Tensor:$a, Aiednn_Tensor:$b, StrAttr:$alg);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$a `,` $b prop-dict attr-dict `:` functional-type(operands, results)";
}

def Aiednn_PreluOp : Aiednn_Op<"prelu", [Pure]> {
  let summary = "y = x > 0 ? x : w * x (w broadcast)";
  let arguments = (ins Aiednn_Tensor:$src, Aiednn_Tensor:$weights);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$src `,` $weights prop-dict attr-dict `:` functional-type(operands, results)";
}

// flags: 1 = scale, 2 = shift (aiednn.h), matching the optional operands.
def Aiednn_BatchNormOp : Aiednn_Op<"batch_norm", [Pure, AttrSizedOperandSegments]> {
  let summary = "batch normalization, inference";
  let arguments = (ins Aiednn_Tensor:$src, Aiednn_Tensor:$mean, Aiednn_Tensor:$variance,
                       Optional<Aiednn_Tensor>:$scale, Optional<Aiednn_Tensor>:$shift, F32Attr:$eps, I64Attr:$flags,
                       // --aiednn-fold-norm-quant: the quantizing reorder after it, folded in (s8 / u8 result)
                       OptionalAttr<F32Attr>:$dst_scale, OptionalAttr<I64Attr>:$dst_zero_point);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = [{
    $src `,` $mean `,` $variance (`scale` $scale^)? (`shift` $shift^)? prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

class Aiednn_NormOp<string mnemonic, dag extra> : Aiednn_Op<mnemonic, [Pure, AttrSizedOperandSegments]> {
  let arguments = !con((ins Aiednn_Tensor:$src, Optional<Aiednn_Tensor>:$scale, Optional<Aiednn_Tensor>:$shift,
                            F32Attr:$eps, I64Attr:$flags), extra);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$src (`scale` $scale^)? (`shift` $shift^)? prop-dict attr-dict `:` functional-type(operands, results)";
}
def Aiednn_LayerNormOp : Aiednn_NormOp<"layer_norm", (ins OptionalAttr<F32Attr>:$dst_scale,
                                                         OptionalAttr<I64Attr>:$dst_zero_point)> {
  let summary = "layer normalization over the last dim (flags 8: RMS, AIEDNN_RMS_NORM)";
}
def Aiednn_GroupNormOp : Aiednn_NormOp<"group_norm", (ins I64Attr:$groups)> {
  let summary = "group normalization";
}

def Aiednn_ConcatOp : Aiednn_Op<"concat", [Pure]> {
  let summary = "concatenate along axis";
  let arguments = (ins Variadic<Aiednn_Tensor>:$srcs, I64Attr:$axis);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$srcs prop-dict attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Aiednn_SumOp : Aiednn_Op<"sum", [Pure]> {
  let summary = "y = sum_i scales[i] * srcs[i]";
  let arguments = (ins Variadic<Aiednn_Tensor>:$srcs, DenseF32ArrayAttr:$scales);
  let results = (outs Aiednn_Tensor:$result);
  let assemblyFormat = "$srcs prop-dict attr-dict `:` functional-type(operands, results)";
}

#endif // AIEDNN_OPS
