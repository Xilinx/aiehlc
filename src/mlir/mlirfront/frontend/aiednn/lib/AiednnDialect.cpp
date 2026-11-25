/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// aiednn dialect: registration and verifiers. The verifiers check what the C
// runtime's create functions would reject (src/aiednn/aiednn_rt.cc), so a model
// that verifies here creates without `unimplemented` / `invalid_arguments`.
#include "AiednnDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace aiednn;

#include "AiednnOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "AiednnOpsAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "AiednnOps.cpp.inc"

void AiednnDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "AiednnOpsAttrDefs.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "AiednnOps.cpp.inc"
        >();
}

static RankedTensorType rt(Value v) { return llvm::cast<RankedTensorType>(v.getType()); }

LogicalResult ViewOp::verify() {
    RankedTensorType s = rt(getSrc()), d = getResult().getType();
    if (s.getElementType() != d.getElementType())
        return emitOpError("a view keeps the element type");
    if (s.hasStaticShape() && d.hasStaticShape() && s.getNumElements() != d.getNumElements())
        return emitOpError("a view keeps the element count (")
               << s.getNumElements() << " vs " << d.getNumElements() << ")";
    return success();
}

/* checks shared by the GEMM family: binary post-ops each consume one post_srcs
 * operand, output scales are per tensor or per output channel */
template <typename OpT> static LogicalResult verifyEpilogue(OpT op, int64_t channels) {
    int64_t nbin = 0;
    if (auto po = op.getPostOpsAttr()) {
        if (po.size() > 4)
            return op.emitOpError("at most 4 post-ops (aiednn_attr_t)");
        for (Attribute a : po) {
            StringRef k = llvm::cast<PostOpAttr>(a).getKind().getValue();
            if (k == "binary")
                nbin++;
            else if (k != "eltwise")
                return op.emitOpError("post-op kind must be eltwise or binary, not ") << k;
        }
    }
    if ((int64_t)op.getPostSrcs().size() != nbin)
        return op.emitOpError("expects one post_srcs operand per binary post-op (") << nbin << ")";
    if (auto sc = op.getOutputScalesAttr())
        if (sc.size() != 1 && (int64_t)sc.size() != channels)
            return op.emitOpError("output_scales must have 1 or ") << channels << " entries, not " << sc.size();
    return success();
}

template <typename OpT> static LogicalResult verifyConvLike(OpT op, bool deconv) {
    RankedTensorType s = rt(op.getSrc()), w = rt(op.getWeights()), d = op.getResult().getType();
    if (s.getRank() != 4 || w.getRank() != 4 || d.getRank() != 4)
        return op.emitOpError("expects 4-D src [N,C,H,W], weights [O,C/g,KH,KW] and dst");
    const int64_t g = op.getGroups();
    if (g < 1 || s.getDimSize(1) % g || d.getDimSize(1) % g)
        return op.emitOpError("groups must divide C and O");
    if (!deconv && (w.getDimSize(0) != d.getDimSize(1) || w.getDimSize(1) * g != s.getDimSize(1)))
        return op.emitOpError("weights must be [O, C/groups, KH, KW]");
    for (auto arr : {op.getStrides(), op.getPadL(), op.getPadR(), op.getDilations()})
        if (arr.size() != 2)
            return op.emitOpError("strides / pads / dilations take 2 entries");
    return verifyEpilogue(op, d.getDimSize(1));
}

LogicalResult ConvOp::verify() { return verifyConvLike(*this, false); }
LogicalResult DeconvOp::verify() { return verifyConvLike(*this, true); }

LogicalResult MatmulOp::verify() {
    RankedTensorType s = rt(getSrc()), d = getResult().getType();
    if (s.getRank() < 2 || d.getRank() < 2)
        return emitOpError("matmul operands are at least 2-D");
    return verifyEpilogue(*this, d.getDimSize(d.getRank() - 1));
}

LogicalResult InnerProductOp::verify() {
    RankedTensorType d = getResult().getType();
    if (d.getRank() != 2)
        return emitOpError("inner product dst is [N, O]");
    return verifyEpilogue(*this, d.getDimSize(1));
}

LogicalResult ReorderOp::verify() {
    RankedTensorType s = rt(getSrc()), d = getResult().getType();
    if (s.getRank() != d.getRank())
        return emitOpError("reorder keeps the rank");
    auto perm = getPerm();
    if (perm && (int64_t)perm->size() != s.getRank())
        return emitOpError("perm has one entry per dim");
    for (int64_t i = 0; i < s.getRank(); i++) {
        const int64_t p = perm ? (*perm)[i] : i;
        if (p < 0 || p >= s.getRank() || s.getDimSize(p) != d.getDimSize(i))
            return emitOpError("dst dim ") << i << " must be src dim " << p;
    }
    return success();
}

LogicalResult ConcatOp::verify() {
    RankedTensorType d = getResult().getType();
    const int64_t ax = getAxis();
    if (getSrcs().empty() || ax < 0 || ax >= d.getRank())
        return emitOpError("needs sources and an axis inside the dst rank");
    int64_t total = 0;
    for (Value v : getSrcs()) {
        RankedTensorType s = rt(v);
        if (s.getRank() != d.getRank())
            return emitOpError("sources have the dst rank");
        for (int64_t i = 0; i < d.getRank(); i++)
            if (i != ax && s.getDimSize(i) != d.getDimSize(i))
                return emitOpError("sources differ from dst outside the axis (dim ") << i << ")";
        total += s.getDimSize(ax);
    }
    if (total != d.getDimSize(ax))
        return emitOpError("sources sum to ") << total << " along the axis, dst has " << d.getDimSize(ax);
    return success();
}

//===----------------------------------------------------------------------===//
// Folders and canonicalization (aiednn-opt --canonicalize / --cse)
//===----------------------------------------------------------------------===//

OpFoldResult ViewOp::fold(FoldAdaptor) {
    if (getSrc().getType() == getResult().getType())
        return getSrc();
    if (auto inner = getSrc().getDefiningOp<ViewOp>()) { // view(view(x)) -> view(x)
        getSrcMutable().assign(inner.getSrc());
        return getResult();
    }
    return {};
}

static bool identityPerm(std::optional<ArrayRef<int32_t>> perm) {
    if (!perm)
        return true;
    for (size_t i = 0; i < perm->size(); i++)
        if ((*perm)[i] != (int32_t)i)
            return false;
    return true;
}

/* same type and no permutation: the runtime copies (scale / zero point unused) */
OpFoldResult ReorderOp::fold(FoldAdaptor) {
    if (getSrc().getType() == getResult().getType() && identityPerm(getPerm()))
        return getSrc();
    return {};
}

/* quantize(dequantize(q)) -> q when both use the same scale / zero point and no
 * permutation: (q - zp) * s is within half an f32 ulp of an integer multiple of s,
 * so nearbyint((q - zp) * s / s) + zp == q -- the round trip is exact */
static LogicalResult foldQuantRoundTrip(ReorderOp op, PatternRewriter &rw) {
    auto deq = op.getSrc().getDefiningOp<ReorderOp>();
    if (!deq || !identityPerm(op.getPerm()) || !identityPerm(deq.getPerm()))
        return failure();
    Type q = deq.getSrc().getType();
    if (q != op.getResult().getType() || !llvm::isa<IntegerType>(getElementTypeOrSelf(q)) ||
        !getElementTypeOrSelf(deq.getResult().getType()).isF32())
        return failure();
    if (deq.getScale() != op.getScale() || deq.getZeroPoint() != op.getZeroPoint())
        return failure();
    rw.replaceOp(op, deq.getSrc());
    return success();
}

void ReorderOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *) {
    results.add(foldQuantRoundTrip);
}
