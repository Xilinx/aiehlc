/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// --aiednn-fuse-epilogue: fold the single-use eltwise / residual-add chain after a
// GEMM-family op (conv, deconv, matmul, inner_product) into its post_ops, at most
// 4 (aiednn_attr_t):
//   * aiednn.eltwise on the result, same type        -> #aiednn.post_op<"eltwise", alg, a, b>
//   * aiednn.binary "add" / "mul" of the result and o -> #aiednn.post_op<"binary", alg, 0, 0>
//     available before the GEMM, broadcastable to the  with o appended to post_srcs
//     result, result type unchanged                    (a view feeding o is hoisted)
#include "AiednnDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

bool broadcastable(RankedTensorType o, RankedTensorType r) {
    if (o.getRank() != r.getRank())
        return false;
    for (int64_t i = 0; i < r.getRank(); i++)
        if (o.getDimSize(i) != 1 && o.getDimSize(i) != r.getDimSize(i))
            return false;
    return true;
}

/* o usable as an input of g: a block argument, or defined before g in g's block
 * (a constant, or a view whose source is usable, is moved up to g) */
bool availableBefore(Value o, Operation *g) {
    if (llvm::isa<BlockArgument>(o))
        return true;
    Operation *d = o.getDefiningOp();
    if (d->getBlock() != g->getBlock())
        return false;
    if (d->isBeforeInBlock(g))
        return true;
    if (llvm::isa<aiednn::ConstOp>(d)) { // no operands: hoisting it is free
        d->moveBefore(g);
        return true;
    }
    if (auto v = llvm::dyn_cast<aiednn::ViewOp>(d))
        if (availableBefore(v.getSrc(), g)) {
            v->moveBefore(g);
            return true;
        }
    return false;
}

FloatAttr f32(MLIRContext *ctx, float v) { return FloatAttr::get(Float32Type::get(ctx), v); }

/* `keep` takes over `gone`: its location becomes the fused location of both, so the plan export
 * lists every source op a primitive computes */
void absorb(Operation *keep, Operation *gone) {
    keep->setLoc(FusedLoc::get(keep->getContext(), {keep->getLoc(), gone->getLoc()}));
}

template <typename OpT> int fuseInto(OpT g) {
    MLIRContext *ctx = g->getContext();
    SmallVector<Attribute> post;
    if (auto p = g.getPostOpsAttr())
        post.append(p.begin(), p.end());
    int fused = 0;
    Value r = g.getResult();
    while (post.size() < 4 && r.hasOneUse()) {
        Operation *u = *r.getUsers().begin();
        if (u->getNumResults() != 1 || u->getResult(0).getType() != r.getType()) // e.g. func.return
            break;
        if (auto e = llvm::dyn_cast<aiednn::EltwiseOp>(u)) {
            post.push_back(aiednn::PostOpAttr::get(ctx, StringAttr::get(ctx, "eltwise"), e.getAlgAttr(),
                                                   e.getAlphaAttr(), e.getBetaAttr()));
        } else if (auto b = llvm::dyn_cast<aiednn::BinaryOp>(u)) {
            if ((b.getAlg() != "add" && b.getAlg() != "mul") || b.getA() == b.getB())
                break;
            Value o = b.getA() == r ? b.getB() : b.getA();
            if (!broadcastable(llvm::cast<RankedTensorType>(o.getType()), llvm::cast<RankedTensorType>(r.getType())) ||
                !availableBefore(o, g))
                break;
            post.push_back(aiednn::PostOpAttr::get(ctx, StringAttr::get(ctx, "binary"), b.getAlgAttr(), f32(ctx, 0.0f),
                                                   f32(ctx, 0.0f)));
            g.getPostSrcsMutable().append(o);
        } else {
            break;
        }
        absorb(g, u);
        u->getResult(0).replaceAllUsesWith(r);
        u->erase();
        fused++;
    }
    if (fused)
        g.setPostOpsAttr(ArrayAttr::get(ctx, post));
    return fused;
}

struct FuseEpiloguePass : public PassWrapper<FuseEpiloguePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseEpiloguePass)
    StringRef getArgument() const final { return "aiednn-fuse-epilogue"; }
    StringRef getDescription() const final {
        return "fold single-use eltwise / residual-add chains into GEMM-family post_ops";
    }
    void runOnOperation() override {
        SmallVector<Operation *> gemms;
        getOperation().walk([&](Operation *op) {
            if (llvm::isa<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(op))
                gemms.push_back(op);
        });
        for (Operation *op : gemms)
            llvm::TypeSwitch<Operation *>(op)
                .Case<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(
                    [](auto g) { fuseInto(g); });
    }
};

/* --aiednn-fold-requant: a GEMM-family op whose f32 result is only quantized
 * (reorder f32 -> s8/u8, no permutation) writes the integer tensor itself:
 * dst_scale / dst_zero_point = the reorder's scale / zero point, result type = the
 * reorder's. Runs after --aiednn-fuse-epilogue (the requant is the last epilogue step). */
template <typename OpT> bool foldRequant(OpT g) {
    Value r = g.getResult();
    if (!r.hasOneUse() || !getElementTypeOrSelf(r.getType()).isF32() || g.getDstScaleAttr())
        return false;
    auto q = llvm::dyn_cast<aiednn::ReorderOp>(*r.getUsers().begin());
    if (!q || q.getPerm())
        return false;
    auto qt = llvm::cast<RankedTensorType>(q.getResult().getType());
    auto it = llvm::dyn_cast<IntegerType>(qt.getElementType());
    if (!it || it.getWidth() != 8 || qt.getShape() != llvm::cast<RankedTensorType>(r.getType()).getShape())
        return false;
    g.setDstScaleAttr(q.getScaleAttr());
    g.setDstZeroPointAttr(q.getZeroPointAttr());
    r.setType(qt);
    absorb(g, q);
    q.getResult().replaceAllUsesWith(r);
    q->erase();
    return true;
}

struct FoldRequantPass : public PassWrapper<FoldRequantPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldRequantPass)
    StringRef getArgument() const final { return "aiednn-fold-requant"; }
    StringRef getDescription() const final {
        return "GEMM-family op followed only by a quantizing reorder writes the int8 tensor directly";
    }
    void runOnOperation() override {
        SmallVector<Operation *> gemms;
        getOperation().walk([&](Operation *op) {
            if (llvm::isa<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(op))
                gemms.push_back(op);
        });
        for (Operation *op : gemms)
            llvm::TypeSwitch<Operation *>(op)
                .Case<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(
                    [](auto g) { foldRequant(g); });
    }
};

/* --aiednn-fold-norm-quant: an f32 batch_norm / layer_norm whose result is only quantized (reorder f32 ->
 * s8/u8, no permutation) writes the integer tensor itself, as --aiednn-fold-requant does for a
 * GEMM: dst_scale / dst_zero_point = the reorder's, result type = the reorder's. Bit-identical:
 * the runtime rounds the norm to f32, then applies the reorder's arithmetic. */
template <typename NormT> void foldNormQuant(NormT b) {
    Value r = b.getResult();
    if (!r.hasOneUse() || !getElementTypeOrSelf(r.getType()).isF32() ||
        !getElementTypeOrSelf(b->getOperand(0).getType()).isF32() || b.getDstScaleAttr())
        return;
    /* r -> reorder, or r -> view -> reorder (sdpa's [B,S,H,D] result reshaped to [B,S,H*D] first):
     * a view is a pure reshape, so the int8 element type moves through it */
    Operation *u = *r.getUsers().begin();
    auto v = llvm::dyn_cast<aiednn::ViewOp>(u);
    Value vin = r;
    if (v) {
        if (!v.getResult().hasOneUse())
            return;
        vin = v.getResult();
        u = *vin.getUsers().begin();
    }
    auto q = llvm::dyn_cast<aiednn::ReorderOp>(u);
    if (!q || q.getPerm())
        return;
    auto qt = llvm::cast<RankedTensorType>(q.getResult().getType());
    auto it = llvm::dyn_cast<IntegerType>(qt.getElementType());
    if (!it || it.getWidth() != 8 || qt.getShape() != llvm::cast<RankedTensorType>(vin.getType()).getShape())
        return;
    b.setDstScaleAttr(q.getScaleAttr());
    b.setDstZeroPointAttr(q.getZeroPointAttr());
    r.setType(RankedTensorType::get(llvm::cast<RankedTensorType>(r.getType()).getShape(), qt.getElementType()));
    if (v)
        vin.setType(qt);
    absorb(b, q);
    q.getResult().replaceAllUsesWith(vin);
    q->erase();
}

struct FoldNormQuantPass : public PassWrapper<FoldNormQuantPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldNormQuantPass)
    StringRef getArgument() const final { return "aiednn-fold-norm-quant"; }
    StringRef getDescription() const final {
        return "f32 batch / layer norm followed only by a quantizing reorder writes the int8 tensor directly";
    }
    void runOnOperation() override {
        SmallVector<Operation *> ns;
        getOperation().walk([&](Operation *op) {
            if (llvm::isa<aiednn::BatchNormOp, aiednn::LayerNormOp, aiednn::SdpaOp>(op))
                ns.push_back(op);
        });
        for (Operation *op : ns)
            llvm::TypeSwitch<Operation *>(op).Case<aiednn::BatchNormOp, aiednn::LayerNormOp, aiednn::SdpaOp>(
                [](auto n) { foldNormQuant(n); });
    }
};

/* --aiednn-fold-head-permute: when q, k, v all come from a [B,S,H,D] -> [B,H,S,D] permute (single use,
 * a pure reorder), sdpa reads [B,S,H,D] directly (layout bit 0); when its result's only user permutes it
 * back, it writes [B,S,H,D] (layout bit 1). Bit-identical. */
static bool isHeadPermute(aiednn::ReorderOp r) {
    auto p = r.getPermAttr();
    if (!p)
        return false;
    auto a = p.asArrayRef();
    return a.size() == 4 && a[0] == 0 && a[1] == 2 && a[2] == 1 && a[3] == 3 && r.getScale().convertToFloat() == 1.0f &&
           r.getZeroPoint() == 0 &&
           getElementTypeOrSelf(r.getSrc().getType()) == getElementTypeOrSelf(r.getResult().getType());
}

struct FoldHeadPermutePass : public PassWrapper<FoldHeadPermutePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldHeadPermutePass)
    StringRef getArgument() const final { return "aiednn-fold-head-permute"; }
    StringRef getDescription() const final { return "fold attention's head split / merge permutes into sdpa"; }
    void runOnOperation() override {
        SmallVector<aiednn::SdpaOp> ss;
        getOperation().walk([&](aiednn::SdpaOp s) { ss.push_back(s); });
        for (aiednn::SdpaOp s : ss) {
            if (s.getLayoutAttr())
                continue;
            int64_t layout = 0;
            SmallVector<aiednn::ReorderOp> in;
            for (Value v : {s.getQ(), s.getK(), s.getV()}) {
                auto r = v.getDefiningOp<aiednn::ReorderOp>();
                if (r && isHeadPermute(r) && v.hasOneUse())
                    in.push_back(r);
            }
            if (in.size() == 3) { // q, k, v: read the [B,S,H,D] tensors
                for (int i = 0; i < 3; i++) {
                    s->setOperand(i, in[i].getSrc());
                    absorb(s, in[i]);
                    in[i]->erase();
                }
                layout |= 1;
            }
            Value r = s.getResult();
            if (r.hasOneUse())
                if (auto u = llvm::dyn_cast<aiednn::ReorderOp>(*r.getUsers().begin()))
                    if (isHeadPermute(u)) { // the result: write [B,S,H,D]
                        r.setType(u.getResult().getType());
                        absorb(s, u);
                        u.getResult().replaceAllUsesWith(r);
                        u->erase();
                        layout |= 2;
                    }
            if (layout)
                s.setLayoutAttr(IntegerAttr::get(IntegerType::get(s->getContext(), 64), layout));
        }
    }
};

/* --aiednn-quantize-before-maxpool: quantize(maxpool(x)) -> maxpool(quantize(x)) when every
 * user of the f32 max-pool is the same quantizing reorder (no permutation). Exact: rounding
 * and saturation are monotonic non-decreasing, so the max commutes with them. */
struct QuantBeforeMaxPoolPass : public PassWrapper<QuantBeforeMaxPoolPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantBeforeMaxPoolPass)
    StringRef getArgument() const final { return "aiednn-quantize-before-maxpool"; }
    StringRef getDescription() const final { return "move a quantizing reorder ahead of a max pooling"; }
    void runOnOperation() override {
        SmallVector<aiednn::PoolingOp> pools;
        getOperation().walk([&](aiednn::PoolingOp p) { pools.push_back(p); });
        for (aiednn::PoolingOp p : pools) {
            if (p.getAlg() != "max" || !getElementTypeOrSelf(p.getSrc().getType()).isF32() || p.getResult().use_empty())
                continue;
            aiednn::ReorderOp q0;
            bool ok = true;
            for (Operation *u : p.getResult().getUsers()) {
                auto q = llvm::dyn_cast<aiednn::ReorderOp>(u);
                if (!q || q.getPerm() || !llvm::isa<IntegerType>(getElementTypeOrSelf(q.getResult().getType())) ||
                    (q0 && (q.getScale() != q0.getScale() || q.getZeroPoint() != q0.getZeroPoint() ||
                            q.getResult().getType() != q0.getResult().getType()))) {
                    ok = false;
                    break;
                }
                q0 = q0 ? q0 : q;
            }
            if (!ok || !q0)
                continue;
            OpBuilder b(p);
            auto srcT = llvm::cast<RankedTensorType>(p.getSrc().getType());
            Type elt = getElementTypeOrSelf(q0.getResult().getType());
            auto qx = b.create<aiednn::ReorderOp>(q0.getLoc(), RankedTensorType::get(srcT.getShape(), elt), p.getSrc(),
                                                  q0.getPermAttr(), q0.getScaleAttr(), q0.getZeroPointAttr());
            auto np = llvm::cast<aiednn::PoolingOp>(b.clone(*p.getOperation()));
            np->setOperand(0, qx.getResult());
            np.getResult().setType(llvm::cast<RankedTensorType>(q0.getResult().getType()));
            for (Operation *u : llvm::make_early_inc_range(p.getResult().getUsers())) {
                u->getResult(0).replaceAllUsesWith(np.getResult());
                u->erase();
            }
            p->erase();
        }
    }
};

/* --aiednn-quantize-residual: a fused residual add (binary post-op) whose f32 operand o is
 * also quantized for some other consumer (reorder f32 -> s8/u8, no permutation, same
 * shape) takes that int8 tensor instead: a new reorder with the same scale / zero point is
 * placed right after o's definition (so --cse merges the existing one into it) and the
 * post-op records the operand's scale / zero point as alpha / beta. Not exact: the add sees the
 * skip at the precision the next layer's input already has. */
template <typename OpT> int quantizeResidual(OpT g) {
    auto post = g.getPostOpsAttr();
    if (!post)
        return 0;
    MLIRContext *ctx = g->getContext();
    SmallVector<Attribute> pos(post.begin(), post.end());
    int k = 0, changed = 0;
    for (Attribute &a : pos) {
        auto po = llvm::cast<aiednn::PostOpAttr>(a);
        if (po.getKind().getValue() != "binary")
            continue;
        const int idx = k++;
        Value o = g.getPostSrcs()[idx];
        if (po.getAlg().getValue() != "add" || !getElementTypeOrSelf(o.getType()).isF32() ||
            o.getType() != g.getResult().getType())
            continue;
        aiednn::ReorderOp q;
        for (Operation *u : o.getUsers()) {
            auto r = llvm::dyn_cast<aiednn::ReorderOp>(u);
            auto it = r ? llvm::dyn_cast<IntegerType>(getElementTypeOrSelf(r.getResult().getType())) : IntegerType();
            if (r && !r.getPerm() && it && it.getWidth() == 8 &&
                llvm::cast<RankedTensorType>(r.getResult().getType()).getShape() ==
                    llvm::cast<RankedTensorType>(o.getType()).getShape()) {
                q = r;
                break;
            }
        }
        if (!q && o.hasOneUse()) {
            /* the skip feeds nothing quantized: quantize it at the scale of the quantizing reorder
             * on this GEMM's own result (dnn_torch._quantize_residuals mirrors this) */
            for (Operation *u : g.getResult().getUsers()) {
                auto r = llvm::dyn_cast<aiednn::ReorderOp>(u);
                auto it =
                    r ? llvm::dyn_cast<IntegerType>(getElementTypeOrSelf(r.getResult().getType())) : IntegerType();
                if (r && !r.getPerm() && it && it.getWidth() == 8 &&
                    llvm::cast<RankedTensorType>(r.getResult().getType()).getShape() ==
                        llvm::cast<RankedTensorType>(o.getType()).getShape()) {
                    q = r;
                    break;
                }
            }
        }
        if (!q)
            continue;
        OpBuilder b(ctx);
        if (Operation *d = o.getDefiningOp())
            b.setInsertionPointAfter(d);
        else
            b.setInsertionPointToStart(llvm::cast<BlockArgument>(o).getOwner());
        auto qo = b.create<aiednn::ReorderOp>(q.getLoc(), q.getResult().getType(), o, q.getPermAttr(), q.getScaleAttr(),
                                              q.getZeroPointAttr());
        g.getPostSrcsMutable()[idx].assign(qo.getResult());
        a = aiednn::PostOpAttr::get(ctx, po.getKind(), po.getAlg(), q.getScaleAttr(),
                                    f32(ctx, (float)(int64_t)q.getZeroPoint()));
        changed++;
    }
    if (changed)
        g.setPostOpsAttr(ArrayAttr::get(ctx, pos));
    return changed;
}

struct QuantizeResidualPass : public PassWrapper<QuantizeResidualPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeResidualPass)
    StringRef getArgument() const final { return "aiednn-quantize-residual"; }
    StringRef getDescription() const final {
        return "fused residual adds take the int8 copy of their skip operand (scale / zp on the post-op)";
    }
    void runOnOperation() override {
        SmallVector<Operation *> gemms;
        getOperation().walk([&](Operation *op) {
            if (llvm::isa<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(op))
                gemms.push_back(op);
        });
        for (Operation *op : gemms)
            llvm::TypeSwitch<Operation *>(op)
                .Case<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(
                    [](auto g) { quantizeResidual(g); });
    }
};

/* --aiednn-sink-permute: binary(reorder_perm(x), c) -> reorder_perm(binary(x, view(c))) when the
 * reorder is a pure f32 permutation used only by that binary, and c is a constant broadcast
 * whose non-singleton dims keep their order under the inverse permutation (so the permuted
 * constant is just a view). Exact. --aiednn-fuse-epilogue can then fuse the binary. */
struct SinkPermutePass : public PassWrapper<SinkPermutePass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkPermutePass)
    StringRef getArgument() const final { return "aiednn-sink-permute"; }
    StringRef getDescription() const final {
        return "move a constant elementwise binary above a pure permutation (towards its producer)";
    }
    static bool isConst(Value v) {
        Operation *d = v.getDefiningOp();
        while (auto vw = llvm::dyn_cast_or_null<aiednn::ViewOp>(d))
            d = vw.getSrc().getDefiningOp();
        return llvm::isa_and_nonnull<aiednn::ConstOp>(d);
    }
    void runOnOperation() override {
        SmallVector<aiednn::BinaryOp> bins;
        getOperation().walk([&](aiednn::BinaryOp b) { bins.push_back(b); });
        for (aiednn::BinaryOp b : bins) {
            if (b.getA() == b.getB())
                continue;
            const bool rFirst = (bool)b.getA().getDefiningOp<aiednn::ReorderOp>();
            Value rv = rFirst ? b.getA() : b.getB(), c = rFirst ? b.getB() : b.getA();
            auto r = rv.getDefiningOp<aiednn::ReorderOp>();
            if (!r || !r.getPerm() || !rv.hasOneUse() || !isConst(c) || r.getScale().convertToFloat() != 1.0f ||
                r.getZeroPoint() != 0)
                continue;
            auto xt = llvm::cast<RankedTensorType>(r.getSrc().getType());
            auto rt = llvm::cast<RankedTensorType>(rv.getType());
            auto ct = llvm::cast<RankedTensorType>(c.getType());
            if (!xt.getElementType().isF32() || !rt.getElementType().isF32() || ct.getRank() != rt.getRank() ||
                b.getResult().getType() != rt)
                continue;
            ArrayRef<int32_t> perm = *r.getPerm(); // r dim i = x dim perm[i]
            const int64_t n = rt.getRank();
            SmallVector<int64_t> cs(n, 1); // c in x's layout: dim perm[i] = c dim i
            for (int64_t i = 0; i < n; i++)
                cs[perm[i]] = ct.getDimSize(i);
            // a view iff the non-singleton dims of c keep their ORDER (which dims, not just their sizes)
            int64_t lastPos = -1;
            bool inOrder = true;
            for (int64_t i = 0; i < n && inOrder; i++)
                if (ct.getDimSize(i) != 1) {
                    inOrder = perm[i] > lastPos;
                    lastPos = perm[i];
                }
            if (!inOrder)
                continue;
            for (int64_t i = 0; i < n; i++)
                if (cs[i] != 1 && cs[i] != xt.getDimSize(i))
                    goto next;
            {
                OpBuilder bld(b);
                auto cv = aiednn::ViewOp::create(bld, b.getLoc(), RankedTensorType::get(cs, ct.getElementType()), c);
                Value lhs = rFirst ? r.getSrc() : cv.getResult(), rhs = rFirst ? cv.getResult() : r.getSrc();
                auto nb = aiednn::BinaryOp::create(bld, b.getLoc(), xt, lhs, rhs, b.getAlgAttr());
                auto nr = aiednn::ReorderOp::create(bld, r.getLoc(), rt, nb.getResult(), r.getPermAttr(),
                                                    r.getScaleAttr(), r.getZeroPointAttr());
                b.getResult().replaceAllUsesWith(nr.getResult());
                b->erase();
                r->erase();
            }
        next:;
        }
    }
};

/* --aiednn-fuse-norm-relu: batch_norm whose only user is a ReLU (alpha 0) of the same type takes
 * AIEDNN_FUSE_NORM_RELU (flags | 4) and the ReLU disappears. Exact. */
struct FuseNormReluPass : public PassWrapper<FuseNormReluPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseNormReluPass)
    StringRef getArgument() const final { return "aiednn-fuse-norm-relu"; }
    StringRef getDescription() const final { return "fold a ReLU into the batch_norm before it (flags | 4)"; }
    void runOnOperation() override {
        SmallVector<aiednn::BatchNormOp> bns;
        getOperation().walk([&](aiednn::BatchNormOp b) { bns.push_back(b); });
        for (aiednn::BatchNormOp b : bns) {
            Value r = b.getResult();
            if (!r.hasOneUse() || (b.getFlags() & 4))
                continue;
            auto e = llvm::dyn_cast<aiednn::EltwiseOp>(*r.getUsers().begin());
            if (!e || e.getAlg() != "relu" || e.getAlpha().convertToFloat() != 0.0f ||
                e.getResult().getType() != r.getType())
                continue;
            b.setFlags(b.getFlags() | 4);
            absorb(b, e);
            e.getResult().replaceAllUsesWith(r);
            e->erase();
        }
    }
};
/* --aiednn-quantize-through-views: move quantizing reorders (f32 -> s8/u8, no permutation) up through
 * concat and slice to the GEMM-family op that produced the f32 values, then give that op one clone per
 * distinct quantization, so --aiednn-fold-requant lets every one of them write int8 on the core:
 *   reorder(concat(x_i)) -> concat(reorder(x_i))        (all inputs f32, the reorder its only user)
 *   reorder(slice(x))    -> slice(reorder(x))           (per reorder user of the slice)
 *   g with users = reorders at k scales -> k copies of g, one per scale
 * Each step is exact: quantization is elementwise, a slice / concat only moves elements, and a clone
 * recomputes the same values. */
bool isQuant(aiednn::ReorderOp q) {
    return q && !q.getPerm() && getElementTypeOrSelf(q.getSrc().getType()).isF32() &&
           llvm::isa<IntegerType>(getElementTypeOrSelf(q.getResult().getType()));
}

RankedTensorType retyped(Type t, Type elt) {
    return RankedTensorType::get(llvm::cast<RankedTensorType>(t).getShape(), elt);
}

bool pushThroughConcat(aiednn::ReorderOp q) {
    auto c = q.getSrc().getDefiningOp<aiednn::ConcatOp>();
    if (!c || !c.getResult().hasOneUse())
        return false;
    for (Value v : c.getSrcs())
        if (!getElementTypeOrSelf(v.getType()).isF32())
            return false;
    OpBuilder b(c);
    Type elt = getElementTypeOrSelf(q.getResult().getType());
    SmallVector<Value> qs;
    for (Value v : c.getSrcs())
        qs.push_back(b.create<aiednn::ReorderOp>(q.getLoc(), retyped(v.getType(), elt), v, q.getPermAttr(),
                                                 q.getScaleAttr(), q.getZeroPointAttr()));
    auto nc = b.create<aiednn::ConcatOp>(c.getLoc(), q.getResult().getType(), qs, c.getAxisAttr());
    q.getResult().replaceAllUsesWith(nc.getResult());
    q->erase();
    c->erase();
    return true;
}

bool pushThroughSlice(aiednn::ReorderOp q) {
    auto sl = q.getSrc().getDefiningOp<aiednn::SliceOp>();
    if (!sl || !getElementTypeOrSelf(sl.getSrc().getType()).isF32())
        return false;
    OpBuilder b(sl);
    Type elt = getElementTypeOrSelf(q.getResult().getType());
    auto qx = b.create<aiednn::ReorderOp>(q.getLoc(), retyped(sl.getSrc().getType(), elt), sl.getSrc(), q.getPermAttr(),
                                          q.getScaleAttr(), q.getZeroPointAttr());
    auto ns = b.create<aiednn::SliceOp>(sl.getLoc(), q.getResult().getType(), qx.getResult(), sl.getOffsetsAttr(),
                                        sl.getStepsAttr());
    q.getResult().replaceAllUsesWith(ns.getResult());
    q->erase();
    if (sl.getResult().use_empty())
        sl->erase();
    return true;
}

/* reorder(op(x)) -> op'(reorder(x)) for a max pooling / nearest resampling (per reorder user): exact,
 * the max and the nearest pick commute with monotonic rounding + saturation, so they run on int8 */
bool pushThroughPick(aiednn::ReorderOp q) {
    Operation *p = q.getSrc().getDefiningOp();
    bool pick = false;
    if (auto pool = llvm::dyn_cast_or_null<aiednn::PoolingOp>(p))
        pick = pool.getAlg() == "max";
    if (auto rs = llvm::dyn_cast_or_null<aiednn::ResamplingOp>(p))
        pick = rs.getAlg() == "nearest";
    if (!pick || !getElementTypeOrSelf(p->getOperand(0).getType()).isF32())
        return false;
    OpBuilder b(p);
    Type elt = getElementTypeOrSelf(q.getResult().getType());
    auto qx = b.create<aiednn::ReorderOp>(q.getLoc(), retyped(p->getOperand(0).getType(), elt), p->getOperand(0),
                                          q.getPermAttr(), q.getScaleAttr(), q.getZeroPointAttr());
    Operation *np = b.clone(*p);
    np->setOperand(0, qx.getResult());
    np->getResult(0).setType(q.getResult().getType());
    q.getResult().replaceAllUsesWith(np->getResult(0));
    q->erase();
    if (p->getResult(0).use_empty())
        p->erase();
    return true;
}

/* g's users all quantizing reorders: merge equal ones, then one clone of g per extra quantization */
void splitPerQuant(Operation *g, bool clone) {
    Value r = g->getResult(0);
    if (r.use_empty() || !getElementTypeOrSelf(r.getType()).isF32())
        return;
    SmallVector<aiednn::ReorderOp> qs;
    for (Operation *u : r.getUsers()) {
        auto q = llvm::dyn_cast<aiednn::ReorderOp>(u);
        if (!isQuant(q))
            return;
        qs.push_back(q);
    }
    SmallVector<aiednn::ReorderOp> keep; // one reorder per (scale, zero point, type)
    for (aiednn::ReorderOp q : qs) {
        aiednn::ReorderOp same;
        for (aiednn::ReorderOp k : keep)
            if (k.getScale() == q.getScale() && k.getZeroPoint() == q.getZeroPoint() &&
                k.getResult().getType() == q.getResult().getType())
                same = k;
        if (!same) {
            keep.push_back(q);
            continue;
        }
        if (q->isBeforeInBlock(same))
            same->moveBefore(q);
        q.getResult().replaceAllUsesWith(same.getResult());
        q->erase();
    }
    if (keep.size() < 2 || !clone)
        return; // one quantization: --aiednn-fold-requant takes it; several stay unless clone=true
    OpBuilder b(g);
    SmallVector<Operation *> copies{g};
    for (size_t i = 1; i < keep.size(); i++) {
        Operation *c = b.clone(*g);
        keep[i]->setOperand(0, c->getResult(0));
        copies.push_back(c);
    }
    /* fold each copy's requant now: identical clones would be merged back by the next --cse */
    for (Operation *c : copies)
        llvm::TypeSwitch<Operation *>(c)
            .Case<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(
                [](auto op) { foldRequant(op); });
}

struct QuantizeThroughViewsPass : public PassWrapper<QuantizeThroughViewsPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeThroughViewsPass)
    QuantizeThroughViewsPass() = default;
    QuantizeThroughViewsPass(const QuantizeThroughViewsPass &o) : PassWrapper(o) {}
    /* opt-in: clones can cost more than the reorders they remove */
    Option<bool> clone{*this, "clone", llvm::cl::desc("one GEMM clone per extra quantization"), llvm::cl::init(false)};
    StringRef getArgument() const final { return "aiednn-quantize-through-views"; }
    StringRef getDescription() const final {
        return "quantizing reorders up through concat / slice / max pool / nearest resampling, equal ones merged "
               "(exact); clone=true: one GEMM clone per extra quantization";
    }
    void runOnOperation() override {
        for (bool changed = true; changed;) { // a concat's new reorders may sit on slices: iterate
            changed = false;
            SmallVector<aiednn::ReorderOp> qs;
            getOperation().walk([&](aiednn::ReorderOp q) {
                if (isQuant(q))
                    qs.push_back(q);
            });
            for (aiednn::ReorderOp q : qs)
                changed |= pushThroughConcat(q) || pushThroughSlice(q) || pushThroughPick(q);
        }
        SmallVector<Operation *> gemms;
        getOperation().walk([&](Operation *op) {
            if (llvm::isa<aiednn::ConvOp, aiednn::DeconvOp, aiednn::MatmulOp, aiednn::InnerProductOp>(op))
                gemms.push_back(op);
        });
        for (Operation *g : gemms)
            splitPerQuant(g, clone);
    }
};
} // namespace

std::unique_ptr<Pass> aiednn::createFuseEpiloguePass() { return std::make_unique<FuseEpiloguePass>(); }
void aiednn::registerFuseEpiloguePass() {
    PassRegistration<FuseEpiloguePass>();
    PassRegistration<FoldRequantPass>();
    PassRegistration<FoldNormQuantPass>();
    PassRegistration<FoldHeadPermutePass>();
    PassRegistration<QuantBeforeMaxPoolPass>();
    PassRegistration<QuantizeResidualPass>();
    PassRegistration<SinkPermutePass>();
    PassRegistration<FuseNormReluPass>();
    PassRegistration<QuantizeThroughViewsPass>();
}
