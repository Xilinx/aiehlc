/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// --aiednn-plan-memory: one arena for every activation of a func.
//
// * a value is live from its defining op to its last use (block order); a view
//   aliases its source's buffer, so a buffer lives until its last view's last use;
//   the func result lives to the end;
// * in place: eltwise / prelu / binary write into an operand buffer of the same type
//   that dies at the op (the runtime's loops read element i before writing it);
// * the rest is greedy first-fit in definition order, 64-byte aligned.
//
// Result: `aiednn.offset` (byte offset of the result's buffer) on every op that
// defines an activation buffer, `aiednn.arena_bytes` on the func, and
// `aiednn.naive_bytes` (sum of all buffers, no reuse) for comparison.
#include "AiednnDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <algorithm>
#include <map>

using namespace mlir;

namespace {

int64_t bytesOf(Type t) {
    auto rt = llvm::cast<RankedTensorType>(t);
    int64_t n = rt.getNumElements() * std::max<int64_t>(1, rt.getElementTypeBitWidth() / 8);
    return (n + 63) / 64 * 64;
}

struct Buffer {
    Value root;
    int64_t size = 0, start = 0, end = 0, offset = -1;
    Buffer *inplaceOf = nullptr;
};

bool canRunInPlace(Operation *op) { return llvm::isa<aiednn::EltwiseOp, aiednn::PreluOp, aiednn::BinaryOp>(op); }

struct PlanMemoryPass : public PassWrapper<PlanMemoryPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanMemoryPass)
    StringRef getArgument() const final { return "aiednn-plan-memory"; }
    StringRef getDescription() const final {
        return "assign every activation a byte offset in one arena (liveness reuse + in-place eltwise)";
    }

    void runOnOperation() override {
        func::FuncOp fn = getOperation();
        Block &body = fn.getBody().front();
        llvm::DenseMap<Value, Value> rootOf; // view result -> buffer root
        llvm::DenseMap<Operation *, int64_t> index;
        std::map<int64_t, Buffer> bufs; // keyed by definition index (ordered)
        llvm::DenseMap<Value, int64_t> bufKey;
        int64_t i = 0;
        auto root = [&](Value v) {
            auto it = rootOf.find(v);
            return it == rootOf.end() ? v : it->second;
        };
        for (Operation &op : body) {
            index[&op] = i;
            if (auto v = llvm::dyn_cast<aiednn::ViewOp>(op))
                rootOf[v.getResult()] = root(v.getSrc());
            else if (!llvm::isa<aiednn::ConstOp, func::ReturnOp>(op))
                for (Value r : op.getResults()) {
                    bufKey[r] = i;
                    bufs[i] = Buffer{r, bytesOf(r.getType()), i, i};
                }
            // operands extend the live range of their buffers
            for (Value o : op.getOperands()) {
                auto it = bufKey.find(root(o));
                if (it != bufKey.end())
                    bufs[it->second].end = std::max(bufs[it->second].end, i);
            }
            if (llvm::isa<func::ReturnOp>(op))
                for (Value o : op.getOperands()) {
                    auto it = bufKey.find(root(o));
                    if (it != bufKey.end())
                        bufs[it->second].end = INT64_MAX; // the graph output outlives everything
                }
            i++;
        }
        // in place: the result reuses an operand buffer that dies here
        for (auto &[k, b] : bufs) {
            Operation *op = b.root.getDefiningOp();
            if (!op || !canRunInPlace(op))
                continue;
            for (Value o : op->getOperands()) {
                auto it = bufKey.find(root(o));
                if (it == bufKey.end() || o.getType() != b.root.getType())
                    continue;
                Buffer &src = bufs[it->second];
                if (src.end == k && !src.inplaceOf && src.root == o) { // dies here, not through a view
                    b.inplaceOf = &src;
                    src.end = b.end; // one buffer, two names
                    break;
                }
            }
        }
        // greedy first-fit in definition order
        std::vector<Buffer *> live;
        int64_t top = 0, naive = 0;
        for (auto &[k, b] : bufs) {
            naive += b.size;
            if (b.inplaceOf)
                continue;
            live.erase(std::remove_if(live.begin(), live.end(), [&](Buffer *x) { return x->end < b.start; }),
                       live.end());
            std::sort(live.begin(), live.end(), [](Buffer *a, Buffer *c) { return a->offset < c->offset; });
            int64_t off = 0;
            for (Buffer *x : live) {
                if (off + b.size <= x->offset)
                    break;
                off = std::max(off, x->offset + x->size);
            }
            b.offset = off;
            live.push_back(&b);
            top = std::max(top, off + b.size);
        }
        Builder bld(fn.getContext());
        for (auto &[k, b] : bufs) {
            Buffer *base = &b;
            while (base->inplaceOf)
                base = base->inplaceOf;
            b.root.getDefiningOp()->setAttr("aiednn.offset", bld.getI64IntegerAttr(base->offset));
            if (b.inplaceOf)
                b.root.getDefiningOp()->setAttr("aiednn.inplace", bld.getUnitAttr());
        }
        fn->setAttr("aiednn.arena_bytes", bld.getI64IntegerAttr(top));
        fn->setAttr("aiednn.naive_bytes", bld.getI64IntegerAttr(naive));
    }
};

} // namespace

void aiednn::registerPlanMemoryPass() { PassRegistration<PlanMemoryPass>(); }
