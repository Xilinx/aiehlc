/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// --aiednn-export-plan="out=<file>": every func.func as a JSON op list, in block
// order, for aiegraph.dnn_mlir.from_plan (which rebuilds the executable graph).
//
//   {"funcs": [{"name": "m", "args": [0], "results": [17],
//               "values": [{"id": 0, "shape": [1,3,224,224], "dtype": "f32"}, ...],
//               "ops": [{"op": "aiednn.conv", "loc": "conv1", "operands": [0, 1, 2],
//                        "segments": [1,1,1,0], "results": [3], "attrs": {...}}]}]}
//
// Floats travel as their IEEE bits ({"f32": <uint32>}), so the round trip is exact.
#include "AiednnDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using mlir::ModuleOp;

namespace {

std::string dtypeOf(Type t) {
    if (t.isF32())
        return "f32";
    if (auto i = llvm::dyn_cast<IntegerType>(t)) {
        if (i.getWidth() == 8)
            return i.isUnsigned() ? "u8" : "s8";
        if (i.getWidth() == 32)
            return "s32";
    }
    std::string s;
    llvm::raw_string_ostream os(s);
    t.print(os);
    return s;
}

llvm::json::Value f32Bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return llvm::json::Object{{"f32", (int64_t)b}};
}

llvm::json::Value attrJson(Attribute a) {
    if (auto b = llvm::dyn_cast<BoolAttr>(a))
        return b.getValue();
    if (auto i = llvm::dyn_cast<IntegerAttr>(a))
        return i.getInt();
    if (auto f = llvm::dyn_cast<FloatAttr>(a))
        return f32Bits((float)f.getValueAsDouble());
    if (auto s = llvm::dyn_cast<StringAttr>(a))
        return s.getValue().str();
    if (auto d = llvm::dyn_cast<DenseI64ArrayAttr>(a))
        return llvm::json::Array(d.asArrayRef());
    if (auto d = llvm::dyn_cast<DenseI32ArrayAttr>(a))
        return llvm::json::Array(d.asArrayRef());
    if (auto d = llvm::dyn_cast<DenseF32ArrayAttr>(a)) {
        llvm::json::Array out;
        for (float f : d.asArrayRef())
            out.push_back(f32Bits(f));
        return out;
    }
    if (auto p = llvm::dyn_cast<aiednn::PostOpAttr>(a))
        return llvm::json::Object{{"kind", p.getKind().getValue().str()},
                                  {"alg", p.getAlg().getValue().str()},
                                  {"alpha", attrJson(p.getAlpha())},
                                  {"beta", attrJson(p.getBeta())}};
    if (auto arr = llvm::dyn_cast<ArrayAttr>(a)) {
        llvm::json::Array out;
        for (Attribute e : arr)
            out.push_back(attrJson(e));
        return out;
    }
    std::string s;
    llvm::raw_string_ostream os(s);
    a.print(os);
    return llvm::json::Object{{"unknown", s}};
}

/* every NameLoc of a (fused) location, in order: the source ops a primitive computes */
llvm::json::Array locNames(Location loc) {
    llvm::json::Array out;
    if (auto n = llvm::dyn_cast<NameLoc>(loc))
        out.push_back(n.getName().str());
    else if (auto f = llvm::dyn_cast<FusedLoc>(loc))
        for (Location l : f.getLocations())
            if (auto n = llvm::dyn_cast<NameLoc>(l))
                out.push_back(n.getName().str());
    return out;
}

std::string locName(Location loc) {
    if (auto n = llvm::dyn_cast<NameLoc>(loc))
        return n.getName().str();
    if (auto f = llvm::dyn_cast<FusedLoc>(loc))
        for (Location l : f.getLocations())
            if (auto n = llvm::dyn_cast<NameLoc>(l))
                return n.getName().str();
    return "";
}

llvm::json::Object funcJson(func::FuncOp fn) {
    llvm::DenseMap<Value, int64_t> ids;
    llvm::json::Array values, args, ops, results;
    auto idOf = [&](Value v) -> int64_t {
        auto it = ids.find(v);
        if (it != ids.end())
            return it->second;
        const int64_t id = (int64_t)ids.size();
        ids[v] = id;
        auto t = llvm::cast<RankedTensorType>(v.getType());
        values.push_back(llvm::json::Object{
            {"id", id}, {"shape", llvm::json::Array(t.getShape())}, {"dtype", dtypeOf(t.getElementType())}});
        return id;
    };
    for (BlockArgument a : fn.getArguments())
        args.push_back(idOf(a));
    for (Operation &op : fn.getBody().front()) {
        if (auto ret = llvm::dyn_cast<func::ReturnOp>(op)) {
            for (Value v : ret.getOperands())
                results.push_back(idOf(v));
            continue;
        }
        llvm::json::Array opnds, res, segs;
        for (Value v : op.getOperands())
            opnds.push_back(idOf(v));
        for (Value v : op.getResults())
            res.push_back(idOf(v));
        llvm::json::Object attrs;
        for (NamedAttribute na : op.getAttrDictionary()) {
            if (na.getName() == "operandSegmentSizes") {
                for (int32_t n : llvm::cast<DenseI32ArrayAttr>(na.getValue()).asArrayRef())
                    segs.push_back(n);
                continue;
            }
            attrs[na.getName().str()] = attrJson(na.getValue());
        }
        llvm::json::Object o{{"op", op.getName().getStringRef().str()},
                             {"loc", locName(op.getLoc())},
                             {"locs", locNames(op.getLoc())},
                             {"operands", std::move(opnds)},
                             {"results", std::move(res)},
                             {"attrs", std::move(attrs)}};
        if (!segs.empty())
            o["segments"] = std::move(segs);
        ops.push_back(std::move(o));
    }
    llvm::json::Object fattrs; // aiednn.* func attributes (arena size, ...)
    for (NamedAttribute na : fn->getDiscardableAttrs())
        if (na.getName().strref().starts_with("aiednn."))
            fattrs[na.getName().str()] = attrJson(na.getValue());
    return llvm::json::Object{{"name", fn.getSymName().str()}, {"attrs", std::move(fattrs)},  {"args", std::move(args)},
                              {"results", std::move(results)}, {"values", std::move(values)}, {"ops", std::move(ops)}};
}

struct ExportPlanPass : public PassWrapper<ExportPlanPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportPlanPass)
    ExportPlanPass() = default;
    ExportPlanPass(const ExportPlanPass &o) : PassWrapper(o) {}
    StringRef getArgument() const final { return "aiednn-export-plan"; }
    StringRef getDescription() const final { return "write every func.func as a JSON op list (aiegraph.dnn_mlir)"; }
    Option<std::string> out{*this, "out", llvm::cl::desc("output JSON file ('-' = stdout)"), llvm::cl::init("-")};

    void runOnOperation() override {
        llvm::json::Array funcs;
        getOperation().walk([&](func::FuncOp fn) { funcs.push_back(funcJson(fn)); });
        llvm::json::Value doc = llvm::json::Object{{"funcs", std::move(funcs)}};
        if (out == "-") {
            llvm::outs() << llvm::formatv("{0:2}", doc) << "\n";
            return;
        }
        std::error_code ec;
        llvm::raw_fd_ostream os(out, ec, llvm::sys::fs::OF_Text);
        if (ec) {
            getOperation().emitError("aiednn-export-plan: cannot write ") << out << ": " << ec.message();
            return signalPassFailure();
        }
        os << llvm::formatv("{0:2}", doc) << "\n";
    }
};

} // namespace

std::unique_ptr<Pass> aiednn::createExportPlanPass() { return std::make_unique<ExportPlanPass>(); }

void aiednn::registerAiednnPasses() {
    PassRegistration<ExportPlanPass>();
    registerFuseEpiloguePass();
    registerPlanMemoryPass();
}
