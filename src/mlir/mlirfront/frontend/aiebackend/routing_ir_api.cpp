/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Routing IR API for agents and tuners (doc/design/routing_ir_reentry.md).
 *
 * The routing IR handed to TilingLinalgPipeline::runPipeline (dumped as
 * ir/dfschedule/0_initial.mlir, read back with AIEHLC_ROUTING_IR_IN) carries
 * the whole tiling decision. These three entry points let Python inspect it,
 * edit it in memory and check an edit, without rebuilding the compiler:
 *
 *   routing_ir_parse(text)          -> dict view (mesh, tensors, partitions with
 *                                      typed per-dim / per-level tiling, func and
 *                                      module attributes)
 *   routing_ir_edit(text, edits)    -> new IR text (typed level edits, or any
 *                                      attribute replaced by MLIR attribute text)
 *   routing_ir_check(text, aie_gen) -> {ok, diagnostics, log, artifacts}: verify,
 *                                      then the whole pipeline in a scratch dir
 *
 * Python wrapper: routing_ir.py (same directory).
 ******************************************************************************/
#include "routing_ir_api.h"

#include "routingmanager.h"
#include "tilinglinalg_pipeline.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace py = pybind11;

namespace {

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &ctx, const std::string &text,
                                              std::vector<std::string> *diags = nullptr) {
    mlir::ScopedDiagnosticHandler h(&ctx, [&](mlir::Diagnostic &d) {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << d.getLocation() << ": " << d.str();
        if (diags)
            diags->push_back(os.str());
        return mlir::success();
    });
    auto m = mlir::parseSourceString<mlir::ModuleOp>(text, &ctx);
    if (m && mlir::failed(mlir::verify(*m)))
        return nullptr;
    return m;
}

std::string toText(mlir::Operation *op) {
    std::string s;
    llvm::raw_string_ostream os(s);
    op->print(os);
    return os.str();
}

std::string attrText(mlir::Attribute a) {
    std::string s;
    llvm::raw_string_ostream os(s);
    a.print(os);
    return os.str();
}

// Plain attribute value for Python: int, str, list of ints, else its MLIR text.
py::object plain(mlir::Attribute a) {
    if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(a))
        return py::int_(i.getValue().getSExtValue());
    if (auto s = mlir::dyn_cast<mlir::StringAttr>(a))
        return py::str(s.getValue().str());
    if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(a)) {
        py::list l;
        for (auto e : arr) {
            auto i = mlir::dyn_cast<mlir::IntegerAttr>(e);
            if (!i)
                return py::str(attrText(a));
            l.append(i.getValue().getSExtValue());
        }
        return l;
    }
    return py::str(attrText(a));
}

py::dict levelDict(routing::LevelAttr l) {
    py::dict d;
    d["base"] = l.getBase();
    d["total"] = l.getTotal();
    d["slice"] = l.getSlice();
    d["step"] = l.getStep();
    d["rounds"] = l.getRounds();
    return d;
}

// Func argument a tensor value comes from (to_tensor / createscheduletensor chain).
int tensorArg(mlir::Value v) {
    for (int guard = 0; guard < 8 && v; ++guard) {
        if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v))
            return (int)ba.getArgNumber();
        mlir::Operation *def = v.getDefiningOp();
        if (!def || def->getNumOperands() == 0)
            return -1;
        v = def->getOperand(0);
    }
    return -1;
}

std::vector<routing::partitiontensor> partitionsOf(mlir::ModuleOp m) {
    std::vector<routing::partitiontensor> out;
    m.walk([&](routing::partitiontensor p) { out.push_back(p); });
    return out;
}

std::vector<routing::createscheduletensor> tensorsOf(mlir::ModuleOp m) {
    std::vector<routing::createscheduletensor> out;
    m.walk([&](routing::createscheduletensor t) { out.push_back(t); });
    return out;
}

// Rebuild `l` with the level at `depth` (0 = outer) changed by `set`.
routing::LevelAttr editLevel(mlir::MLIRContext *ctx, routing::LevelAttr l, int depth, const py::dict &set) {
    if (!l)
        throw std::runtime_error("routing_ir_edit: no tiling level at that depth");
    routing::LevelAttr inner = l.getSliceTiling();
    int64_t base = l.getBase(), total = l.getTotal(), slice = l.getSlice(), step = l.getStep(), rounds = l.getRounds();
    if (depth > 0) {
        inner = editLevel(ctx, inner, depth - 1, set);
    } else {
        for (auto kv : set) {
            std::string k = py::str(kv.first);
            int64_t v = kv.second.cast<int64_t>();
            if (k == "base")
                base = v;
            else if (k == "total")
                total = v;
            else if (k == "slice")
                slice = v;
            else if (k == "step")
                step = v;
            else if (k == "rounds")
                rounds = v;
            else
                throw std::runtime_error("routing_ir_edit: unknown level field '" + k +
                                         "' (base, total, slice, step, rounds)");
        }
    }
    return routing::LevelAttr::get(ctx, base, total, slice, step, rounds, inner);
}

mlir::Attribute parseAttr(mlir::MLIRContext &ctx, const std::string &text) {
    mlir::Attribute a = mlir::parseAttribute(text, &ctx);
    if (!a)
        throw std::runtime_error("routing_ir_edit: cannot parse attribute '" + text + "'");
    return a;
}

routing::partitiontensor findPartition(mlir::ModuleOp m, const py::dict &e) {
    auto parts = partitionsOf(m);
    if (e.contains("partition")) {
        size_t i = e["partition"].cast<size_t>();
        if (i >= parts.size())
            throw std::runtime_error("routing_ir_edit: no partition #" + std::to_string(i));
        return parts[i];
    }
    int arg = e["tensor_arg"].cast<int>();
    for (auto p : parts)
        if (tensorArg(p.getTensor()) == arg)
            return p;
    throw std::runtime_error("routing_ir_edit: no partitiontensor for tensor arg " + std::to_string(arg));
}

void applyEdit(mlir::MLIRContext &ctx, mlir::ModuleOp m, const py::dict &e) {
    std::string target = e["target"].cast<std::string>();
    if (target == "module") {
        m->setAttr(e["attr"].cast<std::string>(), parseAttr(ctx, e["value"].cast<std::string>()));
    } else if (target == "func") {
        auto f = *m.getOps<mlir::func::FuncOp>().begin();
        f->setAttr(e["attr"].cast<std::string>(), parseAttr(ctx, e["value"].cast<std::string>()));
    } else if (target == "tensor") {
        int arg = e["arg"].cast<int>();
        for (auto t : tensorsOf(m))
            if (tensorArg(t.getInitTensor()) == arg) {
                t->setAttr(e["attr"].cast<std::string>(), parseAttr(ctx, e["value"].cast<std::string>()));
                return;
            }
        throw std::runtime_error("routing_ir_edit: no routingcreatescheduletensor for arg " + std::to_string(arg));
    } else if (target == "partition") {
        findPartition(m, e)->setAttr(e["attr"].cast<std::string>(), parseAttr(ctx, e["value"].cast<std::string>()));
    } else if (target == "level") {
        auto p = findPartition(m, e);
        routing::TilingAttr t = p.getTilingAttr();
        if (!t)
            throw std::runtime_error("routing_ir_edit: that partitiontensor has no #routing.tiling");
        size_t dim = e["dim"].cast<size_t>();
        llvm::SmallVector<routing::DimAttr> dims(t.getDims().begin(), t.getDims().end());
        if (dim >= dims.size())
            throw std::runtime_error("routing_ir_edit: tiling has no dim " + std::to_string(dim));
        int depth = e.contains("depth") ? e["depth"].cast<int>() : 0;
        auto outer = editLevel(&ctx, dims[dim].getOuter(), depth, e["set"].cast<py::dict>());
        int64_t axis = e.contains("axis") ? e["axis"].cast<int64_t>() : dims[dim].getAxis();
        dims[dim] = routing::DimAttr::get(&ctx, outer, axis);
        p.setTilingAttr(routing::TilingAttr::get(&ctx, dims));
    } else {
        throw std::runtime_error("routing_ir_edit: unknown target '" + target +
                                 "' (module, func, tensor, partition, level)");
    }
}

// Run `fn` with fds 1 and 2 captured into one string: the pipeline reports
// through llvm::outs/errs, std::cout and printf, which would otherwise flood
// the caller (an agent reads `log` only when it needs to).
template <class F> std::string captureOutput(F fn) {
    auto flushAll = [] {
        std::fflush(stdout);
        std::fflush(stderr);
        std::cout.flush();
        std::cerr.flush();
        llvm::outs().flush();
        llvm::errs().flush();
    };
    flushAll();
    const char *tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/routing_ir_logXXXXXX";
    std::vector<char> path(tmpl.begin(), tmpl.end());
    path.push_back('\0');
    int fd = mkstemp(path.data());
    int saved1 = dup(1), saved2 = dup(2);
    if (fd >= 0) {
        dup2(fd, 1);
        dup2(fd, 2);
    }
    fn();
    flushAll();
    if (fd >= 0) {
        dup2(saved1, 1);
        dup2(saved2, 2);
        close(fd);
    }
    close(saved1);
    close(saved2);
    std::ifstream in(path.data());
    std::stringstream ss;
    ss << in.rdbuf();
    std::remove(path.data());
    return ss.str();
}

} // namespace

py::dict routing_ir_parse(const std::string &text) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    std::vector<std::string> diags;
    auto m = parseModule(ctx, text, &diags);
    if (!m) {
        std::string msg = "routing_ir_parse: IR does not parse / verify";
        for (auto &d : diags)
            msg += "\n  " + d;
        throw std::runtime_error(msg);
    }
    py::dict out;
    py::dict mod;
    for (auto na : (*m)->getAttrs())
        mod[py::str(na.getName().str())] = plain(na.getValue());
    out["module"] = mod;
    py::dict fn;
    for (auto f : m->getOps<mlir::func::FuncOp>()) {
        fn["name"] = f.getName().str();
        py::list args;
        for (auto t : f.getArgumentTypes())
            args.append(attrText(mlir::TypeAttr::get(t)));
        fn["args"] = args;
        py::dict fattrs;
        for (auto na : f->getAttrs())
            if (na.getName() != "sym_name" && na.getName() != "function_type")
                fattrs[py::str(na.getName().str())] = plain(na.getValue());
        fn["attrs"] = fattrs;
        break;
    }
    out["func"] = fn;
    m->walk([&](routing::createhwmesh mesh) {
        py::dict d;
        d["rows"] = mesh.getRow();
        d["cols"] = mesh.getCol();
        if (auto v = mesh.getStartCol())
            d["start_col"] = *v;
        if (auto v = mesh.getEndCol())
            d["end_col"] = *v;
        if (auto v = mesh.getStartRow())
            d["start_row"] = *v;
        if (auto v = mesh.getEndRow())
            d["end_row"] = *v;
        out["mesh"] = d;
    });
    // Outputs: tensors whose partition feeds a routinggatherout (the gather).
    std::set<int> outputs;
    m->walk([&](routing::routinggatherout g) {
        if (auto ex = g.getTensordata().getDefiningOp<routing::extract_data>())
            outputs.insert(tensorArg(ex.getTensor()));
    });
    py::list tensors;
    for (auto t : tensorsOf(*m)) {
        py::dict d;
        d["arg"] = tensorArg(t.getInitTensor());
        d["direction"] = outputs.count(tensorArg(t.getInitTensor())) ? "output" : "input";
        d["shape"] = plain(t.getShapeAttr());
        py::dict attrs;
        for (auto na : t->getAttrs())
            if (na.getName() != "shape" && na.getName() != "dim")
                attrs[py::str(na.getName().str())] = plain(na.getValue());
        d["attrs"] = attrs;
        tensors.append(d);
    }
    out["tensors"] = tensors;
    py::list parts;
    int idx = 0;
    for (auto p : partitionsOf(*m)) {
        py::dict d;
        d["index"] = idx++;
        d["tensor_arg"] = tensorArg(p.getTensor());
        d["direction"] = outputs.count(tensorArg(p.getTensor())) ? "output" : "input";
        auto pa = p.getPartition();
        py::dict pd;
        pd["splitnum"] = pa.getSplitnum();
        pd["splitdim"] = pa.getSplitdim();
        pd["hwAxisOwner"] = pa.getHwAxisOwner().str();
        pd["replicateOn"] = pa.getReplicateOn().str();
        pd["singleTileOwner"] = pa.getSingleTileOwner().str();
        d["partition"] = pd;
        py::list dims;
        if (auto t = p.getTilingAttr())
            for (auto dim : t.getDims()) {
                py::dict dd;
                dd["axis"] = dim.getAxis();
                py::list levels;
                for (auto l = dim.getOuter(); l; l = l.getSliceTiling())
                    levels.append(levelDict(l));
                dd["levels"] = levels;
                dims.append(dd);
            }
        d["tiling"] = dims;
        parts.append(d);
    }
    out["partitions"] = parts;
    return out;
}

std::string routing_ir_edit(const std::string &text, const py::list &edits) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    std::vector<std::string> diags;
    auto m = parseModule(ctx, text, &diags);
    if (!m)
        throw std::runtime_error("routing_ir_edit: input IR does not parse / verify");
    for (auto e : edits)
        applyEdit(ctx, *m, e.cast<py::dict>());
    if (mlir::failed(mlir::verify(*m)))
        throw std::runtime_error("routing_ir_edit: the edited IR does not verify");
    return toText(*m);
}

py::dict routing_ir_check(const std::string &text, const std::string &aieGen, const std::string &workDir) {
    py::dict out;
    py::list diags;
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    std::vector<std::string> d;
    auto m = parseModule(ctx, text, &d);
    for (auto &s : d)
        diags.append(s);
    out["diagnostics"] = diags;
    out["verified"] = (bool)m;
    out["ok"] = false;
    out["log"] = "";
    if (!m)
        return out;
    // The pipeline writes ir/ relative to the cwd and its outputs to outputDir.
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd)))
        throw std::runtime_error("routing_ir_check: getcwd failed");
    if (chdir(workDir.c_str()) != 0)
        throw std::runtime_error("routing_ir_check: cannot chdir to " + workDir);
    bool ok = false;
    std::string log = captureOutput([&] {
        ok = TilingLinalgPipeline::runPipeline(ctx, *m, workDir + "/worklocal", "", "", -1, "", {}, 4096, aieGen);
    });
    llvm::outs().flush();
    if (chdir(cwd) != 0)
        throw std::runtime_error("routing_ir_check: cannot chdir back to " + std::string(cwd));
    out["ok"] = ok;
    out["log"] = log;
    out["artifacts"] = workDir + "/worklocal";
    return out;
}

void bindRoutingIrApi(py::module_ &m) {
    m.def("routing_ir_parse", &routing_ir_parse, py::arg("text"),
          "Routing IR text -> dict view: module/func attrs, mesh, tensors (pp_depth, halo, ...), partitions "
          "with typed per-dim / per-level tiling.");
    m.def("routing_ir_edit", &routing_ir_edit, py::arg("text"), py::arg("edits"),
          "Apply edits (dicts: target = module|func|tensor|partition|level) and return the new IR text.");
    m.def("routing_ir_check", &routing_ir_check, py::arg("text"), py::arg("aie_gen") = "Gen5", py::arg("work_dir"),
          "Verify the IR, then run the whole pipeline in work_dir. Returns {ok, verified, diagnostics, log, "
          "artifacts}.");
}
