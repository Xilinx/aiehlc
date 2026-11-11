/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tilinglinalg_pipeline.h"
#include "kernel_body_emitter.h"
#include "routing_ir_api.h"
#include "aiegraphmanager.h"
#include "lower/AiegraphLowerDriver.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace py = pybind11;

/// Optional per-tensor shim DMA addressing (mirrors DmaAddressing).
/// Tuple layout: (dims, iter_step, iter_wrap, ddrShape, mode)
///   dims:     list of (stride, wrap) pairs — the multi-dim BD descriptor
///   iter_step/iter_wrap: BD iteration dim (im2col OH loop)
///   ddrShape: actual DDR buffer shape (e.g. [H, W, C] for im2col)
///   mode:     0 = flat / im2col-by-dims, 1 = spatial_halo
using DmaSpec = std::tuple<std::vector<std::pair<int, int>>, int, int, std::vector<int64_t>, int>;

static bool run_aie_pipeline(int meshRows, int meshCols,
                             const std::vector<std::tuple<std::vector<int64_t>, int, bool>> &tensorSpecs,
                             const std::string &outputDir, const std::string &userKernelBody,
                             const std::string &userKernelFuncName,
                             const std::vector<std::tuple<int, std::string, std::string>> &splitSpecs = {},
                             const std::vector<DmaSpec> &dmaSpecs = {}) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    std::vector<TensorParam> tensors;
    for (size_t i = 0; i < tensorSpecs.size(); ++i) {
        auto &[shape, bw, isIn] = tensorSpecs[i];
        TensorParam tp{shape, bw, isIn, {}};
        // Attach optional shim DMA addressing (im2col conv etc.) if provided.
        if (i < dmaSpecs.size()) {
            auto &[dims, iterStep, iterWrap, ddrShape, mode] = dmaSpecs[i];
            if (!dims.empty() || mode != 0) {
                DmaAddressing dma;
                dma.dims = dims;
                dma.iter_step = iterStep;
                dma.iter_wrap = iterWrap;
                dma.ddrShape = ddrShape;
                dma.mode = mode;
                tp.shimDma = std::move(dma);
            }
        }
        tensors.push_back(std::move(tp));
    }

    // Build SplitModel from optional split specs
    SplitModel splitModel;
    if (!splitSpecs.empty()) {
        for (auto &[dim, axis, replicate] : splitSpecs)
            splitModel.tensorSplits.push_back({dim, axis, replicate});
    } else {
        splitModel = SplitModel::gemm();
    }

    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, meshRows, meshCols, tensors, splitModel);
    return TilingLinalgPipeline::runPipeline(ctx, module, outputDir, userKernelBody, userKernelFuncName);
}

/// Multi-kernel variant of run_aie_pipeline. Forwards the already-present
/// runPipeline multi-kernel args (hostFuncSuffix, appendMode, numHostDdrArgs)
/// so per-layer host functions can be appended into ONE host.cc. Emits
/// host_canonicalized_<suffix> into outputDir/host.cc; appendMode appends.
/// Returns the number of DDR pointer args on the generated host function.
static int orchestrate_conv_layer(int meshRows, int meshCols,
                                  const std::vector<std::tuple<std::vector<int64_t>, int, bool>> &tensorSpecs,
                                  const std::string &outputDir, const std::string &userKernelBody,
                                  const std::string &userKernelFuncName, const std::string &hostFuncSuffix,
                                  bool appendMode) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    std::vector<TensorParam> tensors;
    for (auto &[shape, bw, isIn] : tensorSpecs)
        tensors.push_back({shape, bw, isIn});
    SplitModel splitModel = SplitModel::gemm();
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, meshRows, meshCols, tensors, splitModel);
    unsigned numHostDdrArgs = 0;
    bool ok = TilingLinalgPipeline::runPipeline(ctx, module, outputDir, userKernelBody, userKernelFuncName,
                                                /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", /*tensors=*/{},
                                                /*maxPingPongBytes=*/4096, /*aieGen=*/"Gen2", hostFuncSuffix,
                                                appendMode, &numHostDdrArgs);
    if (!ok)
        throw std::runtime_error("orchestrate_conv_layer: runPipeline failed");
    return (int)numHostDdrArgs;
}

/// Build a C kernel body string from a list of KernelOp dicts via MLIR EmitC.
///
/// Python passes a list[dict] where each dict has an "op" key plus op-specific
/// int/str fields. This function converts them to C++ KernelOp structs, feeds
/// them to KernelBodyEmitter, which builds EmitC VerbatimOps and calls
/// translateToCpp to produce the final C string.
static std::string build_kernel_body(
    const std::string &kernelName,
    const std::string &elementType,
    int numInputWindows,
    int numOutputWindows,
    const py::list &kernelOpsList)
{
    // Convert Python list[dict] -> vector<KernelOp>
    std::vector<KernelOp> ops;
    for (auto item : kernelOpsList) {
        auto d = item.cast<py::dict>();
        KernelOp op;
        op.op = d["op"].cast<std::string>();
        for (auto &kv : d) {
            std::string key = kv.first.cast<std::string>();
            if (key == "op") continue;
            if (py::isinstance<py::int_>(kv.second))
                op.intArgs[key] = kv.second.cast<int64_t>();
            else if (py::isinstance<py::str>(kv.second))
                op.strArgs[key] = kv.second.cast<std::string>();
        }
        ops.push_back(std::move(op));
    }

    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();

    KernelBodyEmitter emitter(ctx, kernelName, elementType,
                              numInputWindows, numOutputWindows);
    return emitter.emit(ops);
}

// ── aiegraph high-level dialect: build + verify + lower ────────────────────
//
// build_aiegraph_module(ops) -> str
//   Construct one aiegraph.func from a list of op dicts (op kind + geometry +
//   per-op quant params), verify it in C++, and return the textual IR. Each
//   dict carries only ints/strings — dataflow is inferred from list order
//   (result of op i feeds op i+1's first tensor operand). The residual op names
//   its skip source via the "skip_index" field (index of the earlier op whose
//   result is the skip path); -1/absent means "use the immediately-preceding
//   result" (degenerate).
//
// lower_aiegraph(mlir_text) -> list[dict]
//   Parse + verify textual aiegraph IR, walk the func in program order, and
//   return one launch descriptor per op: {op, func_name, index, tensor_specs,
//   weights}. tensor_specs is a list of (shape, bits, is_input) tuples ready to
//   pass straight to run_aie_pipeline.

using AiegraphOpDict = py::dict;

// Fetch an int field with a default.
static int64_t dGetI(const py::dict &d, const char *k, int64_t dflt) {
    if (d.contains(k))
        return d[k].cast<int64_t>();
    return dflt;
}
static std::string dGetS(const py::dict &d, const char *k, const std::string &dflt) {
    if (d.contains(k))
        return d[k].cast<std::string>();
    return dflt;
}

static std::string build_aiegraph_module(const py::list &opsList, const std::string &funcName) {
    mlir::MLIRContext ctx;
    aiegraphmanager::loaddialect(&ctx);

    mlir::OpBuilder builder(&ctx);
    auto loc = builder.getUnknownLoc();
    mlir::ModuleOp m = mlir::ModuleOp::create(loc);

    auto functype = builder.getFunctionType({}, {});
    aiegraph::FuncOp fn =
        builder.create<aiegraph::FuncOp>(loc, builder.getStringAttr(funcName), mlir::TypeAttr::get(functype));
    m.push_back(fn);
    auto &block = fn.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    auto i8 = builder.getIntegerType(8);
    auto i8Tensor = [&](int64_t n) { return mlir::RankedTensorType::get({n}, i8); };

    // Track per-op result Values so dataflow edges can reference any earlier op's
    // output by index. Index i in `results` is the SSA result of the i-th op.
    std::vector<mlir::Value> results;

    // Resolve an explicit producer index to an SSA Value; a negative/out-of-range
    // index means "network input", materialized as a fresh block argument sized to
    // `n` int8 elements.
    auto resolveInput = [&](int64_t producerIdx, int64_t n) -> mlir::Value {
        if (producerIdx >= 0 && (size_t)producerIdx < results.size())
            return results[producerIdx];
        return block.addArgument(i8Tensor(n), loc);
    };

    for (size_t i = 0; i < opsList.size(); ++i) {
        py::dict d = opsList[i].cast<py::dict>();
        std::string op = d["op"].cast<std::string>();
        int64_t mainIdx = dGetI(d, "main_index", -1);

        if (op == "conv_bn_relu" || op == "conv_bn") {
            int64_t H = dGetI(d, "H", 0), W = dGetI(d, "W", 0), Cin = dGetI(d, "Cin", 0), Cout = dGetI(d, "Cout", 0),
                    K = dGetI(d, "K", 0), stride = dGetI(d, "stride", 0);
            int64_t inN = Cin * H * W;
            int64_t out_h = stride ? H / stride : 0, out_w = stride ? W / stride : 0;
            int64_t outN = Cout * out_h * out_w;
            mlir::Value inVal = resolveInput(mainIdx, inN);

            double inScale = d.contains("in_scale") ? d["in_scale"].cast<double>() : 0.0;
            double outScale = d.contains("out_scale") ? d["out_scale"].cast<double>() : 0.0;
            auto quant = aiegraph::quantparamAttr::get(&ctx, builder.getF64FloatAttr(inScale), dGetI(d, "in_zp", 0),
                                                       builder.getF64FloatAttr(outScale), dGetI(d, "out_zp", 0),
                                                       dGetI(d, "bn_scale", 0), dGetI(d, "bn_bias", 0));
            mlir::FlatSymbolRefAttr wsym;
            std::string ws = dGetS(d, "weights", "");
            if (!ws.empty())
                wsym = mlir::FlatSymbolRefAttr::get(&ctx, ws);

            mlir::Value res;
            if (op == "conv_bn_relu") {
                auto o = builder.create<aiegraph::ConvBnReluOp>(loc, i8Tensor(outN), inVal, (uint64_t)H, (uint64_t)W,
                                                                (uint64_t)Cin, (uint64_t)Cout, (uint64_t)K,
                                                                (uint64_t)stride, quant, wsym);
                res = o.getResult();
            } else {
                auto o = builder.create<aiegraph::ConvBnOp>(loc, i8Tensor(outN), inVal, (uint64_t)H, (uint64_t)W,
                                                            (uint64_t)Cin, (uint64_t)Cout, (uint64_t)K,
                                                            (uint64_t)stride, quant, wsym);
                res = o.getResult();
            }
            results.push_back(res);
        } else if (op == "residual_add_relu") {
            int64_t n = dGetI(d, "length", 0);
            mlir::Value inVal = resolveInput(mainIdx, n);
            mlir::Value skipVal = resolveInput(dGetI(d, "skip_index", -1), n);
            auto o = builder.create<aiegraph::ResidualAddReluOp>(loc, i8Tensor(n), inVal, skipVal, (uint64_t)n);
            results.push_back(o.getResult());
        } else if (op == "avgpool_fc") {
            int64_t sh = dGetI(d, "spatial_h", 0), sw = dGetI(d, "spatial_w", 0), ch = dGetI(d, "channels", 0),
                    nc = dGetI(d, "num_classes", 0);
            int64_t inN = ch * sh * sw;
            mlir::Value inVal = resolveInput(mainIdx, inN);
            mlir::FlatSymbolRefAttr wsym;
            std::string ws = dGetS(d, "weights", "");
            if (!ws.empty())
                wsym = mlir::FlatSymbolRefAttr::get(&ctx, ws);
            auto o = builder.create<aiegraph::AvgPoolFcOp>(loc, i8Tensor(nc), inVal, (uint64_t)sh, (uint64_t)sw,
                                                           (uint64_t)ch, (uint64_t)nc, wsym);
            results.push_back(o.getResult());
        } else {
            throw std::runtime_error("unknown aiegraph op: " + op);
        }
    }

    builder.create<aiegraph::YieldOp>(loc);

    if (mlir::failed(mlir::verify(m)))
        throw std::runtime_error("aiegraph module failed verification");

    std::string out;
    llvm::raw_string_ostream os(out);
    m.print(os);
    os.flush();
    return out;
}

static py::list lower_aiegraph(const std::string &mlirText) {
    mlir::MLIRContext ctx;
    bool ok = false;
    auto launches = aiegraph::AiegraphLowerDriver::lowerFromText(ctx, mlirText, ok);
    if (!ok)
        throw std::runtime_error("aiegraph lowering failed (parse/verify)");

    py::list result;
    for (auto &L : launches) {
        py::dict d;
        d["op"] = L.op;
        d["func_name"] = L.funcName;
        d["index"] = L.index;
        d["weights"] = L.weightsSym;
        py::list specs;
        for (auto &ts : L.tensorSpecs) {
            py::list shape;
            for (auto s : ts.shape)
                shape.append(s);
            specs.append(py::make_tuple(shape, ts.elementBitWidth, ts.isInput));
        }
        d["tensor_specs"] = specs;
        result.append(d);
    }
    return result;
}

PYBIND11_MODULE(_aiebackend, m) {
    bindRoutingIrApi(m);
    m.doc() = "AIE backend: pybind11 bindings to TilingLinalgPipeline (shared by the Triton and TVM frontends)";
    m.def("run_aie_pipeline", &run_aie_pipeline, py::arg("mesh_rows"), py::arg("mesh_cols"), py::arg("tensor_specs"),
          py::arg("output_dir"), py::arg("user_kernel_body") = "", py::arg("user_kernel_func_name") = "",
          py::arg("split_specs") = std::vector<std::tuple<int, std::string, std::string>>{},
          py::arg("dma_specs") = std::vector<DmaSpec>{},
          "Run TilingLinalgPipeline: buildRoutingIR + runPipeline.\n"
          "split_specs: optional list of (splitDim, hwAxisOwner, replicateOn) tuples, one per tensor.\n"
          "dma_specs: optional list of (dims, iter_step, iter_wrap, ddr_shape, mode) tuples, one per\n"
          "  tensor, giving non-flat shim DMA addressing (im2col conv). dims is a list of\n"
          "  (stride, wrap) pairs. Empty dims + mode 0 = flat (default).");
    m.def("orchestrate_conv_layer", &orchestrate_conv_layer, py::arg("mesh_rows"), py::arg("mesh_cols"),
          py::arg("tensor_specs"), py::arg("output_dir"), py::arg("user_kernel_body"), py::arg("user_kernel_func_name"),
          py::arg("host_func_suffix"), py::arg("append_mode"),
          "Multi-kernel variant of run_aie_pipeline: emits host_canonicalized_<suffix>\n"
          "into output_dir/host.cc (append_mode appends). Returns numHostDdrArgs.");
    m.def("build_kernel_body", &build_kernel_body,
          py::arg("kernel_name"),
          py::arg("element_type"),
          py::arg("num_input_windows"),
          py::arg("num_output_windows"),
          py::arg("kernel_ops"),
          "Build C kernel body from KernelOp list via MLIR EmitC");
    m.def("build_aiegraph_module", &build_aiegraph_module, py::arg("ops"), py::arg("func_name") = "graph",
          "Build + verify one aiegraph.func from a list of op dicts (op kind +\n"
          "geometry + per-op quant params); returns the textual aiegraph IR.\n"
          "Dataflow is inferred from list order; residual skip edges use the\n"
          "optional 'skip_index' field. Raises on verification failure.");
    m.def("lower_aiegraph", &lower_aiegraph, py::arg("mlir_text"),
          "Parse + verify textual aiegraph IR and return one launch descriptor\n"
          "per op: {op, func_name, index, tensor_specs, weights}. tensor_specs\n"
          "is a list of (shape, bits, is_input) tuples for run_aie_pipeline.");
}
