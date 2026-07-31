/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "tilinglinalg_pipeline.h"

#include "dmaptodmaphop.h"
#include "hw/ResourceManager.h"
#include "kernelconfig.h"
#include "passblueprinttoschedule.h"
#include "passblueprinttoschedulekernel.h"
#include "passdfscheduleprovenancemap.h"
#include "passdfscheduletoapi.h"
#include "passdfscheduletokernelapi.h"
#include "passdmaphopprovenancemap.h"
#include "passdmaphoptodfscheblueprint.h"
#include "passdmaphoptoroutinghw.h"
#include "passroutingprovenancemap.h"
#include "passschedulecanonicalize.h"
#include "passschedulesequentialop.h"
#include "passwaitmerge.h"
#include "routingconstantfold.h"
#include "routingdeadargclean.h"
#include "routinghwlower.h"
#include "routinghwverify.h"
#include "routinglower.h"
#include "routingtodmap.h"
#include "routingunrolling.h"

#include "routingmanager.h"
#include "routinghwmanager.h"
#include "dmapmanager.h"
#include "dmaphopmanager.h"
#include "dfschedulemanager.h"
#include "dfscheblueprintmanager.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <map>

using namespace mlir;

// ---------------------------------------------------------------------------
// Helpers (same as unitest/test.cpp)
// ---------------------------------------------------------------------------

static std::string setupPipelineIRDir(const std::string &subdir) {
    llvm::SmallString<256> cwdPath;
    if (std::error_code EC = llvm::sys::fs::current_path(cwdPath)) {
        llvm::errs() << "Failed to get current directory: " << EC.message() << "\n";
        return "";
    }
    std::string dir = (cwdPath + "/ir/" + subdir).str();
    if (std::error_code EC = llvm::sys::fs::create_directories(dir)) {
        llvm::errs() << "Failed to create IR directory " << dir << ": " << EC.message() << "\n";
        return "";
    }
    std::cout << "IR output directory: " << dir << std::endl;
    return dir;
}

// Resolve IR-sourced builtin calls `<fnName>(<dim>, <portvar>)` in the kernel body
// to integer literals read from the routing.partitiontensor TilingAttr.
//   fnName               -> builtin name, e.g. "aie::get_arg_per_round_size_in_dim"
//                           or "aie::get_arg_total_rounds_in_dim"
//   portVarNames[t]      -> kernel port variable name for tensor index t
//   perTensorData[t][d]  -> resolved value for tensor t, dim d (slice or rounds)
// Unresolvable calls (unknown port, missing dim) are left as-is so the missing
// attribute is visible rather than silently wrong.
static std::string resolveArgDimBuiltin(const std::string &body, const std::string &fnName,
                                        const std::vector<std::string> &portVarNames,
                                        const std::map<int, std::map<int, int64_t>> &perTensorData) {
    const std::string kFn = fnName + "(";
    if (body.find(kFn) == std::string::npos)
        return body;
    // port variable name -> tensor index
    std::map<std::string, int> portToIdx;
    for (int i = 0; i < (int)portVarNames.size(); ++i)
        portToIdx[portVarNames[i]] = i;
    auto trim = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
            s.erase(0, 1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
            s.pop_back();
        return s;
    };
    std::string out = body;
    size_t pos = 0;
    while ((pos = out.find(kFn, pos)) != std::string::npos) {
        size_t argStart = pos + kFn.size();
        size_t argEnd = out.find(")", argStart);
        if (argEnd == std::string::npos)
            break;
        std::string args = out.substr(argStart, argEnd - argStart);
        size_t comma = args.find(',');
        if (comma == std::string::npos) {
            pos = argEnd + 1;
            continue;
        }
        std::string dimStr = trim(args.substr(0, comma));
        std::string portStr = trim(args.substr(comma + 1));
        int dim = -1;
        try {
            dim = std::stoi(dimStr);
        } catch (...) {
            pos = argEnd + 1;
            continue;
        }
        auto pit = portToIdx.find(portStr);
        if (pit == portToIdx.end()) {
            pos = argEnd + 1;
            continue;
        }
        auto tIt = perTensorData.find(pit->second);
        if (tIt == perTensorData.end()) {
            pos = argEnd + 1;
            continue;
        }
        auto dIt = tIt->second.find(dim);
        if (dIt == tIt->second.end()) {
            pos = argEnd + 1;
            continue;
        }
        std::string replacement = std::to_string(dIt->second);
        out.replace(pos, argEnd + 1 - pos, replacement);
        pos += replacement.size();
    }
    return out;
}

static void dumpPipelineIRToFile(mlir::ModuleOp module, const std::string &dir, int stage, const std::string &passName) {
    if (dir.empty())
        return;
    std::string filename = dir + "/" + std::to_string(stage) + "_" + passName + ".mlir";
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Failed to write IR to " << filename << ": " << ec.message() << "\n";
        return;
    }
    module.print(os);
    std::cout << "  IR -> " << filename << std::endl;
}

static bool runPipelineSinglePass(MLIRContext &ctx, mlir::ModuleOp module, std::unique_ptr<mlir::Pass> pass,
                          const std::string &irDir, int &stage, const std::string &passName) {
    mlir::PassManager singlePm(&ctx);
    singlePm.addPass(std::move(pass));
    if (failed(singlePm.run(module))) {
        llvm::errs() << "ERROR: " << passName << " failed!\n";
        return false;
    }
    dumpPipelineIRToFile(module, irDir, stage, passName);
    stage++;
    return true;
}

// ---------------------------------------------------------------------------
// TilingLinalgPipeline implementation
// ---------------------------------------------------------------------------

void TilingLinalgPipeline::registerDialects(mlir::MLIRContext &ctx) {
    // Instantiate dialect managers to trigger registration
    routingmanager mtest;
    routinghwmanager mtesthw;
    dmapmanager mdmaptest;
    dmaphopmanager dmaphoptest;
    dfschedulemanager dfscheduletest;
    dfscheblueprintmanager dfscheblueprinttest;

    mtest.loaddialect(&ctx);
    mtesthw.loaddialect(&ctx);
    mdmaptest.loaddialect(&ctx);
    dmaphoptest.loaddialect(&ctx);
    dfscheduletest.loaddialect(&ctx);
    dfscheblueprinttest.loaddialect(&ctx);

    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
}

mlir::ModuleOp TilingLinalgPipeline::buildRoutingIR(mlir::MLIRContext &ctx, int meshRows, int meshCols,
                                                    const std::vector<TensorParam> &tensors,
                                                    const SplitModel &splitModel, const PartitionDesc &partition,
                                                    const std::string &aieGen) {

    // This is a parameterized version of routingmanager::ops_testNew()
    using namespace routing;

    OpBuilder builder(&ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());

    // Build function type WITH memref arguments — one per tensor.
    // Each memref represents an external DDR pointer provided by the caller.
    SmallVector<Type> argTypes;
    for (const auto &tp : tensors) {
        auto elementType = builder.getIntegerType(tp.elementBitWidth);
        auto memrefType = MemRefType::get(tp.shape, elementType);
        argTypes.push_back(memrefType);
    }
    auto functype = builder.getFunctionType(argTypes, {});
    mlir::func::FuncOp hostFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", functype);

    auto block = hostFunc.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    mlir::IntegerAttr startColAttr, endColAttr, startRowAttr, endRowAttr;
    if (partition.isValid()) {
        startColAttr = builder.getI64IntegerAttr(partition.startCol);
        endColAttr = builder.getI64IntegerAttr(partition.endCol);
        startRowAttr = builder.getI64IntegerAttr(partition.startRow);
        endRowAttr = builder.getI64IntegerAttr(partition.endRow);
    }
    auto mesh = builder.create<createhwmesh>(builder.getUnknownLoc(), meshRows, meshCols, startColAttr, endColAttr,
                                             startRowAttr, endRowAttr);

    // Build all tensor values first
    std::vector<Value> tensorValues;
    std::vector<bool> isInputFlags;
    for (unsigned i = 0; i < tensors.size(); ++i) {
        const auto &tp = tensors[i];

        // Build shape attributes
        SmallVector<Attribute> shape;
        for (int64_t v : tp.shape)
            shape.push_back(builder.getI64IntegerAttr(v));
        ArrayAttr vals = builder.getArrayAttr(shape);
        IntegerAttr dimnum = builder.getI64IntegerAttr(tp.shape.size());

        // Build tensor type
        auto elementType = builder.getIntegerType(tp.elementBitWidth);
        auto tensorType = RankedTensorType::get(tp.shape, elementType);

        // Get the func argument (memref from user's DDR pointer)
        Value memrefArg = block->getArgument(i);

        // Convert memref -> tensor (logical view, zero-copy)
        auto tensorValue = builder.create<bufferization::ToTensorOp>(builder.getUnknownLoc(), tensorType, memrefArg);

        // Feed into createscheduletensor — same as before
        auto tensor = builder.create<createscheduletensor>(builder.getUnknownLoc(), tensorType, tensorValue.getResult(),
                                                           vals, dimnum);
        tensorValues.push_back(tensor);
        isInputFlags.push_back(tp.isInput);
    }

    // Mirror per-tensor spatial-halo descriptors (from TensorParam.shimDma) onto the
    // SplitModel so createroutingfuncBySplitModel can build overlapping partition slices.
    SplitModel effectiveSplit = splitModel;
    for (unsigned i = 0; i < tensors.size() && i < effectiveSplit.tensorSplits.size(); ++i) {
        const auto &sd = tensors[i].shimDma;
        if (sd.mode == 1 && sd.haloSlice > 0) {
            effectiveSplit.tensorSplits[i].haloMode = 1;
            effectiveSplit.tensorSplits[i].haloSlice = sd.haloSlice;
            effectiveSplit.tensorSplits[i].haloStep = sd.haloStep;
            if (sd.splitDim == 0 || sd.splitDim == 1)
                effectiveSplit.tensorSplits[i].splitDim = sd.splitDim;
            // Nested L2 (on-core temporal ROW-split): propagate so
            // createroutingfuncBySplitModel sets the partitiontensor
            // l2_slice/l2_step/l2_rounds attrs.
            effectiveSplit.tensorSplits[i].haloL2Slice = sd.l2Slice;
            effectiveSplit.tensorSplits[i].haloL2Step = sd.l2Step;
            effectiveSplit.tensorSplits[i].haloL2Rounds = sd.l2Rounds;
            // K-contraction accumulate split (independent of the H/row L2 halo):
            // propagate so createroutingfuncBySplitModel sets the HaloAttr k params.
            effectiveSplit.tensorSplits[i].kAccumSlice = sd.kSlice;
            effectiveSplit.tensorSplits[i].kAccumStep = sd.kStep;
            effectiveSplit.tensorSplits[i].kAccumRounds = sd.kRounds;
        }
    }

    // Use SplitModel-driven routing generation
    routingmanager rm;
    rm.createroutingfuncBySplitModel(builder, &ctx, mesh, tensorValues, isInputFlags, meshRows, meshCols,
                                     effectiveSplit);

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    m.push_back(hostFunc);

    // Store per-tensor pp_depth as module attribute so downstream passes can read it.
    // Format: "routing.pp_depth_map" = { tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 1 : i32, ... }
    {
        NamedAttrList ppDepthEntries;
        for (unsigned i = 0; i < splitModel.tensorSplits.size(); ++i) {
            int ppDepth = splitModel.tensorSplits[i].pingPong;
            std::string key = "tensor_" + std::to_string(i);
            ppDepthEntries.append(key, builder.getI32IntegerAttr(ppDepth));
        }
        m->setAttr("routing.pp_depth_map", DictionaryAttr::get(&ctx, ppDepthEntries));
    }

    // Store per-tensor shimDma as module attribute so DmaphopTodfscheblueprintPass
    // can read multi-dim DMA addressing from user-specified DmaTransform.
    // Format: "tensor_N.shim_dma" = { strides = [...], wraps = [...], iter_step = N, iter_wrap = N,
    //                                  ddr_shape = [...], tile_m_alignment = N }
    for (unsigned i = 0; i < tensors.size(); ++i) {
        // Spatial-halo mode (mode == 1) uses flat contiguous shim BDs with overlapping
        // per-tile base offsets — it must NOT emit a multi-dim shim_dma attribute.
        if (tensors[i].shimDma.mode == 1)
            continue;
        if (!tensors[i].shimDma.empty()) {
            NamedAttrList entries;
            SmallVector<int32_t> strides, wraps;
            for (auto &[s, w] : tensors[i].shimDma.dims) {
                strides.push_back(s);
                wraps.push_back(w);
            }
            entries.append("strides", builder.getI32ArrayAttr(strides));
            entries.append("wraps", builder.getI32ArrayAttr(wraps));
            entries.append("iter_step", builder.getI32IntegerAttr(tensors[i].shimDma.iter_step));
            entries.append("iter_wrap", builder.getI32IntegerAttr(tensors[i].shimDma.iter_wrap));
            // Store ddr_shape if available (for tile_m temporal tiling base address computation)
            if (!tensors[i].shimDma.ddrShape.empty()) {
                SmallVector<int64_t> shape(tensors[i].shimDma.ddrShape.begin(), tensors[i].shimDma.ddrShape.end());
                entries.append("ddr_shape", builder.getI64ArrayAttr(shape));
            }
            // Store tile_m_alignment if set (e.g. OW for im2col)
            if (tensors[i].shimDma.tile_m_alignment > 0) {
                entries.append("tile_m_alignment", builder.getI32IntegerAttr(tensors[i].shimDma.tile_m_alignment));
            }
            m->setAttr("tensor_" + std::to_string(i) + ".shim_dma", DictionaryAttr::get(&ctx, entries));
            llvm::outs() << "[TilingLinalg] Stored shim_dma for tensor_" << i << ": " << tensors[i].shimDma.dims.size()
                         << " dims, iter_step=" << tensors[i].shimDma.iter_step
                         << " iter_wrap=" << tensors[i].shimDma.iter_wrap << "\n";
        }
    }

    // Store per-tensor spatial-halo descriptor as module attribute.
    // Format: "tensor_N.halo" = { slice = <haloSlice>, step = <haloStep>, split_dim = <splitDim> }
    // DmaphopTodfscheblueprintPass reads this (or the partitiontensor's step attr) to
    // compute overlapping per-tile DDR base offsets for the flat shim BD.
    for (unsigned i = 0; i < tensors.size(); ++i) {
        const auto &sd = tensors[i].shimDma;
        if (sd.mode == 1 && sd.haloSlice > 0) {
            NamedAttrList entries;
            entries.append("slice", builder.getI32IntegerAttr(sd.haloSlice));
            entries.append("step", builder.getI32IntegerAttr(sd.haloStep));
            entries.append("split_dim", builder.getI32IntegerAttr(sd.splitDim));
            // 2D width-split: carry the per-chunk WIDTH geometry so the shim BD
            // (passdmaphop) and the host round loop (passblueprinttoschedule) can
            // build the 2D narrow-chunk descriptor + (hc,wc) per-round base offset.
            // Emitted only when wRounds > 1 so the height-only path is untouched.
            if (sd.wRounds > 1) {
                entries.append("w_slice", builder.getI32IntegerAttr(sd.wSlice));
                entries.append("w_step", builder.getI32IntegerAttr(sd.wStep));
                entries.append("w_rounds", builder.getI32IntegerAttr(sd.wRounds));
                entries.append("row_pitch", builder.getI32IntegerAttr(sd.rowPitch));
                entries.append("ow_t", builder.getI32IntegerAttr(sd.owT));
            }
            // 2-level (nested) halo: each L1 tile slice is further chunked into
            // l2_rounds on-core temporal rounds (l2_slice rows advancing by l2_step
            // along the split/row dim). Realized downstream via the DMA BD iteration
            // dimension (iter_step=l2_step*row_pitch, iter_wrap=l2_rounds) plus kernel
            // rounds. Emitted only when l2Rounds > 1 so the legacy path is untouched.
            if (sd.l2Rounds > 1) {
                entries.append("l2_slice", builder.getI32IntegerAttr(sd.l2Slice));
                entries.append("l2_step", builder.getI32IntegerAttr(sd.l2Step));
                entries.append("l2_rounds", builder.getI32IntegerAttr(sd.l2Rounds));
                // row_pitch is required by the iteration-dim BD; ensure it is present
                // even when there is no W-split (wRounds <= 1).
                if (sd.wRounds <= 1)
                    entries.append("row_pitch", builder.getI32IntegerAttr(sd.rowPitch));
            }
            // K-contraction accumulate split (independent of the H/row L2 above):
            // the K dim is chunked into k_rounds on-core accumulate rounds of
            // k_slice elements advancing by k_step. Emitted only when kRounds > 1
            // so non-k-accum IR stays byte-identical.
            if (sd.kRounds > 1) {
                entries.append("k_slice", builder.getI32IntegerAttr(sd.kSlice));
                entries.append("k_step", builder.getI32IntegerAttr(sd.kStep));
                entries.append("k_rounds", builder.getI32IntegerAttr(sd.kRounds));
            }
            m->setAttr("tensor_" + std::to_string(i) + ".halo", DictionaryAttr::get(&ctx, entries));
            llvm::outs() << "[TilingLinalg] Stored halo for tensor_" << i << ": slice=" << sd.haloSlice
                         << " step=" << sd.haloStep << " split_dim=" << sd.splitDim;
            if (sd.wRounds > 1)
                llvm::outs() << " [W-SPLIT w_slice=" << sd.wSlice << " w_step=" << sd.wStep
                             << " w_rounds=" << sd.wRounds << " row_pitch=" << sd.rowPitch << " ow_t=" << sd.owT << "]";
            if (sd.l2Rounds > 1)
                llvm::outs() << " [L2-SPLIT l2_slice=" << sd.l2Slice << " l2_step=" << sd.l2Step
                             << " l2_rounds=" << sd.l2Rounds << "]";
            if (sd.kRounds > 1)
                llvm::outs() << " [K-ACCUM k_slice=" << sd.kSlice << " k_step=" << sd.kStep
                             << " k_rounds=" << sd.kRounds << "]";
            llvm::outs() << "\n";
        }
    }

    // Store per-tensor layout_transform as module attribute so DmaphopTodfscheblueprintPass
    // knows whether to apply DmaTransform strides on shim BD (dma_shuffle) or skip them (core_shuffle).
    // Format: "tensor_N.layout_transform" = "dma_shuffle" | "core_shuffle"
    for (unsigned i = 0; i < splitModel.tensorSplits.size(); ++i) {
        const auto &lt = splitModel.tensorSplits[i].layoutTransform;
        if (!lt.empty()) {
            m->setAttr("tensor_" + std::to_string(i) + ".layout_transform", builder.getStringAttr(lt));
            llvm::outs() << "[TilingLinalg] Stored layout_transform for tensor_" << i << ": " << lt << "\n";
        }
    }

    llvm::errs() << m;
    return m;
}

bool TilingLinalgPipeline::runPipeline(mlir::MLIRContext &ctx, mlir::ModuleOp module, const std::string &outputDir,
                                       const std::string &userKernelBody, const std::string &userKernelFuncName,
                                       int runtimeDebugLevel, const std::string &userRewrittenSource,
                                       const std::vector<TensorParam> &tensors, int64_t maxPingPongBytes,
                                       const std::string &aieGen, const std::string &hostFuncSuffix, bool appendMode,
                                       unsigned *numHostDdrArgs, const std::vector<std::string> &portVarNames) {

    // Extract partition bounds from createhwmesh op in the IR (if present)
    int partStartCol = -1, partEndCol = -1, partStartRow = -1, partEndRow = -1;
    module.walk([&](routing::createhwmesh meshOp) {
        if (auto sc = meshOp.getStartCol())
            partStartCol = *sc;
        if (auto ec = meshOp.getEndCol())
            partEndCol = *ec;
        if (auto sr = meshOp.getStartRow())
            partStartRow = *sr;
        if (auto er = meshOp.getEndRow())
            partEndRow = *er;
    });

    // Convert absolute partition columns to 0-based partition-relative columns.
    // XAie_SetupPartitionConfig handles the physical mapping, so the pipeline
    // should generate coordinates relative to the partition origin.
    // Only enable partition mode when both startCol and endCol are specified;
    // otherwise keep relStartCol = -1 so RoutingTopology skips setPartitionBounds.
    int relStartCol = (partStartCol >= 0 && partEndCol >= 0) ? 0 : -1;
    int relEndCol = (partStartCol >= 0 && partEndCol >= 0) ? (partEndCol - partStartCol) : -1;
    RoutingTopology rtopology(aieGen, "", relStartCol, relEndCol, partStartRow, partEndRow);

    std::string irDir = setupPipelineIRDir("dfschedule");
    int stage = 0;

    dumpPipelineIRToFile(module, irDir, stage++, "initial");

    // ── IR-sourced builtin resolution (get_arg_per_round_size_in_dim) ──
    // Read the on-core per-round slice for each (tensor, dim) directly from the
    // routing.partitiontensor TilingAttr *before* RoutingUnrollingLowerPass
    // erases those ops. perTensorTiling[tensorArgIdx][dim] = onCoreRoundSlice
    // (= level.slice_tiling.slice, or level.slice when there is no nested level).
    // The partitiontensor's tensor operand is traced back through
    // createscheduletensor/to_tensor to the func block-argument number, which is
    // the tensor (port) index used by portVarNames.
    std::map<int, std::map<int, int64_t>> perTensorTiling;
    // Parallel map: total on-core rounds to cover the full tensor dim for each
    // (tensor, dim): level.slice_tiling.rounds, or level.rounds when no nested level.
    std::map<int, std::map<int, int64_t>> perTensorRounds;
    {
        // Trace a partitiontensor tensor operand back to its func arg number.
        auto tensorArgIndex = [](mlir::Value v) -> int {
            for (int guard = 0; guard < 8 && v; ++guard) {
                if (auto ba = v.dyn_cast<mlir::BlockArgument>())
                    return (int)ba.getArgNumber();
                mlir::Operation *def = v.getDefiningOp();
                if (!def)
                    return -1;
                if (def->getNumOperands() == 0)
                    return -1;
                v = def->getOperand(0); // createscheduletensor / to_tensor chain
            }
            return -1;
        };
        module.walk([&](routing::partitiontensor pt) {
            auto tiling = pt.getTilingAttr();
            if (!tiling)
                return;
            int tIdx = tensorArgIndex(pt.getTensor());
            if (tIdx < 0)
                return;
            auto dims = tiling.getDims();
            auto &dimMap = perTensorTiling[tIdx];
            auto &roundsMap = perTensorRounds[tIdx];
            for (int d = 0; d < (int)dims.size(); ++d) {
                routing::LevelAttr outer = dims[d].getOuter();
                if (!outer)
                    continue;
                int64_t onCore = outer.getSliceTiling() ? outer.getSliceTiling().getSlice() : outer.getSlice();
                dimMap[d] = onCore;
                int64_t onCoreRounds = outer.getSliceTiling() ? outer.getSliceTiling().getRounds() : outer.getRounds();
                roundsMap[d] = onCoreRounds;
            }
        });
    }

    // Phase 1: routing -> dmap -> dmaphop -> dfscheblueprint
    if (!runPipelineSinglePass(ctx, module, std::make_unique<RoutingUnrollingLowerPass>(), irDir, stage,
                       "RoutingUnrollingLowerPass"))
        return false;
    if (!runPipelineSinglePass(ctx, module, std::make_unique<RoutingToDmapPass>(rtopology), irDir, stage, "RoutingToDmapPass"))
        return false;
    if (!runPipelineSinglePass(ctx, module, std::make_unique<DmapToDmaphopPass>(rtopology), irDir, stage, "DmapToDmaphopPass"))
        return false;

    // Generate provenance map JSON after dmaphop IR is available
    {
        auto provenancePass = std::make_unique<DmaphopProvenanceMapPass>(outputDir, partStartCol, aieGen);
        runPipelineSinglePass(ctx, module, std::move(provenancePass), irDir, stage, "DmaphopProvenanceMapPass");
    }

    // Read the GEMM K-triple from the #routing.tiling op while the partitiontensor op is
    // still alive on `module` (the routing dialect is dropped on the post-conversion
    // hostModule, so the op is gone by the time DfscheduleProvenanceMapPass runs). Values
    // are module-global, so reading here matches hostModule. found=false (fullconnect_auto=0)
    // leaves the K-triple at 0 → DfscheduleProvenanceMapPass falls back to the module attr.
    routing::GemmTilingScalars kTile = routing::readGemmTilingScalars(module);
    int64_t provEffectiveK = kTile.effectiveK;
    int64_t provFullK = kTile.fullK;
    int64_t provKRounds = kTile.kRounds;
    int64_t provTileM = kTile.tileM;
    int64_t provTileN = kTile.tileN;
    int64_t provMRounds = kTile.mRounds;
    int64_t provNRounds = kTile.nRounds;

    // Clone the module at dmaphop stage for the routing path (Phase 5).
    // This preserves the pkt_ids allocated by DmapToDmaphopPass so that
    // routing.cc and host.cc use the same packet IDs.
    mlir::ModuleOp routingDmaphopModule = cast<ModuleOp>(module->clone());

    // Rename @main → @routing in the clone so that routing.cc emits
    // void routing(XAie_DevInst* dev) instead of void main().
    // Prepend a XAie_DevInst* argument at position 0 so that
    // RoutingHWLowerPass patterns can use parentFunc.getArgument(0)
    // instead of calling getOrCreateDeviceInstance().
    // Keep memref func args intact — routing lowering passes need the tensor
    // operands connected through bufferization.to_tensor. RoutingDeadArgPass
    // will strip unused args after lowering.
    for (auto func : routingDmaphopModule.getOps<mlir::func::FuncOp>()) {
        if (func.getName() == "main") {
            func.setName("routing");
            // Insert XAie_DevInst* as arg 0
            auto devInstType = emitc::OpaqueType::get(&ctx, "XAie_DevInst");
            auto devInstPtrType = emitc::PointerType::get(devInstType);
            func.getBody().front().insertArgument(0u, devInstPtrType, func.getLoc());
            // Update the function type to include the new arg
            SmallVector<Type> newArgTypes;
            newArgTypes.push_back(devInstPtrType);
            for (auto t : func.getFunctionType().getInputs())
                newArgTypes.push_back(t);
            func.setFunctionType(FunctionType::get(&ctx, newArgTypes, func.getFunctionType().getResults()));
        }
    }

    if (!runPipelineSinglePass(ctx, module, std::make_unique<DmaphopTodfscheblueprintPass>(), irDir, stage,
                       "DmaphopTodfscheblueprintPass"))
        return false;

    // Clone for host and kernel paths
    mlir::ModuleOp kernelModule = cast<ModuleOp>(module->clone());
    mlir::ModuleOp hostModule = cast<ModuleOp>(module->clone());

    // Initialize ResourceMgr singleton for CoreMemAllocator (BCF/PRX generation)
    {
        auto hwRes = makeResource(aieGen);
        ResourceMgr::init(std::move(hwRes));
    }

    // Early memory check: validate that per-tile buffer requirements fit in tile data memory
    {
        auto hwRes = ResourceMgr::instance()->getrsc();
        uint32_t usableBytes = hwRes->getUsableDataBytes();

        // Estimate per-tile buffer memory from tensor params and ping-pong depth.
        // Each tensor port on a core tile needs ppDepth * bufferSliceBytes.
        // bufferSliceBytes = product(tileShape) * elementBytes where tileShape = shape / meshDim.
        // For a conservative check we use maxPingPongBytes as the per-buffer size.
        uint32_t totalBufferBytes = 0;
        for (const auto &tp : tensors) {
            int ppDepth = 2; // default ping-pong depth
            // Read pp_depth from module attribute if available
            if (auto ppMap = module->getAttrOfType<DictionaryAttr>("routing.pp_depth_map")) {
                unsigned idx = &tp - &tensors[0];
                std::string key = "tensor_" + std::to_string(idx);
                if (auto ppAttr = ppMap.getAs<IntegerAttr>(key))
                    ppDepth = ppAttr.getInt();
            }
            uint32_t bufSize = (maxPingPongBytes > 0) ? maxPingPongBytes : 4096;
            totalBufferBytes += ppDepth * bufSize;
        }

        std::string errMsg;
        if (!hwRes->checkDataMemoryFits(totalBufferBytes, &errMsg)) {
            llvm::errs() << "[TilingLinalg] ERROR: Memory budget exceeded!\n"
                         << "  " << errMsg << "\n"
                         << "  maxPingPongBytes=" << maxPingPongBytes << " numTensors=" << tensors.size() << "\n"
                         << "  Suggestion: reduce maxBufferBytes in SpatialPolicy "
                         << "or reduce ping-pong depth.\n";
            return false;
        }
        std::cout << "[TilingLinalg] Memory check passed: estimated " << totalBufferBytes << " bytes per tile, limit "
                  << usableBytes << " bytes" << std::endl;
    }

    // Phase 2: host path (blueprint -> schedule -> API -> EmitC)
    if (!runPipelineSinglePass(ctx, hostModule,
                               std::make_unique<mlir::BlueprintToSchedulePass>(0.5, maxPingPongBytes, aieGen), irDir,
                               stage, "BlueprintToSchedulePass"))
        return false;
    if (!runPipelineSinglePass(ctx, hostModule, std::make_unique<mlir::ScheduleCanonicalizePass>(), irDir, stage,
                       "ScheduleCanonicalizePass"))
        return false;
    if (!runPipelineSinglePass(ctx, hostModule, std::make_unique<mlir::ScheduleSequentialOpPass>(), irDir, stage,
                               "ScheduleSequentialOpPass"))
        return false;
    if (!runPipelineSinglePass(ctx, hostModule, std::make_unique<mlir::WaitMergePass>(), irDir, stage, "WaitMergePass"))
        return false;

    // Generate low-level dfschedule provenance map after WaitMergePass
    {
        auto dfscheProvenancePass =
            std::make_unique<DfscheduleProvenanceMapPass>(outputDir, partStartCol, aieGen, provEffectiveK, provFullK,
                                                          provKRounds, provTileM, provTileN, provMRounds, provNRounds);
        runPipelineSinglePass(ctx, hostModule, std::move(dfscheProvenancePass), irDir, stage,
                              "DfscheduleProvenanceMapPass");
    }

    if (!runPipelineSinglePass(ctx, hostModule,
                               std::make_unique<mlir::DfscheduleToApiPass>(/*enableDebug=*/true, runtimeDebugLevel),
                               irDir, stage, "DfscheduleToApiPass"))
        return false;
    if (!runPipelineSinglePass(ctx, hostModule, mlir::createCanonicalizerPass(), irDir, stage, "CanonicalizerPass"))
        return false;
    if (!runPipelineSinglePass(ctx, hostModule, std::make_unique<RoutingConstantFoldPass>(aieGen), irDir, stage,
                               "RoutingConstantFoldPass"))
        return false;

    // Phase 3: kernel path (blueprint -> kernel schedule -> kernel API)
    if (!runPipelineSinglePass(ctx, kernelModule,
                               std::make_unique<mlir::BlueprintToScheduleKernelPass>(0.5, maxPingPongBytes), irDir,
                               stage, "BlueprintToScheduleKernelPass"))
        return false;

    // Extract kernel parameter info from KernelModuleOp (before DfscheduleToKernelApiPass lowers it)
    int numInputWindows = 0;
    int numOutputWindows = 0;
    std::string kernelElementType = "int32";
    std::string computeKernelName = "computekernel";
    kernelModule->walk([&](dfschedule::KernelDeclOp declOp) {
        computeKernelName = declOp.getSymName().str();
        auto declAttrs = declOp.getDeclAttrs();
        if (auto inputs = declAttrs.getAs<mlir::ArrayAttr>("inputs"))
            numInputWindows = inputs.size();
        if (auto outputs = declAttrs.getAs<mlir::ArrayAttr>("outputs"))
            numOutputWindows = outputs.size();
    });
    kernelModule->walk([&](dfschedule::KernelConfigDefOp configOp) {
        auto attrs = configOp.getConfigAttrs();
        if (auto typeAttr = attrs.getAs<mlir::TypeAttr>("element_type")) {
            mlir::Type t = typeAttr.getValue();
            if (t.isInteger(32)) kernelElementType = "int32";
            else if (t.isInteger(16)) kernelElementType = "int16";
            else if (t.isInteger(8)) kernelElementType = "int8";
            else if (t.isF32()) kernelElementType = "float";
        }
    });

    // Override compute kernel name with user's __global__ function name if provided.
    // This renames KernelDeclOp in the IR so kernel.cc calls the user's function name
    // and the output file is named accordingly (e.g. matmul.cc instead of computekernel.cc).
    if (!userKernelFuncName.empty() && userKernelFuncName != computeKernelName) {
        std::string newKernelFile = userKernelFuncName + ".cc";
        auto renameKernelInModule = [&](mlir::ModuleOp mod) {
            mod->walk([&](dfschedule::KernelDeclOp declOp) {
                declOp.setSymNameAttr(mlir::StringAttr::get(&ctx, userKernelFuncName));
            });
            mod->walk([&](dfschedule::KernelInvokeOp invokeOp) {
                invokeOp.setKernelRefAttr(SymbolRefAttr::get(&ctx, userKernelFuncName));
            });
            mod->walk([&](dfschedule::KernelConfigDefOp configOp) {
                auto oldAttrs = configOp.getConfigAttrs();
                NamedAttrList newAttrs;
                for (auto &namedAttr : oldAttrs) {
                    if (namedAttr.getName() == "kernel_name")
                        newAttrs.append("kernel_name", StringAttr::get(&ctx, userKernelFuncName));
                    else if (namedAttr.getName() == "kernel_file")
                        newAttrs.append("kernel_file", StringAttr::get(&ctx, newKernelFile));
                    else
                        newAttrs.append(namedAttr);
                }
                configOp.setConfigAttrsAttr(DictionaryAttr::get(&ctx, newAttrs));
            });
        };
        renameKernelInModule(kernelModule);
        renameKernelInModule(hostModule);
        std::cout << "Overriding compute kernel name: " << computeKernelName
                  << " -> " << userKernelFuncName << " (file: " << newKernelFile << ")" << std::endl;
        computeKernelName = userKernelFuncName;
    }

    if (!runPipelineSinglePass(ctx, kernelModule, std::make_unique<mlir::DfscheduleToKernelApiPass>(), irDir, stage,
                       "DfscheduleToKernelApiPass"))
        return false;

    // Create output directory
    if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
        llvm::errs() << "Failed to create directory " << outputDir << ": " << EC.message() << "\n";
        return false;
    }

    // Erase func.func @main from hostModule before EmitC emission.
    // After ScheduleCanonicalizePass, @main only contains dfschedule.launchhost
    // + return; all real logic lives in dfschedule.host @host_canonicalized
    // (now emitc.func @host_canonicalized). The user source provides its own
    // main() via __aie_launch(), so we must not emit a competing main().
    for (auto func : llvm::make_early_inc_range(hostModule.getOps<mlir::func::FuncOp>())) {
        if (func.getName() == "main") {
            func->dropAllUses();
            func.erase();
        }
    }

    // Multi-kernel mode: rename host_canonicalized → host_canonicalized_<suffix>
    // so each kernel gets its own host dispatch function.
    std::string hostFuncName = "host_canonicalized";
    if (!hostFuncSuffix.empty()) {
        hostFuncName = "host_canonicalized_" + hostFuncSuffix;
        for (auto func : hostModule.getOps<emitc::FuncOp>()) {
            if (func.getName() == "host_canonicalized") {
                func.setSymName(hostFuncName);
                std::cout << "Renamed host function: host_canonicalized -> " << hostFuncName << std::endl;
            }
        }
        for (auto func : hostModule.getOps<mlir::func::FuncOp>()) {
            if (func.getName() == "host_canonicalized") {
                func.setName(hostFuncName);
            }
        }
    }

    // Count DDR args on the host function and report back to caller
    {
        unsigned numArgs = 0;
        for (auto func : hostModule.getOps<emitc::FuncOp>()) {
            if (func.getName() == hostFuncName) {
                numArgs = func.getNumArguments();
                break;
            }
        }
        if (numArgs == 0) {
            for (auto func : hostModule.getOps<mlir::func::FuncOp>()) {
                if (func.getName() == hostFuncName) {
                    numArgs = func.getNumArguments();
                    break;
                }
            }
        }
        unsigned ddrArgs = (numArgs > 0) ? numArgs - 1 : 0;
        if (numHostDdrArgs)
            *numHostDdrArgs = ddrArgs;
    }

    // Emit host.cc
    {
        std::string hostPath = outputDir + "/host.cc";
        std::error_code ec;
        auto openFlags = appendMode ? llvm::sys::fs::OF_Append : llvm::sys::fs::OF_None;
        llvm::raw_fd_ostream stream(hostPath, ec, openFlags);
        if (ec) {
            llvm::errs() << "Failed to open " << hostPath << ": " << ec.message() << "\n";
            return false;
        }
        // Emit strong g_runtime_debug_level override if user set #pragma aie_debug_level
        // (only on first write, not in append mode)
        if (!appendMode && runtimeDebugLevel >= 0) {
            stream << "// Override runtime debug level (from #pragma aie_debug_level)\n";
            stream << "int g_runtime_debug_level = " << runtimeDebugLevel << ";\n\n";
        }
        if (appendMode) {
            stream << "\n// ===== Appended host function: " << hostFuncName << " =====\n";
            // In append mode, erase dskernel_receiver from the module to avoid
            // redefinition (it was already emitted by the first pipeline run).
            // Also erase #include "aie_runtime.h" emitc.include ops to avoid duplicate.
            {
                llvm::SmallVector<Operation *, 4> toErase;
                for (auto &op : hostModule.getOps()) {
                    if (auto func = dyn_cast<emitc::FuncOp>(op)) {
                        if (func.getName() == "dskernel_receiver")
                            toErase.push_back(&op);
                    } else if (auto inc = dyn_cast<emitc::IncludeOp>(op)) {
                        toErase.push_back(&op);
                    }
                }
                for (auto *op : toErase)
                    op->erase();
            }
        }
        if (failed(mlir::emitc::translateToCpp(hostModule, stream))) {
            llvm::errs() << "Failed to translate host MLIR to C++.\n";
            return false;
        }

        // In append mode, skip user source and __aie_launch emission — caller handles it
        if (appendMode) {
            stream.close();
            std::cout << "Host function appended to " << hostPath << std::endl;
            goto after_host_emit;
        }

        // Append user's rewritten source code after MLIR-generated functions.
        // The user source provides main() and calls host_canonicalized() via __aie_launch().
        if (!userRewrittenSource.empty()) {
            stream << "\n// ===== User source (preserved from original file) =====\n";

            // Count total args on the host function (includes XAie_DevInst* dev as arg 0)
            unsigned numArgs = 0;
            for (auto func : hostModule.getOps<emitc::FuncOp>()) {
                if (func.getName() == hostFuncName) {
                    numArgs = func.getNumArguments();
                    break;
                }
            }
            if (numArgs == 0) {
                for (auto func : hostModule.getOps<mlir::func::FuncOp>()) {
                    if (func.getName() == hostFuncName) {
                        numArgs = func.getNumArguments();
                        break;
                    }
                }
            }

            // Suppress the Clang-phase stubs (which had wrong arity) and emit
            // correct __aie_launch that forwards DDR pointers.
            stream << "#define AIEHLC_TILING_STUBS_DEFINED\n";
            // Re-emit SpatialPolicy types so user source constexpr definitions compile
            stream << "namespace aie {\n";
            stream
                << "enum class Pattern  { Broadcast = 0, Scatter = 1, Distribute = 1, Multicast = 2, Gather = 3 };\n";
            stream << "enum class Layout   { Row = 0, Col = 1, Grid = 2 };\n";
            stream << "enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };\n";
            stream << "enum class PadMaterialize { DDR = 0, Memtile = 1 };\n";
            stream << "enum class Im2col   { None = 0, Dma = 1 };\n";
            stream << "enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };\n";
            stream << "enum class TileMode { Partition = 0, Overlap = 1 };\n";
            stream << "enum class Objective { MaxArrayUtil = 0, MinLatency = 1, MinDma = 2 };\n";
            stream << "struct Bytes { int value = 0; };\n";
            // tile_level — nested second-level (on-core temporal) split. Must stay in
            // sync with the two Clang-phase emissions in aiehlc.cc (field order
            // tile_size=0, stride=1, rounds=2 is contractual for AST extraction).
            stream << "struct tile_level {\n";
            stream << "  int tile_size = 0;\n";
            stream << "  int stride    = 0;\n";
            stream << "  int rounds    = 0;\n";
            stream << "};\n";
            stream << "struct tile_dim {\n";
            stream << "  int fullsize   = 0;\n";
            stream << "  int tile_round = 0;\n";
            stream << "  int tile_size  = 0;\n";
            stream << "  int stride     = 0;\n";
            stream << "  int padsize    = 0;\n";
            stream << "  tile_level slice_tiling;\n"; // nested inner level (field 5)
            stream << "};\n";
            // SpatialPolicy — 3-part orthogonal policy (map/mat/sched).
            stream << "struct SpatialMap {\n";
            stream << "  Pattern act         = Pattern::Broadcast;\n";
            stream << "  Pattern wgt         = Pattern::Broadcast;\n";
            stream << "  Layout  layout      = Layout::Row;\n";
            stream << "  Flow    merge_order = Flow::Default;\n";
            stream << "  int     mesh_tiling_group1_dim = -1;\n";
            stream << "  int     mesh_tiling_group2_dim = -1;\n";
            stream << "};\n";
            stream << "struct Materialize {\n";
            stream << "  PadMaterialize pad    = PadMaterialize::DDR;\n";
            stream << "  Im2col         im2col = Im2col::None;\n";
            stream << "};\n";
            stream << "struct Schedule {\n";
            stream << "  int   pp_depth  = 2;\n";
            stream << "  Bytes l1_budget = Bytes{32*1024};\n";
            stream << "};\n";
            stream << "struct SpatialPolicy {\n";
            stream << "  SpatialMap  map;\n";
            stream << "  Materialize mat;\n";
            stream << "  Schedule    sched;\n";
            stream << "};\n";
            // GlobalPolicy — per-kernel GLOBAL policy bound by name convention
            // (<kernelName>_policy). fullconnect_auto: 1 (default) = M×N repeat,
            // 0 = no repeat (A/B each sent once).
            stream << "struct GlobalPolicy {\n";
            stream << "  int fullconnect_auto = 1;\n";
            stream << "};\n";
            // GemmSpace — policy + GEMM iteration space
            // (0 policy,1 m,2 n,3 k legacy; 4 d1,5 d2,6 d3,7 d4 per-port 2D/3D/4D).
            stream << "struct GemmSpace {\n";
            stream << "  SpatialPolicy policy;\n";
            stream << "  tile_dim m;\n";
            stream << "  tile_dim n;\n";
            stream << "  tile_dim k;\n";
            stream << "  tile_dim d1;\n";
            stream << "  tile_dim d2;\n";
            stream << "  tile_dim d3;\n";
            stream << "  tile_dim d4;\n";
            stream << "};\n";
            // Conv2dSpace — policy + conv iteration space; ih/iw carry the
            // EXACT input spatial dims (OH/OW derived via forward conv).
            // (0 policy,1 ih,2 iw,3 ic,4 oc,5 kh,6 kw,7 stride,8 pad,9 m).
            // Field 9 (m) is the optional explicit spatial-halo split.
            stream << "struct Conv2dSpace {\n";
            stream << "  SpatialPolicy policy;\n";
            stream << "  tile_dim ih;\n";
            stream << "  tile_dim iw;\n";
            stream << "  tile_dim ic;\n";
            stream << "  tile_dim oc;\n";
            stream << "  tile_dim kh;\n";
            stream << "  tile_dim kw;\n";
            stream << "  int stride = 1;\n";
            stream << "  int pad = 0;\n";
            stream << "  tile_dim m;\n";
            stream << "};\n";
            // ConvGeom + Conv2dSpace_Spatial — declarative spatial-halo conv
            // space (geom 0 in_h,1 in_w,2 cin,3 cout,4 ksize,5 S,6 pad_lo,
            // 7 pad_hi,8 cin_aligned; space 0 geom,1 out_tile_h,2 out_tile_w,
            // 3 objective,4 policy).
            stream << "struct ConvGeom {\n";
            stream << "  int in_h=0; int in_w=0; int cin=0; int cout=0;\n";
            stream << "  int ksize=0; int S=1; int pad_lo=0; int pad_hi=0; int cin_aligned=0;\n";
            stream << "};\n";
            stream << "struct Conv2dSpace_Spatial {\n";
            stream << "  ConvGeom      geom;\n";
            stream << "  int           out_tile_h = 0;\n";
            stream << "  int           out_tile_w = 0;\n";
            stream << "  Objective     objective  = Objective::MaxArrayUtil;\n";
            stream << "  SpatialPolicy policy;\n";
            stream << "  tile_dim      d1;\n";
            stream << "  tile_dim      d2;\n";
            stream << "};\n";
            // DmaTransform — general multi-dim DMA descriptor with factory methods
            stream << "struct DmaTransform {\n";
            stream << "  struct Dim { int stride; int wrap; };\n";
            stream << "  Dim dims[4] = {};\n";
            stream << "  int num_dims = 0;\n";
            stream << "  int iter_step = 0;\n";
            stream << "  int iter_wrap = 0;\n";
            stream << "  int mode = 0;\n";
            stream << "  int halo_slice = 0;\n";
            stream << "  int halo_step = 0;\n";
            stream << "  int split_dim = 0;\n";
            stream << "  int raw_h = 0;\n";
            stream << "  int raw_wc = 0;\n";
            stream << "  int kernel_h = 0;\n";
            stream << "  int kernel_w = 0;\n";
            stream << "  int input_c = 0;\n";
            stream << "  int stride = 0;\n";
            stream << "  int ow = 0;\n";
            stream << "  int oh_per_row = 0;\n";
            stream << "  static constexpr DmaTransform flat() { return {}; }\n";
            stream << "  static constexpr DmaTransform im2col(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    int OW = (W + 2*P - KW) / S + 1;\n";
            stream << "    int OH = (H + 2*P - KH) / S + 1;\n";
            stream << "    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};\n";
            stream << "    d.num_dims = 3;\n";
            stream << "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform dilated_im2col(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P, int D) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;\n";
            stream << "    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;\n";
            stream << "    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};\n";
            stream << "    d.num_dims = 3;\n";
            stream << "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform pool(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P) {\n";
            stream << "    return im2col(H, W, C, KH, KW, S, P);\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P, int G) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    int CPG = C / G;\n";
            stream << "    int OW = (W + 2*P - KW) / S + 1;\n";
            stream << "    int OH = (H + 2*P - KH) / S + 1;\n";
            stream << "    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};\n";
            stream << "    d.num_dims = 3;\n";
            stream << "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform transpose(int rows, int cols) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};\n";
            stream << "    d.num_dims = 2;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    d.dims[0] = {H * W, C};\n";
            stream << "    d.dims[1] = {1, W};\n";
            stream << "    d.num_dims = 2;\n";
            stream << "    d.iter_step = W;\n";
            stream << "    d.iter_wrap = H;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};\n";
            stream << "    d.num_dims = 2;\n";
            stream << "    d.iter_step = 1; d.iter_wrap = C;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "  static constexpr DmaTransform spatial(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P, int R) {\n";
            stream << "    DmaTransform d;\n";
            stream << "    int OH = (H + 2*P - KH) / S + 1;\n";
            stream << "    int oh_per_row = OH / R;\n";
            stream << "    d.mode = 1;\n";
            stream << "    d.halo_slice = (oh_per_row - 1) * S + KH;\n";
            stream << "    d.halo_step  = oh_per_row * S;\n";
            stream << "    d.split_dim  = 0;\n";
            stream << "    d.raw_h  = H;\n";
            stream << "    d.raw_wc = W * C;\n";
            stream << "    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;\n";
            stream << "    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;\n";
            stream << "    return d;\n";
            stream << "  }\n";
            stream << "};\n";
            stream << "struct ConvTiling {\n";
            stream << "  static constexpr DmaTransform spatial(int H, int W, int C,\n";
            stream << "      int KH, int KW, int S, int P, int R) {\n";
            stream << "    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);\n";
            stream << "  }\n";
            stream << "};\n";
            stream << "template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { "
                      "using type = T; };\n";
            stream << "template<typename T> constexpr int get_num_rounds(T) { return 0; }\n";
            stream << "template<typename T> constexpr int get_buffer_size(T) { return 0; }\n";
            stream << "constexpr int get_tile_rows() { return 0; }\n";
            stream << "constexpr int get_tile_cols() { return 0; }\n";
            stream << "constexpr int get_k_dim() { return 0; }\n";
            stream << "constexpr int get_tile_m() { return 0; }\n";
            stream << "constexpr int get_tile_n() { return 0; }\n";
            stream << "constexpr int get_effective_k() { return 0; }\n";
            stream << "constexpr int get_k_rounds() { return 0; }\n";
            stream << "constexpr int get_spatial_m_rounds() { return 0; }\n";
            stream << "constexpr int get_spatial_n_rounds() { return 0; }\n";
            stream << "template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }\n";
            stream << "constexpr int get_kernel_h() { return 0; }\n";
            stream << "constexpr int get_kernel_w() { return 0; }\n";
            stream << "constexpr int get_input_c() { return 0; }\n";
            stream << "constexpr int get_stride() { return 0; }\n";
            stream << "constexpr int get_ow() { return 0; }\n";
            stream << "constexpr int get_oh_per_row() { return 0; }\n";
            stream << "constexpr int get_halo_slice() { return 0; }\n";
            stream << "}\n";
            // aiePartition struct (shared between aieDim and aieMesh)
            stream << "struct aiePartition {\n";
            stream << "    int startCol, endCol, startRow, endRow;\n";
            stream << "};\n";
            // New programming model types: aieMesh + aieArray
            // Note: XAie_DevInst, __Runtime_explicit_init, __Runtime_alloc_buffer,
            // __Runtime_free_buffer are already declared via #include "aie_runtime.h"
            stream << "struct aieMesh {\n";
            stream << "    int rows, cols;\n";
            stream << "    aiePartition partition;\n";
            stream << "    int meshId;\n";
            stream << "};\n";
            stream << "struct aieArray {\n";
            stream << "    int nextMeshId = 0;\n";
            stream << "    XAie_DevInst* _dev = nullptr;\n";
            stream << "    aieMesh partition(aiePartition p, int rows, int cols) {\n";
            stream << "        int meshId = nextMeshId++;\n";
            stream << "        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol + 1);\n";
            stream << "        return aieMesh{rows, cols, p, meshId};\n";
            stream << "    }\n";
            stream << "    aieMesh partition(int rows, int cols) {\n";
            stream << "        int meshId = nextMeshId++;\n";
            stream << "        _dev = __Runtime_init_mesh_partition(meshId, 0, cols);\n";
            stream << "        return aieMesh{rows, cols, {0, cols - 1, 0, rows - 1}, meshId};\n";
            stream << "    }\n";
            stream << "    void* alloc(size_t size) { return __Runtime_alloc_buffer(_dev, size); }\n";
            stream << "    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }\n";
            stream << "    void synchronizecpu(void* ptr, size_t size) { __Runtime_sync_for_cpu(_dev, ptr, size); }\n";
            stream << "};\n";
            // Backward-compatible aieDim (maps to aieMesh internally)
            stream << "struct aieDim {\n";
            stream << "    int rows, cols;\n";
            stream << "    aiePartition partition;\n";
            stream << "    bool hasPartition;\n";
            stream << "    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}\n";
            stream
                << "    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), hasPartition(true) {}\n";
            stream << "};\n";
            stream << "inline void aieSetDevice(int) {}\n";
            stream << "inline void aieDeviceSynchronize() {}\n";
            // numArgs includes the XAie_DevInst* dev param (arg 0); DDR pointer args are numArgs-1
            unsigned numDdrArgs = (numArgs > 0) ? numArgs - 1 : 0;

            // Forward-declare kernel binary symbols so __aie_launch can reference them
            stream << "extern unsigned char _binary_kernel_" << computeKernelName << "_start[];\n";
            stream << "extern unsigned char _binary_kernel_" << computeKernelName << "_end[];\n";
            stream << "extern unsigned int _binary_kernel_" << computeKernelName << "_size;\n\n";

            // Helper: emit __Runtime_sync_for_dev calls for ALL buffers before DMA.
            // On ARM (baremetal), SyncForDev flushes+invalidates cache lines, which is
            // needed for outputs too: dirty cache lines (e.g. zeroed output buffer) must
            // be flushed and invalidated BEFORE DMA writes results to DDR, otherwise a
            // post-DMA invalidate (clean+invalidate) would flush stale zeros over the
            // DMA results. No post-launch sync is needed.
            auto emitSyncCalls = [&](llvm::raw_fd_ostream &os, unsigned nDdrArgs, const std::string &indent,
                                     bool beforeLaunch) {
                if (!beforeLaunch)
                    return; // no post-launch sync needed on ARM
                unsigned limit = nDdrArgs;
                if (limit > tensors.size())
                    limit = tensors.size();
                for (unsigned i = 0; i < limit; ++i) {
                    os << indent << "__Runtime_sync_for_dev(dev, _t" << i << ", _s" << i << ");\n";
                }
            };

            // --- __aie_launch with partition registry (init-once, dispatch by kernel name) ---
            if (numDdrArgs > 0) {
                // aieMesh overload (new programming model)
                stream << "inline void __aie_launch(const char* kernel, aieMesh mesh";
                for (unsigned i = 0; i < numDdrArgs; ++i)
                    stream << ", void* _t" << i << ", size_t _s" << i;
                stream << ", ...) {\n";
                stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << computeKernelName << "_start);\n";
                // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                emitSyncCalls(stream, numDdrArgs, "    ", /*beforeLaunch=*/true);
                stream << "    " << hostFuncName << "(dev";
                for (unsigned i = 0; i < numDdrArgs; ++i)
                    stream << ", _t" << i;
                stream << ");\n";
                // No post-launch sync needed (cache lines already invalidated)
                emitSyncCalls(stream, numDdrArgs, "    ", /*beforeLaunch=*/false);
                stream << "}\n";

                // aieDim overload (backward compatibility)
                stream << "inline void __aie_launch(const char* kernel, aieDim mesh";
                for (unsigned i = 0; i < numDdrArgs; ++i)
                    stream << ", void* _t" << i << ", size_t _s" << i;
                stream << ", ...) {\n";
                stream << "    (void)kernel;\n";
                stream << "    XAie_DevInst* dev;\n";
                stream << "    if (mesh.hasPartition) {\n";
                stream << "        dev = __Runtime_explicit_init_partition(mesh.partition.startCol, "
                          "mesh.partition.endCol - mesh.partition.startCol + 1);\n";
                stream << "    } else {\n";
                stream << "        dev = __Runtime_explicit_init();\n";
                stream << "    }\n";
                stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << computeKernelName << "_start);\n";
                // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                emitSyncCalls(stream, numDdrArgs, "    ", /*beforeLaunch=*/true);
                stream << "    " << hostFuncName << "(dev";
                for (unsigned i = 0; i < numDdrArgs; ++i)
                    stream << ", _t" << i;
                stream << ");\n";
                // No post-launch sync needed (cache lines already invalidated)
                emitSyncCalls(stream, numDdrArgs, "    ", /*beforeLaunch=*/false);
                stream << "    __Runtime_explicit_teardown(dev);\n";
                stream << "}\n";
            } else {
                // aieMesh overload (new programming model)
                stream << "template<typename... Args>\n";
                stream << "inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {\n";
                stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << computeKernelName << "_start);\n";
                stream << "    " << hostFuncName << "(dev);\n";
                stream << "}\n";

                // aieDim overload (backward compatibility)
                stream << "template<typename... Args>\n";
                stream << "inline void __aie_launch(const char* kernel, aieDim mesh, Args... args) {\n";
                stream << "    (void)kernel;\n";
                stream << "    XAie_DevInst* dev;\n";
                stream << "    if (mesh.hasPartition) {\n";
                stream << "        dev = __Runtime_explicit_init_partition(mesh.partition.startCol, "
                          "mesh.partition.endCol - mesh.partition.startCol + 1);\n";
                stream << "    } else {\n";
                stream << "        dev = __Runtime_explicit_init();\n";
                stream << "    }\n";
                stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << computeKernelName << "_start);\n";
                stream << "    " << hostFuncName << "(dev);\n";
                stream << "    __Runtime_explicit_teardown(dev);\n";
                stream << "}\n";
            }

            stream << userRewrittenSource << "\n";
        } else if (!tensors.empty()) {
            // Standalone / unittest mode: generate a default main() that
            // allocates DDR buffers matching the tensor parameters, fills
            // inputs with test data, calls the host function, and prints
            // output. Uses explicit init/teardown (no global g_DevInst).

            // Count total args on the host function (includes XAie_DevInst* dev as arg 0)
            unsigned numArgs = 0;
            for (auto func : hostModule.getOps<emitc::FuncOp>()) {
                if (func.getName() == hostFuncName) {
                    numArgs = func.getNumArguments();
                    break;
                }
            }
            if (numArgs == 0) {
                for (auto func : hostModule.getOps<mlir::func::FuncOp>()) {
                    if (func.getName() == hostFuncName) {
                        numArgs = func.getNumArguments();
                        break;
                    }
                }
            }

            stream << "\n// ===== Auto-generated main() for standalone testing =====\n";
            stream << "#include <stdio.h>\n";
            stream << "#include <stdlib.h>\n";
            stream << "#include <string.h>\n\n";
            stream << "#define __global__\n\n";

            // Forward-declare the host function (first arg is XAie_DevInst* dev)
            stream << "void " << hostFuncName << "(XAie_DevInst* dev";
            // numArgs includes the dev param; DDR pointer args are numArgs-1
            for (unsigned i = 1; i < numArgs; ++i) {
                stream << ", void*";
            }
            stream << ");\n\n";

            // Extern kernel binary symbols
            stream << "extern unsigned char _binary_kernel_" << computeKernelName << "_start[];\n";
            stream << "extern unsigned char _binary_kernel_" << computeKernelName << "_end[];\n";
            stream << "extern unsigned int _binary_kernel_" << computeKernelName << "_size;\n\n";

            // dskernel_receiver stub for the kernel that gets compiled separately
            stream << "__global__ void " << computeKernelName << "(size_t v1) {\n";
            stream << "  return;\n";
            stream << "}\n\n";

            stream << "int main() {\n";
            stream << "    printf(\"------------main--------\\n\");\n\n";

            // Allocate DDR buffers for each tensor
            for (unsigned i = 0; i < tensors.size(); ++i) {
                int64_t totalElements = 1;
                for (auto dim : tensors[i].shape)
                    totalElements *= dim;
                int64_t totalBytes = totalElements * (tensors[i].elementBitWidth / 8);
                stream << "    // Tensor " << i << ": " << (tensors[i].isInput ? "input" : "output") << ", "
                       << totalBytes << " bytes\n";
                stream << "    void* buf_" << i << " = malloc(" << totalBytes << ");\n";
            }
            stream << "\n";

            // Initialize inputs with test data, zero outputs
            for (unsigned i = 0; i < tensors.size(); ++i) {
                int64_t totalElements = 1;
                for (auto dim : tensors[i].shape)
                    totalElements *= dim;
                int64_t totalBytes = totalElements * (tensors[i].elementBitWidth / 8);
                if (tensors[i].isInput) {
                    stream << "    for (int j = 0; j < " << totalBytes << "; j++) ((char*)buf_" << i
                           << ")[j] = (char)(j + " << (i + 1) << ");\n";
                } else {
                    stream << "    memset(buf_" << i << ", 0, " << totalBytes << ");\n";
                }
            }
            stream << "\n";

            // Explicit init → sync ALL buffers → host function → teardown
            stream << "    XAie_DevInst* dev = __Runtime_explicit_init();\n";
            // Flush+invalidate ALL buffers (inputs AND outputs) before DMA.
            // On ARM, output buffers may have dirty cache lines (e.g. from memset to zero)
            // that must be flushed before DMA writes results to DDR.
            for (unsigned i = 0; i < tensors.size(); ++i) {
                int64_t totalElements = 1;
                for (auto dim : tensors[i].shape)
                    totalElements *= dim;
                int64_t totalBytes = totalElements * (tensors[i].elementBitWidth / 8);
                stream << "    __Runtime_sync_for_dev(dev, buf_" << i << ", " << totalBytes << ");\n";
            }
            stream << "    " << hostFuncName << "(dev";
            for (unsigned i = 0; i < tensors.size(); ++i) {
                stream << ", buf_" << i;
            }
            stream << ");\n";
            // No post-launch sync needed — cache lines already invalidated by SyncForDev
            stream << "\n";

            stream << "    printf(\"------------after matmul--------\\n\");\n\n";

            // Print output buffers
            for (unsigned i = 0; i < tensors.size(); ++i) {
                if (!tensors[i].isInput) {
                    int64_t totalElements = 1;
                    for (auto dim : tensors[i].shape)
                        totalElements *= dim;
                    int64_t totalBytes = totalElements * (tensors[i].elementBitWidth / 8);
                    stream << "    printf(\"Output buffer " << i << ":\\n\");\n";
                    stream << "    for (int j = 0; j < " << totalBytes << "; j++) printf(\"  out[%d]=%d\\n\", j, "
                           << "((unsigned char*)buf_" << i << ")[j]);\n";
                }
            }
            stream << "\n";

            // Free buffers
            for (unsigned i = 0; i < tensors.size(); ++i) {
                stream << "    free(buf_" << i << ");\n";
            }
            stream << "    __Runtime_explicit_teardown(dev);\n";
            stream << "    return 0;\n";
            stream << "}\n";
        }

        stream.close();
        std::cout << "Host code written to " << hostPath << std::endl;
    }
after_host_emit:

    // Emit kernel.cc (suffixed per kernel in multi-kernel mode)
    {
        std::string kernelFilename = hostFuncSuffix.empty() ? "kernel.cc" : "kernel_" + hostFuncSuffix + ".cc";
        std::string kernelPath = outputDir + "/" + kernelFilename;
        std::error_code ec;
        llvm::raw_fd_ostream stream(kernelPath, ec, llvm::sys::fs::OF_None);
        if (ec) {
            llvm::errs() << "Failed to open " << kernelPath << ": " << ec.message() << "\n";
            return false;
        }
        if (failed(mlir::emitc::translateToCpp(kernelModule, stream))) {
            llvm::errs() << "Failed to translate kernel MLIR to C++.\n";
            return false;
        }
        stream.close();
        std::cout << "Kernel code written to " << kernelPath << std::endl;
    }

    // Emit computekernel.cc (compute kernel implementation)
    {
        std::string computePath = outputDir + "/" + computeKernelName + ".cc";
        std::error_code ec;
        llvm::raw_fd_ostream stream(computePath, ec, llvm::sys::fs::OF_None);
        if (ec) {
            llvm::errs() << "Failed to open " << computePath << ": " << ec.message() << "\n";
            return false;
        }

        if (!userKernelBody.empty()) {
            // Write the user's __global__ kernel body verbatim, first resolving
            // IR-sourced builtins (get_arg_per_round_size_in_dim /
            // get_arg_total_rounds_in_dim) from the routing partitiontensor
            // TilingAttr captured in perTensorTiling / perTensorRounds above.
            std::string resolvedKernelBody = resolveArgDimBuiltin(userKernelBody, "aie::get_arg_per_round_size_in_dim",
                                                                  portVarNames, perTensorTiling);
            resolvedKernelBody = resolveArgDimBuiltin(resolvedKernelBody, "aie::get_arg_total_rounds_in_dim",
                                                      portVarNames, perTensorRounds);
            stream << "// User-provided compute kernel (extracted from __global__ function)\n";
            stream << resolvedKernelBody << "\n";
            std::cout << "Compute kernel (user-provided) written to " << computePath << std::endl;
        } else {
            // Auto-generate compute kernel
            // Determine type strings from element type
            std::string cType = kernelElementType + "_t"; // e.g. "int32_t"
            std::string vecType = "v4" + kernelElementType; // e.g. "v4int32"
            std::string winInputType = "input_window_" + kernelElementType;  // e.g. "input_window_int32"
            std::string winOutputType = "output_window_" + kernelElementType; // e.g. "output_window_int32"

            stream << "// Auto-generated compute kernel: " << computeKernelName << "\n";
            stream << "// " << numInputWindows << " input(s) + " << numOutputWindows << " output(s)\n";
            stream << "void " << computeKernelName << "(";

            // Generate parameter list
            int paramIdx = 0;
            for (int i = 0; i < numInputWindows; ++i) {
                if (paramIdx > 0) stream << ", ";
                stream << winInputType << " *window_in_" << i;
                paramIdx++;
            }
            for (int i = 0; i < numOutputWindows; ++i) {
                if (paramIdx > 0) stream << ", ";
                stream << winOutputType << " *window_out_" << i;
                paramIdx++;
            }
            stream << ") {\n";

            // Function body
            stream << "    unsigned coreid = get_coreid();\n";
            stream << "    int col = coreid >> 16;\n";
            stream << "    int row = coreid & 0x1F;\n";
            stream << "\n";
            stream << "    for (int k = 0; k < 2; k++) {\n";
            stream << "        klog(\"CENk\", k);\n";

            // Acquire all input windows
            for (int i = 0; i < numInputWindows; ++i) {
                stream << "        " << cType << " *in" << i
                       << " = (" << cType << " *)acquire_input_window(window_in_" << i << ");\n";
            }
            // Acquire all output windows
            for (int i = 0; i < numOutputWindows; ++i) {
                stream << "        " << cType << " *out" << i
                       << " = (" << cType << " *)acquire_output_window(window_out_" << i << ");\n";
            }

            stream << "\n";
            // Debug: print in0 values
            stream << "        // Debug: dump in0 buffer contents\n";
            stream << "        klog(\"IN0\", BUF_SZ_IN_0 * 4);\n";
            stream << "        for (int di = 0; di < BUF_SZ_IN_0 * 4; di++) {\n";
            stream << "            klog(\"IV\", (int)in0[di]);\n";
            stream << "        }\n";
            stream << "\n";
            // The copy/output body references out0 + BUF_SZ_OUT_0, which only exist
            // when the kernel has an output window. Input-only kernels (e.g. the
            // conv2d_spatial halo-input test, numOutputWindows == 0) must skip it,
            // otherwise the body references undeclared identifiers.
            if (numOutputWindows > 0) {
                stream << "        // GEMM kernel: out0[i] = in0[i] * in1[i]\n";
                stream << "        for (int i = 0; i < BUF_SZ_OUT_0; i++) {\n";
                stream << "            " << vecType << " data0 = *((" << vecType << " *)&in0[i * 4]);\n";
                if (numInputWindows > 1) {
                    stream << "            " << vecType << " data1 = *((" << vecType << " *)&in1[i * 4]);\n";
                }
                stream << "            *((" << vecType << " *)&out0[i * 4]) = data0;\n";
                stream << "        }\n";
                stream << "        klog(\"CLOP\", BUF_SZ_OUT_0);\n";
                stream << "\n";
                // Debug: print out0 values
                stream << "        // Debug: dump out0 buffer contents\n";
                stream << "        klog(\"OUT0\", BUF_SZ_OUT_0 * 4);\n";
                stream << "        for (int di = 0; di < BUF_SZ_OUT_0 * 4; di++) {\n";
                stream << "            klog(\"OV\", (int)out0[di]);\n";
                stream << "        }\n";
                stream << "\n";
            }

            // Release all windows
            for (int i = 0; i < numInputWindows; ++i) {
                stream << "        release_input_window(window_in_" << i << ");\n";
            }
            for (int i = 0; i < numOutputWindows; ++i) {
                stream << "        release_output_window(window_out_" << i << ");\n";
            }
            stream << "        klog(\"CEXT\", 1);\n";
            stream << "    }\n";
            stream << "}\n";

            std::cout << "Compute kernel (auto-generated) written to " << computePath << std::endl;
        }

        stream.close();
    }

    // Phase 4: Generate BCF/PRX for kernel compilation
    try {
        auto &allocator = ResourceMgr::instance()->coreMemAllocator();
        const auto &allocations = allocator.getAllocations();

        if (!allocations.empty()) {
            TilingBcf bcf;
            bcf.setStack(0x70000, 0x2800);
            bcf.addReservedDMB(0x40000, 0x10000);
            // This crashes the simulator, commenting out for now
            // bcf.addReservedDMB(0x7F800, 0x800);
            for (const auto &slot : allocations) {
                bcf.addSymbol(slot.symbolName, slot.address);
            }

            // Suffix BCF/PRX with kernel name in multi-kernel mode
            std::string bcfBaseName = hostFuncSuffix.empty() ? "aieml" : "aieml_" + hostFuncSuffix;
            std::string bcfPath = outputDir + "/" + bcfBaseName + ".bcf";
            if (bcf.exportToFile(bcfPath)) {
                std::cout << "BCF written to " << bcfPath << std::endl;
            } else {
                llvm::errs() << "Failed to write BCF to " << bcfPath << "\n";
            }

            TilingPrx prx("kernel", 22);
            prx.setBcfPath(bcfBaseName + ".bcf");
            prx.setKernelLLPath("./build/");

            std::string prxPath = outputDir + "/" + bcfBaseName + ".prx";
            if (prx.exportToFile(prxPath)) {
                std::cout << "PRX written to " << prxPath << std::endl;
            } else {
                llvm::errs() << "Failed to write PRX to " << prxPath << "\n";
            }

            std::cout << "\n=== Core Memory Allocation Summary ===" << std::endl;
            for (const auto &slot : allocations) {
                std::cout << "  " << slot.symbolName << " @ 0x" << std::hex << slot.address
                          << " (size=" << std::dec << slot.size << " bytes)" << std::endl;
            }
            std::cout << "  Free space: " << allocator.getFreeSpace() << " bytes" << std::endl;
        } else {
            std::cout << "No buffer allocations found; skipping BCF/PRX generation." << std::endl;
        }
    } catch (...) {
        std::cout << "ResourceMgr not initialized; skipping BCF/PRX generation." << std::endl;
    }

    // Phase 5: Routing path (dmaphop -> routinghw -> EmitC) for routing.cc
    // Uses the dmaphop module cloned after DmapToDmaphopPass so that packet IDs
    // in routing.cc match those in host.cc (both read from the same dmaphop IR).
    {
        int rstage = 0;
        std::string routingIrDir = setupPipelineIRDir("simplerouting");

        dumpPipelineIRToFile(routingDmaphopModule, routingIrDir, rstage++, "initial");

        // Create a fresh RoutingTopology for Phase 5 so that DmaphopToRoutinghwPass
        // starts with clean resource state. Phase 1 (RoutingToDmapPass/DmapToDmaphopPass)
        // consumed shim/port resources from the original rtopology. Phase 5 reads
        // shim tile info from the dmaphop IR and allocates its own DataIO objects.
        // Use the same 0-based partition-relative columns as the host path.
        RoutingTopology routingPathTopology(aieGen, "", relStartCol, relEndCol, partStartRow, partEndRow);

        if (!runPipelineSinglePass(ctx, routingDmaphopModule,
                                   std::make_unique<DmaphopToRoutinghwPass>(routingPathTopology), routingIrDir, rstage,
                                   "DmaphopToRoutinghwPass"))
            return false;
        if (!runPipelineSinglePass(ctx, routingDmaphopModule, std::make_unique<RoutingHWVerifyPass>(), routingIrDir,
                                   rstage, "RoutingHWVerifyPass"))
            return false;

        // Generate physical routing provenance map JSON while routinghw ops are
        // still present (RoutingHWLowerPass below lowers them into emitc.call).
        {
            auto routingProvenancePass = std::make_unique<RoutingProvenanceMapPass>(outputDir, partStartCol, aieGen);
            runPipelineSinglePass(ctx, routingDmaphopModule, std::move(routingProvenancePass), routingIrDir, rstage,
                                  "RoutingProvenanceMapPass");
        }

        if (!runPipelineSinglePass(ctx, routingDmaphopModule, std::make_unique<RoutingHWLowerPass>(routingPathTopology),
                                   routingIrDir, rstage, "RoutingHWLowerPass"))
            return false;

        if (!runPipelineSinglePass(ctx, routingDmaphopModule, std::make_unique<RoutingDeadArgPass>(), routingIrDir,
                                   rstage, "RoutingDeadArgPass"))
            return false;
        if (!runPipelineSinglePass(ctx, routingDmaphopModule, std::make_unique<RoutingConstantFoldPass>(aieGen),
                                   routingIrDir, rstage, "RoutingConstantFoldPass"))
            return false;
        if (!runPipelineSinglePass(ctx, routingDmaphopModule, mlir::createCanonicalizerPass(), routingIrDir, rstage,
                                   "CanonicalizerPass"))
            return false;

        // After all routing lowering passes, clean up residual memref-related
        // ops (bufferization.to_tensor and routing.routingcreatescheduletensor)
        // that prevent RoutingDeadArgPass from stripping memref func args.
        // Only erase specific known-dead op types — never erase scf/emitc ops.
        for (auto func : routingDmaphopModule.getOps<mlir::func::FuncOp>()) {
            if (func.getBody().empty())
                continue;
            Block &entry = func.getBody().front();
            // Multi-pass: erasing routing ops may make to_tensor results dead
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto &op : llvm::make_early_inc_range(entry.getOperations())) {
                    // Only erase bufferization.to_tensor or routing dialect ops
                    bool isTargetOp = isa<bufferization::ToTensorOp>(&op) ||
                                      (op.getDialect() && op.getDialect()->getNamespace() == "routing");
                    if (isTargetOp && op.use_empty()) {
                        op.erase();
                        changed = true;
                    }
                }
            }
            // Now strip dead block args (memref params no longer used)
            for (int i = (int)entry.getNumArguments() - 1; i >= 0; --i) {
                if (entry.getArgument(i).use_empty())
                    entry.eraseArgument(i);
            }
            // Update function type to match remaining args
            SmallVector<Type> argTypes;
            for (auto arg : entry.getArguments())
                argTypes.push_back(arg.getType());
            func.setFunctionType(FunctionType::get(func.getContext(), argTypes, func.getFunctionType().getResults()));
        }

        std::string routingPath = outputDir + "/routing.cc";
        std::error_code ec;
        llvm::raw_fd_ostream stream(routingPath, ec, llvm::sys::fs::OF_None);
        if (ec) {
            llvm::errs() << "Failed to open " << routingPath << ": " << ec.message() << "\n";
            return false;
        }
        // Emit #include before the translated C++ so int32_t etc. are declared
        stream << "#include <xaiengine.h>\n\n";
        if (failed(mlir::emitc::translateToCpp(routingDmaphopModule, stream))) {
            llvm::errs() << "Failed to translate routing MLIR to C++.\n";
            return false;
        }
        stream.close();
        std::cout << "Routing code written to " << routingPath << std::endl;
    }

    return true;
}
