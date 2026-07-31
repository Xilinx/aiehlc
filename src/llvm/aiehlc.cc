/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "../../include/gcommon.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

//#include "AieFrontEnd.h"
#include "../mlir/mlirfront/AieFrontEnd.h"
#include <boost/algorithm/string.hpp>

#include <ios>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <unordered_map>

#include "routingimplement/include/hw/hwresource.h"
#include "tilinglinalg_pipeline.h"

class AieFrontEnd;

static bool use_llvm_aie = false;
static bool isTilingLinalgMode = false;
static int tilingMeshRows = 0, tilingMeshCols = 0;

struct ParsedTensorInfo {
    std::string varName;
    std::vector<int64_t> shape;
    int elementBitWidth;
    bool isInput;
    std::string spatialTag; // "row_broadcast_in", "col_broadcast_in", etc. or "" for default
    std::string policyName; // "RowBC", "ColBC", "LtoR_Merge", etc. (from aie::port<T, Policy>)
    // Resolved SpatialPolicy struct fields (from Clang AST constexpr evaluation)
    int pattern = 0;      // 0=Broadcast, 1=Scatter, 2=Multicast, 3=Gather
    int distribution = 0; // 0=Row, 1=Col, 2=Grid
    int mergeOrder = 0;   // 0=Default, 1=LeftToRight, 2=RightToLeft
    // Two-level mesh tiling axes (SpatialMap.map fields 4/5). 1-based tensor
    // d-index (d1/d2/d3). group1 = mesh-ROW split axis (implicit today via
    // hwAxisOwner="row"); group2 = within-row mesh-COL split axis (e.g. 3 =
    // channel). -1 / 0 = unset.
    int meshTilingGroup1Dim = -1;
    int meshTilingGroup2Dim = -1;
    int pingPong = 2;
    int maxBufferBytes = 4096;   // max per-buffer size (PP_MAX_BYTES equivalent)
    // Two-level tiling hints (structured per-dimension descriptor).
    // Each tile_dim describes one dimension's split, mapping 1:1 onto a
    // #routing.level. User-facing fields:
    //   fullsize (full PADDED dim length), tile_round (explicit outer round/
    //   group count), tile_size (outer per-tile slice length), stride (outer
    //   step; overlap = tile_size - stride), padsize (per-side pad, metadata/
    //   boundary only), slice_tiling (nested inner level).
    // `groups` maps from tile_round when the user pins it; otherwise it is
    // derived from base (=fullsize) in readTileDim so downstream code
    // (explicitGroups*, validateDim, halo paths) keeps working.
    // size == 0 means "auto-derive from memory budget" (the old tileX == 0).
    struct TileDim {
        int size = 0;    // tile_dim field 2: tile_size (per-tile outer slice)
        int stride = 0;  // tile_dim field 3: outer step (overlap = size - stride)
        int base = 0;    // tile_dim field 0: fullsize (full PADDED dim length)
        int padSize = 0; // tile_dim field 4: padsize (per-side pad, metadata only)
        int groups = 0;  // tile_dim field 1: tile_round (else derived from base)
        // Nested inner level (tile_dim field 5 slice_tiling), mirrors
        // #routing.level: each outer `size` slice is further chunked on-core into
        // l2Groups rounds of l2Size advancing by l2Stride (overlap = l2Size -
        // l2Stride). Zero / <=1 l2Groups keeps the single-level path unchanged.
        int l2Size = 0;   // slice_tiling.tile_size: per-round slice (e.g. 19)
        int l2Stride = 0; // slice_tiling.stride: step between rounds (e.g. 14)
        int l2Groups = 0; // slice_tiling.rounds (or derived): number of rounds (e.g. 4)
    };
    TileDim tdM; // replaces tile_m
    TileDim tdN; // replaces tile_n
    TileDim tdK; // replaces tile_k
    // Per-port 2D GemmSpace dims (GemmSpace fields 4/5). When tdD1.size > 0 the
    // port describes its OWN matrix and the role-aware (d1/d2) extraction path is
    // used instead of the global m/n/k path.
    TileDim tdD1;                    // GemmSpace.d1 (per-port dim 1)
    TileDim tdD2;                    // GemmSpace.d2 (per-port dim 2)
    TileDim tdD3;                    // GemmSpace.d3 (per-port dim 3, e.g. channels);
                                     // bookkeeping only — folded into raw_wc.
    TileDim tdD4;                    // GemmSpace.d4 (per-port dim 4, e.g. padded
                                     // channel coverage for the filter K). Used by the
                                     // structured-dim coalescing path.
    bool perPort2D = false;          // true when d1/d2 were set on this port
    int layoutTransform = 0;         // 0=None, 1=DmaShuffle, 2=CoreShuffle
    int tileMode = 0;                // 0=Partition, 1=Overlap
    bool requireFullCoverage = true; // validation: tiles must cover the dim
    bool policyResolved = false; // true once AST extraction succeeds
    // DmaTransform: general multi-dim DMA descriptor for shim tile addressing
    // Extracted from targs[2] of aie::port<T, Policy, DmaTransform>
    struct ShimDma {
        struct Dim {
            int stride = 0;
            int wrap = 0;
        };
        Dim dims[4];
        int num_dims = 0;
        int iter_step = 0;
        int iter_wrap = 0;
        // Spatial-halo mode (from aie::ConvTiling::spatial). mode==1 selects an
        // overlapping contiguous row-block distribution instead of im2col multi-dim.
        int mode = 0;       // 0 = flat / im2col-by-dims, 1 = spatial_halo
        int halo_slice = 0; // input rows owned by each tile-row (e.g. 61)
        int halo_step = 0;  // row stride between consecutive tile-rows (e.g. 56)
        // When true the halo split was pinned explicitly via Conv2dSpace.m and
        // the Overlap auto-derivation (OH/HW_ROWS) must NOT overwrite it.
        bool haloExplicit = false;
        int split_dim = 0; // tensor dimension carrying the halo split (usually 0)
        int raw_h = 0;     // raw input H (declared A-tensor shape dim 0)
        int raw_wc = 0;    // raw input W*C (declared A-tensor shape dim 1)
        // Conv geometry carried through from ConvTiling::spatial (struct fields 10-15).
        int kernel_h = 0;   // KERNEL_H (e.g. 7)
        int kernel_w = 0;   // KERNEL_W (e.g. 7)
        int input_c = 0;    // INPUT_C (e.g. 3)
        int stride = 0;     // STRIDE (e.g. 2)
        int ow = 0;         // OUTPUT_W (e.g. 112)
        int oh_per_row = 0; // OUTPUT_H / HW_ROWS (e.g. 28)
        // 2D width-split (spatial-halo on-core WIDTH rounds). When w_rounds > 1
        // the WIDTH is chunked into on-core rounds (NOT a mesh axis): each chunk
        // delivers a NARROW slab [halo_slice, w_slice*C] via a 2D shim BD with the
        // PADDED row pitch, and the kernel iterates H_chunks * W_chunks rounds.
        // Zero / 1 keeps the legacy height-only flat contiguous path unchanged.
        int w_slice = 0;   // per-chunk input cols (e.g. TILE_W = 61)
        int w_step = 0;    // halo step between width chunks (e.g. TILE_STRIDE_W = 56)
        int w_rounds = 0;  // number of width chunks (e.g. 4); 0/1 = no width split
        int row_pitch = 0; // PADDED input row pitch in elements (INPUT_W_PAD * C = 920)
        int ow_t = 0;      // per-chunk output cols (e.g. OW_T = 28)
        // Nested L2 (on-core temporal ROW-split) on the SAME (row) axis as the L1
        // halo. Each HW tile owns `halo_slice` rows (L1); when l2Rounds > 1 that
        // slice is further chunked into l2Rounds on-core temporal ROUNDS of
        // `l2Slice` rows advancing by `l2Step` rows (L2 overlap = l2Slice - l2Step).
        // Realized via the BD iteration dim + kernel round multiplier downstream.
        // Zero / 1 keeps the single-level (L1-only) path unchanged.
        int l2Slice = 0;  // input rows per on-core round (e.g. 19)
        int l2Step = 0;   // row stride between L2 rounds (e.g. 14)
        int l2Rounds = 0; // number of L2 on-core rounds (e.g. 4); 0/1 = no L2 split
        // Conv2dSpace-derived geometry (populated when the port carries a
        // Conv2dSpace and the explicit DmaTransform is flat()). These let the
        // post-extraction derivation reconstruct the same im2col/spatial shim
        // DMA that DmaTransform::im2col / ConvTiling::spatial would produce.
        bool fromConvSpace = false;
        int conv_ih = 0; // reconstructed INPUT_H
        int conv_iw = 0; // reconstructed INPUT_W
        int conv_oh = 0; // OUTPUT_H
        int conv_oc = 0; // output channels (num filters)
        int pad = 0;     // PAD
        bool empty() const { return num_dims == 0 && mode == 0; }
    } shimDma;
};
static std::vector<ParsedTensorInfo> parsedTensors;

// DerivedTilingParams is defined in tilinglinalg_pipeline.h
static DerivedTilingParams derivedTilingParams;

static int64_t macroDimM = 0, macroDimN = 0, macroDimK = 0; // GEMM dimensions from launch args or macros

static int parsedDebugLevel = -1; // -1 = not set by user, >=0 = #pragma aie_debug_level value
static std::string userSourceDir; // directory containing the original user source file
static PartitionDesc parsedPartition; // default: invalid (all -1), set by #pragma aie_partition
static std::vector<MeshKernelDesc> parsedMeshKernels; // multi-kernel mode: one per <<<mesh>>> launch
static std::vector<std::string> kernel_name_list;
static std::unordered_map<std::string, const clang::FunctionDecl*> globalKernelFuncs;
// Per-kernel "fullconnect_auto" flag, set from a file-scope constexpr
// `aie::GlobalPolicy <kernelName>_policy = {.fullconnect_auto = 0|1}` (bound by
// name convention). true (default) = full-connect M×N cartesian DMA repeat;
// false = no repeat (A/B each sent once).
static std::unordered_map<std::string, bool> kernelFullConnectAuto;
static std::string userKernelBody;     // raw source text of __global__ function body (last kernel, backward compat)
static std::string userKernelFuncName; // kernel function name from __global__ (last kernel, backward compat)
static std::unordered_map<std::string, std::string> globalKernelBodies; // per-kernel: name -> cleaned body text
static std::vector<std::string> userMacroDefines; // #define lines from user source

using namespace clang;
using namespace clang::tooling;

// Populate SplitModel.tensorSplits[i].gemmTiling for the even-mesh GEMM path
// (fullconnect_auto=1) from the derived M/N/K tiling. Each tensor gets one
// GemmDimTile per tensor dim so routingmanager::buildGemmTilingAttr can emit a
// #routing.tiling that carries the same slice sizes the flat module attrs do.
// Role is derived from the split descriptor (hwAxisOwner/isInput) + shape:
//   input + row  -> A [M, K] : d(splitDim)=M mesh split + on-core tileM;
//                              contraction dim=K (no mesh split) + effectiveK.
//   input + col  -> B [N, K] : d(splitDim)=N mesh split + on-core tileN; d=K.
//   output       -> C [M, N] : d(splitDim)=M mesh split + tileM; other dim=N
//                              (mesh-col gather rounds) + on-core tileN.
// Only 2D tensors get the GEMM tiling; higher-rank (lifted 3D/4D) tensors keep
// full-extent structural levels on the non-split dims. No-op when tiling is
// invalid so conv / non-GEMM kernels are unaffected.
static void populateGemmTiling(SplitModel &splitModel, const std::vector<TensorParam> &tensors,
                               const DerivedTilingParams &dp, int meshRows, int meshCols) {
    if (!dp.valid || meshRows <= 0 || meshCols <= 0)
        return;
    int64_t tileM = dp.tileM > 0 ? dp.tileM : dp.tileRows; // on-core M slice
    int64_t tileN = dp.tileN > 0 ? dp.tileN : dp.tileCols; // on-core N slice
    int64_t effK = dp.effectiveK > 0 ? dp.effectiveK : dp.kDim;
    int64_t mRounds = dp.spatialMRounds > 0 ? dp.spatialMRounds : 1;
    int64_t nRounds = dp.spatialNRounds > 0 ? dp.spatialNRounds : 1;
    int64_t kRounds = dp.kRounds > 0 ? dp.kRounds : 1;

    auto meshLevel = [](int64_t base, int64_t meshSlice, int64_t meshRounds, int64_t coreSlice, int64_t coreRounds) {
        TensorSplitDesc::GemmDimTile t;
        t.base = base;
        t.meshSlice = meshSlice;
        t.meshStep = meshSlice;
        t.meshRounds = meshRounds;
        t.coreSlice = coreSlice;
        t.coreStep = coreSlice;
        t.coreRounds = coreRounds;
        return t;
    };

    for (size_t i = 0; i < splitModel.tensorSplits.size() && i < tensors.size(); ++i) {
        auto &split = splitModel.tensorSplits[i];
        const auto &shape = tensors[i].shape;
        split.gemmTiling.clear();
        // GEMM tiling only for 2D tensors (rank-generic higher dims stay flat).
        if (shape.size() != 2)
            continue;
        int sd = (split.splitDim == 0) ? 0 : 1; // mesh-split tensor dim
        int od = 1 - sd;                        // the other tensor dim
        bool isRow = (split.hwAxisOwner == "row");
        bool isCol = (split.hwAxisOwner == "col");
        std::vector<TensorSplitDesc::GemmDimTile> dims(2);
        if (tensors[i].isInput && isRow) {
            // A [M, K]: split dim = M (mesh rows, on-core tileM); other = K.
            dims[sd] = meshLevel(shape[sd], dp.tileRows, meshRows, tileM, mRounds);
            dims[od] = meshLevel(shape[od], shape[od], 1, effK, kRounds);
        } else if (tensors[i].isInput && isCol) {
            // B [N, K]: split dim = N (mesh cols, on-core tileN); other = K.
            dims[sd] = meshLevel(shape[sd], dp.tileCols, meshCols, tileN, nRounds);
            dims[od] = meshLevel(shape[od], shape[od], 1, effK, kRounds);
        } else if (!tensors[i].isInput) {
            // C [M, N]: split dim = M (mesh rows, tileM); other = N (mesh-col
            // gather rounds, on-core tileN).
            dims[sd] = meshLevel(shape[sd], dp.tileRows, meshRows, tileM, mRounds);
            dims[od] = meshLevel(shape[od], dp.tileCols, meshCols, tileN, nRounds);
        } else {
            continue; // unknown role: leave flat
        }
        split.gemmTiling = std::move(dims);
    }
}

// Extract partition bounds from the 3rd argument of aieDim constructor.
// Handles InitListExpr {2,7,0,10}, CXXConstructExpr (copy/aggregate), and DeclRefExpr (variable).
static void extractPartitionFromExpr(const Expr *partExpr, ASTContext *Context) {
    partExpr = partExpr->IgnoreParenImpCasts();
    // Try InitListExpr: aiePartition{2,7,0,10} or {2,7,0,10}
    if (const auto *IL = dyn_cast<InitListExpr>(partExpr)) {
        if (IL->getNumInits() >= 4) {
            clang::Expr::EvalResult r0, r1, r2, r3;
            if (IL->getInit(0)->EvaluateAsInt(r0, *Context) && IL->getInit(1)->EvaluateAsInt(r1, *Context) &&
                IL->getInit(2)->EvaluateAsInt(r2, *Context) && IL->getInit(3)->EvaluateAsInt(r3, *Context)) {
                parsedPartition.startCol = r0.Val.getInt().getExtValue();
                parsedPartition.endCol = r1.Val.getInt().getExtValue();
                parsedPartition.startRow = r2.Val.getInt().getExtValue();
                parsedPartition.endRow = r3.Val.getInt().getExtValue();
                llvm::outs() << "[TilingLinalg] Partition: [" << parsedPartition.startCol << ","
                             << parsedPartition.endCol << "," << parsedPartition.startRow << ","
                             << parsedPartition.endRow << "]\n";
            }
        }
        return;
    }
    // Try CXXConstructExpr: aggregate or copy constructor
    if (const auto *CE = dyn_cast<CXXConstructExpr>(partExpr)) {
        if (CE->getNumArgs() >= 4) {
            clang::Expr::EvalResult r0, r1, r2, r3;
            if (CE->getArg(0)->EvaluateAsInt(r0, *Context) && CE->getArg(1)->EvaluateAsInt(r1, *Context) &&
                CE->getArg(2)->EvaluateAsInt(r2, *Context) && CE->getArg(3)->EvaluateAsInt(r3, *Context)) {
                parsedPartition.startCol = r0.Val.getInt().getExtValue();
                parsedPartition.endCol = r1.Val.getInt().getExtValue();
                parsedPartition.startRow = r2.Val.getInt().getExtValue();
                parsedPartition.endRow = r3.Val.getInt().getExtValue();
                llvm::outs() << "[TilingLinalg] Partition: [" << parsedPartition.startCol << ","
                             << parsedPartition.endCol << "," << parsedPartition.startRow << ","
                             << parsedPartition.endRow << "]\n";
            }
        } else if (CE->getNumArgs() == 1) {
            // Copy/move ctor — unwrap inner
            extractPartitionFromExpr(CE->getArg(0), Context);
        }
        return;
    }
    // Try DeclRefExpr: variable reference like `part`
    if (const auto *DR = dyn_cast<DeclRefExpr>(partExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
            if (VD->hasInit())
                extractPartitionFromExpr(VD->getInit(), Context);
        }
    }
}

class GlobalFunctionVisitor : public RecursiveASTVisitor<GlobalFunctionVisitor> {
private:
	std::string GetFuncText(FunctionDecl *f) {
		clang::SourceRange functionRange = f->getSourceRange();
		clang::SourceLocation startLocation = functionRange.getBegin();
		clang::SourceLocation endLocation = Rewrite->getSourceMgr().getExpansionLoc(functionRange.getEnd());
		clang::SourceLocation endLocationWithOffset = clang::Lexer::getLocForEndOfToken(endLocation, 0, Rewrite->getSourceMgr(), Rewrite->getLangOpts());
		bool InvalidTemp = false;
		std::string fullkernel;
		clang::StringRef text = Rewrite->getSourceMgr().getBufferData(Rewrite->getSourceMgr().getFileID(startLocation), &InvalidTemp);
		if (!InvalidTemp) {
			const char* begin = text.begin() + Rewrite->getSourceMgr().getFileOffset(startLocation);
			const char* end = text.begin() + Rewrite->getSourceMgr().getFileOffset(endLocationWithOffset);
			std::string functionString(begin, end);
			fullkernel = functionString;
		} else {
			std::cerr << "Error: Unable to access the buffer for the original function text.\n";
			return "";
		}
		std::string str(fullkernel.data(), fullkernel.size());
		return str;
	}
	void ReplaceKernel(FunctionDecl *f) {
		return;
	}
public:
    explicit GlobalFunctionVisitor(Rewriter* Rewrite, ASTContext *Context, AieFrontEnd* Aiefrontend)
        : Context(Context), Rewrite(Rewrite) , Aiefrontend(Aiefrontend){
			 std::ofstream kl(AOUT+std::string("kernel_list"), std::ios::out);
		}

     void ExportFunction(FunctionDecl *FD, std::string kname) {
	     /*
        if (FD->hasBody()) {
            llvm::outs() << FD->getNameAsString() << " Function Body:\n";
            FD->getBody()->printPretty(llvm::outs(), nullptr, PrintingPolicy(Context->getLangOpts()));
            llvm::outs() << "\n\n";
        }*/
			llvm::outs() << "Exporting File: " << AOUT+std::string("kernel_list") << "\n\n";
			 std::ofstream kl(AOUT+std::string("kernel_list"), std::ios::app | std::ios::out);
			 kl << kname << std::endl;
			 kl.close();

			 std::error_code error_code;
			 //change the ext from .c into .cc to fix the aiecompiler compile issue
			llvm::outs() << "Exporting File: " << AOUT+kname + ".cc" << "\n";
			 std::ofstream fd(AOUT+kname + ".cc");
			 auto str = GetFuncText(FD);
			 //std::string toRemove = "__attribute__((annotate(\"__global__\")))";
			//fixme, shoud only remove mem_address and global annotate attribute, correct way is to do
      // one more pass to remove the related attribute
			 std::string toRemove = "__attribute__";
			 size_t pos, prepos=std::string::npos;
			 while((pos = str.find(toRemove)) != std::string::npos){
         //avoid infinite loop
				 if (prepos != std::string::npos && prepos == pos) break;
				 prepos = pos;
         //the remove logic
				 auto nstr = str.substr(pos + toRemove.size());
				 auto npos = nstr.find("(");
				 if (npos == std::string::npos) break;
				 auto cstr = nstr.c_str();
				 int lb = 1;
				 while(lb && npos < nstr.size()) {
					 auto c = cstr[++npos];
					 lb += (c == '(' ? 1 :(c == ')' ? -1 : 0) );
					 //std::cout << "npos is " << npos << " "<<lb << " is lb cstr is " << &cstr[npos]<< std::endl;
				 }
				 str.erase(pos, toRemove.size() + npos + 1);
				 //avoid infinite loop, reset prepos
				 prepos = std::string::npos;
			 }

             // Remove #ifdef KERNEL_COMPILE / #endif guard pairs added during
             // preprocessing.  Only remove #endif lines that were inserted to
             // close the KERNEL_COMPILE guard — leave user #endif directives
             // (e.g. #ifdef DEBUG_OUTPUT_ORDER) intact.
             std::string ifdefToRemove = "#ifdef KERNEL_COMPILE";
             while ((pos = str.find(ifdefToRemove)) != std::string::npos) {
                 // Erase the #ifdef KERNEL_COMPILE line
                 auto endOfLine = str.find("\n", pos);
                 if (endOfLine != std::string::npos) {
                     str.erase(pos, endOfLine - pos + 1);
                 } else {
                     str.erase(pos, ifdefToRemove.size());
                 }
                 // Find and erase the matching #endif (last one in the string,
                 // since the KERNEL_COMPILE guard wraps the entire body).
                 std::string endifToRemove = "#endif";
                 auto lastEndif = str.rfind(endifToRemove);
                 if (lastEndif != std::string::npos) {
                     auto endOfEndifLine = str.find("\n", lastEndif);
                     if (endOfEndifLine != std::string::npos) {
                         str.erase(lastEndif, endOfEndifLine - lastEndif + 1);
                     } else {
                         str.erase(lastEndif, endifToRemove.size());
                     }
                 }
             }

             // Strip spatial type wrappers from kernel source text
             // "aie::row_broadcast_in<input_window_int8 *>" → "input_window_int8 *"
             for (const auto &wrapper : {"row_broadcast_in", "col_broadcast_in", "tiled_in", "row_major_out",
                                         "col_major_out", "row_reduce_out"}) {
                 std::string prefix = "aie::" + std::string(wrapper) + "<";
                 size_t spos;
                 while ((spos = str.find(prefix)) != std::string::npos) {
                     size_t start = spos;
                     size_t angleStart = spos + prefix.size();
                     int depth = 1;
                     size_t end = angleStart;
                     while (end < str.size() && depth > 0) {
                         if (str[end] == '<')
                             depth++;
                         else if (str[end] == '>')
                             depth--;
                         end++;
                     }
                     // Replace "aie::wrapper<INNER>" with "INNER"
                     std::string inner = str.substr(angleStart, end - angleStart - 1);
                     str.replace(start, end - start, inner);
                 }
             }

             // Strip aie::port<T, Policy> → T (extract first template arg only)
             {
                 std::string portPrefix = "aie::port<";
                 size_t spos;
                 while ((spos = str.find(portPrefix)) != std::string::npos) {
                     size_t start = spos;
                     size_t angleStart = spos + portPrefix.size();
                     // Find the top-level comma (separating T from Policy) or closing >
                     int depth = 1;
                     size_t commaPos = std::string::npos;
                     size_t end = angleStart;
                     while (end < str.size() && depth > 0) {
                         if (str[end] == '<')
                             depth++;
                         else if (str[end] == '>')
                             depth--;
                         else if (str[end] == ',' && depth == 1 && commaPos == std::string::npos)
                             commaPos = end;
                         if (depth > 0)
                             end++;
                     }
                     end++; // skip closing >
                     // Extract first template arg (the inner type T)
                     size_t innerEnd = (commaPos != std::string::npos) ? commaPos : (end - 1);
                     std::string inner = str.substr(angleStart, innerEnd - angleStart);
                     // Trim whitespace
                     while (!inner.empty() && inner.front() == ' ')
                         inner.erase(0, 1);
                     while (!inner.empty() && inner.back() == ' ')
                         inner.pop_back();
                     str.replace(start, end - start, inner);
                 }
             }

             // Capture cleaned kernel body for tiling mode computekernel.cc emission
             userKernelFuncName = kname;
             userKernelBody = str;
             globalKernelBodies[kname] = str; // per-kernel map for multi-kernel support
             llvm::outs() << "[TilingLinalg] Captured __global__ kernel body for: " << kname << "\n";

             std::string header = "/******************************************************************************"
                                  "\n* Auto-generated by aiehlc.\n"
                                  "******************************************************************************/\n";
             // llvm aie can not handle these headers yet, results in compile errors
			 if(use_llvm_aie) {
				header += 
				 "\n#include <adf.h>\n\n"
				//  "\n#include <aie_api/aie.hpp>\n"
				//  "\n#include <aie_api/aie_adf.hpp>\n"
				//  "\n#include <aie_api/utils.hpp>\n\n";
				;
			 }
			 else {
                 header += "\n#include <adf.h>"
                           "\n#include <aie_api/aie.hpp>"
                           "\n#include <aie_api/aie_adf.hpp>"
                           "\n#include <aie_api/utils.hpp>\n\n";
             }
             header += "#include \"kernel_log.h\"\n\n";
             // Add user macro definitions before kernel body
             std::string macroBlock;
             if (!userMacroDefines.empty()) {
                 macroBlock = "\n// User macro definitions from source file\n";
                 for (const auto &macro : userMacroDefines) {
                     macroBlock += macro + "\n";
                 }
                 macroBlock += "\n";
             }
             str = header + macroBlock + str;
             fd << str << std::endl;
    }

		 void printSourceRange(const clang::SourceRange &range, const clang::SourceManager &sourceManager) {
			 llvm::outs() << "SourceRange: ";

			 // Get the start and end locations
			 clang::SourceLocation startLoc = range.getBegin();
			 clang::SourceLocation endLoc = range.getEnd();

			 // Print the file name
			 auto& sc = Rewrite->getSourceMgr();
			 llvm::outs() << sourceManager.getFilename(startLoc).str() << ":";
			 llvm::outs() << sourceManager.getFilename(endLoc).str() << ":";

			 ///*
			 // Print the start line and column
			 llvm::outs() << sourceManager.getSpellingLineNumber(startLoc) << ":" << sourceManager.getSpellingColumnNumber(startLoc) << "-";

			 // Print the end line and column
			 llvm::outs() << sourceManager.getSpellingColumnNumber(endLoc) << ":" << sourceManager.getSpellingColumnNumber(endLoc) << "\n";
			 //*/
		 }
		 bool VisitFunctionDecl(FunctionDecl *f) {
			 if (f->hasBody() && f->hasAttr<AnnotateAttr>()) {
                 // Check for streaming annotation
                 bool isStreamingKernel = false;
                 for (auto attr : f->attrs()) {
                     if (auto anno = clang::dyn_cast<clang::AnnotateAttr>(attr)) {
                         if (anno->getAnnotation() == "streaming") {
                             isStreamingKernel = true;
                             llvm::outs() << "Detected streaming kernel annotation\n";
                             break;
                         }
                     }
                 }

                for (auto attr:f->attrs()) {
					if (auto anno = clang::dyn_cast<clang::AnnotateAttr>(attr)){
						if (anno->getAnnotation() == "__global__" || anno->getAnnotation() == "__kernel__") {
							std::string kernelName = f->getNameInfo().getName().getAsString();

							// Store function decl for later tensor parameter extraction
							globalKernelFuncs[kernelName] = f;

                            // GlobalPolicy binding by name convention: look up a file-scope
                            // constexpr `aie::GlobalPolicy <kernelName>_policy = {...}` and read
                            // its fullconnect_auto field (struct field 0). Default = 1 (full
                            // connect) when no matching policy variable is present.
                            {
                                bool fca = true; // default: full-connect
                                bool found = false;
                                // Explicit binding via __global__(<policyVar>) emits an
                                // annotate("aiepolicy:<policyVar>"); prefer it over the
                                // <kernelName>_policy naming convention (fallback).
                                std::string wantName = kernelName + "_policy";
                                for (auto attr2 : f->attrs()) {
                                    if (auto anno2 = clang::dyn_cast<clang::AnnotateAttr>(attr2)) {
                                        std::string ann = anno2->getAnnotation().str();
                                        const std::string prefix = "aiepolicy:";
                                        if (ann.rfind(prefix, 0) == 0) {
                                            wantName = ann.substr(prefix.size());
                                            break;
                                        }
                                    }
                                }
                                if (Context) {
                                    for (auto *d : Context->getTranslationUnitDecl()->decls()) {
                                        auto *vd = clang::dyn_cast<clang::VarDecl>(d);
                                        if (!vd || vd->getName() != wantName)
                                            continue;
                                        auto *rd = vd->getType()->getAsCXXRecordDecl();
                                        if (!rd || rd->getName() != "GlobalPolicy")
                                            continue;
                                        const clang::APValue *apval = vd->getEvaluatedValue();
                                        if (!apval) {
                                            vd->evaluateValue();
                                            apval = vd->getEvaluatedValue();
                                        }
                                        if (apval && apval->isStruct() && apval->getStructNumFields() >= 1) {
                                            int v = (int)apval->getStructField(0).getInt().getExtValue();
                                            fca = (v != 0);
                                            found = true;
                                        }
                                        break;
                                    }
                                }
                                kernelFullConnectAuto[kernelName] = fca;
                                llvm::outs()
                                    << "[TilingLinalg] kernel " << kernelName << " fullconnect_auto=" << (fca ? 1 : 0)
                                    << (found ? " (from " + wantName + ")" : " (default)") << "\n";
                            }

                            // std::cout << "find global " << f->getNameInfo().getName().getAsString() << std::endl;
                            ParmVarDecl *param = f->getParamDecl(0);
							// llvm::outs() << "param->hasAttrs() is " << param->hasAttrs() << "\n";
							//llvm::outs << "range is " << range <<"\n";
							ExportFunction(f, kernelName);
							clang::SourceRange functionRange = f->getSourceRange();
							clang::SourceLocation startLocation = functionRange.getBegin();

							clang::SourceLocation lineStart = Rewrite->getSourceMgr().translateLineCol(
							Rewrite->getSourceMgr().getFileID(startLocation), 
							Rewrite->getSourceMgr().getSpellingLineNumber(startLocation), 1);
							Rewrite->RemoveText(f->getSourceRange());
                            // do other rewrite logic only after the RemoveText work done
                            // Pass streaming flag to frontend
                            Aiefrontend->createKernelDefinitionOp(f, Rewrite, lineStart, isStreamingKernel);
                            if (anno->getAnnotation() == "__global__") {
								std::string globalVars =
									"\n// Global variables for kernel: " + kernelName + "\n" +
									"extern unsigned char _binary_kernel_" + kernelName + "_start[];\n" +
									"extern unsigned char _binary_kernel_" + kernelName + "_end[];\n" +
									"extern unsigned int _binary_kernel_" + kernelName + "_size;\n\n";
								Rewrite->InsertText(lineStart, globalVars, true, true);
							} else if(anno->getAnnotation() == "__kernel__") {
                                // _start is declared in the generated template function; no additional globals needed here.
                            }
						} else {
							std::cerr << "not global func" << std::endl;
						}
						auto range = f->getSourceRange();
						auto& sc = Rewrite->getSourceMgr();
						// std::cout << "range.isValid() " << range.isValid() <<std::endl;
						// printSourceRange(range,  sc);
					}
					for (unsigned int i = 0; i < f->getNumParams(); ++i) {
						ParmVarDecl *param = f->getParamDecl(i);
						// llvm::outs() << "  Param " << (i + 1) << ": type: "
						// 	<< param->getType().getAsString() << "  name:"
						// 	<< param->getNameAsString() << "\n";
					}
				}
			}
			Stmt *body = f->getBody();
			TraverseStmt(body);

			 //PrintCode();
			 // This involves manipulating the AST node of the function
			 return true;
		 }
		 std::string GetCallExprString(const CallExpr *  CE) {
			 std::ostringstream ostr;
			 const FunctionDecl *Callee = CE->getDirectCallee();
			 if (Callee) {
				 ostr << Callee->getNameAsString() << "(";
				 
				 //const Expr *thirdArgExpr = CE->getArg(2)->IgnoreImpCasts();
				 //std::string functionName = "";
				 //if (const clang::StringLiteral *strLit = clang::dyn_cast<clang::StringLiteral>(thirdArgExpr)) {
					// functionName = strLit->getString().str();
					 // Use 'functionName' as needed...
				//	 std::cout<< "load kernel functionName " << functionName << std::endl; 
				 //}
				 auto num = CE->getNumArgs();			
				 for (unsigned i = 0; i < num; i++) {
					 const clang::Expr* arg = CE->getArg(i);
					 clang::QualType argType = arg->getType();
					//  std::cout << "x- Parameter " << i << " type: ";
					//  std::cout << argType.getAsString();
					//  std::cout << std::endl;
					 if (auto *DRE = dyn_cast<DeclRefExpr>(arg->IgnoreImpCasts())) {
						 if (const VarDecl *VD = dyn_cast<const VarDecl>(DRE->getDecl())) {
							 ostr << VD->getNameAsString() ;
							 ostr << ((i<num-1) ? ",":" ");
						 }
					 } else if (auto *CallAsArg = dyn_cast<CallExpr>(arg)) {
							 ostr << GetCallExprString(CallAsArg);
							 ostr << ((i<num-1) ?",":" ");
					 } else if(const clang::PointerType *ptrType = argType->getAs<clang::PointerType>()) {
						 // Check if the pointee type is char (indicating char* or const char*)
						 clang::QualType pointeeType = ptrType->getPointeeType();
						 if (pointeeType->isCharType()) {
							 std::string argStr;
							 llvm::raw_string_ostream os(argStr);
							 arg->printPretty(os, nullptr, PrintingPolicy(LangOptions()));
							//  llvm::outs() << "Pretty printed argument " << i << ": " << os.str() << "\n";
							 ostr<< os.str() <<((i<num-1) ? ",":" ");
							 /*
							 if (const DeclRefExpr *declRef = dyn_cast<const DeclRefExpr>(arg->IgnoreParenImpCasts())) {
								 const ValueDecl *valueDecl = declRef->getDecl();

								 // Get the name of the function or variable
								 std::string paramName = valueDecl->getNameAsString();
								 llvm::outs() << "Pointer parameter name: " << paramName << "\n";
							 }
							 // Evaluate the expression
							 //clang::Expr::EvalResult evalResult;
							 //if (arg->EvaluateAsRValue(evalResult, *Context)) {
							 ///	clang::APValue value = evalResult.Val;
								//auto str = clang::dyn_cast<clang::StringLiteral>(CE->getArg(i)->IgnoreImpCasts());
							  //ostr << str->getString().str() << ((i<num-1) ? ",":" ");
							 //char* charArray = const_cast<char*>(value.getCharPtr().getPointer());
							 //	std::cout << " it is string " << str->getString().str() << " " << value.getCharPtr().getPointer() << std::endl;*/
							 }
						} else if (const CStyleCastExpr *castExpr = dyn_cast<const CStyleCastExpr>(arg)) {
								 // Get the expression being cast
								 const Expr *subExpr = castExpr->getSubExpr();

								 // Check if it's a reference to a variable
								 if (const DeclRefExpr *declRef = dyn_cast<const DeclRefExpr>(subExpr)) {
									 // Get the variable declaration
									 const ValueDecl *valueDecl = declRef->getDecl();

									 // Retrieve and print the name of the variable
									//  llvm::outs() << "Variable name: " << valueDecl->getName() << "\n";
									 ostr<< valueDecl->getName().str() <<((i < num -1) ?",":" ");
								 }
						} else if (argType->isIntegerType()) {
						 clang::Expr::EvalResult xResult;
						 if (arg->EvaluateAsInt(xResult, Callee->getASTContext())) {
							 int v = xResult.Val.getInt().getExtValue();
							 ostr << v ;
							 ostr<< ((i < num -1) ?",":" ");
						 }
					 }else {
						std::string argStr;
						llvm::raw_string_ostream os(argStr);
						arg->printPretty(os, nullptr, PrintingPolicy(LangOptions()));
						ostr << os.str() << ((i < num - 1) ? "," : " ");
					 }
				 }
			 }

			 ostr<< ")";
			 return ostr.str();
		}
		 //*/

		 bool VisitCallExpr(const CallExpr * 	CE) {
			const FunctionDecl *Callee = CE->getDirectCallee();

			// Detect __aie_launch("kernelName", mesh, A, B, C, M, N, K) for tiling mode
			if (Callee && Callee->getNameAsString() == "__aie_launch") {
				isTilingLinalgMode = true;

                // Extract kernel name to check for duplicate AST visits
                std::string currentLaunchKernel;
                if (CE->getNumArgs() >= 1) {
                    if (const auto *SL = dyn_cast<clang::StringLiteral>(CE->getArg(0)->IgnoreParenImpCasts()))
                        currentLaunchKernel = SL->getString().str();
                }

                // Guard: skip if this exact kernel was already parsed (AST visitor may visit twice)
                // Important: do NOT clear globals before this check — a same-name kernel called
                // twice (e.g. matmul<<<mesh>>>(A,B,C); matmul<<<mesh>>>(C,B,A);) must not
                // clobber the state captured from the first visit.
                for (const auto &mk : parsedMeshKernels) {
                    if (mk.kernelName == currentLaunchKernel)
                        return true;
                }

                // For multi-kernel: each launch gets its own parsedTensors / derivedTilingParams.
                // Clear per-kernel state so this kernel starts fresh.
                // This must happen AFTER the duplicate check above.
                bool isFirstKernel = parsedMeshKernels.empty();
                parsedTensors.clear();
                derivedTilingParams = DerivedTilingParams();

                // Arg 0: kernel name (string literal after preprocessing)
				if (CE->getNumArgs() >= 2) {
                    if (!currentLaunchKernel.empty()) {
                        llvm::outs() << "[TilingLinalg] Detected kernel launch: " << currentLaunchKernel << "\n";
                    }
                    // Arg 1: aieDim or aieMesh variable — extract rows/cols
                    const Expr *meshArg = CE->getArg(1)->IgnoreParenImpCasts();

                    // Local vars for this kernel's mesh dims and partition
                    int localMeshRows = 0, localMeshCols = 0;
                    PartitionDesc localPartition;
                    // Helper: extract mesh dims from a CXXConstructExpr (aieDim(rows, cols [, part]))
                    auto extractDimsFromConstruct = [&](const CXXConstructExpr *Construct) {
                        if (Construct->getNumArgs() >= 2) {
                            clang::Expr::EvalResult r0, r1;
                            if (Construct->getArg(0)->EvaluateAsInt(r0, *Context) &&
								Construct->getArg(1)->EvaluateAsInt(r1, *Context)) {
                                localMeshRows = r0.Val.getInt().getExtValue();
                                localMeshCols = r1.Val.getInt().getExtValue();
                            }
                            if (Construct->getNumArgs() >= 3) {
                                extractPartitionFromExpr(Construct->getArg(2), Context);
                                localPartition = parsedPartition;
                            }
                        }
                    };

                    // Helper: extract mesh dims from an aieMesh VarDecl initialized via device.partition(part, rows,
                    // cols)
                    auto extractDimsFromAieMeshVar = [&](const VarDecl *VD) -> bool {
                        if (!VD->hasInit())
                            return false;
                        const Expr *init = VD->getInit()->IgnoreParenImpCasts();
                        // Handle ExprWithCleanups wrapper
                        if (const auto *EWC = dyn_cast<ExprWithCleanups>(init))
                            init = EWC->getSubExpr()->IgnoreParenImpCasts();
                        // Handle MaterializeTemporaryExpr wrapper
                        if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(init))
                            init = MTE->getSubExpr()->IgnoreParenImpCasts();
                        // device.partition({...}, rows, cols) is a CXXMemberCallExpr
                        if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(init)) {
                            if (const auto *MD = MCE->getMethodDecl()) {
                                if (MD->getNameAsString() == "partition" && MCE->getNumArgs() >= 3) {
                                    // Arg 0: aiePartition, Arg 1: rows, Arg 2: cols
                                    extractPartitionFromExpr(MCE->getArg(0), Context);
                                    localPartition = parsedPartition;
                                    clang::Expr::EvalResult r0, r1;
                                    if (MCE->getArg(1)->EvaluateAsInt(r0, *Context) &&
                                        MCE->getArg(2)->EvaluateAsInt(r1, *Context)) {
                                        localMeshRows = r0.Val.getInt().getExtValue();
                                        localMeshCols = r1.Val.getInt().getExtValue();
                                    }
                                    return true;
                                }
                            }
                        }
                        // Fallback: CXXConstructExpr (aieDim or aggregate init)
                        if (const auto *Construct = dyn_cast<CXXConstructExpr>(init)) {
                            extractDimsFromConstruct(Construct);
                            return localMeshRows > 0;
                        }
                        return false;
                    };

                    if (const auto *Construct = dyn_cast<CXXConstructExpr>(meshArg)) {
                        if (Construct->getNumArgs() >= 2) {
                            // Direct construction: __aie_launch("k", aieDim(4,4), ...)
                            extractDimsFromConstruct(Construct);
                        } else if (Construct->getNumArgs() == 1) {
                            // Copy/move constructor: __aie_launch("k", mesh, ...) where mesh is by-value
                            const Expr *inner = Construct->getArg(0)->IgnoreParenImpCasts();
                            if (const auto *DR = dyn_cast<DeclRefExpr>(inner)) {
                                if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
                                    if (!extractDimsFromAieMeshVar(VD)) {
                                        // Try aieDim ctor inside VarDecl init
                                        if (VD->hasInit()) {
                                            if (const auto *IC =
                                                    dyn_cast<CXXConstructExpr>(VD->getInit()->IgnoreParenImpCasts())) {
                                                extractDimsFromConstruct(IC);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if (const auto *DR = dyn_cast<DeclRefExpr>(meshArg)) {
                        // mesh variable already declared — get rows/cols from initializer
                        if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
                            if (!extractDimsFromAieMeshVar(VD)) {
                                // Fallback: try CXXConstructExpr (aieDim)
                                if (VD->hasInit()) {
                                    if (const auto *Construct =
                                            dyn_cast<CXXConstructExpr>(VD->getInit()->IgnoreParenImpCasts())) {
                                        extractDimsFromConstruct(Construct);
                                    }
                                }
                            }
                        }
                    }

                    if (localMeshRows > 0 && localMeshCols > 0) {
                        llvm::outs() << "[TilingLinalg] Mesh: " << localMeshRows << " x " << localMeshCols << "\n";
                        if (localPartition.isValid()) {
                            llvm::outs() << "[TilingLinalg] Partition: [" << localPartition.startCol << ","
                                         << localPartition.endCol << "," << localPartition.startRow << ","
                                         << localPartition.endRow << "]\n";
                        }
                    }

                    // Update globals for backward compat (first kernel only)
                    if (isFirstKernel) {
                        tilingMeshRows = localMeshRows;
                        tilingMeshCols = localMeshCols;
                        parsedPartition = localPartition;
                    }
                    // Extract tensor params from the __global__ kernel function signature
					// Args 2+ of __aie_launch correspond to the kernel function parameters
					std::string launchKernelName;
					if (const auto *SL2 = dyn_cast<clang::StringLiteral>(CE->getArg(0)->IgnoreParenImpCasts())) {
						launchKernelName = SL2->getString().str();
					}

					if (!launchKernelName.empty()) {
						auto it = globalKernelFuncs.find(launchKernelName);
						if (it != globalKernelFuncs.end()) {
							const FunctionDecl *kernelFD = it->second;
							unsigned numKernelParams = kernelFD->getNumParams();

                            // Semantic validation: check launch arg count vs kernel param count
                            // Extra launch args beyond kernel params are dimension scalars (M, N, K)
                            unsigned numLaunchArgs = CE->getNumArgs() - 2; // exclude name and mesh
                            if (numLaunchArgs < numKernelParams) {
                                llvm::errs() << "Error: kernel '" << launchKernelName
											 << "' declares " << numKernelParams << " parameters but launch provides "
											 << numLaunchArgs << " arguments.\n";
								llvm::errs() << "  __global__ void " << launchKernelName << "(...) has "
											 << numKernelParams << " params\n";
								llvm::errs() << "  " << launchKernelName << "<<<mesh>>>(...) provides "
											 << numLaunchArgs << " args\n";
								// Don't abort — continue with available info
                            } else if (numLaunchArgs > numKernelParams) {
                                llvm::outs() << "[TilingLinalg] " << (numLaunchArgs - numKernelParams)
                                             << " extra launch args detected (dimension scalars)\n";
                            }

                            // Helper lambda: check if a QualType is a spatial wrapper or pointer type
                            // (i.e. a tensor parameter, not a scalar dimension).
                            // Spatial wrappers like aie::row_broadcast_in<T*> are struct types
                            // (not pointer types), but they wrap a pointer inside.
                            auto isTensorParam = [](clang::QualType qt) -> bool {
                                if (qt->isPointerType())
                                    return true;
                                // Check for aie:: spatial wrapper struct (legacy aliases + new port<>)
                                std::string typeStr = qt.getUnqualifiedType().getAsString();
                                for (const char *tag :
                                     {"aie::row_broadcast_in", "aie::col_broadcast_in", "aie::tiled_in",
                                      "aie::row_major_out", "aie::col_major_out", "aie::row_reduce_out", "aie::port"}) {
                                    if (typeStr.find(tag) != std::string::npos)
                                        return true;
                                }
                                return false;
                            };

                            // First pass: collect scalar dimension values from kernel params
                            // Launch args start at index 2 (0=name, 1=mesh, 2..=kernel params)
                            std::vector<int64_t> scalarDimValues;
							for (unsigned i = 0; i < numKernelParams; ++i) {
								const ParmVarDecl *kp = kernelFD->getParamDecl(i);
								clang::QualType ptype = kp->getType();
                                if (!isTensorParam(ptype)) {
                                    // Scalar param — try to evaluate from launch arg
									unsigned launchArgIdx = i + 2;
									if (launchArgIdx < CE->getNumArgs()) {
										clang::Expr::EvalResult evalRes;
										if (CE->getArg(launchArgIdx)->EvaluateAsInt(evalRes, *Context)) {
											scalarDimValues.push_back(evalRes.Val.getInt().getExtValue());
										}
									}
                                }
                            }

                            // Second: collect extra launch args beyond kernel params as dimension scalars
                            // e.g. matmul<<<mesh>>>(A, B, C, M, N, K) — M,N,K are at indices numKernelParams+2..
                            for (unsigned i = numKernelParams; i < numLaunchArgs; ++i) {
                                unsigned launchArgIdx = i + 2;
                                if (launchArgIdx < CE->getNumArgs()) {
                                    clang::Expr::EvalResult evalRes;
                                    if (CE->getArg(launchArgIdx)->EvaluateAsInt(evalRes, *Context)) {
                                        int64_t val = evalRes.Val.getInt().getExtValue();
                                        scalarDimValues.push_back(val);
                                        llvm::outs() << "[TilingLinalg] Extra launch arg[" << (i - numKernelParams)
                                                     << "] = " << val << "\n";
                                    }
                                }
                            }

                            // Determine default shape from scalar dims (e.g. M, N, K -> MxN for each tensor)
							// For GEMM: A[M*K], B[K*N], C[M*N] with scalars [M, N, K]
							// General case: use first two scalars as shape for all tensors
							int64_t defaultDim0 = 16, defaultDim1 = 16;
							if (scalarDimValues.size() >= 2) {
								defaultDim0 = scalarDimValues[0];
								defaultDim1 = scalarDimValues[1];
							} else if (scalarDimValues.size() == 1) {
								defaultDim0 = defaultDim1 = scalarDimValues[0];
							}

                            // Map scalar dims to GEMM M, N, K when 3 values available
                            // (from either kernel params or extra launch args)
                            macroDimM = 0;
                            macroDimN = 0;
                            macroDimK = 0;
                            if (scalarDimValues.size() >= 3) {
                                macroDimM = scalarDimValues[0];
                                macroDimN = scalarDimValues[1];
                                macroDimK = scalarDimValues[2];
                                llvm::outs() << "[TilingLinalg] GEMM dims from launch args: M=" << macroDimM
                                             << " N=" << macroDimN << " K=" << macroDimK << "\n";
                                defaultDim0 = macroDimM;
                                defaultDim1 = macroDimK;
                            }
                            // Fallback: when no scalar dims found, try to extract M, N, K from
                            // user #define macros (e.g. #define M 32, #define K 32, #define N 32).
                            // These are the standard GEMM dimension macros used in the kernel source.
                            if (macroDimM == 0 && scalarDimValues.empty()) {
                                for (const auto &macroLine : userMacroDefines) {
                                    // Parse "#define <NAME> <integer>"
                                    std::istringstream mss(macroLine);
                                    std::string tok_define, tok_name, tok_value;
                                    mss >> tok_define >> tok_name >> tok_value;
                                    if (tok_define != "#define" || tok_name.empty() || tok_value.empty())
                                        continue;
                                    // Only accept pure integer values (no expressions)
                                    bool allDigits = true;
                                    for (char c : tok_value) {
                                        if (!std::isdigit(c)) {
                                            allDigits = false;
                                            break;
                                        }
                                    }
                                    if (!allDigits)
                                        continue;
                                    int64_t val = std::stoll(tok_value);
                                    if (tok_name == "M")
                                        macroDimM = val;
                                    else if (tok_name == "N")
                                        macroDimN = val;
                                    else if (tok_name == "K")
                                        macroDimK = val;
                                }
                                if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    llvm::outs() << "[TilingLinalg] GEMM dims from macros: M=" << macroDimM
                                                 << " N=" << macroDimN << " K=" << macroDimK << "\n";
                                    // Update defaults (used when no per-tensor shape can be determined)
                                    defaultDim0 = macroDimM;
                                    defaultDim1 = macroDimK;
                                } else if (macroDimM > 0) {
                                    // Partial: at least M is defined
                                    defaultDim0 = macroDimM;
                                    defaultDim1 = macroDimN > 0 ? macroDimN : macroDimM;
                                }
                            }

                            // Second pass: build ParsedTensorInfo for each tensor parameter
                            // Handles both bare pointer types (input_window_int8*) and
                            // spatial wrapper types (aie::row_broadcast_in<input_window_int8 *>)
                            // Per-port act/wgt resolution: track input-port ordinal so the
                            // 1st input -> map.act, 2nd+ inputs -> map.wgt.
                            int inputOrd = 0;
                            for (unsigned i = 0; i < numKernelParams; ++i) {
								const ParmVarDecl *kp = kernelFD->getParamDecl(i);
								clang::QualType ptype = kp->getType();
                                if (!isTensorParam(ptype))
                                    continue;

                                // Get the full type string for spatial tag extraction
                                std::string fullTypeStr = ptype.getUnqualifiedType().getAsString();

                                // Check for spatial type wrapper: aie::tag_name<inner_type>
                                // Also handles aie::port<inner_type, PolicyName>
                                std::string spatialTag;
                                std::string innerTypeStr;
                                std::string policyName;
                                static const char *spatialTags[] = {
                                    "row_broadcast_in", "col_broadcast_in", "tiled_in", "row_major_out",
                                    "col_major_out",    "row_reduce_out",   "port"};
                                for (const char *tag : spatialTags) {
                                    std::string prefix = std::string("aie::") + tag + "<";
                                    size_t pos = fullTypeStr.find(prefix);
                                    if (pos != std::string::npos) {
                                        // Extract everything between < and matching >
                                        size_t innerStart = pos + prefix.size();
                                        int depth = 1;
                                        size_t innerEnd = innerStart;
                                        while (innerEnd < fullTypeStr.size() && depth > 0) {
                                            if (fullTypeStr[innerEnd] == '<')
                                                depth++;
                                            else if (fullTypeStr[innerEnd] == '>')
                                                depth--;
                                            if (depth > 0)
                                                innerEnd++;
                                        }
                                        std::string templateArgs =
                                            fullTypeStr.substr(innerStart, innerEnd - innerStart);

                                        if (std::string(tag) == "port") {
                                            // aie::port<InnerType, PolicyName>
                                            // Split on top-level comma to get inner type and policy name
                                            int commaDepth = 0;
                                            size_t commaPos = std::string::npos;
                                            for (size_t ci = 0; ci < templateArgs.size(); ++ci) {
                                                if (templateArgs[ci] == '<')
                                                    commaDepth++;
                                                else if (templateArgs[ci] == '>')
                                                    commaDepth--;
                                                else if (templateArgs[ci] == ',' && commaDepth == 0) {
                                                    commaPos = ci;
                                                    break;
                                                }
                                            }
                                            if (commaPos != std::string::npos) {
                                                innerTypeStr = templateArgs.substr(0, commaPos);
                                                policyName = templateArgs.substr(commaPos + 1);
                                                // Trim whitespace
                                                while (!policyName.empty() && policyName.front() == ' ')
                                                    policyName.erase(0, 1);
                                                while (!policyName.empty() && policyName.back() == ' ')
                                                    policyName.pop_back();
                                                // Strip "aie::" namespace prefix if present
                                                if (policyName.substr(0, 5) == "aie::")
                                                    policyName = policyName.substr(5);
                                            } else {
                                                // aie::port<InnerType> with default policy
                                                innerTypeStr = templateArgs;
                                                policyName = "RowBC"; // default policy
                                            }
                                            // Don't set spatialTag for port — policyName is used instead
                                            llvm::outs() << "[TilingLinalg] Policy port: " << policyName
                                                         << " -> inner type: " << innerTypeStr << "\n";
                                        } else {
                                            spatialTag = tag;
                                            innerTypeStr = templateArgs;
                                            llvm::outs() << "[TilingLinalg] Spatial tag: " << spatialTag
                                                         << " -> inner type: " << innerTypeStr << "\n";
                                        }

                                        // Trim whitespace from innerTypeStr
                                        while (!innerTypeStr.empty() && innerTypeStr.front() == ' ')
                                            innerTypeStr.erase(0, 1);
                                        while (!innerTypeStr.empty() && innerTypeStr.back() == ' ')
                                            innerTypeStr.pop_back();
                                        // Remove trailing * if present (the pointer is inside the wrapper)
                                        while (!innerTypeStr.empty() && innerTypeStr.back() == '*') {
                                            innerTypeStr.pop_back();
                                            while (!innerTypeStr.empty() && innerTypeStr.back() == ' ')
                                                innerTypeStr.pop_back();
                                        }
                                        break;
                                    }
                                }

                                // For bare pointer types (no spatial tag or policy), use the pointee type string
                                if (spatialTag.empty() && policyName.empty() && ptype->isPointerType()) {
                                    clang::QualType pointee = ptype->getPointeeType();
                                    innerTypeStr = pointee.getUnqualifiedType().getAsString();
                                }

                                // Determine input/output and bit width from inner type string
                                bool isWindowParam = false;
                                bool isInputWindow = false;
                                int bitWidth = 32; // default

                                if (innerTypeStr.find("input_window") != std::string::npos) {
                                    isWindowParam = true;
                                    isInputWindow = true;
                                } else if (innerTypeStr.find("output_window") != std::string::npos) {
                                    isWindowParam = true;
                                    isInputWindow = false;
                                }

                                if (isWindowParam) {
                                    if (innerTypeStr.find("int8") != std::string::npos ||
                                        innerTypeStr.find("signed char") != std::string::npos ||
                                        innerTypeStr.find("char") != std::string::npos)
                                        bitWidth = 8;
                                    else if (innerTypeStr.find("int16") != std::string::npos ||
                                             innerTypeStr.find("short") != std::string::npos)
                                        bitWidth = 16;
                                } else if (ptype->isPointerType()) {
                                    // Plain pointer type (e.g. const int32_t*)
                                    clang::QualType pointee = ptype->getPointeeType();
                                    if (pointee->isSpecificBuiltinType(clang::BuiltinType::Int))
                                        bitWidth = 32;
                                    else if (pointee->isSpecificBuiltinType(clang::BuiltinType::Short))
                                        bitWidth = 16;
                                    else if (pointee->isSpecificBuiltinType(clang::BuiltinType::SChar) ||
                                             pointee->isSpecificBuiltinType(clang::BuiltinType::Char_S) ||
                                             pointee->isSpecificBuiltinType(clang::BuiltinType::UChar))
                                        bitWidth = 8;
                                    else if (const auto *BT = pointee->getAs<clang::BuiltinType>()) {
                                        bitWidth = Context->getTypeSize(BT);
                                    }
                                }

                                // Determine input/output from spatial tag, policy name, or type info
                                bool isInput;
                                // Determine input/output: always prefer inner type info
                                if (isWindowParam) {
                                    isInput = isInputWindow;
                                } else if (!spatialTag.empty()) {
                                    isInput = (spatialTag.find("_in") != std::string::npos);
                                } else if (ptype->isPointerType()) {
                                    isInput = ptype->getPointeeType().isConstQualified();
                                } else {
                                    isInput = true; // conservative default
                                }

                                ParsedTensorInfo pti;
                                pti.varName = kp->getNameAsString();
                                // Assign per-tensor GEMM shapes from M/N/K macros.
                                // Shape is initially set to defaults; re-assigned after AST
                                // resolution using resolved policy fields (pattern+distribution).
                                if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    bool shapeAssigned = false;
                                    if (!spatialTag.empty()) {
                                        if (spatialTag == "row_broadcast_in") {
                                            pti.shape = {macroDimM, macroDimK};
                                            shapeAssigned = true;
                                        } else if (spatialTag == "col_broadcast_in") {
                                            pti.shape = {macroDimK, macroDimN};
                                            shapeAssigned = true;
                                        } else if (spatialTag == "row_major_out" || spatialTag == "col_major_out") {
                                            pti.shape = {macroDimM, macroDimN};
                                            shapeAssigned = true;
                                        }
                                    }
                                    if (!shapeAssigned)
                                        pti.shape = {defaultDim0, defaultDim1};
                                } else {
                                    pti.shape = {defaultDim0, defaultDim1};
                                }
                                pti.elementBitWidth = bitWidth;
                                pti.isInput = isInput;
                                pti.spatialTag = spatialTag;
                                pti.policyName = policyName;
                                // Input-port ordinal: 0 for the 1st input port (act),
                                // 1+ for subsequent input ports (wgt); -1 for outputs.
                                int thisInputIdx = pti.isInput ? inputOrd++ : -1;

                                // --- AST-based SpatialPolicy struct extraction ---
                                // Extract the constexpr SpatialPolicy struct fields from the
                                // template argument of aie::port<T, Policy>.
                                //
                                // After template instantiation, ptype is a RecordType (the
                                // instantiated aie::port<T,P> struct). We reach the template
                                // args via ClassTemplateSpecializationDecl.
                                //
                                // The policy argument kind depends on the LLVM version:
                                //   - StructuralValue: the APValue is stored directly
                                //   - Declaration: a reference to the constexpr VarDecl;
                                //     we evaluate it to get the APValue
                                if (!policyName.empty()) {
                                    const clang::Type *rawType = ptype.getTypePtr();
                                    rawType = rawType->getUnqualifiedDesugaredType();

                                    const clang::CXXRecordDecl *recDecl = rawType->getAsCXXRecordDecl();
                                    if (auto *ctsd =
                                            dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(recDecl)) {
                                        const auto &targs = ctsd->getTemplateArgs();
                                        if (targs.size() >= 2) {
                                            const auto &policyArg = targs[1];
                                            const APValue *apval = nullptr;

                                            if (policyArg.getKind() == clang::TemplateArgument::StructuralValue) {
                                                apval = &policyArg.getAsStructuralValue();
                                            } else if (policyArg.getKind() == clang::TemplateArgument::Declaration) {
                                                // C++20 struct NTTP: Clang stores the evaluated value as a
                                                // TemplateParamObjectDecl (a unique'd APValue holder).
                                                ValueDecl *decl = policyArg.getAsDecl();
                                                if (auto *tpo = dyn_cast<clang::TemplateParamObjectDecl>(decl)) {
                                                    apval = &tpo->getValue();
                                                } else if (auto *vd = dyn_cast<clang::VarDecl>(decl)) {
                                                    apval = vd->getEvaluatedValue();
                                                    if (!apval)
                                                        vd->evaluateValue(), apval = vd->getEvaluatedValue();
                                                }
                                            } else if (policyArg.getKind() == clang::TemplateArgument::Expression) {
                                                // Fallback: evaluate the expression directly
                                                clang::Expr::EvalResult evalRes;
                                                if (policyArg.getAsExpr()->EvaluateAsConstantExpr(evalRes, *Context)) {
                                                    apval = &evalRes.Val;
                                                }
                                            }

                                            // Resolve the Space record name so we can disambiguate
                                            // GemmSpace vs Conv2dSpace robustly (field counts are a
                                            // fallback only).
                                            std::string spaceTypeName;
                                            {
                                                clang::QualType spaceQT;
                                                if (policyArg.getKind() == clang::TemplateArgument::StructuralValue) {
                                                    spaceQT = policyArg.getStructuralValueType();
                                                } else if (policyArg.getKind() ==
                                                           clang::TemplateArgument::Declaration) {
                                                    ValueDecl *decl = policyArg.getAsDecl();
                                                    if (auto *tpo = dyn_cast<clang::TemplateParamObjectDecl>(decl))
                                                        spaceQT = tpo->getType();
                                                    else if (auto *vd = dyn_cast<clang::VarDecl>(decl))
                                                        spaceQT = vd->getType();
                                                }
                                                if (!spaceQT.isNull())
                                                    if (auto *rd = spaceQT->getAsCXXRecordDecl())
                                                        spaceTypeName = rd->getName().str();
                                            }

                                            auto readTileDim = [](const APValue &f, ParsedTensorInfo::TileDim &out) {
                                                if (!f.isStruct())
                                                    return;
                                                int nf = f.getStructNumFields();
                                                // Field order (contractual for AST extraction):
                                                //   0 fullsize   -> internal base (full PADDED dim length)
                                                //   1 tile_round -> internal groups (explicit outer rounds)
                                                //   2 tile_size  -> internal size  (outer per-tile slice)
                                                //   3 stride     -> internal stride
                                                //   4 padsize    -> internal padSize (metadata/boundary only)
                                                //   5 slice_tiling (nested tile_level)
                                                if (nf >= 1)
                                                    out.base =
                                                        (int)f.getStructField(0).getInt().getExtValue(); // fullsize
                                                if (nf >= 2)
                                                    out.groups =
                                                        (int)f.getStructField(1).getInt().getExtValue(); // tile_round
                                                if (nf >= 3)
                                                    out.size =
                                                        (int)f.getStructField(2).getInt().getExtValue(); // tile_size
                                                if (nf >= 4)
                                                    out.stride = (int)f.getStructField(3).getInt().getExtValue();
                                                if (nf >= 5)
                                                    out.padSize = (int)f.getStructField(4).getInt().getExtValue();
                                                // Nested slice_tiling (struct field 5): a by-value inner
                                                // tile_level { int tile_size; int stride; int rounds; }
                                                // mirroring #routing.level. Realizes the second-level
                                                // (on-core temporal) split on the SAME axis as this dim.
                                                if (nf >= 6 && f.getStructField(5).isStruct()) {
                                                    const APValue &l2 = f.getStructField(5);
                                                    int l2nf = l2.getStructNumFields();
                                                    if (l2nf >= 1)
                                                        out.l2Size = (int)l2.getStructField(0).getInt().getExtValue();
                                                    if (l2nf >= 2)
                                                        out.l2Stride = (int)l2.getStructField(1).getInt().getExtValue();
                                                    if (l2nf >= 3)
                                                        out.l2Groups = (int)l2.getStructField(2).getInt().getExtValue();
                                                }
                                                // Derive groups from base (full padded dim length) when the
                                                // user did NOT pin tile_round, so downstream `.groups` reads
                                                // stay valid. Explicit tile_round wins.
                                                if (out.groups <= 0 && out.base > 0 && out.size > 0) {
                                                    int st = out.stride > 0 ? out.stride : out.size;
                                                    out.groups = (out.base - out.size + st - 1) / st + 1;
                                                }
                                                // Derive L2 rounds when not explicitly given but a
                                                // slice_tiling size/stride was provided: cover the parent
                                                // tile `size` rows. overlap = l2Size - l2Stride.
                                                if (out.l2Groups <= 0 && out.l2Size > 0 && out.size > 0) {
                                                    int l2st = out.l2Stride > 0 ? out.l2Stride : out.l2Size;
                                                    out.l2Groups = (out.size - out.l2Size + l2st - 1) / l2st + 1;
                                                }
                                                // Coverage validation: L2 rounds must tile the parent
                                                // outer slice exactly. (l2Groups-1)*l2Stride + l2Size == size.
                                                if (out.l2Groups > 1 && out.l2Size > 0) {
                                                    int l2st = out.l2Stride > 0 ? out.l2Stride : out.l2Size;
                                                    int covered = (out.l2Groups - 1) * l2st + out.l2Size;
                                                    if (covered != out.size) {
                                                        llvm::errs() << "[slice_tiling] WARNING: L2 coverage mismatch: "
                                                                     << "(rounds-1)*stride + size = " << covered
                                                                     << " != parent tile size " << out.size << "\n";
                                                    }
                                                }
                                            };
                                            // Read the 3-part SpatialPolicy from a struct APValue
                                            // (used for both bare SpatialPolicy and the nested
                                            // policy field of a composed Space). Nested layout:
                                            //   field 0 map  { 0 act, 1 wgt, 2 layout, 3 merge_order }
                                            //   field 1 mat  { 0 pad, 1 im2col }
                                            //   field 2 sched{ 0 pp_depth, 1 l1_budget{ 0 value } }
                                            // Mapping onto the internal pti fields:
                                            //   pattern  <- role-resolved (output->Gather; input
                                            //               ordinal 0->act, 1+->wgt)
                                            //   distribution <- map.layout
                                            //   mergeOrder   <- map.merge_order
                                            //   layoutTransform <- mat.im2col==Dma ? DmaShuffle : None
                                            //   pingPong     <- sched.pp_depth
                                            //   maxBufferBytes <- sched.l1_budget.value
                                            //   requireFullCoverage <- true (default)
                                            auto readPolicy = [&](const APValue &p) {
                                                if (!p.isStruct())
                                                    return;
                                                int nf = p.getStructNumFields();
                                                // (1) map
                                                if (nf >= 1 && p.getStructField(0).isStruct()) {
                                                    const APValue &m = p.getStructField(0);
                                                    int mf = m.getStructNumFields();
                                                    int actPat =
                                                        mf >= 1 ? (int)m.getStructField(0).getInt().getExtValue() : 0;
                                                    int wgtPat =
                                                        mf >= 2 ? (int)m.getStructField(1).getInt().getExtValue() : 0;
                                                    if (mf >= 3)
                                                        pti.distribution =
                                                            (int)m.getStructField(2).getInt().getExtValue();
                                                    if (mf >= 4)
                                                        pti.mergeOrder =
                                                            (int)m.getStructField(3).getInt().getExtValue();
                                                    // Two-level mesh tiling axes (map fields 4/5).
                                                    if (mf >= 5)
                                                        pti.meshTilingGroup1Dim =
                                                            (int)m.getStructField(4).getInt().getExtValue();
                                                    if (mf >= 6)
                                                        pti.meshTilingGroup2Dim =
                                                            (int)m.getStructField(5).getInt().getExtValue();
                                                    // Role-aware pattern resolution.
                                                    if (!pti.isInput)
                                                        pti.pattern = 3; // Gather
                                                    else
                                                        pti.pattern = (thisInputIdx == 0) ? actPat : wgtPat;
                                                }
                                                // (2) mat
                                                if (nf >= 2 && p.getStructField(1).isStruct()) {
                                                    const APValue &mat = p.getStructField(1);
                                                    int matf = mat.getStructNumFields();
                                                    int im2col = matf >= 2
                                                                     ? (int)mat.getStructField(1).getInt().getExtValue()
                                                                     : 0;
                                                    pti.layoutTransform = (im2col == 1) ? 1 /*DmaShuffle*/ : 0;
                                                }
                                                // (3) sched
                                                if (nf >= 3 && p.getStructField(2).isStruct()) {
                                                    const APValue &s = p.getStructField(2);
                                                    int sf = s.getStructNumFields();
                                                    if (sf >= 1)
                                                        pti.pingPong = (int)s.getStructField(0).getInt().getExtValue();
                                                    if (sf >= 2 && s.getStructField(1).isStruct() &&
                                                        s.getStructField(1).getStructNumFields() >= 1)
                                                        pti.maxBufferBytes = (int)s.getStructField(1)
                                                                                 .getStructField(0)
                                                                                 .getInt()
                                                                                 .getExtValue();
                                                }
                                                pti.requireFullCoverage = true;
                                            };

                                            if (apval && apval->isStruct() && apval->getStructNumFields() >= 1) {
                                                // Detect composed Space by the resolved record name.
                                                // Field-0-is-struct no longer distinguishes them: a
                                                // bare 3-part SpatialPolicy ALSO has a struct field 0
                                                // (map). GemmSpace/Conv2dSpace -> composed; bare
                                                // SpatialPolicy -> not. Fallback: a bare Policy has
                                                // exactly 3 fields (map/mat/sched), so nf>=4 ==> composed.
                                                bool composed;
                                                if (spaceTypeName == "GemmSpace" || spaceTypeName == "Conv2dSpace")
                                                    composed = true;
                                                else if (spaceTypeName == "SpatialPolicy")
                                                    composed = false;
                                                else
                                                    composed = apval->getStructNumFields() >= 4;
                                                if (spaceTypeName == "Conv2dSpace_Spatial") {
                                                    // Declarative spatial-halo conv space:
                                                    //   field 0 geom, 1 out_tile_h, 2 out_tile_w,
                                                    //   3 objective, 4 policy (policy is field 4, NOT 0).
                                                    // Derive the spatial-halo split deterministically
                                                    // from raw geometry + the desired output tile,
                                                    // reproducing the legacy GemmSpace d1/d2/d3 halo
                                                    // output EXACTLY (see aiehlc.cc GemmSpace d1 path):
                                                    //   halo_slice = (out_tile_h-1)*S + K - pad_hi
                                                    //   halo_step  = halo_slice - (K - S)
                                                    //   raw_h  = in_h, raw_wc = in_w * cin, mode=Overlap.
                                                    // Done INLINE here (mirroring the GemmSpace d1 lift)
                                                    // — do NOT set fromConvSpace (that triggers the
                                                    // legacy Conv2dSpace post-extraction block).
                                                    readPolicy(apval->getStructField(4));
                                                    const APValue &g = apval->getStructField(0);
                                                    auto gInt = [&](int idx) -> int {
                                                        return (g.isStruct() && g.getStructNumFields() > idx)
                                                                   ? (int)g.getStructField(idx).getInt().getExtValue()
                                                                   : 0;
                                                    };
                                                    int in_h = gInt(0);
                                                    int in_w = gInt(1);
                                                    int cin = gInt(2);
                                                    // cout = gInt(3) — metadata only
                                                    int K = gInt(4);
                                                    int S = gInt(5);
                                                    if (S <= 0)
                                                        S = 1;
                                                    // pad_lo = gInt(6) — metadata only
                                                    int pad_hi = gInt(7);
                                                    // cin_aligned = gInt(8) — channel layout stride.
                                                    // When > cin, the host DDR pre-pads the input with
                                                    // zero channels so the channel-layout dimension is
                                                    // aligned; raw_wc must use the aligned C so every
                                                    // derived stride/length is consistent. Correctness
                                                    // is preserved because padded input and filter
                                                    // channels are both zero.
                                                    int cin_aligned = gInt(8);
                                                    int c_eff = (cin_aligned > cin) ? cin_aligned : cin;
                                                    int oth = (int)apval->getStructField(1).getInt().getExtValue();
                                                    // out_tile_w (field 2), objective (field 3) -> metadata
                                                    int halo_slice = (oth - 1) * S + K - pad_hi;
                                                    int halo_step = halo_slice - (K - S);
                                                    pti.tileMode = 1;
                                                    pti.shimDma.mode = 1;
                                                    pti.shimDma.halo_slice = halo_slice;
                                                    pti.shimDma.halo_step = halo_step;
                                                    pti.shimDma.split_dim = 0;
                                                    pti.shimDma.raw_h = in_h;
                                                    pti.shimDma.raw_wc = in_w * c_eff;
                                                    llvm::outs()
                                                        << "[TilingLinalg] Conv2dSpace_Spatial derived: "
                                                           "halo_slice="
                                                        << halo_slice << " halo_step=" << halo_step << " raw_h=" << in_h
                                                        << " raw_wc=" << in_w * c_eff << "\n";
                                                    // Explicit halo descriptor override (Stage A):
                                                    // fields 5 d1 (height halo), 6 d2 (width*C halo).
                                                    // When d1.tile_size>0 the user has pinned the halo
                                                    // split directly — override the out_tile_h/out_tile_w
                                                    // derivation above. d1 = HEIGHT halo (mesh ROWS):
                                                    //   halo_slice = d1.tile_size, halo_step = d1.stride,
                                                    //   raw_h = d1.base (full padded input H).
                                                    // d2 = WIDTH*C halo (mesh COLS); for Stage A d2 is
                                                    //   non-split (stride==size) so raw_wc = d2.base.
                                                    ParsedTensorInfo::TileDim sd1, sd2;
                                                    int snf = apval->getStructNumFields();
                                                    if (snf >= 6)
                                                        readTileDim(apval->getStructField(5), sd1);
                                                    if (snf >= 7)
                                                        readTileDim(apval->getStructField(6), sd2);
                                                    if (sd1.size > 0) {
                                                        pti.shimDma.halo_slice = sd1.size;
                                                        pti.shimDma.halo_step = sd1.stride > 0 ? sd1.stride : sd1.size;
                                                        if (sd1.base > 0)
                                                            pti.shimDma.raw_h = sd1.base;
                                                        else if (sd1.groups > 0)
                                                            pti.shimDma.raw_h =
                                                                (sd1.groups - 1) * pti.shimDma.halo_step + sd1.size;
                                                        if (sd2.base > 0)
                                                            pti.shimDma.raw_wc = sd2.base;
                                                        else if (sd2.size > 0)
                                                            pti.shimDma.raw_wc = sd2.size;
                                                        llvm::outs()
                                                            << "[TilingLinalg] Conv2dSpace_Spatial d1/d2 override: "
                                                               "halo_slice="
                                                            << pti.shimDma.halo_slice
                                                            << " halo_step=" << pti.shimDma.halo_step
                                                            << " raw_h=" << pti.shimDma.raw_h
                                                            << " raw_wc=" << pti.shimDma.raw_wc << "\n";
                                                    }
                                                } else if (composed) {
                                                    readPolicy(apval->getStructField(0));
                                                    bool isConv;
                                                    if (spaceTypeName == "Conv2dSpace")
                                                        isConv = true;
                                                    else if (spaceTypeName == "GemmSpace")
                                                        isConv = false;
                                                    else
                                                        // GemmSpace now has 7 fields (policy,m,n,k,d1,d2,d3)
                                                        // and Conv2dSpace has 10, so the field-count fallback
                                                        // must use >= 8 to avoid misclassifying GemmSpace.
                                                        isConv = apval->getStructNumFields() >= 8;

                                                    if (isConv) {
                                                        // Conv2dSpace: 1 ih, 2 iw, 3 ic, 4 oc,
                                                        // 5 kh, 6 kw, 7 stride, 8 pad. ih/iw are the
                                                        // EXACT input spatial dims (lossless source).
                                                        ParsedTensorInfo::TileDim ih, iw, ic, oc, kh, kw;
                                                        int nf = apval->getStructNumFields();
                                                        if (nf >= 2)
                                                            readTileDim(apval->getStructField(1), ih);
                                                        if (nf >= 3)
                                                            readTileDim(apval->getStructField(2), iw);
                                                        if (nf >= 4)
                                                            readTileDim(apval->getStructField(3), ic);
                                                        if (nf >= 5)
                                                            readTileDim(apval->getStructField(4), oc);
                                                        if (nf >= 6)
                                                            readTileDim(apval->getStructField(5), kh);
                                                        if (nf >= 7)
                                                            readTileDim(apval->getStructField(6), kw);
                                                        int convStride = 1, convPad = 0;
                                                        if (nf >= 8)
                                                            convStride =
                                                                (int)apval->getStructField(7).getInt().getExtValue();
                                                        if (nf >= 9)
                                                            convPad =
                                                                (int)apval->getStructField(8).getInt().getExtValue();
                                                        if (convStride <= 0)
                                                            convStride = 1;
                                                        // Input geometry is the exact source; derive
                                                        // output via the forward conv formula:
                                                        //   OW = (IW + 2P - KW)/S + 1, OH analogous.
                                                        int IW = iw.size, IH = ih.size;
                                                        int KW = kw.size, KH = kh.size;
                                                        int C = ic.size;
                                                        int OW = (IW + 2 * convPad - KW) / convStride + 1;
                                                        int OH = (IH + 2 * convPad - KH) / convStride + 1;
                                                        pti.shimDma.fromConvSpace = true;
                                                        pti.shimDma.kernel_h = KH;
                                                        pti.shimDma.kernel_w = KW;
                                                        pti.shimDma.input_c = C;
                                                        pti.shimDma.stride = convStride;
                                                        pti.shimDma.pad = convPad;
                                                        pti.shimDma.ow = OW;
                                                        pti.shimDma.conv_oh = OH;
                                                        pti.shimDma.conv_oc = oc.size;
                                                        pti.shimDma.conv_iw = IW;
                                                        pti.shimDma.conv_ih = IH;
                                                        // Field 9 (m): optional explicit spatial-halo
                                                        // split. When present (size>0) it pins the
                                                        // per-tile IFM slab directly, bypassing the
                                                        // OH/HW_ROWS auto-derivation. size=halo slice
                                                        // (input rows/tile), stride=halo step (input-row
                                                        // stride between tiles), groups=number of
                                                        // tile-rows. Stored via ShimDma.haloExplicit so
                                                        // it does NOT touch tdM (no GEMM-tiling side
                                                        // effects).
                                                        if (nf >= 10) {
                                                            ParsedTensorInfo::TileDim mSplit;
                                                            readTileDim(apval->getStructField(9), mSplit);
                                                            // Derive tileMode (Overlap) from an
                                                            // overlapping explicit halo split.
                                                            if (mSplit.size > 0 && mSplit.stride > 0 &&
                                                                mSplit.stride < mSplit.size)
                                                                pti.tileMode = 1;
                                                            if (mSplit.size > 0) {
                                                                pti.shimDma.halo_slice = mSplit.size;
                                                                pti.shimDma.halo_step =
                                                                    mSplit.stride > 0 ? mSplit.stride : mSplit.size;
                                                                pti.shimDma.oh_per_row =
                                                                    mSplit.groups > 0
                                                                        ? OH / mSplit.groups
                                                                        : (mSplit.size - KH) / convStride + 1;
                                                                pti.shimDma.haloExplicit = true;
                                                                llvm::outs()
                                                                    << "[TilingLinalg] Conv2dSpace explicit "
                                                                       "halo: halo_slice="
                                                                    << pti.shimDma.halo_slice
                                                                    << " halo_step=" << pti.shimDma.halo_step
                                                                    << " oh_per_row=" << pti.shimDma.oh_per_row << "\n";
                                                            }
                                                        }
                                                    } else {
                                                        // GemmSpace: 1 m, 2 n, 3 k (legacy/global GEMM dims,
                                                        // also conv-halo); 4 d1, 5 d2 (per-port 2D, role-aware).
                                                        int nf = apval->getStructNumFields();
                                                        if (nf >= 2)
                                                            readTileDim(apval->getStructField(1), pti.tdM);
                                                        if (nf >= 3)
                                                            readTileDim(apval->getStructField(2), pti.tdN);
                                                        if (nf >= 4)
                                                            readTileDim(apval->getStructField(3), pti.tdK);
                                                        if (nf >= 5)
                                                            readTileDim(apval->getStructField(4), pti.tdD1);
                                                        if (nf >= 6)
                                                            readTileDim(apval->getStructField(5), pti.tdD2);
                                                        if (nf >= 7)
                                                            readTileDim(apval->getStructField(6), pti.tdD3);
                                                        if (nf >= 8)
                                                            readTileDim(apval->getStructField(7), pti.tdD4);
                                                        // Per-port 2D path: when d1/d2 carry sizes the port
                                                        // describes its own matrix. We map d1/d2 into tdM/tdN/tdK
                                                        // role-aware (by distribution/IO direction) so the rest
                                                        // of the derivation (explicitTile*, rounds) is unchanged,
                                                        // and the legacy m/n/k reads above are overwritten.
                                                        if (pti.tdD1.size > 0 || pti.tdD2.size > 0 ||
                                                            pti.tdD3.size > 0 || pti.tdD4.size > 0) {
                                                            pti.perPort2D = true;
                                                        }
                                                        // Derive tileMode (Overlap) from the split dim:
                                                        // the policy no longer carries an explicit `mode`,
                                                        // so an overlapping split (stride < size) on the
                                                        // primary split dim (d1, else m) ==> Overlap(1).
                                                        if ((pti.tdD1.size > 0 && pti.tdD1.stride > 0 &&
                                                             pti.tdD1.stride < pti.tdD1.size) ||
                                                            (pti.tdM.size > 0 && pti.tdM.stride > 0 &&
                                                             pti.tdM.stride < pti.tdM.size))
                                                            pti.tileMode = 1;
                                                        else
                                                            pti.tileMode = 0;
                                                        // Spatial-halo via per-port 2D GemmSpace (d1/d2): a
                                                        // 2D operand has only 2 dims, so the halo input is
                                                        // described via d1/d2 instead of legacy m/n/k.
                                                        //   d1 = row split (size=halo_slice, stride=halo_step,
                                                        //        groups=tile-rows); an OVERLAPPING split means
                                                        //        stride < size.
                                                        //   d2 = raw_wc (input row width W*C -> A shape dim 1)
                                                        //   raw_h is DERIVED from coverage:
                                                        //        (groups-1)*stride + size.
                                                        // After lifting we CLEAR tdD1/tdD2 and perPort2D for
                                                        // this port so the halo input does NOT pollute the
                                                        // im2col-GEMM M/N/K tiling (role-based d1/d2 map,
                                                        // explicitTile*, working-set, validateDim), mirroring
                                                        // the legacy m/n/k path below.
                                                        if (pti.tileMode == 1 && pti.tdD1.size > 0 &&
                                                            pti.tdD1.stride > 0 && pti.tdD1.stride < pti.tdD1.size) {
                                                            // Pass-through spatial-halo: the authored d1/d2/d3
                                                            // descriptors map 1:1 onto the #routing.level tiling
                                                            // (no conv OH/kernel-geometry re-derivation). d1 is the
                                                            // HEIGHT (mesh-ROW) outer halo and its nested
                                                            // slice_tiling the on-core L2 rounds; d2 is the WIDTH
                                                            // (mesh-COL) K-accum split; d3 the (unsplit) channel
                                                            // layout stride. Downstream lifts w_slice/w_step ->
                                                            // kSlice/kStep (scaled by C) and l2* -> slice_tiling.
                                                            pti.shimDma.mode = 1;
                                                            pti.shimDma.split_dim = 0;
                                                            // d1 -> outer HEIGHT halo (mesh rows).
                                                            pti.shimDma.halo_slice = pti.tdD1.size;
                                                            pti.shimDma.halo_step = pti.tdD1.stride;
                                                            // d1.slice_tiling -> nested on-core L2 row split.
                                                            if (pti.tdD1.l2Size > 0) {
                                                                pti.shimDma.l2Slice = pti.tdD1.l2Size;
                                                                pti.shimDma.l2Step = pti.tdD1.l2Stride > 0
                                                                                         ? pti.tdD1.l2Stride
                                                                                         : pti.tdD1.l2Size;
                                                                pti.shimDma.l2Rounds =
                                                                    pti.tdD1.l2Groups > 0
                                                                        ? pti.tdD1.l2Groups
                                                                        : (pti.tdD1.size - pti.tdD1.l2Size +
                                                                           pti.shimDma.l2Step - 1) /
                                                                                  pti.shimDma.l2Step +
                                                                              1;
                                                            }
                                                            // Channel-layout stride (d3): the channel dim is not
                                                            // split, so its per-tile coverage IS the padded stride.
                                                            int cDim = pti.tdD3.size > 0 ? pti.tdD3.size : 1;
                                                            pti.shimDma.input_c = cDim;
                                                            // d2 -> WIDTH K-accum split. w_slice/w_step are in COLS;
                                                            // the ShimDma->DmaAddressing lift scales them by cDim
                                                            // into kSlice/kStep ELEMENTS (d1 routing.level slice/
                                                            // step). w_rounds covers d2.base by the halo step.
                                                            if (pti.tdD2.size > 0) {
                                                                pti.shimDma.w_slice = pti.tdD2.size;
                                                                pti.shimDma.w_step = pti.tdD2.stride > 0
                                                                                         ? pti.tdD2.stride
                                                                                         : pti.tdD2.size;
                                                                int wr = pti.tdD2.base > pti.tdD2.size
                                                                             ? (pti.tdD2.base - pti.tdD2.size) /
                                                                                       pti.shimDma.w_step +
                                                                                   1
                                                                             : 1;
                                                                if (wr <= 1) {
                                                                    int mc = localMeshCols > 0 ? localMeshCols
                                                                                               : tilingMeshCols;
                                                                    if (mc > 1)
                                                                        wr = mc;
                                                                }
                                                                pti.shimDma.w_rounds = wr;
                                                                // Per-chunk raw row width (cols*C) + full PADDED row
                                                                // pitch (base*C).
                                                                pti.shimDma.raw_wc = pti.tdD2.size * cDim;
                                                                pti.shimDma.row_pitch =
                                                                    (pti.tdD2.base > 0 ? pti.tdD2.base
                                                                                       : pti.tdD2.size) *
                                                                    cDim;
                                                            } else {
                                                                pti.shimDma.raw_wc = cDim;
                                                            }
                                                            // raw_h = full PADDED input H (d1.base), else coverage.
                                                            if (pti.tdD1.base > 0)
                                                                pti.shimDma.raw_h = pti.tdD1.base;
                                                            else if (pti.tdD1.groups > 0)
                                                                pti.shimDma.raw_h =
                                                                    (pti.tdD1.groups - 1) * pti.tdD1.stride +
                                                                    pti.tdD1.size;
                                                            llvm::outs() << "[TilingLinalg] GemmSpace spatial-halo "
                                                                            "pass-through: "
                                                                         << "halo_slice=" << pti.shimDma.halo_slice
                                                                         << " halo_step=" << pti.shimDma.halo_step
                                                                         << " l2_slice=" << pti.shimDma.l2Slice
                                                                         << " l2_step=" << pti.shimDma.l2Step
                                                                         << " l2_rounds=" << pti.shimDma.l2Rounds
                                                                         << " k_slice=" << pti.shimDma.w_slice * cDim
                                                                         << " k_step=" << pti.shimDma.w_step * cDim
                                                                         << " k_rounds=" << pti.shimDma.w_rounds
                                                                         << " raw_h=" << pti.shimDma.raw_h
                                                                         << " row_pitch=" << pti.shimDma.row_pitch
                                                                         << " raw_wc=" << pti.shimDma.raw_wc << "\n";
                                                            pti.tdD1 = {};
                                                            pti.tdD2 = {};
                                                            pti.tdD3 = {};
                                                            pti.tdD4 = {};
                                                            pti.perPort2D = false;
                                                        }
                                                        // Spatial-halo via GemmSpace: when policy.mode is
                                                        // Overlap and m is an overlapping split, treat this
                                                        // input as a RAW 2D buffer [raw_h, raw_wc] that is
                                                        // row-split into overlapping halo slabs. Convention:
                                                        //   m = row split (size=halo_slice, stride=halo_step,
                                                        //       groups=tile-rows)
                                                        //   n = raw_h  (full input rows -> A shape dim 0)
                                                        //   k = raw_wc (input row width W*C -> A shape dim 1)
                                                        // The conv geometry the kernel needs (KH/KW/C/S/OW/
                                                        // oh_per_row) is NOT carried here — the kernel reads
                                                        // it from compile-time macros instead. After lifting
                                                        // the split into ShimDma we CLEAR tdM/tdN/tdK so they
                                                        // do not pollute the im2col-GEMM M/N/K tiling math
                                                        // (explicitTileM, working-set check, validateDim).
                                                        if (pti.tileMode == 1 && pti.tdM.size > 0 &&
                                                            pti.tdM.stride > 0 && pti.tdM.stride < pti.tdM.size) {
                                                            pti.shimDma.mode = 1;
                                                            pti.shimDma.halo_slice = pti.tdM.size;
                                                            pti.shimDma.halo_step = pti.tdM.stride;
                                                            pti.shimDma.split_dim = 0;
                                                            if (pti.tdN.size > 0)
                                                                pti.shimDma.raw_h = pti.tdN.size;
                                                            if (pti.tdK.size > 0)
                                                                pti.shimDma.raw_wc = pti.tdK.size;
                                                            llvm::outs() << "[TilingLinalg] GemmSpace spatial-halo: "
                                                                         << "halo_slice=" << pti.shimDma.halo_slice
                                                                         << " halo_step=" << pti.shimDma.halo_step
                                                                         << " raw_h=" << pti.shimDma.raw_h
                                                                         << " raw_wc=" << pti.shimDma.raw_wc << "\n";
                                                            pti.tdM = {};
                                                            pti.tdN = {};
                                                            pti.tdK = {};
                                                        }
                                                    }
                                                } else {
                                                    // Bare (legacy) lean SpatialPolicy.
                                                    readPolicy(*apval);
                                                }
                                                pti.policyResolved = true;
                                                llvm::outs()
                                                    << "[TilingLinalg] Policy resolved: pattern=" << pti.pattern
                                                    << " distribution=" << pti.distribution
                                                    << " mergeOrder=" << pti.mergeOrder << " ppDepth=" << pti.pingPong
                                                    << " maxBufferBytes=" << pti.maxBufferBytes;
                                                if (pti.tdM.size > 0 || pti.tdN.size > 0 || pti.tdK.size > 0)
                                                    llvm::outs()
                                                        << " m{size=" << pti.tdM.size << ",stride=" << pti.tdM.stride
                                                        << ",groups=" << pti.tdM.groups << "}"
                                                        << " n{size=" << pti.tdN.size << ",stride=" << pti.tdN.stride
                                                        << ",groups=" << pti.tdN.groups << "}"
                                                        << " k{size=" << pti.tdK.size << ",stride=" << pti.tdK.stride
                                                        << ",groups=" << pti.tdK.groups << "}";
                                                if (pti.shimDma.fromConvSpace)
                                                    llvm::outs()
                                                        << " conv2d{KH=" << pti.shimDma.kernel_h
                                                        << ",KW=" << pti.shimDma.kernel_w
                                                        << ",C=" << pti.shimDma.input_c << ",S=" << pti.shimDma.stride
                                                        << ",P=" << pti.shimDma.pad << ",OH=" << pti.shimDma.conv_oh
                                                        << ",OW=" << pti.shimDma.ow << ",IH=" << pti.shimDma.conv_ih
                                                        << ",IW=" << pti.shimDma.conv_iw << "}";
                                                llvm::outs() << " mode=" << pti.tileMode;
                                                if (pti.layoutTransform > 0)
                                                    llvm::outs() << " layout_transform=" << pti.layoutTransform;
                                                llvm::outs() << "\n";
                                            } else if (!apval) {
                                                llvm::errs()
                                                    << "[TilingLinalg] DEBUG: policy arg kind="
                                                    << (int)policyArg.getKind() << " — could not obtain APValue\n";
                                            }

                                            // --- AST-based DmaTransform struct extraction (targs[2]) ---
                                            // Extract the constexpr DmaTransform struct from the 3rd template
                                            // argument of aie::port<T, Policy, DmaTransform>.
                                            // DmaTransform struct layout:
                                            //   Field 0: dims — array of 4 Dim structs (each: {stride, wrap})
                                            //   Field 1: num_dims — int
                                            //   Field 2: iter_step — int
                                            //   Field 3: iter_wrap — int
                                            //   Field 4: mode — int (0=flat/im2col, 1=spatial_halo)
                                            //   Field 5: halo_slice — int
                                            //   Field 6: halo_step — int
                                            //   Field 7: split_dim — int
                                            //   Field 8: raw_h — int
                                            //   Field 9: raw_wc — int
                                            if (targs.size() >= 3) {
                                                const auto &dmaArg = targs[2];
                                                const APValue *dmaApval = nullptr;

                                                if (dmaArg.getKind() == clang::TemplateArgument::StructuralValue) {
                                                    dmaApval = &dmaArg.getAsStructuralValue();
                                                } else if (dmaArg.getKind() == clang::TemplateArgument::Declaration) {
                                                    ValueDecl *decl = dmaArg.getAsDecl();
                                                    if (auto *tpo = dyn_cast<clang::TemplateParamObjectDecl>(decl))
                                                        dmaApval = &tpo->getValue();
                                                    else if (auto *vd = dyn_cast<clang::VarDecl>(decl)) {
                                                        dmaApval = vd->getEvaluatedValue();
                                                        if (!dmaApval)
                                                            vd->evaluateValue(), dmaApval = vd->getEvaluatedValue();
                                                    }
                                                } else if (dmaArg.getKind() == clang::TemplateArgument::Expression) {
                                                    clang::Expr::EvalResult evalRes;
                                                    if (dmaArg.getAsExpr()->EvaluateAsConstantExpr(evalRes, *Context))
                                                        dmaApval = &evalRes.Val;
                                                }

                                                if (dmaApval && dmaApval->isStruct() &&
                                                    dmaApval->getStructNumFields() >= 4) {
                                                    // Field 0: dims array (APValue::Array of 4 structs)
                                                    const auto &dimsField = dmaApval->getStructField(0);
                                                    // Field 1: num_dims
                                                    int numDims =
                                                        (int)dmaApval->getStructField(1).getInt().getExtValue();
                                                    pti.shimDma.num_dims = numDims;
                                                    // Extract each Dim{stride, wrap} from the array
                                                    if (dimsField.isArray()) {
                                                        for (int di = 0; di < numDims && di < 4; ++di) {
                                                            const auto &dimStruct =
                                                                dimsField.getArrayInitializedElt(di);
                                                            if (dimStruct.isStruct() &&
                                                                dimStruct.getStructNumFields() >= 2) {
                                                                pti.shimDma.dims[di].stride =
                                                                    (int)dimStruct.getStructField(0)
                                                                        .getInt()
                                                                        .getExtValue();
                                                                pti.shimDma.dims[di].wrap =
                                                                    (int)dimStruct.getStructField(1)
                                                                        .getInt()
                                                                        .getExtValue();
                                                            }
                                                        }
                                                    }
                                                    // Field 2: iter_step, Field 3: iter_wrap
                                                    pti.shimDma.iter_step =
                                                        (int)dmaApval->getStructField(2).getInt().getExtValue();
                                                    pti.shimDma.iter_wrap =
                                                        (int)dmaApval->getStructField(3).getInt().getExtValue();

                                                    // Peek the transform mode (field 4) so we can detect a
                                                    // flat() DmaTransform. The port's 3rd template arg always
                                                    // exists (defaulted to flat()), so an unspecified transform
                                                    // shows up here as flat (num_dims==0 && mode==0). A flat
                                                    // transform carries no geometry — applying its all-zero
                                                    // fields would CLOBBER any Conv2dSpace-derived conv geometry
                                                    // (kernel_h/input_c/stride). Skip those overwrites when flat.
                                                    int dmaModePeek = 0;
                                                    if (dmaApval->getStructNumFields() >= 10)
                                                        dmaModePeek =
                                                            (int)dmaApval->getStructField(4).getInt().getExtValue();
                                                    bool dmaIsFlat = (numDims == 0 && dmaModePeek == 0);

                                                    // Fields 4-9: spatial-halo metadata (present when the
                                                    // DmaTransform was built by ConvTiling::spatial). Older
                                                    // DmaTransform structs (4 fields) leave these at 0.
                                                    if (!dmaIsFlat && dmaApval->getStructNumFields() >= 10) {
                                                        pti.shimDma.mode =
                                                            (int)dmaApval->getStructField(4).getInt().getExtValue();
                                                        pti.shimDma.halo_slice =
                                                            (int)dmaApval->getStructField(5).getInt().getExtValue();
                                                        pti.shimDma.halo_step =
                                                            (int)dmaApval->getStructField(6).getInt().getExtValue();
                                                        pti.shimDma.split_dim =
                                                            (int)dmaApval->getStructField(7).getInt().getExtValue();
                                                        pti.shimDma.raw_h =
                                                            (int)dmaApval->getStructField(8).getInt().getExtValue();
                                                        pti.shimDma.raw_wc =
                                                            (int)dmaApval->getStructField(9).getInt().getExtValue();
                                                    }

                                                    // Fields 10-15: conv geometry carried from
                                                    // ConvTiling::spatial (KH/KW/C/S/OW/oh_per_row).
                                                    if (!dmaIsFlat && dmaApval->getStructNumFields() >= 16) {
                                                        pti.shimDma.kernel_h =
                                                            (int)dmaApval->getStructField(10).getInt().getExtValue();
                                                        pti.shimDma.kernel_w =
                                                            (int)dmaApval->getStructField(11).getInt().getExtValue();
                                                        pti.shimDma.input_c =
                                                            (int)dmaApval->getStructField(12).getInt().getExtValue();
                                                        pti.shimDma.stride =
                                                            (int)dmaApval->getStructField(13).getInt().getExtValue();
                                                        pti.shimDma.ow =
                                                            (int)dmaApval->getStructField(14).getInt().getExtValue();
                                                        pti.shimDma.oh_per_row =
                                                            (int)dmaApval->getStructField(15).getInt().getExtValue();
                                                    }

                                                    if (pti.shimDma.mode == 1) {
                                                        llvm::outs()
                                                            << "[TilingLinalg] DmaTransform spatial-halo resolved: "
                                                            << "halo_slice=" << pti.shimDma.halo_slice
                                                            << " halo_step=" << pti.shimDma.halo_step
                                                            << " split_dim=" << pti.shimDma.split_dim
                                                            << " raw_h=" << pti.shimDma.raw_h
                                                            << " raw_wc=" << pti.shimDma.raw_wc << "\n";
                                                    }

                                                    if (!pti.shimDma.empty()) {
                                                        llvm::outs()
                                                            << "[TilingLinalg] DmaTransform resolved: num_dims="
                                                            << numDims;
                                                        for (int di = 0; di < numDims; ++di)
                                                            llvm::outs()
                                                                << " dim" << di << "={" << pti.shimDma.dims[di].stride
                                                                << "," << pti.shimDma.dims[di].wrap << "}";
                                                        if (pti.shimDma.iter_step > 0)
                                                            llvm::outs() << " iter_step=" << pti.shimDma.iter_step
                                                                         << " iter_wrap=" << pti.shimDma.iter_wrap;
                                                        llvm::outs() << "\n";
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (!pti.policyResolved) {
                                        llvm::errs()
                                            << "[TilingLinalg] ERROR: Failed to resolve constexpr SpatialPolicy '"
                                            << policyName << "' from AST\n";
                                    }
                                }

                                // Re-assign shape from resolved policy fields. NOTE: a
                                // spatial-halo input overrides this to the RAW padded DDR
                                // buffer shape below (search "Spatial-halo override").
                                if (pti.policyResolved && macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    if (pti.pattern == 0 && pti.distribution == 0) // Broadcast+Row -> A
                                        pti.shape = {macroDimM, macroDimK};
                                    else if (pti.pattern == 0 && pti.distribution == 1) // Broadcast+Col -> B
                                        pti.shape = {macroDimK, macroDimN};
                                    else // Gather/Scatter -> C
                                        pti.shape = {macroDimM, macroDimN};
                                }

                                // 3D conv OUTPUT shape: when the output port's GemmSpace
                                // declares a full 3D iteration space (d1=H, d2=W, d3=C, each
                                // with a full extent via tile_level.fullsize -> TileDim.base),
                                // emit the genuine 3D output memref <H x W x C> instead of the
                                // flattened 2D [H*W, C]. The #routing.tiling built in
                                // createroutingfuncBySplitModel then follows LtoR_Merge's
                                // d1/d2/d3 directly (H mesh-row split + on-core L2, W on-core
                                // chunk, C group2 mesh-col split). Only the OUTPUT is lifted;
                                // inputs keep their (halo / filter) shapes, and matmul outputs
                                // (no d3 full extent) stay 2D.
                                if (!pti.isInput && pti.perPort2D && pti.tdD1.base > 0 && pti.tdD2.base > 0 &&
                                    pti.tdD3.base > 0) {
                                    pti.shape = {(int64_t)pti.tdD1.base, (int64_t)pti.tdD2.base,
                                                 (int64_t)pti.tdD3.base};
                                    llvm::outs() << "[TilingLinalg] 3D conv output shape: " << pti.varName << " ["
                                                 << pti.shape[0] << "x" << pti.shape[1] << "x" << pti.shape[2]
                                                 << "] (from GemmSpace d1/d2/d3)\n";
                                }

                                // 4D conv FILTER (B) shape: when the col-broadcast filter port
                                // declares the conv kernel sub-dims (ColBC: d1=F, d2=KW, d3=KH,
                                // d4=C via tile_level.fullsize -> TileDim.base), emit the genuine
                                // 4D filter memref <F x KH x KW x C> instead of the flattened 2D
                                // [K, N]. Each mesh COLUMN owns F/HW_COLS filters, so the d0 (F)
                                // split matches the host DDR F-major layout filter[f*K + kk] and
                                // the kernel's B_local[j*K + kk] indexing (the 2D [K,N] form was
                                // the TRANSPOSE of the real DDR buffer). The downstream
                                // split-shape (splitShape[0]/=splitnum) and flat shim BD (product
                                // of all dims) are rank-generic, so [F/cols, KH, KW, C] flows
                                // through unchanged. Only the col-broadcast filter is lifted;
                                // plain matmul B (no d3/d4) stays 2D. Fires only when the conv
                                // sub-dims are present and F*KH*KW*C == K*N (consistency guard).
                                if (pti.policyResolved && pti.isInput && pti.perPort2D && pti.pattern == 0 &&
                                    pti.distribution == 1 && pti.tdD1.base > 0 && pti.tdD2.base > 0 &&
                                    pti.tdD3.base > 0 && pti.tdD4.base > 0) {
                                    int64_t Ff = pti.tdD1.base;  // d1 = F (output filters, col-split)
                                    int64_t KWf = pti.tdD2.base; // d2 = KW
                                    int64_t KHf = pti.tdD3.base; // d3 = KH
                                    int64_t Cf = pti.tdD4.base;  // d4 = C (padded channel)
                                    if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0 && Ff == macroDimN &&
                                        KHf * KWf * Cf == macroDimK) {
                                        pti.shape = {Ff, KHf, KWf, Cf}; // [F, KH, KW, C]
                                        llvm::outs() << "[TilingLinalg] 4D conv filter shape: " << pti.varName << " ["
                                                     << Ff << "x" << KHf << "x" << KWf << "x" << Cf
                                                     << "] (from GemmSpace d1/d3/d2/d4)\n";
                                    }
                                }

                                // Conv2dSpace-derived shim DMA. The Conv2dSpace carries the
                                // conv iteration space; when the port's explicit DmaTransform
                                // is flat() (num_dims==0 && mode==0) we synthesize the same
                                // im2col / spatial-halo shim DMA that DmaTransform::im2col /
                                // ConvTiling::spatial would have produced. An explicit
                                // DmaTransform always wins (single source of truth).
                                if (pti.shimDma.fromConvSpace && pti.shimDma.num_dims == 0 && pti.shimDma.mode == 0) {
                                    const int IH = pti.shimDma.conv_ih;
                                    const int IW = pti.shimDma.conv_iw;
                                    const int C = pti.shimDma.input_c;
                                    const int KH = pti.shimDma.kernel_h;
                                    const int KW = pti.shimDma.kernel_w;
                                    const int S = pti.shimDma.stride > 0 ? pti.shimDma.stride : 1;
                                    const int OH = pti.shimDma.conv_oh;
                                    const int OW = pti.shimDma.ow;
                                    if (pti.tileMode == 1 /*Overlap -> spatial halo*/) {
                                        int convR = localMeshRows > 0 ? localMeshRows : tilingMeshRows;
                                        if (convR <= 0)
                                            convR = 1;
                                        // Honor an explicit Conv2dSpace.m halo split: only
                                        // auto-derive halo_slice/halo_step/oh_per_row when the
                                        // user did NOT pin them via Conv2dSpace.m.
                                        if (!pti.shimDma.haloExplicit) {
                                            int oh_per_row = OH / convR;
                                            pti.shimDma.halo_slice = (oh_per_row - 1) * S + KH;
                                            pti.shimDma.halo_step = oh_per_row * S;
                                            pti.shimDma.oh_per_row = oh_per_row;
                                        }
                                        pti.shimDma.mode = 1;
                                        pti.shimDma.split_dim = 0;
                                        pti.shimDma.raw_h = IH;
                                        pti.shimDma.raw_wc = IW * C;
                                        // Coverage check: the union of all tile slabs must span IH.
                                        if (convR > 0 &&
                                            (convR - 1) * pti.shimDma.halo_step + pti.shimDma.halo_slice < IH)
                                            llvm::errs() << "[TilingLinalg] WARNING: spatial-halo coverage gap: "
                                                         << "(" << convR << "-1)*" << pti.shimDma.halo_step << "+"
                                                         << pti.shimDma.halo_slice << " < IH=" << IH << "\n";
                                        llvm::outs()
                                            << "[TilingLinalg] Conv2dSpace spatial-halo derived: "
                                            << "halo_slice=" << pti.shimDma.halo_slice
                                            << " halo_step=" << pti.shimDma.halo_step << " raw_h=" << pti.shimDma.raw_h
                                            << " raw_wc=" << pti.shimDma.raw_wc
                                            << " oh_per_row=" << pti.shimDma.oh_per_row << "\n";
                                    } else if (IW > 0 && C > 0 && OW > 0) {
                                        // im2col: matches DmaTransform::im2col(IH, IW, C, KH, KW, S, P)
                                        pti.shimDma.dims[0] = {1, KW * C};
                                        pti.shimDma.dims[1] = {IW * C, KH};
                                        pti.shimDma.dims[2] = {S * C, OW};
                                        pti.shimDma.num_dims = 3;
                                        pti.shimDma.iter_step = IW * C * S;
                                        pti.shimDma.iter_wrap = OH;
                                        llvm::outs() << "[TilingLinalg] Conv2dSpace im2col derived: num_dims=3"
                                                     << " dim0={" << pti.shimDma.dims[0].stride << ","
                                                     << pti.shimDma.dims[0].wrap << "}"
                                                     << " dim1={" << pti.shimDma.dims[1].stride << ","
                                                     << pti.shimDma.dims[1].wrap << "}"
                                                     << " dim2={" << pti.shimDma.dims[2].stride << ","
                                                     << pti.shimDma.dims[2].wrap << "}"
                                                     << " iter_step=" << pti.shimDma.iter_step
                                                     << " iter_wrap=" << pti.shimDma.iter_wrap << "\n";
                                    }
                                }

                                // Unify SpatialPolicy Overlap mode into the conv-halo path.
                                // tile_dim describes the *split* (size/stride/groups); the
                                // DmaTransform still provides the windowing math. When the
                                // policy declares m as an Overlap split, drive the existing
                                // halo plumbing from tile_dim (size -> halo_slice,
                                // stride -> halo_step) so it is the single declarative source.
                                if (pti.tileMode == 1 /*Overlap*/ && pti.tdM.size > 0 && pti.tdM.stride > 0 &&
                                    pti.tdM.stride < pti.tdM.size) {
                                    pti.shimDma.mode = 1;
                                    pti.shimDma.halo_slice = pti.tdM.size;
                                    pti.shimDma.halo_step = pti.tdM.stride;
                                }

                                // Spatial-halo override: the A-tensor is declared as the RAW input
                                // [raw_h, raw_wc] (e.g. [224, 672]), NOT the im2col GEMM [M, K].
                                // The partition slices this into overlapping [halo_slice, raw_wc]
                                // row-blocks; the user conv kernel does on-chip windowing.
                                // With a 2D width-split raw_wc holds the PER-CHUNK width, so the
                                // full DDR buffer dim1 is the PADDED row pitch (row_pitch); prefer
                                // it so @main arg1 spans the whole padded buffer (e.g. 230x920).
                                if (pti.shimDma.mode == 1 && pti.shimDma.raw_h > 0 && pti.shimDma.raw_wc > 0) {
                                    int64_t rowPitch = pti.shimDma.row_pitch > 0 ? (int64_t)pti.shimDma.row_pitch
                                                                                 : (int64_t)pti.shimDma.raw_wc;
                                    // 3D conv INPUT shape: the padded row pitch flattens W_pad*C.
                                    // When the channel stride (input_c from RowBC_spatial.d3) evenly
                                    // divides row_pitch, emit the genuine 3D input memref
                                    // <H_pad x W_pad x C> instead of the flattened 2D [H_pad, W_pad*C].
                                    // The DMA is byte-identical (one contiguous padded slab); only the
                                    // rank changes. The halo tiling (routingmanager) and DDR base-offset
                                    // math (flowtransfer_host, rawWc = product of trailing dims) are
                                    // rank-generic, so [H_pad, W_pad, C] flows through unchanged.
                                    int64_t Cf = pti.shimDma.input_c;
                                    if (Cf > 1 && rowPitch % Cf == 0) {
                                        pti.shape = {(int64_t)pti.shimDma.raw_h, rowPitch / Cf, Cf};
                                        llvm::outs() << "[TilingLinalg] 3D conv input shape: " << pti.varName << " ["
                                                     << pti.shape[0] << "x" << pti.shape[1] << "x" << pti.shape[2]
                                                     << "] (H_pad x W_pad x C, from row_pitch/input_c)\n";
                                    } else {
                                        pti.shape = {(int64_t)pti.shimDma.raw_h, rowPitch};
                                    }
                                }

                                parsedTensors.push_back(pti);
                                std::string tagInfo;
                                if (!policyName.empty())
                                    tagInfo = " policy=" + policyName;
                                else if (!spatialTag.empty())
                                    tagInfo = " spatial=" + spatialTag;
                                llvm::outs() << "[TilingLinalg] Tensor param: " << pti.varName << " [";
                                for (size_t sdi = 0; sdi < pti.shape.size(); ++sdi)
                                    llvm::outs() << (sdi ? "x" : "") << pti.shape[sdi];
                                llvm::outs() << "] i" << pti.elementBitWidth << (pti.isInput ? " (input)" : " (output)")
                                             << tagInfo << "\n";
                            }
						}
					}

                    // ---- Compute derived tiling parameters from M/K/N + per-port policy ----
                    int effectiveMeshRows = localMeshRows > 0 ? localMeshRows : tilingMeshRows;
                    int effectiveMeshCols = localMeshCols > 0 ? localMeshCols : tilingMeshCols;
                    if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0 && effectiveMeshRows > 0 &&
                        effectiveMeshCols > 0 && !parsedTensors.empty()) {
                        derivedTilingParams.tileRows = macroDimM / effectiveMeshRows;
                        derivedTilingParams.tileCols = macroDimN / effectiveMeshCols;
                        derivedTilingParams.kDim = macroDimK;

                        // ---- Two-level tiling: compute effective tile_m/tile_n/tile_k ----
                        // Memory budget from HW resource model (same for all current AIE generations)
                        auto hwResForMemCheck = makeResource("Gen2");
                        const int64_t AIE_DATA_MEM_BYTES = hwResForMemCheck->getTileMemoryBytes();
                        const int64_t AIE_USABLE_MEM_BYTES = hwResForMemCheck->getUsableDataBytes();

                        // Collect explicit tiling hints from any resolved policy.
                        // The per-tile length is tile_dim.size (stride/groups feed
                        // coverage validation and round derivation below).
                        int explicitTileM = 0, explicitTileN = 0, explicitTileK = 0;
                        int explicitGroupsM = 0, explicitGroupsN = 0, explicitGroupsK = 0;
                        int elementBytes = 1; // default int8
                        for (const auto &pt : parsedTensors) {
                            if (pt.policyResolved) {
                                if (pt.perPort2D) {
                                    // Per-port 2D GemmSpace: each port describes its OWN
                                    // matrix. Map d1/d2 into the shared M/N/K tile hints by
                                    // role (IO direction + distribution):
                                    //   input  + Row (A=[M,K]) -> d1=M-tile, d2=K-chunk
                                    //   input  + Col (B=[N,K]) -> d1=N-tile, d2=K-chunk
                                    //   output       (C=[M,N]) -> d1=M-tile, d2=N-tile
                                    if (pt.isInput && pt.distribution == 0) { // A (Row)
                                        explicitTileM = pt.tdD1.size;
                                        explicitGroupsM = pt.tdD1.groups;
                                        explicitTileK = pt.tdD2.size;
                                        explicitGroupsK = pt.tdD2.groups;
                                    } else if (pt.isInput && pt.distribution == 1) { // B (Col)
                                        explicitTileN = pt.tdD1.size;
                                        explicitGroupsN = pt.tdD1.groups;
                                        explicitTileK = pt.tdD2.size;
                                        explicitGroupsK = pt.tdD2.groups;
                                    } else if (!pt.isInput) { // C (output)
                                        explicitTileM = pt.tdD1.size;
                                        explicitGroupsM = pt.tdD1.groups;
                                        explicitTileN = pt.tdD2.size;
                                        explicitGroupsN = pt.tdD2.groups;
                                    }
                                } else {
                                    if (pt.tdM.size > 0) {
                                        explicitTileM = pt.tdM.size;
                                        explicitGroupsM = pt.tdM.groups;
                                    }
                                    if (pt.tdN.size > 0) {
                                        explicitTileN = pt.tdN.size;
                                        explicitGroupsN = pt.tdN.groups;
                                    }
                                    if (pt.tdK.size > 0) {
                                        explicitTileK = pt.tdK.size;
                                        explicitGroupsK = pt.tdK.groups;
                                    }
                                }
                            }
                            int eb = pt.elementBitWidth / 8;
                            if (eb > elementBytes)
                                elementBytes = eb;
                        }

                        // ---- Coverage & consistency validation (per dimension) ----
                        // total: M/N partition tiling is temporal *inside* a core
                        // (total = tileRows / tileCols); K is temporal over the full K.
                        auto validateDim = [&](const char *name, const ParsedTensorInfo::TileDim &td, int64_t total) {
                            if (td.size <= 0 || total <= 0)
                                return; // auto-derive: nothing to validate
                            int stride = td.stride > 0 ? td.stride : td.size;
                            int overlap = td.size - stride;
                            int groups = td.groups > 0 ? td.groups : (int)((total - td.size + stride - 1) / stride + 1);
                            int64_t covered = (int64_t)(groups - 1) * stride + td.size;
                            bool isOverlap = overlap > 0;
                            if (isOverlap) {
                                if (stride > td.size)
                                    llvm::errs() << "[TilingLinalg] ERROR: dim " << name << " stride (" << stride
                                                 << ") > size (" << td.size << ") — invalid overlap.\n";
                                // Overlap: tiles must still cover the whole dim.
                                if (covered < total)
                                    llvm::errs() << "[TilingLinalg] ERROR: dim " << name << " Overlap coverage "
                                                 << covered << " < total " << total << " (size=" << td.size
                                                 << " stride=" << stride << " groups=" << groups << ").\n";
                            } else {
                                // Partition: require exact coverage and zero overlap.
                                if (covered != total)
                                    llvm::errs() << "[TilingLinalg] ERROR: dim " << name << " Partition coverage "
                                                 << covered << " != total " << total << " (size=" << td.size
                                                 << " stride=" << stride << " groups=" << groups << ").\n";
                            }
                        };
                        for (const auto &pt : parsedTensors) {
                            if (!pt.policyResolved || !pt.requireFullCoverage)
                                continue;
                            validateDim("m", pt.tdM, derivedTilingParams.tileRows);
                            validateDim("n", pt.tdN, derivedTilingParams.tileCols);
                            validateDim("k", pt.tdK, macroDimK);
                        }

                        int64_t tileM_eff = explicitTileM > 0 ? explicitTileM : derivedTilingParams.tileRows;
                        int64_t tileN_eff = explicitTileN > 0 ? explicitTileN : derivedTilingParams.tileCols;
                        int64_t tileK_eff = explicitTileK > 0 ? explicitTileK : macroDimK;

                        // Memory check: A_local[tileM * tileK] + B_local[tileK * tileN] + C_local[tileM * tileN]
                        auto computeWorkingSet = [&](int64_t tm, int64_t tn, int64_t tk) -> int64_t {
                            return (tm * tk + tk * tn + tm * tn) * elementBytes;
                        };

                        int64_t workingSet = computeWorkingSet(tileM_eff, tileN_eff, tileK_eff);

                        if (explicitTileM > 0 || explicitTileN > 0 || explicitTileK > 0) {
                            // Explicit tiling — validate against memory budget
                            if (workingSet > AIE_DATA_MEM_BYTES) {
                                llvm::errs() << "[TilingLinalg] ERROR: Explicit tiling exceeds AIE data memory!\n"
                                             << "  tile_m=" << tileM_eff << " tile_n=" << tileN_eff
                                             << " tile_k=" << tileK_eff << " working_set=" << workingSet << " bytes"
                                             << " (limit=" << AIE_DATA_MEM_BYTES << ")\n"
                                             << "  A_local=" << tileM_eff * tileK_eff * elementBytes
                                             << " B_local=" << tileK_eff * tileN_eff * elementBytes
                                             << " C_local=" << tileM_eff * tileN_eff * elementBytes << "\n";
                            } else if (workingSet > AIE_USABLE_MEM_BYTES) {
                                llvm::outs() << "[TilingLinalg] WARNING: Explicit tiling uses " << workingSet
                                             << " bytes — close to limit, "
                                             << "little room for stack/ping-pong.\n";
                            }
                            derivedTilingParams.autoTiled = false;
                        } else if (workingSet > AIE_USABLE_MEM_BYTES) {
                            // Auto-tiling: working set exceeds memory budget, need to sub-tile
                            llvm::outs() << "[TilingLinalg] Working set " << workingSet
                                         << " bytes exceeds memory budget " << AIE_USABLE_MEM_BYTES
                                         << " — auto-tiling enabled.\n";

                            // Strategy: first try tiling K (cheapest — no spatial re-launch)
                            // Then reduce M/N if still too large
                            tileK_eff = tileK_eff; // start from full K
                            int64_t budget = AIE_USABLE_MEM_BYTES / elementBytes;

                            // Phase A: reduce K until A+B fit, keeping C stationary
                            // C_local = tileM * tileN (fixed), A_local = tileM * tileK, B_local = tileK * tileN
                            // Constraint: tileM * tileK + tileK * tileN + tileM * tileN <= budget
                            // => tileK * (tileM + tileN) <= budget - tileM * tileN
                            int64_t c_size = tileM_eff * tileN_eff;
                            if (c_size < budget) {
                                int64_t remaining = budget - c_size;
                                int64_t maxK = remaining / (tileM_eff + tileN_eff);
                                if (maxK < tileK_eff && maxK > 0) {
                                    // Round down to power of 2 for alignment
                                    int64_t k = 1;
                                    while (k * 2 <= maxK)
                                        k *= 2;
                                    tileK_eff = k;
                                }
                            }

                            // Phase B: if C_local alone exceeds budget, reduce M and N
                            if (c_size >= budget) {
                                // Solve: 2*tm*tk + tm^2 <= budget (assuming tm=tn, tk=tm for simplicity)
                                // Use heuristic: start with 64 and validate
                                for (int64_t candidate : {256, 128, 64, 32, 16, 8}) {
                                    tileM_eff = std::min((int64_t)candidate, derivedTilingParams.tileRows);
                                    tileN_eff = std::min((int64_t)candidate, derivedTilingParams.tileCols);
                                    // Re-derive tileK with new M/N
                                    c_size = tileM_eff * tileN_eff;
                                    if (c_size < budget) {
                                        int64_t remaining = budget - c_size;
                                        int64_t maxK = remaining / (tileM_eff + tileN_eff);
                                        int64_t k = 1;
                                        while (k * 2 <= maxK && k * 2 <= macroDimK)
                                            k *= 2;
                                        tileK_eff = k;
                                        if (computeWorkingSet(tileM_eff, tileN_eff, tileK_eff) <= AIE_USABLE_MEM_BYTES)
                                            break;
                                    }
                                }
                            }

                            // Final validation
                            workingSet = computeWorkingSet(tileM_eff, tileN_eff, tileK_eff);
                            if (workingSet > AIE_DATA_MEM_BYTES) {
                                llvm::errs() << "[TilingLinalg] ERROR: Auto-tiling failed to fit in memory!\n"
                                             << "  tile_m=" << tileM_eff << " tile_n=" << tileN_eff
                                             << " tile_k=" << tileK_eff << " working_set=" << workingSet << " bytes\n";
                            }
                            derivedTilingParams.autoTiled = true;
                        } else {
                            // Fits in memory without sub-tiling — use full tile dimensions
                            derivedTilingParams.autoTiled = false;
                        }

                        // Compute spatial rounds and K rounds
                        derivedTilingParams.tileM = tileM_eff;
                        derivedTilingParams.tileN = tileN_eff;
                        derivedTilingParams.effectiveK = tileK_eff;
                        // explicitGroups{M,N} come from a per-port .fullsize field and are
                        // therefore GLOBAL counts (over the full M/N dim). The mesh split
                        // (meshRows/meshCols) is applied separately downstream
                        // (populateGemmTiling meshRounds), so convert the global groups to
                        // PER-CORE here by dividing out the mesh factor. When no explicit
                        // groups were given, fall back to the (per-core extent / size) formula.
                        auto perCoreSpatialRounds = [](int globalGroups, int64_t meshFactor, int64_t tileEff,
                                                       int64_t perCoreExtent) -> int64_t {
                            if (globalGroups > 0) {
                                if (meshFactor > 0 && (globalGroups % meshFactor) == 0)
                                    return globalGroups / meshFactor; // global -> per-core
                                return globalGroups;                  // not mesh-divisible: use as-is
                            }
                            return (tileEff > 0 && tileEff < perCoreExtent) ? perCoreExtent / tileEff : 1;
                        };
                        derivedTilingParams.spatialMRounds = perCoreSpatialRounds(
                            explicitGroupsM, effectiveMeshRows, tileM_eff, derivedTilingParams.tileRows);
                        derivedTilingParams.spatialNRounds = perCoreSpatialRounds(
                            explicitGroupsN, effectiveMeshCols, tileN_eff, derivedTilingParams.tileCols);
                        derivedTilingParams.kRounds =
                            explicitGroupsK > 0
                                ? explicitGroupsK
                                : ((tileK_eff > 0 && tileK_eff < macroDimK) ? macroDimK / tileK_eff : 1);

                        // ---- Conv (spatial-halo) geometry: lift from the spatial input tensor ----
                        // The IFM port carries mode==1 with conv geometry captured from
                        // ConvTiling::spatial. Populate derivedTilingParams so the kernel-side
                        // get_kernel_h()/get_halo_slice()/... accessors resolve to real values.
                        bool isSpatialHaloConv = false;
                        // Set when the spatial-halo M-round derivation below detects an
                        // inconsistency (non-divisible tile_rows or C-output geometry mismatch);
                        // propagated to derivedTilingParams.valid so downstream passes skip a
                        // malformed schedule instead of emitting wrong DMA counts.
                        bool convHaloDeriveError = false;
                        for (const auto &pt : parsedTensors) {
                            if (pt.shimDma.mode == 1) {
                                const auto &sd = pt.shimDma;
                                isSpatialHaloConv = true;
                                derivedTilingParams.convHaloSlice = sd.halo_slice;
                                derivedTilingParams.convKernelH = sd.kernel_h;
                                derivedTilingParams.convKernelW = sd.kernel_w;
                                derivedTilingParams.convInputC = sd.input_c;
                                derivedTilingParams.convStride = sd.stride;
                                derivedTilingParams.convOW = sd.ow;
                                derivedTilingParams.convOHPerRow = sd.oh_per_row;
                                // 2D width-split: lift the per-chunk width geometry so the
                                // kernel/output sizing below uses the NARROW per-chunk width.
                                derivedTilingParams.convWRounds = sd.w_rounds;
                                derivedTilingParams.convOWT = sd.ow_t;
                                derivedTilingParams.convWSlice = sd.w_slice;
                                derivedTilingParams.convWStep = sd.w_step;
                                derivedTilingParams.convRowPitch = sd.row_pitch;
                                // Nested L2 (on-core temporal ROW-split) lift: the per-tile
                                // halo_slice rows are chunked into l2Rounds on-core rounds of
                                // l2Slice rows. Drives the kernel A-port round multiplier and
                                // per-round A window below.
                                derivedTilingParams.convL2Rounds = sd.l2Rounds;
                                derivedTilingParams.convL2Slice = sd.l2Slice;
                                derivedTilingParams.convL2Step = sd.l2Step;
                                llvm::outs()
                                    << "[TilingLinalg] Conv spatial-halo params: kernel_h=" << sd.kernel_h
                                    << " kernel_w=" << sd.kernel_w << " input_c=" << sd.input_c
                                    << " stride=" << sd.stride << " ow=" << sd.ow << " oh_per_row=" << sd.oh_per_row
                                    << " halo_slice=" << sd.halo_slice << "\n";
                                if (sd.w_rounds > 1)
                                    llvm::outs()
                                        << "[TilingLinalg] Conv spatial-halo WIDTH-SPLIT lift: w_rounds=" << sd.w_rounds
                                        << " ow_t=" << sd.ow_t << " w_slice=" << sd.w_slice << " w_step=" << sd.w_step
                                        << " row_pitch=" << sd.row_pitch << "\n";
                                break;
                            }
                        }

                        // Capture the OUTPUT (C) tile geometry from its per-port d1/d2
                        // descriptor. For the spatial-halo conv the C output space (LtoR_Merge)
                        // carries d1.tile_size = oh_per_row (output rows per slab) and
                        // d2.tile_size = ow_t (output cols per width-chunk). These supply ow_t
                        // (needed by the output width-split scatter) and cross-check the
                        // halo-derived flat per-slab M-tile below.
                        //
                        // The halo-derived flatTileM (tile_rows/spatialMRounds) is a PER-ON-CORE-
                        // ROUND quantity (one slab produced per (w_round, l2_round) pair). The C
                        // output d1 carries TWO levels: the OUTER per-mesh-row rows (tile_size,
                        // e.g. 28) AND the inner slice_tiling.tile_size = per-round rows (l2Size,
                        // e.g. 7). Use the inner per-round rows for the cross-check so it matches
                        // flatTileM's granularity; fall back to the outer tile_size only when the
                        // output has no on-core L2 split (l2Size == 0).
                        int64_t convOutOhPerRow = 0, convOutOwT = 0;
                        if (isSpatialHaloConv) {
                            for (const auto &pt : parsedTensors) {
                                if (!pt.isInput && pt.perPort2D && pt.tdD1.size > 0 && pt.tdD2.size > 0) {
                                    convOutOhPerRow = pt.tdD1.l2Size > 0 ? pt.tdD1.l2Size : pt.tdD1.size;
                                    convOutOwT = pt.tdD2.size;
                                    break;
                                }
                            }
                        }

                        // Spatial-halo conv: derive the on-core M-round count and the flat
                        // per-slab M-tile DIRECTLY from the per-tensor halo slice info instead
                        // of the fragile ow_t-gated flat tileRows/tileM division. The kernel
                        // (conv2d_spatial) emits ONE [oh_per_row*ow_t, tile_cols] output slab
                        // per (width-round, L2-row-round) pair, so the number of on-core M
                        // rounds is the PRODUCT of the two temporal splits carried by the halo
                        // descriptor:
                        //   spatialMRounds = max(1,w_rounds) * max(1,l2_rounds)   (= 4*4 = 16)
                        //   tile_m         = tile_rows / spatialMRounds           (= 3136/16 = 196)
                        // Both temporal splits are folded into ONE M-round count; the per-port A
                        // spatialRounds below therefore no longer multiplies l2Rounds again. The
                        // result is cross-checked against the C output tile geometry
                        // (oh_per_row * ow_t) and errors on any inconsistency.
                        if (isSpatialHaloConv && derivedTilingParams.tileRows > 0) {
                            int64_t wR = derivedTilingParams.convWRounds > 0 ? derivedTilingParams.convWRounds : 1;
                            int64_t l2R = derivedTilingParams.convL2Rounds > 0 ? derivedTilingParams.convL2Rounds : 1;
                            int64_t spatialMRounds = wR * l2R;
                            if (spatialMRounds <= 0 || (derivedTilingParams.tileRows % spatialMRounds) != 0) {
                                llvm::errs()
                                    << "[TilingLinalg] ERROR: conv spatial-halo tile_rows="
                                    << derivedTilingParams.tileRows
                                    << " not divisible by spatialMRounds=" << spatialMRounds << " (w_rounds=" << wR
                                    << " * l2_rounds=" << l2R << "); cannot derive a consistent tile_m\n";
                                convHaloDeriveError = true;
                            } else {
                                int64_t flatTileM = derivedTilingParams.tileRows / spatialMRounds;
                                // Cross-check against the C output tile geometry when available:
                                // oh_per_row(=C.d1) * ow_t(=C.d2) must equal the flat per-slab
                                // M-tile.
                                if (convOutOhPerRow > 0 && convOutOwT > 0) {
                                    int64_t cTileM = convOutOhPerRow * convOutOwT;
                                    if (cTileM != flatTileM) {
                                        llvm::errs()
                                            << "[TilingLinalg] ERROR: conv spatial-halo M-tile mismatch: "
                                            << "halo-derived tile_m=" << flatTileM << " != C output oh_per_row("
                                            << convOutOhPerRow << ")*ow_t(" << convOutOwT << ")=" << cTileM << "\n";
                                        convHaloDeriveError = true;
                                    }
                                }
                                derivedTilingParams.tileM = flatTileM;
                                tileM_eff = flatTileM;
                                derivedTilingParams.spatialMRounds = spatialMRounds;
                                // ow_t comes from the C output width tile (OW_T); carry it on the
                                // input halo tensor so tensor_N.halo emits a non-zero ow_t and the
                                // output width-split scatter descriptor
                                // (buildOutputTileDescriptor) fires instead of the GEMM
                                // contiguous-M fallback. Also expose it via derivedTilingParams.
                                if (convOutOwT > 0) {
                                    derivedTilingParams.convOWT = convOutOwT;
                                    for (auto &wpt : parsedTensors)
                                        if (wpt.shimDma.mode == 1)
                                            wpt.shimDma.ow_t = (int)convOutOwT;
                                }
                                assert(spatialMRounds * flatTileM == derivedTilingParams.tileRows &&
                                       "conv spatial-halo M coverage: spatialMRounds*tile_m must equal tile_rows");
                                llvm::outs()
                                    << "[TilingLinalg] Conv spatial-halo: derived M-rounds from halo slice info: "
                                    << "w_rounds=" << wR << " * l2_rounds=" << l2R
                                    << " -> spatialMRounds=" << spatialMRounds << ", tile_m=tile_rows("
                                    << derivedTilingParams.tileRows << ")/" << spatialMRounds << "=" << flatTileM
                                    << " (ow_t=" << convOutOwT << ", oh_per_row=" << convOutOhPerRow << ")\n";
                            }
                        }

                        // Spatial-halo conv: the kernel does on-chip im2col with FULL-K dot
                        // products (one pass over k_dim=KH*KW*C), it does NOT k-accumulate.
                        // The GEMM auto-tiler may have split K (effective_k<full_k, k_rounds>1)
                        // based on a GEMM working-set model that does not apply here (the conv
                        // core memory is governed by slab[halo_slice*raw_wc] + local_out, not
                        // A/B/C tiles). Force full-K / single k-round so the filter window
                        // carries the whole [tile_cols, full_k] block the kernel reads.
                        if (isSpatialHaloConv && macroDimK > 0 &&
                            (derivedTilingParams.effectiveK != macroDimK || derivedTilingParams.kRounds != 1)) {
                            llvm::outs() << "[TilingLinalg] Conv spatial-halo: forcing full-K: effective_k="
                                         << derivedTilingParams.effectiveK << "->" << macroDimK
                                         << " k_rounds=" << derivedTilingParams.kRounds << "->1\n";
                            derivedTilingParams.effectiveK = macroDimK;
                            tileK_eff = macroDimK;
                            derivedTilingParams.kRounds = 1;
                        }

                        // Spatial-halo conv: the matmul N (output channels per core) used by
                        // the kernel body's `tile_cols` is the per-mesh-col output-channel
                        // split == tileCols, NOT the flattened output-tile width that the C
                        // output port's d2 (= OW_T * OC_PER_G) carries for transfer geometry.
                        // The C output's d2 inflated explicitTileN (e.g. 128) above, which
                        // would size local_out[oh_per_row*OW*tile_cols] far beyond core
                        // memory. For conv, pin tileN to the GEMM N per core so get_tile_cols()
                        // resolves to the correct matmul inner dimension.
                        if (isSpatialHaloConv && derivedTilingParams.tileCols > 0) {
                            if (derivedTilingParams.tileN != derivedTilingParams.tileCols) {
                                llvm::outs()
                                    << "[TilingLinalg] Conv spatial-halo: overriding tileN="
                                    << derivedTilingParams.tileN << " -> tileCols=" << derivedTilingParams.tileCols
                                    << " (matmul N per core)\n";
                                derivedTilingParams.tileN = derivedTilingParams.tileCols;
                                // Keep the local tileN_eff in sync so the per-port DMA
                                // round/buffer math below (esp. the C-output chunking,
                                // which sizes rounds by tileN_eff) uses the matmul N per
                                // core (tileCols=16), NOT the C-port d2 transfer width
                                // (OW_T*OC_PER_G=128) that inflated bufferSize/numRounds.
                                tileN_eff = derivedTilingParams.tileCols;
                                // N is not sub-tiled within a core for conv (one matmul N pass).
                                derivedTilingParams.spatialNRounds = 1;
                            }
                        }

                        // ---- Compute per-port DMA round/buffer parameters ----
                        // Use effectiveK for DMA buffer sizing (temporal tiling of K)
                        int64_t dmaK = derivedTilingParams.effectiveK;

                        for (auto &pt : parsedTensors) {
                            DerivedTilingParams::PortParams pp;
                            int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                            int maxBuf = pt.maxBufferBytes > 0 ? pt.maxBufferBytes : 4096;

                            if (pt.isInput) {
                                if (pt.shimDma.mode == 1 && pt.shimDma.halo_slice > 0 && pt.shimDma.raw_wc > 0) {
                                    // Spatial-halo IFM: ship the whole contiguous halo slab
                                    // (halo_slice * raw_wc) in a single transfer. On-chip windowing
                                    // requires the contiguous slab, so do NOT cap at maxBuf.
                                    // Nested L2 (on-core temporal ROW-split): when l2Rounds>1 the
                                    // per-round slab is the SMALLER l2Slice rows (not the whole
                                    // halo_slice); the kernel iterates l2Rounds such slabs via the
                                    // BD iteration dim. The per-round A window = l2Slice * raw_wc.
                                    int64_t slabRows = (pt.shimDma.l2Rounds > 1 && pt.shimDma.l2Slice > 0)
                                                           ? (int64_t)pt.shimDma.l2Slice
                                                           : (int64_t)pt.shimDma.halo_slice;
                                    pp.bufferSize = slabRows * pt.shimDma.raw_wc;
                                    pp.numRounds = 1;
                                    // Carry the contiguous slab size to the kernel pass so
                                    // window allocation (BUF_SZ_IN/window_init) matches the
                                    // kernel body's buf_sz (= slabRows * raw_wc), bypassing
                                    // the GEMM flow-view partition size + maxPingPong clamp.
                                    derivedTilingParams.convHaloBufSize = pp.bufferSize;
                                    llvm::outs() << "[TilingLinalg] Spatial-halo IFM buffer: slab_rows=" << slabRows
                                                 << " * raw_wc=" << pt.shimDma.raw_wc << " = " << pp.bufferSize
                                                 << " (numRounds=1" << (pt.shimDma.l2Rounds > 1 ? ", L2 per-round" : "")
                                                 << ")\n";
                                } else if (pt.policyResolved) {
                                    if (pt.pattern == 0 && pt.distribution == 0) {
                                        // Input A: Broadcast+Row — tileM rows × effectiveK per k-round
                                        // pp_depth only controls physical ping-pong buffer count,
                                        // NOT data splitting. Only split when data exceeds maxBuf.
                                        int64_t perKRoundData = tileM_eff * dmaK;
                                        if (perKRoundData <= maxBuf) {
                                            pp.bufferSize = perKRoundData;
                                            pp.numRounds = 1;
                                        } else {
                                            int64_t rowsPerRound = maxBuf / dmaK;
                                            if (rowsPerRound <= 0)
                                                rowsPerRound = 1;
                                            pp.bufferSize = rowsPerRound * dmaK;
                                            pp.numRounds = (tileM_eff + rowsPerRound - 1) / rowsPerRound;
                                            llvm::outs() << "[TilingLinalg] WARNING: Input A tile_m(" << tileM_eff
                                                         << ") * effectiveK(" << dmaK << ") = " << perKRoundData
                                                         << " exceeds max_buffer_bytes(" << maxBuf
                                                         << "), splitting into " << pp.numRounds << " rounds\n";
                                        }
                                    } else if (pt.pattern == 0 && pt.distribution == 1) {
                                        // Input B: Broadcast+Col — tileN cols × effectiveK per k-round
                                        int64_t perKRoundData = tileN_eff * dmaK;
                                        if (perKRoundData <= maxBuf) {
                                            pp.bufferSize = perKRoundData;
                                            pp.numRounds = 1;
                                        } else {
                                            int64_t colsPerRound = maxBuf / dmaK;
                                            if (colsPerRound <= 0)
                                                colsPerRound = 1;
                                            pp.bufferSize = colsPerRound * dmaK;
                                            pp.numRounds = (tileN_eff + colsPerRound - 1) / colsPerRound;
                                            llvm::outs() << "[TilingLinalg] WARNING: Input B tile_n(" << tileN_eff
                                                         << ") * effectiveK(" << dmaK << ") = " << perKRoundData
                                                         << " exceeds max_buffer_bytes(" << maxBuf
                                                         << "), splitting into " << pp.numRounds << " rounds\n";
                                        }
                                    } else {
                                        // Other input patterns: default
                                        int64_t totalElements = 1;
                                        for (auto d : pt.shape)
                                            totalElements *= d;
                                        int64_t perTile = totalElements / (tilingMeshRows * tilingMeshCols);
                                        if (perTile <= maxBuf) {
                                            pp.bufferSize = perTile;
                                            pp.numRounds = 1;
                                        } else {
                                            pp.bufferSize = maxBuf;
                                            pp.numRounds = (perTile + maxBuf - 1) / maxBuf;
                                        }
                                    }
                                } else {
                                    // Unresolved policy — use defaults
                                    int64_t totalElements = pt.shape[0] * pt.shape[1] / effectiveMeshRows;
                                    if (totalElements <= maxBuf) {
                                        pp.bufferSize = totalElements;
                                        pp.numRounds = 1;
                                    } else {
                                        pp.bufferSize = maxBuf;
                                        pp.numRounds = (totalElements + maxBuf - 1) / maxBuf;
                                    }
                                }
                            } else {
                                // Output: use tileM * tileN as per-core output
                                // pp_depth only controls physical buffer count, not data splitting
                                int64_t outputPerCore = tileM_eff * tileN_eff;
                                if (outputPerCore <= maxBuf) {
                                    pp.bufferSize = outputPerCore;
                                    pp.numRounds = 1;
                                } else if (derivedTilingParams.convOW > 0) {
                                    // Conv2d output: the per-core output band is
                                    // [tileM_eff, tileN_eff] where tileM_eff = OH_band * OW
                                    // (M = OH*OW flattened). Chunk along M so each round
                                    // covers a whole number of output-image rows (a multiple
                                    // of convOW), keeping the producer's transfer geometry
                                    // aligned with the shim S2MM consumer (which reassembles
                                    // the band row-by-row). Maximize rows per round under
                                    // maxBuf. Generic 4096-byte chunking would split across
                                    // image-row boundaries (e.g. 4096 vs the conv 3584) and
                                    // desync producer/consumer transfer counts.
                                    int64_t outW = derivedTilingParams.convOW; // image cols
                                    int64_t rowElems = tileN_eff;              // elems per M-row
                                    int64_t maxRows = rowElems > 0 ? maxBuf / rowElems : 0;
                                    // Number of output-IMAGE-rows that fit per round.
                                    int64_t imgRowsPerRound = outW > 0 ? maxRows / outW : 0;
                                    // Total output-image-rows in this band (= oh_per_row).
                                    int64_t totalImgRows = outW > 0 ? tileM_eff / outW : 0;
                                    if (imgRowsPerRound <= 0)
                                        imgRowsPerRound = 1; // at least one image-row band
                                    if (imgRowsPerRound > totalImgRows && totalImgRows > 0)
                                        imgRowsPerRound = totalImgRows;
                                    // Snap DOWN to a divisor of totalImgRows so the rounds tile
                                    // the band EXACTLY (numRounds*bufferSize == outputPerCore).
                                    // Uneven chunking (e.g. 2 img-rows over a 7-row band -> 4
                                    // rounds covering 8 rows) overruns local_out and desyncs the
                                    // S2MM consumer count. oh_per_row is small, so linear scan.
                                    if (totalImgRows > 0) {
                                        while (imgRowsPerRound > 1 && (totalImgRows % imgRowsPerRound) != 0)
                                            imgRowsPerRound--;
                                    }
                                    int64_t rowsPerRound = imgRowsPerRound * outW;
                                    if (rowsPerRound <= 0)
                                        rowsPerRound = outW;
                                    pp.bufferSize = rowsPerRound * rowElems;
                                    pp.numRounds = (tileM_eff + rowsPerRound - 1) / rowsPerRound;
                                    llvm::outs()
                                        << "[TilingLinalg] Conv output chunking: tileM=" << tileM_eff
                                        << " tileN=" << tileN_eff << " outW=" << outW
                                        << " imgRowsPerRound=" << imgRowsPerRound << " rowsPerRound=" << rowsPerRound
                                        << " bufferSize=" << pp.bufferSize << " numRounds=" << pp.numRounds << "\n";
                                } else {
                                    pp.bufferSize = maxBuf;
                                    pp.numRounds = (outputPerCore + maxBuf - 1) / maxBuf;
                                }
                            }
                            // Per-port spatial sub-tile round count for
                            // get_spatial_multiple_rounds(win): an A (input+Row) port
                            // iterates M sub-tiles, a B (input+Col) port iterates N
                            // sub-tiles, and the C (output) port iterates the full
                            // M*N output-tile grid.
                            if (pt.isInput && pt.distribution == 0) {
                                // Both on-core temporal splits (WIDTH rounds AND the nested L2
                                // ROW-split) are ALREADY folded into spatialMRounds
                                // (= w_rounds * l2_rounds) by the halo M-round derivation above,
                                // so the A (Row) port's spatialRounds is spatialMRounds directly.
                                // Do NOT multiply by l2Rounds again here — that double-counted the
                                // L2 split (e.g. 448*4=1792 instead of 16).
                                pp.spatialRounds = derivedTilingParams.spatialMRounds;
                            } else if (pt.isInput && pt.distribution == 1)
                                pp.spatialRounds = derivedTilingParams.spatialNRounds;
                            else if (!pt.isInput)
                                pp.spatialRounds =
                                    derivedTilingParams.spatialMRounds * derivedTilingParams.spatialNRounds;
                            derivedTilingParams.portParams.push_back(pp);
                        }
                        // Mark invalid if the spatial-halo M-round derivation found an
                        // inconsistency (see convHaloDeriveError above) so downstream passes do
                        // not act on a malformed tiling.
                        derivedTilingParams.valid = !convHaloDeriveError;
                        llvm::outs() << "[TilingLinalg] Derived tiling: tileRows=" << derivedTilingParams.tileRows
                                     << " tileCols=" << derivedTilingParams.tileCols
                                     << " kDim=" << derivedTilingParams.kDim << "\n";
                        llvm::outs() << "[TilingLinalg] Two-level tiling: tileM=" << derivedTilingParams.tileM
                                     << " tileN=" << derivedTilingParams.tileN
                                     << " effectiveK=" << derivedTilingParams.effectiveK
                                     << " kRounds=" << derivedTilingParams.kRounds
                                     << " spatialMRounds=" << derivedTilingParams.spatialMRounds
                                     << " spatialNRounds=" << derivedTilingParams.spatialNRounds
                                     << (derivedTilingParams.autoTiled ? " (auto)" : " (explicit/default)") << "\n";
                        llvm::outs() << "[TilingLinalg] Memory estimate: "
                                     << computeWorkingSet(tileM_eff, tileN_eff, tileK_eff) << " bytes"
                                     << " (A=" << tileM_eff * tileK_eff * elementBytes
                                     << " B=" << tileK_eff * tileN_eff * elementBytes
                                     << " C=" << tileM_eff * tileN_eff * elementBytes << ")\n";
                        for (size_t i = 0; i < derivedTilingParams.portParams.size(); ++i) {
                            llvm::outs() << "[TilingLinalg]   port[" << i
                                         << "]: numRounds=" << derivedTilingParams.portParams[i].numRounds
                                         << " bufferSize=" << derivedTilingParams.portParams[i].bufferSize << "\n";
                        }

                        // Check: total ping-pong buffer requirement vs tile data memory
                        {
                            uint32_t totalBufBytes = 0;
                            for (size_t i = 0; i < derivedTilingParams.portParams.size(); i++) {
                                auto &pp = derivedTilingParams.portParams[i];
                                auto &pt = parsedTensors[i];
                                int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                                int elemBytes = pt.elementBitWidth / 8;
                                if (elemBytes <= 0)
                                    elemBytes = 1;
                                totalBufBytes += ppDepth * pp.bufferSize * elemBytes;
                            }
                            std::string errMsg;
                            if (!hwResForMemCheck->checkDataMemoryFits(totalBufBytes, &errMsg)) {
                                llvm::errs() << "[TilingLinalg] ERROR: " << errMsg << "\n";
                                for (size_t i = 0; i < derivedTilingParams.portParams.size(); i++) {
                                    auto &pp = derivedTilingParams.portParams[i];
                                    auto &pt = parsedTensors[i];
                                    int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                                    int elemBytes = pt.elementBitWidth / 8;
                                    if (elemBytes <= 0)
                                        elemBytes = 1;
                                    llvm::errs() << "  Port " << i << " (" << pt.varName << "): " << ppDepth << " x "
                                                 << pp.bufferSize << " x " << elemBytes << " = "
                                                 << ppDepth * pp.bufferSize * elemBytes << " bytes\n";
                                }
                            } else {
                                llvm::outs() << "[TilingLinalg] Buffer memory check passed: " << totalBufBytes << " / "
                                             << hwResForMemCheck->getUsableDataBytes() << " bytes\n";
                            }
                        }

                        // K-round shape adjustment removed: function args now keep full
                        // DDR tensor shapes ({M,K}/{K,N}/{M,N}).  K-round slicing is
                        // handled entirely by module attributes (routing.effective_k,
                        // routing.full_k, routing.k_rounds) consumed by downstream passes.
                    }

                    // ---- Store MeshKernelDesc for multi-kernel support ----
                    {
                        MeshKernelDesc mkd;
                        mkd.kernelName = currentLaunchKernel;
                        mkd.kernelFuncName = currentLaunchKernel;
                        mkd.meshRows = localMeshRows > 0 ? localMeshRows : tilingMeshRows;
                        mkd.meshCols = localMeshCols > 0 ? localMeshCols : tilingMeshCols;
                        mkd.partition = localPartition.isValid() ? localPartition : parsedPartition;
                        mkd.meshId = (int)parsedMeshKernels.size();
                        for (auto &pt : parsedTensors) {
                            DmaAddressing shimDma;
                            if (pt.shimDma.mode == 1) {
                                // Spatial-halo: no multi-dim strides; carry halo descriptor only.
                                shimDma.mode = 1;
                                shimDma.haloSlice = pt.shimDma.halo_slice;
                                shimDma.haloStep = pt.shimDma.halo_step;
                                shimDma.splitDim = pt.shimDma.split_dim;
                                // 2D width-split geometry (only meaningful when w_rounds > 1).
                                shimDma.wSlice = pt.shimDma.w_slice;
                                shimDma.wStep = pt.shimDma.w_step;
                                shimDma.wRounds = pt.shimDma.w_rounds;
                                shimDma.rowPitch = pt.shimDma.row_pitch;
                                shimDma.owT = pt.shimDma.ow_t;
                                // Nested L2 (on-core temporal ROW-split) geometry (only
                                // meaningful when l2Rounds > 1).
                                shimDma.l2Slice = pt.shimDma.l2Slice;
                                shimDma.l2Step = pt.shimDma.l2Step;
                                shimDma.l2Rounds = pt.shimDma.l2Rounds;
                                // K-accum (WIDTH) split -> partitiontensor d1 tiling level.
                                // The width chunks become on-core accumulate rounds; scale
                                // cols by the channel stride so kSlice/kStep are in ELEMENTS
                                // of the padded row pitch (d1 slice=244/step=224/rounds=4).
                                if (pt.shimDma.w_rounds > 1 && pt.shimDma.w_slice > 0) {
                                    int C = pt.shimDma.input_c > 0 ? pt.shimDma.input_c : 1;
                                    shimDma.kSlice = pt.shimDma.w_slice * C;
                                    shimDma.kStep = pt.shimDma.w_step * C;
                                    shimDma.kRounds = pt.shimDma.w_rounds;
                                }
                            } else if (!pt.shimDma.empty()) {
                                for (int i = 0; i < pt.shimDma.num_dims; ++i)
                                    shimDma.dims.push_back({pt.shimDma.dims[i].stride, pt.shimDma.dims[i].wrap});
                                shimDma.iter_step = pt.shimDma.iter_step;
                                shimDma.iter_wrap = pt.shimDma.iter_wrap;
                            }
                            mkd.tensors.push_back({pt.shape, pt.elementBitWidth, pt.isInput, shimDma});
                        }
                        // Store per-kernel body from globalKernelBodies map
                        auto bodyIt = globalKernelBodies.find(currentLaunchKernel);
                        if (bodyIt != globalKernelBodies.end()) {
                            mkd.kernelBody = bodyIt->second;
                        }
                        // Build per-kernel SplitModel from parsedTensors
                        for (auto &pt : parsedTensors) {
                            if (pt.policyResolved) {
                                mkd.splitModel.tensorSplits.push_back(SplitModel::fromPolicyFields(
                                    pt.pattern, pt.distribution, pt.mergeOrder, pt.pingPong, pt.isInput,
                                    pt.maxBufferBytes, pt.layoutTransform));
                            } else if (!pt.spatialTag.empty()) {
                                mkd.splitModel.tensorSplits.push_back(
                                    SplitModel::fromSpatialTag(pt.spatialTag, pt.isInput));
                            } else {
                                mkd.splitModel.tensorSplits.push_back(SplitModel::fromPolicyFields(
                                    pt.isInput ? 0 : 3, 0, pt.isInput ? 0 : 1, 2, pt.isInput, pt.maxBufferBytes));
                            }
                            // Spatial-halo: mirror halo descriptor onto the just-pushed split
                            // so createroutingfuncBySplitModel produces overlapping partitions.
                            if (pt.shimDma.mode == 1 && pt.shimDma.halo_slice > 0) {
                                auto &ts = mkd.splitModel.tensorSplits.back();
                                ts.haloMode = 1;
                                ts.haloSlice = pt.shimDma.halo_slice;
                                ts.haloStep = pt.shimDma.halo_step;
                                ts.splitDim = pt.shimDma.split_dim;
                                // Nested L2 (on-core temporal) split -> partitiontensor
                                // l2_slice/l2_step/l2_rounds attrs in
                                // createroutingfuncBySplitModel.
                                ts.haloL2Slice = pt.shimDma.l2Slice;
                                ts.haloL2Step = pt.shimDma.l2Step;
                                ts.haloL2Rounds = pt.shimDma.l2Rounds;
                            }
                            // Second mesh-axis (group2) split on the OUTPUT (gather)
                            // tensor: within each mesh row, split a *different* tensor
                            // dim (e.g. d3=channel) across mesh cols. Read the parsed
                            // tile for that dim (base=full extent, size=per-col slice)
                            // and stash it on the split desc; createroutingfuncBySplitModel
                            // emits the col=channel #routing.tiling dim from these.
                            if (!pt.isInput && pt.meshTilingGroup2Dim > 0) {
                                auto &ts = mkd.splitModel.tensorSplits.back();
                                const ParsedTensorInfo::TileDim *g2 = nullptr;
                                switch (pt.meshTilingGroup2Dim) {
                                case 1:
                                    g2 = &pt.tdD1;
                                    break;
                                case 2:
                                    g2 = &pt.tdD2;
                                    break;
                                case 3:
                                    g2 = &pt.tdD3;
                                    break;
                                case 4:
                                    g2 = &pt.tdD4;
                                    break;
                                default:
                                    g2 = nullptr;
                                    break;
                                }
                                if (g2 && g2->base > 0 && g2->size > 0) {
                                    // Validate the second-axis split evenly covers the
                                    // full extent across the mesh columns. group2Slice
                                    // (per-col elements) * meshCols must equal group2Full
                                    // (full extent), and group2Full must divide by meshCols.
                                    int cols = mkd.meshCols > 0 ? mkd.meshCols : 1;
                                    bool okDivide = (g2->base % cols == 0);
                                    bool okCover = (g2->size * cols == g2->base);
                                    if (!okDivide || !okCover) {
                                        llvm::errs()
                                            << "[TilingLinalg] ERROR: mesh_tiling_group2_dim=" << pt.meshTilingGroup2Dim
                                            << " invalid split: full=" << g2->base << " slice=" << g2->size
                                            << " meshCols=" << cols
                                            << " (require full % meshCols == 0 and slice * meshCols "
                                               "== full). Skipping group2 split.\n";
                                    } else {
                                        ts.group1Dim = pt.meshTilingGroup1Dim;
                                        ts.group2Dim = pt.meshTilingGroup2Dim;
                                        ts.group2Full = g2->base;
                                        ts.group2Slice = g2->size;
                                        llvm::outs() << "[TilingLinalg] output group2 split: dim=" << ts.group2Dim
                                                     << " full=" << ts.group2Full << " slice=" << ts.group2Slice
                                                     << " meshCols=" << cols << "\n";
                                        // Carry the group1 (mesh-row = d1) on-core
                                        // slice_tiling and d2 (W on-core chunk) into the
                                        // split desc so createroutingfuncBySplitModel can
                                        // build the nested output row-dim #routing.tiling
                                        // chain. Traceability-only (see TensorSplitDesc).
                                        const ParsedTensorInfo::TileDim *g1 = nullptr;
                                        switch (pt.meshTilingGroup1Dim) {
                                        case 1:
                                            g1 = &pt.tdD1;
                                            break;
                                        case 2:
                                            g1 = &pt.tdD2;
                                            break;
                                        case 3:
                                            g1 = &pt.tdD3;
                                            break;
                                        case 4:
                                            g1 = &pt.tdD4;
                                            break;
                                        default:
                                            g1 = nullptr;
                                            break;
                                        }
                                        if (g1 && g1->base > 0) {
                                            ts.group1Full = g1->base;
                                            ts.group1Rounds = g1->groups;
                                            if (g1->l2Groups > 1 && g1->l2Size > 0) {
                                                ts.group1L2Slice = g1->l2Size;
                                                ts.group1L2Step = g1->l2Stride > 0 ? g1->l2Stride : g1->l2Size;
                                                ts.group1L2Rounds = g1->l2Groups;
                                            }
                                        }
                                        if (pt.tdD2.size > 0) { // d2 = W on-core chunk
                                            ts.d2Full = pt.tdD2.base;
                                            ts.d2Slice = pt.tdD2.size;
                                            ts.d2Step = pt.tdD2.stride > 0 ? pt.tdD2.stride : pt.tdD2.size;
                                            ts.d2Rounds = pt.tdD2.groups;
                                        }
                                        llvm::outs()
                                            << "[TilingLinalg] output group1/d2: g1Full=" << ts.group1Full
                                            << " g1Rounds=" << ts.group1Rounds
                                            << " g1L2(slice/step/rounds)=" << ts.group1L2Slice << "/" << ts.group1L2Step
                                            << "/" << ts.group1L2Rounds << " d2(full/slice/step/rounds)=" << ts.d2Full
                                            << "/" << ts.d2Slice << "/" << ts.d2Step << "/" << ts.d2Rounds << "\n";
                                    }
                                }
                            }
                        }
                        // Store per-kernel maxPPBytes (minimum across ports)
                        if (!parsedTensors.empty()) {
                            mkd.maxPPBytes = parsedTensors[0].maxBufferBytes;
                            for (auto &pt : parsedTensors) {
                                if (pt.maxBufferBytes > 0 && pt.maxBufferBytes < mkd.maxPPBytes)
                                    mkd.maxPPBytes = pt.maxBufferBytes;
                            }
                        }
                        // Store per-kernel derivedTilingParams and port var names
                        // for aie::get_*() replacement in multi-kernel mode
                        mkd.derivedParams = derivedTilingParams;
                        // Carry the fullconnect_auto flag parsed from the kernel's
                        // keyword annotation (default true when absent).
                        {
                            auto fcaIt = kernelFullConnectAuto.find(currentLaunchKernel);
                            mkd.fullConnectAuto = (fcaIt != kernelFullConnectAuto.end()) ? fcaIt->second : true;
                        }
                        for (auto &pt : parsedTensors) {
                            mkd.portVarNames.push_back(pt.varName);
                        }
                        parsedMeshKernels.push_back(mkd);
                        llvm::outs() << "[TilingLinalg] Registered MeshKernelDesc: kernel=" << mkd.kernelName
                                     << " mesh=" << mkd.meshRows << "x" << mkd.meshCols << " meshId=" << mkd.meshId
                                     << "\n";
                    }
                }
				return true;
			}

			if (Callee && Callee->getNameAsString() =="XAie_LoadElfMem") {
                if (CE->getNumArgs() >= 3) {
                    const Expr *devExpr = CE->getArg(0);
                    const Expr *tileExpr = CE->getArg(1);
                    const Expr *kernelExpr = CE->getArg(2)->IgnoreParenImpCasts();

                    std::string devStr;
                    llvm::raw_string_ostream devOs(devStr);
                    devExpr->printPretty(devOs, nullptr, PrintingPolicy(LangOptions()));
                    devOs.flush();

                    std::string tileStr;
                    llvm::raw_string_ostream tileOs(tileStr);
                    tileExpr->printPretty(tileOs, nullptr, PrintingPolicy(LangOptions()));
                    tileOs.flush();

                    std::string kernelArgStr;
                    llvm::raw_string_ostream kernelArgOs(kernelArgStr);
                    kernelExpr->printPretty(kernelArgOs, nullptr, PrintingPolicy(LangOptions()));
                    kernelArgOs.flush();
                    boost::algorithm::trim(kernelArgStr);

                    if (kernelArgStr.find("_binary_kernel_") != std::string::npos) {
                        return true;
                    }

                    const Expr *kernelNameExpr = kernelExpr;
                    while (const auto *castExpr = dyn_cast<CastExpr>(kernelNameExpr)) {
                        kernelNameExpr = castExpr->getSubExpr()->IgnoreParenImpCasts();
                    }
                    if (const auto *unaryExpr = dyn_cast<UnaryOperator>(kernelNameExpr)) {
                        if (unaryExpr->getOpcode() == UO_AddrOf) {
                            kernelNameExpr = unaryExpr->getSubExpr()->IgnoreParenImpCasts();
                        }
                    }

                    std::string kernelName;
                    if (const auto *declRef = dyn_cast<DeclRefExpr>(kernelNameExpr)) {
                        kernelName = declRef->getDecl()->getNameAsString();
                    } else {
                        return true;
                    }

                    std::string replacement = "XAie_LoadElfMem(" + devStr + ", " + tileStr +
                                              ", (unsigned char *)_binary_kernel_" + kernelName + "_start)";
                    clang::SourceRange callExprRange = CE->getSourceRange();
                    Rewrite->ReplaceText(callExprRange, replacement);
                }
            } else {
					if (Callee) {
						for (auto x:kernel_name_list) {
							// std::cout << "kernel name is (" << x << ")" <<std::endl;
							if (Callee->getNameAsString() == x) {
								llvm::outs()<< "**************Function name is " << Callee->getNameInfo().getName().getAsString() <<"*********8\n";
							
								if (auto FD = CE->getDirectCallee()) {

									// 2) Check if it’s actually a specialization of a function template:
									if (auto  TSI = FD->getTemplateSpecializationInfo()) {
								
										// 3) Grab the FunctionTemplateDecl (the “template name”):
										auto FTD = FD->getPrimaryTemplate();
										llvm::outs() << "Template name: "
													<< FTD->getNameAsString() << "\n";
									
										// 4) Extract the explicit template-argument list:
										ArrayRef<TemplateArgument> Args = 
											TSI->TemplateArguments->asArray();
									
										llvm::outs() << "Template args: <";
										std::vector<std::string> knames;
										for (size_t i = 0; i < Args.size(); ++i) {
											const TemplateArgument &Arg = Args[i];
											llvm::SmallString<16> Buf;
											switch (Arg.getKind()) {
											case TemplateArgument::Integral:
												
												Arg.getAsIntegral().toString(Buf, 10);
												llvm::outs() << Buf;
												//std::string kname(Buf.begin(), Buf.end());
												//kname = Buf.str().str();
												knames.push_back(Buf.str().str());
												break;
											case TemplateArgument::Type:
												llvm::outs() << Arg.getAsType().getAsString();
												break;
											default:
												llvm::outs() << "(other)";
											}
											if (i + 1 < Args.size()) llvm::outs() << ", ";
										}
										llvm::outs() << ">\n\n";
										Aiefrontend->createKernelFunction(knames);
									}
								}
								
							}
						}
					} else {
						auto Call = CE->getCallee();
						if (Call) {
							//llvm::errs() << "Indirect function call or function pointer dereference\n";
							//Call->dump();  // You can inspect the callee expression
						}
					}
            // If Callee is null, it may be an indirect function call or a function pointer.
            // You can handle those cases accordingly.
            //llvm::outs() << "Indirect function call or function pointer\n";
			}
			return true;
    }
/*
		 bool VisitParmVarDecl(ParmVarDecl *PVD) {
        // Handle function parameters
        if (!PVD->hasInit()) {
          // Parameter does not have an initializer
          llvm::outs() << "Parameter without initializer: " << PVD->getNameAsString() << "\n";
        }
        return true;
      }

      bool VisitVarDecl(VarDecl *VD) {
        // Handle local variables
        if (!VD->hasInit()) {
          // Variable does not have an initializer
          llvm::outs() << "Variable without initializer: " << VD->getNameAsString() << "\n";
        }
        return true;
     }
		  bool VisitDecl(Decl *D) {
        // Check for "unknown type name 'windows'" error
				//std::cout << "unknow VisitDecl" <<std::endl;
				//return false;
				std::cout << "Decl is " << D->getDeclKindName() << std::endl;
        if (isa<UnresolvedUsingTypenameDecl>(D)) {
        }

        return true;
      }
			*/

		 void PrintCode() {
    	// Apply the rewrite to the source file
    	//Rewrite.commit(rewriter::Rewriter::Commit);
			//std::string RewrittenCode = Rewrite.getEditBuffer(Rewrite.getSourceMgr().getMainFileID()).getBuffer();
			 std::cout << "---------****---------" << std::endl;
			 ///*
			 const RewriteBuffer *RewriteBuf = Rewrite->getRewriteBufferFor(Rewrite->getSourceMgr().getMainFileID());
			 if (RewriteBuf) {
        //llvm::outs() << "Rewritten Source Code:\n" << RewriteBuf << "\n";
			//*/
			 //std::ofstream fd("./new_t.c");
				 RewriteBuf->write(llvm::outs());
		 }
			 /*
			 auto& sc = Rewrite.getSourceMgr();
			 auto mid = sc.getMainFileID();
			 llvm::StringRef sourceCode = sc.getBufferData(mid);
			 llvm::outs() << "Original Source Code:\n" << sourceCode << "\n";
			 */
			 std::cout << "------------------" << std::endl;
			//llvm::outs() << "Rewritten Source Code:\n" << RewrittenCode << "\n";
  	}


private:
		AieFrontEnd* Aiefrontend;
    ASTContext *Context;
		Rewriter *Rewrite;
};

class MyASTConsumer : public ASTConsumer {
public:
    //explicit MyASTConsumer(ASTContext *Context)
      //  : Visitor(Context) {}

    explicit MyASTConsumer(Rewriter* TheRewriter, ASTContext *Context, AieFrontEnd* Aiefrontend)
        : Visitor(TheRewriter, Context, Aiefrontend) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
		// std::cout << __FUNCTION__ << " get called " << std::endl;
    }

private:
    GlobalFunctionVisitor Visitor;
};

class AieDebugLevelPragmaHandler : public clang::PragmaHandler {
  public:
    AieDebugLevelPragmaHandler() : PragmaHandler("aie_debug_level") {}

    void HandlePragma(clang::Preprocessor &PP, clang::PragmaIntroducer Introducer, clang::Token &FirstToken) override {
        // Known flag macros (must match aie_runtime.h definitions)
        static const std::unordered_map<std::string, int> knownFlags = {
            {"AIE_DEBUG_FLAG_DISABLE_MULTID_DIM_DMA", 1 << 4}, {"AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN", 1 << 5},
            {"AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER", 1 << 6},   {"AIE_DMA_ISSUE_COUNT", 1 << 7},
            {"AIE_DEBUG_FLAG_CORE_PERF_COUNTER", 1 << 8},      {"AIE_DEBUG_LOG", 1 << 9},
        };

        clang::Token Tok;
        PP.Lex(Tok);

        // Skip optional opening paren: #pragma aie_debug_level(...)
        bool hasParen = Tok.is(clang::tok::l_paren);
        if (hasParen)
            PP.Lex(Tok);

        int result = 0;
        bool valid = false;

        // Parse: term ( '|' term )*
        // Each term is a numeric_constant or a known identifier flag
        while (true) {
            if (Tok.is(clang::tok::numeric_constant)) {
                llvm::SmallString<16> IntegerBuffer;
                bool Invalid = false;
                llvm::StringRef Spelling = PP.getSpelling(Tok, IntegerBuffer, &Invalid);
                if (Invalid)
                    break;
                int val = 0;
                // Support decimal, hex (0x...), octal (0...) via base 0
                if (Spelling.getAsInteger(0, val))
                    break;
                result |= val;
                valid = true;
            } else if (Tok.is(clang::tok::identifier)) {
                std::string name = PP.getSpelling(Tok);
                auto it = knownFlags.find(name);
                if (it == knownFlags.end()) {
                    llvm::errs() << "[aiehlc] Warning: unknown flag '" << name
                                 << "' in #pragma aie_debug_level, ignored\n";
                } else {
                    result |= it->second;
                    valid = true;
                }
            } else {
                break;
            }

            PP.Lex(Tok);
            if (Tok.is(clang::tok::pipe)) {
                PP.Lex(Tok); // consume '|', continue to next term
            } else {
                break;
            }
        }

        // Consume optional closing paren
        if (hasParen && Tok.is(clang::tok::r_paren))
            PP.Lex(Tok);

        if (valid) {
            parsedDebugLevel = result;
            llvm::outs() << "[aiehlc] Detected #pragma aie_debug_level " << parsedDebugLevel << "\n";
        }

        // Consume remaining tokens on the pragma line (if any)
        if (Tok.isNot(clang::tok::eod))
            PP.DiscardUntilEndOfDirective();
    }
};

class MyFrontendAction : public ASTFrontendAction {
public:
		MyFrontendAction() {
			FileMgr = std::make_unique<clang::FileManager>(clang::FileSystemOptions());
			aiefrontend.set_llvm_aie(use_llvm_aie);
		}
		void ReplaceGlobalKeyAndAddInclude(clang::SourceManager &SourceMgr) {
			std::unordered_map<std::string, std::string> replace_map={{"__global__","__attribute__((annotate(\"__global__\")))"},
																	  {"__kernel__","__attribute__((annotate(\"__kernel__\")))"}};
			//auto SourceCodeString = GetKeyReplaceAndAddInclude(SourceMgr, "__global__",
			//		"__attribute__((annotate(\"__global__\")))");
			auto SourceCodeString = GetKeyReplaceAndAddInclude(SourceMgr, replace_map);
			llvm::StringRef ModifiedCode = SourceCodeString;

            // Save a physical copy of the preprocessed file for debugging/inspection
            std::string newfilePath = std::string(AOUT) + "newfile.cpp";
            std::ofstream newfileOut(newfilePath, std::ios::out | std::ios::trunc);
            if (newfileOut.is_open()) {
                newfileOut << SourceCodeString;
                newfileOut.close();
                llvm::outs() << "Saved preprocessed file to: " << newfilePath << "\n";
            }

            std::unique_ptr<llvm::MemoryBuffer> ModifiedBuffer =
					 llvm::MemoryBuffer::getMemBufferCopy(ModifiedCode,"NewCode");
			// std::cout << "new ----------" << code << std::endl;
			// auto src = code.c_str();
			// std::cout << "new ---src-------" << src << std::endl;

			//const char* src = "__attribute__((annotate(\"__global__\"))) int func () {} \n int main() {}";
			//std::unique_ptr<llvm::MemoryBuffer> Buf = llvm::MemoryBuffer::getMemBuffer(src);
			const FileEntryRef SourceFile = FileMgr->getVirtualFileRef("newfile.cpp", ModifiedBuffer->getBufferSize(), 0);
			//FileEntryRef fileEntryRef = SourceMgr.translateFile(SourceFile);

			SourceMgr.overrideFileContents(SourceFile, std::move(ModifiedBuffer));
			FileID MainFileID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
			SourceMgr.setMainFileID(MainFileID);
		}
	 /* do pre-processing change the __global__ into annotation
	  */
	  bool BeginSourceFileAction(CompilerInstance &CI) override {
			//auto& ppopts = CI.getPreprocessorOpts();
			// std::cout <<"File is " << "Filename.str()" << std::endl;
 			clang::SourceManager &SourceMgr = CI.getSourceManager();

			ReplaceGlobalKeyAndAddInclude(SourceMgr);

            clang::Preprocessor &PP = CI.getPreprocessor();
            PP.AddPragmaHandler(new AieDebugLevelPragmaHandler());

            return true;
		}

		std::string GetKeyReplaceAndAddInclude(clang::SourceManager &SourceMgr, std::unordered_map<std::string,std::string>& funcMap) {
			std::string ret;
			// Get the main file entry
			const auto FileEntryRef = SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID());

			if (FileEntryRef) {
				// Get the file name
				llvm::StringRef FileName = FileEntryRef->getName();

                // Capture source directory for include path propagation
                {
                    std::string fnStr = FileName.str();
                    auto lastSlash = fnStr.rfind('/');
                    if (lastSlash != std::string::npos)
                        userSourceDir = fnStr.substr(0, lastSlash);
                    else
                        userSourceDir = ".";
                }

                // Print or use the file name as needed
				// llvm::outs() << "Processing file: " << FileName << "\n";
				clang::FileID MainFileID = SourceMgr.getMainFileID();

				// Get the source code buffer for the main file
				//const llvm::MemoryBuffer *MainFileBuffer = SourceMgr.getBuffer(MainFileID);
				const auto MainFileBufferOp = SourceMgr.getBufferOrNone(MainFileID);
				const llvm::MemoryBufferRef *MainFileBuffer = &MainFileBufferOp.value();
				// Get the source code as a StringRef
				llvm::StringRef SourceCode = MainFileBuffer->getBuffer();
				// std::cout << SourceCode.str() << std::endl;
				std::unique_ptr<llvm::MemoryBuffer> Buffer = llvm::MemoryBuffer::getMemBuffer(SourceCode, "", false);

				llvm::SourceMgr SM;
				SM.AddNewSourceBuffer(std::move(Buffer), llvm::SMLoc());
				//replace keyword with annotate
				//std::string KeywordToReplace = sOld;//"__global__";
				//std::string Replacement = sNew;//"__attribute__((annotate(\"__global__\")))";

				//llvm::StringRef UpdatedSource;
				std::string SourceCodeString = SourceCode.str();

                // Collect user #define macros before any rewriting.
                // Track #if/#else/#endif nesting to skip macros inside
                // conditional blocks (they cause redefinition warnings when
                // both branches are flattened into the kernel file).
                {
                    userMacroDefines.clear();
                    std::istringstream iss(SourceCodeString);
                    std::string line;
                    std::string currentDefine;
                    bool inMultiLine = false;
                    int ifDepth = 0; // nesting depth of #if/#ifdef/#ifndef
                    while (std::getline(iss, line)) {
                        if (inMultiLine) {
                            currentDefine += "\n" + line;
                            if (line.empty() || line.back() != '\\') {
                                if (ifDepth == 0)
                                    userMacroDefines.push_back(currentDefine);
                                inMultiLine = false;
                            }
                            continue;
                        }
                        size_t firstNonSpace = line.find_first_not_of(" \t");
                        if (firstNonSpace == std::string::npos)
                            continue;
                        std::string trimmed = line.substr(firstNonSpace);
                        // Track preprocessor conditional nesting
                        if (trimmed.substr(0, 3) == "#if") {
                            ifDepth++;
                            continue;
                        }
                        if (trimmed.substr(0, 5) == "#else" || trimmed.substr(0, 5) == "#elif") {
                            continue;
                        }
                        if (trimmed.substr(0, 6) == "#endif") {
                            if (ifDepth > 0)
                                ifDepth--;
                            continue;
                        }
                        if (trimmed.substr(0, 7) == "#define") {
                            // Skip aiehlc-internal stub macros
                            if (line.find("AIEHLC_STUBS_DEFINED") != std::string::npos ||
                                line.find("AIEHLC_TILING_STUBS_DEFINED") != std::string::npos) {
                                continue;
                            }
                            // Skip macros inside #if/#else blocks
                            if (ifDepth > 0)
                                continue;
                            currentDefine = line;
                            if (!line.empty() && line.back() == '\\') {
                                inMultiLine = true;
                            } else {
                                userMacroDefines.push_back(currentDefine);
                            }
                        }
                    }
                }

                size_t KeywordPos = 0;
				for (auto x:funcMap ) {
					std::string KeywordToReplace = x.first;//"__global__";
					std::string Replacement = x.second;//"__attribute__((annotate(\"__global__\")))";
					size_t KeywordPos = 0;
					while ((KeywordPos =  SourceCodeString.find(KeywordToReplace, KeywordPos)) != std::string::npos) {
						int pos = KeywordPos;
						bool bgstr = false;
						// check whether the __global_ is inside a '"', actually just check the front '"'
						while(--pos >=0) {
							if (SourceCodeString[pos] == ' ') continue;
							if (SourceCodeString[pos] == '"' || SourceCodeString[pos] == '/' || SourceCodeString[pos] == '*') {
								bgstr = true;
							}
							break;
						}
						if (!bgstr) {
							//process the kernel logic
							auto curkeywordpos = KeywordPos;
							if (KeywordToReplace == "__kernel__") {
								auto bracepos = SourceCodeString.find("(", curkeywordpos);
								auto curpos = bracepos;
								while(--curpos >=0){
									if (SourceCodeString[curpos] == ' ') break;
								}
								curpos++;
							
								auto str = SourceCodeString.substr(curpos, bracepos - curpos);
								while(str[str.size() -1] == ' ') {
									str = str.substr(0, str.size() - 1);
								}
								kernel_name_list.push_back(str);
							}
                            // replace
                            //  Handle explicit policy binding: __global__(<policyVar>)
                            bool didPolicyReplace = false;
                            if (KeywordToReplace == "__global__") {
                                size_t afterKw = KeywordPos + KeywordToReplace.size();
                                size_t p = afterKw;
                                while (p < SourceCodeString.size() && SourceCodeString[p] == ' ')
                                    p++;
                                if (p < SourceCodeString.size() && SourceCodeString[p] == '(') {
                                    size_t closeParen = SourceCodeString.find(")", p);
                                    if (closeParen != std::string::npos) {
                                        std::string policyVar = SourceCodeString.substr(p + 1, closeParen - (p + 1));
                                        // trim surrounding whitespace
                                        size_t s = policyVar.find_first_not_of(" \t");
                                        size_t e = policyVar.find_last_not_of(" \t");
                                        if (s != std::string::npos)
                                            policyVar = policyVar.substr(s, e - s + 1);
                                        else
                                            policyVar.clear();
                                        std::string fullReplacement =
                                            Replacement + " __attribute__((annotate(\"aiepolicy:" + policyVar + "\")))";
                                        size_t spanLen = (closeParen + 1) - KeywordPos;
                                        SourceCodeString.replace(KeywordPos, spanLen, fullReplacement);
                                        KeywordPos += fullReplacement.size();
                                        didPolicyReplace = true;
                                    }
                                }
                            }
                            if (!didPolicyReplace) {
                                SourceCodeString.replace(KeywordPos, KeywordToReplace.size(), Replacement);
                                KeywordPos += Replacement.size();
                            }

                            // For __global__ functions, wrap body with #ifdef KERNEL_COMPILE
                            if (KeywordToReplace == "__global__") {
                                // Find the opening brace of the function body
                                auto openBracePos = SourceCodeString.find("{", KeywordPos);
                                if (openBracePos != std::string::npos) {
                                    // Insert #ifdef KERNEL_COMPILE after opening brace
                                    std::string ifdefGuard = "\n#ifdef KERNEL_COMPILE\n";
                                    SourceCodeString.insert(openBracePos + 1, ifdefGuard);

                                    // Find matching closing brace
                                    int braceCount = 1;
                                    size_t searchPos = openBracePos + 1 + ifdefGuard.size();
                                    while (braceCount > 0 && searchPos < SourceCodeString.size()) {
                                        if (SourceCodeString[searchPos] == '{')
                                            braceCount++;
                                        else if (SourceCodeString[searchPos] == '}')
                                            braceCount--;
                                        if (braceCount > 0)
                                            searchPos++;
                                    }
                                    // Insert #endif before closing brace
                                    if (braceCount == 0) {
                                        std::string endifGuard = "\n#endif\n";
                                        SourceCodeString.insert(searchPos, endifGuard);
                                    }
                                }
                            }

                        } else {
							KeywordPos += KeywordToReplace.size();
						}
					}
				}
				// Transform <<<mesh>>> kernel launch syntax (CUDA-style)
				// Pattern: funcname<<<varname>>>(args) -> __aie_launch("funcname", varname, args)
				{
					size_t launchPos = 0;
					while ((launchPos = SourceCodeString.find("<<<", launchPos)) != std::string::npos) {
						// Find the function name before <<<
						size_t funcEnd = launchPos;
						while (funcEnd > 0 && (std::isalnum(SourceCodeString[funcEnd-1]) || SourceCodeString[funcEnd-1] == '_'))
							funcEnd--;
						std::string funcName = SourceCodeString.substr(funcEnd, launchPos - funcEnd);

						// Find mesh variable name between <<< and >>>
						size_t meshStart = launchPos + 3;
						size_t meshEnd = SourceCodeString.find(">>>", meshStart);
						if (meshEnd == std::string::npos) break;
						std::string meshVar = SourceCodeString.substr(meshStart, meshEnd - meshStart);
						// trim whitespace
						meshVar.erase(0, meshVar.find_first_not_of(" \t"));
						meshVar.erase(meshVar.find_last_not_of(" \t") + 1);

						// Find args between ( and )
						size_t argsOpenParen = SourceCodeString.find("(", meshEnd + 3);
						if (argsOpenParen == std::string::npos) break;
						int depth = 1;
						size_t argsEnd = argsOpenParen + 1;
						while (depth > 0 && argsEnd < SourceCodeString.size()) {
							if (SourceCodeString[argsEnd] == '(') depth++;
							if (SourceCodeString[argsEnd] == ')') depth--;
							if (depth > 0) argsEnd++;
						}
						std::string args = SourceCodeString.substr(argsOpenParen + 1, argsEnd - argsOpenParen - 1);

                        // Build replacement: __aie_launch("funcName", meshVar, args)
                        // Note: byte sizes are injected later in EndSourceFileAction
                        // (post-AST) when parsedMeshKernels is populated.
                        std::string replacement = "__aie_launch(\"" + funcName + "\", " + meshVar + ", " + args + ")";

						// Replace from funcEnd to argsEnd+1 (include closing paren)
						SourceCodeString.replace(funcEnd, argsEnd + 1 - funcEnd, replacement);
						launchPos = funcEnd + replacement.size();

						// Track that we're in tiling mode
						isTilingLinalgMode = true;
					}
				}

                // ret = "#include <stdio.h>\n";
                // ret += "#include \"adf.h\"\n";

                // Add stub declarations for AIE-specific types that clang can't find
                // (because __AIENGINE__ isn't defined, so adf.h uses different code paths)
                // Function body is wrapped with #ifdef KERNEL_COMPILE so only type stubs needed
                ret +=
                    "// Stub type declarations for Clang parsing (function body skipped via #ifdef KERNEL_COMPILE)\n";
                ret += "#ifndef AIEHLC_STUBS_DEFINED\n";
                ret += "#define AIEHLC_STUBS_DEFINED\n";
                ret += "template<typename T> struct input_window {};\n";
                ret += "template<typename T> struct output_window {};\n";
                ret += "typedef int int32;\n";
                ret += "typedef input_window<int32> input_window_int32;\n";
                ret += "typedef output_window<int32> output_window_int32;\n";
                ret += "typedef signed char int8;\n";
                ret += "typedef input_window<int8> input_window_int8;\n";
                ret += "typedef output_window<int8> output_window_int8;\n";
                ret += "typedef short int16;\n";
                ret += "typedef input_window<int16> input_window_int16;\n";
                ret += "typedef output_window<int16> output_window_int16;\n";
                // Kernel intrinsic stubs (body is #ifdef KERNEL_COMPILE guarded)
                ret += "typedef unsigned char uint8_t;\n";
                ret += "typedef unsigned long uintptr_t;\n";
                ret += "typedef int int8_t __attribute__((mode(QI)));\n";
                ret += "typedef int int32_t __attribute__((mode(SI)));\n";
                ret += "typedef int v4int8 __attribute__((vector_size(4)));\n";
                ret += "typedef int v4int32 __attribute__((vector_size(16)));\n";
                ret += "inline unsigned get_coreid() { return 0; }\n";
                ret += "inline void klog(const char*, int) {}\n";
                ret += "template<typename T> inline void* acquire_input_window(T*) { return (void*)0; }\n";
                ret += "template<typename T> inline void* acquire_output_window(T*) { return (void*)0; }\n";
                ret += "template<typename T> inline void release_input_window(T*) {}\n";
                ret += "template<typename T> inline void release_output_window(T*) {}\n";
                ret += "#define BUF_SZ 16\n";
                ret += "#endif\n\n";

                // CUDA-style AIE API stubs for tiling mode
                if (isTilingLinalgMode) {
                    ret += "// CUDA-style AIE API stubs for Clang parsing\n";
                    ret += "#ifndef AIEHLC_TILING_STUBS_DEFINED\n";
                    ret += "#define AIEHLC_TILING_STUBS_DEFINED\n";
                    // SpatialPolicy struct + port<T, Policy> system (C++20 struct NTTP)
                    // Types and port template only — policy constants are user-defined
                    ret += "namespace aie {\n";
                    ret += "enum class Pattern  { Broadcast = 0, Scatter = 1, Distribute = 1, Multicast = 2, Gather = "
                           "3 };\n";
                    ret += "enum class Layout   { Row = 0, Col = 1, Grid = 2 };\n";
                    ret += "enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };\n";
                    ret += "enum class PadMaterialize { DDR = 0, Memtile = 1 };\n";
                    ret += "enum class Im2col   { None = 0, Dma = 1 };\n";
                    // Kept for back-compat (no longer policy fields):
                    ret += "enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };\n";
                    ret += "enum class TileMode { Partition = 0, Overlap = 1 };\n";
                    // Objective — declarative optimization goal for a spatial space
                    // (metadata only today; no autotuner/search yet).
                    ret += "enum class Objective { MaxArrayUtil = 0, MinLatency = 1, MinDma = 2 };\n";
                    ret += "struct Bytes { int value = 0; };\n";
                    // tile_dim — structured per-dimension split descriptor that maps
                    // 1:1 onto a #routing.level. Field order is contractual for AST
                    // extraction:
                    //   fullsize   (0): full PADDED dim length (raw shape source)
                    //   tile_round (1): explicit outer round/group count (0 => derive)
                    //   tile_size  (2): outer per-tile slice length
                    //   stride     (3): outer step (overlap = tile_size - stride)
                    //   padsize    (4): per-side pad (metadata/boundary only)
                    //   slice_tiling (5): nested inner level (tile_level)
                    // tile_level — nested second-level (on-core temporal) split on the
                    // SAME axis as the parent tile_dim, mirroring #routing.level. A
                    // by-value inner member (a struct cannot contain itself by value,
                    // hence the separate type). Field order (tile_size=0, stride=1,
                    // rounds=2) is contractual for AST extraction.
                    ret += "struct tile_level {\n";
                    ret += "  int tile_size = 0;\n";
                    ret += "  int stride    = 0;\n";
                    ret += "  int rounds    = 0;\n";
                    ret += "};\n";
                    ret += "struct tile_dim {\n";
                    ret += "  int fullsize   = 0;\n";
                    ret += "  int tile_round = 0;\n";
                    ret += "  int tile_size  = 0;\n";
                    ret += "  int stride     = 0;\n";
                    ret += "  int padsize    = 0;\n";
                    ret += "  tile_level slice_tiling;\n"; // nested inner level (field 5)
                    ret += "};\n";
                    // SpatialPolicy — 3-part orthogonal policy: array mapping
                    // (map), materialization (mat), and resource/pipeline
                    // schedule (sched). Field order (map=0, mat=1, sched=2) is
                    // contractual for AST extraction. The struct name stays
                    // `SpatialPolicy` so GemmSpace/Conv2dSpace field indices and
                    // the `.policy = {...}` initializer syntax are untouched.
                    ret += "struct SpatialMap {\n";
                    ret += "  Pattern act         = Pattern::Broadcast;\n";
                    ret += "  Pattern wgt         = Pattern::Broadcast;\n";
                    ret += "  Layout  layout      = Layout::Row;\n";
                    ret += "  Flow    merge_order = Flow::Default;\n";
                    ret += "  int     mesh_tiling_group1_dim = -1;\n";
                    ret += "  int     mesh_tiling_group2_dim = -1;\n";
                    ret += "};\n";
                    ret += "struct Materialize {\n";
                    ret += "  PadMaterialize pad    = PadMaterialize::DDR;\n";
                    ret += "  Im2col         im2col = Im2col::None;\n";
                    ret += "};\n";
                    ret += "struct Schedule {\n";
                    ret += "  int   pp_depth  = 2;\n";
                    ret += "  Bytes l1_budget = Bytes{32*1024};\n";
                    ret += "};\n";
                    ret += "struct SpatialPolicy {\n";
                    ret += "  SpatialMap  map;\n";
                    ret += "  Materialize mat;\n";
                    ret += "  Schedule    sched;\n";
                    ret += "};\n";
                    // GlobalPolicy — per-kernel GLOBAL (module-level) policy, NOT a
                    // per-port space. A file-scope constexpr instance named
                    // `<kernelName>_policy` binds by name convention to that kernel.
                    // fullconnect_auto: 1 (default) = full-connect M×N cartesian DMA
                    // repeat (A repeats while B cycles); 0 = no repeat (A and B each
                    // sent once, following the tiling/halo distribution).
                    ret += "struct GlobalPolicy {\n";
                    ret += "  int fullconnect_auto = 1;\n";
                    ret += "};\n";
                    // GemmSpace — composes a SpatialPolicy with the GEMM iteration
                    // space. Field order is contractual for AST extraction:
                    // 0 policy, 1 m, 2 n, 3 k (legacy/global GEMM dims, also used
                    // by conv-halo path), 4 d1, 5 d2, 6 d3, 7 d4 (NEW per-port
                    // 2D/3D/4D dims). When d1/d2 are set the port describes its OWN
                    // matrix (role-aware); otherwise the legacy m/n/k path is used.
                    // Keeping m/n/k means conv2d's positional field indices
                    // (1,2,3) stay untouched. d3/d4 are bookkeeping/coalescing dims
                    // (e.g. d4 carries the padded channel coverage for the filter).
                    ret += "struct GemmSpace {\n";
                    ret += "  SpatialPolicy policy;\n";
                    ret += "  tile_dim m;\n";
                    ret += "  tile_dim n;\n";
                    ret += "  tile_dim k;\n";
                    ret += "  tile_dim d1;\n";
                    ret += "  tile_dim d2;\n";
                    ret += "  tile_dim d3;\n";
                    ret += "  tile_dim d4;\n";
                    ret += "};\n";
                    // Conv2dSpace — composes a SpatialPolicy with the conv iteration
                    // space. Field order is contractual: 0 policy, 1 ih, 2 iw,
                    // 3 ic, 4 oc, 5 kh, 6 kw, 7 stride, 8 pad, 9 m. ih/iw carry the
                    // EXACT input spatial dims (output OH/OW are derived via the
                    // forward conv formula) so strided convs stay lossless. Field 9
                    // (m) is the optional explicit spatial-halo split (size=halo
                    // slice rows, stride=halo step, groups=tile-rows); when omitted
                    // (size==0) the OH/HW_ROWS auto-derivation is used instead.
                    ret += "struct Conv2dSpace {\n";
                    ret += "  SpatialPolicy policy;\n";
                    ret += "  tile_dim ih;\n";
                    ret += "  tile_dim iw;\n";
                    ret += "  tile_dim ic;\n";
                    ret += "  tile_dim oc;\n";
                    ret += "  tile_dim kh;\n";
                    ret += "  tile_dim kw;\n";
                    ret += "  int stride = 1;\n";
                    ret += "  int pad = 0;\n";
                    ret += "  tile_dim m;\n";
                    ret += "};\n";
                    // ConvGeom — raw conv2d geometry for the declarative
                    // Conv2dSpace_Spatial. Field order is contractual for AST
                    // extraction: 0 in_h,1 in_w,2 cin,3 cout,4 ksize,5 S,
                    // 6 pad_lo,7 pad_hi,8 cin_aligned. cin_aligned is metadata
                    // only (does NOT change the contraction K or raw_wc).
                    // NOTE: the kernel-size field is `ksize` (not `K`) to avoid
                    // colliding with a user `#define K ...` (e.g. GEMM K macro).
                    ret += "struct ConvGeom {\n";
                    ret += "  int in_h=0; int in_w=0; int cin=0; int cout=0;\n";
                    ret += "  int ksize=0; int S=1; int pad_lo=0; int pad_hi=0; int cin_aligned=0;\n";
                    ret += "};\n";
                    // Conv2dSpace_Spatial — declarative spatial-halo conv space.
                    // The user describes raw geometry (geom) + a desired output
                    // tile (out_tile_h/out_tile_w) + an objective; the compiler
                    // derives the spatial-halo split deterministically. Field
                    // order is contractual: 0 geom,1 out_tile_h,2 out_tile_w,
                    // 3 objective,4 policy (policy is field 4, not 0). Fields
                    // 5 d1, 6 d2 are the NEW explicit halo descriptors: d1 =
                    // height halo (split across mesh ROWS), d2 = width*C halo
                    // (split across mesh COLS). When d1.tile_size>0 the explicit
                    // descriptor OVERRIDES the out_tile_h/out_tile_w derivation.
                    // Legacy .out_tile_h forms leave d1/d2 zero (no override).
                    ret += "struct Conv2dSpace_Spatial {\n";
                    ret += "  ConvGeom      geom;\n";
                    ret += "  int           out_tile_h = 0;\n";
                    ret += "  int           out_tile_w = 0;\n";
                    ret += "  Objective     objective  = Objective::MaxArrayUtil;\n";
                    ret += "  SpatialPolicy policy;\n";
                    ret += "  tile_dim      d1;\n";
                    ret += "  tile_dim      d2;\n";
                    ret += "};\n";
                    // DmaTransform — general multi-dim DMA descriptor with factory methods
                    ret += "struct DmaTransform {\n";
                    ret += "  struct Dim { int stride; int wrap; };\n";
                    ret += "  Dim dims[4] = {};\n";
                    ret += "  int num_dims = 0;\n";
                    ret += "  int iter_step = 0;\n";
                    ret += "  int iter_wrap = 0;\n";
                    ret += "  int mode = 0;\n";       // 0 = flat/im2col-by-dims, 1 = spatial_halo
                    ret += "  int halo_slice = 0;\n"; // input rows per tile-row
                    ret += "  int halo_step = 0;\n";  // row stride between tile-rows
                    ret += "  int split_dim = 0;\n";  // dim carrying the halo split
                    ret += "  int raw_h = 0;\n";      // raw input H
                    ret += "  int raw_wc = 0;\n";     // raw input W*C
                    ret += "  int kernel_h = 0;\n";   // KERNEL_H
                    ret += "  int kernel_w = 0;\n";   // KERNEL_W
                    ret += "  int input_c = 0;\n";    // INPUT_C
                    ret += "  int stride = 0;\n";     // STRIDE
                    ret += "  int ow = 0;\n";         // OUTPUT_W
                    ret += "  int oh_per_row = 0;\n"; // OUTPUT_H / HW_ROWS
                    ret += "  static constexpr DmaTransform flat() { return {}; }\n";
                    ret += "  static constexpr DmaTransform im2col(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    int OW = (W + 2*P - KW) / S + 1;\n";
                    ret += "    int OH = (H + 2*P - KH) / S + 1;\n";
                    ret += "    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};\n";
                    ret += "    d.num_dims = 3;\n";
                    ret += "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform dilated_im2col(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P, int D) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;\n";
                    ret += "    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;\n";
                    ret += "    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};\n";
                    ret += "    d.num_dims = 3;\n";
                    ret += "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform pool(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P) {\n";
                    ret += "    return im2col(H, W, C, KH, KW, S, P);\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P, int G) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    int CPG = C / G;\n";
                    ret += "    int OW = (W + 2*P - KW) / S + 1;\n";
                    ret += "    int OH = (H + 2*P - KH) / S + 1;\n";
                    ret += "    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};\n";
                    ret += "    d.num_dims = 3;\n";
                    ret += "    d.iter_step = W * C * S; d.iter_wrap = OH;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform transpose(int rows, int cols) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};\n";
                    ret += "    d.num_dims = 2;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    d.dims[0] = {H * W, C};\n";
                    ret += "    d.dims[1] = {1, W};\n";
                    ret += "    d.num_dims = 2;\n";
                    ret += "    d.iter_step = W;\n";
                    ret += "    d.iter_wrap = H;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};\n";
                    ret += "    d.num_dims = 2;\n";
                    ret += "    d.iter_step = 1; d.iter_wrap = C;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    // Spatial-halo conv tiling: distribute the RAW input [H, W*C] across
                    // R tile-rows as overlapping contiguous row-blocks. Each tile-row owns
                    // OH/R output rows -> needs ((OH/R)-1)*S + KH input rows (halo_slice),
                    // advancing by (OH/R)*S input rows per tile-row (halo_step). The shim
                    // BD stays flat; overlap is realized via per-tile DDR base offsets.
                    ret += "  static constexpr DmaTransform spatial(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P, int R) {\n";
                    ret += "    DmaTransform d;\n";
                    ret += "    int OH = (H + 2*P - KH) / S + 1;\n";
                    ret += "    int oh_per_row = OH / R;\n";
                    ret += "    d.mode = 1;\n";
                    ret += "    d.halo_slice = (oh_per_row - 1) * S + KH;\n";
                    ret += "    d.halo_step  = oh_per_row * S;\n";
                    ret += "    d.split_dim  = 0;\n";
                    ret += "    d.raw_h  = H;\n";
                    ret += "    d.raw_wc = W * C;\n";
                    ret += "    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;\n";
                    ret += "    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;\n";
                    ret += "    return d;\n";
                    ret += "  }\n";
                    ret += "};\n";
                    // ConvTiling — convenience namespace-like struct exposing spatial().
                    ret += "struct ConvTiling {\n";
                    ret += "  static constexpr DmaTransform spatial(int H, int W, int C,\n";
                    ret += "      int KH, int KW, int S, int P, int R) {\n";
                    ret += "    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);\n";
                    ret += "  }\n";
                    ret += "};\n";
                    // port — Space NTTP accepts a bare SpatialPolicy (legacy),
                    // a GemmSpace, or a Conv2dSpace (all C++20 structural NTTP types).
                    ret += "template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { "
                           "using type = T; };\n";
                    // Built-in query function stubs — Clang parses these but the compiler
                    // replaces the calls with computed integer literals during kernel body rewriting.
                    ret += "template<typename T> constexpr int get_num_rounds(T) { return 0; }\n";
                    ret += "template<typename T> constexpr int get_buffer_size(T) { return 0; }\n";
                    ret += "constexpr int get_tile_rows() { return 0; }\n";
                    ret += "constexpr int get_tile_cols() { return 0; }\n";
                    ret += "constexpr int get_data_row() { return 0; }\n";
                    ret += "constexpr int get_data_col() { return 0; }\n";
                    ret += "constexpr int get_k_dim() { return 0; }\n";
                    ret += "constexpr int get_tile_m() { return 0; }\n";
                    ret += "constexpr int get_tile_n() { return 0; }\n";
                    ret += "constexpr int get_effective_k() { return 0; }\n";
                    ret += "constexpr int get_k_rounds() { return 0; }\n";
                    ret += "constexpr int get_spatial_m_rounds() { return 0; }\n";
                    ret += "constexpr int get_spatial_n_rounds() { return 0; }\n";
                    ret += "template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }\n";
                    // IR-sourced per-dim per-round on-core slice size for a port.
                    // Resolved from the routing.partitiontensor TilingAttr inside runPipeline
                    // (NOT from frontend scalars), so the stub returns 0 here.
                    ret += "template<typename T> constexpr int get_arg_per_round_size_in_dim(int, T) { return 0; }\n";
                    // IR-sourced per-dim total on-core rounds to cover the full tensor dim
                    // for a port. Resolved from the routing.partitiontensor TilingAttr inside
                    // runPipeline (NOT from frontend scalars), so the stub returns 0 here.
                    ret += "template<typename T> constexpr int get_arg_total_rounds_in_dim(int, T) { return 0; }\n";
                    // Conv accessors (spatial-halo kernel): resolved to integer literals
                    // during kernel body rewriting (DerivedTilingParams).
                    ret += "constexpr int get_kernel_h() { return 0; }\n";
                    ret += "constexpr int get_kernel_w() { return 0; }\n";
                    ret += "constexpr int get_input_c() { return 0; }\n";
                    ret += "constexpr int get_stride() { return 0; }\n";
                    ret += "constexpr int get_ow() { return 0; }\n";
                    ret += "constexpr int get_oh_per_row() { return 0; }\n";
                    ret += "constexpr int get_halo_slice() { return 0; }\n";
                    ret += "}\n";
                    ret += "struct aiePartition {\n";
                    ret += "    int startCol, endCol, startRow, endRow;\n";
                    ret += "};\n";
                    // Forward declarations needed by aieArray::partition()/alloc()/free()/synchronizecpu()
                    ret += "struct XAie_DevInst;\n";
                    ret += "extern \"C\" XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int "
                           "numCols);\n";
                    ret += "extern \"C\" XAie_DevInst *__Runtime_get_partition_dev(int meshId);\n";
                    ret += "extern \"C\" void *__Runtime_alloc_buffer(XAie_DevInst *dev, __SIZE_TYPE__ size_bytes);\n";
                    ret += "extern \"C\" void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);\n";
                    ret += "extern \"C\" void *__Runtime_Alloc(__SIZE_TYPE__ bytes);\n";
                    ret +=
                        "extern \"C\" void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, __SIZE_TYPE__ size);\n";
                    ret += "extern \"C\" void __Runtime_teardown_all();\n";
                    // New programming model types: aieMesh + aieArray
                    ret += "struct aieMesh {\n";
                    ret += "    int rows, cols;\n";
                    ret += "    aiePartition partition;\n";
                    ret += "    int meshId;\n";
                    ret += "};\n";
                    ret += "struct aieArray {\n";
                    ret += "    int nextMeshId = 0;\n";
                    ret += "    XAie_DevInst* _dev = nullptr;\n";
                    ret += "    aieMesh partition(aiePartition p, int rows, int cols) {\n";
                    ret += "        int meshId = nextMeshId++;\n";
                    ret += "        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol + "
                           "1);\n";
                    ret += "        return aieMesh{rows, cols, p, meshId};\n";
                    ret += "    }\n";
                    ret += "    aieMesh partition(int rows, int cols) {\n";
                    ret += "        int meshId = nextMeshId++;\n";
                    ret += "        _dev = __Runtime_init_mesh_partition(meshId, 0, cols);\n";
                    ret += "        return aieMesh{rows, cols, {0, cols - 1, 0, rows - 1}, meshId};\n";
                    ret += "    }\n";
                    ret += "    void* alloc(__SIZE_TYPE__ size) { return __Runtime_alloc_buffer(_dev, size); }\n";
                    ret += "    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }\n";
                    ret += "    void synchronizecpu(void* ptr, __SIZE_TYPE__ size) { __Runtime_sync_for_cpu(_dev, ptr, "
                           "size); }\n";
                    ret += "};\n";
                    // Backward-compatible aieDim
                    ret += "struct aieDim {\n";
                    ret += "    int rows, cols;\n";
                    ret += "    aiePartition partition;\n";
                    ret += "    bool hasPartition;\n";
                    ret +=
                        "    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}\n";
                    ret += "    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), "
                           "hasPartition(true) {}\n";
                    ret += "};\n";
                    ret += "inline void aieSetDevice(int) {}\n";
                    ret += "inline void aieDeviceSynchronize() {}\n";
                    // Stub extern + __aie_launch for Clang AST parsing only.
                    // parsedTensors is empty at this point (pre-AST-visit), so we
                    // emit a no-op template.  The correct forwarding overload is
                    // generated later by TilingLinalgPipeline::runPipeline when
                    // appending user source to host.cc.
                    ret += "extern void host_canonicalized(...);\n";
                    ret += "template<typename... Args>\n";
                    ret += "inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {\n";
                    ret += "    (void)kernel; (void)mesh; (void)sizeof...(args);\n";
                    ret += "}\n";
                    ret += "template<typename... Args>\n";
                    ret += "inline void __aie_launch(const char* kernel, aieDim mesh, Args... args) {\n";
                    ret += "    (void)kernel; (void)mesh; (void)sizeof...(args);\n";
                    ret += "}\n";
                    ret += "#endif\n\n";
                }

                if (!kernel_name_list.empty()) {
                    ret += "#include \"xaiengine.h\"\n";
                    ret += "#ifndef AIEHLC_KERNEL_LAUNCH_HELPERS_DEFINED\n";
                    ret += "#define AIEHLC_KERNEL_LAUNCH_HELPERS_DEFINED\n";
                    ret += "static XAie_DevInst *__aiehlc_global_dev_inst = nullptr;\n";
                    ret += "#define AIEHLC_MAX_LAUNCHED_TILES 1024\n";
                    ret += "static XAie_LocType __aiehlc_launched_tiles[AIEHLC_MAX_LAUNCHED_TILES];\n";
                    ret += "static int __aiehlc_launched_tile_count = 0;\n";
                    ret += "inline void __use_device_instance(XAie_DevInst *DevInst) {\n";
                    ret += "    __aiehlc_global_dev_inst = DevInst;\n";
                    ret += "}\n";
                    ret += "inline void __sync_kernels() {\n";
                    ret += "    if (__aiehlc_global_dev_inst == nullptr) {\n";
                    ret += "        return;\n";
                    ret += "    }\n";
                    ret += "    for (int i = 0; i < __aiehlc_launched_tile_count; ++i) {\n";
                    ret += "        while (XAie_CoreWaitForDone(__aiehlc_global_dev_inst, __aiehlc_launched_tiles[i], "
                           "0) != XAIE_OK) {}\n";
                    ret += "    }\n";
                    ret += "    __aiehlc_launched_tile_count = 0;\n";
                    ret += "}\n";
                    ret += "#endif\n";
                }

                for (auto x:kernel_name_list) {
                    ret += "extern unsigned char _binary_kernel_" + x + "_start[];\n" +
                           "template <int Col, int Row, typename... Args>\n"
                           "inline void " +
                           x +
                           "(Args&&... args) {\n"
                           "    (void)sizeof...(args); \n"
                           "    XAie_DevInst *DevInst = __aiehlc_global_dev_inst;\n"
                           "    if (DevInst == nullptr) {\n"
                           "        return;\n"
                           "    }\n"
                           //   "    XAie_CoreDisable(DevInst, XAie_TileLoc(Col,Row));\n"
                           //   "    XAie_CoreReset(DevInst, XAie_TileLoc(Col,Row));\n"
                           //   "    XAie_LoadElfMem(... _binary_kernel_" + x + "_start);\n"
                           //   "    XAie_CoreUnreset(DevInst, XAie_TileLoc(Col,Row));\n"
                           //   "    XAie_CoreEnable(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_CoreReset(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_CoreUnreset(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_LoadElfMem(DevInst, XAie_TileLoc(Col,Row), (unsigned char *)_binary_kernel_" +
                           x +
                           "_start);\n"
                           "    XAie_CoreEnable(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    if (__aiehlc_launched_tile_count < AIEHLC_MAX_LAUNCHED_TILES) {\n"
                           "        __aiehlc_launched_tiles[__aiehlc_launched_tile_count++] = XAie_TileLoc(Col,Row);\n"
                           "    } else {\n"
                           "        while (XAie_CoreWaitForDone(DevInst, XAie_TileLoc(Col,Row), 0) != XAIE_OK) {}\n"
                           "    }\n"
                           "    return;\n"
                           "};\n";
                }
                ret += SourceCodeString;
				// std::cout << ret << std::endl;
				/*
					 llvm::StringRef ModifiedCode = SourceCodeString;
					 std::unique_ptr<llvm::MemoryBuffer> ModifiedBuffer =
					 llvm::MemoryBuffer::getMemBufferCopy(ModifiedCode," Filename");

					 SourceMgr.overrideFileContents(FileEntry, std::move(ModifiedBuffer));

				// Get the modified file name
				llvm::StringRef ModifiedFileName = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID())->getName();

				// Print modified file name
				llvm::outs() << "Modified file name: " << ModifiedFileName  << ModifiedCode<< "\n";
				*/
			}
			return ret;
		}

		void EndSourceFileAction() override {
			//  std::cout << "---------EndSourceFileAction---------" << std::endl;

            // Stop immediately if Clang reported any compilation errors.
            // Running the MLIR pipeline on an erroneous AST produces misleading output.
            if (getCompilerInstance().getDiagnostics().hasErrorOccurred()) {
                llvm::errs() << "\n=========================================================\n"
                             << "Error: source file has compilation errors. Stopping.\n"
                             << "=========================================================\n";
                return;
            }

             const RewriteBuffer *RewriteBuf = TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID());
			 if (RewriteBuf) {
				 //RewriteBuf->write(llvm::outs());
        //llvm::outs() << "Rewritten Source Code:\n" << RewriteBuf << "\n";
	  		} else {
					llvm::errs() << "RewriteBuf is null\n";
				}

			if (isTilingLinalgMode) {
				// ---- NEW PATH: tilinglinalg pipeline ----
				llvm::outs() << "[TilingLinalg] Running multi-tile pipeline (mesh="
							 << tilingMeshRows << "x" << tilingMeshCols << ")\n";

                // Capture user's rewritten source for merging with pipeline output.
                // The RewriteBuf contains the entire user source with __global__ body
                // removed and <<<mesh>>> syntax replaced by __aie_launch() calls.
                std::string userRewrittenSource;
                if (RewriteBuf) {
                    llvm::raw_string_ostream rso(userRewrittenSource);
                    RewriteBuf->write(rso);
                    rso.flush();
                }

                // Post-process userRewrittenSource: inject byte sizes into __aie_launch() calls.
                // The <<<>>> rewriter ran in BeginSourceFileAction (before AST visit), so
                // parsedMeshKernels was empty at that time and no sizes were injected.
                // Now parsedMeshKernels is populated, so we can do it.
                if (!userRewrittenSource.empty() && !parsedMeshKernels.empty()) {
                    size_t searchPos = 0;
                    while ((searchPos = userRewrittenSource.find("__aie_launch(", searchPos)) != std::string::npos) {
                        // Find the opening paren of __aie_launch(
                        size_t openParen = searchPos + strlen("__aie_launch");
                        // Find matching close paren
                        int depth = 1;
                        size_t pos = openParen + 1;
                        while (depth > 0 && pos < userRewrittenSource.size()) {
                            if (userRewrittenSource[pos] == '(')
                                depth++;
                            if (userRewrittenSource[pos] == ')')
                                depth--;
                            if (depth > 0)
                                pos++;
                        }
                        // Extract full arg string between parens
                        std::string allArgs = userRewrittenSource.substr(openParen + 1, pos - openParen - 1);

                        // Split by commas respecting nested parens
                        auto splitComma = [](const std::string &s) -> std::vector<std::string> {
                            std::vector<std::string> result;
                            int d = 0;
                            size_t start = 0;
                            for (size_t j = 0; j < s.size(); ++j) {
                                if (s[j] == '(')
                                    d++;
                                else if (s[j] == ')')
                                    d--;
                                else if (s[j] == ',' && d == 0) {
                                    std::string a = s.substr(start, j - start);
                                    size_t b = a.find_first_not_of(" \t\n\r");
                                    size_t e = a.find_last_not_of(" \t\n\r");
                                    if (b != std::string::npos)
                                        result.push_back(a.substr(b, e - b + 1));
                                    start = j + 1;
                                }
                            }
                            std::string a = s.substr(start);
                            size_t b = a.find_first_not_of(" \t\n\r");
                            size_t e = a.find_last_not_of(" \t\n\r");
                            if (b != std::string::npos)
                                result.push_back(a.substr(b, e - b + 1));
                            return result;
                        };
                        std::vector<std::string> args = splitComma(allArgs);

                        // args[0] = kernel name string, args[1] = mesh variable, args[2..] = user args
                        // Extract kernel name from the quoted string literal (strip quotes)
                        if (args.size() >= 3) {
                            std::string kernelLit = args[0];
                            // Remove surrounding quotes: "matmul" -> matmul
                            size_t q1 = kernelLit.find('"');
                            size_t q2 = kernelLit.rfind('"');
                            std::string kernelName;
                            if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1)
                                kernelName = kernelLit.substr(q1 + 1, q2 - q1 - 1);

                            const MeshKernelDesc *mkdPtr = nullptr;
                            for (auto &mk : parsedMeshKernels) {
                                if (mk.kernelName == kernelName) {
                                    mkdPtr = &mk;
                                    break;
                                }
                            }

                            if (mkdPtr) {
                                unsigned numPtrArgs = (unsigned)mkdPtr->tensors.size();
                                // Rebuild the call with interleaved sizes
                                std::string newCall = "__aie_launch(" + args[0] + ", " + args[1];
                                for (unsigned ai = 2; ai < args.size(); ++ai) {
                                    newCall += ", " + args[ai];
                                    unsigned tensorIdx = ai - 2;
                                    if (tensorIdx < numPtrArgs) {
                                        int64_t bytes = 1;
                                        for (auto dim : mkdPtr->tensors[tensorIdx].shape)
                                            bytes *= dim;
                                        bytes *= mkdPtr->tensors[tensorIdx].elementBitWidth / 8;
                                        newCall += ", (size_t)" + std::to_string(bytes);
                                    }
                                }
                                newCall += ")";
                                userRewrittenSource.replace(searchPos, pos + 1 - searchPos, newCall);
                                searchPos += newCall.size();
                                continue;
                            }
                        }
                        searchPos = pos + 1;
                    }
                }

                // Derive AIE generation string from the Clang preprocessor's AIE_GEN macro
                // (set via --extra-arg=-DAIE_GEN=<version> at runtime, not compile-time)
                std::string aieGenStr = "Gen2"; // default
                {
                    auto &PP = getCompilerInstance().getPreprocessor();
                    auto *II = PP.getIdentifierInfo("AIE_GEN");
                    if (II && II->hasMacroDefinition()) {
                        auto *MI = PP.getMacroInfo(II);
                        if (MI && MI->getNumTokens() == 1 &&
                            MI->getReplacementToken(0).is(clang::tok::numeric_constant)) {
                            llvm::StringRef valStr = MI->getReplacementToken(0).getLiteralData()
                                                         ? llvm::StringRef(MI->getReplacementToken(0).getLiteralData(),
                                                                           MI->getReplacementToken(0).getLength())
                                                         : "";
                            int aieGen = 2;
                            valStr.getAsInteger(10, aieGen);
                            if (aieGen == 1)
                                aieGenStr = "Gen1";
                            else if (aieGen >= 5)
                                aieGenStr = "Gen5";
                        }
                    }
                }
            llvm::outs() << "[TilingLinalg] AIE generation: " << aieGenStr << "\n";

            mlir::MLIRContext ctx;
            TilingLinalgPipeline::registerDialects(ctx);

            std::string outputDir = std::string(AOUT) + "worklocal/";
            bool isMultiKernel = parsedMeshKernels.size() > 1;

            // ---- Multi-kernel loop: run pipeline per kernel ----
            if (isMultiKernel) {
                llvm::outs() << "[TilingLinalg] Multi-kernel mode: " << parsedMeshKernels.size() << " kernels\n";

                for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                    auto &mkd = parsedMeshKernels[ki];
                    llvm::outs() << "[TilingLinalg] === Pipeline run " << ki << ": kernel=" << mkd.kernelName
                                 << " mesh=" << mkd.meshRows << "x" << mkd.meshCols << " ===\n";

                    int rows = mkd.meshRows > 0 ? mkd.meshRows : 2;
                    int cols = mkd.meshCols > 0 ? mkd.meshCols : 2;

                    // Use per-kernel tensors and splitModel
                    SplitModel &splitModel = mkd.splitModel;
                    if (splitModel.tensorSplits.empty())
                        splitModel = SplitModel::gemm();

                    // Emit per-dim GEMM #routing.tiling for the even-mesh path (only
                    // when fullconnect_auto=1) so the partitiontensor op carries the
                    // on-core per-round slice (get_arg_per_round_size_in_dim + Phase D).
                    if (mkd.fullConnectAuto)
                        populateGemmTiling(splitModel, mkd.tensors, mkd.derivedParams, rows, cols);

                    // Build routing IR for this kernel
                    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, rows, cols, mkd.tensors, splitModel,
                                                                       mkd.partition, aieGenStr);

                    // Set K-round module attributes for downstream DMA passes
                    if (mkd.derivedParams.valid && mkd.derivedParams.kRounds > 1) {
                        mlir::OpBuilder attrBuilder(&ctx);
                        // M×N cartesian repeat / spatial sub-tiling attrs are only meaningful
                        // when fullconnect_auto is enabled. Dropping them for fullconnect_auto=0
                        // lets the downstream passes fall through to the pure-K-round path.
                        // The K-triple (effective_k/k_rounds/full_k) is sourced from the
                        // #routing.tiling op (populateGemmTiling) when fullconnect_auto=1, so it
                        // is only emitted as a module attr on the fullconnect_auto=0 else-branch
                        // (no tiling op exists there).
                        if (mkd.fullConnectAuto) {
                            // tile_m/tile_n, tile_rows/tile_cols, the M/N spatial rounds, and
                            // the K-triple are all sourced from the #routing.tiling op on
                            // partitiontensor (readGemmTilingScalars) by every downstream
                            // consumer, so none are emitted as flat module attrs here.
                            llvm::outs() << "[TilingLinalg] fullconnect_auto=1: tiling scalars sourced from "
                                            "#routing.tiling op (effective_k="
                                         << mkd.derivedParams.effectiveK << " k_rounds=" << mkd.derivedParams.kRounds
                                         << " full_k=" << mkd.derivedParams.kDim
                                         << " tile_rows=" << mkd.derivedParams.tileRows
                                         << " m_rounds=" << mkd.derivedParams.spatialMRounds
                                         << " tile_cols=" << mkd.derivedParams.tileCols
                                         << " n_rounds=" << mkd.derivedParams.spatialNRounds << ")\n";
                        } else {
                            // fullconnect_auto=0 pure-K-round path: no #routing.tiling op is
                            // emitted here, so the K-triple must be carried as module attrs.
                            module->setAttr("routing.effective_k",
                                            attrBuilder.getI64IntegerAttr(mkd.derivedParams.effectiveK));
                            module->setAttr("routing.k_rounds",
                                            attrBuilder.getI64IntegerAttr(mkd.derivedParams.kRounds));
                            module->setAttr("routing.full_k", attrBuilder.getI64IntegerAttr(mkd.derivedParams.kDim));
                            llvm::outs() << "[TilingLinalg] Set K-round module attrs (fullconnect_auto=0, "
                                            "tile_*/m_rounds/n_rounds dropped): effective_k="
                                         << mkd.derivedParams.effectiveK << " k_rounds=" << mkd.derivedParams.kRounds
                                         << " full_k=" << mkd.derivedParams.kDim << "\n";
                        }
                    }

                    // Spatial-halo conv: carry the contiguous IFM slab size so the
                    // kernel pass allocates the input window (BUF_SZ_IN/window_init)
                    // to match the kernel body's buf_sz (= halo_slice * raw_wc).
                    // Set unconditionally (kRounds may be 1 for conv2d), unlike the
                    // K-round attrs above.
                    if (mkd.derivedParams.valid && mkd.derivedParams.convHaloBufSize > 0) {
                        mlir::OpBuilder haloAttrBuilder(&ctx);
                        module->setAttr("routing.spatial_halo_buf_size",
                                        haloAttrBuilder.getI64IntegerAttr(mkd.derivedParams.convHaloBufSize));
                        llvm::outs() << "[TilingLinalg] Set spatial_halo_buf_size=" << mkd.derivedParams.convHaloBufSize
                                     << "\n";
                        // OUTPUT per-slab division count (see single-kernel path for the
                        // rationale): the conv2d_spatial kernel always emits one output
                        // slab per on-core round (spatialMRounds*spatialNRounds) regardless
                        // of fullconnect_auto, so emit the divisor UNCONDITIONALLY.
                        int64_t outRounds = mkd.derivedParams.spatialMRounds * mkd.derivedParams.spatialNRounds;
                        if (outRounds > 1) {
                            module->setAttr("routing.spatial_out_rounds", haloAttrBuilder.getI64IntegerAttr(outRounds));
                            llvm::outs() << "[TilingLinalg] Set spatial_out_rounds=" << outRounds << "\n";
                        }
                    }

                    // Full-connect auto flag (aie::GlobalPolicy <kernel>_policy).
                    // Set UNCONDITIONALLY (applies even when kRounds==1). 1 = default
                    // M×N cartesian DMA repeat; 0 = A/B each sent once.
                    {
                        mlir::OpBuilder fcAttrBuilder(&ctx);
                        module->setAttr("routing.fullconnect_auto",
                                        fcAttrBuilder.getI64IntegerAttr(mkd.fullConnectAuto ? 1 : 0));
                        llvm::outs() << "[TilingLinalg] Set fullconnect_auto=" << (mkd.fullConnectAuto ? 1 : 0)
                                     << " for kernel " << mkd.kernelName << "\n";
                    }

                    // Replace aie::get_*() calls in kernel body with computed integer literals
                    std::string resolvedBody = mkd.kernelBody;
                    if (mkd.derivedParams.valid && !resolvedBody.empty()) {
                        llvm::outs() << "[TilingLinalg] Replacing aie::get_*() calls in kernel body for "
                                     << mkd.kernelName << "\n";

                        // Build port name -> index map from stored varNames
                        std::unordered_map<std::string, size_t> paramToPort;
                        for (size_t pi = 0; pi < mkd.portVarNames.size(); ++pi) {
                            paramToPort[mkd.portVarNames[pi]] = pi;
                        }

                        // Replace per-port queries: aie::get_num_rounds(win_x),
                        // aie::get_buffer_size(win_x), aie::get_spatial_multiple_rounds(win_x)
                        for (const auto &funcName :
                             {"aie::get_num_rounds", "aie::get_buffer_size", "aie::get_spatial_multiple_rounds"}) {
                            std::string prefix = std::string(funcName) + "(";
                            size_t pos = 0;
                            while ((pos = resolvedBody.find(prefix, pos)) != std::string::npos) {
                                size_t argStart = pos + prefix.size();
                                size_t argEnd = resolvedBody.find(")", argStart);
                                if (argEnd == std::string::npos)
                                    break;
                                std::string argName = resolvedBody.substr(argStart, argEnd - argStart);
                                while (!argName.empty() && argName.front() == ' ')
                                    argName.erase(0, 1);
                                while (!argName.empty() && argName.back() == ' ')
                                    argName.pop_back();
                                auto it = paramToPort.find(argName);
                                if (it != paramToPort.end() && it->second < mkd.derivedParams.portParams.size()) {
                                    int64_t val;
                                    if (std::string(funcName) == "aie::get_num_rounds")
                                        val = mkd.derivedParams.portParams[it->second].numRounds;
                                    else if (std::string(funcName) == "aie::get_buffer_size")
                                        val = mkd.derivedParams.portParams[it->second].bufferSize;
                                    else
                                        val = mkd.derivedParams.portParams[it->second].spatialRounds;
                                    std::string replacement = std::to_string(val);
                                    resolvedBody.replace(pos, argEnd + 1 - pos, replacement);
                                    pos += replacement.size();
                                } else {
                                    pos = argEnd + 1;
                                }
                            }
                        }

                        // Replace simple calls: get_tile_rows(), get_tile_cols(), get_k_dim()
                        auto replaceSimpleCall = [&](const std::string &call, int64_t val) {
                            std::string replacement = std::to_string(val);
                            size_t pos = 0;
                            while ((pos = resolvedBody.find(call, pos)) != std::string::npos) {
                                resolvedBody.replace(pos, call.size(), replacement);
                                pos += replacement.size();
                            }
                        };
                        replaceSimpleCall("aie::get_tile_rows()", mkd.derivedParams.tileM > 0
                                                                      ? mkd.derivedParams.tileM
                                                                      : mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_tile_cols()", mkd.derivedParams.tileN > 0
                                                                      ? mkd.derivedParams.tileN
                                                                      : mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_data_row()", mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_data_col()", mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_k_dim()", mkd.derivedParams.kDim);
                        // Two-level tiling query functions
                        replaceSimpleCall("aie::get_tile_m()", mkd.derivedParams.tileM > 0
                                                                   ? mkd.derivedParams.tileM
                                                                   : mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_tile_n()", mkd.derivedParams.tileN > 0
                                                                   ? mkd.derivedParams.tileN
                                                                   : mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_effective_k()", mkd.derivedParams.effectiveK > 0
                                                                        ? mkd.derivedParams.effectiveK
                                                                        : mkd.derivedParams.kDim);
                        replaceSimpleCall("aie::get_k_rounds()", mkd.derivedParams.kRounds);
                        replaceSimpleCall("aie::get_spatial_m_rounds()", mkd.derivedParams.spatialMRounds);
                        replaceSimpleCall("aie::get_spatial_n_rounds()", mkd.derivedParams.spatialNRounds);
                        // Conv (spatial-halo) accessors
                        replaceSimpleCall("aie::get_kernel_h()", mkd.derivedParams.convKernelH);
                        replaceSimpleCall("aie::get_kernel_w()", mkd.derivedParams.convKernelW);
                        replaceSimpleCall("aie::get_input_c()", mkd.derivedParams.convInputC);
                        replaceSimpleCall("aie::get_stride()", mkd.derivedParams.convStride);
                        replaceSimpleCall("aie::get_ow()", mkd.derivedParams.convOW);
                        replaceSimpleCall("aie::get_oh_per_row()", mkd.derivedParams.convOHPerRow);
                        replaceSimpleCall("aie::get_halo_slice()", mkd.derivedParams.convHaloSlice);
                    }

                    // Prepend user macros to kernel body
                    std::string kernelBodyWithMacros = resolvedBody;
                    if (!userMacroDefines.empty() && !resolvedBody.empty()) {
                        std::string macroBlock = "// User macro definitions from source file\n";
                        for (const auto &macro : userMacroDefines) {
                            macroBlock += macro + "\n";
                        }
                        macroBlock += "\n";
                        kernelBodyWithMacros = macroBlock + resolvedBody;
                    }

                    // First kernel writes host.cc fresh; subsequent kernels append
                    bool appendMode = (ki > 0);
                    // In multi-kernel mode, always suffix host function with kernel name
                    std::string hostFuncSuffix = mkd.kernelName;
                    // Don't pass userRewrittenSource in multi-kernel mode — we emit it ourselves after all runs
                    unsigned hostDdrArgs = 0;
                    if (!TilingLinalgPipeline::runPipeline(
                            ctx, module, outputDir, kernelBodyWithMacros, mkd.kernelFuncName, parsedDebugLevel,
                            /*userRewrittenSource=*/"", {}, mkd.maxPPBytes, aieGenStr, hostFuncSuffix, appendMode,
                            &hostDdrArgs, mkd.portVarNames)) {
                        llvm::errs() << "[TilingLinalg] Pipeline FAILED for kernel: " << mkd.kernelName << "\n";
                        std::exit(1);
                    }
                    mkd.numHostDdrArgs = hostDdrArgs;
                    llvm::outs() << "[TilingLinalg] Pipeline completed for kernel: " << mkd.kernelName
                                 << " (hostDdrArgs=" << hostDdrArgs << ")\n";
                }

                // ---- Post-pipeline: append user source + unified __aie_launch to host.cc ----
                {
                    std::string hostPath = outputDir + "/host.cc";
                    std::error_code ec;
                    llvm::raw_fd_ostream stream(hostPath, ec, llvm::sys::fs::OF_Append);
                    if (ec) {
                        llvm::errs() << "Failed to open " << hostPath << " for appending: " << ec.message() << "\n";
                        std::exit(1);
                    }

                    stream << "\n// ===== User source + multi-kernel __aie_launch dispatch =====\n";
                    stream << "#define AIEHLC_TILING_STUBS_DEFINED\n";
                    stream << "#include <cstring>\n";
                    // Re-emit SpatialPolicy types so user source constexpr definitions compile
                    stream << "namespace aie {\n";
                    stream << "enum class Pattern  { Broadcast = 0, Scatter = 1, Distribute = 1, Multicast = 2, Gather "
                              "= 3 };\n";
                    stream << "enum class Layout   { Row = 0, Col = 1, Grid = 2 };\n";
                    stream << "enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };\n";
                    stream << "enum class PadMaterialize { DDR = 0, Memtile = 1 };\n";
                    stream << "enum class Im2col   { None = 0, Dma = 1 };\n";
                    stream << "enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };\n";
                    stream << "enum class TileMode { Partition = 0, Overlap = 1 };\n";
                    stream << "enum class Objective { MaxArrayUtil = 0, MinLatency = 1, MinDma = 2 };\n";
                    stream << "struct Bytes { int value = 0; };\n";
                    // tile_level — nested second-level (on-core temporal) split (see
                    // Clang stub above). Field order (tile_size=0, stride=1, rounds=2).
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
                    // ConvGeom + Conv2dSpace_Spatial — declarative spatial-halo
                    // conv space (geom field order 0 in_h,1 in_w,2 cin,3 cout,
                    // 4 ksize,5 S,6 pad_lo,7 pad_hi,8 cin_aligned; space field
                    // order 0 geom,1 out_tile_h,2 out_tile_w,3 objective,4 policy).
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
                    stream << "    return d;\n";
                    stream << "  }\n";
                    stream << "};\n";
                    stream << "struct ConvTiling {\n";
                    stream << "  static constexpr DmaTransform spatial(int H, int W, int C,\n";
                    stream << "      int KH, int KW, int S, int P, int R) {\n";
                    stream << "    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);\n";
                    stream << "  }\n";
                    stream << "};\n";
                    stream << "template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct "
                              "port { using type = T; };\n";
                    stream << "template<typename T> constexpr int get_num_rounds(T) { return 0; }\n";
                    stream << "template<typename T> constexpr int get_buffer_size(T) { return 0; }\n";
                    stream << "constexpr int get_tile_rows() { return 0; }\n";
                    stream << "constexpr int get_tile_cols() { return 0; }\n";
                    stream << "constexpr int get_data_row() { return 0; }\n";
                    stream << "constexpr int get_data_col() { return 0; }\n";
                    stream << "constexpr int get_k_dim() { return 0; }\n";
                    stream << "constexpr int get_tile_m() { return 0; }\n";
                    stream << "constexpr int get_tile_n() { return 0; }\n";
                    stream << "constexpr int get_effective_k() { return 0; }\n";
                    stream << "constexpr int get_k_rounds() { return 0; }\n";
                    stream << "constexpr int get_spatial_m_rounds() { return 0; }\n";
                    stream << "constexpr int get_spatial_n_rounds() { return 0; }\n";
                    stream << "template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }\n";
                    // IR-sourced per-dim per-round on-core slice size (resolved in runPipeline).
                    stream
                        << "template<typename T> constexpr int get_arg_per_round_size_in_dim(int, T) { return 0; }\n";
                    // IR-sourced per-dim total on-core rounds over the dim (resolved in runPipeline).
                    stream << "template<typename T> constexpr int get_arg_total_rounds_in_dim(int, T) { return 0; }\n";
                    stream << "constexpr int get_kernel_h() { return 0; }\n";
                    stream << "constexpr int get_kernel_w() { return 0; }\n";
                    stream << "constexpr int get_input_c() { return 0; }\n";
                    stream << "constexpr int get_stride() { return 0; }\n";
                    stream << "constexpr int get_ow() { return 0; }\n";
                    stream << "constexpr int get_oh_per_row() { return 0; }\n";
                    stream << "constexpr int get_halo_slice() { return 0; }\n";
                    stream << "}\n";
                    // aiePartition struct
                    stream << "struct aiePartition {\n";
                    stream << "    int startCol, endCol, startRow, endRow;\n";
                    stream << "};\n";
                    // aieMesh + aieArray
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
                    stream << "        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol "
                              "+ 1);\n";
                    stream << "        return aieMesh{rows, cols, p, meshId};\n";
                    stream << "    }\n";
                    stream << "    aieMesh partition(int rows, int cols) {\n";
                    stream << "        int meshId = nextMeshId++;\n";
                    stream << "        _dev = __Runtime_init_mesh_partition(meshId, 0, cols);\n";
                    stream << "        return aieMesh{rows, cols, {0, cols - 1, 0, rows - 1}, meshId};\n";
                    stream << "    }\n";
                    stream << "    void* alloc(size_t size) { return __Runtime_alloc_buffer(_dev, size); }\n";
                    stream << "    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }\n";
                    stream << "    void synchronizecpu(void* ptr, size_t size) { __Runtime_sync_for_cpu(_dev, ptr, "
                              "size); }\n";
                    stream << "};\n";
                    // aieDim (backward compat)
                    stream << "struct aieDim {\n";
                    stream << "    int rows, cols;\n";
                    stream << "    aiePartition partition;\n";
                    stream << "    bool hasPartition;\n";
                    stream << "    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, "
                              "hasPartition(false) {}\n";
                    stream << "    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), "
                              "hasPartition(true) {}\n";
                    stream << "};\n";
                    stream << "inline void aieSetDevice(int) {}\n";
                    stream << "inline void aieDeviceSynchronize() {}\n";

                    // Emit extern declarations for per-kernel binary symbols
                    // (must appear before __aie_launch which references them)
                    for (auto &mkd : parsedMeshKernels) {
                        stream << "extern unsigned char _binary_kernel_" << mkd.kernelName << "_start[];\n";
                        stream << "extern unsigned char _binary_kernel_" << mkd.kernelName << "_end[];\n";
                        stream << "extern unsigned int _binary_kernel_" << mkd.kernelName << "_size;\n";
                    }

                    // Find max DDR args across all kernels for __aie_launch signature
                    unsigned maxDdrArgs = 0;
                    for (auto &mkd : parsedMeshKernels) {
                        if (mkd.numHostDdrArgs > maxDdrArgs)
                            maxDdrArgs = mkd.numHostDdrArgs;
                    }

                    // Helper: emit __Runtime_sync_for_dev calls for ALL buffers before DMA.
                    // On ARM (baremetal), SyncForDev flushes+invalidates cache lines, which is
                    // needed for outputs too: dirty cache lines (e.g. zeroed output buffer) must
                    // be flushed and invalidated BEFORE DMA writes results to DDR, otherwise a
                    // post-DMA invalidate (clean+invalidate) would flush stale zeros over the
                    // DMA results. No post-launch sync is needed.
                    auto emitSyncCalls = [](llvm::raw_fd_ostream &os, const MeshKernelDesc &mkd,
                                            const std::string &indent, bool beforeLaunch) {
                        if (!beforeLaunch)
                            return; // no post-launch sync needed on ARM
                        unsigned limit = mkd.numHostDdrArgs;
                        if (limit > mkd.tensors.size())
                            limit = mkd.tensors.size();
                        for (unsigned i = 0; i < limit; ++i) {
                            os << indent << "__Runtime_sync_for_dev(dev, _t" << i << ", _s" << i << ");\n";
                        }
                    };

                    // Emit unified __aie_launch with strcmp dispatch (aieMesh overload)
                    if (maxDdrArgs > 0) {
                        stream << "inline void __aie_launch(const char* kernel, aieMesh mesh";
                        for (unsigned i = 0; i < maxDdrArgs; ++i)
                            stream << ", void* _t" << i << ", size_t _s" << i;
                        stream << ", ...) {\n";
                        stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                        for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                            auto &mkd = parsedMeshKernels[ki];
                            std::string hostFunc = "host_canonicalized_" + mkd.kernelName;
                            stream << "    " << (ki == 0 ? "if" : "} else if") << " (strcmp(kernel, \""
                                   << mkd.kernelName << "\") == 0) {\n";
                            stream << "        __Runtime_set_kernel_elf(_binary_kernel_" << mkd.kernelName
                                   << "_start);\n";
                            // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                            emitSyncCalls(stream, mkd, "        ", /*beforeLaunch=*/true);
                            stream << "        " << hostFunc << "(dev";
                            for (unsigned i = 0; i < mkd.numHostDdrArgs; ++i)
                                stream << ", _t" << i;
                            stream << ");\n";
                            // No post-launch sync needed (cache lines already invalidated)
                            emitSyncCalls(stream, mkd, "        ", /*beforeLaunch=*/false);
                        }
                        stream << "    }\n";
                        stream << "}\n";

                        // aieDim overload (backward compat)
                        stream << "inline void __aie_launch(const char* kernel, aieDim mesh";
                        for (unsigned i = 0; i < maxDdrArgs; ++i)
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
                        // aieDim only supports single kernel (backward compat) — dispatch first kernel
                        if (!parsedMeshKernels.empty()) {
                            auto &mkd0 = parsedMeshKernels[0];
                            std::string hostFunc = "host_canonicalized_" + mkd0.kernelName;
                            stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << mkd0.kernelName << "_start);\n";
                            // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                            emitSyncCalls(stream, mkd0, "    ", /*beforeLaunch=*/true);
                            stream << "    " << hostFunc << "(dev";
                            for (unsigned i = 0; i < mkd0.numHostDdrArgs; ++i)
                                stream << ", _t" << i;
                            stream << ");\n";
                            // No post-launch sync needed (cache lines already invalidated)
                            emitSyncCalls(stream, mkd0, "    ", /*beforeLaunch=*/false);
                        }
                        stream << "    __Runtime_explicit_teardown(dev);\n";
                        stream << "}\n";
                    } else {
                        // No DDR args variant (template)
                        stream << "template<typename... Args>\n";
                        stream << "inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {\n";
                        stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                        for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                            auto &mkd = parsedMeshKernels[ki];
                            std::string hostFunc = "host_canonicalized_" + mkd.kernelName;
                            stream << "    " << (ki == 0 ? "if" : "} else if") << " (strcmp(kernel, \""
                                   << mkd.kernelName << "\") == 0) {\n";
                            stream << "        __Runtime_set_kernel_elf(_binary_kernel_" << mkd.kernelName
                                   << "_start);\n";
                            stream << "        " << hostFunc << "(dev);\n";
                        }
                        stream << "    }\n";
                        stream << "}\n";
                    }

                    // Append user's rewritten source
                    stream << userRewrittenSource << "\n";
                    stream.close();
                    llvm::outs() << "[TilingLinalg] Multi-kernel __aie_launch + user source appended to host.cc\n";
                }
            } else {
                // ---- Single-kernel path (backward compatible) ----
                // Build tensor params — use defaults matching unitest if not parsed
                std::vector<TensorParam> tensors;
                SplitModel splitModel;
                std::string singleKernelBody = userKernelBody;
                std::string singleKernelFuncName = userKernelFuncName;
                int64_t effectiveMaxPPBytes = 4096;

                if (!parsedMeshKernels.empty()) {
                    // Use data from the single MeshKernelDesc
                    auto &mkd = parsedMeshKernels[0];
                    tensors = mkd.tensors;
                    splitModel = mkd.splitModel;
                    singleKernelBody = mkd.kernelBody;
                    singleKernelFuncName = mkd.kernelFuncName;
                    effectiveMaxPPBytes = mkd.maxPPBytes;
                } else if (parsedTensors.empty()) {
                    tensors.push_back({{16, 16}, 8, true});
                    splitModel = SplitModel::gemm();
                } else {
                    for (auto &pt : parsedTensors) {
                        DmaAddressing shimDma;
                        if (!pt.shimDma.empty()) {
                            for (int i = 0; i < pt.shimDma.num_dims; ++i)
                                shimDma.dims.push_back({pt.shimDma.dims[i].stride, pt.shimDma.dims[i].wrap});
                            shimDma.iter_step = pt.shimDma.iter_step;
                            shimDma.iter_wrap = pt.shimDma.iter_wrap;
                        }
                        tensors.push_back({pt.shape, pt.elementBitWidth, pt.isInput, shimDma});
                    }
                    splitModel = SplitModel::gemm();
                }
                if (splitModel.tensorSplits.empty())
                    splitModel = SplitModel::gemm();

                // Use default mesh if not parsed
                int rows = tilingMeshRows > 0 ? tilingMeshRows : 2;
                int cols = tilingMeshCols > 0 ? tilingMeshCols : 2;

                // Full-connect auto flag (aie::GlobalPolicy <kernel>_policy). Resolved
                // here (before buildRoutingIR) so it can gate both the emitted GEMM
                // #routing.tiling and the M×N tiling module attrs.
                bool singleFullConnectAuto = true;
                {
                    auto fcaItPre = kernelFullConnectAuto.find(singleKernelFuncName);
                    if (fcaItPre != kernelFullConnectAuto.end())
                        singleFullConnectAuto = fcaItPre->second;
                }

                // Emit per-dim GEMM #routing.tiling for the even-mesh path so the IR
                // op carries the on-core per-round slice (consumed by
                // get_arg_per_round_size_in_dim + Phase-D schedule). Only when
                // fullconnect_auto=1 (conv/halo path keeps its own TilingAttr).
                if (singleFullConnectAuto)
                    populateGemmTiling(splitModel, tensors, derivedTilingParams, rows, cols);

                // Build routing IR
                auto module = TilingLinalgPipeline::buildRoutingIR(ctx, rows, cols, tensors, splitModel,
                                                                   parsedPartition, aieGenStr);

                {
                    auto fcaIt = kernelFullConnectAuto.find(singleKernelFuncName);
                    if (fcaIt != kernelFullConnectAuto.end())
                        singleFullConnectAuto = fcaIt->second;
                }

                // Set K-round module attributes for downstream DMA passes
                if (derivedTilingParams.valid && derivedTilingParams.kRounds > 1) {
                    mlir::OpBuilder attrBuilder(&ctx);
                    // M×N cartesian repeat / spatial sub-tiling attrs are only meaningful
                    // when fullconnect_auto is enabled. Dropping them for fullconnect_auto=0
                    // lets the downstream passes fall through to the pure-K-round path.
                    // The K-triple (effective_k/k_rounds/full_k) is sourced from the
                    // #routing.tiling op (populateGemmTiling) when fullconnect_auto=1, so it
                    // is only emitted as a module attr on the fullconnect_auto=0 else-branch
                    // (no tiling op exists there).
                    if (singleFullConnectAuto) {
                        // tile_m/tile_n, tile_rows/tile_cols, the M/N spatial rounds, and the
                        // K-triple are all sourced from the #routing.tiling op on partitiontensor
                        // (readGemmTilingScalars) by every downstream consumer, so none are
                        // emitted as flat module attrs here.
                        llvm::outs() << "[TilingLinalg] fullconnect_auto=1: tiling scalars sourced from "
                                        "#routing.tiling op (effective_k="
                                     << derivedTilingParams.effectiveK << " k_rounds=" << derivedTilingParams.kRounds
                                     << " full_k=" << derivedTilingParams.kDim
                                     << " tile_rows=" << derivedTilingParams.tileRows
                                     << " m_rounds=" << derivedTilingParams.spatialMRounds
                                     << " tile_cols=" << derivedTilingParams.tileCols
                                     << " n_rounds=" << derivedTilingParams.spatialNRounds << ")\n";
                    } else {
                        // fullconnect_auto=0 pure-K-round path: no #routing.tiling op is
                        // emitted here, so the K-triple must be carried as module attrs.
                        module->setAttr("routing.effective_k",
                                        attrBuilder.getI64IntegerAttr(derivedTilingParams.effectiveK));
                        module->setAttr("routing.k_rounds", attrBuilder.getI64IntegerAttr(derivedTilingParams.kRounds));
                        module->setAttr("routing.full_k", attrBuilder.getI64IntegerAttr(derivedTilingParams.kDim));
                        llvm::outs() << "[TilingLinalg] Set K-round module attrs (fullconnect_auto=0, "
                                        "tile_*/m_rounds/n_rounds dropped): effective_k="
                                     << derivedTilingParams.effectiveK << " k_rounds=" << derivedTilingParams.kRounds
                                     << " full_k=" << derivedTilingParams.kDim << "\n";
                    }
                }

                // Spatial-halo conv: carry the contiguous IFM slab size so the kernel
                // pass allocates the input window to match the kernel body's buf_sz
                // (= halo_slice * raw_wc). Set unconditionally (kRounds may be 1).
                if (derivedTilingParams.valid && derivedTilingParams.convHaloBufSize > 0) {
                    mlir::OpBuilder haloAttrBuilder(&ctx);
                    module->setAttr("routing.spatial_halo_buf_size",
                                    haloAttrBuilder.getI64IntegerAttr(derivedTilingParams.convHaloBufSize));
                    llvm::outs() << "[TilingLinalg] Set spatial_halo_buf_size=" << derivedTilingParams.convHaloBufSize
                                 << "\n";
                    // Spatial-halo conv has kRounds==1, so the K-round attr block above is
                    // skipped and routing.tile_m is never set. Emit ONLY the M-dimension
                    // tiling attrs here so classifyTiling returns Multiple(mRounds) and the
                    // host scf.for round loop streams one slab per round. Do NOT emit
                    // tile_n/tile_cols: that would trigger nMode=Multiple and spurious
                    // filter-B N-round iteration (the filter is re-sent unchanged per round).
                    // Gated on fullconnect_auto: a halo kernel that set fullconnect_auto=0
                    // opts out of the M-round repeat entirely.
                    if (singleFullConnectAuto && derivedTilingParams.spatialMRounds > 1 &&
                        derivedTilingParams.tileM > 0 && derivedTilingParams.tileRows > derivedTilingParams.tileM) {
                        module->setAttr("routing.tile_m", haloAttrBuilder.getI64IntegerAttr(derivedTilingParams.tileM));
                        module->setAttr("routing.tile_rows",
                                        haloAttrBuilder.getI64IntegerAttr(derivedTilingParams.tileRows));
                        module->setAttr("routing.m_rounds",
                                        haloAttrBuilder.getI64IntegerAttr(derivedTilingParams.spatialMRounds));
                        llvm::outs() << "[TilingLinalg] Set spatial-halo M-round attrs: tile_m="
                                     << derivedTilingParams.tileM << " tile_rows=" << derivedTilingParams.tileRows
                                     << " m_rounds=" << derivedTilingParams.spatialMRounds << "\n";
                    }
                    // OUTPUT per-slab division count. The conv2d_spatial kernel ALWAYS
                    // emits one [oh_per_row*ow_t, tile_n] output slab per on-core round
                    // (m_rounds = spatialMRounds*spatialNRounds), independent of the
                    // fullconnect_auto DMA-repeat policy (which only governs how the
                    // INPUT filter/A ports are re-sent). The core MM2S output BD must
                    // therefore be sized to ONE slab (perCoreElements / out_rounds) and
                    // iterate out_rounds times. Emit this divisor UNCONDITIONALLY so the
                    // schedule passes divide the output even when fullconnect_auto=0
                    // dropped routing.tile_m/m_rounds above.
                    {
                        int64_t outRounds = derivedTilingParams.spatialMRounds * derivedTilingParams.spatialNRounds;
                        if (outRounds > 1) {
                            module->setAttr("routing.spatial_out_rounds", haloAttrBuilder.getI64IntegerAttr(outRounds));
                            llvm::outs() << "[TilingLinalg] Set spatial_out_rounds=" << outRounds
                                         << " (spatialMRounds=" << derivedTilingParams.spatialMRounds
                                         << " * spatialNRounds=" << derivedTilingParams.spatialNRounds << ")\n";
                        }
                    }
                }

                // Full-connect auto flag (aie::GlobalPolicy <kernel>_policy).
                // Set UNCONDITIONALLY (applies even when kRounds==1). 1 = default
                // M×N cartesian DMA repeat; 0 = A/B each sent once.
                {
                    mlir::OpBuilder fcAttrBuilder(&ctx);
                    module->setAttr("routing.fullconnect_auto",
                                    fcAttrBuilder.getI64IntegerAttr(singleFullConnectAuto ? 1 : 0));
                    llvm::outs() << "[TilingLinalg] Set fullconnect_auto=" << (singleFullConnectAuto ? 1 : 0)
                                 << " for kernel " << singleKernelFuncName << "\n";
                }

                // Replace aie::get_*() calls in kernel body with computed integer literals
                if (derivedTilingParams.valid && !singleKernelBody.empty()) {
                    llvm::outs() << "[TilingLinalg] Replacing aie::get_*() calls in kernel body\n";

                    std::unordered_map<std::string, size_t> paramToPort;
                    for (size_t i = 0; i < parsedTensors.size(); ++i) {
                        paramToPort[parsedTensors[i].varName] = i;
                    }

                    for (const auto &funcName :
                         {"aie::get_num_rounds", "aie::get_buffer_size", "aie::get_spatial_multiple_rounds"}) {
                        std::string prefix = std::string(funcName) + "(";
                        size_t pos = 0;
                        while ((pos = singleKernelBody.find(prefix, pos)) != std::string::npos) {
                            size_t argStart = pos + prefix.size();
                            size_t argEnd = singleKernelBody.find(")", argStart);
                            if (argEnd == std::string::npos)
                                break;
                            std::string argName = singleKernelBody.substr(argStart, argEnd - argStart);
                            while (!argName.empty() && argName.front() == ' ')
                                argName.erase(0, 1);
                            while (!argName.empty() && argName.back() == ' ')
                                argName.pop_back();
                            auto it = paramToPort.find(argName);
                            if (it != paramToPort.end() && it->second < derivedTilingParams.portParams.size()) {
                                int64_t val;
                                if (std::string(funcName) == "aie::get_num_rounds")
                                    val = derivedTilingParams.portParams[it->second].numRounds;
                                else if (std::string(funcName) == "aie::get_buffer_size")
                                    val = derivedTilingParams.portParams[it->second].bufferSize;
                                else
                                    val = derivedTilingParams.portParams[it->second].spatialRounds;
                                std::string replacement = std::to_string(val);
                                singleKernelBody.replace(pos, argEnd + 1 - pos, replacement);
                                pos += replacement.size();
                            } else {
                                pos = argEnd + 1;
                            }
                        }
                    }

                    auto replaceSimpleCall = [&](const std::string &call, int64_t val) {
                        std::string replacement = std::to_string(val);
                        size_t pos = 0;
                        while ((pos = singleKernelBody.find(call, pos)) != std::string::npos) {
                            singleKernelBody.replace(pos, call.size(), replacement);
                            pos += replacement.size();
                        }
                    };
                    replaceSimpleCall("aie::get_tile_rows()", derivedTilingParams.tileM > 0
                                                                  ? derivedTilingParams.tileM
                                                                  : derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_tile_cols()", derivedTilingParams.tileN > 0
                                                                  ? derivedTilingParams.tileN
                                                                  : derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_data_row()", derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_data_col()", derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_k_dim()", derivedTilingParams.kDim);
                    // Two-level tiling query functions
                    replaceSimpleCall("aie::get_tile_m()", derivedTilingParams.tileM > 0
                                                               ? derivedTilingParams.tileM
                                                               : derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_tile_n()", derivedTilingParams.tileN > 0
                                                               ? derivedTilingParams.tileN
                                                               : derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_effective_k()", derivedTilingParams.effectiveK > 0
                                                                    ? derivedTilingParams.effectiveK
                                                                    : derivedTilingParams.kDim);
                    replaceSimpleCall("aie::get_k_rounds()", derivedTilingParams.kRounds);
                    replaceSimpleCall("aie::get_spatial_m_rounds()", derivedTilingParams.spatialMRounds);
                    replaceSimpleCall("aie::get_spatial_n_rounds()", derivedTilingParams.spatialNRounds);
                    // Conv (spatial-halo) accessors
                    replaceSimpleCall("aie::get_kernel_h()", derivedTilingParams.convKernelH);
                    replaceSimpleCall("aie::get_kernel_w()", derivedTilingParams.convKernelW);
                    replaceSimpleCall("aie::get_input_c()", derivedTilingParams.convInputC);
                    replaceSimpleCall("aie::get_stride()", derivedTilingParams.convStride);
                    replaceSimpleCall("aie::get_ow()", derivedTilingParams.convOW);
                    replaceSimpleCall("aie::get_oh_per_row()", derivedTilingParams.convOHPerRow);
                    replaceSimpleCall("aie::get_halo_slice()", derivedTilingParams.convHaloSlice);
                }

                // Prepend user macros to kernel body for multi-tile path
                std::string kernelBodyWithMacros = singleKernelBody;
                if (!userMacroDefines.empty() && !singleKernelBody.empty()) {
                    std::string macroBlock = "// User macro definitions from source file\n";
                    for (const auto &macro : userMacroDefines) {
                        macroBlock += macro + "\n";
                    }
                    macroBlock += "\n";
                    kernelBodyWithMacros = macroBlock + singleKernelBody;
                }

                // Port var names (indexed by tensor order) for IR-sourced builtin
                // resolution (get_arg_per_round_size_in_dim) inside runPipeline.
                std::vector<std::string> singlePortVarNames;
                singlePortVarNames.reserve(parsedTensors.size());
                for (const auto &pt : parsedTensors)
                    singlePortVarNames.push_back(pt.varName);

                // Run pipeline (single kernel — no suffix, no append mode)
                if (!TilingLinalgPipeline::runPipeline(
                        ctx, module, outputDir, kernelBodyWithMacros, singleKernelFuncName, parsedDebugLevel,
                        userRewrittenSource, tensors, effectiveMaxPPBytes, aieGenStr, /*hostFuncSuffix=*/"",
                        /*appendMode=*/false, /*numHostDdrArgs=*/nullptr, singlePortVarNames)) {
                    llvm::errs() << "[TilingLinalg] Pipeline FAILED.\n";
                    std::exit(1);
                }
            }

            llvm::outs() << "[TilingLinalg] Pipeline completed. Output in: " << outputDir << "\n";
            // Copy user #include "..." headers to outputDir so host.cc is self-contained.
            if (!userSourceDir.empty() && !userRewrittenSource.empty()) {
                std::regex includeRegex(R"(#\s*include\s*\"([^\"]+)\")");
                auto begin = std::sregex_iterator(userRewrittenSource.begin(), userRewrittenSource.end(), includeRegex);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) {
                    std::string headerName = (*it)[1].str();
                    // Skip runtime headers already on the compiler include path
                    if (headerName == "aie_runtime.h" || headerName == "aie_runtime_debug.h" ||
                        headerName == "xaiengine.h")
                        continue;
                    std::string srcPath = userSourceDir + "/" + headerName;
                    std::string dstPath = outputDir + "/" + headerName;
                    // Only copy if the file exists in the user source directory
                    std::ifstream src(srcPath, std::ios::binary);
                    if (src.good()) {
                        std::ofstream dst(dstPath, std::ios::binary);
                        dst << src.rdbuf();
                        llvm::outs() << "[TilingLinalg] Copied user header: " << headerName << " -> " << dstPath
                                     << "\n";
                    }
                }
            }
        } else {
            // ---- EXISTING PATH: single-tile ----
            std::error_code error_code;
            llvm::outs() << "Exporting File: " << std::string(AOUT) + "./host.cc" << "\n";
            llvm::raw_fd_ostream outFile(std::string(AOUT) + "./host.cc", error_code, llvm::sys::fs::OF_None);
            if (error_code) {
                llvm::errs() << "Error opening file: " << error_code.message() << "\n";
                return; // Exit early if there's an error
            }
            for (auto &kv : globalKernelFuncs) {
                const std::string &kname = kv.first;
                std::string dmOffsetsPath = std::string(AOUT) + "./kernelcfg/" + kname + "/dm_offsets.h";
                if (llvm::sys::fs::exists(dmOffsetsPath)) {
                    outFile << "#include \"kernelcfg/" << kname << "/dm_offsets.h\"\n";
                    llvm::outs() << "[aiehlc] Injected dm_offsets.h for " << kname << " into host.cc\n";
                }
            }
            // Emit strong g_runtime_debug_level override if user set #pragma aie_debug_level
            if (parsedDebugLevel >= 0) {
                outFile << "// Override runtime debug level (from #pragma aie_debug_level)\n";
                outFile << "int g_runtime_debug_level = " << parsedDebugLevel << ";\n\n";
            }
            // fd.write(RewriteBuf->data(), RewriteBuf->size());
            RewriteBuf->write(outFile);
            /*
            auto& sc = Rewrite.getSourceMgr();
            auto mid = sc.getMainFileID();
            llvm::StringRef sourceCode = sc.getBufferData(mid);
            llvm::outs() << "Original Source Code:\n" << sourceCode << "\n";
            */
            const std::string fname = "aie.mlir";
            auto mlirstr = aiefrontend.dumpir();
            std::ofstream ofs(fname);
            if (ofs.is_open()) {
                ofs << mlirstr;
                ofs.close();
            }
            aiefrontend.RunPass(fname);
        }
        //  std::cout << "----------------end-----------------" << std::endl;
  	}
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef file) override {

				//disable the error information
			    // auto& dg = CI.getDiagnostics();
				// auto& Context = CI.getASTContext();
				// unsigned IgnoredErrorID = Context.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
				// 		"This error is ignored");
				// dg.setClient(new IgnoringDiagConsumer(), /*ShouldOwn=*/true);
				// CI.createDiagnostics(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/false);

				TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());			
        return std::make_unique<MyASTConsumer>(&TheRewriter,&CI.getASTContext(), &aiefrontend);
    }
private:
		std::string code;
		AieFrontEnd aiefrontend;
		Rewriter TheRewriter;
		std::unique_ptr<clang::FileManager> FileMgr;
		//FileManager FileMgr;
};
extern void dumpmlir();

void trycreatefolder(std::string dirPath) {
	///*
	std::string command = "mkdir -p " + dirPath; 
	int result = system(command.c_str());           // Run the command

	if (result == 0) {
		// std::cout << "Directory created." << std::endl;
	} else {
		std::cerr << "Failed to create directory." << std::endl;
	}
}

static llvm::cl::opt<std::string> IncludePath(
			"I", llvm::cl::desc("Specify include path (-I)"), llvm::cl::value_desc("path"));

llvm::cl::OptionCategory MyToolCategory("aiehlc options");
static llvm::cl::opt<bool> LLVMAie(
			"use-llvm-aie", llvm::cl::desc("Enable use of LLVM AIE"), llvm::cl::init(false), llvm::cl::cat(MyToolCategory));

int main(int argc, const char **argv) {

	trycreatefolder(AOUT);
	// std::cout << "main--" << IncludePath.getValue() << std::endl;
	// /*
    //CommonOptionsParser exp(argc, argv, MyToolCategory);
    ///*
	if (argc > 2) {
		if (strcmp(argv[1], "pass") == 0) {
			const char * mlirf = argv[2];
			AieFrontEnd af;
			std::string smlir(mlirf);
			af.Parser(smlir);
			return 0;
		}
	}
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory, llvm::cl::OneOrMore);
    if (!ExpectedParser) {
		llvm::errs() << "Error: " << llvm::toString(ExpectedParser.takeError()) << "\n";
        // Handle error
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
	use_llvm_aie = LLVMAie;
    //*/
    /*
    auto& OriginalCompilations = OptionsParser.getCompilations();
    std::vector<CompileCommand> newCommands;
    for (const auto& SourceFile : OptionsParser.getSourcePathList()) {
        auto CompileCommandsForFile = OriginalCompilations.getCompileCommands(SourceFile);
	for (auto &Command : CompileCommandsForFile) {
            Command.CommandLine.push_back("-x");
            Command.CommandLine.push_back("c++");
	    newCommands.push_back(Command);
	}

    }
    /*
    FixedCompilationDatabase ( "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/", newCommands);
    auto NewCompilations = std::make_shared<FixedCompilationDatabase>(newCommands);
		        //    "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/", newCommands);
    //ClangTool Tool(*NewCompilations,
    //               OptionsParser.getSourcePathList());
    
    std::string Directory = "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/";
    const char* CompileArgs[]= {"-std=c++11", " -x", "c++"};
    int n = 3;
    auto Compilations = FixedCompilationDatabase::loadFromCommandLine(n, CompileArgs, Directory);
    */
		//dumpmlir();
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

	if (!IncludePath.empty()) {
		// Add the -I include path argument before the source file arguments
		Tool.appendArgumentsAdjuster(
				getInsertArgumentAdjuster(("-I" + IncludePath).c_str(), ArgumentInsertPosition::BEGIN));
	}

    // Force C++20 for struct NTTP support in aie::port<T, SpatialPolicy>
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++20", ArgumentInsertPosition::BEGIN));
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
