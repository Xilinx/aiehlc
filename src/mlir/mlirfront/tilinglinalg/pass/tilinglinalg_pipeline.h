/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "passcoretraceinsert/passcoretraceinsert.h"
#include <string>
#include <vector>

/// Generic multi-dim DMA descriptor for shim tile.
/// Mirrors AIEML BD hardware: up to 4 dims + iteration.
struct DmaAddressing {
    std::vector<std::pair<int, int>> dims; // {stride, wrap} per dim
    int iter_step = 0;
    int iter_wrap = 0;
    std::vector<int64_t> ddrShape; // actual DDR buffer shape (e.g. [H, W, C] for im2col)
    int tile_m_alignment = 0;      // tile_m must be multiple of this (e.g. OW for im2col)
    // Spatial-halo distribution (mode 1): the A-tensor is sliced into overlapping
    // contiguous row-blocks. Each tile-row owns `haloSlice` input rows, advancing by
    // `haloStep` rows between tile-rows (so overlap = haloSlice - haloStep). When
    // mode == 1 the shim BD stays flat (no multi-dim strides); the overlap is realized
    // purely through per-tile DDR base offsets (offset_i = i * haloStep * raw_wc).
    int mode = 0;      // 0 = flat / im2col-by-dims, 1 = spatial_halo
    int haloSlice = 0; // input rows owned by each tile-row (e.g. 61)
    int haloStep = 0;  // row stride between consecutive tile-rows (e.g. 56)
    int splitDim = 0;  // tensor dimension that carries the halo split (usually 0)
    // 2D width-split (spatial-halo on-core WIDTH rounds). When wRounds > 1 the conv
    // WIDTH is chunked into on-core temporal rounds: each chunk is a NARROW slab
    // [haloSlice, wSlice*C] cut from the PADDED DDR buffer with row pitch rowPitch.
    // The kernel iterates H_chunks * W_chunks rounds; the shim BD becomes 2D and the
    // host round loop steps a 2D (hc,wc) base offset. Zero / 1 keeps the legacy
    // height-only flat path unchanged.
    int wSlice = 0;   // per-chunk input cols (e.g. TILE_W = 61)
    int wStep = 0;    // width halo step between chunks (e.g. TILE_STRIDE_W = 56)
    int wRounds = 0;  // number of width chunks (e.g. 4); 0/1 = no width split
    int rowPitch = 0; // PADDED input row pitch in elements (INPUT_W_PAD * C = 920)
    int owT = 0;      // per-chunk output cols (e.g. OW_T = 28)
    // 2D L2 (nested temporal) row-split on the SAME split axis as the L1 spatial
    // halo. Each HW tile owns `haloSlice` rows (L1); when l2Rounds > 1 that slice
    // is further chunked into l2Rounds on-core temporal ROUNDS, each `l2Slice`
    // rows advancing by `l2Step` rows (L2 overlap = l2Slice - l2Step). Realized
    // via the BD iteration dim (iter_step = l2Step*rowPitch, iter_wrap = l2Rounds)
    // and a matching kernel round multiplier. Zero / 1 keeps the legacy
    // single-level (L1-only) path unchanged.
    int l2Slice = 0;  // input rows per on-core round (e.g. 19)
    int l2Step = 0;   // row stride between L2 rounds (e.g. 14)
    int l2Rounds = 0; // number of L2 on-core rounds (e.g. 4); 0/1 = no L2 split
    // K-contraction accumulate split (independent of the H/row L2 halo above).
    // The K dim (im2col KH*KW*C) is chunked into kRounds on-core rounds of kSlice
    // advancing by kStep; partial products accumulate. Zero/1 = no K-accum split.
    int kSlice = 0;  // K elements per accumulate round (e.g. 19)
    int kStep = 0;   // K stride between rounds (e.g. 14)
    int kRounds = 1; // number of accumulate rounds; 0/1 = none
    bool empty() const { return dims.empty() && mode == 0; }
};

// The blocks a kernel-only kernel takes on one port, from its traced
// aie::load_buffer / store_buffer calls (-> #routing.kernel_block).
struct KernelBlock {
    std::vector<int64_t> tensorShape;  // the tensor as the KERNEL declares it (the GEMM
                                       // path relabels B as [K, N]; the bytes are [N, K])
    std::vector<int64_t> shape;        // block extent per dim, elements; empty = none
    std::vector<int64_t> stepI, stepJ; // offset per outer / inner round, per dim
    int64_t roundsI = 1, roundsJ = 1;  // blocks per core = roundsI * roundsJ
    bool resident = false;             // {.resident = true}
    int64_t tileR = 0, tileC = 0;      // {.tile = {tileR, tileC}}, 0 = none
    bool empty() const { return shape.empty(); }
};

struct TensorParam {
    std::vector<int64_t> shape;  // e.g. {16, 16}
    int elementBitWidth;         // e.g. 32 for int32_t, 8 for int8_t
    bool isInput;                // true = input, false = output
    DmaAddressing shimDma;       // optional: non-flat DMA addressing for shim
    KernelBlock kernelBlock;     // optional: the kernel's traced blocks (kernel-only sources)
};

/// Per-tensor split descriptor — maps to partitiontensor op attributes.
/// Spatial tags control mesh partitioning (hwAxisOwner, replicateOn), not tensor splitting.
struct TensorSplitDesc {
    int splitDim;            // tensor dimension to split (default: 0). Irrelevant when splitnum=1.
    std::string hwAxisOwner; // "row" | "col" | "" — which mesh axis owns this tensor's partition
    std::string replicateOn; // "row" | "col" | "" — which mesh axis to replicate/broadcast on
    // Spatial policy fields
    std::string pattern; // "broadcast" | "scatter" | "multicast" | "gather"
    std::string flow;    // "ltor" | "rtol" | "default"
    int pingPong = 2;    // ping-pong buffer depth (2, 4, 8)
    int maxBufferBytes = 4096; // max per-buffer size (PP_MAX_BYTES equivalent)
    std::string layoutTransform; // "dma_shuffle" | "core_shuffle" | "" (none)
    // Spatial-halo fields (mirrored from TensorParam.shimDma when mode == 1).
    // When haloMode == 1 the partition slice is overlapping: each tile-row owns
    // `haloSlice` rows along splitDim, advancing by `haloStep` rows per tile-row.
    int haloMode = 0;  // 0 = even split, 1 = overlapping halo split
    int haloSlice = 0; // rows per tile-row (e.g. 61)
    int haloStep = 0;  // row stride between tile-rows (e.g. 56)
    // Nested L2 (on-core temporal) row-split inside each L1 tile slice, on the
    // SAME split axis. When haloL2Rounds > 1 each tile's `haloSlice` rows are
    // chunked into haloL2Rounds rounds of `haloL2Slice` rows advancing by
    // `haloL2Step` (L2 overlap = haloL2Slice - haloL2Step). Mirrored onto the
    // partitiontensor op as l2_slice/l2_step/l2_rounds attrs. Zero / 1 = no L2.
    int haloL2Slice = 0;  // rows per on-core round (e.g. 19)
    int haloL2Step = 0;   // row stride between L2 rounds (e.g. 14)
    int haloL2Rounds = 0; // number of L2 on-core rounds (e.g. 4); 0/1 = no L2
    // K-contraction accumulate split (mirrored from DmaAddressing.kSlice/kStep/
    // kRounds). Independent of the H/row L2 halo above; the K dim is chunked into
    // kAccumRounds on-core rounds of kAccumSlice advancing by kAccumStep.
    int kAccumSlice = 0;
    int kAccumStep = 0;
    int kAccumRounds = 1;
    // Second mesh-axis (group2) split: within each mesh ROW, the output is
    // further split across mesh COLS along a *different* tensor dim than the
    // group1/row split (e.g. channel d3). group2Dim is 1-based; 0/-1 = no
    // second-axis split. Only the OUTPUT (gather) tensor carries this; inputs
    // unchanged. Consumed by the shim S2MM reassembly BD to interleave on the
    // channel dim instead of width. (group1Dim is implicit = splitDim +
    // hwAxisOwner="row"; stored for validation only.)
    int group1Dim = 0;   // 1-based tensor dim split across mesh rows (validation)
    int group2Dim = 0;   // 1-based tensor dim split across mesh cols (e.g. 3 = channel)
    int group2Slice = 0; // elements per col-tile along group2Dim (e.g. 16)
    int group2Full = 0;  // full extent of group2Dim (e.g. 64)
    // Output row-dim (group1 = d1) on-core slice_tiling + d2 (W) chunk. Carried into
    // the output tensor's #routing.tiling row-dim nested levels (mirrors haloL2* for
    // input). group1 = mesh-row split dim (e.g. output H); its slice_tiling records
    // the on-core H rounds. d2 = a non-mesh on-core chunk dim (e.g. output W). These
    // are *logical* descriptors carried for traceability; consuming them to reshape
    // the output reassembly BD is separate downstream work.
    int group1Full = 0;     // d1.fullsize  (output H extent, e.g. 112)
    int group1Rounds = 0;   // d1.tile_round (mesh-row rounds, e.g. 4)
    int group1L2Slice = 0;  // d1.slice_tiling.tile_size (logical H rows/round, e.g. 7)
    int group1L2Step = 0;   // d1.slice_tiling.stride (e.g. 7)
    int group1L2Rounds = 0; // d1.slice_tiling.rounds (e.g. 4); 0/1 = no on-core H split
    int d2Slice = 0;        // d2.tile_size (W chunk cols, e.g. 28)
    int d2Step = 0;         // d2.stride (e.g. 28)
    int d2Rounds = 0;       // d2.tile_round (e.g. 4); 0/1 = no W chunk
    int d2Full = 0;         // d2.fullsize (W extent = flat row pitch, e.g. 112)
    // GEMM (fullconnect_auto=1) per-dim on-core tiling. One GemmDimTile per tensor
    // dim, mirroring the isHalo #routing.tiling builder but for the even mesh/GEMM
    // path. Each dim describes an outer level (mesh split: slice=meshSlice,
    // rounds=meshRounds) plus an optional nested on-core level (slice=coreSlice,
    // rounds=coreRounds; coreRounds<=1 => leaf, no slice_tiling). Empty vector =>
    // no GEMM TilingAttr emitted (keeps conv/halo/group2 paths and non-GEMM
    // kernels unchanged). The on-core per-round slice consumed by
    // get_arg_per_round_size_in_dim(dim, port) is coreSlice (or meshSlice when no
    // nested level).
    struct GemmDimTile {
        int64_t base = 0;       // full tensor extent this dim partitions
        int64_t meshSlice = 0;  // outer slice (mesh split); == base when not mesh-split
        int64_t meshStep = 0;   // outer stride between mesh steps
        int64_t meshRounds = 1; // outer rounds (mesh splitnum); 1 when not mesh-split
        int64_t coreSlice = 0;  // on-core per-round slice (0/absent => no nested level)
        int64_t coreStep = 0;   // on-core stride between rounds
        int64_t coreRounds = 1; // on-core rounds (0/1 => leaf, no slice_tiling)
    };
    std::vector<GemmDimTile> gemmTiling;
};

/// Operation-level split model: one TensorSplitDesc per tensor.
struct SplitModel {
    std::vector<TensorSplitDesc> tensorSplits;

    /// Factory: default GEMM
    /// A: tensor split dim=0, mesh partition by row, broadcast along cols
    /// B: tensor split dim=0, mesh partition by col, broadcast along rows
    /// C: tensor split dim=0, mesh partition by row, gather along cols
    static SplitModel gemm() {
        return {{{0, "row", "col", "broadcast", "default", 2, 4096},
                 {0, "col", "row", "broadcast", "default", 2, 4096},
                 {0, "row", "col", "gather", "ltor", 2, 4096}}};
    }

    /// Construct a TensorSplitDesc from a spatial type tag string.
    /// tag: "row_broadcast_in", "col_broadcast_in", etc.
    /// isInput: used for default when tag is empty.
    static TensorSplitDesc fromSpatialTag(const std::string &tag, bool isInput);

    /// Construct a TensorSplitDesc from resolved SpatialPolicy struct fields.
    /// pattern: 0=Broadcast, 1=Scatter, 2=Multicast, 3=Gather
    /// distribution: 0=Row, 1=Col, 2=Grid
    /// mergeOrder: 0=Default, 1=LeftToRight, 2=RightToLeft
    static TensorSplitDesc fromPolicyFields(int pattern, int distribution, int mergeOrder, int pingPong, bool isInput,
                                            int maxBufferBytes = 4096, int layoutTransform = 0);
};

/// Partition descriptor: confines tile allocation to a sub-region of the AIE array.
/// When invalid (all -1), the full mesh is used.
struct PartitionDesc {
    int startCol = -1, endCol = -1; // column range [startCol, endCol]
    int startRow = -1, endRow = -1; // row range [startRow, endRow]
    bool isValid() const { return startCol >= 0; }
};

/// Derived tiling parameters computed from M/K/N and per-port SpatialPolicy.
/// These are resolved after AST extraction and used to replace aie::get_*() calls
/// in the kernel body with integer literals.
struct DerivedTilingParams {
    int64_t tileRows = 0; // M / HW_ROWS
    int64_t tileCols = 0; // N / HW_COLS
    int64_t kDim = 0;     // K

    // Two-level tiling: L2 temporal tiling within each core
    // When non-zero, the kernel works on sub-tiles of size tile_m x tile_n
    // with K streamed in chunks of effective_k.
    int64_t tileM = 0;          // effective sub-tile rows per core (0 = tileRows, no sub-tiling)
    int64_t tileN = 0;          // effective sub-tile cols per core (0 = tileCols, no sub-tiling)
    int64_t effectiveK = 0;     // K chunk size for temporal tiling (0 = kDim, full K)
    int64_t spatialMRounds = 1; // tileRows / tileM — host re-launch rounds along M
    int64_t spatialNRounds = 1; // tileCols / tileN — host re-launch rounds along N
    int64_t kRounds = 1;        // kDim / effectiveK — K-accumulation rounds inside kernel
    bool autoTiled = false;     // true if compiler auto-derived tileM/tileN/effectiveK

    // Conv (spatial-halo) parameters: resolve aie::get_kernel_h/get_kernel_w/
    // get_input_c/get_stride/get_ow/get_oh_per_row/get_halo_slice in the kernel.
    // Zero when the kernel is not a spatial-halo conv.
    int64_t convKernelH = 0;   // KERNEL_H
    int64_t convKernelW = 0;   // KERNEL_W
    int64_t convInputC = 0;    // INPUT_C
    int64_t convStride = 0;    // STRIDE
    int64_t convOW = 0;        // OUTPUT_W
    int64_t convOHPerRow = 0;  // OUTPUT_H / HW_ROWS
    int64_t convHaloSlice = 0; // input rows owned per tile-row
    // Spatial-halo IFM contiguous slab size in elements (halo_slice * raw_wc).
    // The kernel performs on-chip im2col and needs the whole contiguous slab in
    // one window buffer, so the kernel-side window allocation must use this size
    // (bypassing the GEMM flow-view partition size and the maxPingPong clamp).
    // Zero when the kernel is not a spatial-halo conv.
    int64_t convHaloBufSize = 0; // halo_slice * raw_wc (elements per IFM window)

    // 2D width-split (spatial-halo on-core WIDTH rounds). When convWRounds > 1
    // the conv WIDTH is chunked into on-core rounds: each chunk delivers a NARROW
    // slab [halo_slice, convWSlice*C] via a 2D shim BD with the PADDED row pitch,
    // and the kernel iterates H_chunks * W_chunks (= spatialMRounds*convWRounds)
    // rounds. Zero / 1 keeps the legacy height-only flat contiguous path.
    int64_t convWRounds = 0;  // number of width chunks (e.g. 4); 0/1 = no width split
    int64_t convOWT = 0;      // per-chunk output cols (e.g. OW_T = 28)
    int64_t convWSlice = 0;   // per-chunk input cols (e.g. TILE_W = 61)
    int64_t convWStep = 0;    // width halo step between chunks (e.g. TILE_STRIDE_W = 56)
    int64_t convRowPitch = 0; // PADDED input row pitch in elements (INPUT_W_PAD * C)

    // 2D L2 (nested temporal) ROW-split on the same split axis as the L1 spatial
    // halo. When convL2Rounds > 1 the per-tile A-window row slice is chunked into
    // convL2Rounds on-core rounds of convL2Slice rows; the kernel A-port round
    // count is multiplied by convL2Rounds and the per-round A window is
    // convL2Slice * raw_wc. Zero / 1 keeps the single-level (L1-only) path.
    int64_t convL2Rounds = 0; // number of L2 on-core rounds (e.g. 4); 0/1 = none
    int64_t convL2Slice = 0;  // input rows per L2 on-core round (e.g. 19)
    int64_t convL2Step = 0;   // row stride between L2 rounds (e.g. 14)

    // Per-port derived values (indexed by tensor order: 0=A, 1=B, 2=C for GEMM)
    struct PortParams {
        int64_t numRounds = 0;
        int64_t bufferSize = 0;    // elements per round
        int64_t spatialRounds = 1; // get_spatial_multiple_rounds(win): per-port
                                   // spatial sub-tile count (A->M, B->N, C->M*N)
    };
    std::vector<PortParams> portParams;
    bool valid = false;
};

/// Descriptor for a kernel launch on a specific mesh.
/// Used by multi-kernel mode to associate each <<<meshVar>>> launch with
/// its kernel name, mesh dimensions, partition, and tensor parameters.
struct MeshKernelDesc {
    std::string kernelName;  // e.g. "matmul", "relu"
    std::string meshVarName; // e.g. "meshA", "meshB"
    int meshRows = 0, meshCols = 0;
    PartitionDesc partition;
    std::vector<TensorParam> tensors;
    int meshId = 0; // compiler-assigned ID for partition tracking
    // Per-kernel fields for multi-kernel pipeline
    std::string kernelBody;     // raw source text of __global__ function body
    std::string kernelFuncName; // kernel function name from __global__
    SplitModel splitModel;      // per-tensor data distribution strategy
    int64_t maxPPBytes = 4096;  // max ping-pong buffer bytes
    /// Full-connect auto flag (from aie::GlobalPolicy <kernel>_policy).
    /// true (default) = M×N cartesian DMA repeat; false = A/B each sent once.
    bool fullConnectAuto = true;
    /// Number of DDR args on the generated host function (set after pipeline run)
    unsigned numHostDdrArgs = 0;
    /// Per-port variable names (e.g. "win_a", "win_b", "win_c") for aie::get_*() replacement
    std::vector<std::string> portVarNames;
    /// Derived tiling parameters for this kernel (aie::get_*() replacement)
    DerivedTilingParams derivedParams;
};

class TilingLinalgPipeline {
public:
    /// Register all 6 dialect managers + standard dialects on ctx
    static void registerDialects(mlir::MLIRContext &ctx);

    /// Build initial routing IR (parameterized version of ops_testNew)
    /// meshRows/meshCols -> createhwmesh dimensions
    /// tensors -> createscheduletensor + createroutingfuncBySplitModel
    /// splitModel -> per-tensor data distribution strategy
    /// partition -> optional sub-region for tile allocation
    static mlir::ModuleOp buildRoutingIR(mlir::MLIRContext &ctx, int meshRows, int meshCols,
                                         const std::vector<TensorParam> &tensors,
                                         const SplitModel &splitModel = SplitModel::gemm(),
                                         const PartitionDesc &partition = {}, const std::string &aieGen = "Gen2");

    /// Run the full pipeline and emit files to outputDir:
    ///   host.cc, kernel.cc, routing.cc, aieml.bcf, aieml.prx
    /// If userKernelBody is non-empty, it is written as computekernel.cc
    /// instead of auto-generating the compute kernel. The function name
    /// in userKernelBody (userKernelFuncName) is renamed to match the
    /// pipeline's expected compute kernel name.
    /// hostFuncSuffix: when non-empty, the generated host function is named
    ///   host_canonicalized_<suffix> instead of host_canonicalized.
    ///   Used by multi-kernel mode to generate per-kernel host functions.
    /// appendMode: when true, host.cc is opened in append mode (OF_Append) and
    ///   user source / __aie_launch emission is skipped. Used by multi-kernel mode
    ///   so that each kernel appends its host_canonicalized_<name> function.
    /// numHostDdrArgs (out): if non-null, set to the number of DDR pointer args
    ///   on the generated host function (numArgs - 1, excluding XAie_DevInst* dev).
    /// Returns true on success.
    /// portVarNames: per-port variable names (e.g. "win_a"), indexed by tensor
    ///   order. Used to resolve IR-sourced builtins whose value comes from the
    ///   routing.partitiontensor TilingAttr (get_arg_per_round_size_in_dim(dim,
    ///   port)) rather than frontend scalars. Empty => no IR-sourced resolution.
    /// traceTiles: compute tiles (mesh/partition-relative {col,row}) to core-trace,
    ///   from #pragma aie_trace. When non-empty, CoreTraceInsertPass injects
    ///   __Runtime_core_trace_begin/_end into the generated host.cc. Empty => no-op.
    static bool runPipeline(mlir::MLIRContext &ctx, mlir::ModuleOp module, const std::string &outputDir,
                            const std::string &userKernelBody = "", const std::string &userKernelFuncName = "",
                            int runtimeDebugLevel = -1, const std::string &userRewrittenSource = "",
                            const std::vector<TensorParam> &tensors = {}, int64_t maxPingPongBytes = 4096,
                            const std::string &aieGen = "Gen2", const std::string &hostFuncSuffix = "",
                            bool appendMode = false, unsigned *numHostDdrArgs = nullptr,
                            const std::vector<std::string> &portVarNames = {},
                            const std::vector<TraceTileSpec> &traceTiles = {});
};
