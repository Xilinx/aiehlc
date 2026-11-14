/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel block analysis: derive and verify the DMA of a block-load kernel from
 * the kernel itself.
 *
 *   __global__ void k(aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> x,
 *                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> w,
 *                     aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> y) {
 *       ... aie::program_id(0/1), aie::load_buffer(x, {off}, {shape}), aie::store_buffer(...),
 *           aie::release(x), aie::commit(y) ...
 *   }
 *
 * The kernel body is compiled for x86 with recording stubs under
 * AddressSanitizer and run for every core of the launch mesh, twice with
 * different data. The trace gives each port's block sequence per core:
 *
 *   - checks (KernelTraceResult::errors): blocks inside their tensor, a load on
 *     an input port / a store on an output port, at most pp_depth blocks held,
 *     every block released / committed, every declared port used, the same
 *     schedule on every core and for any data, the output written exactly once,
 *     and -- through ASan -- no access outside a loaded block and none after
 *     its release ("data used without load"); the static bounds analysis
 *     (kernel_static.h) then names the constant behind such an access
 *     ("cause: `oc_t = 15` ... fits with oc_t = 16");
 *   - fit: per port, offset = base_p0*p0 + base_p1*p1 + step_i*i + step_j*j
 *     with rounds i outer, j inner, and a constant shape (the form of a DMA BD);
 *   - kernel-only (no aie::Conv2d / aie::Mapping): the fits are mapped straight
 *     onto the routing IR carriers (trace_routing.h: per dim mesh split, on-core
 *     loop, halo; per port broadcast / gather) and the source is rewritten into
 *     the explicit __global__(policy) + per-port GemmSpace form -- no operator
 *     recognition, so conv2d, GEMM and eltwise kernels route alike. When the
 *     blocks also form a spatial-halo conv the conv derivation is kept as a
 *     cross-check (KernelOnlyResult::derived);
 *   - __global__(conv, map) kernels: the same trace verifies the kernel's
 *     load_buffer / store_buffer blocks against the derived DMA.
 *
 * Design: doc/design/op_mapping_programming_model.md, "Kernel analysis".
 ******************************************************************************/
#pragma once

#include "conv_autotile.h"

#include <map>
#include <string>
#include <vector>

namespace convautotile {

struct TracePort {
    std::string var;        // kernel parameter name
    std::string role;       // IFM / WEIGHTS / OFM
    std::vector<int> shape; // tensor shape, elements
    // Data extent per dim when the tensor is padded up to `shape` (e.g. 3
    // channels aligned to 4); empty = the whole shape is data. The DMA fills
    // shape - extent elements (GemmSpace .padsize).
    std::vector<int> extent;
    int elem_bytes = 1;
    bool output = false;
    // The role tag was omitted (`aie::port<T, aie::shape<...>>`): `role` is
    // inferred -- OFM for an output window, else IFM, refined to WEIGHTS once the
    // trace shows the blocks broadcast along a mesh column. The role only names
    // the routed GemmSpace and lets a conv be recognized; the DMA comes from the
    // traced blocks either way.
    bool roleInferred = false;
};

struct TraceKernel {
    std::string name;
    size_t kwBegin = 0, kwEnd = 0; // `__global__` or `__global__(a, b)`
    size_t paramOpen = 0, paramClose = 0;
    size_t bodyOpen = 0, bodyClose = 0; // braces of the body
    int bodyLine = 1;                   // 1-based source line of bodyOpen
    std::vector<TracePort> ports;
};

// Per-port affine fit of the block sequence (the BlockDesc form).
struct PortFit {
    int rank = 0;
    std::vector<int> shape, base_p0, base_p1, step_i, step_j;
    int rounds_i = 1, rounds_j = 1;
};

// Dataflow options of one port, from its load_buffer / store_buffer calls
// (every block of a port must pass the same). All optional; 0 / false = the
// compiler's default.
//   .depth    the port's core buffers: 1 single, 2 ping-pong
//   .resident one block per core, held for the whole launch, one buffer
//   .tile     {tr, tc} (2-D loads): the DMA delivers the block panel-major --
//             tc-wide column panels one after another, each with all rows --
//             so every tr x tc tile is contiguous (tile (a, b) starts at
//             (b * rows / tr + a) * tr * tc elements)
struct BlockOpts {
    int depth = 0;
    bool resident = false;
    int tile[2] = {0, 0};
    bool any() const { return depth || resident || tile[0] || tile[1]; }
    bool operator==(const BlockOpts &o) const {
        return depth == o.depth && resident == o.resident && tile[0] == o.tile[0] && tile[1] == o.tile[1];
    }
};

struct KernelTraceResult {
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> notes;
    std::map<std::string, PortFit> fits;   // by port variable
    std::map<std::string, BlockOpts> opts; // by port variable, ports that pass options
};

// Trace + check one kernel on a rows x cols mesh. `prelude` is extra C++ the
// body needs (user #defines, op/mapping declarations).
KernelTraceResult traceKernel(const std::string &src, const TraceKernel &k, int rows, int cols,
                              const std::string &prelude, const std::string &srcName, int pp_depth = 2);

// What a logical trace (tensor form, kernel coordinates) found per port.
struct PortUse {
    std::vector<int> lo, hi; // bounding box of every block, may leave the tensor (.pad)
    bool padded = false;     // some load asked for {.pad = aie::pad_value::zero}
    bool whole_last = true;  // every block takes the whole last (channel) dim
};

// Trace in the kernel's own coordinates: the harness checks (bounds unless
// .pad, use without load, release / commit, pp_depth, data-dependent schedule)
// plus {.reuse = N} against the stores made while the block is held. No DMA fit.
KernelTraceResult traceLogical(const std::string &src, const TraceKernel &k, int rows, int cols,
                               const std::string &prelude, const std::string &srcName, int pp_depth,
                               std::map<std::string, PortUse> &use);

// The plain `__global__ <ret> name(aie::port<...> ...)` kernels of a source.
std::vector<TraceKernel> findPlainKernels(const std::string &src, const std::string &srcDir, std::string &err);

// Name of block dim `d` of a port role ("output channels" for WEIGHTS dim 0).
const char *traceDimName(const std::string &role, int d);

// The object-like #define lines of a source (what its kernel TU sees).
std::string traceDefines(const std::string &src);

// One routed port's traced blocks: what the kernel loads / stores per core
// (the routing IR's #routing.kernel_block of that tensor).
struct PortBlock {
    std::vector<int> tensor; // the port's tensor as the kernel declares it (aie::shape)
    PortFit fit;
    BlockOpts opts;
};

struct KernelOnlyResult {
    bool found = false; // the source has a kernel-only __global__ with aie::shape ports
    bool ok = true;
    std::string error; // first error, with the kernel name
    std::vector<std::string> errors, notes;
    std::string report; // human-readable analysis
    // Per kernel, when the blocks are also recognized as a spatial-halo conv
    // whose derivation reproduces them: the conv derivation (est_ms, and the
    // pipeline's tiling cross-check). Not needed to route the kernel.
    std::map<std::string, Derived> derived;
    // Per routed kernel: the kernel-TU declarations (block descriptors) for
    // aie::load_buffer / store_buffer (trace_routing.h routedTUDecls).
    std::map<std::string, std::vector<std::string>> tu_decls;
    // Per routed kernel: the estimated time (ms) -- the conv derivation's when
    // `derived` has the kernel, else the routing cost (routingEstMs).
    std::map<std::string, double> est_ms;
    // Per routed kernel, per port variable: the traced blocks and options.
    std::map<std::string, std::map<std::string, PortBlock>> blocks;
};

// Kernel-only sources: analyze, derive, verify, and rewrite `src` into the
// __global__(conv, map) form (in place). No-op when not applicable.
KernelOnlyResult inferKernelOnly(std::string &src, const std::string &srcDir, const std::string &srcName);

// Hand-written GemmSpaces + block loads: a `__global__(policy)` kernel whose
// ports name `constexpr aie::GemmSpace`s (aie::port<T, Space>) and whose body
// uses aie::load_buffer / store_buffer. The GemmSpaces are the authority and
// stay as written; the kernel is traced, its routing plan rendered through
// routingSpaces must equal every port's GemmSpace (spaceDiff: a conflict is
// an error naming the field), and then the kernel gets the block descriptors
// (tu_decls) its loads check against at run time, with aie::num_programs
// resolved in `src`. No-op (found == false) for any other kernel, and for the
// kernels in `skip` (the ones inferKernelOnly just rewrote into this form).
KernelOnlyResult verifySpaceKernels(std::string &src, const std::string &srcDir, const std::string &srcName,
                                    const std::vector<std::string> &skip = {});

struct KernelDesugar;

// __global__(conv, map) kernels that use aie::load_buffer / store_buffer:
// trace the body and check its blocks against the derived DMA of `k`. Returns
// the errors (empty = consistent, or nothing to check).
std::vector<std::string> verifyKernelBlocks(const std::string &src, const std::string &srcDir,
                                            const std::string &srcName, const KernelDesugar &k,
                                            std::vector<std::string> &notes);

} // namespace convautotile
