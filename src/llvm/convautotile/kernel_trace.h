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
 *   - kernel-only (no aie::Conv2d / aie::Mapping): the fits are recognized as a
 *     spatial-halo conv (mesh split, time tiles, stride, kernel window) and the
 *     source is rewritten into the __global__(conv, map) form the conv
 *     auto-tiling desugar compiles, after checking the kernel's blocks are
 *     exactly what that derivation's DMA delivers (too big / too small / wrong
 *     step are named);
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
    int elem_bytes = 1;
    bool output = false;
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

struct KernelTraceResult {
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> notes;
    std::map<std::string, PortFit> fits; // by port variable
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

struct KernelOnlyResult {
    bool found = false; // the source has a kernel-only __global__ with aie::shape ports
    bool ok = true;
    std::string error; // first error, with the kernel name
    std::vector<std::string> errors, notes;
    std::string report; // human-readable analysis
    std::map<std::string, Derived> derived; // per kernel, when the DMA was derived
};

// Kernel-only sources: analyze, derive, verify, and rewrite `src` into the
// __global__(conv, map) form (in place). No-op when not applicable.
KernelOnlyResult inferKernelOnly(std::string &src, const std::string &srcDir, const std::string &srcName);

struct KernelDesugar;

// __global__(conv, map) kernels that use aie::load_buffer / store_buffer:
// trace the body and check its blocks against the derived DMA of `k`. Returns
// the errors (empty = consistent, or nothing to check).
std::vector<std::string> verifyKernelBlocks(const std::string &src, const std::string &srcDir,
                                            const std::string &srcName, const KernelDesugar &k,
                                            std::vector<std::string> &notes);

} // namespace convautotile
