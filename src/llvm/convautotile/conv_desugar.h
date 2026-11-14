/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source-level desugaring of the declarative conv2d programming model:
 *
 *   constexpr aie::Conv2d  conv = {...};
 *   constexpr aie::Mapping map  = {...};
 *   __global__(conv, map) void k(aie::port<T, aie::IFM> a,
 *                                aie::port<T, aie::WEIGHTS> b,
 *                                aie::port<T, aie::OFM> c) { ... }
 *
 * is rewritten (before aiehlc's own rewrites run) into the explicit form the
 * existing pipeline already consumes:
 *
 *   #define SP_KH 7 ...                         // kernel-visible tile constants
 *   constexpr aie::GemmSpace __aie_k_IFM = {...};  // derived d1/d2/d3 (+halo)
 *   constexpr aie::GemmSpace __aie_k_WEIGHTS = {...};
 *   constexpr aie::GemmSpace __aie_k_OFM = {...};
 *   constexpr aie::GlobalPolicy __aie_k_policy = {.fullconnect_auto = 0};
 *   __global__(__aie_k_policy) void k(aie::port<T, __aie_k_IFM> a, ...)
 *
 * The rewritten buffer is what aiehlc writes to aout/newfile.cpp.
 ******************************************************************************/
#pragma once

#include "conv_autotile.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace convautotile {

struct KernelDesugar {
    std::string kernel;  // kernel function name
    std::string convVar; // aie::Conv2d variable
    std::string mapVar;  // aie::Mapping variable
    Conv2dSpec conv;
    MappingSpec mapping;
    Derived derived;
    // (port variable, role) of each aie::port<T, aie::IFM|WEIGHTS|OFM>, in order.
    std::vector<std::pair<std::string, std::string>> ports;
};

struct DesugarResult {
    bool found = false; // source uses the declarative model
    bool ok = true;
    std::string error;
    std::vector<std::string> warnings;
    std::vector<KernelDesugar> kernels;
};

// Rewrite `src` in place. `srcDir` resolves quoted #include headers so their
// object-like integer macros can be used inside the initializers.
// Kernels that use aie::load_buffer / store_buffer are also traced and their
// blocks verified against the derived DMA (kernel_trace.h); `srcName` names the
// source in those diagnostics.
DesugarResult desugarConvMapping(std::string &src, const std::string &srcDir, const Overrides &ov,
                                 const Target &tgt = Target(), const std::string &srcName = "kernel.cc");

// Parse just the declarations (no rewrite). Used by the aiehlc-convtile CLI.
DesugarResult parseConvMapping(const std::string &src, const std::string &srcDir);

// Kernel-TU declarations for one desugared kernel, for the Triton-style block
// convention (src/mlir/runtime/aie_block_io.h): the op as `<convVar>` (kernel
// reads conv.kh), the block sizes as `<mapVar>` (map.oh_t), and per port the
// block descriptor aiehlc_desc_<port> that aie::load_buffer / store_buffer
// check against. One declaration per entry; aiehlc carries them into the kernel
// TU next to the #defines.
std::vector<std::string> kernelTUDecls(const KernelDesugar &k);

// Source-text helpers (shared with kernel_trace.cpp): comments blanked with
// offsets preserved, bracket matching, whole-token search, the source's
// object-like integer macros (and those of the headers it includes from its own directory), integer expression
// evaluation over them, and the launch mesh of `device.partition({..}, R, C)`.
std::string sourceBlankComments(const std::string &s);
size_t sourceMatchClose(const std::string &s, size_t open, char o, char c);
size_t sourceFindToken(const std::string &s, const std::string &word, size_t from);
std::map<std::string, std::string> sourceMacros(const std::string &src, const std::string &srcDir);
bool sourceEvalInt(const std::string &expr, const std::map<std::string, std::string> &macros, long &v);
// 0x0 when unresolved; `why` (optional) then says what was missing.
std::pair<int, int> sourceLaunchMesh(const std::string &src, const std::string &srcDir, std::string *why = nullptr);

// A hand-written `constexpr aie::GemmSpace <name> = {...};` of `src` as the
// PortSpace it describes (role left empty), with aiehlc's defaults for omitted
// fields (the AIEHLC_TILING_STUBS: act / wgt Broadcast, layout Row, merge
// Default, pad DDR, im2col None, pp_depth 2, l1_budget 32 KB; an omitted
// .stride is 0 and has_stride false). Integers may use the source's #defines
// and `constexpr int` names. Fails closed: .m/.n/.k, a positional or unknown
// field, or a non-constant value is an error, so nothing that shapes the DMA
// is silently ignored. Used to check a hand-written GemmSpace against the
// kernel's aie::load_buffer / store_buffer blocks (kernel_trace.h).
bool sourceGemmSpace(const std::string &src, const std::string &srcDir, const std::string &name, PortSpace &out,
                     std::string &err);

// JSON report for all kernels of a DesugarResult.
std::string reportJson(const DesugarResult &r);

} // namespace convautotile
