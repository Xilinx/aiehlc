/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C++ templates on __global__ kernels.
 *
 *   template <int M, int K, int TK = 64, bool RELU = true>
 *   __global__ void gemm(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<M, K>> a, ...) { ... }
 *   ...
 *   gemm<256, 2304><<<mesh>>>(A, W, C);
 *
 * aiehlc reads a kernel's port shapes and body as SOURCE TEXT in several places
 * (the kernel analysis, the port parser, the kernel extraction for the AIE
 * compiler), and every one of them needs the shapes as constants. So templates
 * are instantiated on the text, before any of them runs -- like
 * inlineSourceIncludes and applyConvAutoTiling:
 *
 *   - every launch `name<args><<<mesh>>>(...)` of a template kernel selects an
 *     instance; defaults fill omitted trailing arguments;
 *   - each distinct instance is the kernel with every template-parameter token
 *     of its signature and body replaced by the argument (an integer argument
 *     is folded to a literal with the source's #defines, so shape parsers see a
 *     number); the `template <...>` header goes;
 *   - one instance keeps the kernel's name; several get one name each
 *     (`name__256_64_2304_64_1`), and each launch is rewritten to its instance.
 *
 * Supported parameters: integral non-type (`int`, `unsigned`, `long`, `bool`,
 * `size_t`, ...) and `typename` / `class`, each with an optional default.
 * Errors (not silently ignored): a template kernel never launched, a launch
 * with too many / too few arguments, an unsupported parameter kind (template
 * template parameters, packs).
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace convautotile {

struct KernelTemplateResult {
    bool found = false; // the source has a template __global__ kernel
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> notes; // one line per instance
};

// Instantiate every template __global__ kernel of `src` in place (see above).
// No-op (found == false) for a source without one.
KernelTemplateResult instantiateKernelTemplates(std::string &src, const std::string &srcDir);

} // namespace convautotile
