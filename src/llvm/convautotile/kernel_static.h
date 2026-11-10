/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Static block-bounds analysis of a block-load kernel body: which source
 * constant makes an access leave its load_buffer / store_buffer block.
 *
 * The x86 trace (kernel_trace.h) catches an access past a block through ASan,
 * but only says WHERE ("data used without being loaded at k.cc:29"). This pass
 * says WHY, from the source alone:
 *
 *   - model: the body's `const int` / macro constants, small const arrays,
 *     `for (int v = a; v < b; v += s)` loops, `if` / `?:` guards, block
 *     pointers (`T *p = aie::load_buffer(port, {off}, {shape})`), pointers
 *     derived from them (`T *r = p + e`) and local arrays (`T a[N]`);
 *   - accesses: `p[e]`, `aie::load_v<N>(p + e)`, `aie::load_unaligned_v<N>(p + e, ..)`,
 *     `aie::store_v(p + e, v)` (N from `extract<N>` / an aie::vector<T, N> value);
 *   - bounds: every needed loop variable (and program_id) is ENUMERATED -- no
 *     interval over-approximation, so correlated values (owB = pass * 16 +
 *     (pass ? 4 : 8)) stay exact -- and [index, index + N) is checked against
 *     the block's element count;
 *   - attribution: the overflowing element is decomposed into block
 *     coordinates, the loops that move the overflowing dim are named, and each
 *     literal constant (`const int oc_t = 15`) is varied alone: the one whose
 *     change makes EVERY access fit is reported as the cause ("fits with
 *     oc_t = 16").
 *
 * Offsets are computed in BYTES: each pointer carries its pointee size, so an
 * `int16_t *` view or a `(const int16_t *)p + e` / `reinterpret_cast<T *>(p) + e`
 * cast scales correctly; an unknown pointee size is a skip reason.
 *
 * An access whose index cannot be evaluated (data-dependent, unknown call,
 * reassigned variable) is skipped, never guessed -- with the reason
 * (StaticAccess::reason); a block pointer used in a form no access models
 * (passed to a helper, `*(p + e)`) is reported the same way.
 ******************************************************************************/
#pragma once

#include "kernel_trace.h"

#include <string>
#include <vector>

namespace convautotile {

struct StaticFinding {
    int line = 0;        // source line of the access
    std::string message; // the access, its block, the overflowing dim and the loops that drive it
};

// Every access the analysis saw, for cross-checking against the trace.
struct StaticAccess {
    int line = 0;
    std::string text;
    char status = 'o';  // o: inside its block, x: out of its block, s: not evaluable, u: use in no modelled form
    std::string reason; // s / u: why
};

struct StaticBoundsResult {
    std::vector<StaticFinding> findings; // one per out-of-block access
    std::string cause;                   // the constant whose change makes every access fit ("" = none found)
    int analysed = 0, skipped = 0;       // accesses checked / not evaluable (incl. unmodelled uses)
    std::vector<StaticAccess> accesses;
};

// Analyze kernel `k` of `src` on a rows x cols mesh. `prelude` supplies
// object-like #defines the body may use; `srcName` labels locations.
StaticBoundsResult staticBlockBounds(const std::string &src, const TraceKernel &k, int rows, int cols,
                                     const std::string &prelude, const std::string &srcName);

} // namespace convautotile
