/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * eager_fmath.h -- scalar f32 math of the eager float templates (feltw, fbinary, ...), shared by the core (through
 * the templates' AIEHLC_KERNEL_PROLOGUE) and their workstation references. The functions are the fixed-megakernel
 * FVEC mode's (src/aietensorop/ops/aie_modes_fvec.h, aie_fvec_prog) in scalar form: exp is 2^n e^r (Cody-Waite,
 * degree-6 e^r), 1/x and 1/sqrt(x) are Newton from a bit-trick seed, log is Newton on e^y = x, erf is
 * Abramowitz-Stegun 7.1.26, sin / cos / tan reduce by 2 pi or pi in two parts. Contract (FVEC's): within
 * 1e-5 + 1e-5 |ref| of ref:'s double arithmetic, not bit-exact.
 *
 * The algorithm codes only; the math itself is eager_fmath_ops.cc's EAGER_FMATH_OPS (a macro of lambdas, so it
 * reaches the core compile and the kernel trace as text). eager_fop(F, x, y, a, b) there: the algorithm F (FVEC's
 * AIE_FVF_* code: the host's fvec_eltwise_fn picks it) of x (unary) or of x and y (binary), alpha / beta.
 ******************************************************************************/
#ifndef EAGER_FMATH_H
#define EAGER_FMATH_H

#include <stdint.h>
#ifndef EAGER_FMATH_OPS /* an aiehlc build has the spliced text already; a workstation build includes it */
#include "eager_fmath_ops.cc"
#endif

#endif /* EAGER_FMATH_H */
