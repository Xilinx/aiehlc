/* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Which optional modes the GEMM megakernel (gemm_uk.cc) compiles. This default enables all of
 * them. A per-app build (aiegraph dnn_build, modes="auto") writes its own aie_modes_cfg.h next to
 * the aiehlc entry, where it shadows this one: the app runs once on x86 with call counting, and
 * only the modes of the entries it called (each AIEK_SPEC's `modes` line) stay in the core
 * program. The host's implementation list is cut to the same entries, so it never sends an
 * opcode the core no longer decodes. Plain GEMM (opcodes 0x00 / 0xA0) is always in. */
#ifndef AIE_MODES_CFG_H
#define AIE_MODES_CFG_H

#ifndef AIEK_MODE_GEMMQ
#define AIEK_MODE_GEMMQ 1 /* 0xF1: GEMM + integer requant on the core, fused s8 residual */
#endif
#ifndef AIEK_MODE_DW
#define AIEK_MODE_DW 1 /* 0xD7: depthwise, int32 out */
#endif
#ifndef AIEK_MODE_DWQ
#define AIEK_MODE_DWQ 1 /* 0xD8: depthwise + integer requant, up to 3 items per core (needs DW) */
#endif
#ifndef AIEK_MODE_POOL
#define AIEK_MODE_POOL 1 /* AIE_MODE_OP_POOL: pooling and trailing-axis reductions */
#endif
#ifndef AIEK_MODE_ELTWISE
#define AIEK_MODE_ELTWISE 1 /* AIE_MODE_OP_ELTWISE: binary of two 8-bit tensors */
#endif
#ifndef AIEK_MODE_LUT
#define AIEK_MODE_LUT 1 /* AIE_MODE_OP_LUT: 8-bit table lookup */
#endif
#ifndef AIEK_MODE_CONV
#define AIEK_MODE_CONV 1 /* direct conv: the core gathers the im2col tile from a raw input region */
#endif
#ifndef AIEK_MODE_MMT
/* opt-in (dnn_build modes="auto" turns it on when the app calls mmt_blk_s8s8): with it, every mode
 * together overflows the core's 16 KB program memory */
#define AIEK_MODE_MMT 0 /* AIE_MODE_OP_MMT: pre-tiled short-K int8 GEMM on mlir-aie's micro-kernel */
#endif
#ifndef AIEK_MODE_FVEC
#define AIEK_MODE_FVEC 0 /* AIE_MODE_OP_FVEC: f32 pointwise / softmax / norm rows (aie_modes_fvec.h), opt-in */
#endif
#ifndef AIEK_MODE_BF16
#define AIEK_MODE_BF16 0 /* AIE_MODE_OP_BF16: bf16 GEMM, f32 accumulation (bf16blk: / bf16x3:), opt-in */
#endif
#ifndef AIEK_MODE_SDPA
#define AIEK_MODE_SDPA 0 /* AIE_MODE_OP_SDPA: attention per (batch, head) item, opt-in (core program memory) */
#endif
#ifndef AIEK_MODE_WQ
#define AIEK_MODE_WQ 0 /* AIE_MODE_OP_WQ: grouped weight-only quant, f32 group-scaled sums (aie_modes_wq.h), opt-in */
#endif
#ifndef AIEK_MODE_FPOOL
#define AIEK_MODE_FPOOL 0 /* AIE_MODE_OP_FPOOL: f32 windowed pooling 2-D / 3-D (aie_modes_fpool.h), opt-in */
#endif
#ifndef AIEK_MODE_STEPS
#define AIEK_MODE_STEPS 1 /* AIE_MODE_OP_STEPS: step tables for monotone ops */
#endif

/* any mode that gathers per-core items from A instead of packing a GEMM operand */
#define AIEK_MODE_ITEMS                                                                                                \
    (AIEK_MODE_DW || AIEK_MODE_DWQ || AIEK_MODE_POOL || AIEK_MODE_ELTWISE || AIEK_MODE_LUT || AIEK_MODE_STEPS ||       \
     AIEK_MODE_SDPA || AIEK_MODE_FVEC || AIEK_MODE_FPOOL)
#define AIEK_MODE_TABLES (AIEK_MODE_LUT || AIEK_MODE_STEPS || AIEK_MODE_FVEC) /* B rows -> dwb per core */

#endif /* AIE_MODES_CFG_H */
