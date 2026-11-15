---
name: aiednnkernel
description: Writing an AIEDNN kernel -- a DNN op (GEMM, conv as im2col GEMM) for the AIE as a parameterised template, one launch per layer, K accumulated in fp32 on the core (`example/aiednn/gemm_kacc_bf16.cc`, `gen_variants.py`, tutorial/aiednn/howto/templatekernel.md). Use for "the core window is N elements x R round(s) ..., but the kernel's blocks are 4N" on a GEMM with M rounds AND K rounds, board `wait_io TIMEOUT` on input shims ch 1 for a K-accumulating GEMM, `{.tile / .resident} need blocks that are whole rows`, `'X' was not declared in this scope` for a template parameter, a launch that is slower with a bigger K chunk, or choosing tile / TK / stationarity / variants.
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# AIEDNN template kernels

Guide: `tutorial/aiednn/howto/templatekernel.md` (from the UNet deployment
report, http://xsjsswsip51:8099/ §15-21). Example: `example/aiednn/`.

## Shape of a kernel
Output-stationary: per core, a loop over its output tiles; per tile, an fp32
accumulator over K chunks (`aie::mmul<8,8,8,bfloat16,bfloat16,accfloat>` under
`__chess__`, scalar `#else` for the x86 trace); epilogue; one store. Host
pre-tiles A / W chunks in mmul operand order. One launch per layer.

## Traps (each fixed or guarded; keep it that way)
- **M rounds AND K rounds:** aiehlc divided every input's core window by
  m_rounds; only A (view [tile_rows, K]) holds M strips. Fixed in
  `passblueprinttoschedulekernel.cpp` + `flowtransfer_kernel.cpp` (from
  unet_m2 8c60bf8). The `#routing.kernel_block` check is what reports it.
- **3-D K BD started k_rounds times** (element-count len) -> input shims never
  finish, `wait_io TIMEOUT ch=1`. `flowtransfer_host.cpp` starts it once (from
  unet_m2 5791cbd) -- ONLY when its len is not already the whole walk:
  `simplematmul2`'s BD is the whole walk and its repeat + iteration step the N
  strips (k_rounds == n_rounds there by coincidence); the unconditional form
  broke it.
- `.tile` / `.resident` on a block that is not whole rows (a K chunk) were
  accepted and then ignored by the DMA; the analysis now refuses them.
- No C++ `template<>` on a `__global__`: the body is analysed as text
  (`'AD_TK' was not declared`). Parameters = plain `#define`s, no trailing
  comments (static bounds analysis), rewritten by a generator.
- All ports one element type; bf16 out (fp32 out = int32-packed ports).
- Cold launch: buffers are kernel BSS, re-written every launch: TK 128 (17 KB
  BSS) 53.6 ms vs TK 64 (9 KB) 33.0 ms on 256x64x2304; kload 98-99 %.
- Check outputs after `Xil_DCacheInvalidateRange` (missing post-DMA invalidate).

## Verify
`build/aiehlc-convtile trace`; x86 with the kemu prelude (scalar branch);
board: `[gemm_kacc] check: N / N bit-exact` + `PASS`; `--profiling` for the
kload / bdcfg / wait_io split. Byte-compare `simplematmul2`,
`simplegemm_kernel_port`, `simpleconv2dkernel_port` after touching the GEMM
M/K-round lowering.
