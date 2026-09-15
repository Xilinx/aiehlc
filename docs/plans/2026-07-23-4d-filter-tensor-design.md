# 4D Filter (B) Tensor Support — Design

Date: 2026-07-23

## Goal

Change the conv2d `@main` filter argument from the flattened 2D
`memref<196x64xi8>` to the genuine 4D `memref<64x7x7x4xi8>`
(`[F=64, KH=7, KW=7, C=4]`), split d0 (F) into `HW_COLS=4` groups of 16 filters
per mesh column (per-col `[16,7,7,4]`), flowing all the way to `host.cc` /
`kernel.cc`.

## Motivation (correctness, not just cosmetics)

The host DDR filter buffer is laid out F-major: `filter[f*K + kk]` with
`K = KH*KW*C = 196`, i.e. `[F=64][K=196]` row-major. The current IR declares the
transpose `[196,64]` (K-major) and splits d0 → `[49,64]` (a K-slice for *all*
filters). The kernel, however, reads `B_local[j*196 + kk]` (16 filters × 196 K).
The byte counts match (`49*64 == 16*196 == 3136`), so nothing crashes, but the
per-column data distribution is wrong.

The 4D `[64,7,7,4]` split on d0 makes column `c` own filters `[c*16, c*16+16)`
with the full `KH*KW*C = 196` contraction — matching both the host F-major DDR
layout (`f*196 + kk`) and the kernel's `B_local[j*196 + kk]` indexing.

## Findings

- The pipeline below `aiehlc.cc` is already rank-generic for `splitDim=0`:
  - `routingmanager.cpp` split-shape copies all dims and divides
    `splitShape[0] /= splitnum`; a 4D `[64,7,7,4]` → `[16,7,7,4]`.
  - `ExtractDataConversion` (passdmaphoptodfscheblueprint) builds N-D
    offsets/sizes/strides.
  - Shim BD flat length and kernel buffer size are the product of *all* dims
    (`16*7*7*4 = 3136`), unchanged from the current `49*64`.
- The only blocker is the frontend: `aiehlc.cc` hardcodes the B filter
  `pti.shape = {macroDimK, macroDimN}` (2D) at ~L1913, with no input-side
  multi-dim lift (only the OUTPUT is lifted to 3D at ~L1928).
- `tdD3` / `tdD4` are NOT consumed for the B/col path (only output group2 and
  the im2col Conv2dSpace branch use them), so adding `d4=KH` is non-disruptive.
- `GemmSpace` already has a `d4` field (contractual field 7).

## Changes

1. `example/tileprogram/ccode/simpleconv2d.cc` — add `.d4 = {KH}` to `ColBC` so
   the filter port declares all four conv sub-dims: `d1=F`, `d4=KH`, `d2=KW`,
   `d3=C`.
2. `src/llvm/aiehlc.cc` — after the 3D output lift, add a 4D **input** lift for
   B (`policyResolved && isInput && perPort2D && pattern==0 && distribution==1`)
   that sets `pti.shape = {F, KH, KW, C}` from `tdD1/tdD4/tdD2/tdD3`, guarded by
   `tdD1/2/3/4.base > 0`, `F == macroDimN`, `KH*KW*C == macroDimK`. Matmul B
   (no `d3/d4`) stays 2D. Also make the rank-2 debug print at ~L2033 rank-safe.
3. Downstream — verify only (rank-generic).

## Verification

- `make -j` in `build/`, then
  `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc` → RC=0.
- `ir/dfschedule/0_initial.mlir`:
  `@main(..., %arg1: memref<64x7x7x4xi8>, ...)`,
  `partitiontensor ... splitdim=0 -> tensor<64x7x7x4xi8>`,
  `extract_data ... -> tensor<16x7x7x4xi8>`.
- `host.cc`: filter shim BD len = 3136 (unchanged).
- `kernel.cc`: `get_buffer_size(win_b)` = 3136, `get_tile_cols()` = 16.
- Regression `simplematmul2.cc`: B stays 2D `[K,N]` (4D guard does not fire).
