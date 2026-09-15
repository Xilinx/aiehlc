# 3D Spatial-Halo Input (A) Tensor Support — Design

Date: 2026-07-23

## Goal

Change the conv2d `@main` A-input argument from the flattened 2D
`memref<230x920xi8>` to the genuine 3D `memref<230x230x4xi8>`
(`[H_pad=230, W_pad=230, C=4]`), split on d0 (H) into overlapping halo slabs
per mesh row (per-row `[61,230,4]`), flowing all the way to `host.cc`.

`RowBC_spatial` (the `win_a` GemmSpace) already declares the three sub-dims:
`d1`=HEIGHT halo (base=230), `d2`=WIDTH (base=230), `d3`=CHANNEL (base=4).
Only the frontend flattens W·C into `row_pitch=920`.

## Motivation

The A DDR buffer is a single contiguous padded slab `[H_pad][W_pad*C]`. Whether
we call it `[230,920]` or `[230,230,4]` the bytes are identical; the shim BD is
byte-for-byte the same. The 3D form makes the W and C dims first-class (matching
`RowBC_spatial.d2`/`d3` and the 3D conv output/4D filter already lifted), so the
whole conv graph is dimension-honest.

## Findings

- The physical DMA is byte-identical; the shim-BD geometry (`row_pitch=920`,
  `k_slice=244`, `slice=61`, `step=56`, …) comes from the **module
  `tensor_0.halo` attr**, which is **rank-independent**.
- `routingmanager.cpp` splitShape (~1069-1092) is already rank-generic for halo
  (only `splitShape[sd]=haloSlice`) → yields `[61,230,4]` unchanged.
- Three rank-sensitive spots:
  1. `routingmanager.cpp:947-950` builds the halo tiling as hardcoded `dims(2)`.
  2. `flowtransfer_host.cpp:139,779` read `rawWc = shape[rank-1]`, which flips
     from `920` (2D) to `4` (3D=C) → wrong DDR base stride.
  3. `aiehlc.cc` spatial-halo override sets 2D `pti.shape = {raw_h, row_pitch}`.

## Changes

1. `src/llvm/aiehlc.cc` — in the spatial-halo override, when
   `row_pitch>0 && input_c>0 && row_pitch%input_c==0`, emit
   `pti.shape = {raw_h, row_pitch/input_c, input_c}` = `[230,230,4]`; else keep
   the 2D `[raw_h, row_pitch]`. Matmul / non-halo untouched.
2. `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp` (~947-950) —
   make the halo tiling rank-aware: `dims(rank)`; `d0`=H halo (unchanged), and
   for rank≥3 fill `d1`=W and `d2`=C as full-extent single-slice levels (the
   blueprint only reads `d0`'s step + the module `.halo`, so d1/d2 are
   structural). splitShape already yields `[61,230,4]`.
3. `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_host.cpp`
   (139, 779, and sibling `rawWc`-derived strides) — replace
   `rawWc = shape[rank-1]` with `rawWc = product of shape dims [1..rank-1]`
   (= 920 for both 2D and 3D), so DDR base offsets stay identical.

## Invariant

Module `.halo` attr + every shim BD length/stride/DDR-offset stays byte-identical
to the 2D baseline. Only tensor ranks change.

## Verification

- `ir/dfschedule/0_initial.mlir`:
  `@main(%arg0: memref<230x230x4xi8>, ...)`,
  `routingcreatescheduletensor ... shape = [230, 230, 4], dim = 3`,
  `partitiontensor ... splitdim=0 -> tensor<230x230x4xi8>`,
  `routingextract_data ... -> tensor<61x230x4xi8>`.
- `host.cc`: A-tensor shim BD lengths + DDR base offsets diff-clean vs the
  current 2D build.
- Regression: `simplematmul2.cc` (no halo) and the conv filter/output paths
  unchanged.
