---
name: blockloadkernel
description: Writing or debugging a Triton-style block-load AIE kernel -- kernel-only (`__global__ void k(aie::port<T, aie::IFM, aie::shape<...>> ...)`, DMA inferred from the kernel) or on top of `__global__(conv, map)` -- with `aie::program_id`, `aie::load_buffer` / `aie::store_buffer` in tensor coordinates, `aie::release` / `aie::commit`. Tensor form (`aie::in<aie::tensor>`, `.pad`, `.reuse`, `aie::autotune`, `[TensorForm]` report). Vector kernels (`aie::mmul` traced through `aie_api_x86emu.h`). Use for `[KernelAnalysis] ERROR` (too big / too small, outside the tensor, data used without being loaded, use after release, never committed, column-outer loops, schedule depends on data, never stored), Use for "no member named 'program_id' in namespace 'aie'", klog BLKE/BLKW/BLKG, a kernel local clashing with an emitted macro (`OH_T`, `OW_T`, `TILE_*`, `SP_*`, `K`), or to emulate the generated kernel on x86 before a board run. Also: hand-written `constexpr aie::GemmSpace` ports (`aie::port<T, MySpace>`) with `aie::load_buffer` -- `use of undeclared identifier 'aiehlc_desc_<port>'`, or `(GemmSpace X) .d1... = A, the kernel's blocks give B`.
---

# Triton-style block-load kernels

Example: `example/tileprogram/ccode/simpleconv2d_new_load.cc`. Design:
`doc/design/op_mapping_programming_model.md` §4 (the derivation itself is
convautotile: `doc/design/conv_autotiling.md`).

## How it lowers
- convautotile desugars `conv`/`map` into GemmSpaces + kernel `#define`s and,
  via `kernelTUDecls`, the kernel-TU `conv`, `map` (block sizes) and per-port
  `aiehlc_desc_<port>` block descriptors (same `Derived` numbers as the DMA).
  `aiehlc.cc` appends those after the `#define` scan (`convKernelDecls`).
- `resolveBlockIoBuiltins` (tilinglinalg_pipeline.cpp) rewrites
  `aie::load_buffer/store_buffer/release/commit` to `aie_block_io.h` and
  `aie::program_id(0|1)` to `get_coreid()` minus the `dmaphop.tile` placement
  base (attribute `tiletype`, printed as `TILETYPE`).

## Traps
- A load does NOT choose the data: the DMA order is fixed; the call takes the
  next window and only CHECKS the block (klog BLKE, visible with kernel logging
  on). Wrong offsets can still give a correct output.
- Kernel locals must not reuse emitted macro names: convautotile emits
  `OH_T`, `OW_T`, `TILE_*`, `SP_*`; the example header defines `M`, `N`, `K`.
  Use lower-case locals (`oh_t`, `ow_t`).
- The op is the math (`224x224x3`, `.pad = 3`), not the padded DDR layout.
- Emulate before the board:
  `bash example/tileprogram/ccode/simpleconv2ddebug/emu_conv2d_spatial.sh aout/worklocal`
  (802,816 outputs vs the CPU reference, plus the BLKE count).

## Kernel-only kernels (no conv / map)
- Ports carry the tensor: `aie::port<T, aie::IFM, aie::shape<230, 230, 4>>`.
  `inferKernelOnly` (kernel_trace.cpp) traces the body on x86 under ASan for
  every core, fits each port to base/step/rounds, recognizes the spatial-halo
  conv, derives with convautotile, compares exactly, then rewrites the source
  into `__global__(__aie_conv_k, __aie_map_k)`. Example:
  `example/tileprogram/ccode/simpleconv2dkernel_port.cc`.
- Iterate with `build/aiehlc-convtile trace <file.cc>` (seconds) instead of a
  full aiehlc.sh run; `AIEHLC_TRACE_KEEP=1` keeps the harness.
- A too-small slab is reported as "data used without being loaded" (ASan), not
  as a shape mismatch: the conv compute reads past it first.
- The body must compile on x86 and the schedule must be data-independent.
- Regression: `python3 src/llvm/convautotile/unitest/kernel_trace_cases.py build/aiehlc-convtile`.

## Hand-written GemmSpaces + block loads
- `__global__(policy) void k(aie::port<T, MySpace> p, ...)` with
  `aie::load_buffer` in the body (`simpleconv2dkernel_space.cc`). Before
  2026-11-10 this failed at kernel compile: `aiehlc_desc_<port>` is referenced
  by every rewritten load but was only emitted for convautotile-derived ports.
- `verifySpaceKernels` traces the body and requires `routingSpaces(trace plan)`
  == each hand GemmSpace (`spaceDiff`). The GemmSpace is never rewritten. Do NOT
  try to invert a GemmSpace into blocks: `tile_round` is a mesh split on one dim
  and a loop on another, and some GemmSpaces are ambiguous (GEMM A: M-split +
  K-loop renders like K-split + M-loop).
- Compared: layout, act/wgt, merge_order, mesh groups, im2col, d1..d4 incl.
  slice_tiling, with aiehlc's defaults filled in (an omitted `.stride` is 0, so
  writing `.stride` where the derived form has none is a conflict). Not compared:
  pad, pp_depth, l1_budget.
- Fails closed: `.m/.n/.k`, unknown fields (`.win`), non-constant values,
  GemmSpace in a header (only the .cc is searched) are errors.
- A kernel-only kernel is rewritten into this form first; aiehlc passes its
  name as `skip` so it is not traced twice.
- Quick check: `build/aiehlc-convtile trace <file.cc>`; regression `SPACE_CASES`.

## Tensor form (aie::in / aie::out, .pad / .reuse, autotune)
- `simpleconv2dkernel.cc`: `kernel_front.cpp::lowerTensorForm` lowers it to the
  kernel-only form above (pass A in kernel coordinates, staged layout, pass B).
  Debug the lowering with `AIEHLC_FRONT_DUMP=/tmp/tf build/aiehlc-convtile trace <file>`
  (writes `/tmp/tf.<values>.logical.cc` / `.staged.cc` per candidate).
- With `aie::autotune` a bug shows on EVERY candidate; the report lists each
  candidate's first error and the ERROR lines are the default's.
- `[TensorForm] ... rejected: ... (a conv border? ...)`: add `{.pad = aie::pad_value::zero}`.
- Locals named like desugar `#define`s (`OH_T`, `OW_T`, `K`, `M`, `N`, `TILE_*`)
  are renamed with a trailing `_` in the generated kernel.
- End-to-end x86 check of a build: `bash example/tileprogram/ccode/simpleconv2ddebug/emu_conv2d_tensor.sh
  example/tileprogram/ccode/simpleconv2dkernel.cc aout/worklocal` -> `GOLDEN PASS`.
- `auto a = aie::alloc<X>(), b = aie::alloc<Y>();` is not valid C++ (two types
  in one `auto`); the lowering splits it and says so in a note.

## Vector kernels (aie::mmul) under the kernel analysis
- The trace compiles the body with g++: AIE API calls come from
  `src/mlir/runtime/aie_api_x86emu.h` (per-element emulation). A call it lacks
  is "does not compile for tracing" -- add it there, matching the AIE API docs.
- `acc32` / `acc64` are GLOBAL tags (`aie::accum<acc32, 64>`); `aie::acc32`
  fails in xchesscc AND in the emulation.
- DMA windows are only 32-byte aligned: `aie::load_v<64>` / `store_v` of 64
  bytes on a window is reported (`needs a 64-byte aligned address`); use
  `load_unaligned_v<64>(p, 4)` or 32-byte halves.
- A 64-byte A-tile load must not run past the block even if the extra lanes
  meet zero weights (ASan reports it, and on the last slab row it reads past
  the buffer): shift the last K-block back and zero those B rows
  (`simpleconv2dkernel_port.cc`, off = 20).
- End to end on x86: `emu_conv2d_spatial.sh aout/worklocal` (includes the
  emulation). Cycles: `kernel.lst` -- count bundles vs VMAC between
  `.begin_of_loop` / `.end_of_loop`.
