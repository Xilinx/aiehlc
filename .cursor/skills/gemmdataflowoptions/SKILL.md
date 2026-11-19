---
name: gemmdataflowoptions
description: Kernel-only block-load kernel with dataflow options -- `aie::load_buffer(p, off, shape, {.depth = 1|2, .resident = true, .tile = {tr, tc}})`, `aie::store_buffer(..., {.depth = n})` -- and the `#routing.kernel_block` carrier behind them. Use for `error: ... the core window is N elements x R round(s) per launch, but the kernel's blocks (#routing.kernel_block) are ...`, `{.resident = true}, but the kernel takes N blocks`, `the port buffers take N B of core memory`, `aie::store_buffer takes only {.depth}`, a GEMM kernel whose B is loaded once / A in full-K strips building rc=0 but hanging, `could not find free space for SpaceSymbol buf_in_ping_0 in memory DMb` on a kernel-only GEMM, or `use of undeclared identifier` for a kernel macro defined under `#ifndef`.
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Dataflow options on kernel-only kernels

Example: `example/tileprogram/ccode/simplegemm_dataflow.cc`. Design:
`doc/design/trace_to_routing.md`, "Dataflow options".

## What each option does
- `.resident = true` (loads): one block per core, held for the whole launch, ONE
  core buffer; the shim sends it once instead of with every round of the other
  ports. Load it before the loop, `aie::release` it after.
- `.depth = 1 | 2`: the port's core buffers. 1 with several rounds = a single BD
  chained to itself (lock init stays 1).
- `.tile = {tr, tc}` (2-D loads): the shim delivers the block **panel-major**
  (tc-wide column panels, each with all rows), so each tr x tc tile is
  contiguous: tile (a, b) at `(b * rows / tr + a) * tr * tc`. Not row-major tiles
  in row order -- that needs 4 BD dims, the shim has 3. It changes the bytes the
  kernel sees: the kernel's indexing must match.
- Order in the initializer: `.reuse, .pad, .depth, .resident, .tile`. Same
  options on every block of a port. The core never sees them
  (`dropBlockOptions`).

## How it lowers
Trace (`checkBlockOpts`) -> `KernelOnlyResult::blocks` -> aiehlc
`kernelBlockFor` -> `TensorParam::kernelBlock` -> `#routing.kernel_block` on
`routingcreatescheduletensor` -> `blueprint_sched::resolveKernelBlock`
(`helper/flowtransfer_common.cpp`), used by host AND kernel lowering:
- **check** every GEMM-path port (`fullconnect_auto = 1`, no halo);
- **follow** inputs whose blocks are row-contiguous: shim BD per block at
  `step_i*i + step_j*j`, core window = block, own `pp_depth`, `.tile` dims, sent
  once when the kernel takes one block. Grep the build log for `[KernelBlock]`.

## Traps
- The GEMM path **relabels B as `[K, N]`** in `TensorParam::shape` while the
  bytes stay `[N, K]`. `kernel_block.tensor_shape` must be the kernel's own
  shape (`PortBlock::tensor`), or a K-chunked B looks row-contiguous and gets
  "followed" (old `simplegemm_kernel_port.cc` host.cc changed).
- **An int8 transpose is not a DMA operation** (32-bit words): a GEMM B for
  `aie::mmul` (K x N tiles) is packed by the host (`gemm_df_pack_b`), like
  conv2dstem's filter. `.tile` only re-orders whole words.
- **Port buffers get 32 KB, not 48** (skill dmbufferregion), and kernel.cc
  declares ping AND pong for every window except a single-buffered followed
  input. A double-buffered 16 KB B + 4 KB A strips links fail; `.resident`
  fixes it. The trace now reports it (`checkPortBuffers`).
- **Kernel macros must be plain `#define`s**: aiehlc copies `#define` lines into
  the kernel, not `#ifndef` blocks (`use of undeclared identifier 'GD_P'`).
- `#define X 16 // comment` defeats the static bounds analysis ("not
  evaluable"); put the comment on its own line.
- Mutating test sources by text: anchor on the call, not `{.resident = true}`
  (the file header mentions it first).
- Never rebuild `build/aiehlc` while an `aiehlc.sh` build is running; the
  queued builds then mix compilers.

## Spatial-halo conv
`simpleconv2dkernel_port_opts.cc`: `.resident` weights (followed on
`fullconnect_auto = 0` too: one buffer, sent once), `.depth = 2` slab/output.
`.depth = 1` on a halo port is an error; `.tile` is 2-D only. Before the fix
any option on a conv gave `the routed carrier differs from the spatial-halo
conv derivation ... (internal error)` (pp_depth was compared).

## Board
palmyra 213 (2026-11-11): `simplegemm_dataflow.cc` GOLDEN PASS (524,288), 79.55 ms
per launch; `--profiling`: kload 96.5 %, bdcfg 2.3 %, wait_io 1.2 % -- the
launch is the per-call kernel ELF reload (26.6 KB of buffer BSS per core over
MMIO), not the DMA. Board runs need the sandbox off (devlock file, ssh).

## Verify
`build/aiehlc-convtile trace <src>` (options listed per port),
`python3 src/llvm/convautotile/unitest/kernel_trace_cases.py build/aiehlc-convtile dataflow`,
`bash src/llvm/convautotile/unitest/run.sh`; byte-compare `kernel.cc routing.cc
host.cc aieml.bcf aieml.prx` of `simplegemm_kernel_port.cc`,
`simpleconv2dkernel_port.cc`, `simplematmul2.cc` against a build before the
change.
