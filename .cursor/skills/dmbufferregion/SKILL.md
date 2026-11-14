---
name: dmbufferregion
description: AIE kernel link fails with `could not find free space for SpaceSymbol buf_in_ping_0 (or buf_in_pong_N) in memory DMb`, or aiehlc stops with `[TilingLinalg] ERROR: Memory budget exceeded! Total buffer requirement ... maxPingPongBytes=... numTensors=3` -- the port buffers do not fit the core's 32 KB buffer region / the pipeline's l1_budget check, although convautotile's old 48 KB model said they did.
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Port buffers vs core data memory

Two limits, both now legality rules in `convautotile::derive`
(`src/llvm/convautotile/conv_autotile.cpp`, doc/design/conv_autotiling.md):

1. **Pipeline check** (`tilinglinalg_pipeline`, "Memory budget exceeded"):
   `pp_depth x l1_budget x 3 tensors <= 49152`. Every tensor's buffers are
   sized at `l1_budget`, not at their real size.
2. **Link** (xchesscc, "could not find free space ... DMb"):
   `CoreMemAllocator(0x78000, 0x8000)` -- the bcf gives the port buffers a
   32 KB region at 0x78000; 0x70000..0x78000 is `DM_stack`. The last 2 KB are
   the klog ring and statics (block descriptors, cursors, `sync_buffer`) share
   it. `kernel.cc` declares a ping AND a pong array for every window whatever
   `pp_depth` says (`v4int8 buf_in_pong_1[...]` exists at pp_depth 1), so:
   `2 x (ifm_slab + weights_block + ofm_tile) + 3072 <= 32768`.
   `pp_depth = 1` passes check 1 but saves nothing at link.

Diagnose: `aout/worklocal/aieml.bcf` (`_symbol buf_* 0x...`, `_stack`) and
`build/kernel.map` ("Memory map for memory 'DMb'", "not enough free space").
Fix: a smaller time tile (fewer output rows/cols per round), or fewer output
channels per core. Kernel-only sources size `l1_budget` from their own blocks
and drop to `pp_depth = 1` only to pass check 1.
