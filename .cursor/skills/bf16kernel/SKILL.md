---
name: bf16kernel
description: bfloat16 tensors in a kernel-only aiehlc kernel (`aie::port<input_window_bfloat16 *, ...>`, `const bfloat16 *x = aie::load_buffer(...)`). Use for "unknown type name input_window_bfloat16", a bf16 kernel whose DMA lengths are twice / half what they should be, `output block bytes per round = X in the derived tiling, but the kernel's blocks give 2X`, `does not compile for tracing` on `bfloat16`, a bf16 conv that no longer fits the core buffers, or a bf16 result where a few hundred outputs read 0 on one run in several.
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# bfloat16 kernel-only kernels

Example `example/tileprogram/ccode/simpleconv2dkernel_port_bf16.cc` (+
`simpleconv2d_bf16.h`). Design: `doc/design/bf16_support.md`.

## How it is carried
16-bit storage end to end: pipeline `i16`, core windows `v4int16`,
`get_input_async_window_int16`. `aie_block_io.h` aliases
`input_window_bfloat16` -> `input_window_int16` on the core (ADF has none). The
kernel body uses `bfloat16` (chess native; RNE x86 type in
`aie_api_x86emu.h`). Host side: `uint16_t` bits + `bf16_to_f32` / `f32_to_bf16`.

## Traps
- Element size is decided in TWO front ends: `aiehlc.cc` (bit width; a type it
  does not know defaults to 32) and `kernel_trace.cpp` `elemBytesOf` (defaults
  to 1). Add a new type to both, or the DMA and the analysis disagree.
- The conv cross-check derives with `Target.elem_bytes`; a 16-bit conv with the
  default 1 fails `checkConvTilingAgainstAnalysis` on bytes.
- One element type per kernel (window helpers are emitted for one type).
- bf16 doubles every block: the int8 7x28 conv tile overflows the 32 KB port
  region (skill dmbufferregion); use 7x14.
- No `accfloat` / bf16 `aie::mmul` in the x86 emulation yet: vector bf16
  kernels do not trace.
- A few hundred outputs reading the host `memset` value on ~1 run in 6 is the
  missing post-DMA invalidate of `__aie_launch` (skill outputtilemajorrace, last
  section), not bf16: invalidate the output after the launch
  (`Xil_DCacheInvalidateRange`) until `arthur/output-post-dma-invalidate` lands.
- Outside the sandbox `$TMPDIR` is unset: helper scripts keyed on it write to `/`.

## Verify
`build/aiehlc-convtile trace <src>`; `kernel_trace_cases.py ... bf16`; x86 with
the kemu prelude; board: `bf16 check: 802816 / 802816 bit-exact` + `PASS`.
