# bfloat16 ports in kernel-only kernels

A kernel-only kernel can take and produce `bfloat16` tensors:

```cpp
__global__ void conv2d_spatial(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
                               aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
                               aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const bfloat16 *w = aie::load_buffer(win_b, {...}, {16, 7, 7, 4}, {.resident = true});
    ...  float acc = ...; acc += (float)x[..] * (float)w[..]; y[..] = (bfloat16)acc;
```

Example: `example/tileprogram/ccode/simpleconv2dkernel_port_bf16.cc` (the bf16
twin of `simpleconv2dkernel_port_less.cc`; host side `simpleconv2d_bf16.h`).

## Model

A bfloat16 tensor travels as **16-bit storage**: the DMA moves bytes, the
pipeline carries the tensor as `i16`, the core windows are `v4int16` and the
generated wrapper passes `get_input_async_window_int16(...)`. Only the kernel
body sees `bfloat16` -- chess's native type on the core, a matching
round-to-nearest-even type in the x86 emulation. ADF has no bfloat16 window, so
`aie_block_io.h` aliases `input_window_bfloat16` / `output_window_bfloat16` to
the int16 windows on the core (`__chess__`).

| Layer | Change |
|---|---|
| frontend port parse (`aiehlc.cc`) | `bfloat16` windows are 16-bit (was: the 32-bit default); parse stubs typedef the bf16 windows |
| kernel analysis (`kernel_trace.cpp` `elemBytesOf`) | 2 bytes (was: the 1-byte default) |
| conv cross-check (`convCrossCheck`) | `derive` with `Target.elem_bytes` = the IFM's element size (was: 1, so the byte check against the pipeline failed for any 16-bit conv) |
| x86 emulation (`aie_api_x86emu.h`) | global `bfloat16` (fp32 upper half, RNE from float, exact to float) |
| block I/O (`aie_block_io.h`) | `load_buffer` / `store_buffer` return a block pointer that converts to the pointer type it is assigned to (`const bfloat16 *`, `int8_t *`, ...), as the trace and kemu already did; bf16 window aliases |
| kemu (`src/aiegraph/kemu.py`) | bf16 windows; the `{.depth / .resident / .tile}` options (it rejected them before) |

## Limits

- **One element type per kernel.** The kernel lowering emits the window helpers
  for one type (`passdfscheduletokernelapi`), so all ports are bf16 (or all int8,
  ...). bf16 in with fp32 out is not supported.
- **Scalar compute only so far.** The x86 emulation has no `accfloat` / bf16
  `aie::mmul`; a vector bf16 kernel does not trace yet. AIE2PS has bf16 mmul
  shapes 4x8x4, 4x8x8, 8x8x8, 4x16x8, 8x1x8 (`aie_api/detail/aie2ps/mmul_fp_fp.hpp`).
- **Twice the bytes per block.** The int8 tile 7x28 does not fit the 32 KB port
  buffer region in bf16; the example uses 7x14 (5016 B slab, 6272 B weights
  `.resident`, 3136 B output).
- MLIR still says `i16`: anything that later needs the real type (an fp
  epilogue in generated code, a typed buffer dump) must add `bf16` there.

## Checking a bf16 result

The example's data are multiples of 1/4, so every fp32 partial sum is exact and
the result does not depend on accumulation order; the check counts bit-exact
outputs and fails beyond 1 ULP. Real data would need a tolerance.

## Verified (2026-11-14)

- `aiehlc-convtile trace`: OK, blocks 2 B per element, conv cross-check passes;
  `kernel_trace_cases.py` BF16_CASES (baseline + an out-of-block read caught).
- x86 (kemu prelude, kernel + host): 802,816 / 802,816 bit-exact.
- Full aiehlc build: BD lengths 5016 / 6272 / 3136 B, `v4int16` windows,
  flowcheck OK. `simplegemm_kernel_port.cc`, `simpleconv2dkernel_port.cc`,
  `simplematmul2.cc`: generated sources byte-identical to a compiler built from
  HEAD without these changes (the bf16 parse stubs are emitted only for sources
  that use `_window_bfloat16`; unconditionally they added 2 lines to every
  host.cc). `_opts`, `_less`, `simplegemm_dataflow` still build with the new
  block pointer.
- palmyra 213: **802,816 / 802,816 bit-exact in 15 of 15 runs** with the
  post-launch invalidate (12, then 3 with the final compiler; 2031 ms, scalar
  fp32 on the core). Without the host's post-launch invalidate, 1 of 6 runs
  showed 191 outputs still reading the host's `memset` 0 -- the known
  missing post-DMA invalidate of the generated `__aie_launch` (it cleans and
  invalidates outputs only before the DMA; fix on
  `arthur/output-post-dma-invalidate`, not on main), not a bf16 issue. The
  example invalidates `output` after the launch until that lands.
