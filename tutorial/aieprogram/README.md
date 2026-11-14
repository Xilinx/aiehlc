# Programming the AIE array with aiehlc -- four tutorials

These tutorials teach the **kernel-only** programming model: you write one C++
file with a `__global__` kernel and a `main()`. The kernel says, in the
coordinates of the whole tensor, which block each core reads and writes. The
compiler traces those blocks and derives the DMA, the stream routing and the
buffers from them. You never write a DMA descriptor.

| # | Tutorial | Example | What it adds |
|---|---|---|---|
| 1 | [GEMM on a 4x4 tile mesh](01_gemm_4x4.md) | `simplegemm_kernel_port.cc` | the model: ports, `program_id`, `load_buffer` / `store_buffer`, `release` / `commit`, the host launch |
| 2 | [Conv2d on a 4x4 tile mesh](02_conv2d_4x4.md) | `simpleconv2dkernel_port_less.cc` | 3-D / 4-D tensors, an output-stationary conv tiling, the input halo |
| 3 | [Low-level scheduling options](03_conv2d_scheduling.md) | `simpleconv2dkernel_port_opts.cc` | per-port `{.depth}` / `{.resident}` (and `{.tile}`): buffers and DMA, same compute |
| 4 | [Vector instructions + scheduling](04_conv2d_vector_scheduling.md) | `simpleconv2dkernel_port.cc`, `simpleconv2dkernel_port_vec_opts.cc` | `aie::mmul<8,8,8>`, an im2col-free A tile, and a schedule (`output_tile_major`) for a kernel that got fast |

All examples live in `example/tileprogram/ccode/`. Read them in order: each
one is the previous one plus one idea.

## Build and run any example

```bash
# from the repo root (aiehlc.sh writes ./aout relative to the current directory)
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplegemm_kernel_port.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf      # palmyra board
```

Every example's `main()` checks its own result and prints `PASS` / `FAIL`.

## Measured on the board (palmyra, AIE2PS, 2026-11-14)

| Example | Result | Launch time |
|---|---|---|
| `simplegemm_kernel_port.cc` (64x64x256) | PASS 4096 / 4096 | 18.4 ms |
| `simpleconv2dkernel_port_less.cc` (scalar conv) | PASS 802,816 / 802,816 | 93.8 ms |
| `simpleconv2dkernel_port_opts.cc` (+ `.resident` weights) | PASS 802,816 / 802,816 | 85.8 ms |
| `simpleconv2dkernel_port_vec_opts.cc` (`aie::mmul` + options + `output_tile_major`) | PASS 802,816 / 802,816 (3 / 3 runs) | 56.4 ms |

A launch re-loads the kernel and re-programs the array, so these times are
mostly configuration, not compute (tutorial 3 explains why `.resident` still
moves them).

## Figures

The SVGs in `img/` come from `img/make_figures.py` (stdlib Python, numbers
taken from the examples): `python3 tutorial/aieprogram/img/make_figures.py`.
