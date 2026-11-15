# Programming the AIE array with aiehlc

These tutorials teach the **kernel-only** programming model: one C++ file with
a `__global__` kernel and a `main()`. The kernel says, in the coordinates of
the whole tensor, which block each core reads and writes. The compiler traces
those blocks and derives the DMA, the stream routing and the buffers. You never
write a DMA descriptor.

They go **by operator family first**, then **optimize one operator, GEMM**,
step by step.

| Chapter | Operator family | Example | Status |
|---|---|---|---|
| [1a](01_contraction/gemm.md) | Contraction: GEMM on a 4x4 mesh -- the whole model, every line, how the compiler gets from `load_buffer` to DMA, routing and C++ | `simplegemm_kernel_port.cc` | runs, board PASS |
| [1b](01_contraction/conv2d.md) | Contraction: conv2d -- 3-D / 4-D tensors, the input halo | `simpleconv2dkernel_port_less.cc` | runs, board PASS |
| [2](02_elementwise/elementwise.md) | Elementwise on 4x4 by redundant broadcast: a mesh row's data broadcast, the work split by column | `simpleeltwise_add.cc` | analysis passes; lowering pending |
| [3](03_reduction/reduction.md) | Reduction on 4x4 in two levels: partial sums in the array, combine on the host | `simplereduce_rowsum.cc` | analysis passes; lowering pending |
| [4](04_data_movement/data_movement.md) | Data movement: transpose on 4x4 -- what the DMA re-orders, what the core does | `simpletranspose.cc` | analysis passes; lowering pending |
| [5](05_indexing/indexing.md) | Indexing on 4x4: crop by host view, stride by footprint -- and why runtime gathers do not fit | `simpleslice.cc` | analysis passes; lowering pending |
| [6.1](06_gemm_optimization/1_low_level_schedule.md) | GEMM optimization: the low-level schedule (`.depth`, `.resident`) | `simplegemm_opt0_baseline.cc`, `simplegemm_opt1_schedule.cc` | runs, board PASS |
| [6.2](06_gemm_optimization/2_vector_instructions.md) | GEMM optimization: vector instructions (`aie::mmul`, `.tile`) | `simplegemm_dataflow.cc` | runs, board PASS |
| [6.3](06_gemm_optimization/3_conv_scheduling.md) | The same scheduling options on the conv2d | `simpleconv2dkernel_port_opts.cc` | runs, board PASS |
| [6.4](06_gemm_optimization/4_conv_vector_scheduling.md) | Vector conv2d + scheduling, and the output race a fast kernel exposes | `simpleconv2dkernel_port_vec_opts.cc` | runs, board PASS |

"Lowering pending": chapters 2-5 show kernels the compiler's kernel analysis
accepts, but the lowering after it still derives core windows from its GEMM
tiling model and stops the build for those shapes. Making it follow every
port's traced blocks is the next compiler step; those chapters get runnable,
board-measured examples when it lands. Their model and limits are current.

## Operator families

| Family | What it does | Examples | Bound by | On the AIE |
|---|---|---|---|---|
| Contraction | multiply and sum over a shared dimension | GEMM, linear, conv, einsum, attention matmuls | compute (inputs reused many times) | all cores, broadcast streams, matrix unit |
| Elementwise | one output from the same position of each input | add, mul, ReLU, cast, requantize, bias / residual add | data movement (one op per byte) | count streams and bytes; fuse into the producer |
| Reduction | fold many elements into fewer | sum, mean, max, argmax, norms; softmax / layernorm statistics | data movement, plus combining partial results | reduce where no core needs another's partial, or combine in two levels |
| Data movement | re-arrange bytes by a shape rule | transpose, permute, reshape, layout change, concat, split, pad, copy | data movement (no arithmetic) | the DMA re-orders words, the core re-orders bytes |
| Indexing | select elements by an index | slice, strided slice, crop; gather, index_select, embedding, scatter | addressing | static (affine) indices: yes; data-dependent: not in this model |

Real networks are mostly contractions surrounded by the other four; the best
place for an elementwise, reduction or layout op is often **inside** the
contraction kernel next to it (fusion), so its bytes never make an extra trip
to DDR.

All examples live in `example/tileprogram/ccode/`.

## Build and run an example

```bash
# from the repo root (aiehlc.sh writes ./aout relative to the current directory)
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplegemm_kernel_port.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf      # palmyra board
```

Every example checks its own result (`PASS` / `GOLDEN PASS`). Only the kernel
analysis, in about a second and without a board:
`./build/aiehlc-convtile trace <file.cc>`.

## Measured on the board (palmyra, AIE2PS)

| Example | Result | Launch |
|---|---|---|
| `simplegemm_kernel_port.cc` (64x64x256) | PASS 4096 / 4096 | 18.4 ms |
| `simplegemm_opt0_baseline.cc` (2048x256x256, scalar) | GOLDEN PASS 524,288 | 90.6 ms |
| `simplegemm_opt1_schedule.cc` (+ `.depth` / `.resident`) | GOLDEN PASS 524,288 | 101.1 ms |
| `simplegemm_dataflow.cc` (+ `aie::mmul`, `.tile`) | GOLDEN PASS 524,288 | 79.6 ms |
| `simpleconv2dkernel_port_less.cc` (scalar conv) | PASS 802,816 | 93.8 ms |
| `simpleconv2dkernel_port_opts.cc` (+ `.resident`) | PASS 802,816 | 85.8 ms |
| `simpleconv2dkernel_port_vec_opts.cc` (`aie::mmul` + options + `output_tile_major`) | PASS 802,816 | 56.4 ms |

A launch re-loads the kernel and re-programs the array, so these times are
mostly configuration; chapters 6.1 and 6.2 measure what each change buys.

## Maintenance

- Each chapter opens with its example's full source, generated from
  `example/tileprogram/ccode/`. After editing an example:
  `python3 tutorial/aieprogram/sync_code.py` (`--check` exits 1 if stale).
- Figures in `img/` come from `img/make_figures.py` (stdlib Python):
  `python3 tutorial/aieprogram/img/make_figures.py`.
