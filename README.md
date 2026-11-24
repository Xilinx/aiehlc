# AIEHLC

AIEHLC (AIE High-Level Compiler) compiles **one C++ file** -- a host `main()`
plus a `__global__` AIE kernel -- into an ELF that runs on an AMD Versal AI
Engine array: VCK190 (AIE), VEK280 (AIE-ML), VEK385 (AIE2PS); bare metal or Linux.

- **No DMA, routing or scheduling to declare.** You write what each core
  computes and which block of each tensor it reads and writes
  (`aie::load_buffer` / `aie::store_buffer`, in whole-tensor coordinates,
  indexed by `aie::program_id`) -- nothing else. The compiler runs the kernel
  on x86, records those data accesses, and derives from them the DMA buffer
  descriptors, stream-switch routing, ping-pong buffers, locks and launch
  schedule. No ADF graph, no PLIO, no hand-written buffer descriptors.
- **CUDA/RCOM-style host code.** Allocate, launch with `kernel<<<mesh>>>(...)`, check
  the result -- the runtime wraps the AIE driver and loads a pre-built hardware
  design (PDI), so no FPGA flow is involved.
- **Checked before the board.** Blocks outside a tensor, data used without
  being loaded, loops a DMA cannot stream, buffers that do not fit are build
  errors with a source line, not hangs; after every build, `flowcheck`
  re-checks that the stream routes deliver each dataflow to the right cores.

## Example: Conv2d on 16 tiles in parallel

A 7x7 stride-2 int8 convolution (224 x 224 x 3 -> 112 x 112 x 64, the ResNet
stem) on a 4 x 4 mesh of AIE tiles. Mesh row `pid_h` takes 28 output rows, mesh
column `pid_c` takes 16 of the 64 output channels; each tile computes 4 x 4
blocks of 7 x 28 x 16 outputs from their 19 x 61 x 4 input footprints. This is
the whole program -- there is no DMA, routing or buffer code to write:

<!-- code: example/tileprogram/ccode/simpleconv2dkernel_port_less.cc nocomments notimer nocheck devicelast; replace , *ref = (int8_t *)__Runtime_Alloc(out_n) => ; after init_conv2d_test_data(: kernel call to parallel compute in 4x4 tiles -->
```cpp
#include "simpleconv2d_new_load.h"
__global__ void conv2d_spatial(aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
                               aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
                               aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const int KH = 7, KW = 7, S = 2, C = 4;
    const int oh_t = 7, ow_t = 28, oc_t = 16;
    const int ih_t = (oh_t - 1) * S + KH;
    const int iw_t = (ow_t - 1) * S + KW;
    const int oh_core = 112 / aie::num_programs(0);
    const int pid_h = aie::program_id(0);
    const int pid_c = aie::program_id(1);
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});

    for (int i = 0; i < oh_core / oh_t; i++) {
        for (int j = 0; j < 112 / ow_t; j++) {
            const int oh0 = pid_h * oh_core + i * oh_t;
            const int ow0 = j * ow_t;
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});
            for (int oh = 0; oh < oh_t; oh++)
                for (int ow = 0; ow < ow_t; ow++)
                    for (int oc = 0; oc < oc_t; oc++) {
                        int acc = 0;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int c = 0; c < C; c++)
                                    acc += x[((oh * S + kh) * iw_t + ow * S + kw) * C + c] *
                                           w[((oc * KH + kh) * KW + kw) * C + c];
                        y[(oh * ow_t + ow) * oc_t + oc] = (int8_t)(acc > 127 ? 127 : acc < -128 ? -128 : acc);
                    }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
    aie::release(win_b);
}
int main() {
    const int in_n = INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN, w_n = K * NUM_FILTERS, out_n = M * N;
    int8_t *input = (int8_t *)__Runtime_Alloc(in_n), *filter = (int8_t *)__Runtime_Alloc(w_n);
    int8_t *output = (int8_t *)__Runtime_Alloc(out_n);
    memset(output, 0, out_n);
    init_conv2d_test_data(input, filter);
    // kernel call to parallel compute in 4x4 tiles
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);
    return 0;
}
```
<!-- /code -->

From `simpleconv2d_new_load.h`: the input is stored padded and channel-aligned
(`INPUT_H_PAD x INPUT_W_PAD x INPUT_C_ALIGN` = 230 x 230 x 4), `K x NUM_FILTERS`
= 196 x 64 filter bytes, `M x N` = 12,544 pixels x 64 channels of output, and
`HW_ROWS x HW_COLS` = 4 x 4. The launch passes `M, N, K` after the buffers:
the conv lowering reads its GEMM dimensions from them (keep them). The full file
([example/tileprogram/ccode/simpleconv2dkernel_port_less.cc](example/tileprogram/ccode/simpleconv2dkernel_port_less.cc))
also times the launch and checks every output against a CPU convolution --
on the board: `PASS: 0 mismatches out of 802816`. A line-by-line walkthrough
is [tutorial 1b](tutorial/aieprogram/01_contraction/conv2d.md).

## Quick start

Prerequisites: AMD Vitis (set `XILINX_VITIS`, `source $XILINX_VITIS/../settings64.sh`),
and for a hardware run a Versal board reachable by `script/test/apppaltest.py`
-- or use the AIE simulator. A pre-built `aiehlc` binary is used unless you
[build your own](doc/build.md).

```bash
source script/setup.sh          # once: builds the aie-rt driver headers, sets PATH
bash script/verify_env.sh       # checks Vitis, the cross compilers, LLVM, board settings

# compile the conv2d above (host ELF with the kernel embedded) into ./aout
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file ./example/tileprogram/ccode/simpleconv2dkernel_port_less.cc

# run it on the board
python3 script/test/apppaltest.py -y --netload ./aout/main.elf
#   [BENCH] aie_wall_ms=93.845
#   PASS: 0 mismatches out of 802816
```

Only the kernel analysis (what the compiler derives from your blocks), in about
a second and without a board: `./build/aiehlc-convtile trace <file.cc>`.
Other platforms (PetaLinux, AIE-ML, llvm-aie) and the simulator:
[doc/reference/build_and_run.md](doc/reference/build_and_run.md).

## Learn

| Start with | What it teaches |
|---|---|
| [tutorial/aieprogram](tutorial/aieprogram/README.md) | The programming model, by operator family: contraction (GEMM, conv2d), elementwise, reduction, data movement, indexing; then GEMM optimization step by step (scheduling options, vector instructions). Every example checks itself on the board. |
| [tutorial/aiednn](tutorial/aiednn/howto/templatekernel.md) | Writing a library-grade ML kernel for AIEDNN (a bf16 GEMM as a C++ template). |
| [tutorial/model](tutorial/model) | Whole models on the AIE: [ResNet18](tutorial/model/resnet18.md), [yolov26](tutorial/model/yolov26.md), [tiny SD UNet](tutorial/model/tiny-sdm-unet.md). |
| [doc/tutorial_aiegraph.md](doc/tutorial_aiegraph.md) | `aiegraph`: take a model from Python (ONNX / PyTorch) to int8 C, offload layers to AIE kernels, build the board ELF. |
| [doc/tutorial_userkernel.md](doc/tutorial_userkernel.md) | Plugging your own AIE kernel into one layer of a model. |
| [tutorial/tutorial.md](tutorial/tutorial.md) | The low-level driver API underneath: one kernel loaded, routed and run by hand. |

A model from Python, in short:

```bash
pip install -e .          # from this checkout, in a venv: aiegraph + the `aiegraph` / `aiehlc` commands
aiegraph doctor           # what works on this machine
```

```python
import aiegraph as ag
module = ag.convert(ag.load("resnet18"), target="a78")             # int8, one C file per layer
module.register_kernel(name="conv2dstem", op="conv2d",
                       source="src/aietensorop/conv2dstem/conv2dstem_uk.cc")
module.assign(region="conv1", backend="aie:conv2dstem")
module.build(target="a78", output="resnet18_a78")                  # main.elf
```

## How it works

![architecture: 6 MLIR dialects](doc/diagram/autorouting_ir_architecture.png)

1. **Kernel analysis.** The kernel body is compiled for x86 and run once per
   core; every `load_buffer` / `store_buffer` is recorded and each port's
   blocks are fitted to `base * program_id + step * loop index`. That fit is a
   DMA pattern: a block that depends only on the mesh row is broadcast along
   the row, one that depends only on the mesh column down the column, an output
   block of each core is gathered back.
2. **MLIR lowering**, one concern per dialect: `routing` (what goes to which
   tiles) -> `dmap` (logical streams) -> `dmaphop` (physical tiles, ports,
   paths) -> `dfscheblueprint` / `dfschedule` (DMA buffer descriptors, locks,
   launch order) and `routinghw` (stream-switch connections). Every stage
   dumps its IR, and you can feed an edited routing IR back in.
3. **Code generation**: EmitC writes `host.cc` (Arm), `kernel.cc` (each core)
   and `routing.cc` (switch setup); the AIE compiler builds the kernel, which
   is embedded in the host ELF.

Step by step, with the IR of a real build at every stage:
[tutorial 1a, section 5](tutorial/aieprogram/01_contraction/gemm.md#5-what-the-compiler-made-of-it).
The C front end: [doc/diagram/architecture.png](doc/diagram/architecture.png);
the passes: [doc/tilinglinalg.md](doc/tilinglinalg.md).

## Supported and limitations

| | |
|---|---|
| Devices | Versal AI Core / AI Edge: AIE (`--aie-version 1`), AIE-ML (`2`), AIE2PS (`5`). Not Ryzen AI. |
| Platforms | bare metal (APU, RPU) and PetaLinux; the AIE simulator |
| Kernel compilers | Synopsys `xchesscc` (default), llvm-aie (experimental) |
| I/O | GMIO (DDR <-> array) with the pre-built PDIs in [tutorial/hw](tutorial/hw). No PLIO: that needs your own hardware design. |
| Operators, end to end | contraction (GEMM, conv2d) |
| Operators, kernel analysis only | elementwise, reduction, data movement, indexing pass the analysis; their lowering is in progress ([status](tutorial/aieprogram/README.md)) |
| Not expressible | data-dependent addressing (gather / scatter by runtime index); per-core input blocks on a 2-D mesh (use a row / column broadcast) |

It is not a VLIW compiler (per-core code generation is Synopsys / llvm-aie) and
not a hardware design tool.

## Performance

A launch costs compute, data movement and control (configuring the array).
At these sizes **control dominates**: a register write over AXI-MM costs
~372 ns, the same write through a control packet ~2.9 ns
([measured](doc/performance/register_write_cost.md)). The tutorial examples
measure what each optimization buys
([GEMM 90.6 -> 79.6 ms](tutorial/aieprogram/06_gemm_optimization/2_vector_instructions.md),
[conv2d 93.8 -> 56.4 ms](tutorial/aieprogram/06_gemm_optimization/4_conv_vector_scheduling.md));
a conv2d configured over the control-packet plane runs one launch in 185 us
([conv2dstem](doc/performance/conv2dstem_perf_review.md)). More numbers:
[doc/reference/performance.md](doc/reference/performance.md).

## Cost model

[`src/costmodel`](src/costmodel/README.md) predicts, without a board, how long one AIE launch, one
site or a whole network takes on a calibrated board, split into AIE time (K), launch overhead (L) and
host glue (N): `T = K + L + N - overlap`. It reads a finished build (it does not change the compiler),
and its per-board parameters are frozen in `src/costmodel/params/` (calibrated on palmyra213).

```bash
cd src
python3 -m costmodel selftest                                      # offline check of the tools and frozen params
python3 -m costmodel estimate --kernel u_lin_1024x320x320           # one kernel (K only)
python3 -m costmodel estimate --net resnet18 --build <ROOT> --cfg off   # a whole network from its build dir
```

Step-by-step usage and runnable examples (single kernel, whole model, offline tuning proposal):
[src/costmodel/README.md](src/costmodel/README.md),
[src/costmodel/example](src/costmodel/example/README.md).

## Reference

| Topic | |
|---|---|
| Build and run: platforms, llvm-aie, simulator | [doc/reference/build_and_run.md](doc/reference/build_and_run.md) |
| `aiehlc.sh` build options | [doc/reference/build_options.md](doc/reference/build_options.md) |
| Runtime debug level (`#pragma aie_debug_level`) | [doc/reference/runtime_debug_level.md](doc/reference/runtime_debug_level.md) |
| Core trace (`#pragma aie_trace`) | [doc/reference/core_trace.md](doc/reference/core_trace.md) |
| Control-plane pragmas | [doc/reference/control_plane_pragmas.md](doc/reference/control_plane_pragmas.md), [doc/controlplane.md](doc/controlplane.md) |
| Kernel config offload (`#pragma KERNELCONFIGOFFLOAD`) | [doc/reference/kernel_config_offload_pragma.md](doc/reference/kernel_config_offload_pragma.md) |
| Debug tools: `aiediag`, the live schedule view / debug UI, `--prettydebug` | [doc/reference/debug_tools.md](doc/reference/debug_tools.md) |
| Building aiehlc itself | [doc/build.md](doc/build.md) |
| Design documents | [doc/](doc), [CLAUDE.md](CLAUDE.md) (architecture overview) |

## Contributing

Set up the clang-format pre-commit hook before you contribute:

```bash
source script/precommitsetup.sh
```

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

<p align="center">Copyright&copy; 2025-2026 Advanced Micro Devices, Inc.</p>
