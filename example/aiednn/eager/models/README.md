# Eager mode: the benchmark models on the AI Engine, one kernel instance per layer shape

aiednn has two ways to run a model on the AIE:

- **Fixed-megakernel mode** (the default). One persistent GEMM megakernel (`src/aietensorop/gemm/gemm_uk.cc` + its
  `aie_modes_*.h`) with fixed 128×128×512 blocks. The host loops every layer over those blocks, and the kernel is
  loaded once.
- **Eager mode** (`dnn_build(mode="eager")`, `run_models.py --eager`, `run.py` here). Every layer runs on an
  **instance of a C++ template kernel**. Its shapes, tiling, operand layout and data types are template arguments
  chosen for that layer. A launch loads the instance's kernel, configures it, runs it and waits; the next layer
  may load a different one.

This directory runs the seven benchmark models in eager mode on the workstation or on a VEK280 board. One page per
model shows the real code at every layer, taken from the board builds. The build directories they link to live
under `worklocal/` on the build machine:

| Model | Page | Primitives | Instances (8×4) | 4×4 | 8×4 | 4×4 warm | 8×4 warm |
|---|---|---|---|---|---|---|---|
| gpt2_small (12 blocks, 128 tokens) | [gpt2_small.md](gpt2_small.md) | 109 | 4 | 6.31 s | 7.52 s | 3.77 s | **2.54 s** |
| tiny_llama (RMSNorm, RoPE, GQA) | [tiny_llama.md](tiny_llama.md) | 49 | 5 | 1.12 s | 1.51 s | | |
| bert_base (12 layers) | [bert_base.md](bert_base.md) | 111 | 4 | 6.36 s | 7.60 s | | |
| resnet50 (ImageNet) | [resnet50.md](resnet50.md) | 58 | 25 | 2.21 s | 2.59 s | 1.50 s | **1.23 s** |
| resnet18 (ImageNet) | [resnet18.md](resnet18.md) | 25 | 19 | 0.79 s | 0.84 s | | |
| mobilenet_v2 (ImageNet) | [mobilenet_v2.md](mobilenet_v2.md) | 56 | 24 | FAIL | FAIL | 1.42 s | **0.97 s** |
| yolov8n (COCO, 320×320) | [yolov8n.md](yolov8n.md) | 150 | 40 | 1.42 s | 1.63 s | | |

All runs are int8 on VEK280 (galerina13). Every run except mobilenet_v2's two baselines PASSES against PyTorch,
with every int8 GEMM checked exact against the CPU reference implementation. mobilenet_v2's baselines ran before
the `gemm_q` N padding fix: two 24-channel layers were wrong, and [mobilenet_v2.md](mobilenet_v2.md) shows the
mismatch. *Warm* is warm launch (below), and the empty cells were not run with it yet. [models.md](models.md) is the generated index with every run,
and `gen_pages.py` regenerates all pages from the build directories.

## Two ways to use eager mode

| | Per-app build (`dnn_build(mode="eager")`, `run.py`, `run_models.py --eager`) | Prebuilt library, cuDNN-style (`dnn-lib --eager`) |
|---|---|---|
| AIE kernels | instances made for **this model's** layer shapes (or `--library`), compiled with the app | the premade 74-instance library, compiled **once per mesh** into `libaiednn.a` |
| App build | aiehlc builds app + runtime + kernels together | plain `aarch64-none-elf-g++` against `aiednn.h`: no aiehlc, no kernel source |
| Kernel choice | the planner picks the instances | at `aiednn_*_create`, dispatch picks among the linked `eager:*` implementations |
| Best for | a fixed model, fewest launches, smallest ELF | any app / framework, no AIE toolchain on the app side |

The prebuilt library is used the way an app uses cuDNN:

```bash
python -m aiegraph dnn-lib --eager --eager-mesh 8x4 --vitis-settings <.../Vitis/settings64.sh>
#   -> worklocal/aiednn/lib_vek280_eager_8x4/{include/aiednn.h, lib/libaiednn.a, link.mk}
make -C example/aiednn/prebuilt DIST=$PWD/worklocal/aiednn/lib_vek280_eager_8x4   # the same app as fixed mode
```

```c
aiednn_memory_desc_init(&ma, 2, (const int64_t[]){M, K}, aiednn_s8);           // describe tensors
aiednn_matmul_forward_create(&p, &ma, &mw, NULL, &mc, 1);                       // create = the query
aiednn_impl_perf_t perf[8];
int n = aiednn_primitive_find(p, AIEDNN_FIND_MEASURE, perf, 8);                 // cudnnFind*: time every candidate
aiednn_primitive_set_impl(p, perf[0].impl);                                     // pin one ("eager:mm", ...)
size_t ws = aiednn_primitive_get_workspace_size(p);                             // caller workspace
aiednn_primitive_execute_async(p, stream, nargs, args, workspace, ws);          // stream execute
```

The API is the same in both ways and in fixed-megakernel mode. `include/aiednn_graph.h` adds the graph API
(partitions, compile, execute), the counterpart of cuDNN's graph / fusion API.

## Running it

```bash
python example/aiednn/eager/models/run.py --model gpt2_small --show      # workstation emulation + the plan and code
python example/aiednn/eager/models/run.py --model gpt2_small --mesh 8x4 --target vek280 \
    --board galerina13 --vitis-settings <.../Vitis/settings64.sh>          # build the ELF and run it on the board
# the benchmark driver, same steps:
python example/aiednn/models/run_models.py --models gpt2_small --modes int8 --app vek280 --board galerina13 \
    --eager --eager-mesh 8x4 --mlir
```

`run.py` is short on purpose. Its five numbered steps are the five layers of code below.

## The layers of code

**1. The PyTorch model.** These are ordinary `nn.Module`s from [`../../models/zoo.py`](../../models/zoo.py): torchvision
resnet18 / resnet50 / mobilenet_v2 with ImageNet weights on a real photo, and the transformer classes `GPT2Small`,
`TinyLlama`, `BertBase`, plus ultralytics yolov8n. Nothing in them is AIE-specific.

**2. aiednn primitives.** `dnn_torch.references` traces the model, quantizes it to int8 (calibrated on the input) and
runs the aiednn MLIR pipeline. The pipeline does epilogue fusion, folds requantization into GEMMs, folds the
attention head permutes and plans memory. `emit_app` writes the result as a C program that calls only
`include/aiednn.h`: per layer one `aiednn_*_create` with exact shapes and a fused epilogue, then
`aiednn_primitive_execute`. gpt2_small becomes 109 such calls, for example:

```c
st = aiednn_matmul_forward_create_attr(&p, &m0, &m1, &m2, &m3, 1, &c5);   // int8 x int8 -> int8, scales + bias
st = aiednn_primitive_execute_args(p, 4, a);                                // dispatch picked eager:mmq
```

This program is the same in both modes. The mode only changes which implementations the library links.

**3. The eager plan.** `dnn_eager.plan(g, model, mesh_rows)` maps every layer to a template instance:

- GEMM-family layers (matmul, conv as im2col, inner product) go to `gemm_q` when the requantize can run on the core,
  `gemm_qr` with an int8 residual add, otherwise `gemm_s8` (int32 out, host epilogue).
- The other ops go to their own templates: `lnorm`, `rope`, `pool`, `dwconv_q` / `dwconv_s8`, `lut` / `lutw`,
  `steps`, `quant`, `frow`, `feltw` / `fbinary` / `faffine`, `bin8`, `red8`, `gemm_bf16`, `gemm_wq`.
- Layers that need the same arguments share one instance, so gpt2_small's 97 AIE layers need 4 instances and yolov8n's 150 primitives need 40.

Two board rules shape the arguments. A weight block must not exceed its A block, so an instance is at most 4·TM
outputs wide and a wider layer runs in slices. A core's 8-bit output row must be whole 32-bit words, so `gemm_q`
pads N to a multiple of 16. `--library` uses the premade [`dnn_eager.LIBRARY`](../../../../src/aiegraph/dnn_eager.py)
(74 instances) instead, so one ELF serves any model.

**4. The instance entry (generated).** `dnn_eager.entry_source(plan)` writes `eager_instances.cc`. Per instance it
has a launch function with the template call, plus the `AIEK_ENTRY` / `AIEK_SPEC` / `AIEK_REFERENCE` that put it
in the aiednn implementation list. The spec carries the instance's exact operand shapes, and the reference is the
x86 emulation:

```c
#define HW_ROWS 8 /* the mesh: the kernels' row split (eager_common.h) */
#include "gemm_q_eager_kernel.cc"
static void gemm_q__128_64_768_256_16_launch(int8_t *A, int8_t *W, int8_t *C) {
    aieSetDevice(0);
    aieArray device;
    __Runtime_partition_reuse_next();               // warm launch: keep the partition
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_q<128, 64, 768, 256, 16><<<mesh>>>(A, W, C);
}
AIEK_ENTRY(eager_gemm_q__128_64_768_256_16) { return eager_gemm_q_entry(args, num_args, 128, 64, 768, 256, ...); }
```

aiehlc compiles every launched template instantiation into its own kernel ELF and routing, all in one host ELF.

**5. The template kernel.** These are hand-written in [`src/aietensorop/eager/`](../../../../src/aietensorop/eager/) (every one is linked
under [Kernel sources](#kernel-sources)) in
aiehlc's kernel-only model. Ports carry their tensor shape. A core asks for blocks in tensor coordinates
(`aie::load_buffer` / `store_buffer`) and finds its place with `aie::program_id`. aiehlc derives the DMA from the
traced block accesses:

```c++
template <int M, int N, int K, int TK = 64, int TM = 16>
__global__ void gemm_q(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K + TK>> a,
                       aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K + TK>> w,
                       aie::port<output_window_int8 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4, ntile = M / (HW_ROWS * TM), nkc = K / TK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);   // mesh row: rows of A; mesh column: N / 4 of C
    ...
        const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
        const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
    ...                                                            // the last W chunk: 8 requant words per channel
```

Each template's `*_eager_ref.h` holds the same arithmetic as the AIEK_REFERENCE, and
`example/aiednn/eager/<template>_eager.cc` is its board self-check: every template has passed on galerina13.

**At run time.** `aiednn_*_create` dispatches over the implementation list. An eager implementation such as
`eager:mmq` ([`src/aiednn/aiednn_rt_eager.h`](../../../../src/aiednn/aiednn_rt_eager.h)) accepts a layer when an
instance covers it. On execute it packs the operands into the instance's layout, calls the instance's
`AIEK_ENTRY` once per tile / slice / chunk (each call is one launch), and unpacks.

## Kernel sources

Every eager template, with links:
- **Kernel:** the AIE kernel template.
- **Reference:** the same arithmetic in plain C++, used as the AIEK_REFERENCE in workstation emulation and as the
  board check's golden.
- **Board self-check:** an app with two instances of different shapes and a PASS/FAIL check. Each has passed on
  galerina13.
- **Runtime implementation:** the code that accepts a primitive, packs its operands into the instance's layout and
  launches the instance.
- **Benchmark models:** which of the seven use the template, from their 8×4 plans.

Shared pieces:
- [eager_common.h](../../../../src/aietensorop/eager/eager_common.h): host staging, mesh and partition.
- [eager_fmath_ops.cc](../../../../src/aietensorop/eager/eager_fmath_ops.cc): the f32 math, spliced into the kernel text.
- [aiednn_rt.cc](../../../../src/aiednn/aiednn_rt.cc): the `eager:*` registrations.
- [dnn_eager.py](../../../../src/aiegraph/dnn_eager.py): the planner, templates, library and entry generator.

| Template | Kernel | Reference | Board self-check | Runtime implementation | Benchmark models |
|---|---|---|---|---|---|
| `gemm_s8` | [gemm_s8_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_s8_eager_kernel.cc) | [gemm_s8_eager_ref.h](../../../../src/aietensorop/eager/gemm_s8_eager_ref.h) | [gemm_s8_eager.cc](../gemm_s8_eager.cc) | eager:mm (any int8 GEMM; host epilogue): `acc_eager_mm` + `run_blocked` in [aiednn_rt_gemm.h](../../../../src/aiednn/aiednn_rt_gemm.h) in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | all 7 |
| `gemm_q` | [gemm_q_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_q_eager_kernel.cc) | [gemm_q_eager_ref.h](../../../../src/aietensorop/eager/gemm_q_eager_ref.h) | [gemm_q_eager.cc](../gemm_q_eager.cc) | eager:mmq, eager:convq (requantize on the core): `run_eager_mmq`, `run_eager_convq` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | gpt2_small, tiny_llama, bert_base, resnet18, resnet50, mobilenet_v2 |
| `gemm_qr` | [gemm_qr_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_qr_eager_kernel.cc) | [gemm_qr_eager_ref.h](../../../../src/aietensorop/eager/gemm_qr_eager_ref.h) | [gemm_qr_eager.cc](../gemm_qr_eager.cc) | eager:mmq, eager:convq (+ int8 residual add): `run_eager_mmq`, `run_eager_convq` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | resnet18, resnet50, mobilenet_v2 |
| `dwconv_q` | [dwconv_q_eager_kernel.cc](../../../../src/aietensorop/eager/dwconv_q_eager_kernel.cc) | [dwconv_q_eager_ref.h](../../../../src/aietensorop/eager/dwconv_q_eager_ref.h) | [dwconv_q_eager.cc](../dwconv_q_eager.cc) | eager:dwq (depthwise + requantize): `run_eager_dwq` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | mobilenet_v2 |
| `dwconv_s8` | [dwconv_s8_eager_kernel.cc](../../../../src/aietensorop/eager/dwconv_s8_eager_kernel.cc) | [dwconv_s8_eager_ref.h](../../../../src/aietensorop/eager/dwconv_s8_eager_ref.h) | [dwconv_s8_eager.cc](../dwconv_s8_eager.cc) | eager:dw (depthwise, int32 + host epilogue): `run_eager_dw` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | - |
| `pool` | [pool_eager_kernel.cc](../../../../src/aietensorop/eager/pool_eager_kernel.cc) | [pool_eager_ref.h](../../../../src/aietensorop/eager/pool_eager_ref.h) | [pool_eager.cc](../pool_eager.cc) | eager:pool (8-bit max / average): `run_eager_pool` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | resnet18, resnet50, yolov8n |
| `quant` | [quant_eager_kernel.cc](../../../../src/aietensorop/eager/quant_eager_kernel.cc) | [quant_eager_ref.h](../../../../src/aietensorop/eager/quant_eager_ref.h) | [quant_eager.cc](../quant_eager.cc) | eager:quant (f32 -> s8 / u8): `run_eager_quant` in [aiednn_rt_eager.h](../../../../src/aiednn/aiednn_rt_eager.h) | resnet18, resnet50, mobilenet_v2, yolov8n |
| `lnorm` | [lnorm_eager_kernel.cc](../../../../src/aietensorop/eager/lnorm_eager_kernel.cc) | [lnorm_eager_ref.h](../../../../src/aietensorop/eager/lnorm_eager_ref.h) | [lnorm_eager.cc](../lnorm_eager.cc) | eager:lnorm (layer / RMS norm): `run_eager_lnorm` in [aiednn_rt_eager_norm.h](../../../../src/aiednn/aiednn_rt_eager_norm.h) | gpt2_small, tiny_llama, bert_base |
| `rope` | [rope_eager_kernel.cc](../../../../src/aietensorop/eager/rope_eager_kernel.cc) | [rope_eager_ref.h](../../../../src/aietensorop/eager/rope_eager_ref.h) | [rope_eager.cc](../rope_eager.cc) | eager:rope: `run_eager_rope` in [aiednn_rt_eager_norm.h](../../../../src/aiednn/aiednn_rt_eager_norm.h) | tiny_llama |
| `frow` | [frow_eager_kernel.cc](../../../../src/aietensorop/eager/frow_eager_kernel.cc) | [frow_eager_ref.h](../../../../src/aietensorop/eager/frow_eager_ref.h) | [frow_eager.cc](../frow_eager.cc) | eager:frow (f32 softmax, reductions, pooling): `run_eager_frow` in [aiednn_rt_eager_fvec.h](../../../../src/aiednn/aiednn_rt_eager_fvec.h) | resnet18, resnet50, mobilenet_v2, yolov8n |
| `feltw` | [feltw_eager_kernel.cc](../../../../src/aietensorop/eager/feltw_eager_kernel.cc) | [felt_eager_ref.h](../../../../src/aietensorop/eager/felt_eager_ref.h) | [felt_eager.cc](../felt_eager.cc) | eager:feltw (f32 eltwise): `run_eager_feltw` in [aiednn_rt_eager_fvec.h](../../../../src/aiednn/aiednn_rt_eager_fvec.h) | yolov8n |
| `fbinary` | [fbinary_eager_kernel.cc](../../../../src/aietensorop/eager/fbinary_eager_kernel.cc) | [felt_eager_ref.h](../../../../src/aietensorop/eager/felt_eager_ref.h) | [felt_eager.cc](../felt_eager.cc) | eager:fbinary (f32 binary, per-element PReLU): `run_eager_fbinary` in [aiednn_rt_eager_fvec.h](../../../../src/aiednn/aiednn_rt_eager_fvec.h) | yolov8n |
| `faffine` | [faffine_eager_kernel.cc](../../../../src/aietensorop/eager/faffine_eager_kernel.cc) | [felt_eager_ref.h](../../../../src/aietensorop/eager/felt_eager_ref.h) | [faffine_eager.cc](../faffine_eager.cc) | eager:faffine (f32 batch norm, PReLU): `run_eager_faffine` in [aiednn_rt_eager_fvec.h](../../../../src/aiednn/aiednn_rt_eager_fvec.h) | - |
| `bin8` | [bin8_eager_kernel.cc](../../../../src/aietensorop/eager/bin8_eager_kernel.cc) | [bin8_eager_ref.h](../../../../src/aietensorop/eager/bin8_eager_ref.h) | [bin8_eager.cc](../bin8_eager.cc) | eager:bin8 (two 8-bit tensors, quantized binary): `run_eager_bin8` in [aiednn_rt_eager_fvec.h](../../../../src/aiednn/aiednn_rt_eager_fvec.h) | - |
| `lut` | [lut_eager_kernel.cc](../../../../src/aietensorop/eager/lut_eager_kernel.cc) | [lut_eager_ref.h](../../../../src/aietensorop/eager/lut_eager_ref.h) | [lut_eager.cc](../lut_eager.cc) | eager:lut (any 8-bit per-channel function): `run_eager_lut` in [aiednn_rt_eager_lut.h](../../../../src/aiednn/aiednn_rt_eager_lut.h) | - |
| `lutw` | [lutw_eager_kernel.cc](../../../../src/aietensorop/eager/lutw_eager_kernel.cc) | [lutw_eager_ref.h](../../../../src/aietensorop/eager/lutw_eager_ref.h) | [lutsteps_eager.cc](../lutsteps_eager.cc) | eager:lut with a 16- / 32-bit dst: `run_eager_lut` in [aiednn_rt_eager_lut.h](../../../../src/aiednn/aiednn_rt_eager_lut.h) | - |
| `steps` | [steps_eager_kernel.cc](../../../../src/aietensorop/eager/steps_eager_kernel.cc) | [steps_eager_ref.h](../../../../src/aietensorop/eager/steps_eager_ref.h) | [lutsteps_eager.cc](../lutsteps_eager.cc) | eager:steps (f32 / s32 -> 8-bit monotone ops): `run_eager_steps` in [aiednn_rt_eager_lut.h](../../../../src/aiednn/aiednn_rt_eager_lut.h) | - |
| `red8` | [red8_eager_kernel.cc](../../../../src/aietensorop/eager/red8_eager_kernel.cc) | [red8_eager_ref.h](../../../../src/aietensorop/eager/red8_eager_ref.h) | [red8_eager.cc](../red8_eager.cc) | eager:red8 (8-bit reductions): `run_eager_red8` in [aiednn_rt_eager_lut.h](../../../../src/aiednn/aiednn_rt_eager_lut.h) | - |
| `gemm_bf16` | [gemm_bf16_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_bf16_eager_kernel.cc) | [gemm_bf16_eager_ref.h](../../../../src/aietensorop/eager/gemm_bf16_eager_ref.h) | [gemm_bf16_eager.cc](../gemm_bf16_eager.cc) | eager:bf16, eager:bf16x3 (float GEMM + f32 composites): `run_eager_bf16` in [aiednn_rt_eager_float.h](../../../../src/aiednn/aiednn_rt_eager_float.h) | - |
| `gemm_wq` | [gemm_wq_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_wq_eager_kernel.cc) | [gemm_wq_eager_ref.h](../../../../src/aietensorop/eager/gemm_wq_eager_ref.h) | [gemm_wq_eager.cc](../gemm_wq_eager.cc) | eager:wq (s8 x 4-bit grouped weights): `run_eager_wq` in [aiednn_rt_eager_float.h](../../../../src/aiednn/aiednn_rt_eager_float.h) | - |

### Example: the kernels gpt2_small runs

gpt2_small's plan ([gpt2_small.md](gpt2_small.md)) has four instances of three templates. Attention
(`gemm+sdpa8`) runs its two int8 GEMMs per head on the same `gemm_s8` instances through `eager:mm`.

| Instance (8×4) | Layers | Template | What runs on the core |
|---|---|---|---|
| `gemm_q<128, 64, 768, 256, 16>` | 36 q / k / v projections | [gemm_q_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_q_eager_kernel.cc) | int8 GEMM + fixed-point requantize to int8 |
| `gemm_s8<128, 64, 768, 256, 16>` | 24 attention-out + MLP-up | [gemm_s8_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_s8_eager_kernel.cc) | int8 GEMM, int32 out (host: scales, bias, GELU) |
| `gemm_s8<128, 64, 3072, 256, 16>` | 12 MLP-down | [gemm_s8_eager_kernel.cc](../../../../src/aietensorop/eager/gemm_s8_eager_kernel.cc) | same, K = 3072 |
| `lnorm<128, 768, 2, 0>` | 25 layer norms | [lnorm_eager_kernel.cc](../../../../src/aietensorop/eager/lnorm_eager_kernel.cc) | f32 layer norm rows (Kahan sums) |

On an 8×4 mesh, mesh row `pm = aie::program_id(0)` takes M/8 rows, and mesh column `pn = aie::program_id(1)` takes
N/4 = 16 output columns. `gemm_s8` is the whole GEMM:

```c++
template <int M, int N, int K, int TK = 64, int TM = 16>
__global__ void gemm_s8(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K>> w,
                        aie::port<output_window_int32 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4;
    const int ntile = M / (HW_ROWS * TM);
    const int nkc = K / TK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int32_t acc[TM * TN];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0;
        for (int j = 0; j < nkc; j++) {                       // K streamed in TK chunks, accumulated in int32
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++) {
                    int32_t s = 0;
                    for (int k = 0; k < TK; k++)
                        s += (int32_t)x[m * TK + k] * (int32_t)v[n * TK + k];
                    acc[m * TN + n] += s;
                }
            aie::release(a);
            aie::release(w);
        }
        int32_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = acc[i];
        aie::commit(c);
    }
}
```

`gemm_q` has the same loop, with one extra K chunk. Its A part is zero, and its W part holds each output channel's 8
requant words (multiplier, 64-bit bias, shift, zero point, clamp). After the K loop each core requantizes its tile:

```c++
        for (int m = 0; m < TM; m++)
            for (int n = 0; n < TN; n++) {
                const int32_t *p = qw + n * 8;
                const int64_t B = (int64_t)(((uint64_t)(uint32_t)p[2] << 32) | (uint32_t)p[1]);
                const int64_t r = (((int64_t)acc[m * TN + n] * (int64_t)p[0] + B) >> p[3]) + p[4];
                y[m * TN + n] = (int8_t)(r < p[5] ? p[5] : r > p[6] ? p[6] : r);
            }
```

`lnorm` is a row op. Each mesh row takes R/8 rows, every core of that row reads the whole row strip, and core `pn`
writes its quarter of the columns. gamma, beta, eps and the row length come in a `.resident` input that is loaded
once:

```c++
template <int R, int L, int TR, int RMS>
__global__ void lnorm(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 4 * L>> x,
                      aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<12, L>> p,
                      aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> y) {
    const int rows = R / HW_ROWS, q = L / 4;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    const int8_t *pp = aie::load_buffer(p, {3 * pn, 0}, {3, L}, {.resident = true});
    ...
        const int8_t *a = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, 4 * L});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * L}, {TR, L});
        // per row: mean and variance (Kahan sums), 1/sqrt(var + eps) by Newton steps, then this core's columns
```

The host side of the same GEMMs (pack A / W into the instance layout, one launch per tile, unpack and epilogue) is
`run_blocked` for `eager:mm` and `run_eager_mmq` for `eager:mmq`.

## Mesh size and warm launch

The templates split rows over `HW_ROWS` mesh rows: 4 (16 cores, partition `{0,8,0,7}`) or 8 (all 8 VEK280 core
rows, 32 cores, `{0,16,0,10}`). The columns stay 4.

Without warm launch, 8×4 was slower for every model: about 4.2 vs 3.3 ms per GEMM launch on gpt2_small. Each launch
set the partition up again, and the setup scales with the core count, while the compute per launch is small.
Two problems were behind it (found by a code investigation of the generated host code):

1. **Two partition re-initializations per launch.** `partition()` passed the first kernel's routing and `__aie_launch`
   its own, so every launch of another instance tore the partition down and set it up twice. The fix is
   `__Runtime_partition_reuse_next()` before `partition()`: a relaunch of the same instance re-initializes nothing.
2. **Full kernel reload.** Every launch reloaded the whole kernel ELF, 88 % of it `.bss` zeros. Eager builds now
   pass `--warm-launch`.

With both, a GEMM launch costs 1.67 ms on 4×4 and **1.09 ms** on 8×4, so the bigger mesh wins:

| gpt2_small, per launch | 4×4 | 8×4 | 4×4 warm | 8×4 warm |
|---|---|---|---|---|
| GEMM, `eager:mm` (1,296 launches) | 3.30 ms | 4.19 ms | 1.67 ms | **1.09 ms** |
| GEMM + requantize, `eager:mmq` (432) | 2.28 ms | 3.03 ms | 1.38 ms | **0.94 ms** |
| layer norm, `eager:lnorm` (25) | 26.9 ms | 16.6 ms | 25.7 ms | **14.2 ms** |
| **model** | 6.31 s | 7.52 s | 3.77 s | **2.54 s** |

Fixed-megakernel mode is still faster on most models. It loads one kernel once and launches large blocks.
Eager mode's remaining per-launch cost is DMA and lock programming, and launches stay small because an instance is
at most 64 outputs wide. Those are the next things to work on.
