# Tutorial 1 -- GEMM on a 4x4 tile mesh

Example: [`example/tileprogram/ccode/simplegemm_kernel_port.cc`](../../example/tileprogram/ccode/simplegemm_kernel_port.cc)
(56 lines, host included). Shared host helpers: `simplegemm_load.h`.

## The whole program

This is the whole program -- kernel and host. There is no DMA, routing or buffer code
anywhere: the compiler derives all of it from the kernel's `load_buffer` / `store_buffer`
calls. The rest of this tutorial explains it line by line.

<!-- code: example/tileprogram/ccode/simplegemm_kernel_port.cc -->
```cpp
/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Kernel-only GEMM: the DMA is derived from the kernel.
 *   C[64, 64] = A[64, 256] x B[64, 256]^T, int8, on the 4x4 mesh
 ******************************************************************************/
#include "simplegemm_load.h"
__global__ void gemm(aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 256>> a,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 256>> b,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<64, 64>> c) {
    const int tm = 16, tn = 16, tk = 64; // C block per core, K chunk per round
    const int pm = aie::program_id(0);   // mesh row: which 16 rows of C
    const int pn = aie::program_id(1);   // mesh col: which 16 columns of C
    int8_t *y = aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
    int acc[tm * tn];
    for (int i = 0; i < tm * tn; i++)
        acc[i] = 0;
    for (int j = 0; j < 256 / tk; j++) { // K chunks
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
        for (int m = 0; m < tm; m++)
            for (int n = 0; n < tn; n++) {
                int s = 0;
                for (int k = 0; k < tk; k++)
                    s += x[m * tk + k] * w[n * tk + k];
                acc[m * tn + n] += s;
            }
        aie::release(a);
        aie::release(b);
    }
    for (int i = 0; i < tm * tn; i++)
        y[i] = (int8_t)acc[i];
    aie::commit(c);
}

int main() {
    printf("=== Kernel-only GEMM (DMA from the kernel's blocks) on AIE %dx%d ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] x B^T[%dx%d], int8\n", GEMM_M, GEMM_N, GEMM_M, GEMM_K, GEMM_K, GEMM_N);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GEMM_N * GEMM_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_N);
    gemm_init(A, B, C);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm<<<mesh>>>(A, B, C);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    const int mismatches = gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
    return mismatches ? 1 : 0;
}
```
<!-- /code -->

```
C[64, 64] = A[64, 256] x B[64, 256]^T        int8 in, int32 accumulate, int8 out
```

You will learn the whole kernel-only model in one small file:

- a **port** declares a tensor the kernel reads or writes;
- `aie::program_id(0/1)` says **which core** of the mesh is running;
- `aie::load_buffer` / `aie::store_buffer` name, in tensor coordinates, the
  **block** this core uses now; `aie::release` / `aie::commit` hand it back;
- the compiler **runs your kernel on x86 for every core**, records the blocks,
  and turns them into DMA programs and stream routes.

## 1. The idea: one output block per core

The AIE array we use is a 4 x 4 mesh of cores. Split C into 4 x 4 blocks of
16 x 16 and give one block to each core. Core `(pm, pn)` computes

```
C[16pm .. 16pm+15, 16pn .. 16pn+15] = sum over j of  A[16pm.., 64j..] x B[16pn.., 64j..]^T
```

It needs a 16-row band of A and a 16-row band of B. Both bands are 16 x 256 =
4 KB, too much to stream as one piece next to everything else, so the core walks
K in **4 chunks of 64**: round `j` brings a `{16, 64}` block of A and of B.

![GEMM tiling and the pid of each block](img/gemm_tiling.svg)

Two facts fall out of this picture, and the compiler finds both by itself:

- every core of mesh **row** `pm` uses the same A blocks, so A can be sent once
  per mesh row and **broadcast** along it;
- every core of mesh **column** `pn` uses the same B blocks: one stream per
  mesh column, broadcast down it.

## 2. The kernel, line by line

```cpp
#include "simplegemm_load.h"
```
Host-side helpers only: the sizes (`GEMM_M/N/K`, `HW_ROWS/COLS` = 4 x 4),
`gemm_init()` (deterministic test data) and `gemm_verify()` (CPU GEMM + compare).
The kernel uses none of it -- aiehlc splits the file into a host part and a
kernel part, and the kernel part sees only the `__global__` function.

```cpp
__global__ void gemm(aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 256>> a,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 256>> b,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<64, 64>> c) {
```
`__global__` marks the function that runs on **every core** of the mesh.
Each parameter is a **port**: `aie::port<WindowType, Role, aie::shape<...>>`.

| Part | Meaning |
|---|---|
| `input_window_int8 *` / `output_window_int8 *` | direction and element type: the core *receives* blocks of an input, *produces* blocks of an output. Always a pointer. |
| `aie::IFM` / `aie::WEIGHTS` / `aie::OFM` | role tag. Optional: `aie::port<T, aie::shape<64, 256>> a` works too (output -> OFM, an input broadcast along mesh columns -> WEIGHTS, other inputs -> IFM). It names the generated descriptor; it does not change the DMA. |
| `aie::shape<64, 256>` | the **whole tensor** as it lies in DDR (row-major). Block offsets below are in these coordinates. |

At the call site the ports are plain host pointers: `gemm<<<mesh>>>(A, B, C)`.

```cpp
    const int tm = 16, tn = 16, tk = 64; // C block per core, K chunk per round
```
The tiling is ordinary C++ in the kernel: a 16 x 16 C block per core, K in
chunks of 64. Change these numbers and the DMA changes with them.

```cpp
    const int pm = aie::program_id(0);   // mesh row: which 16 rows of C
    const int pn = aie::program_id(1);   // mesh col: which 16 columns of C
```
`program_id(0)` is this core's **mesh row** (0..3), `program_id(1)` its **mesh
column** (0..3) -- Triton's `tl.program_id`. On the core they compile to the
core's hardware id minus the first core of the placement (generated code:
`pm = (get_coreid() & 0x1F) - 3`, `pn = (get_coreid() >> 16) - 3`). See
section 4 for which physical tile is which pid.

```cpp
    int8_t *y = aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
```
"This core writes the `{16, 16}` block of C at row `16pm`, column `16pn`."
`store_buffer(port, offset, shape)` takes the offset and shape **in tensor
coordinates** and returns a pointer to a core-local buffer of exactly that
shape (row-major, `y[m * 16 + n]`). The compiler made the DMA that will carry
this buffer to `C` in DDR when you `commit`.

```cpp
    int acc[tm * tn];
    for (int i = 0; i < tm * tn; i++)
        acc[i] = 0;
```
A 16 x 16 int32 accumulator in the core's local data memory (1 KB). It stays
there for all 4 K rounds; only A and B blocks are streamed.

```cpp
    for (int j = 0; j < 256 / tk; j++) { // K chunks
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
```
Round `j` (4 rounds): the `{16, 64}` block of A at row `16pm`, column `64j`,
and the matching block of B at row `16pn`. `load_buffer` waits until the DMA
has filled the core's buffer and returns it (`x[m * 64 + k]`). On the device
a load **does not fetch** anything: the DMA streams the blocks in the order
the compiler derived from your loop, and `load_buffer` takes the next one. That
is why the loops must be regular -- offsets affine in `program_id` and the
loop indices, one block shape per port.

```cpp
        for (int m = 0; m < tm; m++)
            for (int n = 0; n < tn; n++) {
                int s = 0;
                for (int k = 0; k < tk; k++)
                    s += x[m * tk + k] * w[n * tk + k];
                acc[m * tn + n] += s;
            }
```
Plain scalar C++: a 16 x 16 x 64 matrix product on the two blocks, B used
transposed (`w[n * tk + k]` is row `n` of B). Tutorial 4 replaces loops like
this with `aie::mmul`.

```cpp
        aie::release(a);
        aie::release(b);
    }
```
"I am done with this round's A and B blocks." The buffer goes back to the DMA,
which can refill it with round `j + 1`'s block. Every port is **double
buffered** by default (ping-pong), so round `j + 1` is usually already in the
other buffer while you compute round `j`.

```cpp
    for (int i = 0; i < tm * tn; i++)
        y[i] = (int8_t)acc[i];
    aie::commit(c);
}
```
Write the result into the output block and `commit` it: the DMA sends the
16 x 16 block to `C[16pm.., 16pn..]` in DDR.

![One core over its 4 rounds](img/gemm_rounds.svg)

## 3. The host, line by line

```cpp
int main() {
    printf("=== Kernel-only GEMM (DMA from the kernel's blocks) on AIE %dx%d ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] x B^T[%dx%d], int8\n", GEMM_M, GEMM_N, GEMM_M, GEMM_K, GEMM_K, GEMM_N);
```
Runs on the Arm (A78) next to the array; ordinary C++.

```cpp
    aieSetDevice(0);
    aieArray device;
```
Select the device and create the array object (`aieSetDevice` is a no-op kept
for CUDA-like source compatibility).

```cpp
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
```
Reserve a **partition** and describe the mesh the kernel runs on. The first
argument is `{startCol, endCol, startRow, endRow}` of the array: columns 3..6,
four columns, which the runtime initializes (`__Runtime_init_mesh_partition_with`)
and programs with this app's stream routing. `HW_ROWS x HW_COLS` = 4 x 4 is
the mesh: `aie::program_id` ranges over it, and the compiler places the 16
cores in it. The compiler also **reads this call** -- it is how the kernel
analysis knows to trace a 4 x 4 mesh.

```cpp
    int8_t *A = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GEMM_N * GEMM_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_N);
    gemm_init(A, B, C);
```
DDR buffers the AIE DMA can reach, sized like the ports' `aie::shape` (16 KB,
16 KB, 4 KB). `gemm_init` fills A with values in [-4, 4] and B with {-1, 0, 1}
(so a dot product of 256 fits easily) and zeroes C.

```cpp
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm<<<mesh>>>(A, B, C);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
```
The **launch**, CUDA-style. aiehlc rewrites it into
`__aie_launch("gemm", mesh, A, 16384, B, 16384, C, 4096)`, which flushes the
inputs from the cache, loads the kernel ELF into all 16 cores, programs every
DMA buffer descriptor and lock the compiler derived (`host_canonicalized`, the
generated `host.cc`), starts the cores and the shim DMAs, and waits for the
last output block. When it returns, `C` holds the result. `XTime` measures the
whole launch.

```cpp
    const int mismatches = gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
    return mismatches ? 1 : 0;
}
```
Compare with a CPU GEMM (`PASS: all 4096 output elements match.`).

## 4. How the pids land on the hardware

![The mesh: pids, tiles and the derived DMA](img/gemm_mesh.svg)

| pid `(pm, pn)` | array tile `(col, row)` | A rows | B rows | C block |
|---|---|---|---|---|
| (0, 0) | (3, 3) -- partition col 0 | 0..15 | 0..15 | rows 0..15, cols 0..15 |
| (0, 3) | (6, 3) | 0..15 | 48..63 | rows 0..15, cols 48..63 |
| (3, 0) | (3, 6) | 48..63 | 0..15 | rows 48..63, cols 0..15 |
| (pm, pn) | (3 + pn, 3 + pm) | 16pm.. | 16pn.. | (16pm.., 16pn..) |

Mesh row `pm` is array row `3 + pm` (row 3 is the core row closest to the
shim row), mesh column `pn` is partition column `pn` = array column `3 + pn`.
The derived data movement (from the generated `host.cc` and routing):

- **A**: shim tile of column `pm`, MM2S channel 1, streams A rows
  `16pm..16pm+15` in 4 blocks of 16 x 64 and broadcasts them to the 4 cores of
  mesh row `pm`;
- **B**: shim tile of column `pn`, MM2S channel 0, streams B rows
  `16pn..16pn+15` to the 4 cores of mesh column `pn`;
- **C**: the 4 cores of a mesh row are gathered back through one shim S2MM
  channel, each 16 x 16 block to its own place in `C`.

8 input streams instead of 32: the broadcast is the point of the 2-D split.

## 5. What the compiler made of it

The compiler works in two steps. First it finds out **what data each core
touches** by running your kernel, and turns that into a DMA pattern. Then an
MLIR pipeline lowers that pattern, stage by stage, into the three C++ files
that run on the hardware.

### 5.1 From `load_buffer` calls to a DMA pattern

![Trace -> affine fit -> DMA pattern](img/trace_to_dma.svg)

1. **Trace.** The kernel body is compiled for x86 with recording versions of
   `aie::program_id`, `load_buffer`, `store_buffer`, `release` and `commit`
   (under AddressSanitizer), and run once for **every core** of the
   `device.partition(..., 4, 4)` mesh. Each call records
   `(core, port, offset, shape)` -- the kernel's data access, in tensor
   coordinates.
2. **Fit.** Each port's recorded blocks must fit one formula,
   `offset = base_p0*pm + base_p1*pn + step_i*i + step_j*j`, with one block
   shape. Which `base` is non-zero decides the data movement: A moves with `pm`
   only, so every core of a mesh row wants the same blocks (**row
   broadcast**); B moves with `pn` only (**column broadcast**); C moves with
   both, every core its own block (**gather**).
3. **DMA pattern.** That formula is exactly what a DMA buffer descriptor does
   -- a start address plus nested (stride, count) loops. For A: start at row
   `16 pm`, 64 bytes of a row, 16 rows 256 bytes apart, 4 chunks 64 bytes
   apart, one 1 KB window per round.

The build log prints the result of steps 1-2 -- read it after every change:

```
[KernelAnalysis] gemm: traced on the 4x4 mesh
  a (IFM, tensor (64,256)): block (16,64), core offset (16,0)*p0 + (0,0)*p1, rounds 1x4 steps (0,0) / (0,64)
  b (WEIGHTS, tensor (64,256)): block (16,64), core offset (0,0)*p0 + (16,0)*p1, rounds 1x4 steps (0,0) / (0,64)
  c (OFM, tensor (64,64)): block (16,16), core offset (16,0)*p0 + (0,16)*p1, rounds 1x1 steps (0,0) / (0,0)
  routing plan on the 4x4 mesh:
    a (IFM, row broadcast, 1x4 blocks/core): d1[64]=rows/4@16 d2[256]=j4x64@64
    b (WEIGHTS, col broadcast, 1x4 blocks/core): d1[64]=cols/4@16 d2[256]=j4x64@64
    c (OFM, gather, 1x1 blocks/core): d1[64]=rows/4@16 d2[64]=cols/4@16
  => routed: 3 ports on the 4x4 mesh, DMA carriers taken from the traced blocks
```

`d1[64]=rows/4@16` = "dim 1 (64) is split over the 4 mesh rows, 16 apart";
`d2[256]=j4x64@64` = "dim 2 (256) is walked by the inner loop `j`, 4 blocks of
64, step 64". Step 3 is written back into the source as one explicit
descriptor per port (`aout/newfile.cpp`), which is what the rest of the
compiler reads:

```cpp
constexpr aie::GemmSpace __aie_gemm_IFM = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row}, ...
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},
    .d2 = {.fullsize = 256, .tile_round = 4, .tile_size = 64, .stride = 64}};
```

Nothing in these steps knows the kernel is a GEMM -- tutorial 2's conv goes
through exactly the same three steps.

The trace also **checks** the kernel: a block outside the tensor, data used
without being loaded (an index past the block), a block used after `release`,
an output never committed, loops a DMA cannot stream -- each is a build error
with the source line, before anything reaches the board.

### 5.2 From the DMA pattern to host.cc, kernel.cc and routing.cc

![The compiler pipeline](img/compiler_flow.svg)

The descriptors become a **routing IR** module, and six MLIR dialects lower it
one concern at a time. Every stage writes its IR to `ir/dfschedule/` or
`ir/simplerouting/` in your build directory, so you can open each step. Here
is matrix A at each stage of this build:

| Stage (IR dump) | Dialect: what it says | A in this GEMM (excerpt of the real IR) |
|---|---|---|
| `0_initial.mlir` | **routing**: the abstract plan -- mesh, tensors, how each is split, which tile list gets which slice | `routingcreatehwmesh row = 4, col = 4 partition = 3, 6, 0, 6`; `partitiontensor ... splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col"`; per mesh row: `routingextract_data ... -> tensor<16x256xi8>`, `routingmovedatabyio` to that row's tiles |
| `1_RoutingUnrollingLowerPass` | **routing**, unrolled: one op per mesh row / column instead of a loop | the 2 loops (3 `routingmovedatabyio` in their bodies) become 12 separate moves: 4 for A (one per mesh row), 4 for B (one per mesh column), 4 for C |
| `2_RoutingToDmapPass` | **dmap**: logical dataflow -- an IO engine, a core group, a stream between them, push / pull | `define_io_engine {ioattr = "SHIM"}`, `define_core_group {core_count = 4 ...}`, `create_stream src = ..., dst = ...`, `dmap.push` (8 push streams for A and B, 4 pull for C) |
| `3_DmapToDmaphopPass` | **dmaphop**: physical placement -- real tiles, DMA ports and channels, the hop-by-hop path | `dmaphop.tile{TILETYPE = "core", col = 0, row = 3}`, `port ... direction = "In", direction_channel = 1`, hops shim -> (0,3) -> (0,4) -> (0,5) -> (0,6), `create_path {producers = [[@f0_shimPortIn]], consumers = [4 cores]}` |
| `5_DmaphopTodfscheblueprintPass` | **dfscheblueprint**: per flow, what slice of the tensor goes to which tile group in which round | `flow_transfer`, `data_slice @consumer_slice_0_0 ...`, `flowconfig` |
| `6_BlueprintToSchedulePass` ... `12_DfscheduleToApiPass` | **dfschedule (host)**: executable schedule -- DMA BDs, locks, IOs, launch, waits; then runtime API calls | `config.dma_bd(%subview ...)`, `config.create_io`; becomes `__Runtime_dma_bd_config_multidim_ooo(...)` in `host.cc` |
| `18_BlueprintToScheduleKernelPass`, `19_DfscheduleToKernelApiPass` | **dfschedule (kernel)**: the core side of the same flows -- buffers, locks, windows | `buffer_def @buf_in_ping_1 ... {address = 491520}`, `window_init`; becomes `kernel.cc`'s `main()` |
| `simplerouting/1_DmaphopToRoutinghwPass` ... `5_RoutingHWLowerPass` | **routinghw**: stream-switch connections for every hop, shim mux / demux | `connectsinglestreamswitchport {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", ...}`; becomes `XAie_StrmConnCctEnable(...)` in `routing.cc` |

Three things to notice:

- **Each dialect answers one question.** routing: *what goes to which group of
  tiles*; dmap: *which logical streams*; dmaphop: *which physical tiles, ports
  and paths*; dfschedule: *which DMA programs and locks, in what order*;
  routinghw: *which stream-switch ports to connect*.
- **dmaphop is the fork.** The host path, the kernel path and the routing
  path all start from the same dmaphop IR, so the host's buffer descriptors,
  the core's windows and the stream switches describe the same flows. (The
  host and kernel paths program the same core tiles, so they share their
  core-DMA logic in `passblueprintlowering/helper/`.)
- **The last IR of each path is emitted as C++** (MLIR EmitC): `host.cc` and
  `routing.cc` are compiled for the Arm into `main.elf`, `kernel.cc` with the
  AIE compiler into the core image embedded in it. At the end of every build
  `script/test/flowcheck.py` simulates `routing.cc` against the dmaphop flows
  and prints `flowcheck: OK -- 12 flows` (8 push + 4 pull here).

## 6. Run it

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplegemm_kernel_port.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf
```

```
[BENCH] aie_wall_ms=18.413
PASS: all 4096 output elements match.
```

18 ms for 2 M multiply-accumulates is not compute: a launch reloads the
kernel into 16 cores and programs every descriptor over the configuration bus
(~0.4 us per 32-bit register write), and the scalar MAC loop is the slow part
of what remains. Tutorials 3 and 4 work on both.

## 7. Try it

You do not need a full build (or a board) to see what the compiler derives:
`./build/aiehlc-convtile trace <file.cc>` runs only the kernel analysis and
prints the report of section 5 in about a second.

- **Bigger K chunks.** `tk = 128`: the report shows `block (16,128) ... rounds
  1x2` and `d2[256]=j2x128@128`; the A / B windows grow to 2 KB, the loop runs
  2 rounds. Same result.
- **Watch the broadcast follow the kernel.** Load A at `{pn * tm, ...}` and B
  at `{pm * tn, ...}`: the report now says `a (IFM, col broadcast ...)` and
  `b (WEIGHTS, row broadcast ...)`. (That kernel computes a different product,
  so `gemm_verify` would fail -- this is only to see the analysis react.)
- **Break it on purpose.** Load `{tm + 1, tk}` (17 rows): the analysis stops
  with `ERROR: gemm: a block 0 on core (3,0): block dim 0 covers [48, 65)
  outside the tensor extent 64` -- core row 3's block would run past A.

If a board run reports a few hundred outputs as `0` (or stale) in one core's
block and the next run passes, that is a known intermittent host-cache issue
after the output DMA, not your kernel: re-run, and compare against a second
run before debugging the kernel.

Next: [Tutorial 2 -- Conv2d on a 4x4 tile mesh](02_conv2d_4x4.md).
