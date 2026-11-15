# How to write an AIEDNN kernel: one template, many compiled instances

An **AIEDNN kernel** is a DNN operator (GEMM, or a conv lowered to GEMM) written
for the AIE array in aiehlc's kernel-only model: one `__global__` kernel that
says, in tensor coordinates, which block each core loads and stores. The
compiler derives the DMA and routing from it. This guide shows how to write one
that is **fast for a whole layer, not just correct for one tile**, and why each
decision is made the way it is.

It distills sections 15–21 of the UNet deployment report
(<http://xsjsswsip51:8099/>, "How the P1 and P4 blocks were resolved" through
"why an op still needs many kernels"). Its kernel is `gemm_kacc`, the bf16
GEMM that took the UNet's 37 AIE conv layers from 131.2 s to 43.1 s. This guide
also uses its own tested example:

| File | What |
|---|---|
| [`example/aiednn/gemm_kacc_bf16.cc`](../../../example/aiednn/gemm_kacc_bf16.cc) | the template kernel + a self-checking `main()` |
| [`example/aiednn/gemm_kacc_bf16.h`](../../../example/aiednn/gemm_kacc_bf16.h) | host side: test data, pre-tiling, fp32 reference, check, phase report |
| [`example/aiednn/gen_variants.py`](../../../example/aiednn/gen_variants.py) | the generator: layer shapes -> parameter sets -> one source per variant + dispatch table |

Read [tutorial/aieprogram](../../aieprogram/README.md) first if `load_buffer`,
`program_id` or `{.resident}` are new to you.

---

## 1. Why: what one launch carries decides everything

Every AIE launch pays a **fixed cost** before any MAC happens. That cost covers
kernel load, BD and DMA setup, core reset and synchronisation. So the first
design decision is how much work one launch carries. The UNet deployment
measured it (report §17, §18):

| Same work: one 64×64×64 tile (262 k MAC) | Time |
|---|---|
| on the A78 (TVM C) | ~0.29 ms |
| one AIE launch (bf16 vector MAC, warm kernel) | 0.99 ms |

| The whole UNet (37 conv layers on the AIE) | Launches | One UNet call |
|---|---|---|
| A78 only | — | 61.2 s |
| one 64³ tile per launch | 86,600 | 131.2 s (AIE part 85.6 s) |
| **one launch per layer, K accumulated on the core** (`gemm_kacc`) | **37** | **43.1 s (AIE part 0.33 s)** |

With one tile per launch, the AIE was **3.5× slower than the CPU** for the same
work. The fixed cost (~0.9 ms) dwarfs ~0.02 ms of compute. The fix was not a
faster kernel but a different unit of work:

> **Rule 1.** Push the tile loop down into the core and the DMA. The host
> launches once per layer (or once per network), never once per tile.

The example measures the same thing on one launch with the runtime's phase
counters (`--profiling`, section 7). The DMA and the vector MACs (`wait_io`)
are **0.7 %** of the launch. Reloading the kernel (`kload`) is **98 %**.

---

## 2. The skeleton: three loops and an epilogue

Every AIEDNN GEMM / conv kernel has this shape (report §20.0). The comments say
which section of this guide decides each line:

```
for each output tile (m0, n0) this core owns:        <- §4.2 scheduling
    acc = 0                                           <- §4.3 fp32 accumulator, stays on the core
    for each K chunk j:                               <- §4.3 chunk size TK
        A = load_buffer(a, {m0, j*TK}, {TM, TK})
        B = load_buffer(b, {n0, j*TK}, {TN, TK})
        acc += A . B   (aie::mmul)                    <- §4.1 register blocking
        release(a); release(b)
    y = epilogue(acc)                                 <- §4.7 fusion
    store_buffer(c, {m0, n0}, {TM, TN}); commit(c)    <- §4.5 shape tails
```

The **template parameters** are what changes between layers: the shape (M, N,
K), the chunk TK, the epilogue, the data type, the layout. §5 is how they become
compiled kernels.

---

## 3. The example kernel

`C[M][N] = relu(A[M][K] · W[N][K]ᵀ)`, bf16 in and out, fp32 accumulation, one
launch on the 4×4 mesh. A conv layer is this GEMM after im2col: M = output
pixels, K = Cin·kh·kw, N = Cout.

```cpp
#include "gemm_kacc_bf16.h"

// ---- template parameters (gen_variants.py rewrites this block) ----
#define AD_M 256
#define AD_N 64
#define AD_K 2304
#define AD_TK 64
#define AD_RELU 1
// ---- end of template parameters ----
// The output tile, fixed by the 2 x 2 accumulator blocking below.
#define AD_TM 16
#define AD_TN 16

__global__ void gemm_kacc(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<AD_M, AD_K>> a,
                          aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<AD_N, AD_K>> w,
                          aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<AD_M, AD_N>> c) {
    const int ntile = AD_M / (4 * AD_TM); // output tiles per core, walked on the core
    const int nkc = AD_K / AD_TK;          // K chunks per tile
    const int kt = AD_TK / 8;              // 8x8 tiles per chunk row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * AD_TM) + t * AD_TM;
#if defined(__chess__)
        using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
        MM c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>());
        MM c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
#else
        float acc[AD_TM * AD_TN] = {}; // x86 trace / emulation: the same math, scalar
#endif
        for (int j = 0; j < nkc; j++) {
            const bfloat16 *x = aie::load_buffer(a, {m0, j * AD_TK}, {AD_TM, AD_TK});
            const bfloat16 *v = aie::load_buffer(w, {pn * AD_TN, j * AD_TK}, {AD_TN, AD_TK});
#if defined(__chess__)
            for (int kb = 0; kb < kt; kb++) {
                const auto a0 = aie::load_v<64>(x + kb * 64), a1 = aie::load_v<64>(x + (kt + kb) * 64);
                const auto b0 = aie::load_v<64>(v + kb * 64), b1 = aie::load_v<64>(v + (kt + kb) * 64);
                c00.mac(a0, b0);
                c01.mac(a0, b1);
                c10.mac(a1, b0);
                c11.mac(a1, b1);
            }
#else
            for (int m = 0; m < AD_TM; m++)
                for (int n = 0; n < AD_TN; n++)
                    for (int k = 0; k < AD_TK; k++)
                        acc[m * AD_TN + n] += (float)x[((m / 8) * kt + k / 8) * 64 + (m % 8) * 8 + k % 8] *
                                              (float)v[((n / 8) * kt + k / 8) * 64 + (k % 8) * 8 + n % 8];
#endif
            aie::release(a);
            aie::release(w);
        }
        // The tile is done: epilogue on the fp32 accumulator, one rounding, one write-back.
        bfloat16 *y = aie::store_buffer(c, {m0, pn * AD_TN}, {AD_TM, AD_TN});
#if defined(__chess__)
        alignas(64) float acc[AD_TM * AD_TN];
        const MM *blk[4] = {&c00, &c01, &c10, &c11};
        for (int b = 0; b < 4; b++) {
            alignas(64) float t8[64];
            aie::store_v(t8, blk[b]->to_vector<float>());
            for (int i = 0; i < 64; i++)
                acc[((b / 2) * 8 + i / 8) * AD_TN + (b % 2) * 8 + i % 8] = t8[i];
        }
#endif
        for (int i = 0; i < AD_TM * AD_TN; i++) {
            float o = acc[i];
            if (AD_RELU && o < 0.0f)
                o = 0.0f;
            y[i] = (bfloat16)o;
        }
        aie::commit(c);
    }
}
```

The host part of the file packs the inputs, launches once, checks every output
against an fp32 reference, and prints `PERF SUMMARY`, the phase split and
`PASS` / `FAIL`. See the [file](../../../example/aiednn/gemm_kacc_bf16.cc).

Two things that look odd and are deliberate:

- **`#if defined(__chess__)`.** The vector code (`aie::mmul` with an `accfloat`
  accumulator) only exists for the AIE compiler. The x86 emulation that the
  kernel analysis traces has `bfloat16` but no `accfloat` / bf16 `mmul` yet. So
  the `#else` branch computes the same result in scalar code. The analysis
  checks *which blocks* the kernel touches, and both branches touch the same
  ones. This is the workaround `gemm_kacc` used too (report §18, problem 5).
- **The odd index arithmetic** (`(m / 8) * kt + k / 8) * 64 + ...`). The
  blocks arrive pre-tiled in `mmul` operand order (§4.6), and this is how one
  element of a pre-tiled block is addressed.

---

## 4. The decisions and their trade-offs

### 4.1 Output tile and register blocking (TM × TN)

**What.** Each core computes a 16×16 output tile at a time, as a 2×2 grid of
`aie::mmul<8,8,8>` accumulators (`c00 … c11`). Each A tile is used against two
B tiles and each B tile against two A tiles before it leaves the registers.

**Why.** A Tm×Tn×Tk tile does Tm·Tn·Tk MACs on Tm·Tk + Tk·Tn input elements.
A larger output tile reuses every loaded byte more often. The 2×2 blocking is
what the int8 `conv2dpeak` kernel uses to reach 92 % of int8 peak (report §19).

**Trade-off.**

| Larger output tile | Smaller output tile |
|---|---|
| more reuse, fewer loads per MAC | fewer registers, smaller C buffer |
| more accumulator registers (4 × 2048-bit here), larger C window | more loads per MAC: bandwidth-bound sooner |
| more padding waste on small layers (§4.5) | fits small or odd shapes |

**Fixed in this template.** The accumulator count is code, not a number. A
different blocking (1×4, 4×1, …) is a different template, not a parameter (§6).

**Hardware granularity.** bf16 `aie::mmul` on AIE2PS comes in 4×8×4, 4×8×8,
8×8×8, 4×16×8 and 8×1×8 (`aie_api/detail/aie2ps/mmul_fp_fp.hpp`). Tiles are
multiples of the shape you pick.

### 4.2 Scheduling: more tiles than cores

**What.** Work units U = ⌈M/TM⌉ · ⌈N/TN⌉; cores P = 16 (4×4). In the example
each mesh column owns one 16-wide column of C. Each core walks `ntile =
M / (4·TM)` row tiles of its mesh row, **inside one launch**. The DMA streams
them in that same order.

**Why.** This is Rule 1: the fixed per-launch cost is paid once per layer
instead of once per tile.

**Trade-off and options** (report §20.2):

| Distribution | How | Good for |
|---|---|---|
| **block** (this example, `gemm_kacc`) | core (pm, pn) takes `ntile` consecutive tiles: `m0 = pm·ntile·TM + t·TM` | a regular DMA walk (one affine sequence per port); what the kernel analysis can route |
| strided | `for (u = core_id; u < U; u += P)` | load balance when U is not a multiple of P |

The whole layer waits for the slowest core. **Pick TM/TN so U is a multiple of
P.** `gen_variants.py` pads M to a multiple of 4·TM for exactly this reason.

### 4.3 K: accumulate on the core, and how big a chunk (TK)

**What.** K is streamed in chunks of TK. The fp32 accumulator stays in the
`mmul` registers across all K/TK chunks and is rounded to bf16 **once**, after
the last chunk.

**Why one rounding.** Rounding the partial sum to bf16 after every chunk loses
accuracy. On UNet layer c013 (K = 23,040, 360 chunks) it measured cosine
0.999818, against 0.999998 with one final rounding (report §20.4).

**Hard constraints on TK.**
- A multiple of the `mmul` K (8 here).
- The port buffers fit the core's port-buffer region, 29,696 B (32 KB minus
  ~3 KB of klog and statics). Every input window is double-buffered, and so is
  the output window:

  `2·(TM·TK + TN·TK)·2 B  +  2·TM·TN·2 B  ≤  29,696 B`

  At TM = TN = 16 that allows TK up to 224 (`128·TK + 1,024 ≤ 29,696`).

**The trade-off, measured.** A bigger TK means fewer chunks, so fewer lock
round trips and DMA handshakes. But the window buffers are part of the kernel's
BSS, and a cold launch re-writes the BSS of all 16 cores over the config bus.
The same 256×64×2304 GEMM on palmyra:

| TK | kernel BSS per core | launch | kload | bdcfg | wait_io (DMA + compute) |
|---|---|---|---|---|---|
| **64** | 9,228 B | **33.0 ms** | 41.3 M cycles (98.0 %) | 0.48 M | 0.31 M |
| 128 | 17,420 B | 53.6 ms | 68.3 M cycles (99.0 %) | 0.33 M | 0.28 M |

TK 128 really does less work: `bdcfg` and `wait_io` both drop. But the extra
8 KB per core costs ~27 M more cycles of kernel load, which is 60 % slower
end to end. With a **warm** runtime (kernel resident, BSS not re-written: the
UNet's "warm launch", 44.4 → 4.5 ms per launch) the load term disappears, and
the larger TK wins.

> **Rule 2.** There is no universally best TK. Choose the largest one that
> fits, *for the launch model you actually run*, and measure the launch split
> (`--profiling`).

### 4.4 Who stays on the core: A-, B- or C-stationary

| Dataflow | Resident on the core | Streamed | Reused | Fits |
|---|---|---|---|---|
| **C-stationary** (this example, `gemm_kacc`) | the TM×TN accumulator | A and B chunks along K | C over all K/TK chunks, written once | long K (UNet convs: K = 9·Cin up to 23,040) |
| B-stationary (weights) | a TK×TN weight block (`{.resident}`) | activations along M | weights over M/TM tiles | small weights, many pixels |
| A-stationary (input) | a TM×TK activation block | weights along N | activations over N/TN tiles | few rows, many output channels (small-M Linear) |

**How to choose.** A is reused N/TN times, B is reused M/TM times, and C is
reused K/TK times. Keep the most-reused one that still fits (report §20.5).
The loop order *is* the choice:

```
C-stationary:  for (m,n) { acc = 0; for k { acc += A[m,k]·B[k,n] } store C[m,n] }
B-stationary:  for (k,n) { load B[k,n]; for m { C[m,n] += A[m,k]·B[k,n] } }
A-stationary:  for (m,k) { load A[m,k]; for n { C[m,n] += A[m,k]·B[k,n] } }
```

A different stationarity is a different loop nest, so it is a **different
template**, not a parameter (§6).

In aiehlc a resident input is expressed directly. Load the block once, outside
the loop, with `{.resident = true}`. The DMA sends it once and gives it one
buffer, and the analysis checks that the kernel really takes one block
([tutorial 3](../../aieprogram/03_conv2d_scheduling.md)).

### 4.5 Shapes that do not divide the tile

| Option | How | Cost | Use when |
|---|---|---|---|
| A. pad with zeros (host) | M up to a multiple of 4·TM, K up to a multiple of TK | wasted MACs and bytes (M = 64 on TM = 128 wastes 50 %) | waste < ~10 % |
| B. tail inside the kernel | load the full tile, store only `min(TM, M − m0)` rows | a branch; the DMA still moves the full tile | waste is large but one variant is wanted |
| C. a tail variant | main body at TM = 128, the remainder with a TM = 64 instance | one more compiled kernel | performance matters |

**K padding is always safe:** 0 · w = 0 does not change the sum. That is why
the generator pads K, and why layer 3 of the demo (K = 576 → 640 at TK 128)
reports 10 % waste. The best fix for a small M is usually **not** padding but a
variant with a smaller tile. That is the first reason one op ends up with many
kernels (§6).

### 4.6 Layout: who re-orders the data

`aie::mmul<8,8,8>` reads an 8×8 A tile (M × K) and an 8×8 B tile (K × N) as 64
contiguous elements. A row-major block does not have that order. Something has
to re-order it:

| Who | How | Limits |
|---|---|---|
| **the host, once** (this example, `gemm_kacc`) | `ad_pretile()` stores every 16×TK chunk of A and W in operand order inside the same tensor; the DMA moves it unchanged | host time (once per weight set; per call for activations) |
| the shim DMA: `{.tile = {tr, tc}}` | the block arrives panel-major, so each tr×tc tile is contiguous (3 BD dims) | only blocks that are **whole rows** of the tensor, and only re-orders whole 32-bit words: it cannot transpose int8 or bf16. The analysis refuses `.tile` on a K-chunk block |
| the core | `aie::transpose` / shuffles per chunk | vector cycles every chunk |

Here A and W are K-chunked (16×TK of a 2304-wide row), so `.tile` cannot apply.
W also needs a K×N transpose. The host pre-tiles both.

### 4.7 Fusion: the epilogue

**What.** Element-wise work applied to the accumulator before its single
write-back: bias, activation, residual add, scale. Here it is one compile-time
switch:

```cpp
float o = acc[i];
if (AD_RELU && o < 0.0f)   // AD_RELU is a #define: the branch folds away when 0
    o = 0.0f;
y[i] = (bfloat16)o;
```

**Why.** A fused epilogue saves a DDR round trip: the unfused version writes C
and another op reads it back. In the UNet, bias / timestep / residual still run
on the A78 and are part of the 6.1 s of host glue (report §18).

**Trade-offs.**
- **Every extra operand is an extra port.** Bias (per output channel) or a
  residual (same shape as C) is another `aie::port`. A port is part of the
  kernel's signature, so a fused bias+residual kernel is a different kernel
  from a plain one.
- **Order matters.** bias → ReLU → +residual is not +residual → ReLU. The
  epilogue must follow the model graph exactly.
- A policy chain keeps one source for many epilogues (report §20.3,
  illustration):

  ```cpp
  struct Bias { static float f(float x, const EpiArgs &e) { return x + e.bias[e.co]; } };
  struct Relu { static float f(float x, const EpiArgs &)  { return x > 0.f ? x : 0.f; } };
  template <class... Ops> struct Chain {
      static float f(float x, const EpiArgs &e) { ((x = Ops::f(x, e)), ...); return x; }
  };
  using Epi_CBR = Chain<Bias, Relu>;   // conv + bias + relu
  ```

  In aiehlc today the selection is a compile-time `#define`, as in the example:
  the kernel body is analysed and compiled as text with the source's plain
  `#define`s (§5, §8), so whatever selects the epilogue has to be a
  `#define`.

### 4.8 Data type

The example is all bf16: `input_window_bfloat16` / `output_window_bfloat16`
ports, an fp32 `accfloat` accumulator, and one rounding to bf16. Two current
limits shape the ports:

- **One element type per kernel.** All ports are bf16, so C is bf16. The
  UNet's `gemm_kacc` wanted fp32 out, so it declared every port `int32` and
  packed two bf16 per word. That works, but it pushes packing onto the host.
- int8 and bf16 use different vector instructions and have different best
  tiles. A type change is a new instance at least, and often a new template.

---

## 5. Templating: compile-time parameters, a generator, a dispatch table

**Why compile time.** `aie::shape<R, C>` must be a constant. The compiler turns
it, together with the block sizes and loop bounds, into the DMA windows, the
buffer sizes and the BD walk. So a parameter set is one compiled kernel plus
its DMA configuration, and it cannot be chosen at run time.

**Why not one run-time-parameterised kernel** (report §21.2)?
- Loop bounds and tile sizes become variables. The compiler can no longer
  unroll or schedule vector instructions for fixed sizes, and the inner loop
  slows down.
- DMA windows must be sized for the largest shape. Small shapes still occupy
  the full buffers, and large ones may not fit at all.
- That was the UNet's first version: one 64³ tile for every shape, looped by
  the host. It computed correctly in 131 s. Per-layer instances took 43.1 s.

**How, in this repo.** The parameters are a block of plain `#define`s at the
top of the template file. aiehlc analyses and compiles the kernel body as
**text**, together with the source's plain `#define` lines. It does not carry
`#ifndef` blocks, and it does not instantiate a C++ template on a `__global__`.
Tried: `template <int AD_TK> __global__ ...` with `gemm_kacc<64><<<mesh>>>`
fails the analysis with `'AD_TK' was not declared in this scope`. So the
generator rewrites the `#define` block, one copy per parameter set:

```python
def pick(layer):                      # gen_variants.py, abridged
    m = ceil_to(layer["m"], MESH * TM)            # whole 16-row tiles per core (§4.2)
    for cand in (256, 128, 64, 32, 16, 8):        # largest K chunk that fits (§4.3)
        mem = 2 * (TM * cand + TN * cand) * BF16 + 2 * TM * TN * BF16
        if mem <= PORT_BUDGET and cand <= layer["k"]:
            tk = cand
            break
    k = ceil_to(layer["k"], tk)                   # zero K padding is exact (§4.5)
    return {"AD_M": m, "AD_N": layer["n"], "AD_K": k, "AD_TK": tk, "AD_RELU": layer["relu"]}

for i, layer in enumerate(layers):                # de-duplicate: same parameters = same kernel
    params = pick(layer)
    name = variants.setdefault(key(params), f"gemm_kacc_v{len(variants)}")
    table.append({"layer": i, "variant": name, **params})   # -> dispatch.json
```

```
$ python3 example/aiednn/gen_variants.py --out /tmp/variants
layer 0: M=256 N=64 K=2304 relu=1 -> gemm_kacc_v0 (TK 128, padded M 256 K 2304, waste 0.0%)
layer 1: M=256 N=64 K=2304 relu=1 -> gemm_kacc_v0 (TK 128, padded M 256 K 2304, waste 0.0%)
layer 2: M=512 N=64 K=1152 relu=0 -> gemm_kacc_v1 (TK 128, padded M 512 K 1152, waste 0.0%)
layer 3: M=64 N=64 K=576 relu=1 -> gemm_kacc_v2 (TK 128, padded M 64 K 640, waste 10.0%)
4 layers -> 3 variants in /tmp/variants (dispatch.json: layer -> variant)
```

The host then launches each layer with its variant, once per layer. Consecutive
layers that share a variant do not reload the kernel (warm launch). In the UNet
this ran by hand: `kplan.py` picked the parameters and `unet_aie.py` wrote the
sources, giving **37 layers → 21 variants**, plus transposed variants for the
px = 1024 layers.

> The generator's cost model above maximises TK. On the cold-launch runtime of
> §4.3 that is the *slower* choice, because the buffers are reloaded per
> launch. A real generator must take the launch model into account, or better,
> be fed measured launch splits.

---

## 6. Why one op needs many kernels

A single fixed-tile kernel makes every shape **computable**; instances make
every shape **fast** (report §19, §21).

| Level | In the UNet conv path | Who makes it |
|---|---|---|
| template (source) | 1: `gemm_kacc`. One more per *structure*: B-stationary, direct conv, … | written by hand |
| parameter sets (TM/TN/TK, layout, fusion, type) | 37 layers → 21 sets | the generator, from the shapes |
| compiled kernels | 21 ELFs (+ transposed variants) | aiehlc |
| at run time | one launch per layer; no reload when the variant repeats | the host, from the dispatch table |

What a parameter can cover and what needs another template:

| Axis | Parameter? | Why |
|---|---|---|
| shape M/N/K, chunk TK | yes | constants in the same loop nest |
| epilogue on/off (no new operand) | yes | a folded branch |
| epilogue with a new operand (bias, residual) | new instance, new signature | a port is part of the signature |
| tile blocking (2×2 vs 1×4 accumulators) | new template | different register code |
| stationarity (C / B / A) | new template | different loop nest and buffers |
| algorithm (im2col+GEMM vs implicit-GEMM direct conv) | new template | a different way of fetching data |
| data type (int8 / bf16) | new instance, often a new template | different vector instructions |

The variants do not compete for the array. A core holds one kernel at a time,
so more variants only cost compile time and ELF space in DDR.

---

## 7. Build, check, run

```bash
# 1. the analysis (seconds): blocks fit the DMA, every vector access inside its block
build/aiehlc-convtile trace example/aiednn/gemm_kacc_bf16.cc
#   OK: every load_buffer / store_buffer matches the DMA   (11 block accesses checked)

# 2. build (aout/ lands in the current directory) and run on the board
source script/aiehlc.sh --aie-version 5 --runtime-source-file $PWD/example/aiednn/gemm_kacc_bf16.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf
#   [gemm_kacc] PERF SUMMARY launch 32.967 ms, 2.29 GOPS
#   [gemm_kacc] check: 16384 / 16384 bit-exact, max 0 ULP
#   PASS

# 3. where the launch time goes: rebuild with --profiling
source script/aiehlc.sh --aie-version 5 --profiling --runtime-source-file $PWD/example/aiednn/gemm_kacc_bf16.cc
#   [gemm_kacc] PERF phases (PMU cycles): kload 41258745 (98.0%) bdcfg 475839 (1.1%) ... wait_io 305741 (0.7%)

# 4. other shapes: generate the instances and build any of them the same way
python3 example/aiednn/gen_variants.py --out /tmp/variants
```

Results on palmyra (AIE2PS, 4×4 mesh), 2026-11-14:

| Instance | Shape (M×N×K), TK | Result | Launch |
|---|---|---|---|
| template default | 256×64×2304, 64, ReLU | 16,384 / 16,384 bit-exact, 3/3 runs | 33.0 ms |
| `gemm_kacc_v0` | 256×64×2304, 128, ReLU | 16,384 / 16,384 bit-exact, 2/2 | 53.6 ms |
| `gemm_kacc_v1` | 512×64×1152, 128 | 32,768 / 32,768 bit-exact, 2/2 | 53.4 ms |
| `gemm_kacc_v2` | 64×64×640 (K padded from 576), 128, ReLU | 4,096 / 4,096 bit-exact, 2/2 | 52.6 ms |

The test data are multiples of 1/4, so every fp32 partial sum is exact and the
check can demand bit-exact results whatever order the `mmul` accumulates in.
With real data, check against a tolerance.

v2 has ~1/30 of v0's MACs yet takes the same time. **These launches are all
kernel load** (§1, §4.3), which is why a warm, resident kernel is the next
lever, not a faster inner loop.

---

## 8. Pitfalls (each one cost a build or a board run)

| Symptom | Cause | Fix |
|---|---|---|
| `the core window is 256 elements x 144 round(s) ..., but the kernel's blocks are 1024 elements x 144` | aiehlc divided **every** input's window by the M-round count; only A holds M strips (`unet_m2` P4bk) | fixed on main (from `8c60bf8`). The build error is the `#routing.kernel_block` check doing its job: without it, a silent hang |
| board `wait_io TIMEOUT` on the input shims, ch 1 | the A strip BD walks all K chunks in 3-D but was started K-rounds times with an element-count length | fixed on main (from `5791cbd`, narrowed so a BD whose length already is the whole walk keeps its repeat: `simplematmul2` uses that) |
| `{.tile / .resident} need blocks that are whole rows of the tensor` | `.tile` on a K-chunk block: the DMA would have ignored it | pre-tile on the host (§4.6) |
| `use of undeclared identifier` for a parameter | it was defined inside `#ifndef`; aiehlc copies only plain `#define`s into the kernel | plain `#define`s in the template block |
| "static bounds analysis: not evaluable" | a `// comment` on a `#define` line | comments on their own lines |
| a few hundred outputs = the host's `memset` value on ~1 run in 6 | the generated launch invalidates the output only *before* the DMA | `Xil_DCacheInvalidateRange(C)` after the launch (in the example); general fix on `arthur/output-post-dma-invalidate` |
| bf16 kernel "does not compile for tracing" | the x86 emulation has `bfloat16` but no `accfloat` / bf16 `mmul` | `#if defined(__chess__)` vector path + scalar `#else` (§3) |
| N wider than 4·16 | this template gives each mesh column one 16-wide column of C | an N loop in the kernel: another template (`gen_variants.py` refuses) |

---

## 9. Checklist for a new AIEDNN kernel

1. **Unit of work.** One launch per layer. Estimate fixed cost × launches
   against the CPU time for the same work *before* writing the kernel (report
   §17: the rule the UNet project lacked).
2. **Dataflow.** Pick the stationary operand from the reuse counts (§4.4).
3. **Register blocking and tile.** A multiple of the `mmul` shape; U a
   multiple of the core count (§4.1, §4.2).
4. **K.** fp32 accumulation, one rounding; the largest TK that fits the port
   region *and* suits the launch model, measured (§4.3).
5. **Layout.** Who produces `mmul` operand order: host, `.tile`, or core (§4.6).
6. **Epilogue.** Fuse what has no extra operand; count the ports for the rest
   (§4.7).
7. **Template block.** Plain `#define`s, no trailing comments. A generator
   picks the values per layer and de-duplicates (§5).
8. **Verify in layers.** `aiehlc-convtile trace` (seconds), then x86 emulation
   (the scalar branch), then the board with a self-check and `--profiling`.
