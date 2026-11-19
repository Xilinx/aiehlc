# How to write an AIEDNN kernel: one template, many compiled instances

**AIEDNN** is a library of fast AIE kernels for offloading ML operators: GEMM,
convolution (as GEMM after im2col), and fused variants such as conv + bias +
activation. A model compiler or an application calls an AIEDNN kernel instead
of running the operator on the CPU.

This tutorial explains how to **develop an AIEDNN kernel**: how to write it,
why each design decision is made, and what each one trades off. It also covers
how to **integrate the kernel into the AIEDNN library**, so that it serves every
layer shape of an operator, and how to add a **custom op** (§7).

An AIEDNN kernel is written in aiehlc's kernel-only model. One `__global__`
kernel says, in tensor coordinates, which block each core loads and stores, and
the compiler derives the DMA and routing from it. The goal is a kernel that is
**fast for a whole layer, not just correct for one tile**.

The tutorial is built around the library's one GEMM kernel and its compact teaching form:

| File | What |
|---|---|
| [`src/aiednn/gemm/gemm_kacc_kernel.cc`](../../../src/aiednn/gemm/gemm_kacc_kernel.cc) | **the** AIEDNN `gemm_kacc`: a C++ template (`gemm_kacc<CH, KDIM, PX, KW, AROWS, BROWS, TKW, KCH, NKC, NT, S, TR>`) that every aiegraph build instantiates, one instance per layer shape, all in one source (§3.1) |
| [`src/aiednn/__init__.py`](../../../src/aiednn/__init__.py) | `aiednn.template()` (the kernel text, verbatim) and `aiednn.gemm_kacc_launch()` / `gemm_kacc_targs()` (the launch callee of one instance) |
| [`example/aiednn/tpl_two_instances.cc`](../../../example/aiednn/tpl_two_instances.cc) | the same loop nest in its compact teaching form (`tpl2_gemm<M, N, K, TK, RELU>`, §3), launched as two instances + a self-checking `main()` |
| [`example/aiednn/gemm_kacc_bf16.h`](../../../example/aiednn/gemm_kacc_bf16.h) | host side of the compact form: test data, pre-tiling, fp32 reference, check, phase report |
| [`example/aiednn/gen_variants.py`](../../../example/aiednn/gen_variants.py) | the generator: layer shapes -> walk geometry -> template arguments of `gemm_kacc_kernel.cc` -> one source with every distinct instance + dispatch table |

Until sprint aiednn-tpl (2026-11-19) the compact form was also a stand-alone
example, `example/aiednn/gemm_kacc_bf16.cc` (`gemm_kacc<M, N, K, TK, RELU>`, one
instance). It was removed so that the repository has **one** `gemm_kacc`; the
board numbers of §1 and §8 were measured on it (see git `da30c4d`).

Read [tutorial/aieprogram](../../aieprogram/README.md) first if `load_buffer`,
`program_id` or `{.resident}` are new to you.

---

## 1. Why: what one launch carries decides everything

Every AIE launch pays a **fixed cost** before any MAC happens. That cost covers
kernel load, BD and DMA setup, core reset and synchronisation. So the first
design decision is how much work one launch carries.

The example measures it with the runtime's phase counters (`--profiling`, §8).
One launch of a 256×64×2304 bf16 GEMM on the 4×4 mesh breaks down as:

| Phase | PMU cycles | Share |
|---|---|---|
| `kload` (kernel load) | 41.3 M | 98.0 % |
| `bdcfg` (BD configuration) | 0.48 M | 1.1 % |
| `wait_io` (the DMA and the vector MACs) | 0.31 M | 0.7 % |

The actual work is under 1 % of the launch. If the same GEMM were issued as one
launch per output tile, that is 64 tiles here, every launch would pay the fixed
cost again: about 64× the time for the same result. In that regime the AIE can
easily end up slower than the CPU it is meant to offload.

> **Rule 1.** Push the tile loop down into the core and the DMA. The host
> launches once per layer (or once per network), never once per tile.

---

## 2. The skeleton: three loops and an epilogue

Every AIEDNN GEMM / conv kernel has this shape. The comments say
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

The listing is the **compact teaching form** with one instance, as the former
stand-alone example had it; today the same kernel body is `tpl2_gemm` in
[`tpl_two_instances.cc`](../../../example/aiednn/tpl_two_instances.cc) (two
instances, same `main()` pattern). The library kernel that the networks run is
the same loop nest with the walk geometry spelled out as parameters (§3.1).

```cpp
#include "gemm_kacc_bf16.h"

// ---- this program's instance (gen_variants.py rewrites this block per layer) ----
#define GM 256
#define GN 64
#define GK 2304
#define GTK 64
#define GRELU 1
// ---- end of instance ----

// M rows of A / C (output pixels), a multiple of 4 * 16; N columns of C (output
// channels) = 4 mesh cols x 16; K the reduction (Cin * kh * kw), a multiple of
// TK; TK the K chunk in bf16 elements, a multiple of 8 (the mmul K); RELU the
// fused epilogue.
template <int M, int N, int K, int TK = 64, bool RELU = true>
__global__ void gemm_kacc(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<M, K>> a,
                          aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<N, K>> w,
                          aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<M, N>> c) {
    const int TM = 16, TN = 16;     // the output tile: fixed by the 2 x 2 accumulator blocking below
    const int ntile = M / (4 * TM); // output tiles per core, walked on the core
    const int nkc = K / TK;         // K chunks per tile
    const int kt = TK / 8;          // 8x8 tiles per chunk row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
#if defined(__chess__)
        using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
        MM c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>());
        MM c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
#else
        float acc[TM * TN] = {}; // x86 trace / emulation: the same math, scalar
#endif
        for (int j = 0; j < nkc; j++) {
            const bfloat16 *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const bfloat16 *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
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
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    for (int k = 0; k < TK; k++)
                        acc[m * TN + n] += (float)x[((m / 8) * kt + k / 8) * 64 + (m % 8) * 8 + k % 8] *
                                           (float)v[((n / 8) * kt + k / 8) * 64 + (k % 8) * 8 + n % 8];
#endif
            aie::release(a);
            aie::release(w);
        }
        // The tile is done: epilogue on the fp32 accumulator, one rounding, one write-back.
        bfloat16 *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
#if defined(__chess__)
        alignas(64) float acc[TM * TN];
        const MM *blk[4] = {&c00, &c01, &c10, &c11};
        for (int b = 0; b < 4; b++) {
            alignas(64) float t8[64];
            aie::store_v(t8, blk[b]->to_vector<float>());
            for (int i = 0; i < 64; i++)
                acc[((b / 2) * 8 + i / 8) * TN + (b % 2) * 8 + i % 8] = t8[i];
        }
#endif
        for (int i = 0; i < TM * TN; i++) {
            float o = acc[i];
            if (RELU && o < 0.0f)
                o = 0.0f;
            y[i] = (bfloat16)o;
        }
        aie::commit(c);
    }
}

// One launch; the mesh must be declared in the function that launches.
static void gemm_kacc_run(uint16_t *A, uint16_t *W, uint16_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, AD_HW_ROWS, AD_HW_COLS);
    gemm_kacc<GM, GN, GK, GTK, GRELU><<<mesh>>>(A, W, C);
}

int main() {
    printf("=== AIEDNN gemm_kacc<%d, %d, %d, %d, %d>: C = A . W^T, bf16, one launch ===\n", GM, GN, GK, GTK, GRELU);
    const long na = (long)GM * GK, nw = (long)GN * GK, nc = (long)GM * GN;
    uint16_t *A = (uint16_t *)malloc(na * 2), *W = (uint16_t *)malloc(nw * 2), *ref = (uint16_t *)malloc(nc * 2);
    uint16_t *Ad = (uint16_t *)__Runtime_Alloc(na * 2), *Wd = (uint16_t *)__Runtime_Alloc(nw * 2);
    uint16_t *Cd = (uint16_t *)__Runtime_Alloc(nc * 2);
    ad_init(A, W, GM, GN, GK);
    ad_pretile(A, Ad, GM, GK, GTK, 0); // mmul A operand order
    ad_pretile(W, Wd, GN, GK, GTK, 1); // mmul B operand order
    memset(Cd, 0, nc * 2);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_kacc_run(Ad, Wd, Cd);
    XTime_GetTime(&t1);
    // The launch invalidates C only before the DMA; drop lines refetched while the cores wrote it.
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 2);
    const double ms = ad_ms(t0, t1);
    printf("[gemm_kacc] PERF SUMMARY launch %.3f ms, %.2f GOPS\n", ms, 2.0 * GM * GN * GK / (ms * 1e-3) / 1e9);
    ad_report_phases();
    ad_reference(A, W, ref, GM, GN, GK, GRELU);
    return ad_check(Cd, ref, nc);
}
```

That is the whole file. The host side is the last two functions:

- **`gemm_kacc_run()`** is the launch. It opens the device, partitions a 4×4
  mesh (`{3, 6, 0, 6}` is the array region, `AD_HW_ROWS` × `AD_HW_COLS` the
  mesh), and launches the instance `gemm_kacc<GM, GN, GK, GTK, GRELU>` once.
  aiehlc turns that one line into the kernel load, the DMA / routing setup, the
  core start and the wait. The `aieMesh` must be declared in the function that
  launches.
- **`main()`** is the caller:
  1. allocates the logical tensors with `malloc` (for the reference) and the
     device buffers with `__Runtime_Alloc` (what the DMA reads and writes);
  2. fills A and W with test data (`ad_init`) and stores them pre-tiled in
     `mmul` operand order (`ad_pretile`, §4.6): A as the mmul A operand, W as
     the B operand;
  3. times the launch with `XTime`;
  4. invalidates the cache over C **after** the launch: the generated launch
     invalidates the output only before the DMA (§9);
  5. prints `PERF SUMMARY` and the phase split (`ad_report_phases`, filled only
     in a `--profiling` build);
  6. computes the fp32 reference (`ad_reference`) and compares every output
     (`ad_check`), printing `PASS` / `FAIL`.

The helpers are in
[`gemm_kacc_bf16.h`](../../../example/aiednn/gemm_kacc_bf16.h). A library
caller keeps steps 1, 2 (with real data) and the launch, and drops the test
data and the check.

Two things that look odd and are deliberate:

- **`#if defined(__chess__)`.** The vector code (`aie::mmul` with an `accfloat`
  accumulator) only exists for the AIE compiler. The x86 emulation that the
  kernel analysis traces has `bfloat16` but no `accfloat` / bf16 `mmul` yet. So
  the `#else` branch computes the same result in scalar code. The analysis
  checks *which blocks* the kernel touches, and both branches touch the same
  ones.
- **The odd index arithmetic** (`(m / 8) * kt + k / 8) * 64 + ...`). The
  blocks arrive pre-tiled in `mmul` operand order (§4.6), and this is how one
  element of a pre-tiled block is addressed.

### 3.1 The library kernel: `src/aiednn/gemm/gemm_kacc_kernel.cc`

The kernel the aiegraph builds launch (UNet conv / linear / attention, the CNN
chunks) is the same decision set taken further: **C-stationary with the fp32
accumulator in the c window** (so K can be any length, accumulated across
chunks in L1), operands pre-tiled by the host as bf16 pairs in `int32` words,
and the walk geometry fixed by template parameters:

```cpp
template <int CH, int KDIM, int PX, int KW, int AROWS, int BROWS, int TKW, int KCH, int NKC, int NT, int S, bool TR>
__global__ void gemm_kacc(aie::port<input_window_int32 *, aie::IFM, aie::shape<AROWS, KW>> a_port,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<BROWS, KW>> b_port,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<AROWS, BROWS>> c_port);
```

| Parameter | Meaning | In the compact form |
|---|---|---|
| `CH`, `KDIM`, `PX` | channels (N), K in bf16, pixels (M) of the instance | `N`, `K`, `M` |
| `KW` | padded K in int32 words (2 bf16 each) | `K` |
| `AROWS`, `BROWS` | rows of the a / b ports (`CH` / `PX`; swapped when `TR`) | `M`, `N` |
| `TKW`, `KCH`, `NKC` | K chunk in words / in bf16, chunks per tile | `TK`, `K / TK` |
| `NT`, `S` | 16-row a tiles per core, 16-row b sub-tiles per core | `M / (4·16)`, 1 |
| `TR` | transposed: a = X, b = W (the mmul's left operand stays W) | — |

Nobody writes these numbers by hand. The geometry comes from
`aiegraph.cnn_chunk.kacc_geom(ch, k, px)` (which also *refuses* shapes the
kernel cannot cover, with the reason), and
`aiednn.gemm_kacc_launch(ch, k, px, geom)` turns it into the launch callee, e.g.
`gemm_kacc<64, 576, 256, 288, 64, 256, 16, 32, 18, 1, 4, 0>`. A build reads the
template once (`aiednn.template("gemm/gemm_kacc_kernel.cc")`), writes one
`<callee><<<mesh>>>(...)` per distinct layer shape in the same source, and aiehlc
compiles one kernel per instance (§5). The epilogue (bias, activation,
residual) is not in this kernel: the aiegraph host applies it on the fp32 c
window (`src/aiednn/elementwise/`), §4.7 explains the trade-off.

---

## 4. The decisions and their trade-offs

### 4.1 Output tile and register blocking (TM × TN)

**What.** Each core computes a 16×16 output tile at a time, as a 2×2 grid of
`aie::mmul<8,8,8>` accumulators (`c00 … c11`). Each A tile is used against two
B tiles and each B tile against two A tiles before it leaves the registers.

**Why.** A Tm×Tn×Tk tile does Tm·Tn·Tk MACs on Tm·Tk + Tk·Tn input elements.
A larger output tile reuses every loaded byte more often. Register blocking
(several accumulators sharing each loaded operand) is the standard way to raise
that reuse without growing the buffers.

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

**Trade-off and options:**

| Distribution | How | Good for |
|---|---|---|
| **block** (this example) | core (pm, pn) takes `ntile` consecutive tiles: `m0 = pm·ntile·TM + t·TM` | a regular DMA walk (one affine sequence per port); what the kernel analysis can route |
| strided | `for (u = core_id; u < U; u += P)` | load balance when U is not a multiple of P |

The whole layer waits for the slowest core. **Pick TM/TN so U is a multiple of
P.** `gen_variants.py` pads M to a multiple of 4·TM for exactly this reason.

### 4.3 K: accumulate on the core, and how big a chunk (TK)

**What.** K is streamed in chunks of TK. The fp32 accumulator stays in the
`mmul` registers across all K/TK chunks and is rounded to bf16 **once**, after
the last chunk.

**Why one rounding.** bf16 keeps 8 significant bits. Rounding the partial sum
to bf16 after every chunk adds a rounding error per chunk, and with K/TK chunks
those errors accumulate. Long-K layers (thousands of chunks in large convs) lose
visibly more accuracy than with a single rounding at the end.

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
end to end. With a **warm** runtime (kernel resident between launches, BSS
not re-written) the load term disappears, and the larger TK wins.

> **Rule 2.** There is no universally best TK. Choose the largest one that
> fits, *for the launch model you actually run*, and measure the launch split
> (`--profiling`).

### 4.4 Who stays on the core: A-, B- or C-stationary

| Dataflow | Resident on the core | Streamed | Reused | Fits |
|---|---|---|---|---|
| **C-stationary** (this example) | the TM×TN accumulator | A and B chunks along K | C over all K/TK chunks, written once | long K (convs: K = kh·kw·Cin, often thousands) |
| B-stationary (weights) | a TK×TN weight block (`{.resident}`) | activations along M | weights over M/TM tiles | small weights, many pixels |
| A-stationary (input) | a TM×TK activation block | weights along N | activations over N/TN tiles | few rows, many output channels (small-M Linear) |

**How to choose.** A is reused N/TN times, B is reused M/TM times, and C is
reused K/TK times. Keep the most-reused one that still fits.
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
([low-level schedule](../../aieprogram/06_gemm_optimization/1_low_level_schedule.md), [conv scheduling](../../aieprogram/06_gemm_optimization/3_conv_scheduling.md)).

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
| **the host, once** (this example) | `ad_pretile()` stores every 16×TK chunk of A and W in operand order inside the same tensor; the DMA moves it unchanged | host time (once per weight set; per call for activations) |
| the shim DMA: `{.tile = {tr, tc}}` | the block arrives panel-major, so each tr×tc tile is contiguous (3 BD dims) | only blocks that are **whole rows** of the tensor, and only re-orders whole 32-bit words: it cannot transpose int8 or bf16. The analysis refuses `.tile` on a K-chunk block |
| the core | `aie::transpose` / shuffles per chunk | vector cycles every chunk |

Here A and W are K-chunked (16×TK of a 2304-wide row), so `.tile` cannot apply.
W also needs a K×N transpose. The host pre-tiles both.

### 4.7 Fusion: the epilogue

**What.** Element-wise work applied to the accumulator before its single
write-back: bias, activation, residual add, scale. Here it is one template
parameter, `bool RELU`:

```cpp
float o = acc[i];
if (RELU && o < 0.0f)   // RELU is a template parameter: the branch folds away when false
    o = 0.0f;
y[i] = (bfloat16)o;
```

**Why.** A fused epilogue saves a DDR round trip: the unfused version writes C
and another op (often on the CPU) reads it back, applies the element-wise work,
and writes it again.

**Trade-offs.**
- **Every extra operand is an extra port.** Bias (per output channel) or a
  residual (same shape as C) is another `aie::port`. A port is part of the
  kernel's signature, so a fused bias+residual kernel is a different kernel
  from a plain one.
- **Order matters.** bias → ReLU → +residual is not +residual → ReLU. The
  epilogue must follow the model graph exactly.
- A policy chain keeps one source for many epilogues (illustration):

  ```cpp
  struct Bias { static float f(float x, const EpiArgs &e) { return x + e.bias[e.co]; } };
  struct Relu { static float f(float x, const EpiArgs &)  { return x > 0.f ? x : 0.f; } };
  template <class... Ops> struct Chain {
      static float f(float x, const EpiArgs &e) { ((x = Ops::f(x, e)), ...); return x; }
  };
  using Epi_CBR = Chain<Bias, Relu>;   // conv + bias + relu
  ```

  In aiehlc today the selection that is proven on the board is an integral
  template parameter, as in the example (`bool RELU`). aiehlc also substitutes
  `typename` parameters, but a policy type like `Chain<Bias, Relu>` also has to
  be visible to the AIE compile of the kernel, and this example does not
  exercise that.

### 4.8 Data type

The example is all bf16: `input_window_bfloat16` / `output_window_bfloat16`
ports, an fp32 `accfloat` accumulator, and one rounding to bf16. Two current
limits shape the ports:

- **One element type per kernel.** All ports are bf16, so C is bf16. A kernel
  that needs fp32 out can declare every port `int32` and pack two bf16 per
  input word. That works, but it pushes packing onto the host.
- int8 and bf16 use different vector instructions and have different best
  tiles. A type change is a new instance at least, and often a new template.

---

## 5. Templating: compile-time parameters, a generator, a dispatch table

**Why compile time.** `aie::shape<R, C>` must be a constant. The compiler turns
it, together with the block sizes and loop bounds, into the DMA windows, the
buffer sizes and the BD walk. So a parameter set is one compiled kernel plus
its DMA configuration, and it cannot be chosen at run time.

**Why not one run-time-parameterised kernel?**
- Loop bounds and tile sizes become variables. The compiler can no longer
  unroll or schedule vector instructions for fixed sizes, and the inner loop
  slows down.
- DMA windows must be sized for the largest shape. Small shapes still occupy
  the full buffers, and large ones may not fit at all.
- The usual fallback, one fixed tile for every shape looped by the host, is
  Rule 1 broken: one launch per tile.

**How, in this repo.** The kernel is a real C++ template on a `__global__`,
and the launch picks the instance:

```cpp
template <int M, int N, int K, int TK = 64, bool RELU = true>
__global__ void gemm_kacc(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<M, K>> a, ...);

gemm_kacc<GM, GN, GK, GTK, GRELU><<<mesh>>>(A, W, C);   // or gemm_kacc<256, 64, 2304><<<mesh>>>
```

aiehlc reads a kernel's port shapes and body as **source text** in several
places: the kernel analysis, the port parser, and the kernel extraction for the
AIE compiler. Each needs `aie::shape<M, K>` as numbers. So aiehlc instantiates
the template on the text first, before any of them runs
(`src/llvm/convautotile/kernel_template.cpp`):

- Every launch `name<args><<<mesh>>>(...)` selects an instance. Omitted trailing
  arguments take the defaults, which may use earlier parameters.
- An integer argument is folded to a literal using the source's `#define`s
  (`GTK * 2`, `true`, and `2 * 128` all work). Every parameter token in the
  kernel's signature and body is then replaced by the value, and the
  `template <...>` header is removed.
- With one instance, the kernel keeps its name. With several, each instance gets
  its own name (`gemm_kacc__256_64_2304_64_1`), each launch is rewritten to its
  instance, and every instance is compiled into its own kernel ELF in the same
  host ELF.

The build log names every instance:

```
[KernelTemplate] instance gemm_kacc<256, 64, 2304, 64, 1> -> gemm_kacc
```

Misuse is an error, never a silent default: a template kernel that is never
launched, a launch with no template arguments, too many arguments, or a missing
argument with no default. An instance that the kernel cannot cover is caught by
the analysis, the same way as for a plain kernel. For example,
`gemm_kacc<200, ...>` of the compact form reports `512 output element(s) are
never stored`, because 200 is not a multiple of 4·16; for the library kernel,
`NT = 2` on a 64-row a port reports `c_port block 0 on core (2,0): block dim 0
covers [64, 80) outside the tensor extent 64` (the `template:` cases of
`src/llvm/convautotile/unitest/kernel_trace_cases.py` run all of these on
`gemm_kacc_kernel.cc`).

**Parameters:** integral non-type (`int`, `unsigned`, `long`, `bool`, `size_t`,
…) and `typename` / `class`, each with an optional default. Not supported:
packs, template template parameters, and a template launched only through a
pointer.

So one source covers every shape. What is left to the library is *picking* the
arguments per layer. `gen_variants.py` does what the aiegraph builds do: the
walk geometry per layer (`kacc_geom`, which refuses what the kernel cannot
cover), the launch callee (`aiednn.gemm_kacc_launch`), de-duplication, and ONE
source that holds the template once and launches every distinct instance:

```python
for i, layer in enumerate(layers):                # gen_variants.py, abridged
    ch, k, px = layer["n"], layer["k"], layer["m"]
    g = kacc_geom(ch, k, px)                      # K chunk, tiles per core, padded K; or a refusal
    if (ch, k, px) not in variants:               # de-duplicate: same arguments = same kernel
        variants[(ch, k, px)] = (len(variants), aiednn.gemm_kacc_launch(ch, k, px, g))
    table.append({"layer": i, "instance": variants[(ch, k, px)][0], ...})   # -> dispatch.json
src = aiednn.template("gemm/gemm_kacc_kernel.cc") + host   # host: one <callee><<<mesh>>> per instance
```

```
$ python3 example/aiednn/gen_variants.py --out /tmp/variants
layer 0: M=256 N=64 K=576 -> instance 0 gemm_kacc<64, 576, 256, 288, 64, 256, 16, 32, 18, 1, 4, 0> (K chunk 32, 18 chunks, padded K 576)
layer 1: M=256 N=64 K=576 -> instance 0 gemm_kacc<64, 576, 256, 288, 64, 256, 16, 32, 18, 1, 4, 0> (K chunk 32, 18 chunks, padded K 576)
layer 2: M=256 N=128 K=576 -> instance 1 gemm_kacc<128, 576, 256, 288, 128, 256, 16, 32, 18, 2, 4, 0> (K chunk 32, 18 chunks, padded K 576)
layer 3: M=64 N=128 K=1152 -> instance 2 gemm_kacc<128, 1152, 64, 576, 128, 64, 32, 64, 18, 2, 1, 0> (K chunk 64, 18 chunks, padded K 1152)
4 layers -> 3 instances in /tmp/variants/gemm_kacc_variants.cc (dispatch.json: layer -> instance)
$ build/aiehlc-convtile trace /tmp/variants/gemm_kacc_variants.cc
[KernelTemplate] instance gemm_kacc<64, 576, 256, 288, 64, 256, 16, 32, 18, 1, 4, 0> -> gemm_kacc__64_576_256_288_64_256_16_32_18_1_4_0
[KernelTemplate] instance gemm_kacc<128, 576, 256, 288, 128, 256, 16, 32, 18, 2, 4, 0> -> gemm_kacc__128_576_256_288_128_256_16_32_18_2_4_0
[KernelTemplate] instance gemm_kacc<128, 1152, 64, 576, 128, 64, 32, 64, 18, 2, 1, 0> -> gemm_kacc__128_1152_64_576_128_64_32_64_18_2_1_0
OK: every load_buffer / store_buffer matches the DMA
```

Each instance becomes its own kernel ELF inside one host ELF, and the block
descriptors are named per instance (`aiehlc_desc_<instance>_<port>`). This is
how the networks are built: the UNet build launches 38 `gemm_kacc` instances
from one source. Two instances of the compact form in one ELF were verified
on the board (2026-11-15): 16,384 / 16,384 bit-exact each.

The host then launches each layer with its variant, once per layer. Consecutive
layers that share a variant need not reload the kernel (warm launch). On a real
network the de-duplication matters: layers repeat shapes, so dozens of layers
typically collapse to a much smaller set of variants.

> The generator's cost model above maximises TK. On the cold-launch runtime of
> §4.3 that is the *slower* choice, because the buffers are reloaded per
> launch. A real generator must take the launch model into account, or better,
> be fed measured launch splits.

---

## 6. Why one op needs many kernels

A single fixed-tile kernel makes every shape **computable**; instances make
every shape **fast**.

| Level | For the GEMM / conv operator | Who makes it |
|---|---|---|
| template (source) | 1: `gemm_kacc` (`src/aiednn/gemm/gemm_kacc_kernel.cc`). One more per *structure*: B-stationary, direct conv, … | written by hand |
| parameter sets (geometry, K chunk, layout, type) | one per distinct layer shape (demo: 4 layers → 3) | the generator, from the shapes |
| compiled kernels | one ELF per parameter set | aiehlc |
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

## 7. Integrating a kernel into AIEDNN, and adding a custom op

A kernel joins the library as **a template plus the means to instantiate it for
any layer shape**, not as a single compiled binary. Today the integration points
in this repo are the template source, its host-side support, and the generator
with its dispatch table. AIEDNN has no packaged runtime API yet, so these
files *are* the interface.

### What a library kernel delivers

| Deliverable | In the example | Why the library needs it |
|---|---|---|
| the template | `src/aiednn/gemm/gemm_kacc_kernel.cc`: the kernel as a C++ template (`gemm_kacc<CH, …, TR>`), instantiated by aiehlc at each launch | one source covers every shape of the op |
| the host contract | the operand layout and epilogue of the aiegraph host (`src/aiednn/conv2d/kacc_conv_host.cc`); for the compact form `gemm_kacc_bf16.h` (`ad_pretile`, `ad_reference`, check) | callers must lay data out the way the kernel reads it, and every instance must be checkable |
| the instantiation rules | `aiegraph.cnn_chunk.kacc_geom` + `aiednn.gemm_kacc_launch` (used by `gen_variants.py` and the aiegraph builds): a layer shape → a legal, de-duplicated argument set, or a refusal with the reason | the library serves shapes it has never seen, and refuses the ones the template cannot cover instead of mis-computing them |
| the dispatch entry | `dispatch.json`: layer → variant + parameters | how the caller finds the kernel for each layer |
| the evidence | analysis OK, x86 emulation, board self-check, `--profiling` split (§8) | a library kernel must be proven correct and fast, not assumed so |

### Adding a custom op

A custom op is anything the existing templates do not compute, for example
conv + bias + residual add + scale, or a GEMM with an unusual activation. Work
down this list and stop at the first step that covers the op:

1. **Express it as an existing template plus an epilogue.** Most element-wise
   additions after a GEMM or conv are epilogues (§4.7). If the op needs **no
   new operand** (a different activation, a constant scale), it is a new
   template parameter (an `int` / `bool` switch) in the template's epilogue,
   plus a new argument in `pick()`.
2. **New operand → new instance with a new signature.** A per-channel bias, a
   residual tensor or per-channel scales each need an extra `aie::port`. The
   port list is the kernel's signature, so this is a new instance (or a small
   new template) with:
   - the extra `load_buffer` per output tile (a bias block is TN values, a
     residual block is TM×TN);
   - the extra operand in the host contract (layout, reference);
   - an epilogue order that matches the model graph exactly (bias → ReLU →
     +residual is not +residual → ReLU).
3. **New loop structure → new template.** If the op needs a different dataflow
   (B-stationary, a direct conv without im2col, attention-style reductions),
   write a new template following §2–§5. It gets its own generator rules and
   dispatch entries.
4. **Prove it.** The same gates as any library kernel (§8): `aiehlc-convtile
   trace` OK; x86 emulation of the scalar branch against the reference; board
   self-check (bit-exact with exactly representable test data, a tolerance
   otherwise); the `--profiling` split to confirm the launch is not dominated
   by something the kernel can avoid.
5. **Register it.** Add the op's shapes to the generator's rules (what it
   accepts, how it pads, what it refuses) and its entries to the dispatch
   table.

The decision rules of §6 decide between steps 1–3: whether the change is a
parameter, a new instance, or a new template.

## 8. Build, check, run

```bash
# 1. the analysis (seconds) of the library kernel, every `template:` case (instances, defaults, misuse)
python3 src/llvm/convautotile/unitest/kernel_trace_cases.py build/aiehlc-convtile "template:"
#   [KernelTemplate] instance gemm_kacc<64, 576, 256, 288, 64, 256, 16, 32, 18, 1, 4, 0> -> gemm_kacc
#   kernel_trace_cases: 9 passed, 0 failed

# 2. the compact form, two instances, self-checking: build (aout/ lands in the cwd) and run on the board
build/aiehlc-convtile trace example/aiednn/tpl_two_instances.cc
source script/aiehlc.sh --aie-version 5 --runtime-source-file $PWD/example/aiednn/tpl_two_instances.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf
#   [tpl2_gemm<256, 64, 2304, 64, 1>] launch ... ms, then ad_check's line for its 16384 outputs
#   [tpl2_gemm<128, 64, 2304, 64, 0>] launch ... ms, then ad_check's line for its 8192 outputs

# 3. where the launch time goes: rebuild with --profiling
source script/aiehlc.sh --aie-version 5 --profiling --runtime-source-file $PWD/example/aiednn/tpl_two_instances.cc

# 4. your layer shapes: one source with every instance of the library kernel, then trace / build it
python3 example/aiednn/gen_variants.py --out /tmp/variants
build/aiehlc-convtile trace /tmp/variants/gemm_kacc_variants.cc
```

Results on palmyra (AIE2PS, 4×4 mesh), 2026-11-14, of the former stand-alone compact example (`gemm_kacc<M, N, K, TK, RELU>`, git `da30c4d`):

| Instance | Shape (M×N×K), TK | Result | Launch |
|---|---|---|---|
| template default | 256×64×2304, 64, ReLU | 16,384 / 16,384 bit-exact, 4/4 runs (incl. the C++-template source, 2026-11-15) | 33.0 ms |
| two instances in one ELF (`<…, 1>` and `<…, 0>`) | 256×64×2304, 64, ReLU and no ReLU | 16,384 / 16,384 bit-exact each (2026-11-15) | 33.1 ms (first) |
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

## 9. Pitfalls (each one cost a build or a board run)

| Symptom | Cause | Fix |
|---|---|---|
| `the core window is 256 elements x 144 round(s) ..., but the kernel's blocks are 1024 elements x 144` | aiehlc divided **every** input's window by the M-round count; only A holds M strips | fixed in aiehlc. The build error is the `#routing.kernel_block` check doing its job: without it, a silent hang |
| board `wait_io TIMEOUT` on the input shims, ch 1 | the A strip BD walks all K chunks in 3-D but was started K-rounds times with an element-count length | fixed in aiehlc (only where the BD's length is not already the whole walk: `simplematmul2` relies on the other form) |
| `{.tile / .resident} need blocks that are whole rows of the tensor` | `.tile` on a K-chunk block: the DMA would have ignored it | pre-tile on the host (§4.6) |
| `use of undeclared identifier` for a macro used by the kernel | it was defined inside `#ifndef`; aiehlc copies only plain `#define`s into the kernel | plain `#define`s (the instance block, any constants the kernel reads) |
| `template kernel gemm_kacc ... is never launched` / `launched without template arguments` / `N template arguments, the kernel has M` / `no argument and no default for template parameter K` | the launch does not select exactly one instance | launch as `gemm_kacc<args><<<mesh>>>(...)`; defaults fill only trailing parameters |
| "static bounds analysis: not evaluable" | a `// comment` on a `#define` line | comments on their own lines |
| a few hundred outputs = the host's `memset` value on ~1 run in 6 | the generated launch invalidates the output only *before* the DMA | `Xil_DCacheInvalidateRange(C)` after the launch (in the example), until the generated launch invalidates outputs itself |
| bf16 kernel "does not compile for tracing" | the x86 emulation has `bfloat16` but no `accfloat` / bf16 `mmul` | `#if defined(__chess__)` vector path + scalar `#else` (§3) |
| N wider than 4·16 (compact form) | the compact template gives each mesh column one 16-wide column of C | the library kernel walks S sub-tiles per core; `kacc_geom` refuses what it cannot cover |

---

## 10. Checklist for a new AIEDNN kernel

1. **Unit of work.** One launch per layer. Estimate fixed cost × launches
   against the CPU time for the same work *before* writing the kernel.
2. **Dataflow.** Pick the stationary operand from the reuse counts (§4.4).
3. **Register blocking and tile.** A multiple of the `mmul` shape; U a
   multiple of the core count (§4.1, §4.2).
4. **K.** fp32 accumulation, one rounding; the largest TK that fits the port
   region *and* suits the launch model, measured (§4.3).
5. **Layout.** Who produces `mmul` operand order: host, `.tile`, or core (§4.6).
6. **Epilogue.** Fuse what has no extra operand; count the ports for the rest
   (§4.7).
7. **Template parameters.** The shape, chunk and switches as C++ template
   parameters; the launch selects the instance (§5). A generator picks the
   arguments per layer and de-duplicates. Macros the launch uses: plain
   `#define`s, no trailing comments.
8. **Verify in layers.** `aiehlc-convtile trace` (seconds), then x86 emulation
   (the scalar branch), then the board with a self-check and `--profiling`.
9. **Integrate.** Host contract, generator rules (accept / pad / refuse) and
   dispatch entries, so the library can serve every shape of the op (§7).
