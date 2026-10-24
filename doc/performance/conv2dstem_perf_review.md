# conv2dstem performance review — why 31 ms, how far from peak, how to fix

Subject: `src/aietensorop/conv2dstem/conv2dstem.cc` (ResNet-18 stem: 7×7/s2, Cin 3,
Cout 64, 224→112, int8, fused requant+ReLU) on a 4×4 AIE2PS mesh, partition `{2,6}`.

Evidence used (all from the 2026-10-13 build and run):

| Source | What it gives |
|---|---|
| `applog.huaj` (`apppaltest.py` run of `test_conv2d.cc`) | measured wall clock, host share, GOLDEN PASS |
| `aout/worklocal/build/kernel.lst` (chess `darts` listing, `-O2`) | instruction-level cycle count of the core loop |
| `aout/worklocal/conv2d_spatial.cc` | the compiler-resolved constants (`m_rounds`, window sizes) |

## 1. Measured profile

```
PERF first call      31.749 ms   (partition init, staging alloc, weight transpose)
PERF iter 1..5       31.456 – 31.480 ms   mean 31.470 ms
PERF host share      stage_ifm_prepadded 0.168 + collect_ofm_nchwc 0.877 = 1.045 ms (3.3 %)
PERF config+launch   30.425 ms   (= mean − host)
PERF throughput      118.01 MMAC useful / 157.35 MMAC issued per call
GOLDEN PASS 802816/802816
```

The build had no `--profiling`, so the PMU split (kload/bdcfg/coreen/startio/wait_io) is
absent. §2 reconstructs it statically, and it closes to within 1 %.

## 2. Code review of the core: where the 30.4 ms goes

### 2.1 Resolved per-core constants (`conv2d_spatial.cc:91-102`)

`m_rounds = 16` (4 height slabs × 4 width chunks), `buf_sz_a = 4636` (19×244),
`buf_sz_b = 3136` (16 ch × 196), `buf_sz_c = 3136` (7×28×16), one round each.
Per core: 16 × 7 × 28 × 16 = **50 176 outputs** (= 802 816 / 16), each 147 real MACs.

### 2.2 Instruction-level cycle model (`kernel.lst`, one bundle = one cycle)

| Loop (nesting in listing) | Bundles | Trip count / core | Cycles / core |
|---|---|---|---|
| **kh loop, kw×c fully unrolled** (nest 5, `.begin_of_loop` at listing line 1680) — 21 scalar `MAC r8,r8,rA,rB` + 42 `LDA.s8` byte loads | **68** | 7 × 50 176 | 23.88 M |
| per-output epilogue (nest 4: param gather + int64 requant + clamp + store) | **108** (+6 loop setup) | 50 176 | 5.72 M |
| ow / oh / mr loop overhead | 13 / 13 / 46 | 6272 / 112 / 16 | 0.09 M |
| byte-copy loops (`LDA.u8`+`ST.s8`, not pipelined) ×3 | **7 per byte** | (4636+3136)×16 + 3136 = 127 488 B | 0.89 M |
| **Total** | | | **≈ 30.6 M cycles** |

Measured non-host time is 30.43 ms. **At ~1 GHz AIE clock the kernel alone explains
essentially the whole call**: configuration, DMA, ELF load and launch together are in
the noise (< ~0.5 ms). This op is **core-compute-bound on a scalar loop**. (The AIE
clock is not recorded anywhere in the repo; the agreement implies ≈1.0 GHz. Confirm with
`--profiling` — the `wait_io` share should be ≈ 97 %.)

Effective rate: 118.0 M MAC / (16 cores × 30.6 M cycles) = **0.24 MAC/cycle/core**
(4.1 cycles per useful MAC).

### 2.3 Review findings (ordered by cycles lost)

| # | Finding | Code | Cost |
|---|---|---|---|
| R1 | **No vector instructions at all** (see §5). The MAC is the scalar `MAC r,r,r,r` — 1 MAC/cycle at best, vs 256/cycle for `VMAC`/`mmul` | `conv2dstem.cc:447-460` | ~1000× of peak |
| R2 | Loop is **load-bound**, not MAC-bound: 2 byte loads (`LDA.s8`) per MAC on one load unit → 68 cycles for 21 MACs (3.2 cyc/MAC) | same | 78 % of cycles |
| R3 | `if (c < SP_REAL_C)` over a stride-4 channel loop: 3 of 4 lanes useful and the predicate defeats any vectorizer; it exists only because quant params are hidden in the pad channel | `:449-450`, `:205-226` | blocks R1 fix |
| R4 | **Output channel `j` is the innermost of the output loops**: the same 147-byte input patch is re-gathered 16 times (once per channel) | `:428-430` | 16× redundant input loads |
| R5 | **Epilogue re-gathers the 4 quant params byte-by-byte for every output** (16 `LDA.u8` + shifts/ORs) and does a 64-bit multiply — 108 cycles per output, i.e. 74 % of the useful MAC count again | `:470-477` | 5.7 M cycles (19 %) |
| R6 | Window → local **byte copies** at 7 cycles/byte (non-pipelined; window pointer may alias the local array) for A, B and C | `:410-415`, `:420-425`, `:491-496` | 0.9 M cycles (3 %) today; becomes **dominant** once R1 is fixed |
| R7 | Stack VLAs `int8_t slab[buf_sz_a]` etc. — no alignment guarantee, so even a vector rewrite could not use aligned 256/512-bit loads | `:392-409` | blocks R1 fix |
| R8 | Host transposes: `collect_ofm_nchwc` 0.877 ms + `stage_ifm_prepadded` 0.168 ms — work a 4-D DMA BD can do for free | `:627-637`, `:729-740` | 1.05 ms (3.3 % today, ~70 % after kernel fix) |
| R9 | Every call re-programs the mesh (aieSetDevice + partition + config + ELF) | `:884-889` | small today, dominant after R1/R8 |

## 3. GOP/TOPS and distance from theoretical peak

**Work per call**

| | MACs | Ops (2×MAC) |
|---|---|---|
| Useful (147 taps × 802 816 outputs) | 118.0 M | **0.236 GOP** |
| Issued over K = 196 (incl. pad channel) | 157.4 M | 0.315 GOP |

**Achieved**: 0.236 GOP / 31.47 ms = **7.5 GOPS** end to end (7.8 GOPS on the
non-host 30.43 ms).

**Peak of the 4×4 mesh** (int8, `aie::mmul`): 16 tiles × 256 MAC/cycle × 1 GHz =
4.10 TMAC/s = **8.19 TOPS**. (256 MAC/cycle is the AIE-ML figure used in
`doc/design/cost_model_optimizer.md:138`; if AIE2PS's int8 rate is 512, every
percentage below halves and the floor halves.)

| Metric | Value |
|---|---|
| Fraction of peak reached | 7.5 GOPS / 8.19 TOPS = **0.09 %** (≈ 1/1090) |
| Per-core MAC rate | 0.24 of 256 MAC/cycle |
| Compute floor at 100 % peak | 118 M / 4096 MAC/cycle = 28.8 k cycles ≈ **29 µs** |
| Compute floor with K padded to 224 and ~50 % mmul efficiency | ≈ 90 µs |
| DMA floor (≈1.1 MB DDR traffic: 0.8 MB out, ~0.3 MB in with halo; 4 shim columns) | tens of µs, order 50–100 µs (estimate, unmeasured) |
| Today | 31 470 µs (**~350× above a realistic optimized target**) |

## 4. Optimization plan

Ordered by payoff. Estimates are cycle-model projections at 1 GHz, not measurements.

### 4.0 Kernel rewrite (immediately actionable, no compiler change)

| Step | Change | Projected kernel time |
|---|---|---|
| K1 | Hoist the 16 channels' quant params out of all loops into an `int32_t qp[16][4]` decoded once (fixes R5) | 30.6 → ~25 ms |
| K2 | Interchange loops: gather the patch once per (oh,ow), accumulate 16 channels (fixes R4) — still scalar, but loads drop ~2× | → ~8–10 ms (scalar floor 147 MAC × 50 176 ≈ 7.4 M cycles) |
| K3 | **Vectorize with `aie::mmul<4,8,8,int8,int8>`**. Per kh, the 7 taps × 4 ch of one output are **28 contiguous bytes** of the slab (`slab[(oh·2+kh)·244 + ow·8 …+28]`), so A-rows are vector-loadable at stride 8; pad 28→32 gives K = 7×32 = 224. Per round M=196 (7×28), K=224, N=16 → 702 k MAC / 256 ≈ 2.7 k cycles; ×16 rounds ≈ 44 k cycles ideal | → ~0.05–0.1 ms |
| K4 | Vector epilogue: `acc.to_vector` / SRS with per-channel multiplier (or fold mult·2^-shift into one per-channel scale), ReLU via vector max/min | epilogue → ~1 % |
| K5 | Compute directly in acquired window buffers (no copies) or, if a copy is needed, 32-byte vector copies; `alignas(32)` all locals (fixes R6, R7) | removes 0.9 ms |

After K1–K5 the kernel is ~0.1 ms and the call becomes **host + config bound**
(~1.05 ms host + launch overhead) → do §4.1/§4.2.

### 4.1 Compiler (aiehlc / tilinglinalg)

1. **Microkernel generation from the Space.** The compiler already knows M/K/N tile shapes
   from `RowBC_spatial`/`ColBC`/`LtoR_Merge`; it should emit the inner GEMM as an
   `aie::mmul` microkernel (as aietriton's `tl.dot → gemm_body` does) instead of
   compiling a user scalar loop. Choose (r,s,t) from the dtype table; pad K to the
   mmul `s` multiple in the DMA, not in the kernel.
2. **Quant params on their own channel**: a 4th port or an RTP region
   (`aie::param<qparam[16]>`), so the pad channel stays zero and the MAC loop needs no
   `c < 3` mask (unblocks R3). Requires exercising the 4-port path the comment at
   `conv2dstem.cc:192-199` says is unproven.
3. **Aligned, typed window views**: BCF buffer allocation 32/64-byte aligned, and the
   kernel API returns `aie::vector`-loadable views; emit zero-copy acquire→compute→release
   with `pp_depth = 2` so DMA overlaps compute (R6/R7).
4. **DMA-side layout work**: emit the output S2MM BD in NCHW4c order (4-D BD,
   channel blocks of 4) and accept the 3-channel input with a BD that reads 3 B and
   skips — removes both host transforms (R8). Optionally im2col in memtile
   (`Im2col::Dma` + `moplevel = 1`) so the core receives `[M, K_pad]` rows directly.
5. **Loop pragmas**: emit `chess_prepare_for_pipelining` / `chess_loop_range(min,)` on
   generated loops; copy loops currently are not software-pipelined (7 cycles/byte).
6. **Configuration once, launch many** (R9): cache the configured mesh across calls,
   re-arm only shim BDs per call (control-packet AOT window, `control_packet_mode(aot)`);
   this is the persistent-array / `aiecfg` design in `doc/modelsupportevaluate.md` §4.2.
7. **Scale-out after compute is fixed**: VEK385 has 38 columns — a 4×8 or 4×12 mesh is a
   policy change (`HW_COLS`) once the kernel is no longer the bottleneck.

### 4.2 Programming model

1. **Tensor-op kernels, not loop nests.** Let the user declare the op + epilogue
   (`aie::conv2d<KH,KW,S>` / `aie::matmul` + `aie::epilogue::requant_relu`) on the
   Spaces, and let the compiler pick the microkernel. Today the user must hand-write a
   scalar C nest over `input_window_int8*` byte pointers, which is how this kernel
   ended at 0.09 % of peak.
2. **Kernel-visible functions and headers.** "Kernel-visible helpers must be macros"
   (`conv2dstem.cc:212-217`) rules out an `inline` vector microkernel library. Fix the
   extractor to carry `static inline` functions / `#include`d kernel headers into the
   kernel TU.
3. **First-class parameters** (`aie::param`/RTP) for per-channel quant data instead of
   steganography in the pad channel.
4. **Window iterators with alignment and double-buffering in the type**
   (`aie::window_view<int8, 32>`), so zero-copy is the default.
5. **Persistent mesh object.** Allow `aieMesh` to be hoisted/static so a layer is
   configured once and launched N times (`conv2dstem.cc:553-559` documents why it
   cannot be today: the frontend pattern-matches a function-local VarDecl).

### 4.3 Projected end-to-end

| State | Kernel | Host | Config/launch | Call |
|---|---|---|---|---|
| Today | 30.6 ms | 1.05 ms | < 0.5 ms (inferred) | **31.5 ms** |
| + K1–K5 | ~0.1 ms | 1.05 ms | < 0.5 ms | ~1.3–1.6 ms (~20×) |
| + DMA layouts (4.1.4) | ~0.1 ms | ~0 | < 0.5 ms | ~0.3–0.6 ms |
| + configure-once (4.1.6) | ~0.1 ms | ~0 | per-call BD re-arm only | **~0.15–0.2 ms (~150–200×)** |

Validate each step with `./buildtest.sh --profiling` (kload/bdcfg/coreen/startio/wait_io
split) and the golden check — the epilogue must stay bit-exact.

## 5. Does `conv2d_spatial` use vector instructions? — No

From `aout/worklocal/build/kernel.lst` (whole kernel ELF, mnemonic histogram):

* **Zero vector compute or vector load/store**: no `VMAC`/`VMUL`/`VLDA`/`VLDB`/`VST`/
  `VADD`/mmul, and no vector/accumulator register operands (`x*`, `bm*`, `cm*`, `wl*`).
  The only `V` tokens are `NOPV` (empty vector slot in a VLIW bundle).
* The 23 `MAC` instructions are **scalar**: `MAC r8, r8, r23, r30` — 32-bit GPR
  multiply-accumulate on sign-extended bytes from `LDA.s8`.
* The hot loop (listing lines 1680–1766): 68 bundles, 21 MACs, 42 byte loads; the
  vector slot is idle in every bundle.
* Why chess did not auto-vectorize: the `c < SP_REAL_C` predicate on a stride-4 loop,
  int8→int32 widening into a single scalar accumulator, the byte-assembled params, and
  unaligned stack VLAs. Vectorization must be explicit (`aie::mmul` / `aie::mac`), §4.0 K3.

## 6. Implemented: `conv2dstem_vec.cc` (hardware-verified 2026-10-13)

`src/aietensorop/conv2dstem/conv2dstem_vec.cc` is conv2dstem.cc with only the kernel
replaced (host, spaces, windows, DMA, control-plane pragmas byte-identical). Build:
`./buildtest.sh --vec [--skip-trace]` (generates `test_conv2d_vec.cc`).

Implemented from §4.0: K1 (params decoded once into per-lane epilogue vectors),
K2 (one A-tile load feeds both 8-channel N-blocks), K3 (`aie::mmul<4,8,8,int8,int8>`;
each A-tile is 32 contiguous slab bytes at an 8-byte step, K padded 28→32 per kh with
zero weights, pad/param channel zeroed in packed B — no `c<3` mask), K4 (bit-exact
vector requantize, see the kernel header comment: per-channel shift split into two
floor SRS stages), K5 (vector copy into a 256-B-pitch aligned slab; output written
straight into the window).

| | scalar `conv2dstem.cc` | `conv2dstem_vec.cc` |
|---|---|---|
| per-call mean (5 iters) | 31.469 ms | **1.773 ms (17.7×)** |
| host share (stage + collect) | 1.08 ms | 1.09 ms (**62 %** of the call now) |
| config + launch + AIE | 30.4 ms | 0.68 ms |
| GOLDEN | PASS 802816/802816 | PASS 802816/802816 |
| kernel.lst | 0 vector ops, scalar `MAC r` | 27 `VMAC`, 4 `VMUL`, 69 `VLDB`/20 `VLDA`, 22 `VST` |
| effective | 7.5 GOPS (0.09 % peak) | 133 GOPS end to end; ≥ 347 GOPS on the non-host 0.68 ms |

Static listing: the kh loop is 15 cycles for 8 mmul (≈2 cycles/mmul, software-
pipelined); the epilogue is 2 × 62 cycles per 4×16 output block (≈ half of the
kernel's cycles — the emulated int32×int32 multiplies). Next steps, in order of
payoff: move the host transforms into DMA BDs (§4.1.4, −1.09 ms), configure once
(§4.1.6), then cheapen the epilogue (fold mult·2^-shift so one 32×16 multiply
suffices) and measure with `--profiling`.

**Compiler bug found on the way** (fixed; skill **bcfglobalplacement**): the
generated `aieml.bcf` left DMb 0x0..0x3FFFF unreserved, so libme's
`control_rnd`/`control_sat` (which chess reloads into `crRnd`/`crSat` after every
window acquire) were linked at DMb 0x0 — not this tile's memory. First vec run:
726,699 outputs off by +1. Fix: reserve the non-local address space like ADF does
and declare each buffer at its full allocated size.

## 7. conv2d launch under 200 µs (hardware-verified 2026-10-14)

`conv2dstem.cc` now carries the vector kernel (the `_vec` file is the older
snapshot). `test_conv2d.cc` on the board via `apppaltest.py --netload`:
**GOLDEN PASS 802816/802816, conv2d launch 185 µs mean / 182 µs min** (the
`conv2d_spatial` call: cache sync + ELF load + config + array + drain), whole
`conv2d_stem_nchwc()` 0.51 ms (was 1.80 ms; scalar 35.9 ms).

Starting point: the scalar `conv2dstem.cc` (GOLDEN PASS, conv2d launch ~34.9 ms,
whole call 35.9 ms). Each row is the board-measured mean conv2d launch over 5 calls
after that change (tuning took ~3 h 10 min, ~25 build + board cycles):

| # | Change | Launch after | Gain |
|---|---|---|---|
| 0 | Starting point: scalar kernel | ~34,900 µs | – |
| 1 | Vectorized kernel (`aie::mmul<4,8,8>`, from `conv2dstem_vec.cc`) | 590 µs | −34.3 ms |
| 2 | `ctrl_plan_init` snapshot cache (runtime) \* | 509 µs | −81 µs |
| 3 | `#pragma output_tile_major` (one S2MM BD per core, no per-slab host loop) + sinks before sources -- also fixes the race below | 341 µs | −168 µs |
| 4 | Kernel: `mmul<8,8,8>` pairs, aligned A loads, branchless epilogue, host-packed filters (compute 167k → 82k cycles) | 300 µs | −41 µs |
| 5 | AOT control packets + plain shim BD as 1-D AOT site (window batched into one call) | 288 → 270 µs | −30 µs |
| 6 | Deferred launch-arg cache sync (802 KB output invalidate overlaps the array) | 228 µs | −42 µs |
| 7 | Warm kernel-ELF reload (program memory not resent) | 221 µs | −7 µs |
| 8 | `TILE_W` 64 / 232-px staging pitch: kernel reads the DMA window, no slab copy | 214 µs | −7 µs |
| 9 | Shim AXI burst 16 | 210 µs | −4 µs |
| 10 | Non-cacheable input staging + 32-bit word stores | **185 µs** | −25 µs |

\* Step 2 was measured on an intermediate build that still had the output race,
so its gain is approximate; every build from step 3 on is correct, and the final
one is GOLDEN PASS. Step 10 also halved host staging (0.168 → 0.079 ms), so the
whole call went 1.80 ms (vector kernel) → **0.51 ms**.

**The race.** With the dense HWC output the compiler re-programmed the shim
S2MM OOO BDs between 4 height slabs in a host loop. Once the kernel was fast,
cores emitted the next slab before the BDs were re-armed and whole 3136 B tiles
landed at stale addresses (deterministic for a given timing; a 20k-cycle spin per
round made it pass). `#pragma output_tile_major` lays `g_ofm` out
`[mesh row][mesh col][round][7x28x16]` so each core is one linear BD (iter wrap 16),
configured once; the host `collect_ofm_*` de-tiles. Skill **outputtilemajorrace**.

Remaining (measured with `CONV2DSTEM_KCYC 1`): kernel ~157k cycles = compute 82k,
output wait ~23–42k (daisy-chained per-row output merge gives cols 0–1 1/8 of the
link each), core wait for first data ~30k.
