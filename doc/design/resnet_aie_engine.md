# ResNet-18 on the AIE: the persistent conv engine

**Result (board, VEK385 / palmyra, 2026-10-15): 2.81 ms per image**, float image
→ 1000 logits, **bit-identical to TVM's int8 graph** (logits compared as raw
float bits, 10/10 runs, top-5 258 279 259 257 270). Every conv runs on the AIE: the stem, all
16 3×3 convs and all 3 1×1 downsamples. Only input quantize and avgpool + fc stay on the A78.
Starting point: **1270 ms** (TVM C on the A78, stem on the AIE). Code:
`src/aietensorop/resnetengine/` (`gen_engine.py`, `eng/`, `build.sh`).

| stage | where | time |
|---|---|---|
| quantize float → int8 `[230][232][4]` (TVM call 00) | A78, NEON | 0.16 ms |
| stem 7×7 s2 + maxpool + 16 conv layers (TVM calls 01..37) | AIE, 17 engine layers | 2.59 ms |
| avgpool + flatten + dense (TVM calls 38..40) | A78, NEON SDOT | 0.08 ms |
| **total** | | **2.81 ms** |

```bash
# 1. TVM artifacts + per-call dumps (once): deploy_flow.py, then in arm_build/
make local TRACE=1 TRACE_HEX_PARAMS=0 && ./main_local.elf
# 2. engine description: weights/params/headers blob + host table, numpy-proven bit-exact
python3 src/aietensorop/resnetengine/gen_engine.py --out worklocal/resnetengine
# 3. kernel ELF + host ELF, then the board
bash src/aietensorop/resnetengine/build.sh eng
python3 script/test/apppaltest.py -y --netload /tmp/claude/re_eng/aout/worklocal/build/host
```
The run verifies every layer twice, isolated (TVM's golden input) and chained, and
then times 10 images from the float image to logits. It prints `PERF SUMMARY best .. ms`
and `ENG PASS`.

## Runtime modes: persistent vs eager (2026-10-22)

`engine_set_mode(e, eager, layer_major)` in `eng/host.cc` picks the mode at run
time; `k_modes[]` lists the configurations `main()` runs, and the `volatile int
eng_modes` bit mask (default 1 = persistent only; `-DENG_MODES_DEFAULT=<mask>`, e.g. 2 = eager, 3 = both) selects among them:
persistent (one `eng` ELF loaded once), eager (kernel reloaded before every
layer, variants `engstem`/`engconv` from `eng/kernel_variants` -- one ELF with
every feature overflows AIE2PS's 16 KB program memory), and eager layer-major
(B images per layer, B = 1/4/8/16/32: one reload + one memtile-cached weight
fetch serve B images; residuals through DDR, `ENG_F_RESX`/`ENG_F_DOWNOUT`;
7x7 layers row=image, `ENG_F_RI`; per-unit output re-arm gated by `ENG_F_GO`).

**Kernel reverted to the 2026-10-20 23:31 engine** (pre-"ep2"): eager
layer-major B=32 ran ~1.61 ms/image there and never hung. Every faster kernel
since (two-chain epilogue, host-precomputed epilogue constants, 16-bit residual
combine: 1.364 ms/image) hung ~1 run in 3 in the residual layers: a
control-packet task push to an input shim on output columns 3/4 -- and every
later write of that packet -- is lost (skill **shimctrlpushloss**). Suspected
cause: the next unit's input writes are sent right after the go, i.e. into the
output burst the go releases on those same shims; untested fix: send them in
the same batch as the output re-arm, before the go.

---

## 1. Why not one aiehlc library per conv (the `allconv` lesson)

The `allconv` branch offloads every conv through one generic 256×64×256 GEMM tile
(`src/aietensorop/convgemm`) with host im2col. That is 952 launches and **13.0 s per
image**. It is bit-exact, and it proved the parts this engine reuses:

- TVM-tail transplant (`aie_conv_lib.transplant_tail`), now regex-parsed epilogue
  constants in `gen_engine.conv_info`;
- per-layer verification against TVM (`verify_layer`), now the numpy model plus per-layer goldens;
- the graph-driver swap.

Its structure could not reach 4 ms, for three reasons:

- **Launch cost:** each launch costs ~50 µs of setup even after conv2dstem's tuning
  (`doc/performance/conv2dstem_perf_review.md` §7).
- **Isolation:** each aiehlc library has its own runtime, partition cache and `routing()`,
  so libraries cannot alternate on the same columns.
- **Glue on the CPU:** im2col, repacks and the pad/add ops between convs (5.6 ms) stay on the A78.

So the engine keeps **one fabric, one kernel ELF loaded once, and all glue on the AIE**.

## 2. Architecture

* **Fabric**: conv2dstem's routing (`fabric/routing.cc`, partition `{2,6}`, 4×4 mesh on
  cols 3..6 / rows 3..6, control spine on col 2), described in `engine_fabric.h`.
  - Input: shim (k,0) MM2S ch1 → broadcast to mesh row k−1.
  - Weights: MM2S ch0 → broadcast to mesh column k−1.
  - Output: core MM2S ch0 packets → one shim S2MM per mesh row, out-of-order BDs.
* **Kernel** (`eng/kernel.cc`): a persistent core program. Per layer it reads a
  256 B header from the weight stream (`eng_desc.h`) and programs its own three tile
  DMAs (one BD each, *iteration* addressing over ring slots — `aie_kc_encode_bd_nd`).
  It then computes and waits until its output has drained. It is network-agnostic:
  everything layer-specific arrives in the stream.
* **Host** (`eng/host.cc`): per layer, re-arm 24 shim BDs over the
  **control-packet plane** and wait for the output. No ELF reload, core reset or re-route.
* **Activations** in DDR use **L8** = `[H+2][C/8][WST][8]` int8 (value − 128), border
  −128, with WST = roundup8(W+2):
  - An `mmul<8,8,8>` A tile (8 px × 8 ch) is one aligned 64 B load.
  - A C tile is exactly one L8 block, so each layer writes the next layer's padded input directly.
* **Mapping**: core (mesh row r, mesh col c) owns output rows of mesh row r and, in pass
  p, channel blocks `(4p+c)·2 + {0,1}`. Pass p of all four columns is then one contiguous
  64-channel group, which keeps the input chunk, the residual slice and the output packet contiguous.
* **Per layer**:
  - `P = Cout/64` passes of 16 oc per core;
  - `Q = Cin/64` ic chunks;
  - weight chunks of 16 oc × 64 ic × 9 taps (+ params), double-buffered;
  - partial sums in core memory when Q > 1.

## 3. Making every conv a single engine layer (exactness first)

Before writing AIE code, `gen_engine.model_network` runs a numpy model of the
engine's integer semantics over all 16 layers, chained from TVM's maxpool output,
and compares it with TVM's dump of every call. It passed before the first board run, and it
pins down the four decisions that make the layer set small:

1. **Every epilogue is one formula**: `v = clamp(requant(acc) + c0, 0, 255)`, and for the
   eight residual layers `out = clamp(Q(v − szp, sl, sM, ss) + Q(res − rzr, rl, rM, rs), 0, 255)`,
   where `Q(y,l,M,s) = ((y≪l)·M + 2^(s−1)) ≫ s`. TVM's eight fused tails differ only
   in the constants, parsed from the generated C.
2. **Every inter-layer tensor is 8-bit**. TVM's int32 intermediates are either
   already clamped to 0..255, or are a scalar function of a clamped value. Example: conv B's
   output is `Q(clamp(R+92) − zp)`, so the engine stores `clamp(R+92)` and lets the
   consumer apply the producer's `Q`. Layer 37's `Q(v≪2)` becomes a 256-entry table in avgpool.
3. **Downsamples are a 10th tap.** A 1×1 s2 conv reads exactly the centre tap's A tile
   of the 3×3 s2 conv on the same input, so conv1 of a downsample block computes
   both with the same loads. B+C and F+G tail pairs combine in conv2 (the sum is symmetric).
4. **Residuals never touch DDR.** Conv1 of an identity block keeps its core's slice of x
   from the input centre rows (`GRAB`). Conv1 of a downsample block keeps `v_d`. Conv2
   reads them from core memory (`COMB`). Producer and consumer have the same core mapping.

The stem is one more layer kind (`ENG_F_STEM`). It runs 7×7 s2 over `[230][232][4]`:
stride 2 px × 4 B makes each (kh, kw-pair) A tile 64 contiguous bytes. The 3×3 s2
maxpool is fused in. Each mesh row recomputes one halo stem row, and the core emits only
the 56×56 maxpool rows: 4× less DDR traffic, and no CPU maxpool (TVM: 2.4 ms).

## 4. Optimization steps, measured

Each step was measured on the board; per-layer times are host-measured (arm + wait).

| # | step | engine (16 convs) | why it helped |
|---|---|---|---|
| 0 | TVM C on the A78 (19 convs) | 1217 ms | — |
| 1 | E1: one 3×3 layer on a persistent kernel | layer 04: 76 ms → 138 µs | vector `mmul<8,8,8>`, aligned loads + `shuffle_down_fill`, exact 2-stage requant |
| 2 | generic engine, all 16 layers, MMIO shim BDs | 3.56 ms | header-driven kernel, chunked weights, residual on-core |
| 3 | shim BDs over the control-packet plane | 2.75 ms | arm 65 → 20 µs/layer (~372 ns/word MMIO vs ~3 ns/word) |
| 4 | input-ring prefetch + no per-row `%` | (in 3) | a chunk's rows stream while the previous chunk computes |
| 5 | **resident input** for 14/7 px layers | 2.35 ms | deep layers re-streamed their input P = 4..8 times; now once |
| 6 | stem + maxpool on the engine | 2.59 ms incl. stem | stem 0.14 ms; maxpool's 2.4 ms CPU cost gone |
| 7 | NEON quantize, SDOT dense | total 3.55 → **2.81 ms** | 0.51 → 0.16 ms, 0.46 → 0.08 ms |

Experiments that **did not** help, measured so they aren't retried:

- AXI burst 32 instead of 16: 2.35 ms vs 2.33 ms.
- AxCACHE = 3: 2.33 ms.
- Compiling the MAC loops out (`KDEFS=ENG_NO_MAC`): the engine still took 2.59 ms
  against 2.75 ms. **Compute is ~6 % of engine time; the engine is bound by data movement.**

Final per-layer time (chained, µs, includes ~20 µs control-packet arm):

| call | 1 stem | 4 | 6 | 8 | 10 | 12 | 14 | 17 | 19 | 21 | 23 | 26 | 28 | 30 | 32 | 35 | 37 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| µs | 137 | 114 | 127 | 111 | 129 | 122 | 110 | 101 | 110 | 128 | 111 | 104 | 112 | 172 | 259 | 276 | 270 |

## 5. Where the remaining time is (and what would move it)

* **Shim DMA rate.** The AIE clock is measured at **998 MHz** (tile timer vs `XTime`).
  A stream carries 4 B/cycle, yet one shim MM2S channel delivers **~2.3 GB/s** from
  DDR whatever the burst length or AxCACHE. The three 512-channel layers stream
  608 KB of weights per column: ~250 µs each, half of the engine. Total weights are 11.8 MB, so ≈1.3 ms
  is the floor at this rate unless they overlap other layers.
* **Per-layer arm** (~20 µs × 17) is the CPU encoding BDs into control packets. The
  words are identical every image, so record/replay (the runtime's AOT table) would
  make it ~2 µs.
* **Next-layer weight prefetch.** The header plus the first two chunks of layer L+1 could stream
  during layer L; they don't depend on L's output. Shim BDs 12/13 are free for a second bank.
  The OOO output BDs cannot be double-banked: 16 per shim tile, 10 used.

## 6. Traps (each one cost a board run)

* **Kernel stack overflow looks like rounding errors.** With the stem inlined into
  `main`, its frame overflowed the 4.6 KB stack into `ep_main`, the requant constants just
  above it. The result was **+1 errors** in a few pixels and constant output in residual layers.
  `run_*` are `noinline` and the stack is 5 KB; check `kernel.calltree`. Skill **aiecorestackoverflow**.
* **Timer reads move across lock acquires.** Chess schedules
  `aie::tile::current().cycles()` freely around `acquire_greater_equal`, even with
  `chess_separator_scheduler()`. Every "wait" read ~2 cycles and totals included
  host gaps, so A/B builds measure on the host instead. Skill **aiecoretimerlock**.
* **A plain shim BD longer than 1023 words** takes the runtime's fallback path, which
  skips the burst-16 setting. Weights use a 3-D contiguous multidim BD.
* **Header field names**: the generator once looked up `sM` as `sm`. Every residual
  multiplier became 0 and every later layer failed by propagation. Isolated mode (each layer
  fed TVM's golden input) exists so one wrong layer can't hide the others.
* **Core memory is the design constraint.** Arena = `0x71C00..0x80000` (57 KB):
  - `gen_engine.plan` lays out W ring, ring, ACC, OUT and RES per layer;
  - RES sits at the top so the producer and consumer agree on its address;
  - the plan asserts that everything fits.

## 7. Bring-up history

`resnetengine/e0` (persistent kernel + per-layer shim re-arm, synthetic data) and
`resnetengine/e1` (layer 04 alone, bit-exact) are kept as minimal references.
