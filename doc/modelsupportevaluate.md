# Model Support Evaluation — Gemma, ResNet-50, YOLO, MobileNet, BERT, ViT

Status: evaluation + design proposal + milestone plan (2026-10-12).
Scope: what the aiehlc compiler (Part 1 single-kernel aiehlc, Part 2 tilinglinalg,
the `aiegraph` dialect and the TVM / Triton / PyTorch frontends) can program **today**,
what each target model needs, the gap, and a compiler design that closes it — with
emphasis on **runtime reconfiguration over the control-packet plane** and **new
spatial types / dialects**.

All "today" claims below are checked against `main`; citations are `file:line`.

---

## 0. Executive summary

| Model | Verdict today | Fraction of MACs that can run on AIE today | First blocker |
|---|---|---|---|
| ResNet-50 | **Not supported** (only the ResNet-18 stem geometry offloads) | ~2% (stem only, if geometry were generalized) | Single hard-coded conv geometry; one launch per op with full mesh re-program |
| MobileNet v2/v3 | **Not supported** | 0% | Depthwise / grouped conv rejected (`groups != 1`), ReLU6/h-swish epilogue |
| YOLO (v5/v8) | **Not supported** | 0% | SiLU, upsample, concat, maxpool/SPPF, detection head, non-224 input |
| BERT-base | **Not supported** | 0% | No softmax / LayerNorm / GELU, no act×act GEMM in any graph frontend |
| ViT-B/16 | **Not supported** | 0% | Same as BERT + patch-embed conv 16×16/s16 |
| Gemma (2B/7B) | **Not supported** | 0% | No bf16, RMSNorm, RoPE, GeGLU, KV-cache, dynamic seq len; weights ≫ on-chip memory |

What *does* exist, and is the foundation for all of the above:

* A working, hardware-verified **int8 GEMM / conv2d-as-GEMM** spatial compiler on a
  mesh up to 4×4 in practice (Gen5) with DMA-built im2col, halo tiling, tile_k rounds.
* A **control-packet plane** that writes any tile-local address at ~2.9 ns/word versus
  ~372 ns/word over MMIO (`doc/performance/register_write_cost.md`), already used for
  locks, L1 data, shim BDs, kernel ELF load and core enable.
* **KERNELCONFIGOFFLOAD**: a core can program its own DMA BDs/locks via TM writes.
* A TVM BYOC path that is bit-exact for one layer, with CPU fallback for everything else.

The gap is not "one more kernel". It is structural, in four layers:

1. **Op coverage** – only GEMM/conv exist as spatial spaces; every transformer and
   most CNN-modern ops (depthwise, SiLU, softmax, norm, attention) are missing.
2. **Kernel quality** – kernels are scalar (convgemm: 8.2 cycles/MAC vs 256 MAC/cycle
   peak, zero vector ops in the chess listing).
3. **Execution model** – one op = one launch = full mesh re-program + ELF reload.
   The `allconv` experiment shows the consequence: ResNet-18 all-conv offload is correct
   but takes **13 000 ms** over **952 launches**.
4. **Data types & shapes** – int8/16/32 only, static shapes only.

The proposed design (§4) attacks (3) first because it multiplies everything else:
a **persistent array + phase-based reconfiguration** model where layer-to-layer
changes are compiled into **control-packet configuration deltas** and, for
intra-layer changes, **kernels reconfigure their own DMA/stream switch**. Two new
dialects (`aiephase`, `aiecfg`) and five new spatial spaces (`DepthwiseSpace`,
`EltwiseSpace`, `ReduceSpace`, `AttentionSpace`, `PipelineSpace`) carry it.

---

## 1. Current capability inventory

### 1.1 Spatial / tilinglinalg level

| Capability | State | Evidence |
|---|---|---|
| Iteration spaces | `GemmSpace{m,n,k,d1..d3}`, `Conv2dSpace{ih,iw,ic,oc,kh,kw,stride,pad,m}` — **only two** | `doc/design/spatial_space_composition.md:26-107` |
| Data-distribution patterns | Broadcast / Scatter / Multicast / Gather; Layout Row/Col/Grid (Grid ⇒ row ownership) | `routing/routingmanager.cpp:871-874` |
| Reduction across tiles | `row_reduce_out` tag documented as "cascade reduction" but lowered to plain `scatter` — **no real reduction** | `include/aie_spatial_types.h:32-66`, `routingmanager.cpp:843` |
| conv2d | via im2col (DMA multi-dim BD, ≤4 dims) or row-halo split; stride, symmetric pad, K×K incl. 1×1 | `doc/design/conv2d_im2col_design.md:754-842` |
| depthwise / dilation / pool / transpose | `DmaTransform` factories exist, **no caller, untested** | `src/llvm/aiehlc.cc:4300-4345` |
| Datatypes | int8 / int16 / int32 windows; **no bf16 / fp** | `aiehlc.cc:4104-4112` |
| Ports per kernel | 3 (a 4th never exercised) | `conv2dstem.cc:189-191` |
| MemTile staging | path exists, hard-wired off (`moplevel = 0`) | `passroutingtodmap/routingtodmap.cpp:1208` |
| Cascade stream | **no op in any `.td`** | `routing/td/routingop.td` |
| Ops per launch | **1**; multi-kernel source ⇒ independent pipeline + launch per kernel | `aiehlc.cc:4727-4745` |
| Inter-op on-array streaming | **none** | — |
| Practical mesh | 4×4 (`HW_ROWS = HW_COLS = 4`); device model up to 8×38 (Gen2), 4×12 / 4×38 (Gen5) | `hwresource.cpp:118-324` |

### 1.2 Kernel level

| Kernel | Vectorized? | Accumulator | Status |
|---|---|---|---|
| `conv2dstem.cc` (ResNet-18 7×7/s2) | scalar | int32 | HW-verified, bit-exact vs TVM layer 01 |
| `kernels.py` conv_bn(_relu) / residual / avgpool_fc | scalar | **int16 (wraps)** | placeholder quant, not bit-exact |
| `allconv` worktree `convgemm` (generic im2col GEMM) | scalar (0 vector ops) | int32 | correct top-1, 8.2 cyc/MAC |
| `simplematmul_ctrl_pkt*.cc` | `aie::mmul<8,8,8,int8,int8>` | int32 | example only |
| `example/tileprogram/design/ccode/attention.cc`, `kv_cache.cc` | sketch | — | not built by any script |

### 1.3 Frontends

| Frontend | Reach | Limits |
|---|---|---|
| TVM Relay (`src/frontend/tvmrelay`) | ResNet-18 v1 ONNX int8 → baremetal ELF; `--aie-offload` 1 layer | model URL hard-coded (`deploy_flow.py:209-215`); one pattern `aie.qconv_fused`, one emitter `emit_stem` (`aie_codegen.py:235`); `kernel_can_run` rejects groups≠1, non-stem geometry, non-uint8 I/O (`aie_patterns.py:304-341`) |
| `aiegraph` dialect | 4 ops: `conv_bn_relu`, `conv_bn`, `residual_add_relu`, `avgpool_fc`, int8 only | lowers to per-op launch records, not to linalg (`AiegraphLowerDriver.cpp:40-74`) |
| aietriton | `program_id/load/store/for/tl.dot` → GEMM; int8/16/32; static shapes | no softmax/elementwise (`ast_to_kernelops.py:127-246`) |
| pytorchmlir | torchvision ResNet-18 → PT2E int8 → linalg | **stops at MLIR** — no AIE backend |
| rcom | recognize HIP int8 GEMM → fixed template | template only |

Searches over `src/` for rmsnorm, gelu, silu, upsample, embedding, rotary, kv-cache,
bf16 return no implementation (only strings / unit-test placeholders).

### 1.4 Control plane / reconfiguration

| Mechanism | What it can do today | Gap for reconfig |
|---|---|---|
| Control packets | Write any tile address (20-bit, 1–4 beats), read, write-with-return; used for locks, L1, shim BDs, ELF, CoreCtrl | **stream-switch registers never written via ctrl pkt** (always MMIO `XAie_StrmConn*`, `aie_runtime.c:4531`); no MaskWrite in transactions (`controlplane.md:176-192`) |
| Fabric | row/broadcast, ≤4 rows, one spine column | **no per-column unicast** (removed, `controlplane.md:154`) |
| AOT ctrl-pkt (`control_packet_mode(aot)`) | Collect → Emit → `host_ctrlpkt.h` blob with fingerprint fallback | single window, single phase; not enabled for conv2dstem |
| KERNELCONFIGOFFLOAD | core writes own core-tile BDs (1-D), locks, start queues via `TM_W` | values are compile-time constants; no stream-switch encoder; no neighbour writes |
| ELF load over ctrl pkt | `__Runtime_load_kernel_group_16t_ctrl` (broadcast, packed BD in HT mode) | MMIO baseline ≈ 7.4 ms / 16-tile load; ctrl-pkt load **not measured** |
| Packet-switched data routing | `routinghw.connectpktstreamswitchport` (slot/mask/msel/arbiter) | declarative `routinghw.packet_flow` commented out "not implemented" (`routinghwop.td:7-39`); data plane ignores ctrl reservations (`controlplane.md:261`) |
| Runtime parameters (RTP) | — | not implemented (only an XAie API note `doc/aieapi.md:452`) |
| Multi-phase / multi-layer | each layer is its own `host.cc` + partition init, called from `graph_driver.c` | **no persistent layer configuration object, no phase schedule** |

**Measured cost anchors** (use these, do not re-derive):
MMIO write ≈ 372 ns/word; ctrl-pkt ≈ 2.9 ns/word steady-state floor; core BD = 6 words,
shim BD = 9, memtile BD = 8; one config window example 232 writes = 86 µs MMIO;
ELF load MMIO ≈ 7.4 ms/16 tiles; convgemm single-tile launch 13.4 → 2.6 ms with control
packets, after which `wait_io` (real AIE work) is 73 % of the launch.

---

## 2. Model requirements

Representative variants: ResNet-50 v1.5, MobileNet-V2 (V3 noted), YOLOv8n (v5 noted),
BERT-base (seq 128), ViT-B/16 @224, Gemma-2B (prefill + decode). MAC counts are the
standard published figures (rounded).

### 2.1 Operator inventory

| Op family | RN50 | MBv2 | YOLOv8 | BERT | ViT | Gemma | aiehlc today |
|---|---|---|---|---|---|---|---|
| conv K×K, groups=1 | ✔ (3×3, 7×7) | ✔ (3×3 stem) | ✔ (3×3, s1/s2) | | ✔ 16×16/s16 patch | | Conv2dSpace ✔ (1 geometry wired to TVM) |
| conv 1×1 (pointwise) | ✔ dominant | ✔ dominant | ✔ | | | | Conv2dSpace ✔ (allconv only) |
| depthwise conv | | ✔ 17 layers | (v8: no; some heads) | | | | ✘ |
| static-weight GEMM (Linear) | FC | FC | | ✔ QKV/O/FFN | ✔ | ✔ QKV/O/GeGLU/LM head | GemmSpace ✔ |
| activation×activation GEMM | | | | ✔ QKᵀ, PV | ✔ | ✔ | Triton only, static |
| bias + requant epilogue | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | stem epilogue only |
| ReLU / ReLU6 | ✔ / | / ✔ | | | | | ReLU (clip 0..255) only |
| SiLU / h-swish | | V3 | ✔ everywhere | | | | ✘ |
| GELU / GeGLU | | | | ✔ | ✔ | ✔ GeGLU | ✘ |
| softmax | | | DFL head | ✔ | ✔ | ✔ | ✘ |
| LayerNorm / RMSNorm | (BN folded) | (BN folded) | (BN folded) | ✔ LN | ✔ LN | ✔ RMSNorm | ✘ |
| residual add | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | CPU only |
| maxpool / avgpool | ✔ | ✔ | SPPF maxpool | | | | CPU only |
| upsample (nearest) | | | ✔ | | | | ✘ |
| concat / split / chunk | | | ✔ heavy | | | ✔ | ✘ |
| transpose / reshape (head split) | | | | ✔ | ✔ | ✔ | DmaTransform `transpose` untested |
| embedding gather | | | | ✔ | (pos-emb add) | ✔ 256k vocab | ✘ |
| RoPE | | | | | | ✔ | ✘ |
| KV cache, causal mask | | | | | | ✔ | ✘ |
| dynamic shape | | | (multi-res) | seq len | | seq len / position | ✘ |

### 2.2 Scale, datatypes, memory

| Model | Params | MACs / inference | Natural dtype for AIE | Fits on-chip? |
|---|---|---|---|---|
| ResNet-50 | 25.6 M | ~4.1 G | int8 (PTQ well-proven) | weights no (25.6 MB int8); per-layer yes with tiling |
| MobileNet-V2 | 3.4 M | ~0.3 G | int8 (per-channel needed, depthwise is sensitive) | nearly (3.4 MB int8) — candidate for **full on-array residency** across memtiles of a large device |
| YOLOv8n | 3.2 M | ~4.4 G @640 | int8 (SiLU via LUT / piecewise) | nearly; activations 640×640 large |
| BERT-base | 110 M | ~11 G @seq128 | int8 GEMM + bf16 softmax/LN (int8-only accuracy is fragile) | no |
| ViT-B/16 | 86 M | ~17.5 G | int8 GEMM + bf16 softmax/LN | no |
| Gemma-2B | 2.5 B | prefill ~2.5 G/token; decode **memory-bound** (stream all weights per token) | bf16 or int8/int4 weight-only + bf16 act | **no** — 2.5 GB int8; DDR-bandwidth bound |

Per-tile resources (AIE-ML / AIE2PS class): 64 KB core data memory, 512 KB memtile,
int8 peak 256 MAC/cycle/tile with `aie::mmul`. Today's scalar kernels reach
~1/2000 of that peak (convgemm 8.2 cycles/MAC).

---

## 3. Per-model evaluation and gap catalogue

### 3.1 Gap catalogue

| ID | Gap | Layer | Blocks |
|---|---|---|---|
| **G1** | One op = one launch; full mesh re-program and ELF reload per launch; no persistent configuration | runtime / exec model | all (perf) |
| **G2** | Frontend hard-wired to ResNet-18 ONNX; only one BYOC pattern/emitter and one geometry | frontend | all |
| **G3** | Scalar kernels; no vectorized `aie::mmul` library with fused epilogues | kernel | all (perf) |
| **G4** | No depthwise / grouped conv space | spatial | MobileNet, (YOLO heads) |
| **G5** | No elementwise / activation space (SiLU, GELU, h-swish, residual add on array) | spatial + kernel | YOLO, MBv3, BERT, ViT, Gemma |
| **G6** | No reduction space (softmax, LayerNorm, RMSNorm, pooling); `row_reduce_out` is fake; no cascade | spatial + routing | BERT, ViT, Gemma, (pools) |
| **G7** | No act×act GEMM in graph frontends; no attention composition (QKᵀ→softmax→PV) | spatial + frontend | BERT, ViT, Gemma |
| **G8** | No bf16 (or any float) data path | type system + kernel | BERT, ViT, Gemma (accuracy) |
| **G9** | No data-movement ops: concat, split, upsample, transpose, gather/embedding | spatial (DMA) | YOLO, all transformers |
| **G10** | Static shapes only; no runtime parameters (seq length, position, KV length) | IR + runtime | BERT (var seq), Gemma |
| **G11** | No inter-op on-array streaming / fusion; every intermediate goes to DDR | spatial + routing | all (perf, DDR BW) |
| **G12** | MemTile staging disabled; 3 ports/kernel; 4 KB default L1 budget | routing / cost model | large layers, weight reuse |
| **G13** | Control plane cannot write stream switches, no masked write, no per-column unicast, data plane ignores reserved ctrl slots | runtime / ctrl plane | dynamic reconfig (§4.2) |
| **G14** | Weights ≫ on-chip; no weight-streaming schedule, no int4/weight-only quant | spatial + memory | Gemma |
| **G15** | Detection post-processing (DFL, NMS) | — (CPU by design) | YOLO (accept CPU) |

### 3.2 Per-model

**ResNet-50.** Closest to working. Every conv is a Conv2dSpace instance (1×1, 3×3
s1/s2, 7×7 stem); ResNet-18 all-conv offload in the `allconv` worktree already proves
correctness. Blockers: G2 (frontend), G1 (53 convs × ≥1 launch each, with re-program —
the ResNet-18 analogue ran 952 launches / 13 s), G3 (scalar kernel), G11 (bottleneck
blocks write every intermediate to DDR). Residual add / pool can stay on CPU in phase 1
but cost a DDR round trip each. **Difficulty: low–medium. First model to target.**

**MobileNet-V2/V3.** Needs G4 (depthwise 3×3, s1/s2) — the `depthwise_im2col`
DmaTransform exists but depthwise is a *bad* fit for im2col-GEMM (K=9 reduction,
no channel reuse): it needs its own space where channels are partitioned across tiles
and the kernel is a vector MAC over the 3×3 window. ReLU6 = clip(0, 6·scale) is an
epilogue variant; V3's h-swish needs G5. Low MAC count makes **launch overhead (G1)
dominant** — per-layer compute is µs-scale, per-launch overhead is ms-scale today.
Small weights (3.4 MB int8) make it the best candidate for **whole-network residency**
on a large device. **Difficulty: medium.**

**YOLO (v5/v8).** Conv-heavy (good), but SiLU after every conv (G5 — fuse into the
GEMM epilogue as LUT/piecewise), C2f blocks with split + concat (G9 — should be pure
DMA addressing, zero compute), SPPF maxpools (G6), nearest upsample (G9, DMA
replication), 640×640 input (G12 — large activations, need memtile staging and
row-halo tiling). Detection decode / NMS stays on CPU (G15). **Difficulty: medium–high.**

**BERT-base.** GEMM-dominant (QKV, O, FFN: ~95 % of MACs, all GemmSpace-shaped,
M = seq). Needs G7 (QKᵀ and PV are act×act with per-head batching), G6 (softmax,
LayerNorm), G5 (GELU), G8 (bf16 for softmax/LN accuracy), G9 (head split transpose),
G10 (variable seq length — or bucket to 128/384/512 statically). **Difficulty: high.**

**ViT-B/16.** Same as BERT plus a 16×16/s16 patch-embed conv (= reshape + GEMM, fits
Conv2dSpace with K=stride, no halo) and a fixed sequence of 197 — **static shape**,
so ViT is the **better first transformer** than BERT. **Difficulty: high (but
static).**

**Gemma-2B/7B.** Everything BERT needs, plus RMSNorm, RoPE, GeGLU, MQA/GQA, KV cache,
causal mask, 256k-vocab embedding / LM head, and two very different regimes: prefill
(GEMM, compute-bound) and decode (GEMV, **DDR-bandwidth bound**: every token streams all
weights once). Requires G14 weight-streaming dataflow and runtime parameters for
position/KV length (G10). Realistic scope is a single decoder block / prefill first,
and int8 or int4 weight-only quant. **Difficulty: very high.**

---

## 4. Solution design

Guiding principle: **keep the existing GEMM pipeline as the workhorse; add a layer
above it (graph & phase) and a layer beside it (configuration-as-data)**. The design
has five parts:

* 4.1 Graph-level: generalize the frontend and `aiegraph` into a linalg-backed op graph.
* 4.2 Execution model: persistent array + phases, reconfigured over the control plane
  (the core of the proposal).
* 4.3 Kernel self-reconfiguration (on-array, no host).
* 4.4 New spatial spaces.
* 4.5 Kernel library and data types.

```
 ONNX / PyTorch(PT2E) / TVM Relay
            │  (generic importer, quant recipe)
            ▼
   aiegraph  (extended op set, SSA tensors, quant attrs)
            │  fusion + partition (AIE vs CPU), per-op Space selection
            ▼
   aiephase  (NEW)  — phases, resident tensors, residency/streaming plan
            │  per phase: one or more Spaces instantiated on sub-meshes
            ▼
   routing → dmap → dmaphop → dfscheblueprint   (existing, reused per phase)
            │                         │
            ▼                         ▼
   dfschedule (per phase)      aiecfg (NEW) — config state per phase,
            │                  delta = cfg(N+1) ⊖ cfg(N), ctrl-pkt encoding
            ▼                         ▼
   kernel.cc (overlay kernel,   host.cc: phase driver + ctrl-pkt blobs (.rodata/DDR)
   RTP-driven)
```

### 4.1 Graph level: generalized `aiegraph`

* **Importer**: replace the hard-coded ResNet-18 URL with `--model <onnx|torch>` and a
  per-model *recipe* (input shape, quant scheme). Reuse pytorchmlir's linalg output as
  the canonical ingest: lift `linalg.conv_2d_*`, `linalg.matmul/batch_matmul`,
  `linalg.generic` (elementwise, reductions) into `aiegraph` ops.
* **Op set** additions to `aiegraph` (int8 and bf16 element types):
  `conv2d{groups, dilation}` (subsumes depthwise), `matmul{transpose_b, batch}`,
  `eltwise{kind: add|mul|silu|gelu|hswish|relu6, lut?}`, `reduce_norm{kind:
  softmax|layernorm|rmsnorm, axis}`, `pool{max|avg}`, `resize{nearest}`, `concat`,
  `split`, `transpose`, `gather`, `rope`, `attention{heads, kv_heads, causal}`
  (a *composite*, decomposed by §4.4 AttentionSpace), `kv_cache_update`.
* **Epilogue attributes** instead of separate ops: `bias`, `requant{mult,shift,zp}`,
  `act{relu|relu6|silu|gelu}`, `residual` — so the BYOC `kernel_can_run` capability
  check becomes a query against the epilogue set, not geometry equality.
* **Partitioner** keeps the existing CPU fallback (TVM `target="c"`) but decides by
  *cost*, not by whitelist: an op goes to AIE when `t_aie + t_cfg_delta < t_cpu`.

### 4.2 Execution model: persistent array, phases, control-plane reconfiguration

Today: `for each op: init partition → MMIO routing → MMIO BDs → load ELF → launch → wait
→ teardown-ish`. G1 is the dominant cost. The proposal:

**(a) Phase = a static configuration of the mesh.** A phase is one or more Space
instances (possibly fused, §4.4 PipelineSpace) mapped to disjoint sub-meshes, plus the
stream-switch routes, BDs, locks and kernel parameters they need. A model compiles to a
**phase sequence** `P0, P1, … Pn` (ResNet-50 ≈ 20–55 phases depending on fusion).

**(b) Configuration-as-data (`aiecfg` dialect).** Instead of emitting XAie calls, the
host path of each phase emits an `aiecfg` *state*:

```mlir
%s1 = aiecfg.state @phase1 {
  aiecfg.switch_route tile(3,2) master<DMA:0> slave<South:1>   // circuit
  aiecfg.pkt_route    tile(3,2) slot(1) id(5) mask(0x1f) arb(0) msel(1)
  aiecfg.bd           tile(3,2) bd(4) addr(0x400) len(256) lock_acq(0,-1) lock_rel(1,+1)
  aiecfg.lock_init    tile(3,2) lock(0) value(2)
  aiecfg.rtp          tile(*,*) slot(0) value(%layer_desc)      // kernel params
  aiecfg.core_elf     tiles(row 3..6) @overlay_gemm              // only if changed
}
%d = aiecfg.delta %s0 -> %s1          // pass: diff, drop no-op writes
aiecfg.encode %d {mode = ctrlpkt, fabric = row_broadcast}
```

* `AiecfgDeltaPass`: register-level diff between consecutive phases; identical tiles
  in a row/column collapse to **row/broadcast writes** (generalizing today's
  `GroupRegWritePass`, which only does lock inits).
* `AiecfgEncodePass`: lowers a delta to control-packet words using the existing
  `__Runtime_ctrl_pktize_write` encoding, AOT, into a blob per phase transition
  (generalizing `passapitocontrolpacket`'s single window to N windows). Blobs live in
  DDR; the host (or a shim BD chain) streams them in.
* **Cost estimate** (using the measured anchors): a 4×4 phase touching per tile
  ~10 switch words + 4 BDs × 6 + 8 lock words ≈ 42 words → 672 words/phase.
  MMIO: 672 × 0.372 µs ≈ **250 µs**. Ctrl-pkt floor: 672 × 2.9 ns ≈ **2 µs**
  (+ fixed setup, to be measured in M1). With broadcast collapsing, fewer still.
  For ResNet-50 × ~50 phases: ~12.5 ms MMIO vs ≪ 1 ms ctrl-pkt — the difference
  between config-bound and compute-bound.

**(c) Overlay kernels instead of ELF reload.** ELF reload is the most expensive item
(≈7.4 ms/16 tiles over MMIO). Compile a small family of **overlay kernels**
(`overlay_gemm_int8`, `overlay_dwconv`, `overlay_eltwise`, `overlay_norm_bf16`,
`overlay_attention`) whose loop bounds, quant parameters, epilogue selector and window
counts come from a **layer descriptor** in a fixed L1 RTP region, written per phase by
`aiecfg.rtp` (a handful of words). Core program memory is then reloaded only when the
overlay *family* changes between phases (`aiecfg.core_elf` emitted only on change).
Phase ordering pass groups consecutive phases with the same overlay.

**(d) Double-buffered configuration.** While phase N computes, the delta for N+1 is
pushed to the *shadow* resources: alternate BD id banks (BD 0–7 vs 8–15) and alternate
RTP slots; the switchover is one lock release / one broadcast write. Stream-switch
routes cannot be shadowed, so the scheduler classifies deltas into
*shadowable* (BDs, locks, RTP) and *barrier* (routes, ELF) and only barriers stall.
Phases are planned so consecutive phases share routes where possible
(`aiephase` route-reuse heuristic: keep the same broadcast/scatter topology and change
only BD strides).

**(e) Control-plane extensions required (G13).**
1. Stream-switch writes via control packets (master/slave config + packet slot
   registers) — encoding already permits it; add encoders + verification against
   `XAie_StrmConn*` register images (`xaieapiverify`).
2. Masked write (read-modify-write) in transaction capture, or design `aiecfg` so all
   writes are full-register (the compiler owns the whole register image — preferred).
3. Per-column unicast (re-introduce the overlay removed in `controlplane.md:154`) for
   non-uniform deltas.
4. Data-plane allocator must honour `RT_RES_*` reservations so control and data
   routes coexist in the same phase without the release/restore dance
   (`__Runtime_ctrl_plan_release`, `aie_runtime.c:4535-4546`).

### 4.3 Kernel self-reconfiguration (core-driven)

Extends KERNELCONFIGOFFLOAD from *compile-time constant* to *descriptor-driven*:

* **Kernel reprograms its own DMA per sub-phase.** For loops that today require the host
  to re-arm BDs between spatial rounds (~5000 cycles each, `cost_model_optimizer.md:256`)
  the core reads the next BD set from its descriptor and writes it via `TM_W`.
  Requires multi-dim BD encoders in `aie_kernel_runtime.h` (today 1-D only).
* **Kernel reprograms its own stream switch** (TM window already reaches the core
  module; add `core_ss_*` encoders) to switch between e.g. *receive weights* and
  *receive activations* on the same port, or between a "compute" route and a
  "forward-to-neighbour" route. This enables **time-multiplexed ports**, lifting the
  3-port limit (G12) without extra physical routes.
* **Kernel-issued control packets** (stretch): a designated "sequencer" core on the
  control spine emits control packets to its row (core → packet-switched stream →
  CTRL ports). This removes the host from the inner phase loop entirely: the host
  pushes one *program* of phases; the sequencer walks it. Needs packet-route
  reservation from the core's MM2S to the row's CTRL slots (reuse `ResourceMgr`).
* **Safety rules** (inherit from kernel_config_offload.md): BD-id reservation contract
  per phase, lock direction asymmetry, `.TM` presence check in `kernel.lst` as a CI gate.

### 4.4 New spatial spaces (and the `aiephase` dialect)

Each space reuses `tile_dim`, `SpatialPolicy{map, mat, sched}` and lowers to the existing
routing dialect; only the iteration-space → tile-mapping logic and the kernel body are new.

| Space | Iteration space | Mapping | Data movement | Covers |
|---|---|---|---|---|
| **DepthwiseSpace** | `{ih, iw, c, kh, kw, stride, pad, mult}` | channels **scattered** across tiles (no reduction across tiles), rows halo-split | per-tile channel slab; weights tiny ⇒ RTP/L1 resident | MobileNet dw, YOLO dw heads |
| **EltwiseSpace** | `{n}` (flattened) × arity | scatter, no halo | 1–2 inputs, LUT in L1 for nonlinear | residual add, SiLU, GELU, h-swish, scale |
| **ReduceSpace** | `{rows, len, axis}` + `kind` | rows scattered; if `len` > tile then **cascade / neighbour reduction** (real `row_reduce_out`) | two-pass (stats → normalize) or online (softmax) | softmax, LayerNorm, RMSNorm, avg/max pool, GAP |
| **AttentionSpace** | `{heads, q_len, kv_len, d_head, causal}` | head-parallel across columns; flash-style tiling over kv_len inside a column; online softmax in ReduceSpace kernel | Q resident, K/V streamed, P never leaves L1 | BERT, ViT, Gemma prefill |
| **PipelineSpace** (composition) | sequence of Spaces | each stage on a sub-mesh; **stage→stage over stream switch / memtile**, not DDR | producer/consumer locks across tiles (dmaphop already models producer/consumer) | conv→conv fusion (bottleneck), GEMM→GELU→GEMM (FFN), YOLO C2f |
| **MoveSpace** (DMA-only) | addressing transform | no kernel | concat/split = BD offset; upsample = repeat-stride BD; transpose = 4-D BD; gather = indexed BD from RTP | YOLO concat/upsample, head split, embedding |

Two transformer-specific additions to GemmSpace: `transpose_b` (QKᵀ without a
transpose pass, via DMA strides) and **act×act** mode (both operands streamed, no
weight residency), plus **weight-streaming mode** (G14): weights flow shim → memtile →
cores in K-chunks with double buffering while activations stay resident — the decode
GEMV shape for Gemma.

**`aiephase` dialect** (sits between `aiegraph` and `routing`):

```mlir
aiephase.program @resnet50 {
  %x  = aiephase.tensor ddr  : tensor<1x224x224x3xi8>
  %w0 = aiephase.tensor ddr  {weights = @conv1_w}
  aiephase.phase @p0 overlay(@overlay_gemm_int8) mesh(4x4) {
    %y = aiephase.instance Conv2dSpace(%x, %w0) {policy = #stem_policy}
  }
  aiephase.phase @p1 overlay(@overlay_gemm_int8) mesh(4x4) {
    // PipelineSpace: 1x1 → 3x3 → 1x1 bottleneck stays on array
    %a = aiephase.instance Conv2dSpace(%y, %w1) submesh(0..3, 0)
    %b = aiephase.instance Conv2dSpace(%a, %w2) submesh(0..3, 1..2) {in = stream}
    %c = aiephase.instance Conv2dSpace(%b, %w3) submesh(0..3, 3)    {in = stream}
    %r = aiephase.instance EltwiseSpace(%c, %y) {kind = add_relu, fused_epilogue}
  }
  aiephase.reconfig @p0 -> @p1 { shadow = [bd, rtp], barrier = [route] }
}
```

Passes: `AiephasePartitionPass` (graph → phases by sub-mesh capacity + L1/memtile fit
from the existing cost model, fixing its `pp_depth` bug), `AiephaseRouteReusePass`,
`AiephaseToRoutingPass` (per phase, invoking the existing pipeline on a cloned module
— the same clone pattern the host/kernel split already uses), `AiecfgDeltaPass`,
`AiecfgEncodePass`, `AiephaseHostDriverEmit` (host loop: push blob N+1, release
barrier, wait phase N).

### 4.5 Kernel library and data types

* **Vectorized overlay kernels** with `aie::mmul` (int8 4×8×8 / 8×8×8, bf16 4×8×4 on
  AIE-ML/AIE2PS), fused epilogue (bias, requant, activation via LUT/piecewise).
  Target ≥ 50 % of peak MAC/cycle on GEMM-shaped layers; the CI metric is
  cycles/MAC from the existing PMU profiling (`--profiling`, skill **aiehwprofile**).
* **bf16 type path** (G8): extend window type detection (`aiehlc.cc:1202-1222`,
  `4104-4112`) and `routing`/`dmap` element types with `bf16`; mixed precision
  (int8 GEMM → int32 acc → bf16 softmax/LN → int8 requant) expressed as epilogue
  conversions, so the DMA layer sees only byte widths.
* **Nonlinear functions**: LUT + linear interpolation in L1 (SiLU, GELU, exp for
  softmax), shared via RTP-region-resident tables; bit-exactness tested against
  TVM / PyTorch reference per op (reuse the `layeriohex` harness).
* **Runtime shape parameters** (G10): sequence length, KV length, position live in the
  RTP descriptor; Spaces declare which dims are *runtime-bounded* (static max, dynamic
  trip count). The DMA BD lengths become descriptor-driven (§4.3), so the host does not
  re-compile per length — bucketing remains the fallback.

### 4.6 Alternatives considered

| Option | Why not chosen as primary |
|---|---|
| Keep one-launch-per-op, just optimize each launch (ctrl-pkt everywhere) | Removes MMIO cost but not ELF reload, DDR round-trips between ops (G11) or host in the loop; useful as M1 step, not the end state |
| Whole-network spatial residency (every layer on its own tiles) | Only fits small CNNs (MobileNet) on large devices; included as a `aiephase` special case (1 phase), not a general model |
| Generic MLIR-AIE (`aie` dialect / IRON) instead of new dialects | Would discard the existing routing/dmap/dfschedule stack and the control-plane runtime; the proposal stays compatible by keeping `routing` as the per-phase IR |
| Pure CPU fallback for nonlinear ops | Each fallback = DDR round trip + phase barrier; acceptable for YOLO NMS and as a stepping stone, not for softmax/LN inside every transformer block |

---

## 5. Milestone plan

Effort in engineer-weeks (ew), assuming people familiar with this repo. Each milestone
has an **exit criterion measured on hardware** (the repo's verification standard:
`apppaltest.py` + golden compare), not just compile success.

### M0 — Baseline & measurement (2 ew)
1. Merge `allconv` convgemm into `main` behind a flag; keep `verify_aie_stem.py` gate.
2. Measure ctrl-pkt ELF load and ctrl-pkt per-phase fixed overhead (missing data, §1.4).
3. Add per-model op census tool (ONNX → op/shape table, AIE-eligible fraction).
**Exit:** ResNet-18 all-conv on HW, cost breakdown table (cfg / ELF / compute / DDR) per launch.

### M1 — Cheap launches (4 ew) — attacks G1, G13 partially
1. Control-packet stream-switch encoders + register-image verification.
2. Full-register writes only in `aiecfg` (sidestep MaskWrite).
3. Data-plane allocator honours `RT_RES_*` reservations.
4. Enable `control_packet_mode(aot)` for every offloaded layer; multi-window AOT blobs.
**Exit:** ResNet-18 all-conv inference time reduced ≥ 10× vs M0's 13 s, bit-exact logits.

### M2 — `aiecfg` + overlay kernels + persistent array (6 ew) — G1
1. `aiecfg` dialect (td/gen.sh/inc/unitest per repo convention), `AiecfgDeltaPass`,
   `AiecfgEncodePass`.
2. Overlay GEMM/conv kernel with RTP layer descriptor; ELF loaded once.
3. Host phase driver; double-buffered BD banks / RTP slots.
**Exit:** ResNet-18 all-conv with a single ELF load; per-layer reconfig < 50 µs measured.

### M3 — Vectorized kernel library + generalized frontend (6 ew) — G2, G3
1. `aie::mmul` int8 GEMM/conv overlay with fused bias/requant/ReLU/ReLU6/residual.
2. `--model` importer via pytorchmlir linalg → extended `aiegraph`; capability-based
   `kernel_can_run`.
3. Cost-based AIE/CPU partitioner.
**Exit:** **ResNet-50 int8 end-to-end** on HW, all convs on AIE, top-1 matches CPU
reference, ≥ 30 % peak MAC on 3×3 layers.

### M4 — CNN completeness: Depthwise, Eltwise, Move spaces (6 ew) — G4, G5, G9, G12
1. DepthwiseSpace + overlay_dwconv.
2. EltwiseSpace with LUT nonlinearities (SiLU, h-swish).
3. MoveSpace (concat/split/upsample/transpose) — DMA only.
4. Enable MemTile staging (`moplevel`), fix cost-model `pp_depth` bug.
**Exit:** **MobileNet-V2** end-to-end on AIE (CPU only for pre/post); **YOLOv8n**
backbone+neck on AIE, NMS on CPU, mAP within 1 pt of CPU int8.

### M5 — `aiephase` + PipelineSpace (on-array fusion) (8 ew) — G11
1. `aiephase` dialect, partition / route-reuse / to-routing passes.
2. PipelineSpace stage-to-stage streaming via memtile/stream switch.
3. Kernel self-reconfiguration of multi-dim BDs and own stream switch (§4.3).
**Exit:** ResNet-50 bottleneck blocks fused (no DDR intermediates); DDR traffic and
latency reduction reported vs M3.

### M6 — Transformer enablers (8 ew) — G6, G7, G8
1. bf16 type path end-to-end.
2. ReduceSpace (softmax online, LayerNorm, RMSNorm) with real cascade/neighbour reduction.
3. GemmSpace `transpose_b`, act×act mode; AttentionSpace (flash-style).
4. GELU / GeGLU epilogues.
**Exit:** **ViT-B/16** (static 197 tokens) end-to-end, top-1 matches PyTorch int8/bf16
reference; one encoder block cycle breakdown.

### M7 — BERT & dynamic shapes (4 ew) — G10
1. Runtime-bounded dims in Spaces; descriptor-driven BD lengths.
2. Embedding gather via MoveSpace; attention mask.
**Exit:** **BERT-base** seq ≤ 512 with one compiled artifact, accuracy parity on a
GLUE task subset.

### M8 — Gemma (decoder LLM) (10 ew) — G14
1. Weight-streaming GemmSpace (decode GEMV), int8/int4 weight-only.
2. RoPE, MQA/GQA in AttentionSpace, KV-cache update + causal mask.
3. Sequencer-core kernel-issued control packets (host-free phase loop).
4. Prefill / decode as two phase programs.
**Exit:** **Gemma-2B** single-token decode on AIE, tokens/s reported against DDR-BW
roofline; prefill for ≤ 512 tokens.

### Dependency graph

```
M0 ─► M1 ─► M2 ─► M3 ─► M4 ─────────────► (ResNet-50, MobileNet, YOLO)
                   │      │
                   │      └► M5 (fusion; perf for all)
                   └────────► M6 ─► M7 ─► (ViT, BERT)
                                    └──► M8 (Gemma)
```

Total ≈ 54 ew. Critical path to the first new model (ResNet-50): M0→M3 ≈ 18 ew.

### Risks

| Risk | Mitigation |
|---|---|
| Ctrl-pkt per-phase fixed setup larger than the 2.9 ns/word floor | M0 measures it; if high, prefer fewer, larger phases (fusion, M5) |
| Stream-switch reprogramming mid-run races with in-flight packets | barrier class in `aiecfg` (§4.2d); drain + lock handshake; never reconfigure a live route |
| int8-only accuracy for transformers | bf16 for softmax/LN from M6 onward; per-op accuracy harness (layeriohex) |
| Overlay kernel generality vs performance | small family of overlays, each specialized per op class; descriptor fields only for shapes/quant |
| Program-memory size for overlays | measure `kernel.lst` sizes; split families if > PM budget |
| Gemma memory bound | accept DDR roofline as the target; int4 weight-only; scope to 2B |

---

## 6. Immediate next actions

1. M0.2: write a micro-benchmark for ctrl-pkt ELF load and per-window setup (extends
   `doc/performance/register_write_cost.md`).
2. M1.1: prototype a stream-switch ctrl-pkt encoder on one tile, verify the register
   image against MMIO `XAie_StrmConn*` on palmyra.
3. Draft `aiecfg` TableGen (`state`, `switch_route`, `pkt_route`, `bd`, `lock_init`,
   `rtp`, `core_elf`, `delta`, `encode`) under `src/mlir/mlirfront/tilinglinalg/aiecfg/`.
