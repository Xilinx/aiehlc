# 7.4 Special data flow: ping-pong buffers (`.depth`)

> **Data flow: ping-pong.** A port's core buffer is either single (one
> buffer: the DMA and the core take turns) or ping-pong (two buffers: the DMA
> fills one while the core reads the other). The kernel chooses with
> `{.depth = 1}` or `{.depth = 2}` on `aie::load_buffer` / `aie::store_buffer`.
> This chapter follows that one option from the kernel line to the BD/lock
> sequence, the timeline and the buffer lifetimes.

> **Status.** Doc only: no new example. The program is the 6.1 step-1 GEMM
> (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc`). All refs are
> checked against base `43fcb8f` by script/check_refs.sh. The timelines in
> section 4 are **derived, not measured**.

## The whole program

The program is the step-1 GEMM of chapter 6.1; its full listing is in
[6.1, "The whole program (step 1)"](../06_gemm_optimization/1_low_level_schedule.md)
and is not copied here (6.1 keeps it in sync with the source). The three
lines that matter for this chapter
(`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:30-34`):
<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:14 The whole program (step 1) -->
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:30-34 {.depth = 2}); -->

```cpp
    const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true});
    for (int s = 0; s < cm / GD_TS; s++) {                             // 32 strips of 16 rows
        ...
        const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2});
        int8_t *y = aie::store_buffer(c, {r0, pn * cn}, {GD_TS, cn}, {.depth = 2});
```

The strip of A is released and the strip of C committed at the end of every
strip (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:42-43`); that is
the point where the core hands a buffer back to the DMA.
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:42-43 aie::commit(c); -->

## 1. The idea: two buffers, one lock pair each, ownership alternates

With one buffer, the DMA may write strip `s + 1` only after the core has
released strip `s`: DMA and compute never overlap. With two buffers (ping and
pong) the DMA fills one while the core computes on the other, and the two swap
at every release. The price is a second buffer in core data memory and more
buffer descriptors; what it buys is at most the DMA time per strip.

| `.depth` | buffers on the core | DMA during compute | cost |
|---|---|---|---|
| 1 | 1 (single) | no: core waits for every block | none extra |
| 2 | 2 (ping + pong) | yes: block `s + 1` fills while `s` is computed | one more buffer per core + its BDs |

## 2. Kernel: declaration -> buffer count

The rows below are Maxwell's (task `2026-11-21_tut07-pingpong`, P1).

The options struct is `aie::load_opts` (`store_opts` is the same type);
`.depth` defaults to 0, which means "use the kernel-wide pp_depth".
<!-- ref src/llvm/convautotile/kernel_trace.cpp:244 int depth = 0; bool resident = false; -->
<!-- ref src/llvm/convautotile/kernel_trace.h:88 .depth    the port's core buffers: 1 single, 2 ping-pong -->

The trace resolves the count per port in one line:

```c++
const int pp = opt.resident ? 1 : opt.depth > 0 ? opt.depth : g_pp;
```
<!-- ref src/llvm/convautotile/kernel_trace.cpp:281-283 const int pp = opt.resident ? 1 : opt.depth > 0 ? opt.depth : g_pp; -->

| Declaration | Buffers on the core | Rule |
|---|---|---|
| `{.resident = true}` | 1, held for the whole launch | resident wins over everything |
| `{.depth = 1}` on `load_buffer` | 1 (single) | no overlap of DMA and compute |
| `{.depth = 2}` | 2 (ping-pong) | next block arrives during compute |
| no option (`.depth = 0`) | kernel-wide pp_depth, 2 by default | see below |

Legality checks (the compiler rejects these with an error):

- `.depth` must be 0, 1 or 2; a core buffer is single (1) or ping-pong (2).
  <!-- ref src/llvm/convautotile/kernel_trace.cpp:1084-1085 a core buffer is single (1) or ping-pong (2) -->
- `{.depth = 1}` on `aie::store_buffer` is an error: an output window is always
  ping-pong on the core, so an output port always has 2 buffers.
  <!-- ref src/llvm/convautotile/kernel_trace.cpp:1086-1088 {.depth = 1} on aie::store_buffer -->
- `{.resident = true}` with `.depth > 1` is an error: a resident block lives in
  ONE buffer.
  <!-- ref src/llvm/convautotile/kernel_trace.cpp:1089-1091 contradicts it (drop .depth) -->
- `{.depth = 1}` is rejected on a spatial-halo (conv) kernel (7.2): its window
  lowering is ping-pong only.
  <!-- ref src/llvm/convautotile/kernel_trace.cpp:1474-1477 is not supported on a spatial-halo -->

Default: when a port carries no pingPong value, aiehlc sizes it as 2 buffers.
<!-- ref src/llvm/aiehlc.cc:2820 int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2; -->

In the 6.1 program B is resident (one 16 KB buffer), A and C are
`{.depth = 2}` (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:9-11`).
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:9-11 A {.depth = 2}        ping-pong strips -->

## 3. Hardware path and BD / lock sequence

The rows below are Amit's (task `2026-11-21_tut07-pingpong`, P0), pasted unchanged.
They are derived from compiler source and doc/debug, not from a lowered run or
a hardware trace. Locks are named by role only.

Source facts:
- The window op takes separate ping and pong addresses and separate ping / pong lock ids: CreateWindowOperation(dir, size, pingaddr, pongaddr, pinglockid, ponglockid). <!-- ref src/mlir/mlirfront/AieDialect.cc:515-516 arg_ponglockid -->
- The kernel config carries buffer_mode and num_buffers per tile (ping-pong example: buffer_mode = 1, num_buffers = 2). <!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/td/dfscheduleop.td:158 buffer_mode = 1, num_buffers = 2 -->
- pp_depth changes only the number of physical buffers (ping + pong), buffer_mode and num_buffers; it does not change buffer size, round counts or BD iteration. <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:290-300 Number of physical buffer allocations (ping + pong) -->
- buffer_mode 0 = single buffer (pp_depth=1), 1 = ping-pong (pp_depth 2 or more). <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:294 0 = single buffer (pp_depth=1) -->
- Cost of A {.depth = 2} in the 6.1 GEMM: one more kernel buffer per core and 16 more host BDs (88 to 104). <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:158 16 more buffer descriptors -->

| Row | Side | Resource | Rule | Source |
|---|---|---|---|---|
| L0 | lock pair allocation (compiler) | lockA, lockB from the tile's lockPool_ | allocateLockPair marks the first two unused pool entries for one owner (lockA < lockB, not necessarily adjacent); reused from 7.3 ksplit_lock_alloc.md | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186 allocateLockPair --> |
| L1 | Core S2MM, pong BD (emitted first) | pong buffer, pong BD id | next_bd = ping BD id; no linked_bd | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:951 getI32IntegerAttr(pingBdId) --> |
| L2 | Core S2MM, ping BD (emitted second, first to run) | ping buffer, ping BD id | next_bd = pong BD id; linked_bd = pong BD; so the chain is ping, pong, ping, ... | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:979-985 linked_bd = pong BD --> |
| L3 | Core S2MM, BD repeat | ping/pong chain | repeat=1: the next_bd chain re-arms the DMA, no host repeat count | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:683-686 ping-pong BD chaining --> |
| L4 | Lock pair per buffer (window op) | ping lock (pinglockid), pong lock (ponglockid) | each buffer has its own lock: the DMA acquires the free side of the buffer it fills and releases its ready side, the core does the opposite on the other buffer, so DMA fills one buffer while the core reads the other | <!-- ref src/mlir/mlirfront/AieDialect.cc:516 arg_pinglockid --> |
| L5 | Host BD / kernel, input S2MM protocol (per buffer) | lockA / lockB of that buffer | DMA: acquire lockA (val=-1), release lockB (val=1); kernel: acquire lockB (data ready), release lockA (buffer free) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:308-309 acquire lock_id=N+1 (data ready), release lock_id=N (buffer free) --> |
| L6 | .depth=1 (single buffer) | one buffer, one BD, one lock pair | buffer_mode 0, num_buffers 1: one BD, so the DMA can refill only after the core releases the buffer; the core waits for every block and DMA and compute do not overlap. With no next_bd cycling the single BD fires once, so the compiler accepts .depth=1 only when numIterations==1 (else emitError: Single-buffer DMA cannot re-arm) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:490 means single buffer --> |
| L7 | .depth=1 compile check | single BD, numIterations | pp_depth=1 with numIterations above 1 is rejected (the core would deadlock on its second acquire_window); fix is pp_depth 2 or a larger max_buffer_bytes | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:322 pp_depth=1 requires numIterations==1 --> |

**Exemption (reader note).** The check of row L7 skips an input that follows
the kernel's blocks (`c.kb.follow && c.isInput`), the trace-routed input of the
6.1 GEMM. That is why 6.1 step 0 ran A with one buffer for 32 rounds and
passed the golden check (6.1, section 2: `A: one buffer, 32 rounds`). For such
an input `.depth = 1` is legal; it only removes the overlap (section 4).
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:327 !(c.kb.follow && c.isInput) -->
<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:128 A: one buffer, 32 rounds -->

**Which path this chapter describes.** In the dfschedule core-tile lowering
both the ping BD and the pong BD are written with the same
t.bdAcquireLockId / t.bdReleaseLockId (one lock pair shared by both buffers),
while the window op (row L4) and the dfschedulemanager.cpp ping BD keep
per-buffer lock ids. Rows L1-L3 describe the dfschedule BD chain; row L4 the
window op. Either way the protocol of row L5 holds per lock pair.
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:952-954 t.bdAcquireLockId -->
<!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:1239-1241 ping release -->

```
.depth = 2, input A, one core          (derived from rows L1-L5)
shim MM2S --> tile S2MM: BD ping --next_bd--> BD pong --next_bd--> BD ping ...
                 |  acquire free side, write strip, release ready side
                 v
core:  load_buffer(a) = acquire ready side of ping | compute strip s | release(a) = free side
       load_buffer(a) = acquire ready side of pong | compute strip s+1 | release(a)
```

## 4. Timing

The timelines below are Arthur's (task `2026-11-21_tut07-pingpong`, P2).
**derived, not measured.** Every number is arithmetic from 6.1 section 4. No
trace was captured for these pictures.

Inputs (all from 6.1):

- DMA per A strip: 4 KB at ~2.3 GB/s ≈ ~2 us; 32 strips ≈ ~57 us per launch
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:173-175 32 strips ~57 us -->
- Compute per strip: 16 x 64 x 256 = 262,144 scalar MACs per core --
  milliseconds per launch, so each compute block C is far longer than the
  ~2 us DMA block D
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:176-178 262,144 MACs per strip -->
- Cost of depth=2: one more 4 KB buffer per core + its BDs, written over the
  configuration bus at ~0.4 us/word: 16,384 words ≈ ~6 ms per launch
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:179-183 16,384 words, ~6 ms on its own -->

Timelines (derived, not measured; not to scale: C >> D):

```
depth=1 (one A buffer: DMA and core take turns)        derived, not measured
DMA  : |D0|        |D1|        |D2| ...  32 x ~2 us
core :    |---C0---|   |---C1---|   |---C2--- ...
         core waits ~2 us per strip -> at most ~57 us exposed per launch
```

```
depth=2 (ping/pong A buffers: D(i+1) fills while C(i) runs) derived, not measured
CFG  : |== +~6 ms config: extra 4 KB/core image + BDs ==|
DMA  :                                                   |D0|D1|     |D2|    ...
core :                                                      |---C0---|---C1---| ...
         saves <= ~57 us of DMA wait, pays ~6 ms of configuration
```

On this compute-bound kernel the core never waits long for A: ping-pong can
hide at most ~57 us per launch and costs ~6 ms of configuration. Net: slower.
Consistent with that, 6.1 reports the end-to-end launch time: step 0 mean
90.704 ms vs step 1 (depth=2) mean 101.053 ms.
<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:166-167 mean=90.704 -->
That ~10 ms is a whole-launch measurement, not a measured overlap; the lanes
above are derived from the 6.1 arithmetic only.

## 5. Lifecycle: who allocates, owns and frees

Sequence for one launch: **load -> allocate -> configure -> run -> sync -> release**.

| Resource | Allocated by | Owned by (while live) | Freed |
|---|---|---|---|
| ping / pong buffers of A (and C) | compiler, in core data memory: `kernel.cc` gains `buf_in_pong_0`, 4 KB per core (`tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:157-158`) | alternates DMA <-> core under the buffer's lock pair: the DMA from acquiring the free side until it releases the ready side, the core from `aie::load_buffer` until `aie::release(a)` (rows L4, L5) | never at runtime: it is part of the kernel's data image, rewritten on every launch (`tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:179-183`) |
| lock pair per buffer | compiler: `RoutingTile::allocateLockPair` (row L0, `src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186`) | the buffer's owner id, for the whole program | partition teardown (no per-lock release call at 43fcb8f) |
| BDs (ping, pong, chained) | compiler at lowering, emitted into `host.cc`: 88 -> 104 with A `{.depth = 2}` (`tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:157-158`) | the tile DMA, cycling ping -> pong -> ping by next_bd (rows L1-L3) | partition teardown; rewritten with the configuration on every launch |
| resident B buffer (for contrast) | compiler, one 16 KB buffer | the core, the whole launch | released once after the strip loop (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45`) |

<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:157-158 buf_in_pong_0 -->
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45 aie::release(b); -->

Rules that follow from the table:

- Nothing in this chapter is allocated or freed by the host at runtime: the
  buffers, lock pairs and BDs are fixed at compile time. What changes per
  launch is only the configuration write, and `.depth = 2` makes it bigger.
- Ownership of a ping/pong buffer changes at every lock release; a host BD
  repeat count that disagrees with the kernel's acquires hangs (7.3, row L5).
- Choose `.depth = 2` when the DMA time per block is comparable to the compute
  time per block; on a compute-bound kernel (as here) `.depth = 1` on the input
  is cheaper. The default (no option) is 2 buffers, not 1, and an output is
  always 2.
