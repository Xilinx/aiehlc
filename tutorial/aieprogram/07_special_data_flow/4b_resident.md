# 7.4b Special data flow: resident buffers (`.resident`)

> **Data flow: resident.** A resident input is one block per core, loaded once
> into ONE buffer and held by the core for the whole launch. The DMA sends it
> once, not with every round of the other ports. The kernel asks for it with
> `{.resident = true}` on `aie::load_buffer`. This chapter is the second half
> of [7.4 ping-pong buffers](4_pingpong.md): it follows that one option from
> the kernel line to the BD/lock sequence, the cost and the buffer lifetimes.

> **Status.** Doc only: no new example. The program is the 6.1 step-1 GEMM
> (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc`). All refs are
> checked against base `43fcb8f` by script/check_refs.sh. The byte counts in
> section 4 are **derived, not measured**; the one measured number is from
> the resnet engine, not from 6.1.

## The whole program

The program is the step-1 GEMM of chapter 6.1; its full listing is in
[6.1, "The whole program (step 1)"](../06_gemm_optimization/1_low_level_schedule.md)
and is not copied here (6.1 keeps it in sync with the source). The lines that
matter for this chapter are the resident load of B before the strip loop and
its release after it
(`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:30-45`):
<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:14 The whole program (step 1) -->
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:30-45 {.resident = true}); -->

```cpp
const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true}); // this column's 64 rows of B
for (int s = 0; s < cm / GD_TS; s++) {
    const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2});
    ...
    aie::release(a);
    aie::commit(c);
}
aie::release(b);
```
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:30 {.resident = true}); -->
<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45 aie::release(b); -->

## 1. The idea: one buffer, one lock pair, owned by the core all launch

`w` is taken once, before the strip loop, and released once, after it:
`aie::release(b)` is the last statement of the kernel
(`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45`). Between the take
and the release the core owns the buffer; the DMA never refills it. A and C
keep cycling their ping-pong buffers (7.4) while B stays put.

| Declaration | buffers on the core | DMA sends the block | held by the core |
|---|---|---|---|
| `{.depth = 2}` (7.4) | 2 (ping + pong) | once per round | one round |
| `{.depth = 1}` (7.4) | 1 (single) | once per round | one round |
| `{.resident = true}` | 1 | once per launch | the whole launch |

Rule of thumb from 6.1: resident costs nothing extra; it saves a buffer and
re-sending.
<!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:193 it saves a buffer and re-sending -->

## 2. Kernel: declaration -> one buffer

The rows below are Maxwell's (task `2026-11-21_tut07-resident`, P1).

The trace resolves the buffer count per port in one line; resident wins over
everything:

```c++
const int pp = opt.resident ? 1 : opt.depth > 0 ? opt.depth : g_pp;
```
<!-- ref src/llvm/convautotile/kernel_trace.cpp:283 opt.resident ? 1 -->

| Rule | What the trace does at base 43fcb8f | Source |
|---|---|---|
| ONE buffer | the port's buffer count is `opt.resident ? 1 : ...`, so a resident port has exactly one buffer (pp = 1), whatever the kernel-wide ping-pong depth | `src/llvm/convautotile/kernel_trace.cpp:283` |
| whole rows | `.resident` (like `.tile`) is only honoured on an input whose blocks are whole rows of the tensor; otherwise it is an error (the DMA would not apply it, the core would silently get two buffers) | `src/llvm/convautotile/kernel_trace.cpp:1079-1080` |
| no `.depth > 1` | `.resident` with `.depth = 2` is an error: one buffer contradicts ping-pong | `src/llvm/convautotile/kernel_trace.cpp:1089-1091` |
| exactly 1 block per core | `o.resident && blocks != 1` is an error: a resident block is loaded once (outside the loop) and held for the whole launch; a port taken in a strip loop wants `{.depth = 2}` instead | `src/llvm/convautotile/kernel_trace.cpp:1092-1095` |
| conv (spatial halo) | `{.depth = 1}` is refused on a halo kernel; the error points at `{.resident = true}` as the way to get one buffer for a block taken once | `src/llvm/convautotile/kernel_trace.cpp:1474-1477` |
| fit fallback | on a non-halo kernel, ports without `.depth`/`.resident` drop to one buffer when that makes the set fit; if it still does not fit, the error suggests `.resident` for an input taken once | `src/llvm/convautotile/trace_routing.cpp:354-382` |

<!-- ref src/llvm/convautotile/kernel_trace.cpp:1079-1080 whole rows -->
<!-- ref src/llvm/convautotile/kernel_trace.cpp:1089-1091 contradicts it (drop .depth) -->
<!-- ref src/llvm/convautotile/kernel_trace.cpp:1092-1095 o.resident && blocks != 1 -->
<!-- ref src/llvm/convautotile/kernel_trace.cpp:1474-1477 {.resident = true} for a block taken once -->
<!-- ref src/llvm/convautotile/trace_routing.cpp:354-382 load an input that is taken once with {.resident = true} (one buffer) -->

The flag reaches the routing attribute (KernelBlockAttr) from the traced port
options.
<!-- ref src/llvm/aiehlc.cc:270 opts.resident -->

What the errors look like, verbatim from the conv scheduling chapter
(`tutorial/aieprogram/06_gemm_optimization/3_conv_scheduling.md:197-198`):

| Change | Result |
|---|---|
| slab `{.depth = 1}` | `ERROR: ... win_a (IFM): {.depth = 1} is not supported on a spatial-halo (conv) kernel yet -- its window lowering is ping-pong; use {.depth = 2}, or {.resident = true} for a block taken once` |
| slab `{.resident = true}` | `ERROR: ... {.resident = true}, but the kernel takes 16 blocks of this port per core; a resident block is loaded once (outside the loop) and held for the whole launch` |

<!-- ref tutorial/aieprogram/06_gemm_optimization/3_conv_scheduling.md:197-198 a resident block is loaded once (outside the loop) -->

## 3. Hardware path and BD / lock sequence

The rows below are Amit's (task `2026-11-21_tut07-resident`, P0). They are
derived from compiler / runtime source and doc/design, not from a lowered run
or a hardware trace. Locks are named by role only (lockA = buffer free,
lockB = data ready).

Source facts:
- `.resident = true` means one block per core, held for the whole launch, ONE buffer; the DMA sends it once, not with every round of the other ports; it cannot be combined with `.depth = 2`. <!-- ref doc/design/trace_to_routing.md:184 ONE buffer; the DMA sends it once -->
- On `fullconnect_auto = 0` (spatial-halo conv) only a `.resident` input is checked / followed (`resolveKernelBlock`): the conv weights get one 3136 B buffer, sent once. <!-- ref doc/design/trace_to_routing.md:250-256 only a `.resident` input is -->

| Row | Side | Resource | Rule | Source |
|---|---|---|---|---|
| L0 | Which lowering path follows .resident | KernelBlockUse (check / follow) | resolveKernelBlock returns "not followed" when fullconnect_auto = 0 and the input is not resident, so on the spatial-halo conv path ONLY a {.resident} input leaves its schedule to the kernel; on the GEMM path (fullconnect_auto = 1) every kernel-block port is followed, resident or not | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_common.cpp:686-688 getResident() --> |
| L1 | Buffer count (compiler) | one core buffer | for a followed input the core window is the kernel block and ppDepth comes from the port's TensorRoutingInfo; resident gives ppDepth 1 = one buffer (buffer_mode 0, num_buffers 1) | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:264-268 t.ppDepth = static_cast --> |
| L2 | Lock pair allocation (compiler) | lockA, lockB from the tile's lockPool_ | one buffer, so one lock pair (ping-pong needs a pair per buffer); allocateLockPair takes the first two unused pool entries for the owner, as in 7.3 / 7.4 | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186 allocateLockPair --> |
| L3 | Core S2MM BD, queued once per launch | single BD, numIterations | no next_bd cycling, the BD fires exactly once; pp_depth=1 is accepted only with numIterations==1, which holds for a resident block (one block per core), so the BD is queued once per launch, not once per round of the other ports | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_kernel.cpp:322-326 pp_depth=1 requires numIterations==1 --> |
| L4 | DMA / core protocol on the one buffer | lockA / lockB | DMA: acquire lockA (buffer free), write the block, release lockB (data ready) once; core: acquire lockB once and keep the block for the whole launch; the core's release of lockA comes only at `aie::release(b)`, so the DMA never refills it within the launch | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:308-309 acquire lock_id=N+1 (data ready), release lock_id=N (buffer free) --> |
| L5 | Release-only BD form (resnet engine only) | producer BD with LckAcqEn = 0 | acq_id below zero with rel_id set encodes a release-only BD: the producer does not acquire, it only releases the ready lock, because the consumer never hands the slots back within the task (resnet engine resident input); this is the resnet engine form, not the 6.1 output | <!-- ref src/mlir/runtime/aie_kernel_runtime.h:219-221 release-only (LckAcqEn = 0) --> |
| L6 | Check on the resident block (compiler) | core window, round count | a {.resident} input on any path is checked: its final core window and round count must be the kernel's block and block count, else the build fails (otherwise core and shim DMA disagree, a silent wait_io hang) | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passblueprintlowering/helper/flowtransfer_internal.h:218 a {.resident} input on any path --> |

```
{.resident = true}, input B, one core       (derived from rows L1-L4)
shim MM2S --> tile S2MM: one BD, queued once per launch (no next_bd)
                 |  acquire lockA (free), write the 16 KB block, release lockB (ready)  -- once
                 v
core:  load_buffer(b) = acquire lockB      (before the strip loop)
       strip 0 .. 31: compute with w; A and C cycle ping/pong (7.4)
       release(b)     = release lockA      (after the strip loop, end of kernel)
```

## 4. Timing

The numbers below are Arthur's (task `2026-11-21_tut07-resident`, P2). Each is
either cited to a committed line or marked **derived, not measured**.

- Sent once: "one block per core, one buffer, sent once" -- the same thing the
  compiler already did in 6.1 step 0.
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:147 {.resident = true} -->
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:153 sent once -->
- Buffer: the B block is the 16 KB term of the 26 KB per-core budget
  (16 KB B + 2 x 4 KB A ping-pong + 2 x 1 KB C ping-pong).
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:161 26 KB -->
- 32 strips per launch, from step 1's loop ("32 strips ~57 us per launch").
  <!-- ref tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md:175 launch -->

| B handling | bytes per core per launch | bytes, 16 cores | B buffer KB per core | source |
|---|---|---|---|---|
| resident (sent once) | 64 x 256 = 16,384 B | 262,144 B (256 KB) | 16 | block size from 6.1; totals derived, not measured |
| re-sent every round (32 strips) | 32 x 16,384 = 524,288 B | 8,388,608 B (8 MB) | 16 (x2 = 32 if ping-ponged) | derived, not measured |

The "re-sent every round" row is arithmetic (32 strips x one 16 KB B block x
16 cores); no build with a non-resident B was run.

```
B resident vs re-sent, one core          derived, not measured; not to scale
resident : DMA |B 16 KB|                                   (once)
           core      |C0|C1|C2| ... |C31|  release(b)
re-sent  : DMA |B|   |B|   |B|   ...  |B|                  (32 x 16 KB = 512 KB)
           core   |C0|  |C1|  |C2| ...  |C31|
```

The one measured data point is on the resnet engine, not 6.1: making the input
resident for the 14/7 px layers (deep layers had re-streamed their input
P = 4..8 times; now once) took the full network to **2.35 ms** (from 2.75 ms in
the previous row). It shows the direction of the effect, not a 6.1 number.
<!-- ref doc/design/resnet_aie_engine.md:140 resident input -->

## 5. Lifecycle: who allocates, owns and frees

Sequence for one launch: **load -> allocate -> configure -> run -> sync -> release**.

| Resource | Allocated by | Owned by (while live) | Freed |
|---|---|---|---|
| resident B buffer (1 per core, 16 KB) | compiler, in core data memory: ppDepth 1, one buffer (rows L1, section 2 ONE buffer) | the DMA from launch start until it releases the ready lock once (row L4); then the core, from `aie::load_buffer(b, ..., {.resident = true})` until `aie::release(b)` (`example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45`) | never at runtime: part of the kernel's data image; handed back to the free side at `aie::release(b)`, once per launch |
| lock pair (1: lockA free, lockB ready) | compiler: `RoutingTile::allocateLockPair` (row L2, `src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186`) | the buffer's owner id, for the whole program | partition teardown (no per-lock release call at 43fcb8f) |
| core S2MM BD for B (one, no next_bd) | compiler at lowering | the tile DMA; the host issues it once per launch (row L3) | partition teardown; rewritten with the configuration on every launch |
| release-only BD (LckAcqEn = 0) | resnet engine only, not 6.1 (row L5) | the producer: releases the ready lock without acquiring | end of the task; the consumer never hands the slots back within it |

<!-- ref example/tileprogram/ccode/simplegemm_opt1_schedule.cc:45 aie::release(b); -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186 allocateLockPair -->

Rules that follow from the table:

- The DMA fills the buffer once; the core owns it from the take to
  `aie::release(b)`. If the kernel takes the port again inside a loop, it is not
  resident (section 2, exactly 1 block per core) and the build fails.
- Nothing here is allocated or freed by the host at runtime: buffer, lock pair
  and BD are fixed at compile time; per launch the host issues the one BD once.
- Choose `{.resident = true}` for an input whose block per core is the same for
  every round (GEMM weights B, conv weights). Choose `{.depth = 2}`
  ([7.4](4_pingpong.md)) for an input that changes every round.
