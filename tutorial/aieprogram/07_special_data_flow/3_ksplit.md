# 7.3 Special data flow: K-split accumulation (partial sums, two-level combine)

> **Data flow: K-split.** The reduction dimension K of a GEMM is cut into
> chunks. **Level 1:** one core walks the chunks in time and keeps the partial
> sum in its own accumulator. **Level 2:** when K is split across cores, each
> core holds only a partial sum, and someone has to add them. This chapter
> follows both levels from the kernel line to the BD/lock sequence, the buffer
> lifetimes and the cost. The broadcast that delivers the A and B chunks is
> chapter [7.1](1_broadcast.md).

> **Status.** Doc only: no new example. Level 1 is the chapter 1a GEMM
> (`example/tileprogram/ccode/simplegemm_kernel_port.cc`); level 2 is the
> host combine of chapter 3 (`tutorial/aieprogram/03_reduction/reduction.md`).
> All refs are checked against base `43fcb8f` by script/check_refs.sh.

## The whole program

The whole program -- kernel and host, 4x4 mesh (the listing is kept in sync with
the source by `tutorial/aieprogram/sync_code.py`).

<!-- code: example/tileprogram/ccode/simplegemm_kernel_port.cc -->
```cpp
/******************************************************************************
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

## 1. The idea: walk K in time, combine partial sums in space

`C[m, n] = sum_k A[m, k] * B[n, k]`. Every output element needs the whole K
row. When the A and B slices for all of K do not fit in core memory next to
the stationary C block, K is cut into chunks of `tk`:

| level | where K is cut | who holds the partial sum | who adds the partials | in aiehlc at 43fcb8f |
|---|---|---|---|---|
| 1 | in time, inside one core (K rounds) | the core's accumulator `acc[]` | the core, `acc += s` every round | yes: the 1a GEMM, 4 K rounds |
| 2 | in space, across cores | each core, one partial per output | the host after the launch (or a second launch) | host add yes; cascade no |

GEMM today only needs level 1: each core owns whole output elements and walks
all of K itself. Level 2 appears as soon as two cores each sum part of the
same output -- that is the reduction of chapter 3.

## 2. Level 1: the K walk in the core

The core fixes its C block and the K chunk
(`example/tileprogram/ccode/simplegemm_kernel_port.cc:11`):
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:11 tk = 64 -->

```cpp
    const int tm = 16, tn = 16, tk = 64; // C block per core, K chunk per round
```

It zeroes a 16x16 int32 accumulator **once**, before the K walk
(`example/tileprogram/ccode/simplegemm_kernel_port.cc:15-17`), then walks
`256 / tk` = 4 K rounds (`example/tileprogram/ccode/simplegemm_kernel_port.cc:18`):
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:15-17 acc[i] = 0; -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:18 for (int j = 0; j < 256 / tk; j++) -->

```cpp
    for (int j = 0; j < 256 / tk; j++) { // K chunks
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
        ...
                acc[m * tn + n] += s;
        ...
        aie::release(a);
        aie::release(b);
    }
```
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:19-20 j * tk}, {tm, tk}) -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:26 acc[m * tn + n] += s; -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:28-29 aie::release(b); -->

| term | value | step per round | shape | what it means |
|---|---|---|---|---|
| A column offset `j * tk` | 0, 64, 128, 192 | 64 (`j`) | `tk` = 64 | one K chunk of A per round, no overlap |
| B column offset `j * tk` | 0, 64, 128, 192 | 64 (`j`) | `tk` = 64 | the same K chunk of B |
| C offset `{pm * tm, pn * tn}` | fixed | none | 16 x 16 | C is **stationary**: one block for the whole walk |

Only after the last round is `acc` narrowed to int8 and committed **once**
(`example/tileprogram/ccode/simplegemm_kernel_port.cc:31-33`). There is no
partial-sum traffic at level 1: the partial sum never leaves the core.
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:31-33 aie::commit(c); -->

### Where the K chunk and kRounds come from

aiehlc reads the effective K and the number of K rounds from the derived
tiling (`src/llvm/aiehlc.cc:409`, `src/llvm/aiehlc.cc:412`); kRounds = 1 means
no K walk. The chunk is set by the core memory budget: with C stationary
(tileM x tileN), what is left is shared by the A and B slices, so
`maxK = remaining / (tileM + tileN)` (`src/llvm/aiehlc.cc:2549-2550`).
<!-- ref src/llvm/aiehlc.cc:409 int64_t effK = dp.effectiveK -->
<!-- ref src/llvm/aiehlc.cc:412 int64_t kRounds = dp.kRounds -->
<!-- ref src/llvm/aiehlc.cc:2549-2550 int64_t maxK = remaining / (tileM_eff + tileN_eff); -->

The debug note on the larger instance (256x256 int8, `effectiveK=64`,
`kRounds=4`) states the same structure: C stationary, partial products
accumulated across K rounds, one write after all K rounds
(`doc/debug/k_accumulate_c_stationary_shim_stuck.md:14`,
`doc/debug/k_accumulate_c_stationary_shim_stuck.md:46`,
`doc/debug/k_accumulate_c_stationary_shim_stuck.md:56`).
<!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:14 kRounds=4 -->
<!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:46 accumulates partial products across K-rounds -->
<!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:56 one write after all k-rounds -->

### ASCII data path (one core, level 1)

```
DDR A (64 x 256)            DDR B (64 x 256)
  rows pm*16 .. +15           rows pn*16 .. +15
  |k0|k1|k2|k3|  64 each      |k0|k1|k2|k3|  64 each
       |                           |
       v shim MM2S, 4 K rounds     v shim MM2S, 4 K rounds
  row broadcast (7.1)         column broadcast (7.1)
       |                           |
       v tile S2MM, ping/pong      v tile S2MM, ping/pong
  core (pm, pn):  round j:  x = A[.., j*64 ..], w = B[.., j*64 ..]
                  acc[16x16] += x * w^T     <- partial sum stays on the core
                  aie::release(a), aie::release(b)   (buffer free for chunk j+1)
       |  after j = 3 only
       v  y = (int8) acc ;  aie::commit(c)
  tile MM2S -> shim S2MM -> DDR C (64 x 64), block {pm*16, pn*16}
```

The input buffers turn over 4 times; the output buffer once.

## 3. BD / lock sequence

The rows below are Amit's (task `2026-11-21_tut07-ksplit`), pasted unchanged.

PROBE lowered_kacc_bds=no example/tileprogram/ccode script/test/flowcheck_data (43fcb8f: no XAie_DmaSetLock / kRounds in either dir; git grep rc=1)

### K-split flow (K-accumulation, C-stationary): BD / lock table

Derived from doc/debug + kernel source + the compiler's lock allocator, not lowered 1a output.

Lock pair: every buffer's lock pair (lockA, lockB) is handed out by the compiler's
`RoutingTile::allocateLockPair`. It walks the tile's `lockPool_` from index 0 and marks the
first two unused entries as used by the same owner, so lockA < lockB, but the two ids are
not necessarily adjacent (a lock already taken between them is skipped). The debug doc writes
this pair as the symbolic "N / N+1"; read N as lockA and N+1 as lockB, not as consecutive ids.
No concrete lock id or value beyond those on the cited lines is given.

Source facts:
- 256x256 int8 GEMM on a 4x4 mesh, effectiveK=64, kRounds=4. <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:14 kRounds=4 -->
- C is stationary: each core owns a 64x64 output region and accumulates partial products across K-rounds (level 1). <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:46 accumulates partial products across K-rounds -->
- K-round 0..3: A[0:64, k] x B[k, 0:64] -> accum += partial_k (level 1: the core's accumulator). <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:51-54 accum += partial_ -->
- Output written once: saturate(accum) -> C after all k-rounds. <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:56 one write after all k-rounds -->
- Kernel loop: for kr < k_rounds, A acquires num_a_rounds, B acquires num_b_rounds, per window per k-round. <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:64-76 acquire_input_window -->
- Totals: A num_a_rounds x k_rounds, B num_b_rounds x k_rounds, C num_c_rounds (once). <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:87-89 k_rounds -->
- Round count mismatch between host BD repeat and kernel acquires -> hang on wait_io TIMEOUT. <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:4 wait_io TIMEOUT -->

| Row | Side | Resource | Rule | Source |
|---|---|---|---|---|
| L0 | lock pair allocation (compiler) | lockA, lockB from the tile's lockPool_ | allocateLockPair walks lockPool_ and marks the first two unused entries (same ownerId); lockA < lockB, not necessarily adjacent; returns nullopt if fewer than two are free | <!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186-204 allocateLockPair --> |
| L1 | Host BD, input S2MM (A and B windows) | lock N / N+1 (= lockA / lockB) | acquire lock_id=N (val=-1), release lock_id=N+1 (val=1); repeats num_a_rounds x k_rounds (A), num_b_rounds x k_rounds (B) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:307-308 acquire lock_id=N (val=-1), release lock_id=N+1 (val=1) --> |
| L2 | Kernel, input (win_a / win_b, per window per k-round) | same pair, opposite side | acquire lock_id=N+1 (data ready), release lock_id=N (buffer free) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:309 acquire lock_id=N+1 (data ready) --> |
| L3 | Kernel, output (win_c, after all k-rounds) | output lock pair (its own allocateLockPair call) | acquire lock_id=N (buffer free), fill data, release lock_id=N+1 (data ready); num_c_rounds times | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:311-312 release lock_id=N+1 (data ready) --> |
| L4 | Host BD, output MM2S (C drain) | same pair | acquire lock_id=N+1 (val=-1), release lock_id=N (val=1) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:313 acquire lock_id=N+1 (val=-1), release lock_id=N (val=1) --> |
| L5 | Failure mode | any port | lock mismatch -> deadlock (kernel waits for a lock the host never releases) | <!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:316 Lock mismatch --> |

Alloc / own / free: the lock pair is allocated at compile time by allocateLockPair (L0) and
owned by the buffer's ownerId; the accumulator is owned by the core for all kRounds (alloc at
kr=0, free after the single C write); A/B buffers are freed every round by the kernel release
of lockA (L2); the C buffer is freed by the host BD release of lockA (L4).

Note: the debug note's instance has a 64x64 C block per core; the 1a program
above has 16x16 per core and 4 K rounds. The lock rule (rows L1-L4) is the
same; only the repeat counts differ.

## 4. Level 2: combining partial sums across cores

Once K is split **across** cores, each core holds only a partial sum. Where it
is combined decides the cost. Chapter 3 lists the options; the table is quoted
from `tutorial/aieprogram/03_reduction/reduction.md:101-106`:

| Where the partial sums are combined | How | In aiehlc today |
|---|---|---|
| nowhere: each core owns whole rows | split rows across cores | fine on a mesh with one core per row band (4x1); on 4x4, splitting a row band's rows over the 4 cores puts both pids on the output's row dimension, which the router rejects (section 4) |
| on the host, after the launch | each core writes its partial sum, the host adds them | **this chapter** -- a few additions per row on the Arm |
| a second kernel launch | partial sums to DDR, another kernel adds them | works, but every launch re-loads and re-configures the array |
| in the array | neighbouring cores pass accumulators over the **cascade** interface (a direct core-to-core link), or through shared neighbour memory | the AIE way; not expressible in aiehlc yet |
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:101-106 Where the partial sums are combined -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:103 nowhere: each core owns whole rows -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:104 the host adds them -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:105 every launch re-loads and re-configures the array -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:106 the AIE way; not expressible in aiehlc yet -->

("this chapter" in the quote is chapter 3.) The 4x4 instance there
(`tutorial/aieprogram/03_reduction/reduction.md:108-111`):

> On 4x4 we combine the two techniques of chapter 2: mesh row `pm` owns 64 rows
> and receives all of their columns by **broadcast**; core `(pm, pn)` sums only
> **column chunk `pn`** of every row (a quarter of each row) and writes one int32
> **partial sum** per row. The host adds the 4 partial sums of each row.
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:108-111 The host adds the 4 partial sums of each row. -->

So in aiehlc at 43fcb8f a two-level K-split is: level 1 in each core (its K
chunk, walked in rounds as in section 2), level 2 on the host after the
launch (or in a second launch). The cascade combine is **not** expressible
yet; do not plan on it.

```
core (pm, 0) --partial--\
core (pm, 1) --partial---\   tile MM2S -> shim S2MM -> DDR y (int32 partials,
core (pm, 2) --partial---/     4 per output)
core (pm, 3) --partial--/
                               host, after <<<mesh>>> returns:
                               out[r] = y[r][0] + y[r][1] + y[r][2] + y[r][3]
```

## 5. Lifecycle: who allocates, owns and frees

Sequence for one launch: **load -> allocate -> configure -> run -> sync -> release**.

| Resource | Allocated by | Configured by | Owned by (while live) | Released by |
|---|---|---|---|---|
| Partition / mesh (`device.partition`) | host `main` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:41`) | runtime at load | runtime, whole program | `__Runtime_device_teardown` -> `XAie_PartitionTeardown` (`src/mlir/runtime/aie_runtime.c:3291-3298`) |
| DDR A, B, C | host `main` via `__Runtime_Alloc` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:42-44`) | host fills A, B before the launch | host; lent to the shim BDs during `gemm<<<mesh>>>` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:48`) | host `free` after the check (`example/tileprogram/ccode/simplegemm_kernel_port.cc:52-54`) |
| Shim and tile BDs for the 4 K rounds of A and B | compiler at lowering: `RoutingTile::allocateBd` (`src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:125`) | runtime at load: address + length (`src/mlir/runtime/aie_runtime.c:3681`) | the DMAs until the BD chain ends | partition teardown (no per-BD release call at 43fcb8f) |
| Lock pairs guarding the A, B, C tile buffers | compiler: `RoutingTile::allocateLockPair` at lowering (`src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186`) | runtime at load: `XAie_DmaSetLock` on the BD (`src/mlir/runtime/aie_runtime.c:3684-3685`) | per tile, alternately DMA and core (rows L1-L4) | partition teardown |
| Input ping/pong (A chunk, B chunk) | compiler at lowering | runtime at load | DMA while it writes chunk j; the core from `aie::load_buffer` to `aie::release` | the kernel, **every K round**: `aie::release(a)`, `aie::release(b)` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:28-29`) |
| Accumulator `acc[16*16]` (int32 partial sum) | core stack, `int acc[tm * tn]` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:15`) | zeroed once by the kernel (`example/tileprogram/ccode/simplegemm_kernel_port.cc:15-17`) | the core, for **one launch** (all 4 K rounds) | dies with the kernel's stack frame after the last write |
| Output C block | compiler at lowering; `aie::store_buffer` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:14`) | runtime at load | the core for the whole K walk | the kernel, **once**, after the last `j`: `aie::commit(c)` (`example/tileprogram/ccode/simplegemm_kernel_port.cc:31-33`); then the output BD drains it (row L4) |
| Level-2 partials `y` (DDR, int32, 4 per row) | host via `__Runtime_Alloc` (`tutorial/aieprogram/03_reduction/reduction.md:63`) | host zeroes it before the launch | cores write it during the launch; the host owns it afterwards | the host, after the combine loop (`tutorial/aieprogram/03_reduction/reduction.md:80-84`); chapter 3's listing leaves the free to program exit |

<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:41 device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS) -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:42-44 __Runtime_Alloc(GEMM_N * GEMM_K) -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:48 gemm<<<mesh -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:52-54 free(C); -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:15 int acc[tm * tn]; -->
<!-- ref example/tileprogram/ccode/simplegemm_kernel_port.cc:14 aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn}) -->
<!-- ref src/mlir/runtime/aie_runtime.c:3291-3298 XAie_PartitionTeardown(dev) -->
<!-- ref src/mlir/runtime/aie_runtime.c:3681 XAie_DmaSetAddrLen(&DmaInst, dma_addr, (uint32_t)len); -->
<!-- ref src/mlir/runtime/aie_runtime.c:3684-3685 XAie_DmaSetLock(&DmaInst, XAie_LockInit(acquire_lock_id, acquire_lock_val), -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:125 RoutingTile::allocateBd(int ownerId) -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:186 RoutingTile::allocateLockPair(int ownerId, int &lockA, int &lockB) -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:63 __Runtime_Alloc(OPS_R * 16) -->
<!-- ref tutorial/aieprogram/03_reduction/reduction.md:80-84 the level-2 combine: add the 4 partial sums of the row -->
<!-- sym __Runtime_device_teardown -->

Rules that follow from the table:

- The accumulator is the partial sum. It is core stack, lives for one launch,
  and is never seen by the DMA; only its narrowed result is.
- The input ping/pong is released every K round; the C block is released once,
  after the last `j`. A host BD that repeats the input or output BD a
  different number of times than the kernel acquires it hangs in
  `wait_io` (`doc/debug/k_accumulate_c_stationary_shim_stuck.md:4`, row L5).
- At level 2 the host owns the partials after the launch returns: combine
  first, then free.
<!-- ref doc/debug/k_accumulate_c_stationary_shim_stuck.md:4 wait_io TIMEOUT -->

## 6. What it costs

**Level 1.** No extra DDR or stream bytes: every A and B byte of the core's
K range is moved once per core (the broadcast of 7.1 shares it between
cores). The price is core memory for `acc` (16 x 16 x 4 B = 1 KB here) and
one lock round trip per input buffer per K round (4 per buffer here).

**Level 2.** Each core writes an int32 partial per output instead of an int8
result, and the host adds them after the launch (or a second launch re-loads
and re-configures the array). The in-array cascade combine would avoid both
but is not expressible in aiehlc at this base.

Takeaway: K-split is a load whose K offset moves by its shape each round
while the output stays put. Inside one core it costs nothing but an
accumulator; across cores it costs a combine, which today happens on the host.
