# Debug Analysis: K-Accumulation C-Stationary SHIM Stuck

**Date:** 2026-05-26
**Symptom:** SHIM MM2S input channels starve core tiles; kernels acquire fewer DMA rounds than expected; output DMA locks never release; system hangs on `wait_io TIMEOUT`
**Root Cause:** `pp_depth` was incorrectly used to split per-k-round data into sub-rounds, halving the DMA buffer size and doubling the round count. This created a size/count mismatch across three computation paths (frontend, host pass, kernel pass), and propagated to incorrect SHIM BD `iter_wrap`, `iter_step_size`, and `repeat_count`.
**Fix:** Decouple `pp_depth` (physical ping-pong buffer count) from data splitting (driven only by `max_buffer_bytes`). Fix all six bug locations across three files.

---

## 1. Background: K-Accumulation with C-Stationary

### 1.1 The GEMM Data Flow

For a 256x256 int8 GEMM on a 4x4 AIE mesh with K-tiling (`effectiveK=64`, `kRounds=4`):

```
Matrix A (256x256)    Matrix B (256x256)    Matrix C (256x256)
input                 input                 output
broadcast             broadcast              gather
on row               on col                per row

┌─────────┐          ┌─────────┐          ┌─────────┐
│ 256×256 │          │ 256×256 │          │ 256×256 │
└────┬────┘          └────┬────┘          └────┬────┘
    │                    │                    │
partition by row     partition by col     partition by row
(4 groups of 64)     (4 groups of 64)    (4 groups of 64)
    │                    │                    │
┌────▼────┐          ┌────▼────┐          ┌────▼────┐
│ 64×256  │ ×4       │ 64×256  │ ×4       │ 64×256  │ ×4
│per group│          │per group│          │per group│
└────┬────┘          └────┬────┘          └────┬────┘
    │                    │                    │
broadcast to          broadcast to         gather from
4 tiles in row       4 tiles in col       4 tiles in row
    │                    │                    │
┌────▼────┐          ┌────▼────┐          ┌────▼────┐
│ 64×256  │          │ 64×256  │          │ 64×64   │
│per tile │          │per tile │          │per tile │ (C-stationary)
│(full K) │          │(full K) │          │         │
└─────────┘          └─────────┘          └─────────┘
```

### 1.2 C-Stationary with K-Accumulation

C is "stationary" — each core tile owns a fixed 64x64 output region and accumulates partial products across K-rounds:

```
Per core tile data flow (4 k-rounds):

K-round 0:  A[0:64, 0:64]   × B[0:64, 0:64]   → accum += partial_0
K-round 1:  A[0:64, 64:128] × B[64:128, 0:64]  → accum += partial_1
K-round 2:  A[0:64, 128:192]× B[128:192, 0:64] → accum += partial_2
K-round 3:  A[0:64, 192:256]× B[192:256, 0:64] → accum += partial_3

Output:     saturate(accum) → C[0:64, 0:64]     (one write after all k-rounds)
```

### 1.3 DMA Acquire/Release Pattern

The kernel code runs this loop:

```cpp
for (int kr = 0; kr < k_rounds; kr++) {
// Phase 1: Receive A chunk (num_a_rounds acquires)
for (int ra = 0; ra < num_a_rounds; ra++) {
    A_ptr = acquire_input_window(win_a);  // lock acquire
    copy A_ptr → local buffer
    release_input_window(win_a);          // lock release
}
// Phase 2: Stream B chunk (num_b_rounds acquires)
for (int rb = 0; rb < num_b_rounds; rb++) {
    B_ptr = acquire_input_window(win_b);  // lock acquire
    compute partial product, accumulate
    release_input_window(win_b);          // lock release
}
}
// Phase 3: Output (after all k-rounds)
for (int rc = 0; rc < num_c_rounds; rc++) {
out = acquire_output_window(win_c);       // lock acquire
copy local → out
release_output_window(win_c);             // lock release
}
```

Total DMA transactions per core:
- Input A: `num_a_rounds × k_rounds` acquires
- Input B: `num_b_rounds × k_rounds` acquires
- Output C: `num_c_rounds` acquires (once, after all k-rounds)

---

## 2. The Bug: pp_depth Splits Data Incorrectly

### 2.1 Correct Semantics

`pp_depth` controls how many **physical ping-pong buffers** are allocated for DMA/compute overlap. It has NO effect on data splitting:

```
per_k_round_data = tile_m × effectiveK = 64 × 64 = 4096 bytes

if per_k_round_data ≤ max_buffer_bytes:
bufferSize = per_k_round_data      → 4096 (fits, no splitting)
numRounds  = 1                      → 1 acquire per k-round
else:
bufferSize = max_buffer_bytes
numRounds  = ceil(per_k_round_data / max_buffer_bytes)

Physical memory = pp_depth × bufferSize per port (for ping-pong overlap)
```

### 2.2 What the Bug Did

All three computation paths divided buffer size by `pp_depth`:

```
WRONG:  bufferSize = per_k_round_data / pp_depth = 4096 / 2 = 2048
    numRounds  = per_k_round_data / bufferSize = 4096 / 2048 = 2

RIGHT:  bufferSize = per_k_round_data = 4096
    numRounds  = 1
```

### 2.3 The Cascade of Mismatches

The wrong buffer size propagated through the entire pipeline:

```
                WRONG (pp_depth=2)          CORRECT
                ──────────────────          ───────
Frontend (aiehlc.cc):
bufferSize        2048                        4096
numRounds         2                           1

Host pass (passblueprinttoschedule.cpp):
Core BD:
pingPongBufSize 2048 elements               4096 elements
numIterations   2                           1
BD len          2048 bytes                  4096 bytes
SHIM BD (input):
iter_wrap       kRounds×numIter = 4×2 = 8   kRounds×1 = 4
iter_step_size  64 bytes                    64 bytes
SHIM BD (OOO output):
iter_wrap       2                           1
iter_step_size  16384                       0
repeat_count    numTiles×2 = 8              numTiles×1 = 4

Kernel pass (passblueprinttoschedulekernel.cpp):
BUF_SZ_IN_0      512 vectors (2048 bytes)     1024 vectors (4096 bytes)
window_init size  512                         1024
numRounds         2×4 = 8                     1×4 = 4
```

---

## 3. Failure Modes

### 3.1 SHIM MM2S Input Starvation

The SHIM sends data based on its BD `iter_wrap` and `iter_step_size`. The core tile expects data sized by `pingPongBufferSize`:

```
SHIM sends: iter_wrap=8, each transfer = tile_m/2 × effectiveK = 2048 bytes
Core expects: BD len=2048, numIterations=2 per k-round, 4 k-rounds = 8 total

In this case, SHIM and core agree on 2048 per transfer, 8 total.
But the kernel has window_init(size=512, numRounds=8):
512 vectors × 4 bytes/vector = 2048 bytes ← matches core BD ✓
numRounds = 8 ← matches total iterations ✓

So with pp_depth=2 the pipeline is internally consistent but WRONG:
- Each acquire gets only half the data the kernel computation expects
- rows_per_round = 2048/64 = 32 (should be 64)
- The kernel computes on 32 rows per round instead of 64
- Result: incorrect output data
```

### 3.2 Output OOO repeat_count Mismatch (SHIM Hang)

The output gather path is where the system **hangs**:

```
Core tile sends:
1 BD × 1 fire = 1 transaction per core
(output = tile_m × tile_n = 4096 bytes, pp_depth=1, so no splitting)

SHIM expects (WRONG):
repeat_count = numCoreTiles × 2 = 8
(hardcoded ×2, assuming pp_depth always splits)

Result:
4 cores send 4 transactions total
SHIM waits for 8 → waits forever → TIMEOUT
```

### 3.3 iter_step_size with iter_wrap=1 (Benign but Confusing)

When `iter_wrap=1`, `iter_step_size` is meaningless — the BD fires once and never advances. Setting `iter_step_size=16384` when `iter_wrap=1` is harmless but misleading during debugging, making it appear that address advancement is configured when it is not active.

### 3.4 Core BD len vs Kernel buf_sz Mismatch (Data Corruption)

If the host pass and kernel pass disagree on buffer size:

```
Host pass sets core BD len = 2048 bytes
Kernel allocates buf_in_ping[512] = 2048 bytes

Host pass sets core BD len = 4096 bytes (after fix)
Kernel allocates buf_in_ping[512] = 2048 bytes (if kernel pass not fixed)

→ DMA writes 4096 bytes into a 2048-byte buffer → memory corruption
```

All three paths (frontend, host pass, kernel pass) MUST agree on buffer sizes.

---

## 4. Diagnostic Checklist

When debugging K-accumulation SHIM stuck issues, verify these invariants:

### 4.1 Buffer Size Consistency (Critical)

All three must match for each port:

| Source | Parameter | Must Equal |
|--------|-----------|------------|
| Frontend (`aiehlc.cc`) | `pp.bufferSize` | `min(per_k_round_data, max_buffer_bytes)` |
| Host pass (`passblueprinttoschedule.cpp`) | `pingPongBufferSize` | same value |
| Kernel pass (`passblueprinttoschedulekernel.cpp`) | `pingPongBufSize` | same value |

Check in generated code:
```bash
# kernel.cc: BUF_SZ_IN_0 × vector_width × element_bytes = bufferSize
grep "BUF_SZ_IN" kernel.cc

# kernel.cc: window_init size parameter = BUF_SZ_IN
grep "window_init" kernel.cc

# host.cc: core tile BD len = bufferSize × element_bytes
grep "__Runtime_dma_bd_config" host.cc | head -5
```

### 4.2 Round Count Consistency (Critical)

| Source | Parameter | Must Equal |
|--------|-----------|------------|
| Frontend | `pp.numRounds` | `ceil(per_k_round_data / bufferSize)` |
| Host pass | `numIterations` (before kRounds multiply) | same value |
| Kernel pass | `paramInfo.numRounds` (before kRounds multiply) | same value × kRounds |

Check in generated code:
```bash
# kernel.cc: window_init numRounds parameter (last arg)
grep "window_init" kernel.cc
# Should be: numRoundsPerKRound × kRounds

# host.cc: SHIM input startio repeat count
grep "__Runtime_startio" host.cc
# Should be: 1 (for broadcast input) or numIterations×kRounds
```

### 4.3 SHIM Input BD Parameters (K-round iteration)

For broadcast input flows with kRounds > 1:
```
iter_wrap     = kRounds (e.g., 4)
iter_step_size = effectiveK × element_bytes (e.g., 64×1 = 64 bytes)
```

Check in IR:
```bash
grep "iter_wrap\|iter_step_size" ir/dfschedule/5_BlueprintToSchedulePass.mlir
```

### 4.4 SHIM Output BD Parameters (OOO gather)

For many_to_one output flows:
```
iter_wrap      = ooNumIterations (1 if data ≤ max_buffer_bytes)
iter_step_size = 0 (if iter_wrap ≤ 1, since no address advance needed)
repeat_count   = numCoreTiles × ooNumIterations
```

Check in IR:
```bash
grep "repeat_count\|iter_wrap" ir/dfschedule/5_BlueprintToSchedulePass.mlir
```

### 4.5 pp_depth Independence (Invariant)

`pp_depth` must NOT appear in any buffer size or round count calculation. It only affects:
- Number of physical buffer allocations (ping + pong)
- `buffer_mode`: 0 = single buffer (pp_depth=1), 1 = ping-pong (pp_depth>=2)
- `num_buffers`: 1 or 2

Verify `pp_depth` does NOT influence:
- `bufferSize`, `pingPongBufferSize`, `pingPongBufSize`
- `numRounds`, `numIterations`, `ooNumIterations`
- `iter_wrap`, `iter_step_size`, `repeat_count`

### 4.6 Lock Symmetry

For each port, the kernel and host must agree on lock protocol:

```
Input S2MM (host writes to core buffer):
Host BD:   acquire lock_id=N (val=-1), release lock_id=N+1 (val=1)
Kernel:    acquire lock_id=N+1 (data ready), release lock_id=N (buffer free)

Output MM2S (core reads from core buffer):
Kernel:    acquire lock_id=N (buffer free), fill data, release lock_id=N+1 (data ready)
Host BD:   acquire lock_id=N+1 (val=-1), release lock_id=N (val=1)
```

Lock mismatch → deadlock (kernel waits for lock that host never releases).

### 4.7 Total Data Volume

Verify end-to-end data volume:

```
Input A per SHIM:   partition_rows × fullK × elem_bytes = 64 × 256 × 1 = 16384 bytes
= numIterations × pingPongBufSize × elem_bytes
= (1 × kRounds) × 4096 × 1 = 4 × 4096 = 16384 ✓

Input B per SHIM:   partition_cols × fullK × elem_bytes = 64 × 256 × 1 = 16384 bytes
= same calculation ✓

Output C per SHIM:  partition_rows × tile_n × elem_bytes × numCoreTiles
= 64 × 64 × 1 × 4 = 16384 bytes
= numCoreTiles × pingPongBufSize × ooNumIterations × elem_bytes
= 4 × 4096 × 1 × 1 = 16384 ✓
```

---

## 5. The Fix

### 5.1 Files Modified

| File | Location | Change |
|------|----------|--------|
| `src/llvm/aiehlc.cc` | Lines 1234-1310 | Input A/B/default/output: replace `data / ppDepth` with `min(data, maxBuf)`; only split when `data > maxBuf`; add warnings when splitting occurs |
| `src/mlir/.../passblueprinttoschedule.cpp` | Lines 1098-1108 | Core path: remove `effectiveBufferRatio = 1.0/ppDepth`; set `pingPongBufferSize = perCorePerKRound` clamped to maxPingPongBytes |
| `src/mlir/.../passblueprinttoschedule.cpp` | Lines 754-762 | OOO path: remove `shimEffectiveBufferRatio = 1.0/shimPpDepth`; set `ooPingPongSize = ooPerCoreElements` clamped to maxPingPongBytes |
| `src/mlir/.../passblueprinttoschedule.cpp` | Lines 811-814 | Set `iter_step_size = 0` when `iter_wrap ≤ 1` |
| `src/mlir/.../passblueprinttoschedule.cpp` | Lines 1504-1508 | `repeat_count = numCoreTiles × ooNumIterations` (was `numCoreTiles × 2`) |
| `src/mlir/.../passblueprinttoschedulekernel.cpp` | Lines 829-852 | Remove `effectiveBufRatio = 1.0/ppDepth`; set `pingPongBufSize = perCoreSizeForBuf` clamped to maxPingPongBytes |

### 5.2 Before/After Values (256x256, 4x4 mesh, effectiveK=64, kRounds=4, pp_depth=2)

**Input A (per core tile):**

| Parameter | Before (wrong) | After (correct) |
|-----------|----------------|-----------------|
| per_k_round_data | 4096 | 4096 |
| bufferSize | 2048 | 4096 |
| numRounds per k-round | 2 | 1 |
| Total acquires | 8 (2×4) | 4 (1×4) |
| Core BD len | 2048 | 4096 |
| kernel BUF_SZ_IN | 512 | 1024 |

**Output C (per core tile, pp_depth=1):**

| Parameter | Before (wrong) | After (correct) |
|-----------|----------------|-----------------|
| bufferSize | 4096 | 4096 |
| numRounds | 1 | 1 |
| OOO iter_wrap | 1 | 1 |
| OOO iter_step_size | 16384 | 0 |
| SHIM repeat_count | 8 | 4 |

**SHIM Input BD (broadcast A):**

| Parameter | Before | After |
|-----------|--------|-------|
| iter_wrap | 8 (2 rounds × 4 kRounds) | 4 (1 round × 4 kRounds) |
| iter_step_size | 64 | 64 |

### 5.3 Correct Data Flow After Fix

```
SHIM MM2S (Input A, broadcast on row):
DDR partition = 64 × 256 = 16384 bytes
3D addressing: D0=[4,16], D1=[256,64] → reads 64 rows × 64 cols = 4096 bytes per fire
iter_wrap = 4 (kRounds), iter_step_size = 64 (advance to next K chunk)
Total: 4 fires × 4096 bytes = 16384 bytes ✓

Core S2MM (receives input A):
BD len = 4096 bytes, ping-pong = 2 buffers
numIterations = 4 (1 per k-round × 4 k-rounds)
Lock: acquire(free), write 4096, release(ready)

Kernel:
window_init(size=1024, numRounds=4)
for kr in 0..3:
acquire → read 4096 bytes (1024 vectors × 4 bytes) → release
Total: 4 acquires × 4096 bytes = 16384 bytes ✓

Core MM2S (sends output C):
BD len = 4096 bytes, single buffer (pp_depth=1)
numIterations = 1 (output written once after all k-rounds)
Lock: acquire(data ready), send 4096, release(free)

SHIM S2MM (OOO output gather):
4 BDs, each with fixed DDR offset, iter_wrap=1, iter_step_size=0
repeat_count = 4 (one BD completion per core tile)
Total: 4 × 4096 = 16384 bytes ✓
```

---

## 6. Verification

### 6.1 Build and Regenerate

```bash
cd src/mlir/mlirfront/tilinglinalg/pass/unitest/build
make -j4
./test dfschedule
```

### 6.2 Check Generated kernel.cc

```bash
grep "BUF_SZ_IN_0\|BUF_SZ_OUT\|window_init" worklocal/kernel.cc
```

Expected (256x256 case):
```
BUF_SZ_IN_0 = 1024      (4096 bytes / 4 vector width)
BUF_SZ_IN_1 = 1024
BUF_SZ_OUT_0 = 1024
window_init(..., 1024, 4)   (size=1024, numRounds=4 for inputs)
window_init(..., 1024, 1)   (size=1024, numRounds=1 for output)
```

### 6.3 Check Generated host.cc

```bash
grep "__Runtime_dma_bd_config\|__Runtime_startio" worklocal/host.cc | head -20
```

Expected:
- Core tile BD: `len=4096` (not 2048)
- SHIM output startio: `repeat=4` (not 8)

### 6.4 Check Generated IR

```bash
grep "iter_wrap\|iter_step_size\|repeat_count" ir/dfschedule/5_BlueprintToSchedulePass.mlir
```

Expected:
- SHIM input: `iter_wrap = 4`, `iter_step_size = 64`
- SHIM output OOO: `iter_wrap = 1`, `iter_step_size = 0`
- Output startio: `repeat_count = 4`

### 6.5 HW Run

```bash
cd worklocal && source hostcompile.sh   # builds kernel + host
python3 script/test/apppaltest.py build/host
```

Expected:
- Kernel completes all 4 k-rounds
- Output DMA lock releases properly
- `device_teardown done` (no TIMEOUT)

---

## 7. Lessons Learned

1. **pp_depth is a physical resource parameter, not a data-splitting parameter.** It determines how many buffers exist for overlapping DMA with computation (double-buffering). Data splitting is driven solely by whether per-k-round data exceeds `max_buffer_bytes`. Conflating these two concerns caused a subtle but pervasive mismatch.

2. **All three computation paths must agree.** The frontend (`aiehlc.cc`), host pass, and kernel pass independently compute buffer sizes and round counts. If any one path uses a different formula, the generated host.cc and kernel.cc will disagree on DMA transfer sizes, lock protocol, or iteration counts — causing hangs or data corruption.

3. **SHIM repeat_count must equal total BD completions, not a hardcoded multiple.** For OOO output gather: `repeat_count = numCoreTiles × ooNumIterations`. Hardcoding `× 2` (assuming pp_depth always splits) caused the SHIM to wait for transactions that never arrive.

4. **iter_step_size must be 0 when iter_wrap ≤ 1.** A non-zero step size with no iteration is harmless in hardware but misleading during debugging. Always set `iter_step_size = 0` when iteration is disabled.

5. **K-accumulation multiplies the DMA round count, not the buffer size.** With `kRounds=4`, each core acquires `numRoundsPerKRound × kRounds` times, but each acquire is for `bufferSize` bytes (one k-round's worth of data). The SHIM BD uses `iter_wrap = kRounds` to re-fire with DDR address advancement between k-rounds.

6. **C-stationary output is independent of K-rounds.** The output is produced once after all k-rounds complete (the kernel accumulates locally). Output DMA parameters (bufferSize, numRounds, repeat_count) should not incorporate kRounds.

7. **Verify with the invariant: total bytes = partition size.** For each port, `numRounds × bufferSize × elementBytes = partition_rows × fullK × elementBytes` (input) or `partition_rows × tile_n × elementBytes` (output). If this doesn't hold, there's a splitting/counting bug.

8. **Output pp_depth=1 is common for C-stationary.** Since the kernel writes output only once (after accumulation), there's no need for ping-pong overlap on the output port. `pp_depth=1` means single buffer, `buffer_mode=0`, and `num_buffers=1`. The output path should never split by pp_depth.
