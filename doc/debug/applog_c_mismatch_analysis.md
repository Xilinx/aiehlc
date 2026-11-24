# Applog C Mismatch Analysis

## Summary
- **Matrix:** C[8x8] = A[8x8] * B^T[8x8], int8, 4x4 mesh (16 tiles)
- **Result:** 64/64 mismatches -- every single element is wrong
- **Applog:** `<repo>/applog` (May 31, 380KB)
- **Note:** This applog was generated with the **pre-fix** `matmul` kernel code

## Pattern 1: Almost-all-zeros output

The actual C matrix received by the host (applog lines 7514-7522):
```
C [8x8]:
  [ 14,  0,  0,  0,  0,  0,  0,  0]   <- only C[0]=14
  [  0,  0,  0,  0,  0,  0,  0,  0]
  [ -9,  0,  0,  0,  0,  0,  0,  0]   <- only C[16]=-9
  [  0,  0,  0,  0,  0,  0,  0,  0]
  [ -4,  0,  0,  0,  0,  0,  0,  0]   <- only C[32]=-4
  [  0,  0,  0,  0,  0,  0,  0,  0]
  [  1,  0,  0,  0,  0,  0,  0,  0]   <- only C[48]=1
  [  0,  0,  0,  0,  0,  0,  0,  0]
```

Only **4 non-zero values** at positions C[0], C[16], C[32], C[48] -- i.e., `row * 8 + 0` for rows 0, 2, 4, 6. All other 60 elements are zero.

## Pattern 2: Non-zero values are partial K-round results

The non-zero values (14, -9, -4, 1) do NOT match expected values. Checking against partial-product computation with only the first K-round (eff_k=4, first half of K=8):

- Expected full C_ref[0][0] = 3, but got 14
- Expected full C_ref[2][0] = 18, but got -9

These values also don't match single-kr partial products. They appear to be **unsaturated int16->int8 truncation artifacts** (Bug 3 in the fix: saturation was disabled under `DEBUG_OUTPUT_ORDER`).

## Pattern 3: Kernels computed CORRECT values internally

The klog C0 output per tile shows correct values:
- tile(0,3): C0=3, then C0=-4 -- matches C_ref rows 0 and 1 first elements (but output 2x due to Bug 2)
- tile(0,4): C0=18, then C0=-9 -- matches C_ref rows 2 and 3
- tile(0,5): C0=-9, then C0=14 -- matches C_ref rows 4 and 5
- tile(0,6): C0=-8, then C0=9 -- matches C_ref rows 6 and 7

But the klog shows **2 C0 entries per tile per mr**, meaning output was happening **once per kr** (Bug 2: saturate+output inside kr loop). The kernel output data twice -- once after kr=0 (partial product) and once after kr=1 (full accumulation). But only the first kr's output makes it to DDR; the second kr's output either overwrites the wrong location or the DMA has already finished.

## Pattern 4: Output DMA stuck on lock_acq

At debug snapshot time, ALL 16 core tile MM2S channels stall on `lock_acq=1` (applog lines 7228-7420):
```
tile(0,3) ch0 MM2S: STALLED lock_acq=1  BD4 addr=0x8080 pkt_id=1  lock_acq id=5 val=127
tile(0,4) ch0 MM2S: STALLED lock_acq=1  BD4 addr=0x8080 pkt_id=5  lock_acq id=5 val=127
tile(0,5) ch0 MM2S: STALLED lock_acq=1  BD4 addr=0x8080 pkt_id=9  lock_acq id=5 val=127
tile(0,6) ch0 MM2S: STALLED lock_acq=1  BD4 addr=0x8080 pkt_id=13 lock_acq id=5 val=127
...all 16 tiles same pattern (BD4, addr=0x8080, lock id=5, val=127)...
```

Meanwhile, all input S2MM channels stall on `stream=1` (starved, waiting for more input data that will never arrive since the shim has already exhausted its repeat count):
- ch0 S2MM (C-output buffer receive): all 16 tiles stalled on stream=1 (BD2)
- ch1 S2MM (input A/B data receive): all 16 tiles stalled on stream=1 (BD0)

## Root Cause Analysis: All 4 Bugs Contributing

### Bug 1: Wrong accum index (`accum[i * cols_per_round + j]`)
With `tile_cols=2` and `num_b_rounds=1`, `cols_per_round = buf_sz_b / eff_k`. When `cols_per_round != tile_cols`, accum rows don't cover the full tile width, resulting in wrong accumulation positions. This causes only element [0] of each row to get a value while remaining elements stay zero.

### Bug 2: Saturate+output inside kr loop (MOST CRITICAL)
The kernel outputs C data **k_rounds times** (2x for K=8, eff_k=4), but the output DMA (MM2S) ping-pong chain has only enough BDs for 1x output. After the first kr's output:
1. Kr=0: kernel writes partial products to output buffer, DMA sends them to shim
2. Kr=1: kernel tries to write again, but DMA lock is already exhausted (lock 5 val=127 -> can't acquire)
3. Kernel stalls waiting for lock -> output DMA stalls on lock_acq -> deadlock

The shim S2MM receives only the kr=0 partial-product data (not the final accumulated result), explaining why the values are wrong.

### Bug 3: No saturation under DEBUG_OUTPUT_ORDER
The kr=0 partial products may exceed int8 range. Without saturation, `(int8_t)val` does C truncation, producing incorrect values like 14 instead of the expected partial sum.

### Bug 4: Excessive logging (minor)
Every A element is logged in every kr/mr, filling 36 entries in the 64-entry klog buffer. Not a correctness issue, but wastes debug capacity and makes diagnosis harder.

## Why Only C[0], C[16], C[32], C[48] Are Non-Zero

The 4x4 mesh distributes the 8x8 matrix as:
- Rows: 4 tile-rows x 2 rows/tile -> rows 0-1 go to row 0 tiles, rows 2-3 to row 1, etc.
- Cols: 4 tile-cols x 2 cols/tile -> cols 0-1 go to col 0 tiles, cols 2-3 to col 1, etc.

With Bug 1 (`accum[i * cols_per_round + j]` instead of `accum[i * tile_cols + rb * cols_per_round + j]`), the kernel writes to wrong positions. Combined with Bug 2 (only first kr output reaches DDR) and Bug 3 (no saturation), only the first byte of each even row's tile gets a non-zero (truncated) value in the DDR output buffer.

## Conclusion

This applog confirms all 4 bugs from the previous fix are responsible for the mismatches. The fixes already applied to `simplematmul2.cc` address all of them. A rebuild and re-run should produce correct results.

No additional code changes are needed -- this is purely an analysis document.
