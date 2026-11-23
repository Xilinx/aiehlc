# 6.1 GEMM: the low-level schedule (`{.depth}`, `{.resident}`)

Examples, same problem, same scalar compute:

- step 0 [`simplegemm_opt0_baseline.cc`](../../../example/tileprogram/ccode/simplegemm_opt0_baseline.cc)
  -- no options: the compiler chooses every buffer and transfer;
- step 1 [`simplegemm_opt1_schedule.cc`](../../../example/tileprogram/ccode/simplegemm_opt1_schedule.cc)
  -- the same kernel with three schedule options (listing below).

```
C[2048, 256] = A[2048, 256] x B[256, 256]^T     int8, 4x4 mesh, 134 M multiply-accumulates per launch
```

## The whole program (step 1)

The kernel of [1a](../01_contraction/gemm.md) grown to a larger GEMM: each core
keeps its 64 rows of B and walks 512 rows of A in strips of 16, full K. The
three `{...}` arguments are the only difference from step 0.

<!-- code: example/tileprogram/ccode/simplegemm_opt1_schedule.cc -->
```cpp
/******************************************************************************
 *
 * GEMM optimization, step 1: the low-level schedule (tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md).
 *   C[2048, 256] = A[2048, 256] x B[256, 256]^T, int8, on the 4x4 mesh.
 * Step 0 (simplegemm_opt0_baseline.cc) plus three schedule options -- same
 * compute, same result, different buffers and DMA:
 *   B {.resident = true}  one 16 KB buffer, sent once, held all launch
 *   A {.depth = 2}        ping-pong strips: the next arrives during compute
 *   C {.depth = 2}        ping-pong output: the last drains during compute
 * Step 2 (simplegemm_dataflow.cc) adds the vector unit.
 ******************************************************************************/
#include "simplegemm_dataflow.h"

#define GD_M 2048
#define GD_N 256
#define GD_K 256
#define GD_TS 16
#ifndef GD_ITERS
#define GD_ITERS 3
#endif

__global__ void gemm_s1(aie::port<input_window_int8 *, aie::IFM, aie::shape<GD_M, GD_K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<GD_N, GD_K>> b,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<GD_M, GD_N>> c) {
    const int cm = GD_M / 4, cn = GD_N / 4; // per core: 512 rows of C, 64 columns
    const int pm = aie::program_id(0);      // mesh row: which rows of C
    const int pn = aie::program_id(1);      // mesh col: which 64 columns of C
    const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true}); // this column's 64 rows of B
    for (int s = 0; s < cm / GD_TS; s++) {                             // 32 strips of 16 rows
        const int r0 = pm * cm + s * GD_TS;
        const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2});
        int8_t *y = aie::store_buffer(c, {r0, pn * cn}, {GD_TS, cn}, {.depth = 2});
        for (int m = 0; m < GD_TS; m++)
            for (int n = 0; n < cn; n++) {
                int acc = 0;
                for (int k = 0; k < GD_K; k++)
                    acc += x[m * GD_K + k] * w[n * GD_K + k];
                y[m * cn + n] = (int8_t)acc; // low byte, like gemm_df_verify
            }
        aie::release(a);
        aie::commit(c);
    }
    aie::release(b);
}

static void gemm_s1_run(int8_t *A, int8_t *B, int8_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, GD_HW_ROWS, GD_HW_COLS);
    gemm_s1<<<mesh>>>(A, B, C);
}

int main() {
    int8_t *A = (int8_t *)__Runtime_Alloc(GD_M * GD_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GD_N * GD_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GD_M * GD_N);
    gemm_df_init(A, B, GD_M, GD_N, GD_K);
    memset(C, 0, GD_M * GD_N);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_s1_run(A, B, C); // first call: partition init + kernel load
    XTime_GetTime(&t1);
    const double first_ms = gd_ms(t0, t1);
    double sum = 0;
    for (int it = 0; it < GD_ITERS; it++) {
        XTime_GetTime(&t0);
        gemm_s1_run(A, B, C);
        XTime_GetTime(&t1);
        sum += gd_ms(t0, t1);
    }
    const double mean = sum / GD_ITERS;
    printf("PERF SUMMARY gemm_opt1 first=%.3f mean=%.3f gops=%.2f (ms)\n", first_ms, mean,
           2.0 * GD_M * GD_N * GD_K / (mean * 1e-3) / 1e9);
    const long bad = gemm_df_verify(A, B, C, GD_M, GD_N, GD_K);
    free(A);
    free(B);
    free(C);
    return bad ? 1 : 0;
}
```
<!-- /code -->

## 1. The tiling

Core `(pm, pn)` computes C rows `512 pm .. +511` and columns `64 pn .. +63`:

| Port | Block | Blocks per core | Moves with | Data movement |
|---|---|---|---|---|
| B | `{64, 256}` = 16 KB | 1 | `pn` | column broadcast, sent once |
| A | `{16, 256}` = 4 KB | 32 strips | `pm` | row broadcast |
| C | `{16, 64}` = 1 KB | 32 | `pm`, `pn` | gather |

Compared with 1a's 64 x 64 x 256 GEMM this one keeps the B block on the core
for the whole launch and streams A in full-K strips -- far less traffic per
MAC (B is never re-sent).

## 2. Step 0: what the compiler chooses by itself

A core has 32 KB for port buffers (less 3 KB of the compiler's own data). The
default is to double-buffer every port, which here would need
2 x 16 + 2 x 4 + 2 x 1 = 42 KB -- it does not fit. So the compiler single-buffers
the inputs (`kernel.cc` of step 0):

```cpp
  window_init(window_window_in_0, 1, buf_in_ping_0, ..., buf_in_ping_0, ..., 1024, 32);  // A: one buffer, 32 rounds
  window_init(window_window_in_1, 1, buf_in_ping_1, ..., buf_in_ping_1, ..., 4096, 1);   // B: one buffer, 1 block
  window_init(window_window_out_0, 1, buf_out_ping_0, ..., buf_out_pong_0, ..., 256, 32); // C: ping-pong
```

The **output always stays ping-pong**: the core writes the next block while
the last one drains. (Before 2026-11-15 the compiler also single-buffered the
output on the host side while the core still wrote ping / pong; the second
block was never sent and step 0 hung in `wait_io`. The output depth is now
fixed at 2, and `{.depth = 1}` on a `store_buffer` is a build error.)

```
PERF SUMMARY gemm_opt0 first=91.038 mean=90.704 gops=2.96 (ms)
GOLDEN PASS: all 524288 output elements match.
```

## 3. Step 1: choosing the schedule

```cpp
    const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true});
    ...
        const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2});
        int8_t *y = aie::store_buffer(c, {r0, pn * cn}, {GD_TS, cn}, {.depth = 2});
```

- `B {.resident = true}`: one block per core, one buffer, sent once. That is
  what the compiler already did in step 0 -- now it is stated and checked
  (load B twice and the build fails).
- `A {.depth = 2}`: ping-pong, so strip `s + 1` can arrive while strip `s` is
  computed. **This is the one real change**: `kernel.cc` gains `buf_in_pong_0`
  (4 KB per core) and `host.cc` 16 more buffer descriptors (88 -> 104).
- `C {.depth = 2}`: already the case.

The schedule fits: 16 + 2 x 4 + 2 x 1 = 26 KB.

## 4. Measured -- and why step 1 is slower

```
step 0   PERF SUMMARY gemm_opt0 first=91.038 mean=90.704 gops=2.96 (ms)    GOLDEN PASS
step 1   PERF SUMMARY gemm_opt1 first=101.485 mean=101.053 gops=2.66 (ms)  GOLDEN PASS
```

Double-buffering A made the launch **10 ms slower**. Do the arithmetic before
you apply a schedule option:

- **What ping-pong can save** is DMA time hidden behind compute. One A strip
  is 4 KB; at ~2.3 GB/s per shim channel that is ~2 us, 32 strips ~57 us per
  launch -- the most overlap could ever win.
- **What the scalar compute costs** is 16 x 64 x 256 = 262,144 MACs per strip
  per core, milliseconds per launch: the core is never waiting for A, so
  there was nothing to hide.
- **What ping-pong costs** is one more 4 KB buffer per core and its buffer
  descriptors. Every launch writes the kernel's data image (the port buffers
  included) and all descriptors over the configuration bus at ~0.4 us per
  32-bit word (`doc/performance/register_write_cost.md`): 4 KB x 16 cores is
  16,384 words, ~6 ms on its own.

So in a **compute-bound** kernel, double-buffering an input only adds
configuration. It pays off once the compute is fast enough that the core
would wait for the DMA -- which is what 6.2 does to this kernel.

## 5. Rules of thumb

| Option | Use it when | Cost |
|---|---|---|
| `{.resident = true}` | the core needs one block for the whole launch (weights, B) | none -- it saves a buffer and re-sending |
| `{.depth = 2}` on an input | per-round DMA time is comparable to per-round compute | one more buffer + descriptors per core, configured on every launch |
| `{.depth = 1}` on an input | buffers do not fit and the input is cheap to wait for | the core waits for every block |
| `{.tile = {tr, tc}}` | the compute wants tiles (`aie::mmul`) -- 6.2 | a 3-D shim descriptor instead of 2-D |

And: measure. The launch here is dominated by configuring the array, not by
the MACs; that changes which option helps.

Next: [6.2 GEMM: vector instructions](2_vector_instructions.md).
