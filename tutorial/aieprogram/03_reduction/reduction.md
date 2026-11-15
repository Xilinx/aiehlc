# 3. Reduction: row sums on the 4x4 mesh, in two levels

> **Status (2026-11-15).** As for [elementwise](../02_elementwise/elementwise.md):
> the kernel passes the kernel analysis; the lowering does not support this
> shape yet. A runnable example follows the compiler work.

## The whole program

The whole program -- kernel, host, and the host-side combine of the partial sums. It passes the kernel analysis; it does not build yet (status note above).

<!-- code: example/tileprogram/ccode/simplereduce_rowsum.cc -->
```cpp
#include "simpleops_load.h"
// STATUS (2026-11-15): passes the kernel analysis (`build/aiehlc-convtile trace` this file);
// the lowering does not support this op shape yet, so aiehlc.sh stops with
// "the core window is ... but the kernel's blocks (#routing.kernel_block) are ...".
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
// Reduction: s[r] = sum over c of x[r][c], x int8[256, 1024], on the 4x4 mesh by
// redundant broadcast, in two levels:
//   in the array: mesh row pm owns rows 64 pm..; each 16 x 256 block of them is
//     broadcast to the row's 4 cores, and core (pm, pn) sums column chunk pn
//     (columns 256 pn ..) of every row: an int32 PARTIAL sum per row and core;
//   on the host: s[r] = the 4 partial sums of row r.
// y int8[256, 16]: row r's partial from mesh column pn is the int32 at y[r][4 pn..].
// Walkthrough: tutorial/aieprogram/03_reduction/reduction.md
__global__ void rowsum(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                       aie::port<output_window_int8 *, aie::shape<OPS_R, 16>> y) {
    const int tr = 16, tc = 256;                    // input block: 16 rows x 256 columns
    const int rows = OPS_R / aie::num_programs(0); // 64 rows per mesh row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) {           // 4 row strips
        int8_t *o = aie::store_buffer(y, {pm * rows + i * tr, pn * 4}, {tr, 4});
        int acc[tr];
        for (int r = 0; r < tr; r++)
            acc[r] = 0;
        for (int j = 0; j < OPS_C / tc; j++) {      // 4 column chunks, all received
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            if (j == pn)                            // ... this core sums its own
                for (int r = 0; r < tr; r++)
                    for (int c = 0; c < tc; c++)
                        acc[r] += a[r * tc + c];
            aie::release(x);
        }
        for (int r = 0; r < tr; r++) // int32 little-endian
            for (int b = 0; b < 4; b++)
                o[r * 4 + b] = (int8_t)(acc[r] >> (8 * b));
        aie::commit(y);
    }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4);
    const long n = (long)OPS_R * OPS_C;
    int8_t *x = (int8_t *)__Runtime_Alloc(n), *y = (int8_t *)__Runtime_Alloc(OPS_R * 16);
    int8_t *ref = (int8_t *)malloc(OPS_R * 16);
    ops_fill(x, n, 33);
    memset(y, 0, OPS_R * 16);
    XTime t0, t1;
    XTime_GetTime(&t0);
    rowsum<<<mesh>>>(x, y);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    long bad_sum = 0;
    for (int r = 0; r < OPS_R; r++) {
        int total = 0, got = 0;
        for (int q = 0; q < 4; q++) {
            int part = 0;
            for (int c = 0; c < OPS_C / 4; c++)
                part += x[(long)r * OPS_C + q * (OPS_C / 4) + c];
            for (int b = 0; b < 4; b++)
                ref[r * 16 + q * 4 + b] = (int8_t)(part >> (8 * b));
            total += part;
            int32_t v; // the level-2 combine: add the 4 partial sums of the row
            memcpy(&v, &y[r * 16 + q * 4], 4);
            got += v;
        }
        bad_sum += got != total;
    }
    printf("row sums after the host combine: %ld wrong out of %d\n", bad_sum, OPS_R);
    return ops_compare("y", y, ref, OPS_R, 16) || bad_sum ? 1 : 0;
}
```
<!-- /code -->

## 1. The idea: partial sums in the array, the combine somewhere else

A reduction reads a lot and writes a little: row sums of `x[256, 1024]` read
256 KB and write 256 values, one addition per byte. What is new compared with
an elementwise op is the **combine**: as soon as two cores each sum part of the
same row, someone has to add their partial results.

| Where the partial sums are combined | How | In aiehlc today |
|---|---|---|
| nowhere: each core owns whole rows | split rows across cores | fine on a mesh with one core per row band (4x1); on 4x4, splitting a row band's rows over the 4 cores puts both pids on the output's row dimension, which the router rejects (section 4) |
| on the host, after the launch | each core writes its partial sum, the host adds them | **this chapter** -- a few additions per row on the Arm |
| a second kernel launch | partial sums to DDR, another kernel adds them | works, but every launch re-loads and re-configures the array |
| in the array | neighbouring cores pass accumulators over the **cascade** interface (a direct core-to-core link), or through shared neighbour memory | the AIE way; not expressible in aiehlc yet |

On 4x4 we combine the two techniques of chapter 2: mesh row `pm` owns 64 rows
and receives all of their columns by **broadcast**; core `(pm, pn)` sums only
**column chunk `pn`** of every row (a quarter of each row) and writes one int32
**partial sum** per row. The host adds the 4 partial sums of each row.

## 2. The kernel, line by line

```cpp
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
__global__ void rowsum(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                       aie::port<output_window_int8 *, aie::shape<OPS_R, 16>> y) {
```
Two ports -- a kernel has as many as it needs. `y` holds 4 int32 partial sums
per row = 16 bytes (ports carry int8 windows; an int32 is 4 of them): the
partial of row `r` from mesh column `pn` is at `y[r][4 pn .. 4 pn + 3]`.

```cpp
    const int tr = 16, tc = 256;                    // input block: 16 rows x 256 columns
    const int rows = OPS_R / aie::num_programs(0); // 64 rows per mesh row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) {           // 4 row strips
        int8_t *o = aie::store_buffer(y, {pm * rows + i * tr, pn * 4}, {tr, 4});
```
One output block per strip: 16 rows x this core's 4-byte slot. Rows follow
`pm` (dim 0), the slot follows `pn` (dim 1) -- one mesh axis per dimension.
The output block is taken **before** the column loop and committed **after**
it: one output block per 4 input blocks -- ports may have different round
counts.

```cpp
        int acc[tr];
        for (int r = 0; r < tr; r++)
            acc[r] = 0;
        for (int j = 0; j < OPS_C / tc; j++) {      // 4 column chunks, all received
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            if (j == pn)                            // ... this core sums its own
                for (int r = 0; r < tr; r++)
                    for (int c = 0; c < tc; c++)
                        acc[r] += a[r * tc + c];
            aie::release(x);
        }
```
The redundant broadcast of chapter 2: every core of the mesh row receives the 4
chunks of the strip; core `pn` accumulates chunk `pn` in `acc`, which stays in
core memory across the chunks (like the GEMM's K loop in 1a).

```cpp
        for (int r = 0; r < tr; r++) // int32 little-endian
            for (int b = 0; b < 4; b++)
                o[r * 4 + b] = (int8_t)(acc[r] >> (8 * b));
        aie::commit(y);
    }
}
```
Write the 16 partial sums and send the block.

On the host, after the launch -- the second level:

```cpp
        for (int q = 0; q < 4; q++) {
            int32_t v; // the level-2 combine: add the 4 partial sums of the row
            memcpy(&v, &y[r * 16 + q * 4], 4);
            got += v;
        }
```
(`memcpy`, not a cast: `y` is a byte buffer, and an unaligned int32 load can
fault on the A78.)

Kernel analysis:

```
  x (IFM, row broadcast, 4x4 blocks/core): d1[256]=rows/4@64+i4x16@16 d2[1024]=j4x256@256
  y (OFM, gather, 1x4 blocks/core): d1[256]=rows/4@64+j4x16@16 d2[16]=cols/4@4
  => routed: 2 ports on the 4x4 mesh, DMA carriers taken from the traced blocks
```

## 3. Why the partial sums go by column chunk

Each core must reduce a part of the data the mesh row already receives (the
broadcast), and its output must be split over the mesh columns on **its own
dimension**. Column chunk `pn` of every row satisfies both: the input stays a
row broadcast, and the partial-sum slot is a separate output dimension that
`pn` can own.

## 4. What does not work, and why

Giving each core 4 whole rows of every strip (no combine at all) looks
simpler, but the output offset becomes `64 pm + 16 i + 4 pn` -- both pids on
the **same** dimension:

```
ERROR: rowsum: y (OFM): dim 0 moves with both program_id(0) and program_id(1)
(offsets (64,0) / (4,0)); a mesh axis can split one tensor dim only
```

A DMA walk splits one tensor dimension per mesh axis. When you need both, give
the second pid its own dimension -- here, the partial-sum slot.

## 5. Making it fast (once it runs)

- **Vectorize the inner loop**: accumulate whole vectors (`aie::accum`) and
  reduce across lanes once per row at the end, not per element.
- **Fuse**: a reduction over data another kernel just produced (softmax's row
  max and sum, layer norm's mean and variance) is cheapest inside that kernel,
  while the data is still in core memory.
- **Combine in the array** once the compiler can express cascade streams; then
  the partial sums never leave the cores.

Next: [4. Data movement](../04_data_movement/data_movement.md).
