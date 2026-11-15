# 3. Reduction: row sums

> **Status (2026-11-15).** As for [elementwise](../02_elementwise/elementwise.md):
> the kernel passes the kernel analysis; the lowering does not support this
> shape yet (`the core window is 64 elements x 1 round(s) ..., but the
> kernel's blocks ... are 4096 elements x 16`). A runnable example follows the
> compiler work.

## 1. The idea: reduce along a dimension each core owns

A reduction reads a lot and writes a little: row sums of `x[256, 1024]` read
256 KB and write 256 values. Like elementwise ops it is **bandwidth-bound**
(one addition per byte read), and the same rule applies: use the input
streams, not every core.

What is new is the **combine**. Where the partial results meet decides how
hard it is:

| Reduction | Who combines | Today |
|---|---|---|
| along a dimension each core owns completely (row sums, each core its own rows) | nobody: each core finishes its own rows | kernel analysis OK; lowering pending (above) |
| across cores (one sum of the whole tensor) | partial sums from several cores | two launches: partial sums to DDR, then a second kernel or the host combines them -- every launch re-loads and re-configures the array, which costs far more than the data |
| across cores, in the array | neighbouring cores pass accumulators over the **cascade** interface (a direct core-to-core accumulator link), or through shared neighbour memory | the AIE way; not expressible in aiehlc yet |

So: **pick the reduced dimension so that no core needs another core's
partial result.** For row sums, split rows across cores and walk the columns
inside the core.

## 2. The kernel (4x1 mesh)

```cpp
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
// s[r] = sum over c of x[r][c] (int32); the int32 of row r is y[r][0..3]
__global__ void rowsum(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                       aie::port<output_window_int8 *, aie::shape<OPS_R, 4>> y) {
    const int tr = 16, tc = 256;                    // input block: 16 rows x 256 columns
    const int rows = OPS_R / aie::num_programs(0); // 64 rows per core
    const int pm = aie::program_id(0);
    for (int i = 0; i < rows / tr; i++) {           // 4 row strips
        int8_t *o = aie::store_buffer(y, {pm * rows + i * tr, 0}, {tr, 4});
        int acc[tr];
        for (int r = 0; r < tr; r++)
            acc[r] = 0;
        for (int j = 0; j < OPS_C / tc; j++) {      // 4 column chunks: the reduced dimension
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            for (int r = 0; r < tr; r++)
                for (int c = 0; c < tc; c++)
                    acc[r] += a[r * tc + c];
            aie::release(x);
        }
        for (int r = 0; r < tr; r++) // int32 little-endian into the row's 4 bytes
            for (int b = 0; b < 4; b++)
                o[r * 4 + b] = (int8_t)(acc[r] >> (8 * b));
        aie::commit(y);
    }
}
```

- Two ports: a kernel may have as many as it needs.
- The output block `{16, 4}` is taken **before** the column loop and committed
  **after** it: one output block per 4 input blocks. Different ports may have
  different round counts.
- The accumulator `acc` stays in core memory across the column chunks -- the
  same pattern as the GEMM's K loop (1a).
- The int32 result is written as 4 bytes because ports carry int8 windows.

Kernel analysis:

```
  x (IFM, row broadcast, 4x4 blocks/core): d1[256]=rows/4@64+i4x16@16 d2[1024]=j4x256@256
  y (OFM, gather, 1x4 blocks/core): d1[256]=rows/4@64+j4x16@16 d2[4]=whole
  => routed: 2 ports on the 4x1 mesh, DMA carriers taken from the traced blocks
```

## 3. Making it fast (once it runs)

- **Vectorize the inner loop**: accumulate whole vectors (`aie::accum`) and
  reduce across lanes once per row at the end, not per element.
- **Input streams**: x is the only input, so a 4x1 mesh uses 4 of the 8 MM2S
  channels. Using all 8 would mean two cores per row band, each reading half
  the columns: a per-core input block, which the compiler cannot route yet
  ([2, section 4](../02_elementwise/elementwise.md#4-why-not-the-4x4-mesh)),
  plus a combine of the two halves (section 1).
- **Fuse**: a reduction over data another kernel just produced (softmax's row
  max and sum, layer norm's mean and variance) is cheapest inside that kernel,
  while the data is still in core memory.

Next: [4. Data movement](../04_data_movement/data_movement.md).
