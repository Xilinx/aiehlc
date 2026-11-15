# 4. Data movement: transpose

> **Status (2026-11-15).** As for [elementwise](../02_elementwise/elementwise.md):
> the kernel passes the kernel analysis; the lowering does not support this
> shape yet. A runnable example follows the compiler work.

## 1. The idea: who re-arranges the bytes -- the DMA or the core?

A transpose, a layout change, a copy: no arithmetic at all, only bytes in a new
order. On the AIE there are two places to re-order them:

- **the DMA**: a buffer descriptor walks memory with up to 3 nested
  (stride, count) loops, so it can gather a strided block (a column band, a
  sub-matrix) or scatter one, in **32-bit words**;
- **the core**: anything, at the cost of instructions.

An int8 transpose needs both: the DMA moves whole rows and column bands, but
it cannot swap single bytes (it moves words), so the core transposes inside
its buffer.

## 2. The kernel (4x4 mesh)

```cpp
#define OPS_T 256 // y = x^T, int8[256, 256]
__global__ void transpose(aie::port<input_window_int8 *, aie::shape<OPS_T, OPS_T>> x,
                          aie::port<output_window_int8 *, aie::shape<OPS_T, OPS_T>> y) {
    const int tr = 16;                             // strip: 16 rows of x = 16 columns of y
    const int rows = OPS_T / aie::num_programs(0); // 64 x rows per mesh row
    const int qc = OPS_T / aie::num_programs(1);   // 64 x columns per core
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) {
        const int r0 = pm * rows + i * tr;
        const int8_t *a = aie::load_buffer(x, {r0, 0}, {tr, OPS_T});     // a[r][c] = x[r0 + r][c]
        int8_t *o = aie::store_buffer(y, {pn * qc, r0}, {qc, tr});       // o[c][r] = y[pn*qc + c][r0 + r]
        for (int r = 0; r < tr; r++)
            for (int c = 0; c < qc; c++)
                o[c * tr + r] = a[r * OPS_T + pn * qc + c];
        aie::release(x);
        aie::commit(y);
    }
}
```

- **Input: a row strip `{16, 256}`, broadcast along the mesh row** -- the
  redundant broadcast of [chapter 2](../02_elementwise/elementwise.md): the 4
  cores of mesh row `pm` receive the same strips, contiguous in DDR.
- **Output: a `{64, 16}` block of `y`** -- 64 rows of 16 bytes, 256 bytes
  apart. Core `pn` takes columns `64 pn .. +63` of the strip, which become rows
  `64 pn ..` of `y`; the strip's 16 rows become 16 columns starting at `r0`.
  The DMA scatters the block with the 256-byte row pitch, so `store_buffer`
  simply names it in `y`'s coordinates: rows follow `pn`, columns follow `pm`.
- The core does the byte transpose between its two buffers.

Kernel analysis -- the output is split on **both** dimensions, one mesh axis
each:

```
  x (IFM, row broadcast, 1x4 blocks/core): d1[256]=rows/4@64+j4x16@16 d2[256]=whole
  y (OFM, gather, 1x4 blocks/core): d1[256]=cols/4@64 d2[256]=rows/4@64+j4x16@16
  => routed: 2 ports on the 4x4 mesh, DMA carriers taken from the traced blocks
```

## 3. Choosing the blocks

- **Make the DMA's job word-sized.** A block's inner dimension (and the
  tensor's row) must be a multiple of 4 bytes; 16-byte column bands are.
- **Prefer the layout the consumer wants.** Often the best transpose is none:
  produce the data in the layout the next kernel reads (chapter 6.2 has the
  DMA deliver A tile-by-tile for `aie::mmul` with `{.tile}`, and packs B on the
  host once, instead of transposing on every launch).
- Like elementwise ops, data movement is bandwidth-bound: the bytes, not the
  cores, set the pace.

Next: [5. Indexing](../05_indexing/indexing.md).
