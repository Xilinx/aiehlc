# 5. Indexing: slices, strides and why runtime gathers do not fit

> **Status (2026-11-15).** The slicing kernel passes the kernel analysis; the
> lowering does not support this shape yet (as in
> [elementwise](../02_elementwise/elementwise.md)). The limits in sections 2
> and 4 are checked by the compiler today.

## 1. The rule: the schedule is fixed at compile time

Every `load_buffer` offset in this model must be an **affine** function of
`program_id` and the loop counters, and must not depend on data: the compiler
runs the kernel at build time, fits each port's blocks to
`base * pid + step_i * i + step_j * j`, and burns that walk into DMA buffer
descriptors. So:

| Indexing | Example | Expressible |
|---|---|---|
| offset known at compile time, affine in pid / loops | crop, tile, every k-th row block | yes |
| strided element access | `x[::2, ::2]` | yes, by loading the dense footprint and picking in the core |
| offset from data (runtime indices) | `gather(x, idx)`, embedding lookup | **no** -- the DMA order would depend on data |

## 2. Crop: the host passes a view

A port's blocks are tiled from **the tensor's origin**. A crop that starts
elsewhere is rejected:

```
ERROR: slice2: x: core (0,0) starts at (32,64), not at the tensor origin (the DMA tiles a tensor from its origin)
```

The fix is on the host: pass a pointer to the crop's first element and declare
the **view** -- its rows, with the full row pitch of the original tensor.

```cpp
#define OPS_R 256  // x rows
#define OPS_C 1024 // x columns (bytes)
#define SL_R0 32   // y = x[SL_R0 : SL_R0 + 2*64 : 2, SL_C0 : SL_C0 + 2*256 : 2], int8[64, 256]
#define SL_C0 64
__global__ void slice2(aie::port<input_window_int8 *, aie::shape<128, OPS_C>> xv, // the cropped view
                       aie::port<output_window_int8 *, aie::shape<64, 256>> y) {
    const int yr = 16, yc = 128;                // y block per round
    const int pm = aie::program_id(0);
    for (int j = 0; j < 256 / yc; j++) {        // 2 column halves
        const int8_t *a = aie::load_buffer(xv, {2 * yr * pm, 2 * yc * j}, {2 * yr, 2 * yc});
        int8_t *o = aie::store_buffer(y, {yr * pm, yc * j}, {yr, yc});
        for (int r = 0; r < yr; r++)
            for (int c = 0; c < yc; c++)
                o[r * yc + c] = a[(2 * r) * (2 * yc) + 2 * c];
        aie::release(xv);
        aie::commit(y);
    }
}
// host: slice2<<<mesh>>>(x + SL_R0 * OPS_C + SL_C0, y);   // the view starts at the crop origin
```

- `xv` is declared `aie::shape<128, OPS_C>`: 128 rows of the original 1024-byte
  pitch, starting at row 32, column 64. Its blocks start at the view's origin.
- The launch passes `x + SL_R0 * OPS_C + SL_C0`.

## 3. Stride: load the footprint, pick in the core

A load is a dense block. For a stride-2 subsample the core loads the
**footprint** of its 16 x 128 output block -- 32 x 256 of the input -- and
keeps every other element. The DMA moves 4x the bytes the output needs; for a
large stride, prefer a layout where the needed elements are adjacent, or do
the subsample in the kernel that produces `x`.

Kernel analysis:

```
  xv (IFM, row broadcast, 1x2 blocks/core): d1[128]=rows/4@32 d2[1024]=j2x256@256
  y (OFM, gather, 1x2 blocks/core): d1[64]=rows/4@16 d2[256]=j2x128@128
  => routed: 2 ports on the 4x1 mesh, DMA carriers taken from the traced blocks
```

## 4. Runtime indices: not in this model

```cpp
    const int8_t *ix = aie::load_buffer(idx, {pm * 16, 0}, {16, 4});
    for (int r = 0; r < 16; r++) {
        const int8_t *a = aie::load_buffer(x, {(ix[r * 4] & 63), 0}, {1, 256}); // offset from data
```

```
ERROR: gather41_k: the load/store schedule depends on the data: x block 0 on core (0,0) differs between two inputs (DMA order is fixed at compile time)
ERROR: gather41_k: x: the block offsets of a core are not an affine sequence (base + i*step_i + j*step_j); core (0,0) visits (33,0), (29,0) ...
```

The compiler runs the kernel twice with different data to catch exactly this.
Ways around it, depending on the op:

- **gather on the host** before the launch (an embedding lookup of a few rows
  is cheap on the Arm);
- **load the whole table** into core memory when it is small, and index it in
  the core;
- for one-hot style lookups, express them as a **GEMM** with a one-hot matrix
  (chapter 1) -- compute-bound work the array is good at.

Next: [6. GEMM optimization](../06_gemm_optimization/1_low_level_schedule.md).
