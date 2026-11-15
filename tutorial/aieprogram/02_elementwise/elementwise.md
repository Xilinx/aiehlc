# 2. Elementwise: `z = saturate(x + y)`

> **Status (2026-11-15).** The kernel below passes the compiler's kernel
> analysis (trace, fit and routing plan, shown in section 3), but the pipeline
> after it still derives core windows from its GEMM tiling model and stops the
> build for this shape:
> `error: input: the core window is 64 elements x 1 round(s) per launch, but the kernel's blocks (#routing.kernel_block) are 4096 elements x 16`.
> Making the lowering follow every port's traced blocks is the pending
> compiler work; a runnable, board-measured example replaces this note when it
> lands. Everything else in this chapter -- the model and the limits -- holds
> today.

## 1. The idea: an elementwise op is bound by data, not by compute

An elementwise add reads two bytes, writes one and does one addition. A GEMM
uses every byte that reaches a core many times (chapter 1); an elementwise op
uses it once. So the question is not "how many cores?" but **"how fast can the
data get in and out?"**

Data enters the array through the shim tiles. This partition has 4 shim
columns x 2 MM2S (DDR -> array) channels = **8 input streams**, each measured
at about 2.3 GB/s (`doc/design/resnet_aie_engine.md`). A vectorized add on one
core consumes data much faster than one stream delivers it, so for an
elementwise op the streams are the limit -- the schedule should use all 8, and
more cores than streams add nothing.

## 2. The kernel: one core per mesh row (4x1 mesh)

```cpp
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
__global__ void eltwise_add(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                            aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> y,
                            aie::port<output_window_int8 *, aie::shape<OPS_R, OPS_C>> z) {
    const int tr = 16, tc = 256;                       // block: 16 rows x 256 columns (4 KB)
    const int rows = OPS_R / aie::num_programs(0);    // 64 rows per core
    const int pm = aie::program_id(0);
    for (int i = 0; i < rows / tr; i++)                // 4 row strips
        for (int j = 0; j < OPS_C / tc; j++) {         // 4 column chunks
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            const int8_t *b = aie::load_buffer(y, {pm * rows + i * tr, j * tc}, {tr, tc});
            int8_t *o = aie::store_buffer(z, {pm * rows + i * tr, j * tc}, {tr, tc});
            for (int e = 0; e < tr * tc; e++) {
                const int v = a[e] + b[e];
                o[e] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
            }
            aie::release(x);
            aie::release(y);
            aie::commit(z);
        }
}
// host: aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 1);  eltwise_add<<<mesh>>>(x, y, z);
```

- **Sizes are `#define`s in the `.cc`.** The kernel analysis compiles the
  kernel with the object-like `#define`s of the source file itself, not those
  of included headers.
- **No role tags** -- they are optional; inputs become IFM, the output OFM.
- **A 4x1 mesh**: `device.partition(..., 4, 1)`. Each core owns 64 rows and
  walks them in 4 KB blocks, rows outer, columns inner (the order a DMA can
  stream a row-major tensor in). `program_id(1)` is always 0.

## 3. What the compiler makes of it

```
  x (IFM, row broadcast, 4x4 blocks/core): d1[256]=rows/4@64+i4x16@16 d2[1024]=j4x256@256
  y (IFM, row broadcast, 4x4 blocks/core): d1[256]=rows/4@64+i4x16@16 d2[1024]=j4x256@256
  z (OFM, gather, 4x4 blocks/core): d1[256]=rows/4@64+i4x16@16 d2[1024]=j4x256@256
  => routed: 3 ports on the 4x1 mesh, DMA carriers taken from the traced blocks
```

"Row broadcast" with one core per mesh row is one stream per core: x gets 4
streams, y gets 4 -- all 8 input channels of the partition.

## 4. Why not the 4x4 mesh?

Write the same kernel for 4x4 -- core `(pm, pn)` takes its own block of x and
y -- and the analysis stops:

```
ERROR: ... x (IFM): an input block that depends on both program ids is unique per core;
routing supports inputs broadcast along a mesh row (depends on program_id(0) only)
or a mesh col (program_id(1) only)
```

16 cores x 2 inputs would need 32 different input streams at once; there are 8.
The hardware can share a stream between cores (packet switching, or a MemTile
splitting one stream into several), but the compiler routes inputs only as
broadcasts today.

You *can* occupy 16 cores by broadcasting: send each mesh row's blocks to all
4 cores of the row and let core `(pm, pn)` compute and store only column chunk
`pn` (`if (j == pn) { store_buffer ...; commit }`). The analysis accepts it.
The DDR traffic is the same as on 4x1 -- the stream switch copies the stream
to the 4 cores -- but every core receives 4x the data it uses, and the number
of input streams, which was the limit, does not change. For a
bandwidth-bound op that buys nothing.

## 5. What to take away

- **Bandwidth-bound ops** (elementwise, reductions, copies) are parallelized
  by **DMA channels**, not by cores. Contraction ops (chapter 1) are the
  opposite: compute-bound, they want every core, and broadcasting lets 8
  streams feed 16 cores.
- Use all input streams of the partition; one core per stream pair is enough.
- The best elementwise op is one you never run on its own: **fuse** it into
  the kernel that produces the data (a bias add, an activation, a requantize
  in a GEMM or conv epilogue), so the bytes never make an extra DDR round
  trip.

Next: [3. Reduction](../03_reduction/reduction.md).
