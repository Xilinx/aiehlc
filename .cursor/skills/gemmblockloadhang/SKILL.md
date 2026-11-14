---
name: gemmblockloadhang
description: A GEMM (or other non-conv) block-load kernel -- kernel-only `aie::shape` ports or hand-written 2-D GemmSpaces + `aie::load_buffer` -- builds and passes `aiehlc-convtile trace`, but hangs on the board in `wait_io TIMEOUT tile(c,0) ch=0/1 dir=0` (output shim S2MM), or runs but only mesh column 0 is right. Also: `Tensor param: a [16x16]` for a bigger GEMM, `buf_out_ping_0` far smaller than the kernel's C block, a `memref.subview %arg1[16, 0] [64, 64]` style B offset.
---

# GEMM through the block-load path: hang / wrong columns

Reference: `example/tileprogram/ccode/simplegemm_kernel_port.cc` (kernel-only) and
`simplegemm_kernel_space.cc` (hand-written), C[64,64] = A[64,256] x B[64,256]^T on 4x4,
PASS on palmyra 213 since 2026-11-10. History: `doc/design/trace_to_routing.md` step 3.

## Where to look, in order

1. `grep "Tensor param\|GEMM dims" build.log`. All `[16x16]` = M/N/K unknown. The
   GEMM tiling takes M/N/K from extra launch scalars, `#define M/N/K`, or (now)
   `gemmDimsFromSpaces` -- the ports' 2-D GemmSpaces (row bcast A[M,K], col bcast
   B[N,K], gather C[M,N]; all three must be 2-D and agree, else a WARNING).
2. `grep "buf_.*_ping_" aout/worklocal/kernel.cc` against the kernel's blocks. A
   too-small output window = the C block was split twice. A 2-D output with
   `mesh_tiling_group2_dim` must log `is the plain GEMM gather (no group2 split)`;
   the group2 path (and `DmapToDmaphop`'s atomic slab) is 3-D conv output only.
3. `grep "buffer_offset(v<B>" aout/worklocal/host.cc` -- per mesh col the B base must
   step by (N / meshCols) * K bytes. B is declared [K, N] (legacy transpose) but the
   GEMM `#routing.tiling` describes it as [N, K]; `DmaphopTodfscheblueprintPass` may
   take the tiling step only when step != slice (overlap/gap), never as a row count
   of a contiguous split. K == N hides it (simplematmul2 is 256^3).
4. Policy: a plan without halo gets `fullconnect_auto = 1` (`routedDeclsText`); the
   GEMM path with 0 hung. Hand-written form: the GlobalPolicy is the user's.

## Debugging tricks that worked

- Per-core diagnosis on the board from a COPY of the example: count mismatches per
  16x16 block and test which (row block, col block, K-chunk mask) explains a wrong
  block. "Column 0 right, others match nothing" pointed straight at B's offsets.
- Byte-compare generated files against a build from `git show HEAD:<file>` sources
  (`kernel.cc routing.cc host.cc aieml.bcf aieml.prx`): simpleconv2dkernel_port.cc and
  simplematmul2.cc must stay identical after a GEMM fix.
- IR stages to bisect a size: `ir/dfschedule/{0..5}_*.mlir` (`routingextract_data` per
  mesh row, `extract_slice {tag = "producerN"}` per tile, `{tag = "partitionsliceN"}`
  offsets) -- the first stage where the shape goes wrong names the pass.
