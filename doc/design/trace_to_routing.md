# Kernel trace -> routing IR (operator-agnostic DMA derivation)

Status: in progress on `maxwell/trace-to-routing-ir`.

## Problem

Kernel-only sources (`__global__ void k(aie::port<T, ROLE, aie::shape<...>> ...)`,
no `aie::Conv2d` / `aie::Mapping`) are traced on x86 (`kernel_trace.cpp`): every
port's block sequence is fitted to

    offset = base_p0*p0 + base_p1*p1 + step_i*i + step_j*j,  one block shape

and then `recognizeConv()` reads the fits as a spatial-halo conv2d, rebuilds a
`(Conv2dSpec, MappingSpec)` and runs the conv derivation. Anything that is not
that exact conv family (GEMM, eltwise, pool, depthwise, ...) is rejected as
"not recognized", although its fits are just as regular.

## Idea

The fit already is a routing description. Per tensor dim it says which mesh
axis splits the dim (and the step between cores), which on-core loop walks it
(step, rounds), and the block extent -- hence the halo (block > step). Per port
it says whether the block is the same along a mesh row (row broadcast), a mesh
col (col broadcast) or is gathered (output). Those are exactly the fields of the
routing IR tiling carriers (`GemmSpace` d1..d4 `tile_dim` + `slice_tiling` +
`SpatialPolicy`, lowered to `#routing.tiling` / `#routing.halo` /
`#routing.kernel_tiling`, see `routing_ir_tiling_carriers.md`).

`src/llvm/convautotile/trace_routing.{h,cpp}`:

- `planRouting(ports, fits, rows, cols)` -> `RoutingPlan` (per-dim
  `RoutingDim{mesh_axis, mesh_step, loop, rounds, step, block}`), with the
  checks a DMA needs: one mesh axis and one loop per dim, no diagonal walk,
  outer loop on the outer dim ("swap the two loops"), inputs broadcast along
  one mesh axis.
- `routingSpaces(plan, hints)` -> per-port `PortSpace` (the GemmSpace carrier):
  - mesh split dim: `tile_round = parts, tile_size = core slice, stride = mesh
    step`, and an on-core loop on the same dim becomes `slice_tiling`;
  - loop-only dim: `tile_round = rounds, tile_size = block, stride = step`;
  - untouched dim: whole window;
  - policy: row broadcast -> `act=Broadcast, Row`; col broadcast ->
    `wgt=Broadcast, Col`; output -> gather `Row, LeftToRight`,
    `mesh_tiling_group{1,2}_dim` = the dims split by mesh rows / cols.

No rule mentions conv or gemm.

## Verified so far (unit, `src/llvm/convautotile/unitest/run.sh`)

- conv equivalence: the derivation's DMA read back as fits routes to the very
  same carriers -- `emitGemmSpace` text identical for IFM / WEIGHTS / OFM at
  tilings 7x28, 14x16, 28x8;
- GEMM 64x64x256 on 4x4: A -> RowBA (M split, K chunks), B -> ColBB, C ->
  gather (group1 = d1, group2 = d2); eltwise on 4x1;
- rejects: per-core-unique input, swapped loops, diagonal walk;
- real x86 trace of `simpleconv2dkernel_port.cc` routes cleanly.

## Next steps

1. `inferKernelOnly`: replace recognize-conv + `derive()` with
   `planRouting` + `routingSpaces`, rewriting the kernel into the explicit
   `__global__(policy) k(aie::port<T, __aie_k_<port>>)` GemmSpace form
   (+ kernel defines / block descriptors the kernel TU needs); keep the conv
   cost/legality checks (core memory, stack, l1_budget) as generic per-port
   byte checks. Conv kernels must stay byte-identical (aout/newfile.cpp,
   host.cc, kernel.cc, routing.cc).
2. `__global__(conv, map)` (`simpleconv2d_new_load.cc`): cross-check the
   traced plan against the derived carriers (same equality as the unit test).
3. GEMM kernel-only example (`simplematmul2` in load_buffer form) through
   the fullconnect path; GlobalPolicy chosen from the plan (halo ->
   `fullconnect_auto = 0`).
4. Full build + `simpleconv2d_new_load.cc` compile + `apppaltest.py` on board.
