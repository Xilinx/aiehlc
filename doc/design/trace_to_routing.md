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

## Wired into the compiler (`inferKernelOnly`)

Kernel-only sources (`aie::port<T, aie::<ROLE>, aie::shape<...>>`) are now
traced -> `planRouting` -> `routingSpaces(plan, routingHints(plan))` and
rewritten into the explicit form

    constexpr aie::GemmSpace __aie_k_IFM = {...};   // one per port (role, or port var if the role repeats)
    constexpr aie::GlobalPolicy __aie_k_policy = {.fullconnect_auto = 0};
    __global__(__aie_k_policy) void k(aie::port<T, __aie_k_IFM> x, ...)

with no conv recognition. The kernel TU gets `aiehlc_desc_<port>` straight
from the traced fit (`routedTUDecls`). `routingHints` sizes `l1_budget` for
the largest non-halo block and drops to `pp_depth = 1` when double buffering
would overflow 48 KB (the rule the conv recognizer used, now per plan).

Conv recognition survives only as a cross-check: when the blocks also form a
spatial-halo conv whose derivation reproduces them, its carriers must equal
the routed ones (`emitGemmSpace` text) and its `Derived` feeds the pipeline's
tiling check (`convDerivedByKernel`) and the tensor-form autotune cost;
otherwise `routingEstMs` (per-core DMA bytes) ranks candidates.

Behaviour change: a block the conv model called "too big" (a 20-row slab for
a 19-row footprint) is a legal affine walk and is now routed as written.

Checked: `run.sh` (145 + 442), `kernel_trace_cases.py` 27/27 incl. a
kernel-only GEMM fixture (`unitest/gemm_kernel_port.cc`) and the conv kernel
(`simpleconv2dkernel_port.cc`, cross-check equal).

## Full aiehlc build (2026-11-09)

Full `aiehlc` built from this branch (`build/`, venv cmake; this host has no
zstd/libxml2 dev packages, so `zstd_LIBRARY` / `LIBXML2_LIBRARY` point at the
runtime `.so.1` / `.so.2`). `aiehlc.sh --aie-version 5`:

| source | path | result |
|---|---|---|
| `simpleconv2d_new_load.cc` (`__global__(conv, map)`) | conv desugar | `main.elf`, BLK check 0 errors |
| `simpleconv2dkernel_port.cc` (kernel-only) | trace -> routing | `main.elf`; `routing.cc`, `kernel.cc`, `aieml.bcf`, `aieml.prx` **byte-identical** to the new_load build; generated `host.cc` (lines 3-1464) identical, only the echoed user source / debug pragma differ |
| `unitest/gemm_kernel_port.cc` (kernel-only GEMM) | trace -> routing | `main.elf` (0 errors) -- GEMM goes end to end through the same path |

Unit: `run.sh` 145 + 442, `kernel_trace_cases.py` 27/27.

## `__global__(conv, map)` sources go through trace -> routing too (step 4)

`verifyKernelBlocks` (the desugar path of `simpleconv2d_new_load.cc`) now also
runs `planRouting` / `routingSpaces` on the traced fits and compares every
port's carrier with the derivation (`routingCrossCheck`; the scheduling knobs
pp_depth / l1_budget / pad are the declaration's and are copied over first).
The result is a desugar warning/note; the derived carriers stay authoritative.

That check found one real gap: the routed carriers had no channel padding.
`simpleconv2d_new_load.cc` has C = 3 aligned to 4, so the derivation emits
`.padsize = 1` on IFM d3 / WEIGHTS d4. `TracePort::extent` (the data extent
per dim, empty = whole shape) and `RoutingDim::padsize` now carry it, and
`routingSpaces` emits `.padsize`. Result on new_load:

    conv2d_spatial: routing cross-check: trace -> routing gives the derived carriers (3 ports)

Unit: `run.sh` 145 + 725 (`testConvEquivalence` also with C = 3: padsize
checked and rendered text equal), `kernel_trace_cases.py` 27/27. Full rebuild +
`aiehlc.sh --aie-version 5` on new_load: `main.elf`; `routing.cc`, `kernel.cc`,
`aieml.bcf`, `aieml.prx` byte-identical to the build before this change.

## Next steps

1. ~~Full aiehlc build / byte-identity~~ (done above). Still open: generic
   per-port core-memory / stack checks for non-conv plans.
2. ~~`__global__(conv, map)` cross-check~~ (done below).
3. GEMM kernel-only example (`simplematmul2` in load_buffer form) through
   the fullconnect path; GlobalPolicy chosen from the plan (halo ->
   `fullconnect_auto = 0`).
   **2026-11-10, board:** `example/tileprogram/ccode/simplegemm_kernel_port.cc`
   (kernel-only) and `simplegemm_kernel_space.cc` (the same body under
   hand-written GemmSpaces, `verifySpaceKernels`) both build and pass the trace
   check, and both hang on palmyra 213 with `wait_io TIMEOUT tile(3,0) ch=0/1
   dir=0` (output shim). The C tensor's `#routing.tiling` is `d0 = base 16,
   slice 4, step 4, rounds 4` -- each core's 16 output rows split again into 4
   rounds of 4 -- while the kernel commits one 16x16 block per core:
   `tile_round = 4` on the mesh-split d1 is applied as the mesh split AND as an
   on-core round count. `fullconnect_auto = 1` (the GlobalPolicy is the user's in
   the hand-written form) hangs the same way, so the policy alone is not it. The
   earlier "GEMM goes end to end" above meant `main.elf` was produced; it had not
   run on a board. Fix belongs where `routingSpaces` renders a non-halo mesh
   split and where the pipeline derives per-core rounds from `tile_round`
   (`aiehlc.cc`, `perCoreSpatialRounds` / output group1/d2), kept in agreement.
4. ~~Full build + `simpleconv2d_new_load.cc` compile + `apppaltest.py` on board~~
   (done below).

## Board result (palmyra 10.23.224.213, 2026-11-10)

`apppaltest.py --netload` from `xiaojiang/apppaltest-devlock` (devlock +
watchdog), ELFs built by this branch (4267fb3) with `aiehlc.sh --aie-version 5`.
Logs: `aiengineers/maxwell/tasks/2026-11-10_trace_routing_board/`.

| ELF | result |
|---|---|
| `simpleconv2d_new_load.cc` as checked in | rc 0, `aie_wall_ms=164.2`, app exits normally (this source has `verify_conv2d` commented out, so it only checks that the app runs to completion) |
| same source with `verify_conv2d` enabled (copy in /tmp; header returns right after the compare, without the input dump) | rc 0, `aie_wall_ms=164.3`, **`verify: 0 mismatches of 802816`** against the CPU reference conv2d |

The verify build's `routing.cc`, `kernel.cc`, `aieml.bcf` and `aieml.prx` are
byte-identical to the checked-in source's build, so the AIE side checked by
verify is the same one the checked-in source runs.

Board note: after the first new_load run, the next two runs failed apppaltest's
pre-run netload smoke test (`load nbspin5.elf rc=3/5`) even though hw_server
was up and the lock was free. Reloading the supervisor with
`--netload-bootstrap --skip-preflight` fixed it. The cause is not known (the
supervisor may not take the app core back after this app); owner: apppaltest/netboot.
