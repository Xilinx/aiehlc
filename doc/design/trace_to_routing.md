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

**The role tag is optional.** `aie::port<T, aie::shape<...>> x` (no `aie::IFM` /
`aie::WEIGHTS` / `aie::OFM`) is accepted next to the tagged form. The DMA never
came from the tag -- input vs output is the window type, broadcast direction is
how the traced blocks move with `program_id` -- so the tag only names the routed
GemmSpace (`__aie_k_IFM`) and lets `recognizeConv` find the conv's three roles.
Untagged ports get it inferred (`TracePort::roleInferred`): an output window is
OFM, an input is IFM, and `inferRoles()` (after the trace, before planning)
turns an input whose blocks move with `program_id(1)` only -- broadcast along a
mesh column, the B / filter side -- into WEIGHTS. `rewriteKernelRouted` finds a
port by its parameter name and replaces `[aie::ROLE, ]aie::shape<...>`. Checked:
`simpleconv2dkernel_port.cc` and `simplegemm_kernel_port.cc` with the tags
stripped give byte-identical GemmSpaces, policy and signature, and the conv is
still recognized (`test_trace_routing` `testUntaggedPorts`, `kernel_trace_cases.py`
"role tags omitted"). Write the tag when the inference would guess wrong -- an
input that is broadcast along a mesh row but is not the IFM side, or a kernel
with two inputs that both move with `program_id(1)`; tagged ports are never
re-inferred.

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

## Source-level diagnostics before the trace (2026-11-12)

Two mistakes in `example/other/aiehlcpm_gemm1.cc` used to fail with no usable
message. Both are now named where they are detected:

- **Launch mesh unresolved.** The mesh is read from the source text
  (`findMeshShape`), with macros taken from the file and the headers it
  includes from its own directory. Only `#include "x.h"` used to be followed,
  so `device.partition({3,6,0,6}, HW_ROWS, HW_COLS)` with the macros in
  `#include <x.h>` failed as "no device.partition(...) found", although the
  call was there. Angle includes that resolve in the source directory are now
  followed (`aiehlc.sh` puts that directory on the include path, so the
  compile resolves them there too). An unresolved mesh now reports the cause:
  either there is no partition call, or ROWS (or COLS) is not an integer
  constant expression and has no `#define` in the source or in a header it
  includes from its own directory.
- **Window passed by value.** `aie::port<input_window_int8, ...>` traced fine
  on x86 and failed only in xchesscc (`no matching function for call to
  'aiehlc_load_buffer'`). `parsePort` now rejects it: `port a: the window type
  must be a pointer`.

- **Missing header.** `#include <aiehlcpm_gemm1——3.h>` (no such file) showed
  only as the undefined-macro error above, because the source-text passes run
  before Clang and skip headers they cannot open. `aiehlc.cc`
  `checkSourceIncludes` now resolves every include outside `#if` against
  Clang's search path first, and stops with Clang's `fatal error: '...' file
  not found`.

Cases: `kernel_trace_cases.py` "gemm: mesh ROWS/COLS from an #include
<header> macro", "gemm: mesh ROWS is an undefined macro", "gemm: window passed
by value". Skill **kernelanalysisdiagnostics**.

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

## Dataflow options: `.depth` / `.resident` / `.tile` (2026-11-11)

A kernel-only kernel may pass options to its block calls (all optional; the
default is the compiler's derivation as before):

```cpp
const int8_t *w = aie::load_buffer(b, {pn * 64, 0}, {64, 256}, {.resident = true});
const int8_t *x = aie::load_buffer(a, {r0, 0}, {16, 256}, {.depth = 2, .tile = {8, 8}});
int8_t *y = aie::store_buffer(c, {r0, pn * 64}, {16, 64}, {.depth = 2});
```

| Option | Meaning | Checked by the trace |
|---|---|---|
| `.depth = 1 \| 2` | this port's core buffers (single / ping-pong); a single buffer with several rounds re-arms by chaining its BD to itself | 1 or 2; at most that many blocks held |
| `.resident = true` | one block per core, held for the whole launch, ONE buffer; the DMA sends it once, not with every round of the other ports | exactly one block per core; no `.depth = 2` |
| `.tile = {tr, tc}` | loads only, 2-D blocks: the shim DMA delivers the block panel-major (tc-wide column panels, each with all rows), so every tr x tc tile is contiguous -- tile (a, b) at `(b * rows / tr + a) * tr * tc`. Three shim BD dims: d0 one tile row, d1 the rows (row pitch), d2 the panels | divides the block; tile row and tensor row multiples of 4 B (the DMA moves words) |

Designated initializers go in the order `.reuse, .pad, .depth, .resident, .tile`.
On the core the options are dropped (`dropBlockOptions` in
`resolveBlockIoBuiltins`): they configure the DMA and the analysis, a block is
still just the next window. Every block of a port must pass the same options.
Example: `example/tileprogram/ccode/simplegemm_dataflow.cc` (256x256x256 x 8
panels in one launch, `aie::mmul<8,8,8>`, B host-packed in mmul order because
an int8 transpose is not a DMA operation).

**How it reaches the DMA.** The kernel analysis records each port's traced
block (`KernelOnlyResult::blocks`: the kernel's tensor shape, the fit, the
options); aiehlc hands it to the pipeline (`TensorParam::kernelBlock`) and
`attachTensorRoutingInfo` puts it on the tensor as `#routing.kernel_block`
(routing_ir_tiling_carriers.md). `.depth` / `.resident` also set the port's own
GemmSpace `pp_depth` (`RoutingPort::ppDepth`). The blueprint lowering resolves
the attribute once per port, in `blueprint_sched::resolveKernelBlock`, from the
attribute alone -- host (`flowtransfer_host.cpp` `applyKernelBlockShim`,
`flowtransfer_kernel.cpp`) and kernel (`passblueprinttoschedulekernel.cpp`) make
the same decision:

- **check** -- every GEMM-path port (`fullconnect_auto = 1`, no halo): the final
  core window and round count must be the kernel's block and block count, else
  `error: ... the core window is N elements x R round(s) per launch, but the
  kernel's blocks (#routing.kernel_block) are ...`. Before, a disagreement built
  rc=0 and hung on the board.
- **follow** -- an input on that path whose blocks are row-contiguous (cover
  every dim after the first): one shim BD per kernel block at the block's own DDR
  offset (`step_i * i + step_j * j`), one core window per block, the port's own
  `pp_depth`, `.tile` dims; an input taken once is sent once, outside the round
  loop. The GEMM model cannot express either case: it re-sends B with every M
  round, and with full K (`kRounds == 1`) it kept A's core window and shim BD at
  the whole 64-row partition (16 KB per round, no per-round offset).
  The old K-chunked blocks (`simplegemm_kernel_port.cc`, `{16, 64}` of a 256-wide
  tensor) are not row-contiguous: they keep the GEMM schedule, byte-identical.

Two more front-end changes go with it. A dim that is mesh-split AND walked by an
on-core loop in back-to-back blocks (GEMM M strips: `rows/4@512 + j32x16@16`) is
rendered as one flat tile list (`tile_round` = mesh parts x rounds) -- the form
the GEMM schedule reads -- instead of `slice_tiling`, which only the halo path
reads (`RoutingDim::flatLoop`). And the non-halo memory rule now counts what
`kernel.cc` declares (`RoutingPort::declaredBytes`: ping AND pong for every
window, except a single-buffered followed input) against the 32 KB buffer region
less 3 KB (skill dmbufferregion), not 48 KB; ports without options drop to one
buffer when that is what fits, and a set that cannot fit is an analysis error
(`checkPortBuffers`) instead of a chess link error.

Verified: `simplegemm_kernel_port.cc`, `simpleconv2dkernel_port.cc`,
`simplematmul2.cc` give byte-identical `kernel.cc` / `routing.cc` / `host.cc` /
bcf / prx before and after; unit `test_trace_routing` (strip rendering, per-port
depth, buffer rule) and `kernel_trace_cases.py` DATAFLOW_CASES (each option's
error); `simplegemm_dataflow.cc` bit-exact on x86 (kemu-style emulation with the
panel-major gather, 524,288 outputs). Skill **gemmdataflowoptions**.

**Board (palmyra 213, 2026-11-11, `apppaltest.py --netload`).**
`simplegemm_dataflow.cc`: `GOLDEN PASS: all 524288 output elements match`,
79.55 ms per launch (first call 79.94 ms), 3.37 GOPS for 8 x 256^3 -- the
followed A strips (panel-major `.tile` BD, ping-pong) and the once-sent,
single-buffered B work on hardware. `simplegemm_kernel_port.cc` from the same
compiler: `PASS`, 18.46 ms (as before) for 64x64x256. The `--profiling` split of
one launch: **kload 96.5 %**, bdcfg 2.3 %, wait_io 1.2 %, i.e. the DMA + mmul
work is ~1 % (< 1 ms); the launch is the kernel ELF reload -- 3 KB of text but
26.6 KB of BSS (the port buffers) per core, written over the config bus on every
call. That, not the dataflow, is the next lever (warm reload / control-packet
kernel load, as conv2dstem does).

**Spatial-halo conv (2026-11-14).** The same options on a conv kernel:
`example/tileprogram/ccode/simpleconv2dkernel_port_opts.cc` (59 lines,
`simpleconv2dkernel_port_less.cc` + `{.resident = true}` weights, `{.depth = 2}`
slab and output). On `fullconnect_auto = 0` only a `.resident` input is
checked / followed (`resolveKernelBlock`): the weights get one 3136 B buffer,
sent once. The conv cross-check (`convCrossCheck`) ignores the port's
`pp_depth` (a scheduling choice, like `spaceDiff`); `{.depth = 1}` on a halo
port is an analysis error (its window lowering is ping-pong). `.tile` needs a
2-D block, so it does not apply to the 3-D slab. palmyra 213: `PASS: 0
mismatches out of 802816`, 85.8 ms; `simpleconv2dkernel_port.cc` byte-identical.

## Next steps

1. ~~Full aiehlc build / byte-identity~~ (done above). Per-port core-memory
   check for non-conv plans: done (`checkPortBuffers`, above). Still open: stack.
2. ~~`__global__(conv, map)` cross-check~~ (done below).
3. GEMM kernel-only example (`simplematmul2` in load_buffer form) through
   the fullconnect path; GlobalPolicy chosen from the plan (halo ->
   `fullconnect_auto = 0`).
   **Done 2026-11-10** -- `example/tileprogram/ccode/simplegemm_kernel_port.cc`
   (kernel-only) and `simplegemm_kernel_space.cc` (hand-written GemmSpaces) run on
   palmyra 213, 18.5 ms, bit-exact (`PASS: all 4096 output elements match`).
   Before, both built and passed the trace check but hung in `wait_io TIMEOUT
   tile(3,0) ch=0/1 dir=0`. Three compiler bugs, none in the examples:
   1. **No M/N/K.** The GEMM tiling derivation took M/N/K only from extra launch
      scalars (`k<<<mesh>>>(A, B, C, M, N, K)`) or `#define M/N/K`; every tensor
      fell back to 16x16. `gemmDimsFromSpaces` (`aiehlc.cc`) now derives them from
      the ports' own 2-D GemmSpaces (row bcast `A[M,K]`, col bcast `B[N,K]`,
      gather `C[M,N]`, checked to agree) when neither is given.
   2. **2-D output through the conv group2 path.** `mesh_tiling_group2_dim = 2` on
      a 2-D C sent it down the 3-D conv-output path, which cut the per-row view to
      one tile's block and `DmapToDmaphop` then split it across the row again
      (16x16 -> 4x16 per core, 64-B output window). A 2-D output now takes the
      plain GEMM gather.
   3. **B's per-column DDR offset.** `DmaphopTodfscheblueprintPass` moved every
      partition slice by its `#routing.tiling` outer step. The GEMM tiling
      describes the col-broadcast B as [N, K] while the tensor is declared
      [K, N], so mesh col c started at row 16c (1024 B) instead of 64c (4096 B):
      cols 1..3 computed on the wrong B whenever K != N. The tiling step is now
      used only when it differs from the slice (overlap / gap, e.g. the conv halo
      61/56); contiguous slices advance by the slice actually extracted.
   And the plan now picks the GlobalPolicy, as planned here: halo ->
   `fullconnect_auto = 0`, otherwise 1 (`routedDeclsText`).
   Unchanged: `simpleconv2dkernel_port.cc` and `simplematmul2.cc` give
   byte-identical kernel.cc / routing.cc / host.cc / bcf / prx before and after.
   Skill **gemmblockloadhang**.
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
