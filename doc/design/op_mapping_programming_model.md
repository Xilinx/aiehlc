# Op + mapping programming model: `__global__(op, map)`

A kernel declares **what it computes** (the operator) and **how it is mapped**
onto the array (space split, per-core time tiles, hints). Each port names only
its **data role**. The compiler derives every per-port DMA descriptor.

```cpp
constexpr aie::Conv2d conv = {.h = 224, .w = 224, .c = 3, .k = 64, .kh = 7, .kw = 7, .stride = 2, .pad = 3};
constexpr aie::Mapping map = {
    .space = {aie::split(aie::OH, MESH_ROW), aie::split(aie::OC, MESH_COL)},
    .time  = {aie::tile(aie::OH, 7), aie::tile(aie::OW, 28)},
    .hints = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma, .pp_depth = 2,
              .l1_budget = aie::Bytes{4096}}};

__global__(conv, map) void conv2d_spatial(aie::port<input_window_int8 *, aie::IFM> win_a,
                                          aie::port<input_window_int8 *, aie::WEIGHTS> win_b,
                                          aie::port<output_window_int8 *, aie::OFM> win_c) {
    const int kh = conv.kh, kw = conv.kw, s = conv.stride;          // the op
    const int c  = aie::get_arg_per_round_size_in_dim(2, win_a);    // what the port loads
    const int oh_t = aie::get_arg_per_round_size_in_dim(0, win_c);  //   per round
    ...
```

`example/tileprogram/ccode/simpleconv2d_new_load.cc` is this model; `simpleconv2d.cc`
is the same conv written against the per-port model (three hand-computed
`GemmSpace` descriptors and 28 kernel-visible `#define`s). Both produce **byte-identical** IR (all 28 `ir/dfschedule` +
`ir/simplerouting` dumps), `kernel.cc`, `routing.cc`, bcf, prx and the host DMA
program (`host_canonicalized`).

## 1. What is implemented

The derivation is **convautotile** (`src/llvm/convautotile/`, design
[conv_autotiling.md](conv_autotiling.md)): a source-level desugar that runs
before every other aiehlc rewrite, comments out `conv` / `map`, emits the three
derived `GemmSpace`s, a `GlobalPolicy`, and the legacy kernel `#define`s
(`SP_*`, `OH_T`, `TILE_*`), and rewrites `__global__(conv, map)` and the port
tags onto them. It also validates the mapping (`[ConvAutoTile] ERROR`),
autotunes an omitted `time`, and takes tuner overrides (`AIEHLC_CONV_TUNE`,
`src/frontend/convtune`).

On top of it, for kernels written in the Triton block style (Section 4):

| Piece | Where |
|---|---|
| kernel-TU `conv`, `map` (block sizes) and per-port block descriptors, from convautotile's `Derived` | `convautotile::kernelTUDecls`, appended by `aiehlc.cc` after the `#define` scan |
| `aie::program_id`, `aie::load_buffer/store_buffer/release/commit` | `tilinglinalg_pipeline.cpp` (`resolveBlockIoBuiltins`) + `src/mlir/runtime/aie_block_io.h` |

Kernel-body locals must not reuse the emitted macro names (`OH_T`, `OW_T`,
`TILE_*`, `SP_*`) or the example header's `M`/`N`/`K`.

## 2. Compiler design for native support

The desugaring proves the model but keeps its knowledge in a source-text rewrite.
Native support moves it into the compiler, in steps that each keep the
generated code byte-identical on the existing examples.

### Step A — done: desugar to `GemmSpace` (convautotile)

Zero backend change; validation and autotune live in the desugar.

### Step B — the op and the mapping become IR

```mlir
func.func @main(...) attributes {
    routing.op      = #routing.conv2d<h = 224, w = 224, c = 3, k = 64, kh = 7, kw = 7, stride = 2, pad = 3>,
    routing.mapping = #routing.mapping<space = [#routing.split<oh, mesh_rows>, #routing.split<oc, mesh_cols>],
                                       time  = [#routing.tile<oh, 7>, #routing.tile<ow, 28>],
                                       pp_depth = 2, l1_budget = 4096>}
```

- convautotile's parsed `Conv2dSpec` / `MappingSpec` are attached to the
  routing func instead of being rendered as `GemmSpace` text.
- A new **`OpMappingToTilingPass`** (first pass of `runPipeline`) owns the
  derivation in C++: it writes the `#routing.tiling` on each `partitiontensor`,
  the `#routing.halo` on each `routingcreatescheduletensor`, and
  `#routing.kernel_tiling` — the carriers of
  [routing_ir_tiling_carriers.md](routing_ir_tiling_carriers.md). Everything
  downstream is untouched.
- Roles bind by **SSA position**, not by tags in a type: the launch
  `k<<<mesh>>>(input, filter, output)` gives func args 0..2 and the kernel
  signature gives their roles.
- `mesh_rows` / `mesh_cols` are symbolic and bound from the launch mesh
  (`device.partition(..., HW_ROWS, HW_COLS)`), so `MESH_ROW` stops being a
  user macro that must agree with the launch.

Why a pass and not a text desugar: real diagnostics (MLIR `emitError` with the
op/mapping values), the L1 budget check needs the element type and pp depth,
and autotuning (Step C) needs search, not a closed formula.

### Step C — defaults and autotune (convautotile has it)

`time` is optional (`// optional: omit -> autotune under l1_budget`). convautotile
already enumerates the legal tiles and ranks them with an analytic cost model;
`src/frontend/convtune` drives AutoTVM over the same space. What remains is to
move that search into the pass so it can price with IR-level facts. The pass enumerates `(oh_t, ow_t)` divisors, prices each with
`slab_bytes(ih_t, iw_t, c) * pp_depth + out_bytes * pp_depth + weights` against
`l1_budget`, and picks the largest fitting tile (fewest rounds = fewest BD
re-arms; cf. `doc/performance/register_write_cost.md`). The choice is printed
and recorded in `#routing.kernel_tiling`, so it is reproducible.

### Step D — more operators

Each operator is one derivation rule set behind the same `Mapping`:

| op | IFM | WEIGHTS | OFM |
|---|---|---|---|
| `Conv2d` (done) | halo slabs | OC split | tiled gather |
| `Gemm` | M split (row broadcast) | N split (col broadcast) | M×N gather — today's `RowBC`/`ColBC` |
| `DepthwiseConv2d` | halo slabs, C split | C split | tiled gather |
| `Pool2d` | halo slabs | — | tiled gather |
| `Eltwise` | aligned split | — | aligned split |

### Step E — a typed kernel view

Replace string-resolved builtins with a generated per-kernel struct, e.g.
`conv2d_spatial_geom::ifm.round_shape`, emitted into the kernel TU next to the
op. The kernel then cannot ask for a dim that does not exist, and a layout
change is a compile error instead of a silent wrong index.

`GemmSpace` ports remain as the **escape hatch** for mappings the rules do not
cover; a kernel uses one model or the other.

## 3. Evaluation

**It makes sense, and it is the right level for a user-written AIE kernel** —
on the condition that the compiler owns validation and the buffer-layout
contract. Measured on this conv:

| | per-port `GemmSpace` (`simpleconv2d.cc`) | op + mapping (`simpleconv2d_new_load.cc`) |
|---|---|---|
| mapping description | 3 hand-derived descriptors (62 lines: halo 61/56, rounds 19/14/4, width 61/56/4, padded 230...) | 2 `split` + 2 `tile` |
| kernel-side geometry | 28 `#define`s, mirrored by hand so the kernel TU sees them | none; read from `conv` and the ports |
| change `ow_t` 28 → 14 | `OW_T`, `RowBC_spatial.d2.tile_round`, and `tile_round`/`tile_size`/`stride` of `LtoR_Merge.d2` | `tile(OW, 14)` |
| a bad split | runs and produces wrong data | `[ConvAutoTile] ERROR` naming the constraint |
| generated host/kernel/routing | — | byte-identical |

What it gets right:

- **One source of truth.** The halo numbers are consequences of
  `(conv, map)`; writing them by hand is where bugs come from. The `230 / 4`
  edit to this file is exactly that bug in the new syntax: the padded DDR
  layout was written into the *operator*, which would pad twice (OH 115). The
  model makes the distinction explicit — the op is the math, the layout is
  derived — and the derivation can now check it.
- **Ports declare data, not DMA.** "This port loads the IFM" is what the
  kernel author knows; how many BD dims it takes is what the compiler knows.
- **Mapping is a separate, swappable decision** — the property that makes
  autotuning and per-target mappings possible (as in Triton's block sizes,
  IREE/AIR tiling, MLIR-AIE ObjectFIFOs).

Where it is weak, and what to do about it:

1. **The kernel body still bakes in a layout.** It indexes the slab as HWC
   (`slab[ih * raw_wc + iw * c + ch]`), the filter as `B^T [oc][kh][kw][c]`,
   and writes HWC output. Nothing ties that to what `derive` produces. Make the
   role carry the layout (`aie::IFM` ≡ `[ih_t][iw_t][c_align]` row-major per
   round) and generate the kernel view (Step E); otherwise a derivation change
   is a silent wrong-index bug.
2. **Hints can lie.** `im2col = Im2col::Dma` is ignored on the halo path — this
   kernel does im2col on the core. Hints that a rule does not consume should be
   rejected, not ignored.
3. **`MESH_ROW` is a macro that must match the launch mesh** — fixed by Step B.
4. **Coverage is one operator.** Conv2d with an OH×OC split only; other ops or
   splits (e.g. OW across mesh cols, IC accumulation) need rules (Step D).
   Until then `GemmSpace` stays available.
5. **Less low-level control** than hand-written descriptors — acceptable
   because the escape hatch stays, and the generated descriptors are visible in
   `ir/dfschedule/0_initial.mlir`.
6. **Accumulator width is the kernel's** (`int16_t sum` here, which wraps on
   real data — see `conv2dstem.cc`, peak 1,023,225). The op does not carry
   dtypes yet; adding `in/weight/acc/out` dtypes to the op would let the
   compiler flag it.

**Recommendation:** adopt it as the default way to write single-op kernels,
proceed with Step B (op/mapping in IR + `OpMappingToTilingPass`) next — it is
where the remaining weaknesses (diagnostics, mesh binding, autotune, layout
contract) get fixed — and keep `GemmSpace` as the expert escape hatch.

## 4. Triton block-load convention (`aie::load_buffer` / `aie::store_buffer`)

The kernel body of `simpleconv2d_new_load.cc` is written in Triton's style: each
core computes *where* its blocks are from its program ids and loop indices,
in **element coordinates of the whole tensor**, and loads/stores them.

```cpp
const int pid_h = aie::program_id(0), pid_c = aie::program_id(1);  // tl.program_id
const int8_t *w = aie::load_buffer(win_b, {pid_c * OC_T, 0, 0, 0}, {OC_T, KH, KW, C});
for (int i = 0; i < map.oh_per_core / OH_T; i++)
    for (int j = 0; j < map.ow / OW_T; j++) {
        const int oh0 = pid_h * map.oh_per_core + i * OH_T, ow0 = j * OW_T;
        const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {IH_T, IW_T, C});
        int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * OC_T}, {OH_T, OW_T, OC_T});
        ... compute y from x, w ...
        aie::release(win_a); aie::commit(win_c);
    }
aie::release(win_b);
```

`map.*` (block sizes) and `conv.*` are compile-time literals in the kernel TU,
like Triton's `tl.constexpr` block sizes.

### How it is lowered

| Piece | Where |
|---|---|
| `conv`, `map` (block sizes) and per port `aiehlc_desc_<port>` + cursor — block shape, per-core base (×`program_id`) and per-round steps, from the same convautotile `Derived` numbers as the port's DMA | `convautotile::kernelTUDecls` → kernel TU |
| `aie::program_id(0/1)` → `get_coreid()` minus the placement's first core row / column | `tilinglinalg_pipeline.cpp` (`resolveBlockIoBuiltins`, from the `dmaphop.tile` placement) |
| `aie::load_buffer/store_buffer` → acquire the port's next window **and check** that the requested block is the one the DMA delivers next (klog `BLKE`/`BLKW`/`BLKG`); `release/commit` → release the window | `src/mlir/runtime/aie_block_io.h` |

### Verified

- `simpleconv2d_new_load.cc` (Triton form) → all 28 IR dumps, `kernel.cc`,
  `routing.cc`, bcf, prx and the host DMA program are **byte-identical** to the
  hand-written `simpleconv2d.cc`.
- `simpleconv2ddebug/emu_conv2d_spatial.sh aout/worklocal` runs the
  **generated** kernel on x86 for all 16 cores with windows that serve blocks
  exactly as the host shim BDs stream them: **802,816 / 802,816** outputs equal
  the CPU reference, 0 block-check errors. Not yet run on the board.
- Swapping the `i`/`j` loops: the check reports all 768 out-of-order blocks —
  and the output is **still bit-correct**, which is the central point below.

### Evaluation: does the Triton convention make sense on AIE?

**As written, the load does not choose the data — the DMA does.** In Triton,
`tl.load(ptr + offsets)` *is* the memory access. On the AIE the data is
streamed by DMA in an order fixed at compile time; the core can only take the
next window. So `aie::load_buffer(win, offset, shape)` is an **assertion about
what arrives**, not a request — the swapped-loop experiment proves it (wrong
offsets, right output). That has three consequences:

1. **Without the check, the offsets are dead text.** Any mismatch between what
   the kernel believes it holds and what it holds is invisible until the kernel
   uses the offset (boundary handling, per-position bias, debug prints). The
   check makes them honest — but it is a *runtime* check, visible only with
   kernel logging (klog is compiled out by default).
2. **The kernel duplicates the tiling.** The loop bounds and offset formulas
   restate what `(conv, map)` already fixed. They can only be *right* by
   agreeing with it; the user-written version showed how easily they don't
   (row/col swapped, 61·4·230·4 instead of 56·920, step 19 instead of 14, 58
   instead of 56, increments before use, no reset, output reusing the input
   offset, comma-operator shapes).
3. **It cannot express what the DMA cannot deliver.** Random access, a
   different visiting order, or a block shape the BDs do not produce cannot be
   honored — only rejected.

**When it does make sense:** if the compiler **derives the DMA from the loads**
instead of checking them — the real Triton contract. The offsets in this
kernel are affine in `(program_id, i, j)` with constant shapes, which is
exactly what a BD is (base + strides × wraps):

| load | affine form | BD |
|---|---|---|
| `x` | `(56·p0 + 14·i, 56·j, 0)`, shape `(19, 61, 4)` | base 51520·p0, D0 4 B×61, D1 920×19, D2 224×4 (j), iter 12880×4 (i) |
| `y` | `(28·p0 + 7·i, 28·j, 16·p1)`, shape `(7, 28, 16)` | gather BD, same round order |
| `w` | `(16·p1, 0, 0, 0)`, shape `(16, 7, 7, 4)` | one 3136 B BD at 3136·p1 |

That table is the host program the compiler emits today. So a compiler pass
that extracts the loop nest + affine offsets (the same analysis MLIR-AIE/AIR
apply to `affine.load`) could *produce* the tiling from the kernel, making
`map.time` optional and the loads authoritative. Until that exists, the
convention is a readable, checked restatement of the mapping — useful for
humans and debugging, but not a second source of truth.

**Recommendation.**

- Keep the **op + mapping** declaration as the source of truth (Sections 1–3).
- Keep the **Triton block convention** as the kernel's *view* of its data —
  it is the clearest way to say what a block is — but make it impossible to
  disagree with the mapping instead of checked at run time:
  1. near term: resolve the offsets at compile time — the descriptors are
     constants and `i`/`j` are the round loop, so the compiler can generate the
     loop and the offsets (`for (auto [i, j, x, y] : aie::blocks(win_a, win_c))`)
     rather than the user writing them; a hand-written offset becomes a static
     check;
  2. longer term: derive the DMA from the kernel's affine loads (above) and drop
     `map.time` as a separate input.
- Either way, `aie::program_id` and the tensor-coordinate block shapes are
  worth keeping: they make a core's footprint explicit and replace raw byte
  arithmetic, which is where the sketch's bugs came from.

## 5. Kernel analysis: the DMA from the kernel (no `conv`, no `map`)

`example/tileprogram/ccode/simpleconv2dkernel_port.cc` is the conv2d with **no**
`aie::Conv2d` / `aie::Mapping`. The ports declare only their tensors and the
kernel states its own tiling:

```cpp
__global__ void conv2d_spatial(
    aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
    aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
    aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const int KH = 7, KW = 7, S = 2, C = 4, oh_t = 7, ow_t = 28, oc_t = 16;
    const int pid_h = aie::program_id(0), pid_c = aie::program_id(1);
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});
    for (i ...) for (j ...) {
        const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
        int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});
        ... aie::release(win_a); aie::commit(win_c);
    }
    aie::release(win_b);
}
```

### How the DMA is found (`src/llvm/convautotile/kernel_trace.{h,cpp}`)

1. **Trace.** `applyConvAutoTiling` (aiehlc.cc) first calls
   `inferKernelOnly`: the kernel body is compiled for x86 with recording stubs
   (`aie::load_buffer/store_buffer/release/commit/program_id/num_programs`)
   under AddressSanitizer, with a `#line` map so diagnostics name the user's
   source line, and run for every core of the launch mesh
   (`device.partition(..., R, C)`), twice with different data.
2. **Fit.** Each port's block sequence becomes
   `offset = base_p0*p0 + base_p1*p1 + step_i*i + step_j*j`, rounds i outer /
   j inner, one shape -- the form of a DMA buffer descriptor. Per-core starts
   must be affine in `program_id` and start at the tensor origin.
3. **Recognize.** The fits are read as a spatial-halo conv: OH band per mesh
   row (`base_p0`), OC group per mesh col (`base_p1`), output tile =
   time tile, stride = input step / output step, kernel window = the weight
   block, halo = overlap of consecutive input blocks.
4. **Derive + compare.** That `(conv, map)` goes through convautotile's
   `derive()` (so the DMA is the proven one), and every port's fit must equal
   the derived blocks exactly -- too big / too small (per dimension), per-core
   offsets, rounds, steps.
5. **Rewrite.** The source becomes `__global__(__aie_conv_<k>, __aie_map_<k>)`
   with synthesized declarations (`aie::shape` stripped,
   `aie::num_programs` folded); the existing desugar and pipeline take it from
   there. For `simpleconv2dkernel_port.cc` the result is byte-identical to the
   hand-written `simpleconv2d.cc` (28/28 IR dumps, `kernel.cc`, `routing.cc`,
   bcf, prx, host DMA program); x86 emulation 802,816/802,816.
6. **Cross-check the pipeline.** The shim DMA is built from the derived
   descriptors, but the **core** windows (`window_init` size, core MM2S/S2MM BD
   length) and the output round count (`out_rounds`) are re-derived later by
   the GEMM-model tiling in aiehlc.cc. `checkConvTilingAgainstAnalysis` holds
   that result to the analysis -- rounds per core, OFM / IFM / WGT block bytes
   -- and a mismatch is a build error (`[TilingLinalg] ERROR: <k>: output rounds
   per core (out_rounds) = 98 ... the kernel's blocks give 14`), because on the
   board it is a silent `wait_io` hang. Success prints `derived tiling matches
   the kernel analysis (rounds N, OFM .. B, IFM .. B, WGT .. B)`. The bug it
   was added for: the C port's `d2` is the output **width** tile, the GEMM
   model reads it as N, and `spatialNRounds` (= width rounds, already inside
   `spatialMRounds`) was reset to 1 only when `ow_t != oc per core`; a
   14x16x16 tile (ResNet-18 layer 1 prototype) got `out_rounds = 14*7 = 98`
   and a 512 B core output window. It is now reset for every spatial-halo conv.
   Skill **convoutroundsdoublecount**.

`__global__(conv, map)` kernels that use `load_buffer` get the same trace,
compared against their declared mapping (`verifyKernelBlocks`, from
`desugarConvMapping`).

### What the verifier rejects

`src/llvm/convautotile/unitest/kernel_trace_cases.py` mutates the example into
each bug and checks the diagnostic:

| Bug | Diagnostic |
|---|---|
| load bigger than the DMA block | `block (20,61,4) is too big in dim 0 (rows): it takes 20, the DMA block is 19 (a 7x28 output tile of a 7x7 stride-2 conv)` |
| load smaller than the conv footprint | `accesses memory outside the block ... at case.cc:57 on core (0,0): data used without being loaded` |
| index past the loaded block | same, at the reading line |
| use after `aie::release` | `uses a block after aie::release / aie::commit at case.cc:58` |
| offset outside the tensor (`pid_c * oc_t * 20`) | `win_b block 0 on core (0,1): block dim 0 covers [320, 336) outside the tensor extent 64` |
| block never committed / released | `16 block(s) still held at kernel end: never committed (the next DMA round would deadlock)` |
| more than `pp_depth` blocks held | `takes a block while 2 of this port are still held (pp_depth 2)` |
| loops swapped | `blocks are visited column-outer ...; the DMA streams output rows in the outer loop -- swap the two loops` |
| data-dependent schedule | `the load/store schedule depends on the data: win_a block 1 on core (0,0) differs between two inputs` |
| output not fully written | `200704 output element(s) are never stored, first (0,84,0)` (overlaps likewise) |
| port never used | `win_b (WEIGHTS) is never loaded (aie::load_buffer)` |
| store on an input port / load on an output | `aie::store_buffer on an input port` |
| block over the core memory | `block is 211600 B, larger than the core data memory (49152 B)` |

| block sized by a constant the compute ignores (`oc_t = 15`, loops walk 16 channels) | `cause (static analysis): \`oc_t = 15\` (case.cc:8) sizes the block of the access at line 29, whose index never reads oc_t -- with oc_t = 16 every analysed block access fits` |

`aiehlc-convtile trace <src.cc> [--emit]` runs the analysis standalone
(`AIEHLC_TRACE_KEEP=1` keeps the generated harness; the harness goes under
`$TMPDIR`, default `/tmp`).

### Why an access leaves its block: static bounds (`kernel_static.{h,cpp}`)

ASan says *where* ("data used without being loaded at k.cc:29") and stops at
the first fault. `staticBlockBounds()` says *why*, from the source alone, and
runs inside `traceKernel` / `traceLogical` after every trace:

- **Model.** The body's `const int` / `#define` constants, the op/mapping
  struct members of a `__global__(conv, map)` kernel (`map.oh_t`, `conv.kh`,
  from the prelude's `static constexpr aiehlc_kernel_view_t map = {...}`),
  small const arrays (`const int owc[2] = {owA, owB}`), `for (int v = a; v < b;
  v += s)` loops (extra `, kk++` increments allowed), `if` / `?:` guards, block
  pointers (`T *p = aie::load_buffer(port, {off}, {shape})`), pointers derived
  from them (`T *r = p + e`, `(T *)p + e`, `reinterpret_cast<T *>(p) + e`) and
  local arrays. A name assigned after its declaration has no single value and
  is dropped.
- **Bytes, not elements.** Every pointer carries its pointee size, so an
  `int8_t` view of an `int16` block or a cast in a derived pointer scales
  correctly; messages switch from "element" to "byte" when the types differ.
  An unknown pointee (`v16int8`, a struct) is a skip, never a guess.
- **Accesses.** `p[e]`, `aie::load_v<N>(p + e)`, `aie::load_unaligned_v<N>(p + e, ..)`,
  `aie::store_v(p + e, v)` (N from `extract<N>` or an `aie::vector<T, N>`).
- **Exact, not intervals.** Every loop variable the index needs (and
  `program_id`) is *enumerated*. Interval arithmetic gives `owB = pass * 16 +
  (pass ? 4 : 8)` the range [4, 24] and reports a false overflow; the real
  values are {8, 20}. An index that cannot be evaluated is skipped and counted
  (`note: static bounds analysis: 11 block access(es) checked`). Each skip is
  a note with its reason: `` `kk` is reassigned at line 67 ``, `` `helper()` is a
  function call ``, a loop not of the `for (v = a; v < b; v += s)` form, the
  4M-iteration budget, an unknown pointee size. A block pointer used where no
  modelled access accounts for it (`use(w)`, `&w`, `*(w + e)`) is reported as
  `used in a form the analysis does not model`. On `simpleconv2d_new_load.cc`
  2 of 3 accesses are checked; the third indexes by the running counter `kk`.
- **Attribution.** The worst element is decomposed into block coordinates
  (`element 3135 is (15,6,6,3), dim 0 (output channels) index 15 is past its
  extent 15`), the loops that move it are named (`nb < 2` (line 23), `n < 8`
  (line 25)), and each literal constant is varied alone. A constant is a
  **cause** only if it sizes the block while the overflowing index (and its
  loops) never read it, *and* changing it makes every access fit. A constant
  read by both (`ow_t`: block width and row pitch) would only add slack, so it
  is not reported. When nothing qualifies, no cause line is printed; the
  finding points at the index arithmetic.
- **Cross-check with the trace.** A trace fault at line L with no static
  finding at L is classified, not left bare: `static analysis: cannot explain
  line L: <reason>` (it skipped the access), `no block access recognized at
  line L`, or `static analysis DISAGREES: it evaluated ... at line L and found
  it inside its block`. The last one means the static model is wrong for this
  kernel, e.g. `w` moved through an alias (`const int8_t **pw = &w; *pw +=
  2940;`). It tests the analysis against the compiled code on every fault.
- **Merging.** A trace fault at line L gets the static finding for L appended.
  Static findings the trace never reached (`store_v` at line 74 in the
  `oc_t = 15` case) become errors too. Without a trace fault they are only
  notes (`possible out-of-block access`): a guard the model misses may keep the
  access from running.

### Limits

- Recognition covers the spatial-halo conv family the backend implements
  (OH over mesh rows, OC over mesh cols, OH x OW time tiles). Other block
  patterns are traced and checked but reported as not recognized.
- The kernel body must compile on x86 (scalar C++; AIE-API vector code needs
  the AIE API x86 simulation headers in the harness -- not wired yet).
- The schedule must not depend on data (checked).
- The input tensor is the DDR buffer as the kernel reads it (pre-padded);
  `load_opts{.pad}` from the proposal is not implemented.

### Hand-written GemmSpaces with block loads (2026-11-10)

The explicit form -- one `constexpr aie::GemmSpace` per port, `__global__(policy)`,
`aie::port<T, Space>` -- can now be combined with `aie::load_buffer` /
`store_buffer` in the body (`example/tileprogram/ccode/simpleconv2dkernel_space.cc`:
`simpleconv2dkernel_port.cc`'s body under hand-written `RowBC_halo` / `ColBC` /
`LtoR_Merge`). Before, this did not build: the pipeline rewrites every
`aie::load_buffer(P, ...)` to reference `aiehlc_desc_P` (`tilinglinalg_pipeline.cpp`,
`resolveBlockIoBuiltins`), and those descriptors were only emitted for kernels
whose GemmSpaces convautotile had derived itself.

`verifySpaceKernels` (`kernel_trace.cpp`) handles such a kernel. **The GemmSpaces
stay the authority and are not rewritten**:

1. each port's GemmSpace is parsed (`sourceGemmSpace`, `conv_desugar.cpp`; the
   designated-initializer parser of the `aie::Conv2d` path, plus the file's
   `constexpr int` names) with aiehlc's defaults for omitted fields (act / wgt
   Broadcast, layout Row, merge Default, im2col None, an omitted `.stride` is 0).
   It fails closed: `.m/.n/.k`, an unknown or positional field, or a value that
   is not a constant is an error. The tensor (shape, padsize, role) comes from
   d1..d4;
2. the body is traced exactly as for a kernel-only kernel and routed
   (`planRouting`);
3. **the routed plan, rendered through `routingSpaces`, must equal every
   hand-written GemmSpace** (`spaceDiff`, `trace_routing.cpp`), field by field:
   the map (layout, act / wgt, merge order, mesh groups), im2col and d1..d4 with
   their slice_tiling. `pad`, `pp_depth` and `l1_budget` are the user's
   scheduling choices and are not compared (the trace uses the smallest
   `pp_depth` for its held-blocks check);
4. on success the kernel gets `aiehlc_desc_<port>` from the traced fits and
   `aie::num_programs` is resolved; a recognized conv also feeds the pipeline's
   `checkConvTilingAgainstAnalysis`.

**Why compare forward instead of inverting the GemmSpace.** The GemmSpace ->
block mapping is ambiguous for inputs: `tile_round` marks a mesh split or an
on-core loop depending on the dim, and a row-broadcast `A[M, K]` with "d1 split
over 4 mesh rows, d2 walked in 4 rounds" renders to the same GemmSpace as "d1
walked in 4 rounds, d2 split over 4 mesh rows". An inverse would have to guess.
Comparing in GemmSpace space needs no guess: equal GemmSpaces give the pipeline
the same input as the kernel-only build of the same kernel, so the DMA delivers
exactly the traced blocks -- the same soundness argument the kernel-only path
rests on.

A conflict is a build error naming the field, from either side:

```
win_a (GemmSpace RowBC_halo) .d1.slice_tiling.stride = 13, the kernel's blocks give 14
win_c (GemmSpace LtoR_Merge) .d1.slice_tiling.tile_size = 7, the kernel's blocks give 14   <- kernel's oh_t changed
win_a (GemmSpace RowBC_halo) .policy.map.layout = Col, the kernel's blocks give Row
win_c block 0 on core (0,0): offset has 3 and shape 3 dims, but the tensor has 2           <- GemmSpace lost a dim
```

Verified: `simpleconv2dkernel_space.cc` builds with `kernel.cc`, `routing.cc`,
`aieml.bcf/.prx` and the AIE kernel ELF **byte-identical** to
`simpleconv2dkernel_port.cc`'s build, and `host.cc` identical outside the copied
user source; a conflicting GemmSpace stops aiehlc with rc 1 before any ELF.
Unit tests: `unitest/test_trace_routing.cpp` (parse / defaults / errors, spaceDiff,
render -> parse round trip for conv, GEMM and eltwise plans, descriptors equal to
the kernel-only build), `unitest/kernel_trace_cases.py` `SPACE_CASES` (10).

**Open, not caused by this:** on the board (palmyra 213) the example's output
does **not** match `verify_conv2d` -- 128+ mismatches starting in the output
channels of the last two mesh columns (first at f=32 or f=48, varying run to
run). `simpleconv2dkernel_port.cc` gives the same failure (its `main()` never
called `verify_conv2d`, so it went unnoticed); the two builds run identical
kernel and DMA code. `simpleconv2dkernel_space.cc` calls the check and returns
non-zero, so the failure stays visible.

## 6. Tensor form: typed tensors, `.pad` / `.reuse`, block views, autotune

`example/tileprogram/ccode/simpleconv2dkernel.cc` is the full proposal: the
kernel works on the REAL tensors (input `[224, 224, 3]`, no manual border, 3
channels) and states padding, reuse and tile candidates itself:

```cpp
using Ifm = aie::tensor<int8_t, 224, 224, 3>;
template <int OH_T = 7, int OW_T = 28, int OC_T = 16>
__global__ aie::autotune<OH_T, {7, 14, 28}, OW_T, {14, 28, 56}, OC_T, {16, 32}>
void conv2d(aie::in<Ifm> x_t, aie::in<Wgt> w_t, aie::out<Ofm> y_t) {
    auto w = aie::load_buffer(w_t, {pc * OC_T, 0, 0, 0}, {OC_T, KH, KW, C}, {.reuse = ...});
    for (i ...) for (j ...) {
        auto x = aie::load_buffer(x_t, {oh0 * S - P, ow0 * S - P, 0}, {IH_T, IW_T, C},
                                  {.pad = aie::pad_value::zero});
        auto y = aie::store_buffer(y_t, {oh0, ow0, pc * OC_T}, {OH_T, OW_T, OC_T});
        ... y(oh, ow, oc) = aie::saturate<int8_t>(acc of x(..) * w(..));
    }   // x released, y committed here
}
... auto in = aie::alloc<Ifm>(); conv2d<<<mesh>>>(in, wt, out);
```

`lowerTensorForm` (`src/llvm/convautotile/kernel_front.{h,cpp}`) runs first in
`applyConvAutoTiling` and lowers it to the §5 kernel-only form:

1. **Pass A, kernel coordinates** (`traceLogical`): the body (template values
   substituted, `Ofm::dim(k)` / `aie::num_programs` folded, `auto v =
   load_buffer(...)` turned into a block view `v(i, j, c)`, release / commit
   inserted at the end of the view's scope) is traced like §5. Blocks may leave
   the tensor only with `{.pad = aie::pad_value::zero}`; `{.reuse = N}` must
   equal the output blocks stored while the block is held.
2. **Staged layout**: how far the `.pad` loads reach outside the tensor is the
   zero border (symmetric when the far side needs no more: 224 → 230); a
   channel dim that is not 4-byte aligned is padded (3 → 4), which requires
   whole-channel blocks.
3. **Pass B, staged layout**: offsets shift by the border, the channel extent
   becomes the aligned one, ports get `aie::shape<staged>`, and §5's
   `inferKernelOnly` derives the DMA and checks every block against it.
4. **Autotune**: every `aie::autotune` combination (or the explicit
   `conv2d<7, 28, 16><<<mesh>>>` arguments) goes through 1-3 in parallel; the
   legal one with the lowest convautotile cost estimate is kept, and the report
   says why each other one was rejected (e.g. `OC_T=32`: `w_t block covers
   [64, 96) outside the tensor extent 64`).
5. **Host**: `aie::tensor` / `aie::alloc` (a one-line prelude, so line numbers
   stay), and each launch stages its inputs (`aiehlc_stage`: zero border +
   aligned channels) and passes the GEMM `M, N, K`. `auto a = aie::alloc<X>(),
   b = aie::alloc<Y>();` (two types in one `auto`, ill-formed C++) is split.

Kernel locals named like the desugar's `#define`s (`OH_T`, `OW_T`, `K`, ...)
are renamed (`OH_T_`). Source lines are preserved, so ASan / compile errors
name the user's line.

**Verified.** Autotune picks `OH_T=14, OW_T=14` (est. 32.78 ms vs 32.81 ms for
7x28); `main.elf` builds, and `simpleconv2ddebug/emu_conv2d_tensor.sh` (host
staging + windows served from the generated `aiehlc_desc_<port>` + the
example's golden check on the real tensors) passes 802,816/802,816 with 0 block
errors. Pinned to `conv2d<7, 28, 16>`, `routing.cc`, bcf, prx, 27/28 IR dumps
and the generated `host_canonicalized` are identical to the hand-written
`simpleconv2d.cc` build (the 28th differs only in the kernel name).

**Diagnostics** (`kernel_trace_cases.py`, `TENSOR_CASES`): border loaded
without `.pad`, wrong `.reuse`, load too big, data used without load / after
release, part of the channels, weights outside the tensor, output not fully
written, loops swapped.

Limits: one tensor-form kernel per source; roles are inferred (output → OFM,
rank-4 input → WEIGHTS, rank-3 → IFM); `aie::pad_value::zero` only; channel
padding of an output is rejected.

## 7. Vector kernels and the AIE API x86 emulation

`example/tileprogram/ccode/simpleconv2dkernel_port.cc` computes with
`aie::mmul<8, 8, 8, int8_t, int8_t>` instead of scalar loops. Its block-load
contract (`load_buffer` / `store_buffer` blocks, loop order) is unchanged, so
the derived DMA, `routing.cc`, bcf, prx, `host.cc` and all 28 IR dumps are
identical to the scalar build; only the kernel code differs.

**Mapping the conv onto mmul.** C [8 output cols x 8 channels] += A [8 cols x
8 K] * B [8 K x 8 channels]. Stride 2 x 4 channels makes the 7 taps x 4
channels of output column `ow` in kernel row `kh` the 28 contiguous slab bytes
at `(oh*S + kh)*244 + ow*8`, so an A tile is ONE unaligned 64-byte load (no
im2col copy). K per kernel row = 4 blocks of 8: bytes 0-7, 8-15, 16-23 and
20-27 -- the last block starts 4 bytes early so no load runs past the
244-byte row (or the block); its first 4 B rows are zero. Weights are packed
once per launch into `Bp[kh][kb][nb]` B tiles and the weight window is
released right away. Output columns per pass: M-blocks {0, 8} and {16, 20}
(20-23 written twice with identical values) x 2 N-blocks of 8 channels, so
every B load feeds two MACs; the two N-blocks are interleaved with
`interleave_zip` into 128 contiguous output bytes and stored as 32-byte
vectors (DMA windows are only 32-byte aligned). `saturation_mode::saturate`
makes the int8 narrowing equal to the scalar clamp (bit-exact: |sum| <= 588).

**Measured / scheduled.** The kh loop pipelines at one `VMAC` per bundle
(kernel.lst: 15 bundles, 15 VMAC). Scalar: 89 bundles per 28 MACs, ~650
cycles per output, ~32.6 M cycles per core per launch.

**Tracing vector code** (`src/mlir/runtime/aie_api_x86emu.h`). The kernel
analysis compiles the body with g++ under ASan, where the real AIE API has no
implementation (its native mode needs chess). The header emulates the subset
the kernels use per element -- `aie::vector`, `aie::accum<acc32|acc64>`,
`aie::mmul` (row-major A/B/C, acc32 wraps), `load_v` / `load_unaligned_v` /
`store_v`, `zeros`, `concat`, `shuffle_down_fill`, `interleave_zip`,
`bit_and`, `add`, and the SRS rounding / saturation modes -- so:

- ASan sees every byte a vector load touches: a 64-byte load that runs past
  the block is "data used without being loaded" at its source line;
- `load_v` / `store_v` need the vector-size alignment (capped at 64) and
  `load_unaligned_v` its `aligned_elems`; the harness allocates blocks 32-byte
  but not 64-byte aligned (like DMA windows on the core, ASan-poisoned on both
  sides), so `aie::load_v<64>` on a window is reported with the line and the
  fix;
- anything outside the subset is a compile error in the harness -- extend
  the header. The accumulator tags are global (`acc32`), as in the AIE API,
  so the emulation rejects `aie::acc32` just like xchesscc.

The same header drives `simpleconv2ddebug/emu_conv2d_spatial.sh` (the
generated kernel on x86, windows 32-byte aligned): the vector build passes
802,816/802,816 with 0 block errors. `kernel_trace_cases.py` adds two vector
cases (unshifted last K-block, aligned 64-byte load on a window).
