# Plan: Honor the declared tensor dim order in the output DMA descriptor (no HWC/CHW hardcode)

> **For the executing agent:** This is a self-contained spec. You have NO prior conversation
> context. Read the "Background" and "Root Cause" sections first, then execute "Changes" in
> order, then "Build & Verify". All line numbers are from the current tree and may drift a few
> lines — search for the quoted code if so. Do NOT touch anything outside the listed files.

## Background

AIEHLC compiles C++ spatial ops onto AMD AIE tiles. A conv2d example
(`example/tileprogram/ccode/simpleconv2d.cc`) declares its output as an `aie::GemmSpace`
named `LtoR_Merge` with tensor dims `d1=H, d2=W, d3=C` (declared order `[H,W,C]`, i.e. HWC,
channel innermost/contiguous). The multi-tile pipeline (`tilinglinalg`) lowers this through
6 dialects into a host DMA descriptor.

**Bug:** the generated output gather DMA descriptor is **CHW** (channel OUTERMOST), not the
declared **HWC**:
```
num_dims=3
d0: stride=4,     wrap=7      # W contiguous
d1: stride=112,   wrap=7      # H rows
d2: stride=12544, wrap=16     # C planes  ← channel outermost = CHW, WRONG
```
The declared `[d1,d2,d3]` order must become the physical DDR layout, **computed generically**
(row-major, last declared dim contiguous), with the DMA-dimension reversal (tensor `d1,d2,d3`
→ DMA `d2,d1,d0`; innermost tensor dim = DMA `d0` = contiguous). **No `HWC`/`CHW`/`NCHW`
literal may drive the builder.** Re-declaring `[C,H,W]` must yield CHW automatically.

Note: `policy.map.layout` (Row/Col) is NOT a data layout — it is parsed into `pti.distribution`
(mesh axis owner). Do not overload it.

## Root Cause (confirmed)

`buildOutputTileDescriptor` in
`src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_common.cpp`
(rank-3 conv-output branch, ~lines 426-519) **hardcodes NCHW**. Current code:
```cpp
int64_t owT_w = owT / elemsPerWord;
desc.bdLenBytes = ohPerRow * owT * cPerCore * ooElementSizeBytes;
// BD scatter dims, innermost-first (NCHW [C,H,W], W contiguous):
desc.bdDims.push_back({wordBytes, owT_w});                          // D0: W words
desc.bdDims.push_back({OUTPUT_W * elemBytes, ohPerRow});            // D1: H rows
desc.bdDims.push_back({OUTPUT_H * OUTPUT_W * elemBytes, cPerCore}); // D2: C planes (outer)
desc.iterStep = static_cast<int32_t>(owT * elemBytes);
desc.iterWrap = static_cast<int32_t>(wRounds);
desc.totalRounds = hChunks;
desc.roundDims.push_back({hChunks, ohPerRow * OUTPUT_W * elemBytes});
desc.perTileStrideBytes = cPerCore * OUTPUT_H * OUTPUT_W * elemBytes;
```
This reproduces the observed CHW numbers exactly and ignores the declared order.

The authoritative `#routing.tiling` attr (built in `routingmanager.cpp:1085-1130`, attached to
the partition op, read at `flowtransfer_host.cpp:359` via `c.partExtractSlice->getAttr("tiling")`)
carries the declared order as `dims[0..2]` = d1,d2,d3, each a `#routing.dim<outer =
#routing.level<base,total,slice,step,rounds[,slice_tiling=...]>>`. The three output dims are:
- **H** = `dims[0]`: mesh-ROW split (group1). `outer{base=112,slice=rowSlice,rounds=meshRows}`
  with nested `slice_tiling{slice=ohPerRow=7, rounds=hChunks=4}` (on-core H sub-rounds).
- **W** = `dims[1]`: on-core width chunk. `outer{base=112,slice=owT=28,rounds=wRounds=4}`, no slice_tiling.
- **C** = `dims[2]`: mesh-COL split (group2). `outer{base=64,slice=cPerCore=16,rounds=meshCols=4}`, no slice_tiling.

**Ambiguity:** W (`rounds=4`) and C (`rounds=4`) are indistinguishable by value alone, so the
role of each dim (iter vs per-tile) cannot be inferred from the current attr — the current code
only "knows" via hardcoded positions. **Fix: carry the axis role on each `DimAttr`.**

## Design: generic row-major descriptor

Add an `axis` role marker to each declared dim (`0`=on-core, `1`=mesh_row/group1,
`2`=mesh_col/group2). Then in `buildOutputTileDescriptor`, for declared dims `d[0..n-1]`
(= `tiling.getDims()`, already in d1,d2,d3 order):
- `F[k]      = outer.getBase()`                              (full extent, elements)
- `rmStride[n-1]=1; rmStride[k]=rmStride[k+1]*F[k+1]`        (row-major element strides)
- per dim: if `outer.getSliceTiling()` present → `blk[k]=st.getSlice(); rounds[k]=st.getRounds()`
           else → `blk[k]=outer.getSlice(); rounds[k]=outer.getRounds()`
- roles: `rowDim = (axis==1)`, `colDim = (axis==2)`, `iterDim = (axis==0)`
- **BD dims innermost-first** (`k = n-1 … 0`):
  - `k==n-1` (contiguous): `{wordBytes, blk[k]/elemsPerWord}`
  - else: `{rmStride[k]*elemBytes, blk[k]}`
- `bdLenBytes         = product(blk[k]) * ooElementSizeBytes`
- `iterStep           = blk[iterDim]*rmStride[iterDim]*elemBytes; iterWrap = outer[iterDim].getRounds()`
- `totalRounds        = rounds[rowDim]`
- `roundDims          = {rounds[rowDim], blk[rowDim]*rmStride[rowDim]*elemBytes}`
- `perTileStrideBytes = blk[colDim]*rmStride[colDim]*elemBytes`

**Worked check — declared `[H,W,C]=[112,112,64]`, int8 (elemBytes=1, elemsPerWord=4),
axis H=1,W=0,C=2, blk H=7,W=28,C=16:**
`rmStride: H=7168, W=64, C=1` → BD reversed:
`D0{4, 16/4=4}(C)  D1{64, 28}(W)  D2{7168, 7}(H)`;
`iterStep=28*64=1792, iterWrap=4`; `totalRounds=4, roundDims={4, 7*7168=50176}`;
`perTileStrideBytes=16*1=16`. → **HWC, channel now innermost/contiguous.** ✓
Re-declaring `[C,H,W]` recomputes `rmStride` and reverses BD dims → CHW, with roles still
following axis (C=col, H=row, W=iter) — zero code change.

## Changes

### 1. Add `axis` field to `DimAttr` (IR-local role carrier)

**1a.** `src/mlir/mlirfront/tilinglinalg/routing/td/routingattr.td`, `DimAttr` (lines ~100-108):
add a defaulted `axis` parameter after `outer`:
```tablegen
def DimAttr : routingattr<"Dim"> {
  let mnemonic = "dim";
  let summary = "per-dimension tiling descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "routing::LevelAttr":$outer,
    DefaultValuedParameter<"int64_t", "0">:$axis   // 0=on-core, 1=mesh_row(group1), 2=mesh_col(group2)
  );
  let hasCustomAssemblyFormat = 1;
}
```
The default keeps existing 2-arg `DimAttr::get(ctx, outer)` callers valid (TableGen emits a
default-valued builder). If the build reports the 2-arg overload is missing, add an explicit
`, 0` at the plain call sites (see list in "DimAttr::get call sites" below).

**1b.** Regenerate the dialect `.inc` files: run the routing dialect's TableGen generator
(`src/mlir/mlirfront/tilinglinalg/routing/gen.sh`). Confirm the generated `getAxis()` accessor
exists. gen.sh usage:
`bash src/mlir/mlirfront/tilinglinalg/routing/gen.sh --mlir-include <MLIR_INC> --llvm-bin <LLVM_BIN>`
where `<LLVM_BIN>` contains `mlir-tblgen`. If mlir-tblgen cannot be located, the generated
`inc/routingattr.h.inc` + `inc/routingattr.cc.inc` for `DimAttr` must be hand-updated to add the
`axis` storage field, the `getAxis()` accessor, and the 2-arg + 3-arg `get()` builders.

**1c.** `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp`, `DimAttr` custom
print/parse (lines ~222-233): print `axis` only when non-zero (keeps existing IR text stable),
parse it as optional:
```cpp
void routing::DimAttr::print(mlir::AsmPrinter &printer) const {
    printer << "<outer = " << getOuter();
    if (getAxis() != 0) printer << ", axis = " << getAxis();
    printer << ">";
}
mlir::Attribute routing::DimAttr::parse(mlir::AsmParser &parser, mlir::Type) {
    routing::LevelAttr outer;
    if (parser.parseLess() || parser.parseKeyword("outer") || parser.parseEqual() ||
        parser.parseAttribute(outer))
        return {};
    int64_t axis = 0;
    if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("axis") || parser.parseEqual() || parser.parseInteger(axis))
            return {};
    }
    if (parser.parseGreater())
        return {};
    return routing::DimAttr::get(parser.getContext(), outer, axis);
}
```

**1d.** `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp`, the 3D conv-output
group2 tiling builder (lines ~1106-1129): set the axis on each output `DimAttr`:
```cpp
auto hDim = routing::DimAttr::get(ctx2, hOuter, /*axis=mesh_row*/1);
...
auto wDim = routing::DimAttr::get(ctx2, wOuter, /*axis=on-core*/0);
...
auto cDim = routing::DimAttr::get(ctx2, cOuter, /*axis=mesh_col*/2);
```
Leave ALL other `DimAttr::get` sites at default axis 0 (fallback path unaffected).

### 2. Rewrite the rank-3 branch generically

`src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_common.cpp`,
the `if (halo.valid && memrefType.getRank() == 3) { ... }` branch (~lines 426-520).

When the tiling attr is present with 3 dims, replace the hardcoded NCHW BD emission
(~lines 495-519) with the generic algorithm from "Design" (read `dims[k].getOuter()` →
`F/blk/rounds`, read `dims[k].getAxis()` → roles, build `rmStride`, emit BD dims reversed, set
iter/round/perTile). Keep the existing tiling-attr reads (~446-461) if convenient, or replace
them with the array-based reads.

Concrete implementation sketch (adapt names to the existing `desc`/`elemBytes`/`wordBytes`/
`elemsPerWord` already in scope):
```cpp
if (tiling && tiling.getDims().size() == 3) {
    auto dims = tiling.getDims();
    constexpr int n = 3;
    routing::LevelAttr outer[n];
    int64_t F[n], blk[n], rounds[n], axis[n];
    for (int k = 0; k < n; ++k) {
        outer[k] = dims[k].getOuter();
        F[k]     = outer[k].getBase();
        axis[k]  = dims[k].getAxis();
        if (routing::LevelAttr st = outer[k].getSliceTiling()) { blk[k]=st.getSlice(); rounds[k]=st.getRounds(); }
        else { blk[k]=outer[k].getSlice(); rounds[k]=outer[k].getRounds(); }
    }
    int64_t rmStride[n]; rmStride[n-1]=1;
    for (int k=n-2;k>=0;--k) rmStride[k]=rmStride[k+1]*F[k+1];

    int rowDim=-1,colDim=-1,iterDim=-1;
    for (int k=0;k<n;++k){ if(axis[k]==1)rowDim=k; else if(axis[k]==2)colDim=k; else iterDim=k; }
    if (rowDim<0||colDim<0||iterDim<0){ rowDim=0; iterDim=1; colDim=2; } // fallback: old positional

    // word-alignment guard on the contiguous innermost dim
    assert(blk[n-1] % elemsPerWord == 0 && "innermost per-fire block must be word-aligned");

    int64_t prod=1; for(int k=0;k<n;++k) prod*=blk[k];
    desc.bdLenBytes = prod * ooElementSizeBytes;

    for (int k=n-1;k>=0;--k) {
        if (k==n-1) desc.bdDims.push_back({wordBytes, blk[k]/elemsPerWord});          // contiguous
        else        desc.bdDims.push_back({rmStride[k]*elemBytes, blk[k]});
    }
    desc.iterStep    = static_cast<int32_t>(blk[iterDim]*rmStride[iterDim]*elemBytes);
    desc.iterWrap    = static_cast<int32_t>(outer[iterDim].getRounds());
    desc.totalRounds = rounds[rowDim];
    desc.roundDims.push_back({rounds[rowDim], blk[rowDim]*rmStride[rowDim]*elemBytes});
    desc.perTileStrideBytes = blk[colDim]*rmStride[colDim]*elemBytes;
    return desc;
}
// else (no tiling attr): keep the EXISTING attr-less fallback block unchanged.
```
Update the block comment (~lines 427-435) from "NCHW DDR layout" to "row-major of the declared
dim order". Leave the 2D-halo branch (~521-568) and the GEMM branch (~570+) unchanged.

### 3. Make the example USER code match the declared order (HWC)

The kernel write order and host reference are user code and must agree with the declared
`[H,W,C]`. Flip both to HWC:

**3a.** `example/tileprogram/ccode/simpleconv2d.cc` line ~366 (+ comment ~363-365). Current
(CHW, channel `j` outermost):
```cpp
local_out[(j * oh_per_row + oh) * ow_dim + ow] = (int8_t)sum;
```
New (HWC, channel innermost — Phase 3 streams `local_out` flat so MM2S order = HWC):
```cpp
local_out[(oh * ow_dim + ow) * out_c_num + j] = (int8_t)sum;
```
(`out_c_num` is the per-core channel count already computed in the kernel; confirm the exact
variable name in scope near line 366.)

**3b.** `example/tileprogram/ccode/simpleconv2d.h`:
- `scalar_conv2d` line ~132: `output[(oh * OUTPUT_W + ow) * NUM_FILTERS + f] = (int8_t)acc;`
- verify comment ~142-143 → "HWC layout [OH, OW, F]".
- mismatch decode ~171-174: `oh=i/(OUTPUT_W*NUM_FILTERS); ow=(i/NUM_FILTERS)%OUTPUT_W; f=i%NUM_FILTERS;`
- print index ~208/218/228 and commented ref print ~235-244:
  `output_aie[(oh*OUTPUT_W+ow)*NUM_FILTERS + f]`.

**3c.** (Optional, low priority) `example/tileprogram/ccode/simpleconv2ddebug/conv2d_debug.py`
full-output print indexing → HWC. Not fed to host verify.

## DimAttr::get call sites (for schema-change review)
All currently 2-arg `DimAttr::get(ctx, outer)`; only the 3 output-tiling sites get an explicit axis:
- `routingmanager.cpp:232` (parser — updated in 1c to pass `axis`)
- `routingmanager.cpp:902, 1035, 1066, 1169, 1179` — leave default 0
- `routingmanager.cpp:1110 (hDim), 1119 (wDim), 1128 (cDim)` — set axis 1/0/2 (change 1d)
- `pass/unitest/test.cpp:2149,2150,2153,2154` — leave default 0
If TableGen does not emit a 2-arg default builder, append `, 0` to the "leave default" sites.

## Critical Files
- `src/mlir/mlirfront/tilinglinalg/routing/td/routingattr.td` (+ dialect `gen.sh`) — DimAttr `axis`
- `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp` — print/parse + output tiling axis
- `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_common.cpp` — generic builder (real fix)
- `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_host.cpp` — consumes desc; NO change expected (verify only)
- `example/tileprogram/ccode/simpleconv2d.cc`, `simpleconv2d.h` — user code → HWC

## Build & Verify
1. Rebuild / regenerate + run the pipeline:
   `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc`
   (If the `.inc` were regenerated, ensure the standalone unitest/aiehlc build picks them up.)
2. Inspect the generated `host.cc` output shim BD / `__Runtime_dma_bd_config`: expect
   `d0 stride=4 wrap=4`, `d1 stride=64 wrap=28`, `d2 stride=7168 wrap=7`, per-tile offset step 16
   (channel now innermost/contiguous = HWC). Also check the `[OOO ShimBD desc]` stderr line.
3. HW run: `python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1`; expect
   `device_teardown done` and `verify_conv2d` PASS (0 mismatches) — proves descriptor + kernel
   + host ref all agree on the declared HWC order.
4. Genericity check (optional): swap `LtoR_Merge` d1/d2/d3 to `[C,H,W]` and flip the two user
   files to CHW; the SAME builder must emit a CHW descriptor with no code change.
5. On mismatch, use `.claude/commands/data-mismatch-debug.md` for a supply/demand trace.

## Ordered task checklist
1. [ ] Add `axis` param to `DimAttr` in `routingattr.td`; regenerate `.inc`; confirm `getAxis()`.
2. [ ] Update `DimAttr::print`/`parse` (optional `axis`).
3. [ ] Set axis 1/0/2 on hDim/wDim/cDim in the group2 output tiling builder.
4. [ ] Fix any `DimAttr::get` call sites if the default builder is unavailable.
5. [ ] Rewrite the rank-3 branch of `buildOutputTileDescriptor` (generic row-major).
6. [ ] Update the block comment (NCHW → row-major of declared order).
7. [ ] Flip `simpleconv2d.cc` kernel write to HWC.
8. [ ] Flip `simpleconv2d.h` ref/verify/print to HWC.
9. [ ] Build, inspect host.cc BD, HW-run, confirm verify PASS.
10. [ ] (Optional) debug replay HWC; (optional) genericity `[C,H,W]` swap.

## Notes / Risks
- The builder change is the durable fix; the two example files are user-side and only need to
  match the declared order. **All three must agree** — a partial flip = byte-scrambled output.
- Role detection is deterministic from `DimAttr.axis` on the tiling op (no module attr, no
  W-vs-C value ambiguity). Fallback (all axis==0) preserves the old positional behavior.
- Innermost declared dim's per-fire block must be divisible by `elemsPerWord` (int8 C=16 → OK).
- `flowtransfer_common.cpp` other branches (2D-halo, GEMM) are out of scope — do not modify.
- Keep any function under ~200 lines (project rule); factor helpers if the branch grows.

## Environment note (discovered while starting execution)
- Every bash invocation in this shell sources a profile that prints Vitis/PetaLinux env banners;
  ignore that noise.
- The routing dialect `gen.sh` lives at `src/mlir/mlirfront/tilinglinalg/routing/gen.sh` (NOT in
  `td/`). It needs `--mlir-include <path>` and `--llvm-bin <dir with mlir-tblgen>`. Locate
  mlir-tblgen (the same LLVM build used to build aiehlc, `-DLLVM_INSTALL_DIR`) before regenerating.
