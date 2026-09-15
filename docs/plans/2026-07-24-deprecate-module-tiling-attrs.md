# Deprecate Global Module Tiling Attributes — Single-Source `#routing.tiling` Op Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `#routing.tiling` attribute on the `routing.partitiontensor` op the single source of truth for all GEMM/conv tiling geometry, and delete the nine global `routing.tile_*/*_rounds/*_k` module attributes (both their emission in `aiehlc.cc` and every `getAttrOfType` read across the passes).

**Architecture:** Every pass that needs tiling scalars reads them once at pass entry from the live `partitiontensor` tiling op via a single canonical reader (`readGemmTilingScalars`, extended to a full `GemmTilingScalars` struct covering GEMM *and* conv/halo). Helpers stop reading `moduleOp` attrs and instead receive the resolved struct. Once all consumers are converted, the frontend stops emitting the module attrs entirely.

**Tech Stack:** C++17, MLIR (custom `routing` dialect), Clang AST frontend (`aiehlc.cc`), CMake. Tests: per-pass unit test (`pass/unitest/test.cpp`), IR byte-compare of `ir/dfschedule/*.mlir`, and HW run via `script/test/apppaltest.py`.

---

## Current Status Snapshot (as of 2026-07-24)

### The nine module attributes being deprecated

| Module attr | Meaning | Tiling-op equivalent (per `partitiontensor` split-dim `LevelAttr`) |
|-------------|---------|--------------------------------------------------------------------|
| `routing.tile_rows` | mesh M slice per tile | row-owner `dims[splitdim].outer.slice` |
| `routing.tile_m` | on-core M slice | row-owner `dims[splitdim].outer.slice_tiling.slice` |
| `routing.m_rounds` | M rounds | row-owner `...slice_tiling.rounds` (= `tile_rows/tile_m`) |
| `routing.tile_cols` | mesh N slice per tile | col-owner `dims[splitdim].outer.slice` |
| `routing.tile_n` | on-core N slice | col-owner `dims[splitdim].outer.slice_tiling.slice` |
| `routing.n_rounds` | N rounds | col-owner `...slice_tiling.rounds` (= `tile_cols/tile_n`) |
| `routing.full_k` | full K extent | K-dim `outer.total` (dim with `outer.rounds==1` + nested) |
| `routing.effective_k` | on-core K slice | K-dim `outer.slice_tiling.slice` |
| `routing.k_rounds` | K rounds | K-dim `outer.slice_tiling.rounds` |

### The tiling-op / attr structure (already defined)

`src/mlir/mlirfront/tilinglinalg/routing/td/routingattr.td`:
- `LevelAttr` (`#routing.level`): `base,total,slice,step,rounds, slice_tiling(optional nested LevelAttr)`
- `DimAttr` (`#routing.dim`): one `outer` LevelAttr per tensor dim
- `TilingAttr` (`#routing.tiling`): `dims: [DimAttr...]`, attached to `partitiontensor` as `getTilingAttr()`
- `HaloAttr` (`#routing.halo`): conv/halo path descriptor (`sliceSize,step,overlap,l2*,k*`) — **separate** from the GEMM tiling op today.

### Emission (SOURCE — to be deleted in Phase 3)

- Module attrs emitted in `src/llvm/aiehlc.cc`:
  - Multi-kernel GEMM: `4208` (`effective_k`), `4216-4224` (`tile_m/m_rounds/tile_n/tile_cols/n_rounds`)
  - Single-kernel GEMM: `4849` (`effective_k`), `4857-4865` (same set)
  - Single-kernel halo M-round fallback (conv): `4903-4906` (`tile_m`, `m_rounds` only)
- Tiling op emitted via `buildGemmTilingAttr` (`routing/routingmanager.cpp:873`) gated `!isHalo && !isGroup2 && !gemmTiling.empty()` at `routingmanager.cpp:1182-1183` — **GEMM path only**.

### Canonical reader (partial — to be extended)

- `routing::readGemmTilingScalars(moduleOp)` — `routing/routingmanager.cpp:564`, struct `GemmTilingScalars` at `routingmanager.h:78`. Returns `tileM,tileRows,tileN,tileCols,effectiveK,fullK,kRounds,found`. **Gated on `routing.fullconnect_auto==1`** (bails for conv). Does NOT yet expose `mRounds/nRounds`, and does NOT read the conv/halo path.

### Consumers still reading the module attrs directly (`getAttrOfType("routing.tile_*")`)

**A. Host schedule — `BlueprintToSchedulePass`**
- `passblueprinttoschedule/passblueprinttoschedule.cpp:319-325` — reads into `passState` (fallback), then **already** overridden by `readGemmTilingScalars` at `:331-340` for GEMM. ✅ partly converted.
- `helper/flowtransfer_common.cpp:329-330,347-348` — `classifyTiling` (M+N mode/rounds). ❌ raw module attr.
- `helper/flowtransfer_common.cpp:589-591` — per-tile N sub-tiling. ✅ uses `passState`.
- `helper/flowtransfer_host.cpp:167-178, 499-525, 556-573, 719-756, 1160-1174` — shim-BD length / iter_step / iter_wrap for M+N+K sub-tiling. ❌ raw module attr (many sites).

**B. Kernel schedule — `BlueprintToScheduleKernelPass`** (NOT converted at all)
- `passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp:831-834, 863-871, 1004-1052, 1297-1298`. ❌ raw module attr.
- `helper/flowtransfer_kernel.cpp:161-188, 356`. ❌ raw module attr.

**C. Blueprint stage — `DmaphopTodfscheblueprintPass`**
- `passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:985-1012` — 3D K-tiling shim addressing (`effective_k/full_k/tile_m/tile_n`). ❌ raw module attr.

**D. Provenance JSON (report-only, no logic)**
- `passdmaphopprovenancemap/passdmaphopprovenancemap.cpp:910-918` — all 9.
- `passdfscheduleprovenancemap/passdfscheduleprovenancemap.cpp:361-373` — 7.

**E. Unit test**
- `pass/unitest/test.cpp:1070-1078` — `setAttr` all 9 to drive the pass under test.

### Op-lifetime constraint (critical)

The `partitiontensor` op (carrying `#routing.tiling`) is a `routing` op that survives through `DmaphopTodfscheblueprintPass` and `BlueprintToSchedule[Kernel]Pass` and is only erased during later conversion (`routinglower` / `passdfscheduletoapi`). Therefore each consuming pass MUST read the tiling op **once at pass entry (`runOnOperation` top, before `applyPartialConversion`)** into a struct, and helpers must consume that struct — they cannot re-walk `partitiontensor` from inside a conversion pattern reliably.

---

## Target Design

1. **One rich struct + one reader.** Extend `GemmTilingScalars` to carry every scalar any consumer needs (`tileM,tileRows,mRounds,tileN,tileCols,nRounds,effectiveK,fullK,kRounds,found`). Extend `readGemmTilingScalars` to:
   - populate `mRounds/nRounds` (from `slice_tiling.rounds`, fallback `tileRows/tileM`),
   - **also** handle the conv/halo path (see Phase 2) so it is no longer gated on `fullconnect_auto==1`.
2. **Read once, thread down.** Each pass calls the reader at entry into a local `GemmTilingScalars`; helper signatures change from `(ModuleOp moduleOp, ...)` to also take `(const routing::GemmTilingScalars &tiling, ...)`. All `getAttrOfType("routing.tile_*")` reads inside helpers are replaced by struct-field reads.
3. **Conv/halo carries a tiling op too (Phase 2).** Emit a `#routing.tiling` on the conv output/input `partitiontensor` (or teach the reader to synthesize the struct from `HaloAttr` + `tensor_N.halo` dict) so the conv path's `tile_m/m_rounds` come from IR, not module attrs.
4. **Delete emission + reads (Phase 3).** Remove all `module->setAttr("routing.tile_*")` in `aiehlc.cc` and every remaining `getAttrOfType("routing.tile_*")`. Provenance passes read from the reader struct instead. Unit test builds a `partitiontensor` with a tiling op instead of setting module attrs.

**Invariant to protect:** `simplematmul2.cc` (`fullconnect_auto=1`) and `simpleconv2d.cc` (`fullconnect_auto=0`) generated host/kernel `.cc` and `ir/*.mlir` must stay behaviorally identical (byte-identical where the attr is only reformatted away) after each phase.

---

## Pre-flight (do first, every session)

**Step 0.1: Build baseline**
```bash
cd build && cmake .. -DLLVM_INSTALL_DIR=$LLVM_INSTALL_DIR >/tmp/claude/cmake.log 2>&1 && make -j$(nproc) >/tmp/claude/build.log 2>&1; echo RC=$?
```
Expected: `RC=0`.

**Step 0.2: Capture golden IR + generated sources for both examples**
```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/gen_gemm.log 2>&1
cp -r ir /tmp/claude/golden_ir_gemm
cp -r aout/worklocal /tmp/claude/golden_gemm
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc >/tmp/claude/gen_conv.log 2>&1
cp -r ir /tmp/claude/golden_ir_conv
cp -r aout/worklocal /tmp/claude/golden_conv
```
These are the regression baselines. After each task, regenerate and `diff` against them: only the intended attr changes should differ.

---

## Phase 1 — GEMM path: make the tiling op the single source

### Task 1: Extend `GemmTilingScalars` + `readGemmTilingScalars`

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.h:78-95`
- Modify: `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp:564-622`
- Test: `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` (new case)

**Step 1.1:** Add `int64_t mRounds=0, nRounds=0;` to `struct GemmTilingScalars` (`routingmanager.h`).

**Step 1.2:** In `readGemmTilingScalars` (`routingmanager.cpp`), after computing `coreSlice`, also set `mRounds`/`nRounds` from the split-dim `onCore.getRounds()` (fallback `meshSlice/coreSlice` when `coreSlice>0`). Keep the `fullconnect_auto==1` gate for now (Phase 2 removes it).

**Step 1.3: Write a unit test** in `test.cpp` that builds a `partitiontensor` with a two-dim `#routing.tiling` (row-owner: outer slice=`tileRows`, nested slice=`tileM`,rounds=`mRounds`; K-dim: outer rounds=1,total=`fullK`, nested slice=`effectiveK`,rounds=`kRounds`) and asserts `readGemmTilingScalars` returns every field.

**Step 1.4: Build + run unit test**
```bash
cd build && make -j$(nproc) 2>&1 | tail -20; echo RC=$?
```
Expected: `RC=0` and the new assertions pass. (Run the unitest binary per `pass/unitest/` CMake.)

**Step 1.5: Commit**
```bash
git add src/mlir/mlirfront/tilinglinalg/routing/routingmanager.{h,cpp} src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp
git commit -m "refactor(routing): extend GemmTilingScalars with m/n rounds and unit test"
```

### Task 2: Convert `classifyTiling` to the struct

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_common.cpp:325-364`
- Modify: `helper/flowtransfer_internal.h` (signature)

**Step 2.1:** Change `classifyTiling(ModuleOp)` → `classifyTiling(const routing::GemmTilingScalars &t)`. Replace the four `getAttrOfType` reads (`tile_m,tile_rows,tile_n,tile_cols`) with `t.tileM,t.tileRows,t.tileN,t.tileCols`.

**Step 2.2:** Update the single call site in `BlueprintToSchedulePass` to pass the entry-read struct (already available as the `ir` in `passblueprinttoschedule.cpp:331`; hoist it into a pass-scope `GemmTilingScalars`).

**Step 2.3: Build + regenerate GEMM + diff**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/g.log 2>&1
diff -r /tmp/claude/golden_gemm/host.cc aout/worklocal/host.cc; echo DIFF_RC=$?
```
Expected: `RC=0`, `DIFF_RC=0` (host unchanged — value identical, just sourced differently).

**Step 2.4: Commit.**

### Task 3: Convert `flowtransfer_host.cpp` shim-BD sites

**Files:**
- Modify: `helper/flowtransfer_host.cpp:167-178, 499-525, 556-573, 719-756, 1160-1174`

**Step 3.1:** Thread `const routing::GemmTilingScalars &tiling` into the host emit helper(s). Replace every `moduleOp->getAttrOfType("routing.tile_*")` / `effective_k/full_k/k_rounds` read with the struct field. (`effectiveK/fullK/kRounds` already in struct.)

**Step 3.2: Build + regenerate GEMM + full diff**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/g.log 2>&1
diff -r /tmp/claude/golden_gemm aout/worklocal; echo DIFF_RC=$?
```
Expected: `RC=0`, `DIFF_RC=0`.

**Step 3.3: Commit.**

### Task 4: Convert `DmaphopTodfscheblueprintPass` K-tiling block

**Files:**
- Modify: `passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:985-1012` (+ read the struct once at `runOnOperation` entry)

**Step 4.1:** At pass entry read `routing::GemmTilingScalars t = routing::readGemmTilingScalars(moduleOp);`. Replace the four `getAttrOfType` reads in the K-tiling block with `t.effectiveK/t.fullK/t.tileM/t.tileN`.

**Step 4.2: Build + regenerate GEMM + diff `ir/dfschedule`**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/g.log 2>&1
diff -r /tmp/claude/golden_ir_gemm/dfschedule ir/dfschedule; echo DIFF_RC=$?
```
Expected: `RC=0`, `DIFF_RC=0`.

**Step 4.3: Commit.**

### Task 5: Convert the kernel path

**Files:**
- Modify: `passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp:831-1298` (read struct at entry)
- Modify: `helper/flowtransfer_kernel.cpp:161-356`

**Step 5.1:** At `BlueprintToScheduleKernelPass::runOnOperation` entry, read the struct once. Replace all `getAttrOfType("routing.tile_*")`/`k_rounds` reads (both in the pass and in `flowtransfer_kernel.cpp`, threading the struct in) with struct fields.

**Step 5.2: Build + regenerate GEMM + diff kernel `.cc`**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/g.log 2>&1
diff -r /tmp/claude/golden_gemm aout/worklocal; echo DIFF_RC=$?
```
Expected: `RC=0`, `DIFF_RC=0`.

**Step 5.3: Commit.**

### Task 6: GEMM checkpoint — HW verify

**Step 6.1:**
```bash
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```
**Step 6.2:** Read `./applog`; expect `device_teardown done`, no `AIE ERROR`, 0 mismatches for `simplematmul2`.

**Step 6.3: Commit** a checkpoint note if needed.

---

## Phase 2 — conv/halo path: give it a tiling-op source

> After Phase 1, only the conv/halo path (`fullconnect_auto=0`) still needs `routing.tile_m/m_rounds` from module attrs (emitted at `aiehlc.cc:4903-4906`, consumed via `classifyTiling` and the host round-loop). This phase removes that dependency.

### Task 7: Emit a `#routing.tiling` for the conv output partition

**Files:**
- Modify: `src/llvm/aiehlc.cc` (conv/halo emission near `4894-4920`) and/or `routing/routingmanager.cpp` (`buildGemmTilingAttr` gating at `1182`)

**Step 7.1:** Decide the carrier (see design doc `doc/design/fullconnect_auto_pass_tiling_reconstruction.md`): the halo INPUT `partitiontensor` gets `#routing.halo`; the **output** `win_c` `partitiontensor` currently gets empty `TilingAttr{}`. Emit a GEMM-shaped `#routing.tiling` on the **output** partition encoding row-dim `outer.slice=tile_rows`, `slice_tiling.slice=tile_m, rounds=m_rounds` (with `m_rounds = w_rounds × l2_rounds` from the `tensor_N.halo` dict, per the design doc's reconstruction).

**Step 7.2:** Extend `readGemmTilingScalars` to drop the `fullconnect_auto==1` gate: when absent, read the conv output partition's tiling op (row dim → `tileM/tileRows/mRounds`). Keep GEMM behavior identical when `fullconnect_auto==1`.

**Step 7.3: Build + regenerate conv + diff**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc >/tmp/claude/c.log 2>&1
diff -r /tmp/claude/golden_conv aout/worklocal; echo DIFF_RC=$?
```
Expected: `RC=0`; `DIFF_RC=0` (values identical, still sourced from module attrs which are still emitted at this point).

**Step 7.4: Commit.**

### Task 8: Point conv consumers at the struct

**Step 8.1:** With Phase-1 helpers already struct-based, confirm the conv path now flows through the struct (`classifyTiling`, host round loop). Add targeted logging (`[TilingLinalg] conv tiling from op: tile_m=… m_rounds=…`) to confirm the value comes from the op, not the attr, during conv regen.

**Step 8.2: Regenerate conv, confirm the log line + `DIFF_RC=0`. Commit.**

---

## Phase 3 — delete emission + remaining reads

### Task 9: Convert provenance passes to the struct

**Files:**
- Modify: `passdmaphopprovenancemap/passdmaphopprovenancemap.cpp:910-918`
- Modify: `passdfscheduleprovenancemap/passdfscheduleprovenancemap.cpp:361-373`

**Step 9.1:** Replace the `getAttrOfType("routing.tile_*")` reads with `readGemmTilingScalars` struct fields (JSON keys unchanged → provenance JSON byte-identical).

**Step 9.2: Regenerate both examples; diff `*provenancemap.json`. Expected identical. Commit.**

### Task 10: Delete module-attr emission in the frontend

**Files:**
- Modify: `src/llvm/aiehlc.cc:4208-4224, 4849-4865, 4903-4906`

**Step 10.1:** Delete every `module->setAttr("routing.tile_m"/"tile_rows"/"tile_n"/"tile_cols"/"m_rounds"/"n_rounds"/"effective_k"/"full_k"/"k_rounds")`. Leave `routing.fullconnect_auto` and unrelated attrs intact.

**Step 10.2: Build + regenerate BOTH examples + full diff**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc >/tmp/claude/g.log 2>&1
diff -r /tmp/claude/golden_gemm aout/worklocal; echo GEMM_DIFF=$?
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc >/tmp/claude/c.log 2>&1
diff -r /tmp/claude/golden_conv aout/worklocal; echo CONV_DIFF=$?
```
Expected: `RC=0`; the only diffs are the `routing.tile_*` lines disappearing from `ir/*/0_initial.mlir`; generated `host.cc`/kernel `.cc` unchanged (`GEMM_DIFF`/`CONV_DIFF` limited to IR headers).

**Step 10.3: Commit.**

### Task 11: Delete remaining reads + fix the unit test

**Files:**
- Modify: `pass/unitest/test.cpp:1070-1078` (build a `partitiontensor`+tiling op instead of `setAttr` the 9 module attrs)
- Grep-sweep: remove any residual `getAttrOfType("routing.tile_*")` and the now-dead fallback block in `passblueprinttoschedule.cpp:319-325`.

**Step 11.1:** Verify none remain:
```
Grep pattern: getAttrOfType.*routing\.(tile_m|tile_rows|tile_n|tile_cols|m_rounds|n_rounds|effective_k|full_k|k_rounds)
Expected: no matches (except comments).
```
**Step 11.2:** Update `test.cpp` to construct the tiling op; run unitest.

**Step 11.3: Build + both examples + HW run**
```bash
cd build && make -j$(nproc) 2>&1 | tail -5; echo RC=$?
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```
Expected: `RC=0`, `applog` shows `device_teardown done`, 0 mismatches.

**Step 11.4: Commit.**

### Task 12: Docs

**Files:**
- Modify: `doc/design/fullconnect_auto_pass_tiling_reconstruction.md` — mark the module-attr backfill obsolete; note the tiling op is now the single source.
- Modify: `CLAUDE.md` "Additional Documentation" — add a one-line pointer to this plan.

**Step 12.1: Commit.**

---

## Verification Matrix

| Check | Command | Pass criteria |
|-------|---------|---------------|
| Build | `cd build && make -j$(nproc)` | RC 0 |
| No attr reads left | Grep `getAttrOfType.*routing\.(tile_\|.*_rounds\|.*_k)` | 0 matches |
| No attr emission left | Grep `setAttr.*routing\.(tile_\|.*_rounds\|effective_k\|full_k\|k_rounds)` | 0 matches |
| GEMM host/kernel unchanged | regen `simplematmul2.cc` + `diff -r golden_gemm` | only IR-header attr lines differ |
| Conv host/kernel unchanged | regen `simpleconv2d.cc` + `diff -r golden_conv` | only IR-header attr lines differ |
| Provenance JSON unchanged | `diff *provenancemap.json` | identical |
| GEMM HW | `apppaltest.py -y -nonreboot` | `device_teardown done`, 0 mismatch |
| Conv HW | `apppaltest.py -y -nonreboot` | `device_teardown done`, 0 mismatch |

## Residual Risks

1. **Op lifetime** — a helper reading the tiling op after conversion erases `partitiontensor` would crash/return empty. Mitigation: read once at pass entry into the struct (Tasks 4/5), never inside patterns.
2. **Conv output-partition identification** (Task 7) — the design doc flags reliably locating the *output* `partitiontensor`. Verify against generated conv IR before wiring `readGemmTilingScalars` to it.
3. **`m_rounds/n_rounds` divergence** — struct-derived (`slice_tiling.rounds`) vs old attr (`tile_rows/tile_m`). Assert equality with a temporary log during Phase 1 before deleting the attrs.
4. **Not in scope** — the separate module *dictionary* attrs (`tensor_N.halo`, `tensor_N.shim_dma`, `routing.pp_depth_map`) are a different mechanism and are left untouched.
