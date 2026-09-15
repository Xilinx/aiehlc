# Row-Control `plan_init` + Top-Row Return Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the two-step `__Runtime_ctrl_row_open` + per-row `__Runtime_ctrl_row_add` fabric setup with a single `__Runtime_ctrl_plan_init(rows[], nrows)` call, and drop the idle `RET_NORTH` return slot on the topmost configured row.

**Architecture:** A new static one-shot planner (`rt_ctrl_plan_add_rows`) loops the full row list, computing `top_row = max(row)` so the pure planner can flag the top row. The pure planner threads an `is_top` flag into `acr_plan_return_chain`; the top head omits `RET_NORTH` and drives `SOUTH→VRET` with `{local, transit}` only. Ripple updates cover the demo, the debug provenance parser, unit tests, and docs.

**Tech Stack:** C (XAie runtime + pure planner), C++ host unit test (gcc/g++), Python 3 (`xaiehost2provenance.py` + pytest).

**IMPORTANT context for the executor:**
- The working tree already contains **uncommitted** edits to `aie_runtime.c`, `aie_runtime_control_plan.{c,h}`, `ctrlrow_demo.cc`, `test_ctrl_row_plan.cpp`, and `CLAUDE.md`. All line references below are to the **current on-disk state**. Do NOT `git checkout`/reset these files.
- The design doc is `docs/plans/2026-09-13-row-control-plan-init-design.md` (already committed as `076fe9c`).
- The shell profile prints heavy env banner text on every `Bash` call and `git` needs `GIT_PAGER=cat`. Ignore the banner; look past it for real output.
- `never do`: keep every API function < 200 lines.

---

### Task 1: Pure planner — thread `is_top`, drop top-row `RET_NORTH`

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.h` (decls + comments)
- Modify: `src/mlir/runtime/aie_runtime_control_plan.c:181-216` (`acr_plan_return_chain`), `:256-325` (`acr_plan_row_add`)
- Test: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`

**Step 1: Update the unit test extern decls + calls to the new signatures**

In `test_ctrl_row_plan.cpp`, change the two extern "C" decls (lines 8-10):

```c
acr_rc acr_plan_row_add(acr_state *, acr_oplist *, acr_portbook *, uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top);
acr_rc acr_plan_return_chain(acr_oplist *, acr_portbook *, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top);
```

Update every call:
- `test_slot_classes`: `acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, /*ctrl_id*/ 0, /*is_top*/ 1)` (single row ⇒ top).
- `test_forward_slot_table`: `acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, /*ctrl_id*/ 0, /*is_top*/ 1)`.
- `test_return_chain`: `acr_plan_return_chain(&o, &b, 3, 2, 4, /*is_top*/ 0)` (keep the existing 0x7 head assertion — this is the non-top case).
- `test_spine_reuse`: first add row 3 `is_top=0` (row 5 comes later), then re-add row 3 `is_top=0`, then row 5 `is_top=1`.
- `test_shared_head_fanout`: row 3 `is_top=0`, then row 5 `is_top=1`.

**Step 2: Add a top-row return assertion to `test_return_chain`**

Append to `test_return_chain` (after the existing non-top assertions), a second fabric that plans the SAME row as the top row and asserts the head omits `RET_NORTH` and drives SOUTH with `0x3`:

```c
    // Top row (is_top=1): head omits the RET_NORTH slot; SOUTH master pulls
    // {local, transit} only (0x3). Nothing descends from above a top head.
    acr_oplist ot = {};
    acr_portbook bt = {};
    assert(acr_plan_return_chain(&ot, &bt, 3, 2, 4, /*is_top*/ 1) == ACR_OK);
    assert(find_slot(&ot, head, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_ARB_RET) == nullptr);
    assert(count_master(&ot, head, ACR_SOUTH) == 1);
    assert(master_mselen(&ot, head, ACR_SOUTH) == 0x3);
```

**Step 3: Run the unit test to verify it FAILS to compile**

Run: `bash src/mlir/runtime/unitest/build_ctrl_row_plan.sh`
Expected: compile error — `acr_plan_return_chain` / `acr_plan_row_add` called with too many args (signatures not yet updated).

**Step 4: Add `is_top` to the header decls + comments**

In `aie_runtime_control_plan.h`, update the two prototypes (lines ~157-158 and ~170) to add a trailing `int is_top`:

```c
acr_rc acr_plan_row_add(acr_state *s, acr_oplist *o, acr_portbook *b, uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top);
...
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top);
```

Add to the `acr_plan_return_chain` doc comment: "@is_top marks the topmost configured row; when set, the head omits the idle RET_NORTH slot and drives SOUTH→VRET with {local, transit} only (nothing descends from above the top head)."

**Step 5: Implement `is_top` in `acr_plan_return_chain`**

In `aie_runtime_control_plan.c`, change the signature (line 181) to add `int is_top`, then replace the `is_head` branch (lines 198-207) with:

```c
        if (is_head) {
            /* Non-top head: merge the descending upper-spine responses on its
             * NORTH slave. The top head has nothing above it, so it omits this
             * idle slot. */
            if (!is_top &&
                (rc = acr_emit_ret_slot(o, b, c, row, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_MSEL_RET_NORTH)) != ACR_OK)
                return rc;
            /* Head descent: SOUTH master -> VRET, pulling {local, transit?} plus
             * {north} only when a head can sit above (non-top). */
            uint8_t mselen = (uint8_t)((1u << ACR_MSEL_RET_LOCAL) | (has_transit ? (1u << ACR_MSEL_RET_TRANSIT) : 0u) |
                                       (is_top ? 0u : (1u << ACR_MSEL_RET_NORTH)));
            if ((rc = acr_emit_master(o, b, c, row, ACR_SOUTH, mselen, ACR_ARB_RET)) != ACR_OK)
                return rc;
        } else {
```

**Step 6: Thread `is_top` through `acr_plan_row_add`**

Change the `acr_plan_row_add` signature (line 256) to add trailing `int is_top`, and update its return-chain call (line 319) to:

```c
    rc = acr_plan_return_chain(o, b, row, col_lo, col_hi, is_top);
```

(`acr_plan_chain` at line 220 does NOT call the return chain — leave it unchanged.)

**Step 7: Run the unit test to verify PASS**

Run: `bash src/mlir/runtime/unitest/build_ctrl_row_plan.sh`
Expected: `PASS`

**Step 8: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.h src/mlir/runtime/aie_runtime_control_plan.c \
        src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
GIT_PAGER=cat git commit -m "feat(control-plan): drop idle RET_NORTH on top return head"
```

---

### Task 2: Runtime — `__Runtime_ctrl_plan_init` + static one-shot planner

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c:3807-3892` (`__Runtime_ctrl_row_open`, `__Runtime_ctrl_row_add`)
- Modify: `src/mlir/runtime/aie_runtime.h:891-903` (decls)

**Step 1: Replace `__Runtime_ctrl_row_open` with `__Runtime_ctrl_plan_init` + static helpers**

In `aie_runtime.c`, replace the `__Runtime_ctrl_row_open` function (lines 3807-3823) with:

```c
/* Plan one row (forward + return chain) against the fabric's real spine + port
 * book, emit the derived stream-switch ops, and record the chain span when a new
 * row is actually added (an idempotent re-add leaves f->spine.nrows unchanged).
 * @is_top marks the topmost configured row (its return head omits the idle
 * RET_NORTH slot). Static: the fabric is planned once by __Runtime_ctrl_plan_init. */
static AieRC rt_ctrl_plan_add_row(__Runtime_CtrlRowFabric *f, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top) {
    if (f->nrows >= ACR_MAX_ROWS)
        return XAIE_INVALID_ARGS;
    uint8_t nrows_before = f->spine.nrows;
    acr_oplist ops;
    ops.n = 0;
    acr_rc prc = acr_plan_row_add(&f->spine, &ops, &f->book, f->shim_col, row, col_lo, col_hi, f->ctrl_id, is_top);
    if (prc != ACR_OK) {
        printf("[aie_runtime] ctrl_plan: planner rc=%d row=%u [%u..%u]\n", (int)prc, (unsigned)row, (unsigned)col_lo,
               (unsigned)col_hi);
        return XAIE_INVALID_ARGS;
    }
    AieRC rc = __Runtime_ctrl_row_emit(f->dev, &ops);
    if (rc != XAIE_OK)
        return rc;
    if (f->spine.nrows > nrows_before) {
        f->rows[f->nrows].row = row;
        f->rows[f->nrows].col_lo = col_lo;
        f->rows[f->nrows].col_hi = col_hi;
        f->nrows++;
    }
    return XAIE_OK;
}

/* Plan the entire row list once. Rows MUST be given bottom-up (ascending AIE row)
 * so the shared vertical spine extends upward and is reused. Computes the topmost
 * row (independent of order) so its return head can omit the idle RET_NORTH slot.
 * Called once per fabric by __Runtime_ctrl_plan_init. */
static AieRC rt_ctrl_plan_add_rows(__Runtime_CtrlRowFabric *f, const __Runtime_CtrlRowChain *rows, uint8_t nrows) {
    if (!rows || nrows == 0U || nrows > ACR_MAX_ROWS)
        return XAIE_INVALID_ARGS;
    uint8_t top_row = rows[0].row;
    for (uint8_t i = 1; i < nrows; i++)
        if (rows[i].row > top_row)
            top_row = rows[i].row;
    for (uint8_t i = 0; i < nrows; i++) {
        AieRC rc = rt_ctrl_plan_add_row(f, rows[i].row, rows[i].col_lo, rows[i].col_hi, rows[i].row == top_row);
        if (rc != XAIE_OK)
            return rc;
    }
    return XAIE_OK;
}

/* Initialize the row-control fabric and plan the whole row list in one shot. The
 * spine hops and EAST chains are emitted here (was: lazily by row_add). Records
 * the device, spine column, control stream id, and shim S2MM response channel.
 * @rows must be ordered bottom-up (ascending AIE row). No HW is touched until the
 * per-row emit inside rt_ctrl_plan_add_rows. */
AieRC __Runtime_ctrl_plan_init(__Runtime_CtrlRowFabric *f, XAie_DevInst *dev, uint8_t shim_col, int32_t resp_s2mm_ch,
                               uint8_t ctrl_id, const __Runtime_CtrlRowChain *rows, uint8_t nrows) {
    if (!f || !dev)
        return XAIE_INVALID_ARGS;
    memset(f, 0, sizeof(*f));
    f->dev = dev;
    f->shim_col = shim_col;
    f->ctrl_id = ctrl_id;
    f->fwd_vc = 0U; /* single forward/return vertical channel pair for now */
    f->ret_vc = 0U;
    f->resp_s2mm_ch = resp_s2mm_ch;
    return rt_ctrl_plan_add_rows(f, rows, nrows);
}
```

**Step 2: Delete the old `__Runtime_ctrl_row_add` function**

Remove the entire `__Runtime_ctrl_row_add` function (lines 3860-3892, from its doc comment through the closing brace). Its body now lives in `rt_ctrl_plan_add_row`. Leave `rt_ctrl_row_shim_entry` (3834-3858) intact.

**Step 3: Update the header decls**

In `aie_runtime.h`, remove the `__Runtime_ctrl_row_open` decl (lines 891-893) and the `__Runtime_ctrl_row_add` decl (lines 899-903). Replace with:

```c
// Initialize the fabric and plan the whole row list in one shot. @rows (length
// @nrows) lists the EAST chains bottom-up (ascending AIE row); each chain's head
// @col_lo must equal @shim_col. Builds/reuses the shared vertical spine, emits the
// derived stream-switch config, and drains responses via @resp_s2mm_ch. The
// topmost row's return head omits the idle RET_NORTH merge slot.
AieRC __Runtime_ctrl_plan_init(__Runtime_CtrlRowFabric *f, XAie_DevInst *dev, uint8_t shim_col, int32_t resp_s2mm_ch,
                               uint8_t ctrl_id, const __Runtime_CtrlRowChain *rows, uint8_t nrows);
```

(The `__Runtime_CtrlRowChain` struct at lines 865-869 is declared above this, so it is in scope.)

**Step 4: Verify it compiles (planner still builds; runtime compiles under the E2E build in Task 3).**

There is no standalone unit build for `aie_runtime.c`; it is validated by the demo build in Task 3. Do a quick syntax sanity check that no other C source references the removed symbols:

Run: `grep -rn "__Runtime_ctrl_row_open\|__Runtime_ctrl_row_add" src/ example/ --include=*.c --include=*.cc --include=*.h`
Expected: no matches in C/C++ sources except (temporarily) `ctrlrow_demo.cc` (fixed in Task 3). If the header still shows a match, the decl removal is incomplete.

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
GIT_PAGER=cat git commit -m "feat(runtime): __Runtime_ctrl_plan_init one-shot row-list fabric setup"
```

---

### Task 3: Demo — single `plan_init` call

**Files:**
- Modify: `example/tileprogram/ccode/ctrlrow_demo.cc:236-282` (drop `demo_add_row`, rewrite setup)

**Step 1: Remove the `demo_add_row` helper**

Delete the `demo_add_row` static function (lines 236-246).

**Step 2: Replace the open + add + re-add block**

In `run_ctrlrow_demo`, replace lines 257-282 (from the `__Runtime_ctrl_row_open` call through the idempotent re-add `printf`) with:

```c
    /* Plan the whole fabric in one shot: rows A (lower) then B (higher) share the
     * vertical spine (B reuses the shim..A hops). Rows must be bottom-up so the
     * spine extends upward; B is the top row (its return head omits RET_NORTH). */
    static const __Runtime_CtrlRowChain kRows[] = {
        {DEMO_CORE_ROW_A, DEMO_COL_LO, DEMO_COL_HI},
        {DEMO_CORE_ROW_B, DEMO_COL_LO, DEMO_COL_HI},
    };
    AieRC rc = __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, DEMO_S2MM_CH, (uint8_t)DEMO_CTRL_ID, kRows,
                                        (uint8_t)(sizeof(kRows) / sizeof(kRows[0])));
    if (rc != XAIE_OK) {
        printf("[ctrlrow] plan_init rc=%d\n", (int)rc);
        return -1;
    }
    printf("[ctrlrow] plan_init rows=%u spine_top=%u (A=%u,B=%u share spine)\n", (unsigned)fab.nrows,
           (unsigned)fab.spine.spine_top, DEMO_CORE_ROW_A, DEMO_CORE_ROW_B);
```

Then delete the now-unused `uint8_t spine_after_a = ...;` line if it remains (it is inside the replaced block). Leave step 4 (broadcast) onward untouched. The remaining `AieRC rc` uses later in the function reuse this `rc` variable — verify no duplicate `AieRC rc` declaration remains (the original had one at line 257; keep exactly one).

**Step 3: Build + run the E2E flow**

Run:
```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file example/tileprogram/ccode/ctrlrow_demo.cc
python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```
Expected in `applog`: every `[ctrlrow] ... PASS`, `[ctrlrow] DONE PASS (0 multicast failure(s))`, and `device_teardown done` with no `AIE ERROR`.

If the build fails at compile, fix the demo/runtime until it compiles; if it runs but a probe FAILs, use `superpowers:systematic-debugging` before changing routing logic.

**Step 4: Commit**

```bash
git add example/tileprogram/ccode/ctrlrow_demo.cc
GIT_PAGER=cat git commit -m "feat(demo): use __Runtime_ctrl_plan_init row-list setup"
```

---

### Task 4: Debug provenance parser — parse `plan_init` + row array

**Files:**
- Modify: `src/tool/debug/xaiehost2provenance.py:131-140` (regexes + comment), `:268-326` (`extract_ctrl_rows`), `:335-339` (`RE_XAIE_CALL` comment only if it names the old API)
- Test: `src/tool/debug/tests/test_xaiehost2provenance.py:335-415`

**Step 1: Update the three test fixtures + assertions**

In `test_xaiehost2provenance.py`, replace `CTRL_ROW_SRC` (lines 340-348) with the array form:

```python
CTRL_ROW_SRC = """
int run_ctrlrow_demo(XAie_DevInst *dev) {
    __Runtime_CtrlRowFabric fab;
    static const __Runtime_CtrlRowChain rows[] = {
        {3u, 0u, 1u},
        {5u, 0u, 1u},
    };
    __Runtime_ctrl_plan_init(&fab, dev, 0u, 0, (uint8_t)0u, rows, 2u);
}
"""
```

`test_extract_ctrl_rows_open_and_add` assertion stays `[{3,0,1},{5,0,1}]`; update its comment (no more re-add). Replace `CTRL_ROW_DEFINE_SRC` (lines 366-376) with:

```python
CTRL_ROW_DEFINE_SRC = """
#define DEMO_SHIM_COL 0u
#define DEMO_CORE_ROW_A 3u
#define DEMO_COL_LO 0u
#define DEMO_COL_HI 1u
int run_ctrlrow_demo(XAie_DevInst *dev) {
    static const __Runtime_CtrlRowChain rows[] = {
        {DEMO_CORE_ROW_A, DEMO_COL_LO, DEMO_COL_HI},
    };
    __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, 0, (uint8_t)0u, rows, 1u);
}
"""
```

Delete the wrapper-forwarding fixture `CTRL_ROW_WRAPPER_SRC` (lines 393-407) and its test `test_extract_ctrl_rows_follows_wrapper_forwarding` (lines 410-end of that test) — the array form no longer forwards rows through a helper. Add a multi-row-define fixture + test instead:

```python
CTRL_ROW_MULTI_SRC = """
#define DEMO_SHIM_COL 0u
#define DEMO_CORE_ROW_A 3u
#define DEMO_CORE_ROW_B 5u
#define DEMO_COL_LO 0u
#define DEMO_COL_HI 3u
int run_ctrlrow_demo(XAie_DevInst *dev) {
    static const __Runtime_CtrlRowChain rows[] = {
        {DEMO_CORE_ROW_A, DEMO_COL_LO, DEMO_COL_HI},
        {DEMO_CORE_ROW_B, DEMO_COL_LO, DEMO_COL_HI},
    };
    __Runtime_ctrl_plan_init(&fab, dev, DEMO_SHIM_COL, 0, (uint8_t)0u, rows, 2u);
}
"""


def test_extract_ctrl_rows_folds_row_array():
    active = x.strip_comments(x.MacroResolver(5, False).active_source(CTRL_ROW_MULTI_SRC))
    fab = x.extract_ctrl_rows(active, x.collect_defines(active))
    assert fab["shim_col"] == 0
    assert fab["rows"] == [{"row": 3, "col_lo": 0, "col_hi": 3},
                           {"row": 5, "col_lo": 0, "col_hi": 3}]
```

**Step 2: Run the provenance tests to verify they FAIL**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -q -k ctrl_rows`
Expected: FAIL (parser still matches the old `__Runtime_ctrl_row_open`/`row_add`, so `extract_ctrl_rows` returns `None` / wrong rows).

**Step 3: Rewrite the regexes**

In `xaiehost2provenance.py`, replace `RE_ROW_OPEN` / `RE_ROW_ADD` (lines 137-140) with:

```python
# The single __Runtime_ctrl_plan_init(fab, dev, shim_col, ...) picks the shared
# spine (shim) column; its row list is a __Runtime_CtrlRowChain rows[] = {{row,
# col_lo, col_hi}, ...} array. Capture the 3rd positional arg (shim_col) and each
# {row, col_lo, col_hi} triple from the array initializer.
RE_PLAN_INIT = re.compile(
    r"__Runtime_ctrl_plan_init\s*\(\s*[^,]+,\s*[^,]+,\s*([^,]+),")
RE_ROW_ARRAY = re.compile(
    r"__Runtime_CtrlRowChain\s+\w+\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;", re.DOTALL)
RE_ROW_TRIPLE = re.compile(r"\{\s*([^,{}]+),\s*([^,{}]+),\s*([^,{}]+)\}")
```

**Step 4: Rewrite `extract_ctrl_rows`**

Replace the body of `extract_ctrl_rows` (lines 268-326) with an array-based parser (drop the `_resolve` wrapper-forwarding logic — rows are now a static array):

```python
def extract_ctrl_rows(active, defs):
    """Reduce a row-control fabric to {shim_col, rows:[{row,col_lo,col_hi}]}.

    The single __Runtime_ctrl_plan_init gives the shared spine (shim) column; its
    __Runtime_CtrlRowChain rows[] initializer lists the EAST chains as {row,
    col_lo, col_hi} triples. Chains are deduped by (row,col_lo,col_hi). Returns
    None when no fabric is initialized (or its shim column can't be folded);
    triples whose fields don't fold are skipped. Values fold through @defs,
    tolerating u/l suffixes."""
    mo = RE_PLAN_INIT.search(active)
    if not mo:
        return None
    shim_col = _ctrl_int(mo.group(1), defs)
    if shim_col is None:
        return None
    rows, seen = [], set()
    am = RE_ROW_ARRAY.search(active)
    if am:
        for tm in RE_ROW_TRIPLE.finditer(am.group(1)):
            row = _ctrl_int(tm.group(1), defs)
            col_lo = _ctrl_int(tm.group(2), defs)
            col_hi = _ctrl_int(tm.group(3), defs)
            if row is None or col_lo is None or col_hi is None:
                continue
            key = (row, col_lo, col_hi)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"row": row, "col_lo": col_lo, "col_hi": col_hi})
    return {"shim_col": shim_col, "rows": rows}
```

Then check whether `_c_functions` and `_call_args` are used anywhere else:
Run: `grep -n "_c_functions\|_call_args" src/tool/debug/xaiehost2provenance.py`
If they are ONLY used by the old `extract_ctrl_rows`, delete their definitions (dead code). If used elsewhere, leave them.

Update the module comment block at lines 131-136 to describe `plan_init` + the row array (no more per-`row_add`).

**Step 5: Run the provenance tests to verify PASS**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -q`
Expected: all tests pass.

**Step 6: Commit**

```bash
git add src/tool/debug/xaiehost2provenance.py src/tool/debug/tests/test_xaiehost2provenance.py
GIT_PAGER=cat git commit -m "feat(debug): parse __Runtime_ctrl_plan_init row array in provenance"
```

---

### Task 5: Docs — update `CLAUDE.md` control-plane notes

**Files:**
- Modify: `CLAUDE.md` (control-plane paragraph)

**Step 1: Update the API references**

In `CLAUDE.md`, find the control-plane paragraph mentioning the pure row-control planner and runtime send API. Update the setup-API description: replace any mention of the two-step `__Runtime_ctrl_row_open` + per-row `__Runtime_ctrl_row_add` with:

> The fabric is now configured in one shot by `__Runtime_ctrl_plan_init(f, dev, shim_col, resp_s2mm_ch, ctrl_id, rows, nrows)` — `rows` is a `__Runtime_CtrlRowChain[]` (each `{row, col_lo, col_hi}`) ordered bottom-up; a static one-shot planner (`rt_ctrl_plan_add_rows`) computes the top row and plans/emits every chain, reusing the shared spine. The topmost configured row's return head omits the idle `RET_NORTH` merge slot (nothing descends from above it); non-top heads still merge `{local, transit, north}` and drive `SOUTH→VRET`.

**Step 2: Grep for any lingering old-API mention in `CLAUDE.md`**

Run: `grep -n "__Runtime_ctrl_row_open\|__Runtime_ctrl_row_add" CLAUDE.md`
Expected: no matches (all updated).

**Step 3: Commit**

```bash
git add CLAUDE.md
GIT_PAGER=cat git commit -m "docs: update control-plane notes for plan_init + top-row return"
```

---

## Final verification checklist

- [ ] `bash src/mlir/runtime/unitest/build_ctrl_row_plan.sh` → `PASS`
- [ ] `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -q` → all pass
- [ ] E2E demo (`apppaltest.py`) → every probe PASS, `DONE PASS`, `device_teardown done`, no `AIE ERROR`
- [ ] `grep -rn "__Runtime_ctrl_row_open\|__Runtime_ctrl_row_add" src/ example/ CLAUDE.md` → no matches
- [ ] List all changed/new files (Process-transparent rule)

## Notes / references

- Debugging routing failures: `superpowers:systematic-debugging`, and skills `routinghwdebug` / `aiehwdmadebug`.
- Optional doc: `.cursor/skills/debug-ui-framework/reference.md` mentions the old API in prose; update if convenient (not required for correctness).
