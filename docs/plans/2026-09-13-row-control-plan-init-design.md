# Row-Control `plan_init` Row-List API + Top-Row Return Cleanup — Design

Date: 2026-09-13
Status: Approved (brainstorming) — ready for implementation plan

## Problem

The row-control fabric is configured in two steps today:

1. `__Runtime_ctrl_row_open(f, dev, shim_col, resp_s2mm_ch, ctrl_id)` — records
   device / spine column / ids; touches no HW.
2. One `__Runtime_ctrl_row_add(f, row, col_lo, col_hi)` **per row** — lazily
   extends the shared vertical spine and emits that row's forward + return chain.

Because `row_add` sees only one row at a time, the pure planner cannot tell
whether a given row is the **topmost** configured row. As a result every head row
unconditionally arms a `RET_NORTH` return slot and drives `SOUTH→VRET` with
mselen `{local, transit, north}` (0x7) — even the top head, whose `RET_NORTH`
slot never receives traffic (nothing descends from above it). This idle slot is
harmless but wasteful and obscures intent.

## Requirements (as agreed)

1. Make `__Runtime_ctrl_row_add` a **static** helper that is called **once** per
   fabric instance and receives the **entire row list**.
2. Call that helper from the (renamed) init entry point: rename
   `__Runtime_ctrl_row_open` → `__Runtime_ctrl_plan_init`, taking the row list.
3. Now that the planner has the whole row set, handle the **top row** specially in
   the **return path**: the top row's head merges `{local (CTRL→SOUTH), east
   (EAST→SOUTH)}` only; non-top heads keep `{local, east, north}` (CTRL→SOUTH,
   EAST→SOUTH, NORTH→SOUTH). I.e. drop the idle `RET_NORTH` slot on the top row.

(Requirement #3's original phrase "east to north" for the top row was a
transcription slip for "east to south"; north is away from the shim and the
return bus always funnels south.)

## Decisions (from brainstorming)

- **Row-list form:** `const __Runtime_CtrlRowChain *rows, uint8_t nrows`, reusing
  the existing `{row, col_lo, col_hi}` struct (supports per-row column spans).
- **Row order:** callers pass rows **bottom-up (ascending AIE row)**; documented,
  no sort. `top_row` is computed as `max(rows[i].row)` for the `is_top` flag
  (independent of processing order).
- **`__Runtime_ctrl_row_add` public symbol is removed** from the header; the
  idempotent per-row re-add scenario disappears (each fabric is planned once).

## API changes (`aie_runtime.{c,h}`)

New init entry point:

```c
AieRC __Runtime_ctrl_plan_init(__Runtime_CtrlRowFabric *f, XAie_DevInst *dev,
                               uint8_t shim_col, int32_t resp_s2mm_ch,
                               uint8_t ctrl_id,
                               const __Runtime_CtrlRowChain *rows, uint8_t nrows);
```

- Does the old `row_open` init (memset, record dev/shim/ctrl_id/vc/resp channel).
- Then calls the new static helper once:

```c
static AieRC rt_ctrl_plan_add_rows(__Runtime_CtrlRowFabric *f,
                                   const __Runtime_CtrlRowChain *rows,
                                   uint8_t nrows);
```

- `rt_ctrl_plan_add_rows` computes `top_row = max(rows[i].row)`, then for each row
  (in given ascending order) calls the pure planner `acr_plan_row_add(..., is_top
  = (rows[i].row == top_row))`, emits the resulting op list via
  `__Runtime_ctrl_row_emit`, and records the chain span in `f->rows[]` / `f->nrows`
  exactly as `__Runtime_ctrl_row_add` does today.
- The body of the old `__Runtime_ctrl_row_add` moves into a per-row static step
  called from this loop (keeps the <200-line rule).

`__Runtime_ctrl_row_close`, the send/read/write-ack APIs, and the emit layer are
unchanged.

## Planner changes (`aie_runtime_control_plan.{c,h}`)

Thread an `is_top` flag down to the return chain:

```c
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row,
                             uint8_t col_lo, uint8_t col_hi, int is_top);
acr_rc acr_plan_row_add(acr_state *s, acr_oplist *o, acr_portbook *b,
                        uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top);
```

In `acr_plan_return_chain`, at the head tile (`c == col_lo`):

- **Non-top (`is_top == 0`):** unchanged — arm the `RET_NORTH` slot on the NORTH
  slave, SOUTH master mselen = `{local, transit?, north}` (0x7 / 0x5 when single
  column).
- **Top (`is_top == 1`):** do **not** arm `RET_NORTH`; SOUTH master mselen =
  `{local, transit?}` (0x3, or 0x1 when `col_lo == col_hi`).

`acr_plan_row_add` forwards `is_top` to `acr_plan_return_chain`. The forward chain
and spine-extension logic are untouched: the spine still extends only to
`row-1`, so no pass-through return CCT ever sits above the top head.

`acr_plan_chain` (plain chain, no spine/return) is unaffected.

## Ripple updates

- **`example/tileprogram/ccode/ctrlrow_demo.cc`**: replace `row_open` + the two
  `demo_add_row` calls + the idempotent re-add block (step 3) with a single
  `__Runtime_ctrl_plan_init` passing a static array `{{3,0,3},{5,0,3}}`. Drop the
  `demo_add_row` helper and the re-add PASS/FAIL check. Keep all probes
  (broadcast, whole-row, read, write-ack) and the final spine_top/nrows log.
- **`src/tool/debug/xaiehost2provenance.py`** (+ `tests/test_xaiehost2provenance.py`):
  `RE_ROW_OPEN` matches `__Runtime_ctrl_plan_init` (shim_col still the 3rd positional
  arg). Replace the per-`__Runtime_ctrl_row_add` `RE_ROW_ADD` extraction with parsing
  the `__Runtime_CtrlRowChain rows[]` array initializer (fold `{{r,lo,hi}, ...}`
  triples through `defs`, deduped by `(row,col_lo,col_hi)`). Update the test fixture
  to the new call shape.
- **`src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`**: thread `is_top` through
  `acr_plan_row_add` / `acr_plan_return_chain` calls. Add assertions: top head has
  **no** `RET_NORTH` slot and SOUTH mselen `0x3`; a non-top head keeps `RET_NORTH`
  and SOUTH mselen `0x7`. Existing forward-chain / spine-reuse assertions unchanged.
- **`CLAUDE.md`**: update the control-plane paragraph — `plan_init` row-list API,
  static one-shot planning, and the top-row return (no idle `RET_NORTH`).

## Testing

- Host unit test: `src/mlir/runtime/unitest/build_ctrl_row_plan.sh` → `PASS`.
- Provenance test: `src/tool/debug/tests/test_xaiehost2provenance.py` → pass.
- E2E: `source script/aiehlc.sh --aie-version 5 --runtime-source-file
  example/tileprogram/ccode/ctrlrow_demo.cc` then
  `python3 script/test/apppaltest.py -y -nonreboot` — every probe PASSes and the
  run reaches teardown with no AIE ERROR.

## Files touched

- `src/mlir/runtime/aie_runtime.c` — `plan_init` + static `rt_ctrl_plan_add_rows`.
- `src/mlir/runtime/aie_runtime.h` — declare `plan_init`; remove `row_add` /
  `row_open` decls.
- `src/mlir/runtime/aie_runtime_control_plan.c` / `.h` — `is_top` in
  `acr_plan_row_add` / `acr_plan_return_chain`; top-row `RET_NORTH` omission.
- `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp` — `is_top` + top-row asserts.
- `example/tileprogram/ccode/ctrlrow_demo.cc` — single `plan_init` call.
- `src/tool/debug/xaiehost2provenance.py` (+ test) — parse `plan_init` + row array.
- `CLAUDE.md` — control-plane notes.
