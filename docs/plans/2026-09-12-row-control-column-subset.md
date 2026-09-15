# Column-Subset Row-Multicast Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three column-subset delivery classes (all-but-last / only-last / whole-row) to the row-control plane, keyed by packet id `[3:2]`, with a 4-row-index space in id `[1:0]`.

**Architecture:** The pure planner (`aie_runtime_control_plan.c`) arms per-tile-role slot tables (interior vs last-tile) whose masks/ids implement the class semantics; the runtime (`aie_runtime.c`) resolves a physical row to its add-order index and sends `id=(class<<2)|rowidx`. Broadcast (`id[4]=1`) is unchanged.

**Tech Stack:** C11 pure planner + host C++ unit test (`gcc` C linkage + `g++` test), XAie runtime C, AIE stream-switch packet routing.

**Design doc:** `docs/plans/2026-09-12-row-control-column-subset-design.md`

**Build/run the host unit test (used in every task):**
```bash
mkdir -p /tmp/claude \
 && gcc -I src/mlir/runtime -c src/mlir/runtime/aie_runtime_control_plan.c -o /tmp/claude/acp.o \
 && g++ -I src/mlir/runtime -c src/mlir/runtime/unitest/test_ctrl_row_plan.cpp -o /tmp/claude/acp_test.o \
 && g++ /tmp/claude/acp_test.o /tmp/claude/acp.o -o /tmp/claude/acr_test \
 && /tmp/claude/acr_test
```
Expected on success: `PASS`.

---

## Task 1: New constants in `aie_runtime_control_plan.h`

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.h:14-44`

**Step 1: Replace the row-id cap + two-mode constants block**

Replace lines 14 and 16-44 (the `ACR_MAX_ROW_ID`, the two-class doc comment, and the
`ACR_SLOT_*` / `ACR_MSEL_*` / id / mask / `ACR_MSELEN_*` defines) with:

```c
#define ACR_MAX_ROW_IDX 3 /* target row index lives in id[1:0] (0..3 => up to 4 rows) */

/* Three column-subset delivery classes select which columns of the target row
 * consume a non-broadcast packet, keyed by id[3:2]; id[1:0] = target row index
 * (add order). Broadcast is id[4]=1 (every column incl. last, every row).
 *   id = (class << 2) | rowidx.
 *   class 00 = all-but-last : every column EXCEPT col_hi
 *   class 01 = only-last    : only col_hi
 *   class 10 = whole-row    : every column incl. col_hi
 * Interior tiles (col < col_hi) arm a masked consume slot (classes 00,10) + a
 * broadcast slot + a transit-east slot (class 01, forwarded east but not
 * consumed); spine-head tiles add a transit-north climb slot. Last tiles (col_hi)
 * arm two exact consume slots (only-last, whole-row) + broadcast. */
#define ACR_CLASS_ALL_BUT_LAST 0
#define ACR_CLASS_ONLY_LAST 1
#define ACR_CLASS_WHOLE_ROW 2

/* Interior-tile slot indices (== MSel on that tile). */
#define ACR_SLOT_CONSUME 0   /* classes {00,10} @row K -> CTRL + EAST */
#define ACR_SLOT_BCAST 1     /* id[4]=1 -> CTRL + EAST (+NORTH on head) */
#define ACR_SLOT_TRANSIT_N 2 /* any row-mcast -> NORTH climb (spine head only) */
#define ACR_SLOT_TRANSIT_E 3 /* class 01 @row K -> EAST only (forward past) */

/* Last-tile slot indices (== MSel on that tile). */
#define ACR_SLOT_ONLY_LAST 0  /* class 01 @row K -> CTRL */
#define ACR_SLOT_WHOLE 1      /* class 10 @row K -> CTRL */
#define ACR_SLOT_BCAST_LAST 2 /* id[4]=1 -> CTRL */

/* MSel select value each slot injects (mirrors the slot index above). */
#define ACR_MSEL_CONSUME 0
#define ACR_MSEL_BCAST 1
#define ACR_MSEL_TRANSIT_N 2
#define ACR_MSEL_TRANSIT_E 3
#define ACR_MSEL_ONLY_LAST 0
#define ACR_MSEL_WHOLE 1
#define ACR_MSEL_BCAST_LAST 2

#define ACR_ARB_CTRL 0 /* single shared arbiter for all classes */

/* Reserved-bit id/mask scheme (5-bit stream id):
 *   [4]=bcast marker; [3:2]=class; [1:0]=target row index when [4]=0. */
#define ACR_ID_BCAST 0x10    /* [4]=1 */
#define ACR_MASK_BCAST 0x10  /* match on [4] only */
#define ACR_ID_TRANSIT 0x00  /* transit-north matches any row-mcast (bit4=0) */
#define ACR_MASK_CLASS 0x10  /* match on [4] only: row-mcast (0) vs broadcast (1) */
#define ACR_MASK_EXACT 0x1F  /* full 5-bit exact match */
#define ACR_MASK_CONSUME 0x17 /* match [4]=0,[2]=0,[1:0]=K ; ignore [3] -> classes 00 & 10 */

/* Per-master MSelEn bitmaps. */
#define ACR_MSELEN_CTRL ((1u << ACR_MSEL_CONSUME) | (1u << ACR_MSEL_BCAST))                          /* 0x3 */
#define ACR_MSELEN_EAST ((1u << ACR_MSEL_CONSUME) | (1u << ACR_MSEL_BCAST) | (1u << ACR_MSEL_TRANSIT_E)) /* 0xB */
#define ACR_MSELEN_NORTH ((1u << ACR_MSEL_BCAST) | (1u << ACR_MSEL_TRANSIT_N))                       /* 0x6 */
#define ACR_MSELEN_CTRL_LAST ((1u << ACR_MSEL_ONLY_LAST) | (1u << ACR_MSEL_WHOLE) | (1u << ACR_MSEL_BCAST_LAST)) /* 0x7 */
```

Note: `ACR_NUM_SLOTS` (=4) at line 13 stays. Keep `acr_plan_chain` / `acr_plan_row_add`
prototypes; their doc comments are updated in Task 3.

**Step 2: Verify it compiles (header only)**

Run: `gcc -I src/mlir/runtime -fsyntax-only -x c src/mlir/runtime/aie_runtime_control_plan.c`
Expected: FAIL — the `.c` still references old `ACR_SLOT_ROWMCAST` etc. (that is fixed in Task 2). This step just confirms the header itself has no syntax error; ignore the "undeclared old-name" errors, treat a *header* parse error as a real failure.

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.h
git commit -m "feat(control-plan): add column-subset class constants (all-but-last/only-last/whole-row)"
```

---

## Task 2: Rewrite `acr_plan_chain_ex` slot/master emission

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.c:98-152`

**Step 1: Replace the per-column loop body**

Replace the body of `acr_plan_chain_ex` (the guards at 100-105 and the `for` loop at
107-150) with the role-split emission below. Keep the function signature and the
`acr_emit_slot` / `acr_emit_master` / `acr_emit_op` helpers unchanged.

```c
    if (col_hi < col_lo || col_hi >= 64)
        return ACR_ERR_BOUNDS;
    if (rowidx > ACR_MAX_ROW_IDX) /* target row index lives in id[1:0] */
        return ACR_ERR_BOUNDS;
    if (ACR_NUM_SLOTS < 4) /* spine head arms consume+bcast+transitN+transitE */
        return ACR_ERR_SLOTS;

    for (uint8_t c = col_lo; c <= col_hi; c++) {
        acr_port sport = (c == col_lo) ? head_ingress : ACR_WEST;
        int is_last = (c == col_hi);
        int has_east = !is_last;
        int is_spine_head = (shim_col >= 0) && (c == (uint8_t)shim_col) && (c == col_lo);

        acr_rc rc;
        /* Book the ingress slave *port* (idx 0) once; slots are sub-resources. */
        if ((rc = acr_book_port(b, c, row, sport, 0, /*master*/ 0)) != ACR_OK)
            return rc;

        if (is_last) {
            /* Last tile (col_hi): consume only-last + whole-row; no EAST/NORTH. */
            uint8_t only_id = (uint8_t)((ACR_CLASS_ONLY_LAST << 2) | rowidx);
            uint8_t whole_id = (uint8_t)((ACR_CLASS_WHOLE_ROW << 2) | rowidx);
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_ONLY_LAST, only_id, ACR_MASK_EXACT,
                                    ACR_MSEL_ONLY_LAST, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_WHOLE, whole_id, ACR_MASK_EXACT, ACR_MSEL_WHOLE,
                                    ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_BCAST_LAST, ACR_ID_BCAST, ACR_MASK_BCAST,
                                    ACR_MSEL_BCAST_LAST, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
        } else {
            /* Interior tile (col < col_hi): masked consume (classes 00,10) +
             * broadcast + transit-east (class 01, EAST-only) [+ transit-north on
             * the spine head]. */
            uint8_t consume_id = (uint8_t)((ACR_CLASS_ALL_BUT_LAST << 2) | rowidx); /* == rowidx */
            uint8_t te_id = (uint8_t)((ACR_CLASS_ONLY_LAST << 2) | rowidx);
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_CONSUME, consume_id, ACR_MASK_CONSUME,
                                    ACR_MSEL_CONSUME, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_BCAST, ACR_ID_BCAST, ACR_MASK_BCAST, ACR_MSEL_BCAST,
                                    ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if (is_spine_head &&
                (rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_TRANSIT_N, ACR_ID_TRANSIT, ACR_MASK_CLASS,
                                    ACR_MSEL_TRANSIT_N, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_TRANSIT_E, te_id, ACR_MASK_EXACT, ACR_MSEL_TRANSIT_E,
                                    ACR_ARB_CTRL)) != ACR_OK)
                return rc;
        }

        acr_op slaveen = {.kind = ACR_OP_SLAVE_EN, .col = c, .row = row, .sport = sport, .sidx = 0, .pkt_id = ctrl_id};
        if ((rc = acr_emit_op(o, &slaveen)) != ACR_OK)
            return rc;

        if (is_last) {
            if ((rc = acr_emit_master(o, b, c, row, ACR_CTRL, (uint8_t)ACR_MSELEN_CTRL_LAST, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
        } else {
            if ((rc = acr_emit_master(o, b, c, row, ACR_CTRL, (uint8_t)ACR_MSELEN_CTRL, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if (has_east &&
                (rc = acr_emit_master(o, b, c, row, ACR_EAST, (uint8_t)ACR_MSELEN_EAST, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
            if (is_spine_head &&
                (rc = acr_emit_master(o, b, c, row, ACR_NORTH, (uint8_t)ACR_MSELEN_NORTH, ACR_ARB_CTRL)) != ACR_OK)
                return rc;
        }
    }
    return ACR_OK;
```

**Step 2: Rename the `row` parameter usage to carry a row index**

`acr_plan_chain_ex` currently takes `uint8_t row` as both the physical AIE row (for
`col`/`row` op coords) AND the slot id. Split them: add a new parameter `uint8_t rowidx`
after `row`. Update the signature at `:98-99`:

```c
static acr_rc acr_plan_chain_ex(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t rowidx, uint8_t col_lo,
                                uint8_t col_hi, uint8_t ctrl_id, int shim_col, acr_port head_ingress) {
```

The op coordinate is still `row` (physical); the slot ids use `rowidx`. `is_spine_head`
compares `c` to `shim_col` (column), unaffected.

**Step 3: Update `acr_plan_chain` wrapper (`:156-158`)**

A plain chain has a single row => rowidx 0. Pass `rowidx=0`:

```c
acr_rc acr_plan_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, uint8_t ctrl_id) {
    return acr_plan_chain_ex(o, b, row, /*rowidx=*/0, col_lo, col_hi, ctrl_id, /*shim_col=*/-1, ACR_WEST);
}
```

**Step 4: Verify the `.c` compiles alone**

Run: `gcc -I src/mlir/runtime -fsyntax-only src/mlir/runtime/aie_runtime_control_plan.c`
Expected: PASS (no diagnostics). (`acr_plan_row_add` still calls the old 8-arg form — that
call is fixed in Task 3; if syntax check fails only on that call, that is expected and
fixed next.)

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.c
git commit -m "feat(control-plan): emit interior/last-tile column-subset slot tables"
```

---

## Task 3: `acr_plan_row_add` passes the row index

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.c:190-235`
- Modify: `src/mlir/runtime/aie_runtime_control_plan.h:33-100` (doc comments only)

**Step 1: Compute + pass the add-order row index**

In `acr_plan_row_add`, the index of a newly added row is `s->nrows` (its position in
`s->rows[]` before the append). Guard it and pass it. Change the bounds guard at `:192`
and the `acr_plan_chain_ex` call at `:229`:

```c
    if (row == 0 || shim_col >= 64)
        return ACR_ERR_BOUNDS;
    if (s->nrows > ACR_MAX_ROW_IDX) /* row index would exceed id[1:0] */
        return ACR_ERR_BOUNDS;
```

(Keep the `col_lo != shim_col` guard and the idempotent `acr_row_is_head` early-return
untouched. Note the old `row > ACR_MAX_ROW_ID` check is removed — physical row is no
longer the id; only the index is bounded.)

At the chain call (`:229`):

```c
    acr_rc rc = acr_plan_chain_ex(o, b, row, /*rowidx=*/s->nrows, col_lo, col_hi, ctrl_id,
                                  /*shim_col=*/(int)shim_col, ACR_SOUTH);
```

**Step 2: Update the doc comments**

- `aie_runtime_control_plan.h:89-100` — rewrite the `acr_plan_chain` / `acr_plan_row_add`
  comments to describe the three classes + row-index scheme (mirror the header block from
  Task 1). Keep them under ~10 lines each.
- `aie_runtime_control_plan.c:88-97` and `:168-189` — update the block comments: the head
  NORTH master (`0x6`) climbs broadcast + any transiting row-mcast; interior EAST (`0xB`)
  forwards consume + broadcast + only-last transit so every class reaches its columns;
  CCT spine rows stay class-agnostic.

**Step 3: Verify the `.c` compiles**

Run: `gcc -I src/mlir/runtime -fsyntax-only src/mlir/runtime/aie_runtime_control_plan.c`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.c src/mlir/runtime/aie_runtime_control_plan.h
git commit -m "feat(control-plan): plan_row_add threads add-order row index into slot ids"
```

---

## Task 4: Rewrite host unit test assertions

**Files:**
- Modify: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`

**Step 1: Update the extern decl + helpers**

The `extern "C"` block already declares `acr_plan_chain` / `acr_plan_row_add` with their
current signatures (unchanged). Keep `count_kind`, `count_master`, `count_slots`,
`master_mselen`. Add a helper to fetch a slot's fields by (col, slot):

```cpp
static const acr_op *find_slot(const acr_oplist *o, uint8_t col, uint8_t slot) {
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == ACR_OP_SLOT && o->ops[i].col == col && o->ops[i].slot == slot)
            return &o->ops[i];
    return nullptr;
}
```

**Step 2: Replace `test_slot_classes`**

A row_add chain cols 2..4 on the spine (head col 2). Head+interior (2,3) are
interior-type; col 4 is the last tile.

```cpp
static int test_slot_classes() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o = {};
    assert(acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, 0) == ACR_OK); // head row 3, cols 2..4, rowidx 0
    // Spine head (col 2): consume + bcast + transit-north + transit-east = 4 slots.
    assert(count_slots(&o, 2) == 4);
    // Interior non-head (col 3): consume + bcast + transit-east = 3 slots.
    assert(count_slots(&o, 3) == 3);
    // Last tile (col 4): only-last + whole-row + bcast = 3 slots.
    assert(count_slots(&o, 4) == 3);
    // CTRL masters: interior 0x3, last-tile 0x7.
    assert(master_mselen(&o, 2, ACR_CTRL) == 0x3);
    assert(master_mselen(&o, 3, ACR_CTRL) == 0x3);
    assert(master_mselen(&o, 4, ACR_CTRL) == 0x7);
    // Last tile has no EAST and no NORTH.
    assert(count_master(&o, 4, ACR_EAST) == 0);
    assert(count_master(&o, 4, ACR_NORTH) == 0);
    // Plain chain: interior tiles 3 slots, last-tile 3 slots, no NORTH, no transit-north, no CCT.
    acr_oplist op = {};
    acr_portbook pb = {};
    assert(acr_plan_chain(&op, &pb, 4, 2, 3, 0) == ACR_OK); // cols 2,3 ; rowidx 0
    assert(count_slots(&op, 2) == 3); // interior: consume+bcast+transit-east
    assert(count_slots(&op, 3) == 3); // last: only-last+whole-row+bcast
    assert(count_kind(&op, ACR_OP_CCT) == 0);
    for (int i = 0; i < op.n; i++)
        assert(!(op.ops[i].kind == ACR_OP_MASTER_EN && op.ops[i].mport == ACR_NORTH));
    return 0;
}
```

**Step 3: Replace `test_east_forward`**

Interior EAST MSelEn is now `0xB` (consume+bcast+transit-east); last tile no EAST.

```cpp
static int test_east_forward() {
    acr_oplist o = {};
    acr_portbook b = {};
    assert(acr_plan_chain(&o, &b, 4, 2, 4, 0) == ACR_OK); // cols 2,3,4
    for (uint8_t c = 2; c <= 3; c++) { // interior
        assert(count_master(&o, c, ACR_EAST) == 1);
        assert(master_mselen(&o, c, ACR_EAST) == 0xB);
    }
    assert(count_master(&o, 4, ACR_EAST) == 0); // last tile
    return 0;
}
```

**Step 4: Replace `test_slot_table`**

Verify each slot's pkt_id/mask/msel per role (rowidx 0). Head col 2, interior col 3,
last col 4.

```cpp
static int test_slot_table() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o = {};
    assert(acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, 0) == ACR_OK); // rowidx 0
    // Interior consume slot (col 3): pkt == rowidx(0), mask 0x17, msel 0.
    const acr_op *cons = find_slot(&o, 3, ACR_SLOT_CONSUME);
    assert(cons && cons->pkt_id == 0 && cons->mask == ACR_MASK_CONSUME && cons->msel == ACR_MSEL_CONSUME);
    // Interior transit-east (col 3): class 01 => pkt 4, exact mask, msel 3.
    const acr_op *te = find_slot(&o, 3, ACR_SLOT_TRANSIT_E);
    assert(te && te->pkt_id == ((ACR_CLASS_ONLY_LAST << 2) | 0) && te->mask == ACR_MASK_EXACT &&
           te->msel == ACR_MSEL_TRANSIT_E);
    // Head transit-north (col 2): pkt 0x00, mask 0x10, msel 2.
    const acr_op *tn = find_slot(&o, 2, ACR_SLOT_TRANSIT_N);
    assert(tn && tn->pkt_id == ACR_ID_TRANSIT && tn->mask == ACR_MASK_CLASS && tn->msel == ACR_MSEL_TRANSIT_N);
    // Last tile (col 4): only-last pkt 4 exact msel0 ; whole-row pkt 8 exact msel1 ; bcast msel2.
    const acr_op *ol = find_slot(&o, 4, ACR_SLOT_ONLY_LAST);
    assert(ol && ol->pkt_id == ((ACR_CLASS_ONLY_LAST << 2) | 0) && ol->mask == ACR_MASK_EXACT &&
           ol->msel == ACR_MSEL_ONLY_LAST);
    const acr_op *wh = find_slot(&o, 4, ACR_SLOT_WHOLE);
    assert(wh && wh->pkt_id == ((ACR_CLASS_WHOLE_ROW << 2) | 0) && wh->mask == ACR_MASK_EXACT &&
           wh->msel == ACR_MSEL_WHOLE);
    const acr_op *bl = find_slot(&o, 4, ACR_SLOT_BCAST_LAST);
    assert(bl && bl->pkt_id == ACR_ID_BCAST && bl->mask == ACR_MASK_BCAST && bl->msel == ACR_MSEL_BCAST_LAST);
    // Last tile does NOT arm an all-but-last consume (mask 0x17) slot.
    for (int i = 0; i < o.n; i++)
        assert(!(o.ops[i].kind == ACR_OP_SLOT && o.ops[i].col == 4 && o.ops[i].mask == ACR_MASK_CONSUME));
    return 0;
}
```

**Step 5: Update `test_spine_reuse` + `test_shared_head_fanout`**

Both assert head NORTH `MSelEn == 0x6` — unchanged value, keep those asserts. Only ensure
they still call `acr_plan_row_add` with a second row that keeps rowidx <= 3 (they add 2-3
rows; fine). Update any comment that says "row 3 slot id == row" to "consume pkt == row
index". Verify the second-row rowidx: `test_spine_reuse` adds row 3 then row 5 — indices 0
then 1 — both <= 3. `test_shared_head_fanout` adds row 3 then row 5 — same. No numeric
assert on the consume pkt in those tests, so no change needed beyond comments.

**Step 6: Run the host unit test**

Run the build/run block from the plan header.
Expected: `PASS`.

**Step 7: Commit**

```bash
git add src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
git commit -m "test(control-plan): assert column-subset slot tables (interior/last/head)"
```

---

## Task 5: Runtime send API — three class wrappers

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c:3919-3971` (replace `__Runtime_ctrl_row_multicast_write`)
- Modify: `src/mlir/runtime/aie_runtime.h:912-921`

**Step 1: Add a physical-row → index resolver + core send in `aie_runtime.c`**

Replace `__Runtime_ctrl_row_multicast_write` (`:3919-3971`) with a static row-index
resolver, a static core send, and three wrappers. The core send mirrors the existing
multicast body but takes a full `send_id`:

```c
/* Resolve a configured physical @row to its add-order index (0..3), or -1. */
static int rt_ctrl_row_index(const __Runtime_CtrlRowFabric *f, uint8_t row) {
    for (uint8_t i = 0; i < f->nrows; i++)
        if (f->rows[i].row == row)
            return (int)i;
    return -1;
}

/* Fire-and-forget WRITE with an explicit 5-bit @send_id to @row's chain. */
static AieRC rt_ctrl_row_send(__Runtime_CtrlRowFabric *f, uint8_t row, uint8_t send_id, uint32_t tile_addr,
                              const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch, int log) {
    if (!f || !f->dev || !data || nwords == 0U || f->nrows == 0U)
        return XAIE_INVALID_ARGS;
    AieRC rc = rt_ctrl_row_shim_entry(f, mm2s_ch);
    if (rc != XAIE_OK)
        return rc;
    uint32_t cap = nwords * 2U + 8U;
    uint32_t *pkt = (uint32_t *)__Runtime_alloc_buffer(f->dev, (size_t)cap * sizeof(uint32_t));
    if (!pkt) {
        printf("[aie_runtime] ctrl_row_send ERROR: request buffer alloc failed\n");
        return XAIE_ERR;
    }
    uint32_t pw = __Runtime_ctrl_pktize_write(pkt, cap, send_id, tile_addr, data, nwords,
                                              /*lastwriteack=*/0, /*ret_stream_id=*/0U, NULL);
    if (pw == 0U) {
        __Runtime_free_buffer(f->dev, pkt);
        return XAIE_ERR;
    }
    __Runtime_sync_for_dev(f->dev, pkt, (size_t)pw * sizeof(uint32_t));
    __Runtime_CtrlInstance inst = {
        .dev = f->dev,
        .shim_col = f->shim_col,
        .dest_col = f->rows[f->nrows - 1U].col_lo,
        .dest_row = row,
        .stream_id = send_id,
        .bd_id = bd_id,
        .mm2s_ch = mm2s_ch,
        .s2mm_ch = 0,
        .token = NULL,
        .resp_words = 0U,
    };
    rc = __Runtime_ctrl_push(&inst, pkt, pw, /*block=*/0, log);
    __Runtime_free_buffer(f->dev, pkt);
    return rc;
}

/* Compose id = (class<<2)|rowidx for a configured @row and send. */
static AieRC rt_ctrl_row_class_write(__Runtime_CtrlRowFabric *f, uint8_t cls, uint8_t row, uint32_t tile_addr,
                                     const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch, int log) {
    int idx = rt_ctrl_row_index(f, row);
    if (idx < 0) {
        printf("[aie_runtime] ctrl_row class_write: row=%u not configured\n", (unsigned)row);
        return XAIE_INVALID_ARGS;
    }
    uint8_t send_id = (uint8_t)((cls << 2) | ((uint8_t)idx & 0x3U));
    return rt_ctrl_row_send(f, row, send_id, tile_addr, data, nwords, bd_id, mm2s_ch, log);
}

AieRC __Runtime_ctrl_row_all_but_last_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                            const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                            int log) {
    return rt_ctrl_row_class_write(f, ACR_CLASS_ALL_BUT_LAST, row, tile_addr, data, nwords, bd_id, mm2s_ch, log);
}

AieRC __Runtime_ctrl_row_only_last_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                         const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                         int log) {
    return rt_ctrl_row_class_write(f, ACR_CLASS_ONLY_LAST, row, tile_addr, data, nwords, bd_id, mm2s_ch, log);
}

AieRC __Runtime_ctrl_row_whole_row_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                         const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                         int log) {
    return rt_ctrl_row_class_write(f, ACR_CLASS_WHOLE_ROW, row, tile_addr, data, nwords, bd_id, mm2s_ch, log);
}
```

Each function stays well under 200 lines (CLAUDE.md rule).

**Step 2: Update prototypes in `aie_runtime.h`**

Replace the `__Runtime_ctrl_row_multicast_write` prototype + comment (`:912-921`) with:

```c
// Column-subset row-multicast WRITE (fire-and-forget, block=0). The 5-bit stream
// id = (class<<2) | rowindex, rowindex = @row's add-order index (0..3). Requires a
// prior __Runtime_ctrl_row_add for @row.
//   all_but_last: every column of @row EXCEPT col_hi
//   only_last   : only col_hi of @row
//   whole_row   : every column of @row incl. col_hi (old row-multicast)
AieRC __Runtime_ctrl_row_all_but_last_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                            const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                            int log);
AieRC __Runtime_ctrl_row_only_last_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                         const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                         int log);
AieRC __Runtime_ctrl_row_whole_row_write(__Runtime_CtrlRowFabric *f, uint8_t row, uint32_t tile_addr,
                                         const uint32_t *data, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                         int log);
```

`__Runtime_ctrl_row_broadcast_write` (`:909-910`) is unchanged.

**Step 3: Grep for stale references**

Run: `grep -rn "ctrl_row_multicast_write" src example`
Expected: matches ONLY in the demo (fixed in Task 6) — none left in `aie_runtime.c` /
`aie_runtime.h`.

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(runtime): column-subset row sends (all-but-last/only-last/whole-row)"
```

---

## Task 6: Update the demo `ctrlrow_demo.cc`

**Files:**
- Modify: `example/tileprogram/ccode/ctrlrow_demo.cc`

**Step 1: Replace the probe helper**

Replace `demo_probe_row` (multicast + cross-row verify) with `demo_probe_class(fab, dev,
row, cls, val, col_lo, col_hi)` that calls the right class wrapper and verifies the exact
column subset via `XAie_DataMemBlockRead`:
- `ACR_CLASS_ALL_BUT_LAST`: expect `val` on cols `col_lo..col_hi-1`, unchanged on `col_hi`.
- `ACR_CLASS_ONLY_LAST`: expect `val` only on `col_hi`.
- `ACR_CLASS_WHOLE_ROW`: expect `val` on all cols `col_lo..col_hi`.
Read back with a small poll loop (reuse the existing DIAG read pattern already in the file).

**Step 2: Update call sites + defines**

Replace the two `demo_probe_row(...)` calls with per-class probes on `DEMO_CORE_ROW_A`
(e.g. whole-row with `DEMO_MC_VAL_A`, only-last, all-but-last with distinct sentinels).
Keep the broadcast sentinel test. Add `#include`/constants as needed; drop unused
`DEMO_MC_*` that no longer apply, add `DEMO_ABL_VAL` / `DEMO_OL_VAL` sentinels.

**Step 3: Build the runtime with the demo (env-dependent)**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc`
Expected: builds clean (no compile errors referencing the removed multicast symbol).

**Step 4: Commit**

```bash
git add example/tileprogram/ccode/ctrlrow_demo.cc
git commit -m "example(ctrlrow): probe all-but-last/only-last/whole-row column subsets"
```

---

## Task 7: Docs

**Files:**
- Modify: `docs/plans/2026-09-08-row-control-connection.md:25-71` (the 2026-09-12 update section)
- Modify: `CLAUDE.md:47` (control-plane paragraph)

**Step 1: Update the design-doc update section**

Replace the "two-mode superset" slot table + master list with the three-class tables from
`docs/plans/2026-09-12-row-control-column-subset-design.md` (interior/last split, id
`[3:2]`=class + `[1:0]`=row index, EAST `0xB`, last-tile CTRL `0x7`). Keep the multi-match
note (now: `00`/`10` at target head match consume + transit-north; robust either way).

**Step 2: Update the CLAUDE.md control-plane paragraph**

Rewrite the "persistent two-mode slot superset" sentence group to describe the three
column-subset classes, the id `[3:2]`=class / `[1:0]`=row-index (add-order, ≤4 rows)
scheme, the interior vs last-tile slot tables, masters (CTRL `0x3`/`0x7`, EAST `0xB`,
NORTH `0x6`), and the send API (`_all_but_last_write` / `_only_last_write` /
`_whole_row_write` + unchanged `_broadcast_write`).

**Step 3: Re-run the host unit test (guard against doc-time regressions)**

Run the build/run block from the plan header. Expected: `PASS`.

**Step 4: Commit**

```bash
git add docs/plans/2026-09-08-row-control-connection.md CLAUDE.md
git commit -m "docs: column-subset row-control classes (slot tables + send API)"
```

---

## Task 8: End-to-end verification (env-dependent)

**Files:** none (verification only).

**Step 1: E2E on sim/HW**

Run: `python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1; tail -30 applog`
Expected: `device_teardown done`, no `AIE ERROR`. Exercise (via the demo): broadcast,
whole-row, only-last (only col_hi changes), all-but-last (col_hi unchanged), and a class
targeting a row above an intervening head (spine climb).

**Step 2: pmap overlay check (aiedebug Load control plan)**

Each spine head shows 4 slots (consume `K·0x17·msel0`, bcast `0x10·0x10·msel1`,
transit-north `0x00·0x10·msel2`, transit-east `(4|K)·0x1F·msel3`); interior non-head 3
slots (no transit-north); last tile 3 slots (only-last `4|K`, whole-row `8|K`, bcast).
CTRL `0x3`/`0x7`, EAST `0xB`, NORTH `0x6`.

---

## Notes for the executor

- **DRY:** `rt_ctrl_row_send` is the single send body; the three wrappers only pick the
  class. Do not duplicate the pktize/push logic.
- **YAGNI:** do not implement the reserved class `11` or >4 rows.
- **Slot budget:** the spine head arms exactly 4 slots — if you add a slot, something must
  give. Re-check `ACR_NUM_SLOTS`.
- **200-line rule:** keep every runtime function under 200 lines (CLAUDE.md).
- **Per CLAUDE.md:** after each task, list changed/created files.
