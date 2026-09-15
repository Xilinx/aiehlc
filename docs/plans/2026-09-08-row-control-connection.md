# Row-Based Control Connection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stateful runtime API that configures a control connection across a
row of AIE tiles (shim vertical spine + EAST daisy-chain), where each tile can
consume at CTRL, forward EAST, or both (broadcast), with single-target
read/ack responses drained to a fixed memtile, and incremental row-add reusing
the shared spine.

**Architecture:** Split the risky logic into a **pure planner** (no XAie calls:
derives per-tile packet-switch ops and spine hops from high-level intent, and
books ports) that is host-unit-testable, plus a **thin emit layer** that
translates planner output into `XAie_StrmPktSw*` / `XAie_StrmConnCctEnable`
calls and is verified on sim/HW. State lives in a `__Runtime_CtrlRowFabric`
instance that owns the resource booking.

**Tech Stack:** C (runtime, `src/mlir/runtime/aie_runtime*.c`), XAie driver
(aie-rt), host C++ for unit tests, `apppaltest.py` for HW/sim E2E.

**Design doc:** `docs/plans/2026-09-08-row-control-connection-design.md`

---

## Update (2026-09-12): column-subset row-multicast classes (supersedes two-mode)

The per-tile derivation arms a **persistent column-subset slot superset** on every
tile's ingress slave port; the runtime packet **id** then selects the delivery mode
**and target row** at send time (no per-chain `target_col`). Unicast is **removed**.
A row-multicast can target a **column subset** of a row, keyed by `id[3:2]`:
- **all-but-last** (class `00`): every column of the target row **except** `col_hi`.
- **only-last** (class `01`): **only** `col_hi`.
- **whole-row** (class `10`): every column **including** `col_hi` (old row-multicast).

Broadcast (`id[4]=1`) is unchanged (every column of every configured row).
`acr_plan_chain_ex` takes `int shim_col` (‑1 ⇒ plain chain, no NORTH climb) and a
`uint8_t rowidx`; `acr_plan_chain` passes `rowidx=0`.

**Reserved-bit 5-bit stream id:** `[4]=broadcast marker`; when `[4]=0`,
`[3:2]=class`, `[1:0]=target row index` (add-order, `0..3`, up to 4 rows —
`ACR_MAX_ROW_IDX`). `id = (class<<2)|rowidx`. `acr_plan_row_add` assigns each
configured row its add-order index (`s->nrows`); the runtime resolves a physical
row → index the same way (`rt_ctrl_row_index`).

**Interior tile** (`col < col_hi`, includes the spine head):

| MSel | Slot class | PktId | Mask | Feeds masters |
|---|---|---|---|---|
| 0 | consume `{00,10}` @row K | `K` (rowidx) | `0x17` (match `[4]=0,[2]=0,[1:0]=K`; ignore `[3]`) | CTRL + EAST |
| 1 | broadcast | `0x10` | `0x10` | CTRL + EAST (+NORTH on head) |
| 2 | transit-north *(spine head only)* | `0x00` | `0x10` (any row-mcast) | NORTH climb |
| 3 | transit-east `{01}` @row K | `4\|K` | `0x1F` exact | EAST only (forward past, not consumed) |

Mask `0x17` covers all-but-last (`00`) and whole-row (`10`) for row K (both have
`id[2]=0`); only-last (`01`, `id[2]=1`) is excluded from CTRL/consume and instead
rides the transit-east slot (EAST only) through interior tiles to `col_hi`.

**Last tile** (`col_hi`; no EAST, no NORTH):

| MSel | Slot class | PktId | Mask | Feeds masters |
|---|---|---|---|---|
| 0 | only-last `{01}` @row K | `4\|K` | `0x1F` exact | CTRL |
| 1 | whole-row `{10}` @row K | `8\|K` | `0x1F` exact | CTRL |
| 2 | broadcast | `0x10` | `0x10` | CTRL |

Two separate exact consume slots because `01` and `10` share no maskable bits; the
last tile arms **no** all-but-last (`00`) slot.

**Per-tile masters** (all `keep_header=1`, arbiter 0):
- Interior **CTRL** `MSelEn = 0x3` (consume + broadcast).
- Interior **EAST** `MSelEn = 0x0B` (consume + broadcast + transit-east) — forwards
  **all** classes east so any subset reaches its columns.
- Interior **NORTH** (spine head only) `MSelEn = 0x6` (broadcast + transit-north climb).
- Last-tile **CTRL** `MSelEn = 0x7` (only-last + whole-row + broadcast).

**Slot budget:** spine head arms 4 (consume, bcast, transit-north, transit-east);
other interior arm 3; last tile arms 3; a single-column chain (head == last) arms
only-last + whole-row + bcast + transit-north = 4.

`acr_plan_row_add` passes `shim_col = col_lo` so the head emits its own NORTH climb
master (`0x6`). Spine extension through an already-configured head row emits nothing
(that head climbs via its own NORTH master); pure pass-through spine rows stay
circuit `ACR_OP_CCT` (class-agnostic — every class + broadcast climb through
unchanged).

**Multi-match assumption (sim/HW):** at a target head K a `00`/`10` packet matches
both the consume slot (CTRL+EAST) and transit-north — robust to either HW
slot-priority rule (consumed here; if it also climbs, higher heads have a different
K and drop it). only-last (`01`) at head K matches transit-east (EAST) and, if
present, transit-north — it rides east to `col_hi` and any harmless climb drops at
higher heads. Broadcast never multi-matches (bit 4 partitions it).

Constants live in `aie_runtime_control_plan.h` (`ACR_MAX_ROW_IDX`, `ACR_CLASS_*`,
interior/last `ACR_SLOT_*` + `ACR_MSEL_*`, `ACR_ARB_CTRL`, `ACR_ID_*`, `ACR_MASK_*`
incl. `ACR_MASK_CONSUME 0x17`, `ACR_MSELEN_*`). The emit layer
(`__Runtime_ctrl_row_emit`) is unchanged — it forwards slot/mask/msel/arbiter/mselen
generically, and the pmap provenance (`rt_pmap_port_ex`) surfaces arb/msel/mask
automatically. Runtime send: `__Runtime_ctrl_row_{all_but_last,only_last,whole_row}_write`
resolve `row → rowidx` and send id `(class<<2)|rowidx`; `broadcast_write` unchanged.

---

## Conventions

- Tile coords are `XAie_TileLoc(Col, Row)`.
- The pure planner lives in a **new** file pair so it compiles standalone for
  unit tests without pulling the full XAie device: `aie_ctrl_row_plan.h` /
  `aie_ctrl_row_plan.c`.
- The emit layer + public API live in the existing runtime translation unit
  (`aie_runtime.c`) with the struct in `aie_runtime.h`.
- Every task is TDD where the logic is pure; emit/HW tasks verify on sim.
- Commit after every green step.

---

## Task 1: Planner data types (pure, no XAie)

**Files:**
- Create: `src/mlir/runtime/aie_ctrl_row_plan.h`

**Step 1: Write the header with plain-data types**

```c
#ifndef AIE_CTRL_ROW_PLAN_H
#define AIE_CTRL_ROW_PLAN_H
#include <stdint.h>

#define ACR_MAX_ROWS      16
#define ACR_MAX_OPS       256
#define ACR_NUM_SLOTS      4   /* NumSlaveSlots per port (xaie_ss.c:755) */

/* Direction/port tags, decoupled from XAie enums for pure testing. */
typedef enum { ACR_WEST, ACR_EAST, ACR_NORTH, ACR_SOUTH, ACR_CTRL } acr_port;

/* One emitted stream-switch operation (translated to XAie by the emit layer). */
typedef enum { ACR_OP_SLOT, ACR_OP_SLAVE_EN, ACR_OP_MASTER_EN, ACR_OP_CCT } acr_op_kind;
typedef struct {
    acr_op_kind kind;
    uint8_t col, row;
    acr_port sport;   uint8_t sidx;   /* slave side */
    acr_port mport;   uint8_t midx;   /* master side */
    uint8_t  slot, pkt_id, mask, msel, arbiter, mselen;
    uint8_t  keep_header;             /* 1 => DONOT_DROP_HEADER */
} acr_op;

typedef struct { acr_op ops[ACR_MAX_OPS]; int n; } acr_oplist;

/* Per-tile used-port bitmaps for conflict rejection (requirement #7). */
typedef struct {
    uint16_t master_used[ACR_MAX_ROWS+1][64]; /* [row][col] bit i => master idx i used */
    uint16_t slave_used [ACR_MAX_ROWS+1][64];
} acr_portbook;

typedef enum { ACR_OK=0, ACR_ERR_PORT_CONFLICT, ACR_ERR_SLOTS, ACR_ERR_BOUNDS,
               ACR_ERR_TILETYPE } acr_rc;
#endif
```

**Step 2: Verify it compiles standalone**

Run: `gcc -c -I. src/mlir/runtime/aie_ctrl_row_plan.h -o /tmp/claude/acr_hdr.o 2>&1 || gcc -fsyntax-only -x c src/mlir/runtime/aie_ctrl_row_plan.h`
Expected: no errors.

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_ctrl_row_plan.h
git commit -m "feat(ctrl-row): planner plain-data types"
```

---

## Task 2: Port booking (pure, TDD)

**Files:**
- Create: `src/mlir/runtime/aie_ctrl_row_plan.c`
- Test: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`

**Step 1: Write the failing test**

```cpp
// test_ctrl_row_plan.cpp
#include <cassert>
extern "C" {
#include "aie_ctrl_row_plan.h"
acr_rc acr_book_port(acr_portbook*, uint8_t col, uint8_t row, acr_port, uint8_t idx, int is_master);
}
int test_book() {
    acr_portbook b = {};
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 0, /*master*/1) == ACR_OK);
    // same port+idx again => conflict
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 0, 1) == ACR_ERR_PORT_CONFLICT);
    // different idx ok
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 1, 1) == ACR_OK);
    return 0;
}
int main(){ return test_book(); }
```

**Step 2: Run to verify it fails (link error / missing symbol)**

Run: `g++ -I src/mlir/runtime src/mlir/runtime/unitest/test_ctrl_row_plan.cpp src/mlir/runtime/aie_ctrl_row_plan.c -o /tmp/claude/acr_test 2>&1 | head`
Expected: FAIL — `acr_book_port` undefined (file not yet created / empty).

**Step 3: Implement minimal `acr_book_port`**

```c
// aie_ctrl_row_plan.c
#include "aie_ctrl_row_plan.h"
static uint16_t *slot_for(acr_portbook *b, uint8_t col, uint8_t row, int is_master) {
    uint16_t (*t)[64] = is_master ? b->master_used : b->slave_used;
    return &t[row][col];
}
acr_rc acr_book_port(acr_portbook *b, uint8_t col, uint8_t row, acr_port p,
                     uint8_t idx, int is_master) {
    (void)p; /* idx is unique per (col,row,is_master) domain for this API */
    uint16_t *cell = slot_for(b, col, row, is_master);
    if (*cell & (1u << idx)) return ACR_ERR_PORT_CONFLICT;
    *cell |= (1u << idx);
    return ACR_OK;
}
```

**Step 4: Run to verify pass**

Run: `g++ -I src/mlir/runtime src/mlir/runtime/unitest/test_ctrl_row_plan.cpp src/mlir/runtime/aie_ctrl_row_plan.c -o /tmp/claude/acr_test && /tmp/claude/acr_test && echo PASS`
Expected: `PASS`

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_ctrl_row_plan.c src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
git commit -m "feat(ctrl-row): port booking with conflict rejection"
```

---

## Task 3: Per-tile consume/forward/broadcast derivation (pure, TDD)

**Files:**
- Modify: `src/mlir/runtime/aie_ctrl_row_plan.c`
- Test: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp` (append)

The function derives the ops for one EAST chain given: `row`, `col_lo`, `col_hi`,
the target column(s), and mode. Encodes the §5 table.

**Step 1: Write failing tests for the three cases**

```cpp
extern "C" acr_rc acr_plan_chain(acr_oplist*, acr_portbook*,
    uint8_t row, uint8_t col_lo, uint8_t col_hi,
    int target_col /* -1 => broadcast all */, uint8_t ctrl_id);

static int count(const acr_oplist*o, acr_op_kind k, uint8_t col, acr_port mport){
    int c=0; for(int i=0;i<o->n;i++) if(o->ops[i].kind==k && o->ops[i].col==col &&
        (k!=ACR_OP_MASTER_EN || o->ops[i].mport==mport)) c++; return c; }

int test_unicast() {           // target is col_lo+1; col_lo forwards, target consumes
    acr_oplist o={}; acr_portbook b={};
    assert(acr_plan_chain(&o,&b, 4, 2, 3, /*target*/3, 5)==ACR_OK);
    // col 2 = forward-only: has EAST master, no CTRL master
    assert(count(&o,ACR_OP_MASTER_EN,2,ACR_EAST)==1);
    assert(count(&o,ACR_OP_MASTER_EN,2,ACR_CTRL)==0);
    // col 3 = consume-only: CTRL master keep_header, no EAST master
    assert(count(&o,ACR_OP_MASTER_EN,3,ACR_CTRL)==1);
    assert(count(&o,ACR_OP_MASTER_EN,3,ACR_EAST)==0);
    return 0;
}
int test_broadcast() {          // all tiles consume+forward except last (consume only)
    acr_oplist o={}; acr_portbook b={};
    assert(acr_plan_chain(&o,&b, 4, 2, 4, /*broadcast*/-1, 5)==ACR_OK);
    for (uint8_t c=2;c<=3;c++){   // interior: CTRL + EAST both on arb0/msel0
        assert(count(&o,ACR_OP_MASTER_EN,c,ACR_CTRL)==1);
        assert(count(&o,ACR_OP_MASTER_EN,c,ACR_EAST)==1);
    }
    assert(count(&o,ACR_OP_MASTER_EN,4,ACR_CTRL)==1);  // last consumes
    assert(count(&o,ACR_OP_MASTER_EN,4,ACR_EAST)==0);  // no further forward
    return 0;
}
int test_msel_rule() {          // every master MSelEn == 1<<msel of its slot
    acr_oplist o={}; acr_portbook b={};
    acr_plan_chain(&o,&b, 4, 2, 3, 3, 5);
    for(int i=0;i<o.n;i++) if(o.ops[i].kind==ACR_OP_MASTER_EN)
        assert(o.ops[i].mselen == (uint8_t)(1u<<o.ops[i].msel));
    return 0;
}
```
Update `main()` to call all tests.

**Step 2: Run — expect FAIL (undefined `acr_plan_chain`)**

Run: same g++ command as Task 2.
Expected: FAIL undefined symbol.

**Step 3: Implement `acr_plan_chain`**

Encode §5. For each column `c` in `[col_lo..col_hi]`:
- `is_target = (target_col<0) || (c==target_col)`
- `has_east_target = (target_col<0 && c<col_hi) || (target_col>=0 && c<target_col)`
- Emit incoming slave slot + slave-port-enable on `ACR_WEST` (except the head tile,
  whose incoming is the spine — handled by the spine planner in Task 4; here for
  `c==col_lo` still emit a WEST slot representing the spine tap input).
- If `is_target`: emit `ACR_OP_MASTER_EN` `ACR_CTRL idx0 keep_header=1 arb0 msel0
  mselen=1`.
- If `has_east_target`: emit `ACR_OP_MASTER_EN` `ACR_EAST idxE keep_header=1`. For
  **broadcast** interior tiles reuse `arb0/msel0/mselen=1` (consume+forward share
  the slot); for **forward-only** unicast pass-through use `arb1/msel1/mselen=2`
  with the slot's `msel=1`.
- Book every master/slave idx via `acr_book_port`; propagate conflict/bounds/slot
  errors. Return `ACR_ERR_SLOTS` if a port would need >4 slots.

Keep the function < 200 lines (CLAUDE.md rule).

**Step 4: Run — expect PASS**

Run: same g++ command; expect `PASS`.

**Step 5: Commit**

```bash
git add -A src/mlir/runtime/aie_ctrl_row_plan.c src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
git commit -m "feat(ctrl-row): derive per-tile consume/forward/broadcast ops"
```

---

## Task 4: Spine planner + idempotent row-add (pure, TDD)

**Files:**
- Modify: `src/mlir/runtime/aie_ctrl_row_plan.c`
- Test: append to `test_ctrl_row_plan.cpp`

Models the shared vertical spine and idempotence (requirement #5). A pure
`acr_plan_state` mirrors the instance's spine fields.

**Step 1: Failing test**

```cpp
typedef struct { uint8_t spine_top; uint8_t rows[ACR_MAX_ROWS]; uint8_t nrows; } acr_state;
extern "C" acr_rc acr_plan_row_add(acr_state*, acr_oplist*, acr_portbook*,
    uint8_t shim_col, uint8_t row, uint8_t col_lo, uint8_t col_hi, uint8_t ctrl_id);

int test_spine_reuse() {
    acr_state s={}; acr_portbook b={};
    acr_oplist o1={}; acr_plan_row_add(&s,&o1,&b, 2, 3, 2, 4, 5); // rows 1..3 spine
    int cct1 = 0; for(int i=0;i<o1.n;i++) if(o1.ops[i].kind==ACR_OP_CCT) cct1++;
    assert(cct1 > 0);
    acr_oplist o2={}; acr_plan_row_add(&s,&o2,&b, 2, 3, 2, 4, 5); // re-add same row
    // idempotent: no new spine CCT hops, no new chain ops
    assert(o2.n == 0);
    acr_oplist o3={}; acr_plan_row_add(&s,&o3,&b, 2, 5, 2, 4, 5); // higher row reuses spine 1..3
    int cct3 = 0; for(int i=0;i<o3.n;i++) if(o3.ops[i].kind==ACR_OP_CCT) cct3++;
    assert(cct3 == 2); // only rows 4,5 spine hops added
    return 0;
}
```

**Step 2: Run — FAIL undefined.**

**Step 3: Implement `acr_plan_row_add`**
- If `row` already in `s->rows` → emit nothing, return `ACR_OK` (idempotent).
- If `row > s->spine_top`: emit `ACR_OP_CCT` pass-through (NORTH↔SOUTH `fwd_vc`)
  for each new spine row `spine_top+1..row`; set `s->spine_top=row`.
- Emit the spine→row head tap (WEST-in from spine to the left tile) then call
  `acr_plan_chain` for `[col_lo..col_hi]`.
- Record `row` in `s->rows`.

**Step 4: Run — PASS.**

**Step 5: Commit**
```bash
git add -A && git commit -m "feat(ctrl-row): spine planner with idempotent row reuse"
```

---

## Task 5: Emit layer — translate `acr_oplist` to XAie calls

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c`
- Modify: `src/mlir/runtime/aie_runtime.h` (declare instance struct + API)

**Step 1: Add `__Runtime_CtrlRowFabric` to `aie_runtime.h`** (from design §3), plus
API prototypes (design §4).

**Step 2: Implement `acr_emit(XAie_DevInst*, const acr_oplist*)`** mapping:
- `ACR_OP_SLOT` → `XAie_StrmPktSwSlaveSlotEnable(dev, loc, port(sport), sidx, slot,
  XAie_PacketInit(pkt_id,0), mask, msel, arbiter)`
- `ACR_OP_SLAVE_EN` → `XAie_StrmPktSwSlavePortEnable(dev, loc, port, sidx)`
- `ACR_OP_MASTER_EN` → `XAie_StrmPktSwMstrPortEnable(dev, loc, port(mport), midx,
  keep_header?XAIE_SS_PKT_DONOT_DROP_HEADER:XAIE_SS_PKT_DROP_HEADER, arbiter, mselen)`
- `ACR_OP_CCT` → `XAie_StrmConnCctEnable(dev, loc, port(sport), sidx, port(mport), midx)`

Map `acr_port`→`StrmSwPortType` (WEST/EAST/NORTH/SOUTH/CTRL). Return on first
non-`XAIE_OK`, logging `[aie_runtime] ctrl_row:` with tile coords.

**Step 3: Build check** — compile the runtime via the normal flow:

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc 2>&1 | tail -20`
Expected: builds without errors in `aie_runtime.c`.

**Step 4: Commit**
```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(ctrl-row): emit layer mapping planner ops to XAie calls"
```

---

## Task 6: Instance lifecycle — `row_open` / `row_close`

**Files:** Modify `src/mlir/runtime/aie_runtime.c`

**Step 1:** Implement `__Runtime_ctrl_row_open` — zero the fabric, store
`dev/shim_col/ctrl_id/resp_memtile/resp_s2mm_ch/resp_bd`, choose `fwd_vc/ret_vc`,
`spine_top=0`, `nrows=0`. Do not touch HW.

**Step 2:** Implement `__Runtime_ctrl_row_close` — free `resp_buf` if allocated
(`__Runtime_free_buffer`), zero state. (Route teardown: best-effort; document that
partition teardown clears switches.)

**Step 3:** Build check (Task 5 command). Commit.
```bash
git commit -am "feat(ctrl-row): instance open/close lifecycle"
```

---

## Task 7: `row_add` + `broadcast_write` (write-only path, sim-verified)

**Files:** Modify `src/mlir/runtime/aie_runtime.c`; add example
`example/tileprogram/ccode/ctrlrow_demo.cc`.

**Step 1:** Implement `__Runtime_ctrl_row_add` — mirror `acr_plan_row_add` against
the instance's real spine fields + `PortBook`, then `acr_emit`.

**Step 2:** Implement `__Runtime_ctrl_row_broadcast_write` — `acr_plan_chain`
(broadcast) for the row if needed, build words via `__Runtime_ctrl_pktize_write(...,
stream_id=f->ctrl_id, lastwriteack=0, ret_sid=0, ...)`, push via
`__Runtime_ctrl_push(inst-like, buf, n, block=0, log)`.

**Step 3:** Write `ctrlrow_demo.cc`: open fabric on a shim col, `row_add` one core
row, `broadcast_write` a sentinel to a scratch register on all tiles.

**Step 4: Sim/HW run**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc && python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1; tail -30 applog`
Expected: `device_teardown done`, no `AIE ERROR`.

**Step 5:** Commit.
```bash
git add -A && git commit -m "feat(ctrl-row): row_add + broadcast_write (write-only)"
```

---

## Task 8: `unicast_write` + `unicast_read` with memtile response

**Files:** Modify `src/mlir/runtime/aie_runtime.c`

**Step 1:** Implement `__Runtime_ctrl_row_unicast_write` — plan chain with
`target_col=col` (forward-only pass-through before it, consume at it), pktize
write, push `block=0`.

**Step 2:** Implement the **return path** helper — dest CTRL slave → WEST →
down `ret_vc` → `resp_memtile` S2MM. Arm the memtile S2MM with
`XAie_DmaChannelSetStartQueue(dev, resp_memtile, resp_s2mm_ch, DMA_S2MM, resp_bd,
1, XAIE_DISABLE)` + Finish-on-TLAST (pattern `aie_runtime.c:1148`). Allocate
`resp_buf` sized `resp_words`.

**Step 3:** Implement `__Runtime_ctrl_row_unicast_read` — pktize read
(`__Runtime_ctrl_pktize_read`, ret_sid=f->ctrl_id), setup return path, push
`block=1`, poll memtile drain, `rt_ctrl_read_extract` into `out`.

**Step 4: Sim/HW run** — extend `ctrlrow_demo.cc`: unicast_write a value to one
core tile, unicast_read it back via the memtile, print/compare.

Run: same as Task 7 Step 4.
Expected: readback equals written value; `device_teardown done`.

**Step 5:** Commit.
```bash
git add -A && git commit -m "feat(ctrl-row): unicast write/read with memtile response"
```

---

## Task 9: E2E verification + docs

**Files:** Modify `docs/plans/2026-09-08-row-control-connection-design.md` (status),
update `CLAUDE.md` control-packet section if API is user-facing, add skill note.

**Step 1:** Add a demo that: `row_add` row A, broadcast_write; `row_add` a higher
row B (asserts spine reuse — check log shows no re-emit of shared hops); unicast_read
a tile in each row from the memtile.

**Step 2: Run** the E2E (Task 7 Step 4 command); confirm both rows deliver and
readbacks match.

**Step 3:** Update the design doc status to "implemented", list changed/created
files per the Process-transparency rule. Commit.
```bash
git add -A && git commit -m "docs(ctrl-row): mark implemented; E2E demo + notes"
```

---

## Files created / modified (summary)

- Create: `src/mlir/runtime/aie_ctrl_row_plan.h`, `aie_ctrl_row_plan.c`
- Create: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`
- Create: `example/tileprogram/ccode/ctrlrow_demo.cc`
- Modify: `src/mlir/runtime/aie_runtime.h` (instance + API), `aie_runtime.c` (emit
  layer, lifecycle, write/read/broadcast, return path)
- Modify: `docs/plans/2026-09-08-row-control-connection-design.md`, `CLAUDE.md`

## Verification checklist

- [ ] Pure planner unit tests pass (`/tmp/claude/acr_test`): booking, unicast,
      broadcast, MSel rule, spine reuse/idempotence.
- [ ] Runtime builds via `script/aiehlc.sh`.
- [ ] Broadcast write E2E: `device_teardown done`, no `AIE ERROR`.
- [ ] Unicast read-back through memtile matches written value.
- [ ] Second-row add reuses spine (log shows no duplicate spine hops).
