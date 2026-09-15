# Design: column-subset row-multicast classes (row-control plane)

**Date:** 2026-09-12
**Status:** approved (brainstorm), ready for implementation plan
**Supersedes id scheme in:** `docs/plans/2026-09-08-row-control-connection.md` (2026-09-12 two-mode update)

## Goal

Extend the pure row-control planner so a non-broadcast (row-multicast) packet can
target a **column subset** of a row, keyed by the packet stream id. Three classes,
selected by `id[3:2]`:

- **all-but-last** (id 0..3): every column of the target row **except** the last
  (`col_hi`). The last tile does **not** accept.
- **only-last** (id 4..7): **only** the last column (`col_hi`) of the target row.
- **whole-row** (id 8..11): every column **including** the last (equivalent to the
  old row-multicast).

Broadcast (`id[4]=1`) is **unchanged**: every column (incl. last) of every
configured row.

## Id scheme (5-bit stream id)

| Bits | Field | Values |
|---|---|---|
| `id[4]` | broadcast marker | `1` ⇒ broadcast |
| `id[3:2]` | column-subset class (when `id[4]=0`) | `00`=all-but-last, `01`=only-last, `10`=whole-row, `11`=reserved |
| `id[1:0]` | target row **index** | `0..3` = Nth configured row in `row_add` order |

`id = (class << 2) | rowidx`. A fabric supports up to **4 rows**
(`ACR_MAX_ROW_IDX = 3`). The row **index** (not the physical row) is what travels
in `id[1:0]`; the planner assigns each configured row its add-order index, and the
runtime fabric maps a physical row → index the same way (add order in
`__Runtime_ctrl_row_add`).

## Confirmed HW model

`thirdparty/alib/aie-rt/driver/src/stream_switch/xaie_ss.c`:
- Slave slot `{PktId, Mask, MSel, Arbitor}` matches when `(id & Mask) == (PktId & Mask)`,
  injecting into `Arbitor` with select `MSel`.
- Master `{Arbitor, MSelEn}` pulls a matched packet iff `master.Arbitor == slot.Arbitor
  && ((MSelEn >> slot.MSel) & 1)`.
- Limits: `NumSlaveSlots = 4`, `MSel 0..3`, single shared arbiter 0, 5-bit id/mask.

## Per-tile slot tables

`K` = this tile's row index. Bit layout `[4]=bcast [3:2]=class [1:0]=row`.

### Interior tile (any column `< col_hi`, includes the spine head)

| MSel | Slot class | PktId | Mask | Feeds masters |
|---|---|---|---|---|
| 0 | consume `{00,10}` @row K | `K` | `0x17` (match `[4]=0,[2]=0,[1:0]=K`; ignore `[3]`) | CTRL + EAST |
| 1 | broadcast | `0x10` | `0x10` | CTRL + EAST (+NORTH on head) |
| 2 | transit-north *(spine head only)* | `0x00` | `0x10` (any row-mcast) | NORTH climb |
| 3 | transit-east `{01}` @row K | `4\|K` | `0x1F` exact | EAST only (forward past, not consumed) |

Mask `0x17` matches all-but-last (`00`) and whole-row (`10`) for row K because both
have `id[2]=0`; only-last (`01`) has `id[2]=1` and is excluded from CTRL/consume.
The **transit-east** slot re-admits only-last packets to the EAST master alone, so
they ride through interior tiles (not consumed) to `col_hi`.

### Last tile (`col_hi`; no EAST, no NORTH)

| MSel | Slot class | PktId | Mask | Feeds masters |
|---|---|---|---|---|
| 0 | only-last `{01}` @row K | `4\|K` | `0x1F` exact | CTRL |
| 1 | whole-row `{10}` @row K | `8\|K` | `0x1F` exact | CTRL |
| 2 | broadcast | `0x10` | `0x10` | CTRL |

Two separate exact consume slots because `01` and `10` share no maskable common
bits. The last tile does **not** arm an all-but-last (`00`) slot.

### Masters (all `keep_header=1`, arbiter 0)

- Interior CTRL `MSelEn = (1<<0)|(1<<1) = 0x3` (consume + broadcast).
- Interior EAST `MSelEn = (1<<0)|(1<<1)|(1<<3) = 0x0B` (consume + broadcast +
  transit-east) — forwards **all** classes east so any subset reaches its columns.
- Interior NORTH (spine head only) `MSelEn = (1<<1)|(1<<2) = 0x6` (broadcast +
  transit-north climb).
- Last-tile CTRL `MSelEn = (1<<0)|(1<<1)|(1<<2) = 0x7`.

## Slot budget

| Tile role | Slots | Fits (≤4) |
|---|---|---|
| spine head (interior + NORTH climb) | consume, bcast, transit-north, transit-east = **4** | yes (exactly) |
| other interior | consume, bcast, transit-east = **3** | yes |
| last (`col_hi`) | only-last, whole-row, bcast = **3** | yes |
| single-column chain (head == last) | only-last, whole-row, bcast, transit-north = **4** | yes |

For a single-column chain, "all-but-last" targets no tile, so that tile arms the
only-last + whole-row consume slots (plus its spine transit-north). It does not arm
an all-but-last slot — consistent with the class definition.

## Spine climb (reach any row)

The vertical spine is class-agnostic: pure pass-through rows stay circuit `ACR_OP_CCT`
(SOUTH→NORTH), carrying every id up unchanged; a configured head climbs via its own
NORTH master (broadcast + transit-north). A row-mcast for row index `K` therefore
climbs to head K, where the consume slot (`0x17`, row K) fans CTRL+EAST and — for
`00`/`10` — delivers the subset. only-last (`01`) at head K matches transit-east
(EAST only) and rides to `col_hi`.

**Multi-match:** at target head K a `00`/`10` packet matches both consume and
transit-north — robust to either HW slot-priority rule (consumed here; if it also
climbs, higher heads have a different K and drop it). Broadcast never multi-matches
(bit 4 partitions it).

## Runtime send API

Track configured rows in add-order in `__Runtime_CtrlRowFabric` (already stored) so a
physical row resolves to its index. Replace `__Runtime_ctrl_row_multicast_write`
with three thin wrappers over a core send:

- `__Runtime_ctrl_row_send(f, class, rowidx, tile_addr, data, nwords, bd_id, mm2s_ch, log)`
  — builds id `(class<<2)|(rowidx&3)`, fire-and-forget (`block=0`).
- `__Runtime_ctrl_row_all_but_last_write(f, row, ...)` — class `00`.
- `__Runtime_ctrl_row_only_last_write(f, row, ...)` — class `01`.
- `__Runtime_ctrl_row_whole_row_write(f, row, ...)` — class `10` (old multicast).
- `__Runtime_ctrl_row_broadcast_write` — unchanged (id `0x10`).

Each wrapper resolves `row` → `rowidx` via the fabric's add-order table (error if the
row is not configured or index > 3).

## Constants (`aie_runtime_control_plan.h`)

- `ACR_MAX_ROW_IDX 3`
- Classes: `ACR_CLASS_ALL_BUT_LAST 0`, `ACR_CLASS_ONLY_LAST 1`, `ACR_CLASS_WHOLE_ROW 2`
- Slot indices / MSel: interior `ACR_SLOT_CONSUME 0`, `ACR_SLOT_BCAST 1`,
  `ACR_SLOT_TRANSIT_N 2`, `ACR_SLOT_TRANSIT_E 3`; last-tile `ACR_SLOT_ONLY_LAST 0`,
  `ACR_SLOT_WHOLE 1`, `ACR_SLOT_BCAST_LAST 2` (MSel == slot index per tile type).
- Masks: `ACR_MASK_CONSUME 0x17`, `ACR_MASK_EXACT 0x1F`, `ACR_MASK_BCAST 0x10`,
  `ACR_MASK_CLASS 0x10`.
- Ids: `ACR_ID_BCAST 0x10`, `ACR_ID_TRANSIT 0x00`; class base `(class<<2)`.
- MSelEn: `ACR_MSELEN_CTRL 0x3`, `ACR_MSELEN_EAST 0x0B`, `ACR_MSELEN_NORTH 0x6`,
  `ACR_MSELEN_CTRL_LAST 0x7`.

## Testing

- **Host unit test** (`test_ctrl_row_plan.cpp`): interior consume mask `0x17` @ row
  index; transit-east slot present on interior, absent on last; last tile has two
  exact consume slots (`4|K`, `8|K`) and no all-but-last slot; spine head arms exactly
  4 slots; CTRL/EAST/NORTH MSelEn = `0x3`/`0x0B`/`0x6`; last-tile CTRL `0x7`.
- **Demo** (`ctrlrow_demo.cc`): for each class, send to a target row and verify via
  `XAie_DataMemBlockRead` that exactly the intended columns changed (all-but-last:
  cols `col_lo..col_hi-1`; only-last: `col_hi`; whole-row: all), and no other row
  changed.
- **E2E** `apppaltest.py` on sim/HW: broadcast + each class + a class targeting a row
  above an intervening head (spine climb).

## Out of scope

- >4 configured rows (id `[1:0]` cap) — documented `ACR_MAX_ROW_IDX`.
- Reserved class `11`.
- Row-control read-back (removed with unicast earlier).
