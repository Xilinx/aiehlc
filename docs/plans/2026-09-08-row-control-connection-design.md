# Design: Row-Based Control Connection Runtime API

**Status:** design approved (no code yet)
**Date:** 2026-09-08
**Scope:** A new *runtime* control-network API that configures a control connection
across a horizontal row of AIE tiles. A control packet climbs a shim vertical
spine to the row's left tile, then daisy-chains EAST tile-to-tile. Each tile can
(a) consume the packet at its CTRL port, (b) forward it to its right (EAST)
neighbor, or (c) both (broadcast). Single-target read / write-with-return
responses are collected at a fixed **memtile** location. The connection is a
stateful instance that owns its resource information and supports incrementally
adding new rows while reusing the shared vertical spine. Initial tile support:
**core** and **memtile** only.

Chosen approach: **A — extend `__Runtime_CtrlInstance` into a stateful row-fabric
instance** (see "Approaches considered").

---

## 1. Requirements

1. Configure a control connection for a specific **row**.
2. A packet can be **consumed** by the CTRL port **or forwarded** to the next
   right (EAST) neighbor (selective / unicast).
3. A packet can be **consumed and forwarded** to the next neighbor to support
   **broadcast**.
4. The connection handles the control-packet **response** to a **memtile**
   location (single-target read / write-with-return ack only).
5. Support **adding a new row** and **reusing** the existing connection.
6. Initially support **core** and **memtile** tiles only.
7. Maintain the **instance** holding the resource information.

### Locked design decisions (from brainstorming)

- **Row entry:** shim → vertical climb up the left column → row's leftmost tile →
  EAST daisy-chain.
- **Response model:** only single-target reads and write-with-return acks generate
  a response; it is routed back to **one fixed memtile** held in the instance.
  Broadcast writes are **write-only** (no response).
- **Reuse model:** one **shared vertical spine** per column, kept in the instance;
  adding a row taps the spine at that row's left tile and extends a new EAST chain.
- **API level:** **high-level intent** (unicast / broadcast); the runtime derives
  each tile's slot / arbiter / MSel and books ports in the instance.

---

## 2. Architecture

A stateful **row-fabric instance** owns:

- one **forward vertical spine** — shim(row 0) up a single column to the highest
  configured row, and
- N **EAST chains** — one per configured row, spreading right from the spine tap,
  plus
- one **return spine** — response path down the same column to a fixed memtile.

Requests are **packet-switched** control packets injected at the spine top and
daisy-chained EAST. Each tile either consumes at CTRL, forwards EAST, or both.
Single-target read / write-with-return responses turn WEST, drop down the return
spine, and drain into the fixed memtile S2MM.

Physical routing follows the existing runtime convention (`rt_ctrl_route_setup_col`,
`aie_runtime.c:3454`): **packet-switched** at taps (slave-slot + slave-port +
master enables) and **circuit-switched** on pure pass-through hops.

```
 shim(col c,row0) --fwd spine (vertical)--> row r left tile --EAST chain-->
                                              |  \consume at CTRL
                                              |   \forward EAST
 memtile(resp) <--return spine (vertical)-- WEST turn <-- dest CTRL slave (response)
```

---

## 3. Instance struct

```c
typedef struct {
    uint8_t  row;                  // AIE row of this EAST chain
    uint8_t  col_lo, col_hi;       // inclusive column span built on this row
    uint8_t  head_tapped;          // spine->row left-tile tap already emitted
} RowChain;

typedef struct {
    // per-tile used master/slave port-idx bitmaps, keyed by (col,row)
    // used to reject double-booking a physical port.
    ...
} PortBook;

typedef struct {
    XAie_DevInst *dev;             // partitioned device instance
    uint8_t  shim_col;             // vertical spine column (= row left edge)
    uint8_t  spine_top;            // highest row the fwd spine reaches (0 = none)
    uint32_t fwd_vc, ret_vc;       // vertical stream channels for fwd / return

    // fixed response sink:
    XAie_LocType resp_memtile;     // memtile that drains responses
    int32_t  resp_s2mm_ch;         // memtile S2MM channel
    int32_t  resp_bd;              // memtile S2MM BD
    uint32_t *resp_buf;            // DDR/local response buffer
    uint32_t resp_words;           // expected response length (0 => 1)

    RowChain rows[MAX_ROWS];       // configured EAST chains
    uint8_t  nrows;

    PortBook book;                 // resource booking (requirement #7)
    uint8_t  ctrl_id;              // 5-bit stream id used for consume-matching
} __Runtime_CtrlRowFabric;
```

`MAX_ROWS` is a small fixed cap (device rows). `PortBook` is the resource-info
owner satisfying requirement #7; it is consulted on every route emit and rejects
conflicts loudly.

---

## 4. API surface

```c
// Initialize the instance and the fixed memtile response sink. No HW config yet.
AieRC __Runtime_ctrl_row_open(__Runtime_CtrlRowFabric *f, XAie_DevInst *dev,
                              uint8_t shim_col, XAie_LocType resp_memtile,
                              int32_t resp_s2mm_ch, int32_t resp_bd,
                              uint8_t ctrl_id);

// Ensure the vertical spine reaches `row`, then build/extend the EAST chain
// [col_lo..col_hi] on that row. Idempotent: re-adding an existing row/segment
// is a no-op. Network only (no packets pushed).
AieRC __Runtime_ctrl_row_add(__Runtime_CtrlRowFabric *f,
                             uint8_t row, uint8_t col_lo, uint8_t col_hi);

// Single-target write (consume at exactly one tile).
AieRC __Runtime_ctrl_row_unicast_write(__Runtime_CtrlRowFabric *f,
                             uint8_t row, uint8_t col,
                             uint32_t addr, const uint32_t *data, uint32_t n);

// Single-target read; response drained at resp_memtile, extracted into out[].
AieRC __Runtime_ctrl_row_unicast_read(__Runtime_CtrlRowFabric *f,
                             uint8_t row, uint8_t col,
                             uint32_t addr, uint32_t n, uint32_t *out);

// Broadcast write to every tapped tile on `row` (consume+forward). Write-only.
AieRC __Runtime_ctrl_row_broadcast_write(__Runtime_CtrlRowFabric *f,
                             uint8_t row, uint32_t addr,
                             const uint32_t *data, uint32_t n);

// Tear down all routes, free resp_buf.
AieRC __Runtime_ctrl_row_close(__Runtime_CtrlRowFabric *f);
```

The write/read calls reuse the existing packet builders
(`__Runtime_ctrl_pktize_write` / `__Runtime_ctrl_pktize_read`,
`aie_runtime.c:3035/3134`) and push via `__Runtime_ctrl_push`
(`aie_runtime.c:3255`). `stream_id` passed to the pktizer must equal
`f->ctrl_id` so headers match the armed slave slots.

---

## 5. Per-tile config derivation (consume / forward / broadcast)

For each tile on a chain the runtime derives the packet-switch config from
whether the tile is a **target** and whether any tile further EAST is a target.

| Case | WEST slave slot(s) | Master(s) enabled |
|------|--------------------|-------------------|
| **Consume only** (unicast dest, nothing further east) | slot0 `id=ctrl_id` → `arb0/msel0` | CTRL master idx 0, `XAIE_SS_PKT_DONOT_DROP_HEADER` |
| **Forward only** (pass-through toward a dest further east) | slot1 `id=*` (mask covers forwarded ids) → `arb1/msel1` | EAST master, `DONOT_DROP_HEADER` |
| **Consume + forward** (broadcast) | slot0 `id=ctrl_id` → `arb0/msel0` | CTRL master **and** EAST master, both on `arb0/msel0`, `DONOT_DROP_HEADER` |

Hardware facts enforced by the derivation (verified against the aie-rt driver):

- **4 slots per slave port** (`NumSlaveSlots = 4`, `xaiemlgbl_reginit.c:2090`,
  `xaie_ss.c:755`). More than 4 distinct match ids on one port → error.
- **MSel ↔ MSelEn rule:** a master pulls a slot only when
  `arbiter_master == arbiter_slot` **and** `MSelEn & (1<<slot_msel) != 0`. The
  derivation always sets master `MSelEn = 1<<slot_msel`
  (`XAIE_SS_MASTER_PORT_MSELEN_*`, `xaie_ss.c:47-53`, ranges: arbiter 0..7,
  msel 0..3, MSelEn 0..0xF).
- **`DONOT_DROP_HEADER`** on every forwarding master (next tile must still match
  the 5-bit id) and at CTRL sinks (in-tile decoder reads addr/beats out of the
  preserved header).

Emitted primitives per tapped tile (same as
`routinghwlower.cpp:270-337` / `rt_ctrl_route_setup_col`):
`XAie_StrmPktSwSlaveSlotEnable` + `XAie_StrmPktSwSlavePortEnable` +
`XAie_StrmPktSwMstrPortEnable`. Pure pass-through hops use
`XAie_StrmConnCctEnable`.

---

## 6. Response-to-memtile path (single-target only)

Only `unicast_read` / write-with-return arms a return path:

1. Dest tile CTRL **slave** port emits the response packet
   (`XAie_StrmPktSwSlaveSlotEnable(CTRL,…, Mask=0)` +
   `XAie_StrmPktSwSlavePortEnable`), then a **WEST** master drives it back with
   `DONOT_DROP_HEADER`.
2. WEST turn along the row to the spine column, then **down the return spine**
   (`ret_vc`) via circuit-switched pass-through hops.
3. Into `resp_memtile` S2MM via `XAie_DmaChannelSetStartQueue(dev, mt,
   resp_s2mm_ch, DMA_S2MM, resp_bd, /*repeat=*/1, …)` with Finish-on-TLAST
   (pattern already used for mem-trace drain, `aie_runtime.c:1148`).
4. Host reads `resp_buf`; the per-access stream header is stripped by
   `rt_ctrl_read_extract` (`aie_runtime.c:3804`).

Broadcast writes skip all of the above and push with `block=0` (no response
drain).

> Single memtile S2MM drain BD ⇒ one response packet (a ≤4-word access not
> crossing a 128-bit boundary). Larger reads needing one BD per packet are out of
> scope for the first increment.

---

## 7. Reuse / spine state machine

`__Runtime_ctrl_row_add(f, row, col_lo, col_hi)`:

1. If `row > f->spine_top`: extend the forward spine (and return spine) from
   `spine_top+1 .. row` with circuit-switched pass-through hops; set
   `spine_top = row`.
2. If this row's left-tile tap is not yet emitted, emit the spine→row tap and
   mark `head_tapped`.
3. Build only the **new** EAST segment `[col_lo..col_hi]`, booking ports in
   `PortBook`.
4. Re-adding an already-built row/segment is **idempotent** (checked via
   `rows[]` + `PortBook`), so the shared spine hops are never re-emitted.

This directly realizes requirement #5 ("add a new row, reuse existing
connection").

---

## 8. Error handling & resource booking

- `PortBook` rejects any route that would reuse an occupied master/slave idx →
  return an error (never silently overwrite a port).
- Tile-type gate: only `Core` / `Mem` accepted as consume targets initially;
  shim / other tile types → `XAIE_INVALID_TILE`.
- `>4` distinct match slots required on one slave port → error.
- `col_hi` beyond device bounds, or an EAST chain crossing a memtile horizontal
  gap (memtiles do not route East/West per `hwresource.cpp`) → error.
- All emit helpers return `AieRC`; the first non-`XAIE_OK` aborts and is logged
  with tile coordinates (matching existing `[aie_runtime] ctrl_route:` style).

---

## 9. Testing

- **Unit (host, mockable emit layer):** feed a target set and assert the derived
  sequence of `SlaveSlotEnable` / `SlavePortEnable` / `MstrPortEnable` /
  `StrmConnCctEnable` calls matches the derivation table in §5 (consume /
  forward / consume+forward), including `MSelEn == 1<<msel` and header-keep flags.
- **HW / sim:**
  1. Unicast write + read one core tile; verify the readback lands in
     `resp_memtile` and matches.
  2. Broadcast write a row; read back each tile via unicast read to confirm all
     tapped tiles saw the write.
  3. Add a second row; assert the shared spine hops are **not** re-emitted
     (idempotence) and both rows deliver.

---

## 10. Approaches considered

- **A (chosen):** extend `__Runtime_CtrlInstance` into a stateful row-fabric
  instance. Only approach that satisfies all requirements — runtime instance owns
  resource info (#7), incremental row reuse via shared spine (#5), high-level
  consume/forward/broadcast + memtile response (#1–#4, #6).
- **B:** stateless helpers + caller-held config. Loses reuse/resource safety;
  weakly meets #5/#7.
- **C:** reuse the MLIR `StreamType::CONTROL` compile-time multicast
  (`docs/plans/2026-09-03-control-packet-anytile-multicast-design.md`). Not a
  *runtime* API; cannot add a row live. Useful only as a broadcast validation
  oracle.

---

## 11. File references (verified, no edits)

| Ref | What |
|-----|------|
| `include/../aie_runtime.h:757-769` | current `__Runtime_CtrlInstance` (single-target, same-column) |
| `aie_runtime.c:3454` | `rt_ctrl_route_setup_col` — packet-tap + circuit pass-through pattern |
| `aie_runtime.c:3552-3575` | CTRL slave response emit + `DONOT_DROP_HEADER` master |
| `aie_runtime.c:3583-3596` | vertical pass-through (circuit) hops |
| `aie_runtime.c:1148` | memtile S2MM `SetStartQueue` drain (response-sink pattern) |
| `aie_runtime.c:3035/3134/3255` | `pktize_write` / `pktize_read` / `ctrl_push` |
| `aie_runtime.c:3804` | `rt_ctrl_read_extract` (strips per-access header) |
| `xaie_ss.c:755`, `xaiemlgbl_reginit.c:2090` | `NumSlaveSlots = 4` (slots per port) |
| `xaie_ss.c:45-53` | arbiter/MSel/MSelEn field ranges (0..7 / 0..3 / 0..0xF) |
| `routinghwlower.cpp:270-337` | reference packet-switch emit (slot/port/master) |

**See also:** `docs/plans/2026-09-03-control-packet-anytile-multicast-design.md`
(compile-time CONTROL multicast) and
`docs/plans/2026-09-03-control-network-bootstrap-config-design.md`.
