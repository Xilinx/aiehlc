# Design: Control-Packet Delivery to Any Tile / Whole Row / Whole Col from Any Shim

**Status:** design only (no code changes in this doc)
**Date:** 2026-09-03
**Scope (Q1):** Reuse the existing MLIR routing engine to deliver a control-packet
register write to (a) any AIE tile from any shim column, (b) every tile in a row,
or (c) every tile in a column, as a **single multicast packet** (write-only, no
per-tile ack). Must not change the existing `routing.cc` data-plane logic.

---

## 1. Summary

The MLIR routing engine already contains the exact primitive we need: a
`StreamType::CONTROL` flow. It builds a shim → N-target **packet-switched
fan-out** (the same BFS / broadcast machinery as `StreamType::BROADCAST`), and
each target tile terminates the packet at its **CTRL master** stream-switch port
with the packet **header preserved**. Because every target taps the same 5-bit
stream id, one control packet pushed from the source shim lands an identical
register write on every tapped tile. That is precisely "one multicast packet to a
row / col / set of tiles".

Two small, clearly-scoped relaxations are needed (documented here, **not** coded):

1. Relax the `TileType::Core`-only gate on CONTROL termination
   (`passdmaphoptoroutinghw.cpp:472`) so a memtile/shim can also be a CONTROL sink
   when a register write targets it.
2. A front-end convenience to expand `row(r)` / `col(c)` / `rect(...)` into the
   destination-tile set fed to the fan-out.

Everything else — cross-column pathfinding, port-budget accounting, EmitC
emission — is reused unchanged. Single-target **read/ack** stays on the runtime
same-column path (`__Runtime_ctrl_read_target`), untouched.

---

## 2. The reusable primitive: `StreamType::CONTROL`

### 2.1 Where CONTROL is created

A shim→core dmaphop marked `control` becomes a CONTROL stream during
dmaphop→routinghw lowering:

- `passdmaphoptoroutinghw.cpp:1045` sets the default `streamtype` for a
  shim→core (MM2S) delivery to `StreamType::BROADCAST`.
- `passdmaphoptoroutinghw.cpp:1049-1051` upgrades it to `StreamType::CONTROL`
  when the op carries a `control` bool attr:

  ```cpp
  if (isShimToCore && op->hasAttrOfType<BoolAttr>("control") &&
      op->getAttrOfType<BoolAttr>("control").getValue()) {
      streamtype = StreamType::CONTROL;
  }
  ```

### 2.2 CONTROL fans out like BROADCAST, terminates at CTRL

`ParseTheCCTRoutingPath` (`passdmaphoptoroutinghw.cpp:492`) handles both
BROADCAST and CONTROL through the same path builder `GetSeqPath`. Two facts make
it a control multicast:

- **Shared stream id.** `passdmaphoptoroutinghw.cpp:514-515`:

  ```cpp
  bool isControl = (streamtype == StreamType::CONTROL);
  int ctrlPktId = static_cast<int>(dioid & 0x1f);   // 5-bit stream-header id
  ```

  Every fan-out target uses the same `ctrlPktId`, so one packet with that id in
  its stream header is accepted by every tapped slave slot → identical write on
  all targets. This IS the "one multicast packet".

- **Per-target termination at CTRL.** The termination is marked with
  `localsinkport = "CTRL"`; in `routinghwlower.cpp` the pattern
  `ConnectStreamPktSwitchPortpattern` reads it at line 196
  (`isCtrlSink = (localsinkportstr == "CTRL")`) and forces the master port to be
  the tile's CTRL master, idx 0, header preserved:

  - `routinghwlower.cpp:303-304`
    `effMasterDirStr = isCtrlSink ? "CTRL" : masterportdirectionstr;`
    `effMasterIdx    = isCtrlSink ? 0 : masterportidx;`
  - `routinghwlower.cpp:323-332` CTRL sinks always emit
    `XAIE_SS_PKT_DONOT_DROP_HEADER`, because the in-tile CTRL decoder must read
    the control-info (addr/beats) word out of the preserved header.

  The emitted calls per tapped tile are therefore (from the same pattern):
  - `XAie_StrmPktSwSlaveSlotEnable` + `XAie_StrmPktSwSlavePortEnable`
    (`routinghwlower.cpp:270-278`) — arm the incoming slot for `ctrlPktId`,
  - `XAie_StrmPktSwMstrPortEnable(... CTRL, 0, XAIE_SS_PKT_DONOT_DROP_HEADER ...)`
    (`routinghwlower.cpp:335-337`) — terminate into the CTRL master.

### 2.3 Current gate (the only place to relax)

CONTROL (and BROADCAST) termination fires only on Core tiles today:

`passdmaphoptoroutinghw.cpp:472-474`

```cpp
if (rm->getrsc()->tileType(p.r, p.c) == TileType::Core &&
    (StreamType::BROADCAST == streamtype || StreamType::CONTROL == streamtype) &&
    dsttiles.find(p) != dsttiles.end()) {
```

To allow a register write to land on a **memtile or shim** CTRL port, this gate
must additionally accept `TileType::Mem` (and, if ever needed, the shim types).
The CTRL master enable itself is tile-type agnostic in `routinghwlower.cpp`, so
no other change is required for the termination.

---

## 3. Any shim source (cross-column origin)

A CONTROL flow originates on whatever NOC-shim column its `DataIO` is allocated
on. The `hasShim` branch (`passdmaphoptoroutinghw.cpp:1061-1090`) either reuses
an existing `DataIO` for a given shim column+channel
(`findDataIOByShimChannel`) or creates a new one; `shimcol = dio->colpos()`
(`:1080`, `:1088`) pins the source column.

The legal source columns are exactly the NOC-shim columns enumerated by the
resource model: `ResourceMgr::InitSHIMNocList()`
(`ResourceManager.cpp:256-261`) walks `resource_->getShimNoc()` and registers a
`ShimTile(0, col, 2, 2)` per NOC-shim column. So "different shim tiles" ⇒ pick a
different source column in the front-end op; **no engine change**.

---

## 4. Any AIE tile (cross-column route)

BFS pathfinding already produces EAST / WEST / NORTH / SOUTH routes with
resource tracking, so a `shim(col_s,0) → tile(row_d,col_d)` route with
`col_d != col_s` needs **no new pathfinding**. The route is created by
`router_.createPath()` and its intermediate tiles are materialized in
`ParseTheCCTRoutingPath` phase 2 (`passdmaphoptoroutinghw.cpp:532-539`), which
inserts a `routinghw::TileCreate` for any tile on the path not already in
`dsttiles`.

Path priority is Memory > SHIM > Core (per `RoutingPath` in
`routingimplement/routing/`), and every hop's port usage is booked through
`ResourceManager` / `PortTemplate`, so cross-column CONTROL routes automatically
respect the same limits as data flows (section 6).

---

## 5. Whole row / whole col (multicast tree)

Express the target as a set:

- `row(r)` → `{ (r, c) : c in legal columns }`
- `col(c)` → `{ (r, c) : r in legal rows }`
- `rect(r0..r1, c0..c1)` → the product set.

Feed that set as `dsttiles`. The fan-out engine (`GetSeqPath` +
`ParseTheCCTRoutingPath`) builds the multicast tree from the source shim to every
target; each target arms its slave slot for the shared `ctrlPktId` and terminates
at its CTRL master. One pushed packet ⇒ identical register write on all targets.

**Proposed front-end convenience (design only):** a routing-dialect op or attr
that expands `row(r)` / `col(c)` / `rect(...)` into the `dsttiles` set before the
dmaphop lowering runs. This is pure sugar over the existing set input — no change
to the fan-out engine.

**Fan-out width bound:** the per-tile port budget (section 6) can limit how wide
a single fan-out tree is at any one tile; when a tile runs out of the relevant
master/slave ports, `ResourceManager` reports exhaustion and the route fails
loudly rather than silently overwriting a port.

---

## 6. Port budget is honored automatically

`hwresource.cpp` `defaultPortTemplates()` encodes the asymmetric budget the BFS
must respect:

| Tile | N-Master | S-Master | N-Slave | S-Slave | E/W (M/S) | DMA (M/S) |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| Core (`:128-138`) | 4 | 4 | 4 | 4 | 4 / 4 | 4 / 4 |
| Mem (`:152-162`)  | **6** | 4 | 4 | **6** | 0 / 0 | **6 / 6** |
| NocShim (`:163-173`) | 4 | 2 (mux 3,7) | 4 | 2 (demux 1,3) | 4 / 4 | 4 / 4 |

The memtile North-Master 6 / South-Slave 6 / DMA 6 numbers are the wide vertical
channel the CONTROL climb rides; horizontal (East/West) is capped at 4 on cores
and 0 on memtiles (memtiles do not route horizontally). `ResourceManager` tracks
per-direction usage via `PortTemplate`, so any CONTROL route is bounded by these
limits automatically.

---

## 7. Runtime split (network vs. packet words)

The MLIR engine emits only the **network** — the extra CONTROL packet-switch
rounds in `routing.cc` (slave-slot / slave-port / CTRL-master enables). It does
**not** emit the packet payload.

At run time the host still:

1. Builds the control-packet words with
   `__Runtime_ctrl_pktize_write(out, cap, stream_id = ctrlPktId, tile_addr, data,
   nwords, lastwriteack = 0, ret_sid = 0, &resp_words)`
   (`aie_runtime.c:3035`). For a write-only multicast, `lastwriteack = 0` so no
   write-with-return word is appended and `resp_words = 0`.
2. Pushes the buffer through the **source shim MM2S** with
   `__Runtime_ctrl_push(inst, buf, nwords, block = 0, log)`
   (`aie_runtime.c:3255`). `block = 0` because a multicast write has **no S2MM
   response drain** (write-only, no per-tile ack).

The `stream_id` passed to `pktize_write` must equal the flow's `ctrlPktId`
(`dioid & 0x1f`) so the packet header matches the slave slots the engine armed.

> Note on the encoding: the control-packet word format (op / addr [19:0] /
> beats-1 [21:20] / stream id / odd parity [31]) is documented inline above
> `__Runtime_ctrl_pktize_write` / `__Runtime_ctrl_pktize_read`
> (`aie_runtime.c:3035`, `:3115-3134`) and in `doc/controlpkt.txt` (present in
> the `cnn` worktree; keep a copy in `doc/` when this work lands).

---

## 8. Why this does not break `routing.cc`

CONTROL flows are **additive**:

- They use their own stream ids (`ctrlPktId = dioid & 0x1f`) and their own ports,
  all allocated by the same `ResourceManager` that allocates data-plane ports, so
  there is no aliasing with existing data rounds.
- They are emitted as independent EmitC rounds (each round is a distinct block in
  `routing.cc`, e.g. the per-`dioid` blocks visible in
  `aout/worklocal/routing.cc:8-45`). Adding CONTROL rounds appends new blocks; it
  does not alter existing circuit-switch (`XAie_StrmConnCctEnable`) or packet
  rounds.
- The current `aout/worklocal/routing.cc` contains only data-plane circuit rounds
  (all `XAie_StrmConnCctEnable`; no CTRL master enable), confirming CONTROL
  emission is opt-in and absent unless a `control` flow is declared.

---

## 9. Single-target read / ack stays on the runtime path

When verification / readback is needed, use the unchanged runtime same-column
path `__Runtime_ctrl_read_target(...)` which builds READ packets
(`__Runtime_ctrl_pktize_read`, `aie_runtime.c:3134`), sets up the vertical
forward + return route (`rt_ctrl_route_setup_col`, `aie_runtime.c:3454`) and
drains the response on the shim S2MM. This is **circuit-switched** and
**same-column only** (`__Runtime_ctrl_setup_routing` rejects
`dest_col != shim_col`, `aie_runtime.c:3735`), and is orthogonal to the multicast
write path above.

---

## 10. Worked example — shim(col 2) → whole row 4, CONTROL multicast

Goal: write one 32-bit value V to register offset `A` on every tile in row 4,
from the NOC-shim in column 2, as one packet.

**Target set:** `row(4)` → `{(4,0),(4,1),(4,2),(4,3),(4,4),(4,5), ...}`
(the legal columns of the device).

**Engine build (per target tile T = (4, cT)):**
1. BFS route shim(2,0) → T. For `cT = 2` the route is the straight vertical
   climb rows 1→4 (memtiles at rows 1..k then cores), riding memtile
   North-Master / South-Slave ports (budget 6 each, `hwresource.cpp:153,159`).
   For `cT != 2` the route climbs to row 4 then goes East/West along row 4 using
   core East/West ports (budget 4, `hwresource.cpp:131,136`).
2. At each hop `ParseTheCCTRoutingPath` arms the packet slave slot for the shared
   `ctrlPktId = dio->id() & 0x1f` and forwards to the next master
   (`routinghwlower.cpp:270-337`).
3. At T the termination is a CTRL sink:
   `XAie_StrmPktSwSlaveSlotEnable(dev, XAie_TileLoc(cT,4), <in-dir>, <in-idx>,
   slot, XAie_PacketInit(ctrlPktId,0), mask, msel, arbiter)`,
   `XAie_StrmPktSwSlavePortEnable(...)`, then
   `XAie_StrmPktSwMstrPortEnable(dev, XAie_TileLoc(cT,4), "CTRL", 0,
   XAIE_SS_PKT_DONOT_DROP_HEADER, arbiter, msel)`
   (`routinghwlower.cpp:335-337`).

   > Requires the `TileType::Core` gate relaxation from §2.3 only for the row-4
   > tiles that are memtiles/shims; row-4 cores already pass the gate today.

**Runtime push (once):**
```c
uint32_t words[8], resp = 0;
uint32_t n = __Runtime_ctrl_pktize_write(words, 8,
                 /*stream_id=*/ctrlPktId, /*tile_addr=*/A,
                 &V, /*nwords=*/1, /*lastwriteack=*/0,
                 /*ret_sid=*/0, &resp);          // resp == 0 (write-only)
__Runtime_ctrl_push(&inst /*shim col 2 MM2S*/, words, n, /*block=*/0, log);
```

**Result:** every tile in row 4 whose slave slot was armed for `ctrlPktId` writes
`V` to offset `A`. One packet, no acks.

**Port-legality check:** the vertical climb uses memtile North-Master idx in
0..5 and South-Slave idx in 0..5 (both budget 6, `hwresource.cpp:153,159`); the
row-4 horizontal spread uses core East/West master/slave idx in 0..3 (budget 4,
`hwresource.cpp:131,136`); the CTRL termination uses master idx 0. All within the
templates.

---

## 11. Verification checklist (for the future implementation phase)

- [ ] Relax `passdmaphoptoroutinghw.cpp:472` gate to include `TileType::Mem`
      (and shim if targeted) for CONTROL termination; confirm cores unaffected.
- [ ] Add front-end `row/col/rect` → `dsttiles` expansion (sugar only).
- [ ] Emit a CONTROL flow in a small example; diff the resulting `routing.cc` to
      confirm existing data rounds are byte-identical and only new CTRL rounds are
      appended.
- [ ] Push one multicast write via `apppaltest.py`; read back a sentinel register
      on one target with `__Runtime_ctrl_read_target` to confirm delivery (this
      readback is the only ack, since the multicast itself is write-only).

---

## 12. File references (verified, no edits)

| Ref | What |
|-----|------|
| `passdmaphoptoroutinghw.cpp:1045-1051` | shim→core MM2S default BROADCAST; `control` attr ⇒ CONTROL |
| `passdmaphoptoroutinghw.cpp:1061-1090` | `hasShim` source-column selection / `DataIO` reuse |
| `passdmaphoptoroutinghw.cpp:466-487` | per-tile CONTROL/BROADCAST termination gate (Core-only today) |
| `passdmaphoptoroutinghw.cpp:492-519` | `ParseTheCCTRoutingPath` + `ctrlPktId = dioid & 0x1f` |
| `passdmaphoptoroutinghw.cpp:532-539` | intermediate path-tile materialization |
| `routinghwlower.cpp:124-347` | `ConnectStreamPktSwitchPortpattern` (packet emit) |
| `routinghwlower.cpp:196,303-304,323-337` | `isCtrlSink` → CTRL master idx 0, DONOT_DROP_HEADER |
| `hwresource.cpp:128-173` | Core/Mem/NocShim port templates (6/6/4 budget) |
| `ResourceManager.cpp:256-261` | `InitSHIMNocList` — legal NOC-shim source columns |
| `aie_runtime.c:3035` | `__Runtime_ctrl_pktize_write` |
| `aie_runtime.c:3134` | `__Runtime_ctrl_pktize_read` |
| `aie_runtime.c:3255` | `__Runtime_ctrl_push` |
| `aie_runtime.c:3454` | `rt_ctrl_route_setup_col` (same-column vertical spine, circuit-switched) |
| `aie_runtime.c:3735` | `__Runtime_ctrl_setup_routing` rejects `dest_col != shim_col` |
| `aie_runtime.h:757-769` | `__Runtime_CtrlInstance` |
| `aout/worklocal/routing.cc:8-45` | current data-plane circuit rounds (no CTRL) |

**See also:** `docs/plans/2026-09-03-control-network-bootstrap-config-design.md`
(Q2 — using a control network to configure the `routing.cc` network).
