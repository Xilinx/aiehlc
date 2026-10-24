# Control-Packet Plane

Runtime control path for AIE register access over the stream fabric, as an
alternative to the host config bus. Lives in `src/mlir/runtime/` (`aie_runtime.c`,
`aie_runtime_control_plan.c`, `aie_runtime_resource.c`) and is built into `mlirtestlib`
and the host link (`hostcompile.sh` / `aiehlc.sh` `RUNTIME_SRCS`).

---

## 1. Packet encoding and parsing

| API | Purpose |
|-----|---------|
| `__Runtime_ctrl_pktize_write(out, cap, stream_id, tile_addr, data, nwords, lastwriteack, ret_sid, &resp_words)` | Build WRITE control-packet words |
| `__Runtime_ctrl_pktize_read(out, cap, req_sid, ret_sid, tile_addr, nwords, &resp_words)` | Build READ control-packet words |
| `__Runtime_ctrl_parse_ctrl_hdr(hdr, &addr, &op, &beats, &ret_sid)` | Decode a control-info word |
| `__Runtime_ctrl_parse_pkt_hdr(hdr, &id, &type, &src_row, &src_col)` | Decode a packet-switched stream header |

**Write with ack.** When `lastwriteack` and `nwords > 0`, the LAST written word is
re-emitted as a single-word WRITE-WITH-RETURN access (op `0b10`, return stream id
[28:24]). Its header-only response acks all preceding writes, so `resp_words` = 1.

**Read encoding.** op `01`, byte addr [19:0], beats-1 [21:20], return stream id
[28:24], odd parity [31]; no data payload.

**Response word accounting.** `resp_words` counts, per access, 1 stream header word
*plus* its data words — the return-route master KEEPS each response packet's stream
header (`XAIE_SS_PKT_DONOT_DROP_HEADER`). The per-access header is stripped host-side
in `rt_ctrl_read_extract`.

**Control-info word fields:** addr [19:0], op [23:22] (`00` write / `01` read /
`10` write-with-return), beats = [21:20]+1 = 1..4, return sid [28:24].
**Stream header fields:** id [4:0], type [14:12] (7 = SLVERR), source row [20:16],
source col [27:21].

Both parsers accept NULL out-pointers and return 1 iff the odd-parity bit [31] over
[30:0] is consistent.

---

## 2. Single-target send/receive (`__Runtime_CtrlInstance`)

The send context is a `__Runtime_CtrlInstance` struct: shim col, dest tile, stream id,
DMA channels/BD, `resp_words`, and an internal response buffer `token`.

| API | Purpose |
|-----|---------|
| `__Runtime_ctrl_setup_routing(inst)` | Program shim→dest CTRL forward route + dest CTRL slave→shim S2MM return route; allocate the `resp_words` response buffer; arm the shim S2MM |
| `__Runtime_ctrl_push(inst, buf, nwords, block, log)` | Push a packet buffer via a SHIM MM2S BD; `block != 0` also waits for the response drain |
| `__Runtime_ctrl_tct_poll(inst, print)` | Poll the S2MM drain; return the first response word |
| `__Runtime_ctrl_read_target(dev, shim_col, dest_col, dest_row, req_sid, ret_sid, tile_addr, nwords, out_data, bd_id, mm2s_ch, s2mm_ch)` | Full register read: pktize → setup_routing → push → poll → extract |
| `__Runtime_ctrl_push_target(...)` | setup_routing → push → poll around one packet buffer |

**Completion barrier.** The return path is the *control-packet response*, not a DMA
TCT token: the dest CTRL slave port emits the response (read data, or write-with-return
ack), circuit-switched down to the shim S2MM. The drain landing `resp_words` words is
the barrier.

**Limit.** A single shim S2MM drain BD implies one response packet (a ≤4-word access
not crossing a 128-bit boundary). Larger reads need one BD per packet — not yet
implemented.

---

## 3. Row-control fabric (`aie_runtime_control_plan.c`)

A pure planner (no HW access), host-unit-tested via `unitest/test_ctrl_row_plan.cpp` +
`build_ctrl_row_plan.sh`.

### Governing model

Arm a **uniform persistent forward-slot superset** on every chain tile's ingress slave
port, so a single row-multicast reaches EVERY column of the target row. All columns
respond — there is no per-column last-tile special casing.

### Stream id (5 bits)

- `[4]` = broadcast marker. `[4]=1` ⇒ `ACR_ID_BCAST` (whole array).
- `[4]=0` ⇒ **row-multicast**; `[1:0]` = target row **add-order index** (0..3,
  `ACR_MAX_ROW_IDX`) — the Nth row in the `plan_init` list gets index N.
- `[3:2]` = legacy column-subset class (`ACR_CLASS_*`), no longer separately routed.

### Forward chain

Every tile arms, on arbiter 0:

| Slot | pkt / mask | msel | Fan-out |
|------|-----------|------|---------|
| slot0 CONSUME | `rowidx` / `ACR_MASK_CONSUME 0x17` (matches row K for classes 00 **and** 10) | 0 | CTRL + EAST |
| slot1 BCAST | `0x10` / `0x10` | 1 | CTRL + EAST (+ NORTH on head) |
| slot2 TRANSIT_N *(spine-head tiles only)* | `0x00` / `0x10` | 2 | NORTH climb |

Per-tile masters (arb0, keep_header): CTRL `MSelEn=0x3` on **every** tile; EAST
`MSelEn=0xB` when not the last col; NORTH `MSelEn=0x6` at the head-on-spine-column.
Slot budget: spine head = 3, other tiles = 2 (all ≤ 4).

> The old `0x7` CTRL_LAST plus the ONLY_LAST / WHOLE / TRANSIT_E slots are gone — last
> tiles are no longer special.

### Return chain (`acr_plan_return_chain`, arbiter `ACR_ARB_RET=1`)

Every target-row column injects its CTRL-slave response (RET_LOCAL slot, accept-any
pkt=0 / mask=0, msel0). Responses merge **east→west**: each tile with an east neighbor
also arms an EAST-slave RET_TRANSIT slot (msel1) to pull the neighbor's westbound bus.

- Non-head tiles drive the merged bus WEST (`MSelEn` interior `0x3`, last col `0x1`).
- The head (`col_lo`, on the spine column) drives SOUTH→VRET (`MSelEn 0x7`), also
  merging descending upper-spine responses via a NORTH-slave RET_NORTH slot (msel2).

`acr_plan_return_chain(o, b, row, col_lo, col_hi, is_top)` — the **topmost** configured
row's return head OMITS the idle RET_NORTH slot (nothing descends from above it) and
drives SOUTH→VRET with `{local, transit}` only; non-top heads keep the full
`{local, transit, north}` merge.

### Chain assembly

- `acr_plan_row_add(..., is_top)` builds the forward chain then the return chain per
  row, threading `is_top`.
- Pure pass-through spine rows emit BOTH a forward circuit `ACR_OP_CCT` (SOUTH→NORTH,
  class-agnostic) and a return circuit CCT (NORTH→SOUTH), so responses descend the VRET
  spine to the shim.
- `acr_plan_chain_ex(o, b, row, rowidx, col_lo, col_hi, ctrl_id, shim_col, head_ingress)`
  takes `int shim_col` (−1 ⇒ plain chain, no NORTH climb). `acr_plan_row_add` passes
  `rowidx = s->nrows` and `shim_col = col_lo` so the head emits its own NORTH climb
  (bcast + transit, so any row-multicast reaches any row). Spine extension through an
  already-configured head row emits nothing — that head climbs itself.

### One-shot init

`__Runtime_ctrl_plan_init(f, dev, shim_col, resp_s2mm_ch, ctrl_id, rows, nrows)` where
`rows` is a `__Runtime_CtrlRowChain[]` (each `{row, col_lo, col_hi}`) ordered bottom-up.
The static planner `rt_ctrl_plan_add_rows` computes the top row and plans/emits every
chain, reusing the shared spine.

`plan_init` also programs the **shim return leg** at plan time
(`rt_ctrl_row_shim_return_route`: shim NORTH/VRET→SOUTH circuit + S2MM stream-port
enable + provenance) whenever `resp_s2mm_ch >= 0` — so the return spine reaches the shim
(row 0) up front, symmetric with the per-send forward entry (`rt_ctrl_row_shim_entry`),
and visible on the device map without needing a read/write-ack send. Only the
response-buffer-dependent S2MM **BD arming** stays per-send, inside
`rt_ctrl_row_shim_return` (which re-runs the idempotent route helper).

### Send / receive APIs

| API | Purpose |
|-----|---------|
| `__Runtime_ctrl_row_broadcast_write` | Whole-array write (id `0x10`) |
| `__Runtime_ctrl_row_whole_row_write(f, row, ...)` | Row-multicast write; resolves physical `row`→add-order index via `f->rows[]`, builds `sid=(ACR_CLASS_WHOLE_ROW<<2)\|rowidx`, targets `col_lo` |
| `__Runtime_ctrl_row_read(f, row, tile_addr, nwords, out_vals, bd_id, mm2s_ch)` | Per-column register read: whole-row read packet, arms `ncols` shim S2MM BDs `bd_id+1..bd_id+ncols` (one per response packet), polls the drain, routes each landed packet into `out_vals[ci*nwords]` by parsing its kept stream-header `src_col` |
| `__Runtime_ctrl_row_write_ack(f, row, tile_addr, data, nwords, bd_id, mm2s_ch)` | Whole-row write whose last word is a write-with-return so every column acks; asserts all `ncols` acks drained |

The legacy `__Runtime_ctrl_row_all_but_last_write` / `__Runtime_ctrl_row_only_last_write`
wrappers still compile but resolve onto the same uniform superset. The unicast
write/read overlay is removed.

---

## 4. Transaction capture

An XAie-transaction-like flow (in `aie_runtime.c`) that turns WRITE-only register ops
into a BROADCAST/MULTICAST control-packet word buffer, routed against a
`__Runtime_CtrlRowFabric`. Host-unit-tested via `unitest/test_ctrl_txn.cpp` +
`build_ctrl_txn.sh` (9 cases).

**`__Runtime_control_start_transaction(dev, fab, row)`** wraps
`XAie_StartTransaction(XAIE_TRANSACTION_DISABLE_AUTO_FLUSH)` so captured
`XAie_Write32` / `XAie_BlockWrite32` calls never touch HW, and records the cast target
into `fab->txn_row` (`row < 0` ⇒ whole-array broadcast; `row >= 0` ⇒ row-multicast).

**`__Runtime_control_write_pkt_commit_transaction(dev, fab, out, out_cap, &nwords)`** is
**stateless w.r.t. the caller** — it reads `fab->txn_row` back, calls
`XAie_ExportSerializedTransaction`, then hands off to
`rt_ctrl_txn_translate(dev, buf, sid, target_row, out, cap, &nwords, lastwriteack, ret_sid)`,
which walks the serialized `XAie_TxnHeader`+ops buffer and:

1. Decodes each op's `RegOff` via `dev->DevProp.RowShift` / `.ColShift` into
   `(col, row, tile_addr)`.
2. **Verifies every op is a write op** (Write32 / BlockWrite32). MaskWrite32 / MaskPoll
   / unknown are rejected.
3. Validates the tile against the cast target (`rt_ctrl_txn_check_tile`): broadcast ⇒
   every op must land on a CORE tile per `XAie_GetTileTypefromLoc`; multicast ⇒ every op
   must land on `target_row`.
4. Emits one WRITE control packet per op via `__Runtime_ctrl_pktize_write`, stamped with
   the single cast stream id (`rt_ctrl_txn_cast_sid`: broadcast ⇒ `ACR_ID_BCAST`
   id[4]=1; multicast ⇒ `(ACR_CLASS_WHOLE_ROW<<2)|rowidx` id[4]=0).

With `lastwriteack` (commit passes 1, `ret_sid=0`), **only the LAST emitted write
packet** becomes a WRITE-WITH-RETURN (op `0b10`) so one header-only response acks the
whole batch; every earlier op stays fire-and-forget (op `0b00`).

Rejected: MaskWrite32 / MaskPoll / unknown ops, tile-type mismatches, unconfigured rows,
capacity overflow — all return nonzero.

**`__Runtime_control_push(fab, out, nwords, bd_id, mm2s_ch, block, log)`** pushes the
committed buffer to HW. It reads `fab->txn_row` to resolve the diagnostic dest tile
(broadcast ⇒ last-added row head + `ACR_ID_BCAST`; multicast ⇒ that row's chain
`col_lo` + whole-row sid), **copies `out` into a device DMA buffer** (`out` may be a
stack array, which `__Runtime_ctrl_push` cannot DMA), arms the shim spine entry
(`rt_ctrl_row_shim_entry`), builds a `__Runtime_CtrlInstance`, and pushes via a SHIM
MM2S BD.

> `rt_ctrl_txn_translate` takes the already-exported buffer (only a tile-type query on
> broadcast), so it is host-testable with a hand-built transaction, a `make_dev()` grid,
> and a `XAie_GetTileTypefromLoc` stub.

### On-HW demos (`example/tileprogram/ccode/ctrlrow_demo.cc`)

- **`run_ctrol_trasaction` / `demo_txn_multicast_row`** — per core row: `start` with
  `row`, `XAie_BlockWrite32` to a core L1 scratch on that row, `commit` into a
  control-packet buffer, byte-compare against a reference `__Runtime_ctrl_pktize_write`
  with the same row-multicast sid, deliver via `__Runtime_control_push`, verify every
  column landed the payload via `XAie_DataMemBlockRead`.
- **`demo_txn_multipl_row`** — 4×4 broadcast: `plan_init` configures 4 core rows (3..6)
  × cols 0..3 on one spine; a single-`XAie_BlockWrite32` transaction is `start`ed with
  `row=-1`, committed (yielding exactly the `ACR_ID_BCAST` packet words, byte-compared
  to a reference pktize), and pushed — delivering to all 16 tiles, verified per-tile.

---

## 5. Resource reservation table (`aie_runtime_resource.{c,h}`)

Pure C, `extern "C"`-guarded so it is C++-safe. Host-unit-tested via
`unitest/test_ctrl_resource.cpp` + `build_ctrl_resource.sh`.

**Single source of truth** for the stream-switch resources (packet slots, arbiters,
msel, 5-bit pkt-ids/masks) the control plane occupies, keyed by device gen (`rt_res_gen`
GEN1/2/5) + port-type. The static `rt_res_table` enumerates:

- **Forward fabric** — ingress-slave WEST/SOUTH slots CONSUME rowidx 0..3 / BCAST /
  TRANSIT_N on arb0; masters CTRL / EAST / NORTH.
- **Return fabric** — CTRL/EAST/NORTH slave slots RET_LOCAL / RET_TRANSIT / RET_NORTH on
  arb1; masters WEST / SOUTH.

| API | Purpose |
|-----|---------|
| `__Runtime_res_port_reserved(gen, col, row, port, is_master, out, cap)` | List reserved resources on one stream port |
| `__Runtime_res_reserved_pktid_mask` / `_arbiter_mask` / `_slot_mask` | Aggregate masks |
| `__Runtime_res_is_reserved(gen, port, is_master, slot, arbiter, msel, pkt_id)` | Validation predicate; master matching is an mselen **subset** test, since return masters emit subsets 0x1 / 0x3 / 0x7 |
| `__Runtime_res_gen_from_name("Gen#")` | Gen lookup by name |

**No drift by construction.** All numeric constants live here as `RT_RES_*`;
`aie_runtime_control_plan.h` `#define`s the legacy `ACR_*` macros as aliases of them.
The planner's `acr_emit_slot` / `acr_emit_master` gain **opt-in**
`assert(__Runtime_res_is_reserved(...))` checks under `-DACR_VALIDATE_RESERVED` (defined
by `build_ctrl_row_plan.sh`, off in production) — proving the planner only emits
reserved resources.

### MLIR compile-time reservation

`ResourceMgr::reserveControlPlaneResources(rt_res_gen)` is **opt-in**: it runs only when
the user source contains `#pragma control_plan_op_control_packet` (a bare marker pragma
parsed in `aiehlc.cc`, carried into the module as attr
`routing.control_plan_op_control_packet = 1 : i64`). Each reserve call site gates on it;
default off.

When enabled it is called right after each `ResourceMgr::init` and reads the table to
mark reserved pkt-ids used in `pktIdPool_` (sentinel owner `kControlPlaneOwner = -2`, so
`allocatePktId` never hands them out), and records `reservedArbiterMask()` /
`reservedSlotMask(port, is_master)` accessors.

> **Follow-up:** wiring data-plane slot/arbiter picking to actually consult these
> accessors is not done yet.

---

## 6. Provenance and the debug UI

### Static provenance

`xaiehost2provenance.py` produces debug provenance for a control-packet `host.cc`
statically — it recognizes `__Runtime_CtrlInstance` sends (shim + vertical pass-through
+ dest tiles, two flows per send), so `schedule_debug_server.py` can open a debug GUI
without a `Work/` tree.

### PMAP applog lines

When `__Runtime_ctrl_pmap_enable(1)` is set before the first `aie_ctrl*` `setup_routing`
/ `plan_init`, routing setup emits one line per programmed stream port (same-column climb
+ row fabric):

```
CONTROLPAN-PMAP col=.. row=.. port=.. idx=.. dir=fwd|ret ms=master|slave id=.. sw=pkt|circuit slot=.. arb=.. msel=.. mask=..
```

`arb` / `msel` / `mask` are the stream-switch packet-routing params: a packet slave slot
carries mask + msel + arbiter; a packet master carries mselen (in the `msel` field) +
arbiter. Circuit ports and legacy lines emit −1 for all three via `rt_pmap_port_ex`, so
old applogs stay backward compatible.

### `dir` comes from the planner, not the port type

`dir` is read from the planner's own `acr_op.is_ret` fabric tag (via `rt_acr_dir`). The
return chain arms its slots on CTRL/EAST **slaves** and its non-head master on WEST —
all ports the forward chain also uses — so a port-based guess mislabels every non-spine
return port as `fwd`.

Slot/master ops derive `is_ret` from `arbiter == ACR_ARB_RET` (the two fabrics use
disjoint arbiters by construction). Slave-enable and circuit ops carry no arbiter, so
`acr_emit_ret_slot` and the return CCT set it explicitly.

> **Invariant:** a slot and every master that pulls it must share a `dir`, because
> `tile_switch_view` buckets ports by direction *before* linking slot→master — a mismatch
> silently drops the link from the switch-detail view.
> Locked in by `test_ctrl_row_plan.cpp::test_is_ret_tagging`.

### Device-map overlay

The aiedebug device-map **Load control plan** button parses the PMAP lines
(`controlpan_pmap.py` + the `/ctrlplan/load` endpoint) and overlays control-plan routing
on the device map.

After the plan is loaded, clicking a tile that carries control-plan ports renders that
tile's stream-switch connection detail as a card in the right-side Info panel
(`showTileSwitchDetail` in `schedule_view.py` pushes a keyed `switch:` panel card whose
`wireBody` fetches `/ctrlplan/tile` + `controlpan_pmap.tile_switch_view`; the clicked
tile gets an amber ring highlight). The card is an enlarged SVG with one section per
DIRECTION (fwd/ret): merged physical slave input ports → the packet slots each arms →
merged master output fan-out.

`tile_switch_view` returns:

```
{col, row, dirs: [{dir,
   slaves:  [{port, idx, sw, id, slots: [{slot, pkt_id, mask, msel, arb}]}],
   masters: [{port, idx, dest, sw, id, arb, mselen}]}]}
```

- A physical `(port, idx)` is merged **once per direction** — a slave arming N slots is
  one box carrying N slots, not N groups.
- Per AIE stream-switch semantics the packet-routing params live on the **slot**: each
  slot box is labeled `slot N · pkt_id X · mask 0xY` with an `arb·msel` sub-line.
- Each master is sub-labeled `arb·mselen`, where `mselen` is a BITMASK of accepted `msel`
  values.
- A slot→master link is drawn for EVERY master that pulls that slot
  (`m.arb == slot.arb && ((m.mselen >> slot.msel) & 1)`) across all slots/slaves — so one
  CTRL/EAST/NORTH master fans out to whichever slots its mselen selects (legitimate
  multicast).
- Circuit slaves carry no slot and link straight to the circuit master with the matching
  id.
- Master `dest` pairs by neighbor `(port, idx, dir)` — id-agnostic, since a packet master
  feeds a whole neighbor slave port.
