# Row-Fabric Control Return Path (Write-Ack + Read) — Design

Date: 2026-09-13
Status: Approved (brainstorming) — ready for implementation plan

## Problem

The row-multicast control fabric (`__Runtime_ctrl_row_*`, planner
`acr_plan_row_add` / `acr_plan_chain_ex`) builds **forward-only** routing: a
control packet climbs a shared vertical spine and daisy-chains EAST across a
target row. It is fire-and-forget — there is no return path, so a WRITE-ack
(write-with-return) or a register READ response cannot flow back to the shim.

The only return-path logic today lives in the **unicast** per-column function
`rt_ctrl_route_setup_col` (`aie_runtime.c:3569-3659`): it wires the dest tile's
CTRL slave → SOUTH(VRET) master → pass-through NORTH→SOUTH hops → shim
NORTH(VRET)→S2MM demux. That logic must move to the row planner, and the row
fabric must gain read/write-ack send APIs plus a demo.

## Requirements (as agreed)

1. Remove the ack/return-route logic from `rt_ctrl_route_setup_col`.
2. In `acr_plan_row_add`, add logic to connect each target-row tile's CTRL slave
   into a WEST-going (right→left) return chain that funnels down the spine to the
   shim, to carry write-ack and read responses.
3. Add read + write-ack handling to the demo (`ctrlrow_demo.cc`).
4. Remove the column-subset "last-tile" class probes (only-last, all-but-last)
   from the demo.

## Decisions (from brainstorming)

- **Target file for #3/#4:** `example/tileprogram/ccode/ctrlrow_demo.cc` (there is
  no literal `controlexample.c`).
- **Return scope:** EVERY tile in the target row responds; the responses MERGE
  onto the westbound packet bus (revised from single-target — user directive
  "all the tile should return response or value").
- **No only-last / last-tile logic anywhere.** Forward delivery is uniform
  whole-row/broadcast (reaches every tile), which the already-simplified planner
  `acr_plan_chain_ex` does today (CONSUME mask `0x17` + BCAST per tile). This also
  satisfies #4 at the planner level.
- **#4 removal scope:** drop the only-last / all-but-last `demo_probe_class`
  calls and the `is_last` expectation branch; keep broadcast + whole-row.
- **#1 method:** EXTRACT the return-route logic into a standalone helper so the
  unicast read/ack callers keep working; `rt_ctrl_route_setup_col` becomes
  forward-only.
- **Multi-packet drain:** N columns ⇒ N response packets (each keeps its stream
  header, which carries source col/row). The shim S2MM is armed with one BD per
  packet (queued), and the host demuxes per-column values by parsing headers.

## Current-code note (important)

The working-tree `acr_plan_chain_ex` (`aie_runtime_control_plan.c:102-156`) is
already simplified: it emits only CONSUME (`mask 0x17` → classes 00 & 10) + BCAST
(+ TRANSIT_N on the spine head) per tile, with CTRL master `0x3` and EAST master
`0xB`. It has NO transit-east/only-last/whole slots and NO last-tile `0x7` CTRL
branch (`te_id` is computed but unused). Consequently the stale `test_ctrl_row_plan.cpp`,
the header comments, and `CLAUDE.md` still describe last-tile logic that no longer
exists — these MUST be updated to match (uniform per-tile forward + the new return
chain). only-last (class 01) is currently dropped; the all-tiles-respond model
does not need it.

## Approach (chosen vs. rejected)

- **Chosen — planner-owned return chain.** `acr_plan_row_add` emits a WEST-going,
  packet-switched return chain (col_hi→…→col_lo→down VRET→shim S2MM). The emit
  layer is already fully general and `rt_acr_chan` (`aie_runtime.c:3716`) already
  maps `SOUTH master`/`NORTH slave` → `VRET`, so the spine descent needs no emit
  changes.
- **Rejected — reuse unicast `rt_ctrl_route_setup_col` per target:** contradicts
  requirements #1/#2.
- **Rejected — circuit-only return:** a shared westbound bus must merge each
  tile's CTRL-slave response with the east-transit stream, which requires packet
  arbitration (circuit is strictly 1:1).

## Part 1 — Extract return route out of `rt_ctrl_route_setup_col`

- Move `aie_runtime.c:3569-3659` (CTRL-slave→SOUTH(VRET) packet enable, the
  pass-through NORTH→SOUTH return loop, the shim NORTH(VRET)→S2MM demux leg, and
  their pmap logs) into a new static helper `rt_ctrl_col_return_route(const
  __Runtime_CtrlInstance *c, int port_evt)`.
- `rt_ctrl_route_setup_col` keeps only the forward path (shim MM2S → spine up →
  dest CTRL master) and no longer calls the return logic.
- `__Runtime_ctrl_setup_routing` calls `rt_ctrl_col_return_route` explicitly
  (before/after `rt_tct_s2mm_arm` as ordering requires) so `__Runtime_ctrl_read_target`,
  `__Runtime_ctrl_push_target`, and `example/debug/aieml_controlperf.cc` still work.

## Part 2 — Return chain in the planner (`aie_runtime_control_plan.{c,h}`)

New pure-planner logic (new `acr_plan_return_chain_ex`, invoked by
`acr_plan_row_add` after the forward chain). No XAie dependency; unit-testable.

Constants (header):
- `ACR_ARB_RET 1` — return arbiter, distinct from `ACR_ARB_CTRL 0`.
- `ACR_SLOT_RET_LOCAL 0`, `ACR_SLOT_RET_TRANSIT 1`, `ACR_SLOT_RET_NORTH 2` —
  return slot indices (local response, east-neighbor transit, spine-above descent).
- `ACR_MSEL_RET_LOCAL 0`, `ACR_MSEL_RET_TRANSIT 1`, `ACR_MSEL_RET_NORTH 2`.
- `ACR_MSELEN_RET_HEAD 0x7` (local+transit+north), `ACR_MSELEN_RET_MID 0x3`
  (local+transit), `ACR_MSELEN_RET_LOCALONLY 0x1` (last tile only).
- Return response id/mask: `pkt_id = 0`, `mask = 0` (accept any response id,
  mirroring the unicast `XAie_PacketInit(stream_id,0)` / Mask=0), `keep_header = 1`.

Per tile `c` in `[col_lo..col_hi]`, east→west — EVERY tile injects its own
response and forwards its east neighbor's, so all N responses merge west:
- CTRL slave → `RET_LOCAL` slot (its own response source) — on every tile.
- EAST slave → `RET_TRANSIT` slot when `c < col_hi` (forwards the eastern
  neighbor's westbound merged responses).
- Master (keep_header, `ACR_ARB_RET`):
  - `c > col_lo`: WEST master pulling {local, transit} (`0x3`; last tile `c==col_hi`
    has no transit ⇒ `0x1`).
  - `c == col_lo` (head, on spine): SOUTH master (→ VRET) pulling {local, transit}
    plus `RET_NORTH` when a head sits above this one on the spine (see descent).

Merge semantics: the return bus is packet-switched with `keep_header`, so each
tile's response packet (header carries source col/row per
`__Runtime_ctrl_parse_pkt_hdr`) traverses west intact and the arbiter interleaves
them into one stream of N packets down VRET to the shim.

Spine descent (multi-head aware): the demo configures rows 3 and 5 on one spine,
so row 5's descending return MUST merge through row 3 (a head). Descent rules,
folded into the existing spine-extension loop in `acr_plan_row_add`:
- **Pure pass-through row** (not a configured head): `NORTH-slave → SOUTH-master`
  CCT (circuit; maps to VRET via `rt_acr_chan`). Packet framing rides transparently
  over the circuit hop. Symmetric to the forward SOUTH→NORTH up-hops; reuse
  `spine_top`.
- **Every head row**: arm a `RET_NORTH` packet slot on its NORTH slave and have
  its SOUTH master pull {local, transit, north} (`ACR_MSELEN_RET_HEAD 0x7`).
  Arming `RET_NORTH` unconditionally is harmless: on the topmost head (nothing
  above) that slot simply never receives traffic. This keeps
  `acr_plan_return_chain_ex` fully local — no dependency on `spine_top` or a later
  `row_add`, so the head return master is emitted exactly once (no re-emit, no
  `acr_book_port` double-book). The pass-through descent CCTs (which DO depend on
  the growing spine) stay in the incremental spine loop, skipping heads.

Port/slot safety:
- Return ports (EAST-slave, CTRL-slave, NORTH-slave, WEST-master, SOUTH-master)
  are disjoint from forward ports (WEST-slave/SOUTH-slave, EAST-master,
  CTRL-master, NORTH-master); `acr_portbook` tracks master/slave domains
  separately, so no double-book. (Head forward ingress is SOUTH *slave*; return
  descent uses NORTH *slave* + SOUTH *master* — all distinct.)
- Each return slot lives on its OWN slave port (RET_LOCAL on CTRL-slave,
  RET_TRANSIT on EAST-slave, RET_NORTH on NORTH-slave) ⇒ 1 slot/port, far within
  `ACR_NUM_SLOTS` (4).

## Part 3 — Runtime shim return leg + row read/ack API (`aie_runtime.{c,h}`)

Every column in the target row responds, so the shim must drain **N response
packets** (N = `col_hi - col_lo + 1`), each keeping its stream header (source
col/row). The host demuxes per-column by parsing headers.

- `rt_ctrl_row_shim_return(const __Runtime_CtrlRowFabric *f, int32_t s2mm_ch, int npkt)`:
  `XAie_StrmConnCctEnable(shim, NORTH, VRET, SOUTH, rport)` +
  `XAie_EnableAieToShimDmaStrmPort(shim, rport)` + arm **N** S2MM BDs (one per
  response packet, queued) via the existing `rt_tct_s2mm_arm` (mirrors
  `aie_runtime.c:3642-3659`, extended to a per-packet BD queue).
- `__Runtime_ctrl_row_read(f, row, tile_addr, nwords, out_vals, bd_id, mm2s_ch)`:
  resolve `rowidx`; broadcast/whole-row a READ control packet (reaches every
  column) with a return stream id; program shim entry + `rt_ctrl_row_shim_return`
  (npkt = ncols); `__Runtime_ctrl_push(block=1)`; drain the N packets via
  `__Runtime_ctrl_tct_poll`; parse each packet header (`__Runtime_ctrl_parse_pkt_hdr`)
  to place its data into `out_vals[col - col_lo]` via `rt_ctrl_read_extract`.
- `__Runtime_ctrl_row_write_ack(f, row, tile_addr, data, nwords, bd_id, mm2s_ch)`:
  whole-row WRITE with `lastwriteack=1`; every column returns a header-only ack;
  drain all N acks (barrier that all columns applied the write).
- Keep each new API function under 200 lines (project rule).

## Part 4 — Demo (`ctrlrow_demo.cc`)

- Remove the only-last and all-but-last `demo_probe_class` calls; keep broadcast +
  whole-row. Simplify `demo_probe_class` to drop the `is_last` expectation branch
  (whole-row ⇒ every column expects SET).
- Add a read demo: whole-row write a per-column sentinel across the row, then
  `__Runtime_ctrl_row_read` and compare the returned `out_vals[col]` against each
  column's expected value.
- Add a write-ack demo: `__Runtime_ctrl_row_write_ack` across the row, assert all N
  acks drained.

## Testing

- Host unit test `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`: assert the new
  return-chain ops — per-tile CTRL-local return slot on EVERY tile, EAST-transit
  return slot on `c < col_hi`, WEST return masters (`c > col_lo`), SOUTH return
  master on the head, VRET descent CCTs, arbiter `ACR_ARB_RET`, and no port-book
  conflicts against the forward chain. Drop the stale last-tile forward assertions
  (only-last / whole / transit-e slots, CTRL `0x7`) to match the simplified planner.
- E2E (`ctrlrow_demo.cc` via `apppaltest.py`): per-column read values match the
  prior per-column writes; all N write-acks drain; existing broadcast/whole-row
  probes still PASS to teardown with no AIE ERROR.

## Files touched

- `src/mlir/runtime/aie_runtime.c` — extract return helper (Part 1); shim return
  leg + row read/ack API (Part 3).
- `src/mlir/runtime/aie_runtime.h` — declare new row read/ack API.
- `src/mlir/runtime/aie_runtime_control_plan.c` / `.h` — return chain planner +
  constants (Part 2).
- `example/tileprogram/ccode/ctrlrow_demo.cc` — demo changes (Part 4).
- `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp` — return-chain unit tests.
- `CLAUDE.md` — update control-plane architecture notes (Document rule).
