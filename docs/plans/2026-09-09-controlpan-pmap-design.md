# CONTROLPAN-PMAP: control-plan provenance map + device-map overlay

Date: 2026-09-09

## Problem

The `aie_ctrl*` runtime API (`aie_runtime.c`) builds a *control plan*: stream-switch
routes that carry control packets from a shim column up a vertical spine (and, for
the row fabric, out along EAST chains) to CTRL ports on destination tiles, plus the
response return route back to the shim S2MM. Today these routes are programmed by
`XAie_*` calls with only scattered diagnostic prints (`[aie_runtime] ctrl_route ...`).
There is no machine-parseable record of *which ports on which tiles* the control plan
uses, so the aiedebug device-map page cannot show the control-plan routing.

## Goal

1. **C runtime:** an opt-in flag that makes the `aie_ctrl*` API emit a text
   provenance map to stdout (the applog). One line per stream port, tagged with the
   keyword `CONTROLPAN-PMAP`, carrying tile location, stream port number, port type,
   direction, master/slave, and the control/stream id.
2. **Debug UI:** a "Load control plan" button on the device-map page that parses the
   applog for `CONTROLPAN-PMAP` lines and overlays the parsed control-plan routing on
   the existing device map.

## Decisions (from brainstorming)

- **Flag mechanism:** global setter `__Runtime_ctrl_pmap_enable(int)` — no signature
  churn on existing/generated call sites.
- **Emit location (Option A):** emit at the XAie call sites inside
  `rt_ctrl_route_setup_col` and `__Runtime_ctrl_row_emit` (+ `rt_ctrl_row_shim_entry`),
  so the map mirrors exactly what was programmed. No re-derivation from structs.
- **Granularity:** one line per port.
- **Coverage:** both the same-column single-target send AND the row fabric.
- **Line fields:** col, row, port type, port idx, direction (fwd/ret), master/slave,
  id.
- **UI update:** overlay on the current device map (toggle-able), data-flow edges
  left intact.
- **Applog source:** server-side `st.applog` via a new POST endpoint.

## Line format

```
CONTROLPAN-PMAP col=<c> row=<r> port=<SOUTH|NORTH|CTRL|EAST|WEST> idx=<n> dir=<fwd|ret> ms=<master|slave> id=<id>
```

- `dir`: forward = climb / shim→dest; return = drain / dest→shim.
  From `RT_CTRL_SEL_FWD` (0) / `RT_CTRL_SEL_RET` (1).
- `ms`: `master` / `slave` = `XAIE_STRMSW_MASTER` / `XAIE_STRMSW_SLAVE`.
- `id`: `stream_id` (same-column) or `ctrl_id` (row fabric).

## C runtime design

### Global flag + emit helper (`aie_runtime.c`)

```c
static int g_ctrl_pmap = 0;
void __Runtime_ctrl_pmap_enable(int on) { g_ctrl_pmap = on; }  /* decl in aie_runtime.h */

static void rt_pmap_port(uint8_t col, uint8_t row, const char *ptype,
                         uint8_t pidx, const char *dir, const char *ms, uint32_t id) {
    if (g_ctrl_pmap)
        printf("CONTROLPAN-PMAP col=%u row=%u port=%s idx=%u dir=%s ms=%s id=%u\n",
               (unsigned)col, (unsigned)row, ptype, (unsigned)pidx, dir, ms, (unsigned)id);
}
```

Both new symbols stay well under the 200-line function limit.

### Emit call sites — same-column (`rt_ctrl_route_setup_col`)

Forward (climb):
- shim: SOUTH `fport` master (fwd) — the shim MM2S drop onto the mux.
- each pass-through row `r` in `1..dest_row-1`: NORTH `vfwd` slave + SOUTH? (the fwd
  climb is NORTH master on the tile below → NORTH `vfwd` on the pass-through). Emit
  the CCT endpoints actually programmed (`SOUTH vfwd` slave → `NORTH vfwd` master).
- dest tile: SOUTH `vfwd` slave (fwd) + CTRL master (fwd, the consume).

Return (drain):
- dest tile: CTRL slave (ret, the response emitter) + SOUTH `vret` master (ret).
- each pass-through row: NORTH `vret` slave → SOUTH `vret` master (ret).
- shim: NORTH `vret` slave → SOUTH `rport` demux master (ret) → shim S2MM.

Each `rt_pmap_port` call is placed beside its `XAie_StrmConnCctEnable` /
`XAie_StrmPktSw*` / `XAie_EnableShimDma*` call, using the same `col,row,port,idx`
already passed to the XAie call, so the map cannot drift from the programmed route.

### Emit call sites — row fabric (`__Runtime_ctrl_row_emit`)

The `acr_oplist` already carries `col,row,sport/mport,sidx/midx,pkt_id` for every op:
- `ACR_OP_CCT`: emit slave line (`sport`,`sidx`) + master line (`mport`,`midx`).
- `ACR_OP_SLAVE_EN` / `ACR_OP_SLOT`: emit slave line (`sport`,`sidx`).
- `ACR_OP_MASTER_EN`: emit master line (`mport`,`midx`).

Map `acr_port`→string via a small `rt_acr_port_name()` mirroring the existing
`rt_acr_port()`. `dir` for the row fabric: derive from the remapped vertical channel
(VFWD⇒fwd, VRET⇒ret) or default `fwd` for horizontal (EAST/WEST/CTRL) taps. `id` =
`op->pkt_id` (falls back to fabric `ctrl_id`). Also emit the shim entry programmed in
`rt_ctrl_row_shim_entry` (SOUTH `fport` master, fwd).

Enable is gated by a single `__Runtime_ctrl_pmap_enable(1)` call before the first
`setup_routing` / `row_add`.

## UI design

### Server (`schedule_debug_server.py`)

- **Standalone parser** `parse_controlpan_pmap(text)` (module-level, unit-testable):
  scans lines beginning with `CONTROLPAN-PMAP `, parses `key=val` pairs into
  `{col,row,port,idx,dir,ms,id}` (tolerant of malformed/partial lines), and synthesizes
  device-map **edges**: pair a `master` port on `(col,row)` with the `slave` port on the
  neighbor in that port's direction:
  - SOUTH master ↔ NORTH slave of tile below? (SOUTH points down; the AIE convention
    here: SOUTH master drives downward). Use the same `_OPPOSITE`/`_DELTA` convention
    already in `schedule_view._load_comm_paths`.
  - EAST master ↔ WEST slave of tile to the right.
  - CTRL = tile-local consume/emit marker (no edge; highlight tile).
  Returns `{"ports":[...], "edges":[{"from":[c,r],"to":[c,r],"dir","id"}], "count":N}`.
- **New POST endpoint** `/ctrlplan/load` in `do_POST`: reads `self.applog`, runs
  `parse_controlpan_pmap`, returns the dict via `_send_json`. Mirrors existing
  `/applog` handling (uses `self.applog`, JSON response).

### Frontend (`schedule_view.py` HTML/JS)

- `<button id="dmLoadCtrlPlan">Load control plan</button>` in `#devmap-topbar`
  (beside Scan/Clear).
- On click → `POST /ctrlplan/load` → store edges in a `ctrlPlanEdges` overlay layer →
  draw as a distinct-color/dashed overlay on the device-map SVG, on top of the
  existing `DATA.comm_paths` edges (which remain intact). Tiles referenced by the plan
  get a highlight; `dir=fwd/ret` styles the arrow direction.
- Companion toggle checkbox `ctrl plan` to show/hide the overlay; `Clear` also clears
  it.

## Testing

- **Python parser:** unit tests for `parse_controlpan_pmap` — sample multi-line applog
  → expected ports + synthesized edges (SOUTH/NORTH spine pairing, EAST chain pairing,
  CTRL consume markers, malformed-line tolerance, non-PMAP lines ignored).
- **C runtime:** no host-only test (needs XAie); verify by generating an applog on
  HW/sim with the flag on and grepping `CONTROLPAN-PMAP` lines match the programmed
  route. The format correctness is covered by the parser tests over captured output.
- **UI:** manual verify via browser MCP (navigate → Load control plan → screenshot
  overlay).

## Files touched

- `src/mlir/runtime/aie_runtime.c` — global flag, `rt_pmap_port` helper,
  `rt_acr_port_name`, emit calls in `rt_ctrl_route_setup_col`,
  `__Runtime_ctrl_row_emit`, `rt_ctrl_row_shim_entry`.
- `src/mlir/runtime/aie_runtime.h` — declare `__Runtime_ctrl_pmap_enable`.
- `src/tool/debug/schedule_debug_server.py` — `parse_controlpan_pmap` +
  `/ctrlplan/load` endpoint.
- `src/tool/debug/schedule_view.py` — button, overlay layer + toggle, draw logic.
- `src/tool/debug/tests/` — parser tests.
- Docs: `.cursor/skills/debug-ui-framework/reference.md`, `CLAUDE.md` (control-packet
  paragraph), this design doc.

## Non-goals

- No change to the actual control-plane routing behavior — pmap is observe-only.
- No client-side file picker; the button loads the session's `st.applog`.
- Larger multi-packet reads / additional fabric shapes beyond what the runtime already
  programs are out of scope.

## Status: DONE

Implemented via `docs/plans/2026-09-09-controlpan-pmap.md` (8-task TDD plan), executed
with subagent-driven-development (implementer + spec/quality review per task).

### Files changed

- `src/mlir/runtime/aie_runtime.h` — declare `__Runtime_ctrl_pmap_enable`.
- `src/mlir/runtime/aie_runtime.c` — `g_ctrl_pmap` flag, `__Runtime_ctrl_pmap_enable`,
  `rt_pmap_port` helper; emit calls in `rt_ctrl_route_setup_col` (same-column climb),
  `rt_acr_port_name` + `rt_acr_dir` + per-op emits in `__Runtime_ctrl_row_emit`, and
  the shim-entry emits in `rt_ctrl_row_shim_entry` (row fabric).
- `src/tool/debug/controlpan_pmap.py` — standalone parser: `parse_ports`, `parse_edges`,
  `parse` → `{ports, edges, count}`.
- `src/tool/debug/tests/test_controlpan_pmap.py` — 6 parser tests (ports, malformed
  tolerance, SPINE NORTH↔SOUTH pairing, EAST↔WEST chain, CTRL non-edge, summary counts).
- `src/tool/debug/schedule_debug_server.py` — `import controlpan_pmap` + `/ctrlplan/load`
  POST endpoint (reads `st.applog`, returns `controlpan_pmap.parse(text)`).
- `src/tool/debug/schedule_view.py` — **Load control plan** button + `ctrl plan` toggle,
  `ctrlPlanEdges` overlay layer, `drawCtrlPlanOverlay` / `loadCtrlPlan`, `dmClearAll` reset,
  and `.ctrlplan-edge` / `.ctrlplan-ret` CSS.
- Docs: `.cursor/skills/debug-ui-framework/reference.md` (Device map overlay bullet),
  `CLAUDE.md` (control-packet paragraph), this design doc.
