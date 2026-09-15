# CONTROLPAN-PMAP detail: env-gate + switch-type/slot overlay

Date: 2026-09-09

## Problem

The device-map **Load control plan** button does nothing because:

1. `__Runtime_ctrl_pmap_enable(1)` is never called, so `g_ctrl_pmap` stays 0 and **no
   `CONTROLPAN-PMAP` lines are ever emitted**. The captured applog has 0 of them, so
   the parser returns empty and the overlay stays blank.
2. Even when emitted, the line format only carries `col row port idx dir ms id` — it
   lacks the **switch type (pkt/circuit)** and **slot number**, so the overlay cannot
   show "which master/slave, port#, slot#, pkt vs circuit, CTRL vs spine".

## Decisions (from brainstorming)

- **Emit gating:** env-var auto-gate. Runtime resolves `AIE_CTRL_PMAP` from the
  environment on first use (tri-state `g_ctrl_pmap=-1`). `__Runtime_ctrl_pmap_enable`
  still forces it explicitly. No code regen to toggle; export the var and re-run.
- **Detail scope:** full. Add `sw=<pkt|circuit>` and `slot=<n>` (`-1` = N/A) to each
  line. Overlay renders per-hop labels + tooltips: master→slave, port#, slot#,
  pkt/circuit, and highlights CTRL consume/emit tiles.

## Line format (extended, backward compatible)

```
CONTROLPAN-PMAP col=<c> row=<r> port=<SOUTH|NORTH|CTRL|EAST|WEST> idx=<n> \
    dir=<fwd|ret> ms=<master|slave> id=<id> sw=<pkt|circuit> slot=<n>
```

Parser tolerates old lines missing `sw`/`slot` (defaults `sw=circuit`, `slot=-1`).

## Per-hop switch type (authoritative, from aie_runtime.c)

Same-column (`rt_ctrl_route_setup_col`):
- forward climb (shim SOUTH→NORTH, pass-through, dest SOUTH→CTRL): `circuit`, slot=-1.
- return dest `CTRL slave` + `SOUTH(vret) master`: `pkt`, slot=0 (StrmPktSw* @3572/3594).
- return pass-through + shim NORTH→SOUTH: `circuit`, slot=-1.

Row fabric (`__Runtime_ctrl_row_emit`), by `op->kind`:
- `ACR_OP_SLOT`: `pkt`, slot=`op->slot`.
- `ACR_OP_SLAVE_EN` / `ACR_OP_MASTER_EN`: `pkt`, slot=-1.
- `ACR_OP_CCT`: `circuit`, slot=-1 (both slave and master lines).

## Tasks

### Task 1 — parser: sw/slot fields + edge enrichment (TDD)

- Test first (`tests/test_controlpan_pmap.py`): a line with `sw=pkt slot=0` parses
  `sw`,`slot`; a legacy line without them defaults `sw="circuit"`,`slot=-1`; an edge
  carries `sw`,`slot`,`from_idx`,`to_idx`.
- `controlpan_pmap.py`: add `sw`,`slot` to `parse_ports` (slot int; tolerant), carry
  `sw`,`slot`,`from_idx` (master idx),`to_idx` (slave idx) onto each edge.

### Task 2 — C runtime: env-gate + rt_pmap_port(sw,slot) + call sites

- `g_ctrl_pmap=-1`; `rt_ctrl_pmap_on()` resolves `AIE_CTRL_PMAP` once.
- `rt_pmap_port(...,const char *sw,int slot)` prints `sw=%s slot=%d`.
- Update all same-column + row-fabric call sites with the switch types above.
- Keep every function < 200 lines.

### Task 3 — frontend overlay: labels, tooltips, pkt/circuit styling, CTRL tiles

- `drawCtrlPlanOverlay`: style `pkt` solid vs `circuit` dashed; `ret` cyan / `fwd`
  pink; midpoint label `port from_idx→to_idx sw [sN]`; `<title>` tooltip with full
  detail; arrow marker for direction.
- Store `j.ports`; highlight CTRL tiles (consume=fwd / emit=ret) with a marker+title.
- CSS for pkt/circuit/label.

### Task 4 — verify (DONE)

- Parser unit tests green: 7 passed (`tests/test_controlpan_pmap.py`).
- Representative applog sample in the new format (`/tmp/claude/ctrlplan_sample.log`,
  read-with-return climb) → launched `schedule_debug_server.py aout/worklocal
  --applog <sample>` → `POST /ctrlplan/load` returns 12 ports / 5 edges with
  `sw`,`slot`,`from_idx`,`to_idx` (3 fwd circuit climb NORTH, 2 ret pkt slot=0
  descent SOUTH).
- Served page (`GET /`) confirmed to contain `drawCtrlPlanOverlay`, `ctrlPlanPorts`
  decl, `loadCtrlPlan` port store, `dmClearAll` port reset, and CSS
  `.ctrlplan-lbl/.ctrlplan-ctrl/.ctrlplan-ctrl-emit`. (Live browser-MCP screenshot
  skipped — mcp-browser tools not loaded this session; endpoint + served-page checks
  cover the data path.)
- Real HW regeneration: `AIE_CTRL_PMAP=1` in the board run env (harness propagates the
  var to the remote host ELF), re-run controlperf → applog has the lines.

## Files touched

- `src/tool/debug/controlpan_pmap.py`, `src/tool/debug/tests/test_controlpan_pmap.py`
- `src/mlir/runtime/aie_runtime.c`
- `src/tool/debug/schedule_view.py`
- Docs: `.cursor/skills/debug-ui-framework/reference.md`, `CLAUDE.md`, this doc.

## Non-goals

- No routing-behavior change (observe-only).
- Harness env-var propagation to the board is noted, not auto-wired here.
