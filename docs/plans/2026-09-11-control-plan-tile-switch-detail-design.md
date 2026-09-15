# Control-plan tile stream-switch detail — Design

Date: 2026-09-11

## Goal

Extend the aiedebug device-map **Load control plan** feature: after a control plan
is loaded and the control-plan routing overlay is shown, clicking a tile opens an
enlarged modal popup that draws that tile's stream-switch internals — slave input
ports → packet-slot match → master output fan-out — mirroring the reference
sketch:

```
            ┌──────────── stream switch ────────────┐
West0 ──→   │ slot1 match → (arb0, msel1)           │
 (slave)    │                  │                    │
            │        ┌─────────┴─────────┐          │
            │        ↓ copy              ↓ copy     │
            │   Ctrl master         East0 master    │
            └────────┼───────────────────┼─────────┘
                     ↓                   ↓
               Tile Ctrl module     next column tile's West0
             (register write, end)  (matches slot1 again, repeats)
```

## Data source (no new hardware data)

Reuse the existing `CONTROLPAN-PMAP` provenance already parsed by
`controlpan_pmap.py`. Each port record is
`{col,row,port,idx,dir,ms,id,sw,slot}`.

Within a single tile, ports that share the same `(dir, id)` belong to one route
through the switch:

- `ms == "slave"` ports are the switch **inputs**.
- `ms == "master"` ports are the fan-out **outputs**.
- `slot` / `sw` describe the packet-slot match (the `arb`/`msel` node).

This maps 1:1 to the reference: `West0` slave with a given `id` matches its
`slot`, then fans out to `Ctrl` master + `East0` master sharing the same `id`.

`CONTROLPAN-PMAP` does not record explicit `arb`/`msel` indices, so the arbiter
node is labelled `slot N (sw)` — no fabricated arb/msel numbers.

## Components

### 1. `controlpan_pmap.py` — `tile_switch_view(text, col, row)` (pure, tested)

- Parse ports (`parse_ports`) and edges (`parse_edges`).
- Filter ports to `(col,row)`; group by `(dir, id)`.
- Each group →
  `{dir, id, slot, sw, slaves:[{port,idx}], masters:[{port,idx,dest}]}`.
  - `dest` for a master is resolved from `parse_edges`: the neighbor tile
    `"(c,r) PORT"`; `CTRL` master → `"CTRL (local endpoint)"`; unpaired → `"—"`.
- Return `{col,row, groups:[...]}` (groups sorted for stable output).

### 2. `schedule_debug_server.py` — GET `/ctrlplan/tile?col=&row=`

Reads the applog text (same source as `/ctrlplan/load`) and returns
`controlpan_pmap.tile_switch_view(text, col, row)` as JSON.

### 3. `schedule_view.py` (frontend)

- New modal `#swDetailModal` (dark card + backdrop) with CSS; closes on `Esc`
  or backdrop click.
- `showTileSwitchDetail(tc,tr)`: `await api('/ctrlplan/tile?col=..&row=..')`,
  then render an SVG inside the modal:
  - left column: slave input ports,
  - middle: one `slot N (sw)` arbiter node per group,
  - right column: master output ports, each labelled with its `dest`,
  - arrows slave → slot → masters.
  Reuses the existing `svgN` builder.
- Tile click hook (existing handler at schedule_view.py:~6515): after the current
  selection logic, **only when a control plan is loaded and this tile has
  control-plan ports**, open the modal. Behavior is unchanged when no plan is
  loaded.

## Error handling

- No groups for the tile → modal shows "no control-plan ports on this tile".
- API error → `alert(...)`, matching `loadCtrlPlan`.

## Testing

- Python unit tests for `tile_switch_view`: grouping by `(dir,id)`, CTRL dest,
  neighbor-tile dest, multi-master fan-out, empty tile.
- Frontend verified via mcp-browser screenshot of the modal after Load control
  plan + tile click.

## Files touched

- `src/tool/debug/controlpan_pmap.py`
- `src/tool/debug/tests/test_controlpan_pmap.py`
- `src/tool/debug/schedule_debug_server.py`
- `src/tool/debug/schedule_view.py`
- Docs: `CLAUDE.md`, `.cursor/skills/debug-ui-framework/reference.md`
