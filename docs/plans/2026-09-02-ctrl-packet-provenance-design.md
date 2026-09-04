# Control-Packet Provenance Map — Design

**Date:** 2026-09-02
**Topic:** Make `schedule_debug_server.py` start a debug web GUI for control-packet
(`__Runtime_ctrl_setup_routing`) apps by extending the static provenance generator.

## Problem

Running:

```
python3 src/tool/debug/schedule_debug_server.py aout/worklocal --elf aout/worklocal/build/host --aie-version 5 --open
```

fails with:

```
[AppRegistry] warning: .../aout/worklocal is neither a provenance bundle nor an app directory containing Work/
error: explicit app has no schedule_view.json and no usable Work/ tree
```

### Root cause

`aout/worklocal/host.cc` drives the AIE through the **control-packet** runtime API
(`__Runtime_CtrlInstance` + `__Runtime_ctrl_setup_routing` / `__Runtime_ctrl_read_target`
/ `__Runtime_ctrl_push_target`), *not* the `XAie_LoadElfMem` / `XAie_MoveDataExternal2Aie`
/ `XAie_MoveDataAie2External` / `XAie_Route` calls that the static generator
`src/tool/debug/xaiehost2provenance.py` matches.

`script/aiehlc.sh:50` runs `xaiehost2provenance.py`, but it finds zero recognized
calls, prints "no XAie routing found - skipping provenance", and writes nothing. With
no `dfscheduleprovenancemap.json` there is no `schedule_view.json`, so
`AppRegistry._resolve_explicit` rejects the directory.

## Chosen approach

**Extend the Python parser** (`xaiehost2provenance.py`) to recognize control-packet
routing and emit the *same* `dfscheduleprovenancemap.json` + `dmaphopprovenacemap.json`
schema. Build-time only. No C changes to `__Runtime_ctrl_setup_routing`. Downstream
(`schedule_view.py`, `schedule_debug_server.py`) is reused unchanged.

Rejected alternatives:
- *Emit JSON from the C runtime at run time* — requires a board run + pulling the file
  back; heavier and inconsistent with the existing static provenance model.
- *Both* — unnecessary for the stated goal.

## Control-packet topology (from `rt_ctrl_route_setup_col` in `src/mlir/runtime/aie_runtime.c`)

Same-column vertical, `dest_col == shim_col`:

- **Forward (request):** shim `(col,0)` MM2S → SOUTH mux → climb NORTH through rows
  `1..dest_row-1` → dest `(col,dest_row)` CTRL master. Circuit-switched.
- **Return (response):** dest `(col,dest_row)` CTRL slave → SOUTH → climb down through
  rows `dest_row-1..1` → shim `(col,0)` S2MM. Header dropped; `resp_words` data words.

No kernel ELF is loaded on this path (register read/write), so `kernel_placements`
stays empty.

## Design

Single-file change to `xaiehost2provenance.py` (+ tests). Merges into the existing
`extract_model` → `build_dfschedule` / `build_dmaphop` pipeline.

### 1. Parse control-packet sends

Recognize two source forms and reduce each to a send tuple
`(shim_col, dest_row, resp_words)` (with `dest_col == shim_col`):

- **Designated-initializer struct:**
  `__Runtime_CtrlInstance <name> = { .dev=..., .shim_col=C, .dest_col=C, .dest_row=R, ..., .resp_words=W };`
  Parse the `.field = value` pairs; resolve each value with the existing `eval_int`
  + `defs` machinery so `#define`s and `u32 x = ...;` locals fold. `resp_words`
  defaults to 1 when absent.
- **Composite call form:** `__Runtime_ctrl_read_target(dev, shim_col, dest_col, dest_row, ...)`
  and `__Runtime_ctrl_push_target(dev, shim_col, dest_col, dest_row, ...)` — pull
  `shim_col`/`dest_col`/`dest_row` from the positional args.

Dedupe sends by `(shim_col, dest_row)`.

### 2. Tiles (geometry-aware typing)

For each send add: shim `(col,0)`, dest `(col,dest_row)`, and the vertical
pass-through tiles rows `1..dest_row-1` (they carry the circuit-switched climb).

Tile type by row, per AIE gen:

```
core_start = {1:1, 2:2, 5:3}[gen]   # mirrors C rt_port_evt_base / aiediag.AIE_TILE_ROW_START
row == 0            -> "shim"
0 < row < core_start -> "memtile"
row >= core_start   -> "core"
```

(Gen5 / AIE2PS: MemTileNumRows=2 → rows 1,2 are memtile, cores start at row 3.)

### 3. Flows (two per send)

- Forward `shim(col,0) → dest(col,dest_row)`, `direction = "S2MM"` (push up),
  `len = resp_words*4` placeholder.
- Return `dest(col,dest_row) → shim(col,0)`, `direction = "MM2S"` (drain down),
  `len = resp_words*4`.

These merge into the same `flows` list `build_dfschedule` / `build_dmaphop` already
consume, so `dma_channels` / `communication_paths` fall out unchanged.

### 4. dmaphop hops

Emit the real vertical hop chain per flow (shim→…→dest per intermediate row) instead
of one direct hop, since the control climb is a known straight vertical path (unlike
`XAie_Route` which is runtime-decided). `direction`: forward = `push`, return = `pull`.

### 5. Wiring

- `extract_model`: after the existing XAie extraction, run control-packet extraction
  and append to the same `tiles`/`flows`. Pass `gen` for typing.
- `find_entry_fn`: also treat `__Runtime_ctrl_` calls as body markers so
  `host_entry_fn` is surfaced for the ctrl-packet host.cc entry function.
- `main`/writers unchanged; `model_is_empty` becomes false so JSONs are written.

## Data flow (downstream unchanged)

```
host.cc (ctrl-packet) → xaiehost2provenance.py → dfscheduleprovenancemap.json
                                                + dmaphopprovenacemap.json
                       → schedule_view.py       → schedule_view.json + host_schedule.html
                       → schedule_debug_server.py loads the bundle
```

## Error handling / graceful degrade

- No XAie *and* no ctrl-packet matches → keep existing "no routing found — skipping".
- A send whose `shim_col`/`dest_row` cannot be `eval_int`-resolved → skip that send
  (same policy as current tile-loc resolution failures).

## Testing

Add to `src/tool/debug/tests/test_xaiehost2provenance.py`:
- designated-init `__Runtime_CtrlInstance` → expected shim/dest tiles + 2 flows.
- `__Runtime_ctrl_read_target(...)` call form.
- geometry-aware memtile-row dest typing (gen5 dest_row=2 → "memtile", 3 → "core").
- pass-through tiles present for `dest_row=3`.
- graceful degrade unchanged when neither pattern present.

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -v`

## Post-implementation regeneration

Regenerate against the existing bundle so the target command works immediately:

```
python3 src/tool/debug/xaiehost2provenance.py aout/worklocal/host.cc \
    --out-dir aout/worklocal --aie-version 5 --platform baremetal \
    --artifacts-dir aout/worklocal
python3 src/tool/debug/schedule_view.py aout/worklocal
```

## Files

- Modify: `src/tool/debug/xaiehost2provenance.py`
- Modify: `src/tool/debug/tests/test_xaiehost2provenance.py`
- Regenerate: `aout/worklocal/dfscheduleprovenancemap.json`,
  `aout/worklocal/dmaphopprovenacemap.json`, `aout/worklocal/schedule_view.json`,
  `aout/worklocal/host_schedule.html`
