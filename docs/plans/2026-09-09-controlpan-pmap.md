# CONTROLPAN-PMAP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in runtime flag that emits per-port control-plan provenance lines (`CONTROLPAN-PMAP`) from the `aie_ctrl*` API, plus a device-map "Load control plan" button that parses the applog and overlays the control-plan routing.

**Architecture:** In `aie_runtime.c`, a global toggle (`__Runtime_ctrl_pmap_enable`) gates a small `rt_pmap_port()` helper that prints one keyword-tagged line per stream port, called beside every stream-switch programming call in the same-column (`rt_ctrl_route_setup_col`) and row-fabric (`__Runtime_ctrl_row_emit`, `rt_ctrl_row_shim_entry`) paths. On the UI side, a standalone Python parser (`controlpan_pmap.py`) turns those lines into device-map edges; a new `/ctrlplan/load` server endpoint reads the session applog and returns them; a device-map button draws them as a toggle-able overlay.

**Tech Stack:** C (XAie driver), Python 3 (stdlib http.server, pytest), inline HTML/JS in `schedule_view.py`.

Design doc: `docs/plans/2026-09-09-controlpan-pmap-design.md`.

---

## Notes for the implementer

- **Env-banner noise:** every shell command prints a shell-profile banner to stdout. Redirect pytest to a file and read it: `python3 -m pytest <args> > /tmp/claude/out.txt 2>&1; cat /tmp/claude/out.txt`.
- **Run pytest from repo root** so `sys.path.insert(0, ..)` in the test resolves `import controlpan_pmap`.
- **Do not build/run HW** for these tasks; the C emit is verified by generating an applog separately (out of plan scope) and by the parser tests over sample lines.
- **`never do` rule:** no runtime function may exceed 200 lines. The new helpers are tiny; the emit calls add ≤2 lines each at existing call sites.
- Reference skills: @kernelcodegen and @hostcodegen are NOT needed; this is runtime + debug tooling. Use @debug-ui-framework for the UI file map.

---

## Task 1: C runtime global flag + emit helper

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` (add near the other `rt_ctrl_*` statics, before `rt_ctrl_route_setup_col` ~line 3454)
- Modify: `src/mlir/runtime/aie_runtime.h` (declare next to `__Runtime_ctrl_setup_routing`, ~line 786)

**Step 1: Declare the enable in the header**

In `aie_runtime.h`, after the `__Runtime_ctrl_setup_routing` declaration (~line 786), add:

```c
// Enable (on!=0) or disable per-port control-plan provenance logging. When on,
// the aie_ctrl* routing setup prints one `CONTROLPAN-PMAP ...` line per stream
// port it programs (tile location, port type/idx, direction, master/slave, id)
// to stdout (the applog). Call once before the first setup_routing / row_add.
void __Runtime_ctrl_pmap_enable(int on);
```

**Step 2: Add the flag + helper in the .c**

In `aie_runtime.c`, just above `rt_ctrl_route_setup_col` (~line 3454), add:

```c
/* Control-plan provenance map (CONTROLPAN-PMAP). Opt-in via
 * __Runtime_ctrl_pmap_enable. Emits one line per programmed stream port so the
 * aiedebug device-map can overlay the control-plan routing. */
static int g_ctrl_pmap = 0;
void __Runtime_ctrl_pmap_enable(int on) { g_ctrl_pmap = on; }

static void rt_pmap_port(uint8_t col, uint8_t row, const char *ptype, uint8_t pidx, const char *dir, const char *ms,
                         uint32_t id) {
    if (g_ctrl_pmap)
        printf("CONTROLPAN-PMAP col=%u row=%u port=%s idx=%u dir=%s ms=%s id=%u\n", (unsigned)col, (unsigned)row, ptype,
               (unsigned)pidx, dir, ms, (unsigned)id);
}
```

**Step 3: Compile-check the runtime translation unit**

The runtime is normally compiled as part of the host build (needs XAie headers/toolchain), which is heavy. Instead do a syntax-only check that the new symbols parse by grepping they exist and are balanced:

Run: `grep -n "__Runtime_ctrl_pmap_enable\|rt_pmap_port" src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h`
Expected: the decl in the .h and the two definitions + helper in the .c.

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(aie_runtime): add CONTROLPAN-PMAP flag + rt_pmap_port helper"
```

---

## Task 2: Emit CONTROLPAN-PMAP in the same-column route (`rt_ctrl_route_setup_col`)

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` — inside `rt_ctrl_route_setup_col` (~lines 3454-3616)

**Step 1: Add forward-path emit calls**

Beside each forward programming call in `rt_ctrl_route_setup_col`, add a matching `rt_pmap_port`. Use the variables already in scope (`shim_col`, `dest_row`, `fport`, `vfwd`, `stream_id`, loop `r`). Concretely:

- After the shim MM2S drop (`XAie_EnableShimDmaToAieStrmPort` / the SOUTH `fport`→NORTH `vfwd` CCT near the top of the function):
  ```c
  rt_pmap_port(shim_col, 0, "SOUTH", fport, "fwd", "master", stream_id);
  rt_pmap_port(shim_col, 0, "NORTH", vfwd, "fwd", "master", stream_id);
  ```
- Inside the forward pass-through loop (rows `1..dest_row-1`), after the `SOUTH vfwd -> NORTH vfwd` CCT:
  ```c
  rt_pmap_port(shim_col, (uint8_t)r, "SOUTH", vfwd, "fwd", "slave", stream_id);
  rt_pmap_port(shim_col, (uint8_t)r, "NORTH", vfwd, "fwd", "master", stream_id);
  ```
- After the dest `SOUTH vfwd -> CTRL` CCT (line ~3507):
  ```c
  rt_pmap_port(shim_col, dest_row, "SOUTH", vfwd, "fwd", "slave", stream_id);
  rt_pmap_port(shim_col, dest_row, "CTRL", 0, "fwd", "master", stream_id);
  ```

(Adjust port/idx names to the actual arguments passed to the neighboring XAie call — the map must mirror the programmed call exactly.)

**Step 2: Add return-path emit calls**

- After the dest CTRL slave slot/port + SOUTH `vret` master enables (lines ~3553-3575):
  ```c
  rt_pmap_port(shim_col, dest_row, "CTRL", 0, "ret", "slave", stream_id);
  rt_pmap_port(shim_col, dest_row, "SOUTH", vret, "ret", "master", stream_id);
  ```
- Inside the return pass-through loop (line ~3583), after the `NORTH vret -> SOUTH vret` CCT:
  ```c
  rt_pmap_port(shim_col, (uint8_t)r, "NORTH", vret, "ret", "slave", stream_id);
  rt_pmap_port(shim_col, (uint8_t)r, "SOUTH", vret, "ret", "master", stream_id);
  ```
- After the shim `NORTH vret -> SOUTH rport` CCT (line ~3596):
  ```c
  rt_pmap_port(shim_col, 0, "NORTH", vret, "ret", "slave", stream_id);
  rt_pmap_port(shim_col, 0, "SOUTH", rport, "ret", "master", stream_id);
  ```

**Step 3: Verify placement + line-count budget**

Run: `grep -n "rt_pmap_port" src/mlir/runtime/aie_runtime.c`
Expected: ~12 calls inside `rt_ctrl_route_setup_col`, none breaking the surrounding `if (rc != XAIE_OK)` blocks. Confirm the function still ends at its original closing brace and stays under 200 lines (it already was; we add ≤12 one-liners).

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c
git commit -m "feat(aie_runtime): emit CONTROLPAN-PMAP for same-column ctrl route"
```

---

## Task 3: Emit CONTROLPAN-PMAP in the row fabric (`__Runtime_ctrl_row_emit` + shim entry)

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` — `rt_acr_port` (~3619), `__Runtime_ctrl_row_emit` (~3655), `rt_ctrl_row_shim_entry` (~3727)

**Step 1: Add an acr_port → name mapper**

Just below `rt_acr_port` (~line 3633) add:

```c
/* Human-readable port name for CONTROLPAN-PMAP (mirrors rt_acr_port). */
static const char *rt_acr_port_name(acr_port p) {
    switch (p) {
    case ACR_WEST: return "WEST";
    case ACR_EAST: return "EAST";
    case ACR_NORTH: return "NORTH";
    case ACR_SOUTH: return "SOUTH";
    case ACR_CTRL:
    default: return "CTRL";
    }
}

/* fwd/ret for a remapped vertical spine channel; horizontal taps default fwd. */
static const char *rt_acr_dir(acr_port p, uint8_t chan) {
    if (p == ACR_NORTH || p == ACR_SOUTH)
        return chan == RT_CTRL_VRET ? "ret" : "fwd";
    return "fwd";
}
```

**Step 2: Emit per-op inside `__Runtime_ctrl_row_emit`**

Inside the `for` loop in `__Runtime_ctrl_row_emit` (after `midx`/`sidx` are computed, ~line 3660), before the `switch`, add per-kind emits using the already-computed remapped `sidx`/`midx`:

```c
switch (op->kind) {
case ACR_OP_SLOT:
case ACR_OP_SLAVE_EN:
    rt_pmap_port(op->col, op->row, rt_acr_port_name(op->sport), sidx, rt_acr_dir(op->sport, sidx), "slave",
                 op->pkt_id);
    break;
case ACR_OP_MASTER_EN:
    rt_pmap_port(op->col, op->row, rt_acr_port_name(op->mport), midx, rt_acr_dir(op->mport, midx), "master",
                 op->pkt_id);
    break;
case ACR_OP_CCT:
    rt_pmap_port(op->col, op->row, rt_acr_port_name(op->sport), sidx, rt_acr_dir(op->sport, sidx), "slave",
                 op->pkt_id);
    rt_pmap_port(op->col, op->row, rt_acr_port_name(op->mport), midx, rt_acr_dir(op->mport, midx), "master",
                 op->pkt_id);
    break;
default:
    break;
}
```

Place this new `switch` immediately BEFORE the existing `switch (op->kind)` that does the XAie calls (a separate observe-only switch), so it never alters control flow. Keep it gated by `g_ctrl_pmap` (already inside `rt_pmap_port`).

**Step 3: Emit the shim entry**

In `rt_ctrl_row_shim_entry` (~line 3736), after the successful `SOUTH fport -> NORTH VFWD` CCT, add:

```c
rt_pmap_port(f->shim_col, 0, "SOUTH", fport, "fwd", "master", f->ctrl_id);
rt_pmap_port(f->shim_col, 0, "NORTH", RT_CTRL_VFWD, "fwd", "master", f->ctrl_id);
```

**Step 4: Verify**

Run: `grep -n "rt_acr_port_name\|rt_acr_dir\|rt_pmap_port" src/mlir/runtime/aie_runtime.c`
Expected: the two new static helpers + the per-op emits in `__Runtime_ctrl_row_emit` + the two shim-entry emits. `__Runtime_ctrl_row_emit` stays under 200 lines.

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c
git commit -m "feat(aie_runtime): emit CONTROLPAN-PMAP for row-fabric ctrl route"
```

---

## Task 4: Python parser `controlpan_pmap.py` — parse lines to ports (TDD)

**Files:**
- Create: `src/tool/debug/controlpan_pmap.py`
- Create: `src/tool/debug/tests/test_controlpan_pmap.py`

**Step 1: Write the failing test (parse ports)**

Create `src/tool/debug/tests/test_controlpan_pmap.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import controlpan_pmap as c

SAMPLE = """
some unrelated log line
CONTROLPAN-PMAP col=0 row=0 port=SOUTH idx=3 dir=fwd ms=master id=1
CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1
[aie_runtime] ctrl_route ok: ...
CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=ret ms=slave id=1
"""

def test_parse_ports_basic():
    ports = c.parse_ports(SAMPLE)
    assert len(ports) == 3
    assert ports[0] == {"col": 0, "row": 0, "port": "SOUTH", "idx": 3,
                        "dir": "fwd", "ms": "master", "id": 1}
    assert ports[1]["port"] == "CTRL" and ports[1]["ms"] == "master"
    assert ports[2]["dir"] == "ret"

def test_parse_ports_ignores_malformed():
    bad = "CONTROLPAN-PMAP col=x row= port=SOUTH\nCONTROLPAN-PMAP col=1 row=2 port=EAST idx=0 dir=fwd ms=master id=7\n"
    ports = c.parse_ports(bad)
    assert ports == [{"col": 1, "row": 2, "port": "EAST", "idx": 0,
                      "dir": "fwd", "ms": "master", "id": 7}]
```

**Step 2: Run to verify it fails**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py -v > /tmp/claude/pmap.txt 2>&1; cat /tmp/claude/pmap.txt`
Expected: FAIL — `ModuleNotFoundError: No module named 'controlpan_pmap'`.

**Step 3: Write minimal parser**

Create `src/tool/debug/controlpan_pmap.py`:

```python
"""Parse CONTROLPAN-PMAP applog lines into device-map ports and edges.

A CONTROLPAN-PMAP line (emitted by the aie_ctrl* runtime when
__Runtime_ctrl_pmap_enable is on) describes one programmed stream port:

    CONTROLPAN-PMAP col=<c> row=<r> port=<SOUTH|NORTH|CTRL|EAST|WEST> idx=<n> \
        dir=<fwd|ret> ms=<master|slave> id=<id>

parse_ports() returns the port dicts; parse_edges() pairs master ports with the
slave port on the adjacent tile to synthesize device-map routing edges.
"""
import re

_TAG = "CONTROLPAN-PMAP"
_KV = re.compile(r"(\w+)=(\S+)")
_INT_KEYS = ("col", "row", "idx", "id")

# Master-port direction -> neighbor tile delta (col,row). AIE grid: row 0 = shim
# at the bottom; NORTH points up (+row), SOUTH down (-row), EAST +col, WEST -col.
_DELTA = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
_OPP = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


def parse_ports(text):
    ports = []
    for line in text.splitlines():
        i = line.find(_TAG)
        if i < 0:
            continue
        kv = dict(_KV.findall(line[i + len(_TAG):]))
        if not {"col", "row", "port", "idx", "dir", "ms", "id"} <= set(kv):
            continue
        try:
            rec = {k: (int(kv[k]) if k in _INT_KEYS else kv[k])
                   for k in ("col", "row", "port", "idx", "dir", "ms", "id")}
        except ValueError:
            continue
        ports.append(rec)
    return ports


def parse_edges(text):
    """Pair each master port with the opposite-direction slave port on the
    neighbor tile. CTRL master/slave are tile-local consume/emit markers (no
    edge). Returns a list of {from:[c,r], to:[c,r], dir, id, port} edges."""
    ports = parse_ports(text)
    slaves = {(p["col"], p["row"], p["port"]): p
              for p in ports if p["ms"] == "slave"}
    edges = []
    for p in ports:
        if p["ms"] != "master" or p["port"] not in _DELTA:
            continue
        dc, dr = _DELTA[p["port"]]
        nb = (p["col"] + dc, p["row"] + dr, _OPP[p["port"]])
        if nb in slaves:
            edges.append({"from": [p["col"], p["row"]],
                          "to": [nb[0], nb[1]], "dir": p["dir"],
                          "id": p["id"], "port": p["port"]})
    return edges


def parse(text):
    """Convenience: {ports, edges, count}."""
    ports = parse_ports(text)
    return {"ports": ports, "edges": parse_edges(text), "count": len(ports)}
```

**Step 4: Run to verify parse_ports tests pass**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py -v > /tmp/claude/pmap.txt 2>&1; cat /tmp/claude/pmap.txt`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/tool/debug/controlpan_pmap.py src/tool/debug/tests/test_controlpan_pmap.py
git commit -m "feat(debug): add controlpan_pmap parser (ports)"
```

---

## Task 5: Parser edge synthesis tests (TDD)

**Files:**
- Modify: `src/tool/debug/tests/test_controlpan_pmap.py`

**Step 1: Add failing edge tests**

Append to `test_controlpan_pmap.py`:

```python
SPINE = """
CONTROLPAN-PMAP col=0 row=0 port=NORTH idx=0 dir=fwd ms=master id=1
CONTROLPAN-PMAP col=0 row=1 port=SOUTH idx=0 dir=fwd ms=slave id=1
CONTROLPAN-PMAP col=0 row=1 port=NORTH idx=0 dir=fwd ms=master id=1
CONTROLPAN-PMAP col=0 row=2 port=SOUTH idx=0 dir=fwd ms=slave id=1
"""

def test_edges_spine_pairs_north_master_to_south_slave():
    edges = c.parse_edges(SPINE)
    assert {"from": [0, 0], "to": [0, 1], "dir": "fwd", "id": 1, "port": "NORTH"} in edges
    assert {"from": [0, 1], "to": [0, 2], "dir": "fwd", "id": 1, "port": "NORTH"} in edges
    assert len(edges) == 2

EAST = """
CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=fwd ms=master id=2
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=0 dir=fwd ms=slave id=2
"""

def test_edges_east_chain_pairs_east_master_to_west_slave():
    edges = c.parse_edges(EAST)
    assert edges == [{"from": [0, 3], "to": [1, 3], "dir": "fwd", "id": 2, "port": "EAST"}]

def test_edges_ctrl_is_not_an_edge():
    ctrl = "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1\n"
    assert c.parse_edges(ctrl) == []

def test_parse_summary_counts():
    r = c.parse(SPINE)
    assert r["count"] == 4 and len(r["edges"]) == 2
```

**Step 2: Run to verify pass (parser already handles these)**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py -v > /tmp/claude/pmap.txt 2>&1; cat /tmp/claude/pmap.txt`
Expected: 6 passed. (If EAST/SPINE pairing fails, the `_DELTA`/`_OPP` maps in Task 4 need fixing — fix minimally, re-run.)

**Step 3: Commit**

```bash
git add src/tool/debug/tests/test_controlpan_pmap.py
git commit -m "test(debug): controlpan_pmap edge synthesis"
```

---

## Task 6: Server endpoint `/ctrlplan/load`

**Files:**
- Modify: `src/tool/debug/schedule_debug_server.py` — import (top), `do_POST` dispatch (~line 5303+)

**Step 1: Import the parser**

Near the other local imports at the top of `schedule_debug_server.py`, add:

```python
import controlpan_pmap
```

(The debug tools run with `src/tool/debug` on `sys.path`; if not, add `sys.path` insert mirroring the existing pattern in that file.)

**Step 2: Add the endpoint in `do_POST`**

In `do_POST`, add a new branch alongside the others (e.g. after the `/timeline` branch, ~line 5335):

```python
elif u.path == "/ctrlplan/load":
    try:
        with open(st.applog, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        self._send_json({"error": f"cannot read applog: {e}",
                         "ports": [], "edges": [], "count": 0})
        return
    self._send_json(controlpan_pmap.parse(text))
```

**Step 3: Smoke-test the endpoint logic without a live board**

Write a tiny scratch check (not committed) that the branch is reachable: grep it exists and the parser import resolves.

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -c "import sys; sys.path.insert(0,'src/tool/debug'); import controlpan_pmap; print(controlpan_pmap.parse('CONTROLPAN-PMAP col=0 row=0 port=NORTH idx=0 dir=fwd ms=master id=1\nCONTROLPAN-PMAP col=0 row=1 port=SOUTH idx=0 dir=fwd ms=slave id=1'))" > /tmp/claude/ep.txt 2>&1; cat /tmp/claude/ep.txt`
Expected: a dict with `count=2`, one edge `(0,0)->(0,1)`.

Run: `grep -n "/ctrlplan/load\|import controlpan_pmap" src/tool/debug/schedule_debug_server.py`
Expected: both present.

**Step 4: Commit**

```bash
git add src/tool/debug/schedule_debug_server.py
git commit -m "feat(debug-server): /ctrlplan/load parses applog CONTROLPAN-PMAP"
```

---

## Task 7: Device-map "Load control plan" button + overlay (frontend)

**Files:**
- Modify: `src/tool/debug/schedule_view.py` — device-map topbar HTML (~line 3326-3332), device-map JS (edge-draw section, search for `devmap-svg` / `comm_paths` draw)

**Step 1: Add the button + toggle to the topbar**

In the `#devmap-vp` toolbar row (near `dmSwWrap`, ~line 3328), add:

```html
<button id="dmLoadCtrlPlan" title="parse applog CONTROLPAN-PMAP lines and overlay the control-plan routing">Load control plan</button>
<label id="dmCtrlPlanWrap" title="show/hide the parsed control-plan overlay" hidden><input type="checkbox" id="dmCtrlPlanToggle" checked> ctrl plan</label>
```

**Step 2: Add the JS overlay state + fetch + draw**

In the device-map `<script>` section (where `DATA.comm_paths` edges are drawn onto `#devmap-svg`), add a module-level overlay array and a draw function. Find the existing edge-draw routine (grep `devmap-svg` and the function that iterates `comm_paths`) and add a sibling that draws `ctrlPlanEdges` with a distinct dashed stroke:

```javascript
let ctrlPlanEdges = [];   // [{from:[c,r], to:[c,r], dir, id, port}]

async function loadCtrlPlan(){
  const btn = document.getElementById('dmLoadCtrlPlan');
  btn.disabled = true;
  try{
    const r = await fetch('/ctrlplan/load', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'});
    const j = await r.json();
    if(j.error){ alert('Load control plan: '+j.error); return; }
    ctrlPlanEdges = j.edges || [];
    document.getElementById('dmCtrlPlanWrap').hidden = ctrlPlanEdges.length===0;
    document.getElementById('dmCtrlPlanToggle').checked = true;
    drawDevmap();   // existing full redraw entry point
  } finally { btn.disabled = false; }
}

function drawCtrlPlanOverlay(svg){
  if(!document.getElementById('dmCtrlPlanToggle')?.checked) return;
  for(const e of ctrlPlanEdges){
    // reuse the same tile-center helper the data-flow edges use:
    const a = tileCenter(e.from[0], e.from[1]);
    const b = tileCenter(e.to[0],   e.to[1]);
    if(!a||!b) continue;
    const ln = document.createElementNS('http://www.w3.org/2000/svg','line');
    ln.setAttribute('x1',a.x); ln.setAttribute('y1',a.y);
    ln.setAttribute('x2',b.x); ln.setAttribute('y2',b.y);
    ln.setAttribute('class', 'ctrlplan-edge '+(e.dir==='ret'?'ctrlplan-ret':'ctrlplan-fwd'));
    ln.setAttribute('stroke-dasharray','4 3');
    svg.appendChild(ln);
  }
}
```

**Note for implementer:** `tileCenter` and `drawDevmap` are illustrative names. Grep the actual helper that maps `(col,row)` → svg coords and the actual redraw entry point in `schedule_view.py`, and wire `drawCtrlPlanOverlay(svg)` into the end of that redraw so the overlay is drawn on top. Add matching CSS (`.ctrlplan-edge{stroke:#e91e63;stroke-width:2;opacity:.85} .ctrlplan-ret{stroke:#00bcd4}`) in the device-map `<style>` block.

**Step 3: Wire the button + toggle handlers**

Where other device-map controls bind (grep `dmScanBtn`), add:

```javascript
document.getElementById('dmLoadCtrlPlan').onclick = loadCtrlPlan;
document.getElementById('dmCtrlPlanToggle').onchange = drawDevmap;
```

Ensure `dmClearBtn` also clears the overlay: in the Clear handler, add `ctrlPlanEdges=[]; document.getElementById('dmCtrlPlanWrap').hidden=true;`.

**Step 4: Verify the page still renders (server not required)**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -c "import sys; sys.path.insert(0,'src/tool/debug'); import schedule_view; print('import ok')" > /tmp/claude/sv.txt 2>&1; cat /tmp/claude/sv.txt`
Expected: `import ok` (no syntax error in the edited module).

Run: `grep -n "dmLoadCtrlPlan\|ctrlPlanEdges\|drawCtrlPlanOverlay" src/tool/debug/schedule_view.py`
Expected: button HTML + JS state + draw fn + handlers all present.

**Step 5: Manual browser verification (optional, needs a live applog with CONTROLPAN-PMAP lines)**

Using the browser MCP workflow (see @debug-ui-framework): start `schedule_debug_server.py` on a bundle whose applog contains `CONTROLPAN-PMAP` lines → navigate → click **Load control plan** → screenshot → confirm the dashed overlay edges appear and the `ctrl plan` toggle hides/shows them.

**Step 6: Commit**

```bash
git add src/tool/debug/schedule_view.py
git commit -m "feat(devmap): Load control plan button + CONTROLPAN-PMAP overlay"
```

---

## Task 8: Docs

**Files:**
- Modify: `.cursor/skills/debug-ui-framework/reference.md`
- Modify: `CLAUDE.md`
- Modify: `docs/plans/2026-09-09-controlpan-pmap-design.md` (mark done + files-changed)

**Step 1: debug-ui-framework reference**

Add a short subsection documenting `controlpan_pmap.py`, the `/ctrlplan/load` endpoint, and the device-map "Load control plan" overlay button.

**Step 2: CLAUDE.md**

In the control-packet API paragraph (Part 1), add one sentence: the `aie_ctrl*` API can emit a `CONTROLPAN-PMAP` per-port provenance map to the applog when `__Runtime_ctrl_pmap_enable(1)` is set; the aiedebug device-map "Load control plan" button parses it (`controlpan_pmap.py` + `/ctrlplan/load`) and overlays the routing.

**Step 3: Design doc closeout**

Check off the design doc's non-goals/scope and add a "Files changed" list.

**Step 4: Commit**

```bash
git add .cursor/skills/debug-ui-framework/reference.md CLAUDE.md docs/plans/2026-09-09-controlpan-pmap-design.md
git commit -m "docs: document CONTROLPAN-PMAP runtime flag + device-map overlay"
```

---

## Final verification

Run the full debug parser suite plus the existing provenance suite to confirm nothing regressed:

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py src/tool/debug/tests/test_xaiehost2provenance.py -v > /tmp/claude/all.txt 2>&1; tail -30 /tmp/claude/all.txt`
Expected: all controlpan_pmap tests + all existing provenance tests pass.
