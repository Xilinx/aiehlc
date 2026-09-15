# Control-plan tile stream-switch detail Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After **Load control plan**, clicking a tile opens an enlarged modal drawing that tile's stream-switch internals (slave inputs → slot match → master output fan-out) from CONTROLPAN-PMAP provenance.

**Architecture:** A tested pure Python grouping function `tile_switch_view()` in `controlpan_pmap.py` groups a tile's ports by `(dir, id)` into slave-inputs / slot-node / master-outputs (masters annotated with their neighbor destination from `parse_edges`). A new POST `/ctrlplan/tile` endpoint returns it. The frontend adds a modal + `showTileSwitchDetail()` that draws the SVG, hooked into the existing tile click handler only when a control plan is loaded and the tile has control-plan ports.

**Tech Stack:** Python 3 (stdlib `http.server`), pytest, vanilla JS + inline SVG (`svgN` builder), mcp-browser for UI verification.

---

## Task 1: `tile_switch_view()` grouping function

**Files:**
- Modify: `src/tool/debug/controlpan_pmap.py` (append new function after `parse`)
- Test: `src/tool/debug/tests/test_controlpan_pmap.py` (append)

**Step 1: Write the failing tests**

Append to `src/tool/debug/tests/test_controlpan_pmap.py`:

```python
# tile_switch_view: group one tile's ports by (dir,id) into slave inputs,
# slot node, and master outputs (masters annotated with neighbor dest).
SWITCH = """
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=4 dir=fwd ms=slave  id=2 sw=pkt slot=1
CONTROLPAN-PMAP col=1 row=3 port=CTRL idx=0 dir=fwd ms=master id=2 sw=pkt slot=1
CONTROLPAN-PMAP col=1 row=3 port=EAST idx=0 dir=fwd ms=master id=2 sw=pkt slot=1
CONTROLPAN-PMAP col=2 row=3 port=WEST idx=4 dir=fwd ms=slave  id=2 sw=pkt slot=1
"""

def test_tile_switch_view_fanout():
    v = c.tile_switch_view(SWITCH, 1, 3)
    assert v["col"] == 1 and v["row"] == 3
    assert len(v["groups"]) == 1
    g = v["groups"][0]
    assert g["dir"] == "fwd" and g["id"] == 2 and g["slot"] == 1 and g["sw"] == "pkt"
    assert g["slaves"] == [{"port": "WEST", "idx": 4}]
    ports = {m["port"]: m for m in g["masters"]}
    assert ports["CTRL"]["dest"] == "CTRL (local endpoint)"
    assert ports["EAST"]["dest"] == "(2,3) WEST"

def test_tile_switch_view_empty_tile():
    assert c.tile_switch_view(SWITCH, 9, 9) == {"col": 9, "row": 9, "groups": []}

def test_tile_switch_view_unpaired_master_dest_dash():
    # EAST master with no neighbor slave -> dest "—"
    txt = "CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=fwd ms=master id=5 sw=pkt slot=1\n"
    v = c.tile_switch_view(txt, 0, 3)
    assert v["groups"][0]["masters"][0]["dest"] == "\u2014"

def test_tile_switch_view_splits_fwd_ret():
    # Same tile, same id, different dir -> two groups.
    txt = (
        "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1 sw=pkt slot=0\n"
        "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=ret ms=slave  id=1 sw=pkt slot=0\n"
    )
    v = c.tile_switch_view(txt, 0, 3)
    assert {g["dir"] for g in v["groups"]} == {"fwd", "ret"}
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py -q`
Expected: FAIL with `AttributeError: module 'controlpan_pmap' has no attribute 'tile_switch_view'`.

**Step 3: Write minimal implementation**

Append to `src/tool/debug/controlpan_pmap.py`:

```python
def tile_switch_view(text, col, row):
    """Group one tile's CONTROLPAN-PMAP ports into stream-switch routes.

    Groups the (col,row) ports by (dir, id): the slave ports are the switch
    inputs, the master ports the fan-out outputs. Each master is annotated with
    its destination — the neighbor tile "(c,r) PORT" from parse_edges, or
    "CTRL (local endpoint)" for a CTRL master, or "\u2014" if unpaired.

    Returns {col, row, groups:[{dir,id,slot,sw,slaves:[{port,idx}],
    masters:[{port,idx,dest}]}]}.
    """
    ports = [p for p in parse_ports(text)
             if p["col"] == col and p["row"] == row]
    # master (col,row,port) -> neighbor "(c,r) PORT" from synthesized edges.
    dest_of = {}
    for e in parse_edges(text):
        if e["from"] == [col, row]:
            dest_of[(e["port"], e["from_idx"], e["dir"], e["id"])] = \
                "({},{}) {}".format(e["to"][0], e["to"][1], _OPP[e["port"]])
    groups = {}
    for p in ports:
        key = (p["dir"], p["id"])
        g = groups.setdefault(key, {"dir": p["dir"], "id": p["id"],
                                    "slot": p["slot"], "sw": p["sw"],
                                    "slaves": [], "masters": []})
        if p["ms"] == "slave":
            g["slaves"].append({"port": p["port"], "idx": p["idx"]})
        else:
            if p["port"] == "CTRL":
                dest = "CTRL (local endpoint)"
            else:
                dest = dest_of.get((p["port"], p["idx"], p["dir"], p["id"]), "\u2014")
            g["masters"].append({"port": p["port"], "idx": p["idx"], "dest": dest})
    ordered = sorted(groups.values(), key=lambda g: (g["dir"], g["id"]))
    return {"col": col, "row": row, "groups": ordered}
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m pytest src/tool/debug/tests/test_controlpan_pmap.py -q`
Expected: PASS (10 tests: 6 existing + 4 new).

**Step 5: Commit**

```bash
git add src/tool/debug/controlpan_pmap.py src/tool/debug/tests/test_controlpan_pmap.py
git commit -m "feat(debug): controlpan_pmap tile_switch_view grouping"
```

---

## Task 2: `/ctrlplan/tile` server endpoint

**Files:**
- Modify: `src/tool/debug/schedule_debug_server.py` (add branch in `do_POST` dispatch after the `/ctrlplan/load` block at ~5345)

**Step 1: Add the endpoint**

Insert immediately after the `/ctrlplan/load` `self._send_json(...)` line (~5345):

```python
        elif u.path == "/ctrlplan/tile":
            try:
                with open(st.applog, "r", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                self._send_json({"error": f"cannot read applog: {e}",
                                 "groups": []})
                return
            try:
                col = int(body.get("col")); row = int(body.get("row"))
            except (TypeError, ValueError):
                self._send_json({"error": "bad col/row", "groups": []})
                return
            self._send_json(controlpan_pmap.tile_switch_view(text, col, row))
```

**Step 2: Byte-compile check**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m py_compile src/tool/debug/schedule_debug_server.py && echo OK`
Expected: `OK`.

**Step 3: Commit**

```bash
git add src/tool/debug/schedule_debug_server.py
git commit -m "feat(debug-server): /ctrlplan/tile returns tile_switch_view"
```

---

## Task 3: Frontend modal + CSS

**Files:**
- Modify: `src/tool/debug/schedule_view.py` (CSS near the `.ctrlplan-*` block ~2925; modal HTML; JS)

**Step 1: Add CSS**

After the `.ctrlplan-ctrl-lbl-emit` rule (~2925) add:

```css
  #swDetailModal { position:fixed; inset:0; z-index:200; display:none;
    align-items:center; justify-content:center; background:rgba(0,0,0,.55); }
  #swDetailModal.show { display:flex; }
  #swDetailCard { background:#1c1f26; border:1px solid #3a3f4b; border-radius:8px;
    padding:14px 16px; max-width:92vw; max-height:88vh; overflow:auto;
    box-shadow:0 8px 30px rgba(0,0,0,.5); }
  #swDetailCard .swd-hdr { display:flex; align-items:center; gap:10px;
    font-size:13px; font-weight:700; color:#e4e4e4; margin-bottom:8px; }
  #swDetailCard .swd-close { margin-left:auto; cursor:pointer; border:none;
    background:#2a2f3a; color:#e4e4e4; border-radius:4px; padding:2px 9px; font-size:13px; }
  #swDetailCard .swd-empty { color:#b0bec5; font-size:12px; padding:12px; }
  .swd-slave { fill:#4a7fd4; }
  .swd-slot  { fill:#ffb300; }
  .swd-master{ fill:#e91e63; }
  .swd-lbl   { font-size:10px; fill:#e4e4e4; font-family:monospace; }
  .swd-dest  { font-size:9px; fill:#b0bec5; font-family:monospace; }
  .swd-link  { stroke:#8a90a0; stroke-width:1.5; fill:none; }
```

**Step 2: Add modal HTML**

Find the device-map container markup (search `id="devmap"` around the `dmLoadCtrlPlan` button ~3352) and add this modal near the end of the `<body>` region (anywhere top-level in the returned HTML; put it just before the closing template of the device-map panel or alongside other overlays). Minimal safe location: immediately after the `dmLoadCtrlPlan` button's containing toolbar `</div>`:

```html
<div id="swDetailModal"><div id="swDetailCard">
  <div class="swd-hdr"><span id="swDetailTitle">Stream switch</span>
    <button class="swd-close" id="swDetailClose">✕</button></div>
  <div id="swDetailBody"></div>
</div></div>
```

**Step 3: Add JS render + wiring**

After `loadCtrlPlan()` (~6086) add:

```javascript
// Enlarged stream-switch detail modal. Uses the loaded control-plan ports
// (via /ctrlplan/tile) to draw one tile's routes: slave inputs -> slot node ->
// master output fan-out. Only opened from a tile click when a plan is loaded
// and the tile has control-plan ports (see the tile click handler).
function swDetailClose(){ document.getElementById('swDetailModal')?.classList.remove('show'); }

async function showTileSwitchDetail(tc, tr){
  const modal = document.getElementById('swDetailModal');
  const body = document.getElementById('swDetailBody');
  const title = document.getElementById('swDetailTitle');
  if(!modal||!body) return;
  title.textContent = 'Stream switch ('+tc+','+tr+')';
  body.innerHTML = 'loading…';
  modal.classList.add('show');
  let j;
  try { j = await api('/ctrlplan/tile', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({col:tc,row:tr})}); }
  catch(e){ body.textContent = 'error: '+e; return; }
  if(j.error){ body.textContent = 'error: '+j.error; return; }
  const groups = j.groups||[];
  if(!groups.length){ body.innerHTML =
    '<div class="swd-empty">no control-plan ports on this tile</div>'; return; }
  // Layout: each group is a horizontal band; slaves left, slot node middle,
  // masters right, connected slave -> slot -> master.
  const COLX={slave:30, slot:230, master:430}, W=760, ROWH=26, BANDPAD=18;
  let y=20, svgParts=[];
  const box=(x,yy,cls,txt)=>{
    svgParts.push('<rect x="'+x+'" y="'+(yy-11)+'" width="150" height="20" rx="4" class="'+cls+'" opacity="0.9"/>');
    svgParts.push('<text x="'+(x+6)+'" y="'+(yy+3)+'" class="swd-lbl">'+txt+'</text>');
  };
  const link=(x1,y1,x2,y2)=>svgParts.push('<path class="swd-link" d="M'+x1+','+y1+' C'+((x1+x2)/2)+','+y1+' '+((x1+x2)/2)+','+y2+' '+x2+','+y2+'"/>');
  const esc=s=>String(s).replace(/[&<>]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[ch]));
  groups.forEach(g=>{
    const rows=Math.max(g.slaves.length, g.masters.length, 1);
    const slotY=y+(rows-1)*ROWH/2;
    box(COLX.slot, slotY, 'swd-slot', 'slot '+g.slot+' ('+esc(g.sw)+')');
    svgParts.push('<text x="'+COLX.slot+'" y="'+(slotY-15)+'" class="swd-dest">'+esc(g.dir)+' id'+g.id+'</text>');
    g.slaves.forEach((s,i)=>{ const sy=y+i*ROWH;
      box(COLX.slave, sy, 'swd-slave', esc(s.port)+' '+s.idx+' (slave)');
      link(COLX.slave+150, sy, COLX.slot, slotY); });
    if(!g.slaves.length) box(COLX.slave, slotY, 'swd-slave', '(no slave)');
    g.masters.forEach((m,i)=>{ const my=y+i*ROWH;
      box(COLX.master, my, 'swd-master', esc(m.port)+' '+m.idx+' (master)');
      svgParts.push('<text x="'+COLX.master+'" y="'+(my+18)+'" class="swd-dest">→ '+esc(m.dest)+'</text>');
      link(COLX.slot+150, slotY, COLX.master, my); });
    y += rows*ROWH + BANDPAD;
  });
  body.innerHTML = '<svg width="'+W+'" height="'+(y+10)+'" '+
    'xmlns="http://www.w3.org/2000/svg">'+svgParts.join('')+'</svg>';
}
```

**Step 4: Wire close handlers**

In the DOM-ready init area (search for where `dmLoadCtrlPlan` button is wired, or the main init block that binds buttons — near `document.getElementById('dmLoadCtrlPlan')`), add:

```javascript
document.getElementById('swDetailClose')?.addEventListener('click', swDetailClose);
document.getElementById('swDetailModal')?.addEventListener('click', e=>{
  if(e.target && e.target.id==='swDetailModal') swDetailClose(); });
document.addEventListener('keydown', e=>{ if(e.key==='Escape') swDetailClose(); });
```

If the `dmLoadCtrlPlan` button is not yet click-bound nearby, bind it too:
`document.getElementById('dmLoadCtrlPlan')?.addEventListener('click', loadCtrlPlan);`
(only add if a binding does not already exist — search first).

**Step 5: Hook the tile click**

In the tile click handler, at the end of the `g.addEventListener('click', ...)` body (after the `else` selection branch closes, before the handler's closing `});` at ~6552), add:

```javascript
      // Control-plan mode: if a plan is loaded and this tile has control-plan
      // ports, open the enlarged stream-switch detail modal.
      if((ctrlPlanPorts||[]).some(p=>p.col===tc && p.row===tr)){
        showTileSwitchDetail(tc, tr);
      }
```

**Step 6: Byte-compile check**

Run: `cd /scratch/staff/huaj/aiehlc/aiehlcopensource/aiehlchj && python3 -m py_compile src/tool/debug/schedule_view.py && echo OK`
Expected: `OK`.

**Step 7: Commit**

```bash
git add src/tool/debug/schedule_view.py
git commit -m "feat(debug-ui): stream-switch detail modal on tile click after Load control plan"
```

---

## Task 4: Browser verification

**Step 1: Start the debug server against an applog with CONTROLPAN-PMAP lines**

Run (background): `python3 src/tool/debug/schedule_debug_server.py <args matching an existing applog that contains CONTROLPAN-PMAP lines>`.
If no live applog is handy, reuse the sample from the tests to confirm the parser path, but full UI verification needs a real applog.

**Step 2: Drive the UI with mcp-browser**

- `browser_navigate` to `http://localhost:<port>`
- open the Device Map, `browser_click` **Load control plan**
- `browser_screenshot` — confirm the control-plan overlay appears
- `browser_click` a tile that has control-plan ports
- `browser_screenshot` — confirm the modal shows slave → slot → master fan-out with destinations
- `browser_click` the ✕ (or press Escape) — confirm the modal closes

Expected: modal renders the stream-switch bands; closes cleanly. Capture a screenshot for the record.

**Step 3: No commit** (verification only). Fix any defects found and re-commit under Task 3 if needed.

---

## Task 5: Docs

**Files:**
- Modify: `CLAUDE.md` (extend the CONTROLPAN-PMAP paragraph: mention click-tile stream-switch detail modal + `/ctrlplan/tile` + `tile_switch_view`)
- Modify: `.cursor/skills/debug-ui-framework/reference.md` (add the modal + endpoint to the control-plan feature map)

**Step 1: Update docs** — one or two sentences each, matching the existing style.

**Step 2: Commit**

```bash
git add CLAUDE.md .cursor/skills/debug-ui-framework/reference.md
git commit -m "docs: control-plan tile stream-switch detail modal"
```

---

## Notes / conventions

- Run pytest from the repo root; the test file inserts `..` on `sys.path`.
- Every Bash call prints the shell-profile banner to stdout — read tool output past it.
- Keep any new Python function under 200 lines (project rule); `tile_switch_view` is ~30.
- No changes to `aie_runtime.c` emission — this is a pure consumer of existing CONTROLPAN-PMAP lines.
