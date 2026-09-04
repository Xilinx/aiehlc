# Control-Packet Provenance Map Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach the static provenance generator `xaiehost2provenance.py` to recognize control-packet routing (`__Runtime_CtrlInstance` / `__Runtime_ctrl_setup_routing` / `__Runtime_ctrl_read_target` / `__Runtime_ctrl_push_target`) so `schedule_debug_server.py` can start a debug web GUI for control-packet apps.

**Architecture:** Add a control-packet extractor to `xaiehost2provenance.py` that reduces each send to `(shim_col, dest_row, resp_words)` (same-column vertical climb), then emits shim + pass-through + dest tiles (geometry-aware typing) and two flows per send (forward S2MM up, return MM2S down). These merge into the existing `tiles`/`flows` model so the existing `build_dfschedule` / `build_dmaphop` / `schedule_view.py` pipeline is reused unchanged.

**Tech Stack:** Python 3, `re`, pytest. No C changes.

---

## Background (read before starting)

- Design doc: `docs/plans/2026-09-02-ctrl-packet-provenance-design.md`
- File to modify: `src/tool/debug/xaiehost2provenance.py`
- Tests: `src/tool/debug/tests/test_xaiehost2provenance.py`
- Run tests from repo root: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -v`

Key existing pieces in `xaiehost2provenance.py`:
- `eval_int(expr, defs)` — folds `#define`s / locals to an int, returns `None` if unresolvable.
- `collect_defines(src)` — object-like `#define NAME body` → dict.
- `extract_model(raw_src, aie_gen, aiesim)` (lines ~155-218) — builds
  `{"tiles", "kernel_placements", "flows", "entry_fn"}`. Has a local `add_tile(loc)`
  closure that types tiles as `"shim" if row==0 else "core"`. It already collects
  `defs` (from `collect_defines`, `shimcol`, and `u32 name = ...;` locals).
- `find_entry_fn(active)` (lines ~132-152) — names the enclosing function; keys off
  `RE_XAIE_CALL = re.compile(r"XAie_(LoadElfMem|MoveData\w+|Route)\b")`.

The real control-packet initializer in `aout/worklocal/host.cc` looks like:

```c
__Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u, .stream_id = 0u,
                              .bd_id = RAW_BD_SLOT, .mm2s_ch = 0, .s2mm_ch = 1, .token = NULL,
                              .resp_words = _rspcap};
```

(Multi-line, one or more fields per line, `u`-suffixed literals, `resp_words` may be a
local that does not fold — default to 1 in that case.)

---

## Task 1: Geometry-aware tile-type helper

**Files:**
- Modify: `src/tool/debug/xaiehost2provenance.py` (add helper near `hw_gen_str`, ~line 221)
- Test: `src/tool/debug/tests/test_xaiehost2provenance.py`

**Step 1: Write the failing test**

Append to the test file:

```python
def test_ctrl_tile_type_geometry():
    # gen5/AIE2PS: rows 1,2 memtile, cores from row 3
    assert x.ctrl_tile_type(0, 5) == "shim"
    assert x.ctrl_tile_type(2, 5) == "memtile"
    assert x.ctrl_tile_type(3, 5) == "core"
    # gen2: cores from row 2
    assert x.ctrl_tile_type(1, 2) == "memtile"
    assert x.ctrl_tile_type(2, 2) == "core"
    # gen1: cores from row 1 (no memtile rows)
    assert x.ctrl_tile_type(1, 1) == "core"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py::test_ctrl_tile_type_geometry -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'ctrl_tile_type'`

**Step 3: Write minimal implementation**

Add above `hw_gen_str` (~line 221):

```python
# AIE-core row start per generation (row 0 shim; 0<row<start memtile; row>=start
# core). Mirrors the C rt_port_evt_base geometry and aiediag.AIE_TILE_ROW_START:
# gen5/AIE2PS have 2 memtile rows (cores from 3), gen2 one (from 2), gen1 none.
_CORE_ROW_START = {1: 1, 2: 2, 5: 3}


def ctrl_tile_type(row, aie_gen):
    if row == 0:
        return "shim"
    start = _CORE_ROW_START.get(int(aie_gen), 3)
    return "memtile" if row < start else "core"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py::test_ctrl_tile_type_geometry -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tool/debug/xaiehost2provenance.py src/tool/debug/tests/test_xaiehost2provenance.py
git commit -m "feat(xaiehost2provenance): add geometry-aware ctrl_tile_type helper"
```

---

## Task 2: Parse control-packet sends into send tuples

**Files:**
- Modify: `src/tool/debug/xaiehost2provenance.py` (add `extract_ctrl_sends`, near the
  other RE_* patterns ~line 89-100)
- Test: `src/tool/debug/tests/test_xaiehost2provenance.py`

**Step 1: Write the failing test**

Append:

```python
CTRL_STRUCT_SRC = """
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u, .stream_id = 0u,
                                  .bd_id = RAW_BD_SLOT, .mm2s_ch = 0, .s2mm_ch = 1, .token = NULL,
                                  .resp_words = 2u};
    AieRC _rrc = __Runtime_ctrl_setup_routing(&_ri, 1);
"""

CTRL_CALL_SRC = """
    __Runtime_ctrl_read_target(dev, 0u, 0u, 2u, 5u, 6u, 0x1000u, 1u, out, RAW_BD_SLOT, 0, 1);
"""


def test_extract_ctrl_sends_struct():
    active = x.strip_comments(x.MacroResolver(5, False).active_source(CTRL_STRUCT_SRC))
    defs = x.collect_defines(active)
    sends = x.extract_ctrl_sends(active, defs)
    assert sends == [{"shim_col": 0, "dest_row": 3, "resp_words": 2}]


def test_extract_ctrl_sends_call():
    active = x.strip_comments(x.MacroResolver(5, False).active_source(CTRL_CALL_SRC))
    defs = x.collect_defines(active)
    sends = x.extract_ctrl_sends(active, defs)
    # read_target positional: (dev, shim_col, dest_col, dest_row, ...)
    assert sends == [{"shim_col": 0, "dest_row": 2, "resp_words": 1}]


def test_extract_ctrl_sends_dedup_and_unresolved_respwords():
    # resp_words references an unfoldable local -> defaults to 1; duplicate
    # (shim_col,dest_row) collapses to one send.
    src = CTRL_STRUCT_SRC.replace(".resp_words = 2u", ".resp_words = _rspcap") + CTRL_STRUCT_SRC
    active = x.strip_comments(x.MacroResolver(5, False).active_source(src))
    sends = x.extract_ctrl_sends(active, x.collect_defines(active))
    assert sends == [{"shim_col": 0, "dest_row": 3, "resp_words": 1}]
```

**Step 2: Run to verify it fails**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -k extract_ctrl_sends -v`
Expected: FAIL with `AttributeError: ... has no attribute 'extract_ctrl_sends'`

**Step 3: Write minimal implementation**

Add these module-level regexes next to the other `RE_*` (after `RE_ROUTE`, ~line 100):

```python
# Control-packet send: a __Runtime_CtrlInstance designated-initializer block, or a
# composite __Runtime_ctrl_read_target / __Runtime_ctrl_push_target call. Both are
# same-column (dest_col == shim_col); a send reduces to (shim_col, dest_row,
# resp_words). The struct body may span many lines (DOTALL); the call form gives
# shim_col/dest_col/dest_row as positional args 2/3/4.
RE_CTRL_STRUCT = re.compile(
    r"__Runtime_CtrlInstance\s+\w+\s*=\s*\{(.*?)\}", re.DOTALL)
RE_CTRL_FIELD = re.compile(r"\.(\w+)\s*=\s*([^,}]+)")
RE_CTRL_CALL = re.compile(
    r"__Runtime_ctrl_(?:read|push)_target\s*\(\s*[^,]+,\s*"
    r"([^,]+),\s*([^,]+),\s*([^,]+),")
```

Add the extractor near the other extract helpers (e.g. after `resolve_tileloc`, ~line 117):

```python
def _ctrl_int(expr, defs, default=None):
    """eval_int with a fallback for unfoldable control-packet field exprs."""
    v = eval_int(expr.strip(), defs)
    return default if v is None else v


def extract_ctrl_sends(active, defs):
    """Reduce every control-packet send in @active to a same-column send dict
    {shim_col, dest_row, resp_words}, deduped by (shim_col, dest_row). Values fold
    through @defs; resp_words defaults to 1 when unresolvable. Sends whose shim_col
    or dest_row cannot be resolved are skipped (mirrors tile-loc resolution)."""
    sends, seen = [], set()

    def add(shim_col, dest_row, resp_words):
        if shim_col is None or dest_row is None:
            return
        key = (shim_col, dest_row)
        if key in seen:
            return
        seen.add(key)
        sends.append({"shim_col": shim_col, "dest_row": dest_row,
                      "resp_words": resp_words if resp_words else 1})

    for m in RE_CTRL_STRUCT.finditer(active):
        fields = {k: v for k, v in RE_CTRL_FIELD.findall(m.group(1))}
        add(_ctrl_int(fields.get("shim_col", ""), defs),
            _ctrl_int(fields.get("dest_row", ""), defs),
            _ctrl_int(fields.get("resp_words", "1"), defs, default=1))
    for m in RE_CTRL_CALL.finditer(active):
        add(_ctrl_int(m.group(1), defs), _ctrl_int(m.group(3), defs), 1)
    return sends
```

**Step 4: Run to verify it passes**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -k extract_ctrl_sends -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/tool/debug/xaiehost2provenance.py src/tool/debug/tests/test_xaiehost2provenance.py
git commit -m "feat(xaiehost2provenance): parse control-packet sends"
```

---

## Task 3: Merge control-packet sends into the model (tiles + two flows)

**Files:**
- Modify: `src/tool/debug/xaiehost2provenance.py` (`extract_model`, ~lines 155-218)
- Test: `src/tool/debug/tests/test_xaiehost2provenance.py`

**Step 1: Write the failing test**

Append:

```python
CTRL_MODEL_SRC = """
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u, .stream_id = 0u,
                                  .bd_id = RAW_BD_SLOT, .mm2s_ch = 0, .s2mm_ch = 1, .token = NULL,
                                  .resp_words = 2u};
    AieRC _rrc = __Runtime_ctrl_setup_routing(&_ri, 1);
"""


def test_extract_model_ctrl_packet():
    model = x.extract_model(CTRL_MODEL_SRC, aie_gen=5, aiesim=False)
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    # shim, pass-through memtile rows 1,2, and core dest row 3
    assert tiles[(0, 0)] == "shim"
    assert tiles[(0, 1)] == "memtile"
    assert tiles[(0, 2)] == "memtile"
    assert tiles[(0, 3)] == "core"
    # two flows: forward shim->dest S2MM (up), return dest->shim MM2S (down)
    dirs = {(f["src"], f["dst"], f["direction"]) for f in model["flows"]}
    assert ((0, 0), (0, 3), "S2MM") in dirs
    assert ((0, 3), (0, 0), "MM2S") in dirs
    # len tracks resp_words*4
    fwd = next(f for f in model["flows"] if f["direction"] == "S2MM")
    assert fwd["len"] == 8
    # no kernel loaded on the control path
    assert model["kernel_placements"] == {}


def test_extract_model_ctrl_memtile_dest():
    src = CTRL_MODEL_SRC.replace(".dest_row = 3u", ".dest_row = 2u")
    model = x.extract_model(src, aie_gen=5, aiesim=False)
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    assert tiles[(0, 2)] == "memtile"
```

**Step 2: Run to verify it fails**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -k ctrl_packet -v`
Expected: FAIL — model has no ctrl tiles/flows yet.

**Step 3: Write minimal implementation**

In `extract_model`, the `add_tile` closure currently hardcodes the type. Change it to
accept an explicit type so ctrl tiles can be memtile/core geometry-aware, while XAie
callers keep the old row-based default. Replace the existing `add_tile` definition
(~lines 168-173):

```python
    def add_tile(loc, ttype=None):
        if loc is None or loc in seen:
            return
        seen.add(loc)
        if ttype is None:
            ttype = "shim" if loc[1] == 0 else "core"
        tiles.append({"col": loc[0], "row": loc[1], "type": ttype})
```

Then, immediately before the final `return {...}` of `extract_model` (~line 217),
insert the control-packet merge:

```python
    # Control-packet sends: same-column vertical climb shim(col,0) -> dest(col,row).
    # Add shim, the vertical pass-through tiles, and the dest (geometry-aware type),
    # then two flows per send: forward S2MM up (request) + return MM2S down (response).
    for s in extract_ctrl_sends(active, defs):
        col, drow = s["shim_col"], s["dest_row"]
        length = s["resp_words"] * 4
        for r in range(0, drow + 1):
            add_tile((col, r), ctrl_tile_type(r, aie_gen))
        shim, dst = (col, 0), (col, drow)
        flows.append({"src": shim, "dst": dst, "direction": "S2MM", "len": length})
        flows.append({"src": dst, "dst": shim, "direction": "MM2S", "len": length})
```

**Step 4: Run to verify it passes**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -k ctrl -v`
Expected: PASS. Also run the full file to confirm no regressions:
`python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -v`
Expected: all PASS.

**Step 5: Commit**

```bash
git add src/tool/debug/xaiehost2provenance.py src/tool/debug/tests/test_xaiehost2provenance.py
git commit -m "feat(xaiehost2provenance): merge control-packet sends into model"
```

---

## Task 4: Surface entry function for control-packet host.cc

**Files:**
- Modify: `src/tool/debug/xaiehost2provenance.py` (`RE_XAIE_CALL`, ~line 128)
- Test: `src/tool/debug/tests/test_xaiehost2provenance.py`

**Step 1: Write the failing test**

Append:

```python
CTRL_ENTRY_SRC = """
int controlperf_main(XAie_DevInst *dev)
{
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u,
                                  .resp_words = 1u};
    __Runtime_ctrl_setup_routing(&_ri, 1);
}
"""


def test_ctrl_entry_fn():
    model = x.extract_model(CTRL_ENTRY_SRC, aie_gen=5, aiesim=False)
    assert model["entry_fn"] == "controlperf_main"
    doc = x.build_dfschedule(model, aie_gen=5)
    assert doc["host_entry_fn"] == "controlperf_main"
```

**Step 2: Run to verify it fails**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py::test_ctrl_entry_fn -v`
Expected: FAIL — `entry_fn` is `None` (RE_XAIE_CALL doesn't match `__Runtime_ctrl_`).

**Step 3: Write minimal implementation**

Extend `RE_XAIE_CALL` (~line 128) to also match control-packet calls:

```python
RE_XAIE_CALL = re.compile(
    r"(XAie_(LoadElfMem|MoveData\w+|Route)|__Runtime_ctrl_setup_routing"
    r"|__Runtime_ctrl_(read|push)_target)\b")
```

**Step 4: Run to verify it passes**

Run: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py::test_ctrl_entry_fn -v`
Expected: PASS. Then full file: `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -v` — all PASS.

**Step 5: Commit**

```bash
git add src/tool/debug/xaiehost2provenance.py src/tool/debug/tests/test_xaiehost2provenance.py
git commit -m "feat(xaiehost2provenance): detect control-packet host entry fn"
```

---

## Task 5: Regenerate provenance for aout/worklocal and verify the server loads

**Files:**
- Regenerate (outputs): `aout/worklocal/dfscheduleprovenancemap.json`,
  `aout/worklocal/dmaphopprovenacemap.json`, `aout/worklocal/schedule_view.json`,
  `aout/worklocal/host_schedule.html`

**Step 1: Generate the provenance JSONs from the existing host.cc**

Run:
```bash
python3 src/tool/debug/xaiehost2provenance.py aout/worklocal/host.cc \
    --out-dir aout/worklocal --aie-version 5 --platform baremetal \
    --artifacts-dir aout/worklocal
```
Expected stdout: `[xaiehost2provenance] wrote provenance to aout/worklocal`
(NOT "no XAie routing found - skipping").

**Step 2: Verify the JSON content**

Run:
```bash
python3 -c "import json; d=json.load(open('aout/worklocal/dfscheduleprovenancemap.json')); print(sorted((t['col'],t['row'],t['type']) for t in d['tiles']))"
```
Expected: includes `(0, 0, 'shim')`, `(0, 1, 'memtile')`, `(0, 2, 'memtile')`, `(0, 3, 'core')`.

**Step 3: Build the schedule view**

Run: `python3 src/tool/debug/schedule_view.py aout/worklocal`
Expected: writes `aout/worklocal/schedule_view.json` and `aout/worklocal/host_schedule.html` with no error.

**Step 4: Verify the server accepts the bundle (no browser)**

Run:
```bash
timeout 8 python3 src/tool/debug/schedule_debug_server.py aout/worklocal \
    --elf aout/worklocal/build/host --aie-version 5 --no-llm --no-mcp-probe \
    --port 8199 2>&1 | head -20
```
Expected: NO "neither a provenance bundle" warning and NO "no usable Work/ tree" error;
instead a normal startup line (e.g. serving/listening on a port). `timeout` ends it.

If `aout/worklocal/build/host` does not exist, drop `--elf` for this verification step
(the server still loads the bundle from schedule_view.json).

**Step 5: Commit the regenerated bundle**

```bash
git add aout/worklocal/dfscheduleprovenancemap.json aout/worklocal/dmaphopprovenacemap.json \
        aout/worklocal/schedule_view.json aout/worklocal/host_schedule.html
git commit -m "chore(aout): regenerate control-packet provenance bundle for worklocal"
```

Note: confirm `aout/` is not git-ignored first (`git check-ignore aout/worklocal` — if
ignored, skip this commit; the artifacts are still generated on disk for the server).

---

## Task 6: Update docs

**Files:**
- Modify: `.cursor/skills/debug-ui-framework/reference.md` (xaiehost2provenance section, ~line 75-92)
- Modify: `CLAUDE.md` (control-packet API note already documents the C side; add a line
  that provenance for control-packet host.cc is emitted by xaiehost2provenance.py)

**Step 1: Update the debug-ui-framework reference**

In `.cursor/skills/debug-ui-framework/reference.md`, in the `## xaiehost2provenance.py`
section, add a sentence: it now also recognizes control-packet routing
(`__Runtime_CtrlInstance` / `__Runtime_ctrl_setup_routing` / `__Runtime_ctrl_read_target`
/ `__Runtime_ctrl_push_target`), emitting shim + vertical pass-through + dest tiles and
two flows per send (forward S2MM up, return MM2S down).

**Step 2: Update CLAUDE.md**

Under the Part 1 control-packet API paragraph, add one sentence: "The debug provenance
for a control-packet host.cc is produced statically by `xaiehost2provenance.py` (it
recognizes `__Runtime_CtrlInstance` sends), so `schedule_debug_server.py` can open a
debug GUI without a Work/ tree."

**Step 3: Commit**

```bash
git add .cursor/skills/debug-ui-framework/reference.md CLAUDE.md
git commit -m "docs: note control-packet provenance in debug-ui reference and CLAUDE.md"
```

---

## Done criteria

- [x] `python3 -m pytest src/tool/debug/tests/test_xaiehost2provenance.py -v` — all pass (20 passed).
- [x] `python3 src/tool/debug/xaiehost2provenance.py aout/worklocal/host.cc --out-dir aout/worklocal --aie-version 5 --platform baremetal --artifacts-dir aout/worklocal` writes JSONs (not "skipping").
- [x] `aout/worklocal/schedule_view.json` exists after `schedule_view.py` (4 tiles).
- [x] `schedule_debug_server.py aout/worklocal ... ` loads the bundle (`startcol: 0 (from provenance JSON) device=pal aie=2ps`, HTTP 200); no "neither a provenance bundle" warning.
- [x] Docs updated.

## Files changed / created (after execution)

- Modify: `src/tool/debug/xaiehost2provenance.py` — `ctrl_tile_type` (Task 1),
  `extract_ctrl_sends`/`_ctrl_int`/`RE_CTRL_*` (Task 2), control-packet merge in
  `extract_model` + `add_tile(ttype=)` (Task 3), `RE_XAIE_CALL` control-packet
  calls (Task 4), `MacroResolver` inline `#define`/`#undef` + `_eval_cond`
  literal replacement (Task 5 root-cause fix).
- Modify: `src/tool/debug/tests/test_xaiehost2provenance.py` — control-packet and
  MacroResolver-inline-define tests.
- Generate (git-ignored `aout/`): `aout/worklocal/{dfscheduleprovenancemap.json,dmaphopprovenacemap.json,schedule_view.json,host_schedule.html}`
- Modify: `.cursor/skills/debug-ui-framework/reference.md`, `CLAUDE.md`

## Execution note

The real controlperf host.cc gates its `__Runtime_CtrlInstance` sends behind a
bare in-file `#define _CONTROL_WRITE_TEST_` directly followed by `#ifdef`. The
original `MacroResolver` ignored inline `#define`/`#undef`, so it dropped the
guarded block → empty model → "no XAie routing found". Task 5 therefore added a
root-cause fix (not in the original plan): honor inline define/undef in active
regions, plus a `_eval_cond` fix so macro bodies with backslashes (multi-line
`BENCH` defines) substitute literally instead of raising `re.error: bad escape`.
Commits: Task 3 `38c4d3d`, Task 4 `408aaea`, Task 5 fix `b4b6b4b`, Task 6 `89401f8`.
