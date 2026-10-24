<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `src/tool/debug` — AIE Live Debug & Schedule Visualization Tools

This directory centralizes the **HTML AIE-schedule / live-debug** solution: a set
of Python tools that turn the compiler's provenance JSON + generated `host.cc`
into a readable schedule view, and connect a browser (or Claude Code) to live AIE
hardware over JTAG for DMA/core/event inspection.

All tools are pure-Python (stdlib only, except `aiemcp.py` which needs the `mcp`
package). None of them modify `host.cc` or any generated source.

## Files

| File | Role | Depends on |
|------|------|------------|
| `aiediag.py` | DMA diagnostic **library + CLI** — register offsets, `aiedbg` reg reads, status decoders, provenance-map cross-reference | none (leaf) |
| `aiegdb.py` | Scoped **GDB-like CLI** over `aiedbg` (partition → tile → channel) | `aiediag` |
| `aiemcp.py` | **MCP server** exposing `aiegdb` to Claude Code (in-process, persistent scope) | `aiediag`, `aiegdb`, `mcp` |
| `schedule_view.py` | Generates `schedule_view.json` + `host_schedule.html` from provenance JSON + `host.cc` | none (stdlib) |
| `ensure_aiedbg.py` | Clone/install/update `aiedbg` + write `.aiehlc/aiedbg_env.sh` for PATH | none (stdlib) |
| `schedule_debug_server.py` | **Live debug daemon** (`http.server`) behind `host_schedule.html`; deploys ELF, tails `applog`, overlays live DMA/core/event status, drives an LLM tab | `aiediag`, `aiegdb`, `ensure_aiedbg`; spawns `aiemcp.py` |
| `streamswitch_crossref.py` | **Accuracy check + CLI** — cross-references the UI's per-tile Stream switch panel against the `XAie_Strm*` calls in generated source | `schedule_view`, `node` |
| `flow_crossref.py` | **Attribution check + CLI** — checks that every tile and flow is shown the right information, against `routing.cc` groups + `dmaphop` endpoints | `streamswitch_crossref`, `schedule_view`, `node` |
| `switch_scan.py` | **Live switch read + diff** — reads the stream-switch registers off the board/sim and diffs them against the routing map; backs the UI's `Switch` scan | `aiediag`, `streamswitch_crossref` |
| `switch_reconstruct.py` | **Routing rebuilt from the board** — turns those same registers into flows with no provenance map involved, for when the map is absent, stale, or was changed at runtime; backs the UI's `Dynamic` routing source | `aiediag` |

### Dependency graph

```
aiediag  <-- aiegdb  <-- aiemcp
   ^          ^            ^  (spawned as subprocess via --mcp-config)
   |          |            |
   +----------+---- schedule_debug_server  --> host_schedule.html (served)
                    schedule_view --> host_schedule.html + schedule_view.json
```

---

## Prerequisites

- **Provenance JSONs** in a workdir (e.g. `aout/worklocal/`):
  `dfscheduleprovenancemap.json`, `dmaphopprovenacemap.json`, and `host.cc`.
  These are produced by the tilinglinalg pipeline (see the repo `CLAUDE.md`).
- **`aiedbg` CLI** for live hardware reads. The debug server bootstraps it
  automatically on first launch (clone + `pip install --user`); see
  [aiedbg bootstrap](#aiedbg-bootstrap) below.
- **Live HW debug** also requires a JTAG target, set via either:
  - `export AIEDBG_TARGET=xsdb://<host>:3121`, or
  - `~/.aiedbg_env` (written by `aiedbg-setup`), or
  - a `--target xsdb://<host>:3121` flag.
- `aiemcp.py` additionally needs: `pip install "mcp[cli]"`.

All live tools accept `--dry-run` (or `AIEMCP_DRY_RUN=1`) to print the `aiedbg`
commands **without a board** — useful for testing offline.

---

## aiedbg bootstrap

`schedule_debug_server.py` and `aiehlc.sh --prettydebug` call
`ensure_aiedbg.py` on startup when `aiedbg` is not on `PATH`:

1. `git clone` into `thirdparty/aiedbg` (or use `$AIEHLC_AIEDBG_SRC`)
2. `python3 -m pip install --user .`
3. Write `.aiehlc/aiedbg_env.sh` so future shells prepend `~/.local/bin`

Manual install / update:

```bash
script/debug/bootstrap_aiedbg.sh           # install if missing
script/debug/bootstrap_aiedbg.sh --update  # git pull + pip upgrade
```

`aiehlc.sh` sources `.aiehlc/aiedbg_env.sh` automatically when present.

Board JTAG is **not** configured by the bootstrap — set `AIEDBG_TARGET` (see
`script/test/envlocal.sh`) or run `aiedbg-setup --board <name>` separately.

Flags on the debug server: `--skip-aiedbg-bootstrap`, `--update-aiedbg`.

Environment overrides:

| Variable | Meaning |
|----------|---------|
| `AIEHLC_AIEDBG_REPO` | git clone URL (default: `AMD-AECG-SSWSIP/aiedbg` on GitHub) |
| `AIEHLC_AIEDBG_SRC` | use an existing clone instead of `thirdparty/aiedbg` |

---

## 1. `schedule_view.py` — generate the readable schedule view

Consumes `<workdir>/dfscheduleprovenancemap.json` + `<workdir>/host.cc` and writes
`<workdir>/schedule_view.json` and `<workdir>/host_schedule.html` (a self-contained
static page — all HTML/CSS/JS is generated inline).

```bash
python3 src/tool/debug/schedule_view.py [workdir]      # workdir default: aout/worklocal
# → wrote <workdir>/schedule_view.json
# → wrote <workdir>/host_schedule.html
```

Open `host_schedule.html` in a browser for an offline view. This step is run
automatically by `script/aiehlc.sh` after a build.

---

## 2. `aiediag.py` — flow-aware DMA diagnostics

Reads DMA status registers from live hardware, decodes them, and cross-references
provenance JSON to name the connected tiles / routing paths.

```bash
# Dig into a tile's DMA channel (logical col, physical row; startcol optional)
python3 src/tool/debug/aiediag.py dig <col> <row> [startcol N] [--dry-run]

# Read a core's PC and map it to kernel source file:line
python3 src/tool/debug/aiediag.py pc <col> <row> [--dry-run]
```

Common flags: `--json-dir DIR`, `--aie-version {5,2ps}`, `--target xsdb://...`,
`--device pal`, `--dry-run`. It is also imported as a **library** by the other
tools (offsets, `run_aiedbg_reg_read`, decoders, `startcol_from_jsons`).

---

## 3. `aiegdb.py` — scoped GDB-like CLI

A stateful REPL that keeps a *current scope* (partition → tile → channel). Scope
persists across commands; tile scope auto-injects `phys_col = col + startcol`,
channel scope also auto-injects direction/channel.

```bash
# Interactive REPL
python3 src/tool/debug/aiegdb.py --target xsdb://<host>:3121

# Non-interactive: run ';'-separated commands, then exit
python3 src/tool/debug/aiegdb.py -c "target tile 0 3; target channel mm2s0; dma status"

# Run a command script
python3 src/tool/debug/aiegdb.py --script cmds.txt

# No board — print aiedbg commands only
python3 src/tool/debug/aiegdb.py --dry-run -c "target tile 0 3; dma status"
```

Inside the REPL: `target tile <col> <row>`, `target channel <dir_ch>` (e.g.
`mm2s0`), `up`, `top`, `where`, `?` (per-scope command list), plus channel
helpers `dma status`, `bd`, `event`, `pc`. Flags mirror `aiediag`
(`--startcol`, `--aie-version`, `--json-dir`, `--device`). `--server` runs a
framed stdin/stdout REPL used by the debug daemon.

### Tile scope: `show switch` / `scan switch` (stream-switch read-back)

Two **read-only** tile-scope commands decode the AIE stream switch from its
configuration registers (`aiediag.read_switch` / `format_switch`):

```bash
# Decode THIS tile's stream switch: every enabled master, the slave feeding it,
# circuit vs packet mode, enabled slaves and their packet slots.
python3 src/tool/debug/aiegdb.py -c "tile 0 3; show switch"   # alias: switch

# Flow-trace every configured connection through this tile: BFS the reachable
# set of tiles along enabled directional ports (up + downstream), read each
# switch, then print the assembled end-to-end flows, e.g.
#   DMA0@(0,3) -> NORTH0@(0,3) -> NORTH0@(0,4) -> CORE0@(0,5)
python3 src/tool/debug/aiegdb.py -c "tile 0 3; scan switch"
```

A master-config register's `CONFIGURATION` field is the *physical slave index*
feeding it (driver `_XAie_StreamSwitchConfigureCct`), so the decode maps each
enabled master directly back to its source slave. Register bases are identical
for gen5 (aieml) and gen 2ps: core/shim master `0x3F000` / slave `0x3F100` /
slot `0x3F200`; memtile `0xB0000` / `0xB0100` / `0xB0200`. Port maps and
inter-tile wiring (NORTH↔SOUTH, EAST↔WEST; shim SOUTH = PL/NoC/DDR terminal;
memtiles have no EAST/WEST) are transcribed from `xaie2psgbl_reginit.c`. Both
commands are non-intrusive (no writes) and cycle-safe (BFS `visited` +
per-branch recursion guard).

### `COMMAND_SPEC` — the one grammar definition

`COMMAND_SPEC` (module level, keyed `universal` / `partition` / `tile` /
`channel`) is the machine-readable command grammar. Each entry carries `name`,
`aliases`, `args`, `summary`, `intrusive` (writes hardware) and `passthrough`
(forwarded raw to `aiedbg`).

It exists so no front-end has to hand-copy the grammar and let it drift:

```bash
python3 src/tool/debug/aiegdb.py --dry-run -c "spec"   # prints {scope, spec} as JSON
```

`_commands()` (the `?` listing) is *rendered from* the spec, the daemon serves it
at `GET /aiegdb/spec`, and `schedule_view.py` bakes a copy into the generated
HTML — so adding a command here makes it appear in the CLI listing, the browser
autocomplete and the command palette at once.

Two things to know when editing it: unknown verbs still fall through to `aiedbg`
passthrough, so a command missing from the spec *works* but is never suggested;
and `_commands()` formats as `f"  {sig:<36}  {summary}"` where the two trailing
spaces are deliberate, because the browser splits the listing on that gap.

---

## 4. `aiemcp.py` — MCP server for Claude Code

Exposes `aiegdb` to Claude Code as MCP tools, keeping **one long-lived** `AieGdb`
instance so scope persists across tool calls. Registered as `aiegdb` in the
repo-root `.mcp.json`.

Tools: `aie_exec(cmd)` (primary — e.g. `target tile 0 3`, `dma status`, `bd`,
`pc`), `aie_scope()` (`where`), `aie_commands()` (`?`), `aie_help()` (`help`).

Config via environment (set in `.mcp.json`):

| Env var | Meaning | Default |
|---------|---------|---------|
| `AIEDBG_TARGET` | JTAG target `xsdb://host:port` | `~/.aiedbg_env` |
| `AIEMCP_DEVICE` | `aiedbg -d` device | `pal` |
| `AIEMCP_STARTCOL` | physical column offset | from provenance JSON |
| `AIEMCP_AIE_VERSION` | register offsets `5` \| `2ps` | from provenance JSON |
| `AIEMCP_JSON_DIR` | provenance JSON dir | auto-detect |
| `AIEMCP_DRY_RUN` | print `aiedbg` cmds, no board | off |

```bash
# Run standalone (stdio transport)
python3 src/tool/debug/aiemcp.py

# Protocol smoke test (needs mcp[cli]); no board
AIEMCP_DRY_RUN=1 mcp dev src/tool/debug/aiemcp.py
```

---

## 5. `schedule_debug_server.py` — live debug daemon

An `http.server` daemon that backs the `host_schedule.html` page. It deploys the
ELF via `apppaltest.py -y -nonreboot`, tails the repo-root `applog`, exposes
polling JSON endpoints (`/run`, `/runstate`, `/applog`, `/grid`, `/cmd`) so the browser shows
live DMA/core/event overlays, and drives an in-browser LLM tab wired to the
`aiegdb` MCP server. The daemon issues **read-only** hardware ops only.

```bash
python3 src/tool/debug/schedule_debug_server.py <workdir> \
    --elf <path/to/main.elf> --aie-version 5 --open

# Open one aiecompiler app directly and generate worklocal/ when needed.
python3 src/tool/debug/schedule_debug_server.py \
    --app ../naiebaremetal/example/example_oob_4x4 --open
```

Key flags: `--port 8091`, `--host 0.0.0.0`, `--target xsdb://...`, `--device pal`,
`--startcol N`, `--apppaltest PATH` (default `script/test/apppaltest.py`),
`--applog PATH` (default repo-root `applog.$USER`, or `$SCHEDULE_DEBUG_APPLOG`), `--no-llm`, `--password` /
`--no-password`, `--no-mcp-probe`. `--app PATH` selects exactly one app and
accepts either its root directory or an existing provenance bundle;
`--app-root DIR` remains the multi-app discovery mode. Launched automatically by
`script/aiehlc.sh --prettydebug`.

### Run deploys through netboot; Stop is soft unless [force reset]

For the pal board, `start_run` (the UI's **Run**, `POST /run`) launches
`apppaltest.py -y -nonreboot --netload <elf>`. It first probes the netboot
supervisor (`_netboot_probe`: a STATUS through `netboot_proxy.py` on PALIP, over
the multiplexed ssh `apppaltest.py` also uses; stages the proxy once if missing).
- **Supervisor answers:** fast path. The ELF streams through the proxy and the
  parked app core is released. Measured UI Run to `status: pass`: **1.9-2.0 s**.
- **No answer:** still `--netload`. apppaltest bootstraps the supervisor once
  (power cycle, BOOT.BIN, `dow netboot.elf`, ~40-50 s) and later runs are fast.
  "Use netboot only if one already answers" would never leave the JTAG path,
  because every JTAG `-nonreboot` run power-cycles the board and kills the supervisor.

The applog records the choice (`# deploy: netboot (--netload) -- …`) and
`/runstate` returns it as `deploy`. `AIEHLC_NETLOAD=off` restores the JTAG flow.

**Stop run** is **soft** by default: it kills the local apppaltest process group,
writes `[stop: board left as is (soft stop) …]`, and re-enables Run and the
profile timeline. The board and its netboot supervisor are untouched, so the
next Run is fast (measured 2.0 s right after a mid-run Stop). The **[force reset]**
checkbox next to Stop (`POST /stop {"force_reset": true}`) arms a one-shot reset:
the next Run adds `--netload-bootstrap` and starts from a clean board (measured
41.7 s, then back to 2.0 s). The box clears itself after use. Use it when the
app hangs, because a hung app core cannot be reclaimed any other way. Force reset
also works with no run in progress.

`[Program completed after` counts as a pass marker, so netload runs of apps that
print no teardown line (they end on the supervisor's exit report) show `pass`.

**The daemon must be restarted to pick up these changes.** A daemon started
before them keeps the old JTAG-every-run behaviour.

### Session provenance — a target is not a connection

`AIEDBG_TARGET` is exported by `script/test/envlocal.sh`, so the daemon holds a
target from the instant it starts, with no user action — and a reachable board
still contains registers from whatever ran on it last. Treating "has a target" as
"is connected" is therefore wrong, and it once led the embedded agent to read a
previous run's registers and a leftover `applog` and report a successful run that
had never happened in that session.

The daemon tracks how the session earned its access, via one builder
(`session_state()` / `session_summary()`):

| State | Earned by | Meaning |
|---|---|---|
| `none` | startup default, target notwithstanding | reads refused |
| `connected` | **Connect** → a passing `/ping` | link real; registers are pre-existing state |
| `attached` | **Open Current Session** → `POST /attach` | real run, started outside the UI; earlier history unknown |
| `ran` | **Run** → `start_run` | live state *and* `applog` describe the current run |

`hw_authorized()` gates `/grid` and the `aiegdb` MCP server's device commands
(navigation, `help` and `?` stay open so the console works offline).
`applog_provenance()` classifies the log `current` / `predates_session` /
`foreign` / `absent` by mtime versus session start — necessary because `applog` is
a fixed repo-root path the manual CLI flow also writes, so its mere existence
proves nothing. `get_applog` prefixes a `[STALE: …]` / `[FRESH: …]` banner
accordingly.

The state ships in `backend_status.json` (`session`, `session_summary`) so the
out-of-process MCP servers can gate without a round-trip, is injected into every
LLM message, and is enforced by a precondition block in the agent's system prompt.

### App capabilities are detected, not declared

The startup listing tags each registered app `sim`, `hw`, `sim,hw` or
`view-only`. That tag reports only what run profiles the daemon can find for
that app — it is not a restriction, and a `view-only` app still renders its
schedule view, device map, aiegdb console and LLM tab, and still runs on the
board through the built-in `palmyra` / `vek385` options (which are baked into
the page, not per-app; `select_app()` repoints the ELF via
`_resolve_default_elf`).

Simulators used to be *declared* in a per-app `worklocal/debug_ui_config.json`.
For the naiebaremetal IPC flow that file was boilerplate — `runsim_ipc.sh
<example_dir>` is the entire contract, so every copy held the same script path
and a different example dir — and in practice almost nobody wrote one, so
sim-ready examples showed up as `view-only`. Both simulators are now detected
from the build artifacts they leave behind, through one resolver
(`_detect_sim_device`) shared by `App.caps()` and `_load_app_profile()` so the
list and `/devices` cannot disagree:

| Profile | Detected by | Runner |
|---|---|---|
| sim — aiehlc (`aie2pssimmsm`) | `sim_config.sh` in the bundle or its parent | `<repo>/script/runsim.sh`, `sim_kind='aiesim'` |
| sim — naiebaremetal (IPC) | `<example>/ipc/build_sim.env` **and** `<example>/Work/ps/c_rts/systemC` | `script/runsim_ipc.sh`, found by walking up from the example, `sim_kind='ipc'` |
| hw — VEK385 | `<example>/build/vek385.BIN` (mandatory; the runner exits without it) | `src/tool/test/runhw_vek385.py`, found by walking up |

Each predicate is the runner's own precondition, so detection can never
advertise a profile that dies on launch. Hardware *looked* per-app because its
`hw_env` is a wall of board settings — but `VEK385IP`, `USERNAME`, `VEK385PDI`
and `AIEDBG_TARGET` all come from the repo-wide `hwlocal.sh`, which
`runhw_vek385.py` sources by itself. The only genuinely per-app parts are
`build/vek385.BIN` and the optional `vek385.elf`, both at fixed conventional
paths. So an example becomes hw-capable by being **built** — `./build.sh 5
-bootgen` — not by being registered.

#### The board is per-run, never part of the profile

A detected profile names **no board**. Which board a run targets is a per-run
choice — developers move between boards session to session, and the UI's
hostname box is editable before every run — so freezing a host into the profile
is exactly what makes a written config go stale. Detection pins only the two
artifacts; everything board-shaped is resolved at launch by `_board_run_env()`:

```
host = UI hostname box  →  $VEK385IP  →  hwlocal.sh
```

**Live sources only.** A profile never contributes a board — it contributes the
runner and the app's artifacts. Anything board-shaped in a declared `hw_env`
(`_BOARD_KEYS` = `VEK385IP`, `AIEDBG_TARGET`) is dropped on overlay and
recomputed, because that file is a *snapshot*: `run_debug_ui.sh` copies
`$VEK385IP`/`hwlocal.sh` into it at write time, so those keys are stale by
construction. If nothing names a board the daemon sets neither variable and
leaves it to `runhw_vek385.py`, which sources `hwlocal.sh` itself — the same
answer one level down, rather than a guess.

Everything derived from a host then follows that one value. Previously only
`VEK385IP` followed the box, so a profile carrying its own `AIEDBG_TARGET` kept
it: switching boards programmed one machine and pointed live debug at another —
a half-switch that reads as the new board misbehaving. The session's target
survives only while it still names the resolved host (preserving a non-default
`hw_server` port); the moment the host changes it is recomputed. The chosen
board and target go into the applog header, so "which board was that run on?" is
answerable from the log alone.

The **label** is part of this. `run_debug_ui.sh` writes `"VEK385 (${VEK385IP})"`,
so the dropdown advertised whichever board the generator saw — a config written
against `portobello13` still offered "VEK385 (portobello13)" as the only hardware
option on a machine set to `crimini2`. `_device_label()` strips the profile's own
host (and only that host, so a genuinely custom label like `VEK385 rev B (extra
DDR)` survives) and falls back to the device value, so both declared and detected
profiles render as plain `vek385` — the same string as the baked-in option,
rather than varying with whether a profile happened to be found. The board is
shown, and chosen, in the hostname box beside the dropdown; `/config` prefills it
from the same chain, per selected app, as a **prefill, not a binding**.

An explicit `debug_ui_config.json` still wins over detection for *structure*
(which runner, which artifacts) — just never for the board. Since a board named
in one is silently ignored, `App.caps()` says so rather than leaving someone to
wonder why editing the file changes nothing:

```
  example_oob_4x4        sim,hw       /…/example_oob_4x4/worklocal
                                      ! debug_ui_config.json names VEK385IP=portobello13 —
                                        ignored; the board comes from the UI box / $VEK385IP /
                                        hwlocal.sh (currently crimini2)
```

A remaining `view-only` row prints the artifact it is missing (`no
ipc/build_sim.env …`, `no sim_config.sh …`) via `App._no_sim_reason()`.

Note the page bakes in generic `vek385` / `palmyra` options, so a profile naming
one of those **replaces that option's label** rather than appending a second row
with the same value (which is what produced two indistinguishable `vek385`
entries before).

### One device namespace, not two

`device` is nothing but the `-d` flag forwarded to `aiedbg`, which uses it to pick
array geometry and register tables from its own `DEVICE_CONFIGS`. It reaches the
board through four paths: the `/grid` scan, `aiediag`'s single-tile reads,
the `aiegdb` subprocess, and `AIEMCP_DEVICE` in the LLM tab's MCP config.

It used to be a *separate namespace* from the board dropdown (`palmyra` /
`vek385` / `simulator`), and nothing connected them — the dropdown only fed
`resolve_target()`'s `xsdb://` URL. So picking `vek385` in the UI left aiedbg on
its startup `pal`, which describes a **12-column** array instead of **38**;
full-array scans (`tile list`, `aieshow` grids, `enumerate_tiles`) silently
stopped at column 11. Register decoding was unaffected — `pal` and `vek385`
share `reg_dir: aie2ps` — which is why single-tile reads always looked right.
Three things kept it stuck: the dropdown never touched `st.device`,
`select_app()` re-derived `startcol`/`aie_version` from provenance but not the
device, and `run_debug_ui.sh` passes `--device` only on the fresh-launch path
(registering into an already-running daemon via `/apps/add` passes nothing).

The dropdown now **carries aiedbg's own device names**, derived at render time
from `aiedbg.device_config.DEVICE_CONFIGS` (`_aiedbg_devices()`, with a fixed
fallback for static generation) — so it cannot drift, and a device aiedbg adds
appears on its own. `resolve_device()` maps the selection to `-d`, `/grid`
threads it per request, and `retarget()` (Connect) sets `st.device` so the
aiegdb subprocess, the MCP config and `backend_status.json` all follow.

Two fallthroughs had to close with it, both of which used to treat *anything not
`vek385`* as palmyra: `start_run()` would have deployed the palmyra test to a
vek280 selection (it now refuses with "no run script for …"), and the hw_server
recovery path would have ssh'd to `$PALIP` instead of the chosen board.

`vrk160` is deliberately absent — aiedbg rejects it (`Unknown device type`).
Note also that `vek280` and `vck190` carry `reg_dir: None`, so aiedbg refuses
register reads on them outright; they are selectable for the run/connect path.

### Source viewer — click a reference, read the code

`GET /source?path=&line=` returns a file Pygments-highlighted as JSON; the browser
opens it as a `src:` item in the Info pane, scrolled to the line with that line
highlighted. Clickable from all four places a path appears: the LLM chat, the
aiegdb console's `PC -> file:line`, the tile `code piece:` banner, and the
kernel / `.bcf` / dfschedule headers. **Pygments 2.20 is already installed** — no
JS bundle, no CDN, so the page stays self-contained and `file://` still works
(`SRC.on` is false there and clicking reports that the daemon is needed).

Three Pygments details the implementation depends on, each measured:

- **`get_style_defs()` leaks unscoped rules** — `pre { line-height:125% }`,
  `span.linenos`, `td.linenos …` — which would restyle `pre.code`, `.bdpretty`,
  `.md-block` and the console app-wide. `_source_css()` forces every selector
  under `.srcview`; there is a test asserting nothing escapes.
- **`linespans`, never `hl_lines`.** `hl_lines` is slice-relative while the gutter
  follows `linenostart`, so a windowed file highlights the wrong line
  (`linenostart=400, hl_lines=[3]` → line 402). `linespans='SL'` emits
  `<span id="SL-412">`, so highlighting is a client-side `classList.add` and the
  render is line-independent and cacheable.
- **`linenos='inline'`, not `'table'`** — gutter inside the line span, so the
  highlight covers the whole row.

In the chat only the **basename** is shown — an absolute app path runs to ~60
characters and swamps the sentence around it. A reference may carry a **range** — `graph.cpp:138-145` (hyphen or en dash) —
which highlights every line in it and scrolls to the first. Scrolling uses
`getBoundingClientRect()` deltas, **not `offsetTop`**: `#panel` and every
ancestor up to `<body>` are `position:static`, so `offsetTop` measures from the
top of the document and the computed `scrollTop` overshot to the end of the
file, opening the right file at the wrong place.

The full path rides in `data-p`
for the click and in `title` for hover; the viewer's own banner shows it in
full once open. The console keeps aiegdb's line verbatim.

Path handling: absolute, relative to any root, or a bare basename resolved
through a two-tier index. Tier 1 is the paths the view names (`host.cc`,
`kernel.cc`, the `.bcf`, per-tile `code_file`) so a generated file the tools
actually cite wins its name. Tier 2 is a **bounded walk of the app's own dirs**
— app root, bundle, `Work/` — which is what makes hand-written sources
clickable: the model cites `graph.cpp:138`, and that lives at
`<app>/src/graph.cpp`, so joining a bare name against a root never found it. The
walk skips `.git`/`ipc`/`aiesimulator_output`/etc., caps depth and file count,
and costs ~6 ms for a full naiebaremetal example (0.03 ms cached). It is cached
on a TTL rather than an mtime because a directory tree has no single mtime. The
repo root and the aiedbg clone are deliberately not walked — too large, and
citations into them are repo-relative rather than bare.

**App selection is what scopes it.** The registry registers `<example>/worklocal`,
and `app_paths()` derives `app_dir` by going *up* from a `worklocal` basename —
so selecting an app in the UI points the walk at that app's source tree, and
every example's own `src/graph.cpp` resolves to its own copy. Relatedly,
`AppRegistry.default()` now prefers an explicitly named app (positional workdir,
then `--app`) over `list()`'s newest-by-mtime ordering: a cold
`run_debug_ui.sh` passes the app's worklocal positionally, and it used to lose
to whichever unrelated app was rebuilt most recently — which silently scoped
bare-filename lookups to the wrong app. Going further: when the invocation
*names* an app (positional workdir or `--app`) and passes no `--app-root`, the
`aout/` + `example/` auto-scan is skipped entirely. That scan exists so a bare
`schedule_debug_server.py` finds the newest build on its own; once you have said
what to debug and asked for no tree to browse, its hits are noise from whichever
repo the daemon was started in — launching on a naiebaremetal bundle listed this
repo's `aout` and `example/stream` in the App dropdown. The positional `workdir`
therefore defaults to `None`, not `"aout/worklocal"`: whether the user named a
work dir is the deciding input, and a default string makes every bare invocation
look explicit (`_DEFAULT_WORKDIR` is the fallback when the registry is empty).
An explicit positional path or `--app PATH` may name either a provenance bundle
or an app directory containing `Work/`. The app-directory form regenerates
`worklocal` when `aie_control_config.json` is newer, writes
`schedule_view.json` plus `host_schedule.html`, and registers only that app.
`--app` is singular; `--app-root` remains the multi-app discovery mode.
The browser follows from the app *count*, not a launch flag — `loadApps()` sets
`#ctrlbar[data-apps="one"]` at `apps.length <= 1` and CSS drops the whole App row
(label, select, path), so a later `POST /apps/add` brings it back on the next load
with no extra plumbing. `--sim-only` is the same shape for the Board row:
`/config` reports `sim_only`, the page sets `#ctrlbar[data-simonly="true"]` and CSS
drops `#row-board`, `Connect`, `Attach existing run` and the start-hw_server hint;
when both rows go, `.run-fields` goes with them. The buttons are NOT in that block
— they sit on the pane-title row (`.run-head`), since they are the pane's point and
under sim-only + one app they are all that is left; `#connstatus` is pushed right
with `margin-left:auto` and wraps to its own line on a narrow pane rather than
crushing them. Below the buttons is one flex `.run-row` per selector, Board then
App, deliberately not a grid: a hidden grid row still leaves its `row-gap` behind,
a hidden flex child does not. `#deviceSel`
stays in the DOM pinned to `simulator` because every connect/run path in the page
branches on its value — hiding the row, not deleting the control, is what keeps
those paths untouched. Hiding is not enforcement, so the daemon refuses `/ping`,
`/targets`, `/run`, `/attach`, `/settarget` and `/launch_hwserver` with
`sim_only: true` (an MCP client, a page loaded before the flag, and curl all reach
the same handlers), `session_state()` publishes `sim_only`, and `session_summary()`
gains a sim-only "NO SESSION" wording so the model does not suggest Connect.
`Connect` goes too, leaving `Run` as the only button — but Connect is what
*unlocks* Run, so `applyBoardDefaults()` calls `testConnect()` on load when
`sim_only`. Its simulator branch touches no board (it probes `/sim/status`, reveals
the console and enables the run buttons), so firing it automatically is safe, and
it runs after `updateDeviceUI()` because that resets exactly the state it sets.
`Stop run` stays: a simulator you cannot stop is a trap. The status line drops the
word "activated" under sim-only — it names a step the user no longer takes — for
`simulator ready — press "Run"`. With
the App row gone the identity moves to `#appbadge` beside the `AIE Debug` pane
title, and `document.title` carries it too. Roots are recomputed per
request because `select_app` reassigns `workdir`; nested roots are kept
deliberately — they are also the join bases for a relative citation, and dropping
`aout/worklocal` because it sits under the repo root made `worklocal/host.cc`
unresolvable. Containment uses `commonpath` after `realpath` — `startswith` would
accept `/a/bc` for root `/a/b`, and realpath-first is what defeats `..` and a
symlink pointing out of the tree. Errors never echo the requested path.

**Security.** `/source` is the first read-any-file-under-root primitive here and
the daemon binds `0.0.0.0` by default, so it is gated on `_check_llm_auth` (the
browser's `api()` already sends the header) and startup warns when bound wide
with no password. `POST /apps/add` — unauthenticated, takes an arbitrary path,
and with `select=true` rewrites the very allowlist above — is now *auth or
loopback*: `run_debug_ui.sh` registers over `127.0.0.1` with no header and keeps
working, anything off-box needs the password.

**Performance.** `_aiedbg_paths()` scans dist metadata at ~150 ms and
`_source_roots()` calls it, so `_serve_source` hoists the roots instead of
recomputing them 2–3× per request, `_aiedbg_paths()` is memoised, and the ~2 MB
view parse behind the basename index is memoised on mtime. Warm requests went
**320 ms → 0.6 ms**. Cold render of a 1586-line `host.cc` is ~495 ms / 726 KB,
which is why `_SRC_FULL_LINES` is 2000 and larger files are windowed ±400 lines.

### `aieprofile.sh` — AIE hardware performance counters on a running board

`script/debug/aieprofile.sh` + `aieprofile.tcl` drive the Vitis `aieprofile`
package (`$XILINX_VITIS/scripts/vitis/util/aie_profile.tcl`, 4717 lines) against
a board over JTAG. It is the only Vitis profiling path that reaches a
**baremetal** run: `aiesimulator --profile` is simulator-only, and both
`aiecompiler --event-trace` and `adf::event::start_profiling` need an adf graph,
which the aiehlc tiling flow does not build.

```bash
source script/test/envlocal.sh                 # XILINX_VITIS + VEK385IP
python3 ../naiebaremetal/src/tool/test/runhw_vek385.py -nonreboot &
./script/debug/aieprofile.sh --example ../naiebaremetal/example/example_oob_4x4
vitis_analyzer aout/profile/example_oob_4x4_*/aie_trace_profile.run_summary
```

It is a **second JTAG client** on the board's `hw_server`, exactly like the debug
UI's aiedbg reads — the app is started separately and this attaches while it
runs. Unlike aiedbg it *writes*: configuring a counter is a register write
(`mwr -force`), so it must not run alongside anything else owning the
performance counters. The vendor script says so itself (`aie_profile.tcl:1476`,
"turn off other usages of counters (e.g., ECC, trace)"); in this repo the
conflict to avoid is `aiegdb`'s `dma counter setup`.

Board host resolution is the same live chain the daemon uses — `--host` →
`$VEK385IP` → `hwlocal.sh` — because a board is a per-run choice and must never
be baked into a config file. Metric sets come from `aieprofile start -help`:
`heat_map`, `stalls`, `execution`, `floating_point`, `stream_put_get` and the
throughput sets for AIE tiles; `conflicts`, `dma_locks`, `dma_stalls_mm2s`,
`dma_stalls_s2mm` for the memory module; stall/throughput/`packets`/
`start_to_bytes_transferred` for interface tiles; channel and conflict stats for
memory tiles. Pass them with repeatable `--metric "-flag value"`.

Four vendor behaviours the wrapper exists to handle, each verified:

- **`-interval` does not default to 20 ms.** The help says it does, but the
  option carries no `default` clause, so an omitted `-interval` falls through to
  `0` and the loop polls flat out. The wrapper always passes it explicitly.
- **`start` connects to *localhost*.** It calls bare `connect` only when
  `[connect -list]` is empty — and that same branch is what selects the target,
  so connecting first (as we must, to reach a remote board) means we also have
  to select the target ourselves. Worse, a bare `connect` to a machine with no
  hw_server **launches one locally**, so the failure is a silent connection to
  the wrong device rather than an error.
- **Output goes to CWD, not `-work-dir`.** `write_run_summary` carries the
  comment "NOTE: for now, use pwd", so the wrapper cds into `--out` first.
  Default is `aout/profile/<example>_<stamp>/` in *this* repo, since the
  naiebaremetal checkout is often on a read-only mount.
- **`targets -filter {name =~ "$var"}` really does substitute.** Braces would
  normally prevent it, and xsdb echoes the filter *pre*-substitution so a failed
  match prints `name =~ "$targetName"` verbatim and reads like a bug. Confirmed
  by discriminator: an undefined variable in the same position fails with
  `can't read "...": no such variable`. Do not "fix" it to an interpolated
  string.

Three sibling packages ship in the same directory and take the same shape:
`aie_trace.tcl` (`aietrace start|stop|clearconfig`), `aie_status.tcl`
(`aiestatus examine` — AMD's own decode of the registers `aiediag.py`
reimplements, and a useful second opinion when a decode is in doubt), and
`aie_debug.tcl` / `aie_mem_dump.tcl`, alongside per-generation register maps
(`aie2ps_registers.h`, `aie2ps_attributes.h`).

#### The ordering constraint, and `aierun_retrigger.tcl`

`aieprofile start` configures the counters and then **blocks in its own polling
loop**, so it cannot start the application; and a PDI reprogram reconfigures the
AIE array, wiping any counter setup that preceded it. Both facts together fix the
order:

```
1. program once            runhw_vek385.py            PDI + ELF + run
2. reload ELF, stay halted aierun_retrigger.tcl load  slow — before the profiler
3. configure + poll        aieprofile.sh
4. resume                  aierun_retrigger.tcl go    instant
```

`aierun_retrigger.tcl` exists only for steps 2 and 4: halt the A78, `dow -force`
the host ELF, and later `con` — no PDI reprogram, so the counters survive. The
download is deliberately step 2 rather than folded into step 4 because it runs at
JTAG rate and would otherwise consume the whole sampling window.

Getting this wrong produces an **all-zero CSV**: every tile present in
`METRIC_SETS`, sampled the full count, every value 0 — the counters were
configured after the application had already finished. Verified on
`example_oob_4x4`: the first capture (kept as `aout/profile/probe/`) is exactly
that, and the corrected ordering gave 15/16 tiles of live data.

Also note the board's `hw_server` is commonly bound `tcp:127.0.0.1:3121`, so an
external client cannot reach it — tunnel (`ssh -N -L 13121:127.0.0.1:3121 <board>`,
then `--host 127.0.0.1 --port 13121`) or start one with `-stcp:0.0.0.0:3121`.

#### A fourth trap: the module offset

Event ids carry their module as a **decimal offset** — `memory_event_offset 1000`,
`shim_event_offset 2000` (`aie_profile.tcl:77-78`). Load-bearing rather than
cosmetic: interface tiles are `(0,0)`/`(1,0)`, core tiles are `(0,0)`..`(3,3)`,
they collide in the CSV's `column,row`, and **the CSV has no module column**.
Without splitting on the offset a shim port counter is silently added to a core
tile's row. So `1031` is memory `dma_stall_s2mm_chan0` (31) and `2134` is shim
`port_running_0` (134).

#### Rendering: `aieprofile_summary.py` and `aieprofile_report.py`

Two readers over the same captures, both decoding event names from the Vitis
install's own `aie2ps_attributes.h` rather than a copied table, so a decode
cannot drift from the tool that produced the numbers:

- **`aieprofile_summary.py <dir>`** — tabular. One section per run: what it
  enabled (module → metric set → tile count) and the per-tile counters.
- **`aieprofile_report.py <dir>`** — visual. Hero figure, KPI tiles, the array
  drawn as the grid it physically is with a metric toggle, a ranked stall bar
  chart, and computed findings. Follows the house data-viz method: one sequential
  hue with a toggle rather than several ramps at once, palette validated in both
  modes, status colours reserved for callouts.

Two things the visual pass got wrong first, both invisible until rendered. The
ramp was anchored at 0, but these counters cluster in a narrow band far above it
(active cycles vary 670k–700k, a 4% spread) so the array came out as one flat
wash — it now spans the **non-zero data range**, with exact zero given its own
neutral swatch, since "never incremented" is a different statement from "small",
and the legend states the domain. And the ramp **reverses on the dark surface**:
"darker is more" is right on white where the high end is furthest from the
surface, but on near-black it points the largest values at the background and the
busiest tiles recede.

### Source grounding — the assistant cites the app's code, not just registers

Three layers describe an AIE design, and only two of them were reachable by the
embedded assistant:

| Layer | Answers | Comes from |
|---|---|---|
| live registers | what the hardware **did** | `aie_exec` |
| compiled schedule | what the tools **built** | `tile_info`, `get_flow_detail` |
| application source | what the developer **asked for** | the app's own `.cc` / `.cpp` |

The first two arrive free from tool calls, so answers settled there — accurate
about BD chains and DMA channels, and unusable, because nothing in them is a line
the user can change. `app_paths()` named the app *directory*, but a directory is
not an inventory: "read the kernel for this tile" meant guessing a filename, and a
wrong guess is indistinguishable from the app having no sources, so the assistant
quietly stopped trying.

`_app_source_manifest()` builds the inventory. It walks `app_dir` (skipping
`build/`, `Work/`, `worklocal/`, `aiesimulator_output/`, `.Xil/`, `arch/`, `ipc/`)
for `.cc`/`.cpp`/`.h`, drops anything whose banner says a tool wrote it, and
groups the rest as **Application**, **Headers**, **Build**, **Entry source** and
**Generated**. It is published three ways — inline in the system prompt, as
`backend_status.json > app_sources` (structured) and `app_sources_text`
(rendered, so the out-of-process MCP server needs no second formatter, exactly as
`session` / `session_summary` already pair), and through the
`mcp__debugui__app_sources()` tool, which is what refreshes it after an app
switch. Measured at ~2 ms for a naiebaremetal example, ~12 ms for `aout/`
(more files to read banners from), cached 20 s.

**Kernel name → definition site** is the part that makes it operational.
`_app_kernel_defs()` takes the kernel names the schedule attributes to tiles and
finds each one's definition, so `tile_info` saying `kernel: conv2` becomes
`src/convolution2.cc:19` — one Read away from an answer that quotes the user's
loop. `_app_def_line()` prefers a match whose line does not end in `;`, which
separates a signature with a body from a prototype or a call site; verified
against `conv2`, `stream_accum` and `matmul`, all landing exactly on the
definition. It is a name match, not a parse, and both the prompt and the
`source-grounding` skill say so — the line is a jump target to confirm, not a
citation to emit unread.

**Two flows, and only one of them can be walked.** A naiebaremetal example keeps
its sources in `<app>/src/`, so the walk finds them. The aiehlc flow's `app_dir`
is `aout/` — generated code only — while its real source is whatever
`--runtime-source-file` pointed at, anywhere in the tree. Nothing recorded that,
so `aiehlc.sh` now writes `worklocal/app_source.txt` at build time and
`_app_entry_source()` reads it (falling back to `sim_config.sh`'s `HOST_SRC` for
builds that predate it). Without it that flow's inventory is generated files and
nothing else. Its per-tile kernel labels are roles (`dskernel_receiver`), which no
source defines, so `_app_kernel_names()` also takes `view.kernel.function` — the
function the frontend actually lifted, and the only name that appears in the
user's file.

**Generated files stay in the inventory, labelled.** `host.cc`, `kernel.cc`, the
`.bcf` and the dfschedule MLIR are the best evidence of what the compiler decided
and the static tools already quote them — but they are overwritten on the next
build, so proposing an edit to one costs the user their time twice. The prompt
states that as a rule, and the group header repeats it where it is read.

The behavioural half lives in the prompt (a *Ground every explanation in the
application's source* section, plus a final workflow step) and in the
`source-grounding` skill, which carries the four chains worth following —
transfer size ↔ window declaration, repetition count ↔ graph run count, lock ids
↔ acquire/release calls, BD address ↔ the buffer the kernel writes — and two
rules that override the rest: never cite a line you have not read, and when the
source and the registers disagree, report the gap rather than reconciling it,
because that gap is usually the bug.

### Device map — one lane per flow

Flow lines are drawn as straight segments between tile centres, offset
laterally so parallel flows do not overlap. The offset used to be computed
**per edge**, from how many flows shared that particular hop — so a flow's own
offset changed from segment to segment as the sharing changed along its route,
and it arrived at a tile on one lane and left on another. That is the reported
gaps and jogs; in `example_external_buffer`:

```
net3  (6,4)->(5,4)  shared_by=1  offset=(0,0)
      (5,4)->(5,3)  shared_by=4  offset=(-12,-8)   <-- 12px break mid-route
net5  (5,0)->(5,1)  shared_by=2  offset=(-4,-2)
      (5,1)->(5,2)  shared_by=4  offset=(4,2)      <-- jumps across the tile
net6  (5,2)->(5,1)  shared_by=4  offset=(12,8)
      (5,1)->(5,0)  shared_by=2  offset=(4,2)
```

`flowLane(fi)` now assigns one lane per flow from its index in `dmFlowIds`, held
for the whole path, so a flow is one continuous line. It also removes an overlap
the old scheme allowed: two flows that never shared an edge both got offset 0
and drew on top of each other. The band is the width the surrounding code
already reserves (`streamMaxX/Y = OX_STEP*(N-1)/2`), and `OX_STEP`/`OY_STEP` now
scale down with the flow count so the fan stays ~40% of the tile pitch — at a
fixed 5px, 12 flows would span 55px of a 72px row step.

### Pane names and empty-selection state

The four quadrants each carry a `.pane-title`: **AIE Debug** (top-left, the
existing `h1`, bumped to 17px), **Run** (bottom-left: app/board selection, the
run buttons, the applog tail), **Info** (top-right, the tile/net detail panel)
and **Tools** (bottom-right: aiegdb / LLM / Search).

None of them takes a row of its own — each rides the pane's existing first row,
so naming the panes costs zero vertical space:

| Pane | Rides |
|---|---|
| AIE Debug | `#lefttop-header`, already a flex row beside the Grid/Device Map switcher |
| Run | the first `.ctrlrow`, ahead of `App:` |
| Info | `#panel-hdr`, a new flex row holding the title **and** `#panel-itemtabs` |
| Tools | `#contabs`, inline-block ahead of the folder tabs |

`#panel-hdr` also moves the item-tab strip **out** of `#panel`. `#panel` is the
scroll container, so the strip used to scroll away with the body; hoisting it
costs no extra height and keeps both the title and the tabs reachable. **Tools**
stays a child of `#cmdconsole` so it hides with the console, as `#rhsplitter`
already does.

That relocation exposed a latent bug in the "Show global / kernel-group" button:
it assigned to `#panel.innerHTML`, destroying the child containers
`panelRenderTabs`/`panelRenderBody` write into, so the next tile click hit a
null. It now writes to `#panel-body`.

The device map draws the whole array, including tiles absent from
`DATA.tiles`. Clicking one used to leave `panelItems` untouched — the tile
highlighted, but the Info pane still showing the *previously* clicked tile's
detail as if it belonged to the one just clicked. `panelClearTiles(note)` now
drops the `tile:` entries and shows `tile (c,r) — no schedule info`. Net entries
survive: they are a separate selection, and ctrl+click is unaffected because
adding a tile with no info correctly adds nothing.

The names are load-bearing for the embedded agent too — a user saying "the Info
pane" has to resolve to a tool call. They are published in three places, all
naming the same four panes:

- the LLM **system prompt**, as a pane → contents → *which tool answers it*
  table, plus a per-pane breakdown. This block also corrected two stale claims:
  the run controls were documented as top-right (they are bottom-left) and
  Device Map as a right-panel tab (it is a view of the top-left pane).
- **`list_panes()`**, which now prints the window layout before the `get_pane`
  ids, and each `get_pane` id names the pane it belongs to
  (`tile.hi` → "Info pane (top-right), tile High tab").
- the **`debugui-tools` skill**, with the same table and the pane column added
  to its `get_pane` reference.

`get_ui_state()` resolves what is actually selected, and now reports
`view` (`"grid"` | `"map"`) — `switchView()` pushes it, so the agent can tell
which view of the top-left pane the user is on instead of inferring it.

### Scan controls — mode is a selection, Scan is the action

Both toolbars (Grid `#overlayctl`, Device Map `#devmap-scan`) share one selection
via `LIVE.what`, and the three controls have distinct jobs: the **mode pills**
pick what to read, **Scan** reads it once, **live** re-reads every 2s. Picking a
mode used to read the board immediately — and in the Grid view it also silently
ticked the live checkbox on, starting a poll the user never asked for.

`pickOverlayWhat()` now scans only when `LIVE.enabled`; with live off it just
moves the selection, clears the previous mode's tints (leaving DMA colors under
a "Cores" selection would read as data that was never read) and says `click
"Scan" to read <mode>`. `scanOnce()`'s mid-flight retry is gated the same way:
a swallowed click always retries, but a mode changed during a scan only retries
under live.

The Grid toolbar had no Scan button — gating it there would have left no way to
trigger a one-shot — so it gets one (`#gridScanBtn`), sharing `#dmScanBtn`'s
styling and routed through the same `runScanNow()`.

### LLM tab — tool calls render as one row each

The daemon emits tool activity as line markers wrapped in blank lines
(`\n[tool: <name> <args>]\n`, `\n[tool result: …]\n`). The bubble is
`white-space:pre-wrap`, so every call cost three lines, its result another
three, long args wrapped across more, and the result never visually paired with
the call that produced it.

`llmProse()` now collapses a run of those markers — blank lines included — into
one `.llm-tools` block with a single `.llm-tc` row per call:

```
AIE exec       cmd target tile 5 0                      ✓ 3 lines
Tile info      col 0   row 3   section hi               ✓ 41 lines
Read           file_path …/aiediag.py   offset 1200     ✗ error
```

- **Name** — `llmToolName()` strips the `mcp__<server>__` prefix, splits
  `snake_case` and `CamelCase`, sentence-cases the result and upper-cases known
  acronyms (`aie_exec` → `AIE exec`, `get_ui_state` → `Get UI state`). Rendered
  in the sans stack; the raw name and full args stay in `title`.
- **Args** — `llmToolArgs()` parses the JSON and emits one background-filled
  `.tg` chip per key, label and value side by side, instead of raw JSON. Values
  over 60 chars ellipsis; unparseable args fall back to raw text. No chip border
  — it would grow the row.
- **Result** — folded onto the same row as `✓ 41 lines` / `✗ error`, or `…`
  while running. Left border grey in flight, accent when done, red on error.

**Pairing is FIFO.** Claude issues tools in parallel batches, so the stream is N
calls back to back and then N results — never strict alternation. Filling the
*last* unresolved row left every earlier call stuck on `…` and the surplus
results orphaned as nameless rows (which rendered as the tool named "Tool").
Each result now fills the **first** unresolved call. A result with no open call
at all — its call was separated into an earlier block by prose — renders as a
`↳` continuation row rather than a fake tool name.

Two parsing details. `LLM_TOOL_CALL` takes the **last** `]` on the line, because
a JSON list in the args embeds one; and the name stops at whitespace *or* `{`,
so `name{...}` parses the same as `name {...}`. On the daemon side
`_llm_tool_args()` truncates each **value** (80 chars) rather than the finished
string, so what reaches the browser is always parseable JSON — the old
whole-string 200-char cut produced fragments that could never render as chips.
It still sends only a tag for the result (line count / `empty` / `error` from
`is_error`), never the payload: results run to register dumps, and the point is
to spend less vertical space, not more.

One ordering constraint: block HTML is now restored **after**
`llmColorizeMarkers()`, not inside `llmProse()`. The colorizer's file-path regex
was rewriting the inside of a row's `title="…"`, injecting a `<span class=…>`
whose quotes terminated the attribute. `_llmBlocks` is therefore render-scoped
(reset at the top of `llmRenderText`) and placeholders survive until the last
step.

### Context pills — placed inline, and kept in the transcript

"+ Add context" used to park every selection in a row above the input and
prepend them all on send, which left two gaps: the blocks vanished from the
transcript the moment they were drained (no record of what the model was given),
and there was no way to say *where* in the sentence a snippet belonged.

The prompt box is a **`contenteditable` div, not a `<textarea>`** — a textarea
cannot hold elements, and the pills are now real chips sitting inline in the
message where the caret was:

```
you>  why does [Info pane · tile(0,3) ×] stall while [aiegdb output ×] is running?
```

There is no separate pill row any more. Each chip is `contenteditable="false"`
and `data-cid` points at the stored snippet.

Chips **select, copy, cut and paste like text**. Only the `×` carries
`user-select:none`, so a drag-selection takes the chip whole without dragging
the glyph along. `copy`/`cut` write two flavours: plain text via the same
walker the sender uses (a chip reads as `[[label]]`, never as its `×`), and
HTML, so pasting back into the box rebuilds a real chip rather than dropping to
a token. Paste reconstructs chips from `data-cid` and turns **everything else
into plain text**, so foreign markup can never enter the box; the HTML is read
with `DOMParser`, whose document is inert, rather than `innerHTML`, where a
pasted `<img onerror>` would fire. A chip pasted from another session, whose
snippet is gone, degrades to its `[[label]]` token.

Newlines on paste are the fiddly part, because a copy out of a contenteditable
comes back wrapped. A break is emitted **before** a block and never after —
emitting one after each block added a trailing blank line per wrapper. And a
chip is inline text that must never start or end a line of its own, so
`dropNl()` removes a break the wrapper put immediately before one and
`justPill` swallows the one that follows; an explicit `<br>` is user intent and
always stands:

```
pill wrapped in a div         newlines=0   ZW <PILL> ZW
text then div-wrapped pill    newlines=0   "a" ZW <PILL> ZW
div-pill then div-text        newlines=0   ZW <PILL> ZW "b"
two real text lines           newlines=1   "a" \n "b"
pill then explicit br         newlines=1   ZW <PILL> ZW \n "x"
inline pill between text      newlines=0   "a " ZW <PILL> ZW " b"
```

Three things make the caret behave. **The caret is remembered**: a
`selectionchange` listener stores the last range that was inside `#llmin`,
because selecting text in another pane moves the document selection out of the
box — so at the moment "+ Add context" is clicked there is no live caret, which
is why chips used to land at the end no matter where you were typing. **Chips
are flanked by zero-width spaces**, giving the caret a text position on both
sides; without them the browser cannot place a caret before a chip that starts
the message, or between two adjacent chips. **Backspace and Delete are
intercepted** so a chip goes in one press — left to the browser the first press
eats the flanking ZWSP and looks like it did nothing. Only that ZWSP counts as
"nothing in between": a space you typed is real text and gets deleted first, so
the behaviour stays predictable.

`ctxReadInput(pill)` is the single walker that reads the box back out, with the
`pill` callback deciding what a chip becomes: the full `[context from …]` block
for the outgoing prompt, or `[[label]]` for the transcript echo and the
"is there anything to send" check. It normalises `<br>` and any block the
browser produced into newlines, and the `&nbsp;` inserted after each chip back
into a space.

Labels are assigned once at creation (`ctxUniqueLabel`) and never renumbered —
they used to collapse `#1/#2` back to bare when a sibling was removed, which is
fine for a chip in a staging row but wrong for one the user has placed in a
sentence.

One trap the switch sprang: a document-level capture-phase handler suppresses
Space to stop the page scrolling while the device map is up, and it exempted
typeable focus **by tag name** (`INPUT`/`TEXTAREA`). The prompt box is a `div`
now, so every space typed into it was swallowed — and only in Device Map view,
which is the default. The guard tests `isContentEditable` too.

Three things the editable box needs that a textarea gave for free: Enter is
intercepted to send and **Shift+Enter** uses `insertLineBreak` (left alone the
browser builds `<div>` soup the reader would have to undo), **paste is forced to
plain text** (the default pastes markup, so copying from the transcript would
inject spans), and there is no `autoGrow` — it sizes itself, with an `input`
listener re-pinning the transcript via `llmKeepBottom`.

`llmSend(prompt, fromInput)` takes a flag because the box is now the source of
truth: only an interactive send drains it, so a programmatic caller cannot
drain a half-written message out from under the user.

The DMA scan no longer auto-sends anything. It used to fingerprint the issue
set and fire a `[auto] DMA scan detected N issues…` prompt at the LLM on first
detection, switching to the LLM tab unprompted. `_updateIssueBar` now just
renders the issue bar; clicking a row still focuses the offending tile, and
asking the model about it is the user's call.

`llmAddSentCtx()` echoes what went out as read-only pills above the `you`
bubble; clicking one expands the exact text sent. The prompt itself is echoed
with its tokens intact — showing the expanded blocks there would bury the
question.

### LLM tab — the transcript stays pinned to the bottom

`#llmmsg` is `flex:1 1 0` in a column it shares with the think dots, the context
pills and the input box, so anything that changes *their* height shrinks the
transcript. A view scrolled to the bottom keeps its `scrollTop`, so those pixels
come off the bottom and the last lines slide out of sight — 15px of clipped text
every time the dots appear.

`llmKeepBottom(mutate)` records whether the list was at the bottom (same `< 60`
threshold the append path uses), runs the mutation, and re-pins if it was.
Reading `scrollHeight` after the mutation forces the reflow, so it scrolls
against the new geometry. Used by `llmShowThink`, the context-pill row, and the
textarea's `autoGrow` — all three had the identical bug. A user who has scrolled
up is left where they are.

### LLM tab — the working indicator tracks the turn

`_llm_active` is set when a prompt is written to the `claude` subprocess and
cleared only on the stream's `{"type":"result"}` event, so `/llm/poll`'s `active`
flag spans the whole turn — tool calls, tool results and every text block. The
browser was hiding `#llmthink` on the *first* data chunk instead, so the dots
vanished the moment the opening tokens landed and the rest of the turn looked
idle. `llmPollOnce()` now drives the indicator from `active` alone
(`llmShowThink(!done)`), hiding it once at the `result` event, and on error or a
dead daemon.

`probeLLM()` had the matching gap: reloading mid-turn skipped to the tail and
never resumed, so the in-flight answer landed in a transcript nobody was tailing.
It now adopts an `active` turn — new pending message, indicator on, poll loop
restarted.

### Run state — the UI must not keep its own copy

The browser used to learn about a board run from exactly one place: the `/applog`
tail *it* started after clicking **Run**. Nothing else told it a run existed.
So a page reload, a dropped tail, or a run started from another tab left the
browser believing the board was free while `_run_proc` was still live in the
daemon — and every recovery path is gated on the daemon's belief, not the
browser's:

| Control | Refused mid-run because | Result when the two disagree |
|---|---|---|
| **Connect** | `/ping` returns "disabled during run" | reads as a dead link → the UI ssh'd to the board to restart `hw_server` (touching hardware for nothing), retried, failed again |
| **Run** | `start_run` → "a run is already in progress" | dead end |
| **Force stop** | *not* refused — but the button is only ungrayed by a successful Connect | unclickable, so the one control that fixes it is unreachable |

The result was a UI that could only be unwedged by restarting the daemon.
Three changes close it:

- **`GET /runstate`** — `running`, `run_id`, `pid`, `device`, `started_iso`,
  `age_s`, `status`, `stale` (live process, applog gone quiet), `debuggable`,
  plus `session`. No JTAG, so the browser polls it every 5s and on page load, and
  adopts whatever the daemon reports. `applyRunState()` is edge-triggered on
  `running`, so it can't fight the tail `pollLog` already drives.
- **`busy` on the mid-run refusals** — `/ping`, `/attach` and `/settarget` now
  answer `{ok:false, busy:true, run:{…}}`, and `/run`'s error carries `run` too.
  `busy` means "a run owns the link", which is *not* a dead link: the browser
  shows the run and points at Force stop instead of ssh-ing to the board.
- **`stop_run()` always releases the handle** — a child parked in an
  uninterruptible syscall survives SIGTERM *and* SIGKILL, and `poll()` then
  returns `None` forever. The daemon now clears `_run_proc` regardless and
  reports `abandoned: true`; "Force stop" means the daemon stops believing this
  run owns the board, which is what makes the state recoverable without a
  restart.

Force stop is consequently gated on the daemon's run state alone
(`updateRunButtons()`), never on `LIVE.connected` — a run this page never
started is precisely when it is needed.

### Path anchoring

This file resolves the repo root as three levels up (`src/tool/debug` → repo
root) and locates sibling tools (`aiegdb.py`, `aiemcp.py`) in its own directory.
`apppaltest.py` stays under `script/test/` and `applog` is written at the repo
root.

---

## 6. `streamswitch_crossref.py` — is the Stream switch panel telling the truth?

The tile panel's **Stream switch** section is three hops away from hardware:
`routingprovenancemap.json` → `_load_comm_paths()` → browser JS. Nothing in that
chain reads the code that actually programs the switch. This tool closes the
loop by deriving the same facts from the generated source and diffing them.

```bash
python3 src/tool/debug/streamswitch_crossref.py aout/worklocal
python3 src/tool/debug/streamswitch_crossref.py aout/worklocal -v   # + view artifacts
```

Two extractors, no shared input:

| Side | Input | Method |
|------|-------|--------|
| source | `routing.cc`, `host.cc` | parse `XAie_StrmConnCctEnable`, `XAie_StrmPktSwSlaveSlotEnable`, `XAie_StrmPktSwSlavePortEnable`, `XAie_StrmPktSwMstrPortEnable`, `XAie_Enable{Shim,Aie}*StrmPort` |
| ui | `routingprovenancemap.json` | run the real `renderTileRoutingSection()` from `schedule_view.HTML_TEMPLATE` under `node`, parse the rows back out |

Packet routes are resolved the way the switch resolves them: a slave slot drives
every master port on its tile whose **arbiter** matches and whose **MSelEn** has
the slot's **msel** line set. The UI instead borrows a "shared forward master"
from a sibling provenance record — the two agree only while that heuristic is
right, which is exactly what this checks.

Exit code is 0 on MATCH, 1 on deviation. Reported separately:

- **missing from UI** — programmed in source, never rendered
- **the UI invents** — rendered with nothing behind it
- **rows no flow focus can reach** — visible unfocused, lost under every flow filter
- **shim port enables absent from every UI surface** — dropped before any view
- **NO UI DATA** — `routingprovenancemap.json` absent, so the panel is empty by
  construction; a pipeline gap rather than a misreporting UI

`tests/test_streamswitch_crossref.py` runs this against two committed fixtures
(`tests/fixtures/streamswitch/{tiling,rawxaie}`) plus the parser and pairing unit
tests. Known deviations live in `KNOWN_DEVIATIONS` with their root cause and are
asserted in both directions, so fixing one makes the test ask to be updated.
Point it at a fresh build with:

```bash
AIE_XREF_WORKDIR=aout/worklocal python3 -m pytest \
  src/tool/debug/tests/test_streamswitch_crossref.py -q
```

Static parsing sees generated code only — switch writes issued at runtime with
computed arguments (core-trace setup, shim loopback) are out of scope.

---

## 7. `flow_crossref.py` — is the right info on the right tile and flow?

Section 6 asks whether a connection is real. This asks whether the UI puts it
on the flow and the tile it belongs to — the failure a user actually notices,
because a connection shown under the wrong flow looks entirely plausible.

```bash
python3 src/tool/debug/flow_crossref.py aout/worklocal
python3 src/tool/debug/flow_crossref.py aout/worklocal -v   # full lists
```

**The oracle.** `routing.cc` emits one `if (v2) { ... }` block per routing
group, in the same order as `routing_groups`, and each block's
`XAie_EnableAieToShimDmaStrmPort` splits it exactly where `shim_aie_to_ext`
splits the group — so a block yields a push record set and a pull record set.
Blocks are joined to flows on DMA endpoints (tile **and** port index) plus shim
column, taken from `dmaphopprovenacemap.json`. Neither input is what the UI
uses: groups carry no `flow_index` in the tiling flow, so `_load_comm_paths`
falls back to a frozenset of DMA tiles, which its own comments call ambiguous.

**What it checks**

| Check | Question |
|---|---|
| group structure | does each provenance group hold exactly its source block's connections? |
| flow connections | does flow N's panel show flow N's connections and no others? |
| flow tiles | does flow N's tile set match the tiles its block programs? |
| tile flows | do the Flows table, the focused Stream switch rows, and source agree per tile? |
| DMA badges | does each channel badge name the flow that channel serves? |
| focus badges | after focusing flow N, does any row still wear flow M's badge? |
| net panel badges | does the net panel agree with the tile panel about the same connection? |
| hop coverage | does every stream hop land on a tile in the flow? |
| shmem flows | does a tile listing a shared-memory hop for flow N admit to carrying N? |
| shim attribution | is every shim port enable owned by exactly one flow? |
| search anchor | does a search hit point at a tile the flow touches? |
| group key collisions | can two groups be confused by the DMA-tile key the UI joins on? |

It degrades rather than errors: without `routing.cc` blocks only whole-design
attribution is checkable, and without `dfscheduleprovenancemap.json` the DMA
badge checks are skipped. Both cases are stated in the output.

`tests/test_flow_crossref.py` pins the checks that pass today as regression
guards and the known defects in `KNOWN_DEFECTS` with their root cause, asserted
in both directions.

---

## 8. `Switch` scan — what the hardware is actually programmed with

Sections 6 and 7 check the UI against the *source*. This checks it against the
**board**: a fourth live-overlay mode next to DMA / Cores / Events that reads
the stream-switch registers off every tile.

Pick **Switch**, press **Scan**. The payoff is the `Dynamic (n)` routing source
that appears next to the scan button (§9) — a map rebuilt from what the array is
actually programmed with. Tiles are left untinted, except `unreachable` for any
whose registers could not be read.

There is deliberately no "this tile has switch config" tint: that is true of
nearly every tile in the array, so it would be a colour that never varies.
Comparing the two maps is the **diff** checkbox in §9.

**Cost.** aiegdb's `show switch` reads one register per `aiedbg reg read`
subprocess — 228 of them for a core tile. The switch register block is
contiguous and fits inside aiedbg's 256-word limit, so the scan issues a single
`aiedbg mem read` per tile instead. The simulator uses its per-register IPC
reader directly.

**Not polled.** Switch configuration is static — only the host program changes
it — so selecting `Switch` turns the 2 s live poll off. Re-scan explicitly
after a run programs the array.

Decoding matches the driver: a **circuit** master's `CONFIGURATION` field is
the source slave index, but a **packet** master's is `arbiter[2:0] |
msel_en[6:3]` ([xaie_ss.c:45-48](../../../thirdparty/alib/aie-rt/driver/src/stream_switch/xaie_ss.c))
and names no slave at all. Packet routes are then resolved the way the arbiters
do — a slave slot drives every master whose arbiter matches and whose MSelEn
has the slot's msel line set.

`tests/test_switch_scan.py` encodes the frozen fixture's real `routing.cc` into
a register image, decodes it back through the same helpers the board path uses,
and asserts all 28 tiles verify against the provenance map — so a wrong
register layout, field split or pairing rule fails the suite without a board.

---

## 9. Dynamic routing — rebuilding the map from the board

The `Switch` scan above can only answer *relative to the routing map*. That
leaves three cases where it has nothing to say:

- there is no `routingprovenancemap.json` (raw-XAie flows, a bare workdir),
- the map is stale — the board holds a different binary than the workdir
  describes,
- routing was reprogrammed at runtime, so the map was never right.

The same registers a scan already reads are enough to rebuild the flows from
scratch, so every `Switch` scan also reconstructs one. When it finds anything, a
**routing: `diff` | `Static` | `Dynamic (n)` | `Save JSON`** control appears next
to the scan button.

| Source | What you are looking at |
|---|---|
| `Static` | the map the compiler emitted — `routingprovenancemap.json` |
| `Dynamic` | flows rebuilt from the live stream-switch registers, with no reference to that map |

Switching source repoints every routing panel at once — device map, net panels
and each tile's **Stream switch** section — because they all read the same
`comm_paths`. **Clear** returns to `Static`.

**The `diff` checkbox owns every verdict.** With it off you are reading one map,
plainly: no badges on the rows and no tint on the tiles. Tick it and the *same*
rows — still the map you selected — get annotated against the other one, and the
tiles go `sw ok` / `sw ≠`.

Diff is relative to the source you picked, so it reads in the direction you are
looking:

| Selected | Rows drawn from | Badge on a row the other map lacks | Extra rows |
|---|---|---|---|
| `Static` | the compiler's map | `static only` | `dynamic only` |
| `Dynamic` | the rebuild from registers | `dynamic only` | `static only` |

Note that `dynamic` *is* the hardware, rebuilt from the registers the scan read,
so "board vs static map" and "dynamic vs static" are the same comparison — which
is why there is one control for it rather than a verdict that follows you around.
The two maps disagreeing is the ordinary case (a flow that never moved data is
enough to do it), so it stays behind the checkbox instead of painting the array
red on every scan.

**How a flow is recovered.** Intra-tile edges come from the master config
(circuit) or the arbiter/msel pairing (packet); inter-tile edges from the fixed
port wiring. A DFS from every terminal slave gives one fan-out tree per source,
and trees are then grouped by their **shim endpoint** — which is what the
compiler's map calls a flow. A broadcast to four cores is one push flow with
four sinks; four DMAs draining to one shim port are one pull flow, not four.

**Identity.** A reconstructed flow that matches a static one tile-for-tile and
edge-for-edge adopts its `id` and `flow_index`, so the DMA and lock tables keyed
off that index keep working. A flow that does *not* match keeps its own — it is
genuinely not that flow, and an empty BD table is the honest answer.

**Artifact.** Each scan writes `routingprovenancemap.dynamic.json` into the
workdir, and `Save JSON` downloads the same thing. It is emitted in the standard
`routing_groups` shape, so `streamswitch_crossref.py` and `flow_crossref.py`
read it like any other map.

Two limits worth knowing: a slot register carries no packet type, so `pkttype`
is reported as `0`; and a flow crossing a tile that could not be read splits
into fragments marked `partial` rather than being bridged over the gap.

`tests/test_switch_reconstruct.py` drives three round trips off the fixture's
own `routing.cc`: the rebuilt flows must reproduce the static `comm_paths`
edges (12/12 on `tiling`, 2/2 on `rawxaie`), the production `scan_tile` diff
must come back clean against the rebuilt map, and the written artifact must
survive being reloaded — that last one pins the connection *ordering*, which
`routing_edges_for_flow` is sensitive to.

---

## Quick reference

```bash
# Offline: generate the static schedule view
python3 src/tool/debug/schedule_view.py aout/worklocal

# Offline: check the Stream switch panel against the generated source
python3 src/tool/debug/streamswitch_crossref.py aout/worklocal

# Offline: check per-tile / per-flow attribution
python3 src/tool/debug/flow_crossref.py aout/worklocal

# Offline: dry-run the live tools (no board)
python3 src/tool/debug/aiegdb.py --dry-run -c "target tile 0 3; dma status"
AIEMCP_DRY_RUN=1 mcp dev src/tool/debug/aiemcp.py

# Live: scoped CLI
export AIEDBG_TARGET=xsdb://xxx.xxx.xxx.213:3121
python3 src/tool/debug/aiegdb.py

# Live: full browser debug server
python3 src/tool/debug/schedule_debug_server.py aout/worklocal --elf aout/worklocal/build/host --open
```

## See also

- `doc/design/live_debug_framework.md` — daemon design
- `doc/design/aiegdb_live_debug_framework.md` — scoped CLI / MCP design
- `.cursor/skills/aiehwdmadebug/SKILL.md` — live DMA debug workflow
- repo-root `.mcp.json` — registers `aiemcp.py` as the `aiegdb` MCP server
