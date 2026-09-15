# Design: Route core event-trace to the same-column MemTile

## Problem

`__Runtime_core_trace_setup` (in `src/mlir/runtime/aie_runtime.c`) armed the AIE
core trace unit and drained it with a **single intra-tile** stream connection —
`XAie_StrmConnCctEnable(dev, tile, TRACE, 0, DMA, s2mm_ch)` — into the core
tile's *own* data memory. That has two drawbacks:

1. The core tile's data memory is small, so a deep trace competes with the
   kernel's own input/output buffers for space.
2. The trace S2MM channel can collide with the movedata S2MM channel that feeds
   the kernel input into the same tile.

The request: route the trace stream DOWN through the intervening core tiles into
the same-column top **MemTile** (512 KB on AIE2PS), using multiple tiles to wire
the stream, and read it back from the MemTile.

## Approach (chosen)

**Replace** the intra-tile connect with a multi-hop circuit-switched route on a
single physical stream channel `strm_ch`, terminating in the top MemTile's S2MM
DMA. The MemTile is auto-selected (same column, top MemTile row); the caller
supplies the stream channel and the S2MM channel.

### Why "auto same-column, caller gives ports"

- Same-column keeps the route to a straight vertical drop (SOUTH/NORTH hops on
  one channel index) — a tile's SOUTH master port *k* is physically the NORTH
  slave port *k* of the tile directly below it, so the same index chains with no
  per-hop remap.
- The top MemTile (row `XAIE_AIE_TILE_ROW_START - 1`) is the fewest hops below
  the core rows.
- Leaving `strm_ch`/`s2mm_ch` to the caller keeps resource allocation where the
  rest of the flow already decides it, avoiding a hidden allocator in the runtime.

### Route (worked example, core tile `(4,4)`, gen5)

```
(4,4) core:    TRACE port 0        -> SOUTH master strm_ch
(4,3) core:    NORTH slave strm_ch -> SOUTH master strm_ch   (pass-through)
(4,2) memtile: NORTH slave strm_ch -> DMA master (S2MM s2mm_ch)
               S2MM BD -> [buf_addr, buf_addr+buf_len) in MemTile memory
```

For gen5, `XAIE_AIE_TILE_ROW_START = 3`, so the top MemTile is row 2. Cores are
rows 3..6, so `(4,4)` drops through `(4,3)` into memtile `(4,2)`.

## API change

```c
// before
AieRC __Runtime_core_trace_setup(XAie_DevInst *dev, XAie_LocType tile,
                                 uint32_t buf_addr, uint32_t buf_len,
                                 uint8_t s2mm_ch, uint8_t bdnum);
// after  (added strm_ch)
AieRC __Runtime_core_trace_setup(XAie_DevInst *dev, XAie_LocType tile,
                                 uint32_t buf_addr, uint32_t buf_len,
                                 uint8_t strm_ch, uint8_t s2mm_ch,
                                 uint8_t bdnum = 0);
```

- `strm_ch` — physical stream channel for every hop; must be `0..3` (core
  SOUTH-master / NORTH-slave and MemTile NORTH-slave port range).
- `buf_addr`/`buf_len` — now bytes into **MemTile** memory (the DMA-view address).

`__Runtime_core_trace_read` is **unchanged**: `XAie_DataMemBlockRead` is
loc-generic (dispatches on tile type internally and supports
`XAIEGBL_TILE_TYPE_MEMTILE`, `xaie_mem.c:71-72`), so the caller simply passes the
**MemTile** loc instead of the core loc.

## Data flow

```
core trace unit (window ACTIVE_CORE..DISABLED_CORE, EVENT_TIME)
  -> TRACE port -> vertical strm_ch hops -> MemTile NORTH slave
  -> MemTile S2MM DMA (BD bdnum) -> MemTile memory [buf_addr, +buf_len)
  -> (host, after CoreWaitForDone + CoreDisable)
  -> __Runtime_core_trace_read(memtile_loc) -> __Runtime_core_trace_decode
```

## Error handling

- gen1 guard: `XAIE_RES_TILE_NUM_ROWS == 0` (no MemTiles) returns `XAIE_ERR`
  with a diagnostic before any connect.
- **MemTile BD/channel parity guard**: unlike a core tile, the MemTile DMA
  couples the BD number to the S2MM channel parity
  (`_XAieMl_MemTileDmaCheckBdChValidity`, `xaie_dma_aieml.c:1420`): an **even**
  channel (0/2/4) requires BD `< 24`, an **odd** channel (1/3/5) requires BD
  `>= 24`. `__Runtime_core_trace_setup` validates `(s2mm_ch, bdnum)` against this
  and returns `XAIE_INVALID_ARGS` with a precise message rather than letting the
  driver emit its opaque `[AIE ERROR] ... Invalid BdNum, ChNum combination`.
- Every `XAie_StrmConnCctEnable` and DMA call is rc-checked; the first failure
  prints tile coords + channel and returns the driver rc.

### Caller obligation (from a live HW failure)

The first board run used `s2mm_ch = 1` (odd) with `bdnum = 4` (< 24) and hit the
driver check above. The core-tile version had no such rule. `aieml_perf.cc` now
uses `TRC_S2MM_CH = 0` (even) with BD 4 — the movedata channel-0 clash the old
comment warned about was a *core-tile* concern and does not apply to the
otherwise-unused trace MemTile.

## Testing / verification

1. **Syntax/type check** (done): `g++ -std=c++17 -fsyntax-only -DAIE_GEN=5
   -D__AIESIM__ ... src/mlir/runtime/aie_runtime.c` → exit 0, no diagnostics.
   The trace function is outside any `__AIESIM__` guard, so this exercises the
   edited code; all symbols (`TRACE/SOUTH/NORTH/DMA`, `XAIE_ERR`,
   `XAIE_RES_TILE_NUM_ROWS`, `XAIE_AIE_TILE_ROW_START`, DMA BD APIs) resolve.
2. **Caller migration**: `example/perf/aieml_perf.cc` — setup call passes the new
   `strm_ch` (`TRC_STRM_CH = 0`); read call takes the MemTile loc `(4,2)`.
3. **End-to-end (HW)**: build via the single-kernel flow
   (`source script/aiehlc.sh --aie-version 5 --runtime-source-file
   example/perf/aieml_perf.cc`), run on the board, confirm the decoded trace
   timeline prints from the MemTile.

## Open caveat

The MemTile DMA-master **port index** is taken to equal `s2mm_ch`, mirroring the
single-hop `TRACE -> DMA` convention this replaces. Like the trace packet decode,
that mapping is worth confirming against one hardware capture.

## Files changed

- `src/mlir/runtime/aie_runtime.c` — `__Runtime_core_trace_setup` multi-hop
  route + MemTile S2MM BD; doc comment; `strm_ch` param.
- `src/mlir/runtime/aie_runtime.h` — declaration + doc.
- `example/perf/aieml_perf.cc` — `TRC_STRM_CH` define, setup call, MemTile read
  loc, comments.

## Follow-up: resource-map-driven resource selection

The multi-hop route above still took `strm_ch`/`s2mm_ch`/`bdnum` and the packet
id (hard-coded `1`) purely by convention. In a full tiling run the data plane
already occupies stream-switch ports, packet ids, and MemTile DMA channels along
the same column the trace drops through, so a convention pick can silently
collide with a live data flow. `__Runtime_core_trace_setup` now consults the
generated routing resource map (when one is available) to pick trace resources
that avoid the recorded data-plane routing, and falls back to the convention
values unchanged when no map is passed.

### Delivery: by parameter, NULL ⇒ fallback

`__Runtime_core_trace_setup` gained two trailing (defaulted) parameters:

```c
AieRC __Runtime_core_trace_setup(XAie_DevInst *dev, XAie_LocType tile,
                                 uint32_t buf_addr, uint32_t buf_len,
                                 uint8_t strm_ch, uint8_t s2mm_ch, uint8_t bdnum = 0,
                                 const struct AieResourceEntry *resmap = 0,
                                 int resmap_count = 0);
```

- `resmap == NULL || resmap_count <= 0` ⇒ behaviour is exactly as before
  (raw/single-kernel flow, or any caller that has no map).
- The only real caller, `__Runtime_core_trace_begin_ch`, passes the globally
  generated table guarded by the feature macro:
  ```c
  #ifdef AIE_HAVE_RESOURCE_MAP
      const struct AieResourceEntry *rm = __aie_resource_map;
      int rmn = __aie_resource_map_count;
  #else
      const struct AieResourceEntry *rm = 0; int rmn = 0;
  #endif
  ```
  `AIE_HAVE_RESOURCE_MAP` is defined by the `__has_include(<aie_resource_map.h>)`
  guard at the top of `aie_runtime.c`, so the map is used only when the tiling
  flow has emitted one into `worklocal/`.

### Header enrichment (circuit ports now queryable)

`src/tool/debug/resource_json_to_header.py` previously carried only
`packet_connect` ports (recv / local_dma / forward_master) and dropped
`circuit_connect` and `circuit_connect_pair`. The trace route is a chain of
*circuit-switched* pass-through hops, so those were exactly the entries needed to
detect collisions. `struct AieResourceEntry` gained generic slave/master fields:

```c
/* circuit_connect / circuit_connect_pair / shim_* generic ports */
const char *slave_dir;  int slave_idx;
const char *master_dir; int master_idx;
```

`build_entries` is now kind-aware: `circuit_connect` emits one row (slave/master);
`circuit_connect_pair` — which spans two tiles — emits **two per-tile rows**
(one for `src_tile`, one for `dst_tile`), each carrying that tile's own
slave/master; `shim_*` carry their single port in the master fields to keep the
table complete. `packet_connect` rows keep the new fields empty (NONE/-1). The
deterministic sort key and `__Runtime_print_resource_map` were extended to
include the new fields.

### Selection

An `#ifdef AIE_HAVE_RESOURCE_MAP` block of small `static` query helpers scans the
passed table filtered by `(col,row)`:

- `resmap_master_used(...)` — a master port in use (checks `fwd_*` and `master_*`).
- `resmap_slave_used(...)` — a slave slot in use (checks `recv_*`, `local_dma`
  with `dir=="DMA"`, and `slave_*`).
- `resmap_pktid_used(...)` — a packet id already taken on the tile.
- `resmap_dma_used(...)` — any port with `dir=="DMA" && idx==ch` (MemTile channel).

`resmap_pick_trace_resources(...)` uses them to choose, for the source core at
`(col,srcRow)` dropping to the top MemTile at `(col,mtRow)`:

- `strm_ch` ∈ 0..3 — free SOUTH-master on the source, free NORTH-slave **and**
  SOUTH-master on every pass-through core row (`srcRow-1 .. mtRow+1`), and free
  NORTH-slave on the MemTile — **and** not already claimed by an earlier trace
  setup on this column (see claim tracker below). (One physical channel index
  chains the whole vertical drop, so it must be free at every hop.)
- `s2mm_ch` ∈ 0..3 — first MemTile DMA channel with no `dir=="DMA"` usage and not
  already claimed by an earlier trace setup on this column.
- `pktid` ∈ 1..31 — first not used on the source tile.
- `bd` — parity-correct **and distinct per channel**: even `s2mm_ch` → BD `4+ch`,
  odd `s2mm_ch` → BD `24+ch`, honouring the MemTile BD/channel parity rule
  documented above and matching the convention `k_bd_for_slot {4,25,6,27}`. (A
  fixed `4/25` collapsed same-parity channels onto one BD; distinct `s2mm_ch`
  therefore now yields distinct BDs.)

`resmap_apply_trace_resources(...)` wraps the picker (extracted so `_setup` stays
under the 200-line rule): a no-op when `resmap` is NULL, otherwise it overrides
`strm_ch`/`s2mm_ch`/`bdnum`/`pkt_id` in place and logs the choice, or logs that
no free candidate was found and keeps the convention values. It **only selects**;
it does not claim. The chosen `pkt_id` flows into both `XAie_PacketInit(pkt_id, 1)`
and the TRACE slave-slot enable, and is reported in the applog (see below) so
tooling sees the value actually programmed.

**Claiming is deferred to the point of physical occupation.** The claim is *not*
recorded at selection time; instead `_setup` calls `resmap_claim_trace(col,
strm_ch, s2mm_ch)` at the **end of section 3**, after the source packet-switch
(3a), every pass-through `XAie_StrmConnCctEnable` (3b), and the MemTile
`XAie_StrmConnCctEnable` (3c) have all succeeded. This records the FINAL values
(map-picked or convention fallback) so the next trace setup on the same column
avoids them, and — because it runs only after the whole vertical route is actually
enabled — a route that fails part-way (early `return rc` from any 3a/3b/3c hop)
leaks no claim on a channel it never occupied. The call is guarded by
`#ifdef AIE_HAVE_RESOURCE_MAP` (matching `resmap_claim_trace`'s definition) and
records even when `resmap` was NULL at runtime, since the convention values still
physically occupy the channel.

### Per-column claim tracker (mutable state complementing the const map)

The generated `__aie_resource_map` is `const` and records only the **data-plane**
routing; it cannot record the trace path's *own* resource usage. Without extra
state, two traced tiles in the same column would each re-pick the same first-free
`strm_ch`/`s2mm_ch` and collide on the shared pass-through segment and the MemTile
NORTH-slave / S2MM channel. (The convention path in `_begin_ch` avoids this via
`s_trace_col_used[col]` handing each tile a distinct slot; the map-driven override
had no equivalent memory.)

Two small mutable per-column bitmasks bridge the gap:

- `s_trace_strm_claim[col]` — bit `ch` set ⇒ `strm_ch == ch` taken on this
  column's vertical route.
- `s_trace_s2mm_claim[col]` — bit `ch` set ⇒ `s2mm_ch == ch` taken on this
  column's MemTile.

`col` is partition-relative, bounded by `AIE_TRACE_CLAIM_COLS` (64, matching the
`_begin_ch` column width); an out-of-range `col` simply skips the claim check
(the map + convention slot allocation still apply). `resmap_strm_claimed()` /
`resmap_s2mm_claimed()` gate the picker's `strm_ch` / `s2mm_ch` loops;
`resmap_claim_trace()` (called from the applier) sets the bits after each setup.
This is deliberately per-run process state (reset only on process restart), which
is the correct lifetime — it must persist across the multiple `_setup` calls of a
single multi-tile trace session.

Because one physical channel index chains the whole vertical drop (an upper
tile's SOUTH-master idx **is** the NORTH-slave idx of the tile below), the picker
validates that single `strm_ch` at *every* hop — source SOUTH-master, each
pass-through core's NORTH-slave + SOUTH-master, and the MemTile NORTH-slave — and
all three `XAie_StrmConnCctEnable` / packet-switch sites (3a/3b/3c) consume that
one validated value. There is deliberately no per-hop resmap query at each
`XAie_StrmConnCctEnable`: a different channel per hop is physically impossible, so
selection is done once up-front over the entire route.

### Applog: `[TRACESTREAMCONFIG]` marker

`__Runtime_core_trace_setup` emits the programmed routing to the applog via the
`trace_emit_stream_config(...)` helper (also extracted for the 200-line rule),
which prints two forms:

- a grep-friendly one-liner tagged **`[TRACESTREAMCONFIG]`** carrying the
  packet-switch slot config on the source TRACE port (`in_port=TRACE:0`,
  `out_port=SOUTH:<strm_ch>`, `pkt_id`, `slot=0`, `mask=0x1F`, `msel=0`,
  `arbiter=0`) plus the MemTile landing (`mt_in_port=NORTH:<strm_ch>`,
  `dma=S2MM:<s2mm_ch>`, `bd`, `buf_addr`, `buf_len`). The `slot/mask/msel/arbiter`
  literals mirror the `XAie_StrmPktSw*` enable calls in section 3a;
- the existing `core_trace_stream_json` line (now also carrying `"mask":"0x1F"`)
  with the full hop-by-hop `hops` array.

### Caveat carried forward

The map records no BD numbers, so a BD-vs-data-DMA collision on the MemTile
remains undetectable — behaviour is unchanged from the convention path and noted
in a code comment. Parity correctness (even/odd channel ↔ BD range) is still
enforced.

### Files changed (follow-up)

- `src/tool/debug/resource_json_to_header.py` — `_new_row` helper; kind-aware
  `build_entries` (circuit_connect / circuit_connect_pair / shim_*); enriched
  `struct AieResourceEntry`; extended row format, sort key, and print function.
- `src/mlir/runtime/aie_runtime.h` — `struct AieResourceEntry;` forward decl and
  two defaulted params on `__Runtime_core_trace_setup`.
- `src/mlir/runtime/aie_runtime.c` — `resmap_*` query helpers, picker, and
  applier; per-column claim tracker (`s_trace_strm_claim` / `s_trace_s2mm_claim`
  + `resmap_strm_claimed` / `resmap_s2mm_claimed` / `resmap_claim_trace`);
  distinct-per-channel BD; `trace_emit_stream_config` helper emitting the
  `[TRACESTREAMCONFIG]` one-liner + `core_trace_stream_json`; `pkt_id` plumbed
  through `XAie_PacketInit` / slave-slot enable / `core_trace_stream_json`;
  `_begin_ch` passes the global map.

## Follow-up: memory-module DMA trace (packet id 2) + per-pkt-id demux

The core stream above traces only the **core module** (ACTIVE / stall / lock
core-state). A compute tile also has a **memory-module** trace unit that can
watch the tile's DMA engine (BD start/finish, stream/lock stalls, lock
acquire/release). This follow-up arms that second unit and merges its stream
onto the *same* vertical drain, distinguished from the core stream by packet id.

### Hardware basis

The AIE-ML compute tile exposes **two** trace stream-switch slave ports feeding
its stream switch: `AIE_TRACE` (port index 0 = CORE module) and `MEM_TRACE`
(port index 1 = MEMORY module). Both can packet-tag their output, so two logical
trace streams can share one physical drain and be separated downstream by packet
id. `AieMlTileTraceMod[]` (thirdparty/aie-rt reginit) defines index0 = MEMORY
(`XAIE_MEM_MOD`, own `.PktId`), index1 = CORE — one arbiter, distinct MSel.

### Chosen design (SEPARATE setup API, shared drain, shared buffer)

- **Core stream = pkt id 1** (TRACE port 0, `XAIE_CORE_MOD`), unchanged.
- **Mem stream = pkt id 2** (TRACE port 1, `XAIE_MEM_MOD`), armed by a dedicated
  `__Runtime_mem_trace_setup(dev, tile, dma_kind, dma_ch, arbiter, msel)`.
- Both streams share **ONE** packet-switch arbiter on the source SOUTH master,
  each with its own MSel; the master's `MSelEn` bitmask is the OR of the two
  MSels so the arbiter round-robins the two packet ids onto the single
  `strm_ch`. `__Runtime_core_trace_setup` enables the SOUTH master with the
  combined MSelEn and calls `__Runtime_mem_trace_setup` before enabling the core.
- The mem stream rides the **same** TRACE→SOUTH→MemTile-S2MM route and lands in
  the **same** MemTile buffer as the core stream; there is no second drain and no
  second buffer.
- Wiring is exposed through defaulted trailing params on
  `__Runtime_core_trace_setup` — `mem_dma_kind = AIE_TRACE_DMA_S2MM`,
  `mem_dma_ch = 0` — so existing callers automatically get an S2MM-ch0 mem trace;
  `AIE_TRACE_DMA_NONE` opts out.

### Parametric DMA channel + slot map (mem unit, 8 slots)

`mem_dma_kind` (`AIE_TRACE_DMA_S2MM` / `_MM2S`) and `mem_dma_ch` (0/1) pick which
DMA channel's events fill the 8 mem trace slots. The `TASK` event variants are
used for BD start/finish. Slot → event (S2MM direction):

| slot | mem `XAIE_EVENT_*_MEM`            | generic decoder name |
|------|----------------------------------|----------------------|
| 0    | DMA_S2MM_c_START_TASK            | `DMA_START`          |
| 1    | DMA_S2MM_c_FINISHED_TASK         | `DMA_FINISH`         |
| 2    | DMA_S2MM_c_STALLED_LOCK          | `DMA_STALL_LOCK`     |
| 3    | DMA_S2MM_c_STREAM_STARVATION     | `STREAM_STALL`       |
| 4    | DMA_S2MM_c_MEMORY_BACKPRESSURE   | `MEM_BP`             |
| 5    | GROUP_LOCK                       | `LOCK_GRP`           |
| 6    | LOCK_SEL0_ACQ_GE                 | `LOCK_ACQ`           |
| 7    | LOCK_0_REL                       | `LOCK_REL`           |

> **gen2 lock-event asymmetry.** On AIE-ML / AIE2PS the lock *acquire* event was
> split into selectable-lock variants (`LOCK_SEL0_ACQ_EQ`=44, `LOCK_SEL0_ACQ_GE`=45),
> so the classic gen1 `LOCK_0_ACQ` enum maps to `XAIE_EVENT_INVALID` and
> `XAie_TraceEvent` rejects it — slot 6 therefore arms `LOCK_SEL0_ACQ_GE`. The
> *release* event was **not** split: the plain `LOCK_0_REL` (event 46) survives on
> the compute-tile memory module for both gen2 variants (only the MemTile has a
> `LOCK_SEL0_REL`), so slot 7 arms `LOCK_0_REL` and the trace captures real lock
> releases. (An earlier revision mistakenly armed both ACQ_EQ and ACQ_GE and
> claimed the compute-tile mem module had no release event; that was wrong.)

For `_MM2S` the direction-specific slots swap: slot3 → `STREAM_BACKPRESSURE`
(still decoded as `STREAM_STALL`), slot4 → `MEMORY_STARVATION` (`MEM_BP`). The
window uses `TRUE_MEM`/`NONE_MEM` (free-running from arm); both units share the
tile timer set by the Start frame, so the mem stream shares the core stream's
time base.

### Read-side demux by packet id

The trace framing tags every 8-word packet's header with its packet id
(`word & 0x1F`). Both decoders now **group packet payloads by packet id** before
decoding, and decode each group with its own 8-slot name table:

- pkt id 1 → core table `s_core_trace_slot_name` (C) / `SLOT_NAMES` (Python).
- pkt id 2 → mem table `s_mem_trace_slot_name` (C) / `MEM_SLOT_NAMES` (Python)
  = `("DMA_START","DMA_FINISH","DMA_STALL_LOCK","STREAM_STALL","MEM_BP",
  "LOCK_GRP","LOCK_ACQ","LOCK_REL")`.

Groups are decoded in ascending pkt-id order (core before mem). A legacy
single-id buffer (including id 0) decodes as one group with the core table —
exactly the pre-demux behaviour — so old captures still read correctly.

- Python: `core_trace_decode.py` gained `MEM_SLOT_NAMES`, `table_for_id(pkt_id)`,
  and a `stream` arg on `event_category(name, stream)`; `decode()` demuxes by id.
- C: `__Runtime_core_trace_decode` gained a per-pkt-id context/group walk
  (`__core_trace_ctx`, `__core_trace_decode_group`, `__core_trace_table_for_id`);
  `__core_trace_names` now takes the active name table.

### Timeline: a distinct "mem dma" lane despite the STREAM_STALL name clash

`STREAM_STALL` appears in **both** the core slot table (slot 2) and the mem slot
table (slot 3), so the off-device timeline cannot tell the two streams apart by
name alone. To keep the mem DMA events in their own lane, the unified dump
(`__Runtime_aie_trace_profile_dump`) tags mem-stream interval lines with a
backward-compatible `stream=mem` discriminator:

```
[TIMESYNC] trace tile=4,4 2005  DMA...                 (core: no stream= tag)
[TIMESYNC] trace tile=4,4 stream=mem 2005  DMA_START   (mem-module DMA stream)
```

The discriminator is threaded through the timeline pipeline: `AieTraceInterval`
carries the producing stream's `names` table (set on flush) so the dump can pick
`s_mem_trace_slot_name` and emit `stream=mem`; `host_aie_timeline.py`'s
`_TRACE_IV` regex captures the optional `stream=` group into a 4-tuple
`(s_cyc, e_cyc, names, stream)`, and `correlate()` routes `stream=="mem"`
intervals into a separate `"tile C,R mem dma"` lane (core/port lanes unchanged).
`timeline_gui.py` adds mem-DMA colors (`DMA_START`/`DMA_FINISH`/… ) and a
`MEM_STALL_ORDER`, plus friendly legend labels.

### Tests

- `test_core_trace_decode.py` — extracts `s_mem_trace_slot_name[8]` from the C
  source alongside the core table; adds mixed-buffer demux cases
  (`test_demux_core_and_mem_streams`, `_mem_slot_names_and_category`,
  `_core_and_mem_categories_distinct`) and `test_profile_dump_tags_mem_stream`
  (core lines carry no `stream=`, mem lines carry `stream=mem`). 24 pass.
- `test_core_trace_decode_cython.py` — mirrors the demux case in-process
  (`test_cy_demux_core_and_mem_streams`). 15 pass.
- `test_host_aie_timeline.py` — updated to the current 4-tuple / `"tile C,R core"`
  lane naming and adds `test_mem_stream_gets_own_lane` (asserts a
  `tile C,R mem dma` lane and that the mem STREAM_STALL does not leak into the
  core lane). 11 pass.
- `test_timeline_gui.py` — 15 pass (mem colors/labels covered by self-test).

### Files changed (mem-module trace follow-up)

- `src/mlir/runtime/aie_runtime.h` — `enum { AIE_TRACE_DMA_NONE/_S2MM/_MM2S }`;
  `__Runtime_mem_trace_setup` decl; `mem_dma_kind`/`mem_dma_ch` defaulted params
  on `__Runtime_core_trace_setup`; `names` field on `AieTraceInterval`.
- `src/mlir/runtime/aie_runtime.c` — `s_mem_trace_slot_name[8]`;
  `mem_trace_events_for_chan` / `mem_trace_program_unit` /
  `__Runtime_mem_trace_setup`; shared-arbiter MSelEn wiring in
  `__Runtime_core_trace_setup`; per-pkt-id C decoder demux; `names` set on flush;
  `stream=mem` tag in the profile dump.
- `src/tool/debug/core_trace_decode.py` — `MEM_SLOT_NAMES`, `table_for_id`,
  `event_category(stream=...)`, pkt-id demux in `decode()`.
- `src/tool/debug/host_aie_timeline.py` — `stream=` in `_TRACE_IV`; 4-tuple
  intervals; `mem dma` lane in `correlate()`; self-test mem lines.
- `src/tool/debug/timeline_gui.py` — mem-DMA colors, `MEM_STALL_ORDER`,
  `event_color` update, legend labels.
- `src/tool/debug/tests/test_core_trace_decode.py`,
  `test_core_trace_decode_cython.py`, `test_host_aie_timeline.py` — demux and
  mem-lane cases; lane-naming/tuple updates.

---

## Pragma-driven mem-module DMA/stream selection (`#pragma aie_trace` 2nd tuple)

The runtime already threads `mem_dma_kind`/`mem_dma_ch` into
`__Runtime_core_trace_setup` / `__Runtime_mem_trace_setup`. This follow-up
exposes that selection from the front-end via an optional SECOND tuple on
`#pragma aie_trace`, in two forms:

```c
#pragma aie_trace((0, 3), (STREAM, "s2mm", 0))   // explicit dir + channel
#pragma aie_trace((0, 3), (STREAM, "mm2s", 1))
#pragma aie_trace((0, 3), (PARAMETER, "win_a"))  // named window -> resolved
#pragma aie_trace(0, 3)                            // unchanged: default S2MM ch0
```

### Grammar & backward compatibility

- Flat form `#pragma aie_trace(col, row)` is unchanged (no second tuple). The
  parser detects the nested form by a second `(` opening immediately after the
  outer `(` (i.e. `((col,row),(SEL,...))`).
- `STREAM` tuple: `(STREAM, "s2mm"|"mm2s", idx)` — direction string is
  case-insensitive; `idx` is the DMA channel (0/1). Maps to
  `dmaKind = S2MM(1)/MM2S(2)`, `dmaCh = idx`.
- `PARAMETER` tuple: `(PARAMETER, "win_x")` — a kernel window/port name resolved
  later to the physical `(direction, channel)` the tiling flow assigned on the
  traced tile.
- The selection applies to every `(col,row)` expanded from the first tuple
  (ranges still allowed). A malformed second tuple warns and falls back to
  Default (S2MM ch0) without dropping the tile. Dedup is on `(col,row)`; a later
  pragma's sel overrides an earlier Default.

### PARAMETER resolution

`resolveTraceParameterSpecs` (in `tilinglinalg_pipeline.cpp`) runs on the host
module just before `DfscheduleToApiPass` (while `dfschedule.config.create_io`
provenance still exists). It:

1. walks `create_io` and collects, per traced tile, the S2MM and MM2S channels
   in IR order;
2. maps the port name -> direction via `tensors[i].isInput` over the
   tensor-ordered `portVarNames` (input windows drain via **S2MM**, output via
   **MM2S**), and to the ordinal within that direction;
3. stamps the resolved `dmaKind`/`dmaCh` back into the `TraceTileSpec`.

Unresolvable names (not a window, or tile lacks a matching create_io) warn and
fall back to Default (S2MM ch0). `tensors`/`portVarNames` are populated on the
single-kernel tiling flow; multi-kernel mode passes empty `tensors`, so
PARAMETER there falls back to Default — STREAM works in both flows.

### Carrier type & runtime wrapper

- `TraceTileSpec` (in `passcoretraceinsert.h`) replaces `pair<int,int>` as the
  trace-tile carrier: `{col,row, TraceDmaSel sel, dmaKind, dmaCh, paramName}`.
  `TraceDmaSel = {Default, Stream, Parameter}`.
- `CoreTraceInsertPass` emits `__Runtime_core_trace_begin_dma(dev, col, row,
  dma_kind, dma_ch)` per tile (Default tiles pass S2MM(1)/0 — identical to the
  old `__Runtime_core_trace_begin`).
- `__Runtime_core_trace_begin_dma` (new) shares `trace_begin_impl` with
  `__Runtime_core_trace_begin` / `_begin_ch`; only the trailing
  `mem_dma_kind`/`mem_dma_ch` args of `__Runtime_core_trace_setup` differ.

### Verified

- `((0,3),(STREAM,"mm2s",1))` -> `__Runtime_core_trace_begin_dma(dev,0,3,2,1)`.
- `((0,3),(PARAMETER,"win_a"))` -> resolved `S2MM ch1` (win_a is the first input;
  first S2MM create_io on tile (0,3) is ch1) ->
  `__Runtime_core_trace_begin_dma(dev,0,3,1,1)`.
- `(0,3)` (no 2nd tuple) -> `__Runtime_core_trace_begin_dma(dev,0,3,1,0)` (S2MM
  ch0), semantically unchanged from before.

### Files changed (pragma-driven selection follow-up)

- `src/mlir/mlirfront/tilinglinalg/pass/passcoretraceinsert/passcoretraceinsert.h`
  — `TraceDmaSel` enum, `TraceTileSpec` struct, pass ctor/field type.
- `src/mlir/mlirfront/tilinglinalg/pass/passcoretraceinsert/passcoretraceinsert.cpp`
  — emit `__Runtime_core_trace_begin_dma` with per-spec kind/ch.
- `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.{h,cpp}` — signature
  type change; `resolveTraceParameterSpecs` before `DfscheduleToApiPass`.
- `src/llvm/aiehlc.cc` — `parsedTraceTiles` type; `AieTracePragmaHandler` 2nd-tuple
  parse.
- `src/mlir/runtime/aie_runtime.c` — `trace_begin_impl` refactor +
  `__Runtime_core_trace_begin_dma`.
- `src/mlir/runtime/aie_runtime.h` — `__Runtime_core_trace_begin_dma` decl.

## Compile-time validation of the mem-DMA selection

The 2nd tuple is validated so a typo or an unused channel **fails the build**
(non-zero exit) instead of silently arming the wrong DMA. Three checks, split
across two stages:

1. **Invalid STREAM direction** (parse-time, `aiehlc.cc`
   `AieTracePragmaHandler::HandlePragma`). A recognizable STREAM tuple whose
   direction is neither `"s2mm"` nor `"mm2s"` (e.g. `"s3mm"`) sets a
   `static bool parsedTraceFatal`. Both `runPipeline` entry points (multi-kernel
   ~4620, single-kernel ~5287) check it and `std::exit(1)` with
   `Aborting: invalid #pragma aie_trace directive(s).` This works for both flows
   because it needs no IR provenance.

2. **STREAM channel not used by the app** (resolver-time,
   `resolveTraceParameterSpecs` in `tilinglinalg_pipeline.cpp`). The requested
   `(direction, channel)` is compared against the channels the flow actually
   assigned on the traced tile (collected from `dfschedule.config.create_io`
   `DeclareTileOp`/`ConfigCreateIoOp`). A channel that is hardware-invalid or
   valid-but-unused (e.g. `mm2s ch1` on a tile that only drives `mm2s ch0`) makes
   the resolver return `false`, so `runPipeline` returns `false` and `aiehlc.cc`
   exits 1. The error lists the available channels.

3. **Unknown PARAMETER name** (resolver-time). A `paramName` that is not a
   declared kernel window/port makes the resolver return `false` with
   `PARAMETER "<name>" is not a kernel window/port name.` A known name whose
   direction ordinal has no matching create_io channel on the tile is likewise
   rejected.

`resolveTraceParameterSpecs` now returns `bool`. A **tile-presence guard**
(`if (!s2mmCh.count(key) && !mm2sCh.count(key)) continue;`) skips specs whose
tile has no create_io in the current module so multi-kernel meshes are not
spuriously failed by a tile another kernel owns. The `Default` form
(`aie_trace(col,row)`) is not validated (no regression).

### Verified (validation)

- `((0,3),(STREAM,"s3mm",0))` -> parse-time
  `Error: ... STREAM direction "s3mm" is invalid` + `Aborting` + EXIT=1.
- `((0,3),(PARAMETER,"win_adasfdsa"))` -> resolver
  `Error: PARAMETER "win_adasfdsa" is not a kernel window/port name.` +
  `Pipeline FAILED` + EXIT=1.
- `((0,3),(STREAM,"mm2s",1))` (tile only drives `mm2s ch0`) ->
  `Error: STREAM mm2s ch1 is not used by the app on tile(0,3). Available mm2s
  channels: ch0` + EXIT=1.
- `((0,3),(STREAM,"mm2s",0))` (valid) -> EXIT=0, emits
  `__Runtime_core_trace_begin_dma(v1,0,3,2,0)`.
- `((0,3),(PARAMETER,"win_a"))` (valid) -> EXIT=0, resolves `S2MM ch1`, emits
  `__Runtime_core_trace_begin_dma(v1,0,3,1,1)`.

### Files changed (validation follow-up)

- `src/llvm/aiehlc.cc` — `parsedTraceFatal` flag; STREAM branch hard-errors on
  invalid direction; abort checkpoints before both `runPipeline` calls.
- `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp` —
  `resolveTraceParameterSpecs` returns `bool`, validates STREAM channel usage
  and PARAMETER name, tile-presence guard; call site returns `false` on failure.
