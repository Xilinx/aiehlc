# Row-Fabric Control Return Path (Write-Ack + Read) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the row-multicast control fabric a return path so every target-row
column's WRITE-ack / register-READ response merges west down the shared spine to
the shim, and expose row read / write-ack runtime APIs + a demo.

**Architecture:** A new pure-planner routine (`acr_plan_return_chain_ex`) emits a
WEST-going, packet-switched return chain per row (each tile injects its own CTRL
response and forwards its east neighbor's), plus VRET spine-descent hops. The
unicast return route is extracted into a reusable helper so `rt_ctrl_route_setup_col`
becomes forward-only. Runtime row read/ack APIs drive N-packet shim S2MM drains.

**Tech stack:** C (host runtime + pure planner), C++ host unit test, XAie stream-
switch API, `apppaltest.py` E2E on PAL hardware.

**Design doc:** `docs/plans/2026-09-13-row-control-return-path-design.md`

> **ENVIRONMENT BLOCKER (read first):** the interactive shell login profile in this
> workspace hangs after its `XILINX_VITIS`/`PETALINUX` banner, so `git`, builds,
> and tests cannot run from the agent shell right now. Every task lists its exact
> build/test command; run them once the shell profile is fixed (e.g. run commands
> through a non-interactive `bash --noprofile --norc`, or fix the hanging profile).
> Do NOT mark a task complete on unrun tests — see verification-before-completion.

---

## Ground-truth references (verified against current tree)

- Planner: `src/mlir/runtime/aie_runtime_control_plan.c`
  - `acr_emit_slot` (`:39`), `acr_emit_master` (`:66`, books master port, keep_header=1),
    `acr_emit_op` (`:28`), `acr_book_port` (`:16`, master/slave domains separate).
  - `acr_plan_chain_ex` (`:102`) — **already simplified**: per tile emits CONSUME
    (mask `0x17`) + BCAST (+ TRANSIT_N on spine head) + SLAVE_EN + CTRL master `0x3`
    + EAST master `0xB` (non-last) + NORTH master `0x6` (spine head). `te_id`
    computed but UNUSED; NO last-tile branch / only-last / whole / transit-e slots.
  - `acr_plan_row_add` (`:196`) — idempotent; forward spine up-hops SOUTH→NORTH CCT
    (skip heads) to `row-1`; calls `acr_plan_chain_ex(rowidx=s->nrows, ACR_SOUTH)`.
- Header: `src/mlir/runtime/aie_runtime_control_plan.h` — `ACR_ARB_CTRL 0`,
  `ACR_NUM_SLOTS 4`, `acr_port {ACR_WEST,ACR_EAST,ACR_NORTH,ACR_SOUTH,ACR_CTRL}`,
  `acr_op`/`acr_oplist`/`acr_portbook`/`acr_state`.
- Emit layer: `__Runtime_ctrl_row_emit` (`aie_runtime.c:3728`) translates SLOT/
  SLAVE_EN/MASTER_EN/CCT generically; `rt_acr_chan` (`:3716`) maps SOUTH-master→VRET,
  NORTH-slave→VRET, NORTH-master→VFWD, SOUTH-slave→VFWD.
- Unicast route: `rt_ctrl_route_setup_col` (`aie_runtime.c:3489`); return block to
  extract = **`:3589`–`:3659`** (`(void)s2mm_ch;` through the S2MM enable + pmap).
- Row fabric: `__Runtime_ctrl_row_open` (`:3787`, stores `resp_s2mm_ch`),
  `rt_ctrl_row_shim_entry` (`:3816`, forward mux only), `__Runtime_ctrl_row_add`
  (`:3842`), `__Runtime_ctrl_row_broadcast_write` (`:3878`), `rt_ctrl_row_class_write`
  (`:3945`), `rt_ctrl_row_index` (`:3922`).
- Unit test harness: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp` +
  `build_ctrl_row_plan.sh` (currently STALE — asserts removed last-tile logic).

---

## Task 1: Return-chain constants in the planner header

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.h` (after the `ACR_ARB_CTRL`
  block, ~`:51`).

**Step 1: Add constants**

```c
/* --- Return chain (write-ack / read response) ---------------------------- */
#define ACR_ARB_RET 1 /* return arbiter, distinct from ACR_ARB_CTRL */

/* Return slots each live on their OWN slave port, so slot index == MSel. */
#define ACR_SLOT_RET_LOCAL 0   /* this tile's CTRL-slave response */
#define ACR_SLOT_RET_TRANSIT 1 /* east neighbor's westbound merged responses */
#define ACR_SLOT_RET_NORTH 2   /* upper spine head's descending responses */
#define ACR_MSEL_RET_LOCAL 0
#define ACR_MSEL_RET_TRANSIT 1
#define ACR_MSEL_RET_NORTH 2

/* Per-master MSelEn bitmaps for the return SOUTH/WEST masters. */
#define ACR_MSELEN_RET_HEAD                                                                                            \
    ((1u << ACR_MSEL_RET_LOCAL) | (1u << ACR_MSEL_RET_TRANSIT) | (1u << ACR_MSEL_RET_NORTH)) /* 0x7 */
#define ACR_MSELEN_RET_MID ((1u << ACR_MSEL_RET_LOCAL) | (1u << ACR_MSEL_RET_TRANSIT))       /* 0x3 */
#define ACR_MSELEN_RET_LOCALONLY (1u << ACR_MSEL_RET_LOCAL)                                  /* 0x1 */
```

**Step 2: Declare the new planner entry (near `acr_plan_row_add` decl, ~`:126`)**

```c
/* Emit the WEST-going packet return chain for [col_lo..col_hi] on @row: every
 * tile injects its CTRL-slave response (RET_LOCAL) and forwards its east
 * neighbor's (RET_TRANSIT, c<col_hi); the head (col_lo) drives SOUTH→VRET and
 * also merges the descending upper-spine responses (RET_NORTH). keep_header=1,
 * pkt_id/mask=0 (accept any response id). Ports are disjoint from the forward
 * chain. */
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi);
```

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.h
git commit -m "feat(control-plan): return-chain constants + acr_plan_return_chain decl"
```

---

## Task 2: Failing unit test for the return chain

**Files:**
- Modify: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`

**Step 1: Add a test asserting the return-chain ops** (2 rows on spine col 0,
`col_lo=0 col_hi=3`, matching the demo geometry). Assert, over the oplist produced
by two `acr_plan_row_add` calls (row 3 then row 5):

- Every tile `c` in `[0..3]` of each row has a SLOT on `ACR_CTRL` slave with
  `slot==ACR_SLOT_RET_LOCAL, msel==ACR_MSEL_RET_LOCAL, arbiter==ACR_ARB_RET,
  mask==0, keep? (slot has no keep, ignore)`.
- Every tile `c` in `[0..2]` has a SLOT on `ACR_EAST` slave with
  `slot==ACR_SLOT_RET_TRANSIT, msel==ACR_MSEL_RET_TRANSIT, arbiter==ACR_ARB_RET`.
- Each head (`c==0`) has a SLOT on `ACR_NORTH` slave `slot==ACR_SLOT_RET_NORTH`
  and a MASTER_EN on `ACR_SOUTH` with `mselen==ACR_MSELEN_RET_HEAD,
  arbiter==ACR_ARB_RET, keep_header==1`.
- Each non-head `c>0` has a MASTER_EN on `ACR_WEST` with
  `mselen == (c<col_hi ? ACR_MSELEN_RET_MID : ACR_MSELEN_RET_LOCALONLY)`.
- A return-descent CCT `NORTH-slave→SOUTH-master` exists on the pass-through rows
  (rows 1,2,4) at col 0, and NOT on the head rows (3,5).
- No `acr_book_port` conflict is returned (planner rc == ACR_OK for both adds).

Use the existing helper style in the file (scan `o.ops[0..o.n)` for matches). Add
a `count_ops(...)` predicate if the file lacks one.

**Step 2: Run — expect FAIL (link error: `acr_plan_return_chain` undefined)**

```bash
cd src/mlir/runtime/unitest && bash build_ctrl_row_plan.sh
```
Expected: undefined reference to `acr_plan_return_chain` (or assertion failures).

**Step 3: Commit the failing test**

```bash
git add src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
git commit -m "test(control-plan): assert row return chain + VRET descent (failing)"
```

---

## Task 3: Implement `acr_plan_return_chain` + wire into `acr_plan_row_add`

**Files:**
- Modify: `src/mlir/runtime/aie_runtime_control_plan.c`

**Step 1: Add the return-chain function** (after `acr_plan_chain_ex`, before
`acr_plan_chain`). Books slave ports explicitly (slots do not book); masters book
via `acr_emit_master`.

```c
/* Emit one return SLOT on tile (c,row)'s @sport slave: accept-any (pkt=0,mask=0)
 * into arbiter ACR_ARB_RET with @msel, then enable that slave port. Books the
 * slave port (idx 0) first. */
static acr_rc acr_emit_ret_slot(acr_oplist *o, acr_portbook *b, uint8_t c, uint8_t row, acr_port sport, uint8_t slot,
                                uint8_t msel) {
    acr_rc rc = acr_book_port(b, c, row, sport, 0, /*master*/ 0);
    if (rc != ACR_OK)
        return rc;
    if ((rc = acr_emit_slot(o, c, row, sport, slot, /*pkt=*/0, /*mask=*/0, msel, ACR_ARB_RET)) != ACR_OK)
        return rc;
    acr_op se = {.kind = ACR_OP_SLAVE_EN, .col = c, .row = row, .sport = sport, .sidx = 0, .pkt_id = 0};
    return acr_emit_op(o, &se);
}

/* Return chain for [col_lo..col_hi] on @row (east->west packet merge). Every tile
 * injects its CTRL-slave response; c<col_hi forwards the east neighbor's; the head
 * (col_lo) drives SOUTH->VRET and additionally merges the descending upper-spine
 * responses on its NORTH slave (harmless when nothing is above). */
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi) {
    if (col_hi < col_lo || col_hi >= 64)
        return ACR_ERR_BOUNDS;
    for (uint8_t c = col_lo; c <= col_hi; c++) {
        int is_head = (c == col_lo);
        int has_transit = (c < col_hi);
        acr_rc rc;
        /* Local response source. */
        if ((rc = acr_emit_ret_slot(o, b, c, row, ACR_CTRL, ACR_SLOT_RET_LOCAL, ACR_MSEL_RET_LOCAL)) != ACR_OK)
            return rc;
        /* East neighbor's westbound merged responses. */
        if (has_transit &&
            (rc = acr_emit_ret_slot(o, b, c, row, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_MSEL_RET_TRANSIT)) != ACR_OK)
            return rc;
        if (is_head) {
            /* Descending upper-spine responses (unconditional; idle if none). */
            if ((rc = acr_emit_ret_slot(o, b, c, row, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_MSEL_RET_NORTH)) != ACR_OK)
                return rc;
            /* Head descent: SOUTH master -> VRET, merge {local,transit,north}. */
            uint8_t mselen = (uint8_t)((1u << ACR_MSEL_RET_LOCAL) | (has_transit ? (1u << ACR_MSEL_RET_TRANSIT) : 0u) |
                                       (1u << ACR_MSEL_RET_NORTH));
            if ((rc = acr_emit_master(o, b, c, row, ACR_SOUTH, mselen, ACR_ARB_RET)) != ACR_OK)
                return rc;
        } else {
            /* Interior/last: WEST master merges {local, transit?}. */
            uint8_t mselen = (uint8_t)((1u << ACR_MSEL_RET_LOCAL) | (has_transit ? (1u << ACR_MSEL_RET_TRANSIT) : 0u));
            if ((rc = acr_emit_master(o, b, c, row, ACR_WEST, mselen, ACR_ARB_RET)) != ACR_OK)
                return rc;
        }
    }
    return ACR_OK;
}
```

**Step 2: Call it from `acr_plan_row_add`** — after the forward `acr_plan_chain_ex`
succeeds (after `:241`, before `s->rows[s->nrows++]`):

```c
    rc = acr_plan_return_chain(o, b, row, col_lo, col_hi);
    if (rc != ACR_OK)
        return rc;
```

**Step 3: Add the VRET spine-descent CCTs** — in the spine up-hop loop
(`acr_plan_row_add` `:219`), for each pure pass-through row `r` also emit the
return descent hop (NORTH-slave → SOUTH-master CCT). Inside the existing
`for (r ...) { if (acr_row_is_head(s,r)) continue; ...forward CCT... }` block, add
after the forward CCT emit:

```c
            acr_op rcct = {.kind = ACR_OP_CCT,
                           .col = shim_col,
                           .row = r,
                           .sport = ACR_NORTH,
                           .sidx = 0,
                           .mport = ACR_SOUTH,
                           .midx = 0};
            if ((crc = acr_emit_op(o, &rcct)) != ACR_OK)
                return crc;
```

Note: `rt_acr_chan` maps NORTH-slave→VRET and SOUTH-master→VRET, so this realizes
`XAie_StrmConnCctEnable(thru, NORTH, vret, SOUTH, vret)` at emit time.

**Step 4: Run — expect PASS**

```bash
cd src/mlir/runtime/unitest && bash build_ctrl_row_plan.sh
```
Expected: all return-chain assertions pass, no port-book conflicts.

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime_control_plan.c
git commit -m "feat(control-plan): row return chain + VRET spine descent"
```

---

## Task 4: Refresh the stale forward-path unit assertions

**Files:**
- Modify: `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp`

**Step 1:** Remove/replace assertions that reference the removed last-tile forward
logic (CTRL `0x7`, `ACR_SLOT_ONLY_LAST`, `ACR_SLOT_WHOLE`, `ACR_SLOT_TRANSIT_E`,
`ACR_MSELEN_CTRL_LAST`). The current forward planner emits, per tile: CONSUME
(mask `0x17`) + BCAST (+TRANSIT_N on spine head) + CTRL `0x3` + EAST `0xB`
(non-last) + NORTH `0x6` (spine head). Assert that instead.

**Step 2: Run — expect PASS**

```bash
cd src/mlir/runtime/unitest && bash build_ctrl_row_plan.sh
```

**Step 3: Commit**

```bash
git add src/mlir/runtime/unitest/test_ctrl_row_plan.cpp
git commit -m "test(control-plan): align forward-chain asserts with simplified planner"
```

---

## Task 5: Part 1 — extract the unicast return route into a helper

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c`

**Step 1:** Add a static helper `rt_ctrl_col_return_route(const __Runtime_CtrlInstance *c,
int port_evt)` containing the body currently at `aie_runtime.c:3589`–`:3659` (the
`(void)s2mm_ch;` line through the final `rt_pmap_port(... "SOUTH", rport ...)`),
re-deriving locals it needs (`dev, shim_col, dest_row, stream_id, s2mm_ch, shim,
dst, rport, vret`) from `c`. Return `AieRC`.

**Step 2:** In `rt_ctrl_route_setup_col`, delete lines `:3569`–`:3659` (the return
block) so it ends after the forward CTRL master setup + `port_evt` selects at
`:3567`, then `return XAIE_OK;`. Keep the forward-only pmap logs.

**Step 3:** In `__Runtime_ctrl_setup_routing` (the unicast composer at ~`:4129`),
call `rt_ctrl_col_return_route(inst, /*port_evt=*/1)` at the same point the return
route was previously established (before `rt_tct_s2mm_arm`; verify ordering by
reading the function). `__Runtime_ctrl_read_target` / `__Runtime_ctrl_push_target`
/ `example/debug/aieml_controlperf.cc` keep working unchanged.

**Step 4: Build the host** (unicast regression path; no logic change expected):

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/debug/aieml_controlperf.cc
```
Expected: builds clean; no new warnings around `rt_ctrl_route_setup_col`.

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c
git commit -m "refactor(runtime): extract unicast return route into rt_ctrl_col_return_route"
```

---

## Task 6: Part 3 — shim return leg (N-packet drain)

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c`, `src/mlir/runtime/aie_runtime.h`

**Step 1:** Add `static AieRC rt_ctrl_row_shim_return(const __Runtime_CtrlRowFabric *f,
int32_t s2mm_ch, int npkt)`:
- `rport = rt_shim_s2mm_port(f->dev, s2mm_ch)`.
- `XAie_StrmConnCctEnable(f->dev, shim, NORTH, RT_CTRL_VRET, SOUTH, rport)` +
  `XAie_EnableAieToShimDmaStrmPort(f->dev, shim, rport)` (mirrors `:3643`–`:3652`).
- Arm `npkt` S2MM BDs (one per response packet) via the existing `rt_tct_s2mm_arm`
  pattern (`:4018`). If `rt_tct_s2mm_arm` arms a single BD, loop it over `npkt`
  distinct BD ids / queued on `s2mm_ch` (read `rt_tct_s2mm_arm` first to match its
  BD-queue convention).
- pmap: `rt_pmap_port(f->shim_col,0,"NORTH",VRET,"ret","slave",...)` +
  `"SOUTH",rport,"ret","master"`.

**Step 2:** No standalone test (covered by the E2E in Task 10). Build:

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc
```

**Step 3: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c
git commit -m "feat(runtime): row-fabric shim return leg with N-packet S2MM drain"
```

---

## Task 7: Part 3 — `__Runtime_ctrl_row_read` + `__Runtime_ctrl_row_write_ack`

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c`, `src/mlir/runtime/aie_runtime.h`

**Step 1: `__Runtime_ctrl_row_read(f, row, tile_addr, nwords, out_vals, bd_id, mm2s_ch)`**
(≤200 lines). Structure (mirror `rt_ctrl_row_class_write` + the unicast
`__Runtime_ctrl_read_target`):
- Resolve `rowidx`/chain via `rt_ctrl_row_index`; `ncols = col_hi-col_lo+1`.
- `rt_ctrl_row_shim_entry(f, mm2s_ch)` (forward mux).
- `rt_ctrl_row_shim_return(f, f->resp_s2mm_ch, ncols)` (N-packet drain).
- Build a whole-row READ packet (`sid = (ACR_CLASS_WHOLE_ROW<<2)|rowidx`, id[4]=0)
  with a return stream id via `__Runtime_ctrl_pktize_read`; every column consumes
  and responds.
- Build `__Runtime_CtrlInstance` (dest = `col_lo`, `s2mm_ch=f->resp_s2mm_ch`,
  `resp_words` = per-packet count × ncols, allocate `token`).
- `__Runtime_ctrl_push(block=1)` then drain via `__Runtime_ctrl_tct_poll`.
- For each of the N drained packets, parse the stream header
  (`__Runtime_ctrl_parse_pkt_hdr` → `src_col`) and place its data via
  `rt_ctrl_read_extract` into `out_vals[src_col - col_lo]`.

**Step 2: `__Runtime_ctrl_row_write_ack(f, row, tile_addr, data, nwords, bd_id, mm2s_ch)`**
(≤200 lines). Same setup, but build a whole-row WRITE with `lastwriteack=1`; every
column returns a header-only ack; drain all `ncols` acks as the completion barrier.

**Step 3: Build**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc
```

**Step 4: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(runtime): row-fabric read + write-ack APIs (all-columns respond)"
```

---

## Task 8: Part 4 — demo `ctrlrow_demo.cc`

**Files:**
- Modify: `example/tileprogram/ccode/ctrlrow_demo.cc`

**Step 1:** Remove the `demo_probe_class` calls for only-last and all-but-last
(the class-subset probes, ~`:257`–`:271`); keep broadcast + whole-row. Simplify
`demo_probe_class` (`:130`–`:177`) to drop the `is_last` expectation branch
(whole-row ⇒ every column expects SET).

**Step 2:** Add a **read** demo: whole-row write a per-column sentinel across the
row, then `__Runtime_ctrl_row_read(...)` and compare each returned `out_vals[col]`
to the expected per-column value; print PASS/FAIL per column.

**Step 3:** Add a **write-ack** demo: `__Runtime_ctrl_row_write_ack(...)` across the
row; assert all N acks drained (return code + drained count).

**Step 4: Build**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc
```

**Step 5: Commit**

```bash
git add example/tileprogram/ccode/ctrlrow_demo.cc
git commit -m "test(demo): row read + write-ack; drop last-tile class probes"
```

---

## Task 9: Docs — CLAUDE.md + planner header comments

**Files:**
- Modify: `CLAUDE.md` (control-plane architecture paragraph),
  `src/mlir/runtime/aie_runtime_control_plan.h` (stale last-tile slot comments
  `:23`–`:26`, `:83`–`:97`, `:112`–`:127`).

**Step 1:** Update both to describe (a) the simplified uniform forward chain
(CONSUME `0x17` + BCAST; no only-last / last-tile logic) and (b) the new return
chain (per-tile CTRL-local + EAST-transit merge west, head SOUTH→VRET with
RET_NORTH merge, VRET spine descent) + the row read/ack APIs.

**Step 2: Commit**

```bash
git add CLAUDE.md src/mlir/runtime/aie_runtime_control_plan.h
git commit -m "docs(control-plan): document simplified forward + new return chain"
```

---

## Task 10: E2E verification on hardware

**Step 1: Generate + run**

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/ctrlrow_demo.cc
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```

**Step 2: Verify** — `./applog` shows the per-column read values matching the
per-column writes, all N write-acks drained, existing broadcast/whole-row probes
PASS, and ends with `device_teardown done` and NO `AIE ERROR`.

**Step 3:** If a return packet stalls, inspect `CONTROLPAN-PMAP` lines for the
`ret` ports (CTRL slave / EAST slave / NORTH slave / WEST master / SOUTH master)
and confirm arbiter `arb=1` and the VRET descent CCTs; use the aiedebug **Load
control plan** overlay.

---

## Files touched (summary)

- `src/mlir/runtime/aie_runtime_control_plan.h` — return constants + decl + comment refresh.
- `src/mlir/runtime/aie_runtime_control_plan.c` — `acr_plan_return_chain` + row_add wiring + VRET descent.
- `src/mlir/runtime/aie_runtime.c` — extract `rt_ctrl_col_return_route`; `rt_ctrl_row_shim_return`; row read/ack APIs.
- `src/mlir/runtime/aie_runtime.h` — declare row read/ack APIs.
- `src/mlir/runtime/unitest/test_ctrl_row_plan.cpp` — return-chain asserts + forward refresh.
- `example/tileprogram/ccode/ctrlrow_demo.cc` — read + write-ack demos; drop last-tile probes.
- `CLAUDE.md` — control-plane architecture notes.
