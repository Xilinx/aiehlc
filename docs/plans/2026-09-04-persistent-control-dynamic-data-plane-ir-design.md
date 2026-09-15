# Design: New IR/ops for a PERMANENT control plane + DYNAMIC (runtime-reconfigurable) data plane

**Status:** Proposed
**Date:** 2026-09-04
**Scope:** Design decision / IR proposal only — no code, no TableGen changes.

## 1. Context

The multi-tile GEMM pipeline (`tilinglinalg`) compiles data flows onto a single
physical stream-switch fabric, allocated once at compile time by one
`ResourceManager`. Control packets — register/config writes — are today used the
same way: a **one-shot**, compile-time-static bootstrap that configures the
network before data runs.

The accepted ADR
[`2026-09-03-control-vs-data-network-sharing-adr.md`](2026-09-03-control-vs-data-network-sharing-adr.md)
chose a **config-only, time-multiplexed** regime: carve a *temporary* reserved
control spine (forward `RT_CTRL_VFWD=4`, return `RT_CTRL_VRET=3`), control-write
the data network's config registers, then **release** the spine ports so the data
plane reuses them. Control and data never run concurrently.

This document covers the **different regime the user selected**:

- The control spine is **never released** — it is a **permanent (persistent)**
  control routing plane.
- Data-plane flows are **dynamic**: the host issues control packets **at runtime,
  during execution**, to **install / teardown / swap** data flows over the
  standing control plane.

This is the deliberate inverse of the ADR's release-after-bootstrap model. It
trades a permanent slice of the per-tile port budget (fwd loses idx 4, ret loses
idx 3) for the ability to reconfigure the data plane without recompiling or
re-loading a PDI.

**User decisions (locked):**
1. *Dynamic* = **runtime reconfig** — host issues control packets during
   execution to install/teardown/swap data flows.
2. *Layers* = **full stack** — new ops/attrs in `routing`,
   `routinghw`/`ResourceManager`, and `dfschedule` (plus a blueprint manifest).
3. *Deliverable* = **design doc only**, matching the two prior control docs.

## 2. Today's pipeline & the gap (verified against the repo)

Control-packet support already exists end to end, but it is **compile-time-static
and one-shot**:

- **Control intent op.** `routing.routingcontrolwrite` delivers config/register
  writes to a set of target tiles via stream-switch control packets; a
  `broadcast` attr selects shared vs individual addressing
  (`routing/td/routingop.td:66-86`). It is a *one-shot* delivery, not a standing
  plane.
- **Marker propagation.** A `control` `BoolAttr` is set by `tagControl` in
  `routingtodmap.cpp:936-939` and flows dmap→dmaphop; it is consumed in
  `passdmaphoptoroutinghw.cpp:1049-1051`, which promotes a shim→core flow to
  `StreamType::CONTROL`.
- **StreamType enum.** `FORWARDONLY, BROADCAST, CONTROL`
  (`routingimplement/include/hw/ResourceManager.h:117-124`). `CONTROL` reuses the
  `BROADCAST` packet fan-out engine and terminates at each target tile's CTRL
  master port (`routinghwlower.cpp:196`).
- **Allocator.** Ports are single-owner; reservation exists only at **tile**
  granularity (`reserveTile`/`isTileReserved`,
  `ResourceManager.cpp:479-498`). There is **no port-level permanent
  reservation**.
- **dfschedule runtime ops.** The executable schedule dialect has
  `config.dma_bd` (`dfscheduleop.td:142`), `config.create_io` (`:183`),
  `schedule.start_io` (`:254`), `schedule.launch_kernel_group` (`:236`),
  `schedule.wait` (`:267`), and locks. **None** issue a runtime control-packet
  register write or install/teardown a flow (see the full op inventory —
  `dfscheduleop.td` has no `ctrl_write`, `install_flow`, or `teardown_flow`).
- **Runtime C API already present** (the new ops would lower onto these):
  `__Runtime_ctrl_pktize_write` (`aie_runtime.c:3035`),
  `__Runtime_ctrl_push_target` (`aie_runtime.c:3762`),
  `__Runtime_ctrl_read_target` (`aie_runtime.c:3815`),
  `__Runtime_ctrl_setup_routing` (`aie_runtime.c:3734`),
  `rt_ctrl_route_setup_col` (`aie_runtime.c:3454`), spine consts
  `RT_CTRL_VFWD=4` / `RT_CTRL_VRET=3` (`aie_runtime.c:3375-3376`), and the
  `__Runtime_CtrlInstance` send context.

**Gap.** Everything is compile-time-static and one-shot. There is no notion of
(a) flow **lifetime** (persistent vs dynamic), (b) **permanent** port
reservation, or (c) **runtime** flow install/teardown driven by control packets.
The three gaps map one-to-one to the three groups of new IR below.

## 3. Cross-cutting: a lifetime/phase attribute

Add an enum attribute that distinguishes the forever control plane from
runtime-installable data flows:

```
#routing.lifetime<persistent | dynamic>
```

- `persistent` — allocated once, pinned for the whole program (the control
  spine and its gateway/reach tiles).
- `dynamic` — (re)installable at runtime; its ports come from a pool that
  excludes pinned ports and are returned on teardown.

Propagation reuses the existing `control`-marker mechanism: set on the abstract
routing op, forwarded onto dmap streams (as `tagControl` does in
`routingtodmap.cpp:937`), then onto dmaphop `create_path`, and finally read in
`passdmaphoptoroutinghw.cpp` next to the `control` check
(`passdmaphoptoroutinghw.cpp:1049-1051`). `lifetime` is orthogonal to
`StreamType`: a flow can be `CONTROL`+`persistent` (the spine) or
`FORWARDONLY`/`BROADCAST`+`dynamic` (a runtime data flow).

## 4. New ops by dialect

### 4.0 Layering principle: WHERE the control plane is set up

"Control-plane setup" is **not one thing** — it is three sub-concerns, and each
lands in a different layer. The deciding line between the two lowering dialects:

- **routinghw → `routing.cc`** = *static physical stream-switch wiring*, emitted
  once at init, compile-time-known (`XAie_StrmConnCctEnable`; see the control
  spine primitives at `aie_runtime.c:3475,3507,3596`).
- **dfschedule → `host.cc`** = *runtime host actions* (buffer alloc, DMA BD, S2MM
  arm, launches).

Note that today's runtime helper `__Runtime_ctrl_setup_routing`
(`aie_runtime.c:3734`) is **two things glued together**: (1)
`rt_ctrl_route_setup_col` programs the spine's circuit connections
(`aie_runtime.c:3740` → `XAie_StrmConnCctEnable`), and (2) it allocates the
response buffer + arms the shim S2MM drain (`aie_runtime.c:3744-3753`). Because
the plane here is **permanent** (never released, unlike the ADR), the *wiring*
(1) is compile-time-known and belongs in the **static routing layer**, while only
the *host send context* (2) is a genuine runtime resource. The setup therefore
splits:

| Sub-concern | Layer | Rationale |
|-------------|-------|-----------|
| Pin spine ports out of the dynamic pool (fwd 4 / ret 3) | **routinghw / ResourceManager** (§4.2) | Only place the port budget is tracked. |
| Wire the permanent spine's circuit connections | **routinghw → `routing.cc`** (§4.2), static, once | Plane is permanent & compile-time-known → emit statically with the rest of the fabric (same `StreamType::CONTROL` mechanism, `routinghwlower.cpp:196`), just marked `persistent`. |
| Allocate host send context (`ctrl_instance`: response buffer + BD + S2MM arm) | **dfschedule → `host.cc`** (§4.4) | Genuine runtime resource (`aie_runtime.c:3744-3753`). |

This is why there is **no** standalone `dfschedule.ctrl_plane_setup` op that
programs the route: for a permanent plane the routing is static. dfschedule only
allocates/arms the host context and drives the *dynamic* ops.

### 4.1 routing dialect (abstract intent)

| New op | Purpose | Key operands / attrs | Lowers toward |
|--------|---------|----------------------|---------------|
| `routing.control_plane` | Declare the **persistent** control plane as standing infrastructure (generalizes the one-shot `routingcontrolwrite`). | region op; `spine_fwd`/`spine_ret` port indices (default 4/3), gateway/reach tile set, `lifetime=persistent` | **routinghw** (static spine wiring in `routing.cc`, §4.2) **+** `dfschedule.ctrl_instance_init` (host context, §4.4) |
| `routing.install_data_flow` | Abstract **runtime intent**: install a named data flow over the control plane by delivering its stream-switch config as control writes. | ref to the target data flow; `lifetime=dynamic` | `dfschedule.install_flow` |
| `routing.teardown_data_flow` | Abstract **runtime intent**: uninstall a named data flow (disable its ports via control writes). | ref/handle to the installed flow; `lifetime=dynamic` | `dfschedule.teardown_flow` |

`routing.control_plane` is the persistent counterpart of
`routing.routingcontrolwrite` (`routingop.td:66-86`): where the latter is a
one-shot register delivery, the former declares a standing plane that later
`install_data_flow`/`teardown_data_flow` ops target repeatedly. It lowers across
**two** layers (§4.0): its spine wiring goes to routinghw (static), its host
context to dfschedule (runtime).

### 4.1.1 dmap & dmaphop — conversion carriers (NO new ops)

The pipeline passes through two intermediate dialects between `routing` and
`routinghw`/`dfscheblueprint`: **dmap** (logical dataflow) and **dmaphop**
(physical hops). For the control plane these are **pure attribute carriers** —
they need **no new ops**. This mirrors exactly how the existing `control` marker
already flows today:

| Conversion | Pass | What it does with the marker |
|------------|------|------------------------------|
| routing → dmap | `RoutingToDmapPass` | `tagControl` sets the `control` BoolAttr on the produced dmap stream ops (`routingtodmap.cpp:936-939`) |
| dmap → dmaphop | `DmapToDmaphopPass` | forwards `control` onto `dmaphop.create_path` (`dmaptodmaphop.cpp:465-487`, `path->setAttr("control", ...)` at `:486`) |
| dmaphop → routinghw | `DmaphopTodfscheblueprintPass` / `passdmaphoptoroutinghw` | reads `control` → `StreamType::CONTROL` (`passdmaphoptoroutinghw.cpp:1049-1051`) |

**The new `#routing.lifetime` attr rides the identical carrier** — set alongside
`control` in `RoutingToDmapPass`, forwarded onto `create_path` in
`DmapToDmaphopPass`, and read in `passdmaphoptoroutinghw` next to the existing
`control` check. Concretely:

- The **persistent spine** (`routing.control_plane`) lowers through dmap/dmaphop
  the same way `routingcontrolwrite` does today (a `control`+`persistent` flow),
  terminating at the CTRL sink — the only delta is the extra `persistent` bit the
  allocator honors (§4.2).
- A **dynamic data flow** (`install_data_flow`) lowers through dmap/dmaphop as an
  ordinary data flow (no change); it **only diverges at the blueprint stage**,
  where its physical-hop config is captured as a `ctrl_regdiff_manifest` (§4.3)
  instead of being emitted as static schedule config.

So dmap and dmaphop require **no new ops** — the "full stack" scope adds ops in
routing, routinghw, dfscheblueprint, and dfschedule, but **carries** the
`lifetime` attr through dmap/dmaphop on the existing flow ops.

**Nuance (not "zero change" — a lowering fork).** "Carrier" means no new *ops*,
not no new *pass logic*. The `dmaphop → routinghw` / `dmaphop → dfscheblueprint`
lowering must **branch on `lifetime`**:

- `persistent` (the spine) → allocate + **pin** its ports (§4.2) and **statically
  emit** its connections into `routing.cc`, exactly like a normal flow.
- `dynamic` (a runtime flow) → still run path-finding to compute its physical
  config, but **capture that config as a `ctrl_regdiff_manifest`** (§4.3) and
  **skip static emission** — the connections are installed later by control
  packets at runtime, not baked into `routing.cc`.

**Alternative considered — dedicated dmap/dmaphop ops (rejected).** Adding
first-class `dmap.control_plane` / `dmaphop.ctrl_spine` (etc.) ops would give
uniform representation across every dialect, but they would be **structural
clones** of the existing stream / `create_path` ops — same BFS path-finding, same
`ResourceManager`, same port assignment — differing only by the `lifetime` enum.
That duplicates all path-finding logic across two more passes plus TableGen,
builders, verifiers, and per-dialect unit tests for **no new physical behavior**:
`lifetime` is a *property* of a path (when it is programmed, whether its ports are
pinned), not a new *shape* of path. New ops are therefore added only where the
semantics stop being "move data along a path" — the runtime *actions*
(`install_flow`/`teardown_flow`/`ctrl_write`/`ctrl_read`, §4.4) and the register
*diff capture* (`ctrl_regdiff_manifest`, §4.3), neither of which has a hop-path
analog. This also follows the precedent already set by the shipped `control`
marker.

### 4.2 routinghw + ResourceManager (permanent port reservation)

| New op / capability | Purpose | Key operands / attrs | Notes |
|---------------------|---------|----------------------|-------|
| `routinghw.reserve_ctrl_port` (or a `persistent` bit honored by the allocator) | **Permanently** exclude the spine port indices (fwd 4 / ret 3) from the dynamic data pool. | tile, `PortDirection`, port index, `persistent=true` | Inverse of the ADR's time-multiplex *release* — the crux change. |
| `routinghw.ctrl_spine` (persistent `StreamType::CONTROL` flow) | **Statically wire** the permanent spine's circuit connections into `routing.cc`, emitted once at init. | fwd/ret port indices (4/3), gateway/reach tiles, `lifetime=persistent` | Reuses the existing CONTROL fan-out (`passdmaphoptoroutinghw.cpp:468-519`) + CTRL sink emission (`routinghwlower.cpp:196`), emitting `XAie_StrmConnCctEnable` (cf. `aie_runtime.c:3475,3507,3596`) statically instead of at runtime. **This is where the "control plane setup" wiring lives** (§4.0). |

**Allocator change (design note, not a new op).** Extend the existing
**tile-level** reservation (`reserveTile`, `ResourceManager.cpp:487`) to a
**port-level, persistent** reservation. Introduce a **two-phase allocation
model**:

1. **Phase A — pin the control plane.** Allocate the persistent spine and mark
   its ports `persistent`; they are removed from the dynamic pool for the whole
   program.
2. **Phase B — dynamic data flows.** Allocate each runtime data flow from a pool
   that **excludes pinned ports**. On `teardown_data_flow`, return the flow's
   ports to the dynamic pool — but **never** the pinned spine.

This is contrasted with the ADR, which releases the spine after bootstrap so the
data plane reclaims fwd 4 / ret 3; here those indices are lost for the program's
lifetime (see Risks §7).

### 4.3 dfscheblueprint (register-diff manifest)

| New op | Purpose | Key operands / attrs | Parallels |
|--------|---------|----------------------|-----------|
| `dfscheblueprint.ctrl_regdiff_manifest` | Capture a data flow's stream-switch config as `(tile_addr, value)` register diffs, to be packetized as control writes at install time. | list of `(tile_addr, value)` entries; ref to the owning flow | `dfscheblueprint.transfer_manifest` (`dfscheblueprintop.td:58`), `dfscheblueprint.flowconfig` (`:190`) |

The register diff is what the control plane physically delivers: installing a
dynamic flow == writing that flow's stream-switch route/BD config registers via
control packets; tearing it down == writing the disabling diff.

### 4.4 dfschedule (executable runtime) — the KEY new ops

Per §4.0, dfschedule owns only the **runtime** pieces — it does **not** program
the spine route (that is static routinghw). `ctrl_instance_init` therefore only
allocates + arms the host send context; the route is already wired by
`routing.cc`.

| New op | Purpose | Key operands / results | Lowering target (runtime C API) |
|--------|---------|------------------------|---------------------------------|
| `dfschedule.ctrl_instance_init` | Allocate + arm the host **send context** for the (already-wired) permanent spine: response buffer + shim S2MM drain. Does **not** program the route. | `!dfschedule.ctrl_instance` result | response-buffer alloc + `rt_tct_s2mm_arm` portion of `__Runtime_ctrl_setup_routing` (`aie_runtime.c:3744-3753`); the route half (`rt_ctrl_route_setup_col`, `:3740`) is emitted statically by routinghw instead |
| `dfschedule.ctrl_write` | Runtime control-packet register write. | ctrl_instance, tile_addr, data words | `__Runtime_ctrl_push_target` (`:3762`) / `__Runtime_ctrl_pktize_write` (`:3035`) |
| `dfschedule.install_flow` | Install a data flow at runtime: dispatch its `ctrl_regdiff_manifest` as control writes over the persistent plane. | ctrl_instance, manifest sym; returns `!dfschedule.flow_handle` | sequence of `__Runtime_ctrl_push_target` |
| `dfschedule.teardown_flow` | Disable an installed flow's ports via control writes. | consumes `!dfschedule.flow_handle` | sequence of `__Runtime_ctrl_push_target` |
| `dfschedule.ctrl_read` | Readback-verify a config register (layered-growth safety check). | ctrl_instance, tile_addr; returns read words | `__Runtime_ctrl_read_target` (`:3815`) |

**New types:**
- `!dfschedule.ctrl_instance` — wraps the `__Runtime_CtrlInstance` send context
  (shim col, dest tile, stream id, DMA channels/BD, response buffer).
- `!dfschedule.flow_handle` — an installed-flow token; produced by
  `install_flow`, consumed by `teardown_flow` to make lifetime explicit in the
  IR and enable use-after-teardown checking.

These are justified by the inventory in §2: the executable schedule dialect today
has **no** op that issues a runtime control write or installs/tears down a flow.

## 5. Allocator change: port-level persistent reservation vs the ADR

| Aspect | ADR (config-only) | This design (persistent + dynamic) |
|--------|-------------------|-------------------------------------|
| Spine lifetime | Temporary; **released** after bootstrap | **Permanent**; pinned for program lifetime |
| Reservation granularity needed | none new (release restores pool) | **port-level, persistent** (extends tile-level `reserveTile`, `ResourceManager.cpp:487`) |
| Data-plane change | compile-time-static | **runtime** install/teardown |
| Port budget cost | none permanent | fwd loses idx 4, ret loses idx 3 for good |
| Concurrency | control & data never overlap | control plane stands **while** data runs |

## 6. End-to-end example

### 6.1 Building the control plane in IR

The snippets below are **illustrative** MLIR — for the *new* ops the exact
assembly formats are TBD at TableGen time; the *existing* ops follow their real
formats: `routingcreatedataio` (`routingop.td:22-35`, `$iotype , $direction ->
type`), `routingcreatetilearray` (`:8-20`, `$rownum $colnum : type type -> type`,
**no commas**), and `routingcontrolwrite` (`:66-86`, `$io , $tilearray : types
attr-dict -> type`). Note every routing op returns `I32:$output`, so the abstract
`install_data_flow` returns an `i32` handle here — the typed
`!dfschedule.flow_handle` only appears **after lowering** (snippet B).

**(A) Routing-dialect input IR** — what the frontend emits; only routing ops.

```mlir
// The reach set: the tiles the plane may address; its shim gateway is col 0.
// iotype="control" selects the CTRL sink path (StreamType::CONTROL).
%ctrl_io = routing.routingcreatedataio "control", "input" -> !routing.dataio
%reach   = routing.routingcreatetilearray %rows %cols : i32 i32
             -> !routing.tilearray

// Declare the FOREVER spine on the high vertical ports (fwd 4 / ret 3). Marking
// it lifetime=persistent tells routinghw to (a) PIN those ports out of the
// dynamic pool and (b) STATICALLY wire the spine into routing.cc (never
// released) — §4.0 / §4.2. Persistent counterpart of routingcontrolwrite.
%plane = routing.control_plane %ctrl_io, %reach
             : !routing.dataio, !routing.tilearray {
  spine_fwd   = 4 : i32,               // RT_CTRL_VFWD  (aie_runtime.c:3375)
  spine_ret   = 3 : i32,               // RT_CTRL_VRET  (aie_runtime.c:3376)
  gateway_col = 0 : i32,               // shim column that drives the spine
  broadcast   = true,                  // multicast reach (reuses CONTROL fan-out)
  lifetime    = #routing.lifetime<persistent>
} -> i32

// Install a DYNAMIC data flow (@flowA, an ordinary data flow declared elsewhere)
// over the plane. Its ports come from the pool that EXCLUDES the pinned spine.
%flowA = routing.install_data_flow @flowA {
           lifetime = #routing.lifetime<dynamic>
         } -> i32
// ... run the data plane against @flowA ...
// Tear it down by symbol ref; ports return to the dynamic pool (never spine).
routing.teardown_data_flow @flowA { lifetime = #routing.lifetime<dynamic> }
```

**(B) After lowering** — the persistent spine wiring is emitted statically by
routinghw into `routing.cc`; the runtime pieces become dfschedule ops driving the
control C API (§4.4).

```mlir
// Host send context for the (already statically-wired) spine: alloc response
// buffer + arm shim S2MM drain — NOT the route (that is in routing.cc). §4.4.
%ci = dfschedule.ctrl_instance_init { gateway_col = 0 : i32 }
        -> !dfschedule.ctrl_instance

// @flowA's stream-switch config, captured at blueprint time (§4.3), is dispatched
// as control writes over the persistent plane; install_flow returns the typed
// installed-flow token.
%h = dfschedule.install_flow %ci, @flowA_diff
       : (!dfschedule.ctrl_instance) -> !dfschedule.flow_handle
// Optional readback-verify a config register before relying on the flow.
%v = dfschedule.ctrl_read %ci, 0x0001D000 : (!dfschedule.ctrl_instance) -> i32
// ... run the data plane ...
dfschedule.teardown_flow %h : !dfschedule.flow_handle
```

Region-scoped variant (optional): `routing.control_plane` may instead carry a
body region so that `install_data_flow`/`teardown_data_flow` ops are lexically
scoped to the plane they target — making the "flows live inside this plane"
relationship explicit and enabling verification that no dynamic flow escapes a
plane it was not installed on.

### 6.2 Lowering trace (IR → runtime C API)

```
routing.control_plane {spine_fwd=4, spine_ret=3, lifetime=persistent}
        │
        ├─(compile-time, routinghw)──────────────────────────────────────────┐
        │   Phase A: pin spine ports (port-level persistent reservation)      │
        │   routinghw.ctrl_spine ──► routing.cc: XAie_StrmConnCctEnable × N    │
        │   (static spine wiring, emitted once at init — NOT a runtime op)     │
        │                                                                     ◄┘
        ├─(runtime, dfschedule)
        ▼
dfschedule.ctrl_instance_init  ──► response-buffer alloc + rt_tct_s2mm_arm
        │  → %ci : !dfschedule.ctrl_instance   (host send context only; route already wired)
        ▼
routing.install_data_flow(@flowA)             (Phase B: allocate from pool minus pinned)
        │
        ▼
dfscheblueprint.ctrl_regdiff_manifest @flowA_diff = [(tile_addr,val), ...]
        │
        ▼
dfschedule.install_flow %ci, @flowA_diff       ──► __Runtime_ctrl_push_target × N
        │   → %h : !dfschedule.flow_handle       (each = __Runtime_ctrl_pktize_write + push)
        │
        │   [optional] dfschedule.ctrl_read %ci, tile_addr  ──► __Runtime_ctrl_read_target
        │                                          (verify the writes landed)
        ▼
   ... run data plane (existing schedule.start_io / launch_kernel_group / wait) ...
        │
        ▼
routing.teardown_data_flow(@flowA)
        ▼
dfschedule.teardown_flow %h                    ──► __Runtime_ctrl_push_target × N (disable diff)
        │   (return flowA ports to dynamic pool; spine stays pinned)
        ▼
   ... optionally install_flow @flowB, reusing the same persistent %ci ...
```

## 7. Risks

1. **Shadow-state desync.** The aie-rt driver keeps its own register shadow; a
   control-packet write bypasses it. After a runtime `ctrl_write`/`install_flow`,
   the driver's shadow no longer matches hardware. Mitigation: treat the control
   plane as the single writer for the registers it owns, or explicitly resync the
   shadow after each write batch.
2. **Self-clobber ordering.** Because the persistent spine shares the fabric,
   reconfiguring a flow that overlaps the delivery path can cut the path
   mid-write. Deliver register diffs **downstream-first** (farthest tile first)
   so the spine to the tile being reconfigured is never torn out from under an
   in-flight write.
3. **Silent no-ack multicast.** A broadcast/multicast control write has no
   per-target ack. Mitigation: pair risky installs with
   `dfschedule.ctrl_read` (→ `__Runtime_ctrl_read_target`, `aie_runtime.c:3815`)
   as a layered-growth safety check before relying on the new flow.
4. **Permanent port-budget loss.** Pinning the spine costs fwd idx 4 and ret idx
   3 for the whole program (`RT_CTRL_VFWD=4`/`RT_CTRL_VRET=3`,
   `aie_runtime.c:3375-3376`). Dense data topologies that needed those indices
   must be re-routed by the two-phase allocator, or fail allocation — a
   deliberate trade vs the ADR's release model.

## 8. Cross-references

- ADR (config-only, released spine):
  [`2026-09-03-control-vs-data-network-sharing-adr.md`](2026-09-03-control-vs-data-network-sharing-adr.md)
- Control-packet delivery to any tile / row / col (multicast engine reuse):
  [`2026-09-03-control-packet-anytile-multicast-design.md`](2026-09-03-control-packet-anytile-multicast-design.md)
- Bootstrapping the `routing.cc` network via control-packet register writes
  (layered growth, port-asymmetry ordering):
  [`2026-09-03-control-network-bootstrap-config-design.md`](2026-09-03-control-network-bootstrap-config-design.md)
