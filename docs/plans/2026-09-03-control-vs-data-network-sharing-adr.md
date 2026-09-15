# ADR: Share one stream-switch network for data + control (config/bootstrap only)

**Status:** Accepted
**Date:** 2026-09-03
**Scope:** Design decision only — no code changes.

## Context

The multi-tile GEMM pipeline (`tilinglinalg`) routes data flows across the AIE
array through a single physical stream-switch fabric, allocated by one
`ResourceManager`. We now also send **control packets** — register/config writes
that bootstrap the network before data runs.

The question: **do we reuse the `routing.cc` data network (one fabric, one
routing engine) for control-packet flows, or stand up a separate control stream
config with its own allocation?**

Two constraints frame the answer:

1. There is exactly **one** physical stream-switch fabric with a **scarce
   per-tile port budget** (e.g. Core = 4 ports/direction).
2. Control packets are **config/bootstrap only**: they configure the network,
   then hand off to the data plane. Control and data never need to run
   concurrently.

## Decision

**Reuse one shared stream-switch fabric and one routing engine for both data and
control-packet flows. Do NOT build a separate control network or a second
allocator.**

For the bootstrap phase, carve a **temporary reserved control spine** on fixed
high vertical ports — forward `RT_CTRL_VFWD=4` (up, NORTH-master / SOUTH-slave,
valid range 0–5) and return `RT_CTRL_VRET=3` (down, SOUTH-master / NORTH-slave,
valid range 0–3). Use the spine to control-write the data network's config
registers, then **release the spine ports** so the data plane reuses them.
Because control is config-only, there is no concurrent contention and no
permanent port budget is lost.

## Rationale

The following facts (verified against the repo) support the decision:

1. **There is only one physical fabric and one allocator.** A "separate network"
   is not physically realizable — it would be a hand-maintained duplicate
   fighting the same per-tile port budget without the engine's conflict tracking.
   All ports come from one `defaultPortTemplates()` set consumed by one
   `ResourceManager`.
   - `routingimplement/hw/hwresource.cpp:128-173` — single port-template set:
     Core 4/direction (`:128-138`); Mem North-Master 6, South-Slave 6, DMA 6
     (`:153,159,162`); NocShim South mux `{3,7}` / demux `{1,3}`
     (`:165,170`).

2. **The CTRL sink is already separate — for free.** `PortDirection` includes a
   `Control` value (`hwresource.h:42`), but `defaultPortTemplates()` gives
   `Control` **no budget** (it has no `PortTemplate` entry). So the CTRL master
   (idx 0) is a fixed terminal that `ResourceManager` never allocates — control
   *termination* never competes with data ports.
   - `routinghwlower.cpp:196` — `isCtrlSink` when `localsinkport == "CTRL"`.
   - `routinghwlower.cpp:303-304` — CTRL sink routes to CTRL master idx 0.
   - `routinghwlower.cpp:323-337` — CTRL sinks force header preservation
     (`nodropheader`) so the CTRL decoder can read the target address/size.

3. **Transport hops are exclusively owned → a real (temporary) reservation.**
   `allocate()` marks a port slot `used` for exactly one `ioId`; two flows can
   never share a port index. So the control spine's N/S/E/W/DMA hops occupy real
   data ports **while active**.
   - `ResourceManager.cpp:182-193` — `occupyport()` scans the bank.
   - `ResourceManager.cpp:195-210` — `allocate()` sets `used=true` + `ioId`
     (single-owner).

4. **Config-only ⇒ time-multiplex, not permanent reservation.** Control flows run
   first (bootstrap), then data flows are allocated afterward and may reuse the
   exact ports the spine used. No permanent budget loss; no runtime contention.

5. **Spine ports avoid self-clobbering.** During bootstrap the packet rewrites the
   ports it rides, so the spine must sit on reserved indices disjoint from the
   config writes it carries: vertical fwd 4 (0–5 range) and ret 3 (0–3 range),
   within the Mem North-Master 6 / South-Slave 6 and South-Master 4 / North-Slave
   4 budgets.
   - `hwresource.cpp:153` (North-Master 6), `:154` (South-Master 4),
     `:158` (North-Slave 4), `:159` (South-Slave 6).
   - `aie_runtime.c:3375-3376` — `RT_CTRL_VFWD=4` / `RT_CTRL_VRET=3`, with the
     port-asymmetry rationale (`:3370-3374`).

Control flows also reuse the existing BROADCAST fan-out engine rather than a new
code path: `ParseTheCCTRoutingPath` treats `StreamType::CONTROL` like BROADCAST,
terminating at each target's CTRL master with the header preserved
(`passdmaphoptoroutinghw.cpp:492-519`).

## What NOT to do

- Do **not** add a second `ResourceManager`/allocator or a parallel dialect for
  control routing — reuse `StreamType::CONTROL`, which already fans out via the
  BROADCAST engine (`passdmaphoptoroutinghw.cpp:492-519`).
- Do **not** permanently reserve control ports on cores (only 4/direction); the
  spine is released after bootstrap.

## Consequences

- **+** No duplicate allocator; conflict tracking stays in one place
  (`ResourceManager`).
- **+** CTRL sink is already free — it is not in the port budget, so control
  termination costs nothing.
- **+** Full data-plane port budget is restored after bootstrap (time-multiplex
  of the spine ports).
- **−** Bootstrap and data flows cannot overlap in time — acceptable because
  control is config-only.
- **−** Spine teardown/handoff must be **explicit** so the released ports are
  genuinely reusable by the data plane.

## Cross-references

- [2026-09-03-control-packet-anytile-multicast-design.md](2026-09-03-control-packet-anytile-multicast-design.md)
  — control-packet delivery to any tile / whole row / whole col via the reused
  BROADCAST multicast engine.
- [2026-09-03-control-network-bootstrap-config-design.md](2026-09-03-control-network-bootstrap-config-design.md)
  — bootstrapping the `routing.cc` network via control-packet register writes
  (layered growth, port-asymmetry ordering).
