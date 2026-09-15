# Design: Bootstrapping the `routing.cc` Network via Control Packets

**Status:** design only (no code changes in this doc)
**Date:** 2026-09-03
**Scope (Q2):** Instead of the host issuing hundreds of memory-mapped
`XAie_Strm*` register writes to build the network described by `routing.cc`, send
those same register writes as **control-packet WRITE payloads** over a minimal,
pre-established control network. This matches production CDO / control-code style
and works when the host only has shim-DMA reach into the array. Reuse the routing
engine as the source of truth; do not break the existing XAie-call emitter.

Prerequisite reading: `docs/plans/2026-09-03-control-packet-anytile-multicast-design.md`
(Q1 — the CONTROL multicast primitive this doc builds on).

---

## 1. Register-diff model

Every network-config call in `routing.cc` reduces to a small set of
stream-switch config-register writes `(tile_addr, value)`:

- `XAie_StrmConnCctEnable(dev, loc, sdir, sport, mdir, mport)` — a circuit
  connection ⇒ writes the master/slave mux-select config register(s) for that
  port pair.
- `XAie_StrmPktSwSlaveSlotEnable` / `XAie_StrmPktSwSlavePortEnable` /
  `XAie_StrmPktSwMstrPortEnable` — packet-switch arming ⇒ writes the slot-config,
  slave-port-enable (bit[31] of `0x3F100+port*4`, per the inline note at
  `routinghwlower.cpp:275`), and master-port config registers.

Concretely, `aout/worklocal/routing.cc:8-45` is a list of such calls; each maps
to a handful of 32-bit config writes at a tile-local byte address.

**Extraction options (design choice deferred):**
1. **Record backend** — run a thin `XAie_Write32` shim that logs
   `(loc, offset, value)` while executing the existing `routing()` function once
   on host, capturing the exact diff the driver would apply. Lowest risk: the
   values are whatever aie-rt actually computes.
2. **Precomputed offsets** — derive `(offset, value)` directly from the aie-rt
   SW-config register layout for the target gen. Faster at runtime, but must
   track per-gen register maps (see §5 risks).

Each captured `(tile_addr, value)` is then packetized with
`__Runtime_ctrl_pktize_write(out, cap, stream_id, tile_addr, &value, 1,
lastwriteack, ret_sid, &resp)` (`aie_runtime.c:3035`). Multiple consecutive
register writes to the same tile pack into one buffer (up to 4 words per access,
per the 128-bit-boundary chunking in `rt_ctrl_chunk_words`,
`aie_runtime.c:3104`).

---

## 2. Bootstrap: the chicken-and-egg and its solution

**The problem:** to control-write tile T's config registers, a shim→T CTRL stream
must already exist. But that stream is itself made of the config writes we are
trying to deliver.

**The solution — layered growth:**

- **(a) Minimal mem-mapped spine.** The host mem-maps only a minimal
  circuit-switched control **spine** to reach a first ring of gateway tiles'
  CTRL ports — exactly what `rt_ctrl_route_setup_col` (`aie_runtime.c:3454`) does
  today: a straight vertical NORTH climb from the shim, terminating at a tile's
  CTRL master (`XAie_StrmConnCctEnable(dev, dst, SOUTH, vfwd, CTRL, 0)`,
  `aie_runtime.c:3507`). This is the only part that uses memory-mapped writes.

- **(b) Control-packet growth.** Control packets sent over that spine write the
  SW-config registers of the **next** tiles (enabling their slave/master ports),
  extending control reach outward, tile by tile, until the full `routing.cc`
  network is configured. Two natural orders:
  - **Column-spine-first:** one CONTROL multicast per column writes each column's
    vertical-connect registers (bring every column's spine up), then per-row
    multicasts fill in the horizontal connections.
  - **Spine then rings:** bring up the column spine, then write the first ring of
    tiles, then use those tiles as new gateways for the next ring.

The mem-mapped spine is small and fixed; the bulk of the network is then built
in-fabric by control packets.

---

## 3. Port-asymmetry ordering rule (the crux)

The port budget is asymmetric (see Q1 §6 / `hwresource.cpp:128-173`) and the
control spine must not clobber the very port the in-flight control packet is
riding.

**Direction → port bank (matches the runtime constants):**

- **Forward / up:** NORTH-master + SOUTH-slave, indices **0-5**
  (`RT_CTRL_VFWD = 4`, `aie_runtime.c:3375`). Memtile budget is 6 here
  (`hwresource.cpp:153` N-Master 6, `:159` S-Slave 6).
- **Return / down:** SOUTH-master + NORTH-slave, indices **0-3**
  (`RT_CTRL_VRET = 3`, `aie_runtime.c:3376`). Memtile S-Master 4
  (`hwresource.cpp:154`), N-Slave 4 (`hwresource.cpp:158`).
- **Horizontal EAST/WEST:** indices **0-3** (core E/W 4, `hwresource.cpp:131-132,
  136-137`; memtiles do not route horizontally, E/W = 0, `:155-156,160-161`).

**Rules to avoid self-clobbering:**

1. **Reserve a dedicated high port for the control spine.** Pin the spine to the
   runtime's `RT_CTRL_VFWD = 4` (forward) and `RT_CTRL_VRET = 3` (return). These
   indices are within the 6/6 (fwd) and 4/4 (ret) budgets and are held by the
   bootstrap for the whole config phase, so config-plane writes never touch the
   spine's own ports. Config writes may freely program ports 0-3/0-5 *other than*
   the reserved spine indices.

2. **Order writes downstream-first along the flow direction.** A tile's ports are
   only (re)written *after* the control packet has already left that tile, so
   reprogramming a port never yanks the path out from under an in-flight packet.
   For an upward spine this means: write the farthest (top) tile first, then work
   back toward the shim.

3. **Horizontal spine has only 0-3.** East/West offers no spare high port to
   reserve, so config and data compete on the same 4 ports. Therefore schedule
   **row (horizontal) config only after the column spine is fully up**, and
   document the reduced horizontal fan-out budget: a horizontal control multicast
   can occupy at most 4 E/W ports per tile, so wide rows may need multiple passes
   or multiple source columns.

---

## 4. Reuse, don't break: an alternate emitter

The routing engine stays the **single source of truth** for the target network.
Q2 adds an **alternate emitter** beside the existing XAie-call emitter in
`routinghwlower.cpp` (`ConnectStreamSingleSwitchPortpattern` = circuit,
`ConnectStreamPktSwitchPortpattern` = packet, lines 124-347):

- **Existing emitter (unchanged):** lowers each connection op to
  `XAie_Strm*` EmitC calls → `routing.cc` → memory-mapped writes on host.
- **New emitter (config-packet):** lowers the same connection ops to
  `(tile_addr, value)` register-diffs, packetized as control-packet WRITE
  payloads, delivered over the CONTROL multicast primitive from Q1.

The mem-mapped path remains the **fallback and the bootstrap spine** — it is not
removed. A build flag selects which emitter runs (or both, for a spine + packet
split).

---

## 5. Risks / open items

1. **Per-gen SW-config register offsets.** The exact stream-switch config-register
   offsets differ across gens (AIE2PS vs AIE-ML). The record backend (§1 option 1)
   sidesteps this; the precomputed-offset path (option 2) must carry a per-gen map
   and be validated against aie-rt.

2. **Parity / beats packing for multi-register writes.** `__Runtime_ctrl_pktize_write`
   packs up to 4 words per access and never crosses a 128-bit boundary
   (`rt_ctrl_chunk_words`, `aie_runtime.c:3104`); each control-info word carries
   beats-1 in bits[21:20] and odd parity in bit[31] (`aie_runtime.c:3053-3055`).
   Config-register diffs must be grouped so each access stays within a tile's
   contiguous, boundary-aligned register block.

3. **Silent failures on no-ack multicast.** A write-only multicast (`block = 0`,
   no S2MM drain) gives no confirmation that every tile accepted the write, so a
   config error is silent. **Mitigation:** after each layer, do a same-column
   readback of a sentinel config register with
   `__Runtime_ctrl_read_target(...)` (`aie_runtime.c`, response drained on shim
   S2MM) to verify the layer took effect before growing outward.

4. **Driver shadow-state interaction.** aie-rt keeps internal shadow state for
   some registers. If both emitters run (mem-mapped bootstrap + control-packet
   growth), control-packet writes bypass the driver's shadow, so any later
   mem-mapped call that reads-modifies-writes could use stale shadow values.
   Design must either (a) route *all* config for a given tile through one emitter,
   or (b) resynchronize the driver shadow after control-packet writes.

5. **Ordering correctness.** The downstream-first rule (§3.2) must be enforced by
   the emitter's traversal order, not left to chance; the engine already knows the
   ordered path points (`ParseTheCCTRoutingPath` `orderedPathPoints`,
   `passdmaphoptoroutinghw.cpp:525`), which can be walked in reverse for the
   config-packet emitter.

---

## 6. Relationship to Q1

Q1 gives the delivery primitive (one multicast control packet to any
tile/row/col). Q2 uses that primitive to deliver **the network's own config
register writes**, bootstrapped from a minimal mem-mapped spine. Q1's port-budget
and stream-id reuse arguments (Q1 §6, §8) apply directly: the config-plane
control flows are additive and allocated by the same `ResourceManager`, and the
reserved spine ports (fwd 4 / ret 3) keep the bootstrap channel disjoint from the
config writes it carries.

---

## 7. File references (verified, no edits)

| Ref | What |
|-----|------|
| `aout/worklocal/routing.cc:8-45` | the `XAie_Strm*` calls to be re-expressed as register diffs |
| `routinghwlower.cpp:124-347` | packet emitter (alternate-emitter insertion point) |
| `routinghwlower.cpp:275` | slave-port-enable = bit[31] of `0x3F100+port*4` (a config diff) |
| `routinghwlower.cpp:349+` | `ConnectStreamSingleSwitchPortpattern` (circuit emitter) |
| `passdmaphoptoroutinghw.cpp:525` | `orderedPathPoints` — traversal order for downstream-first |
| `hwresource.cpp:128-173` | port templates: fwd 6/6, ret 4/4, horizontal 4/4 |
| `aie_runtime.c:3035` | `__Runtime_ctrl_pktize_write` (register-diff packetizer) |
| `aie_runtime.c:3053-3055` | ctrl-info word: beats-1 [21:20], odd parity [31] |
| `aie_runtime.c:3104` | `rt_ctrl_chunk_words` — 128-bit-boundary access chunking |
| `aie_runtime.c:3375-3376` | `RT_CTRL_VFWD = 4`, `RT_CTRL_VRET = 3` (reserved spine ports) |
| `aie_runtime.c:3454` | `rt_ctrl_route_setup_col` — minimal mem-mapped bootstrap spine |
| `aie_runtime.c:3507` | dest `SOUTH,vfwd → CTRL,0` termination |
| `aie_runtime.c:3735` | `__Runtime_ctrl_setup_routing` same-column guard (readback path) |

**See also:** `docs/plans/2026-09-03-control-packet-anytile-multicast-design.md`.
