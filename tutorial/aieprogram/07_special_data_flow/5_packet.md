# 7.5 Special data flow: packet-switched shared stream (ID/mask, one port)

> **Data flow: packet merge (gather).** Several cores send their results on ONE
> stream port. Each core's DMA puts a packet header with its own packet ID in
> front of the data; each stream switch slot matches the ID with a mask and
> forwards the packet to the shared master port. Four streams, one port, one
> shim channel. This is the core->shim mirror of [7.1 broadcast](1_broadcast.md).

> **Status.** Doc only: no new example. The routing shown is the flowcheck
> fixture `script/test/flowcheck_data/good/routing.cc` (an example fixture, not
> the 1a output). All refs are checked against base `43fcb8f` by
> script/check_refs.sh. Every cycle number in section 5 is **derived, not
> measured**.
>
> This is the gather the team asked about; s12dma scatter is not on main.

## The whole program

There is no kernel line for this flow: the compiler picks packet merge for a
core->shim path with several producer cores (section 4). The part of the
program that matters is the packet-switched block of the fixture's routing:
four DMA slots with IDs 1..4, one catch-all WEST slot per hop, and one
master port per tile
(`script/test/flowcheck_data/good/routing.cc:79-100`):
<!-- ref script/test/flowcheck_data/good/routing.cc:79-100 XAie_StrmPktSwSlaveSlotEnable -->

```cpp
XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(0,3), DMA, 0, 0, XAie_PacketInit(1, 0), 31, 0, 0);
XAie_StrmPktSwSlavePortEnable(v1, XAie_TileLoc(0,3), DMA, 0);
XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(0,3), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1,3), WEST, 1, 0, XAie_PacketInit(0, 0), 0, 0, 0);
XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(1,3), DMA, 0, 0, XAie_PacketInit(2, 0), 31, 0, 0);
XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1,3), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
...                                       // (2,3) ID 3, (3,3) ID 4
XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(3,0), 1);
XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3,3), SOUTH, 1, XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
XAie_StrmConnCctEnable(v1, XAie_TileLoc(3,2), NORTH, 1, SOUTH, 1);   // then (3,1), (3,0)
```
<!-- ref script/test/flowcheck_data/good/routing.cc:79 XAie_PacketInit(1, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:97 XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3,3), SOUTH, 1 -->

## 1. The idea: N sources, one port, the ID says who sent it

| Flow | sources -> destinations | what is shared | cost shape |
|---|---|---|---|
| 7.1 broadcast | 1 shim -> N cores | one stream, read by all | lockstep: slowest consumer |
| 7.5 packet merge | N cores -> 1 shim | one stream port, written in turns | port time-sharing: N x (B + 1) words |

A circuit route gives each core its own port all the way to the shim; a
packet route lets the four cores in row 3 share one EAST 1 / SOUTH 1 port and
one shim S2MM channel. The price is a 1-word header per packet and turn-taking
on the port (section 5). The ID in the header is the only thing that tells the
consumer which core a packet came from.

## 2. Data path (hardware)

The routing used here is an **example fixture, not the 1a output**: `script/test/flowcheck_data/good/routing.cc`
and `script/test/flowcheck_data/good/host.cc` (base 43fcb8f). It shows one packet-switched
stream shared by four cores in row 3. The packet header (ID and mask) is set in
routing.cc. host.cc only gives the DMA channel and BD numbers; it has no packet tokens.

Four core tiles, (0,3), (1,3), (2,3) and (3,3), send their results out of core MM2S
channel 0, BD 4. <!-- ref script/test/flowcheck_data/good/host.cc:27-30 channel_id=0, bd_id=4, tile=(0,3), direction=MM2S -->
All four streams merge onto one east-going packet-switched port in row 3, turn south at
column 3, and land in shim (3,0) S2MM channel 0, BD 2.
<!-- ref script/test/flowcheck_data/good/host.cc:26 channel_id=0, bd_id=2, tile=(3,0), direction=S2MM -->

```
 core(0,3)        core(1,3)        core(2,3)        core(3,3)
 MM2S ch0 bd4     MM2S ch0 bd4     MM2S ch0 bd4     MM2S ch0 bd4
 hdr id=1         hdr id=2         hdr id=3         hdr id=4
    |                |                |                |
 [DMA0 slot]      [DMA0 slot]      [DMA0 slot]      [DMA0 slot]
 [            ]   [WEST1 slot]     [WEST1 slot]     [WEST1 slot]
    |  EAST1 ------> |  EAST1 ------> |  EAST1 ------> |
    |                |                |                | SOUTH1
                                                    (3,2) NORTH1->SOUTH1  (circuit)
                                                    (3,1) NORTH1->SOUTH1  (circuit)
                                                    (3,0) NORTH1->SOUTH1  (circuit)
                                                    shim (3,0) S2MM ch0 bd2
```

Hop by hop:

1. **(0,3)**: the slave slot on the DMA 0 port matches packet ID 1 with mask 31, and the
   port is enabled in packet mode. <!-- ref script/test/flowcheck_data/good/routing.cc:79-80 XAie_PacketInit(1, 0), 31 -->
   Master EAST 1 is enabled in packet mode and keeps the header (`XAIE_SS_PKT_DONOT_DROP_HEADER`),
   so the next switch can still route by ID. <!-- ref script/test/flowcheck_data/good/routing.cc:81 XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(0,3), EAST, 1, XAIE_SS_PKT_DONOT_DROP_HEADER -->
2. **(1,3)**: two slave ports feed the same master. WEST 1 carries the traffic from (0,3)
   and has a catch-all slot (ID 0, mask 0). <!-- ref script/test/flowcheck_data/good/routing.cc:82-83 XAie_PacketInit(0, 0), 0 -->
   DMA 0 adds the local core with ID 2, mask 31. <!-- ref script/test/flowcheck_data/good/routing.cc:84-85 XAie_PacketInit(2, 0), 31 -->
   Both go to master EAST 1, so this is where two cores merge onto one port.
   <!-- ref script/test/flowcheck_data/good/routing.cc:86 XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(1,3), EAST, 1 -->
3. **(2,3)**: the same pattern. WEST 1 is catch-all and DMA 0 adds ID 3; output goes to EAST 1.
   <!-- ref script/test/flowcheck_data/good/routing.cc:87-91 XAie_PacketInit(3, 0), 31 -->
4. **(3,3)**: WEST 1 is catch-all and DMA 0 adds ID 4. <!-- ref script/test/flowcheck_data/good/routing.cc:92-95 XAie_PacketInit(4, 0), 31 -->
   The merged stream leaves on master SOUTH 1 instead of EAST.
   <!-- ref script/test/flowcheck_data/good/routing.cc:97 XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3,3), SOUTH, 1 -->
5. **Column 3, going down**: (3,2), (3,1) and (3,0) are plain circuit connections, NORTH 1 to SOUTH 1.
   <!-- ref script/test/flowcheck_data/good/routing.cc:98-100 NORTH, 1, SOUTH, 1 -->
   Shim stream port 1 is connected to the shim DMA, so the stream reaches shim S2MM.
   <!-- ref script/test/flowcheck_data/good/routing.cc:96 XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(3,0), 1) -->

Resources used: 4 packet-switched slave slots for the DMA ports and 3 for WEST 1; one
EAST 1 master per tile in columns 0 to 2; SOUTH 1 at (3,3); one shim S2MM channel. Only one
physical port carries all four streams across row 3, instead of four parallel circuits.

## 3. BD / packet-ID sequence

The match rule for a slave slot is `(incoming_PktId & Mask) == (slot_PktId & Mask)`.
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:26 (incoming_PktId & Mask) == (slot_PktId & Mask) -->
Mask 31 (0x1F, all five ID bits) is an exact match. <!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:28 this is an **exact equality** check -->
Mask 0 matches any ID, which is why the WEST 1 pass-through slots use `XAie_PacketInit(0, 0)` with mask 0.

| # | Source tile / DMA (host.cc) | BD | Packet ID in header (routing.cc slot) | Mask | Enters merged stream at |
|---|---|---|---|---|---|
| 1 | (0,3) MM2S ch0 | 4 | 1 | 31 | (0,3) DMA 0 → EAST 1 |
| 2 | (1,3) MM2S ch0 | 4 | 2 | 31 | (1,3) DMA 0 → EAST 1 |
| 3 | (2,3) MM2S ch0 | 4 | 3 | 31 | (2,3) DMA 0 → EAST 1 |
| 4 | (3,3) MM2S ch0 | 4 | 4 | 31 | (3,3) DMA 0 → SOUTH 1 |
| – | shim (3,0) S2MM ch0 | 2 | receives IDs 1..4, header kept | – | sink |

<!-- ref script/test/flowcheck_data/good/routing.cc:79 XAie_PacketInit(1, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:84 XAie_PacketInit(2, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:89 XAie_PacketInit(3, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:94 XAie_PacketInit(4, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/host.cc:27-30 bd_id=4 -->
<!-- ref script/test/flowcheck_data/good/host.cc:26 channel_id=0, bd_id=2, tile=(3,0) -->

Order of events in one round:

1. Each core MM2S BD 4 must put the same packet ID in its header that routing.cc gives the
   DMA slot on its tile (1, 2, 3 or 4). The ID comes from routing.cc; host.cc only names the
   channel and BD.
2. The four BDs may start in any order. The EAST 1 masters arbitrate packet by packet, so
   packets from different cores are interleaved but never cut in the middle of a packet.
3. Shim S2MM BD 2 receives every packet with its header. The consumer uses the ID to know
   which core a packet came from.
4. Sync rule (no fixed numbers, since host.cc has comments only): each core waits on its
   own output buffer's lock #k (producer side) before BD 4 runs, and the host waits on the shim
   S2MM BD 2 completion before reading. No lock is shared between the four cores; they
   share only the stream port.

## 4. Compiler: how packet merge is chosen

This section mirrors the broadcast section of `1_broadcast.md`: broadcast is the
shim→core fan-out, packet merge is the core→shim fan-in. All code below is cited at
base `43fcb8f` and lives under `src/mlir/mlirfront/tilinglinalg/`.

### 4.1 The decision: `passdmaphoptoroutinghw`

`DmaphopPathConversionPattern` in
`src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp`
first decides the direction of a `dmaphop::create_path` (`isShimToCore`, see the
broadcast section). For the other direction it sets `isCoreToShim = !isShimToCore`.
When the path is core→shim **and** has producer core tiles, it calls
`GatherPktRoutingPathCreate`
(`src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:1156-1167`).
The comment above the call names this case processing_type 2 (core→shim, gather by
packet merge); the call itself is on line 1159 (line 1157 is only the comment). It
passes the allocated shim point, the `DataIO`, the path and `producerCoreTiles`, and
returns the packet-routing tile map `lastPkttilemap`. It is a consumer of the packet IDs: it does not allocate them; it
reads the `dmapktid` that `DmapToDmaphopPass` already put on each core port (section 6).
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:1159 lastPkttilemap = GatherPktRoutingPathCreate( -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:1156-1167 isCoreToShim && !producerCoreTiles.empty() -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:1157 core-to-shim packet routing (processing_type=2) -->

### 4.2 Three definitions of `GatherPktRoutingPathCreate`

At `43fcb8f` the function is defined three times:

| Definition | Status at 43fcb8f |
|---|---|
| `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:139` | compiled; inside the anonymous `namespace {` that starts at line 32, so it is local to this file. The comment above it says it was copied from `routinglower.cpp`. |
| `src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:13` | not compiled: inside the `/* ... */` block that runs from line 12 to 394. |
| `src/mlir/mlirfront/tilinglinalg/pass/routinglower.cpp:14` | compiled; this is the original copy. |

<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:139 GatherPktRoutingPathCreate -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:32 namespace { -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:12-13 GatherPktRoutingPathCreate -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routinglower.cpp:14 GatherPktRoutingPathCreate -->

All three source files are listed in the build
(`src/mlir/mlirfront/CMakeLists.txt:126-132`), so whether a copy counts depends on
the comment blocks above, not on the build list.
<!-- ref src/mlir/mlirfront/CMakeLists.txt:126-132 routinglower.cpp -->

There is a second caller in another pass:
`RoutingmovedatabyioConvert` in `routinglower.cpp` calls its own copy
(`src/mlir/mlirfront/tilinglinalg/pass/routinglower.cpp:1129`) and hands the result
to `ParseTheCCTRoutingPath` with `StreamType::FORWARDONLY` on the next line. This
section does not claim which of the two passes a given pipeline runs; the code only
shows that both call sites are compiled.
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routinglower.cpp:1129 auto lastPkttilemap = GatherPktRoutingPathCreate( -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routinglower.cpp:958 struct RoutingmovedatabyioConvert -->

### 4.3 Lowering of the routing op: `RoutingToDmapPass`

In `src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp`,
`RoutingmovedatabyioConvertdmap` starts with `processing_type = 0` (broadcast). It
switches to `processing_type = 2`, gather/packet merge, only when operand 0 is
defined by a `routing::routinggatherout` op; it then takes `extract_data` from that
op's operand 1
(`src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:970-975`).
This is the exact mirror of the broadcast case, which keeps type 0.
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:970-975 processing_type = 2; -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:970 0 broadcast //1 tbd //2 gather an packet merge -->

### 4.4 The packet ID in the BD: `dfschedule`

The packet header the merged stream carries is set per buffer descriptor.
`dfschedule::ConfigDmaBdOp` has a `packetIdOperand` next to the BD id, offset and
out-of-order BD id
(`src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:314-316`).
<!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:314-316 packetIdOperand -->
The builder example in the same file creates a shim BD with `packet_id = 10`,
`out_of_order_bd_id = -1`, `len = 1024` and `enable_packet = true`
(`src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:941-950`).
With `enable_packet` the DMA puts a packet header with this ID in front of the
data, which is what lets several cores share one stream port.
<!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:941-950 // packet_id (SSA Value) -->
<!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:941-950 // enable_packet -->

### 4.5 Dead code: do not use as a reference

- `ParseTheCCTRoutingPath` at
  `src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:284` is
  inside the same `/* ... */` block (12–394) as the third definition, so it is not
  compiled in that file. The live callers use the copies in
  `passdmaphoptoroutinghw.cpp` and `routinglower.cpp`.
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:284 ParseTheCCTRoutingPath -->

<!-- sym DmaphopPathConversionPattern -->
<!-- sym RoutingmovedatabyioConvertdmap -->

## 5. Cost (timing)

**Status of every number below: derived, not measured.** At base 43fcb8f there is
no measured packet-switched timeline in the repo. The matmul profile under
`data/profile/matmul` is one core lane on a circuit-switched route, so it is not
used here. Cycle counts are ASSUMED round numbers chosen to show the shape.

**What it costs — serialization on one port.** In a packet merge, the stream
switch lets N core MM2S streams share one output port
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp:970 0 broadcast //1 tbd //2 gather an packet merge -->
(mode 2 in `src/mlir/mlirfront/tilinglinalg/pass/passroutingtodmap/routingtodmap.cpp`).
The port carries one packet at a time, so the N streams take turns: with N = 4
cores each sending B words, the port is busy for about 4 x (B + 1) words, where
the +1 is the packet header the DMA prepends to each packet
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:49 DMA prepends packet header {PktId=0, PktType=0} -->.
At the ASSUMED 1 word/cycle port rate and B = 64, that is 4 x 65 = 260 cycles
for the merged port versus 65 cycles for one core alone (derived, not measured).
Each core's MM2S waits while the others own the port; that wait is stream
backpressure, not a lock wait.

**Contrast with 7.1 broadcast.** Broadcast is 1 source to N destinations: one
stream feeds four cores at once and the cost is lockstep (the slowest consumer
holds the buffer, seen as `LOCK_STALL`). Packet merge is N sources to 1
destination: the cost is time-sharing of the port (4x the port time, plus one
header per packet), seen as stream stalls on the senders.

**4-to-1 timeline (illustrative, ASSUMED cycles, derived, not measured):**
```
cycle      0      65     130    195    260
port   :  |H c0..|H c1..|H c2..|H c3..|     H = packet header (1 word)
core0  :  |send  |wait (backpressure)  |
core1  :  |wait  |send  |wait          |
core2  :  |wait         |send  |wait   |
core3  :  |wait                |send   |
```

## 6. Lifecycle: who allocates, owns and frees

Sequence for one launch: **load -> allocate -> configure -> run -> sync -> release**.

| Resource | Allocated by | Owned by (while live) | Freed |
|---|---|---|---|
| packet IDs 1..4 for this stream (1..16 for the whole fixture, section 8) | `DmapToDmaphopPass`: `router.getRM()->allocatePktId(-1)` for each core `corePortOut`, Pull flows only (`src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:261`). This is ONE counter for the whole design, not one per stream, so IDs are unique across all streams (stream A gets 1..4, B 5..8, C 9..12, D 13..16); two streams that end on the same shim tile are kept apart by using different S2MM channels, not by reusing IDs; pool assert "max 31" at `:262`; stored as the port's `dmapktid` attribute at `:263` (passed positionally at `:267`). Push flows get no ID (`basePktId = 0`) | the `dmapktid` attribute on the `dmaphop.port`; fixed at compile time | never: `ResourceMgr::releasePktId` (`src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:920`) has no caller at 43fcb8f |
| slave slots (DMA 0 with ID k, mask 31; WEST 1 catch-all, mask 0) | routing pass; emitted as `XAie_StrmPktSwSlaveSlotEnable` in routing.cc | routing.cc, for the launch (`script/test/flowcheck_data/good/routing.cc:79-95`) | nothing is freed per launch |
| shared master port (EAST 1 in cols 0-2, SOUTH 1 at (3,3)) | routing pass; `XAie_StrmPktSwMstrPortEnable` in routing.cc | routing.cc, for the launch (`script/test/flowcheck_data/good/routing.cc:81`, `:97`); the four cores take it packet by packet | nothing is freed per launch |
| core MM2S BD 4 with packet_id = k | compiler (`dfschedule::ConfigDmaBdOp` packetIdOperand, `src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:314-316`) | the tile DMA; issued once per round. **Rule: BD packet_id must equal the slot ID** (section 3, pitfall 1) | nothing is freed per launch |
| shim S2MM ch0 BD 2 (sink) | compiler; listed in host.cc (`script/test/flowcheck_data/good/host.cc:26`) | the shim DMA, then the host after it waits for completion | nothing is freed per launch |
| teardown of slots / master port | **not in repo**: no slot-disable or port-disable call for this flow at 43fcb8f | - | not in repo |

<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:261 allocatePktId -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:262 pkt_id pool exhausted (max 31) -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:263 pktIdAttr = rewriter.getI32IntegerAttr -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:267 pktIdAttr -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/routingimplement/hw/ResourceManager.cpp:920 releasePktId -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:469 basePktId = 0; // Push flows are circuit-switched -->
<!-- ref script/test/flowcheck_data/good/routing.cc:79-95 XAie_StrmPktSwSlaveSlotEnable -->
<!-- ref script/test/flowcheck_data/good/routing.cc:81 XAie_StrmPktSwMstrPortEnable -->
<!-- ref script/test/flowcheck_data/good/routing.cc:97 XAie_StrmPktSwMstrPortEnable -->
<!-- ref src/mlir/mlirfront/tilinglinalg/dataflowmap/dfschedule/dfschedulemanager.cpp:314-316 packetIdOperand -->
<!-- ref script/test/flowcheck_data/good/host.cc:26 bd_id=2, tile=(3,0) -->

Rules that follow from the table:

- IDs, slots and the master port are all compile-time; the host allocates
  nothing for this flow and frees nothing per launch.
- The four cores share no lock; they share only the stream port. Each core's
  sync is on its own buffer (lock #k, by role only).
- The one runtime invariant is ID equality: BD packet_id == slot ID. Both come from
  ONE attribute, `dmapktid`, written once by `allocatePktId`. Routing reads it
  (`src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:222-231`,
  fallback 1 at `:224`) and so does the BD side
  (`src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:148`,
  used at `:312`). Note: `git grep dmapktid` finds no writer, because the writer passes
  `pktIdAttr` positionally; search for `allocatePktId` instead.
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:222-231 getDmapktid -->
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp:224 int pktId = 1; // fallback: 1-based -->
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:148 getDmapktid -->
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:312 getBasePktIdFromProducers -->

## 7. Pitfalls

### Pitfall 1: 0-based BD packet ID versus 1-based routing ID
This fixture has already failed once. The routing engine gives packet IDs starting at 1, but
the compiler (BlueprintToSchedulePass) wrote BD packet IDs starting at 0. With mask 31 no slot
matched, the switch held back TREADY and all four DMA channels stalled with MM2S stream
backpressure. <!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:15 Packet ID Mismatch Between host.cc and routing.cc -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:34-35 1 (DMA slot, Mask=31) -->
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:39-41 This off-by-one mismatch caused all 4 DMA channels to stall -->
Check: for each tile, the BD packet ID must equal the routing.cc DMA slot ID (1-based). If a
packet-switched flow hangs with MM2S backpressure, compare these two numbers before anything else.

### Pitfall 2: an unmatched packet ID looks like backpressure, forever
 The switch
matches `(incoming_PktId & Mask) == (slot_PktId & Mask)`
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:24-28 XAie_StrmPktSwSlaveSlotEnable -->;
with Mask=31 that is exact equality. If no slot matches the header the DMA sent,
the switch deasserts TREADY and the DMA raises `DMA_MM2S_STREAM_BACKPRESSURE`
(Event ID 148)
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:15-20 DMA_MM2S_STREAM_BACKPRESSURE -->.
In the recorded case all 4 core MM2S channels stalled with `STALL_STREAM=1` and
the host hung at `__Runtime_wait`
<!-- ref src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md:7-9 STALL_STREAM=1 -->
(see `src/mlir/mlirfront/tilinglinalg/pass/docs/streamdebug.md`). So the same
event means two things: normal turn-taking (it clears after a few hundred
cycles) or an ID mismatch (it never clears and the shim S2MM shows starvation).
Check the BD `pkt_id` against the slot PktId before reading it as a cost.

## 8. The whole fixture: four merged streams, IDs 1-16

Sections 1-7 follow stream A (row 3). The fixture has four such merged streams,
one per round, in rows 3, 4, 5 and 6. Each one is the same pattern: four core
MM2S ch0 BD 4 sources, each with its own DMA 0 slot (mask 31, exact match), one
packet-switched path east along the row, then circuit hops down to a shim S2MM
channel. The sink of each stream is read from the route (which shim tile and port
the path leaves at), not assumed from the order in host.cc. All refs are at base
43fcb8f.

| Stream | Row | Packet IDs (DMA 0 slots, mask 31) | Packet masters, west to east | Circuit hops to the shim | Shim port | Sink (host.cc) |
|---|---|---|---|---|---|---|
| A | 3 | (0,3) XAie_PacketInit(1, 0)<br>(1,3) XAie_PacketInit(2, 0)<br>(2,3) XAie_PacketInit(3, 0)<br>(3,3) XAie_PacketInit(4, 0) | EAST 1, EAST 1, EAST 1, SOUTH 1 | col 3 down: (3,2) (3,1) (3,0), N1->S1 | (3,0) port 1 | host.cc:26 (3,0) S2MM ch0 bd2 |
| B | 4 | (0,4) XAie_PacketInit(5, 0)<br>(1,4) XAie_PacketInit(6, 0)<br>(2,4) XAie_PacketInit(7, 0)<br>(3,4) XAie_PacketInit(8, 0) | EAST 0, EAST 1, EAST 1, SOUTH 1 | col 3 down: (3,3) N1->S2, (3,2) (3,1) N2->S2, (3,0) N2->S3 | (3,0) port 3 | host.cc:36 (3,0) S2MM ch1 bd7 |
| C | 5 | (0,5) XAie_PacketInit(9, 0)<br>(1,5) XAie_PacketInit(10, 0)<br>(2,5) XAie_PacketInit(11, 0)<br>(3,5) XAie_PacketInit(12, 0) | EAST 1, EAST 1, EAST 1, SOUTH 1 | (3,4) N1->S2, (3,3) N2->W2, then col 2 down: (2,3) E2->S2, (2,2) (2,1) N2->S2, (2,0) N2->S1 | (2,0) port 1 | host.cc:46 (2,0) S2MM ch0 bd3 |
| D | 6 | (0,6) XAie_PacketInit(13, 0)<br>(1,6) XAie_PacketInit(14, 0)<br>(2,6) XAie_PacketInit(15, 0)<br>(3,6) XAie_PacketInit(16, 0) | EAST 1, EAST 1, EAST 1, SOUTH 1 | (3,5) N1->S2, (3,4) N2->W2, (2,4) E2->S1, then col 2 down: (2,3) N1->S3, (2,2) (2,1) (2,0) N3->S3 | (2,0) port 3 | host.cc:56 (2,0) S2MM ch1 bd8 |

<!-- ref script/test/flowcheck_data/good/routing.cc:79 XAie_PacketInit(1, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:84 XAie_PacketInit(2, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:89 XAie_PacketInit(3, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:94 XAie_PacketInit(4, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:96 XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(3,0), 1) -->
<!-- ref script/test/flowcheck_data/good/host.cc:26 channel_id=0, bd_id=2, tile=(3,0) -->
<!-- ref script/test/flowcheck_data/good/routing.cc:118 XAie_PacketInit(5, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:123 XAie_PacketInit(6, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:128 XAie_PacketInit(7, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:133 XAie_PacketInit(8, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:135 XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(3,0), 3) -->
<!-- ref script/test/flowcheck_data/good/host.cc:36 channel_id=1, bd_id=7, tile=(3,0) -->
<!-- ref script/test/flowcheck_data/good/routing.cc:160 XAie_PacketInit(9, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:165 XAie_PacketInit(10, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:170 XAie_PacketInit(11, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:175 XAie_PacketInit(12, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:177 XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(2,0), 1) -->
<!-- ref script/test/flowcheck_data/good/host.cc:46 channel_id=0, bd_id=3, tile=(2,0) -->
<!-- ref script/test/flowcheck_data/good/routing.cc:206 XAie_PacketInit(13, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:211 XAie_PacketInit(14, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:216 XAie_PacketInit(15, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:221 XAie_PacketInit(16, 0), 31 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:223 XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(2,0), 3) -->
<!-- ref script/test/flowcheck_data/good/host.cc:56 channel_id=1, bd_id=8, tile=(2,0) -->

```
 row 6  D: ids 13 14 15 16 --E1--E1--E1--> (3,6) S1 -+
 row 5  C: ids  9 10 11 12 --E1--E1--E1--> (3,5) S1 -+-> (3,5) N1->S2 [D]
 row 4  B: ids  5  6  7  8 --E0--E1--E1--> (3,4) S1  |   (3,4) N1->S2 [C]; N2->W2 [D] -> (2,4) E2->S1 [D]
 row 3  A: ids  1  2  3  4 --E1--E1--E1--> (3,3) S1  |   (3,3) N1->S2 [B]; N2->W2 [C] -> (2,3) E2->S2 [C]
                                                         (2,3) N1->S3 [D]
 column 3 down to shim (3,0):                column 2 down to shim (2,0):
   A: N1->S1 ... (3,0) S1 -> port 1            C: N2->S2 ... (2,0) N2->S1 -> port 1
   B: N2->S2 ... (3,0) N2->S3 -> port 3        D: N3->S3 ... (2,0) N3->S3 -> port 3

 sinks:  A -> host.cc:26  (3,0) S2MM ch0 bd2     C -> host.cc:46  (2,0) S2MM ch0 bd3
         B -> host.cc:36  (3,0) S2MM ch1 bd7     D -> host.cc:56  (2,0) S2MM ch1 bd8
```

<!-- ref script/test/flowcheck_data/good/routing.cc:120 XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(0,4), EAST, 0 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:137 NORTH, 1, SOUTH, 2 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:140 NORTH, 2, SOUTH, 3 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:180 NORTH, 2, WEST, 2 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:181 EAST, 2, SOUTH, 2 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:184 NORTH, 2, SOUTH, 1 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:226 NORTH, 2, WEST, 2 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:227 EAST, 2, SOUTH, 1 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:228 NORTH, 1, SOUTH, 3 -->
<!-- ref script/test/flowcheck_data/good/routing.cc:231 NORTH, 3, SOUTH, 3 -->

What the four streams show together:

- **IDs are unique across the whole design.** `allocatePktId` is one counter for
  the design (section 6), so stream A gets 1-4, B 5-8, C 9-12 and D 13-16. No two
  DMA 0 slots in the fixture share an ID, so a header always names exactly one core.
  <!-- ref src/mlir/mlirfront/tilinglinalg/pass/passdmaptodmaphop/dmaptodmaphop.cpp:261 allocatePktId -->
- **Two streams per shim tile, one S2MM channel each.** A and B both end at shim
  (3,0), on ports 1 and 3 and S2MM ch0 / ch1; C and D both end at shim (2,0), again
  on ports 1 and 3 and S2MM ch0 / ch1. The streams never merge with each other: each
  keeps its own circuit path and its own shim channel.
- **(0,4) is the one head on EAST 0.** At (0,3), (0,5) and (0,6) master EAST 0 is
  taken by the circuit broadcast of the same round, so those heads use EAST 1.
  <!-- ref script/test/flowcheck_data/good/routing.cc:72 XAie_TileLoc(0,3), SOUTH, 1, EAST, 0 -->
  <!-- ref script/test/flowcheck_data/good/routing.cc:153 XAie_TileLoc(0,5), SOUTH, 1, EAST, 0 -->
  <!-- ref script/test/flowcheck_data/good/routing.cc:199 XAie_TileLoc(0,6), SOUTH, 1, EAST, 0 -->
  Stream B's next hop therefore listens on WEST 0 at (1,4), not WEST 1.
  <!-- ref script/test/flowcheck_data/good/routing.cc:121 XAie_TileLoc(1,4), WEST, 0, 0, XAie_PacketInit(0, 0) -->
- **Pitfall 1 applies per stream.** Each core's BD 4 packet_id must equal its own
  slot ID (1..16 above), not its position in the row (1..4).
