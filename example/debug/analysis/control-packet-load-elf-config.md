# Control-Packet Register-Write Support in Routing (Load ELF / Config)

Add stream-switch **control packet** support so `routing.cc` can route packets that
perform in-tile **register/memory writes** — the mechanism used by
`_XAie_LoadElfStrmSwFromMemPkt` / `XAie_LoadElfStrmSwPkt` in
`/proj/xsjsswstaff/gregoryw/kernel-loading/aie-rt/driver/src/core/xaie_elfloader.c`.

Use cases: **kernel-ELF load acceleration AND generic config writes** → one general
control-packet infrastructure.

---

## 1. How the reference works (`xaie_elfloader.c`)

`_XAie_CtrlPktizeElfPkt` (lines ~1180-1259) turns each ELF `PT_LOAD` segment into a
DDR buffer of 32-bit words, with **two headers per ≤4-word chunk**:

- **Stream packet header**: `bits[4:0]=stream id`, `bit31=odd parity`. The stream switch
  routes on this id. The reference encodes `StreamId = LocIdx<<2 | neighborBits` so a single
  DMA stream can target up to 8 tiles' local / N / S / E-W memories.
- **Control packet header**: `bits[19:0]=tile-local address`, `bits[21:20]=size-1`,
  `bit31=parity`. The tile's **CTRL** stream-switch port decodes this and writes `data` to
  that address.

The buffer is DMA'd from DDR through SHIM MM2S into the stream switch; the switch delivers
each stream id to the target tile's `CTRL` master port → register write. `CTRL` is a real
port type: `thirdparty/aie-rt/driver/src/global/xaiegbl.h:263`
(`StrmSwPortType { CORE, DMA, CTRL, ... }`).

---

## 2. What changes in `routing.cc`

Today `routing.cc` (`aout/worklocal/routing.cc`) only routes packets to the **`DMA`** master
port (S2MM data into a buffer). Control packets are identical plumbing but the destination
master port is **`CTRL`** and the header must be **preserved** (never `DROP_HEADER`, since the
CTRL decoder needs the control header).

A control-packet route to a target tile emits (per tile on the path):

```c
// incoming slave (from shim/neighbor) accepts the stream id
XAie_StrmPktSwSlaveSlotEnable(v1, Loc, <slavedir>, <idx>, slot,
                              XAie_PacketInit(pktid,0), mask, msel, arb);
XAie_StrmPktSwSlavePortEnable (v1, Loc, <slavedir>, <idx>);
// route into the tile control port instead of DMA
XAie_StrmPktSwMstrPortEnable  (v1, Loc, CTRL, 0, XAIE_SS_PKT_DONOT_DROP_HEADER, arb, mselEn);
// (intermediate tiles) forward to next tile master, header preserved
XAie_StrmPktSwMstrPortEnable  (v1, Loc, <fwddir>, <idx>, XAIE_SS_PKT_DONOT_DROP_HEADER, arb, mselEn);
```

Plus the shim external-input enable (`XAie_EnableShimDmaToAieStrmPort`) at the source,
exactly as done today for the input path. The host side then DMA-pushes the control-packet
buffer into that shim port.

The existing lowering (`routinghwlower.cpp`) already emits
`XAie_StrmPktSwMstrPortEnable(dev, loc, <masterdir>, ...)`. By making the local sink port be
`"CTRL"` and forcing `preserveheader=true`, **no new XAie API call is needed** — only the op
attributes change.

---

## 3. Routing model (targeting) & relationship to S2MM

Control packets are an **input-direction (shim→core) flow carrying config**, orthogonal to the
compute S2MM path. We **do NOT** convert existing S2MM (core→shim gather) routing to packet.
S2MM stays as-is. Control packets replace/augment (a) per-core `XAie_LoadElfMem` register
writes and (b) direct host register config. We reuse the packet engine that already exists for
the S2MM gather, but point it at `CTRL`.

Targeting modes (all expressible with the current path/fan-out engine in
`passdmaphoptoroutinghw.cpp`):

- **Single core (unicast)**: one path shim→core, unique packet/stream id; tile enables `CTRL`
  master with that id's slot. The *control header address* selects the register inside the tile.
- **Row / col / multi-core broadcast**: reuse the existing **fan-out** in
  `ParseTheCCTRoutingPath` (`MasterSendToNextTileDirection2`, secondary masters, lines
  ~619-646). A slave port fans to several master ports; every tile on the broadcast taps the
  stream into its `CTRL` port with the **same** packet id → identical register writes on all
  tiles (matches `XAie_LoadElfStrmSwBroadcast`).
- **Selective multicast (subset of a broadcast)**: give groups distinct stream ids and use the
  slave-slot **mask/msel** matching (already plumbed via `ConnectStreamPktSwitchPort`
  `receiveslavemask/msel`) so a tile only accepts its ids — this is the reference's
  `LocIdx<<2 | neighbor` scheme.

Routing shape: shim MM2S source → (optional mem-tile hop) → packet fan-out tree over the
target core tiles → each target routes to `CTRL`. Different-address / different-value writes to
different cores are handled by **payload content + stream id**, not by extra routes.

---

## 4. IR / op design: extend `routinghw` + `routing`, no new dialect

The control-packet path is ~90% the existing packet mechanism; a new dialect would duplicate
tile/port/packet modeling. Concretely:

1. **routinghw** (`routinghw/td/routinghwop.td`): add a control-destination selector to
   `ConnectStreamPktSwitchPort` instead of hard-coding `DMA`:
   - `StrAttr:$localsinkport` (default `"DMA"`, may be `"CTRL"`) — which local port the packet
     terminates into.
   Lowering (`routinghwlower.cpp` `ConnectStreamPktSwitchPortpattern`) uses this string for the
   local termination and forces `DONOT_DROP_HEADER` when `localsinkport=="CTRL"`. Minimal,
   backward compatible.

2. **routing** (`routing/td/routingop.td`): add a high-level intent op
   `routing.routingcontrolwrite` (abstract, like `routingcreatebroadcast`) with:
   - operand: target `tilearray` / tile-list (or a mesh partition), an `iotype="control"`.
   - attrs: `broadcast : bool`, packet-id policy.
   Expresses "deliver control writes to these tiles" without hardware detail.

3. **dmap / dmaphop / StreamType**: add a `CONTROL` variant to `StreamType`
   (`ResourceManager.h:117`) and mark the shim producer as a control source so
   `passdmaphoptoroutinghw.cpp` (`GetSeqPath` / `ParseTheCCTRoutingPath`) chooses `CTRL` as the
   local sink and always packet-mode + preserve-header. The path/fan-out logic is unchanged.

4. **Payload build (runtime C builder)** in `src/mlir/runtime/aie_runtime.c`: add
   `__Runtime_ctrl_pktize(...)` that, given (tile-address, value) pairs or an ELF, builds the
   packet buffer with the two-header format (mirroring `_XAie_CtrlPktizeElfPkt`), and
   `__Runtime_ctrl_push(dev, shim, buf)` that configures a SHIM MM2S BD (reuse existing
   `__Runtime_dma_bd_config`) and starts it. The compiler emits `routing.cc` (switch setup) +
   a small manifest (which tiles/ids); runtime fills the payload. Flexible for both ELF-load
   and generic config, and matches the driver.

---

## 5. Files to modify

| Area | File | Change |
|------|------|--------|
| HW op def | `src/mlir/mlirfront/tilinglinalg/routinghw/td/routinghwop.td` | add `$localsinkport` attr to `ConnectStreamPktSwitchPort`; regen `.inc` |
| HW lowering | `src/mlir/mlirfront/tilinglinalg/pass/routinghwlower.cpp` | route local packet to `CTRL` when `localsinkport=="CTRL"`; force `DONOT_DROP_HEADER` |
| Abstract op | `src/mlir/mlirfront/tilinglinalg/routing/td/routingop.td` | add `routing.routingcontrolwrite`; regen |
| Stream type | `.../pass/routingimplement/include/hw/ResourceManager.h` | add `StreamType::CONTROL` |
| dmaphop→hw | `.../pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp` | for CONTROL flow, set local sink=`CTRL`, packet mode, preserve header; reuse fan-out for broadcast |
| routing→dmap | `.../pass/passroutingtodmap/routingtodmap.cpp` | lower `routingcontrolwrite` to dmap control producer/consumers |
| Runtime | `src/mlir/runtime/aie_runtime.c` | `__Runtime_ctrl_pktize` + `__Runtime_ctrl_push`; optionally switch kernel load to control-packet |
| Pipeline | `.../pass/tilinglinalg_pipeline.cpp` | ensure control routes flow through the routing.cc emit path |

---

## 6. Implementation status (Batch 1 — HW core)

**Done and verified (compiles cleanly):**

1. `routinghw/td/routinghwop.td` — added
   `DefaultValuedStrAttr<StrAttr, "DMA">:$localsinkport` to `ConnectStreamPktSwitchPort`
   (backward compatible; assemblyFormat uses `attr-dict`).
2. Regenerated `routinghw/inc/routinghwop.{cc,h}.inc` via `mlir-tblgen`
   (getter `getLocalsinkport()` present). Note: `gen.sh` has a pre-existing arg-quoting quirk;
   ran `mlir-tblgen` directly:
   ```
   TBG=/scratch/staff/huaj/llvm-project/build/bin/mlir-tblgen
   INC=/scratch/staff/huaj/llvm-project/mlir/include
   $TBG -gen-op-defs  td/routinghwop.td -dialect=routinghw -Itd -I. -I$INC -o inc/routinghwop.cc.inc
   $TBG -gen-op-decls td/routinghwop.td -dialect=routinghw -Itd -I. -I$INC -o inc/routinghwop.h.inc
   ```
3. `pass/routinghwlower.cpp` — in `ConnectStreamPktSwitchPortpattern`:
   - read new attr into `localsinkportstr` / `isCtrlSink`;
   - when `isCtrlSink`, override the local master port to `"CTRL"` (idx 0) via
     `effMasterDirStr` / `effMasterIdx`;
   - force header preservation for CTRL:
     `headerPolicy = (isLastPktHop && !preserveHeader && !isCtrlSink) ? dropheader : nodropheader`.
   - Verified: `routinghwlower.cpp.o` compiles clean in the main `build/` tree.
4. `pass/unitest/test.cpp` — added self-contained `testControlPacketCtrlSink()` +
   `lowerCtrlSinkAndEmit()` helper and a `ctrlwrite` CLI subcommand. Builds a minimal routinghw
   module (textual IR parsed via `parseSourceString`), runs `RoutingHWLowerPass`, translates to
   C++ and asserts:
   - CTRL case → emits `XAie_StrmPktSwMstrPortEnable(..., CTRL, ..., XAIE_SS_PKT_DONOT_DROP_HEADER, ...)`, no `DROP_HEADER`.
   - DMA case → does not target `CTRL` (backward-compat).
   - `test.cpp.o` compiles clean.

**Blocker (pre-existing, unrelated to this feature):**
The unit-test `CMakeLists.txt` (`pass/unitest/CMakeLists.txt`) and the stale `utbuild/` cache
do **not** include `../passroutingresourcemap/`, so `tilinglinalg_pipeline.cpp:20`
(`#include "passroutingresourcemap.h"`) fails to compile in the unit-test build. This gap
predates this work (the include dir was never added when `passroutingresourcemap` joined the
pipeline). To run the `ctrlwrite` unit test, add:
```
include_directories(../passroutingresourcemap/)
# and, if link errors follow, add the .cpp to SOURCE_FILES:
${CMAKE_CURRENT_SOURCE_DIR}/../passroutingresourcemap/passroutingresourcemap.cpp
```
then reconfigure with a fresh `build/` (per `piplinerun.sh rebuild`).

---

## 7. Remaining work (Batch 2 & 3)

**Batch 2 (plumbing):**
- `ResourceManager.h` — add `StreamType::CONTROL`.
- `passdmaphoptoroutinghw.cpp` — for CONTROL flows set the `localsinkport="CTRL"` attr on the
  emitted `ConnectStreamPktSwitchPort`, force packet mode + preserve header; reuse fan-out for
  broadcast/multicast.
- `routing/td/routingop.td` — add `routingcontrolwrite`; regen.
- `passroutingtodmap/routingtodmap.cpp` — lower `routingcontrolwrite` to a dmap control
  producer/consumers.

**Batch 3 (runtime + pipeline + HW verify):**
- `aie_runtime.c` — `__Runtime_ctrl_pktize` + `__Runtime_ctrl_push`; opt-in kernel-ELF-load
  switch behind an env flag (e.g. `AIEHLC_SKIP_BSS`), keeping `XAie_LoadElfMem` as fallback.
- `tilinglinalg_pipeline.cpp` — ensure control routes flow through the routing.cc emit path.
- Full build + `source script/aiehlc.sh --aie-version 5 --runtime-source-file
  ./example/tileprogram/ccode/simplematmul.cc`.
- HW/sim: `python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1`;
  pass = `device_teardown done`, fail = `AIE ERROR`.

---

## 8. Verification checklist

1. **Build**: `mkdir build && cd build && cmake .. -DLLVM_INSTALL_DIR=... && make -j`.
2. **Generate**: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc`.
3. **Inspect `routing.cc`** (`aout/worklocal/routing.cc`): control routes emit
   `XAie_StrmPktSwMstrPortEnable(..., CTRL, 0, XAIE_SS_PKT_DONOT_DROP_HEADER, ...)` and matching
   slave slots; broadcast tiles share one packet id, unicast tiles have distinct ids.
4. **Static verify**: skills `xaieapiverify` / `routinghwdebug`
   (`.cursor/skills/routinghwdebug/references/op-mapping.md`).
5. **HW/sim run**: pass = `device_teardown done`; for ELF-load use case, verify cores execute
   (compute output correct) with control-packet load instead of `XAie_LoadElfMem`.
6. **Unit test**: `./test ctrlwrite` checks emitted `CTRL` master enable + preserve-header.

---

## 9. Open decisions (defaults taken)

- Payload built at **runtime** (recommended) — flexible for ELF + generic writes.
- **Extend routinghw/routing** (no new dialect).
- Keep existing S2MM path unchanged; control packets are a separate input-direction flow.
- Kernel-ELF-load switch to control-packet is **opt-in** (env flag, e.g. `AIEHLC_SKIP_BSS`),
  keeping `XAie_LoadElfMem` as fallback.

---

## Reference anchors

- `_XAie_CtrlPktizeElfPkt`, `_XAie_LoadElfStrmSwFromMemPkt`, `XAie_LoadElfStrmSwPkt`,
  `XAie_LoadElfStrmSwBroadcast` — `.../aie-rt/driver/src/core/xaie_elfloader.c`.
- `StrmSwPortType { CORE, DMA, CTRL, FIFO, ... }` — `thirdparty/aie-rt/driver/src/global/xaiegbl.h:263`.
- `ConnectStreamPktSwitchPort` op — `routinghw/td/routinghwop.td:170`.
- Master-port emit + header policy — `pass/routinghwlower.cpp` (`ConnectStreamPktSwitchPortpattern`).
- Fan-out / secondary masters — `passdmaphoptoroutinghw.cpp` `ParseTheCCTRoutingPath`.
- `StreamType` enum — `pass/routingimplement/include/hw/ResourceManager.h:117`.
