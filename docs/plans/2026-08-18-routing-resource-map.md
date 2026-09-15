# Plan: Routing resource map — JSON → header → app → runtime print

## Goal
Add a routing-IR pass that emits a **resource JSON** describing which tile / which stream
port / pkt id / **pkt mask** each routing connection uses. Convert that JSON into a C
header, compile the header into the app, and have the app **load and print** the resource
info from runtime init.

Four deliverables, mirroring the four parts of the request:
1. New MLIR pass `RoutingResourceMapPass` → `worklocal/routingresourcemap.json`
2. Python converter → `worklocal/aie_resource_map.h`
3. Header compiled into the app (already on `-I${WORKLOCAL_DIR}`, invoked from `aiehlc.sh`)
4. Runtime-init load + print in `aie_runtime.c`

---

## Part 1 — New pass `RoutingResourceMapPass`

**Model on** the existing `passroutingprovenancemap/passroutingprovenancemap.{h,cpp}`
(reuse the self-contained `JsonWriter`, `getIntAttr`/`getStrAttr`, `resolveTile`,
`writeTileRef`, `writePort`). This pass differs in one important way: it **derives the
pkt mask** (which the provenance pass omits), so the JSON is a complete resource record.

New files:
- `src/mlir/mlirfront/tilinglinalg/pass/passroutingresourcemap/passroutingresourcemap.h`
- `src/mlir/mlirfront/tilinglinalg/pass/passroutingresourcemap/passroutingresourcemap.cpp`

`.h`: copy the provenance header, rename class → `RoutingResourceMapPass`,
`getArgument()` → `"routing-resource-map"`, keep the 4 constructors
`(), (outputDir), (outputDir,startCol), (outputDir,startCol,aieGen)` and the same
`getDependentDialects` (routinghw, routing, func, arith, scf).

`.cpp`: walk `routing::RoutingCreate` groups exactly as the provenance pass does; for each
`routinghw` connection op emit a resource record. For the packet-switch op
`ConnectStreamPktSwitchPort` emit tile + `recv_slave{dir,idx,pktid,pkttype,mask}` +
`local_dma{dir,idx,pktid,pkttype,mask}` + `forward_master{dir,idx}` + `preserve_header`.
**Mask derivation** (matches `routinghwlower.cpp` EmitC): recv/forward slave slot
`mask = 0x0` (forward-all), local-DMA slave slot `mask = 0x1f` (exact 5-bit match).
Also handle the same circuit/shim ops the provenance pass handles (`kind:"circuit_*"`,
shim enable ops) so the resource map is complete, but those carry no pkt mask/id.
Output file: `outputDir + "/routingresourcemap.json"`.

**Mask derivation is verified** against `routinghwlower.cpp`:
- `routinghwlower.cpp:237` — recv/forward slave slot: `mask = 0` (forward-all)
- `routinghwlower.cpp:243` — local-DMA slave slot: `dmamask = 0x1f` (exact 5-bit match)

**Register in** `src/mlir/mlirfront/CMakeLists.txt`:
- add `.../passroutingresourcemap/passroutingresourcemap.cpp` to `SOURCE_LIB_FILES`
  (after line 141, the provenance source)
- add `include_directories(./tilinglinalg/pass/passroutingresourcemap/)`
  (after line 202, the provenance include dir)

**Wire into** `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp` right
after the existing `RoutingProvenanceMapPass` block (closes at line 1540), running on
`routingDmaphopModule` (must run BEFORE `RoutingHWLowerPass` at line 1542, while the
routinghw connection ops still exist):
```cpp
{
    auto routingResourcePass = std::make_unique<RoutingResourceMapPass>(outputDir, partStartCol, aieGen);
    runPipelineSinglePass(ctx, routingDmaphopModule, std::move(routingResourcePass), routingIrDir, rstage, "RoutingResourceMapPass");
}
```
Add `#include "passroutingresourcemap.h"` alongside the provenance include (line 18).

### JSON shape (one entry per connection)
```json
{
  "partition_start_col": 3,
  "aie_gen": "5",
  "connections": [
    { "kind": "packet_connect",
      "tile": {"col": 0, "row": 2, "kind": "core"},
      "recv_slave":  {"dir": "SOUTH", "idx": 0, "pktid": 1, "pkttype": 0, "mask": 0},
      "local_dma":   {"dir": "DMA",   "idx": 0, "pktid": 1, "pkttype": 0, "mask": 31},
      "forward_master": {"dir": "NORTH", "idx": 0},
      "preserve_header": false }
  ]
}
```

---

## Part 2 — JSON → C header converter

New file: `src/tool/debug/resource_json_to_header.py`
- CLI: `resource_json_to_header.py <routingresourcemap.json> --out <aie_resource_map.h>`
- Emits a self-contained header:
  - `#ifndef AIE_RESOURCE_MAP_H` guard
  - a `struct AieResourceEntry` (col,row,kind, recv dir/idx/pktid/mask, dma dir/idx/pktid/mask, fwd dir/idx, preserve)
  - `static const struct AieResourceEntry __aie_resource_map[]` table + count
  - `static inline void __Runtime_print_resource_map(void)` that iterates the table and
    prints each entry (tile, stream port, pkt id, pkt mask) — plain `printf`, no deps.
- Deterministic output (stable ordering) so rebuilds are diff-clean.

---

## Part 3 — Compile the header into the app

The header lands in `worklocal/`, which `hostcompile.sh` already puts on the include path
(`INCLUDE_OPTS ... -I${WORKLOCAL_DIR}`, line 338). Only need to invoke the converter.

**Invoke from** `script/aiehlc.sh` tiling branch, before the `hostcompile.sh` call
(around line 528, next to where user `*.h` are copied into `WORKLOCAL_DIR` and
`app_source.txt` is written), **non-fatal** (mirror the `schedule_view.py` precedent at
lines 556-563): if `worklocal/routingresourcemap.json` exists, run the converter to
produce `worklocal/aie_resource_map.h`; on any failure, warn and continue (never change
build exit code).

---

## Part 4 — Runtime-init load + print

**Edit** `src/mlir/runtime/aie_runtime.c`:
- Near the top, guarded include so the raw/single-kernel flow (no header) still compiles
  (aie_runtime.c is built with g++ -std=c++17, so `__has_include` is available):
  ```c
  #if defined(__has_include)
  #  if __has_include("aie_resource_map.h")
  #    include "aie_resource_map.h"
  #    define AIE_HAVE_RESOURCE_MAP 1
  #  endif
  #endif
  ```
- In `__Runtime_routing_init(XAie_DevInst *dev)` (line 1411), after `routing(dev);`:
  ```c
  #ifdef AIE_HAVE_RESOURCE_MAP
      __Runtime_print_resource_map();
  #endif
  ```
Compiles to nothing when the header is absent.

---

## Files created / modified
- **new** `src/mlir/mlirfront/tilinglinalg/pass/passroutingresourcemap/passroutingresourcemap.h`
- **new** `src/mlir/mlirfront/tilinglinalg/pass/passroutingresourcemap/passroutingresourcemap.cpp`
- **new** `src/tool/debug/resource_json_to_header.py`
- **edit** `src/mlir/mlirfront/CMakeLists.txt` (register pass source + include dir)
- **edit** `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp` (include + wire pass)
- **edit** `script/aiehlc.sh` (invoke converter, non-fatal, tiling branch)
- **edit** `src/mlir/runtime/aie_runtime.c` (guarded include + print call)

---

## Verification
1. **Rebuild aiehlc**: `cd build && make -j$(nproc)` — confirm the new pass compiles/links.
2. **Run the tiling flow**:
   ```
   source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
   ```
   - Confirm `aout/worklocal/routingresourcemap.json` is produced and contains
     tile/port/pktid/**mask** entries.
   - Confirm `aout/worklocal/aie_resource_map.h` is generated and compiles into the host
     (no build error from hostcompile.sh).
3. **Standalone converter test**: run `resource_json_to_header.py` on the generated JSON,
   diff the header, sanity-check the struct + table + print function.
4. **Runtime print**: run on board/sim
   (`python3 script/test/apppaltest.py -y -nonreboot`) and confirm the resource-map lines
   appear in `applog` at init.
5. **Raw single-kernel flow still builds** (no `aie_resource_map.h`): confirm the
   `#if __has_include` guard makes it a no-op.
