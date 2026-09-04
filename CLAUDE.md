## Role
- MLIR and Spatial computing expert

# AIEHLC Project Architecture

AIEHLC (AIE High-Level Compiler) is a compilation/deployment solution for AMD Versal AI Engine. It has **two parts**:

1. **aiehlc** — Clang-based tool that compiles C++ (AIE driver C API) into host + kernel binaries for single-kernel AIE apps.
2. **tilinglinalg** — MLIR progressive lowering pipeline that offloads a GEMM operation across multiple AIE tiles, generating host, kernel, and routing C++ code through 6 custom dialects.

Both share the AIE runtime (`include/aie_runtime.h`) and target Versal AI Core Series with pre-built PDI.

## Directory Layout

```
aiehlc/
├── include/                   # aie_runtime.h, aie_device_map.h
├── src/
│   ├── llvm/aiehlc.cc         # Main aiehlc Clang tool
│   └── mlir/
│       ├── runtime/aie_runtime.c  # Runtime wrappers over XAie_* APIs
│       └── mlirfront/
│           ├── AieFrontEnd.cc     # Clang AST → MLIR
│           ├── AieDialect.cc      # AIE dialect (LoadKernel, etc.)
│           └── tilinglinalg/      # ★ Multi-tile GEMM pipeline
│               ├── routing/       # Abstract routing dialect
│               ├── routinghw/     # Physical routing dialect
│               ├── dataflowmap/   # dmap, dmaphop, dfscheblueprint, dfschedule dialects
│               └── pass/          # All lowering passes + unitest/
├── script/                    # setup.sh, aiehlc.sh, kc.sh, test/apppaltest.py
├── example/                   # AIE examples (perf, matmul, multi-kernel)
└── thirdparty/alib/           # XAie driver (excluded from this doc)
```

## Part 1: aiehlc (Single Kernel)

Compiles a user C++ file into a host ELF (ARM) + kernel ELF (AIE core):

```
User C++ → aiehlc (Clang AST) → AieFrontEnd (MLIR)
  → Kernel: xchesscc → xchessmk → kernel ELF (embedded in host via ld -r -b binary)
  → Host: aarch64-g++ (host.cc + aie_runtime.c + routing + kernel.o) → host ELF
```

Key runtime API: `__Runtime_device_init`, `__Runtime_load_kernel_group`, `__Runtime_launch_kernel_group`, `__Runtime_dma_bd_config`, `__Runtime_wait_event`, `__Runtime_device_teardown`.

Control-packet API: `__Runtime_ctrl_pktize_write(out, cap, stream_id, tile_addr, data, nwords, lastwriteack, ret_sid, &resp_words)` (build WRITE control-packet words; when `lastwriteack` and `nwords>0`, the LAST written word is re-emitted as a single-word WRITE-WITH-RETURN access — op=0b10, return stream id [28:24] — whose header-only response acks all preceding writes, so `resp_words`=1), `__Runtime_ctrl_pktize_read(out, cap, req_sid, ret_sid, tile_addr, nwords, &resp_words)` (build READ control-packet words per doc/controlpkt.txt: op=01, byte addr [19:0], beats-1 [21:20], return stream id [28:24], odd parity [31]; no data payload; reports the expected response word count — now counting, per access, 1 stream header word + its data words, since the return-route master KEEPS each response packet's stream header via XAIE_SS_PKT_DONOT_DROP_HEADER; the per-access header is stripped host-side in `rt_ctrl_read_extract`). `__Runtime_ctrl_push(inst, buf, nwords, block, log)` pushes a packet buffer via a SHIM MM2S BD; the send context — dev, shim col, bd_id, mm2s_ch — comes from a `__Runtime_CtrlInstance *`, with `buf`/`nwords` the per-call payload; `block!=0` also waits for the response drain via `__Runtime_ctrl_tct_poll`. The return path is the **control-packet response**: the dest CTRL slave port emits the response (read data / write-with-return ack), circuit-switched down to the shim S2MM (the drain landing `resp_words` words is the completion barrier — no DMA TCT token issue). The send context is a `__Runtime_CtrlInstance` struct (shim col, dest tile, stream id, DMA channels/BD, `resp_words`, internal response buffer `token`); `__Runtime_ctrl_setup_routing(inst)` programs the shim→dest CTRL forward route + dest CTRL slave→shim S2MM return route, allocates the response buffer (`resp_words` words), and arms the shim S2MM; `__Runtime_ctrl_tct_poll(inst, print)` polls the S2MM drain and returns the first response word. `__Runtime_ctrl_read_target(dev, shim_col, dest_col, dest_row, req_sid, ret_sid, tile_addr, nwords, out_data, bd_id, mm2s_ch, s2mm_ch)` composes it all for a register read: build read packets → setup_routing → ctrl_push → tct_poll → extract data words into `out_data` (the return route keeps each response's stream header, so `rt_ctrl_read_extract` skips the per-access header word when copying out the data). Single shim S2MM drain BD ⇒ one response packet (a ≤4-word access not crossing a 128-bit boundary); larger reads need one BD per packet (not yet implemented). `__Runtime_ctrl_push_target` composes setup_routing → ctrl_push → tct_poll around one packet buffer. Two header parsers invert the pktize encoding: `__Runtime_ctrl_parse_ctrl_hdr(ctrl_hdr, &addr, &op, &beats, &ret_sid)` decodes a control-info word (addr [19:0], op [23:22] 00/01/10, beats = [21:20]+1 = 1..4, return sid [28:24]); `__Runtime_ctrl_parse_pkt_hdr(pkt_hdr, &id, &type, &src_row, &src_col)` decodes an AIE packet-switched stream header (id [4:0], type [14:12] with 7=SLVERR, source row [20:16], source col [27:21]). Any out-pointer may be NULL; both return 1 iff the odd-parity bit [31] over [30:0] is consistent. The debug provenance for a control-packet host.cc is produced statically by `xaiehost2provenance.py` (it recognizes `__Runtime_CtrlInstance` sends — shim + vertical pass-through + dest tiles and two flows per send), so `schedule_debug_server.py` can open a debug GUI without a `Work/` tree.

Platforms: baremetal (`aarch64-none-elf-g++`) or Linux (`aarch64-linux-gnu-g++`).

## Part 2: tilinglinalg (Multi-Tile GEMM)

### Six Custom MLIR Dialects

| Dialect | Purpose |
|---------|---------|
| **routing** | Abstract tile arrays, data IO, broadcast, mesh partitioning |
| **routinghw** | Physical tiles, stream switch ports, packet flows |
| **dmap** | Logical dataflow: ports, streams, push/pull |
| **dmaphop** | Physical hops: tile-to-tile paths, DMA, buffers |
| **dfscheblueprint** | Schedule blueprint: transfer manifests, flow configs |
| **dfschedule** | Executable schedule: DMA BD, kernel launch, locks |

### Pass Pipeline

**Shared stages** (produces dfscheblueprint IR, then module is cloned):

1. `RoutingUnrollingLowerPass` — unroll abstract routing into per-tile ops
2. `RoutingToDmapPass` — routing → logical dataflow
3. `DmapToDmaphopPass` — logical → physical hops
4. `DmaphopTodfscheblueprintPass` — hops → schedule blueprints

**Host path** → `host.cc`:

5. `BlueprintToSchedulePass` → `ScheduleCanonicalizePass` → `DfscheduleToApiPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `host.cc`

**Kernel path** → `kernel.cc`:

5. `BlueprintToScheduleKernelPass` → `DfscheduleToKernelApiPass` → EmitC → `kernel.cc`

**Routing path** (alternative, Path A) → `routing.cc`:

1. `RoutingUnrollingLowerPass` → `RoutingLowerPass` → `RoutingHWLowerPass` → `RoutingDeadArgPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `routing.cc`

### Routing Implementation (`pass/routingimplement/`)

- **RoutingTopology**: Gen2 AIE tile topology model
- **RoutingPath**: BFS path finding (priority: Memory > SHIM > Core)
- **ResourceManager**: Tracks link/port usage to avoid conflicts

## Build

```bash
# Main aiehlc binary
mkdir build && cd build && cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build && make -j$(nproc)

# TilingLinalg unitest (standalone)
 source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc
 python3 ./script/test/apppaltest.py -y  -nonreboot > ./applog 2>&1
```

Each dialect has `td/` (TableGen), `gen.sh` (runs mlir-tblgen), and `inc/` (generated .inc files).

## Test and Verification

### Unitest CLI

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
```

### End-to-End Flow

```
1. Generate    source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
2. HW run      python3 script/test/apppaltest.py aout/worklocal/build/host  → SSH+xsdb+console
3. Verify      script/test/verify_host.sh → pass: "device_teardown done", fail: "AIE ERROR"
```

### Per-Dialect Unit Tests

Each dialect has its own `unitest/` directory with independent CMake build:
- `routing/unitest/`, `routinghw/unitest/`
- `dataflowmap/{dmap,dmaphop,dfschedule,dfscheblueprint}/unitest/`
- `pass/routingimplement/{routing,hw}/unitest/`

## Additional Documentation

### Design docs

- **[doc/module_analysis.md](doc/module_analysis.md)** — 9-module project breakdown (M1–M9), key files, dependencies, data-flow diagram
- **[doc/aieapi.md](doc/aieapi.md)** — XAie driver API guide (single-tile, multi-tile manual, production AEG patterns)
- **[doc/tilinglinalg.md](doc/tilinglinalg.md)** — TilingLinalg deep dive: dialects, passes, routing engine, build/HW-run flow
- **[doc/lowering.md](doc/lowering.md)** — Concrete IR lowering trace with snippets and op-to-API mapping
- **[doc/debug/tutorial_aiehlc.md](doc/debug/tutorial_aiehlc.md)** — aiehlc simulator + debug UI (`--platform sim`, `--sim-only`)
- **[doc/debug/tutorial_baremetal.md](doc/debug/tutorial_baremetal.md)** — naiebaremetal VEK385 boot + debug UI
- **[doc/design/tile_dim_structured_design.md](doc/design/tile_dim_structured_design.md)** — Structured `tile_dim` for `aie::SpatialPolicy`
- **[doc/design/spatial_space_composition.md](doc/design/spatial_space_composition.md)** — Composition-based spatial op spaces (GemmSpace, Conv2dSpace)
- **[doc/design/aiegdb_live_debug_framework.md](doc/design/aiegdb_live_debug_framework.md)** — Live debug framework design (static view + daemon)
- **[doc/llvm_mlir_pitfalls.md](doc/llvm_mlir_pitfalls.md)** — Known LLVM/MLIR API pitfalls
- **[doc/aiedifferentview.md](doc/aiedifferentview.md)** — Memory and lock semantics

### Agent skills (`.cursor/skills/<name>/SKILL.md`)

Read the matching skill when the task fits:

| Topic | Skill |
|-------|-------|
| Debug UI, daemon, live session, browser UI features | **debug-ui-framework** (+ [reference.md](.cursor/skills/debug-ui-framework/reference.md)) |
| Embedded LLM context loss on retarget | debugui-llm-reset |
| Static XAie API verify (routing.cc, host.cc) | xaieapiverify |
| Routing debug (IR → generated code) | routinghwdebug |
| DMA BD verify in host.cc | dmabdverify |
| Pre-HW data correctness | datacorrectness |
| XAie driver internals | aiedriverkb |
| Live HW DMA stall debug | aiehwdmadebug |
| Shim BD stuck on wrong/locked BD (index overflow into channel-control regs) | shimbdindexoverflow |
| Sim PS.so load segfault | aiesimloaddebug |
| HW performance counters | aiehwprofile |
| Raw-XAie sim debug bundle | raw-xaie-sim-debug-bundle |
| Sim build/run separation | sim-build-run-separation |
| hostcompile / missing compile_kernel.sh | hostcompile-entrypoint |
| AEG IPC sim C++ headers | aeg-sim-cxx-headers |
| Host codegen | hostcodegen |
| Kernel codegen | kernelcodegen |

**Embedded LLM plugin** (`src/tool/debug/dbg_llm_skills/`): nine skills for the browser LLM tab. Launch locally with `claude --plugin-dir src/tool/debug/dbg_llm_skills`. Listed in **debug-ui-framework** reference.

**External:** aiedbg clone at `/scratch/staff/bkirinci/aiedbg` — see plugin skill `aiedbg-reference`.

**Command:** [data-mismatch-debug](.claude/commands/data-mismatch-debug.md) — systematic DMA data-mismatch triage.

### Browser automation (MCP)

`mcp-browser` is configured in `.mcp.json` (built at `thirdparty/mcp-browser/`). It gives Claude live Playwright-based browser control via MCP tools.

**When to use it:**
- Testing or verifying the debug UI (`schedule_debug_server.py` / `schedule_view.py`) in a real browser
- Reproducing UI bugs: navigate to the UI, screenshot, inspect element state
- Adding or debugging frontend features: navigate → interact → screenshot to confirm the feature works
- Checking that a server-side change is reflected in the rendered page without manual intervention

**Key tools:** `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_type`, `browser_extract_text`, `browser_wait_for_element`, `browser_execute_script`

**Typical debug UI workflow:**
1. Ensure the debug server is running (e.g., `python3 src/tool/debug/schedule_debug_server.py ...`)
2. `browser_navigate` to `http://localhost:<port>`
3. `browser_screenshot` to see current state
4. Interact and verify with `browser_click` / `browser_extract_text`

**Re-building after changes:**
```bash
cd thirdparty/mcp-browser && npm run build
```
Restart Claude Code (or reload MCP) to pick up the rebuilt server.

### Debug tools

- **[src/tool/debug/README.md](src/tool/debug/README.md)** — user guide and CLI for all debug tools
- **Skill: debug-ui-framework** — implementation map for `schedule_debug_server.py`, `schedule_view.py`, `aiegdb.py`, `aiemcp.py`, `aiediag.py`, `xaiehost2provenance.py` (detail in [reference.md](.cursor/skills/debug-ui-framework/reference.md))

### Build notes

- **[script/verify_env.sh](script/verify_env.sh)** — validate Vitis, LLVM, toolchain, board vars before build
- **[script/hostcompile.sh](script/hostcompile.sh)** — kernel build via `compile_one_kernel()` → `kc.sh`; do not restore deleted `compile_kernel.sh` (skill: hostcompile-entrypoint)
- **[script/aiehlc.sh](script/aiehlc.sh)** — `--platform sim` is build-only; launch sim separately via `runsim.sh` or debug UI **Run** (skills: sim-build-run-separation, raw-xaie-sim-debug-bundle)

## Key Terms

| Term | Definition |
|------|------------|
| **PDI** | Pre-built hardware design; decouples HW/SW development |
| **GMIO** | Global Memory I/O; DDR ↔ AIE via NoC |
| **Shim tile** | Row-0 tile bridging NoC/DDR and AIE array |
| **MemTile** | Large-memory tile for caching between DDR and compute tiles |
| **BD** | Buffer Descriptor; configures a DMA transfer |
| **DSKernel** | Data-streaming kernel (receives via DMA, computes, outputs via DMA) |
| **EmitC** | MLIR dialect for C/C++ emission; final stage before `translateToCpp` |
| **xchesscc** | Synopsys compiler for AIE cores (from Vitis) |
| **PAL** | Board environment for running ELFs on real AIE hardware |

## never do
- the api function > 200 lines
## Learn rule
- create skill when a issue fixed
- when do somthing wrong that is fix by user guide create related skill
## Document rule
- Maintain and update architecture doc and keep update after do some changes
## Process transperent rule
- after each task done, list all files that change or new created
