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
│                   └── passblueprintlowering/  # host+kernel blueprint lowering
│                       ├── helper/             # SHARED core-tile DMA/lock config
│                       ├── passblueprinttoschedule/       # → host.cc
│                       └── passblueprinttoschedulekernel/ # → kernel.cc
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

**Control-packet plane** — register access over the stream fabric instead of the host
config bus. See **[doc/controlplane.md](doc/controlplane.md)** for the full API,
encoding, and fabric layout. In brief:

| Layer | Entry points |
|-------|--------------|
| Packet encode/parse | `__Runtime_ctrl_pktize_write` / `_pktize_read`, `_parse_ctrl_hdr` / `_parse_pkt_hdr` |
| Single-target send | `__Runtime_CtrlInstance` + `_setup_routing` / `_push` / `_tct_poll`; composed by `_read_target` / `_push_target` |
| Row/broadcast fabric | `__Runtime_ctrl_plan_init` (one-shot planner, `aie_runtime_control_plan.c`), then `_row_broadcast_write` / `_row_whole_row_write` / `_row_read` / `_row_write_ack` |
| Transaction capture | `__Runtime_control_start_transaction` → `_write_pkt_commit_transaction` → `_control_push` (WRITE-only ops → broadcast/multicast packet buffer) |
| Resource reservation | `aie_runtime_resource.{c,h}` — single source of truth for slots/arbiters/msel/pkt-ids; `RT_RES_*` constants aliased by the planner's `ACR_*` |

Key invariants worth knowing before touching it:

- Completion is the **control-packet response** draining at the shim S2MM
  (`resp_words` words), not a DMA TCT token.
- The return route KEEPS each response's stream header
  (`XAIE_SS_PKT_DONOT_DROP_HEADER`); it is stripped host-side in `rt_ctrl_read_extract`.
- Forward routing arms a **uniform slot superset** on every chain tile — all columns of
  a row respond; there is no last-tile special case.
- Provenance `dir=fwd|ret` comes from the planner's `acr_op.is_ret` tag, **not** the port
  type (both fabrics use the same ports). A slot and every master pulling it must share
  a `dir`, or the debug switch-detail view silently drops the link.
- `ResourceMgr::reserveControlPlaneResources` is **opt-in** via
  `#pragma control_plan_op_control_packet`; default off.

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

5. `BlueprintToSchedulePass` → `ScheduleCanonicalizePass` → (`GroupRegWritePass`, opt-in) → `DfscheduleToApiPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `host.cc`

`GroupRegWritePass` is **gated** on `routing.control_plan_group_reg_write`, published by
aiehlc only for `#pragma CONTROL_PLAN_GROUP_REG_WRITE`. Off (the default), per-tile lock
inits emit as individual `XAie_LockSetValue` writes instead of coalesced control-packet
group writes (`__Runtime_ctrl_row_write_ack`), and the pipeline logs
`GroupRegWritePass skipped`.

**Kernel path** → `kernel.cc`:

5. `BlueprintToScheduleKernelPass` → (`DfscheduleKernelAggregationPass`, offload-only) → `DfscheduleToKernelApiPass` → EmitC → `kernel.cc`

`DfscheduleKernelAggregationPass` (`pass/passdfschedulekernelaggregation/`, gated on
`routing.kernel_config_offload`) collapses the per-core-tile DMA config onto
`dfschedule.declaretile.self` — one BD group per window instead of one per tile
(96 → 6 `dma_bd`). `packet_id` / `out_of_order_bd_id` are **SSA operands, not
attributes**, and must stay `arith.constant`-foldable. See
**[doc/design/kernel_dma_aggregation.md](doc/design/kernel_dma_aggregation.md)** — it
covers why `declaretile.self` is its own op (fail-closed on `getDefiningOp`), why
`ooo_bd_id` needs the `ResourceMgr` singleton fallback on the kernel clone, and why
erasure must be a fixpoint sweep.

**KERNELCONFIGOFFLOAD** (gated on `routing.kernel_config_offload`, set by
`#pragma KERNELCONFIGOFFLOAD`, default off, Gen5 only): the core self-configures all of
its own core-tile DMA from `kernel.cc` via MMIO instead of the host programming it over
the config bus, using `src/mlir/runtime/aie_kernel_runtime.h` (encoders + `core_reg_*`
apply, the latter going through `kernel_tm.h`'s `TM_W` — a plain pointer cast never
reaches the processor bus). See
**[doc/design/kernel_config_offload.md](doc/design/kernel_config_offload.md)** for that
TM rule, the direction asymmetry, the cross-clone plan channel, and the BD-id reservation
contract — each one deadlocks or silently corrupts registers if broken.

**Pass layout.** Both blueprint-lowering passes live under `pass/passblueprintlowering/`, with the shared `helper/` hoisted **above** them (`passblueprintlowering/helper/`) rather than nested inside the host pass. The two paths describe the same physical core tiles — one BD bank, one lock array per tile — so whoever programs it must agree with whoever doesn't. Anything both paths must agree on belongs in `helper/`. See `passblueprintlowering/README.md`.

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
- **[doc/controlplane.md](doc/controlplane.md)** — Control-packet plane: encoding, `CtrlInstance` send/recv, row/broadcast fabric, transaction capture, resource reservation, provenance
- **[doc/performance/register_write_cost.md](doc/performance/register_write_cost.md)** — **Measured** host↔AIE register access cost: ~372 ns per 32-bit `XAie_Write32` (non-posted Device-nGnRnE NoC round trip; an N-word BD is N serial writes) vs ~2.9 ns via control packets. Cite this instead of deriving ns/write from a timeline span.
- **[doc/tilinglinalg.md](doc/tilinglinalg.md)** — TilingLinalg deep dive: dialects, passes, routing engine, build/HW-run flow
- **[doc/design/kernel_config_offload.md](doc/design/kernel_config_offload.md)** — `#pragma KERNELCONFIGOFFLOAD`: core self-configured DMA, lock asymmetry, BD-id reservation
- **[doc/design/kernel_dma_aggregation.md](doc/design/kernel_dma_aggregation.md)** — `DfscheduleKernelAggregationPass`, `declaretile.self`, `packet_id`/`ooo_bd_id` operands
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
| Core register write succeeds on-core but host reads 0 / DMA never starts (`ST` vs `ST.TM`) | coretmregisterwrite |
| Shim BD stuck on wrong/locked BD (index overflow into channel-control regs) | shimbdindexoverflow |
| Sim PS.so load segfault | aiesimloaddebug |
| aiesim live debug register socket | aiesim-debug-socket |
| HW performance counters | aiehwprofile |
| Raw-XAie sim debug bundle | raw-xaie-sim-debug-bundle |
| Sim build/run separation | sim-build-run-separation |
| Build fails on missing snap cmake / libz.so / ZLIB::ZLIB; fast single-file compile check | mlirbuildsandbox |
| Gen2-only build break: missing `xpseudo_asm_armclang.h`, or `XPAR_CPU_TIMESTAMP_CLK_FREQ` undeclared | bspheadergen2 |
| hostcompile / missing compile_kernel.sh | hostcompile-entrypoint |
| AEG IPC sim C++ headers | aeg-sim-cxx-headers |
| Host codegen | hostcodegen |
| Kernel codegen | kernelcodegen |

**Embedded LLM plugin** (`src/tool/debug/dbg_llm_skills/`): nine skills for the browser LLM tab. Launch locally with `claude --plugin-dir src/tool/debug/dbg_llm_skills`. Listed in **debug-ui-framework** reference.

**External:** aiedbg clone at `/scratch/staff/bkirinci/aiedbg` — see plugin skill `aiedbg-reference`.

**Command:** [data-mismatch-debug](.claude/commands/data-mismatch-debug.md) — systematic DMA data-mismatch triage.

### Browser automation (MCP)

`mcp-browser` (configured in `.mcp.json`, built at `thirdparty/mcp-browser/`) gives live
Playwright browser control. **Use it for any debug-UI work** — verifying
`schedule_debug_server.py` / `schedule_view.py` renders, reproducing UI bugs, and
confirming a server-side change actually reaches the page.

Workflow: start the debug server → `browser_navigate` to `http://localhost:<port>` →
`browser_screenshot` → interact with `browser_click` / `browser_extract_text`.
Other tools: `browser_type`, `browser_wait_for_element`, `browser_execute_script`.

After changing it: `cd thirdparty/mcp-browser && npm run build`, then restart Claude Code
to reload the MCP server.

### Debug tools

- **[src/tool/debug/README.md](src/tool/debug/README.md)** — user guide and CLI for all debug tools
- **Skill: debug-ui-framework** — implementation map for `schedule_debug_server.py`, `schedule_view.py`, `aiegdb.py`, `aiemcp.py`, `aiediag.py`, `xaiehost2provenance.py` (detail in [reference.md](.cursor/skills/debug-ui-framework/reference.md))

### Build notes

- **[script/verify_env.sh](script/verify_env.sh)** — validate Vitis, LLVM, toolchain, board vars before build
- **[script/hostcompile.sh](script/hostcompile.sh)** — kernel build via `compile_one_kernel()` → `kc.sh`; do not restore deleted `compile_kernel.sh` (skill: hostcompile-entrypoint)
- **[script/aiehlc.sh](script/aiehlc.sh)** — `--platform sim` is build-only; launch sim separately via `runsim.sh` or debug UI **Run** (skills: sim-build-run-separation, raw-xaie-sim-debug-bundle)
- **`include/bspcompat/`** — shims for standalone-BSP headers present only on the armclang branch (`thirdparty/alib/include/` is a flattened copy of *one* BSP, picked by `--aie-version`, wiped every run, gitignored). `aiehlc.sh` appends `-I${AIEHLC_DIR}/include/bspcompat` **last** in `AIEHLC_ARGS` so the real Gen5 header still wins — add it to the **front-end only**, never to the host/kernel compiles. Details and the related `AIE_GEN > 2` XTime gate: skill **bspheadergen2**.
- **[script/kc.sh](script/kc.sh)** — after linking the kernel ELF, `strip_kernel_debug_loc` drops `.debug_loc` + the `.debug_info` group via `llvm-objcopy` (13.7 MB → 82 KB on matmul; this is JTAG download time, since the kernel ELF is embedded into the host ELF and `dow -force`d). `.debug_line` is kept, so `kernel.linemap.json` / aiediag pc are unaffected; the full-DWARF original is parked at `<out>/kernel_debug`. `--keep-debug-loc` opts out. Must be `llvm-objcopy` — GNU binutils rejects the chess `e_machine 0x108`.

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

## Working rules

- **Never** let an API function exceed 200 lines.
- **Create a skill** whenever an issue is fixed, or whenever the user has to correct
  something you did wrong.
- **Keep the architecture docs current** — update them as part of the change, not after.
- **After each task, list every file changed or created.**
