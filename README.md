# AIEHLC

App Writing and Deployment Tutorial: [tutorial.md](tutorial/tutorial.md)

## What is the aiehlc?

AIEHLC (AIE High‑Level Compiler) is an LLVM & MLIR-based, lightweight, and efficient high-level
compiler tailored for AIE (AI Engine) applications on the Versal AI Core Series. By leveraging spatial types and policies declared within the kernel code, the compiler automatically generates routing, tiling, and scheduling logic, eliminating the need to manually construct spatial compute graph connections.

Furthermore, the compiler features a high-level runtime that wraps the AIE driver API, ensuring seamless interoperability with CUDA/ROCm-compatible API use cases. This runtime solution integrates with bare-metal platforms to support direct ELF deployment on bare-metal hardware such as APUs and RPUs. It also co-operates with pre-built hardware designs (PDIs) to effectively decouple hardware and software application development, ultimately delivering a comprehensive, end-to-end deployment solution.

Key features:

> **Simplifies** the AIE application development learning curve for AIE development

> **Native provenance mapping** for easier debugging and handling abstraction leaks.

> **Kernel-requirement-driven** programming with automatic routing, tiling, and scheduling.

> **Automatic ping-pong** buffering and lock synchronization, overlaps DMA and compute without manual effort.

> **Memory-aware auto scheduling**, validates per-tile buffer budgets at compile time, catching capacity violations before they reach hardware.

> **Collective transfer primitives**, — built-in one-to-many broadcast and many-to-one gather for efficient multi-tile data distribution.

> **Six-level progressive MLIR lowering**, letting developers tune at whichever abstraction layer matches their expertise.


AIEHLC enables developers to easily build AIE applications by relying solely on the AIE driver C API and CUDA/ROCm-compatible APIs, eliminating the need to learn additional domain-specific languages (DSLs). This streamlined approach target to accelerates the development process and reduces barriers to entry for AIE application development.

### Emaple code conv2d

```c
//kernel code 
constexpr aie::GlobalPolicy conv_policy = {.fullconnect_auto = 0};
__global__(conv_policy) void conv2d_spatial(
    aie::port<input_window_int8 *, RowBC_spatial> win_a, // raw slab [halo_slice, W*C] (derived)
    aie::port<input_window_int8 *, ColBC> win_b,         // filter [K, tile_N]
    aie::port<output_window_int8 *, LtoR_Merge> win_c    // output [oh_per_row*OW, tile_N]
) {
   .....
}

//host code 
int main() {
    //...
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    //...
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);
    //...
}
```

### Architecture

![architecture diagram 6 IR](doc/diagram/autorouting_ir_architecture.png)

![architecture diagram C frontend](doc/diagram/architecture.png)

### What it is Not

1. It is not a VLIW processor compiler; instead, it relies on Synopsys or llvm-aie to provide processor-level support.
2. It focuses solely on software compilation and does not function as a hardware design compilation tool.
3. It currently supports only Versal AI Core Series devices and does not provide compatibility with Ryzen AI SOCs.

### What are the Limitations

Currently only support GMIO and no PLIO support yet.

### What are the Trade-Offs

1. The pre-built PDI (hardware design) supports only GMIO interfaces. For PLIO (FPGA) support, users must implement a customized hardware design with the appropriate PLIO logic components.

2. The primary programming language is C. However, users who wish to perform low-level tuning can interact directly with the MLIR programming layer.

## Examples

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --runtime-source-file example/perf/aieml_perf.cc 
```

Using llvm-aie (experimental):

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --use-llvm-aie --runtime-source-file example/other/multikernel.cc
```

AIE2PS Compilation Support

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --aie-version 5 --runtime-source-file example/perf/aieml_perf.cc
```

PetaLinux Support

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --platform linux --aie-version 2 --runtime-source-file tutorial/example.cpp
```

AIE Simulator Support (no board required)

Select the simulator with `--platform sim`. Key args:

- `--aie-version`: `1` (AIE), `2` (AIE-ML), or `5` (AIE2PS)
- `--sim-tiles "col:row,..."`: tiles to load a stub kernel ELF onto (comma-separated logical-`col`:physical-`row`, e.g. `0:3`). Omit to stub the whole array (default, but seems to crash the simulator often).

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

# stub all tiles (seems to crash the simulator often)
source script/aiehlc.sh --platform sim --aie-version 5 --runtime-source-file tutorial/example.cpp
# or explicitly stub only tiles you used, e.g. col=0 row=3
source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles "0:3" --runtime-source-file tutorial/example.cpp

# aiehlc only builds the artifacts; launch separately
bash script/runsim.sh
```

`runsim.sh` can be invoked three ways:

```bash
# 1. Default: no args -> reads ./aout/sim_config.sh (same CWD you ran aiehlc.sh from)
bash script/runsim.sh

# 2. Specific aout dir: point it at any aout/ containing a sim_config.sh
bash script/runsim.sh path/to/aout

# TilingLinalg writes its config in the provenance bundle
bash script/runsim.sh aout/worklocal

# 3. Direct (no config file): pass the inputs yourself (not recommended)
bash script/runsim.sh \
    --host-src /path/to/host.cc \
    --kernel-objs "/path/k1.o /path/k2.o" \
    --kernel-names "k1 k2" \
    --aie-gen 5 \
    --stub-all            # or: --stub-tiles "0:3"
```

On success the sim ends with `Sucess: CPU result matches AIE.` / `Sim result: 0`.

## rcom: ROCm/HIP GEMM Front End

`rcom` is a standalone Python front end that reads a canonical ROCm/HIP int8
GEMM source, recognizes it as a GEMM, and emits a CUDA-style `<name>.cc/.h`
(the same dialect as `example/tileprogram/ccode/simplematmul2.cc`). It then
hands the generated `.cc` to the unchanged `script/aiehlc.sh` pipeline to build
host + kernel ELFs.

```
matmul_hip.cpp --(rcom.py)--> <name>.cc + <name>.h --(aiehlc.sh)--> aout/main.elf
```

The HIP kernel body is *recognized and templated* (the proven AIE cache-A /
stream-B matmul), not translated line-for-line. v1 supports **int8 only** and
defaults to the HW-validated `M=N=K=256`, `4x4` mesh config; `M/N/K` and mesh
are overridable but deviations are heuristic (untested on HW) and emit a
warning.

```bash
source script/setup.sh

# One-shot: HIP source -> generated .cc/.h -> ELF (aout/main.elf)
source script/rcom.sh --rocm-source-file example/tileprogram/rocm/matmul_hip.cpp --aie-version 5

# Emit only (no build), e.g. for inspection:
python3 src/tool/frontend/rcom.py example/tileprogram/rocm/matmul_hip.cpp \
    --emit-only --out ./gen

# Overrides: kernel name, mesh, dims
python3 src/tool/frontend/rcom.py example/tileprogram/rocm/matmul_hip.cpp \
    --name gemm --mesh 4x4 --M 256 --N 256 --K 256 --out ./gen
```

## Runtime Debug Level

Control runtime diagnostic verbosity and feature flags from your source file using `#pragma aie_debug_level`:

```cpp
#pragma aie_debug_level 2

__global__ void mykernel(const int32_t *A, int32_t *B) {
    // kernel code
}

int main() {
    // host code
}
```

The debug level value is a bitfield:
- **Bits 0-3**: verbosity level (0-15)
- **Bits 4-31**: feature flags

### Verbosity Levels (bits 0-3)

| Level | Behavior |
|-------|----------|
| `0` | Silent (default when pragma is absent) |
| `1` | BD tracking and IO logs |
| `2` | Full diagnostics: DMA address log, write pattern, readback |

### Feature Flags (bits 4+)

| Flag | Bit | Value | Behavior |
|------|-----|-------|----------|
| `DISABLE_MULTID_DIM_DMA` | 4 | `16` | Suppress multi-dimensional DMA; force linear `__Runtime_dma_bd_config` |
| `DISABLE_PARTITIONTEARDOWN` | 5 | `32` | Skip `XAie_PartitionTeardown` in `__Runtime_device_teardown` |
| `MM2SBDFINISH_COUNTER` | 6 | `64` | Arm probe-tile MM2S ch0/ch1 BD-finished counters; read via `__Runtime_perfcnt_read_mm2s_probe`. Requires `--profiling` |
| `AIE_DMA_ISSUE_COUNT` | 7 | `128` | Use `XAie_PartitionInitialize_v2`/`_v2` teardown to arm DMA txn perf counters (whole partition, MM2S ch0) |
| `CORE_PERF_COUNTER` | 8 | `256` | Arm core-module cycle counters (active / vector-instr / stream-stall / lock-stall) on the probe tile. Requires `--profiling` |

### Examples

| Pragma Value | Verbosity | Flags | Effect |
|-------------|-----------|-------|--------|
| `0` | 0 | none | Silent, multi-dim DMA enabled |
| `1` | 1 | none | BD tracking, multi-dim DMA enabled |
| `2` | 2 | none | Full diagnostics, multi-dim DMA enabled |
| `16` | 0 | `DISABLE_MULTID_DIM_DMA` | Silent, multi-dim DMA suppressed |
| `18` | 2 | `DISABLE_MULTID_DIM_DMA` | Full diagnostics + multi-dim DMA suppressed |
| `128` | 0 | `AIE_DMA_ISSUE_COUNT` | Silent, DMA txn counters armed via `PartitionInitialize_v2` (MM2S ch0) |

The pragma is detected during preprocessing and emits a strong symbol override of `g_runtime_debug_level` into the generated `host.cc`. The runtime in `aie_runtime.c` defines this variable as a weak symbol with default `0`, so the linker picks the user-specified value when present.

This works for both single-tile (`aiehlc`) and multi-tile (`tilinglinalg`) compilation paths.

## Core Trace

Arm the AIE per-tile trace units from your source file using `#pragma aie_trace`.
Each pragma selects a tile `(col, row)` whose core-module and memory-module trace
units are set up in the generated `host.cc` (via `__Runtime_core_trace_begin_dma`).
The memory-module trace stream is drained through one of the tile's DMA channels;
the pragma lets you pick which one.

### Forms

```cpp
// 1. Default: trace tile (0,3); mem-trace drains via S2MM channel 0.
#pragma aie_trace(0, 3)

// 2. Ranges: trace every tile in col 1..2, row 3 (all get the default S2MM ch0).
#pragma aie_trace(1:2, 3)

// 3. STREAM: pick the mem-trace DMA direction + channel explicitly.
#pragma aie_trace((0, 3), (STREAM, "s2mm", 0))
#pragma aie_trace((0, 3), (STREAM, "mm2s", 1))

// 4. PARAMETER: name a kernel window; the compiler resolves the physical
//    (direction, channel) the tiling flow assigned to it on that tile.
//    Input windows drain via S2MM, output windows via MM2S.
#pragma aie_trace((0, 3), (PARAMETER, "win_a"))
```

### Selection semantics

| Form | `mem_dma_kind` | `mem_dma_ch` |
|------|----------------|--------------|
| `aie_trace(col, row)` | `S2MM` | `0` (default) |
| `(STREAM, "s2mm", ch)` | `S2MM` | `ch` |
| `(STREAM, "mm2s", ch)` | `MM2S` | `ch` |
| `(PARAMETER, "win_x")` | resolved: input→`S2MM`, output→`MM2S` | resolved channel |

- The direction string is case-insensitive (`"s2mm"` / `"S2MM"`).
- `PARAMETER` resolution reads the `dfschedule.config.create_io` provenance to
  map a window name to the DMA channel the flow physically assigned on the traced
  tile. It is available in the multi-tile (`tilinglinalg`) flow; if a name cannot
  be resolved (unknown window, or no matching create_io on that tile) the compiler
  warns and falls back to the default `S2MM ch0` so the trace still arms.
- Pragmas are repeatable and ranges expand to one trace-setup call per tile; the
  selected mem-DMA applies to every tile expanded from that pragma.
- The legacy flat form `#pragma aie_trace(0, 3)` (no inner parens) is still
  accepted and is identical to the default `S2MM ch0` behaviour.

Each pragma emits `__Runtime_core_trace_begin_dma(dev, col, row, dma_kind, dma_ch)`
into `host.cc`. STREAM selection works in both single-tile and multi-tile flows;
PARAMETER resolution requires the multi-tile provenance and otherwise falls back
to the default.

### Validation

The mem-DMA selection is checked at compile time and the build **fails with an
error** (non-zero exit) rather than silently mis-tracing when:

| Case | Example | Error |
|------|---------|-------|
| Invalid STREAM direction | `(STREAM, "s3mm", 0)` | `STREAM direction "s3mm" is invalid (expected "s2mm" or "mm2s")` |
| STREAM channel not used by the app on that tile | `(STREAM, "mm2s", 1)` when the tile only drives `mm2s ch0` | `STREAM mm2s ch1 is not used by the app on tile(0,3). Available mm2s channels: ch0` |
| PARAMETER name is not a kernel window | `(PARAMETER, "win_xyz")` | `PARAMETER "win_xyz" is not a kernel window/port name.` |

The STREAM channel check compares the requested `(direction, channel)` against the
channels the tiling flow actually assigned on the traced tile (from the
`dfschedule.config.create_io` provenance), so it rejects both hardware-invalid
indices and valid-but-unused channels. Tiles a given kernel does not own are
skipped so multi-kernel meshes are not spuriously failed. The `Default` form
(`aie_trace(col, row)`, S2MM ch0) is not validated.

## Control-Plane Pragmas

Two opt-in marker pragmas (no arguments) enable control-plane features in the
multi-tile (`tilinglinalg`) flow. Both are off by default; add the pragma to your
source file to turn the feature on. They are detected during preprocessing and
published as module attributes that gate the corresponding MLIR pipeline stage.

```cpp
#pragma control_plan_op_control_packet
#pragma CONTROL_PLAN_GROUP_REG_WRITE

__global__ void mykernel(const int32_t *A, int32_t *B) {
    // kernel code
}

int main() {
    // host code
}
```

| Pragma | Module attr | Effect when present |
|--------|-------------|---------------------|
| `#pragma control_plan_op_control_packet` | `routing.control_plan_op_control_packet` | The routing/scheduling pipeline **reserves** (excludes) control-plane stream-switch resources (pkt-ids / arbiters / slots) so the control-plane fabric and the data plane do not collide. Absent => reservation is skipped. |
| `#pragma CONTROL_PLAN_GROUP_REG_WRITE` | `routing.control_plan_group_reg_write` | The host pipeline runs `GroupRegWritePass`, which coalesces identical core-tile lock-init register writes into **control-packet group writes** (broadcast / row-multicast, lowered to `__Runtime_ctrl_row_write_ack`). Absent => the pass is skipped and lock inits are emitted as individual `XAie_LockSetValue` register writes. |

### `#pragma control_plan_op_control_packet`

Reserves control-plane resources so a control-packet fabric (see the row-control
planner APIs in `aie_runtime.h`) can share the array with the data plane without
resource conflicts. The pipeline gates `reserveControlPlaneResources(...)` on the
published attr.

### `#pragma CONTROL_PLAN_GROUP_REG_WRITE`

Enables `GroupRegWritePass`. In a tiled GEMM the per-tile lock inits are identical
across the mesh; when this pass runs it clusters them by `(tile-local addr, value)`
and folds a cluster that covers all configured tiles into a single `broadcast`
group write, or all columns of one row into a `row` group write. Each group write
lowers to a blocking `__Runtime_ctrl_row_write_ack` call in `host.cc`. Without the
pragma the pass does not run (the pipeline logs
`GroupRegWritePass skipped (enable with #pragma CONTROL_PLAN_GROUP_REG_WRITE)`)
and every lock init is emitted individually.

Both pragmas work in the single-kernel and multi-kernel `tilinglinalg` paths.

## Build Options

Two host-build toggles, both off by default and both compile-time (they change how
`aie_runtime.c` is compiled, so they need a rebuild to take effect):

| Flag | Define | Effect |
|------|--------|--------|
| `--profiling` | `-DAIEHLC_PROFILING=1` | Compile in the PS PMU profiling layer: launch-phase timers (kload / bdcfg / coreen / startio / wait_io) and the `[PERF]` summary. Off, the `RT_PROF_*` macros expand to nothing and the reader functions return zeros. |
| `--skip-bss` | `-DAIEHLC_SKIP_BSS_DEFAULT=1` | Load only PT_LOAD segments with file bytes, skipping pure-`.bss` zero-init on kernel load. Safe only when no global is read before being written — the DMA-fed window buffers qualify. |

```bash
source script/aiehlc.sh --platform baremetal --aie-version 5 --profiling --skip-bss \
    --runtime-source-file ./example/tileprogram/ccode/simplematmul2_prof.cc
```

Note the `MM2SBDFINISH_COUNTER` and `CORE_PERF_COUNTER` pragma bits above are inert
without `--profiling`: the counters are armed by code that only exists in a profiling
build, so the pragma alone reports zeros.

`--skip-bss` must be the compile define rather than the `AIEHLC_SKIP_BSS` environment
variable on baremetal, where newlib `getenv()` always returns `NULL`.

## Debug Tools

### aiediag — Flow-Aware DMA Diagnostic

`aiediag` (`src/tool/debug/aiediag.py`) is a post-deployment diagnostic tool for debugging DMA stalls and data flow issues on live AIE hardware. It reads DMA status registers via `aiedbg`, cross-references them with compile-time provenance JSONs to identify connected tiles and routing paths, and prints an automated diagnosis.

#### When to Use

- A DMA channel is stuck (stream starvation, backpressure, lock stall)
- Output data is missing or incomplete after a HW run
- You need to determine whether `start_io` was actually issued on a shim tile
- You want to trace a data flow from shim through core tiles and identify where it stalls

#### Usage

```bash
# Basic: diagnose MM2S ch0 on logical tile (1,4) with startcol offset 3
python3 src/tool/debug/aiediag.py dig 1 4 -mm2s0 startcol 3 -dev pal

# Dry-run (no HW access, prints what would be read)
python3 src/tool/debug/aiediag.py dig 1 4 -mm2s0 startcol 3 --dry-run

# Specify AIE version and custom JSON directory
python3 src/tool/debug/aiediag.py dig 0 3 -s2mm1 --aie-version 2ps --json-dir ./my_worklocal

# Custom shim events JSON location
python3 src/tool/debug/aiediag.py dig 0 0 -mm2s0 --shim-events-json /path/to/shimtile_events.json -dev pal
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `COL` | Logical column (from IR/JSON, 0-based) |
| `ROW` | Physical row (0=shim, 1=memtile, 3-6=core for AIEML) |
| `-mm2s0`, `-s2mm1`, etc. | Direction and channel number |
| `startcol N` | Physical column offset; physical_col = COL + startcol |
| `--aie-version` | `5` (AIEML, default) or `2ps` |
| `-dev`, `--device` | Device type passed to aiedbg (e.g., `pal`) |
| `--target` | Target passed to aiedbg (e.g., `baremetal://192.168.0.1:9999`) |
| `--json-dir` | Directory containing provenance JSON files |
| `--shim-events-json` | Path to `shimtile_events.json` (default: `~/aiejson/shimtile_events.json`) |
| `--dry-run` | Print aiedbg commands without executing |

#### Diagnostic Steps

The tool runs 7 steps in sequence:

1. **DMA Status** — Read the queried tile's DMA status register (running/idle/stalled, stall cause, current BD, queue size). For core tiles (row != 0), also decode memory-module DMA start/finish/error events for all 4 channels and any active core-module error events.
2. **BD Chain** — Show the buffer descriptor chain from JSON (BD IDs, lengths, locks, ping-pong, packet IDs, repeat count), the total intended data volume (sum of BD lengths × repeat), and an intended-vs-real comparison of each BD's `Buffer_Length` read back from hardware (`OK` / `MISMATCH` / `hw not configured`)
3. **Connected Tiles** — Use the flow summary to find all senders/receivers in the same data flow
4. **Connected Tile DMA Status** — Read DMA status registers of all connected tiles
4b. **Shim Event Status** — For shim tiles (row=0), read hardware event status registers to determine whether DMA start/finish/stall events have fired
5. **Connected Tile BD Chains** — Show BD chains of connected tiles
6. **Routing Path** — Show the physical routing hops from dmaphop provenance map
7. **Diagnosis** — Automated root-cause analysis combining all data sources

#### Prerequisites

##### aiedbg

`aiedbg` must be in `PATH`. It is the Xilinx/AMD AIE debug CLI that reads/writes AIE tile registers over JTAG or network. It is part of the Vitis installation (`$XILINX_VITIS/bin/aiedbg`) or can be installed separately.

The live debug server (`schedule_debug_server.py` / `aiehlc.sh --prettydebug`) **bootstraps** `aiedbg` automatically on first launch: clones into `thirdparty/aiedbg`, runs `pip install --user`, and writes `.aiehlc/aiedbg_env.sh` for PATH. Manual install/update: `script/debug/bootstrap_aiedbg.sh` (`--update` to refresh). Use `--skip-aiedbg-bootstrap` to skip auto-install (e.g. when using Vitis's copy).

The tool invokes it as:
```
aiedbg [--target TARGET] [--device DEVICE] reg read PHYS_COL ROW 0xOFFSET
```

##### Provenance Map JSONs

These JSON files are generated by the tilinglinalg compilation pipeline and describe the compiled data flow. The tool auto-searches these directories (in order):

1. `./worklocal/`
2. `./aout/worklocal/`
3. `./src/mlir/mlirfront/tilinglinalg/pass/unitest/build/worklocal/`
4. `./src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/`

Or use `--json-dir` to specify explicitly.

**`dfscheduleprovenancemap.json`** — Generated by `DfscheduleToApiPass` / `DfscheduleToKernelApiPass`. Contains:
- `tiles[]` — Every tile with its DMA channels, BD chains (bd_id, buffer_offset, len, locks, packet_id, next_bd, dim_strides, iter_wrap), start_io repeat counts, and contracts
- `flow_summary[]` — Each data flow with its participants (tile_col, tile_row, io_direction, channel, repeat_count, bd_len)

**`dmaphopprovenacemap.json`** (or `provenance_map.json`) — Generated by `DmaphopTodfscheblueprintPass`. Contains:
- `communication_paths[]` — Each path with producer/consumer tiles, physical hop chain, tensor shape, partition info, and port symbols

##### shimtile_events.json

Default location: **`~/aiejson/shimtile_events.json`**

This file maps hardware event IDs to human-readable names for the shim tile PL module. It is derived from the AIE driver source (`aie_runtime_debug.c :: s_pl_evt_names`). The file structure:

```json
{
  "module": "PL",
  "description": "Shim tile (PL module) events ...",
  "source": "aie_runtime_debug.c :: s_pl_evt_names[117]",
  "max_id": 116,
  "events": {
    "0": "NONE",
    "14": "DMA_S2MM_0_START_TASK",
    "15": "DMA_S2MM_1_START_TASK",
    "16": "DMA_MM2S_0_START_TASK",
    "17": "DMA_MM2S_1_START_TASK",
    "18": "DMA_S2MM_0_FINISHED_TASK",
    "19": "DMA_S2MM_1_FINISHED_TASK",
    "...": "..."
  }
}
```

Key DMA event IDs used by aiediag (all in the shim PL event status registers at `0x34200`-`0x34204`):

| Event ID | Name | What It Means |
|----------|------|---------------|
| 14-17 | `DMA_*_START_TASK` | DMA channel was started (start_io was issued) |
| 18-21 | `DMA_*_FINISHED_TASK` | DMA channel completed all programmed transfers |
| 22-25 | `DMA_*_STALLED_LOCK` | DMA stalled waiting for lock |
| 26-27 | `DMA_S2MM_*_STREAM_STARVATION` | S2MM has no data arriving from stream |
| 28-29 | `DMA_MM2S_*_STREAM_BACKPRESSURE` | MM2S cannot push data (downstream full) |
| 30-33 | `DMA_*_MEMORY_*` | Memory-side stall (backpressure or starvation) |

The `~/aiejson/` directory also contains event JSONs for other tile types (`core_events.json`, `memtile_events.json`) and register address maps (`aie2ps*.json`). If `shimtile_events.json` is missing, aiediag prints a warning and skips the shim event check; all other diagnostic steps still run.

### schedule_debug_server — Live Schedule View + Debug/Test Daemon

`schedule_debug_server` (`src/tool/debug/schedule_debug_server.py`) is a
zero-dependency local daemon (stdlib `http.server`, bound to `127.0.0.1`) that turns
the static `host_schedule.html` schedule view into a live debug/test console. It serves
the enhanced HTML, exposes JSON endpoints, imports `aiediag.py` as a library for register
offsets/decoders/provenance, and orchestrates `apppaltest.py`. Design rationale:
[doc/design/aiegdb_live_debug_framework.md](doc/design/aiegdb_live_debug_framework.md).

#### Multi-app mode

The daemon is not tied to one compiled app. It discovers app *bundles* — any directory
holding a `schedule_view.json` — and injects the selected app's data into the page when
serving it, so switching apps in the **App** dropdown reloads the whole UI against a
different build. One daemon can therefore serve builds from several repos at once.

```bash
# standalone: auto-discovers aout/** and opens the most recently compiled app
python3 src/tool/debug/schedule_debug_server.py

# add apps from another repo (naiebaremetal examples) and a saved snapshot
python3 src/tool/debug/schedule_debug_server.py \
    --app-root ../naiebaremetal/example \
    --app doc/debug/int32_deadlock_snapshot/worklocal=int32-deadlock
```

| flag | meaning |
|---|---|
| *(positional)* `workdir` | still accepted; registered as an app (backwards compatible) |
| `--app PATH[=LABEL]` | register one app explicitly, repeatable, may live in another repo |
| `--app-root DIR` | scan a tree for app workdirs, repeatable |

Switching an app also switches that app's **run profile** — the `extra_devices`
(simulator / board, `hw_env`, PDI and ELF paths) from its `debug_ui_config.json` — plus
`startcol` and `aie_version` from its provenance JSONs. Switching is refused while a run
is in progress, since the profile carries board and image paths.

Endpoints: `GET /apps`, `POST /apps/select {id}`, `POST /apps/add {path,label,select}`,
`GET|POST /uistate`. Note these are unauthenticated, like the daemon's other non-LLM
endpoints (`--password` gates only the LLM tab).

Both producer flows feed the same consumer:

```
aiehlc_aiesim: compiler passes ────────────────► provenance JSONs ─┐
                                                                   ├─► schedule_view.py ─► app bundle
naiebaremetal: aiecompiler Work/ ─► work2provenance.py ────────────┘
```

`schedule_view.py <workdir> --json-only` writes just `schedule_view.json` (+ the per-app
code cache) and skips the ~1.6 MB standalone HTML, which is what the daemon-served flow
wants. naiebaremetal's `src/tool/run_debug_ui.sh` uses this and registers the result with
an already-running daemon via `/apps/add` instead of starting a second server.

Note the per-tile code cache lives under `<workdir>/debugcache/code` so two apps cannot
overwrite each other's pieces (the paths are handed to the embedded agent to read).

#### debug_ui_mcp — the embedded agent's view of the UI

`src/tool/debug/debug_ui_mcp.py` is the MCP server the browser's LLM tab talks to. It
follows whatever app the human has selected (the daemon tells it via
`DEBUGUI_SERVER_URL`) and can read both the UI's contents and its current state:

| tool | purpose |
|---|---|
| `list_apps` / `current_app` / `select_app` | see and change which compiled app is loaded |
| `app_sources` | that app's own source files, grouped, plus the `file:line` defining each kernel the schedule runs — so an answer can cite the user's code instead of stopping at the schedule |
| `list_panes` | which panes are readable and what selector each needs |
| `get_pane(pane, col, row, flow, query)` | the content of a pane: `grid`, `tile.hi`, `tile.mid`, `tile.lo`, `tile.kernel`, `tile.supply`, `net.flow`, `search` |
| `get_ui_state` | what the user has open right now — selected tile, active tile tab, net tab, console pane, channel, flow |
| `tile_info`, `tile_list`, `symbol_search`, `get_design_overview`, `get_flow_detail`, `get_applog`, `get_sim_log`, `get_backend_status` | existing static-schedule tools, now app-aware |

The browser reports selection changes to `POST /uistate`, so the agent can answer
questions about the view in front of you rather than guessing. Tools degrade gracefully
when an app lacks optional data (e.g. naiebaremetal bundles have no `invariant_checks`,
and `backend_status.json` is only present once a run profile exists).

##### Answers cite your source, not just registers

The agent is told what the app is *made of*, not only where it lives: the same
inventory `app_sources` returns is inlined into its system prompt, including the
file and line defining each kernel the schedule runs (`kernel: conv2` →
`src/convolution2.cc:19`). Its instructions and the `source-grounding` skill make
reading and quoting that source the default rather than an extra step, on the
principle that a register value is the symptom while the user's source is where
the cause lives and where the fix has to go. Citations are written `<file>:<line>`,
which the UI turns into a click that opens the file in the Info pane.

Hand-written and generated files are kept apart, because proposing an edit to
`host.cc` or `kernel.cc` wastes the user's time twice — those are overwritten on
the next build, so they are evidence of what the compiler decided, never the place
to fix anything. For the aiehlc flow, whose `aout/` holds no hand-written code at
all, `aiehlc.sh` records the `--runtime-source-file` it built from in
`worklocal/app_source.txt`; without that the agent has nothing of yours to read.

It lets you, from a single browser page:

1. **Run the test** — deploy the compiled ELF via `apppaltest.py -nonreboot` and watch the
   console + pass/fail/hang verdict.
2. **Live overlay** — poll each tile's **DMA / core / event** status and colorize the grid
   in near real time (switchable tabs, ~2s interval).
3. **Per-tile console** — click a tile and issue whitelisted read-only ops
   (`dma`, `core`, `event`, `pc`, `reg <off>`) to inspect its registers.

When the daemon is not running, opening `host_schedule.html` from disk still works — it
degrades gracefully to the static offline schedule view (live controls become inert).

#### Quickest path: `--prettydebug`

Add `--prettydebug` to the tiling build and the server launches automatically (and opens
your browser) once the schedule view is generated:

```bash
source script/aiehlc.sh --aie-version 5 \
  --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc --prettydebug
```

This runs the full flow, then serves `aout/worklocal/host_schedule.html` live. Ctrl-C
stops the server.

#### Manual launch

```bash
# Live reads (grid overlay + per-tile /cmd) need a JTAG target. Set it once:
export AIEDBG_TARGET=xsdb://10.23.224.213:3121   # or pass --target below

# Serve an existing worklocal (must contain host_schedule.html + provenance JSONs)
python3 src/tool/debug/schedule_debug_server.py aout/worklocal --open

# Point at a specific ELF and shim startcol offset
python3 src/tool/debug/schedule_debug_server.py aout/worklocal \
  --elf aout/main.elf --startcol 3 --aie-version 5 --device pal --port 8091
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `workdir` | Dir with `host_schedule.html` + `schedule_view.json` + provenance JSONs (default `aout/worklocal`) |
| `--elf` | ELF to deploy via the Run button (default `<workdir>/../main.elf`) |
| `--port` | HTTP port (default `8091`) |
| `--aie-version` | Register offset set: `5` (AIEML, default) or `2ps` |
| `--device` | `aiedbg --device` (default `pal`) |
| `--target` | `aiedbg --target` (e.g. `xsdb://10.23.224.213:3121`); resolution order: `--target` → `$AIEDBG_TARGET` → `~/.aiedbg_env` (the file `aiedbg-setup` writes). Without any, live grid/`/cmd` reads fail with "timed out" / "connection closed" |
| `--startcol` | Physical column offset: `phys_col = col + startcol` |
| `--apppaltest` | Path to `apppaltest.py` (default `script/test/apppaltest.py`) |
| `--open` | Open the served URL in a browser after binding |

#### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serves the enhanced `host_schedule.html` |
| `GET` | `/schedule_view.json` | The static `DATA` blob |
| `POST` | `/run` | Spawn the board test `-y -nonreboot <elf>` (`-u` unbuffered) → `applog`. Body `{device, board_host}`: `palmyra` → `apppaltest.py` (inherit env); `vek385` → `appvek385.py` with env `USERNAME=getpass.getuser()` + `VEK385IP=<board_host>` (host required) |
| `POST` | `/stop` | Force-kill the running test's process group (SIGTERM→SIGKILL); appends a `[force-stop]` line to `applog` |
| `GET` | `/applog?offset=N` | Realtime tail of the `applog` file → `{data, next, running, status}`; poll stops on `running=false` (process exit), not on derived `pass\|fail` |
| `GET` | `/ping?device=&host=` | Connection test → `{ok, aiedbg, target, detail}`. Confirms `aiedbg` is in PATH **and** the resolved JTAG target actually answers (one read-only register read on the first schedule tile). Drives the UI's "Connect" gating; a passing probe records session mode `connected` |
| `POST` | `/attach` | "Open Current Session": same link probe as `/ping`, but records session mode `attached` — for a run the user started outside the UI (CLI, or an already-programmed board). Body `{device, board_host}` → `{ok, detail, session}` |
| `GET` | `/aiegdb/spec` | `aiegdb.COMMAND_SPEC` as JSON — the console autocomplete's command grammar. Served from the imported module (not the aiegdb subprocess) so it answers before any command has run |
| `GET` | `/grid?what=dma\|cores\|events&device=&host=` | Whole-array status survey for the overlay; `device`/`host` pick the aiedbg target (`palmyra` → `xsdb://$PALIP:3121`, `vek385` → `xsdb://<host>:3121`) |
| `POST` | `/cmd` | Whitelisted op (`dma`/`core`/`event`/`pc`/`reg`/`chans`/`chanevent`); body may carry `device`/`host` for device-aware target. `chans` lists a tile's channels + coarse live DMA state; `chanevent` (needs `dir_ch`) decodes per-channel start/finish/(stall/error) events → `{events, summary}` |

The browser UI has a **Board selector** (`palmyra` / `vek385`). Selecting a device
enables a **"Connect"** button (and, for `vek385`, reveals a board-hostname
text box). Clicking **Connect** hits `/ping`; only on a passing test does the
"Live status overlay" checkbox unlock **and** the drill-down console appear. The
"Run" button (spawns `apppaltest`) is enabled as soon as a device is chosen.

#### Session provenance — why a target is not a connection

`AIEDBG_TARGET` is exported by `script/test/envlocal.sh`, so the daemon has a target
from the moment it starts, with no user action. That is *not* evidence that a board is
live or that anything has been run — and a physically reachable board still holds
registers from whatever ran on it last, possibly days ago or by another user. The
daemon therefore tracks how the current session earned its access:

| State | Earned by | Meaning for live reads and logs |
|---|---|---|
| `none` | default at startup, even with `AIEDBG_TARGET` set | reads refused; nothing may be inferred about the board |
| `connected` | **Connect** (a verified `/ping`) | link is real, but registers are pre-existing state, not the result of a run here |
| `attached` | **Open Current Session** (`/attach`) | a real run is in play, started outside the UI — the daemon cannot vouch for what came before |
| `ran` | **Run** | only here are live state and the `applog` the current run |

`hw_authorized()` gates the live overlay (`/grid`) and the `aiegdb` MCP server's device
commands; navigation, `help` and `?` stay available so the console is still useful
offline. The state is published in `backend_status.json` (`session`, `session_summary`),
injected into every LLM message, and enforced by a precondition block in the agent's
system prompt.

Log provenance is tracked the same way. `applog` is a fixed repo-root path that the
manual CLI flow also writes, so a file being present proves nothing — `get_applog`
prefixes a banner such as:

```
[STALE: written 2026-08-04 17:12, BEFORE this debug session started (2026-08-05 09:40).
 This describes a PREVIOUS run, not the current one.]
```

Without it, a leftover `PASS: all 65536 elements match` reads exactly like a fresh
result — which is precisely how the embedded agent once reported a successful run that
had never happened in that session.

#### aiegdb console tab

The `aiegdb` tab renders each command as a foldable block (command as header, output
below), colorized by line kind — scope changes, errors, warnings, `OK`/`PASS` verdicts,
`PC -> file:line`, hex literals and `key:` labels. The `[registers read] { … }` appendix
that aiegdb appends after most decoded commands collapses into a `<details>`.

Typing offers scope-aware suggestions (name · args · summary), with a red **WRITES HW**
badge on the intrusive commands (`reg write`, `dma counter setup`). Tab/Enter fills the
command and leaves the caret for its arguments; ↑/↓ moves through suggestions when the
popup is open and through command history when it is closed. **⌘ Commands** opens a
searchable palette of everything valid at the current scope.

The grammar comes from `aiegdb.COMMAND_SPEC` — the same dict that renders the CLI's `?`
listing — fetched from `/aiegdb/spec`, with a copy baked into the generated HTML so
autocomplete still works when opening `host_schedule.html` with no daemon.

The right panel has a **drill-down command console** (pinned bottom, revealed after
a passing connection test): clicking a tile sets it as the console TARGET and
auto-lists its channels; clicking a channel narrows the target to that channel,
after which `status` (DMA status) and `event` (per-channel DMA events) operate on
it. Commands: `channels | status | event | core | pc | reg <off>`. The console
persists while the overlay checkbox is toggled off (it is tied to the connection,
not the overlay), and is hidden again when the device changes or the connection
test fails.

#### Safety

- Binds `127.0.0.1` only; `/cmd` uses an op **whitelist** (no arbitrary shell; `reg` offset
  parsed as an integer).
- Issues **read-only** register reads only (never `stop`/`con`/writes), so it cannot perturb
  a running target sharing the same JTAG bridge. The two commands that *do* write —
  `reg write` and `dma counter setup` — are reachable only by typing them explicitly in the
  aiegdb console, and are flagged **WRITES HW** wherever they are suggested.
- Live reads require a **board session** (Connect / Run / Open Current Session). A
  target inherited from `$AIEDBG_TARGET` does not authorize reads, so neither the overlay
  nor the embedded agent can silently report another run's leftover register state as
  current — see *Session provenance* above.
- Reuses aiediag's `--json` `value_hex` parsing (avoids the offset-as-value Pitfall). If
  `aiedbg` is not in `PATH`, live reads are disabled and endpoints report `unreachable`
  rather than fake zeros.

## Enable realtime debug with --prettydebug flag

```
source script/aiehlc.sh --prettydebug  --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc
```
## Compiling aiehlc (Optional)

Building aiehlc is only necessary if you intend to develop or compile aiehlc itself. If your aim is simply to use aiehlc, this step is not required.

Build Tutorial: [build.md](doc/build.md)

## Contributing

If you plan to contibute code then make sure you set up the clang-format pre-commit hook.

```bash
source script/precommitsetup.sh
```

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

<p align="center">Copyright&copy; 2025-2026 Advanced Micro Devices, Inc.</p>
