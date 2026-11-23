# Kernel Config Offload (KERNELCONFIGOFFLOAD) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Under an opt-in `#pragma KERNELCONFIGOFFLOAD`, offload core-tile DMA
configuration (incoming/ongoing S2MM BD, outgoing MM2S BD, lock init, DMA
channel-start) from the HOST (`host.cc`) into the AIE CORE (`kernel.cc`), which
self-programs its own DMA/lock registers via raw MMIO writes.

**Architecture:** Reuse the existing two-module pipeline. A new module attr
`routing.kernel_config_offload` (published by aiehlc from the pragma) gates:
(a) `DfscheduleToApiPass` to SKIP host-side core-tile BD/lock/channel-start
emission, (b) the kernel path (`DfscheduleToKernelApiPass`) to EMIT raw-MMIO
self-config into `kernel.cc`, and (c) the runtime to call
`XAie_CoreProcessorBusEnable` per core tile so the core can write its own
memory-module DMA/lock registers. Default OFF ⇒ byte-identical to today.

**Tech Stack:** C++ MLIR passes (LLVM/MLIR), Clang pragma handler, C runtime
(XAie driver), EmitC codegen. AIE2PS (gen5) register offsets. Design doc:
`docs/plans/2026-09-14-kernel-config-offload-design.md`.

---

## Ordering rationale

Task 1 is a **HW spike** that de-risks the one residual unknown (the exact
core-visible register address + DMA re-arm behavior) BEFORE the codegen
refactor. Do NOT proceed to Task 3+ codegen until Task 1 confirms data lands.

Tasks 2 (runtime helper) and 4 (pragma plumbing) are independent of the spike
outcome and can be built in parallel with confidence; but the spike MUST pass
before the codegen tasks (5-7) that depend on the exact MMIO encoding.

---

## Task 1: HW spike — core self-programs its own S2MM DMA via MMIO

**Purpose:** Prove a core tile, after host `XAie_CoreProcessorBusEnable`, can
write its own memory-module BD/lock/start-queue registers via volatile MMIO and
land incoming data. This validates: (a) the correct core-visible register base
for the tile-local `0x1D000`/`0x1E000`/`0x1F000` offsets, (b) the BD field
encoding, (c) DMA channel re-arm from inside the core.

**Files:**
- Create: `example/tileprogram/ccode/spike_kernelcfg/host_spike.cc` (hand-written)
- Create: `example/tileprogram/ccode/spike_kernelcfg/kernel_spike.cc` (hand-written)
- Reference: `docs/plans/2026-09-14-kernel-config-offload-design.md` register map
- Reference: `src/mlir/runtime/aie_runtime.c` `__Runtime_dma_bd_config` (host BD encoding to mirror)
- Reference: `thirdparty/alib/aie-rt/driver/src/core/xaie_core.c:1113` `XAie_CoreProcessorBusEnable`

**Step 1: Read the reference host BD encoding**

Read `src/mlir/runtime/aie_runtime.c` `__Runtime_dma_bd_config` fully to capture
the exact S2MM BD field packing (address, length, lock acquire/release ids+vals,
next_bd, packet enable/id, out-of-order bd id) that the MMIO writes must mirror.

**Step 2: Write the spike host (`host_spike.cc`)**

Minimal single-core host that:
- `dev = __Runtime_explicit_init()` (or `_partition`).
- Calls `XAie_CoreProcessorBusEnable(dev, XAie_TileLoc(col, row))` for ONE core tile.
- Programs ONLY the shim BD to feed data to that core (host still owns shim).
- Loads + starts the core (which self-programs its S2MM DMA in `kernel_spike.cc`).
- Verifies the incoming data landed in the core's buffer (read back via `XAie_DataMemBlockRead` or an output DMA).

**Step 3: Write the spike kernel (`kernel_spike.cc`)**

Core `main()` that, BEFORE `kernel_invoke`, self-programs via volatile MMIO:
- S2MM BD0 at core-visible base + `0x1D000` (address, length, lock ids/vals, next_bd=-1, packet disable).
- Lock init at core-visible base + `0x1F000 + id*0x10`.
- S2MM channel start-queue at core-visible base + `0x1DE04` (BD id + repeat).
Use `volatile uint32_t *` writes. The core-visible base is the unknown to
resolve here (try tile-local direct first: `0x1D000` in the core's own memory
map; if that faults, subtract the `0x70000` processor-view delta noted in
`flowtransfer_kernel.cpp`).

**Step 4: Build + run the spike on HW**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/spike_kernelcfg/host_spike.cc`
then `python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1`.
Expected: `verify_host.sh` pass ("device_teardown done"), incoming data matches.

**Step 5: Record the confirmed encoding**

Update `docs/plans/2026-09-14-kernel-config-offload-design.md` "Risk / validation"
section with the CONFIRMED core-visible register base + any re-arm quirk found.
This is the encoding the codegen tasks (5-7) must emit.

**Step 6: Commit**

```bash
git add example/tileprogram/ccode/spike_kernelcfg/ docs/plans/2026-09-14-kernel-config-offload-design.md
git commit -m "spike: core self-programs S2MM DMA via MMIO after CoreProcessorBusEnable"
```

**GATE:** If the spike fails to land data, STOP and report. Do not proceed to
codegen until the MMIO encoding is confirmed.

---

## Task 2: Runtime helper `__Runtime_enable_core_proc_bus`

**Files:**
- Modify: `src/mlir/runtime/aie_runtime.c` (add helper near `__Runtime_partition_initialize`, ~line 2571; call after `__Runtime_partition_initialize(dev)` at 2630 and 2701)
- Modify: `src/mlir/runtime/aie_runtime.h` (add prototype)
- Test: `src/mlir/runtime/unitest/` (host-native build like `build_ctrl_txn.sh`)

**Step 1: Add the prototype to `aie_runtime.h`**

```c
/* Enable each AIE core tile's processor bus so the core can write its own
 * memory-module DMA/lock registers (required for KERNELCONFIGOFFLOAD). Iterates
 * the partition's core tiles (row >= core-row-min) calling
 * XAie_CoreProcessorBusEnable. Gen5 baremetal only; no-op elsewhere. */
AieRC __Runtime_enable_core_proc_bus(XAie_DevInst *dev);
```

**Step 2: Implement the helper in `aie_runtime.c` (< 200 lines)**

Place after `__Runtime_partition_initialize` (~line 2571). Iterate the
partition's columns (`dev->StartCol .. dev->StartCol+dev->NumCols`) and core
rows (row from the first core row to `dev->NumRows-1`), calling
`XAie_CoreProcessorBusEnable(dev, XAie_TileLoc(col, row))`. Skip on any
non-AIETILE (the API rejects those). Gate the body on
`#if AIE_GEN == 5 && !defined(__AIESIM__)`; else return `XAIE_OK`.

```c
AieRC __Runtime_enable_core_proc_bus(XAie_DevInst *dev) {
#if AIE_GEN == 5 && !defined(__AIESIM__)
    AieRC rc = XAIE_OK;
    for (u8 col = dev->StartCol; col < dev->StartCol + dev->NumCols; col++) {
        for (u8 row = dev->AieTileRowStart;
             row < dev->AieTileRowStart + dev->AieTileNumRows; row++) {
            AieRC r = XAie_CoreProcessorBusEnable(dev, XAie_TileLoc(col, row));
            if (r != XAIE_OK) rc = r;
        }
    }
    return rc;
#else
    (void)dev;
    return XAIE_OK;
#endif
}
```

Verify the exact field names for core-row start/count on `XAie_DevInst` (grep
`AieTileRowStart` / `AieTileNumRows` in the driver headers; adjust if named
`.DevProp` sub-fields).

**Step 3: Wire the call into both init paths**

After `RC = __Runtime_partition_initialize(dev);` at line 2630
(`__Runtime_explicit_init`) and line 2701 (`__Runtime_explicit_init_partition`),
add — gated so default flows are unaffected — a call:
```c
if (RC == XAIE_OK) RC = __Runtime_enable_core_proc_bus(dev);
```
Note: this is safe to call unconditionally (it only enables the proc bus; the
core only writes registers when the offloaded kernel does). Keeping it
unconditional avoids threading the pragma flag into the prebuilt runtime.

**Step 4: Build to confirm no compile regression**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc`
Expected: runtime compiles; existing app still builds (default OFF unaffected).

**Step 5: Commit**

```bash
git add src/mlir/runtime/aie_runtime.c src/mlir/runtime/aie_runtime.h
git commit -m "feat(runtime): __Runtime_enable_core_proc_bus for kernel config offload"
```

---

## Task 3: Pragma plumbing — `#pragma KERNELCONFIGOFFLOAD` → `routing.kernel_config_offload`

Mirror Task E (`CONTROL_PLAN_GROUP_REG_WRITE`) exactly.

**Files:**
- Modify: `src/llvm/aiehlc.cc` (bool ~187; handler class ~3521; register ~3574; setAttr at 4563 and 5244)

**Step 1: Add the parse flag**

After `static bool parsedControlPlanGroupRegWrite = false;` (line 187) add:
```c
// Set by #pragma KERNELCONFIGOFFLOAD. When true, core-tile DMA config (BD, lock
// init, channel-start) is offloaded from host.cc into kernel.cc (self-config via
// raw MMIO). Absent (default) => host programs core DMA as today.
static bool parsedKernelConfigOffload = false;
```

**Step 2: Add the pragma handler class**

After `AieControlPlanGroupRegWritePragmaHandler` (line 3521) add:
```c
// Bare marker pragma: #pragma KERNELCONFIGOFFLOAD (no arguments). Opts core-tile
// DMA config into the kernel path (kernel.cc self-programs via MMIO).
class AieKernelConfigOffloadPragmaHandler : public clang::PragmaHandler {
  public:
    AieKernelConfigOffloadPragmaHandler() : PragmaHandler("KERNELCONFIGOFFLOAD") {}
    void HandlePragma(clang::Preprocessor &PP, clang::PragmaIntroducer, clang::Token &Tok) override {
        parsedKernelConfigOffload = true;
        llvm::outs() << "[aiehlc] Detected #pragma KERNELCONFIGOFFLOAD\n";
        if (Tok.isNot(clang::tok::eod))
            PP.DiscardUntilEndOfDirective();
    }
};
```

**Step 3: Register the handler**

After `PP.AddPragmaHandler(new AieControlPlanGroupRegWritePragmaHandler());`
(line 3574) add:
```c
            PP.AddPragmaHandler(new AieKernelConfigOffloadPragmaHandler());
```

**Step 4: Publish the module attr at BOTH setAttr sites**

After the `routing.control_plan_group_reg_write` setAttr at line 4563
(multi-kernel) and 5244 (single-kernel), add:
```c
                        module->setAttr("routing.kernel_config_offload",
                                        fcAttrBuilder.getI64IntegerAttr(parsedKernelConfigOffload ? 1 : 0));
```
(single-kernel site uses `fcAttrBuilder` at the same indent as its neighbors.)

**Step 5: Build**

Run: `make -C <repo>/build -j$(nproc)`
Expected: aiehlc builds clean.

**Step 6: Verify the flag round-trips**

Add `#pragma KERNELCONFIGOFFLOAD` to a test kernel, run aiehlc, confirm
`[aiehlc] Detected #pragma KERNELCONFIGOFFLOAD` prints and the module attr is
set (grep the dumped IR for `routing.kernel_config_offload`).

**Step 7: Commit**

```bash
git add src/llvm/aiehlc.cc
git commit -m "feat(pragma): #pragma KERNELCONFIGOFFLOAD -> routing.kernel_config_offload"
```

---

## Task 4: Host-side skip in DfscheduleToApiPass (offload ON)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletoapi/passdfscheduletoapi.cpp` (`ConfigDmaBdInnerPattern::matchAndRewrite` ~1197; lock-init emit ~1505; channel-start emit)

**Step 1: Read the module-attr access pattern**

Confirm how the pass reads module attrs (e.g. the existing
`dfschedule.grouped_lock_inits` fold check). The offload flag is
`routing.kernel_config_offload` on the top-level ModuleOp.

**Step 2: Gate the core-tile BD emission**

In `ConfigDmaBdInnerPattern::matchAndRewrite`, when
`routing.kernel_config_offload != 0` AND the BD targets a CORE tile
(row >= core-row-min), ERASE the op WITHOUT emitting the host
`__Runtime_dma_bd_config` call (shim BDs stay host-side). Mirror the existing
core-vs-shim tile classification already used in the pass.

**Step 3: Gate lock-init + channel-start emission**

Similarly skip `XAie_LockSetValue` (core-tile lock init) and
`XAie_DmaChannelSetStartQueue` (core-tile channel start) when offload is ON, for
core tiles only.

**Step 4: Build**

Run: `make -C <repo>/build -j$(nproc)`
Expected: builds clean.

**Step 5: Verify host.cc has no core BD when offload ON; unchanged when OFF**

- OFF (default): diff generated `host.cc` vs baseline — must be byte-identical.
- ON: generated `host.cc` has no `__Runtime_dma_bd_config`/`XAie_LockSetValue`/
  `XAie_DmaChannelSetStartQueue` for core tiles (shim BDs remain).

**Step 6: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletoapi/passdfscheduletoapi.cpp
git commit -m "feat(host): skip core-tile DMA config when kernel_config_offload on"
```

---

## Task 5: Kernel-path incoming (S2MM) + ongoing (ping/pong) BD + lock init (#1)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_kernel.cpp` (route core BD/lock ops to KERNEL module when offload ON)
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp` (emit MMIO writes for core S2MM BD + lock init)

**Step 1: Read how FlowTransferConversion selects the target module**

Read `flowtransfer_kernel.cpp` `emitCoreTileConfigs` + `finalizeKernelConfig` to
find where the `ConfigDmaBdOp`/lock ops are inserted (currently HOST module).
When `routing.kernel_config_offload` is ON, these S2MM (input) BD + lock-init
ops must be emitted into the KERNEL path instead.

**Step 2: Emit S2MM BD chain + lock init as MMIO in kernel path**

In `passdfscheduletokernelapi.cpp`, when offload ON, lower the core S2MM
`ConfigDmaBdOp` (single: next_bd=-1; ping/pong: pong→ping chain) into raw
`volatile uint32_t*` writes at the confirmed core-visible base + `0x1D000`
(BD block, per-BD stride) and lock init at `0x1F000 + id*0x10`, emitted BEFORE
`kernel_invoke`. Use the encoding confirmed in Task 1. Keep each emit helper < 200 lines.

**Step 3: Build**

Run: `make -C <repo>/build -j$(nproc)`

**Step 4: Verify kernel.cc has the S2MM BD + lock MMIO writes when ON**

Generate with `#pragma KERNELCONFIGOFFLOAD`; inspect `kernel.cc` for the
volatile MMIO S2MM BD chain + lock init before `kernel_invoke`.

**Step 5: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/helper/flowtransfer_kernel.cpp src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp
git commit -m "feat(kernel): offload S2MM BD chain + lock init into kernel.cc (MMIO)"
```

---

## Task 6: Kernel-path outgoing (MM2S) BD via per-(col,row) if/else (#2)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp`

**Step 1: Confirm the per-core divergence**

The MM2S out-of-order target BD id (`coreOooBdId`) differs per core tile
(`flowtransfer_kernel.cpp` sets it from `shimPerTileBdIds[tileIndex]` for output
flows). So the shared kernel binary needs a runtime `(col,row)` branch.

**Step 2: Resolve the core's own (col,row) at runtime**

Determine the aie_api / adf intrinsic that yields the core's own tile position
(the pipeline builds ONE shared kernel binary, so a runtime query is required —
NOT a per-tile binary). Grep the AIE headers for a tile-position intrinsic.

**Step 3: Emit the if/else MM2S BD chain**

For each output core tile, emit a branch keyed on the resolved `(col,row)` that
programs that tile's MM2S BD (address, length, packet enable/id, out-of-order bd
id `coreOooBdId`) via MMIO at core-visible base + `0x1D000` (its BD) and
channel ctrl at `0x1DE10`.

**Step 4: Build + verify**

Run: `make -C <repo>/build -j$(nproc)`
Inspect `kernel.cc` (offload ON): an `if/else` chain over `(col,row)` each
programming that tile's MM2S BD.

**Step 5: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp
git commit -m "feat(kernel): offload MM2S BD per-(col,row) if/else into kernel.cc"
```

---

## Task 7: Kernel-path DMA channel start (#4)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp`

**Step 1: Locate the deferred StartIoOp / channel-start**

The host today emits `XAie_DmaChannelSetStartQueue` for core tiles. When offload
ON, emit the S2MM (and MM2S) channel start-queue from kernel.cc instead: write
`S2MM_0_START_QUEUE` (`0x1DE04`) with the first BD id + repeat.

**Step 2: Emit the start-queue MMIO write(s)**

After the BD chain is programmed and before `kernel_invoke`, write the
start-queue register(s) via MMIO with the first BD id + repeat count from the
schedule.

**Step 3: Build + verify**

Run: `make -C <repo>/build -j$(nproc)`
Inspect `kernel.cc` (offload ON): start-queue MMIO write present; host.cc has no
core channel-start.

**Step 4: End-to-end HW run (offload ON)**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file <offload-kernel.cc>`
then `python3 script/test/apppaltest.py -y -nonreboot > ./applog 2>&1`.
Expected: `verify_host.sh` pass; data correct with core self-configured DMA.

**Step 5: End-to-end HW run (offload OFF — regression)**

Run the same flow WITHOUT the pragma. Expected: byte-identical host.cc/kernel.cc
to baseline; HW pass. This proves default behavior is unchanged.

**Step 6: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp
git commit -m "feat(kernel): offload DMA channel start into kernel.cc (MMIO)"
```

---

## Task 8: Documentation (#5)

**Files:**
- Modify: `README.md` (Control-Plane Pragmas section — add KERNELCONFIGOFFLOAD)
- Modify: `CLAUDE.md` (note the pragma alongside the other control-plane pragmas)

**Step 1: Document the pragma in README.md**

In the existing "## Control-Plane Pragmas" section, add a `#pragma
KERNELCONFIGOFFLOAD` entry: what it does (offloads core DMA config into
kernel.cc via MMIO), default OFF, gen5 only, requires `XAie_CoreProcessorBusEnable`
(auto-called by the runtime), scope (core tiles only; shim/memtile stay host-side).

**Step 2: Note in CLAUDE.md**

Add a one-line reference next to the other control-plane pragma notes.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document #pragma KERNELCONFIGOFFLOAD"
```

---

## Verification summary

- **Default OFF regression:** generated host.cc/kernel.cc byte-identical to
  baseline for every existing example; HW pass. (Tasks 4, 7.)
- **Offload ON:** host.cc has no core-tile BD/lock/channel-start; kernel.cc has
  MMIO S2MM chain + lock init + MM2S if/else + channel start; runtime enables
  proc bus; HW data correct. (Tasks 5-7.)
- **Spike-first:** Task 1 confirms the MMIO encoding before any codegen.

## Out of scope (v1)

- MemTile / shim self-config (shim BDs stay host-side).
- Non-AIE2PS generations (gate on gen5).
- Dynamic reconfiguration mid-run (one-shot config at kernel entry only).
