# KERNELCONFIGOFFLOAD — S2MM Codegen Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Emit the (already byte-verified) `aie_kernel_runtime.h` MMIO encoder calls into
`kernel.cc main()` so an offloaded AIE core self-configures its **incoming S2MM** BD
chain + lock inits + S2MM channel-start, replacing the host's core-tile S2MM config.
MM2S (outgoing) stays host-side for now.

**Architecture:** Gate on module attr `routing.kernel_config_offload`. Kernel path
(`passdfscheduletokernelapi.cpp::convertMainToEmitC`) emits raw `volatile uint32_t*`
writes at tile-local config offsets, using the encoder in `src/mlir/runtime/aie_kernel_runtime.h` (then `include/aie_kernel_config.h`).
The core sources its own BD base address and length from its already-declared C buffer
symbols (`buf_in_ping_0`, `sizeof(...)`), so the S2MM config is uniform across all core
tiles and needs no runtime `(col,row)`. The host's `removeCoreTileHostDmaChain` becomes
**S2MM-selective** (keeps MM2S). BD-id coordination: kernel allocates S2MM bd-ids in the
same order the host resource manager did for inputs (0..2·nIn−1), so they never collide
with host MM2S bd-ids (which the host allocated *after* inputs, at higher ids).

**Tech Stack:** MLIR (dfschedule dialect, EmitC), C (aie_kernel_runtime.h encoder,
xchesscc kernel), aie-rt gen5 driver (host).

## Decisions (confirmed with user, 2026-09-15)

1. **BD base address / length** in kernel.cc come from the core's own C buffer symbols:
   `addr = (uintptr_t)buf_in_ping_0`, `len = sizeof(buf_in_ping_0)` (bytes). No host
   plumbing of numeric L1 offsets. (One HW check pending: core data-mem pointer == DMA
   BD base address.)
2. **Defer MM2S.** This task offloads only the uniform S2MM incoming BD + lock init +
   S2MM channel-start. MM2S outgoing BD stays host-side. (No `get_coreid` intrinsic
   exists — deferring MM2S avoids that blocker.)

## Ground-truth facts (verified this session)

- Kernel `main()` injection point: `convertMainToEmitC`
  (`passdfscheduletokernelapi.cpp:247`); emit the config block right after
  `klog_init();` and BEFORE the first `window_init` (so the DMA is armed at entry).
- Kernel already declares each window's L1 buffers as C symbols
  (`v4int8 buf_in_ping_0[256];` …) and lock-id macros (`#define LOCK_window_in_0_ACQ 48`).
- `translateToCpp(kernelModule)` runs with NO cleanup pass; leftover top-level
  `dfschedule` config ops are simply not emitted (confirmed by inspecting current
  `aout/worklocal/kernel.cc`, which contains only window_init/kernel_invoke/done).
- S2MM lock init values (host, `passdfscheduletoapi.cpp:1560`): acquire lock =
  `isSingleBuffer ? 1 : 2`; release lock = 0 (default).
- Per-tile BD chain (host `emitCorePingPongBd`,
  `passblueprinttoschedule/helper/flowtransfer_kernel.cpp:679`): pong BD (next→ping) then
  ping BD (next→pong); both acquire `bdAcquireLockId` val −1, release `bdReleaseLockId`
  val 1; input flow ⇒ enable_packet=false, ooo=−1. `coreBdLen = pingPongBufferSize *
  elementSizeBytes` (== `sizeof(buf_in_ping_0)`).
- `removeCoreTileHostDmaChain` (`passdfscheduletoapi.cpp:3220`) currently removes ALL
  core-tile BDs (both directions) — must be made S2MM-selective.
- `ConfigCreateIoOp` carries `channel:I32` + `direction:Str` ("MM2S"/"S2MM") — used to
  classify a BD's direction (trace bd_handle → create_io).

## Open detail to resolve during Task 3

- Exact S2MM channel-start `repeat_count` + `enable_token_issue` the host used: read the
  host channel-start emission (`ConfigCreateIoOp`/`StartIoOp` lowering in
  `passdfscheduletoapi.cpp` ~1700-1830 and the start-queue call it emits) and mirror it.

---

## Task 1: Make `removeCoreTileHostDmaChain` S2MM-selective (host side)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletoapi/passdfscheduletoapi.cpp:3220-3293`

**Change:** classify each core-tile `StartIoOp`/`ConfigCreateIoOp`/`ConfigDmaBdOp` by
direction (via `create_io.direction`), and only remove the **S2MM** chain. Keep MM2S
start_io/create_io/bd and their `schedule.wait` events. The lock-init suppression in
`ConfigDmaBdInnerPattern` must likewise only suppress S2MM (input) lock inits — MM2S
output lock inits (release lock init=2) stay host-side.

**Verify:** build the MLIR test lib; run an existing app flow WITHOUT the pragma → host.cc
byte-identical to baseline (offload off path untouched). With the pragma → host.cc still
contains MM2S core BD + MM2S start-queue + MM2S lock init, but no S2MM core BD/lock/start.

**Commit** immediately after clean build (before any pipeline run).

## Task 2: Plumb S2MM window config into the kernel module

**Files:**
- Modify: `passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp` (KernelParamInfo
  → window_def attrs): add per-input-window `direction`, `dma_channel`, `single_buffer`
  onto the `WindowDefOp` (or `KernelConfigDefOp`) so the kernel codegen can read them.
- Modify: `passdfscheduletokernelapi.cpp` `WindowInfo` + its builder: carry `isInput`,
  `channel`, `singleBuffer` (already has pingBuffer/pongBuffer/acquireLock/releaseLock/
  bufferSize/numRounds).

**Verify:** dump kernel IR; confirm each input `window_def` carries direction=in +
channel. Build clean. **Commit.**

## Task 3: Emit S2MM MMIO config block into kernel.cc main() (gated)

**Files:**
- Modify: `passdfscheduletokernelapi.cpp` `convertMainToEmitC` + `KernelModuleToEmitCPattern`
  (read `routing.kernel_config_offload` off the top-level ModuleOp).
- Reference: `src/mlir/runtime/aie_kernel_runtime.h` (then `include/aie_kernel_config.h`) (encoder, already committed & verified).

**Emitted block** (only when offload on), after `klog_init();`, before first window_init:

```c
#include "aie_kernel_runtime.h"           // emitted once near top of file
// ---- KERNELCONFIGOFFLOAD: self-configure incoming S2MM DMA ----
{
  AieKcReg _kc[8];
  int _n;
  // per input window i (ping bd = 2*i, pong bd = 2*i+1):
  //   pong BD: next -> ping
  _n = aie_kc_encode_bd(_kc, /*bd*/1, (uintptr_t)buf_in_pong_0, sizeof(buf_in_pong_0),
                        /*next*/0, /*acq*/48,-1, /*rel*/49,1, /*pkt*/0,0,0, /*ooo*/-1);
  for (int k=0;k<_n;k++) *(volatile uint32_t*)(uintptr_t)_kc[k].off = _kc[k].val;
  //   ping BD: next -> pong
  _n = aie_kc_encode_bd(_kc, /*bd*/0, (uintptr_t)buf_in_ping_0, sizeof(buf_in_ping_0),
                        /*next*/1, 48,-1, 49,1, 0,0,0, -1);
  for (int k=0;k<_n;k++) *(volatile uint32_t*)(uintptr_t)_kc[k].off = _kc[k].val;
  //   lock inits: acquire=2 (ppdepth) / 1 (single), release=0
  _n = aie_kc_encode_lock(_kc, 48, 2);
  *(volatile uint32_t*)(uintptr_t)_kc[0].off = _kc[0].val;
  _n = aie_kc_encode_lock(_kc, 49, 0);
  *(volatile uint32_t*)(uintptr_t)_kc[0].off = _kc[0].val;
  //   S2MM channel start: start_bd = ping bd, repeat/token per host
  _n = aie_kc_encode_s2mm_start(_kc, /*ch*/0, /*start_bd*/0, /*repeat*/R, /*token*/T);
  *(volatile uint32_t*)(uintptr_t)_kc[0].off = _kc[0].val;
}
```

bd-id scheme: input window index `i` → ping bd `2*i`, pong bd `2*i+1`. Values (lock ids,
channel, buffer symbols, single/ppdepth) come from the per-window info plumbed in Task 2.

**Verify:** regenerate kernel.cc for `simplematmul.cc` WITH pragma → block present, correct
symbols/lock ids/channel; WITHOUT pragma → kernel.cc byte-identical to baseline. Build
clean. **Commit.**

## Task 4: End-to-end HW run + the one address assumption check

**Verify:** run the pragma'd app on HW (`apppaltest.py`). Confirm data-correct (validates
the `(uintptr_t)buf == DMA BD base` assumption). If wrong, fall back to plumbing host L1
offsets (decision fork option B).

## Task 5: Docs

**Files:** `README.md` + `CLAUDE.md` — document `#pragma KERNELCONFIGOFFLOAD` (default off),
scope (S2MM offloaded, MM2S host-side), and the `aie_kernel_runtime.h` encoder.

## Out of scope (this plan)

- MM2S per-`(col,row)` offload (needs a core-position mechanism — separate design).
- MemTile / shim self-config. Non-gen5. Mid-run reconfiguration.
