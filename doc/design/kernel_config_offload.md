# KERNELCONFIGOFFLOAD — core self-configured DMA

Gated on the `routing.kernel_config_offload` module attr, set by
`#pragma KERNELCONFIGOFFLOAD`. **Default off. Gen5 (AIE2PS) only.**

When on, the core self-configures **all of its own core-tile DMA** — BD chains, lock
inits, lock config and channel starts, in **both** directions — from `kernel.cc` via
MMIO, instead of the host programming it over the config bus. Encoders live in
`src/mlir/runtime/aie_kernel_runtime.h`: `aie_kc_encode_bd` / `_lock` / `_s2mm_start` /
`_mm2s_start`. `script/kc.sh` already puts that directory on the kernel include path.

## Encode, then apply through TM

The header has two halves, and only the first is portable:

- **Encoders** (`aie_kc_encode_*`) are pure computation over `<stdint.h>` — they fill an
  `AieKcReg[]` of `(tile-local offset, value)` pairs. Host-testable; byte-diffed against
  real aie-rt driver output by `unitest/test_kernel_config.cpp`.
- **Apply** (`core_reg_write` / `_read` / `_write_block`) must reach the tile's registers
  over the **processor bus**, so it delegates to `TM_W`/`TM_R` from
  `src/mlir/runtime/kernel_tm.h`.

> **The address does not select the interface.** The compiler picks it from the pointer's
> memory space, which a numeric cast cannot express. A raw
> `*(volatile uint32_t *)(0x80000 + off) = v` assembles to a plain `ST` that never leaves
> the core — and reads back correctly on-core, so a same-core readback "confirms" a write
> the host still sees as 0. Only `TM_W` lowers to `ST.TM`.
>
> Verify the lowering, not the value: `grep -cE '\.TM' <build>/obj/kernel.lst`. Zero means
> every access degraded and the DMA was never programmed.

Shared register offsets (`TM_BASE_ADDR`, `TM_DMA_BD0_0`, `TM_DMA_S2MM_0_START_QUEUE`,
`TM_LOCK0_VALUE`) are **aliased** from `kernel_tm.h` rather than redefined, so the two
headers cannot drift.

Because `kernel_tm.h` compiles its macros out on a host parse, the one header serves both
the xchesscc kernel build and the host golden-diff test.

See also `pass/passblueprintlowering/README.md`.

---

## Why the two directions differ

There is exactly one `kernel.cc` / one ELF, broadcast to every core tile.

- **S2MM** config is *identical* on all tiles (circuit-switched: no packet id, no
  out-of-order bd) — so it emits straight-line.
- **MM2S** is *not*: `packet_id` is `basePacketId + tileIndex`, and `ooo_bd_id` comes
  from **shim-side** BD allocation. Both are distinct per tile. MM2S therefore emits
  inside a `get_coreid()`-keyed dispatch (`col = id >> 16`, `row = id & 0x1F`), one arm
  per tile.

> **`get_coreid()` is the absolute hardware column; plan columns are partition-relative.**
> The dispatch therefore emits `_col = (_cid >> 16) - <partition start col>`
> (`DfscheduleToKernelApiPass(partStartCol)`). Without the rebase, any partition
> not starting at column 0 sends cores to the wrong arm (wrong `packet_id`) or to no
> arm at all (MM2S never armed, mesh deadlocks with `wait_io TIMEOUT`). See
> `doc/debug/troubleshoot_aie_offload_timeout.md`.

> **MM2S BD length comes from the host plan, not `sizeof()`.** The arm emits
> `CoreOffloadEntry::bdLenBytes` (the host's BD length) and falls back to
> `sizeof(buf)` only when the plan has none. A buffer's C declaration can understate
> its BCF slot (conv2dstem: `v4int8[196]` = 784 B against a 3136 B slot). The core
> then sends short chunks into shim out-of-order BDs that never complete.

## Lock direction asymmetry

This mirrors the host exactly — **swapping it deadlocks**.

| Direction | DMA acquires | DMA releases | Initialized to ppdepth |
|-----------|--------------|--------------|------------------------|
| S2MM | window ACQ lock | REL (init 0) | ACQ |
| MM2S | *swapped* | *swapped* | the **release** lock (the kernel's acquire lock); the DMA's acquire lock stays at 0 |

For MM2S the DMA must wait for the kernel to produce — hence the inverted init.

## Cross-clone channel

The host and kernel passes run on separate `module->clone()`s, so the kernel pass cannot
see what the host computed — and `ooo_bd_id` is not recomputable on the kernel clone at
all.

The host publishes a per-tile plan through the `ResourceMgr::instance()` **singleton**
(`coreOffloadPlan()` / `addCoreOffloadTile`, `CoreOffloadTileConfig`) — the same channel
`coreMemAllocator` already uses.

> **Note:** `BlueprintToSchedulePass` also holds its *own* pass-local `ResourceMgr` for
> BD/lock allocation. Writing the plan there would strand it.

`BlueprintToScheduleKernelPass::attachCoreOffloadPlan` copies the plan onto the
`KernelModuleOp` as the `dfschedule.core_offload_plan` array attr, so it is visible in
the `ir/*.mlir` dumps.

## Who decides, who formats

**`BlueprintToScheduleKernelPass::runOnOperation`** reads the gate **once** — cached
alongside the tiling scalars, before `applyPartialConversion` can strip module attrs.
That pass owns the decision *and* the resources it implies:

- Core-tile BD ids from `KernelResourceManager::allocateCoreWindowBdId` — a counter
  distinct from the per-flow `allocateBdId`. Inputs are numbered first from 0, then
  outputs continue the same counter, so the two never alias.
- The 48+N kernel-intrinsic lock ids converted to 0..15 **hardware** indices
  (`toHardwareLockId`).

It stamps `dfschedule.kernel_config_offload` plus `ping_bd_id` / `pong_bd_id` /
`acquire_lock_hw_id` / `release_lock_hw_id` on **both** input and output `window_def`s.

**`DfscheduleToKernelApiPass`** then only *formats*: `convertMainToEmitC` reads the gate
off the `KernelModuleOp` it operates on; `emitCoreDmaConfigBlocks` reads the BD/lock ids
off `window_def` and the per-tile facts off the plan attr. `emitWindowBdAndLocks` is
shared by both directions. `window_init()` still takes the intrinsic macro names, and BD
base+len come from the core's own C buffer symbols (`(uintptr_t)buf_in_ping_0`,
`sizeof(...)`).

### Fail closed, don't emit stray MMIO

All of these return `LogicalResult` and **fail the build** rather than emitting MMIO
writes to unrelated registers:

- a missing BD id
- a hardware lock id outside 0..15 — `aie_kc_encode_lock` addresses
  `LOCK0_VALUE + id*0x10`, valid only through LOCK15
- an MM2S window with no per-tile plan entries

The BD `LOCK_ACQ_ID` / `LOCK_REL_ID` fields are 4 bits wide: a 48 there would run past
the lock array into `LOCKS_EVENT_SELECTION` and silently truncate.

## BD-id agreement

Two allocators cover one physical BD bank: `KernelResourceManager` on the kernel clone,
and `ResourceMgr::allocateTileBd` (first-free from 0) on the host clone.

They stay disjoint because the host **reserves** what it no longer emits —
`reserveOffloadedCoreBds` (`helper/flowtransfer_kernel.cpp`) allocates-and-discards the
BDs the core self-programs, so a host-programmed flow on the same tile is never handed
one.

> Its count must mirror the kernel's per-window claim: **2 per window, unconditionally**
> — a single-buffer window still burns its pong slot. Keep the two in step if either
> side's BD shape changes.
