# Kernel Config Offload

(Moved from the root README.md.)

`#pragma KERNELCONFIGOFFLOAD` (no arguments, off by default) moves the core
tile's **incoming S2MM** DMA configuration off the host and into the AIE core
itself. Add the pragma to your `tilinglinalg` source file to enable it:

```cpp
#pragma KERNELCONFIGOFFLOAD
```

| Pragma | Module attr | Effect |
|--------|-------------|--------|
| `#pragma KERNELCONFIGOFFLOAD` | `routing.kernel_config_offload` | The core self-configures its incoming S2MM DMA from `kernel.cc` via raw MMIO instead of the host programming it over the config bus. The host's `removeCoreTileHostDmaChain` becomes S2MM-selective (drops the core-tile S2MM BD chain + lock inits + channel-start), while MM2S stays host-side. Absent => the host programs all core-tile DMA as before. |

## What gets offloaded

When enabled, `convertMainToEmitC` emits a raw-MMIO block into `kernel.cc`
`main()` — after `klog_init()` and before the first `window_init` — that
self-configures every **incoming (S2MM)** window:

- **Ping/pong BD chain** (or a single BD for single-buffer windows). Input
  window `i` (in `window_def` declaration order) claims ping bd `2*i` /
  pong bd `2*i+1` — the same `0..2·nIn−1` range the host resource manager used
  for inputs, so kernel-side S2MM bd-ids never collide with host-side MM2S
  bd-ids (allocated at higher ids).
- **Lock inits**: acquire lock = ping-pong depth (`2`, or `1` single-buffer),
  release lock = `0` — mirroring the host `emitCorePingPongBd`.
- **S2MM channel-start** on the window's DMA channel.

BD base address and length come from the core's own C buffer symbols
(`(uintptr_t)buf_in_ping_0`, `sizeof(buf_in_ping_0)`), so the block is uniform
across all core tiles and needs no runtime `(col,row)`. The encoder lives in
[`src/mlir/runtime/aie_kernel_runtime.h`](../../src/mlir/runtime/aie_kernel_runtime.h)
(`aie_kc_encode_bd` / `aie_kc_encode_lock` / `aie_kc_encode_s2mm_start`), which
applies the encoded words through `core_reg_write` →
[`kernel_tm.h`](../../src/mlir/runtime/kernel_tm.h)'s `TM_W` (a plain pointer cast
would never reach the processor bus).

## Scope and limitations

- **S2MM only.** The outgoing **MM2S** BD stays host-side (per-`(col,row)`
  offload needs a core-position mechanism — a separate design). MemTile / shim
  self-config, non-gen5 targets, and mid-run reconfiguration are out of scope.
- Gen5 (AIE2PS) only.
