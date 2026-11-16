---
name: kernelperfcheck
description: Why is an offloaded AIE layer slow, and what to change -- read the measured per-layer time (emod.profile), the kernel source, the generated host.cc / kernel.cc / kernel.lst / kernel ELF and aiehlc.log, attribute the time (config-bus writes, kernel ELF + BSS load, partition init/teardown, DMA, compute), and rank fixes with predicted savings. Used by emod.AI_perf_analysis("layerN") and emod.AI_kernel_accelerate("layerN", report); also by hand when a layer's PROF row is far above the conv2dstem reference (185 us launch).
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Kernel performance check (one offloaded layer)

## 0. Is it computing at all?
Before any timing: the board log's `[aie-offload] layer NN: EXIT rc=..` line.
`rc=-4` (`AIEK_ERR_KERNEL`) / all-zero `out[0..7]` / `hash=00000000` = the
kernel is still a backbone (`TODO(compute)`, `_run` returns -1). Its time is
pure launch overhead; say so first -- a later compute fill-in ADDS time.

## 1. Evidence (all produced by the flow; `AI_perf_analysis` collects it)
| What | Where |
|---|---|
| measured ms per call (mean per graph_run) | `emod.profile(ag.ALL)` <- `run(profile=True)`, `arm_build/run_<board>.log` `PROF` rows |
| kernel source, pragmas, partition, mesh | the layer's `source` in `layers/partition.json` |
| generated host program (what the A78 writes) | `<aie_dir>/host.cc`, `<aie_dir>/routing.cc` |
| core code, window sizes | `<aie_dir>/kernel.cc`, `<aie_dir>/build/kernel.lst` |
| core memory image (what the loader writes) | `<aie_dir>/build/kernel` PT_LOAD: text / data / **bss** |
| which pragmas aiehlc honoured | `aiek/<name>/aiehlc.log`: `[aiehlc] Detected #pragma` |

## 2. Cost constants (measured -- cite, do not re-derive)
- **372 ns** per 32-bit `XAie_Write32` over the config bus (non-posted NoC
  round trip): `doc/performance/register_write_cost.md`. An N-word BD = N writes.
- **~2.9 ns** per word over control packets (stream fabric).
- AIE clock 998 MHz; ~2.3 GB/s per shim MM2S channel.
- Reference: conv2dstem 7x7/2 stem on the same 4x4 mesh, **185 us** per launch
  (`doc/performance/conv2dstem_perf_review.md` sect. 7); scalar kernel ~31 ms,
  `mmul<8,8,8>` vector kernel ~0.1 ms compute.

## 3. Checklist (cause -> how to see it -> fix -> typical saving)
1. **Kernel ELF + BSS over MMIO.** No `control_plan_op_control_packet` pragma ->
   `__Runtime_load_kernel_group` (not `_ctrl`) in host.cc. Cost = cores x
   (text+data+bss)/4 x 372 ns -- the BSS (window ping/pong, statics) is written
   word by word too. Fix: the control-plane pragma set below (load moves onto the
   fabric), and/or `aiehlc.sh --skip-bss`. Saving: most of it (tens of ms).
2. **Partition init + teardown on every launch.** No
   `AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN` in `#pragma aie_debug_level` ->
   `XAie_PartitionTeardown` zeroes the partition memories each call, then the next
   call re-initializes. Fix: `#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)`.
   Saving: unmeasured per layer; ms-scale. Measure before/after.
3. **Config writes (BDs, locks, routing) over MMIO.** Count `__Runtime_dma_bd_config*`
   (~8 words each, multidim ~12), `startio`, lock inits in host.cc.
   Fix: `CONTROL_PLAN_GROUP_REG_WRITE`, `control_plan_shim_bd_ctrl`,
   `control_plan_dedicated_shim`, `KERNELCONFIGOFFLOAD` (core programs its own DMA).
   Saving: ~1 ms -> ~0.1 ms.
4. **Control plane prerequisites** (each one silently does nothing if missed):
   partition needs **one extra column** (`{2, 6}` for a 4-col mesh, data plane
   stays on the same columns), `__Runtime_ctrl_high_throughput_enable(1)` before
   device init (conv2dstem's `conv2dstem_control_plane_init()`), the pragma must be
   `Detected` in aiehlc.log (stale aiehlc binary drops unknown pragmas -- skill
   **stalepragmabinary**). `KERNELCONFIGOFFLOAD` on a start col != 0: skill
   **kcocoreidpartition**.
5. **Scalar compute.** kernel.lst with no `VMAC`/`VMUL` and scalar `MAC` = scalar
   kernel (~0.09 % of peak). Fix: `aie::mmul<8,8,8,int8,int8>` + vector requantize
   (conv2dstem.cc / conv2dstem_vec.cc). Saving: ~30 ms -> ~0.1 ms for the stem.
6. **Per-slab host loop re-programming shim S2MM.** host.cc `scf.for`-style loop
   around shim BD config (`OOO output outerRounds loop` in aiehlc.log). Fix:
   `#pragma output_tile_major` (+ host de-tiles). Also a correctness race (skill
   **outputtilemajorrace**).
7. **Host staging / layout transforms** (NCHWc <-> HWC, weight packing) on the A78
   every call. Cached-weight packing (pack once), non-cacheable staging, 32-bit stores.
8. **Core output window != store_buffer block** (out_rounds mismatch) -> hang, not
   slowness; skill **convoutroundsdoublecount**.
9. **Measurement artifacts inside the timed call.** The generated `aie_<symbol>`
   adapter prints `[aie-offload] ... ENTER` / `EXIT ... hash=` on the console
   INSIDE the `PROF` window: at 115200 baud that is ~87 us per character (~14 ms
   for the two lines), plus a serial hash of the whole output (~1.6 ms for 802 KB).
   Subtract them before blaming the AIE; fix by printing outside the timed region.
10. **Count calls, not declarations.** host.cc *declares* `__Runtime_teardown_all`
   but the aieMesh launch path never calls it -- a grep for the name is not
   evidence of a per-call teardown.

## 4. Report (what AI_perf_analysis asks for -- one fenced ```json block)
```json
{"summary": "...", "computing": true,
 "bottlenecks": [{"what": "...", "evidence": "file:line or measured fact", "est_ms": 37.0}],
 "suggestions": [{"id": "S1", "change": "...", "where": "file / pragma / flag",
                  "expected_saving_ms": 35.0, "risk": "low|med|high", "effort": "low|med|high"}],
 "predicted_ms_after": 2.0}
```
Rules: every `est_ms` either measured (say so) or from the constants above with
the arithmetic shown; rank suggestions by saving/effort; never suggest anything
that changes the layer's numerics; a backbone kernel's prediction must say that
compute will be added.

## 5. Optimizing (AI_kernel_accelerate)
Edit only the staged copy. Gates after every round, run by aiegraph, not by trust:
`offload_check` READY (spec, calibrate, emulate) and an aiehlc build of the copy;
then the facts are re-read from the NEW build and the model re-predicted. A
predicted gain is not a measured one: `AI_kernel_accelerate(..., board="palmyra")`
measures it (`run(profile=True)` with the copy offloaded).
