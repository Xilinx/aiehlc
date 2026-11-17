---
name: outputtilemajorrace
description: Spatial-halo conv (conv2dstem, or a kernel-only aie::shape-port conv such as simpleconv2dkernel_port.cc) gives GOLDEN FAIL only after the kernel got faster (e.g. switched to aie::mmul) while x86 emulation passes (whole 3136-byte output tiles wrong, pattern per mesh column, passes when the core is slowed with a spin); host.cc has an scf.for re-programming shim S2MM OOO BDs per height slab. Also: KCYC cycle stamps, mismatch map, deferred cache sync, warm ELF reload, non-cacheable staging.
---

# Output BD re-programming race (conv2dstem)

## Symptom
`GOLDEN FAIL` with whole rounds wrong (test_conv2d `MISMAP` shows ~2.5k mismatches
per round for some (col,row,round)), plausible values, identical between runs of
the same build, gone when each kernel round is slowed (20k-cycle spin).

## Cause
Dense HWC output = 2-D per-core round walk (width x height); a BD has one
iteration dim, so `buildOutputTileDescriptor` folds width into the BD and makes
height a host `scf.for` that rewrites the S2MM OOO BDs between slabs. Fast cores
send the next slab while BDs are rewritten -> data lands at stale addresses.

## Fix
`#pragma output_tile_major` (aiehlc.cc -> `routing.output_tile_major` ->
`flowtransfer_common.cpp`): each core's rounds contiguous, one BD per core,
`totalRounds=1`, straight-line schedule. Host converts layout (`collect_ofm_*`).
`ScheduleSequentialOpPass` starts S2MM (sinks) before MM2S (sources).

## Measuring
- `#define CONV2DSTEM_KCYC 1` (unconditional literal; the kernel extractor drops
  `#define`s inside `#if`). Cores stamp cycles into their last tile; test prints
  `KCYC core N`. GOLDEN then fails on exactly those bytes.
- Never chain a board run after a failed build: `apppaltest -y` silently picks a
  stale `debugcache/kco_scratch` ELF that can hang the board.

## Runtime pieces added alongside (aie_runtime.c)
- `__Runtime_sync_for_dev_deferred` / `__Runtime_dsync_drain` (emitted by both
  `__aie_launch` emitters). Do NOT drain inside `__Runtime_sync_for_cpu`: the
  runtime calls it for every ctrl ack token (kload 28 -> 84 us).
- `__Runtime_buffer_device_written` (invalidate-only, overlapped with wait_io).
- `__Runtime_alloc_uncached` (2 MB NC block below 4 GB) for CPU-written inputs.
- Warm ELF reload (skips program-memory accesses; keyed on blob/fabric/dev/
  partition gen/`g_elf_load_seq`); `ctrl_plan_init` snapshot cache.

## Kernel-only kernels hit it too (2026-11-14)
`example/tileprogram/ccode/simpleconv2dkernel_port.cc` (aie::mmul<8,8,8>, dense
HWC output) with a golden-checking host: **FAIL 483,593 / 802,816** on palmyra,
while `emu_conv2d_spatial.sh` on the same build says 0 mismatches (x86 has no DMA
timing). Every wrong output equals the right value of the same position in
another round (core (0,3) round (0,0) held round (1,0)'s block): the generated
`host.cc` still has the per-height-slab `for` loop re-programming the S2MM BDs.
The scalar twin (`_less`, ~2 ms per round) never races. `#pragma
output_tile_major` works unchanged on a kernel-only source (the kernel's
`store_buffer` keeps image coordinates); the host de-tiles
`[mesh row][mesh col][round i*4+j][7][28][16]`:
`simpleconv2dkernel_port_vec_opts.cc` PASS 3/3, 56.4 ms. The original
`simpleconv2dkernel_port.cc` host prints only `[BENCH]`, so it never showed.
Tutorial: `tutorial/aieprogram/06_gemm_optimization/conv_vector_scheduling.md` section 5.

Telling it apart from the intermittent post-DMA stale-cache issue: the race is
**reproducible** (hundreds of thousands of mismatches every run, values from
another round); the cache issue is a few hundred outputs reading as the host's
`memset` value, on one run in several (`jane/post-dma-inval`). Seen again on
`simpleconv2dkernel_port_bf16.cc` (2026-11-14): 191 outputs = 0 on 1 of 6 runs,
12/12 bit-exact with `Xil_DCacheInvalidateRange(output)` after the launch.
