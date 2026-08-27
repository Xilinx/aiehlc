# Control-Plane Performance Analysis — `aieml_controlperf.cc`

**Date:** 2026-08-26
**Build:** `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/debug/aieml_controlperf.cc --debug-syms --prettydebug`
**Run:** `python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1`
**Target:** Versal AIE2PS (AIE_GEN=5), baremetal (APU/RPU) backend
**Inputs used:** `applog` (control-plane microbenchmark), `data/profile/matmul/*/timeline.{csv,json}`, `aout/worklocal/host.cc`, XAie driver under `thirdparty/alib/aie-rt/`.

---

## TL;DR (root cause, one line)

The workload is **host control-plane latency bound**, not compute- or DMA-bandwidth
bound. Every buffer descriptor (BD), lock, and DMA start is programmed as a series
of **individual, blocking, non-posted 32-bit MMIO writes** from the ARM across the
NoC (~365 ns each, ~9 per shim BD). With ~240 such config/start ops per matmul
iteration (~290 µs of serial host issue) and **no batching**, the AIE cores idle in
`LOCK_STALL` **~98%** of the time and compute only **~1%**.

---

## 1. What the AIE core actually does (timeline)

From `data/profile/matmul/mm2s0/timeline.csv`, aggregating event durations over the
458.8 µs captured window on compute tile `0,3`:

| Core / DMA state                         | Time      | % of window |
|------------------------------------------|-----------|-------------|
| `ACTIVE\|LOCK_STALL` (stalled on a lock) | 450.9 µs  | **98.3%**   |
| pure `ACTIVE` (real compute)             | 4.6 µs    | **1.0%**    |
| DMA MM2S actually running                | ~0 µs     | ~0%         |
| DMA MM2S `STALLED_LOCK`                   | 215.5 µs  | 47%         |

The compute tile does real work ~1% of the time. The remaining ~98% it is parked in
`LOCK_STALL`, waiting for a lock release that only arrives after the host finishes
programming and starting the next DMA. The DMA data movement itself is a tiny sliver,
and the DMA is *itself* lock-stalled ~47% of the time.

**Conclusion:** the tile is starved by the control plane, not limited by compute or
data bandwidth.

---

## 2. Why it is starved — host control cost (microbenchmark)

Every `__Runtime_*` config call is a burst of AXI-MM register writes to the device.
The benchmark (`applog`, `==== AIE control-plane API microbenchmark ====`) isolates
the unit cost:

| API                                | us/call | ≈ # register writes |
|------------------------------------|---------|---------------------|
| `XAie_TileLoc` (CPU only)          | 0.0032  | 0                   |
| `XAie_DmaDescInit` (CPU only)      | 0.0183  | 0 (host struct build) |
| `dma_createio_4` (CPU struct)      | 0.0397  | 0                   |
| `XAie_LockSetValue+LockInit`       | 0.3702  | 1 → **~365 ns / single 32-bit write** |
| `dma_channel_enable_ooo`           | 0.3621  | 1                   |
| `startio` (SetStartQueue)          | 0.3645  | 1                   |
| `wait_io` (idle poll)              | 0.8800  | ~1 read + poll      |
| `dma_bd_config (core, 6-word BD)`  | 2.2875  | ~6                  |
| `dma_bd_config (shim, 9-word BD)`  | 3.3385  | ~9                  |
| `dma_bd_config_multidim (3D)`      | 3.3307  | ~9                  |
| `XAie_DmaWriteBd (raw write only)` | 3.3361  | ~9                  |
| `load_kernel_group_16t (ELF x16)`  | 7368.96 | one-time setup      |
| `launch_kernel_group (enable x16)` | 13.8131 | ~16 enables         |

compare runtime api with the raw register write

```
==== AIE control-plane API microbenchmark (AIE_GEN=5) ====
Each __Runtime_* config call = a burst of AXI-MM register writes to the device.
API                                   iters     total_us      us/call
-------------------------------------------------------------------------
XAie_TileLoc (cpu)                     1000         3.19       0.0032
XAie_DmaDescInit (cpu)                 1000        19.09       0.0191
dma_createio_4 (cpu struct)            1000        39.90       0.0399
-------------------------------------------------------------------------
API vs raw write32              nW     api_us     raw_us   delta_us     ns/wr
-------------------------------------------------------------------------
dma_bd_config (shim,1024)        9     3.3388     3.3522    -0.0134     372.5
dma_bd_config (core,1024)        6     2.2914     2.3012    -0.0098     383.5
dma_bd_config_multidim (3D)      9     3.3306     3.3606    -0.0300     373.4
dma_bd_config_multidim_ooo       9     3.3394     3.3517    -0.0123     372.4
XAie_DmaWriteBd (shim)           9     3.3366     3.3536    -0.0170     372.6
XAie_LockSetValue+LockInit       1     0.3712     0.3838    -0.0126     383.8
dma_channel_enable_ooo           1     0.3622     0.3714    -0.0092     371.4
startio (SetStartQueue)          1     0.3645     0.3684    -0.0039     368.4
wait_io (idle poll, read)        1     0.8796     0.4143     0.4653     414.3
load_kernel_group_16t (ELF x16)          20    147415.26    7370.7631
launch_kernel_group (enable x16)         20       276.15      13.8076
-------------------------------------------------------------------------
--- control overhead / matmul iter (120 bd_config + 60 createio+startio) ---
      api=290.11 us   raw(780 write32)=296.04 us   host_overhead=-5.93 us   raw=379.5 ns/write
==== control-plane microbenchmark done ====
XAie_UpdateNpiAddr()
XAie_UpdateNpiAddr(0xf6d50000)
before XAie_PartitionInitialize

==== AIE control-plane API microbenchmark (AIE_GEN=5) ====
Each __Runtime_* config call = a burst of AXI-MM register writes to the device.
API                                   iters     total_us      us/call
-------------------------------------------------------------------------
XAie_TileLoc (cpu)                     1000         3.19       0.0032
XAie_DmaDescInit (cpu)                 1000        19.09       0.0191
dma_createio_4 (cpu struct)            1000        39.90       0.0399
-------------------------------------------------------------------------
API vs raw write32              nW     api_us     raw_us   delta_us     ns/wr
-------------------------------------------------------------------------
dma_bd_config (shim,1024)        9     3.3388     3.3522    -0.0134     372.5
dma_bd_config (core,1024)        6     2.2914     2.3012    -0.0098     383.5
dma_bd_config_multidim (3D)      9     3.3306     3.3606    -0.0300     373.4
dma_bd_config_multidim_ooo       9     3.3394     3.3517    -0.0123     372.4
XAie_DmaWriteBd (shim)           9     3.3366     3.3536    -0.0170     372.6
XAie_LockSetValue+LockInit       1     0.3712     0.3838    -0.0126     383.8
dma_channel_enable_ooo           1     0.3622     0.3714    -0.0092     371.4
startio (SetStartQueue)          1     0.3645     0.3684    -0.0039     368.4
wait_io (idle poll, read)        1     0.8796     0.4143     0.4653     414.3
load_kernel_group_16t (ELF x16)          20    147415.26    7370.7631
launch_kernel_group (enable x16)         20       276.15      13.8076
-------------------------------------------------------------------------
--- control overhead / matmul iter (120 bd_config + 60 createio+startio) ---
      api=290.11 us   raw(780 write32)=296.04 us   host_overhead=-5.93 us   raw=379.5 ns/write
==== control-plane microbenchmark done ====
```

Two decisive facts:

1. **`XAie_DmaWriteBd` (3.336) ≈ `dma_bd_config` (3.339)** → ~100% of a BD-config's
   cost is the raw register write. The CPU-side descriptor build (`XAie_DmaDescInit`
   = 0.018) is **< 1%**. Optimizing the host struct math is pointless.
2. The single-register ops (Lock / channel-enable / startio) are all ~0.365 µs →
   **one 32-bit MMIO store costs ~365 ns**, dominated by NoC round-trip latency, not
   data volume. BD cost scales linearly with word count:
   - core/tile BD = 6 words × 0.365 ≈ 2.19 µs (measured 2.29)
   - shim BD = 9 words × 0.365 ≈ 3.29 µs (measured 3.34)

---

## 3. Driver confirmation — a BD is N blocking stores, no batching

Path for a shim BD write (AIE2PS):

- `XAie_DmaWriteBd` → `DmaMod->WriteBd` → `_XAie2PS_ShimDmaWriteBd`
  (`thirdparty/alib/aie-rt/driver/src/dma/xaie_dma_aie2ps.c:496`) builds a **9-word**
  array (`XAIE2PS_SHIMDMA_NUM_BD_WORDS = 9`; tile DMA = 6, memtile = 8) and hands it
  to the backend via `XAie_RunOp(..., XAIE_BACKEND_OP_CONFIG_SHIMDMABD, ...)`.
- The baremetal "block write" is **not** a hardware burst — it is a software loop
  (`thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:380`):
  ```c
  static AieRC XAie_BaremetalIO_BlockWrite32(void *IOInst, u64 RegOff,
          const u32 *Data, u32 Size) {
      for (u32 i = 0U; i < Size; i++) {
          XAie_BaremetalIO_Write32(IOInst, RegOff + (u64)(i * 4U), *Data);
          Data++;
      }
      return XAIE_OK;
  }
  ```
- Each `Write32` is one blocking device store
  (`thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:193`):
  ```c
  Xil_Out32(BaremetalIOInst->BaseAddr + RegOff, Value); // non-posted AXI-MM write
  ```

So a shim BD = 9 × ~365 ns ≈ 3.34 µs, **exactly** the measured `dma_bd_config`/
`XAie_DmaWriteBd`. The ARM stalls on every single word; there is no batching.

---

## 4. Per matmul iteration (host.cc)

`aout/worklocal/host.cc` issues, per iteration (verified by grep):

- **120** `__Runtime_dma_bd_config`
- **60** `__Runtime_dma_createio_4`
- **60** `__Runtime_startio`
- (0 `wait_io` in this generated host)

The benchmark aggregates this composite to:

```
--- control overhead / matmul iter (120 bd_config + 60 createio+startio) --- 289.62 us
```

≈ **290 µs of pure serial host register-writing per iteration** — this is what keeps
the core pinned in 98% `LOCK_STALL`. Setup ops are heavier but one-time:
`load_kernel_group_16t` = 7369 µs/call (streaming 16 kernel ELFs into tiles the same
word-by-word way).

---

## 5. Reconciling the "DMA write only ~10%" observation

It is actually more extreme than 10%: the *data plane* (DMA transfer + compute) is
~1–2% of wall time. The ~90%+ elsewhere is **not** hidden work — it is the host
serially issuing hundreds of blocking single-word register writes, during which the
tile sits in `LOCK_STALL`. The "DMA write" looks small because a BD is only 9 words;
its cost is 9 NoC round-trips, not the transfer size.

---

## 6. Optimization levers (in impact order)

1. **Batch register writes into a command/transaction buffer** instead of per-word
   blocking stores. Use the driver transaction API (`XAie_StartTransaction` /
   `XAie_SubmitTransaction`) or a control-code / IPU command-queue backend so one NoC
   transaction programs a whole BD (or a whole iteration's BDs) rather than 9
   round-trips each. This attacks the ~365 ns × N term directly.
2. **Reuse / pre-program BDs.** 120 BD writes/iter means BDs are reprogrammed every
   iteration. Program the BD ring once and re-arm via the task-queue / OOO enqueue
   (`startio` only), dropping most of the 120 × 3.3 µs.
3. **Avoid shim (9-word) BDs on the hot path** where a 6-word tile BD suffices, and
   collapse multi-dim BDs where possible.
4. **Overlap host issue with compute** (double-buffer: issue next-iter BDs while the
   current iter computes) so the ~290 µs host time hides behind DMA/compute instead of
   serializing in front of it.

---

## 7. Source references

| Item | Location |
|------|----------|
| Microbenchmark source | `example/debug/aieml_controlperf.cc` |
| Perf numbers | `applog` (lines ~233–277) |
| Core timeline (stall %) | `data/profile/matmul/mm2s0/timeline.csv`, `timeline.json` |
| Per-iter control calls | `aout/worklocal/host.cc` (120 bd_config / 60 createio / 60 startio) |
| Shim BD word count (=9) | `thirdparty/alib/aie-rt/driver/src/dma/xaie_dma_aie2ps.c:32-34, :496` |
| Baremetal block write loop | `thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:380` |
| Baremetal single MMIO store | `thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:193` |
