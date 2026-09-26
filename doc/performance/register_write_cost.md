# Host↔AIE Register Access Cost

**The number:** one host-issued 32-bit AIE register write costs **~365–384 ns**
on VEK385 / AIE2PS baremetal. A read costs **~384–414 ns**.

Delivering the same write as a **DMA-pushed control packet** costs **~2.9 ns**
— roughly **128× cheaper** in steady state.

Everything below is measured on real hardware, not estimated. If you need to
justify a control-plane design decision, cite this page rather than re-deriving
a number from a timeline span (see [Pitfalls](#pitfalls)).

| | Path | ns / 32-bit write |
|---|---|---|
| Today | host `XAie_Write32` (serial MMIO) | **~372** |
| Batched | control packet via shim DMA | **~2.9** |

Companion docs: [controlperf_analysis.md](controlperf_analysis.md) (why the
array starves), [controlperf_timeline_crosstrack.md](controlperf_timeline_crosstrack.md)
(per-iteration cross-track walkthrough).

---

## 1. Measurement

**Benchmark:** `example/debug/aieml_controlperf.cc`
**Raw capture:** `example/debug/data/aximmwriteperf.txt` (`//vek385-4 REVA`)
**Target:** Versal AIE2PS (`AIE_GEN=5`), baremetal APU/RPU, 1000 iterations/row

```bash
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file ./example/debug/aieml_controlperf.cc \
    --debug-syms --prettydebug
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```

`BENCH2` times each `__Runtime_*` API, then re-issues the **identical word
count** as raw `XAie_Write32` into the same register file. `ns/wr = raw_us / nW`
is therefore one real 4-byte AXI-MM write.

```
API vs raw write32              nW     api_us     raw_us   delta_us     ns/wr
dma_bd_config (shim,1024)        9     3.3377     3.4297    -0.0919     381.1
dma_bd_config (core,1024)        6     2.2876     2.2798     0.0079     380.0
dma_bd_config_multidim (3D)      9     3.3295     3.4398    -0.1102     382.2
dma_bd_config_multidim_ooo       9     3.3385     3.4292    -0.0908     381.0
XAie_DmaWriteBd (shim)           9     3.3350     3.4326    -0.0977     381.4
XAie_LockSetValue+LockInit       1     0.3698     0.3636     0.0062     363.6
dma_channel_enable_ooo           1     0.3620     0.3621    -0.0001     362.1
startio (SetStartQueue)          1     0.3644     0.3819    -0.0175     381.9
wait_io (idle poll, read)        1     0.8798     0.4143     0.4655     414.3
```

The **single-word rows** are the cleanest evidence: lock-set, channel-enable and
startio are each exactly one register write, all landing at ~0.365 µs.

`delta_us` (`api_us − raw_us`) is ~0 or negative throughout ⇒ the `__Runtime_*`
wrapper adds no measurable cost. **Optimizing host-side descriptor math is
pointless**; `XAie_DmaDescInit` is 0.019 µs, under 1% of a BD config.

### Cost is 100% bus, 0% software

`BENCH4` peels off every software layer against the same register:

```
API vs xaie/xil/volatile      nW    api_us   xaie_us    xil_us    vol_us   ns/acc
dma_bd_config (shim,1024)      9    3.3425    3.3508    3.3350    3.3502    372.2
dma_bd_config (core,1024)      6    2.2863    2.2974    2.2815    2.2979    383.0
XAie_DmaWriteBd (shim)         9    3.3347    3.3542    3.3340    3.3506    372.3
dma_channel_enable_ooo (1w)    1    0.3609    0.3712    0.3533    0.3720    372.0
read32 (status, 1w)            1    0.8788    0.4143    0.4059    0.3839    383.9
```

| Layer | What it is | 9-word cost |
|---|---|---|
| `api_us` | full `__Runtime_*` wrapper | 3.3425 µs |
| `xaie_us` | `XAie_Write32` (driver + backend dispatch) | 3.3508 µs |
| `xil_us` | `Xil_Out32` (BSP inline, one volatile access) | 3.3350 µs |
| `vol_us` | bare `*(volatile u32*)` — zero software | 3.3502 µs |

All four are within noise. A bare compiler store costs the same as the full
driver call, so `ns/acc = vol_us / nW` **is** the pure AXI-MM bus cost.

---

## 2. Why ~372 ns

Two independent causes, both recorded in the capture.

### The aperture is Device-nGnRnE (non-posted)

From the `PAR_EL1`-after-`AT`-translate probe at the top of `aximmwriteperf.txt`:

```
[AIE BD reg] VA=0x02000031d000  S1E1R PAR=0x000002000031d900 -> Device-nGnRnE
[DDR local ] VA=0x00004026b79c  S1E3R -> Normal memory (cacheable)
```

`Device-nGnRnE` means non-gathering, non-reordering, **no early write
acknowledgement** — the ARM stalls on each store for a full NoC round trip. The
cost is latency, not data volume, which is why a 1-word write and a 9-word BD
cost the same *per word*.

### `BlockWrite32` is a software loop, not a burst

`thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:380`:

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

each iteration reaching a single blocking store at `:193`:

```c
Xil_Out32(BaremetalIOInst->BaseAddr + RegOff, Value); // non-posted AXI-MM
```

So **an N-word block write is N serial round trips.** There is no batching
anywhere in this path.

### Consequence: cost scales linearly with word count

| Descriptor | Words | Predicted (N × 0.365) | Measured |
|---|---|---|---|
| Core/tile BD | 6 | 2.19 µs | 2.29 µs |
| Shim BD | 9 | 3.29 µs | 3.34 µs |
| MemTile BD | 8 | 2.92 µs | — |

Word counts: `XAIE2PS_TILEDMA_NUM_BD_WORDS = 6`, `SHIMDMA = 9`, `MEMTILEDMA = 8`
(`driver/src/dma/xaie_dma_aie2ps.c:32-34`).

**Estimating rule:** `cost ≈ (number of register words) × 0.37 µs`.

---

## 3. The control-packet path (~128× faster)

Recorded in commit **`ec05657`** (2026-09-03):

> the perf is 10000 4 bytes configure spend 29us,
> 1.38GB without count pkt header and control header,
> 2GB after count per 4 control data have 1 control header and 1 pkt header

**10 000 register writes in 29 µs ⇒ ~2.9 ns per write**, 1.38 GB/s of payload
(2 GB/s counting packet + control headers).

Instead of N blocking MMIO stores, the host builds a packet buffer once and the
**shim DMA** streams it into the array, where the tile's CTRL port applies each
write. The ARM issues one descriptor, not N round trips.

Driver path: `__Runtime_ctrl_pktize_write` → `__Runtime_ctrl_push`, or the
transaction API (`__Runtime_control_start_transaction` →
`_write_pkt_commit_transaction` → `__Runtime_control_push`). See
[../controlplane.md](../controlplane.md).

### What the 29 µs includes — read this before extrapolating

The timer (`aieml_controlperf.cc`, the `MINI-A` block) brackets **only**
`__Runtime_ctrl_push` plus the completion drain-poll. It **excludes**:

- `__Runtime_ctrl_setup_routing` (fabric programming — one-time, amortized)
- the host-side fill of the packet buffer
- `__Runtime_alloc_buffer` / `sync_for_dev`

It is a **steady-state throughput** number over a large batch. 10 000 writes
amortize the fixed per-push cost far better than a few hundred will, so small
batches land meaningfully worse than 2.9 ns/write. Treat 2.9 ns as a floor and
measure your own batch size before promising a specific speedup.

---

## 4. Worked example — deriving a ~68 µs config window

A KERNELCONFIGOFFLOAD matmul build (`simplematmul2_offload.cc`, 4×4 mesh,
3 windows). This walks the whole derivation, so you can repeat it on your own
build.

### Step 1 — bound the region

The round-loop body in the generated `host.cc` runs from the `for` at line 83 to
the wait at line 286. The two `__Runtime_core_trace_event` markers bracketing it
are `dma_start` (emitted at line 93) and `wait_start` (line 286).

### Step 2 — count the calls

```bash
sed -n '83,286p' aout/worklocal/host.cc \
  | grep -oE '__Runtime_(dma_bd_config_multidim_ooo|dma_bd_config_multidim|dma_channel_enable_ooo|startio|dma_createio_4)\b' \
  | sort | uniq -c
```

```
      4 __Runtime_dma_bd_config_multidim
     20 __Runtime_dma_bd_config_multidim_ooo
      4 __Runtime_dma_channel_enable_ooo
     12 __Runtime_dma_createio_4
     12 __Runtime_startio
```

### Step 3 — convert calls to register writes

Words per call come from the BD width of the **target tile**. All 24 BD configs
here target row 0 (verify with `XAie_TileLoc(c, 0)` — see step 5), so each is a
**9-word shim BD**:

| Call | Count | × writes each | = writes |
|---|---|---|---|
| `__Runtime_dma_bd_config_multidim_ooo` | 20 | 9 (shim BD) | 180 |
| `__Runtime_dma_bd_config_multidim` | 4 | 9 (shim BD) | 36 |
| `__Runtime_dma_channel_enable_ooo` | 4 | 1 | 4 |
| `__Runtime_startio` | 12 | 1 (start queue) | 12 |
| `__Runtime_dma_createio_4` | 12 | 0 — CPU-only struct build | 0 |
| **Total** | | | **232** |

### Step 4 — two independent predictions

**(a) writes × unit cost** — uses only the ns/write number:

```
232 writes × 0.372 µs/write = 86.3 µs
```

**(b) calls × measured per-call cost** — uses the `api_us` column directly, so
it also captures the CPU-only `createio`:

```
20 × 3.3385 µs (bd_config_multidim_ooo) = 66.77 µs
 4 × 3.3295 µs (bd_config_multidim)     = 13.32 µs
 4 × 0.3620 µs (channel_enable_ooo)     =  1.45 µs
12 × 0.3644 µs (startio)                =  4.37 µs
12 × 0.0399 µs (createio_4, CPU only)   =  0.48 µs
                                          -------
                                           86.4 µs
```

The two agree to 0.1 µs (86.3 vs 86.4), which is the useful cross-check: it
confirms the word-count model *and* shows the CPU-side work is negligible
(0.48 µs of 86.4 = 0.6%).

### Step 5 — compare against the measurement

From the timeline CSV (`src/tool/debug/timeline.py <applog>`):

```
iter0.dma_start   = 5923.949 µs
iter0.wait_start  = 5992.400 µs
                    ---------
measured span     =   68.45 µs
```

**68.45 µs measured vs 86.3 µs predicted — the measurement is ~21% lower.**

The predictions are *upper* bounds and the measured span is a *lower* bound; they
bracket the truth rather than contradicting each other:

- The markers do not bracket exactly the same work as the tight benchmark loop
  (`dma_start` is emitted at line 93, after the first few calls).
- The benchmark runs 1000 iterations of one call type; the real loop interleaves
  five, with different cache/branch-predictor behaviour.

**Do not invert this to get ns/write.** `68.45 ÷ 232 = 295 ns` is ~21% below the
real 372 ns, for exactly the reasons above. See [Pitfalls](#pitfalls).

### Step 6 — sanity-check against the AIE side

If the window really is host-bound, the array must be idle through it. Aggregating
the tile-lane rows over the same 5923.9→5992.4 µs span:

| Lane | State | Share of window |
|---|---|---|
| tile 0,3 core | `ACTIVE\|LOCK_STALL` | **99.8%** |
| tile 0,3 south slave 0 | `PORT_STALLED_0` | 94.7% |
| tile 0,3 dma s2mm 1 | `STREAM_STARVATION` | 79.1% |
| tile 0,3 core | pure `ACTIVE` (real compute) | **0.4%** |

The core computes 0.4% of the window and waits the rest — confirming the 68 µs is
host-issue time, not AIE time. (De-duplicate rows first if the applog contains
more than one `[TIMESYNC]` dump, or the overlap totals double-count.)

The larger workload in [controlperf_analysis.md](controlperf_analysis.md) shows
the same shape at 5× the scale: 120 `bd_config` + 60 `startio` per iteration,
98.3% lock-stall, 1.0% compute, ~290 µs of serial host issue.

### If it were batched

```
232 writes × 0.0029 µs = 0.67 µs   (floor — see §3 on what the 29 µs excludes)
```

### Why KERNELCONFIGOFFLOAD did not remove this

It moves *core-tile* DMA config into `kernel.cc` — and it worked: zero core-tile
BD configs remain in `host.cc`. But every BD counted above targets **row 0
(shim)**, which a core cannot program (it has no path to another tile's
registers, and `ooo_bd_id` is allocated shim-side).

Resolve the tile each BD config targets — this is step 3's word-count assumption,
so check it before trusting the 9:

```bash
h=aout/worklocal/host.cc
for v in $(sed -n '83,286p' $h \
    | grep -oE '__Runtime_dma_bd_config[a-z_]*\(v1, v[0-9]+' \
    | grep -oE 'v[0-9]+$' | sort -u); do
  grep -m1 -oE "XAie_LocType $v = XAie_TileLoc\([0-9]+, *[0-9]+\)" $h
done
```

```
XAie_LocType v13 = XAie_TileLoc(0, 0)
XAie_LocType v19 = XAie_TileLoc(1, 0)
XAie_LocType v25 = XAie_TileLoc(2, 0)
XAie_LocType v31 = XAie_TileLoc(3, 0)
```

All row 0 ⇒ all shim ⇒ 9 words each. (A row > 0 target would be a core tile at
6 words, or a MemTile at 8 — see §2.)

See [../design/kernel_config_offload.md](../design/kernel_config_offload.md).

---

## 5. Optimization levers

1. **Batch config into control packets** — the only lever that attacks the
   ~372 ns unit cost. ~128× in steady state.
2. **Issue fewer writes.** Hoist loop-invariant BD fields; if only the buffer
   address changes per round, most of the 9 words are rewritten with identical
   values every iteration.
3. **Do not optimize host-side struct math.** Measured at <1% of BD cost.
4. **Re-typing the aperture** (`AIE_ATTR_PATCH` in `aieml_controlperf.cc:1002`,
   an EL3 page-table experiment, default off) — **no recorded result**. Unknown
   whether it helps; non-posted ordering is likely required for correctness.

---

## Pitfalls

- **Do not derive ns/write by dividing a timeline span by a write count.** Doing
  that on the example above yields 295 ns (68.45 µs ÷ 232) — ~21% low, because
  trace markers bracket different work than the benchmark. The microbenchmark is
  the authority.
- **Reads are not cheaper than writes** (~384–414 ns). A status poll costs about
  the same as a config write, which is why `wait_io` polling is expensive.
- **Word count drives cost, not payload size.** A 1024-byte and a 4-byte shim BD
  both cost 9 writes.

---

## Source references

| What | Where |
|---|---|
| Benchmark source | `example/debug/aieml_controlperf.cc` (`BENCH2` ~:338, `BENCH4` ~:364, column defs :355-362) |
| Raw board capture | `example/debug/data/aximmwriteperf.txt` (untracked) |
| Control-packet result | commit `ec05657` message; example at `aieml_controlperf.cc:541` (`WDATA_NUM 10000`) |
| Memory-type rationale | `aieml_controlperf.cc:965-969`, `:1002` |
| Block-write loop | `thirdparty/alib/aie-rt/driver/src/io_backend/ext/xaie_baremetal.c:380`, `:193` |
| BD word counts | `thirdparty/alib/aie-rt/driver/src/dma/xaie_dma_aie2ps.c:32-34` |
| Analysis docs | commits `d389dc9` (created), `f8a2c8b` (table), `a106c35` (timeline) |

### Not yet captured

- No `[PERF]` PMU output is committed anywhere, though the profiling layer
  exists (`--profiling` / `-DAIEHLC_PROFILING=1`; phases `PH_KLOAD` / `PH_BDCFG`
  / `PH_COREEN` / `PH_STARTIO` in `src/mlir/runtime/aie_runtime.c:104`,
  consumers in `example/tileprogram/ccode/simplematmul2_prof.cc`).
- The control-packet 2.9 ns figure exists **only** in a commit message — no test
  output file. Re-running `aieml_controlperf.cc` and capturing it alongside
  `aximmwriteperf.txt` would close that gap.
