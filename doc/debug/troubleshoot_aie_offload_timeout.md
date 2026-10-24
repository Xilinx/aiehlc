# Troubleshooting: `--aie-offload` main.elf hangs with `wait_io TIMEOUT`

Symptom: `deploy_flow.py --aie-offload` builds `main.elf` (layer 01, the conv
stem, on the AIE), but

```
python3 script/test/apppaltest.py -y -nonreboot worklocal/tvmrelay_deploy/arm_build/main.elf > applog 2>&1
```

never prints `top1`. The console shows `[aie-offload] layer 01 ... ENTER` with no
matching `EXIT`, then `wait_io TIMEOUT` on shim channels and a PLM EAM error.

## 1. Read the hang correctly

```
[aie-offload] layer 01 ...: ENTER aiehlc conv2dstem (...), 9 args
[aie_runtime] wait_io TIMEOUT tile(4,0) ch=0 dir=0 pending=1
```

* **ENTER appears twice: that is not two launches.** apppaltest captures stdout
  and the com0 UART, so every console line can appear twice. `main.c` calls
  `graph_run()` exactly once.
* `dir=0` is S2MM: the host is waiting for **output** that never arrives.
* The dump (enable it with `#pragma aie_debug_level(AIE_DEBUG_LOG)`, see §5) showed
  the shape of the deadlock:
  * core tiles: `LOCK_STALL_E: YES`; their input S2MM channels `STALLED stream=1`;
  * a shim MM2S on the feed side parked on an **empty BD** (`addr=0x0 len=0`);
  * the shim S2MM output channels are idle and wait on a stream that never comes.

  **Input never reaches the cores → cores block on locks → no output → timeout.**
  This is a real array deadlock. Raising the timeout will not fix it.

## 2. Code review before touching the board

| Question | How | Finding |
|---|---|---|
| Did `allconv` fix this? | `git log main..allconv`, `git diff main allconv` | `allconv` = main + 2 commits. Its `wait_io TIMEOUT` fix (`convgemm.cc:36-50`) is for the **convgemm** kernel: a widened int32 C port made buffer sizes disagree with the windows. Carrying int32 inside the proven int8 layout fixed it. Its `conv2dstem.cc` is **byte-identical to main's committed one**. |
| What does the board actually run? | `git diff -- src/aietensorop/conv2dstem/` | Uncommitted WIP in `conv2dstem.cc`: five control-plane pragmas (`control_plan_op_control_packet`, `CONTROL_PLAN_GROUP_REG_WRITE`, `control_plan_dedicated_shim`, `control_plan_shim_bd_ctrl`, `KERNELCONFIGOFFLOAD`) and partition `{3,6}` → `{2,6}`. |
| Was a pragma silently dropped (stale aiehlc)? | `grep "Detected #pragma" layers/01_*/aiehlc_stem.log`; `grep -cE '\.TM' .../kernel.lst` | All five detected, `control plane: dedicated shim col 0 ... data plane uses cols [1,4]`, `.TM` = 144. The build did what the source asked. |
| Is the offload glue wrong? | read `aie_<kernel>()` at the end of `layers/01_*/aie/host.cc` | Arg decode, qparam fold and the call to `conv2d_stem_nchwc()` are correct. |

From this review the suspect is the control-plane WIP, not the TVM side.

## 3. Bisect on hardware

| Run | Stem source | Context | Result |
|---|---|---|---|
| A | WIP (control plane, `{2,6}`) | `--aie-offload` main.elf | **hang**, `wait_io TIMEOUT` |
| B | committed (no control plane, `{3,6}`) | `--aie-offload` main.elf | **pass**: `EXIT rc=0`, `top1 class=258 Samoyed logit=12.017`, `device_teardown done` |
| C | WIP (control plane, `{2,6}`) | standalone `test_conv2d.cc` | **hang**, same TIMEOUT signature as A |

Swapping only `conv2dstem.cc` flips the result (A vs B), so the WIP is the cause.
C shows the hang does not depend on the TVM offload. The control-plane
configuration of this mesh deadlocks on its own.

Recipe for run B, without losing the WIP:

```bash
cp src/aietensorop/conv2dstem/conv2dstem.cc debugcache/offload_timeout/   # backup
git stash push -- src/aietensorop/conv2dstem/conv2dstem.cc
python3 src/frontend/tvmrelay/deploy_flow.py --aie-offload
python3 script/test/apppaltest.py -y -nonreboot worklocal/tvmrelay_deploy/arm_build/main.elf > applog 2>&1
git stash pop
```

Recipe for run C: `buildtest.sh --skip-trace` stops at step 3 when
`arm_build/layeriohex/` is gone. If the data headers are already in the app
directory, source aiehlc directly:

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file src/aietensorop/conv2dstem/test_conv2d.cc
python3 script/test/apppaltest.py -y -nonreboot aout/main.elf > applog 2>&1
```

## 4. Root cause and fix

### 4.1 Pragma bisection (standalone, run C)

| Pragmas on (partition `{2,6}`) | Result |
|---|---|
| all five | hang |
| without `shim_bd_ctrl` + `dedicated_shim` | hang |
| without those **and** `KERNELCONFIGOFFLOAD` | **GOLDEN PASS 802816/802816** |
| `op_control_packet` + `GROUP_REG_WRITE` + `dedicated_shim` + `shim_bd_ctrl` (KCO off) | **GOLDEN PASS 802816/802816**, 31.7 ms/call |

So the trigger is **`KERNELCONFIGOFFLOAD` (KCO)**. All three control-packet
features (ELF load + launch, grouped lock writes, shim-BD control) work on
`{2,6}`. KCO is not a control-packet feature: the core programs its own DMA over
its local bus.

### 4.2 Defect 1 (fixed): KCO dispatches on the absolute column

With KCO, `kernel.cc` arms each core's output MM2S inside a `get_coreid()`
dispatch, one arm per tile, carrying that tile's `packet_id` / `ooo_bd_id`.
`get_coreid()` returns the **absolute** hardware column, but the arms were
emitted at **partition-relative** columns (the host's coordinate space). With
partition start col 2 they are off by 2:

| Core (rel col) | hw col | Arm taken | MM2S in the hang dump |
|---|---|---|---|
| 1, 2 | 3, 4 | the arm for **col 3 / col 4**: wrong packet id | `Paused`, armed |
| 3, 4 | 5, 6 | **none** | `Idle running=0` |

The pattern held for all 16 cores. Under the old `{3,6}` partition KCO was
never enabled, so the mismatch never surfaced. It is the kernel-side twin of
`de26c81` (host `__Runtime_enable_core_proc_bus`, relative vs absolute columns).

**Fix:** `DfscheduleToKernelApiPass` now takes the partition start column
(`tilinglinalg_pipeline.cpp` passes `partStartCol`) and emits
`int _col = (int)(_cid >> 16) - <start>;`. After the fix all 16 cores arm
(`Paused running=1`), so the fix is confirmed on the board.

### 4.3 Defect 2 (fixed): KCO MM2S BD length came from `sizeof()`

After fix 1, cores sent their first buffers, then stalled on output
backpressure. Diffing the KCO-off host program for core (1,3) against the KCO
arm found it:

| | host (KCO off) | core arm (KCO on) |
|---|---|---|
| output MM2S BD 4/5 length | **3136** | `sizeof(buf_out_ping_0)` = **784** |
| input S2MM lengths | 4636 / 3136 | 4636 / 3136 (match) |

`kernel.cc` declares `v4int8 buf_out_ping_0[196]` (784 B). The real buffer is
3136 B: the BCF places ping/pong 0xC40 apart, and the host plan publishes
`len=3136`. KCO was the only consumer of `sizeof()`, so it alone sent 784-byte
chunks into shim OOO BDs that expect 3136 and never complete. **Fix:**
`emitWindowBdAndLocks(..., lenBytes)` takes the host plan's
`CoreOffloadEntry::bdLenBytes` for the MM2S arms and falls back to `sizeof()`
only when the plan has none. Verified standalone: all five pragmas,
`GOLDEN PASS 802816/802816`, 31.47 ms per call (31.73 ms with KCO off).

### 4.4 Red herrings ruled out on the way

* **Column off-by-one in host/routing.** A first look showed cols 0..3, but those
  files came from the committed-stem build that had overwritten them. The
  control-plane build uses cols 1..4 everywhere. Always check *which* build
  produced a file before reading it.
* **Empty shim BD 2 / 6 on the ch1 feed** (`RT_RES_CTRL_BD` range). It persists
  with `shim_bd_ctrl` off, and it is downstream of the core stall.
* `rt_cpe_seg_write` burst encoding, row-0 port booking, and the ctrl ELF
  loader (which runs on the spine column, not on data shims).

### 4.5 Result on the target command

`deploy_flow.py --aie-offload`, then
`apppaltest.py -y -nonreboot worklocal/tvmrelay_deploy/arm_build/main.elf`:

| Check | Value |
|---|---|
| layer 01 on the AIE | `ENTER` … `EXIT rc=0`, hash `2aeeab65` (same as the committed stem) |
| control plane | `op_control_packet`, `GROUP_REG_WRITE`, `dedicated_shim`, `shim_bd_ctrl` and `KERNELCONFIGOFFLOAD` detected |
| classification | top-1 `class=258 Samoyed logit=12.017338`, same top-5 order as the onnxruntime reference |
| `graph_run()` | **1271 ms** (was 2232 ms; see 4.6) |
| teardown | `device_teardown done`, no `wait_io TIMEOUT` |

### 4.6 Inference time: the board ELF was built `-Os`

With the hang fixed, `graph_run()` took 2232 ms, of which the AIE stem is about
32 ms. The rest is TVM's C on the A78, and the generated Makefile compiled it
`-Os`. At `-O3` it takes 1282 ms with bit-identical logits. The default is now
`-O3` in `arm_build.py`'s Makefile template. To try other flags, edit the
generated Makefile and run **`make clean` first**: the objects do not depend
on `CFLAGS`, so a plain `make` relinks stale `-Os` objects.

Follow-up on the board, with KCO on and output bit-identical in every case:

| CFLAGS | `graph_run()` |
|---|---|
| `-O3` | 1281.9 ms |
| `-O3 -flto` | 1281.2 ms |
| `-O3 -funroll-loops` | **1270.4 ms**, now the default |

KCO itself saves about 0.26 ms per stem call (31.47 vs 31.73 ms), which is below
the noise of the whole graph. Essentially all of `graph_run()` is TVM's CPU C.

## 5. Logging: off by default, one flag to turn it on

`rt_wait_io_timeout_dump()` (`src/mlir/runtime/aie_runtime.c`) used to print
per-tile core status, DMA channels, BD contents and locks with no gate. One hang
produced about 1.6 MB of UART text. It is now gated on the existing runtime log
flag:

* default: only the one-line `wait_io TIMEOUT tile(c,r) ch dir pending` plus
  `DMA status dump suppressed (enable: #pragma aie_debug_level(AIE_DEBUG_LOG))`;
* to enable: OR `AIE_DEBUG_LOG` (bit 9) into the app's pragma, for example
  `#pragma aie_debug_level(2 | AIE_DEBUG_LOG | ...)`. aiehlc then emits
  `g_runtime_debug_level` with that value.

Checked on the board in run C: the dump collapsed to the single suppressed line.

## 6. Checklist for the next occurrence

1. Look for `EXIT rc=` after `ENTER`. If it is missing, the hang is inside the AIE launch.
2. Ignore duplicated console lines. apppaltest has two capture channels.
3. Before suspecting the toolchain, read `layers/01_*/aiehlc_stem.log` for
   `Detected #pragma` and check that the `.TM` count is non-zero.
4. Swap in the committed `conv2dstem.cc` (§3 run B). If that passes, the
   regression is in the stem source.
5. Reproduce standalone (§3 run C). It is faster, and it separates "kernel or
   config" from "offload integration".
6. Turn on `AIE_DEBUG_LOG` only for the run whose dump you will actually read.
