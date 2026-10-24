# ResNet-18 layer 01 on AIE: `wait_io TIMEOUT` debug summary

Step-by-step record of how the `--aie-offload` hang was found and fixed.
Reference detail (dumps, tables, recipes) is in
[debug/troubleshoot_aie_offload_timeout.md](debug/troubleshoot_aie_offload_timeout.md).

## Problem

```
python3 src/frontend/tvmrelay/deploy_flow.py --aie-offload
python3 script/test/apppaltest.py -y -nonreboot worklocal/tvmrelay_deploy/arm_build/main.elf > applog 2>&1
```

Layer 01 (the 7x7/s2 conv stem, `conv2d_stem_nchwc()` on a 4x4 AIE mesh) printed
`[aie-offload] ... ENTER` but never `EXIT`. The console then showed
`wait_io TIMEOUT` on shims (3,0)/(4,0), a PLM EAM error, and about 1.6 MB of
DMA dump. No classification was printed.

## Final result

| Requirement | Result |
|---|---|
| no timeout | `EXIT rc=0`, `device_teardown done`, 0 `wait_io TIMEOUT` |
| control packets kept on | `op_control_packet`, `GROUP_REG_WRITE`, `dedicated_shim`, `shim_bd_ctrl` active, plus `KERNELCONFIGOFFLOAD` (after root cause 2) |
| classification | top-1 `class=258 Samoyed logit=12.017338`, top-5 matches the onnxruntime reference |
| inference | `graph_run()` **1271 ms** with KCO + `-O3 -funroll-loops` (was 2232 ms; 1282 ms at `-O3`) |

## Steps

### 1. Read the symptom precisely

* `ENTER` without `EXIT` means the hang is inside the AIE launch, not in TVM.
* Each console line appears twice because apppaltest captures stdout and the
  UART. It is one run, not two launches.
* `dir=0` is S2MM: the host is waiting for output.
* The dump showed cores at `LOCK_STALL_E`, core S2MM starving, and shim MM2S
  backpressured. That is a real deadlock; a longer timeout would not help.

### 2. Code review before using the board

* **allconv branch** (`git log main..allconv`): its `wait_io TIMEOUT` fix was
  for the *convgemm* kernel (a widened int32 port whose buffer sizes disagreed
  with the windows). Its `conv2dstem.cc` is identical to main's committed file,
  so that fix does not apply here.
* **What the board runs:** an uncommitted `conv2dstem.cc` adding five
  control-plane pragmas and widening the partition `{3,6}` → `{2,6}`.
* **Stale tool?** No. All pragmas showed `Detected`, the dedicated shim was
  active, and the `.TM` count was non-zero.
* **Offload glue** (`aie_<kernel>()` in the layer's `host.cc`): correct.

### 3. Bisect source against context on hardware

| Run | Stem | Context | Result |
|---|---|---|---|
| A | control-plane WIP | offload | hang |
| B | committed stem | offload | pass, Samoyed |
| C | control-plane WIP | standalone `test_conv2d.cc` | hang |

The WIP is the cause (A vs B), and the TVM offload is not involved (C).
Standalone builds are faster to iterate: `source script/aiehlc.sh --aie-version 5
--runtime-source-file src/aietensorop/conv2dstem/test_conv2d.cc`, then run
apppaltest on `aout/main.elf`.

### 4. Bisect the pragmas (standalone, `{2,6}`)

| Pragmas on | Result |
|---|---|
| all five | hang |
| without `shim_bd_ctrl` / `dedicated_shim` | hang |
| without those and `KERNELCONFIGOFFLOAD` | **pass** |
| the four control-packet pragmas, KCO off | **pass**, 31.7 ms/call |

The trigger is **`KERNELCONFIGOFFLOAD` (KCO)**, where the core programs its own
DMA. It is not a control-packet feature.

### 5. Root cause 1 (fixed): KCO dispatched on the absolute column

With KCO, `kernel.cc` arms each core's output DMA inside a `get_coreid()`
dispatch, one arm per tile. `get_coreid()` returns the **absolute** hardware
column, but the arms were emitted at **partition-relative** columns. With
partition start 2:

* cores in rel cols 1-2 took the arm for cols 3-4 (wrong packet id);
* cores in rel cols 3-4 matched no arm, so their MM2S stayed `Idle running=0`.

The dump showed exactly this split on all 16 cores. **Fix:**
`DfscheduleToKernelApiPass(partStartCol)` emits
`_col = (get_coreid() >> 16) - <partition start col>`. After rebuilding
`build/aiehlc`, all 16 cores armed on the board. This is the kernel-side twin
of commit `de26c81`.

### 6. Root cause 2 (fixed): KCO output BD length taken from `sizeof()`

After fix 1, cores sent their first buffers, then stalled. Diffing the host's
core-tile program (KCO off) against the KCO arm for one tile: the output MM2S
BD is **3136 B** from the host but `sizeof(buf_out_ping_0)` = **784 B** in the
arm. `kernel.cc` declares the buffer `v4int8[196]`, while the BCF slot and the
host plan say 3136. **Fix:** the arm uses the host plan's `bdLenBytes`. KCO now
passes with all five pragmas, `GOLDEN PASS`, 31.47 ms/call versus 31.73 ms
without KCO.

### 7. Inference time

Once it ran, `graph_run()` was 2232 ms, and only about 32 ms of that is the AIE
stem. The board Makefile compiled TVM's C at **`-Os`**. At **`-O3`** it takes
1282 ms with bit-identical logits. That is now the default in
`src/frontend/tvmrelay/arm_build.py`. When trying flags by hand, run
`make clean` first: objects do not depend on `CFLAGS`.

Follow-up on the board, with KCO on and output bit-identical in every case:

| CFLAGS | `graph_run()` |
|---|---|
| `-O3` | 1281.9 ms |
| `-O3 -flto` | 1281.2 ms |
| `-O3 -funroll-loops` | **1270.4 ms**, now the default |

KCO itself saves about 0.26 ms per stem call (31.47 vs 31.73 ms), which is below
the noise of the whole graph. Essentially all of `graph_run()` is TVM's CPU C.

### 8. Logging default off

`rt_wait_io_timeout_dump()` (`src/mlir/runtime/aie_runtime.c`) printed the full
per-tile dump unconditionally. It is now gated on the existing flag: by default
you get the one-line `wait_io TIMEOUT ...` plus a "suppressed" hint. To get the
full dump, add `AIE_DEBUG_LOG` to the app's `#pragma aie_debug_level(...)`.

## Lessons

1. Check `ENTER`/`EXIT` first to locate the hang.
2. Swap in the committed source before suspecting the toolchain.
3. Reproduce standalone. It is faster and isolates kernel/config from integration.
4. Bisect pragmas one at a time and confirm each build with `Detected #pragma`.
5. Before reading a generated file, check *which build* wrote it. An early
   "column off-by-one" lead came from files overwritten by another build.
6. When a channel runs a BD or arm the code did not choose, suspect
   coordinate or register mapping, not lock protocol (skills
   **kcocoreidpartition**, **shimbdindexoverflow**).
7. Measure where time goes before optimizing. Here it was CPU compile flags,
   not the AIE.

## Files changed

* `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.{h,cpp}`,
  `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp`: KCO column rebase
* `src/mlir/runtime/aie_runtime.c`: timeout dump gated on `AIE_DEBUG_LOG`
* `src/frontend/tvmrelay/arm_build.py`: board `CFLAGS` `-Os` → `-O3`
* `src/aietensorop/conv2dstem/conv2dstem.cc`, `README.md`: KCO off, documented
* `doc/debug/troubleshoot_aie_offload_timeout.md`, `doc/design/kernel_config_offload.md`,
  `.cursor/skills/kcocoreidpartition/SKILL.md`, `CLAUDE.md`
