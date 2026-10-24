<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# conv2dstem — the ResNet-18 stem convolution on the AIE array

A fused int8 7×7/s2 convolution (3 → 64 channels, 224×224 → 112×112) running on
a mesh of AIE core tiles, plus the host-side library that stages its buffers.

It exists twice over, and that is the point:

* as a **standalone app** (`test_conv2d.cc`) that reproduces one layer of the
  compiled ResNet-18 graph and checks itself against that layer's real output;
* as a **static library** (`libconv2dstem.a`) that TVM's BYOC path calls in
  place of its own kernel when you run `deploy_flow.py --aie-offload`.

Both are built from the same `conv2dstem.cc`.

---

## Quick start

```bash
cd src/aietensorop/conv2dstem
./buildtest.sh                      # regenerate the data headers, then build
python3 script/test/apppaltest.py aout/worklocal/build/host    # run on hardware
```

Expect, on the console:

```
[conv2dstem] PERF first call: 148.512 ms  (includes partition init, staging
             allocation and the weight transpose -- not the layer's cost)
[conv2dstem] PERF iter 1/5: 31.204 ms
...
[conv2dstem] PERF SUMMARY iters 5  first 148.512  min 31.198  mean 31.240  max 31.301  host 2.914  (ms)
[conv2dstem] GOLDEN PASS 802816/802816
conv2dstem done (rc=0).
... device_teardown done
```

`GOLDEN PASS 802816/802816` means every output byte matches what TVM's own
kernel produced for the same layer. Anything else is a real failure — the
comparison is bit-exact, not approximate. (The `PERF` numbers above are
illustrative formatting, not a reference measurement.)

---

## The files

### `conv2dstem.cc` — the kernel and the host library

The AIE kernel (`conv2d_spatial`, launched over a mesh) **and** everything the
host does around it: allocating DMA-capable staging buffers, packing weights
into the layout the kernel indexes, scattering the image into the
channel-aligned input buffer, and copying the result back out.

It has **no `main()`**, so building it directly archives it into
`libconv2dstem.a` instead of linking an ELF (skill: **hostlibrarymode**):

```bash
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file src/aietensorop/conv2dstem/conv2dstem.cc
```

That is what the TVM BYOC path consumes. You do **not** need this for the
standalone app — `test_conv2d.cc` `#include`s the whole file.

Three public entry points, differing **only in buffer layout** — the
arithmetic is identical:

| entry | input | weights | output | used by |
|---|---|---|---|---|
| `conv2d_stem()` | `int8[224,224,3]` raw | `int8[64,7,7,3]` OHWI | `uint8[112,112,64]` HWC | callers that want the library to pad |
| `conv2d_stem_prepadded()` | `int8[230,230,3]` already padded | same | same | **TVM BYOC** (`--aie-offload`) |
| `conv2d_stem_nchwc()` | `int8[230,230,3]` already padded | `int8[16,7,7,3,4]` OIHW3i4o | `uint8[16,112,112,4]` NCHW4c | **`test_conv2d.cc`** |

Plus `conv2d_stem_verify()` (on-board CPU cross-check),
`conv2d_stem_invalidate_weights()` and `conv2d_stem_release()`.

`conv2d_stem_nchwc()` speaks TVM's *packed* layouts so the app can be fed the
graph's own tensors verbatim. It is a **separate entry on purpose**: it and
`conv2d_stem_prepadded()` differ only in layout, so merging them would silently
reinterpret every weight and output byte for the offload path — which fails as
wrong pixels, not as a link error.

#### Control plane — host→array configuration off the config bus

A 32-bit `XAie_Write32` is a non-posted Device-nGnRnE NoC round trip at **~372 ns**
([doc/performance/register_write_cost.md](../../../doc/performance/register_write_cost.md)),
and an N-word BD is N serial writes, so on this layer the host's register traffic
is a first-class cost term, not an error bar. Five marker pragmas take it off the
config bus, each gating one MLIR pipeline stage:

| Pragma | What stops being MMIO | Proof in the generated code |
|---|---|---|
| `control_plan_op_control_packet` | the plan itself: reserves control-plane slots/arbiters/msel/pkt-ids, and lowers **kernel ELF load + core launch** onto the fabric | `__Runtime_ctrl_plan_init`, `__Runtime_load_kernel_group_16t_ctrl`, `__Runtime_launch_kernel_group_ctrl` in `host.cc` |
| `CONTROL_PLAN_GROUP_REG_WRITE` | per-tile lock inits, folded into broadcast/row group writes | `__Runtime_ctrl_row_write_ack` |
| `KERNELCONFIGOFFLOAD` | **all** core-tile DMA — BD chains, lock init/config, channel starts, both directions — the core self-programs it over the processor bus | `KERNELCONFIGOFFLOAD S2MM/MM2S` blocks in `kernel.cc`; **`grep -cE '\.TM' aout/worklocal/build/kernel.lst` must be non-zero** |
| `control_plan_shim_bd_ctrl` | the shim BDs, out-of-order channel enable and start-queue writes, sent along row 0 as control packets | `__Runtime_ctrl_shim_bd_begin` / `_commit` |
| `control_plan_dedicated_shim` | nothing — a **force flag** (see below) | build fails instead of warning |

Two non-obvious things, each of which silently does nothing if you get it wrong:

**The partition is `{2, 6}` — five columns for a four-column mesh.** The spare
column is not slack. `planControlColumn` grants a dedicated control spine only
when `partition cols >= mesh cols + 1`, and the spine is what
`control_plan_shim_bd_ctrl` runs its row-0 packets along. The data plane does not
move: the planner sets `dataStartCol = relStartCol + 1`, so the mesh still
occupies absolute columns 3..6 exactly as it did under the old `{3, 6}`. Narrow it
back and shim BDs silently return to MMIO — which is why
`control_plan_dedicated_shim` is there: alone, `control_plan_shim_bd_ctrl` only
prints a WARNING and falls back. Confirm in the build log:

```
[TilingLinalg] control plane: dedicated shim col 0 (MM2S ch0 / S2MM ch0;
    shim-row BD control on MM2S ch1 / S2MM ch1); data plane uses cols [1,4]
```

**`__Runtime_ctrl_high_throughput_enable(1)` is required, not a tuning knob.** On
AIE2PS it disables CTRL TLAST errors at partition init and packs all
self-delimiting accesses into one shim MM2S BD with a single final TLAST.
`control_plan_shim_bd_ctrl` does not work without it, and the ELF load degrades to
one BD and one TLAST per access. It must run *before* device init, so
`conv2dstem_control_plane_init()` is called at the top of both launch sites, ahead
of their `aieSetDevice`/`partition` pair. It also pins the provenance map off —
that defaults to reading the `AIE_CTRL_PMAP` env var, so a stray export in the
operator's shell would print a line per packet and wreck the `PERF` numbers.

**`control_plan_shim_bd_ctrl` needs a current `aiehlc`.** It postdates
`release/prebuild/aiehlc` and older `build/aiehlc` trees; an unknown pragma is a
warning, not an error, so a stale tool accepts the file and silently omits the
feature. Check `[aiehlc] Detected #pragma ...` — all **five** must appear (`KERNELCONFIGOFFLOAD` on `{2,6}` relies on two compiler fixes; see `doc/debug/troubleshoot_aie_offload_timeout.md`) — or
`strings build/aiehlc | grep control_plan_shim_bd_ctrl`. Rebuild with
`make -C build CMAKE_COMMAND=/scratch/staff/huaj/aiehlc/.venv/bin/cmake aiehlc -j16`
(skill: **mlirbuildsandbox**).

Not enabled here: `#pragma control_packet_mode(aot)`, which precomputes the
compile-time part of every control packet at build time (skill:
**ctrlpkt-aot-elf**).

### `conv2dstem_vec.cc` — the vectorized kernel

Same file as `conv2dstem.cc` outside the kernel section. The kernel uses
`aie::mmul<4,8,8,int8,int8>` and a bit-exact vector requantize; 1.77 ms per call
against 31.47 ms for the scalar kernel, golden-exact on the board. Build with
`./buildtest.sh --vec`, which generates `test_conv2d_vec.cc` (gitignored) from
`test_conv2d.cc` with the one kernel `#include` swapped — a macro cannot choose,
because aiehlc splices `.cc` includes before the preprocessor runs. See
`doc/performance/conv2dstem_perf_review.md` §6.

### `conv2dstem.h` — the library contract

Geometry macros, the `conv2dstem_qparam` struct, the entry declarations and the
error codes. Read this first; it documents each layout and the requantization
convention in detail.

### `test_conv2d.cc` — the standalone app

The **aiehlc entry file**: it owns `main()` and `#include`s `conv2dstem.cc`, so
one command produces `host.cc`, the extracted kernel and a linked ELF. (The
frontend splices `.cc` includes into the text before its rewrites run — skill
**aiehlcincludekernel**.)

It runs the deployed graph's **layer 01**,

```
01_contrib_conv2d_NCHWc_subtract_add_subtract_fixed_point_multiply_per_axi_1a1eefcf0256215e
```

on the AIE and compares against that layer's own output. So this is not a model
of the operator — it is the operator, on the real data the network carries.

What it does: decode the layer's six int32 parameter arrays into
`conv2dstem_qparam[64]` (`fold_qparams`), call `conv2d_stem_nchwc()`, **measure**
(`run_perf`), then compare all 802,816 output bytes and print one
machine-readable verdict line (`data/groundtruth.py --applog` parses it back off
the console, which is the only channel a baremetal board under xsdb has).

#### What `run_perf()` measures

Three numbers, because quoting the wrong one is the usual way an AIE layer gets
misreported:

| Number | What is inside it |
|---|---|
| **first call** | partition init over the config bus, the staging allocations, the 12544-element weight transpose. A bring-up number, never the layer's cost. |
| **steady state** (`CONV2DSTEM_PERF_ITERS`, default 5) | input staging, the *full* DMA/BD + kernel-ELF configuration, the launch, the wait, the output repack. aiehlc re-programs the mesh on **every** launch, so repeating amortizes none of that away — this is the per-inference latency of calling `conv2d_stem_nchwc()`, not an array throughput. |
| **host share** | `stage_ifm_prepadded()` + `collect_ofm_nchwc()` timed alone. The remainder is reported as `config+launch`, *not* as "AIE compute": at ~372 ns per 32-bit `XAie_Write32` ([doc/performance/register_write_cost.md](../../../doc/performance/register_write_cost.md)) the register traffic is a first-class term, not an error bar. |

Wall clock is the Arm generic timer (`XTime` / `COUNTS_PER_SECOND`, reached via
`conv2dstem.cc`'s `xiltimer.h` — deliberately **not** `aie_timer.h`, which
branches on `AIE_GEN`, a macro the host compile defines but the aiehlc frontend
parse does not).

The perf loop runs **before** the golden check, so the check validates the last
launch rather than only the first; the inputs never change, so `ofm` holds the
same bytes either way. A failing iteration aborts the loop and poisons `rc` —
perf numbers for a failed launch would be worse than none.

A second block — per-launch `kload` / `bdcfg` / `coreen` / `startio` / `wait_io`
PMU cycles — prints only when the host was built with
`source script/aiehlc.sh --profiling ...` (`-DAIEHLC_PROFILING=1`). The
accessors always link; without the flag every counter reads 0, which the report
says in words instead of printing as a row of zeros. The counters read
`pmccntr_el0`, which the standalone BSP boot never starts; the runtime's
`__Runtime_init` constructor enables it under the flag (this file cannot call
`aie_timer.h`'s `__ps_pmccntr_enable()` itself). Built with the flag but still all
zero is reported separately as "built in but ... not running". Those are PMU cycles and
the wall clock is the generic timer, so they are shown as a share of the
accounted total and never mixed into one ratio.

Set `-DCONV2DSTEM_PERF_ITERS=0` to time only the first call. Raising it
multiplies board time, and `apppaltest.py` watchdogs a run at 300 s without
console output / 600 s total.

### `buildtest.sh` — build the app, regenerating its data first

`test_conv2d.cc`'s two data headers are **generated artifacts**, not source, so
building by hand fails on a missing `#include`. This script produces them and
then builds:

| step | what |
|---|---|
| 1 | `make clean-local && make local TRACE=1` in `worklocal/tvmrelay_deploy/arm_build` |
| 2 | run `./main_local.elf` there — it writes `layeriohex/` |
| 3 | copy `l01_…_in.h` / `_out.h` in as `conv2d_stem_in_weight.h` / `conv2d_stem_out_golden.h` |
| 4 | `source script/aiehlc.sh` on `test_conv2d.cc` |

```bash
./buildtest.sh                 # all four steps
./buildtest.sh --skip-trace    # reuse the layeriohex already on disk (1-2 are slow)
./buildtest.sh --no-aiehlc     # stop once the headers are in place
./buildtest.sh -h
```

Env: `AIE_VERSION` (default 5), `TRACE_HEX_PARAMS` (default 0 — the standalone
`param_*.h` set is ~72 MB and is not needed, because every parameter this layer
uses is already one of its own inputs).

It needs `worklocal/tvmrelay_deploy/arm_build` to exist; if it does not, run the
TVM flow once:

```bash
PYTHONPATH=src python3 src/frontend/tvmrelay/deploy_flow.py --local
```

### `parameter.h` — vestigial

The same geometry macros (`INPUT_H`, `KERNEL_H`, `OUTPUT_H`, `K`, …) that
`conv2dstem.h` already defines. **Nothing includes it** — verified by grep
across `src/` and `script/`. Do not add it to a translation unit that already
has `conv2dstem.cc`: the two spell the same values with different token
sequences, which is a macro redefinition error, not a silent duplicate.

### `data/` and `ref/`

`data/` generates the older hand-built fixture (real photo → preprocessed,
quantized, zero-point-padded image; real conv1 weights; CPU ground truth) and
holds the cross-checks against TVM's layer 0. It has its own
[README](data/README.md) — read that before touching anything in it.

`ref/` is the CPU reference used during bring-up: `stem_ref.py` (the ground
truth), `conv2dstem_x86.c` and `verify_pack.c`.

---

## The generated `*.h`

**Nothing in this table is source.** All of it is produced by tooling and
overwritten; do not hand-edit, and regenerate rather than patch.

| file | shape / dtype | size | produced by |
|---|---|---|---|
| `conv2d_stem_in_weight.h` | layer 01's **8 inputs** (see below) | ~1.0 MB | `buildtest.sh` step 3, from `layeriohex/` |
| `conv2d_stem_out_golden.h` | `uint8[1,16,112,112,4]` — the golden | ~4.8 MB | same |

Both are `layeriohex` dumps: `graph_driver.c` writes every kernel input and
output as a compilable C hex header under `make TRACE=1` (see
`worklocal/tvmrelay_deploy/arm_build/README.md`).

`conv2d_stem_in_weight.h` carries the whole layer:

| var | dtype / shape | role |
|---|---|---|
| `…_in0` | `int8[1,1,230,230,3]` | activation, already spatially padded with the **input zero-point** (−15 = uint8 113 − 128), not with zero |
| `…_in1_p0_weight` | `int8[16,1,7,7,3,4]` | weights, OIHW3i4o |
| `…_in2_p1_zero_point` | `int32[64]` | subtracted |
| `…_in3_p2_bias` | `int32[64]` | added |
| `…_in4_p3_zero_point` | `int32[64]` | subtracted — **all zero** for this model |
| `…_in5_p4_requant_multiplier` | `int32[64]` | Q31 multiplier |
| `…_in6_p5_unused` | `int32[64]` | the kernel never reads it |
| `…_in7_p6_requant_shift` | `int32[64]` | shift, applied as `shift + 31` |

The older fixture headers — `conv2dstem_image.h`, `conv2dstem_weights.h`,
`conv2dstem_golden.h`, `conv2dstem_image_tvm.h` — come from `data/*.py` and are
**not used by `test_conv2d.cc`**. See `data/README.md` if you need them.

> Generated headers are untracked but **not** `.gitignore`d, so a blanket
> `git add .` will commit ~6 MB of them. There are also stale copies inside
> `data/` from before `buildtest.sh` existed; those are not read by anything.

---

## How to test

### On hardware (the real check)

```bash
./buildtest.sh
python3 script/test/apppaltest.py aout/worklocal/build/host
```

Pass is `GOLDEN PASS 802816/802816` **and** `device_teardown done`. The run also
prints a `PERF SUMMARY ... (ms)` line; for the per-launch cost breakdown rebuild
with `./buildtest.sh --profiling`. Mismatch
coordinates are printed as NCHW4c (`occ, oh, ow, ocb`, with `oc = occ*4 + ocb`),
capped at 16 lines — the cap is not cosmetic, since 802,816 lines over a UART
would blow through apppaltest's watchdog.

### On the host, without a board

The entire host-side chain — parameter decode, weight packing, output scatter —
can be checked with a plain `gcc` by extracting the real functions from
`conv2dstem.cc` and `test_conv2d.cc` by name and substituting the library's own
`scalar_conv2d_ref()` for the AIE. That separates a layout bug from a kernel
bug, which are otherwise indistinguishable in the output. Recipe in skill
**conv2dstemtvmlayer**.

### Not a regression test

`buildtest.sh` builds and flashes; it does not assert. Read the console.

---

## Things that silently produce wrong results

1. **A generated header in a subdirectory.** `aiehlc.sh` collects user headers
   with a *non-recursive* `*.h` glob flattened to basename, so a header left in
   `data/` parses in the frontend and then fails at cross-g++. Keep them flat in
   this directory (skill: **conv2dstemfixture**).
2. **`make local TRACE=1` without `clean-local` first.** The Makefile's objects
   do not depend on `CFLAGS`, so a previous *untraced* build is simply relinked,
   compiles none of the trace code and writes no dumps at all. `buildtest.sh`
   handles this.
3. **`aiehlc.sh` writes `aout/` relative to the current working directory**, not
   to the entry file. Run it from this directory and you get
   `src/aietensorop/conv2dstem/aout/` while an older `$REPO/aout/` sits
   untouched — so a "did it build" check can pass against a *stale* ELF.
   `buildtest.sh` pins the cwd and requires the ELF to be newer than the run.
4. **Padding with 0 instead of the zero-point.** In the quantized domain, real
   zero is the zero-point (−15 here), not integer 0. Padding with 0 injects
   ≈ +0.279 around the whole border.
5. **A non-zero second zero-point.** TVM subtracts *two* (`acc + bias − zp_a −
   zp_b`) while `conv2dstem_qparam` has one field; the fold is only exact
   because `zp_b` is identically zero for this quantization.
   `fold_qparams()` **checks this per channel and refuses** rather than
   computing wrong pixels — if it ever fires, the algebra needs redoing, not the
   check removing.

## See also

- [`data/README.md`](data/README.md) — the fixture generators and their cross-checks
- `worklocal/tvmrelay_deploy/arm_build/README.md` — `make TRACE=1` and the `layeriohex` dumps
- Skills: **conv2dstemtvmlayer**, **conv2dstemfixture**, **hostlibrarymode**,
  **aiehlcincludekernel**, **byocaieoffload**
