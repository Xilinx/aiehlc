## Role
- MLIR and Spatial computing expert

# AIEHLC Project Architecture

AIEHLC (AIE High-Level Compiler) is a compilation/deployment solution for AMD Versal AI Engine. It has **two parts**:

1. **aiehlc** — Clang-based tool that compiles C++ (AIE driver C API) into host + kernel binaries for single-kernel AIE apps.
2. **tilinglinalg** — MLIR progressive lowering pipeline that offloads a GEMM operation across multiple AIE tiles, generating host, kernel, and routing C++ code through 6 custom dialects.

Both share the AIE runtime (`include/aie_runtime.h`) and target Versal AI Core Series with pre-built PDI.

## Directory Layout

```
aiehlc/
├── include/                   # aie_runtime.h, aie_device_map.h
├── src/
│   ├── llvm/aiehlc.cc         # Main aiehlc Clang tool
│   └── mlir/
│       ├── runtime/aie_runtime.c  # Runtime wrappers over XAie_* APIs
│       └── mlirfront/
│           ├── AieFrontEnd.cc     # Clang AST → MLIR
│           ├── AieDialect.cc      # AIE dialect (LoadKernel, etc.)
│           ├── frontend/aiebackend/  # _aiebackend pybind over TilingLinalgPipeline (Triton + TVM share it)
│           └── tilinglinalg/      # ★ Multi-tile GEMM pipeline
│               ├── routing/       # Abstract routing dialect
│               ├── routinghw/     # Physical routing dialect
│               ├── dataflowmap/   # dmap, dmaphop, dfscheblueprint, dfschedule dialects
│               └── pass/          # All lowering passes + unitest/
│                   └── passblueprintlowering/  # host+kernel blueprint lowering
│                       ├── helper/             # SHARED core-tile DMA/lock config
│                       ├── passblueprinttoschedule/       # → host.cc
│                       └── passblueprinttoschedulekernel/ # → kernel.cc
├── script/                    # setup.sh, aiehlc.sh, kc.sh, test/apppaltest.py
├── example/                   # AIE examples (perf, matmul, multi-kernel)
└── thirdparty/alib/           # XAie driver (excluded from this doc)
```

## Part 1: aiehlc (Single Kernel)

Compiles a user C++ file into a host ELF (ARM) + kernel ELF (AIE core):

```
User C++ → aiehlc (Clang AST) → AieFrontEnd (MLIR)
  → Kernel: xchesscc → xchessmk → kernel ELF (embedded in host via ld -r -b binary)
  → Host: aarch64-g++ (host.cc + aie_runtime.c + routing + kernel.o) → host ELF
```

Key runtime API: `__Runtime_device_init`, `__Runtime_load_kernel_group`, `__Runtime_launch_kernel_group`, `__Runtime_dma_bd_config`, `__Runtime_wait_event`, `__Runtime_device_teardown`.

**Entry file may `#include` its kernel.** The `__global__` kernel does not have to
sit in the file passed to `--runtime-source-file`: a thin `main.cc` may
`#include "kernel.cc"`. `inlineSourceIncludes()` (`src/llvm/aiehlc.cc`) splices
quoted `.cc`/`.cpp` includes into the text *before* the rewrites run, so the
kernel is annotated, body-guarded and launch-lowered as if written inline. The
artifact kind follows `main()`: present → ELF, absent → `lib<app>.a` (skill:
hostlibrarymode). `src/aietensorop/conv2dstem/` ships both entries over one
kernel. The splice is deliberately **opaque** (no `#line`) because kernel export
and the host `RewriteBuffer` both key off the kernel's `FileID` — see skill
**aiehlcincludekernel** before changing it.

**Build it with `src/aietensorop/conv2dstem/buildtest.sh`**, not by calling
aiehlc directly: the entry file's two data headers are *generated* artifacts
(`layeriohex` dumps) and are not committed, so the script regenerates them —
`make local TRACE=1` in `worklocal/tvmrelay_deploy/arm_build` → run
`main_local.elf` → copy `l01_..._in.h`/`_out.h` in as
`conv2d_stem_in_weight.h`/`conv2d_stem_out_golden.h` → source `aiehlc.sh`.
`--skip-trace` reuses the dumps on disk, `--no-aiehlc` stops after the copy.
Two traps it exists to handle: the Makefile's objects **don't depend on
CFLAGS**, so without a `clean-local` first a previous *untraced* build is
relinked and writes no dumps at all (silent no-op); and **`aiehlc.sh` writes its
`aout/` tree relative to the CWD**, not to the entry file, so running it from
the app directory produces `src/aietensorop/conv2dstem/aout/` while any older
`$REPO/aout/` sits there and makes a success check pass against a *stale* ELF —
the script pins the cwd and additionally requires the ELF to be newer than the
run.

**`test_conv2d.cc` runs TVM's layer 01, not a hand-built fixture.** It drives
`conv2d_stem_nchwc()` from `conv2d_stem_in_weight.h` — the `layeriohex` dump of
call [01] (`01_contrib_conv2d_NCHWc_subtract_add_subtract_fixed_point_multiply_per_axi`)
— and self-checks against that layer's own output (`conv2d_stem_out_golden.h`),
so the app reproduces an operator the deployed graph actually runs rather than a
model of it. **The kernel's arithmetic was already correct**: all 802,816 outputs
recompute exactly from these headers using the existing
`(acc + bias − zp) · mult >> (shift+31)` epilogue. Only *layout* differed, and
both conversions are host-side — the AIE core, the `GemmSpace` descriptors and
the DMA are untouched:

| | TVM layer 01 | library | bridge |
|---|---|---|---|
| input | `int8[230,230,3]`, zp-padded (**−15**, = uint8 113−128) | same | **none** — `stage_ifm_prepadded()` already widens 3→4 |
| weights | `int8[16,7,7,3,4]` OIHW3i4o | `int8[64,7,7,3]` OHWI | `stage_weights_nchwc()` |
| output | `uint8[16,112,112,4]` NCHW4c | `uint8[112,112,64]` HWC | `collect_ofm_nchwc()` |

`conv2d_stem_nchwc()` is a **new entry alongside** `conv2d_stem_prepadded()`,
which is untouched (94 insertions, 0 deletions) so `--aie-offload` is unaffected.
Two traps: TVM subtracts **two** zero-point terms (`acc + bias − zp_a − zp_b`)
while `conv2dstem_qparam` has one field — the fold is only valid because `zp_b`
and the `unused` param measure identically zero here, so `main.cc:fold_qparams()`
**checks** it per channel and refuses rather than computing wrong pixels; and the
header's `_data[]` is `unsigned char[]`, so reading the int32 params via a
`(const int32_t *)` cast is an unaligned load that faults on the A78 — assemble
the bytes (`rd_i32`). Mismatch coordinates decode as NCHW4c (`occ,oh,ow,ocb`),
not HWC. Skill: **conv2dstemtvmlayer**.

It also **measures** (`run_perf`, before the golden check so the check validates
the last launch): the **first** call separately — it carries partition init,
staging allocation and the weight transpose — then `CONV2DSTEM_PERF_ITERS`
(default 5) steady-state calls, then `stage_ifm_prepadded`/`collect_ofm_nchwc`
alone so the host share is measured rather than assumed. **Repeating amortizes
nothing**: aiehlc re-programs the mesh on every launch, so the steady-state
number is the per-call latency of `conv2d_stem_nchwc()`, and the non-host
remainder is labelled `config+launch`, not "AIE compute" —
`doc/performance/register_write_cost.md` puts a 32-bit `XAie_Write32` at ~372 ns.
Repeat calls are safe because `__Runtime_init_mesh_partition` is init-once per
`meshId` and the function-local `aieArray` restarts at `meshId` 0. Wall clock is
`xiltimer.h`'s `XTime`, deliberately **not** `aie_timer.h` (it branches on
`AIE_GEN`, which the host compile defines but the frontend parse does not). The
per-launch `kload/bdcfg/coreen/startio/wait_io` PMU split needs
`./buildtest.sh --profiling`; without it the counters read 0 and the report says
so instead of printing zeros. The counters read `pmccntr_el0`, which the BSP
boot never starts — `aie_runtime.c`'s `__Runtime_init` constructor enables it
under `AIEHLC_PROFILING` (baremetal only); before that, a `--profiling` build of
`test_conv2d.cc` still read all zeros (skill **pmccntrnotenabled**).
Machine-readable line: `PERF SUMMARY ... (ms)`.

**`conv2dstem_vec.cc` is the vectorized kernel** (`./buildtest.sh --vec`): same host/DMA/pragmas, kernel on `aie::mmul<4,8,8,int8,int8>` + a bit-exact two-stage vector requantize; 1.77 ms vs 31.47 ms, GOLDEN PASS on the board. Host transforms (1.09 ms) now dominate. `conv2dstem.cc` stays the `--aie-offload` source. Details: `doc/performance/conv2dstem_perf_review.md` §6.

**conv2d launch is 185 µs** (`test_conv2d.cc`, GOLDEN PASS): vector kernel now in
`conv2dstem.cc` (`mmul<8,8,8>`, filter pre-packed in mmul order by the host,
2×32-bit-split exact requantize), `#pragma output_tile_major` (fixes a shim-BD
re-programming race; `g_ofm` is tile-major, host de-tiles), `control_packet_mode(aot)`,
`TILE_W` 64 with a 232-px staging pitch, non-cacheable input staging. Runtime:
deferred launch-arg cache sync, warm ELF reload, plan cache, shim AXI burst 16.
Details `doc/performance/conv2dstem_perf_review.md` §7; skill **outputtilemajorrace**.

**conv2dstem configures the array over the control plane, not the config bus.**
`conv2dstem.cc` carries four active marker pragmas — `control_plan_op_control_packet`
(the plan; also moves **kernel ELF load + core launch** onto the fabric →
`__Runtime_load_kernel_group_16t_ctrl` / `_launch_kernel_group_ctrl`),
`CONTROL_PLAN_GROUP_REG_WRITE` (lock inits → `__Runtime_ctrl_row_write_ack`),
`control_plan_shim_bd_ctrl` (shim BDs + channel enable + start queue along row 0 →
`__Runtime_ctrl_shim_bd_begin`/`_commit`), and `control_plan_dedicated_shim`.
Justified by `doc/performance/register_write_cost.md`: ~372 ns per `XAie_Write32`,
N words = N serial writes. Three traps. The partition is **`{2, 6}` — five columns
for a four-column mesh**: `planControlColumn` grants the control spine only when
`partition cols >= mesh cols + 1`, and *the data plane does not move*
(`dataStartCol = relStartCol + 1`, so still absolute cols 3..6, as under the old
`{3,6}`). `control_plan_dedicated_shim` is a **force flag**, not a feature — alone,
`control_plan_shim_bd_ctrl` only WARNs and falls back to MMIO.
`__Runtime_ctrl_high_throughput_enable(1)` is **required** (AIE2PS: disables CTRL
TLAST errors, packs self-delimiting accesses into one shim MM2S BD) and must run
*before* device init — hence `conv2dstem_control_plane_init()` at the top of both
launch sites. **`KERNELCONFIGOFFLOAD` is on** and needed two compiler fixes to
work on `{2,6}` (otherwise `wait_io TIMEOUT`): KCO's `get_coreid()` dispatch is
rebased by the partition start column, and its MM2S BD length comes from the host
plan (3136 B), not `sizeof(buf_out_ping_0)` (784 B). Skill **kcocoreidpartition**;
`doc/debug/troubleshoot_aie_offload_timeout.md`. Verify, don't assume: **five**
`[aiehlc] Detected #pragma` lines, non-zero `grep -cE '\.TM' aout/worklocal/build/kernel.lst`, and
`control plane: dedicated shim col 0 (... shim-row BD control ...)`. An unknown
pragma is a *warning*, so `control_plan_shim_bd_ctrl` is silently dropped by
`release/prebuild/aiehlc` and by `build/aiehlc` trees older than it — rebuild the
tool (skill: **mlirbuildsandbox**). Not enabled: `control_packet_mode(aot)`.

**conv2dstem runs on real data.** `src/aietensorop/conv2dstem/data/` generates
three committed-or-derived headers from a photo and the int8-quantized
ResNet-18: `conv2dstem_image.h` (`int8[230,230,4]`, already preprocessed,
quantized, **zero-point**-padded and channel-aligned — byte-for-byte
`g_ifm_pad`, so the board does no scatter), `conv2dstem_weights.h` (real conv1
weights + 64 folded qparams), and `conv2dstem_golden.h` (CPU ground truth, which
`main.cc` self-checks against and `groundtruth.py --applog` reads back; a
baremetal board under xsdb has no channel but the console). This matters: with
real weights the peak accumulator is **1,023,225**, so an int16 accumulator
wraps — the old synthetic `[-4,4]×{-1,0,1}` fixture could not show that.
`conv2dstem_image.h` and **TVM's layer 0**
(`tvmgen_default_fused_divide_round_add_clip_cast_subtract_layout_transform`) are
the *same* quantization — bit-exact over all 150,528 values — in different
representations: `−128` vs `−zp(113)` (a constant +15), int8 vs **int16** (TVM's
`98 ff` is LE int16 `−104`, not two bytes), `[230,230,4]` pre-padded vs
`[224,224,3]` unpadded. `make_image_header.py --tvm-layer0` reimplements the
kernel's two steps (`tvm_quantize_u8` → `tvm_layer0`, literals **parsed from the
generated C**, C `roundf` tie rule) and emits `conv2dstem_image_tvm.h`;
`--verify-kernel` compiles and runs the real kernel and diffs it. Skill
**conv2dstemfixture**.

Generated headers must land **flat** in the app source dir, not in `data/` —
`aiehlc.sh:532` copies user headers with a non-recursive `*.h` glob flattened to
basename and the host compile only gets `-I<worklocal>`, so a `data/` header
parses in the frontend and then fails at cross-g++. See
`src/aietensorop/conv2dstem/data/README.md` and skill **conv2dstemfixture**.

**Control-packet plane** — register access over the stream fabric instead of the host
config bus. See **[doc/controlplane.md](doc/controlplane.md)** for the full API,
encoding, and fabric layout. In brief:

| Layer | Entry points |
|-------|--------------|
| Packet encode/parse | `__Runtime_ctrl_pktize_write` / `_pktize_read`, `_parse_ctrl_hdr` / `_parse_pkt_hdr` |
| Single-target send | `__Runtime_CtrlInstance` + `_setup_routing` / `_push` / `_tct_poll`; composed by `_read_target` / `_push_target` |
| Row/broadcast fabric | `__Runtime_ctrl_plan_init` (one-shot planner, `aie_runtime_control_plan.c`), then `_row_broadcast_write` / `_row_whole_row_write` / `_row_read` / `_row_write_ack` |
| Transaction capture | `__Runtime_control_start_transaction` → `_write_pkt_commit_transaction` → `_control_push` (WRITE-only ops → broadcast/multicast packet buffer) |
| Resource reservation | `aie_runtime_resource.{c,h}` — single source of truth for slots/arbiters/msel/pkt-ids; `RT_RES_*` constants aliased by the planner's `ACR_*` |

Key invariants worth knowing before touching it:

- Completion is the **control-packet response** draining at the shim S2MM
  (`resp_words` words), not a DMA TCT token.
- The return route KEEPS each response's stream header
  (`XAIE_SS_PKT_DONOT_DROP_HEADER`); it is stripped host-side in `rt_ctrl_read_extract`.
- Forward routing arms a **uniform slot superset** on every chain tile — all columns of
  a row respond; there is no last-tile special case.
- Provenance `dir=fwd|ret` comes from the planner's `acr_op.is_ret` tag, **not** the port
  type (both fabrics use the same ports). A slot and every master pulling it must share
  a `dir`, or the debug switch-detail view silently drops the link.
- `ResourceMgr::reserveControlPlaneResources` is **opt-in** via
  `#pragma control_plan_op_control_packet`; default off.

Platforms: baremetal (`aarch64-none-elf-g++`) or Linux (`aarch64-linux-gnu-g++`).

## Part 2: tilinglinalg (Multi-Tile GEMM)

### Six Custom MLIR Dialects

| Dialect | Purpose |
|---------|---------|
| **routing** | Abstract tile arrays, data IO, broadcast, mesh partitioning |
| **routinghw** | Physical tiles, stream switch ports, packet flows |
| **dmap** | Logical dataflow: ports, streams, push/pull |
| **dmaphop** | Physical hops: tile-to-tile paths, DMA, buffers |
| **dfscheblueprint** | Schedule blueprint: transfer manifests, flow configs |
| **dfschedule** | Executable schedule: DMA BD, kernel launch, locks |

### High-level frontend dialect (`aiegraph`)

Sits **above** `routing` — a fused/quantized op-level graph (TVM/Relay → aiegraph
→ per-op `run_aie_pipeline`). Not part of the GEMM pass pipeline; it lowers each
op to an independent launch on the existing backend.

**Runtime op split.** The `run_aie_pipeline` backend implements only the conv2d
family (`conv_bn`, `conv_bn_relu`). The other aiegraph ops (`residual_add_relu`,
`avgpool_fc`) are **not** sent to AIE; they are emitted as bit-exact CPU C by TVM
(`target="c"`, `src/frontend/tvm/cpu_codegen.py`). Non-conv ops still build/verify/
lower in the aiegraph IR — only the emit path forks (dispatch via
`cpu_codegen.is_aie_op`). See `doc/design/tvm_frontend.md` §"CPU fallback".

| Dialect | Purpose |
|---------|---------|
| **aiegraph** | Fused int8 tensor ops (`conv_bn_relu`, `conv_bn`, `residual_add_relu`, `avgpool_fc`) as SSA def-use over `tensor<Nxi8>`, with per-op quant attrs + weights `SymbolRefAttr`; `func`/`yield` container. Buffer wiring is verified SSA, not string names. |

Location: `src/mlir/mlirfront/frontend/aiegraph/` (`td/`, `gen.sh`, `inc/`,
`aiegraphmanager.{h,cpp}`, `lower/AiegraphLowerDriver.{h,cpp}`, `unitest/`).
pybind: `build_aiegraph_module(ops)` (build+verify → textual IR) and
`lower_aiegraph(mlir_text)` (walk → per-launch `tensor_specs`) in
`aiebackend/aiebackend_pybind.cpp`. Python entry: `_compiler.compile_plan(..., via_aiegraph=True)`.

### PyTorch → PT2E int8 → MLIR (`src/frontend/pytorchmlir/`)

A **second model-ingest frontend**, independent of TVM: torchvision ResNet-18
**v1** → `torch.export` → PT2E int8 (`NPUQuantizer`, per-channel symmetric
weights / affine activations) → `torch_mlir.fx` → torch dialect →
linalg-on-tensors. **Scope ends at MLIR** — no aiegraph, no AIE backend.
Needs no compiled TVM, so it coexists with `tvmrelay`.

```bash
PYTHONPATH=src python src/frontend/pytorchmlir/deploy_torch.py
python src/frontend/pytorchmlir/setup_torchmlir.py --verify-only
```

Verified on torch 2.10.0 / torchvision 0.25.0 / torchao 0.18.0 /
torch-mlir 20261001: 20 int8 convs + int8 fc, fp32-vs-int8 top-1 match with
identical top-5 order. Details in
**[src/frontend/pytorchmlir/README.md](src/frontend/pytorchmlir/README.md)**.

Three non-obvious things, each of which silently misleads if forgotten:

- **`0 fused *_q` is expected, not a failure.** torch-mlir registers
  `quantized_decomposed.*` as native ODS ops, but `torch-match-quantized-custom-ops`
  matches only the *unregistered* `torch.operator` spelling — so the matcher
  never fires and no `!torch.qint8` appears. int8 still reaches linalg as
  explicit `arith.extsi/subi/mulf` around `linalg.conv_2d_nchw_fchw`.
- **`--output-type tosa` needs `--per-tensor`** — TOSA marks
  `dequantize_per_channel` illegal with no lowering pattern.
- **PT2E lives in `torchao`**, not `torch.ao` (deprecated, deletion planned).
  All PT2E symbols resolve in `torch_deps.pt2e_api()`.
- Do **not** reuse `example/model/resnet18py/resnet18.py:resnet18()` here — it
  builds ResNet-18 **v2** from ONNX weights. Only `classify.preprocess` is shared.

**Each AIE layer builds its own `lib<layer>.a`.** `run_aie_pipeline` stops at
source (`host.cc`/`kernel.cc`/`routing.cc`) exposing only
`host_canonicalized(dev, t0, t1, t2)` — no device init, no allocation, no
caller, no kernel ELF. `aie_layer_lib.py` appends a generated op entry (the
counterpart of `conv2dstem.cc`'s hand-written `stem_run()`; the pipeline never
sees that file, so it cannot emit it) **into `host.cc`** — a sibling `.cc`
would be dropped, since `hostcompile.sh` compiles a fixed source set — then
runs `hostcompile.sh` in that directory to archive
`layers/NN_op/aie/build/libNN_op.a`. `arm_build` links them by wildcard
(`AIE_LAYER_LIBS`), and `partition.json` records `archive` / `archive_reason`.
They coexist with `libconv2dstem.a` despite both carrying the AIE runtime,
because `ld` pulls only members that resolve an undefined symbol. **Wired for
the stem only:** a layer whose `partition.json` entry carries `entry` /
`replaces` (today layer 01, built from `conv2dstem.cc`) is called through
`aie_<kernel>()` in `graph_driver.c` under `#ifdef GRAPH_AIE_OFFLOAD`. Only the
board Makefile defines that macro (`arm_build.aie_entries` → `AIE_OFFLOAD_DEFS`),
so `make local` builds the CPU kernel from the *same* driver and
`main_local.elf` is the reference for `main.elf`. Every other layer's archive
still links but contributes nothing.

**qnn zero-point folding is the DEFAULT (`--no-fold-qnn-zp` opts out).** A
quantized conv lowers to `term1 − term2 − term3 + term4`; `Conv2DCombineTerms`
(`tvm-0.16/src/relay/qnn/op/convolution.cc:633`) drops **term2 and term4 when the
kernel zero point is zero** — which it is for all 20 convs here, the quantizer
being *symmetric* per-channel. It did not fire, for a narrow reason: the integer
it tests is only read when `IsConstScalar(kernel_zero_point)`
(`convolution.cc:747`), and `Constant::is_scalar()` is strictly **`ndim == 0`**
(`include/tvm/relay/expr.h:80`). Per-channel quant makes the zero point a `[64]`
**tensor**, so the test fails, `dynamic_zp` is set, both zero points fall back to
`-1`, and the `else` branch emits term2 unconditionally. The elision keys on
**scalar-ness, not on the values being zero** — per-*tensor* symmetric takes the
fast path, per-*channel* symmetric misses it. term2 depends on the live input, so
`FoldConstant` cannot remove it; it lowers to a real
`cast_sum → multiply → avg_pool2d → repeat_multiply_layout_transform` chain
**per conv**. `qnn_fold_zp.fold_scalar_zero_points` rewrites an all-**equal**
(not merely all-zero) constant zero point to rank-0 before
`qnn.CanonicalizeOps`, so TVM's own elision fires. Must run before `build_c`
**and** before `partition_for_aie` — both canonicalize internally, after which
there is nothing left to fold. Measured on int8 ResNet-18: **120 → 41 kernel
calls**, 42 → 0 correction nodes, buffers 18.58 → 14.29 MB, logits bit-identical.
`kernel_scale` stays per-channel — the requantization genuinely is.

**int8 operands are the DEFAULT (`--no-pe-int8` opts out).** The quantizer emits genuine
int8 (all 11,678,912 ResNet-18 weights measure inside `[-127,127]`, 255 levels);
what widens them is **legalization**, which describes the *hardware*, not the
numbers. `Target("c")` has `keys=['cpu']`, so the Intel rule fires,
`is_fast_int8_on_intel()` is false for a C target, and
`helper_no_fast_int8_hw_legalization` casts data **and** weights to int16.
`pe_target.py` says otherwise via a target key — `Target("c -keys=pe_int8,cpu")`,
so `generic_func` picks the int8 rule while `cpu` still serves schedules and
`conv2d_alter_op`; patching `register("cpu", override=True)` works too but
retargets every x86 build in the process. The rule is TVM's own
`helper_change_dtypes_to_int8` = `x_i8 = x_u8 − 128`, `zp −= 128`, zero-point
folded into the bias by `QnnConv2DCanonicalize` — the same algebra as
conv2dstem's `+128·Σw`. Measured: output **bit-identical**, `weights.bin`
23.5 → 11.9 MB, board ELF 24.4 → 12.8 MB, against 30 → 120 graph nodes.
**ON for the AIE paths too** (it used to auto-disable there). The old guard was
that int8 renumbers `layers/` — the stem moves from **index 1 to 6** — while
`--aie-layers` selected positionally. Selection now resolves a layer to its conv
by **weight fingerprint** (`byoc.aie_annotate.weight_fingerprint`: values sorted,
cast int64, hashed), which is layout- and dtype-independent — the same weights
hash identically as int8/NCHW, int16/NCHW and int16/NCHWc — and `check_targets`
demands one Relay conv matching fingerprint *and* geometry. Keeping int8 off was
also wrong on its own terms: the AIE kernel eats int8, so int16 legalization left
the offloaded conv's operands the one thing not in the hardware's width.
`--aie-layers` **defaults to 6** and now governs **both** `--aie-offload` and
`--aiegraph`, so the two ways in target the same conv; a stale index now reports
`not a conv -- only convs offload` rather than offloading the wrong op.
`--aiegraph` used to take no selection at all — `partition_layers(records,
aie_ops, layers)` forces every layer outside the selection to CPU with
`not in the --aie-layers selection (N)` as its `partition.json` verdict, *after*
the whole graph is still lifted and verified. It still maps 0 layers under int8,
but it already did (`byoc_aie_plan.md:160`, "0/28 eligible, confirmed") —
it keys on fused *names* (`conv2d_add_relu`) that legalization renames. **Bias,
requant multiplier and shift stay int32 and must** — bias lives at the
`s_x·s_w` accumulator scale (23 bits here), the multiplier is a fixed-point
scale in `[2³⁰,2³¹)`; together 0.5% of the blob.

**`network.md` documents the compiled graph.** One generator —
`network_md.write_network_md`, pure and byte-reproducible — reached three ways:
`deploy_flow.py` stage 6 (automatic, no flag), the `network_md.py` CLI, or a
direct import. They never differ in logic, only in *what is on disk* when they
run; deploy_flow calls it after the offload/aiegraph stages so it reflects the
*final* graph. Missing `<stem>_params.bin` makes every constant reclassify as an
activation (`parameters | 0`, no `param` rows) — now warned about loudly instead
of silently. `network.md` and `graph_driver.c` both carry a `graph-sha` line;
`grep -m1 graph-sha network.md arm_build/graph_driver.c` says whether they are
the same build. It turns `resnet18_graph.json` into `worklocal/tvmrelay_deploy/network.md`:
summary, a **Mermaid** dataflow chart with residual skips as dotted edges, a
per-call layer table carrying `dtype[shape]` for every input, output **and
parameter**, a per-argument detail table, a parameter rollup by dtype, and what
buffer aliasing saves. **Parameter roles** (`weight` / `bias` / `zero-point` /
`requant multiplier` / `requant shift` / `unused`) are read out of the generated
C by `kernel_param_usage`, keyed on the kernel's **arg slot** (layer 29's graph
`p131..p136` are `p1..p6` inside the kernel) and splitting the two multiplies by
**cast width** `int32_t` vs `int64_t` — a rank-based rule calls the rank-3 dense
weight a multiplier. Indices printed are **entry indices**
(`node_row_ptr[node]+k`) so rows line up with `make TRACE=1` dumps, and layer
folders are matched by **kernel symbol** — stage 5 writes one folder per distinct
kernel while the graph makes more calls, so positional matching mislabels
everything after the first repeated kernel. **Three numbering spaces are all
called "layer 01"**: the call index (`network.md`'s `#`, and `[01]` in a
`TRACE=1` dump), the folder's `NN_` prefix (distinct kernels, first-use order),
and `--aie-layers` (the folder index *of the first build*). `layers/` also
accumulates folders across builds — `_clear_stale` keeps non-empty dirs on
purpose, so AIE `aie/` artifacts survive — so two `01_*` folders can coexist;
stage 5 names the leftovers and `manifest.json` lists them in `stale_dirs`. The
manifest is the authority, not `ls layers/`.

**Tracing between layers: `make TRACE=1`, never a hand edit.** `graph_driver.c`
carries a per-node dump of every kernel input/output behind `#ifdef GRAPH_TRACE`
(`TRACE_ELEMS=N` for width); a normal build compiles none of it. It decodes the
dtype — `DLDataType` is a 4-byte **struct**, so `printf("%d", t.dtype)` is UB that
prints `code | bits<<8 | lanes<<16`, i.e. int16 shows as `69632`, uint8 as `67585`
— and dumps **elements, not bytes** (a byte dump of int16 reads `98 ff` = `-104`).
`dl_dtype_str`/`graph_dump_tensor` are `static inline` and always emitted.

`TRACE=1` *also* writes **every kernel input and output in full** as a
compilable C hex header (`GRAPH_TRACE_HEX`), into **`layeriohex/` next to the
running ELF** — so `make local run-local` fills `arm_build/layeriohex/`.
**Two files per call, not one per tensor**: all inputs merge into
`l06_<op>_in.h` and all outputs into `l06_<op>_out.h`, each tensor under its own
variable prefix, so a conv's activation + weight + six quant constants are one
`#include` (`graph_hex_file_begin` / `_tensor` / `_file_end`;
`graph_dump_hex` = all three, for the one-per-file `param_*.h`). File stem =
`l<call>_<op>_<in|out>`; variable prefix adds the slot and, for a parameter, its
name and role — `l06_nn_contrib_conv2d_NCHWc_in1_p0_weight`. The op slug is deliberately
**short** (`_HEX_NAME_FN_CHARS` = 30, cut at a word boundary, `tvmgen_default_fused_`
prefix and trailing content hash stripped): the call index already makes the stem
unique, so the symbol is description not identity, and at full length 95 identical
characters pushed the distinguishing part off the edge of a terminal. The full
symbol stays on the first line **inside** each file. 417 files (240 in/out
groups + 177 params) holding 446 tensors, all stems unique. `graph_dump_hex` is
`static inline` — under `TRACE_HEX_PARAMS=0` nothing calls it, and a plain
`static` warns; **`-fsyntax-only` does not report `-Wunused-function`**, so the
trace flag matrix must be compiled for real (`-c`) to catch it.
Header carries `_DTYPE`/`_NDIM`/`_ELEMS`/`_BYTES` + `_shape[]` so it decodes on
its own; the payload is **bytes** (the on-the-wire buffer image) where
`graph_dump_tensor` gives elements. Knobs: `TRACE_HEX=0` off, `TRACE_HEX_MAX=N`
cap (sets `_TRUNCATED`), `make clean-hex`. ~100 MB on int8 ResNet-18 — params are
kernel inputs and are dumped per call, deliberately.

**Activations carry their producer**, because a layer can take several and a
bare `(in)` cannot tell them apart (`_dataflow`/`_flow_note`): `in, from call
[05] repeat_multiply_layout`, `in, network input`, `out, read by call [07]`.
The trap this closes: layer 06's `in[2]` is `int32[1,16,112,112,4]`, the **same
geometry as the conv's own output** (`uint8`), so it reads like a feedback edge.
It is not — it is call [05]'s output, the tail of the `cast_sum → multiply →
avg_pool2d → repeat_multiply` chain from call [02], which is qnn's
**weight-zero-point correction term** (`term2`); the kernel subtracts it, hence
the `subtract` in its fused name. Dtype is the giveaway, not shape. All 148
producer claims verified against the graph JSON.

**Parameters are named, not just numbered.** A conv's `_in3`/`_in4`/`_in6` are
indistinguishable, so every input that is a graph parameter gets
`_<name>_<role>` (`_in1_p0_weight`, `_in6_p5_requant_multiplier`) plus
`weights.bin@<off>+<n>` in the provenance, and `graph_dump_params()` writes each
constant **once** by name (`param_p0_weight.h`) — deduplicated, including ones no
call reads. `TRACE_HEX_PARAMS=0` drops that set (~72 MB). Roles come from
`network_md.kernel_param_usage`/`param_role`, the same classifier behind
`network.md`'s tables — verified to agree for all 41 parameter-bearing calls.
Two traps: `pN` is literally `args[N]` in the generated C, so a classifier slot
indexes the driver's input list directly; and **a role may only be applied to a
slot that is also a parameter** — the classifier labels activations too, and
layer 06's image input lands in an `add`, so it would file as a "bias". A slot
absent from `kernel_param_usage` means `unused`, which is a verdict, not a
missing entry — `_param_roles` fills every slot so the two cannot be confused.

**Only the hosted build gets
`-DGRAPH_TRACE_HEX_FILES`**: a baremetal board has no filesystem, so it streams
the identical text over the console between `===BEGIN layeriohex/...===` markers
and `arm_build.split_layeriohex(log)` cuts it back into byte-identical files
(verified). Linking `fopen`/`mkdir` into the BSP build would compile, then fail
at run time.

Hand edits to `graph_driver.c` are overwritten by the next `deploy_flow.py` run;
change `arm_build._trace_lines` / `_DEBUG_HELPERS_C` instead. All of it is restated in the
**generated** `arm_build/README.md` (`arm_build.write_readme` / `_README_MD`), which
carries each run's real numbers — edit the template, never the copy.

**`--local` runs the same C on x86.** `deploy_flow.py --local` (or `make local` /
`make run-local` in `arm_build/`) links `main_local.elf` from the *same* generated
sources with the host gcc and runs it, printing the same top-5 the board would, so
stage 7's onnxruntime reference is checkable without flashing a board — the flow
prints `local vs cpu: MATCH/MISMATCH`. It needs no Vitis toolchain (so it composes
with `--no-arm`, and is built *before* the cross-toolchain gate). It is **not** the
cross build retargeted: the BSP and the baremetal link recipe are dropped, and two
flags must be **added** — `-D__AIESIM__` (else `aie_timer.h` includes the BSP-only
`xtime_l.h`) and `-std=gnu11` not `-std=c11` (else POSIX `CLOCK_MONOTONIC` is
hidden *inside* that header). Objects land in `arm_build/localobj/`, never `local/`
— that is the phony target's name and make drops the order-only dir dependency as
circular. An AIE-offloaded build cannot link here and fails with a named error.
Skill: **tvmlocalhostelf**.

**The board ELF links from the per-layer split, not the monolithic C.** Stage 5
(`split_layers.py`, on by default; `--no-split` disables) writes one translation
unit per operator to `worklocal/tvmrelay_deploy/layers/NN_op/NN_op.c`; stage 6
(`arm_build.py`) compiles each to a `.o` **beside its source** and archives them
into `arm_build/liblayers.a`, which `graph_driver.c` + `main.c` link against.
Editing one operator recompiles one TU. `resnet18.c` remains the fallback when
`layers/` is absent; exactly one of the two is wired in, and `make` stops with a
named error if neither resolves. Cost measured on ResNet-18 int8: **+0.02%**
`.text`. Because every layer includes `layers_common.h`, that header now also
carries mid-file material (a BYOC prelude) and a synthesized prototype for any
function defined but not pre-declared — so **BYOC codegen must keep weights as
function-scope statics**, or they duplicate into all 65 TUs and collide at link.

**Whole ResNet-18 on the AIE: the persistent conv engine** (`src/aietensorop/resnetengine/`).
**2.81 ms per image** on the board, float image → logits **bit-identical to TVM**
(baseline 1270 ms): stem 7×7 + fused maxpool + all 19 other convs on the AIE as 17
engine layers, only quantize and avgpool+fc on the A78 (NEON). One fabric (conv2dstem's
routing), **one kernel ELF loaded once**; each layer's 256 B header arrives on the
weight stream and the core programs its own DMAs, so the host only re-arms 24 shim BDs
per layer over the control-packet plane. `gen_engine.py` extracts weights and every
epilogue constant from TVM's graph/params/generated C and **proves the fused design
bit-exact in numpy before any board run**: downsamples are a 10th tap of the 3×3 s2
conv, residuals stay in core memory (GRAB/COMB), every inter-layer tensor is 8-bit.
Measured limits: AIE clock 998 MHz, ~2.3 GB/s per shim MM2S channel (burst/AxCACHE
don't move it), compute only ~6 % of engine time. Design + step-by-step optimization
log: **[doc/design/resnet_aie_engine.md](doc/design/resnet_aie_engine.md)**; skill
**resnetengine**. Runtime mode switch `engine_set_mode(e, eager, layer_major)`: persistent,
eager (kernel reloaded per layer), eager layer-major (B images per layer). Kernel
reverted to the 2026-10-20 engine (~1.61 ms/image at B=32): the faster kernels
(1.364 ms) hang ~1 run in 3 in residual layers (skill **shimctrlpushloss**).

### TVM `--aie-offload` → AIE via BYOC

`deploy_flow.py --aie-offload --aie-layers N` (default 1, the stem) puts layer N
on the AIE: after the CPU build, Relay `MergeComposite` → `PartitionGraph` →
`relay.ext.aie` codegen, the C is rebuilt, and TVM's graph executor calls the
generated wrapper → `conv2d_stem_prepadded()` in `aout/libconv2dstem.a` (built
from `src/aietensorop/conv2dstem/conv2dstem.cc` by `aiehlc.sh` in library mode —
skill **hostlibrarymode**; aiehlc generates the init/partition/launch caller).
`--byoc-aie` was merged into this flag. Code: `src/frontend/tvmrelay/byoc/`
(`aie_patterns`, `aie_fuse`, `aie_annotate`, `aie_codegen`, `aie_byoc`) +
`aie_offload.py`; design
**[doc/design/aie_offload_byoc.md](doc/design/aie_offload_byoc.md)**; check
`byoc/verify_aie_stem.py` (bit-exact vs TVM, incl. all 1000 logits).

**`--aie-offload` also runs the aiegraph lift** over the same `--aie-layers`
selection, because BYOC writes no per-layer artifacts. That is what puts
`layers/01_*/aie/` on disk, as `aiegraph.conv_bn` H=230 Cin=3 Cout=64 K=7 s2.
A conv with exactly that stem geometry is **not** built from the generic
placeholder body (`kernels.conv_bn_body`: int16 acc, dummy quant): `aie_stem_lib`
runs `aiehlc.sh` on `conv2dstem.cc` in a scratch cwd and installs
`aout/worklocal/` as the layer's `aie/` (`conv2d_spatial.cc`, `host.cc`,
`kernel.cc`, `routing.cc`, `build/libconv2dstem.a`). That is the logic
`test_conv2d.cc` verifies bit-exact: `kernel.cc`/`routing.cc`/bcf/prx are
byte-identical to its build, and `host.cc` is a strict subset (it lacks only
`test_conv2d.cc`'s `main()` driver). `partition.json` records `backend: aiehlc`
+ `source`. It adds ~80 s per run. Appended to that `host.cc`, then
re-archived by `hostcompile.sh`: `aie_<kernel>()`, which has TVM's packed-call
signature and performs `test_conv2d.cc`'s `fold_qparams` + `conv2d_stem_nchwc`.
It prints `[aie-offload] layer 01 ... ENTER` / `EXIT rc=.. hash=..` on the
console, which is the proof the board ran layer 01 on the array. Test:
`src/frontend/tvmrelay/test/run_test.sh` (check 4 `[elf ]` demands main.elf
is fresh and links the entry, because deploy_flow exits 0 even when `make`
fails). It is additive (every `.c` is
untouched), so `src/frontend/tvmrelay/test/run_test.sh` holds `layers/` to
the **default-flow** `layers-golden/` and checks `aie/` separately. aiegraph
used to lift 0 convs under int8 because `_conv_geometry` read 4-D only; it now
folds NCHWc through `aie_offload._layer_geometry`. Skill: **aieoffloadaiegraph**.

**Four steps, one per file, and only the first is TVM's.** `partition_for_aie` is
orchestration over **1 fuse** (`aie_fuse`: canonicalize → `MergeComposite` → tag
`aie.weight_fp`) → **2 select** (`aie_annotate`: fingerprint ∈ selection *and*
`kernel_can_run` → tag `aie.layer_id`) → **3 partition** (`MergeCompilerRegions`
→ `PartitionGraph`) → **4 codegen** (`aie_codegen`: `relay.ext.aie`, dispatched
on the composite name via `_EMITTERS`). **"Fuse" names two different things and
they cannot be reordered**: step 1 is `MergeComposite`, while TVM's `FuseOps` is
TE-level fusion and is the *last* pass of the second `relay.build`, necessarily
after partitioning so it only fuses what stayed on the CPU — "byoc before fuse"
is correct, not a bug. Fusion is deliberately **geometry-agnostic**: the stem
gate used to sit in the `MergeComposite` checker, so only the one offloadable
layer ever fused, "malformed" and "not the stem" were the same silence, and the
inline-back path was unreachable. `check_fused_qconv` is shape-only;
`kernel_can_run` is capability and returns `(ok, reason)`, which is what
separates `info["unmatched"]` (did not fuse) from `info["rejected"]` (fused,
unrunnable, with the reason). The cost: **inline-back is now hot** — ~12 convs
fuse and are inlined back per run, and the whole C is rebuilt from that module,
so `verify_aie_stem.py`'s whole-network logit check is the regression gate, not
a formality. Skill: **byocfusesplit**.

Non-obvious and each one silently wrong if broken — see skill **byocaieoffload**:
the subgraph is the **fused** op (conv → requantize → ReLU → uint8), because that
is all the AIE kernel can emit; the uint8 input (zero-point 113) goes in as
**`x − 128`** with `+128·Σw` folded into the bias — a plain cast corrupts every
pixel ≥ 128; TVM pads with the **zero-point**, so the library takes the
**pre-padded** 230×230 tensor; layers map onto Relay convs by **weight
fingerprint**, not position (shortcut convs run in a different order); the
accumulator must be **int32**; per-channel quant params hide in the filter
window's **pad channel**; kernel-visible helpers must be **macros**; and the link
needs the BSP's `libxil.a` de-duplicated against `libxaienginea78.a`.

### `aiegraph` — the Python front door (`pip install -e .`)

`src/aiegraph/` (+ root `pyproject.toml`): `ag.load("resnet18" | "x.onnx" | nn.Module)` →
`ag.embedded_convert(m, "a78")` (drives `deploy_flow.run(arm=False, calib=...)` then
`arm_build.build_arm_elf(run_make=False)` into `worklocal/aiegraph/<model>/`, cached by a
model-hash stamp `aiegraph.json`) → `emod.detect / offload_check / offload / build /
run_local` (all `aiekernel.flow` + `arm_build`), one-call `ag.compile(m, target, offload)`,
`ag.open(workdir)` adopts any deploy tree, CLI `aiegraph …`, plus an `aiehlc` command
(`bash -c 'source script/aiehlc.sh "$@"'` in the caller's cwd). **One install command:**
`pip install -e .` — verified in a fresh venv: no `PYTHONPATH`, no setup step, `doctor`
green, quickstart converts/checks/offloads/links. Traps: **editable only** — the flow
needs the checkout (`repo_root()` = `parents[2]`, refuses otherwise); `src/frontend` has
no `__init__.py`, so packaging needs `namespaces = true`; **TVM 0.16 has no wheel** —
`_env.attach_tvm()` puts `thirdparty/tvm-0.16/python` on `sys.path` *and* `PYTHONPATH`
(children) when `import tvm` is not 0.16, probing in a child process; importing anything
under `frontend.tvmrelay` runs `ensure_tvm016()` which *provisions* (10–25 min) if TVM is
missing, so `doctor` imports with `AIEHLC_TVM_AUTO_INSTALL=0`; `deploy_flow` caches by
fixed file names (`resnet18-v1-7.*.onnx`), so each model gets its own workdir and a stamp
mismatch wipes it (refusing a non-aiegraph dir); never run two `aiehlc.sh` at once
(`thirdparty/alib/lib` is wiped per run). Inputs other than `[1,3,224,224]` are refused at
`load`. Guide **[doc/tutorial_aiegraph.md](doc/tutorial_aiegraph.md)**, skill **aiegraphinstall**.

**Region API** (the headline form; the layer-index calls above still work):
`ag.convert` (= `embedded_convert`, also `model.convert`) → `module.register_kernel(name,
source, op, target, constraints)` → `module.check_offload(region=, kernel=)` →
`module.assign(region=, backend="aie:<name>"|"cpu")` → `module.partition(fallback="cpu",
optimize_for="latency")` → `module.build(target="a78"|"x86", output=dir)`. Code:
`region.py` / `contract.py` / `partition.py`. A **region** is a weighted ONNX node of
`<workdir>/*.int8.qdq.onnx` (`Conv`, `Gemm`), named from the node (`/layer1/layer1.0/conv1/Conv`
→ `layer1.0.conv1`; a scope repeats only its *parent's* attribute, so
`/downsample/downsample.0/` → `downsample.0`), and the zoo's gluon names also get
torchvision aliases (`resnetv15_stage2_conv2` → `layer2.0.downsample.0` — by the `convK`
index and the 1×1 kernel, **not node order**, which puts the downsample first). It maps to
a TVM call by **weight fingerprint**, never by position: 21/21 on ResNet-18 (zoo and torch).
Four traps: a TVM layer can serve several regions (`layer1.0.conv1` + `layer1.1.conv1` =
layer 4) and the driver swaps **by symbol**, so they move together — `check_offload` notes
it, `partition` reports the sibling as *dragged* and fails a split assignment as a
conflict; `region="layer1"` is torchvision's **stage** (every region under it), whereas the
old `offload_check("layer1", …)` means **TVM kernel 1** (the stem); the contract is
**model-level** (ONNX `Conv` is NCHW/OIHW whatever TVM picks) while the `AIEK_SPEC` is
call-level — `register_kernel` refuses only what the spec contradicts (op family, weight
dtype); and `partition()` is the **whole** placement — it resets every AIE row of
`partition.json` to CPU before offloading, and `optimize_for` has no cost model (it says
so in the report). Skill **aiegraphregion**.

### User kernels: `--aie-kernel` / `frontend.tvmrelay.aiekernel`

Bring-your-own AIE kernel for one layer or every layer a spec fits. **C:**
`include/aie_user_kernel.h` — an aiehlc library source adds `AIEK_ENTRY(name)`
(`const aiek_tensor *args` in **TVM call order**: inputs incl. constants as the
graph passes them, then outputs), `AIEK_SPEC(name, "in|param|out <dtype> [dims]
[require=..|tol=..]\n"...)` (dims int / `*` / NAME bound once; `require=` checked
on real `params.bin` values per call), and optionally an x86
`AIEK_REFERENCE(name)`. **Python:** `python -m frontend.tvmrelay.aiekernel
{lint,detect,check,offload}` or `deploy_flow.py --aie-kernel SRC[@L,..]`. Six
gates, none touching a board: **lint** (L0xx — every known aiehlc silent trap),
**detect** (signature + values, all calls of a layer), **calibrate** (`<name>_ref`
vs the layer's stage-5 TVM C on real constants + adversarial activations,
outputs poisoned differently per side), **emulate** (whole network on x86 through
the *generated* `aie_<symbol>` adapter — same text as the board's, callee
swapped — logits byte-compared, ENTER count checked, failing layer bisected),
**build** (aiehlc → `layers/<first>/aie/build/libaiek_<name>.a`, adapters
appended to `host.cc`, `build/host.o` deleted first because `hostcompile.sh`
skips existing objects), **link**. Then `partition.json` (`backend:
aiehlc-user`) and `arm_build.aie_entries` swap the call, as for the stem.
Verified: stem (`conv2dstem_uk.cc`) bit-exact + logits identical, built, isolated,
`main.elf` linked; `example/userkernel/conv3x3_requant_ref.c` (reference-only)
matches 7 layers / 8 calls, all bit-exact, logits identical. **One aiehlc
library per ELF:** `__Runtime_routing_init` calls ONE global `routing()` and
partition init is cached per `meshId`, so `isolate_archive` localizes everything
except adapters, entry and the runtime hooks (kept *strong* so a second library
collides instead of silently running with the first one's routing) and the link
gate rejects later kernels. Tutorial: **[doc/tutorial_userkernel.md](doc/tutorial_userkernel.md)**;
skill **aieuserkernel**.

### Pass Pipeline

**Shared stages** (produces dfscheblueprint IR, then module is cloned):

1. `RoutingUnrollingLowerPass` — unroll abstract routing into per-tile ops
2. `RoutingToDmapPass` — routing → logical dataflow
3. `DmapToDmaphopPass` — logical → physical hops
4. `DmaphopTodfscheblueprintPass` — hops → schedule blueprints

**Host path** → `host.cc`:

5. `BlueprintToSchedulePass` → `ScheduleCanonicalizePass` → (`GroupRegWritePass`, opt-in) → `DfscheduleToApiPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `host.cc`

`GroupRegWritePass` is **gated** on `routing.control_plan_group_reg_write`, published by
aiehlc only for `#pragma CONTROL_PLAN_GROUP_REG_WRITE`. Off (the default), per-tile lock
inits emit as individual `XAie_LockSetValue` writes instead of coalesced control-packet
group writes (`__Runtime_ctrl_row_write_ack`), and the pipeline logs
`GroupRegWritePass skipped`.

**Kernel path** → `kernel.cc`:

5. `BlueprintToScheduleKernelPass` → (`DfscheduleKernelAggregationPass`, offload-only) → `DfscheduleToKernelApiPass` → EmitC → `kernel.cc`

`DfscheduleKernelAggregationPass` (`pass/passdfschedulekernelaggregation/`, gated on
`routing.kernel_config_offload`) collapses the per-core-tile DMA config onto
`dfschedule.declaretile.self` — one BD group per window instead of one per tile
(96 → 6 `dma_bd`). `packet_id` / `out_of_order_bd_id` are **SSA operands, not
attributes**, and must stay `arith.constant`-foldable. See
**[doc/design/kernel_dma_aggregation.md](doc/design/kernel_dma_aggregation.md)** — it
covers why `declaretile.self` is its own op (fail-closed on `getDefiningOp`), why
`ooo_bd_id` needs the `ResourceMgr` singleton fallback on the kernel clone, and why
erasure must be a fixpoint sweep.

**KERNELCONFIGOFFLOAD** (gated on `routing.kernel_config_offload`, set by
`#pragma KERNELCONFIGOFFLOAD`, default off, Gen5 only): the core self-configures all of
its own core-tile DMA from `kernel.cc` via MMIO instead of the host programming it over
the config bus, using `src/mlir/runtime/aie_kernel_runtime.h` (encoders + `core_reg_*`
apply, the latter going through `kernel_tm.h`'s `TM_W` — a plain pointer cast never
reaches the processor bus). See
**[doc/design/kernel_config_offload.md](doc/design/kernel_config_offload.md)** for that
TM rule, the direction asymmetry, the cross-clone plan channel, and the BD-id reservation
contract — each one deadlocks or silently corrupts registers if broken.

**Pass layout.** Both blueprint-lowering passes live under `pass/passblueprintlowering/`, with the shared `helper/` hoisted **above** them (`passblueprintlowering/helper/`) rather than nested inside the host pass. The two paths describe the same physical core tiles — one BD bank, one lock array per tile — so whoever programs it must agree with whoever doesn't. Anything both paths must agree on belongs in `helper/`. See `passblueprintlowering/README.md`.

**Routing path** (alternative, Path A) → `routing.cc`:

1. `RoutingUnrollingLowerPass` → `RoutingLowerPass` → `RoutingHWLowerPass` → `RoutingDeadArgPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `routing.cc`

### Routing Implementation (`pass/routingimplement/`)

- **RoutingTopology**: Gen2 AIE tile topology model
- **RoutingPath**: BFS path finding (priority: Memory > SHIM > Core)
- **ResourceManager**: Tracks link/port usage to avoid conflicts

## Build

```bash
# Main aiehlc binary
mkdir build && cd build && cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build && make -j$(nproc)

# TilingLinalg unitest (standalone)
 source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc
 python3 ./script/test/apppaltest.py -y  -nonreboot > ./applog 2>&1
```

Each dialect has `td/` (TableGen), `gen.sh` (runs mlir-tblgen), and `inc/` (generated .inc files).

## Test and Verification

### Unitest CLI

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
```

### End-to-End Flow

```
1. Generate    source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
2. HW run      python3 script/test/apppaltest.py aout/worklocal/build/host  → SSH+xsdb+console
3. Verify      script/test/verify_host.sh → pass: "device_teardown done", fail: "AIE ERROR"
```

### Per-Dialect Unit Tests

Each dialect has its own `unitest/` directory with independent CMake build:
- `routing/unitest/`, `routinghw/unitest/`
- `dataflowmap/{dmap,dmaphop,dfschedule,dfscheblueprint}/unitest/`
- `pass/routingimplement/{routing,hw}/unitest/`

## Additional Documentation

### Design docs

- **[doc/module_analysis.md](doc/module_analysis.md)** — 9-module project breakdown (M1–M9), key files, dependencies, data-flow diagram
- **[doc/aieapi.md](doc/aieapi.md)** — XAie driver API guide (single-tile, multi-tile manual, production AEG patterns)
- **[doc/controlplane.md](doc/controlplane.md)** — Control-packet plane: encoding, `CtrlInstance` send/recv, row/broadcast fabric, transaction capture, resource reservation, provenance
- **[doc/performance/register_write_cost.md](doc/performance/register_write_cost.md)** — **Measured** host↔AIE register access cost: ~372 ns per 32-bit `XAie_Write32` (non-posted Device-nGnRnE NoC round trip; an N-word BD is N serial writes) vs ~2.9 ns via control packets. Cite this instead of deriving ns/write from a timeline span.
- **[doc/performance/conv2dstem_perf_review.md](doc/performance/conv2dstem_perf_review.md)** — Why conv2dstem takes 31 ms: kernel.lst cycle model (scalar `MAC r`, 68 cycles / 21 MACs, 108-cycle per-output epilogue, 7 cycles/byte copies) closes to the measurement; 7.5 GOPS = 0.09 % of 4×4 int8 peak; **no vector instructions**; kernel/compiler/programming-model fixes
- **[doc/tilinglinalg.md](doc/tilinglinalg.md)** — TilingLinalg deep dive: dialects, passes, routing engine, build/HW-run flow
- **[doc/modelsupportevaluate.md](doc/modelsupportevaluate.md)** — Gap analysis for Gemma / ResNet-50 / YOLO / MobileNet / BERT / ViT (none run on AIE today); proposed `aiephase` + `aiecfg` dialects (phase-based control-plane reconfiguration, overlay kernels), new Depthwise/Eltwise/Reduce/Attention/Pipeline/Move spaces, milestones M0–M8
- **[doc/design/resnet_aie_engine.md](doc/design/resnet_aie_engine.md)** — whole int8 ResNet-18 on the AIE in 2.81 ms, bit-exact: persistent engine architecture, exactness proofs (fusions, 8-bit tensors, on-core residuals), measured optimization steps 1270 → 2.81 ms, what did not help, remaining limits
- **[doc/design/kernel_config_offload.md](doc/design/kernel_config_offload.md)** — `#pragma KERNELCONFIGOFFLOAD`: core self-configured DMA, lock asymmetry, BD-id reservation
- **[doc/design/aie_offload_byoc.md](doc/design/aie_offload_byoc.md)** — `--aie-offload`: TVM BYOC → `libconv2dstem.a`, fused boundary, uint8 shift, weight-fingerprint layer mapping
- **[doc/design/kernel_dma_aggregation.md](doc/design/kernel_dma_aggregation.md)** — `DfscheduleKernelAggregationPass`, `declaretile.self`, `packet_id`/`ooo_bd_id` operands
- **[doc/debug/tutorial_aiehlc.md](doc/debug/tutorial_aiehlc.md)** — aiehlc simulator + debug UI (`--platform sim`, `--sim-only`)
- **[doc/debug/tutorial_baremetal.md](doc/debug/tutorial_baremetal.md)** — naiebaremetal VEK385 boot + debug UI
- **[doc/tutorial_aiegraph.md](doc/tutorial_aiegraph.md)** — `aiegraph` user guide: `pip install -e .`, `aiegraph doctor`, ONNX/PyTorch steps, `ag.compile`, API + CLI reference, limits
- **[doc/design/aiegraph_python_api.md](doc/design/aiegraph_python_api.md)** — Design (M0–M3 implemented in `src/aiegraph/`): user-facing `import aiegraph as ag` facade — `ag.load` (ONNX/PyTorch) → `ag.convert(m, "a78")` (int8 + per-layer C) → `register_kernel` → `check_offload(region=, kernel=)` → `assign` → `partition` → `build(target=, output=)` (§4.8; the older `offload_check("layer1", k.cc)` → `offload(...)` remains), one-call `ag.compile(m, "aie", "layer1=k.cc")`; prior art (PyTorch/ExecuTorch, TVM BYOC, MLIR/IREE, Triton, ORT); maps onto `deploy_flow` + `aiekernel`; milestones M0–M5
- **[doc/tutorial_userkernel.md](doc/tutorial_userkernel.md)** — Bring your own AIE kernel: `aie_user_kernel.h` contract (`AIEK_ENTRY/SPEC/REFERENCE`), lint → detect → calibrate → emulate → build → link, stem + 3×3 "bunch of layers" walkthroughs, one-library-per-ELF limit
- **[doc/design/lwip_elf_bootloader.md](doc/design/lwip_elf_bootloader.md)** — netboot, **implemented** (`src/tool/netboot/`, `apppaltest.py --netload`): lwIP TCP supervisor resident on A78 core 0 at `0x7800_0000` runs each ELF on core 1 via xilpm; 15 MB ELF in ~0.2 s vs ~150 s JTAG; the target ELF is streamed from the dev box through `netboot_proxy.py` on PALIP over ssh (no scp), and a finished app parks core 1 (patched `_exit`) so the next run needs no bootstrap. Section 12 = board facts: `0x8_0000_0000` mirrors `0x0`; `Xil_SetTlbAttributes` >4 GB rewrites a 1 GB block; `p_memsz` includes the 1 GB heap; this PDI denies stopping core 1 (auto re-bootstrap). Section 13 = protocol v2: the board acks each request's header before the ELF is sent (`ok` / `busy` + time left / `overtime` / `stuck` / `serving`), per-run `max_run_ms` (default 60 s), `--force` preempts only an app past its max (on palmyra: `apppaltest` frees core 1 over JTAG -- `stop; rst -proc; pc=park_addr; con` -- ~17 s instead of a ~60 s re-bootstrap; the supervisor is polled, because that reset disables the GIC); `netload.py --wait/--force/--max-run-ms`, `apppaltest --netload-wait/--netload-max-run-ms/--netload-no-force`
- **[doc/resnetlaye1timeoutdebug.md](doc/resnetlaye1timeoutdebug.md)** — Step-by-step summary of the ResNet-18 layer-01 `--aie-offload` timeout debug: symptom, allconv review, source/pragma bisection, KCO absolute-column root cause + fix, `-O3` (2232 → 1282 ms), lessons
- **[doc/debug/troubleshoot_aie_offload_timeout.md](doc/debug/troubleshoot_aie_offload_timeout.md)** — `--aie-offload` main.elf hangs (`ENTER` without `EXIT`, `wait_io TIMEOUT`): reading the deadlock, allconv review, HW bisection (committed stem passes, control-plane WIP hangs standalone too), `AIE_DEBUG_LOG`-gated timeout dump
- **[doc/design/tile_dim_structured_design.md](doc/design/tile_dim_structured_design.md)** — Structured `tile_dim` for `aie::SpatialPolicy`
- **[doc/design/spatial_space_composition.md](doc/design/spatial_space_composition.md)** — Composition-based spatial op spaces (GemmSpace, Conv2dSpace)
- **[doc/design/aiegdb_live_debug_framework.md](doc/design/aiegdb_live_debug_framework.md)** — Live debug framework design (static view + daemon)
- **[doc/llvm_mlir_pitfalls.md](doc/llvm_mlir_pitfalls.md)** — Known LLVM/MLIR API pitfalls
- **[doc/aiedifferentview.md](doc/aiedifferentview.md)** — Memory and lock semantics

### Agent skills (`.cursor/skills/<name>/SKILL.md`)

Read the matching skill when the task fits:

| Topic | Skill |
|-------|-------|
| Debug UI, daemon, live session, browser UI features | **debug-ui-framework** (+ [reference.md](.cursor/skills/debug-ui-framework/reference.md)) |
| Embedded LLM context loss on retarget | debugui-llm-reset |
| Debug UI grid: every tile red border + ⚠ "supply/demand mismatch" on every launch | flowbalancebroadcast |
| Debug UI grid: big blank column on the left of the AIE array (data plane starts at col 1, e.g. `{2,6}` control-plane partition) | debuguigridleftgap |
| Static XAie API verify (routing.cc, host.cc) | xaieapiverify |
| Routing debug (IR → generated code) | routinghwdebug |
| DMA BD verify in host.cc | dmabdverify |
| Pre-HW data correctness | datacorrectness |
| XAie driver internals | aiedriverkb |
| Live HW DMA stall debug | aiehwdmadebug |
| Core register write succeeds on-core but host reads 0 / DMA never starts (`ST` vs `ST.TM`) | coretmregisterwrite |
| Shim BD stuck on wrong/locked BD (index overflow into channel-control regs) | shimbdindexoverflow |
| Vector kernel (aie_api SRS/requantize) off by +1 / data-dependent rounding on HW but exact on host; `control_rnd`/`sync_buffer` at DMb 0x0 in kernel.map | bcfglobalplacement |
| netboot (`src/tool/netboot`, `apppaltest --netload`): supervisor dies on LOAD_RUN, uploads crawl, "app core busy" / PLM 0x7DB 0x7D2, watchdog fires at launch; board busy / how long to wait / max run time / forcing an overdue app (netload exit 3/4/5); `0x8_0000_0000` mirrors `0x0` on palmyra | netbootsupervisor |
| `KERNELCONFIGOFFLOAD` + partition start col ≠ 0 hangs; high core columns' MM2S never armed (`get_coreid()` absolute vs plan relative) | kcocoreidpartition |
| Sim PS.so load segfault | aiesimloaddebug |
| aiesim live debug register socket | aiesim-debug-socket |
| HW performance counters | aiehwprofile |
| GOLDEN FAIL only once the kernel is fast; whole output tiles at stale addresses; per-slab host loop re-programming shim S2MM BDs; KCYC cycle stamps | outputtilemajorrace |
| `--profiling` build, yet every runtime phase (kload/bdcfg/coreen/startio/wait_io) reads 0 / "counters not built in" (`pmccntr_el0` never started) | pmccntrnotenabled |
| Raw-XAie sim debug bundle | raw-xaie-sim-debug-bundle |
| Sim build/run separation | sim-build-run-separation |
| Sim build (`pr_sim_test.sh`, `-D__AIESIM__`) fails in aie_runtime.c with "use of undeclared identifier" while the HW build is fine | aiesimguardedsymbol |
| Build fails on missing snap cmake / libz.so / ZLIB::ZLIB; fast single-file compile check | mlirbuildsandbox |
| A `#pragma` you added is never `Detected` in the aiehlc log and its feature silently does nothing (build still rc=0) — stale `aiehlc` binary | stalepragmabinary |
| Gen2-only build break: missing `xpseudo_asm_armclang.h`, or `XPAR_CPU_TIMESTAMP_CLK_FREQ` undeclared | bspheadergen2 |
| hostcompile / missing compile_kernel.sh | hostcompile-entrypoint |
| App source with no `main()` → static lib; "return-statement with a value, in function returning 'void'" | hostlibrarymode |
| Entry file `#include`s a kernel `.cc`; `unknown type name '__global__'` or `acquire_input_window` errors pointing at an INCLUDED file | aiehlcincludekernel |
| Pipeline "succeeds" but emits an EMPTY module (0 routing connections, no BCF/PRX → `Couldn't open aie2ps.prx`): `__global__` in a comment, or a prototype above the kernel | aiesourcetextrewrite |
| `deploy_flow.py` emits fp32 instead of the default int8, `target.build.llvm is not enabled`, missing `onnx`, or stage-5 split silently skipped | tvmrelaynollvm |
| Run the TVM-generated ResNet C on x86 (`--local` / `make local`): `xtime_l.h: No such file`, `'CLOCK_MONOTONIC' undeclared`, `Circular ... dependency dropped`, stack-smash in `graph_run` | tvmlocalhostelf |
| Quantized graph has ~3x the kernel calls of its operator count (120 vs 41); dead `cast_sum → multiply → avg_pool2d → repeat_multiply` chains per conv computing/subtracting zeros; a conv input with the same shape as its own output | qnnscalarzeropoint |
| int8 model whose conv weights/activations are **stored int16** (`weights.bin` 23.5 MB, `network.md` says `int16`); telling TVM the PE takes int8 (`--pe-int8`, `Target("c -keys=pe_int8,cpu")`); why bias/multiplier/shift are int32 by design | tvmint8legalize |
| PT2E int8 → torch-mlir yields no `!torch.qint8` / no `linalg.*_q` (fusion passes look like no-ops); TOSA "failed to legalize `dequantize_per_channel`"; `pip install torchvision` upgrading torch | torchmlirquantfusion |
| TVM `--aie-offload` (BYOC → AIE): fused boundary, uint8 shift, zero-point padding, weight-fingerprint layer mapping, conv2dstem int32/epilogue, duplicate `XAie_*` at link | byocaieoffload |
| BYOC four-step split: "fuse" means two things (`MergeComposite` vs TVM's `FuseOps`); adding a second AIE kernel/pattern; `--aie-layers N` silently offloads nothing and says nothing about why; CPU layers change when only the AIE layer should have | byocfusesplit |
| conv2dstem real-data fixture: generated header can't live in a subdir; zero-point border; `shift = -exponent`; golden self-check over the console | conv2dstemfixture |
| Drive conv2dstem from a real TVM layer's `layeriohex` dump (`conv2d_stem_nchwc`, OIHW3i4o/NCHW4c bridges); plausible-but-wrong pixels after a layout change; int32 quant params read as garbage or fault on the A78; mismatch coords name the wrong pixel | conv2dstemtvmlayer |
| `--aie-offload` says `non-4D shapes; not a conv2d` for every layer, or a layer gets another layer's geometry (`K=0`); TVM 5-D NCHWc vs aiehlc's d1..d4 | nchwclayoutfold |
| `Option '...' registered more than once` / `Option 'basic' already exists` when TVM flow loads `_aiebackend` (two LLVMs) | aiebackendtvmllvm |
| `run_test.sh` check 3: no `aie/` under the offloaded layer; `--aiegraph` lifts 0 convs under int8 (`non-4D shapes`); may `--aie-offload` change `layers/` vs golden | aieoffloadaiegraph |
| Frontend prints "OVER BUDGET" / gates offload on tile memory — don't; offload is blind, aiehlc tiles | aieoffloadblind |
| `pip install -e .` / `import aiegraph`: TVM 0.16 not found or provisioning unexpectedly, `no aiehlc checkout`, `exists and was not written by aiegraph`, stale results after changing model/calib, packaging `frontend` namespace | aiegraphinstall |
| `aiegraph` region API: `no region matches 'layer1.0.conv1'`, a region offloads its sibling too (`move together` / *dragged*), `region="layer1"` vs `offload_check("layer1")` mean different things, `partition` undid an earlier offload, `register_kernel` "contract contradicts" | aiegraphregion |
| Plug a user-written kernel into a TVM layer (`AIEK_ENTRY`/`AIEK_SPEC`, `--aie-kernel`, `aiekernel check/offload`); an L0xx lint code, calibrate MISMATCH, emulate FAIL, or `rejected at link -- ... routing()` | aieuserkernel |
| Run/extend the persistent ResNet-18 engine (`resnetengine`: gen_engine.py model mismatch, core memory overflow, ENG FAIL, new layer kind) | resnetengine |
| resnetengine eager layer-major hangs ~1 run in 3 in a residual layer: `shim_bd_ctrl TIMEOUT ... owing an ack`, input shim with no task; or a few wrong logits | shimctrlpushloss |
| AIE kernel: +1 errors / constant garbage right after ADDING a code path; statics above `DM_stack` overwritten | aiecorestackoverflow |
| Kernel cycle counters around `acquire_greater_equal` read ~2 cycles; measuring AIE clock | aiecoretimerlock |
| AEG IPC sim C++ headers | aeg-sim-cxx-headers |
| Host codegen | hostcodegen |
| Kernel codegen | kernelcodegen |

**Embedded LLM plugin** (`src/tool/debug/dbg_llm_skills/`): nine skills for the browser LLM tab. Launch locally with `claude --plugin-dir src/tool/debug/dbg_llm_skills`. Listed in **debug-ui-framework** reference.

**External:** aiedbg clone at `/scratch/staff/bkirinci/aiedbg` — see plugin skill `aiedbg-reference`.

**Command:** [data-mismatch-debug](.claude/commands/data-mismatch-debug.md) — systematic DMA data-mismatch triage.

### Browser automation (MCP)

`mcp-browser` (configured in `.mcp.json`, built at `thirdparty/mcp-browser/`) gives live
Playwright browser control. **Use it for any debug-UI work** — verifying
`schedule_debug_server.py` / `schedule_view.py` renders, reproducing UI bugs, and
confirming a server-side change actually reaches the page.

Workflow: start the debug server → `browser_navigate` to `http://localhost:<port>` →
`browser_screenshot` → interact with `browser_click` / `browser_extract_text`.
Other tools: `browser_type`, `browser_wait_for_element`, `browser_execute_script`.

After changing it: `cd thirdparty/mcp-browser && npm run build`, then restart Claude Code
to reload the MCP server.

### Debug tools

- **[src/tool/debug/README.md](src/tool/debug/README.md)** — user guide and CLI for all debug tools
- **Skill: debug-ui-framework** — implementation map for `schedule_debug_server.py`, `schedule_view.py`, `aiegdb.py`, `aiemcp.py`, `aiediag.py`, `xaiehost2provenance.py` (detail in [reference.md](.cursor/skills/debug-ui-framework/reference.md))

### Build notes

- **[script/verify_env.sh](script/verify_env.sh)** — validate Vitis, LLVM, toolchain, board vars before build
- **[script/hostcompile.sh](script/hostcompile.sh)** — kernel build via `compile_one_kernel()` → `kc.sh`; do not restore deleted `compile_kernel.sh` (skill: hostcompile-entrypoint). Picks `HOST_ENTRY_KIND` from the generated `host.cc`: `int main()` / `void main()` link an ELF, **no main() archives `lib<app>.a`** instead (skill: hostlibrarymode). The `return;`→`return 0;` fixup is main-only — it rewrites the *first* `return;` in the file and corrupts `host_canonicalized` if applied without a main.
- **[script/aiehlc.sh](script/aiehlc.sh)** — `--platform sim` is build-only; launch sim separately via `runsim.sh` or debug UI **Run** (skills: sim-build-run-separation, raw-xaie-sim-debug-bundle)
- **`include/bspcompat/`** — shims for standalone-BSP headers present only on the armclang branch (`thirdparty/alib/include/` is a flattened copy of *one* BSP, picked by `--aie-version`, wiped every run, gitignored). `aiehlc.sh` appends `-I${AIEHLC_DIR}/include/bspcompat` **last** in `AIEHLC_ARGS` so the real Gen5 header still wins — add it to the **front-end only**, never to the host/kernel compiles. Details and the related `AIE_GEN > 2` XTime gate: skill **bspheadergen2**.
- **[script/kc.sh](script/kc.sh)** — after linking the kernel ELF, `strip_kernel_debug_loc` drops `.debug_loc` + the `.debug_info` group via `llvm-objcopy` (13.7 MB → 82 KB on matmul; this is JTAG download time, since the kernel ELF is embedded into the host ELF and `dow -force`d). `.debug_line` is kept, so `kernel.linemap.json` / aiediag pc are unaffected; the full-DWARF original is parked at `<out>/kernel_debug`. `--keep-debug-loc` opts out. Must be `llvm-objcopy` — GNU binutils rejects the chess `e_machine 0x108`.

## Key Terms

| Term | Definition |
|------|------------|
| **PDI** | Pre-built hardware design; decouples HW/SW development |
| **GMIO** | Global Memory I/O; DDR ↔ AIE via NoC |
| **Shim tile** | Row-0 tile bridging NoC/DDR and AIE array |
| **MemTile** | Large-memory tile for caching between DDR and compute tiles |
| **BD** | Buffer Descriptor; configures a DMA transfer |
| **DSKernel** | Data-streaming kernel (receives via DMA, computes, outputs via DMA) |
| **EmitC** | MLIR dialect for C/C++ emission; final stage before `translateToCpp` |
| **xchesscc** | Synopsys compiler for AIE cores (from Vitis) |
| **PAL** | Board environment for running ELFs on real AIE hardware |

## Working rules

- **Never** let an API function exceed 200 lines.
- **Create a skill** whenever an issue is fixed, or whenever the user has to correct
  something you did wrong.
- **Keep the architecture docs current** — update them as part of the change, not after.
- **After each task, list every file changed or created.**
