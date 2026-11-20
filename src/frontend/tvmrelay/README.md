# `tvmrelay` — the Relay-era TVM frontend

Targets **TVM 0.16**, the last release with a complete `tvm.relay`.

`src/frontend/tvm` targets the modern Relax-era TVM (0.25+). **The two are
mutually exclusive** — one Python environment holds one TVM — so exactly one of
them works at a time. Which one is live is whichever `setup_tvm016.py` left
behind.

## Why 0.16

Relay was removed in the Relax/Unity transition. On 0.26 this is unconditional:

```
>>> from tvm import relay
ImportError: cannot import name 'relay' from 'tvm'
```

That is also why `src/frontend/tvm/relay_import.py` has been silently inert —
its `tvm_available()` catches that ImportError and returns False on every run,
so `walk.build_plan` always falls through to `model.layer_plan()` and the ONNX
graph never influences the result.

## Provisioning

```bash
python src/frontend/tvmrelay/setup_tvm016.py --verify-only   # report, change nothing
python src/frontend/tvmrelay/setup_tvm016.py --dry-run       # print every command
python src/frontend/tvmrelay/setup_tvm016.py --yes           # uninstall + build + install
```

It detects the installed TVM (out-of-process), uninstalls anything newer than
0.16, tries a 0.16 wheel, and falls back to a source build at tag `v0.16.0` in
`thirdparty/tvm-0.16/` (gitignored). It finishes by **importing a real ONNX
model through Relay** — not by reading a version string.

No 0.16 wheel exists on PyPI or tlcpack, so the source build is the real path.
Budget 10–25 min. Every removal is recorded to
`thirdparty/tvm-0.16/uninstalled.json` with a one-line restore command.

Flags: `--yes --src-dir PATH --jobs N --llvm[=PATH] --force-source --dry-run
--verify-only --allow-system`.

`USE_LLVM` is **OFF** by default. The Relay ONNX importer, the graph walk and
`target="c"` codegen need no LLVM; only `target="llvm"` JIT does. This host
carries LLVM 19, which postdates TVM 0.16. Pass `--llvm` to opt in.

## Importing it

Follow the `src/frontend` convention — put `.../src` on `sys.path` and import
`frontend.tvmrelay`:

```python
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import frontend.tvmrelay        # installs the onnx.mapping shim
from tvm import relay
```

Putting `.../src/frontend` on the path instead shadows the real `tvm` and `onnx`
with the empty sibling directories of those names, and fails as
`module 'tvm' has no attribute '__version__'`.

### Importing self-provisions

`import frontend.tvmrelay` calls `ensure_tvm016()`, which checks that the
importable TVM is 0.16 **with a working `tvm.relay`** and, if it is not, runs
`setup_tvm016.provision()` to install it. On a healthy environment this costs
about a second; otherwise the import uninstalls the current TVM and runs the
source build.

That is a large side effect for an `import`, so it announces itself on stderr,
and:

```bash
AIEHLC_TVM_AUTO_INSTALL=0 python your_script.py   # report and continue instead
```

After a successful install the new TVM is made importable **in the same
process** — the `.pth` file is only read at interpreter startup, so
`ensure_tvm016` adds the path and purges any stale `tvm` from `sys.modules`
itself. No restart needed.

`ensure_tvm016()` never fires recursively: `setup_tvm016.verify()` runs a
subprocess that imports this package, so it sets `_TVMRELAY_PROVISIONING=1` in
that subprocess's environment, and `ensure_tvm016` refuses to provision while
that sentinel is set. Without it, verify → import → provision → verify loops
forever.

Use `tvm016_status()` for a cheap `(ok, detail)` check with no side effects.

### The onnx.mapping shim

Importing the package installs the `onnx.mapping` compatibility shim
(`onnx_compat.py`). That is **required, not decorative**: onnx 1.16 deleted
`onnx.mapping`, TVM 0.16 imports `TENSOR_TYPE_TO_NP_TYPE` from it, and without
the shim `relay.frontend.from_onnx` raises `ImportError` on any current onnx.
The shim rebuilds the table from `onnx.helper` rather than pinning onnx back
below 1.16, which would put `torch` and `onnxruntime` — sharing this
environment — into a version fight over one dict.

## `deploy_flow.py` — ONNX ResNet-18 → Relay → int8 → C → layers → board ELF → CPU check

```bash
source script/setup.sh --path-set-only        # stage 6 needs the cross toolchain
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --image cat.jpg
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --skip-quantize
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --no-fuse
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --no-arm
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aie-offload
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aiegraph
```

Seven stages: `ensure_tvm016()` → download + Relay import → int8 quantize →
`target="c"` codegen (then a `gcc -fsyntax-only` check) → split into
per-layer folders → baremetal aarch64 `main.elf` → CPU reference
classification.

Flags: `--out-dir --image --skip-quantize --relay-ptq --global-scale
--no-split --no-fuse --flat --no-arm --aie-offload --aie-layers --aie-ops
--aie-mesh --aiegraph --aiegraph-ops`.

### Stage 3: which quantizer

**Default is ONNX PTQ** (`onnx_ptq.py`) — the model is quantized *before* Relay
sees it, so the import is already int8.

| | default (ONNX PTQ) | `--relay-ptq` |
|---|---|---|
| weights | symmetric int8, **per-channel** | symmetric int8, per-tensor |
| activations | **asymmetric** (zero-point) | symmetric |
| first conv | **quantized** | fp32 (`skip_conv_layers=[0]`) |
| dense layer | quantized | fp32 (`skip_dense_layer=True`) |
| input image | **quantized** | fp32 |
| needs LLVM | yes (in codegen, not quantize) | yes (in quantize) |
| top-5 vs fp32 | **order identical** | top-1 kept, 2–5 reordered |
| `main.elf` | 24.4 MB | **14.2 MB** |

`relay.quantize` *cannot* do asymmetric: its annotation op is
`simulated_quantize(data, scale, clip_min, clip_max)` — a scale and a clip
range, with no zero-point in the op, in `QConfig`, or in the C++ pass. That is
also why it skips the first conv, where raw-pixel dynamic range hurts a
symmetric scale most.

**Why the default is the bigger ELF — conv operands are stored int16.** The
values are genuine int8 (all 22 weight tensors verify within `[-127, 127]`);
this is storage width, not a quantization failure. It costs real bytes:
**99.5%** of `weights.bin` (23.4 of 23.5 MB) and **94%** of the activation
buffers (24.2 of 25.7 MB) are int16.

It is *not* the generic `qnn_conv2d_legalize` — that returns `None`.
`Target("c")` carries `keys=['cpu']`, so the **Intel** registration
`_qnn_conv2d_legalize_intel_cpu` fires (`legalizations.py:520`). Its gate is
`is_fast_int8_on_intel()` → `target_has_features("sse4.2")`, false for a
C-source target, so it falls to `helper_no_fast_int8_hw_legalization`, which
casts **data and kernel** to int16 and subtracts the zero-points eagerly
(`legalizations.py:192`).

**int8 operands are the default** (`pe_target.py`); `--no-pe-int8` restores the
int16 behaviour for A/B comparison. The right way to say "my PE takes int8" is
to give the target its own key, not to patch the x86 rule:

```python
Target("c -keys=pe_int8,cpu", host="c")       # pe_target.pe_target()
```

`qnn_conv2d_legalize` is a `tvm.target.generic_func`, so it tries the keys in
order: `pe_int8` wins, and `cpu` stays available for everything else —
operator strategies, schedules, `conv2d_alter_op` — which are all registered
there and must still resolve. Monkeypatching `register("cpu", override=True)`
also works but silently retargets every x86 build in the process.

The rule registered on that key is TVM's own `helper_change_dtypes_to_int8`,
written for Nvidia `dp4a`, and it is exactly the AIE convention:

```
x_i8  = x_u8 - 128
zp_i8 = zp   - 128        # 113 -> -15
```

with the shifted zero-point folded into the bias by `QnnConv2DCanonicalize` —
the same algebra as `conv2dstem`'s `+128·Σw` bias fold, reached from the Relay
side instead of by hand. It returns `None` once the operands are already int8,
which is what stops the legalize pass looping.

Measured end to end (`deploy_flow.py --local`, ResNet-18, dog.jpg):

| | `--no-pe-int8` (old) | **default** |
|---|---|---|
| conv weight dtype | `int16` | **`int8`** |
| `weights.bin` | 23,484,752 B | **11,851,606 B** |
| activation buffers | 25.7 MB | **18.6 MB** |
| `main.elf` (board) | 24,409,664 B | **12,795,600 B** |
| `main_local.elf` (x86) | 24,253,696 B | **12,654,648 B** |
| graph nodes / unique kernels | 30 / 28 | 120 / 64 |
| top-1 | 258, logit 12.017338 | **identical** |

The output is bit-identical because this is algebra on the same int8 values,
not a different quantization — the weights measure `[-127, 127]` with 255
distinct levels either way. The cost is the four-term expansion made explicit:
4× the graph nodes, an extra `nn_pad` and `cast_sum` reduction per conv, and
int32 scratch for the zero-point correction terms.

**What stays int32, and must.** Only the *operands* narrow. Measured by role:

| role | dtype | min | max |
|---|---|---|---|
| weight | `int8` | -127 | 127 |
| bias | `int32` | -3,834,368 | 2,873,728 |
| requant multiplier | `int32` | 1,073,764,180 | 2,147,291,785 |
| requant shift | `int32` | 6 | 27 |

The bias lives at the **accumulator** scale `s_x·s_w`, so it is ~10⁶× the real
value — 23 bits here. The multiplier is a fixed-point scale in `[2³⁰, 2³¹)`,
int32 by construction. Both are int32 in every int8 scheme (TFLite, ONNX QDQ,
PyTorch); narrowing them would be arithmetically wrong, not an optimization.
Together they are 0.5% of the blob — the 99.5% is the weights.

Note this is the *CPU* path reaching the same convention the AIE offload
already uses by hand: shift by 128 so the activation fits int8, with `128·Σw`
folded into the bias (skill **byocaieoffload**).

### It is switched OFF under `--aie-offload` / `--aiegraph`

int8 legalization makes the four-term expansion explicit, which restructures
the graph — 30 → 120 nodes, 28 → 64 layer folders — and **renumbers `layers/`**.
The ResNet stem moves from index **1** to index **6**. `--aie-layers` selects by
that index, so leaving int8 on would make the documented default
(`--aie-layers 1`, "the 7x7/s2 stem") pick a `layout_transform` and offload
nothing, silently.

So the default is three-state, not a boolean:

| invocation | operands | why |
|---|---|---|
| *(nothing)* | **int8** | the default |
| `--no-pe-int8` | int16 | forced off |
| `--pe-int8` | int8 | forced on |
| `--aie-offload` / `--aiegraph` | int16 | auto-off, with a printed reason |
| `--aie-offload --pe-int8` | int8 | forced on — **re-check `--aie-layers`** against `layers/` first |

The auto-off prints what it did:

```
[1/7] pe     : int8 operand legalization OFF -- it renumbers layers/ and --aie-layers selects by index
[1/7]          (pass --pe-int8 to force it; re-check your --aie-layers against layers/ if you do)
```

**`FakeQuantizationToInteger` is not optional** on this path. A QDQ import is
*simulated* quantization: left alone it stays `dequantize → fp32 op →
quantize`, the weights get folded back to fp32 constants, and codegen emits
`float*` kernels over a 46 MB blob — while every accuracy check still passes.
`to_integer_ops()` runs the pass and reports the op counts so a partial
conversion is visible rather than assumed.

`--skip-quantize` overrides both and emits fp32.

### The image, and what the two ends print

The ELF classifies a **real photo**, baked in as `input_image.h`. Default is
the pytorch/hub dog; `--image path-or-url` takes any other. Preprocessing is
*called*, not reimplemented — `example/model/resnet18py/classify.py:preprocess`,
the same function the reference classifier uses (resize shorter side to 256,
center-crop 224, `/255`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`,
NCHW). Reimplementing it would be the easiest way to make the board and the CPU
disagree for a reason that has nothing to do with the compiler.

`imagenet_labels.h` rides along so the board prints a name, not an index.

Stage 7 then classifies the same image with **onnxruntime on the same folded
ONNX** — an independent implementation, not a second call into the generated C
— and prints the top-5 in the same format the ELF does:

```
[7/7] cpu    : onnxruntime reference on dog.jpg
               1. class=258  Samoyed                        logit=12.4359
               2. class=279  Arctic fox                     logit=8.6504
               ...
[7/7]          the ELF prints these same lines; top1 must be class 258 (Samoyed)
```

Logits are printed, not just the class: a class index alone hides the small
numeric drift a miscompiled kernel actually produces.

Verified by compiling the ELF's exact sources for x86 and running them — both
images match the oracle on all five entries including logits (dog → Samoyed
12.4359; cat → Egyptian cat 15.6039).

## `--aie-offload` — layers onto the AIE via TVM BYOC

```bash
# once (and after any aiehlc.sh run for another source -- it resets aout/):
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file ./src/aietensorop/conv2dstem/conv2dstem.cc
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aie-offload            # layer 1 = stem
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aie-offload --aie-layers 1
PYTHONPATH=src python3 src/frontend/tvmrelay/byoc/verify_aie_stem.py               # bit-exact check
```

Off by default. After the CPU build (stages 4–5), the selected layers are
partitioned out of the Relay graph with BYOC, the C is rebuilt, and TVM's graph
executor calls the generated wrapper → `conv2d_stem_prepadded()` in
`aout/libconv2dstem.a` in their place. `--byoc-aie` was merged into this flag.
Design, and why each step is the way it is: **`doc/design/aie_offload_byoc.md`**.

`aie_offload.py` maps `--aie-layers` (stage-5 numbering) onto convs by **weight
fingerprint**, not position — the graph executor and Relay order the ResNet
shortcut convs differently — and refuses to offload if any target does not map
onto exactly one Relay conv with the same geometry.

Selection: `--aie-layers` takes an index, a comma list, or `all` (default `1`,
the 7×7/s2 stem); `--aie-ops` filters by kind. Non-conv layers are reported as
skipped with the reason. Only the stem has an AIE kernel today: any other
selected conv is reported ("no AIE kernel for this conv yet") and stays on the
APU.

Partitioning canonicalizes the whole graph before `relay.build`, so the CPU
remainder is built differently (65 kernels instead of 28) — still bit-exact:
`verify_aie_stem.py` compares all 1000 logits against the unpartitioned build.

### No tile-budget check — offload is blind

The frontend hands every selected layer to aiehlc as-is and does **not** predict
whether it fits a tile. Spatial tiling, halo, mesh partitioning and the per-tile
memory budget belong to aiehlc, which reports what does not fit
(`doc/design/byoc_aie_plan.md`). A Python-side estimate would duplicate — and
drift from — the backend's own policy, so the former `check_layer` /
`feasible` / "OVER BUDGET" report has been removed from both `aie_offload.py`
and `aiegraph_partition.py`.

## `aiegraph_partition.py` — whole-graph lift, then AIE/CPU partition

```bash
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aiegraph
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py \
    --aiegraph --aiegraph-ops conv_bn_relu,conv_bn,residual_add_relu,avgpool_fc
```

The other way into aiegraph, and the difference from `--aie-offload` is that it
takes **no layer selection**. It lifts the *entire* graph into one verified
`aiegraph.func` whose SSA edges are the model's real dataflow, then partitions:
a layer goes to the aiehlc kernel backend when **every** aiegraph op it expands
to is in `--aiegraph-ops`, and otherwise **reuses the TVM-generated CPU C**
from stage 5 untouched. Every layer gets a verdict and a reason in
`layers/partition.json`; the `.c` files are byte-identical with the flag on and
off.

On ResNet-18 that is 29 aiegraph ops over 24 graph invocations — 12 AIE / 12
CPU with the default conv-family op set, 21 / 3 with all four ops enabled.

| TVM fused layer | aiegraph | default verdict |
|---|---|---|
| `conv2d_add_relu` | `conv_bn_relu` | **AIE** |
| `conv2d_add` | `conv_bn` | **AIE** |
| `conv2d_add_add_relu` | `conv_bn` + `residual_add_relu` | CPU — see below |
| `global_avg_pool2d` + `dense_add` | one `avgpool_fc` (folded) | CPU — no AIE kernel |
| `max_pool2d` | *(none)* | CPU — not in the 4-op dialect |
| `batch_flatten` | *(none)* | CPU — dead kernel, graph elides it |

### Each AIE layer gets its own `lib<layer>.a`

`run_aie_pipeline` stops at source — `host.cc`, `kernel.cc`, `routing.cc`,
`aieml.bcf/prx` — and what it emits is only

```c
void host_canonicalized(XAie_DevInst* dev, void* t0, void* t1, void* t2);
```

the DMA/lock/launch body. No device init, no buffer allocation, no caller, no
compiled kernel ELF: nothing a host program could call, which is why those
folders used to hold artifacts and no library.

`aie_layer_lib.py` supplies the two missing pieces per layer:

1. **An op entry** — the generated counterpart of `conv2dstem.cc`'s
   `stem_run()`. `stem_run` is hand-written for one op and the pipeline never
   sees it (`run_aie_pipeline` is driven by tensor shapes and a kernel-body
   string, not by a C++ file), so the same wrapping is generated from the shape
   the aiehlc frontend emits for `__aie_launch`: partition → `set_kernel_elf` →
   `sync_for_dev` per input → `host_canonicalized` → `sync_for_cpu` per output
   → teardown. It is **appended to `host.cc`**, not written as a sibling file,
   because `hostcompile.sh` compiles a *fixed* set of sources and a standalone
   `.cc` would be silently left out of the archive. Re-running replaces the
   block rather than stacking copies.
2. **The archive** — `hostcompile.sh` with the layer's `aie/` as its working
   directory builds `kernel.cc` into the core ELF, embeds it (`_binary_kernel_
   <func>_*`), sees `host.cc` has no `main()` and therefore archives
   (`HOST_ENTRY_KIND=library`, skill **hostlibrarymode**).

```
layers/01_conv2d/aie/
  host.cc  kernel.cc  routing.cc  aieml.bcf
  build/lib01_conv2d.a          <- self-contained: op entry + host + kernel ELF
```

Verified on one layer: `T aie_01_..._run` (the entry), `T _Z18host_canonicalized…`
(the pipeline body), and the embedded core ELF closing *within* the archive
(`U _binary_kernel_conv_bn_0_start` in `host.o`, `D` in `kernel.o`).

`partition.json` records `archive` per layer, or `archive_reason` when it could
not be built — a box without the Vitis cross toolchain is a normal place to run
the earlier stages, so that is reported, not raised.

`arm_build` discovers them by wildcard and links them:

```make
AIE_LAYER_LIBS := $(sort $(wildcard $(LAYERS)/*/aie/build/*.a) \
                        $(wildcard $(LAYERS)/*/aie/*/build/*.a))
```

(the second pattern catches the `aie/conv_bn/`, `aie/residual_add_relu/`
subdirectories a multi-op layer produces). They link cleanly alongside
`libconv2dstem.a` despite both carrying the AIE runtime, because `ld` pulls only
the archive members that resolve an undefined symbol.

> **Not yet wired:** `graph_driver.c` still calls the **CPU** kernel for those
> layers, so the archives link but no member is pulled and `.text` is unchanged.
> Redirecting the call needs buffer marshalling — TVM's graph buffers are plain
> `static` arrays, while the entry requires `__Runtime_alloc_buffer` memory —
> and it changes the ELF's numerical output. That is the BYOC path's job today
> (`tvmgen_default_aie_main_0` → wrapper → `conv2d_stem_prepadded`).

**Partial eligibility is not offloadable.** A residual block's conv half is an
eligible `conv_bn`, but the TVM C for that layer is a *single fused function*
computing conv+bias+residual+relu, so there is no seam to split it at — taking
the conv to AIE would mean the residual add never runs. The whole layer stays
on the APU unless `residual_add_relu` is enabled too, in which case both
kernels are built into `aie/conv_bn/` and `aie/residual_add_relu/`
subdirectories. Single-op layers keep the flat `aie/` layout.

**Two counting traps this module is built around.** ResNet-18 has 22 distinct
generated symbols but **24** graph invocations — two shape-identical 64ch/56×56
kernels are each called twice — and dataflow lives on the *nodes*, not the
symbols. So the IR is built by walking `graph["nodes"]`, and layers are keyed
back by symbol. (`aie_offload.select_layers` instead zips manifest index *i*
against `call_nodes[i]` positionally, which desynchronizes at the first
repeated symbol and hands every layer from index 4 on **another layer's
geometry**; `--aiegraph` does not share that path.) Counts in `partition.json`
are over invocations, with `aie_dirs`/`cpu_dirs` giving the folder counts.

**The block args do not alias.** Two consumers of the same CPU-produced tensor
each get their own block argument, since `-1` ("not from an aiegraph op") is
the only thing the pybind boundary can say and it allocates a fresh one every
time. The IR is faithful about which ops run where and about every edge between
two aiegraph ops, but under-shares at the CPU boundary — harmless, because each
launch is standalone and the APU owns the buffers between them.

### What a no-LLVM TVM costs you

This tree's TVM 0.16 is built `USE_LLVM=OFF`. Four separate things break, and
each fails in a way that does not name LLVM:

| Symptom | Cause | Handled by |
|---|---|---|
| `AttributeError: ... no attribute 'target_has_feature'` | `topi.x86.dense_alter_op` probes AVX-512 via an LLVM-only FFI symbol. Fires even at `opt_level=0` on a 2-line model. | `_shim_target_features()` → False |
| `RuntimeError: LLVM version is not available` | `topi.x86.conv2d_int8.is_int8_hw_support` calls `llvm_version_major()` | same shim → 0 |
| `target.build.llvm is not enabled` | `target_host` defaults to `llvm`; the host module is built separately from the kernels | `Target("c", host="c")` |
| `target.build.llvm is not enabled` (again) | `FoldConstant` JIT-executes constant subgraphs against a target hardcoded in C++ (`fold_constant.cc:403`) | disable `FoldConstant` + `AlterOpLayout` |

**`--relay-ptq` is genuinely unavailable without LLVM.** `relay.quantize` →
`prerequisite_optimize` → `FoldConstant`, and that target is not reachable from
Python. Disabling `FoldConstant` does not help either: calibration then asserts
on `isinstance(expr.args[0], Constant)` (`_calibrate.py:154`) because it
*requires* folded constants. So that path detects this, prints the root cause
and the fix, and continues in fp32 rather than mislabelling the output.

**The default path still needs LLVM too, for a different reason.** Quantizing
outside TVM removes the dependency from *quantization* — stage 3 gets all the
way to `top-5 order identical` on a `USE_LLVM=OFF` build — but codegen then
dies with the same `target.build.llvm is not enabled`. The culprit is
`FakeQuantizationToInteger`, which calls `FoldConstantExpr` directly
(`fake_quantization_to_integer.py:35`), so `disabled_pass=["FoldConstant"]`
cannot reach it. Verified by swapping the no-LLVM `libtvm.so` back in.

So: LLVM is required either way. To enable `--relay-ptq`:

```bash
sudo apt install zlib1g-dev libzstd-dev     # the only two things missing
python src/frontend/tvmrelay/setup_tvm016.py --yes \
    --llvm <llvm-build>/bin/llvm-config
```

Host LLVM 19 is already detected as compatible — zlib/zstd were the sole cmake
failures.

### Two non-obvious choices

**ResNet-18 v1, not v2.** With `FoldConstant` off, TVM's `SimplifyInference`
rewrites BatchNorm into arithmetic containing `sqrt` over the variance
constants and leaves them unevaluated → `Unresolved call Op(tir.sqrt)`. So BN
is folded in *ONNX* first, via onnxruntime at `ORT_ENABLE_BASIC`. That works
only for v1 (post-activation, Conv→BN, all 19 BN nodes fold into the preceding
conv). v2 is pre-activation (BN→Conv) with nothing to fold into — 10 BN nodes
survive and `sqrt` returns. `ORT_ENABLE_EXTENDED` is also avoided: it emits
`FusedConv`, a Microsoft-domain op the Relay importer rejects.

**`tir.disable_vectorize`.** Without it the emitted C contains `int32_t16`,
which is not a C type; gcc rejects it. The file otherwise looks correct — right
size, right kernel count — so `verify_c` compiles it rather than trusting it.

Verified output: 340,219 chars, 22 kernels, compiles clean.

## `split_layers.py` — stage 5, one folder per layer

`relay.build` emits every kernel into one 4.5k-line translation unit. Stage 5
splits it into `layers/` beside it — **one folder per layer, in graph execution
order**:

```
layers/
  layers_common.h              includes + every forward declaration
  00_conv2d_add_relu/          stem conv
  01_max_pool2d/
  02_conv2d_add_relu/          ─┐
  03_conv2d_add_add_relu/       ┘ residual block 1
  04_conv2d_add_relu/  05_conv2d_add/  06_conv2d_add_add_relu/
  ...
  19_global_avg_pool2d/  20_dense_add/  21_batch_flatten/
  manifest.json  Makefile  README.md
```

The folder is a **per-layer working directory**, which is the point: reading
`layers/` top to bottom is reading the network in the order it executes, and
each folder is where that layer's AIE artifacts belong — `.bcf`, kernel
sources, the generated `host.cc`/`routing.cc` — sitting next to the C it
replaces. Those are per-layer, not per-op-kind, which is why the folder is
keyed by execution index rather than by operator.

`--flat` gives a single directory of `.c` files with no per-layer folders;
useful only when nothing else will ever sit beside them.

Each file is a **complete translation unit** — it includes
`../layers_common.h` and compiles alone, which the splitter verifies per file
with `gcc -fsyntax-only` rather than assuming. In `layers/`, `make` builds all
22 and `make 00_conv2d_add_relu` builds one layer.

These are not just for reading: **stage 6 links the board ELF from them.** Each
layer becomes a `.o` beside its source, archived into
`arm_build/liblayers.a` — so editing one operator recompiles one translation
unit rather than the whole module. See [`arm_build.py`](#arm_buildpy--stage-6-the-board-elf).

`layers_common.h` carries two things beyond TVM's own prologue, both needed
only once the layers are compiled *separately*:

- **mid-file material**, such as the BYOC prelude `aie_codegen.py` appends
  after TVM's kernels (its `conv2d_stem_prepadded` declaration). It sits past
  the first `extern "C"` guard, so it is not part of the prologue;
- **a synthesized prototype for every function defined but not pre-declared** —
  TVM declares its packed-ABI kernels up front, but a BYOC helper is only ever
  *defined*. In one file the definition preceded its caller; split apart, the
  AIE shim (layer 1) calls a wrapper defined in layer 64.

Without either, those layers compiled with an **implicit declaration** —
linkable on aarch64 since the pointer arguments pass in registers regardless,
but unprototyped and a hard error under `-Werror`. Note the contract this
implies: what gets hoisted is **declarations**. A BYOC codegen must keep its
weights as *function-scope* statics (as `aie_codegen._FUNC_TMPL` does), because
file-scope data here would be duplicated into all 65 translation units and
collide at link.

Folder numbers are **graph execution order**, read from `resnet18_graph.json`,
so the listing reads like the network — stem conv, maxpool, eight residual
blocks, avgpool, dense.

Names are the fused op list with `nn_` stripped and `expand_dims` dropped (it
is a bias broadcast, not a layer). So `conv2d_add_add_relu` reads as conv,
bias, residual, activation. The full TVM symbol is never lost — it is in each
file's header comment and in `manifest.json`.

Run it standalone against an already-generated tree:

```bash
PYTHONPATH=src python src/frontend/tvmrelay/split_layers.py worklocal/tvmrelay_deploy
PYTHONPATH=src python src/frontend/tvmrelay/split_layers.py worklocal/tvmrelay_deploy --flat
```

Re-splitting is safe to repeat: it clears previously-generated `.c`/`.o` from
`layers/` and its immediate subfolders, then removes whatever folders that
leaves empty. Files you add yourself — an AIE kernel beside the generated C,
a notes file — are left alone, and their folder survives with them.

## `network_md.py` — `network.md`, the compiled graph as a document

Written on every run, next to the generated C:

```bash
PYTHONPATH=src python src/frontend/tvmrelay/network_md.py worklocal/tvmrelay_deploy
```

`resnet18_graph.json` holds every shape, dtype and edge in the network, but as
400 KB of flat JSON whose connectivity is `node_row_ptr` arithmetic.
`network.md` is the readable form:

| section | what |
|---|---|
| Summary | counts, input/output tensor, which kernels are invoked more than once |
| Dataflow | a **Mermaid flowchart** — real topology, residual skips as dotted edges, every edge labelled with the dtype and shape it carries |
| Layers | one row per kernel call: folder, kernel, producers, and **`dtype[shape]` for every input, output and parameter, each parameter named by role** (`weight`, `bias`, …) |
| Layer detail | every argument of every call, with its **role**, kernel arg slot, graph name, entry index, dtype, shape, storage id and bytes |
| Parameters | the 137 constants rolled up by dtype and the 10 largest — where `weights.bin`'s 23.5 MB goes |
| Buffer reuse | what TVM's storage aliasing saves, and which storage ids are shared |

### One generator, three ways in

There is exactly **one** implementation — `network_md.write_network_md` — and
it is pure: given the same files on disk it emits byte-identical output
(verified: all three routes produce the same md5).

| route | when it runs |
|---|---|
| `deploy_flow.py` | automatically, stage 6, no flag — `deploy_flow.py:984` |
| `python network_md.py <out_dir>` | on demand, over whatever is on disk |
| `from ... import write_network_md` | same, from Python |

So they never disagree about *logic*. What differs is **what is on disk when
they run**:

- `deploy_flow.py` calls it **after** `build_c`, the layer split, and the
  offload/aiegraph stages, so the graph, `weights.bin`, `manifest.json` and
  the per-layer `.c` are all fresh and mutually consistent — it describes the
  graph that was *finally* compiled, not the first one.
- A standalone call reads the tree as it stands. On a complete tree that is
  identical; on a half-built or mixed one it faithfully describes what it
  finds.

It needs four inputs, and degrades differently for each:

| input | if absent |
|---|---|
| `<stem>_graph.json` | returns `None`, prints `skipped` — nothing to describe |
| `<stem>_params.bin` | **every constant reclassifies as an activation** → `parameters \| 0`, no `param` rows |
| `layers/manifest.json` | folder column shows `—`; roles fall back to the monolithic `resnet18.c` |
| the kernel `.c` | roles become a bare `param` |

The second one used to be silent, and it reads exactly like a model with no
weights. It now says so, on stdout and at the top of the document:

```
[net] WARNING: parameter blob missing resnet18_params.bin -- every weight is
      reported as an activation, so the document says 0 parameters.
```

Both the document and `arm_build/graph_driver.c` also carry a `graph-sha`
line, so "are these two from the same build?" is one command:

```bash
grep -m1 graph-sha network.md arm_build/graph_driver.c
```

Three things it is deliberately careful about, each a real source of confusion:

- **Entry indices, not node indices.** A node's k-th output is tensor
  `node_row_ptr[node] + k`, and that is what the shape/dtype arrays — and
  `graph_driver.c`'s `tensors[]` — are indexed by. So a row lines up with a
  `make TRACE=1` dump line with no translation.
- **Layer folders are matched by kernel symbol, not position.** Stage 5 writes
  one folder per *distinct* kernel while the graph makes more calls than that,
  because repeated fused groups share one function. Matching positionally
  mislabels every row after the first repeat.

### Three different things are called "layer 01"

They are separate numbering spaces and they only coincide by accident:

| number | what it counts | where you see it |
|---|---|---|
| **call index** | kernel calls in execution order (121) | `network.md`'s `#` column, and `[01]` in a `make TRACE=1` dump |
| **layer index** | *distinct* kernels, in first-use order (65) | the `NN_` prefix on a `layers/NN_*` folder |
| `--aie-layers` | the layer index **of the first build** | `--aie-offload` |

So `[01] tvmgen_default_fused_layout_transform` in a trace and a folder named
`01_contrib_conv2d_...` can both exist and have nothing to do with each other.
`network.md` pairs them for you: its `#` is the call index, and the `layer`
column is the folder that call's kernel actually lives in.

**And `layers/` accumulates folders across builds.** `_clear_stale` removes
generated `.c`/`.o` and then any directory that leaves empty — deliberately
keeping non-empty ones, because that is how an AIE-offloaded layer's `aie/`
artifacts survive a re-split. The cost is that a folder from an older,
differently-shaped graph lingers once anything else drops a file into it, so
`ls layers/` can show two `01_*` folders. Stage 5 now names them:

```
[5/7]          NOTE 2 stale folder(s) from an earlier build are still on disk
               and are NOT this graph: 01_contrib_conv2d_..., ...
```

and `manifest.json` records them under `stale_dirs`. The manifest is the
authority on what the current graph contains; the directory listing is not.
- **Parameters are excluded from the diagram.** 137 of the 143 buffers are
  weights; drawing them buries the topology. They are still accounted for in
  the per-layer tables.

### Parameter roles are read out of the C, not guessed

TVM names constants `p0, p1, …` by argument slot, which says nothing about
whether a tensor is a weight or a requantization shift. `kernel_param_usage`
classifies each one by **how the generated kernel uses it**:

```c
conv MAC    ... * ((int32_t)((int16_t*)p1_1)[...])            -> weight
epilogue    acc + p2_1[c]                                     -> bias
                - p3_1[c]                                     -> zero-point
            * ((int64_t)p4_1[c])                              -> requant multiplier
            >> ((int64_t)(p6_1[c] + 31))                      -> requant shift
```

Three things this gets right that a plausible shortcut does not:

- **Slot, not name.** Layer 29 takes `p131..p136` from the graph and calls them
  `p1..p6` inside the kernel. Keying on the graph name mislabels every layer
  after the constants stop being numbered from 1.
- **Cast width, not rank.** The two multiplies are told apart by `int32_t` vs
  `int64_t`, because the accumulator multiply must widen and the MAC must not.
  A "rank ≥ 4 means weight" rule looks fine on convs (`[16,1,7,7,3,4]`) and
  silently calls the **dense** weight (`[125,512,8]`, rank 3) a multiplier —
  that bug gave 20 weights for a network that has 21.
- **`unused` is reported, not named.** Every quantized layer has a slot the
  body never reads (the left-shift operand of
  `fixed_point_multiply_per_axis`, constant-folded to zero). Calling it
  "left shift" would be a tidier lie. For the same reason `copied` is its own
  role — the kernel re-lays-out that constant into NCHWc without doing
  arithmetic on it, so it has no additive or multiplicative role to report.

Roles are read from the per-layer split when stage 5 ran, and from the
monolithic `resnet18.c` otherwise, so `--no-split` is covered too. If neither
is on disk the role is a bare `param`: with no C to read, `unused` in
particular must not be guessed, since it asserts the kernel never touches the
tensor.

Coverage is checkable: 21 weight (20 convs + 1 dense), 21 bias, 21 requant
multiplier, 21 requant shift, 21 unused, 32 zero-point — 137, every parameter
accounted for, none falling through to a generic label.

## `arm_build.py` — stage 6, the board ELF

```bash
source script/setup.sh --path-set-only        # cross toolchain + XILINX_VITIS
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py
PYTHONPATH=src python src/frontend/tvmrelay/arm_build.py worklocal/tvmrelay_deploy
PYTHONPATH=src python src/frontend/tvmrelay/arm_build.py --no-make   # generate only
PYTHONPATH=src python src/frontend/tvmrelay/arm_build.py --local     # + x86 ELF
```

`relay.build(target="c")` gives kernels but **not a program**: no `main`, no
weights in memory, nothing calling the kernels in order. TVM normally supplies
that from its C++ graph *runtime*, which is host-side. Stage 6 generates the
missing pieces from `resnet18_graph.json` instead, into `arm_build/`:

| file | what it is |
|---|---|
| `graph_driver.c` | one static buffer per `storage_id`, weights pointed into the blob, one kernel call per graph node in order |
| `tvm_runtime_shim.c` | the four runtime symbols the kernels reference — a bump allocator plus an error sink |
| `main.c` | entry point, feeds `input_image.h`, prints the top-5 with names and logits, the per-phase timing, and the `device_teardown done` line `verify_host.sh` greps |
| `input_image.h` | the preprocessed photo as `static const float[150528]` |
| `imagenet_labels.h` | the 1000 class names |
| `tvm/runtime/c_*_api.h` | baremetal stand-ins so `resnet18.c` is used **verbatim**, not patched |
| `weights.bin` | `resnet18_params.bin`, linked in via `ld -r -b binary` |
| `liblayers.a` | the per-layer objects from stage 5, one per operator — **the kernels** |
| `Makefile` | the build recipe |
| `README.md` | generated too — the build targets, the file inventory with this run's actual numbers, `TRACE=1`, the `DLDataType` packing, and the `make local` flags, written next to the sources so it cannot drift from the Makefile it documents |

### Where the kernels come from

By default, **the per-layer sources stage 5 split out**: `layers/NN_op/NN_op.c`
→ `NN_op.o` beside its source → `liblayers.a`. Editing one operator recompiles
one translation unit:

```bash
make -C worklocal/tvmrelay_deploy/arm_build          # 1 TU + archive + link
```

The monolithic `resnet18.c` is the fallback, used when `layers/` is absent
(`--no-split`). Exactly one of the two is wired into the Makefile, so no kernel
is compiled twice; if neither resolves, `make` stops with a named error rather
than a wall of undefined `tvmgen_default_*` references. Override by hand:

```bash
make -C ... LAYERS=/nonexistent KERNELS=/path/to/resnet18.c
```

Cost of splitting, measured on ResNet-18 int8 with `--aie-offload`: 1,312,864
vs 1,312,608 bytes of `.text` (**+0.02%**, from slightly less cross-TU
inlining), identical 65 `tvmgen_*` symbols.

`graph_driver.o` pulls the members it calls out of the archive, and one member
may call another — the AIE packed-ABI shim calls the BYOC wrapper, a separate
layer. GNU ld rescans an archive until no new undefined symbols are resolved,
so intra-archive references need no `--start-group`.

### What the ELF prints

After the top-5, `main.c` reports what each phase cost, measured with the Arm
generic timer via `include/aie_timer.h` — the same `XTime` /
`COUNTS_PER_SECOND` helpers the tilinglinalg host code uses:

```
resnet18: timing (timer 100000000 Hz, 1 tick = 10.0 ns)
  init            1.234 ms
  inference      24.567 ms   <- graph_run()
  top5            0.089 ms
  total          25.890 ms
  throughput      40.70 inferences/s
```

Three numbers rather than one total, because only one of them is the model:
`init` is a one-off weight-pointer setup, `top5` is the argmax, and
**`inference` is `graph_run()`** — the number an AIE offload has to beat. The
image copy sits outside the window deliberately: that is the harness feeding
the model, not the model running.

The timer frequency is printed because a tick count means nothing without it,
and on this part the generic timer runs nowhere near the CPU clock. `-DAIE_GEN=5`
in `CFLAGS` selects `aie_timer.h`'s `xiltimer.h` branch, which is what this
cortexa78 BSP ships — the default branch wants `xtime_l.h`, which it does not
have.

`device_teardown done` stays the **last** line, so `verify_host.sh` is
unaffected.

### Python generates, `make` builds

The recipe lives in the generated `Makefile`, not inside Python, so the build
the flow ran is the same one you re-run by hand:

```bash
make -C worklocal/tvmrelay_deploy/arm_build                     # rebuild changed
make -C worklocal/tvmrelay_deploy/arm_build clean && make -C ... # from scratch
make -C ... CROSS=aarch64-linux-gnu- CPU=cortex-a72             # retarget
make -C ... -j8
```

Edit a generated `.c`, run `make`, and only that file recompiles — the 340 KB
kernel translation unit is not rebuilt. `--no-make` (or `run_make=False`) stops
after generating. Without `setup.sh` sourced the sources and Makefile are
**still written**; only the build is skipped, with the `make -C ...` line to run
once the toolchain is on PATH. No traceback.

Toolchain matches `script/hostcompile.sh` and `quant/build_demo.build_aarch64`:
`aarch64-none-elf-gcc`, `-mcpu=cortex-a78`, cortexa78 BSP, `--specs=nosys.specs`,
BSP linker script.

One Makefile subtlety worth not re-discovering: `LDLIBS` must stay a single
unbroken token. The commas are `-Wl` separators, so a `\` line continuation
splits it and ld goes looking for a library literally named
`-lxilstandalone,...`.

### Tracing tensors between layers — `make TRACE=1`

`graph_driver.c` carries a dump of every kernel's inputs and outputs behind
`#ifdef GRAPH_TRACE`. A normal build compiles none of it (`.text` is
byte-identical); `TRACE=1` turns it on for either target:

```bash
make TRACE=1                       # board ELF, traced
make local TRACE=1 run-local       # host ELF, traced, run it
make local TRACE=1 TRACE_ELEMS=16  # 16 elements per tensor instead of 8
```

```
[00] tvmgen_default_fused_divide_round_add_clip_cast_subtract_layout_transform
   in  tensors[0] float32 [1,3,224,224] = -1.929532 -1.929532 -1.912407 ...
   in  tensors[1] int16 [] = 113
   out tensors[2] int16 [1,1,224,224,3] = -104 -98 -87 -104 -98 -86
[01] tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_..._a7e8b96f394601f9_
   in  tensors[2] int16 [1,1,224,224,3] = -104 -98 -87 -104 -98 -86
   in  tensors[3] int16 [16,1,7,7,3,4] = -2 0 -52 25 -18 0
   out tensors[9] uint8 [1,16,112,112,4] = 46 97 14 0 30 89
```

Two things this gets right that a hand-added `printf` does not:

**The dtype is decoded.** `DLDataType` is a 4-byte *struct*
`{code, bits, lanes}`, so `printf("%d", t.dtype)` is undefined behaviour that
happens to print the little-endian packing `code | bits<<8 | lanes<<16` —
int16 comes out as `69632` (`0x011000`) and uint8 as `67585`. `dl_dtype_str()`
prints `int16`. The codes are 0 int, 1 uint, 2 float, 4 bfloat.

**It dumps elements, not bytes.** A byte dump of an int16 tensor reads
`98 ff 9e ff`, which is little-endian `-104, -98` — the single most common way
to misread this output. `graph_dump_tensor()` switches on `(code, bits)` and
prints decoded values, clamping to the element count so a scalar parameter
(`ndim 0`) is not read past the end of its 2-byte buffer.

`dl_code_name`, `dl_dtype_str` and `graph_dump_tensor` are `static inline` and
always emitted, so a `printf` you drop into `graph_run()` by hand can use them
immediately. But prefer `TRACE=1`: **`graph_driver.c` is generated**, and
`deploy_flow.py` overwrites hand edits on the next run. Change
`arm_build._trace_lines`/`_DEBUG_HELPERS_C` if the trace itself needs to change.

### `--local` — the same C, run on this host

```bash
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --local
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --local --no-arm
make -C worklocal/tvmrelay_deploy/arm_build local       # build main_local.elf
make -C worklocal/tvmrelay_deploy/arm_build run-local    # build it and run it
```

`main_local.elf` is the **same generated C** — same `graph_driver.c`, same
`liblayers.a` sources, same `weights.bin`, same `main.c` — linked for x86 with
the host gcc. It prints the same top-5 with logits the board would, so stage
7's onnxruntime reference can be checked in a second instead of by flashing a
board, and `deploy_flow.py` prints the verdict:

```
      local vs cpu: MATCH class=258 (Samoyed) logit 12.0173 vs 12.4359
```

It needs **no Vitis toolchain**, which is why it composes with `--no-arm`, and
it is built *before* the cross-toolchain gate for exactly that reason.

It is not the cross build with a different `-mcpu`. The BSP include/lib paths
and the whole baremetal link recipe (`--specs=nosys.specs`,
`--defsym end=__bss_end__`, `-T lscript.ld`) are **dropped**, and two flags
have to be **added** — the ones a hand-rolled `gcc resnet18.c` always misses:

| flag | without it |
|---|---|
| `-D__AIESIM__` | `aie_timer.h` falls through to `#include "xtime_l.h"`, a BSP header that does not exist off-target → `fatal error: xtime_l.h: No such file` |
| `-std=gnu11` (**not** `-std=c11`) | `clock_gettime`/`CLOCK_MONOTONIC` are POSIX, which strict ISO mode hides → `'CLOCK_MONOTONIC' undeclared` **inside** `aie_timer.h` |
| `-lm` | the BSP supplied libm inside `-lxil`'s group; here it is explicit |

Objects go to `arm_build/localobj/` with flattened basenames, so a local build
never clobbers the aarch64 objects (which sit next to their `.c` under
`layers/`). The directory is deliberately **not** called `local` — that is the
phony target's name, and make would read `localobj/x.o: ... | local` as
circular, drop the order-only prerequisite, and then fail with
`can't create local/x.o: No such file or directory`. `vpath` is what lets a
flat `localobj/NN_op.o` find its source back under `layers/NN_op/`.

An **AIE-offloaded build cannot be linked here** and says so up front: the
archives are aarch64 baremetal and their kernels execute on the array. Re-run
without `--aie-offload`/`--aiegraph` for a local ELF.

This is also what caught `TVMValue args[8]` in `graph_driver.c`: the NCHWc
convs take **10** arguments, so `args[8]`/`args[9]` ran off the end into
`codes[0..1]`. The kernels ignore `arg_type_ids`, which is the only reason the
board tolerated it — on x86 it trips the stack protector immediately. Both
widths are now computed from the graph (`graph_widths`, emitted as
`GRAPH_MAX_NDIM`/`GRAPH_MAX_ARGS`).

**Why not TVM's AOT executor.** `Executor("aot")` does emit a real
`tvmgen_default_run`, which looks like the obvious answer. It was tried and
rejected: AOT inlines every weight into the C source as initializer text — a
**188 MB** `.c` for ResNet-18 — and `link-params=False` does not prevent it.
Reading weights from a linked blob keeps the source at 340 KB and the weights
in `.rodata`.

**Verified numerically, not just linked.** The same generated driver compiled
for x86 and run against the same ramp input matches onnxruntime on the folded
ONNX: identical top-5 (`858, 539, 430, 852, 553`), max abs diff **7.2e-06**.
A driver that links but computes nonsense would pass every other check here.

Footprint: 69 MB `.bss` (52 MB graph buffers + 16 MB kernel workspace) and
46 MB `.data` of weights, topping out at 0x470971C0 — inside the 2 GB DDR the
BSP maps, with ~956 MB spare. Note `size` reports ~1.1 GB of bss because it
counts the BSP's 1 GB `.heap` reservation, which this program does not use.

**No AIE yet.** This is the CPU path only. `aie_runtime.c`, `routing.cc` and an
embedded kernel ELF are what `hostcompile.sh` adds, and no layer has them yet —
when a layer folder grows a `.bcf` and a kernel, that layer's call site in
`graph_driver.c` is what changes.

### There is no `relu.c` by default, and that is not a bug

TVM's `FuseOps` welds elementwise ops into their producer, so the compiled unit
genuinely *is* conv2d+bias+residual+relu: one loop nest, one function. No
amount of text splitting recovers a separate relu — it was never emitted. For
true per-op files, build with fusion off:

```bash
PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --no-fuse
# -> 30 layers: 00_conv2d.c  01_add.c  02_relu.c  03_max_pool2d.c ...
```

giving folders `conv2d/`, `add/`, `relu/`, `max_pool2d/`, …

`--no-fuse` is **not** `disabled_pass=["FuseOps"]`. FuseOps does two jobs —
grouping ops, and wrapping each group in a primitive `Function` — and the graph
executor codegen needs the second unconditionally; disabling the pass outright
fails with `Expected the operator to be a global var, but got Op`
(`graph_executor_codegen.cc:452`). Instead `build_c` runs
`FuseOps(fuse_opt_level=0)` itself, which returns right after `InitGroups`
(`graph_partitioner.cc:104`) so each node is its own group *and still wrapped*,
then disables the build's own FuseOps so it cannot re-fuse.

### Two details worth knowing

**`21_batch_flatten.c` has zero callers.** Its header says so
(`invoked : never -- not reached by the graph (dead kernel)`). The graph elides
the flatten into a `__nop` because the reshape is layout-free. Uncalled kernels
are emitted anyway, after the reachable ones — worth seeing, not hiding.

**A stem can repeat; the index disambiguates.** ResNet has eight
`conv2d_add_relu` kernels, so files are `00_`, `02_`, `04_`… — the execution
index is what makes each filename unique, not the name.

## Verified state

```
tvm 0.16.0 | numpy 2.2.6 | onnx 1.22.0 (shim)
relay.frontend.from_onnx(resnet18-v2-7.onnx) ->
  20 nn.conv2d, 19 nn.batch_norm, 18 nn.relu, 9 add,
  1 nn.max_pool2d, 1 nn.global_avg_pool2d, 1 reshape, 1 nn.dense
  ret type: Tensor[(1, 1000), float32]
```

Note the ONNX initializers (99 of them) arrive as inline `relay.Constant` nodes,
**not** in the returned `params` dict — which comes back empty, and stays empty
under `freeze_params=True`. Anything walking this IR for weights must read the
constants out of the graph.
