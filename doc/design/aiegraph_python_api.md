# Design: `aiegraph` — a user-facing Python API for model → int8 C → AIE offload

Status: **implemented** for M0–M3 except where noted (2026-10-14): `src/aiegraph/`,
installed by `pip install -e .` (`pyproject.toml`); user guide
[doc/tutorial_aiegraph.md](../tutorial_aiegraph.md). Differences from this design:
M0 was not needed — `deploy_flow.run` gained one keyword (`calib=`) instead of
being split, and `aiegraph` drives it whole; `ag.open(workdir)` was added to adopt
an existing deploy tree; `ag.new_kernel`, `emod.ir()`, `_repr_html_` and the native
PT2E lowering (M4/M5) are not implemented. TVM 0.16 is attached from
`thirdparty/tvm-0.16` at run time rather than installed by pip (no wheel exists).
Date: 2026-10-14.

## 1. What this is for

Today a user who wants a network on the A78 with some layers on the AIE has to
know four entry points, three numbering schemes and a dozen flags:

```bash
PYTHONPATH=src python3 src/frontend/tvmrelay/deploy_flow.py --aie-offload --aie-layers 1 ...
PYTHONPATH=src python3 -m frontend.tvmrelay.aiekernel check src/.../conv2dstem_uk.cc@1
PYTHONPATH=src python3 src/frontend/pytorchmlir/deploy_torch.py      # stops at MLIR
```

The goal is one small Python package, `aiegraph`, that reads like this:

```python
import aiegraph as ag

model = ag.load("resnet18")
module = ag.convert(model, target="a78")

# Register or describe the kernel contract.
module.register_kernel(
    name="conv2dstem",
    source="src/aietensorop/conv2dstem/conv2dstem_uk.cc",
    op="conv2d",
    target="a78",
    constraints={
        "input_layout": "NCHW",
        "weight_layout": "OIHW",
        "dtype": "int8",
        "quantization": "per_channel",
    },
)

# Inspect whether the named graph region is compatible.
check = module.check_offload(
    region="layer1.0.conv1",
    kernel="conv2dstem",
)
print(check.summary())

# Explicit assignment for bring-up; later this can be automatic.
module.assign(region="layer1.0.conv1", backend="aie:conv2dstem")

# Make fallback policy explicit.
plan = module.partition(
    fallback="cpu",
    optimize_for="latency",
)
print(plan.report())

artifact = module.build(
    target="a78",
    output="resnet18_a78",
)
```

This runs as written (§4.8). `conv2dstem` fits only the stem, so `layer1.0.conv1`
(a 3×3, 64→64 conv) is reported `NOT READY` at detect, and `partition` keeps it on the
A78 (`fallback="cpu"`). With `region="conv1"` the stem goes to the AIE.

or, in one line:

```python
module = ag.compile(model, "aie", "layer1=aie/src/conv2dstem.cc")
```

Printed tables in this document are **illustrative** — they show the intended
format, filled with numbers from today's ResNet-18 runs where those exist
(`check`/`offload` lines are real `aiekernel` output), not captured from code
that does not exist yet.

Design principles (§3) come from the frameworks users already know (§2);
everything under the hood already exists (§5) — this is a front door, not a new
compiler.

### Non-goals

* No new compiler passes, dialect ops or runtime code. The facade calls
  `frontend.tvmrelay`, `frontend.pytorchmlir` and `frontend.tvmrelay.aiekernel`.
* Never touches a board implicitly. Running on hardware is an explicit,
  separate call (`emod.run(board=...)`, out of scope for v1).
* No automatic writing of AIE kernels. Offload means "use THIS kernel for
  THAT layer"; the facade checks and wires it.

---

## 2. Prior art — what we borrow and what we deliberately do not

| framework | the API users know | what `aiegraph` borrows | what it does not |
|---|---|---|---|
| **PyTorch 2** | `torch.compile(model, backend=...)` — one call, sensible defaults, a backend string; `torch.export` → `ExportedProgram` as the stable captured graph | the **one-call `compile(model, target)`** with a short target string; capture once into an explicit graph object the user can inspect | dynamo's implicit re-tracing / guards: our graph is static and int8, so capture is explicit and happens once |
| **PyTorch PT2E / ExecuTorch** | `prepare_pt2e(m, quantizer)` → calibrate → `convert_pt2e`; ExecuTorch `to_edge(...)` then `to_backend(Partitioner)` where a partitioner *tags* nodes for a delegate | quantization as a **named, swappable step with a calibration set**; offload as **tagging layers for a delegate** (our "delegate" = a user kernel library), then lowering | requiring the user to write a Partitioner class: here the partition is a dict `{"layer1": "kernel.cc"}` or a selector |
| **TVM** | `relay.frontend.from_onnx` → `relay.build(mod, target, params)`; BYOC = `MergeComposite` → `AnnotateTarget` → `PartitionGraph` → `relay.ext.<name>` codegen; target strings like `"c -keys=..."` | the **frontend → IR → build(target)** staging; BYOC's idea that offload is **per subgraph with a fallback to CPU**; target strings | exposing Relay passes, `PassContext`, or pattern tables to the user. TVM stays an implementation detail (and the only place a TVM version can leak is `ag.doctor()`) |
| **MLIR / IREE** | `iree.compiler.compile_file(path, target_backends=["llvm-cpu"])`; textual IR at every stage; pass pipelines as strings | **every stage leaves an inspectable artifact** (`emod.ir("aiegraph")`, `emod.layer(1).c_source`); one function from model to artifact with a backend list | asking users to compose pass pipelines |
| **Triton** | `@triton.jit` kernel + `kernel[grid](...)`; `triton.testing.assert_close`; autotune configs | **low ceremony for the kernel author**: a user kernel is one file, checked against a reference with one call (`offload_check`), like Triton's "write kernel, compare to torch" loop. The repo's own `aietriton` already uses the `@jit` idiom | Python-defined AIE kernels (that is `aietriton`'s job, and its output can be an `aiegraph` kernel later, §8) |
| **ONNX Runtime** | `InferenceSession(path, providers=["XEP", "CPUExecutionProvider"])` — ordered fallback; `quantize_static(model, out, calibration_data_reader)` | **ordered fallback per layer**: anything not offloaded runs on the A78 in TVM's C, always; static int8 PTQ with a calibration reader (already used by `onnx_ptq.py`) | runtime provider negotiation: placement is decided at compile time and printed |

The common lesson: **one obvious call, a printed summary, inspectable stages,
and an escape hatch at each stage.**

---

## 3. Design principles for "easy"

1. **One obvious path, three lines.** `load → embedded_convert → offload` covers
   the common case; `compile` collapses it to one call.
2. **Defaults that work on this repo's reference model.** `ag.load("resnet18")`
   + `embedded_convert(m)` reproduces today's `deploy_flow.py` default (int8,
   per-layer split, A78 baremetal driver) with no flags.
3. **Everything prints itself.** Every object has a readable `__repr__` (a
   table) and `_repr_html_` for notebooks. A user never needs to open
   `partition.json` or `manifest.json`.
4. **Errors say what to do next.** Every failure is an `AiegraphError` with
   `.stage`, `.reason` and `.hint`, e.g.
   `OffloadCheckError: layer4 stopped at detect: arg 0 dim 1 is 16, kernel says 1 — this kernel fits layer1 only; try ag.detect("./aie/conv2d_fuse.cc")`.
5. **Names a human can type.** `"layer1"`, `1`, `"layer4-9"`, `"conv*"`, a
   `Layer` object — all accepted wherever a layer is expected (§4.4).
6. **Progressive disclosure.** Simple calls take strings; every call also takes
   keyword options for experts (`calib=`, `quantizer=`, `trials=`, `force=`),
   and every object exposes the underlying paths (`emod.workdir`, `.artifacts`).
7. **Safe by default.** `offload_check` writes nothing outside the workspace;
   `offload` refuses anything that failed a gate unless `force=True`; nothing
   runs on a board unless asked.
8. **Idempotent and cached.** Re-running a notebook cell does not redo int8
   calibration or an aiehlc build whose inputs did not change (hash-keyed, as
   `aiekernel.build` already does for aiehlc).

---

## 4. The API

### 4.1 Loading a model (`ag.load`, `ag.from_onnx`, `ag.from_torch`)

```python
m = ag.load("resnet18")                         # name -> model zoo (ONNX, cached under ~/.cache/aiegraph)
m = ag.load("resnet18", source="torchvision")   # same net from torchvision weights
m = ag.load("path/to/model.onnx")               # a path: ONNX by suffix
m = ag.from_onnx("model.onnx")                  # explicit
m = ag.from_torch(torch_module, example_inputs=(torch.randn(1, 3, 224, 224),))
```

Returns an `ag.Model`: framework-neutral, fp32, not yet compiled.

```text
>>> m
Model resnet18  (onnx, opset 13, fp32)  input data float32[1,3,224,224] -> output float32[1,1000]
  69 ops: Conv x20, Relu x17, Add x8, MaxPool, GlobalAveragePool, Flatten, Gemm
  next: ag.embedded_convert(m, target="a78")
```

`ag.download_model(name)` is an alias of `ag.load(name)` that only fetches
(the user asked for `onnx.downloadmodel("resnet18")`; it lives in our namespace
so it can carry the repo's cache and checksum).

**PyTorch path.** v1 exports to ONNX (`torch.onnx.export`, opset 13) and joins
the ONNX path, because only that path reaches per-layer C today
(`pytorchmlir` stops at linalg MLIR). The choice is recorded on the model
(`m.lowering == "torch->onnx"`). A native path (PT2E int8 → torch-mlir →
aiegraph/C) is milestone M4 (§7); the API does not change when it lands —
`from_torch(..., lowering="pt2e")` selects it.

### 4.2 Converting for an embedded target (`ag.embedded_convert`)

```python
emod = ag.embedded_convert(m, target="a78")                    # all defaults
emod = ag.embedded_convert(m, target="a78", quantize="int8",
                           calib=["dog.jpg", "cat.jpg"],      # paths, arrays, or a DataLoader
                           workdir="worklocal/resnet18_a78")
```

What it does (all existing stages of `deploy_flow.run`, minus the AIE ones):
quantize to int8 (ONNX PTQ by default, `quantizer="pt2e"` for torch models),
fold zero points, build TVM C with `pe_int8`, split one C file per layer,
generate the A78 driver/Makefile, compute the onnxruntime reference. It does
**not** run `make` until `emod.build()`.

Targets:

| target | meaning |
|---|---|
| `"a78"` | Cortex-A78 baremetal C (today's board ELF), all layers on the CPU |
| `"x86"` / `"cpu"` | the same C for this host (`main_local.elf`) — always available, used for checks |
| `"aie"` | `"a78"` plus offload; only meaningful with an offload plan |

Returns an `ag.EmbeddedModel`:

```text
>>> emod
EmbeddedModel resnet18 -> a78   int8 (onnx_ptq, 1 calib image)   workdir worklocal/resnet18_a78
 int8 vs fp32 top-5: identical order (Samoyed 12.15 vs 12.44)
 layer    calls  op                              in -> out                             on
 layer0   1      quantize+pad (layout)           float32[1,3,224,224] -> int8[1,1,230,230,3]   cpu
 layer1   1      conv2d 7x7/2 requant relu       int8[1,1,230,230,3] -> uint8[1,16,112,112,4]  cpu
 layer2   1      max_pool2d                      ...                                           cpu
 layer4   2      conv2d 3x3/1 requant relu       int8[1,16,58,58,4] -> int8[1,16,56,56,4]      cpu
 ...
 33 layers, 41 calls, 11.9 MB weights
 next: ag.offload_check("layer1", "<kernel.cc>")   or   emod.detect("<kernel.cc>")
```

Useful members: `emod.layers` (list of `Layer`), `emod.layer("layer1")`,
`emod.layer(1).c_source` (path to the TVM C), `emod.network_md`,
`emod.build()`, `emod.run_local()` (x86, returns top-k),
`emod.accuracy()`, `emod.export(dir)`.

### 4.3 Checking and offloading a layer

```python
rep = emod.offload_check("layer1", "./aie/conv2d_fuse.cc")
rep = ag.offload_check("layer1", "./aie/conv2d_fuse.cc")       # same, on the current model (§4.6)
```

Runs the `aiekernel` gates that need no Vitis and no board — lint, detect,
calibrate, whole-network emulate — and returns an `ag.CheckReport`:

```text
>>> rep
offload_check conv2dstem (./aie/conv2d_fuse.cc) -> layer1      READY
  lint       clean
  detect     layer1 matches (1 call); also fits: -
  calibrate  BIT-EXACT, 4 trials x 802,816 outputs
  emulate    logits bit-identical (Samoyed 12.0173), swap fired 1/1
  next: ag.offload(("layer1", "./aie/conv2d_fuse.cc"))
>>> rep.ok
True
```

On failure `rep.ok` is `False` and the report names the gate, the reason and a
hint; `rep.raise_for_status()` turns it into an exception for scripts.

```python
emod.offload(("layer1", "./aie/conv2d_fuse.cc"))                 # one pair
emod.offload({"layer1": "aie/stem.cc", "layer4": "aie/c3.cc"})  # a plan
emod.offload("./aie/conv3x3.cc")                                 # every layer it fits ("auto")
emod.offload(("layer4-9", "./aie/conv3x3.cc"))                   # a range
ag.offload(("layer1", "./aie/conv2d_fuse.cc"))                   # current model
```

`offload` = `offload_check` + aiehlc build + link gate + driver swap
(`aiekernel.offload`). It returns an `ag.OffloadReport` whose table shows the
new placement column (`cpu` / `aie:<kernel>`), and then `emod.build()` links
`main.elf`. Rules the report states up front rather than at link time:

* a layer offloads all of its calls (layer4 is called twice; both swap);
* **one aiehlc library per ELF** (the runtime's single `routing()`): a plan
  naming two different kernel *sources* is rejected at plan time with that
  reason and the hint "put both kernels in one source";
* without Vitis the build is skipped and the report says so; `offload_check`
  is still fully useful.

Discovery helpers:

```python
emod.detect("./aie/conv3x3.cc")       # which layers fit, and why the others don't
ag.lint("./aie/conv3x3.cc")           # static checks only, no model needed
ag.new_kernel("conv3x3", like="layer4", dir="./aie")   # scaffold: spec + reference stub (§8)
```

### 4.4 Naming layers

A layer reference is resolved by one function, `emod.resolve(ref)`, and is
accepted everywhere:

| ref | means |
|---|---|
| `"layer1"`, `1`, `"01"` | stage-5 layer index (`layers/01_*`) |
| `"layer4-9"`, `[4, 9]`, `range(4, 10)` | several |
| `"conv*"`, `"*7x7*"` | glob over the human op label shown in the table |
| `"call[08]"` | the layer owning that call (what `TRACE=1` dumps print) |
| `"auto"` / `None` | every layer the kernel's spec matches |
| a `Layer` object | itself |

The three numbering spaces of CLAUDE.md (call / layer / node) stay internal;
the user sees `layerN` everywhere and `call[NN]` only where a dump is involved.

### 4.5 The one-call form: `ag.compile`

```python
emod = ag.compile(model, "cpu")                                  # == embedded_convert(target="a78") + build
emod = ag.compile(model, "aie", "layer1=aie/src/conv2dstem.cc")  # + offload_check + offload + build
emod = ag.compile(model, "aie", {"layer1": "aie/src/conv2dstem.cc"}, calib=images)
emod = ag.compile("resnet18", "aie", "aie/src/conv3x3.cc")       # "auto": every layer it fits
```

Signature: `compile(model, target="cpu", offload=None, *, calib=None,
quantize="int8", workdir=None, build=True, strict=True, **opts)`.
`model` may be a name, a path, an `ag.Model` or a `torch.nn.Module`
(`example_inputs=` then required). Offload specs accept the string form
`"layer=path"` (several separated by `;`), a tuple, a dict, or a list of these.
With `strict=True` (default) any gate failure raises; `strict=False` keeps the
failing layers on the CPU and reports — the ORT "fallback provider" behaviour.

### 4.6 The "current model" convenience

`embedded_convert` and `compile` set `ag.current()`; module-level
`ag.offload_check`, `ag.offload`, `ag.detect`, `ag.build` act on it, which is
what makes the four-line script in §1 work. Methods on `EmbeddedModel` are the
explicit form and are what the docs use for anything longer than a quickstart
(pyplot vs. object-oriented matplotlib is the model). Calling a module-level
function with no current model raises with the hint `call ag.embedded_convert(...) first`.

### 4.7 Discoverability

* `ag.guide()` — prints the step the user is at and the next call, based on
  `ag.current()` (no model → `ag.load`; converted, nothing offloaded →
  `offload_check`; offloaded, not built → `build`).
* `ag.doctor()` — environment report: TVM 0.16 + LLVM, onnxruntime, torch/
  torchao/torch-mlir, host gcc, Vitis cross toolchain, `_aiebackend`, each
  with the exact fix (wraps `setup_tvm016.verify`, `setup_torchmlir.py
  --verify-only`, `arm_build.toolchain_status`).
* `python -m aiegraph {convert,check,offload,build,doctor,guide}` — the same
  calls from a shell, mirroring the function names one-to-one.

### 4.8 Regions, kernel contracts, partition (the §1 form)

Status: **implemented** (`src/aiegraph/region.py`, `contract.py`, `partition.py`;
methods on `EmbeddedModel`). The §4.3 layer-index calls stay; this is the form the
§1 example uses. Verified on ResNet-18: the §1 script runs end to end, and with
`region="conv1"` the stem is placed on the AIE and `main.elf` links its adapter.

| Call | Does |
|------|------|
| `ag.convert(model, target)` / `model.convert(target)` | `embedded_convert` under its short name; the object is the *module* |
| `module.regions` | every weighted op of the int8 model by name, with the TVM layer/call behind it |
| `module.register_kernel(name, source, op, target, constraints)` | lint the source, pick its `AIEK_SPEC` (by `name`, or the only one), refuse a contract the spec contradicts |
| `module.check_offload(region=, kernel=)` | contract per region, then lint/detect/calibrate/emulate (x86) for the TVM layers behind the regions that fit; writes only `aiek/` |
| `module.assign(region=, backend=)` | record `"cpu"` or `"aie:<name>"`; builds nothing |
| `module.partition(fallback=, optimize_for=)` | the whole placement: AIE rows reset to CPU, then `aiekernel.flow.offload` for the assigned layers; failures run on `fallback` (`None` raises) |
| `module.build(target=, output=)` | `main.elf` (`a78`) or `main_local.elf` (`x86`); `output=` copies the ELF(s), `partition.json`, `network.md` |

**What a region is.** One `Conv` / `Gemm` node of `<workdir>/*.int8.qdq.onnx` — the
model the flow actually compiled — named from the node: torch exports give
`layer1.0.conv1` (`/layer1/layer1.0/conv1/Conv`; a scope repeats only its parent's
attribute, so `/downsample/downsample.0/` is `downsample.0`), the zoo's gluon ResNet
gives `resnetv15_stage1_conv0` **plus** the torchvision alias `layer1.0.conv1`, so a
script runs on either source. The gluon aliases come from the `convK` index and the
1×1 kernel (`downsample.0`), not node order, because the ONNX graph puts the downsample
conv before `conv1`. References: an exact name or alias; a prefix (`"layer1"` = every
region under torchvision's stage 1 — *not* TVM kernel 1 as in §4.4); a glob
(`"layer2.*.conv1"`); `"call[08]"`.

**Region → TVM call, by weight fingerprint.** Values sorted, cast int64, hashed —
`byoc.aie_annotate.weight_fingerprint`. Layout and dtype drop out, so ONNX
`int8[64,3,7,7]` and TVM's `int8[16,1,7,7,3,4]` hash identically. All 21 regions of
ResNet-18 (20 convs + `fc`) map one-to-one, zoo and torch. Position is never used.

**The kernel is the unit that moves.** A TVM layer is a distinct kernel and the board
driver swaps it by symbol for every call. `layer1.0.conv1` and `layer1.1.conv1` are one
kernel (TVM layer 4), so offloading either offloads both. `check_offload` prints a
`move together` note, `partition` reports an unassigned sibling as *dragged*, and two
siblings assigned to different backends is a conflict that falls back.

**Two descriptions of one kernel.** The contract (`op`, `constraints`) is model-level:
`input_layout`/`weight_layout` describe the ONNX region (a `Conv` is NCHW/OIHW whatever
layout TVM later picks), `dtype` is the region's weight dtype, `quantization` is read from
the weight scale (`[Cout]` → `per_channel`). The `AIEK_SPEC` is call-level (exact TVM
tensors) and is enforced by detect/calibrate/emulate. `register_kernel` refuses only what
the spec itself contradicts: op family (`op` must appear in the spec's `op`/`match`) and
weight dtype. Kernel `target` is `"aie"` (where it runs) or `"a78"` (the ELF that
launches it); both mean the board build.

**`optimize_for`** accepts `"latency"` or `None`. There is no cost model yet, so the
placement is exactly the assignments that passed every gate, and the report says so.

---

## 5. How it maps onto what exists

```
ag.load / from_onnx / from_torch        tvmrelay.deploy_flow.fetch_model, prefold_onnx
                                        torch.onnx.export (v1) | pytorchmlir.quantize_pt2e_flow (M4)
ag.embedded_convert                     deploy_flow._stage_quantize, qnn_fold_zp, pe_target,
                                        build_c, verify_c, _stage_split, arm_build.build_arm_elf(run_make=False),
                                        cpu_reference, network_md
EmbeddedModel.layers / layer(...)       aiekernel.graph.Network (calls, manifest, params)
emod.detect / ag.lint                   aiekernel.match.detect_layers, aiekernel.lint.lint_source
emod.offload_check                      aiekernel.flow.check
emod.offload                            aiekernel.flow.offload
emod.build / run_local                  arm_build.build_arm_elf(run_make=True) / build_local_elf
```

The facade is therefore thin: new code is the object model, name resolution,
argument normalization, reports, and the PyTorch→ONNX bridge. The one real
refactor it needs is splitting `deploy_flow.run` (one 230-line function, seven
stages) into reusable stage functions the facade can call one at a time — the
CLI keeps calling them in order, so its behaviour does not change.

Process model, already solved and reused: `_aiebackend` and TVM each carry an
LLVM and must not share a process (skill **aiebackendtvmllvm**), so anything
touching the backend goes through `aiebackend.spawn()`; torch and TVM can share
the interpreter. aiehlc itself runs as a subprocess (`aiehlc.sh`).

Package layout (proposed): `src/aiegraph/`

| file | contents |
|---|---|
| `__init__.py` | the public surface: `load, from_onnx, from_torch, download_model, embedded_convert, compile, offload_check, offload, detect, lint, current, guide, doctor` |
| `model.py` | `Model` (fp32 graph handle) and the two importers |
| `embedded.py` | `EmbeddedModel`, `Layer`, `resolve()` |
| `plan.py` | offload-spec parsing (`"layer1=a.cc;layer4=b.cc"`, tuples, dicts) and plan-time rules (one library per ELF) |
| `report.py` | `CheckReport`, `OffloadReport`, `AiegraphError` + hints; text and HTML rendering |
| `stages.py` | adapters onto `deploy_flow` stage functions |
| `__main__.py` | the CLI |

Naming note: `aiegraph` is also the MLIR dialect. That is intentional — the
dialect is the IR this package's graph lowers to — and `emod.ir("aiegraph")`
returns its textual form. The Python package lives at `src/aiegraph/`, the
dialect stays under `src/mlir/mlirfront/frontend/aiegraph/`; no import clash.

---

## 6. Guided steps (what the tutorial and `ag.guide()` will show)

**From ONNX**

```python
import aiegraph as ag
m = ag.load("resnet18")                                   # 1 get a model
emod = ag.embedded_convert(m, target="a78", calib=["dog.jpg"])   # 2 int8 + per-layer C
print(emod)                                               # 3 look at the layers
emod.detect("./aie/conv2d_fuse.cc")                       # 4 which layers can my kernel take
emod.offload_check("layer1", "./aie/conv2d_fuse.cc")      # 5 prove it on x86
emod.offload(("layer1", "./aie/conv2d_fuse.cc"))          # 6 build + wire it
emod.build()                                              # 7 main.elf
```

**From PyTorch**

```python
import torch, torchvision, aiegraph as ag
net = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
m = ag.from_torch(net, example_inputs=(torch.randn(1, 3, 224, 224),))
emod = ag.embedded_convert(m, target="a78", calib=loader)  # same from here on
```

**Writing the kernel** (bridges to `doc/tutorial_userkernel.md`)

```python
ag.new_kernel("conv3x3", like="layer4", dir="./aie")   # writes conv3x3_ref.c with the spec filled
emod.offload_check("auto", "./aie/conv3x3_ref.c")      # calibrate the reference on every match
# ... write the AIE kernel, then:
emod.offload_check("layer4", "./aie/conv3x3.cc")
```

`new_kernel(..., like=layer)` fills `AIEK_SPEC` from the layer's real call
signature (it already exists as `aiekernel.graph.LayerCall`) and writes a
reference stub with the arguments unpacked and named — the most error-prone
part of starting a kernel (argument order, the two zero-points) is generated.

---

## 7. Milestones

| | scope | exit criterion |
|---|---|---|
| M0 | refactor `deploy_flow.run` into stage functions; CLI unchanged | `deploy_flow.py` output byte-identical on `layers-golden/` (`run_test.sh`) |
| M1 | `src/aiegraph`: `load`, `embedded_convert`, `EmbeddedModel` table, `detect`, `offload_check`, `offload`, `build`, `current`, reports | §6 "From ONNX" runs end to end on ResNet-18 with `conv2dstem_uk.cc`; same `partition.json` as `aiekernel offload` |
| M2 | `compile()` one-call form, offload-spec strings, layer selectors, `guide`, `doctor`, CLI | `ag.compile("resnet18", "aie", "layer1=...")` == M1 result |
| M3 | `from_torch` via ONNX export; calibration from a `DataLoader` | torchvision ResNet-18 v1 reaches the same per-layer C path; int8 top-5 order matches fp32 |
| M4 | native PyTorch lowering (PT2E → torch-mlir → aiegraph → C) behind `lowering="pt2e"` | per-layer C from the PT2E graph; offload_check bit-exact on the stem |
| M5 | `new_kernel` scaffolding; notebook (`_repr_html_`) polish; `aietriton` kernels accepted as offload sources | a Triton-style kernel goes through `offload_check` unchanged |

---

## 8. Open questions (decisions needed before M1)

1. **Package name.** `aiegraph` (as requested, shares the dialect's name) vs
   `aiehlc` (the product name). Recommendation: `aiegraph`, as above.
2. **Default model source.** ONNX model zoo (today's `deploy_flow` default,
   needs no torch) vs torchvision (needs torch). Recommendation: ONNX by
   default; `source="torchvision"` opt-in.
3. **`strict` default for `compile`.** Raise on any gate failure (proposed) vs
   fall back to CPU silently-with-report. Raising is safer for scripts; the
   notebook flow uses `offload_check` first anyway.
4. **Workspace layout.** One `workdir` per `EmbeddedModel` (proposed:
   `worklocal/<model>_<target>/`, same tree `deploy_flow` writes today) vs a
   global cache. Proposed keeps every existing tool (`make`, `network.md`,
   `run_test.sh`) working on the facade's output.
5. **Board execution.** `emod.run(board=...)` wrapping `apppaltest.py` is
   deliberately out of scope; confirm it stays a separate, explicit step.

## 9. Testing plan

* Unit: spec-string parsing, layer resolution (every row of §4.4), report
  rendering, error hints — no TVM needed (fake `Network`, as
  `aiekernel/test` does).
* Integration (skip if no deploy tree / no TVM): §6 ONNX steps on ResNet-18;
  `compile(...)` equals the step-by-step result (same `partition.json`, same
  `graph-sha`); PyTorch path reaches identical layer signatures.
* Golden: `deploy_flow.py` CLI after the M0 refactor is byte-identical on
  `layers-golden/`.
* Never: anything that runs on a board.
