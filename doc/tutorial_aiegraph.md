# aiegraph — model → int8 C on the A78, layers on the AIE, from Python

`aiegraph` turns an ONNX or PyTorch image classifier into int8 C for the
Cortex-A78 (one C file per layer, plus a driver and Makefile), and moves the
layers you choose onto your own AIE kernels after proving them on x86. It
never runs anything on a board; it hands you `main.elf`.

Design and rationale: [doc/design/aiegraph_python_api.md](design/aiegraph_python_api.md).
Writing an AIE kernel for it: [doc/tutorial_userkernel.md](tutorial_userkernel.md).

## 1. Install — one command

From the repo root, in a virtualenv:

```bash
pip install -e .                 # or  pip install -e ".[torch]"  for torchvision models
aiegraph doctor                  # what works on this machine, and the exact fix for what does not
```

That installs the `aiegraph` Python package, its Python dependencies, and two
commands: `aiegraph` (the CLI below) and `aiehlc` (the compiler driver —
`aiehlc --aie-version 5 --runtime-source-file k.cc` is `source script/aiehlc.sh …`
without the `source`).

It must be **editable** (`-e`): the flow uses the checkout itself — the
`aiehlc` binary and scripts, `include/`, the TVM build in `thirdparty/tvm-0.16`.

What pip cannot install, and how it is handled:

| piece | how |
|---|---|
| TVM 0.16 (no wheel exists) | **automatic**: aiegraph attaches the build in `thirdparty/tvm-0.16`. If there is none, `aiegraph setup` builds it once (10–25 min) — or the first conversion does. |
| Vitis (AIE compiler `xchesscc`, aarch64 cross gcc) | licensed, install it yourself and `source <Vitis>/settings64.sh`. Needed only for `offload` (the aiehlc build) and `build` (linking `main.elf`). Everything else works without it. |
| a board | never used by aiegraph |

`aiegraph doctor` on a working machine:

```text
aiegraph doctor
  ok   aiehlc checkout          /…/aiehlchj
  ok   numpy                    2.2.6
  ok   onnx                     1.22.0
  ok   onnxruntime              1.23.2
  ok   PIL                      12.1.1
  ok   torch                    2.10.0+cu128
  ok   TVM 0.16 (Relay)         tvm 0.16.0
  ok   torchvision (optional)   0.25.0+cu128
  ok   host gcc                 /usr/bin/gcc
  ok   aiehlc binary            /…/aiehlchj/build/aiehlc
  ok   Vitis (xchesscc)         XILINX_VITIS=/…/Vitis, xchesscc=/…/aietools/bin/xchesscc
  ok   aarch64 cross toolchain  /…/aarch64-none-elf-gcc
  ready for: convert, check, offload
```

## 2. From ONNX, step by step

```python
import aiegraph as ag

m = ag.load("resnet18")                       # zoo name, "model.onnx", or a torch.nn.Module
emod = ag.embedded_convert(m, target="a78")   # int8 + one C file per layer (a few minutes, cached)
print(emod)                                   # the layer table below

kernel = "src/aietensorop/conv2dstem/conv2dstem_uk.cc"
print(emod.detect(kernel))                    # which layers the kernel fits, and why not the others
print(emod.offload_check("layer1", kernel))   # lint, calibrate vs TVM, whole-network emulation (x86)
print(emod.offload(("layer1", kernel)))       # aiehlc build + swap the call      (needs Vitis)
print(emod.build())                           # link main.elf                       (needs Vitis)
```

The same as a script: `python example/aiegraph/quickstart_onnx.py`.

`print(emod)`:

```text
EmbeddedModel resnet18 -> a78   int8   workdir …/worklocal/aiegraph/resnet18
  layer    calls  op                            in -> out                                       on
  layer0   1      divide round add clip cast a  float32[1,3,224,224] -> int8[1,1,230,230,3]     cpu
  layer1   1      conv7x7/2 ->uint8             int8[1,1,230,230,3] -> uint8[1,16,112,112,4]    cpu
  layer2   1      max pool2d                    uint8[1,16,112,112,4] -> uint8[1,16,56,56,4]    cpu
  layer4   2      conv3x3/1 ->int8              int8[1,16,58,58,4] -> int8[1,16,56,56,4]        cpu
  layer6   1      conv3x3/1 +res ->int32        uint8[…] int8[1,16,58,58,4] -> int32[…]         cpu
  …
  33 layers, 41 calls
  next: print(module.regions)  /  module.register_kernel(name=..., source=..., op=...)
```

`print(emod.offload_check("layer1", kernel))`:

```text
offload_check  READY
  layer1   ok  conv2dstem_uk.cc: lint ok | detect 1 call(s) | calibrate BIT-EXACT (1 call(s))
  emulate: PASS -- logits BIT-IDENTICAL; top1 baseline (258, 12.017338) vs emulated (258, 12.017338); swapped L01 x1/1
  next: emod.offload(('layer1', 'conv2dstem_uk.cc'))
```

A failing check names the gate and the reason, e.g.
`layer4 NO conv2dstem_uk.cc: stopped at detect: call [04]: arg 0 (ifm_pad): dim 1 is 16, spec says 1`,
and `rep.raise_for_status()` turns it into an exception for scripts.

### 2a. The same, by region name

Name the model's own ops (`conv1`, `layer1.0.conv1`, `fc`), describe the kernel once,
and let `partition` decide what actually goes to the AIE:

```python
import aiegraph as ag

model = ag.load("resnet18")
module = ag.convert(model, target="a78")
print(module.regions)                         # every conv + fc, and the TVM layer/call behind it

module.register_kernel(
    name="conv2dstem",
    source="src/aietensorop/conv2dstem/conv2dstem_uk.cc",
    op="conv2d",
    target="a78",
    constraints={"input_layout": "NCHW", "weight_layout": "OIHW",
                 "dtype": "int8", "quantization": "per_channel"},
)
check = module.check_offload(region="conv1", kernel="conv2dstem")
print(check.summary())
module.assign(region="conv1", backend="aie:conv2dstem")
plan = module.partition(fallback="cpu", optimize_for="latency")
print(plan.report())
artifact = module.build(target="a78", output="resnet18_a78")
```

Script: `python example/aiegraph/quickstart_regions.py`. `print(check.summary())`:

```text
check_offload conv2dstem -> conv1 (resnetv15_conv0)   READY
  conv1 (resnetv15_conv0)  (onnx resnetv15_conv0_fwd; tvm layer1 call [01])
    contract: ok
    gates:    ok -- lint ok | detect 1 call(s) | calibrate BIT-EXACT (1 call(s))
  emulate: PASS -- logits BIT-IDENTICAL; top1 baseline (258, 12.017338) vs emulated (258, 12.017338); swapped L01 x1/1
  next: module.assign(region=..., backend='aie:conv2dstem')
```

`print(plan.report())` when a region does **not** fit (here `layer1.0.conv1`, a 3×3 conv):

```text
partition  fallback=cpu  optimize_for=latency  FELL BACK
  region                                   assigned         placed           tvm
  layer1.0.conv1 (resnetv15_stage1_conv0)  aie:conv2dstem   cpu              layer4 call [04]
                                             stopped at detect: call [04]: arg 0 (ifm_pad): dim 1 is 16, spec says 1 ([1, 16, 58, 58, 4] vs [1,1,230,230,3])
  on the AIE: nothing; every other layer runs on the A78
```

What to know:

* **Names.** The zoo model's gluon names (`resnetv15_stage1_conv0`) also answer to
  torchvision's (`layer1.0.conv1`); a torch export has the torchvision names
  directly. `region="layer1"` means every region under torchvision's stage 1. It is
  *not* `emod.offload_check("layer1", …)`, where `layer1` is TVM kernel 1 (the stem).
* **Regions that share a kernel move together.** `layer1.0.conv1` and
  `layer1.1.conv1` run the same TVM kernel, and the driver swaps kernels, not calls.
  `check_offload` notes it, and `partition` lists the unassigned one as dragged.
* **`partition()` is the whole placement.** It first puts every layer back on the A78,
  then offloads what you assigned. Each failure runs on `fallback`; with
  `fallback=None`, a failure raises instead.
* **The contract describes the model op** (an ONNX `Conv` is NCHW/OIHW). The
  `AIEK_SPEC` describes the TVM call and is enforced by the gates.
  `register_kernel` refuses a contract that its own spec contradicts (op family,
  weight dtype).
* `optimize_for="latency"` is recorded, but there is no cost model yet. The plan says so.

## 3. From PyTorch

```python
import torch, torchvision, aiegraph as ag

net = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
m = ag.from_torch(net, example_inputs=(torch.randn(1, 3, 224, 224),))
emod = ag.embedded_convert(m, target="a78")          # identical from here on
```

`from_torch` exports to ONNX (opset 13) and joins the ONNX path — the one that
reaches per-layer C today. Script: `python example/aiegraph/quickstart_torch.py`.

Different weights, different verdict: with torchvision's `IMAGENET1K_V1`
weights the stem kernel is **refused** for layer1 —
`stopped at detect: arg 7 (shift): require=range(-30,32), but values span [7, 50]`.
Seven channels quantize to a requant shift that makes the epilogue shift an
int64 by 67–81 bits (undefined behaviour in C, in TVM's code too). With
`strict=False` the layer stays on the A78 and `main.elf` still links. This is
why offload checks run on the *real* parameters of *your* model.

## 4. One call

```python
emod = ag.compile("resnet18", "cpu")                                   # convert + link main.elf
emod = ag.compile("resnet18", "x86")                                   # convert + build/run main_local.elf here
emod = ag.compile("resnet18", "aie", "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc")
emod = ag.compile(m, "aie", {"layer1": "a.cc"}, calib=["dog.jpg", "cat.jpg"], strict=False)
```

`strict=True` (default) raises at the first failed gate; `strict=False` keeps
failing layers on the A78 and leaves the report in `emod.offload_report`.

## 5. Reference

| call | does |
|---|---|
| `ag.load(x)` / `ag.from_onnx(path)` / `ag.from_torch(module)` / `ag.download_model(name)` | get an fp32 `Model` (input must be `[1,3,224,224]`) |
| `ag.embedded_convert(m, target="a78", calib=None, image=None, workdir=None)` | int8 + per-layer C + A78 driver; cached by model hash + options |
| `ag.open(workdir)` | adopt an existing `deploy_flow.py` tree |
| `emod.layers`, `emod.layer(ref)`, `emod.resolve(ref)` | layers; `ref` = `"layer1"`, `1`, `"layer4-9"`, `"conv3x3*"`, `"call[08]"`, a list |
| `emod.detect(kernel)` | which layers fit |
| `emod.offload_check(spec)` | x86 gates; never writes outside `<workdir>/aiek/` |
| `emod.offload(spec, build=True, make=False, force=False)` | + aiehlc build + link gate + driver swap |
| `emod.run_local()` | build and run `main_local.elf` on x86 |
| `ag.convert(m, target)` / `m.convert(target)` | `embedded_convert` by its short name |
| `module.regions`, `module.regions.find(ref)` | model ops by name; `ref` = `"conv1"`, `"layer1.0.conv1"`, `"layer1"` (all under it), `"layer2.*.conv1"`, `"call[08]"` |
| `module.register_kernel(name, source, op, target="aie", constraints=None)` | a named kernel contract (`op`: `conv2d`/`dense`; constraints: `input_layout`, `weight_layout`, `dtype`, `quantization`) |
| `module.check_offload(region=, kernel=)` | contract + x86 gates per region; `.summary()`, `.ok`, `.raise_for_status()` |
| `module.assign(region=, backend="aie:<name>" or "cpu")` | record a placement |
| `module.partition(fallback="cpu", optimize_for="latency")` | apply all assignments; `.report()`, `.ok`, `.on_aie` |
| `module.build(target="a78", output=None, local=False)` | `main.elf` (`a78`) / `main_local.elf` (`x86`); `output=` copies ELF(s) + `partition.json` + `network.md` |
| `ag.compile(model, target, offload)` | all of the above in one call |
| `ag.offload_check / offload / detect / build / run_local` | the same on `ag.current()` (the last converted/opened model) |
| `ag.lint(kernel)`, `ag.doctor()`, `ag.setup()`, `ag.guide()` | kernel checks, environment, TVM setup, "what next" |

Offload specs: `"layer1=k.cc"`, `"layer1=a.cc; layer4-9=b.cc"`, `"k.cc"` (every
layer it fits), `("layer1", "k.cc")`, `{"layer1": "k.cc"}`, or a list of these.

CLI — the same functions:

```bash
aiegraph convert resnet18
aiegraph layers
aiegraph detect src/aietensorop/conv2dstem/conv2dstem_uk.cc
aiegraph check  "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc"
aiegraph offload "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc"
aiegraph build
aiegraph compile resnet18 aie "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc"
```

(`--workdir` defaults to `worklocal/aiegraph/resnet18`.)

## 6. Limits today

* Image classifiers with a `float32[1,3,224,224]` input (ImageNet preprocessing).
  Other inputs are refused at `load` with that reason.
* One aiehlc library per ELF (the runtime has one `routing()`): offloading layers
  to two different kernel *sources* is rejected at the link gate. Several layers
  of one kernel are fine.
* PyTorch goes through ONNX export; the native PT2E → MLIR path is a later milestone.
* Board runs are yours: copy `main.elf` over (`script/test/apppaltest.py`).
