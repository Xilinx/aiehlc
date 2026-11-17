# aiegraph — model → int8 C on the A78, layers on the AIE, from Python

`aiegraph` turns an ONNX or PyTorch image classifier into int8 C for the
Cortex-A78 (one C file per layer, plus a driver and Makefile), and moves the
layers you choose onto your own AIE kernels after proving them on x86. It
hands you `main.elf`, and `emod.run()` runs it on a lab board (palmyra by
default, or `emod.run(ag.VEK385, "hostname")`) -- nothing else touches a board.

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
| a board | only for `emod.run()`: `script/test/apppaltest.py` (palmyra: `USERNAME`/`PALIP`/`BOARDNAME`) or `appvek385.py` (VEK385: `USERNAME`/`VEK385IP`), from the env or `script/test/envlocal.sh`; `pexpect` |

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

## 4b. Before writing a kernel: a backbone for the layer

```python
print(ag.offload_prototype())                   # the file format and rules (no model needed)
proto = emod.offload_prototype("layer4")        # TVM layer 4 (or region="layer1.0.conv1")
print(proto)                                    # args[] in call order, roles, value facts, AIEK_SPEC
proto.write("my_kernels/")                      # aiek_layer04.cc + _ref.c + test_aiek_layer04.cc
# fill in the TODO(...)s, then
print(emod.offload_check("layer4", "my_kernels/aiek_layer04.cc"))
```

`aiek_layer04.cc` is a **backbone** with the architecture of
`src/aietensorop/conv2dstem/conv2dstem_uk.cc` (+ `conv2dstem.cc`), specialized
to the layer -- what is the layer's is done, what is the algorithm's is yours:

| part | done | TODO (yours) |
|---|---|---|
| `aiek_layer04_kernel` (the aiehlc kernel) | ports with the staged shapes (`aie::shape<58,58,64>`, ...), mesh split by `aie::program_id`, the block loop with `load_buffer` / `store_buffer` / `release` / `commit` on a tile the kernel analysis accepts (so aiehlc derives the DMA) | `TODO(compute)`: the MACs and the epilogue |
| `aiek_layer04_run(...)` (`conv2d_stem_nchwc`'s role) | every TVM argument in call order, typed; staging buffers; mesh partition + launch | `TODO(stage input / weights / parameters)`, `TODO(unpack output)` |
| `AIEK_ENTRY(aiek_layer04)` | all arguments -> `aiek_layer04_run` | -- |
| `AIEK_SPEC(aiek_layer04, ...)` | proven against every call | -- |
| `aiek_layer04_ref.c` | arguments, TVM layouts; for the conv + requantize layers of these int8 networks the **whole reference** (bit-exact) | the op, for any other layer |
| `test_aiek_layer04.cc` | a standalone board test: `main()` + `#include "aiek_layer04.cc"`, all tensors bound in TVM call order, the entry called `AIEKT_ITERS` (5) times, each **timed**, one `[aiek_layer04] TEST PASS/FAIL` line | -- |

So `proto.write("./")` then `emod.offload_check("./aiek_layer04.cc")` is READY
as written (calibrate BIT-EXACT, whole-network emulation identical): the
reference is the layer's math, i.e. the spec of what your AIE code must
compute. Until the `.cc` TODOs are filled `_run` returns -1 and the entry
`AIEK_ERR_KERNEL`, so `offload` builds but the board run reports the error.
The backbone builds as is (`aiehlc.sh` -> `libaiek_layer04.a`; verified for
ResNet-18 layers 1 and 4) and lints clean. For each argument the report shows
`in` / `param` / `out`, dtype, shape, the role the layer's own TVM C gives it
(weight, bias, zero-point, requant multiplier / shift, unused) and facts over
every call (`all zero` -> `require=zero`). Kernel frames exist for TVM NCHWc
conv2d layers; any other op gets the entry + spec skeleton.

**The test.** `test_aiek_layer04.cc` is an aiehlc *entry file* (as
`test_conv2d.cc` is for the stem): aiehlc splices the `#include`d kernel in, and
`main()` makes it an ELF instead of `libaiek_layer04.a`:

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file $PWD/my_kernels/test_aiek_layer04.cc
python3 script/test/apppaltest.py aout/worklocal/build/host
```

The path must be **absolute** (a relative one resolves to `/test_...cc`), and
`aout/` lands in the current directory. If the deploy was built with `make TRACE=1`
(`arm_build/layeriohex/` exists), `write` also emits `aiek_layer04_test_data.h`
and copies that layer's first call's dumps `lNN_*_in.h` / `_out.h` beside it, so
the test runs **TVM's real tensors** and byte-compares every output against
**TVM's own result** (`TEST PASS golden`). A `static_assert` per tensor catches a
stale dump at compile time. Without dumps it is a **smoke** test on seeded
synthetic tensors (ranges follow each argument's role and `require=`): the entry
returns `AIEK_OK`, every output gets written (outputs are poisoned before each
call), and two launches give identical bytes. Until the `.cc` TODOs are
filled the test FAILs with `entry returned -4`.

Every call is **timed** (wall clock around the entry, i.e. what TVM's graph
executor waits; `XTime`, as `test_conv2d.cc`, or `CLOCK_MONOTONIC` without the
BSP): the first call is reported apart (partition init, staging allocation, ELF
load), then min / mean / max of the repeats -- which is per-call latency
*including* array configuration, since aiehlc re-programs the mesh on every
launch -- GOPS for a conv frame, and under `aiehlc.sh --profiling` the PMU split
`kload / bdcfg / coreen / startio / wait_io` per call. Machine-readable:
`[aiek_layer04] PERF SUMMARY iters 5  first ..  min ..  mean ..  max ..  (ms)`.
`-DAIEKT_ITERS=N` changes the count. Both forms build as generated
(verified: layer 1 golden, layer 4 smoke). `write(dir, test=False)` /
`--no-test` skip it.

CLI: `aiegraph prototype [layer1] [--region R] [--entry NAME] [--write DIR] [--no-test]`.

## 4c. Let Claude fill the backbone: `ai_offload_prototype`

```python
res = emod.ai_offload_prototype("layer4")        # backbone -> claude -p fills the TODOs
print(res)                                       # CORRECT / NOT CORRECT + the gates
emod.offload("layer4", res)                      # path-like: the filled aiek_layer04.cc
print(emod.prototype_check("layer4", "my_kernels/"))   # the same verdict on any directory
```

It runs `offload_prototype(layer).write(...)` into
`<workdir>/aiproto/layerNN_prototype/` (an existing directory is moved aside to
`.prev-<time>`, never deleted), then gives `claude -p` the layer, its TVM C, the
backbone and the current verdict. Claude may edit only that directory and run one
command, `aiegraph protocheck`. After each round **our own** `prototype_check`
decides; a failure goes back to the same session, at most `rounds=3` times:

| gate | passes when |
|---|---|
| `todo` | no `TODO(` / `TODO:` marker left in `<entry>.cc` / `<entry>_ref.c` |
| `kemu` | the kernel **and** host code, run on x86, give every output byte of TVM's own output (`make TRACE=1` dumps; without them `<entry>_ref` on synthetic tensors) |
| `trace` | `aiehlc-convtile trace`: every `load_buffer` / `store_buffer` matches the derived DMA |
| `check` | `offload_check` READY (lint, spec, reference bit-exact, whole network) |
| `build` | last, once the others pass (`build=False` / `--no-build` skips): aiehlc builds `test_<entry>.cc` into a board ELF |

**`kemu` is the gate that judges the kernel.** `offload_check` alone cannot:
calibrate and emulate run the `_ref.c` reference, so an *unfilled* backbone
with a complete reference is READY there. `kemu` (`src/aiegraph/kemu.py`)
compiles your `<entry>.cc` with g++ under ASan. Each `aie::port` is a view of
the host buffer the launch passed. `aie::load_buffer` gathers the requested
block into an exactly-sized allocation, so ASan flags any read past it, and
`aie::commit` scatters the block back. `kernel<<<mesh>>>(...)` runs the kernel
once per core with `aie::program_id` set, with vector code going through
`aie_api_x86emu.h`. A block held past the kernel's end, or outside the tensor,
is an error. It takes ~2 s for the 7×7 stem, and catches wrong rounding, a
wrong output offset, a missing commit and an out-of-block read. It models
*which data* each core sees, not DMA timing, which is `trace`'s job.

CLI: `aiegraph aiprototype layer4 [--rounds N] [--no-build] [--out DIR] [--model M]`,
`aiegraph protocheck layer4 DIR [--build]` (exit 0 = correct).

## 4a. When a check says NOT READY: let Claude port the kernel

```python
kernel = "src/aietensorop/conv2dstem/conv2dstem_uk.cc"     # a 7x7/2 stem kernel
print(emod.offload_check("layer4", kernel))                # NOT READY: stopped at detect ...
new_kernel = emod.offload_AI_proto("layer4", kernel)       # claude -p ports it
print(emod.offload(("layer4", new_kernel)))                # path-like: use it like any source
```

`offload_AI_proto` runs `offload_check` first and returns the kernel unchanged if
it is already READY. Otherwise it copies the kernel's directory (flat) to
`<workdir>/aiproto/layer4_<kernel>/`, renames the entry to `<kernel>_l4.cc`, and
runs `claude -p` there with the layer's TVM C, the refusal and
`doc/tutorial_userkernel.md`. Claude may read anything but edit only that
directory, and the one command it may run is `aiegraph check`. After each round
**aiegraph runs `offload_check` itself** -- Claude's word is not the verdict -- and
sends any remaining failure back to the same session, for up to `rounds=3`.

| argument | default | |
|---|---|---|
| `rounds` | 3 | Claude turns, each followed by our own `offload_check` |
| `model` | Claude Code's default | `--model` for `claude` |
| `claude` | `"claude"` | the CLI (must be logged in: `claude -p hi`) |
| `timeout_s` | 1800 | per round |
| `out_dir` | `<workdir>/aiproto/<layer>_<kernel>` | wiped and re-staged on every call |
| `quiet` | False | hide the live `[claude] Edit: ...` progress lines |

The result has `.ok`, `.source`, `.report` (the last `CheckReport`), `.rounds`,
`.log` (prompt + raw event stream) and `.session` (`claude --resume <id>` to carry
on by hand); `raise_for_status()` raises when it is not READY. The original kernel
and the model are never modified. READY means bit-exact on x86 against TVM, as for
any kernel; it does not prove that aiehlc compiles it for AIE2PS -- `offload()`
(needs Vitis) is that gate.

## 5. Reference

| call | does |
|---|---|
| `ag.load(x)` / `ag.from_onnx(path)` / `ag.from_torch(module)` / `ag.download_model(name)` | get an fp32 `Model` (input must be `[1,3,224,224]`) |
| `ag.embedded_convert(m, target="a78", calib=None, image=None, workdir=None, force=False, keep_offloads=False)` | int8 + per-layer C + A78 driver; the conversion is cached by model hash + options, but it starts FRESH: earlier offloads go back to the CPU unless `keep_offloads=True`; `force=True` redoes the conversion |
| `ag.open(workdir)` | adopt an existing `deploy_flow.py` tree as it is (earlier offloads included) |
| `emod.offload_reset(layers=None)` / `ag.offload_reset(...)` | put offloaded layers (all, or `"layer4"` etc.) back on the CPU; driver regenerated |
| `emod.offload("layer1", False)` | the same for one spec: undo `offload("layer1", k.cc)`, layer back on the CPU (returns the layers reset, `[]` if it was not on the AIE) |
| `emod.layers`, `emod.layer(ref)`, `emod.resolve(ref)` | layers; `ref` = `"layer1"`, `1`, `"layer4-9"`, `"conv3x3*"`, `"call[08]"`, a list |
| `print(emod.sources())` / `emod.source_path("layer1")` | the code each layer runs, as **absolute** paths: `layers/NN_*/NN_*.c` (TVM C) on the CPU, the user kernel source on the AIE (`sources(full=False)`: as partition.json records them -- workdir-relative when the file is inside the workdir, e.g. an `aiopt/` copy, else repo-relative); `print(emod)` also shows `aie:<kernel>  <- <source>` |
| `ag.offload_prototype()` | the kernel-file format and rules (no model needed) |
| `emod.offload_prototype(layer, region=None, entry=None)` | a kernel-file backbone for that layer: kernel frame + host run function + entry + proven `AIEK_SPEC`, reference frame; `.write(dir)` (section 4b) |
| `emod.detect(kernel)` | which layers fit |
| `emod.offload_check(spec)` | x86 gates; never writes outside `<workdir>/aiek/` |
| `emod.offload(spec, build=True, make=False, force=False)` | + aiehlc build + link gate + driver swap |
| `emod.ai_offload_prototype(layer, rounds=3, build=True, model=None)` | `offload_prototype` backbone → `claude -p` fills the TODOs until `prototype_check` passes (section 4c) |
| `emod.prototype_check(layer, dir, build=False)` | gates `todo` / `kemu` (kernel + host on x86 vs TVM) / `trace` / `check` [/ `build`] (section 4c) |
| `emod.offload_AI_proto(layers, kernel, rounds=3, model=None)` | NOT READY → `claude -p` ports a copy until `offload_check` is READY (section 4a) |
| `emod.run_local()` | build and run `main_local.elf` on x86 |
| `emod.run(board="palmyra", host=None, *, elf=None, build=True, profile=False, netload=None, reboot=None, timeout=1800)` | run `main.elf` on a lab board, (re)building it if missing or older than the driver / `partition.json` / AIE archives; returns a `RunReport` (`.ok`, `.top1`, `.top5`, `.inference_ms`, AIE `ENTER`/`EXIT` counts, `.log`). `emod.run(ag.VEK385, "crimini2")` for a VEK385; `host` overrides `$PALIP` / `$VEK385IP`. Palmyra defaults to `--netload` (a 12–16 MB ELF fails JTAG `dow`); VEK385 powers off afterwards (its `-nonreboot` holds the board until killed). Refuses while another run of the same board script of yours is active. CLI: `aiegraph run [palmyra\|vek385] [HOST]` |
| `emod.profile(ag.ALL, *, by="call", top=None)` | each layer's time on the board from the last `run()`: one row per call (`by="layer"`: per layer, its calls summed), layer name, op, `cpu`/`aie`, mean ms per `graph_run()`, % of the total; `emod.profile("layer1")` / `top=5` filter. Needs `emod.run(profile=True)` (a `make PROFILE=1` ELF, two timer reads per call; the default `profile=False` builds a plain one, switching rebuilds); otherwise it prints `profiling not enabled` and returns `None`. Reads `last_run`, else the newest `arm_build/run_*.log`. CLI: `aiegraph profile [LAYER ...] [--by-layer] [--top N]` |
| `report = emod.AI_perf_analysis("layer1", ai=True)` | why an offloaded layer is slow (skill **kernelperfcheck**): deterministic facts (measured ms of the last `run(profile=True)`, EXIT rc, pragmas, generated host.cc call counts, kernel ELF text/data/BSS, kernel.lst vector ops, aiehlc.log), a cost model (372 ns per config-bus word, 2.9 ns over control packets), the skill's rules, then `claude -p` read-only returns bottlenecks + ranked suggestions with predicted savings. `ai=False`: no AI. Saved to `<workdir>/aiperf/<layer>/report.{md,json}` |
| `result = emod.AI_kernel_accelerate("layer1", report, rounds=2, board=None)` | `claude -p` applies the suggestions to a COPY of the kernel the layer runs NOW (`<workdir>/aiopt/layer<N>_<kernel>/`, `..._r2` when that kernel is itself an aiopt copy; an earlier copy is moved to `<dir>.prev-<time>`, never deleted; the original is never edited); every round is gated by our own `offload_check` + an aiehlc build of the copy (failures go back via `--resume`); the result compares facts/model of the old and NEW build: `result.saved_ms`, `result.predicted_ms`. `board=` also offloads the copy and measures it (`measured_after`). Path-like: `emod.offload(("layer1", result))` |
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
* Several kernel *sources* in one ELF need libraries built by the current aiehlc
  (per-library routing; `print(emod)` keeps both layers on `aie:`). A library
  from an older aiehlc still owns the global `routing()` and is displaced by the
  next offload. Switching between libraries re-initializes the partition per
  launch.
* PyTorch goes through ONNX export; the native PT2E → MLIR path is a later milestone.
* Board runs are yours: copy `main.elf` over (`script/test/apppaltest.py`).
