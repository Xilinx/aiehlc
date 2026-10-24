# Bring your own AIE kernel: offload one layer, or a bunch of them

This tutorial shows how to plug **your own** AIE kernel into the TVM → aiehlc
flow and have it replace one layer — or every layer it fits — of a deployed
network. It covers the C contract you implement, the Python tooling that
checks it, and the two worked examples that ship with the repo.

Everything here runs on the build host. **No step needs a board**; only the
final `build` step needs Vitis (it compiles and links, it never runs).

```
your kernel.cc ──lint──► detect ──► calibrate ──► emulate ──► build ──► link ──► partition.json
 (aiehlc source)   L0xx    which     ref == TVM    whole net   aiehlc     one lib     graph_driver.c calls
 + reference.c             layers    per layer     on x86      archive    per ELF     aie_<symbol>()
                           fit       bit-exact     bit-exact   (Vitis)
```

| You write | The tooling generates / checks |
|---|---|
| the AIE kernel + mesh launch (as in `conv2dstem.cc`) | lint: the aiehlc silent-failure traps |
| `AIEK_ENTRY(name)` — TVM's args in, your launch called | the TVM packed-call adapter `aie_<tvm symbol>()` |
| `AIEK_SPEC(name, "...")` — signature + value requirements | which layers match, and why the others don't |
| `AIEK_REFERENCE(name)` — the op in plain x86 C | bit-exact diff vs TVM's own kernel, per call; whole-network logits |
| — | aiehlc build, archive, symbol isolation, `partition.json`, driver swap |

Code: `include/aie_user_kernel.h` (C) and `src/frontend/tvmrelay/aiekernel/` (Python).
Examples: `src/aietensorop/conv2dstem/conv2dstem_uk.cc` (+ `_ref.c`) and
`example/userkernel/conv3x3_requant_ref.c`.

---

## 1. Before you start

You need a deployed network — the output of `deploy_flow.py` (graph JSON,
params, and the stage-5 per-layer split):

```bash
PYTHONPATH=src python3 src/frontend/tvmrelay/deploy_flow.py      # -> worklocal/tvmrelay_deploy
export PYTHONPATH=src                                            # for everything below
```

Every command below defaults to `--out-dir worklocal/tvmrelay_deploy`. `check`
writes only under `<out-dir>/aiek/`; `offload` also writes `layers/` and
`arm_build/`, so point it at a copy if you want to keep a pristine tree.

Know your layer before you write anything. `network.md` lists every call with
its arguments; `detect --all` (§5) prints the same from the tool's point of view.
Three numbering spaces matter (CLAUDE.md, "three numbering spaces"):

| name | meaning | where you see it |
|---|---|---|
| **layer** | stage-5 folder index `NN_` = one TVM kernel *symbol* | `--layers`, `@1,4`, `partition.json` |
| **call** | execution-order index of a kernel invocation | `[NN]` in `TRACE=1` dumps, `layeriohex/lNN_*` |
| node | graph-JSON node id | `graph_node` in `partition.json` |

A layer can be called more than once (ResNet-18 calls layer 04's kernel twice,
calls `[04]` and `[08]`, with different weights). The driver swaps by symbol,
so **offloading a layer offloads every call of it**, and every call is checked.

---

## 2. The C contract — `include/aie_user_kernel.h`

A user kernel is **one aiehlc source in library mode** (no `main()`): your
`__global__` kernel and its `kernel<<<mesh>>>(...)` launch, exactly as in
`src/aietensorop/conv2dstem/conv2dstem.cc`, plus three things from the header.

### 2.1 The entry

```c
#include "aie_user_kernel.h"

AIEK_ENTRY(my_kernel) {          // extern "C" int my_kernel(const aiek_tensor *args, int num_args)
    const int8_t *x = AIEK_PTR(&args[0], const int8_t);
    ...
    return run_on_aie(...) == 0 ? AIEK_OK : AIEK_ERR_KERNEL;
}
```

`args` are the layer's tensors **in TVM's call order**: every input as the
graph passes it — activations and constants (weights, bias, quant params)
interleaved — then every output. Each is an `aiek_tensor`:

```c
typedef struct {
    void *data;            // graph executor's buffer, 64-byte aligned, not retained
    int32_t ndim;
    uint8_t code, bits;    // AIEK_INT / AIEK_UINT / AIEK_FLOAT, width
    uint16_t lanes;
    const int64_t *shape;
    uint64_t nbytes;
} aiek_tensor;
```

Return `AIEK_OK`, or `AIEK_ERR_PARAM` (a parameter value you cannot express —
refuse rather than compute wrong pixels), `AIEK_ERR_ALLOC`, `AIEK_ERR_KERNEL`.
You do **not** check dtypes or shapes: the generated adapter has already done
that against the layer's concrete shapes before your entry is called.

### 2.2 The spec

```c
AIEK_SPEC(my_kernel,
    "kernel    my_kernel\n"                     // display name (default: entry)
    "op        conv2d_requant_relu_u8\n"        // free-form label
    "match     contrib_conv2d_NCHWc\n"          // substring(s) of the TVM symbol
    "reference my_kernel_ref.c\n"               // x86 reference, relative to this file
    "in    int8  [1,1,230,230,3]   ifm_pad\n"
    "param int8  [16,1,7,7,3,4]    weight\n"
    "param int32 [1,16,1,1,4]      zero_point_b  require=zero\n"
    "param int32 [16,4]            shift         require=range(-30,32)\n"
    "out   uint8 [1,16,112,112,4]  ofm\n");
```

One directive per line (`#` starts a comment):

| line | meaning |
|---|---|
| `in <dtype> [dims] [name]` | an activation (produced by another kernel or the network input) |
| `param <dtype> [dims] [name] [require=...]` | a graph constant; its **values** are checked |
| `out <dtype> [dims] [name] [tol=N]` | an output; `tol=N` allows `|ref − tvm| ≤ N` |
| `match <s>...` | every substring must occur in the TVM kernel symbol |
| `reference <file>` | the x86 reference (`AIEK_REFERENCE`) |
| `export <sym>...` | keep these globals visible in the archive (default: only the entry + adapters) |
| `kernel` / `op` | names for reports |

Dims are an integer, `*` (anything), or a **NAME** bound on first use and
required to agree everywhere after — `[1,CIC,HI,WI,4]` … `[COC,CIC,3,3,4,4]` …
`[1,COC,HO,WO,4]` says "the weight's input-channel blocks equal the
activation's". dtypes: `int8..int64`, `uint8..uint64`, `float16/32/64`.

`require=` (repeatable) is checked on the constant's **real values** from
`params.bin`, per call: `zero`, `nonzero`, `uniform`, `nonneg`,
`range(lo,hi)`. This is what makes a "same shape, different numbers" layer
fail detection instead of failing on the board — the stem's epilogue has one
zero-point field while TVM subtracts two, so the second must be all-zero
(`require=zero`).

`AIEK_SPEC` expands to a `static const char[]` marked `used`, so the spec also
travels inside the archive and the ELF (`strings libaiek_<name>.a`).

### 2.3 The reference

A plain-C file the host gcc compiles:

```c
#include "aie_user_kernel.h"
AIEK_REFERENCE(my_kernel) {      // defines my_kernel_ref(), same args, same results
    ...
    return AIEK_OK;
}
```

Write it to model **what your AIE path computes** (its folds, its single
zero-point, its rounding, its layouts), not a textbook convolution. It is then
diffed against TVM — they agree only if your model of the AIE path is right.
For a reference-only source (§6) the spec can live in the reference file
itself and the `reference` line can be omitted.

### 2.4 Rules the source must follow

These all break the aiehlc frontend **silently** (rc=0, wrong or empty output);
`lint` checks each one:

* never write the kernel keyword or the `<<<` launch token in a comment or a
  string (L011/L012) — the source rewriter matches text;
* nothing that reaches the launch may be named above the kernel definition —
  declare/define the entry and launch functions after it (L014);
* the launch and its `device.partition(...)` live in the same function (L015);
* helpers the kernel body calls must be `#define` macros written in the `.cc`
  itself (L017/L018);
* headers sit next to the entry file (flat copy, L019); spliced `.cc`
  includes resolve against the entry file's directory (L020);
* no `main()` (L016) — the layer links a `lib*.a`.

---

## 3. The Python side

```python
from frontend.tvmrelay import aiekernel

aiekernel.lint_source("kernel.cc")                       # LintReport (findings, specs)
aiekernel.detect(out_dir, "kernel.cc")                   # [LayerVerdict] for every layer
aiekernel.check(out_dir, ["kernel.cc@1,4"])              # lint+detect+calibrate+emulate
aiekernel.offload(out_dir, ["kernel.cc@1"], make=True)   # + build + partition.json + main.elf
```

```bash
python3 -m frontend.tvmrelay.aiekernel lint    SRC...
python3 -m frontend.tvmrelay.aiekernel detect  SRC [--all]
python3 -m frontend.tvmrelay.aiekernel check   SRC[@L,..]... [--layers L] [--no-emulate]
python3 -m frontend.tvmrelay.aiekernel offload SRC[@L,..]... [--no-build] [--make] [--force]
python3 src/frontend/tvmrelay/deploy_flow.py --aie-kernel SRC[@L,..]   # inside the full flow
```

`SRC@1,4` selects layers 1 and 4; without `@` (and without `--layers`) every
layer the spec matches is selected — the "bunch of layers" case. Exit status
is 0 when every kernel has at least one layer through every gate that ran. The
full report lands in `<out-dir>/aiek/report.json`.

| module | job |
|---|---|
| `spec.py` | read the source the way aiehlc does (`.cc` include splice), parse `AIEK_SPEC` |
| `lint.py` | the L0xx rules |
| `graph.py` | the deployed graph as typed kernel calls — TVM-free (graph JSON + `params.bin`) |
| `match.py` | detect: signature + `require=` values |
| `calibrate.py` | reference vs TVM's kernel, per call |
| `emulate.py` | whole network on x86 through the generated adapter |
| `adapter.py` | the `aie_<symbol>` TVM packed-call glue |
| `build.py` | aiehlc + `hostcompile.sh` + symbol isolation |
| `flow.py` | the gates in order; `partition.json`; `arm_build/` regeneration |

---

## 4. Walkthrough A — the stem, one layer, real AIE kernel

`src/aietensorop/conv2dstem/conv2dstem_uk.cc` adds no AIE code: it splices the
HW-verified `conv2dstem.cc` (`#include "conv2dstem.cc"`, as `test_conv2d.cc`
does) and adds the contract around `conv2d_stem_nchwc()`:

```c
/* abridged -- see the file */
#include "conv2dstem.cc"
#include "aie_user_kernel.h"

AIEK_ENTRY(conv2dstem_uk) {
    const int8_t *ifm = AIEK_PTR(&args[0], const int8_t);   // [1,1,230,230,3] zp-padded
    const int8_t *wts = AIEK_PTR(&args[1], const int8_t);   // [16,1,7,7,3,4] OIHW3i4o
    const int32_t *zp_a = ..., *bias = ..., *zp_b = ..., *mult = ..., *unused = ..., *shift = ...;
    uint8_t *ofm = AIEK_PTR(&args[8], uint8_t);              // [1,16,112,112,4] NCHW4c
    conv2dstem_qparam qp[CONV2DSTEM_NUM_FILTERS];
    for (int oc = 0; oc < CONV2DSTEM_NUM_FILTERS; oc++) {
        if (zp_b[oc] != 0 || unused[oc] != 0) return AIEK_ERR_PARAM;   // one zp field
        qp[oc] = {bias[oc], zp_a[oc], mult[oc], shift[oc]};
    }
    return conv2d_stem_nchwc(ifm, wts, qp, ofm) == CONV2DSTEM_OK ? AIEK_OK : AIEK_ERR_KERNEL;
}
AIEK_SPEC(conv2dstem_uk, ... "param int32 [1,16,1,1,4] zero_point_b require=zero\n" ...);
```

Compare `frontend/tvmrelay/aie_stem_lib.py:_ENTRY_TMPL`, which hard-codes the
same nine arguments in Python. Here the fold is your C and the glue is
generated. The reference, `conv2dstem_uk_ref.c`, models the same path on x86:
refuse a live second zero-point, int32 accumulation in TVM's packed layouts,
`acc += bias; acc -= zero_point`, `(acc·mult + 2^(s+30)) >> (s+31)`, clamp.

### 4.1 lint

```
$ python3 -m frontend.tvmrelay.aiekernel lint src/aietensorop/conv2dstem/conv2dstem_uk.cc
src/aietensorop/conv2dstem/conv2dstem_uk.cc: OK (AIE kernel conv2d_spatial)
  lint: clean
  conv2dstem (entry conv2dstem_uk, op conv2d_requant_relu_u8)
    arg 0: in int8[1,1,230,230,3] ifm_pad
    arg 1: param int8[16,1,7,7,3,4] weight
    ...
    arg 8: out uint8[1,16,112,112,4] ofm
```

What a failure looks like — the kernel keyword left in a `// TODO` comment:

```
ERROR   L011 my.cc:2: the kernel keyword appears inside a comment or string; aiehlc's
        source rewriter matches text, so it rewrites the wrong code and the build
        still succeeds (skill: aiesourcetextrewrite). Reword it.
```

### 4.2 detect

```
$ python3 -m frontend.tvmrelay.aiekernel detect src/aietensorop/conv2dstem/conv2dstem_uk.cc
  layer 01 MATCH calls [01]: matches (1 call(s))
  layer 04 ----- calls [04], [08]: call [04]: arg 0 (ifm_pad): dim 1 is 16, spec says 1 ([1, 16, 58, 58, 4] vs [1,1,230,230,3])
  layer 06 ----- calls [06]: call [06]: 11 args (10 in + 1 out), spec has 9 (8 + 1)
  ...
1 layer(s) match: 01
```

Every non-match says which argument, which dim, and what the layer actually
has — that is the list of things your kernel would have to grow to cover it.

### 4.3 check = calibrate + emulate

```
$ python3 -m frontend.tvmrelay.aiekernel check src/aietensorop/conv2dstem/conv2dstem_uk.cc
[aiek] conv2dstem_uk.cc: lint -- 0 error(s), 0 warning(s)
[aiek] conv2dstem: detect -- 1 of 33 layers fit the spec: 01
        layer 01: selected -- matches (1 call(s))
  [calib] layer 01 call [01]: BIT-EXACT on 4 trial(s) x 802,816 outputs (rand0, rand1, min, max)
  [emul] emulate: PASS -- logits BIT-IDENTICAL; top1 baseline (258, 12.017338) vs emulated (258, 12.017338); swapped L01 x1/1
VERDICT conv2dstem: READY -- layer(s) 01 passed lint, detect, calibrate, emulate
```

**Calibrate** links one x86 program from the layer's stage-5 TVM C
(`layers/01_*/01_*.c` — the code `main_local.elf` runs), your reference, and a
fixed harness; feeds both the call's **real** constants plus adversarial
activations (two seeded full-range draws, all-min, all-max); and compares
every output byte. Outputs are poisoned with different bytes before each side
runs, so an element your reference forgets to write cannot pass. A one-LSB
rounding slip (`2^(s+29)` instead of `2^(s+30)`) is reported as:

```
  [calib] layer 01 call [01]: MISMATCH in trial 'rand0': 118,722/802,816 outputs differ, max |d|=1.0, first at [0, 0, 0, 0, 2]
```

(coordinates are in the output's own layout — NCHW4c here).

**Emulate** runs the whole network twice on x86 — TVM's kernels everywhere,
then with every selected call routed to `aie_<symbol>()` → your reference
**through the same generated adapter the board build uses** — on the real
image, and byte-compares the 1000 logits. It also counts the adapter's
`[aie-offload] ... ENTER` lines against the graph's call count, so a swap that
never fired cannot pass. If several layers are swapped and the logits move, it
re-runs each alone and rejects the one responsible.

### 4.4 offload

```
$ python3 -m frontend.tvmrelay.aiekernel offload src/aietensorop/conv2dstem/conv2dstem_uk.cc@1 --make
  ... lint / detect / calibrate / emulate as above ...
  [build] isolate libaiek_conv2dstem.a: 2 entry symbol(s) + 2 runtime hook(s) (_Z7routingP12XAie_DevInst, g_runtime_debug_level) stay global, 13 made local
[aiek] link gate: 1 AIE archive(s) called by the driver, 0 collision(s)
[aiek] arm_build: main.elf linked: .../arm_build/main.elf
VERDICT conv2dstem: READY -- layer(s) 01 passed lint, detect, calibrate, emulate, build, link
OFFLOAD conv2dstem: libaiek_conv2dstem.a -> graph_driver.c calls 1 adapter(s) under -DGRAPH_AIE_OFFLOAD
```

(≈ 90 s for the aiehlc run, cached afterwards by a hash of the source and the
headers beside it; then `hostcompile.sh`.) What changed on disk:

* `layers/01_*/aie/` — aiehlc's `worklocal/` tree; `host.cc` ends with the
  generated adapter block (`/* ---- aiek adapters ... */`);
  `build/libaiek_conv2dstem.a` is the isolated archive;
* `layers/partition.json`, row of layer 01:
  `target: aie, backend: aiehlc-user, user_kernel: conv2dstem, source:
  src/aietensorop/conv2dstem/conv2dstem_uk.cc, entry: aie_tvmgen_..._, archive: layers/01_*/aie/build/libaiek_conv2dstem.a`;
* `arm_build/graph_driver.c` — the call is wrapped in `#ifdef GRAPH_AIE_OFFLOAD`
  (`arm_build.aie_entries`, the same mechanism as the stem), and the Makefile
  sets `AIE_OFFLOAD_DEFS := -DGRAPH_AIE_OFFLOAD`;
* `main.elf` (`--make` only) — `nm` shows `T aie_tvmgen_..._` and
  `T conv2dstem_uk`, while `conv2d_stem_nchwc` and
  `_binary_kernel_conv2d_spatial_start` are now local.

`make local` / `main_local.elf` are unchanged: the local build never defines
`GRAPH_AIE_OFFLOAD`, so it stays the CPU reference for the same driver.

The same thing inside the full flow:

```bash
python3 src/frontend/tvmrelay/deploy_flow.py --aie-kernel src/aietensorop/conv2dstem/conv2dstem_uk.cc@1
```

It runs after `--aie-offload`/`--aiegraph` (so its verdict is the last word in
`partition.json`) and before stage 6, which then generates the swapped driver.

### 4.5 When you do take it to a board

Not part of this tutorial's steps, but this is what to look for: each
offloaded call prints

```
[aie-offload] layer 01 nn_contrib_conv2d_NCHWc_subtract_add_sub: ENTER conv2dstem (conv2dstem_uk), 9 args
[aie-offload] layer 01: EXIT rc=0, out[0..7]= 2e 61 0e 00 1e 59 2b 1c, hash=2aeeab65
```

and the emulated run already printed the reference's line for the same image
(`<out>/aiek/emul/emulated.log`). The `hash` is the same `h = h*31 + byte`
over the output bytes on both, so **the board's EXIT hash must equal the
emulated one** — that is the bit-exactness check for the AIE kernel itself.

---

## 5. Walkthrough B — a bunch of layers from one spec

`example/userkernel/conv3x3_requant_ref.c` is **reference-only** (no AIE code
yet): a shape-generic 3×3 int8 conv + requant + ReLU − 128, with symbolic
dims. This is step 1 of writing a new kernel — find out what it would cover
and pin down the arithmetic *before* any AIE work.

```
$ python3 -m frontend.tvmrelay.aiekernel check example/userkernel/conv3x3_requant_ref.c
[aiek] conv3x3_requant: detect -- 7 of 33 layers fit the spec: 04, 09, 14, 16, 21, 23, 28
        layer 04: selected -- matches (2 call(s))
        ...
  [calib] layer 04 call [04]: BIT-EXACT on 4 trial(s) x 200,704 outputs (rand0, rand1, min, max)
  [calib] layer 04 call [08]: BIT-EXACT on 4 trial(s) x 200,704 outputs (rand0, rand1, min, max)
  [calib] layer 09 call [12]: BIT-EXACT on 4 trial(s) x 100,352 outputs (rand0, rand1, min, max)
  ...
  [calib] layer 28 call [35]: BIT-EXACT on 4 trial(s) x 25,088 outputs (rand0, rand1, min, max)
  [emul] emulate: PASS -- logits BIT-IDENTICAL; top1 baseline (258, 12.017338) vs emulated (258, 12.017338); swapped L04 x2/2, L09 x1/1, L14 x1/1, L16 x1/1, L21 x1/1, L23 x1/1, L28 x1/1
VERDICT conv3x3_requant: READY -- layer(s) 04, 09, 14, 16, 21, 23, 28 passed lint, detect, calibrate, emulate (reference-only: write the AIE kernel before offload)
```

One spec, seven layers, eight calls — both strides, four channel widths — and
the network's logits do not move with all eight replaced. The non-matches are
informative too: the residual convs (`11 args (10 in + 1 out), spec has 9`)
take the skip tensor as an extra input, and layer 25 produces `int32`, not
`int8`.

Pick a subset with `@`: `check example/userkernel/conv3x3_requant_ref.c@4,9`.
Mix kernels in one run: `check conv2dstem_uk.cc conv3x3_requant_ref.c@4,9`
emulates the stem and the 3×3 layers together.

**Step 2 — the AIE kernel.** Write an aiehlc source in the
`conv2dstem_uk.cc` pattern: the kernel, its GemmSpace descriptors, a host
function with the partition + launch, and `AIEK_ENTRY(conv3x3_requant)`. Move
the spec there with `reference conv3x3_requant_ref.c` pointing back. Two
constraints shape it:

* aiehlc bakes **one geometry** per build, so the spec of a buildable kernel
  should be concrete (lint L027 warns otherwise). Layer 04's two calls share a
  geometry and are covered by one build; layers 09/14/… are not the same
  geometry.
* **One aiehlc library per ELF.** The runtime calls a single global
  `routing()` (`__Runtime_routing_init`) and caches partition init per
  `meshId`, so two independently built libraries cannot both be right in one
  program. `offload` keeps the first kernel and rejects the rest at the `link`
  gate with that reason; it also displaces an `--aie-offload` stem library
  when a user kernel builds. To cover several geometries in one ELF, put the
  kernels in one source (one library) or offload them in separate builds.

---

## 6. What each gate proves — and what it does not

| gate | proves | does not prove |
|---|---|---|
| lint | the source avoids every known aiehlc silent-failure trap | that it compiles (only `build` does) |
| detect | arity, kinds, dtypes, dims and `require=` values fit, for every call | anything about the numbers your kernel computes |
| calibrate | your reference == TVM's kernel bit-for-bit on real constants and adversarial inputs | that the **AIE kernel** == your reference |
| emulate | the adapter, the driver swap and your reference on the real activations of a real image | the AIE path (it never runs here) |
| build | aiehlc + chess + host compile succeed; adapters are in the archive | runtime behaviour |
| link | `main.elf` links; one library per ELF; no colliding symbols | anything on the array |

The gap between "reference" and "AIE kernel" is closed the way the stem
closed it: make the kernel match the reference on the board
(`conv2d_stem_verify` pattern), and compare the adapter's `EXIT hash` with the
emulated run's. Calibration's job is to make sure that the thing you are
matching is right in the first place — layouts, the two zero-point terms,
rounding, int32 accumulation are where every bring-up in this repo went wrong.

---

## 7. Lint rules

| code | level | what | skill |
|---|---|---|---|
| L001 | error | no `AIEK_SPEC` | — |
| L002 | error | malformed spec (message names the spec line) | — |
| L003 | error | the spec's entry is not defined in this source | — |
| L004 | error | `aie_user_kernel.h` not included | — |
| L005 | error | entry not a C identifier, or starts with `aie_` | — |
| L010 | info | no AIE kernel — reference-only | — |
| L011 | error | kernel keyword in a comment/string | aiesourcetextrewrite |
| L012 | error | `<<<` in a comment/string | aiesourcetextrewrite |
| L014 | error | something reaching the launch is named above the kernel (empty module) | aiesourcetextrewrite |
| L015 | error | launch without `device.partition(...)` in the same function | — |
| L016 | error | `main()` defined (would link an ELF, not a library) | hostlibrarymode |
| L017 | error | kernel body calls a function (helpers must be macros) | byocaieoffload |
| L018 | warning | kernel body uses a macro not `#define`d in this source | byocaieoffload |
| L019 | error | quoted header from a subdirectory | conv2dstemfixture |
| L020 | error | spliced `.cc` include does not resolve | aiehlcincludekernel |
| L021 | error | control-plane pragma without `control_plan_op_control_packet` | — |
| L022 | warning | `control_plan_shim_bd_ctrl` without `control_plan_dedicated_shim` | — |
| L023 | error | `control_plan_shim_bd_ctrl` without `__Runtime_ctrl_high_throughput_enable(1)` | — |
| L025 | warning | int32 read through a cast of a byte array (A78 unaligned fault) | conv2dstemtvmlayer |
| L027 | warning | symbolic dims in a spec that also has an AIE kernel | — |
| L028 | warn/err | no `reference` (detect only) / reference file missing | — |
| L029 | error | reference file lacks `AIEK_REFERENCE(entry)` | — |
| L031 | warning | int16 accumulator in the kernel body | byocaieoffload |

---

## 8. Troubleshooting

| symptom | meaning / fix |
|---|---|
| `detect -- 0 of N layers fit` | read the per-layer reason; most often a dim or an extra input (`10 in + 1 out`) |
| `rejected at detect -- ... require=zero, but 3/64 values are non-zero` | this layer's constants break an assumption of your kernel; generalize the kernel or leave the layer on the APU |
| `MISMATCH in trial 'max'` only | saturation/clamp or accumulator overflow; check int32 accumulation and the clamp order |
| `MISMATCH ... first at [..]` on every trial | layout: compare your index arithmetic with the TVM layer C (`layers/NN_*/NN_*.c`) |
| `rc tvm=0 ref=-2` | your reference refused a parameter (`AIEK_ERR_PARAM`) that `require=` did not exclude — add the `require=` |
| `emulate: FAIL -- ... 0 ENTER / 1 calls` | the driver swap did not fire: the TVM symbol changed since `arm_build/` was generated (re-run `deploy_flow.py`) |
| `rejected at build -- no toolchain` | source Vitis (`source script/setup.sh --path-set-only`) or use `--no-build` to record the decision only |
| `rejected at link -- ... already owns this ELF's aiehlc library` | one library per ELF (§5); merge the kernels into one source |
| `rejected at build -- ... after isolation conv2d_x not defined` | the adapter/entry did not compile into `host.o` — read `layers/NN_*/aie/hostcompile.log` |
| `compile ... failed` in calibrate | your reference does not compile with `gcc -std=gnu11`; the message has the compiler's lines |

---

## 9. File map

| file | role |
|---|---|
| `include/aie_user_kernel.h` | C contract: `aiek_tensor`, `AIEK_ENTRY/REFERENCE/SPEC`, adapter helpers |
| `src/frontend/tvmrelay/aiekernel/` | Python side (table in §3); `python -m frontend.tvmrelay.aiekernel` |
| `src/frontend/tvmrelay/aiekernel/test/test_aiekernel.py` | tests (`python3 -m pytest -q src/frontend/tvmrelay/aiekernel/test`) |
| `src/aietensorop/conv2dstem/conv2dstem_uk.cc`, `conv2dstem_uk_ref.c` | example A: the stem through the interface |
| `example/userkernel/conv3x3_requant_ref.c` | example B: reference-only, seven layers |
| `src/frontend/tvmrelay/deploy_flow.py --aie-kernel` | the same, inside the full flow |
| `<out>/aiek/` | scratch: `report.json`, `calib/`, `emul/` (logs, logits), `_host/` objects, aiehlc cache |
