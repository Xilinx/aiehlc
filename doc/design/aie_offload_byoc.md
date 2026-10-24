# `--aie-offload`: TVM BYOC → aiehlc AIE library (as implemented)

`deploy_flow.py --aie-offload --aie-layers N` puts layer *N* of the ResNet-18
int8 graph on the AIE and links it into the same `main.elf` the CPU path builds.
TVM's graph executor still schedules everything; for the offloaded layer it
calls a generated C wrapper, which calls the aiehlc-built AIE library.
`--byoc-aie` no longer exists — it was merged into this flag.

Supersedes the `run_aie_pipeline` emit that `--aie-offload` used to do (no
caller, no halo tiling, whole-feature-map kernel) and the raw-int32 boundary of
the old `--byoc-aie` (whose "AIE" entry was a scalar loop on the APU).
`doc/design/byoc_aie_plan.md` is the earlier plan.

## Flow

```
stage 4  relay.build (CPU)  → resnet18.c, resnet18_graph.json, resnet18_params.bin
stage 5  split              → layers/NN_<name>/            (what --aie-layers numbers)
stage 6a aie_offload.resolve_conv_targets   layer N → weight fingerprint
         byoc.aie_byoc.partition_for_aie     Relay, BEFORE relay.build:
           1 fuse      aie_fuse       InferType → FoldConstant → qnn.CanonicalizeOps
                                      → FoldConstant → MergeComposite → InferType
                                      → tag aie.weight_fp on every composite
           2 select    aie_annotate   fingerprint ∈ selection AND kernel_can_run
                                      → tag aie.layer_id, compiler_begin/end
                                      (everything else inlined back to the CPU)
           3 partition aie_byoc       MergeCompilerRegions → PartitionGraph
         aie_offload.check_targets           exactly 1 Relay conv per target, same geometry
         relay.build again
           4 codegen   aie_codegen    relay.ext.aie, dispatched on the composite
                                      name → wrapper C in resnet18.c
stage 5  split again
stage 6  arm_build: per-layer .o → liblayers.a; wrapper calls
         conv2d_stem_prepadded → links aout/libconv2dstem.a
```

**"Fuse" names two different things here, and they cannot be reordered.**
Step 1 is `MergeComposite`, which groups ops into the unit the AIE takes. TVM's
own `FuseOps` is TE-level operator fusion, is the last pass of the second
`relay.build` (`relay/backend/build_module.cc:350`), and *must* follow
partitioning so that it only fuses what stayed on the CPU. "BYOC runs before
fuse" describes the second one and is correct.

Note also that stage 6a runs *temporally* after the CPU `relay.build`, because it
needs `layers/` to resolve `--aie-layers` — but it partitions the **pre-build
module snapshot**, so the CPU build and the BYOC path start from the same
un-fused, un-legalized qnn graph. Partitioning has to precede `AlterOpLayout`
(`relay/backend/utils.cc:276`, inside `relay.build`) or the conv becomes the 5-D
`contrib_conv2d_NCHWc` that neither the pattern nor the kernel recognizes.

The library is built separately — aiehlc generates its device init, mesh
partition (`aieMesh` preamble → `__Runtime_init_mesh_partition`), the
`__aie_launch` → `host_canonicalized` caller, routing and the kernel ELF:

```bash
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file ./src/aietensorop/conv2dstem/conv2dstem.cc   # → aout/libconv2dstem.a
```

## Decisions, each one load-bearing

**Fusion is geometry-agnostic; capability belongs to selection.** The stem
geometry gate used to live in the `MergeComposite` checker, so the only layer
that ever fused was the one layer the AIE could already run — fusing and
offloading were a single decision. Three things were wrong with that: a layer
the kernel could not take never became a composite, so "malformed" and "not the
stem" were the same silence and `--aie-layers 7` could say nothing useful; the
inline-back path in `aie_annotate` was unreachable code; and there was nowhere
to stand to ask what a second kernel would need. `check_fused_qconv` now tests
shape only, and `kernel_can_run` — called during selection — returns
`(ok, reason)`, which is what lets `deploy_flow` distinguish *did not fuse*
(`info["unmatched"]`) from *fused but unrunnable* (`info["rejected"]`, with the
reason).

The cost is that **inline-back is now on the hot path**: ~12 of ResNet-18's
convs fuse and are then inlined back on every offload run, and because stage 6a
rebuilds the *whole* C from the partitioned module, a non-bit-identical
round-trip would silently change every CPU layer. `verify_aie_stem.py`'s
whole-network logit check is the gate for exactly this.

**A layer carries its identity in the IR.** Step 1 stamps `aie.weight_fp` (the
sorted-weight hash) on each composite; step 2 adds `aie.layer_id` to the ones it
selects. Previously the layer number was dropped at `partition_for_aie`'s
boundary — `deploy_flow` built a `{fingerprint: layer}` map and passed only its
keys — so a codegen error could name `tvmgen_default_aie_main_0` but never the
`--aie-layers` argument that caused it. `ConvSelectAnnotator` still recomputes
the fingerprint if the tag is missing, because trusting the attribute alone
would turn a lost tag into "selected nothing", which looks exactly like a
pattern miss.

**The boundary is the fused op.** The AIE kernel (`conv2d_spatial`) does conv +
bias + zero-point + per-channel requantize + ReLU + uint8 in one pass and can
produce nothing else, so the composite covers the canonicalized chain
`nn.conv2d → [− wzp·Σx] → +C2 → bias_add → [cast] → −C4 →
fixed_point_multiply_per_axis → clip(0,255) → cast(uint8)`
(`byoc/aie_patterns.py`). `extract_fused` is the one place that names each
constant; the checker and the codegen both use it.

**The epilogue folds into the kernel's existing `{bias, zp, mult, shift}`.**
`bias = C2 + C3 − C4 + 128·Σw[f]`, `zp = 0`, `mult = C5`, `shift = C7 (rshift)`.
TVM's `QMultiplyShift` is `(x·y + 2^(rs+30)) >> (rs+31)` in int64 — the core's
formula. Rejected by the checker: non-zero weight zero-point (C1 makes the
result depend on Σx), any left shift, a clip other than [0,255].

**uint8 input on an int8 kernel: shift, don't cast.** The stem input is
asymmetric uint8 (zero-point 113). The wrapper passes `x − 128` and the
`+128·Σw` above restores it exactly. A plain reinterpret corrupts every pixel
≥ 128 — measured: 613,203 / 802,816 outputs wrong on full-range input.

**The library takes TVM's padded tensor (`conv2d_stem_prepadded`).** TVM hoists
the padding into its own `nn.pad` and fills with the zero-point, not 0;
re-padding with zero (what `conv2d_stem()` does) is wrong on every border pixel.

**A layer is identified by its weights, not its position.** Relay's post-order
and the graph executor's order differ: TVM fuses each ResNet shortcut conv with
the residual add, so it runs after its main branch, while Relay visits it
first (convs 5–7, 10–12, 15–17 swap). `weight_fingerprint` hashes the sorted
weight values — independent of the CPU build's NCHWc-blocked int16 storage, and
unique for all 20 convs. `check_targets` fails closed: on any mismatch nothing
is offloaded.

**Weights and qparams are baked into the wrapper** as function-scope
`static const` arrays (Constants of the composite), so nothing publishes them
at run time and `split_layers` still gets a self-contained layer file. This is
a hard requirement now that stage 6 links the ELF from the *per-layer* objects:
file-scope data here would be copied into all 65 translation units and collide
at link. `qp` is a flat `int32[OC*4]` with `conv2dstem_qparam`'s layout.

**The wrapper's prelude is carried, as of the per-layer link.** The
`conv2d_stem_prepadded` declaration `aie_codegen` emits sits *after* TVM's own
kernels, past the first `extern "C"` guard, so it is not part of the prologue
`split_layers` hoists. It — and a synthesized prototype for the wrapper itself,
which TVM never pre-declares — now go into `layers_common.h`
(`split_module`'s `interstitial`). Before that, the AIE shim (layer 1) and the
wrapper (layer 64) each compiled with an **implicit declaration**: linkable on
aarch64 because the pointer arguments pass in registers either way, but
unprototyped and a hard error under `-Werror`.

**One link symbol, one definition.** `arm_build.AIE_ENTRY`
(`"conv2d_stem_prepadded"`) is what the link hook greps for and what the codegen
emits.

## Verification

```bash
PYTHONPATH=src python3 src/frontend/tvmrelay/byoc/verify_aie_stem.py
```

1. wrapper + `conv2dstem/ref/conv2dstem_x86.c` (the core's arithmetic) vs the
   composite run by TVM on LLVM: **0 / 802,816** mismatches on full-range uint8
   and on the dog image;
2. whole network on LLVM, partitioned vs unpartitioned: **0 / 1000** logits
   differ, top-5 identical (258 Samoyed).

Not covered here: the on-core DMA/window path
(`conv2dstem/ref/verify_pack.c`) and execution on hardware.

### The same op, on the board

`src/aietensorop/conv2dstem/data/` reaches the identical folding from the
**ONNX** side instead of the Relay side, and ships it to hardware as generated
headers (`conv2dstem_image.h`, `conv2dstem_weights.h`, `conv2dstem_golden.h`).
Because the two derivations are independent, they cross-check each other: the
`bias = bias_q + (128 − in_zp)·Σw`, `zp = 0`, `mult/shift = fixed_point(in_s·w_s
/ out_s)` here is `aie_codegen.fold_qparams`'s `C2 + C3 − C4 + 128·wsum` there.

Measured off-board: the generated headers through `ref/conv2dstem_x86.c` give
**0 / 802,816** against the CPU ground truth, and that ground truth agrees with
**onnxruntime** running the real quantized layer to **2 / 802,816** (max
difference 1 — ORT rounds half-to-even, the core half-up).

On the board `main.cc` self-checks against the golden array and prints one
verdict line, which `groundtruth.py --applog` reads back; a baremetal board
under xsdb has no channel but the console. See
`src/aietensorop/conv2dstem/data/README.md` and skill **conv2dstemfixture** —
particularly for why a generated header cannot live in a subdirectory, and why
the stored shift is the negation of `frexp`'s exponent.

## Scope today

Only the ResNet-18 stem geometry has an AIE kernel; any other selected conv is
reported ("no AIE kernel for this conv yet") and stays on the APU.
