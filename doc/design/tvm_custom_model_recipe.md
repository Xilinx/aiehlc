# Recipe: custom model → TVM Relay IR → aiehlc AIE code

## 1. Overview & when to use this

AIEHLC has **three** frontends into `TilingLinalgPipeline`:

| Frontend | Input | Reaches the pipeline via |
|----------|-------|--------------------------|
| **aiehlc** (C++) | C++ using `aie::SpatialPolicy` + Clang AST | `AieFrontEnd.cc` builds routing IR in-process |
| **aietriton** (Python) | `@aie_triton.jit` GEMM kernel | AST → tensor specs + C body → `_aiebackend.run_aie_pipeline` |
| **tvm** (Python) | a model via ONNX → Relay | fused-graph walk → per-launch tensor specs + C body → `_aiebackend.run_aie_pipeline` |

This document is the **"bring your own model" how-to** for the TVM frontend
(`src/frontend/tvm/`). It is a companion to the frontend deep-dive
in [`tvm_frontend.md`](tvm_frontend.md): that doc explains *how the existing
scaled ResNet-18 frontend works*; this doc explains *how to take a customized
model of your own through the same path* and where the current path stops you.

**Scope / outcome.** The recipe targets a **structural offload verified against
the numpy CPU oracle**, not real ImageNet accuracy. Weights are the frontend's
deterministic placeholder scheme (conv weights alternate ±1, `bn_scale=64`≈0.5 Q7,
biases 0; fc weights 1) so the AIE result and the numpy reference stay bit-exact.
There is **no real-weight quantization** in this path. See §13 (Limitations).

The full ImageNet ResNet-18 that `example/model/resnet18py/classify2.py` runs
(`resnet18.py`: 224×224×3, up to 512 channels, 1000 classes, 7×7 stride-2 stem,
maxpool, pre-activation v2, real ONNX weights) is used throughout as the
"unmodified big model" contrast — §3(b) and §12 spell out exactly which of its
properties must be scaled/adapted before the existing int8 path accepts it.

## 2. The pipeline in one diagram

```
model.export_onnx()          torch.nn model ─► ONNX            model.py:266
  └─ relay_import.import_relay ONNX ─► Relay IR                relay_import.py:46
        (InferType → SimplifyInference[BN fold] → FoldConstant → FuseOps → InferType)
        └─ walk.build_plan     fused graph ─► LayerOp plan     walk.py:103
              (recover_signatures + validate_against_canonical; returns model.layer_plan())
              └─ _compiler.compile_plan  each LayerOp ─► run_aie_pipeline   _compiler.py:230
                    (tensor_specs + kernels.kernel_body_for → host.cc/kernel.cc/routing.cc)
              └─ _compiler.cpu_reference  bit-exact numpy oracle             _compiler.py:115
```

The public entry point is `run_resnet(...)` (`__init__.py:52`), which wires all of
the above together and returns a `RunResult` (`plan`, `logits`,
`predicted_class`, `used_tvm`, `emitted`).

## 3. Two starting points

### (a) Structural / scaled — recommended, covered end-to-end here

Define your network in the `model.py` style (small torch.nn module + a matching
canonical `layer_plan()`), so the exported ONNX drives a **validated** plan and
every dimension fits the int8 constraints of §13. Steps 1–7 below walk this path.

### (b) Bring an existing ONNX (e.g. `resnet18py`)

You *can* point `import_relay()` at any ONNX file — the Relay import and the
`recover_signatures` walk will run. But `build_plan` **returns
`model.layer_plan()`**, not a plan synthesised from your graph (see §6), and the
concrete blockers below stop the big ImageNet model:

- **int8 config range** — every conv dim `{H,W,Cin,Cout,K,stride}` and fc dim
  `{sp_h,sp_w,channels,nclass}` is packed as `int8` and read as `uint8`
  (`model.pack_config` at `model.py:167`; kernels read `(uint8_t)` at
  `kernels.py:58`). 512 channels and 1000 classes do **not** fit ≤ 255.
- **missing ops** — `resnet18py` needs an input-BN, a 7×7 stride-2 conv stem, and
  a 3×3 maxpool. Only `conv_bn_relu / conv_bn / residual_add_relu / avgpool_fc`
  exist (`kernels.py:181`). Also note `resnet18py` is **pre-activation v2**
  (BN→ReLU→Conv, `resnet18.py:57`) while the frontend model is **post-activation**
  (`model.py:240`); the op-fusion pattern the walk keys on differs.
- **real weights** — this path has no PTQ; only the placeholder buffers of
  `make_conv_params` / `make_fc_params`.

Generalizing the frontend to accept (b) directly is future scope (§13).

## 4. Step 1 — Express the model

Edit two functions in `model.py` and keep them in **structural lock-step**
(the walk validates one against the other, §6):

1. `build_torch_model()` (`model.py:217`) — the torch.nn graph that
   `export_onnx()` serialises. Structure, not weights, is what matters.
2. `layer_plan()` (`model.py:105`) — the canonical `LayerOp` launch sequence,
   including buffer wiring (`input / feat1 / feat2 / feat3 / tmp1 / tmp2 /
   skip_ds / logits`) and residual element counts.

`LayerOp` (`model.py:51`) carries the op name, output/input buffer names, and the
per-op dims; `LayerOp.signature()` (`model.py:85`) is the structural fingerprint
used for validation.

Op-mapping table (Relay pattern after `SimplifyInference` → `LayerOp`):

| Relay pattern | `LayerOp.op` | dims used |
|---------------|--------------|-----------|
| `nn.conv2d` + folded BN `multiply`/`add` + `nn.relu` | `conv_bn_relu` | `H,W,Cin,Cout,K,stride` |
| `nn.conv2d` + folded BN `multiply`/`add`             | `conv_bn`      | `H,W,Cin,Cout,K,stride` |
| `add` + `nn.relu` (residual join)                    | `residual_add_relu` | `length` |
| `nn.global_avg_pool2d` + `nn.dense` (+ bias)         | `avgpool_fc`   | `spatial_h,spatial_w,channels,num_classes` |

The scaled default is 8×8×1 input, channels 4→8→16→32, 4 classes
(`model.py:35-38`). To customize, change those constants and mirror the change in
both `build_torch_model()` and `layer_plan()`. Keep every dimension ≤ 255 (§3a)
and each launch's windows small enough to fit AIE tile memory (§13).

## 5. Step 2 — ONNX → Relay IR

`export_onnx(path)` (`model.py:266`) writes the torch model as ONNX (opset 13,
input name `"input"`). `import_relay(onnx_path, input_name, input_shape)`
(`relay_import.py:46`) then runs:

```
relay.frontend.from_onnx
  → InferType
  → SimplifyInference    # folds BatchNorm into scale/shift == multiply + add
  → FoldConstant
  → FuseOps              # wraps ops in inner functions
  → InferType
```

`SimplifyInference` is why a conv "block" appears in the fused graph as
`conv2d → multiply → add [→ relu]`: BN in inference is an affine per-channel
scale+shift, and folding it to `multiply`+`add` is what lets the walk key on a
stable pattern. TVM and onnx are **optional**: `tvm_available()` /
`onnx_available()` (`relay_import.py:28,38`) gate the path; when absent the
frontend falls back to the canonical plan.

Inspect the IR while developing:

```python
from frontend.tvm.model import export_onnx, INPUT_C, INPUT_H, INPUT_W
from frontend.tvm.relay_import import import_relay
mod, params = import_relay(export_onnx("/tmp/claude/m.onnx"),
                           input_shape=(1, INPUT_C, INPUT_H, INPUT_W))
print(mod)   # dump the fused Relay IRModule
```

## 6. Step 3 — Graph walk → LayerOp plan

`build_plan(onnx_path, strict=False)` (`walk.py:103`):

1. `canonical = model.layer_plan()`.
2. If `onnx_path` is `None` or TVM is unavailable → return `canonical`
   (graceful degradation).
3. Else `recover_signatures(onnx_path)` (`walk.py:59`) walks the fused module with
   a `relay.ExprVisitor` in dataflow (post-)order, recording conv geometry,
   `relu`, `add`, `gap`, `dense` signatures. Conv geometry is read from
   `checked_type` shapes: `(Cout,Cin,K,K)` from the weight (OIHW,
   `_conv_signature` at `walk.py:43`), `(H,W)` from the input (NCHW, `_feature_hw`
   at `walk.py:53`), stride from `attrs.strides[0]`.
4. `validate_against_canonical(recovered, canonical)` (`walk.py:134`) compares the
   ordered conv `(H,W,Cin,Cout,K,stride)` tuples, checks `add` count ≥ residual
   count, and checks GAP+dense presence when the plan has an `avgpool_fc`.
5. On match → return `canonical` (buffer wiring intact). On mismatch → return
   `canonical`, unless `strict=True`, which **raises** and prints the recovered
   convs — use `strict=True` while developing a custom model to catch divergence
   between your torch graph and your `layer_plan()`.

> **Key property:** the walk **validates**, it does not synthesise. A custom model
> is expressed by editing `model.py`; the graph is the cross-check, not the source
> of the buffer wiring. This is why (b) in §3 does not "just work".

## 7. Step 4 — Param buffers & tensor specs

Placeholder param buffers (`model.py`):

- `make_conv_params(H,W,Cin,Cout,K,stride)` (`model.py:171`):
  `[config:6={H,W,Cin,Cout,K,stride}][weights:Cin·Cout·K·K][bn_scale:Cout][bn_bias:Cout]`.
  Weights alternate 1/−1; `bn_scale=64`; `bn_bias=0`.
- `make_fc_params(sp_h,sp_w,channels,nclass)` (`model.py:187`):
  `[config:4][weights:channels·nclass][bias:nclass]`. Weights 1, bias 0.
- `fc_params_no_header(...)` (`model.py:200`) is the headerless variant the CPU
  reference uses (intentional divergence, see `tvm_frontend.md`).

The **int8 config header is the hard limit**: `pack_config` stores `int8`
(`model.py:167`) and every kernel reads config as `(uint8_t)` (`kernels.py:58`,
`:147`), so all packed values must be ≤ 255.

Per-launch `tensor_specs` are built in `_tensor_specs(op)` (`_compiler.py:157`) as
flat int8 `([shape], bits, isInput)` tuples per window:

| Op | windows: `([shape], 8, isInput)` |
|----|----------------------------------|
| `conv_bn_relu` / `conv_bn` | `([Cin·H·W], in)`, `([param_sz], in)`, `([Cout·outH·outW], out)` |
| `residual_add_relu` | `([n], in)`, `([n], in)`, `([n], out)` |
| `avgpool_fc` | `([channels·sp_h·sp_w], in)`, `([param_sz], in)`, `([num_classes], out)` |

## 8. Step 5 — Kernel bodies

All four bodies (`kernels.py`) use the **two-input / one-output window ABI**
(`window_in_0`, `window_in_1`, `window_out_0`) and read their config from the
param-buffer header, so **one body serves every launch of that type**:

- `conv_bn_relu_body` / `conv_bn_body` (`kernels.py:47`)
- `residual_add_relu_body` (`kernels.py:114`) — length from the pipeline macro
  `BUF_SZ_OUT_0 * 4` (no param buffer)
- `avgpool_fc_body` (`kernels.py:138`) — fixed `int8_t pooled[256]`, so
  channels ≤ 256

Two int16 rules are **load-bearing for bit-exactness** with the CPU oracle:

- the conv accumulator is `int16_t` and wraps at each `+=` (`kernels.py:84`);
- Q7 BN is `(int16_t)(acc * bn_scale) >> 7` — the *product* is truncated to int16
  **before** the shift (`kernels.py:90`), matching numpy's
  `(np.int16(s)*np.int16(scale))>>7` (`_compiler.py:75`).

**Adding a new op** (e.g. maxpool) requires five coordinated edits:

1. a new body generator + a `KERNEL_BODIES` entry (`kernels.py:181`);
2. a `walk.py` mapping so the pattern is recovered/validated;
3. a `_compiler._tensor_specs` arm (`_compiler.py:157`);
4. a `_compiler.cpu_reference` arm (`_compiler.py:129`) — the bit-exact oracle;
5. a `model.LayerOp` field set + `signature()` case if new dims are needed.

## 9. Step 6 — Emit AIE code

`compile_launch(op, idx, out_root, mesh)` (`_compiler.py:211`) makes
`out_root/<idx:02d>_<op>/` and calls:

```python
run_aie_pipeline(mesh_rows, mesh_cols, tensor_specs, out_dir, body, func_name
                 [, split_specs, dma_specs])
```

`_aiebackend` is the **shared** pybind extension in
`src/mlir/mlirfront/frontend/aiebackend/` (in-process via `aiebackend.load()`,
or out of process via `aiebackend.spawn()` when TVM is loaded); build it (cmake with
`LLVM_INSTALL_DIR` + MLIR) before emitting. Each call writes one file set into its
directory: `host.cc`, `kernel.cc`, `<kernel>.cc`, `routing.cc`, `aieml.bcf`,
`aieml.prx`. `compile_plan` (`_compiler.py:230`) iterates the whole plan (~29
launches for the default scaled ResNet).

**Optional im2col conv path.** The default conv is a *direct* convolution whose
loop nest lives in the kernel body, so `dma_specs` is left empty.
`im2col_dma_spec(H,W,Cin,K,stride)` (`_compiler.py:174`) builds the multi-dim shim
DMA addressing the extended pybind `dma_specs` argument accepts. That argument is
the Python surface of the generic `DmaAddressing` on `TensorParam`
(`aiebackend_pybind.cpp:21`), a per-tensor tuple:

```cpp
using DmaSpec = std::tuple<std::vector<std::pair<int,int>>,  // dims (stride,size)
                           int,                              // iter_step
                           int,                              // iter_wrap
                           std::vector<int64_t>,             // ddrShape
                           int>;                             // mode
```

Non-empty entries populate `TensorParam::shimDma`; empty dims + mode 0 = flat
(`aiebackend_pybind.cpp:38-49`). Defaulted to `{}` so the Triton path is
unaffected. See [`conv2d_im2col_design.md §11`](conv2d_im2col_design.md).

## 10. Step 7 — Verify

1. **CPU oracle** — `cpu_reference(plan, input_data)` (`_compiler.py:115`) is a
   pure-numpy interpreter of the plan; it is the bit-exact reference the AIE
   result is checked against. Needs only numpy.
2. **End-to-end (no build)** — `run_resnet(emit_aie=False)` (`__init__.py:52`)
   returns the plan, logits, and predicted class.
3. **Unit tests** — `python src/frontend/tvm/test_tvm_frontend.py`.
   Parts needing TVM or the built extension are **skipped**, not failed.
4. **AIE HW/sim run** (per `CLAUDE.md`): generate with `emit_aie=True`, then
   drive the per-launch `host` through the standard flow —
   `source script/aiehlc.sh --aie-version 5 --runtime-source-file <cc>` then
   `python3 script/test/apppaltest.py aout/worklocal/build/host`, verifying with
   `script/test/verify_host.sh` (pass: `device_teardown done`, fail: `AIE ERROR`).

## 11. Running the real `classify.py` model through TVM (framework-level)

"Convert `classify.py` to TVM" has **two** distinct meanings — don't conflate them:

1. **Framework port (this section).** Run the *real* ImageNet ResNet-18 through
   TVM's own Relay runtime (`relay.build` + `graph_executor`) instead of torch.
   Real weights, real accuracy, ~15 lines of change, **no aiehlc involvement** —
   the AIE pipeline is never touched.
2. **AIE offload (the next section, §12).** Scale the network down and remap it so
   the existing int8 AIE frontend accepts it. Placeholder weights, structural
   result only. That is the scale-down / generalize-frontend problem of §12 and
   the Limitations of §13.

This section covers meaning (1).

### What `classify.py` does today

`example/model/resnet18py/classify.py` is a pure PyTorch classifier:

- downloads `resnet18-v2-7.onnx` (`classify.py:126-128`);
- builds a from-scratch torch model and **loads the real ONNX weights into it**
  (`classify.py:131` → `resnet18(onnx_path=...)` at `resnet18.py:173-177`);
- preprocesses the image to a float32 NCHW `[1,3,224,224]` tensor
  (`preprocess()` at `classify.py:94-112`);
- runs `model(x)` → `softmax` → `topk` and prints labels
  (`classify.py:133-143`).

The ONNX file already exists on disk after `resolve()`, so TVM's native
`relay.frontend.from_onnx` can consume it **directly** — the from-scratch torch
model is not needed for the framework port.

### The framework port, step by step

1. **Reuse the front matter unchanged.** `resolve()` (`classify.py:66`) and
   `preprocess()` (`classify.py:94`) already give you the numpy input `x`
   (`x.numpy()`, shape `(1,3,224,224)`, float32). No change.
2. **Load the ONNX and read its input name dynamically.** The real model's input
   is named **`"data"`**, *not* `"input"` (which is the *scaled* model's ONNX name
   from `model.py`). Never hardcode it — `verify_onnx.py:29-31` already reads it
   from onnxruntime (`sess.get_inputs()[0].name`); the Relay equivalent is:

   ```python
   import onnx
   onnx_model = onnx.load(weights_path)          # weights_path from resolve()
   in_name = onnx_model.graph.input[0].name       # typically "data"
   ```
3. **Import to Relay with the correct shape dict.**

   ```python
   from tvm import relay
   mod, params = relay.frontend.from_onnx(
       onnx_model, shape={in_name: (1, 3, 224, 224)})
   ```
4. **(Optional) run the same passes as the frontend.** `relay_import.py:66-74`
   applies `InferType → SimplifyInference → FoldConstant → FuseOps → InferType`
   under `opt_level=3`. For a plain classifier you can skip these and just build at
   `opt_level=3` — `relay.build` runs its own optimisation.
5. **Build and run with `graph_executor`.** `import_relay` itself is
   **analysis-only** (it stops after the passes, no `relay.build`), so it cannot
   run the model. Build explicitly instead:

   ```python
   import tvm
   from tvm.contrib import graph_executor

   with tvm.transform.PassContext(opt_level=3):
       lib = relay.build(mod, target="llvm", params=params)
   gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))
   gm.set_input(in_name, x.numpy())
   gm.run()
   logits = gm.get_output(0).numpy()[0]           # shape (1000,)
   ```
6. **Tail is identical to `classify.py`.** Feed `logits` through the same
   `softmax` + `topk` + label lookup as `classify.py:136-143`.

### Notes

- **`import_relay` is analysis-only.** It runs the optimisation passes and returns
  `(mod, params)` for the graph walk (`relay_import.py:46-75`); it never calls
  `relay.build`, so use the snippet in step 5 to actually classify.
- **Cross-check the result.** Compare TVM's logits against onnxruntime exactly as
  `verify_onnx.py` compares torch vs onnxruntime (`verify_onnx.py:29-38`); the
  argmax and top-k should agree.
- **Contrast in one line.** The framework port is a ~15-line swap (the ONNX
  already exists; only the input name — `"data"` vs `"input"` — differs, read
  dynamically). The AIE offload is a different, harder problem: the scale-down
  mapping of §12 and the frontend-generalization limits of §13.

## 12. Adapting `classify2.py`'s ResNet specifically

`classify2.py` runs the **full ImageNet ResNet-18 v2** (`resnet18.py`). To bring
it onto the existing int8 path, scale it down and drop/approximate the features
the op set and int8 header cannot express:

| Real model (`resnet18.py`) | Scaled variant the path accepts | Why the change |
|----------------------------|---------------------------------|----------------|
| 224×224×3 input | 8×8×1 (`model.py:35`) | window must fit a tile; config ≤ 255 |
| input BN (`bn0`) | dropped | no input-BN kernel body |
| 7×7 stride-2 conv stem | 3×3 stride-1 stem (`model.py:249`) | no 7×7/stride-2 stem body; `K` ≤ 255 but window/compute must fit |
| 3×3 stride-2 maxpool | dropped | no maxpool op in `KERNEL_BODIES` |
| channels 64→128→256→512 | 4→8→16→32 (`model.py:37`) | 512 > 255 int8 config; buffer size |
| 1000 classes | 4 (`model.py:36`) | 1000 > 255; `avgpool_fc pooled[256]` |
| pre-activation v2 (BN→ReLU→Conv) | post-activation (`model.py:240`) | matches the walk's `conv→BN→[relu]` fusion pattern |
| head `BN→ReLU→GAP→FC` | `GAP→FC` (`avgpool_fc`) | no pre-GAP BN/ReLU body |
| real ImageNet ONNX weights | placeholder ±1 / 1, `bn_scale=64` | no PTQ in this path; keeps AIE == numpy oracle |

The result is a *structural* offload of ResNet-18's residual topology, verified
bit-exact against the numpy oracle — not an accurate ImageNet classifier.

## 13. Limitations & future work

Constraints a custom model must respect on the existing path:

1. **int8/uint8 config** — all `{H,W,Cin,Cout,K,stride}` and
   `{sp_h,sp_w,channels,nclass}` ≤ 255 (`model.py:167`, `kernels.py:58`).
2. **walk validates, does not synthesise** — a custom model is `model.py`'s
   `layer_plan()` + `build_torch_model()` kept in lock-step (`walk.py:103`).
3. **fixed op set** — `conv_bn_relu / conv_bn / residual_add_relu / avgpool_fc`
   only (`kernels.py:181`); no maxpool / 7×7 stem / input-BN.
4. **placeholder weights only** (`make_conv_params` / `make_fc_params`); bit-exact
   vs the oracle, no real quantization.
5. **buffers must fit a tile** — scaled dims (8×8 down to 1×1, ≤ 32 ch) are chosen
   so each launch's windows fit AIE tile memory.
6. **`avgpool_fc` fixed `pooled[256]`** → channels ≤ 256 (`kernels.py:155`).

A true "generalize the frontend" effort would add: real **plan synthesis** from
the fused graph (buffer wiring + residual lengths recovered, not just validated);
**int32 config** headers to lift the 255 limit; **PTQ weights** from the real ONNX
initializers; new **maxpool / 7×7-stem / input-BN** kernel bodies; and
**tiling/splitting** of large layers across the mesh.

## 14. See also

- [`tvm_frontend.md`](tvm_frontend.md) — the frontend deep-dive this recipe complements.
- [`conv2d_im2col_design.md`](conv2d_im2col_design.md) §11 — the generic
  `DmaAddressing` on `TensorParam` that `dma_specs` exposes to Python.
- `src/frontend/tvm/README.md` — module usage.
- `src/mlir/mlirfront/frontend/aietriton/README.md` — the sibling Triton frontend
  and the `_aiebackend` pybind bridge this frontend reuses.
- `example/tileprogram/design/triton/resnet18_triton.py` — the hand-written
  template whose launch sequence, kernel bodies, and CPU references the frontend
  reproduces.
- `example/model/resnet18py/` (`classify2.py`, `resnet18.py`) — the full ImageNet
  ResNet-18 v2 used as the "unmodified big model" contrast in §3(b) / §12.
