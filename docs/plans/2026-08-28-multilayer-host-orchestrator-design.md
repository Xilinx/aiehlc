# Multi-layer host orchestrator (TVM frontend → single `main.elf`)

Status: **design (approved approach: A2 primary → A1 reconstruction, both independently
buildable)**. Companion to [`tvmflow.md`](tvmflow.md) and
[`../../../doc/design/tvm_frontend.md`](../../../doc/design/tvm_frontend.md).

## 1. Problem

The TVM demo (`demo_flow.py` Stage 4) emits **~29 independent per-launch directories**:
each conv layer → its own `host.cc`/`kernel.cc`/`routing.cc` (via `run_aie_pipeline`),
each non-conv layer → its own `<func>.c` (via `cpu_codegen`). There is:

- no `main()`, no program-order execution;
- no single ELF — the emitted host is a **fixed-named** `host_canonicalized(XAie_DevInst*,
  void*, void*, void*)` (collides across layers);
- no DDR buffer chaining (layer *k*'s output = layer *k+1*'s input);
- no ARM integration of the CPU ops.

Goal: produce **one `aout/main.elf`** that, on the board, runs each conv layer on the AIE
array and each CPU op on ARM, in ResNet program order, chaining DDR buffers per the plan.

## 2. Key discovery — the mechanism already exists

The C++ `aiehlc` **multi-kernel mode** already implements exactly this pattern:

| Piece | Location | Role |
|---|---|---|
| `hostFuncSuffix` | `tilinglinalg_pipeline.cpp:736`, `runPipeline` arg | rename `host_canonicalized` → `host_canonicalized_<name>` |
| `appendMode` | `runPipeline` arg | append each layer's host func into ONE `host.cc` |
| `numHostDdrArgs` (out) | `runPipeline` arg | DDR-pointer arity per layer |
| `__aie_launch(kernel, mesh, bufs…)` | `aiehlc.cc:4946-5017` | `__Runtime_set_kernel_elf` + cache sync + call `host_canonicalized_<name>(dev, bufs…)` |
| `_binary_kernel_<name>_start` | `kc.sh:72`, `hostcompile.sh:104-137` | per-kernel ELF embed + `kernel_<name>.o` |

**But** the pybind `run_aie_pipeline` (`aietriton_pybind.cpp:37`) calls `runPipeline` with
only 4 args — it does **not** expose `hostFuncSuffix`/`appendMode`/`numHostDdrArgs`. That
single gap is why the TVM path can only emit isolated per-layer dirs.

The C++ multi-kernel loop is driven by the **Clang frontend parsing a `.cc`** with N
`__global__` kernels (`aiehlc.cc:4416 parsedMeshKernels.size() > 1`). Reproducing that
`.cc` for 29 conv layers (rich `aie::Conv2dSpace` literals) is the expensive part — and
unnecessary for the primary path, because the demo already proves `run_aie_pipeline(2, 2,
tensor_specs, …)` produces a correct per-layer `host.cc`/`kernel.cc` from the *simple*
`tensor_specs` the plan carries.

## 3. Approach: A2 primary → A1 reconstruction (both independently buildable)

Two independent build paths that must agree bit-for-bit; either alone yields a runnable
`main.elf`. Divergence between them is a troubleshooting signal.

```
                         plan (List[LayerOp])  +  lower_aiegraph() launches
                              {op, func_name, index, weights, tensor_specs}
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │  A2 (PRIMARY)                                          │  A1 (RECONSTRUCTED)
              ▼                                                        ▼
  orchestrate_plan pybind:                              A2→A1 parser (Python):
   per conv layer runPipeline(...,                       read A2 artifacts (per-layer
     hostFuncSuffix=<layer>, appendMode=(i>0),            host_canonicalized_<name>,
     &numHostDdrArgs) → ONE dir:                          kernel bodies, tensor_specs,
       host.cc  (N appended host funcs)                   weights, buffer wiring)
       kernel_<name>.cc  (N kernels)                           │  emit ONE standalone
   Python orchestrator.py emits:                               ▼  resnet.cc:
       main.cc  (DDR alloc + program order)               N __global__ conv kernels
       __aie_launch dispatcher (mirror :4946)             + main() calling
       <cpu_op>.c  (plain-C, bit-exact)                    conv_layerN<<<mesh>>>(…)
              │                                            + inline CPU C
              ▼                                                  │  run `aiehlc resnet.cc`
   multi-kernel hostcompile.sh → main.elf  (A2)                 ▼  (existing Clang path)
                                                          multi-kernel build → main.elf (A1)
```

- **A2** is the reference path: it reuses the *exact input the working demo produces*
  (`tensor_specs`) and only adds wiring (suffix + append + one driver + CPU-op C).
- **A1** is a *reconstruction* from A2's artifacts into a hand-editable standalone
  `resnet.cc` that builds through the *entire* existing `aiehlc` Clang path — useful for
  human inspection, hand-tweaking one layer, and cross-checking A2.
- Both feed the **same** multi-kernel `hostcompile.sh` build backend; both produce
  `main.elf`. A test builds both and diffs the run outputs.

## 4. What each launch carries (input to both paths)

`lower_aiegraph(ir)` returns per op (`aietriton_pybind.cpp:249-263`):

```
{ "op": str,               # conv_bn_relu | conv_bn | residual_add_relu | avgpool_fc
  "func_name": str,        # unique per launch (e.g. conv_bn_relu_0)
  "index": int,            # program order
  "weights": str,          # SymbolRefAttr name for this op's params
  "tensor_specs": [ (shape:list[int], bits:int, is_input:bool), … ] }
```

`plan[i]` (LayerOp) additionally carries the **named buffer wiring** (input/feat1/…/
logits) already resolved to SSA producer indices — this drives DDR buffer chaining.

## 5. Component design

### 5.1 New pybind: `orchestrate_conv_layer` (thin)
Expose the already-present `runPipeline` multi-kernel args. Signature mirrors
`run_aie_pipeline` **plus** `host_func_suffix`, `append_mode`; returns `num_host_ddr_args`
(so the Python driver knows each layer's buffer arity). ≤ ~40 lines; just forwards to
`runPipeline(..., hostFuncSuffix, appendMode, &numHostDdrArgs, …)` exactly as
`aiehlc.cc:4624` does. No new lowering logic.

### 5.2 New Python module: `src/frontend/tvm/orchestrator.py`
- `orchestrate_plan(plan, launches, out_dir) -> build_dir` — the A2 driver:
  1. For each conv launch *i*: call `orchestrate_conv_layer(2, 2, tensor_specs, out_dir,
     body, func_name, host_func_suffix=func_name, append_mode=(first_conv==False))`.
     Accumulate `num_host_ddr_args[func_name]`.
  2. Emit **plain-C CPU ops** for non-conv layers (`cpu_codegen` gains a `plain_c` path —
     see §5.3).
  3. Emit `main.cc`:
     - allocate DDR buffers (`__Runtime_Alloc`) sized from `tensor_specs`, one per distinct
       plan buffer (input/feat*/tmp*/skip_ds/logits);
     - in program order, for each layer emit either `__aie_launch("<func>", mesh, <in>,
       <out>, …)` (conv) or a direct `<func>(<in>, <out>)` call (CPU op), wiring buffers
       per the plan's named edges;
     - fill layer-0 input from the demo `demo_input`; read back `logits`.
  4. Emit the `__aie_launch` dispatcher (mirror of `aiehlc.cc:4946`: `strcmp` chain →
     `set_kernel_elf(_binary_kernel_<name>_start)` + sync + `host_canonicalized_<name>(dev,
     bufs…)`).
  5. Return the build dir; caller invokes multi-kernel `hostcompile.sh`.
  Each function ≤ 200 lines (project rule); split emit helpers per concern (buffers,
  launches, dispatcher, cpu-calls).

### 5.3 CPU ops as plain aarch64 C (`cpu_codegen.py` extension)
Add `plain_c_source(op, func_name) -> str` emitting a self-contained
`void <func_name>(const int8_t* a, const int8_t* b, int8_t* out)` (residual) /
`void <func_name>(const int8_t* feat, const int8_t* w, const int8_t* bias, int8_t* out)`
(avgpool_fc), computing the **same int16-accumulate/clamp** semantics as
`_compiler._cpu_add_relu` / `_cpu_avgpool_fc`. No TVM FFI runtime on the board. A unit test
compiles this C with the host compiler and asserts elementwise-equal to the numpy oracle
(reuses the existing `test_cpu_codegen_bit_exact` harness with `target="c"` swapped for the
plain-C compile+run). The existing FFI-packed `.c` emit stays for the standalone demo.

### 5.4 A2→A1 parser: `src/frontend/tvm/a2_to_a1.py`
- `reconstruct_resnet_cc(build_dir, plan, launches) -> resnet_cc_path`:
  read the A2 per-layer kernel bodies + `tensor_specs` + `weights` + buffer wiring, and
  emit ONE standalone `resnet.cc`: N `__global__` conv kernels (body = the same
  `kernels.kernel_body_for` used by A2, wrapped in an `aie::Conv2dSpace`/`GemmSpace` port
  decl derived from `tensor_specs`), a `main()` calling `conv_layerN<<<mesh>>>(…)` in order
  with inline CPU-op C. Builds via `aiehlc resnet.cc` → same multi-kernel backend.
- Independence: `resnet.cc` has no dependency on the A2 build dir once emitted.

## 6. DDR buffer chaining (shared by A2 and A1)

The plan's named edges (already resolved to producer SSA indices) define a small buffer
graph. The orchestrator allocates one DDR region per distinct buffer name, sizes it from
the producing layer's output `tensor_specs`, and passes `(ptr, size)` pairs into
`__aie_launch` / CPU calls. Cache sync is handled by the emitted `__aie_launch` (it already
calls `__Runtime_sync_for_dev` per buffer, `aiehlc.cc:4942`); CPU ops run on ARM directly
on the same DDR pointers (invalidate before/flush after around each CPU call).

## 7. Build & run

- **A2:** `orchestrate_plan` writes `main.cc` + `host.cc` (appended) + `kernel_<name>.cc` +
  `<cpu_op>.c` into one dir → run multi-kernel `hostcompile.sh` → `aout/main.elf`.
- **A1:** `reconstruct_resnet_cc` → `aiehlc resnet.cc` (existing Clang multi-kernel path) →
  `aout/main.elf`.
- **HW run:** `python test/mainelfpaltest.py` (board `palmyra`, ELF `~/aiehlc/main.elf`).

## 8. Verification

1. **Unit — plain-C CPU bit-exact:** compile `plain_c_source(...)` with host CC, run on
   random int8, assert elementwise-equal to numpy oracle.
2. **Unit — orchestrate emits ONE host.cc** with N `host_canonicalized_<name>` funcs +
   `__aie_launch` dispatcher + `main`; no per-layer dir collisions.
3. **Unit — A2→A1 parser** produces a `resnet.cc` that parses (aiehlc dry-run) and declares
   N kernels + main.
4. **Integration — dual build parity:** build A2 `main.elf` and A1 `main.elf`; run both in
   sim (or board) on the same `demo_input`; assert identical `logits` and identical to
   `cpu_reference(plan, demo_input)` (bit-exact end-to-end).
5. **HW smoke:** `mainelfpaltest.py` → expect `device_teardown done`, no `AIE ERROR`.

## 9. Out of scope

- BYOC graph partitioning; generic Relay→C for arbitrary ops.
- Routing *reuse* across layers (each conv layer reconfigures routing via its own
  `host_canonicalized_<name>` — same as the C++ multi-kernel path today).
- Autotuning tile/halo geometry (uses the plan's `tensor_specs` as-is).

## 10. Files (planned)

New:
- `src/frontend/tvm/orchestrator.py` — A2 driver (main.cc + dispatcher + build glue).
- `src/frontend/tvm/a2_to_a1.py` — A2→A1 `resnet.cc` reconstructor.

Modify:
- `src/mlir/mlirfront/frontend/aietriton/aietriton_pybind.cpp` — add
  `orchestrate_conv_layer` exposing `hostFuncSuffix`/`appendMode`/`numHostDdrArgs`.
- `src/frontend/tvm/cpu_codegen.py` — add `plain_c_source` (plain aarch64 C path).
- `src/frontend/tvm/demo_flow.py` — Stage 5: call `orchestrate_plan` (+ optional A1).
- `src/frontend/tvm/test_tvm_frontend.py` — tests §8.1–8.4.
- docs: this file, `tvmflow.md`, `doc/design/tvm_frontend.md`, `CLAUDE.md`.
