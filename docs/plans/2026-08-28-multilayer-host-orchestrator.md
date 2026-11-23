# Multi-layer Host Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce one runnable `aout/main.elf` that runs all ~29 TVM-frontend ResNet layers in program order (conv on AIE, CPU ops on ARM, DDR buffers chained), via an **A2 primary** path (drive the existing multi-kernel runtime plumbing) plus an **A1 reconstruction** path (standalone `resnet.cc`), both independently buildable.

**Architecture:** Reuse the C++ `aiehlc` multi-kernel mechanism that already exists (`hostFuncSuffix` rename + `appendMode` in `runPipeline`, the `__aie_launch` dispatcher pattern from `aiehlc.cc:4946`, `__Runtime_set_kernel_elf`, `_binary_kernel_<name>_start` embed). A2: a thin new pybind exposes those existing `runPipeline` args; Python `orchestrator.py` appends per-layer host funcs into one dir and emits `main.cc` + dispatcher + plain-C CPU ops. A1: `a2_to_a1.py` reconstructs a standalone `resnet.cc` from A2 artifacts. Both feed the same multi-kernel `hostcompile.sh`.

**Tech Stack:** MLIR/C++ (`TilingLinalgPipeline::runPipeline`), pybind11 (`_aietriton_core`), Python 3.10 (`src/frontend/tvm/`), TVM `target="c"` (existing) + plain-C emit (new), aarch64 cross-compile via `script/hostcompile.sh`.

**Design doc:** `src/frontend/tvm/design/orchestrator.md` (== `docs/plans/2026-08-28-multilayer-host-orchestrator-design.md`).

**Reference before touching code:**
- `src/llvm/aiehlc.cc:4416-5022` — multi-kernel loop + `__aie_launch` dispatcher emit (the pattern A2 mirrors).
- `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h:288-317` — `runPipeline` signature (`hostFuncSuffix`, `appendMode`, `numHostDdrArgs`).
- `src/mlir/mlirfront/frontend/aietriton/aietriton_pybind.cpp:15-38, 241-295` — `run_aie_pipeline` + `lower_aiegraph` launch dict.
- `src/frontend/tvm/_compiler.py` — `_cpu_add_relu` (:87), `_cpu_avgpool_fc` (:96), `compile_plan_via_aiegraph` (:294), `cpu_reference`.
- `src/frontend/tvm/cpu_codegen.py` — `AIE_OPS`, `is_aie_op`, `_residual_te`, `_avgpool_fc_te`, `emit_cpu_launch`.
- `example/tileprogram/design/ccode/worklocal/host.cc:282-329` — emitted `__aie_launch` + user `main` shape.
- `script/hostcompile.sh:97-140` — multi-kernel build (`kernel_<name>.cc` detect → `kernel_<name>.o`).

**Conventions:**
- No function > 200 lines (project rule). Split emit helpers.
- Tests are pytest-style `test_*` funcs in `src/frontend/tvm/test_tvm_frontend.py`; TVM/pybind-dependent tests **skip** (not fail) when unavailable (mirror `test_cpu_codegen_bit_exact`, `test_emit_aie`).
- Run all Bash with sandbox disabled (git-worktree `.git` breaks bwrap).
- After each task: `git add <specific files>` + commit; list changed/created files (project "Process transparent rule").

---

## Task 1: Expose multi-kernel args in the pybind (`orchestrate_conv_layer`)

**Files:**
- Modify: `src/mlir/mlirfront/frontend/aietriton/aietriton_pybind.cpp` (add fn near `run_aie_pipeline` at `:15-38`; register near `:270`)
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_orchestrate_conv_layer_appends`)

**Step 1: Write the failing test**

```python
def test_orchestrate_conv_layer_appends():
    """orchestrate_conv_layer appends host_canonicalized_<suffix> into ONE host.cc."""
    try:
        core = _compiler._core()
    except Exception as e:
        print("SKIP (pybind not built):", e); return
    import tempfile
    plan = build_plan(None)
    launches = core.lower_aiegraph(_compiler.build_aiegraph_ir(plan))
    convs = [(op, L) for op, L in zip(plan, launches) if cpu_codegen.is_aie_op(op.op)][:2]
    with tempfile.TemporaryDirectory() as d:
        for i, (op, L) in enumerate(convs):
            specs = [(list(s), int(b), bool(x)) for (s, b, x) in L["tensor_specs"]]
            body = kernels.kernel_body_for(op.op, L["func_name"])
            n = core.orchestrate_conv_layer(2, 2, specs, d, body, L["func_name"],
                                            host_func_suffix=L["func_name"],
                                            append_mode=(i > 0))
            assert isinstance(n, int) and n >= 1
        host = open(os.path.join(d, "host.cc")).read()
        for _, L in convs:
            assert f"host_canonicalized_{L['func_name']}(" in host
        # exactly one host.cc, both funcs appended (no collision)
        assert host.count("host_canonicalized_") >= 2
```

**Step 2: Run to verify it fails**

Run: `cd <repo>/.worktrees/cnn && python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_conv_layer_appends -v` (sandbox off)
Expected: FAIL — `AttributeError: 'module' has no attribute 'orchestrate_conv_layer'` (or SKIP if pybind unbuilt — in that case build it first, Step 3b).

**Step 3: Add the pybind function**

In `aietriton_pybind.cpp`, add after `run_aie_pipeline` (mirrors it, forwarding the already-present `runPipeline` multi-kernel args):

```cpp
static int orchestrate_conv_layer(
    int meshRows, int meshCols,
    const std::vector<std::tuple<std::vector<int64_t>, int, bool>> &tensorSpecs,
    const std::string &outputDir, const std::string &userKernelBody,
    const std::string &userKernelFuncName, const std::string &hostFuncSuffix,
    bool appendMode) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    std::vector<TensorParam> tensors;
    for (auto &[shape, bw, isIn] : tensorSpecs) tensors.push_back({shape, bw, isIn});
    SplitModel splitModel = SplitModel::gemm();
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, meshRows, meshCols, tensors, splitModel);
    unsigned numHostDdrArgs = 0;
    bool ok = TilingLinalgPipeline::runPipeline(
        ctx, module, outputDir, userKernelBody, userKernelFuncName,
        /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", /*tensors=*/{},
        /*maxPingPongBytes=*/4096, /*aieGen=*/"Gen2", hostFuncSuffix, appendMode,
        &numHostDdrArgs);
    if (!ok) throw std::runtime_error("orchestrate_conv_layer: runPipeline failed");
    return (int)numHostDdrArgs;
}
```

Register near `:270`:

```cpp
    m.def("orchestrate_conv_layer", &orchestrate_conv_layer,
          py::arg("mesh_rows"), py::arg("mesh_cols"), py::arg("tensor_specs"),
          py::arg("output_dir"), py::arg("user_kernel_body"), py::arg("user_kernel_func_name"),
          py::arg("host_func_suffix"), py::arg("append_mode"),
          "Multi-kernel variant of run_aie_pipeline: emits host_canonicalized_<suffix>\n"
          "into output_dir/host.cc (append_mode appends). Returns numHostDdrArgs.");
```

**Step 3b: Rebuild the pybind extension**

Run the extension build (locate its build script/CMake target first):
Run: `cd .worktrees/cnn && find . -name '*.so' -path '*aietriton*'` and check `src/mlir/mlirfront/frontend/aietriton/` for a `build.sh`/CMake. Rebuild so `_aietriton_core.*.so` picks up the new symbol. Confirm: `python -c "from aietriton import _aietriton_core as c; print(hasattr(c,'orchestrate_conv_layer'))"` → `True`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_conv_layer_appends -v`
Expected: PASS (or SKIP only if pybind genuinely cannot build in this env).

**Step 5: Commit**

```bash
git add src/mlir/mlirfront/frontend/aietriton/aietriton_pybind.cpp src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): expose orchestrate_conv_layer pybind (multi-kernel append)"
```

---

## Task 2: Plain-C CPU op emit (`cpu_codegen.plain_c_source`)

**Files:**
- Modify: `src/frontend/tvm/cpu_codegen.py` (add `plain_c_source`, `emit_cpu_launch_plain`)
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_plain_c_cpu_bit_exact`)

**Step 1: Write the failing test** (compiles the emitted C with host cc, runs, compares to numpy oracle)

```python
def test_plain_c_cpu_bit_exact():
    """plain_c_source emits self-contained C that is bit-exact with the numpy oracle."""
    import shutil, subprocess, ctypes, tempfile
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        print("SKIP (no C compiler)"); return
    plan = build_plan(None)
    res = next(op for op in plan if op.op == "residual_add_relu")
    src = cpu_codegen.plain_c_source(res, "res_test")
    assert "void res_test(" in src
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "res.c"); sopath = os.path.join(d, "res.so")
        open(cpath, "w").write(src)
        subprocess.run([cc, "-shared", "-fPIC", "-O2", cpath, "-o", sopath], check=True)
        lib = ctypes.CDLL(sopath)
        n = 64
        a = (np.random.randint(-128, 128, n)).astype(np.int8)
        b = (np.random.randint(-128, 128, n)).astype(np.int8)
        out = np.zeros(n, dtype=np.int8)
        p = ctypes.POINTER(ctypes.c_int8)
        lib.res_test(a.ctypes.data_as(p), b.ctypes.data_as(p), out.ctypes.data_as(p),
                     ctypes.c_int(n))
        ref = _compiler._cpu_add_relu(a, b)
        assert np.array_equal(out, ref)
```

**Step 2: Run to verify it fails**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_plain_c_cpu_bit_exact -v`
Expected: FAIL — `AttributeError: module 'cpu_codegen' has no attribute 'plain_c_source'`.

**Step 3: Implement `plain_c_source`** (mirror `_compiler._cpu_add_relu` / `_cpu_avgpool_fc` semantics exactly: int16 accumulate, clamp, int8 store)

```python
def _plain_c_residual(func_name, n):
    return (f"#include <stdint.h>\n"
            f"void {func_name}(const int8_t* a, const int8_t* b, int8_t* out, int n) {{\n"
            f"  for (int i = 0; i < n; ++i) {{\n"
            f"    int16_t s = (int16_t)a[i] + (int16_t)b[i];\n"
            f"    if (s > 127) s = 127; if (s < 0) s = 0;\n"   # relu: lo=0
            f"    out[i] = (int8_t)s;\n  }}\n}}\n")

def _plain_c_avgpool_fc(func_name, sh, sw, ch, nc):
    # pooled[c] = int8( trunc( sum_spatial int16(feat) / (sh*sw) ) ); feat>=0 => floor
    # logits[j] = int8(clamp( sum_c int16(pooled[c])*int16(w[c*nc+j]) + int16(bias[j]), -128,127))
    sp = sh * sw
    return (f"#include <stdint.h>\n"
            f"void {func_name}(const int8_t* feat, const int8_t* w, const int8_t* bias,"
            f" int8_t* out) {{\n"
            f"  int C={ch}, NC={nc}, SP={sp};\n"
            f"  int8_t pooled[{ch}];\n"
            f"  for (int c=0;c<C;++c){{ int16_t acc=0; for(int s=0;s<SP;++s)"
            f" acc += (int16_t)feat[c*SP+s]; pooled[c]=(int8_t)(acc/SP); }}\n"
            f"  for (int j=0;j<NC;++j){{ int16_t acc=(int16_t)bias[j];"
            f" for(int c=0;c<C;++c) acc += (int16_t)pooled[c]*(int16_t)w[c*NC+j];"
            f" if(acc>127)acc=127; if(acc<-128)acc=-128; out[j]=(int8_t)acc; }}\n}}\n")

def plain_c_source(op, func_name):
    if op.op == "residual_add_relu":
        n = int(np.prod(op.out_shape))   # resolve length from op metadata
        return _plain_c_residual(func_name, n)
    if op.op == "avgpool_fc":
        sh, sw, ch, nc = _avgpool_dims(op)   # reuse the same dim extraction as _avgpool_fc_te
        return _plain_c_avgpool_fc(func_name, sh, sw, ch, nc)
    raise ValueError(f"plain_c_source: not a CPU op: {op.op}")
```

> NOTE at implementation time: read `cpu_codegen._residual_te` / `_avgpool_fc_te` and
> `_compiler._cpu_avgpool_fc` to copy the EXACT dim-extraction and the exact
> clamp/trunc/accumulate order. The C must be byte-identical to the oracle. Verify sign of
> `acc/SP` truncation matches numpy (feat can be negative post-conv? oracle uses trunc — if
> feat>=0 by construction, floor==trunc; confirm from `_cpu_avgpool_fc`).

**Step 4: Run to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_plain_c_cpu_bit_exact -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/frontend/tvm/cpu_codegen.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): plain-C bit-exact emit for residual_add_relu/avgpool_fc"
```

---

## Task 3: DDR buffer graph from the plan (`orchestrator._buffer_graph`)

**Files:**
- Create: `src/frontend/tvm/orchestrator.py`
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_buffer_graph_chains`)

**Step 1: Write the failing test**

```python
def test_buffer_graph_chains():
    """_buffer_graph maps each layer to (in_bufs, out_buf) with sizes, chaining producers."""
    from frontend.tvm import orchestrator
    plan = build_plan(None)
    g = orchestrator._buffer_graph(plan)
    assert g.entry_buffer is not None            # layer-0 input
    assert g.logits_buffer is not None           # final output
    # every consumed buffer (except entry) is produced by an earlier layer
    for layer in g.layers:
        for b in layer.in_bufs:
            assert b == g.entry_buffer or b in g.produced_before(layer.index)
    # sizes are positive ints
    for name, sz in g.sizes.items():
        assert isinstance(sz, int) and sz > 0
```

**Step 2: Run to verify it fails**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_buffer_graph_chains -v`
Expected: FAIL — module `orchestrator` missing / `_buffer_graph` undefined.

**Step 3: Implement `_buffer_graph`** — read the LayerOp named-edge fields (input/feat*/tmp*/skip_ds/logits already resolved to producer indices; confirm exact field names from `model.py`/`walk.py`). Build a small dataclass graph: per layer `{index, op, func_name, in_bufs:[name], out_buf:name}` + `sizes[name] = prod(shape)` from `tensor_specs`. Provide `entry_buffer`, `logits_buffer`, `produced_before(idx)`.

**Step 4: Run to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_buffer_graph_chains -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/frontend/tvm/orchestrator.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): plan buffer graph for DDR chaining"
```

---

## Task 4: Emit `main.cc` + `__aie_launch` dispatcher (A2 driver)

**Files:**
- Modify: `src/frontend/tvm/orchestrator.py` (`_emit_dispatcher`, `_emit_main`, `orchestrate_plan`)
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_orchestrate_plan_emits_driver`)

**Step 1: Write the failing test**

```python
def test_orchestrate_plan_emits_driver():
    """orchestrate_plan writes ONE host.cc (N funcs), main.cc (program order), dispatcher, cpu .c."""
    try:
        core = _compiler._core()
    except Exception as e:
        print("SKIP (pybind not built):", e); return
    import tempfile
    plan = build_plan(None)
    launches = core.lower_aiegraph(_compiler.build_aiegraph_ir(plan))
    with tempfile.TemporaryDirectory() as d:
        from frontend.tvm import orchestrator
        bd = orchestrator.orchestrate_plan(plan, launches, d)
        host = open(os.path.join(bd, "host.cc")).read()
        main = open(os.path.join(bd, "main.cc")).read()
        assert "__aie_launch(" in host and "__Runtime_set_kernel_elf(" in host
        assert "int main(" in main
        # program order: conv launches appear as __aie_launch, cpu ops as direct calls
        conv_names = [L["func_name"] for op, L in zip(plan, launches)
                      if cpu_codegen.is_aie_op(op.op)]
        for cn in conv_names[:3]:
            assert cn in main
        # cpu op .c present
        assert any(f.endswith(".c") for f in os.listdir(bd))
```

**Step 2: Run to verify it fails**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_plan_emits_driver -v`
Expected: FAIL — `orchestrate_plan` undefined.

**Step 3: Implement the driver** — `orchestrate_plan(plan, launches, out_dir)`:
1. First conv sets `append_mode=False`; subsequent convs append. Call `core.orchestrate_conv_layer(...)`; record `numHostDdrArgs` per func.
2. For CPU layers, write `cpu_codegen.plain_c_source(op, func)` → `<func>.c`.
3. `_emit_dispatcher(host_cc, convs, ddr_args)` — append a `__aie_launch(const char* kernel, aieMesh mesh, void* _t0, size_t _s0, …)` mirroring `aiehlc.cc:4946`: `strcmp` chain → `__Runtime_set_kernel_elf(_binary_kernel_<name>_start)` + `__Runtime_sync_for_dev` per buf + `host_canonicalized_<name>(dev, _t0, …)`.
4. `_emit_main(buffer_graph, launches)` — `__Runtime_Alloc` per distinct buffer; per layer in program order emit `__aie_launch("<func>", mesh, in, sin, out, sout, …)` (conv) or `<func>(in, out[, …])` (cpu); fill entry from a provided input array; read `logits`.
Keep each emit fn < 200 lines.

**Step 4: Run to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_plan_emits_driver -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/frontend/tvm/orchestrator.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): emit A2 main.cc + __aie_launch dispatcher + cpu glue"
```

---

## Task 5: Build A2 `main.elf` via multi-kernel hostcompile

**Files:**
- Modify: `src/frontend/tvm/orchestrator.py` (`build_main_elf`)
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_orchestrate_builds_elf`, gated on toolchain)

**Step 1: Write the failing test**

```python
def test_orchestrate_builds_elf():
    """A2 build produces aout/main.elf (skips if cross-toolchain/pybind absent)."""
    try:
        core = _compiler._core()
    except Exception as e:
        print("SKIP (pybind not built):", e); return
    import shutil
    if shutil.which("aarch64-linux-gnu-g++") is None:
        print("SKIP (no aarch64 toolchain)"); return
    import tempfile
    plan = build_plan(None)
    launches = core.lower_aiegraph(_compiler.build_aiegraph_ir(plan))
    with tempfile.TemporaryDirectory() as d:
        from frontend.tvm import orchestrator
        bd = orchestrator.orchestrate_plan(plan, launches, d)
        elf = orchestrator.build_main_elf(bd)
        assert elf and os.path.exists(elf)
```

**Step 2: Run to verify it fails**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_builds_elf -v`
Expected: FAIL — `build_main_elf` undefined (or SKIP if toolchain absent).

**Step 3: Implement `build_main_elf(build_dir)`** — arrange the dir to match the multi-kernel `hostcompile.sh` contract (`kernel_<name>.cc` per conv, `host.cc`, `main.cc` as user source, `<cpu>.c` compiled for aarch64 and linked), invoke `script/hostcompile.sh` (read `:97-140, :349-490`), return the produced `aout/main.elf` path. Reuse existing script rather than re-implementing linking.

**Step 4: Run to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_orchestrate_builds_elf -v`
Expected: PASS (or SKIP without toolchain).

**Step 5: Commit**

```bash
git add src/frontend/tvm/orchestrator.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): build A2 main.elf via multi-kernel hostcompile"
```

---

## Task 6: A2→A1 reconstruction (`a2_to_a1.reconstruct_resnet_cc`)

**Files:**
- Create: `src/frontend/tvm/a2_to_a1.py`
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_a2_to_a1_reconstructs`)

**Step 1: Write the failing test**

```python
def test_a2_to_a1_reconstructs():
    """reconstruct_resnet_cc emits a standalone resnet.cc: N kernels + main, no A2-dir deps."""
    try:
        core = _compiler._core()
    except Exception as e:
        print("SKIP (pybind not built):", e); return
    import tempfile
    plan = build_plan(None)
    launches = core.lower_aiegraph(_compiler.build_aiegraph_ir(plan))
    with tempfile.TemporaryDirectory() as d:
        from frontend.tvm import orchestrator, a2_to_a1
        bd = orchestrator.orchestrate_plan(plan, launches, d)
        cc = a2_to_a1.reconstruct_resnet_cc(bd, plan, launches, os.path.join(d, "resnet.cc"))
        src = open(cc).read()
        n_conv = sum(1 for op in plan if cpu_codegen.is_aie_op(op.op))
        assert src.count("__global__") >= n_conv
        assert "int main(" in src
        assert "<<<" in src  # kernel-launch syntax
```

**Step 2: Run to verify it fails**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_a2_to_a1_reconstructs -v`
Expected: FAIL — module `a2_to_a1` missing.

**Step 3: Implement `reconstruct_resnet_cc`** — from A2 artifacts (`kernel_<name>.cc` bodies, `tensor_specs`, `weights`, buffer graph) emit ONE `resnet.cc`: per conv an `aie::Conv2dSpace`/`GemmSpace` port decl derived from `tensor_specs` (shapes → ih/iw/ic/oc etc.; reuse `kernels.kernel_body_for` for the body), a `main()` mirroring `orchestrator._emit_main` but using `conv_layerN<<<mesh>>>(…)` launch syntax + inline plain-C CPU ops. Standalone (no A2 dir refs after emission).

> NOTE: deriving correct `Conv2dSpace` literals from bare `tensor_specs` is the hard part
> (see design §3 "Con"). Start with the simplest space that the pipeline accepts for these
> shapes; if geometry can't be reconstructed from `tensor_specs` alone, extend
> `lower_aiegraph` to also return per-conv geometry (kh/kw/stride/pad) and thread it through
> — small, additive.

**Step 4: Run to verify it passes**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_a2_to_a1_reconstructs -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/frontend/tvm/a2_to_a1.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): reconstruct standalone resnet.cc from A2 artifacts (A1 path)"
```

---

## Task 7: Dual-build parity + demo Stage 5

**Files:**
- Modify: `src/frontend/tvm/demo_flow.py` (Stage 5: call `orchestrate_plan`, optional A1)
- Test: `src/frontend/tvm/test_tvm_frontend.py` (new `test_dual_build_parity`, sim-gated)

**Step 1: Write the failing test** (parity: A2 and A1 `main.elf` produce identical logits, both == `cpu_reference`)

```python
def test_dual_build_parity():
    """A2 and A1 main.elf produce identical logits, both == cpu_reference (sim/hw gated)."""
    # Gate on the availability of a runnable substrate (sim or board); otherwise SKIP.
    if os.environ.get("AIE_RUN_SUBSTRATE") is None:
        print("SKIP (no AIE_RUN_SUBSTRATE=sim|hw)"); return
    # ... build A2 elf, build A1 elf, run both on demo_input, compare logits to
    #     _compiler.cpu_reference(plan, demo_input). Assert all three equal.
```

**Step 2: Run to verify it fails/skips**

Run: `python -m pytest src/frontend/tvm/test_tvm_frontend.py::test_dual_build_parity -v`
Expected: SKIP (no substrate) initially; becomes runnable under sim/hw.

**Step 3: Implement demo Stage 5** — after Stage 4 emits per-layer artifacts, add:
```python
# Stage 5: orchestrate all launches into one main.elf (A2), optionally reconstruct A1.
from frontend.tvm import orchestrator, a2_to_a1
bd = orchestrator.orchestrate_plan(plan, list(core.lower_aiegraph(ir)), OUT)
print("A2 build dir:", bd)
# elf = orchestrator.build_main_elf(bd)     # when cross-toolchain present
# a1  = a2_to_a1.reconstruct_resnet_cc(bd, plan, launches, os.path.join(OUT, "resnet.cc"))
```
Fill in the parity test body to run both elfs when substrate is set.

**Step 4: Run demo + parity**

Run: `python src/frontend/tvm/demo_flow.py` — expect a Stage-5 A2 build dir with one `host.cc`/`main.cc` + `kernel_<name>.cc` + `<cpu>.c`.
Run: `AIE_RUN_SUBSTRATE=sim python -m pytest ...::test_dual_build_parity -v` when sim is wired.
Expected: demo PASS; parity PASS under sim.

**Step 5: Commit**

```bash
git add src/frontend/tvm/demo_flow.py src/frontend/tvm/test_tvm_frontend.py
git commit -m "feat(tvm): demo Stage 5 orchestrate + dual-build parity test"
```

---

## Task 8: HW smoke + docs

**Files:**
- Modify: `src/frontend/tvm/design/orchestrator.md`, `doc/design/tvm_frontend.md`, `src/frontend/tvm/design/tvmflow.md`, `CLAUDE.md`

**Step 1: HW smoke run** (board `palmyra`)

Run: build A2 `main.elf`, copy to `~/aiehlc/main.elf`, `python test/mainelfpaltest.py`.
Expected: `device_teardown done`, no `AIE ERROR`.

**Step 2: Update docs** — mark orchestrator design "implemented"; add a "Multi-layer orchestrator (A2/A1)" subsection to `tvm_frontend.md` and `tvmflow.md`; note in `CLAUDE.md` under the aiegraph section that the TVM frontend now emits a single `main.elf` via the multi-kernel path. Per "Process transparent rule", list all changed/created files.

**Step 3: Commit**

```bash
git add src/frontend/tvm/design/orchestrator.md doc/design/tvm_frontend.md src/frontend/tvm/design/tvmflow.md CLAUDE.md
git commit -m "docs(tvm): multi-layer orchestrator (A2/A1) design + arch updates"
```

---

## Full test run (after all tasks)

Run: `cd .worktrees/cnn && python -m pytest src/frontend/tvm/test_tvm_frontend.py -v` (sandbox off)
Expected: all prior 11 tests + 8 new tests PASS or SKIP (never FAIL). New: `test_orchestrate_conv_layer_appends`, `test_plain_c_cpu_bit_exact`, `test_buffer_graph_chains`, `test_orchestrate_plan_emits_driver`, `test_orchestrate_builds_elf`, `test_a2_to_a1_reconstructs`, `test_dual_build_parity`.

## Risk register

- **R1 (Conv2dSpace reconstruction, Task 6):** `tensor_specs` may lack conv geometry. Mitigation: additively extend `lower_aiegraph` to also emit kh/kw/stride/pad per conv.
- **R2 (avgpool trunc sign):** confirm `acc/SP` C truncation == numpy oracle when feat can be negative. Mitigation: match `_cpu_avgpool_fc` exactly; add explicit floor if needed.
- **R3 (routing re-config between layers):** each `host_canonicalized_<name>` reconfigures its own routing (same as C++ multi-kernel today) — verify no cross-layer stream-switch residue on HW; covered by Task 8 smoke.
- **R4 (DDR lifetime):** all layer buffers co-resident in DDR; size the alloc set from the buffer graph; if too large, add buffer reuse (out of scope v1).
