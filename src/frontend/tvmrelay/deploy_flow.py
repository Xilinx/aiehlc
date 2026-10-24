###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Deploy a real ONNX ResNet-18 through Relay to a board ELF, stage by stage.

    ensure_tvm016()  ->  download resnet18-v1-7.onnx  ->  Relay
                     ->  int8 quantize  ->  target="c" codegen
                     ->  per-layer folders  ->  baremetal aarch64 main.elf
                     ->  CPU reference classification

Seven stages, matching the seven things this flow has to prove:

1. **Environment** — ``onnx_compat.ensure_tvm016()`` guarantees TVM 0.16 with a
   working ``tvm.relay`` (Relay was removed in the Relax transition, so the
   modern TVM in ``src/frontend/tvm`` cannot do this at all), and
   ``ensure_onnx_mapping()`` restores the ``onnx.mapping`` module that onnx 1.16
   deleted and TVM 0.16 still imports. Both come from importing the package.

2. **Import** — fetch the pretrained ImageNet ResNet-18 (opset 7, 46 MB) and run
   ``relay.frontend.from_onnx``. The 99 initializers arrive as inline
   ``relay.Constant`` nodes, *not* in the returned ``params`` dict (which comes
   back empty even under ``freeze_params=True``), so anything wanting weights
   must read them out of the graph.

3. **Quantize** — int8 PTQ **in ONNX**, before Relay sees the graph
   (``onnx_ptq.py``): symmetric per-channel int8 weights + **asymmetric**
   activations, covering every conv *including the first* and the input image
   itself. The quantized QDQ model is then imported and run through
   ``FakeQuantizationToInteger`` so the graph is genuinely integer rather than
   ``dequantize -> fp32 op -> quantize``.

   The quantization itself needs no LLVM (it happens in onnxruntime), but the
   build still does: ``FakeQuantizationToInteger`` calls ``FoldConstantExpr``
   directly (``fake_quantization_to_integer.py:35``), which
   ``disabled_pass=["FoldConstant"]`` cannot reach, so codegen on a
   ``USE_LLVM=OFF`` build dies with ``target.build.llvm is not enabled``
   *after* stage 3 has already succeeded.

   ``--relay-ptq`` selects the older ``relay.quantize`` path instead. It is
   symmetric-only -- its annotation op is
   ``simulated_quantize(data, scale, clip_min, clip_max)``, with no zero-point
   anywhere -- and it leaves an fp32 head and tail
   (``skip_conv_layers=[0]``, ``skip_dense_layer=True``). It is also **gated on
   LLVM**: see the note below; when the gate is shut the stage is skipped and
   the flow continues in fp32 rather than emitting something mislabelled as
   int8.

   The ONNX path currently produces a *larger* ELF (~24 MB vs ~14 MB) because
   conv **operands are stored int16**. The values are genuine int8; only the
   storage width is not. It is still the default because it quantizes the
   whole network and preserves the fp32 top-5 ordering, which the symmetric
   path does not.

   The int16 is **not** the generic legalization -- that one returns ``None``.
   ``Target("c")`` carries ``keys=['cpu']``, so the **Intel** registration
   ``_qnn_conv2d_legalize_intel_cpu`` is what fires
   (``qnn/op/legalizations.py:520``); its gate ``is_fast_int8_on_intel()`` is
   ``target_has_features("sse4.2")``, false for a C-source target, so it takes
   ``helper_no_fast_int8_hw_legalization``, which casts **data and kernel** to
   int16 and subtracts the zero-points eagerly. Registering a ``"cpu"``
   legalization that returns ``None`` instead reaches
   ``QnnConv2DCanonicalize``, whose four-term expansion keeps uint8 x int8 and
   folds the zero-points into the bias. Measured: output **bit-identical**
   (top1 258, logit 12.017338), params 23.5 -> 11.9 MB, local ELF 24.3 -> 12.6
   MB, ~7% faster -- against 30 -> 120 graph nodes, 28 -> 59 kernels, and 0.1
   -> 5.1 MB of int32 scratch for the zero-point reduction terms. Not wired to
   a flag.

4. **Codegen** — ``relay.build(target="c")`` and write the C source + header,
   then ``gcc -fsyntax-only`` it (TVM can emit C that does not compile).

5. **Split** — one folder per layer under ``layers/``, in graph execution
   order, so each layer has a working directory for its AIE artifacts
   (``split_layers.py``).

6. **ARM** — link a baremetal aarch64 ``main.elf`` for the board
   (``arm_build.py``). The generated C is kernels only -- no ``main``, no
   weights in memory, nothing calling the kernels in order -- so that module
   generates the graph driver, a small TVM runtime shim, and an entry point,
   then links them with the cortexa78 BSP using the same recipe as
   ``script/hostcompile.sh``. Needs ``source script/setup.sh --path-set-only``
   for the cross toolchain; without it the stage reports and is skipped.

   The image the ELF classifies is baked in as ``input_image.h``: a real photo
   (the pytorch/hub dog by default, ``--image`` for another) run through the
   standard ImageNet preprocessing, plus ``imagenet_labels.h`` so the board
   prints "Samoyed" rather than "class 258". Preprocessing is *called*, not
   reimplemented -- ``example/model/resnet18py/classify.py:preprocess`` -- so
   the board and the CPU reference cannot disagree over normalization.

   **``--local``** links the *same* generated C into ``main_local.elf`` for
   this host instead (``make local``) and runs it, so the top-5 can be diffed
   against stage 7 in seconds rather than by flashing a board. It needs no
   Vitis toolchain -- it composes with ``--no-arm`` -- and it is not the cross
   build with a different ``-mcpu``: the BSP and the baremetal link recipe are
   dropped, and two flags have to be *added*, ``-D__AIESIM__`` (so
   ``aie_timer.h`` takes its portable ``clock_gettime`` branch rather than
   including the BSP's ``xtime_l.h``) and ``-std=gnu11`` instead of
   ``-std=c11`` (``CLOCK_MONOTONIC`` is POSIX, which strict ISO mode hides).
   An AIE-offloaded build cannot be linked here and says so.

   **``make TRACE=1``** compiles in the generated driver's per-node
   instrumentation, which this stage emits behind ``#ifdef GRAPH_TRACE`` on
   every build -- a normal build carries none of it. Two dumps per tensor:
   the first few DECODED elements to the console, and **every kernel input and
   output in full** as a compilable C header of hex bytes, one file per tensor
   per call, into ``layeriohex/`` next to the running ELF (so ``--local`` fills
   ``arm_build/layeriohex/``). ``TRACE_HEX=0`` drops the headers,
   ``TRACE_HEX_MAX=N`` caps each at N bytes. Only the hosted build writes
   files; a baremetal board has no filesystem, so there the identical text
   streams over the console between ``===BEGIN layeriohex/...===`` markers and
   ``arm_build.split_layeriohex(log)`` cuts it back into the same files.
   Instrumenting a node therefore never means editing ``graph_driver.c`` --
   that file is rewritten on every run of this script.

   **AIE offload** (``--aie-offload``, off by default) puts the selected
   layers (``--aie-layers``, default 6 = the 7x7/s2 conv stem; it was 1 before
   int8 legalization split each conv into a zero-point chain and renumbered
   ``layers/``) on the AIE through
   TVM BYOC: the convs are partitioned out of the Relay graph, the C is
   rebuilt, and the graph executor calls the generated wrapper -> the aiehlc
   library ``aout/libconv2dstem.a`` in their place. The library is built
   separately (``source script/aiehlc.sh --aie-version 5 --runtime-source-file
   src/aietensorop/conv2dstem/conv2dstem.cc``); aiehlc generates its device
   init / mesh partition / launch. Offload is blind: no tile-budget check on
   this side. See ``aie_offload.py`` and ``byoc/``.

   **Whole-graph aiegraph** (``--aiegraph``) is the other way in: it lifts the
   *entire* graph into one ``aiegraph.func`` whose SSA edges are the model's
   real dataflow, then **partitions** it -- layers that are in ``--aie-layers``
   *and* whose aiegraph ops are all in ``--aiegraph-ops`` (default the conv2d
   family) go to the aiehlc kernel backend, and every other
   layer **reuses the TVM-generated CPU C** from stage 5 unchanged. Every layer
   gets a verdict and a reason in ``layers/partition.json``, including the ones
   the 4-op dialect cannot express (``max_pool2d``, ``batch_flatten``). The
   whole graph is lifted and verified regardless of the selection -- it only
   decides which verified layers get an ``aie/`` build, so ``--aie-layers``
   means the same thing here as under ``--aie-offload`` and both default to the
   conv stem at layer 6. See ``aiegraph_partition.py``.

7. **CPU** — classify the same image with onnxruntime on the same folded ONNX
   and print the top-5. This is the answer the ELF has to reproduce, from an
   **independent** implementation rather than a second call into the generated
   C, so matching means the compile path is right rather than self-consistent.
   The ELF prints the same five lines in the same format, logits included --
   a class index alone would hide the small numeric drift a miscompiled kernel
   actually produces.

Two host-specific fixups this flow has to make, both load-bearing:

``target_has_feature``
    ``topi.x86.dense_alter_op`` calls ``target_has_features(["avx512bw", ...])``
    during legalization, which reaches an FFI symbol that only exists in an
    LLVM-enabled build. Without a shim *any* ``relay.build(target="c")`` dies
    with ``AttributeError: module 'tvm.target._ffi_api' has no attribute
    'target_has_feature'`` — even for a two-line dense model, and even at
    ``opt_level=0``. ``_shim_target_features()`` answers False (no AVX-512),
    which is the right answer for a C-source target anyway.

``FoldConstant`` needs LLVM
    ``relay.quantize`` calls ``prerequisite_optimize`` ->  ``FoldConstant``,
    which *JIT-executes* constant subgraphs against a target hardcoded as
    ``Target eval_cpu_target_{"llvm"}`` (``src/relay/transforms/fold_constant.cc:403``)
    — not overridable from Python. On a ``USE_LLVM=OFF`` build that raises
    ``target.build.llvm is not enabled``. Disabling ``FoldConstant`` does not
    help: calibration then asserts on ``isinstance(expr.args[0], Constant)``
    (``_calibrate.py:154``), because it *requires* folded constants. So int8
    quantization is genuinely unavailable without LLVM, and this module reports
    that rather than pretending. ``llvm_status()`` is the single predicate.

Run::

    source script/setup.sh --path-set-only          # for stage 6's toolchain
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --image cat.jpg
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --skip-quantize
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --no-arm
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --local
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --local --no-arm
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aie-offload
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py \\
        --aie-offload --aie-layers 6      # 6 is the default: the conv stem
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --aiegraph
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py \\
        --aiegraph --aie-layers all --aiegraph-ops conv_bn_relu --no-arm
    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --relay-ptq
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Repo convention: .../src on sys.path, packages reached as frontend.<name>.
# Putting .../src/frontend on the path instead shadows the real `tvm` and `onnx`
# with the empty sibling directories of those names.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: Pretrained ImageNet ResNet-18, **v1**. The choice of v1 over v2 is
#: load-bearing on a no-LLVM build, not cosmetic -- see ``prefold_onnx``.
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "classification/resnet/model/resnet18-v1-7.onnx"
)
MODEL_NAME = "resnet18-v1-7.onnx"
FOLDED_NAME = "resnet18-v1-7.folded.onnx"
INPUT_NAME = "data"
INPUT_SHAPE = (1, 3, 224, 224)

#: Passes ``relay.build`` must not run on a no-LLVM build. ``FoldConstant``
#: JIT-executes constant subgraphs against a hardcoded ``"llvm"`` target
#: (``fold_constant.cc:403``); ``AlterOpLayout`` rewrites conv into x86 layouts
#: whose schedules then need the LLVM codegen. Both are optimizations, so
#: dropping them costs performance in the emitted C, not correctness.
NO_LLVM_DISABLED_PASSES = ["AlterOpLayout", "FoldConstant"]

DEFAULT_OUT = Path("./worklocal/tvmrelay_deploy")

#: Conv op kinds. ``--aie-ops`` filters ``--aie-offload``'s layer selection by
#: them (a selected conv still needs a matching AIE kernel -- ``byoc/aie_patterns``
#: decides that), and the ``--aiegraph`` partition is decided against the same
#: list, where each op also needs a ``frontend.tvmrelay.kernels`` body.
AIE_OP_KINDS = ("conv_bn_relu", "conv_bn")


# ═══════════════════════════════════════════════════════════════════════════
#  Host fixups
# ═══════════════════════════════════════════════════════════════════════════

def _shim_target_features() -> list:
    """Supply the LLVM-only FFI probes x86 legalization calls. Returns names added.

    TVM's x86 legalization asks two questions that only an LLVM-enabled build
    can answer, and both are fatal on a ``USE_LLVM=OFF`` build even when the
    target is ``"c"``:

    * ``target_has_feature`` — ``topi.x86.dense_alter_op`` probing for AVX-512
      (``AttributeError`` otherwise, even at ``opt_level=0``);
    * ``llvm_version_major`` — ``topi.x86.conv2d_int8.is_int8_hw_support``
      (``RuntimeError: LLVM version is not available`` otherwise).

    Both are shimmed to "this host has no LLVM vector ISA", which is the
    correct answer when emitting portable C: it steers legalization away from
    AVX-512-specific int8 rewrites rather than faking support for them.
    """
    from tvm.target import _ffi_api

    added = []
    if not hasattr(_ffi_api, "target_has_feature"):
        _ffi_api.target_has_feature = lambda feature, target=None: False
        added.append("target_has_feature")
    if not hasattr(_ffi_api, "llvm_version_major"):
        # 0 => "older than every version gate", so int8 HW paths stay off.
        _ffi_api.llvm_version_major = lambda: 0
        added.append("llvm_version_major")
    return added


def llvm_status() -> tuple:
    """Return ``(has_llvm, detail)`` for the *running* TVM. Never raises.

    This is the gate for stage 3. It asks the build config rather than trying a
    quantize, so callers can report the situation before spending minutes.
    """
    try:
        import tvm

        cfg = {}
        try:
            cfg = dict(tvm.support.libinfo())
        except Exception:
            pass
        use_llvm = str(cfg.get("USE_LLVM", "")).strip()
        if use_llvm and use_llvm.upper() not in ("OFF", "FALSE", "0", ""):
            return True, f"USE_LLVM={use_llvm}"
        # Fall back to asking the codegen registry directly.
        enabled = bool(tvm.get_global_func("target.build.llvm", allow_missing=True))
        return enabled, (f"USE_LLVM={use_llvm or 'OFF'}"
                         f"{' but target.build.llvm is registered' if enabled else ''}")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_LLVM_HINT = """\
  int8 quantization needs an LLVM-enabled TVM. relay.quantize runs
  prerequisite_optimize -> FoldConstant, which JIT-executes constant
  subgraphs against a target hardcoded as "llvm" in C++
  (src/relay/transforms/fold_constant.cc:403), so it cannot be retargeted
  from Python. Disabling FoldConstant does not work either -- calibration
  asserts on isinstance(expr.args[0], Constant) (_calibrate.py:154).

  To enable it, rebuild against the host LLVM 19:

      python src/frontend/tvmrelay/setup_tvm016.py --yes \\
          --llvm /scratch/staff/huaj/llvm-project/build/bin/llvm-config

  LLVM links against zlib/zstd/libxml2. If the dev packages are missing and
  you have no sudo, point cmake at the runtime libs with locally-staged
  headers instead -- ``ZLIB_LIBRARY``/``ZLIB_INCLUDE_DIR`` and
  ``-DCMAKE_SHARED_LINKER_FLAGS=-L<stub>/lib`` -- using headers whose version
  matches the installed ``libz.so.N``/``libzstd.so.N``. libxml2 is only
  referenced by LLVM's Windows-manifest code, which is unreachable here, so a
  stub exporting its 16 symbols satisfies the link.

  ``setup_tvm016.apply_source_fixups`` handles the two source incompatibilities
  automatically (LLVM 19 removed ``StringRef::startswith``; NumPy 2.0 removed
  ``np.math``, which breaks quantize calibration).
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — environment
# ═══════════════════════════════════════════════════════════════════════════

def stage_env(verbose: bool = True) -> bool:
    """Guarantee TVM 0.16 + Relay + the ``onnx.mapping`` shim are in place.

    Importing ``frontend.tvmrelay`` is what does the work: its ``__init__``
    calls ``ensure_tvm016()`` (auto-installing TVM 0.16 if absent -- disable
    with ``AIEHLC_TVM_AUTO_INSTALL=0``) and then ``ensure_onnx_mapping()``.
    """
    import frontend.tvmrelay as tvmrelay  # noqa: F401  (side effects are the point)

    ok, detail = tvmrelay.tvm016_status()
    if not ok:
        print(f"[1/7] TVM 0.16 unavailable: {detail}", file=sys.stderr)
        return False

    shimmed = _shim_target_features()
    has_llvm, llvm_detail = llvm_status()
    if verbose:
        import onnx

        print(f"[1/7] env    : tvm {detail} | onnx {onnx.__version__} | "
              f"llvm {'yes' if has_llvm else 'no'} ({llvm_detail})")
        if shimmed:
            print(f"[1/7]          shimmed {', '.join(shimmed)} (no-LLVM build)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 — download + Relay import
# ═══════════════════════════════════════════════════════════════════════════

def fetch_model(out_dir: Path, url: str = MODEL_URL, verbose: bool = True) -> Path:
    """Download the ONNX model into ``out_dir``, reusing a cached copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MODEL_NAME
    if path.exists() and path.stat().st_size > 0:
        if verbose:
            print(f"[2/7] model  : cached {path} ({path.stat().st_size:,} B)")
        return path
    if verbose:
        print(f"[2/7] model  : downloading {url}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    if verbose:
        print(f"[2/7]          saved {path} ({path.stat().st_size:,} B)")
    return path


def prefold_onnx(model_path: Path, verbose: bool = True) -> Path:
    """Fold BatchNorm into Conv **in ONNX**, before Relay ever sees the graph.

    This exists because of the no-LLVM constraint. TVM's own BN removal
    (``SimplifyInference``) rewrites BN into arithmetic containing ``sqrt`` over
    the variance *constants*, and leaves evaluating them to ``FoldConstant`` --
    which needs LLVM. With ``FoldConstant`` disabled those survive into codegen
    and it dies with ``Unresolved call Op(tir.sqrt)``. Folding in ONNX first
    means there is no BN, hence no ``sqrt``, hence nothing to constant-fold.

    onnxruntime's BASIC optimization level is used deliberately: EXTENDED emits
    ``FusedConv``, a Microsoft-domain op the Relay importer does not know.
    BASIC keeps the graph to standard ONNX ops.

    **This only works on ResNet-v1.** v1 is post-activation (Conv -> BN), so BN
    folds into the preceding conv's weights and all 19 BN nodes disappear. v2 is
    pre-activation (BN -> Conv); there is no preceding conv to fold into, 10 BN
    nodes survive BASIC folding, and the ``sqrt`` problem comes back. That is
    why ``MODEL_URL`` points at v1.
    """
    folded = model_path.with_name(FOLDED_NAME)
    if folded.exists() and folded.stat().st_size > 0:
        if verbose:
            print(f"[2/7] fold   : cached {folded.name}")
        return folded

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    if verbose:
        import onnx

        counts = {}
        for node in onnx.load(str(folded)).graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        kept = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[2/7] fold   : BN folded into Conv -> {kept}")
    return folded


def import_relay(model_path: Path, verbose: bool = True):
    """ONNX -> Relay. Returns ``(mod, params)``.

    ``params`` comes back **empty**: this model's 99 initializers are imported
    as inline ``relay.Constant`` nodes rather than free variables, and stay that
    way under ``freeze_params=True``. That is expected, not a failure -- but it
    means a weight-extracting walk must read the graph constants.
    """
    import onnx
    from tvm import relay

    model = onnx.load(str(model_path))
    mod, params = relay.frontend.from_onnx(
        model, shape={INPUT_NAME: INPUT_SHAPE}, freeze_params=True
    )
    mod = relay.transform.InferType()(mod)
    if verbose:
        print(f"[2/7] relay  : imported | params={len(params)} "
              f"(initializers are inline Constants) | {_op_summary(mod)}")
    return mod, params


def _op_summary(mod, top: int = 6) -> str:
    """Compact ``op=count`` summary of a Relay module's calls."""
    from tvm import relay

    counts = {}

    def visit(expr):
        if isinstance(expr, relay.expr.Call) and hasattr(expr.op, "name"):
            counts[expr.op.name] = counts.get(expr.op.name, 0) + 1

    relay.analysis.post_order_visit(mod["main"], visit)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return " ".join(f"{name}={n}" for name, n in ranked)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 — int8 quantization (gated on LLVM)
# ═══════════════════════════════════════════════════════════════════════════

def quantize_int8(mod, params, *, global_scale: float = 8.0,
                  skip_conv_layers=(0,), verbose: bool = True):
    """Quantize to int8 with ``relay.quantize``. Returns ``(mod, did_quantize)``.

    The ``--relay-ptq`` path, **not** the default -- stage 3 now quantizes in
    ONNX (``onnx_ptq.py``). Two limits keep this one optional: it is
    symmetric-only (its annotation op carries a scale and a clip range, no
    zero-point), and it leaves the first conv and the dense layer in fp32.

    Returns the module **unchanged** with ``did_quantize=False`` when the
    running TVM has no LLVM, because ``relay.quantize`` cannot run at all in
    that case (see the module docstring). Reporting that is the point -- a flow
    that silently emitted fp32 while claiming int8 would be worse than one that
    stops.

    ``global_scale`` calibration is used rather than KL-divergence because it
    needs no calibration dataset; ``skip_conv_layers=(0,)`` leaves the first
    conv in fp32, the usual accuracy-preserving convention.
    """
    from tvm import relay

    has_llvm, detail = llvm_status()
    if not has_llvm:
        if verbose:
            print(f"[3/7] int8   : SKIPPED -- this TVM has no LLVM ({detail})")
            print(_LLVM_HINT, end="")
            print("[3/7]          continuing in fp32; stage 4 output is NOT int8")
        return mod, False

    with relay.quantize.qconfig(
        calibrate_mode="global_scale",
        global_scale=global_scale,
        weight_scale="power2",
        skip_conv_layers=list(skip_conv_layers),
    ):
        qmod = relay.quantize.quantize(mod, params)
    if verbose:
        print(f"[3/7] int8   : quantized (global_scale={global_scale}, "
              f"skip_conv_layers={list(skip_conv_layers)}) | {_op_summary(qmod)}")
    return qmod, True


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 4 — C codegen
# ═══════════════════════════════════════════════════════════════════════════

def _collect_c_source(lib) -> str:
    """Concatenate the C source of a module tree, not just the root.

    ``lib.get_source()`` is enough for a plain ``target="c"`` build, but a BYOC
    build returns a COMPOSITE module: the TVM kernels and each offloaded
    subgraph's wrapper are separate child modules, and calling ``get_source()``
    on the root raises

        Module[const_loader] does not support GetSource

    So walk ``imported_modules`` and take every child whose ``type_key`` is
    ``"c"``. Without this the wrappers are silently missing from ``resnet18.c``
    and the link fails on an undefined ``tvmgen_default_aie_main_*``.
    """
    seen, parts = set(), []

    def walk(m):
        if id(m) in seen:
            return
        seen.add(id(m))
        if m.type_key == "c":
            try:
                parts.append(m.get_source())
            except Exception:  # a "c" module that cannot emit is not fatal
                pass
        for child in getattr(m, "imported_modules", []):
            walk(child)

    walk(lib)
    if not parts:  # non-composite build: the old path still applies
        return lib.get_source()
    return "\n".join(parts)


def build_c(mod, params, out_dir: Path, *, opt_level: int = 3, fuse: bool = True,
            target=None, verbose: bool = True) -> Path:
    """``relay.build(target="c")`` and write the C source. Returns its path.

    ``fuse=False`` gives one kernel per Relay op instead of per fused group --
    a real ``relu`` kernel, a real ``add`` kernel -- which is what stage 5
    needs to emit genuinely per-op layer files. It is slower code: fusion
    exists to keep intermediates in registers.

    Note it is **not** implemented as ``disabled_pass=["FuseOps"]``. FuseOps
    does two jobs -- grouping ops, and wrapping each group in a primitive
    ``Function`` -- and the graph executor codegen needs the second one
    unconditionally (``graph_executor_codegen.cc:452``:
    ``Expected the operator to be a global var, but got Op``). Instead we run
    ``FuseOps(fuse_opt_level=0)`` ourselves, which short-circuits at
    ``graph_partitioner.cc:104`` after ``InitGroups`` so every node is its own
    group, still wrapped -- then disable the build's own FuseOps so it cannot
    re-fuse what we just split.

    ``target="c"`` emits portable C rather than machine code, so it needs no
    LLVM -- with two caveats, both of which bite on a ``USE_LLVM=OFF`` build:

    * the FFI shims from stage 1 are required to get past x86 legalization;
    * ``target_host`` must be set to ``"c"`` explicitly. It defaults to
      ``llvm``, and the *host* module is built separately from the operator
      modules, so leaving it implicit fails in ``TIRToRuntime`` with
      ``target.build.llvm is not enabled`` even though every kernel compiled;
    * ``NO_LLVM_DISABLED_PASSES`` must be dropped (only when LLVM is absent --
      with LLVM they are kept, since they make the output faster);
    * ``tir.disable_vectorize`` is required for the C to *compile*. TVM's x86
      schedules vectorize, and the C backend renders a 16-lane int32 as the
      type ``int32_t16``, which is not a C type -- gcc rejects it with
      ``unknown type name 'int32_t16'``. The emitted file looks fine and is
      only caught by actually compiling it, so ``verify_c`` does.
    """
    import tvm
    from tvm import relay

    # ``target`` overrides the default C target -- ``pe_target.pe_target()``
    # passes one carrying an extra key, which is how the int8 QNN legalization
    # is selected without disturbing the ``cpu`` fallbacks (``--pe-int8``).
    if target is None:
        target = tvm.target.Target("c", host="c")

    out_dir.mkdir(parents=True, exist_ok=True)
    disabled = [] if llvm_status()[0] else list(NO_LLVM_DISABLED_PASSES)
    if not fuse:
        with tvm.transform.PassContext(opt_level=opt_level,
                                       disabled_pass=disabled):
            mod = relay.transform.InferType()(mod)
            mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
        disabled.append("FuseOps")
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled,
                                   config={"tir.disable_vectorize": True}):
        lib = relay.build(mod, target=target, params=params)

    source = _collect_c_source(lib.lib)
    c_path = out_dir / "resnet18.c"
    c_path.write_text(source)

    graph_path = out_dir / "resnet18_graph.json"
    graph_path.write_text(lib.get_graph_json())

    params_path = out_dir / "resnet18_params.bin"
    params_path.write_bytes(relay.save_param_dict(lib.get_params()))

    if verbose:
        # Count definitions, not occurrences of the signature: every kernel is
        # named twice, once forward-declared and once defined, so a plain
        # `source.count(...)` reports exactly double.
        from frontend.tvmrelay.split_layers import iter_c_functions
        n_fns = sum(1 for _ in iter_c_functions(source))
        print(f"[4/7] codegen: {c_path} ({len(source):,} chars, {n_fns} kernels)")
        print(f"[4/7]          {graph_path.name}, {params_path.name} "
              f"({params_path.stat().st_size:,} B)")
    return c_path


def verify_c(c_path: Path, verbose: bool = True) -> bool:
    """Syntax-check the emitted C with the host compiler.

    Worth doing as a step rather than trusting the file size: TVM will happily
    emit C containing vector types like ``int32_t16`` that no C compiler
    accepts (see ``build_c``). That output *looks* correct -- right length,
    right kernel count -- and only fails when someone finally compiles it.
    """
    import shutil
    import subprocess

    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        if verbose:
            print("[4/7] verify : skipped (no gcc/cc on PATH)")
        return True

    tvm_root = Path(__file__).resolve().parents[3] / "thirdparty" / "tvm-0.16"
    cmd = [cc, "-fsyntax-only",
           "-I", str(tvm_root / "include"),
           "-I", str(tvm_root / "3rdparty" / "dlpack" / "include"),
           str(c_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["(no output)"])[0]
        print(f"[4/7] verify : FAILED -- emitted C does not compile: {first}",
              file=sys.stderr)
        return False
    if verbose:
        print(f"[4/7] verify : {c_path.name} compiles ({cc} -fsyntax-only)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  Stage 7 — CPU reference classification
# ═══════════════════════════════════════════════════════════════════════════

def cpu_reference(out_dir: Path, image=None, topk: int = 5,
                  verbose: bool = True):
    """Classify the same image on the CPU and print the top-k. Returns it.

    This is the answer the board ELF has to reproduce. It is deliberately an
    **independent** implementation -- onnxruntime on the same folded ONNX, not
    a second call into the generated C -- so agreeing means the compile path
    is right, rather than only self-consistent.

    Compare it against the ELF by eye, or diff the printed logits: the ELF
    prints the same five lines in the same format. A class index alone would
    hide a small numeric drift, and drift is what a miscompiled kernel
    produces.
    """
    onnx_path = Path(out_dir) / FOLDED_NAME
    if not onnx_path.is_file():
        if verbose:
            print(f"[7/7] cpu    : skipped -- {onnx_path.name} not found")
        return None
    try:
        from frontend.tvmrelay import image_input

        tensor, path = image_input.preprocess_image(image, verbose=False)
        labels = image_input.load_labels(verbose=False)
        top = image_input.classify_reference(tensor, onnx_path, labels,
                                             topk=topk)
    except Exception as exc:                     # torch/PIL/onnxruntime absent
        if verbose:
            print(f"[7/7] cpu    : skipped -- {type(exc).__name__}: {exc}")
        return None

    if verbose:
        print(f"[7/7] cpu    : onnxruntime reference on {Path(path).name}")
        for rank, idx, name, logit in top:
            print(f"               {rank}. class={idx:<4d} {name:<30s} "
                  f"logit={logit:.4f}")
        print(f"[7/7]          the ELF prints these same lines; "
              f"top1 must be class {top[0][1]} ({top[0][2]})")
    return top


def _stage_split(c_path, out_dir: Path, *, split: bool, compiles: bool,
                 flat: bool, verbose: bool):
    """Stage 5: one folder per layer, in execution order. Returns the result or None."""
    if not split:
        if verbose:
            print("[5/7] split  : skipped (--no-split)")
        return None
    if not compiles:
        # Splitting C that does not compile just multiplies the broken file.
        if verbose:
            print("[5/7] split  : skipped -- the emitted C does not compile")
        return None
    from frontend.tvmrelay.split_layers import split_layers

    layers = split_layers(c_path, out_dir=out_dir / "layers",
                          group=not flat, verbose=False)
    if verbose:
        check = layers.get("syntax_check")
        note = ("all compile" if check == "pass"
                else f"{len(check)} FAILED" if isinstance(check, list)
                else str(check))
        print(f"[5/7] split  : {layers['layer_count']} layers -> "
              f"{out_dir / 'layers'}/ ({note})")
        key = "path" if flat else "dir"
        shown = [f"{e[key]}{'' if flat else '/'}" for e in layers["layers"][:4]]
        print(f"[5/7]          {', '.join(shown)}, ... "
              f"({layers['layer_count']} in execution order)")
        # Folders from an earlier, differently-shaped graph survive on disk
        # (see split_layers._clear_stale). Unreported, two builds leave two
        # `01_*` folders and `ls layers/` disagrees with the manifest.
        stale = layers.get("stale_dirs") or []
        if stale:
            print(f"[5/7]          NOTE {len(stale)} stale folder(s) from an "
                  f"earlier build are still on disk and are NOT this graph: "
                  f"{', '.join(stale[:2])}"
                  f"{', ...' if len(stale) > 2 else ''}")
    return layers


def _stage_aie_offload(mod, params, out_dir: Path, aie_layers, aie_ops, *,
                       layers_ok: bool, fuse: bool, target=None,
                       verbose: bool) -> dict:
    """Stage 6a: offload the selected layers to AIE through TVM BYOC.

    1. Layer indices -> conv weight fingerprints, from the CPU build just
       written (``aie_offload.resolve_conv_targets``).
    2. ``partition_for_aie`` on the pre-build Relay module. Each target must be
       exactly one Relay conv with the same geometry (``check_targets``), or
       nothing is offloaded -- the wrong conv on the AIE computes garbage.
    3. Rebuild the C. Returns ``{"count", "c_path", ...}``; ``count == 0``
       means the CPU build stands unchanged.
    """
    from frontend.tvmrelay import aie_offload as _aie
    from frontend.tvmrelay.byoc import aie_byoc
    from frontend.tvmrelay.byoc.aie_codegen import AIE_ENTRY

    if not layers_ok:
        if verbose:
            print("[6/7] aie    : skipped -- needs stage 5's layers/ to resolve "
                  "--aie-layers")
        return {"count": 0, "reason": "no layers/"}
    res = _aie.resolve_conv_targets(out_dir, aie_layers, tuple(aie_ops))
    if not res.get("ok"):
        if verbose:
            print(f"[6/7] aie    : skipped -- {res['reason']}")
        return {"count": 0, **res}
    targets = res["targets"]
    if verbose:
        sel = "all" if aie_layers is None else ",".join(str(i) for i in aie_layers)
        print(f"[6/7] aie    : offloading layer(s) {sel} to AIE via BYOC")
        for s in res["selections"]:
            if not s.get("eligible"):
                print(f"  [aie] layer {s['index']:02d} {s['kind']}: skipped -- "
                      f"{s['reason']}")
            else:
                print(f"  [aie] layer {s['index']:02d} {s['kind']}: {s['H']}x"
                      f"{s['W']}x{s['Cin']} -> {s['Cout']}ch K{s['K']}s{s['stride']}")
    if not targets:
        return {"count": 0, **res}

    # The mapping, not just its keys: the layer number rides along into the IR as
    # aie.layer_id, so codegen and errors can name the --aie-layers argument.
    fp_layer = {t["fingerprint"]: layer for layer, t in targets.items()}
    pmod, info = aie_byoc.partition_for_aie(mod, params, convs=fp_layer,
                                            verbose=verbose)
    bad = _aie.check_targets(targets, info["convs"])
    if bad:
        if verbose:
            print("[6/7]          selected layers do not map onto the Relay graph "
                  "-- offloading NOTHING:")
            for line in bad[:5]:
                print(f"              {line}")
        return {"count": 0, "mapping_errors": bad, **res}
    if verbose:
        # Two different outcomes, and they used to look the same. "unmatched" =
        # the layer did not even fuse (the pattern does not describe its shape);
        # "rejected" = it fused, and the kernel said why it cannot run it.
        for fp, why in info.get("rejected", {}).items():
            print(f"  [aie] layer {fp_layer[fp]:02d}: fused, but no AIE kernel "
                  f"can run it -- {why}; stays on the APU")
        for fp in info["unmatched"]:
            print(f"  [aie] layer {fp_layer[fp]:02d}: does not match any AIE "
                  f"pattern -- stays on the APU (AIE_BYOC_DEBUG=1 for the reason)")
    if not info["count"]:
        return {"count": 0, **res, "byoc": info}

    if verbose:
        print(f"[6/7]          rebuilding the C with {info['count']} AIE "
              f"subgraph(s) (calls {AIE_ENTRY}() in aout/libconv2dstem.a)")
    c_path = build_c(pmod, None, out_dir, fuse=fuse, target=target,
                     verbose=verbose)
    return {**res, "count": info["count"], "functions": info["functions"],
            "c_path": c_path, "byoc": info}


def _stage_quantize(raw_path, model_path, out_dir: Path, *,
                    skip_quantize: bool, relay_ptq: bool,
                    global_scale: float, image, verbose: bool, calib=None):
    """Stage 3. Returns ``(mod, params, quantized, quantizer, accuracy)``.

    Four mutually exclusive paths, in precedence order: ``skip_quantize``
    (fp32), ``relay_ptq``, the no-LLVM fallback, and the ONNX-PTQ default.
    Lifted out of :func:`run` so each stays readable; the comments on the
    branches are the reason this is not a lookup table.
    """
    quantized = False
    accuracy = None
    quantizer = None
    if skip_quantize:
        # fp32: overrides both quantizers.
        mod, params = import_relay(model_path, verbose=verbose)
        if verbose:
            print("[3/7] int8   : skipped (--skip-quantize)")
    elif relay_ptq:
        mod, params = import_relay(model_path, verbose=verbose)
        mod, quantized = quantize_int8(mod, params,
                                       global_scale=global_scale,
                                       verbose=verbose)
        quantizer = "relay.quantize" if quantized else None
    elif not llvm_status()[0]:
        # DEGRADED PATH -- not the intended default. ONNX PTQ *is* the default
        # and works fine; it needs an LLVM-enabled TVM, which this one is not.
        #
        # Importing the QDQ graph is what breaks: relay.frontend.from_onnx
        # calls fold_constant() on EVERY node (onnx.py:6973 -> common.py:515),
        # and folding a q/dq node JIT-executes it against the hardcoded "llvm"
        # target, so the import dies with "target.build.llvm is not enabled".
        # It is unreachable from Python (disabled_pass does not apply inside
        # the frontend), and the fp32 model imports fine -- so the gate belongs
        # here, before we spend minutes quantizing something we cannot import.
        #
        # Rebuilding TVM with LLVM is the fix, and _LLVM_HINT gives the exact
        # command (including the stub-header workaround when the zlib/zstd/
        # libxml2 dev packages are missing). Falling back to fp32 only avoids
        # a crash; it does NOT produce the artifact this flow is meant to.
        if verbose:
            print(f"[3/7] int8   : SKIPPED -- this TVM has no LLVM "
                  f"({llvm_status()[1]})")
            print(_LLVM_HINT, end="")
            print("[3/7]          FALLING BACK to fp32 -- stage 4 output is "
                  "NOT int8. Rebuild TVM with LLVM (above) to get the "
                  "default int8 path.")
        mod, params = import_relay(model_path, verbose=verbose)
    else:
        # Default. Quantize in ONNX *before* Relay sees the graph, so the
        # import is already int8 -- including layer 0 and the input image,
        # which relay.quantize cannot reach. See onnx_ptq.py.
        from frontend.tvmrelay import onnx_ptq as _ptq

        # `calib` (a list) wins over the single ELF `image`: calibration wants
        # several representative inputs, the ELF classifies exactly one.
        qdq_path = _ptq.quantize_onnx_int8(raw_path, out_dir, images=(
            list(calib) if calib else [image] if image else None), verbose=verbose)
        accuracy = _ptq.compare_topk(model_path, qdq_path, image=image,
                                     verbose=verbose)
        mod, params = import_relay(qdq_path, verbose=verbose)
        # Required, not optional: without this the QDQ graph stays
        # dequantize->fp32 op->quantize and codegen emits float kernels over a
        # 46 MB fp32 param blob, while every accuracy check still passes.
        mod = _ptq.to_integer_ops(mod, verbose=verbose)
        quantized = True
        quantizer = "onnx_ptq"

    return mod, params, quantized, quantizer, accuracy


def run(out_dir: Path = DEFAULT_OUT, *, skip_quantize: bool = False,
        global_scale: float = 8.0, fuse: bool = True, split: bool = True,
        flat: bool = False, arm: bool = True, image=None,
        aie_offload: bool = False, aie_layers=(1,),   # keep in step with --aie-layers
        aie_ops=("conv_bn_relu", "conv_bn"), mesh=(2, 2),
        aiegraph: bool = False, aiegraph_ops=AIE_OP_KINDS,
        relay_ptq: bool = False, local: bool = False, local_run: bool = True,
        pe_int8=None, fold_qnn_zp: bool = True, aie_kernels=(),
        calib=None, verbose: bool = True) -> dict:
    """Run all seven stages. Returns a dict of what happened.

    ``aie_offload`` offloads ``aie_layers`` (stage-5 layer indices; ``None``
    means every eligible layer) to AIE through TVM BYOC: after the CPU build,
    the selected convs are partitioned out of the Relay graph and the C is
    rebuilt, so the graph executor calls the generated wrapper -> the aiehlc
    AIE library in place of TVM's own kernel (``_stage_aie_offload``). It also
    runs the ``aiegraph`` lift below over the same selection, which is what
    writes ``layers/NN_op/aie/`` -- BYOC itself writes no per-layer artifacts.

    ``aiegraph`` instead lifts the **whole** graph into one verified
    ``aiegraph.func`` and partitions it: layers in ``aie_layers`` whose aiegraph
    ops are all in ``aiegraph_ops`` are offloaded to the aiehlc kernel backend,
    and the rest reuse the TVM-generated CPU C. Also additive. ``aie_layers`` is
    the **same** selection ``aie_offload`` uses -- both default to layer 6, the
    7x7/s2 conv stem under the int8-legalized graph -- so the two ways in target
    the same conv instead of one offloading a layer and the other the graph.

    **Quantizer.** Stage 3 defaults to ONNX PTQ (``onnx_ptq.py``): the model is
    quantized *before* Relay sees it, giving symmetric per-channel int8 weights
    and **asymmetric** activations, and covering every conv including the first
    plus the input image itself. ``relay_ptq=True`` selects the older
    ``relay.quantize`` path instead -- symmetric-only, and it leaves an fp32
    head and tail (``skip_conv_layers=[0]``, ``skip_dense_layer=True``). See
    ``quantize_int8`` for what that costs and why it needs LLVM.
    ``skip_quantize`` overrides both and emits fp32.

    ``fold_qnn_zp`` (on; ``--no-fold-qnn-zp`` opts out) collapses each conv's
    all-equal per-channel kernel zero point to a rank-0 scalar so TVM's own
    four-term elision fires. Output-preserving, and it removes the dead
    ``cast_sum -> multiply -> avg_pool2d -> repeat_multiply`` chains: measured
    **120 -> 41 kernel calls** on int8 ResNet-18. See ``qnn_fold_zp.py``.

    ``local`` additionally links the same generated C into ``main_local.elf``
    for *this* host and runs it, so stage 6's output can be checked against
    stage 7's reference without a board. It needs no Vitis toolchain, so it
    composes with ``arm=False``; it cannot link an AIE-offloaded build.

    ``calib`` is a list of calibration images (paths/URLs) for the ONNX PTQ
    quantizer; default is ``[image]``. ``image`` stays the one the ELF
    classifies. Used by the ``aiegraph`` facade; the CLI does not expose it.

    ``aie_kernels`` (``--aie-kernel SRC[@layers]``) offloads layers with USER
    kernels (``include/aie_user_kernel.h``, ``frontend.tvmrelay.aiekernel``):
    after the split (and after ``aie_offload``/``aiegraph``, so it gets the last
    word in ``partition.json``) each source is linted, matched against every
    layer, calibrated against TVM's own kernel and emulated across the whole
    network on x86, then built by aiehlc. Stage 6 then swaps the call exactly
    as it does for the stem. One aiehlc library per ELF: a user kernel that
    builds displaces an ``--aie-offload`` stem library (the runtime has one
    ``routing()``). See ``doc/tutorial_userkernel.md``.
    """
    if not stage_env(verbose=verbose):
        return {"ok": False, "stage": "env"}

    # Declare the PE's operand width BEFORE any build: the rule is selected
    # by a target key, so it must be registered before relay.build runs.
    #
    # `pe_int8=None` means "default: on", for every path including the AIE ones.
    #
    # This used to auto-disable under --aie-offload / --aiegraph, because int8
    # legalization restructures the graph (30 -> 120 nodes) and renumbers
    # `layers/` -- the ResNet stem moves from index 1 to 6 -- while
    # `--aie-layers` was read as a position into the Relay graph. That guard is
    # obsolete: selection now resolves a layer to its conv by WEIGHT FINGERPRINT
    # (`byoc.aie_annotate.weight_fingerprint` -- sorted values hashed as int64),
    # which is layout- and dtype-independent by construction, and
    # `check_targets` demands exactly one Relay conv matching both the
    # fingerprint and the geometry before anything is offloaded. Verified: the
    # same weights hash identically as int8/NCHW, int16/NCHW and int16/NCHWc.
    # `--aiegraph` now takes the same `--aie-layers` selection (it used to take
    # none), so both ways in target the same conv: layer 1, the stem.
    #
    # Keeping it off was also the wrong default on its own terms: the AIE kernel
    # consumes int8, so legalizing to int16 meant the offloaded conv's operands
    # were the one thing NOT in the width the hardware wants.
    #
    # `--aie-layers` indices DO still move with any pass that adds or removes
    # graph nodes -- the stem has been index 1 (fp32), then 6 (int8 legalized),
    # then 1 again (int8 + the qnn zero-point fold, today's default). That was
    # never the thing protecting correctness: a wrong index that lands on a
    # non-conv reports "not a conv -- only convs offload" and offloads nothing.
    # It does NOT protect against landing on a DIFFERENT conv, which resolves
    # fine and then matches no pattern -- check the geometry the stage prints.
    #
    # `--aiegraph` maps layers by fused op list (`conv2d` present -> conv_bn /
    # conv_bn_relu). It used to lift 0 convs here, not because of the names but
    # because the int8 graph is 5-D NCHWc and its geometry reader took 4-D only
    # ("non-4D shapes; not a conv2d"); it now folds NCHWc via
    # aie_offload._layer_geometry, so all 20 convs lift and layer 1 (the stem,
    # 230x230x3 -> 64ch K7s2) gets an aie/ build.
    if pe_int8 is None:
        pe_int8 = True
    target = None
    if pe_int8:
        from frontend.tvmrelay import pe_target as _pe

        target = _pe.enable(verbose=verbose)

    raw_path = fetch_model(out_dir, verbose=verbose)
    model_path = prefold_onnx(raw_path, verbose=verbose)

    (mod, params, quantized, quantizer,
     accuracy) = _stage_quantize(raw_path, model_path, out_dir,
                                 skip_quantize=skip_quantize,
                                 relay_ptq=relay_ptq,
                                 global_scale=global_scale,
                                 image=image, verbose=verbose, calib=calib)

    # Before build_c, and before _stage_aie_offload's partition_for_aie: both
    # run qnn.CanonicalizeOps internally, and once that has expanded the four
    # terms there is nothing left to fold.
    if fold_qnn_zp:
        from frontend.tvmrelay import qnn_fold_zp

        mod = qnn_fold_zp.fold_scalar_zero_points(mod, verbose=verbose)

    c_path = build_c(mod, params, out_dir, fuse=fuse, target=target,
                     verbose=verbose)
    compiles = verify_c(c_path, verbose=verbose)

    layers = _stage_split(c_path, out_dir, split=split, compiles=compiles,
                          flat=flat, verbose=verbose)

    # AIE offload via BYOC: partition the selected convs out of the Relay
    # graph and rebuild, so main.elf calls the AIE library for them.
    aie = None
    if aie_offload:
        aie = _stage_aie_offload(mod, params, out_dir, aie_layers, aie_ops,
                                 layers_ok=layers is not None, fuse=fuse,
                                 target=target, verbose=verbose)
        if aie.get("count"):
            c_path = aie["c_path"]
            compiles = verify_c(c_path, verbose=verbose)
            layers = _stage_split(c_path, out_dir, split=split,
                                  compiles=compiles, flat=flat, verbose=verbose)

    # Whole-graph aiegraph + AIE/CPU partition. Also additive: CPU layers keep
    # reusing the stage-5 C, and AIE layers keep theirs too so the ELF links.
    #
    # --aie-offload runs it too, over the same --aie-layers selection: BYOC
    # above only swaps the call in resnet18.c and writes no per-layer AIE
    # artifacts, while this lifts the selected conv into aiegraph and emits
    # layers/NN_op/aie/ (host.cc/kernel.cc/routing.cc). Every layer's .c is left
    # exactly as stage 5 wrote it, so layers/ still matches the default flow.
    graph_part = None
    if aiegraph or aie_offload:
        if layers is None:
            if verbose:
                print("[6/7] aiegrph: skipped -- needs stage 5's layers/")
        else:
            from frontend.tvmrelay import aiegraph_partition

            if verbose:
                sel = ("all" if aie_layers is None
                       else ",".join(str(i) for i in aie_layers))
                print(f"[6/7] aiegrph: lifting the whole graph, offloading "
                      f"layer(s) {sel} (AIE ops: {', '.join(aiegraph_ops)})")
            try:
                graph_part = aiegraph_partition.run_aiegraph(
                    out_dir, aie_ops=tuple(aiegraph_ops), mesh=mesh,
                    layers=aie_layers, verbose=verbose)
            except RuntimeError as exc:
                # Under --aiegraph this IS the stage, so fail loudly. Under
                # --aie-offload alone it is an additive extra: before it ran
                # here, `--aie-offload --no-fold-qnn-zp` worked, and that graph
                # makes the whole-graph lift fail verification (its unfolded
                # conv names carry two `add`s, read as a residual). Report and
                # keep the CPU build instead of losing the whole run.
                if aiegraph:
                    raise
                graph_part = {"ok": False, "reason": f"aiegraph lift failed, "
                              f"no aie/ built: {exc}"}
            if verbose and not graph_part.get("ok"):
                print(f"[6/7]          {graph_part.get('reason', 'partition failed')}")

    # User kernels (--aie-kernel). After the two paths above so its verdicts
    # are the last write to partition.json, and before stage 6 so the driver
    # stage 6 generates already swaps what it built. regenerate=False: stage 6
    # below is the regeneration.
    user_kernels = None
    if aie_kernels:
        if layers is None:
            if verbose:
                print("[6/7] aiek   : skipped -- needs stage 5's layers/")
        else:
            from frontend.tvmrelay import aiekernel

            if verbose:
                print(f"[6/7] aiek   : {len(aie_kernels)} user kernel(s): "
                      f"{', '.join(str(k) for k in aie_kernels)}")
            user_kernels = aiekernel.offload(out_dir, list(aie_kernels),
                                             regenerate=False, verbose=verbose)

    # Written after every graph-changing stage above (offload and aiegraph
    # both rebuild the C), so it always describes the graph that was finally
    # compiled rather than the first one.
    from frontend.tvmrelay.network_md import write_network_md

    network_md = write_network_md(out_dir, c_name=Path(c_path).name,
                                  verbose=verbose)

    elf = None
    local_res = None
    if not arm and not local:
        if verbose:
            print("[6/7] arm    : skipped (--no-arm)")
    elif not compiles:
        if verbose:
            print("[6/7] arm    : skipped -- the emitted C does not compile")
    else:
        from frontend.tvmrelay.arm_build import build_arm_elf

        # Generation is shared: --local --no-arm still writes the driver, the
        # image header and the Makefile, then builds only the host ELF.
        if verbose:
            print("[6/7] arm    : " + ("linking baremetal aarch64 ELF" if arm
                                       else "generating sources (--no-arm)"))
        built = build_arm_elf(out_dir, c_name=Path(c_path).name,
                              image=image, run_make=arm, local=local,
                              local_run=local_run, verbose=verbose)
        if built.get("ok"):
            elf = built["elf"]
        elif verbose and built.get("stderr"):
            print(f"[6/7]          {built['reason']}:")
            for line in built["stderr"].strip().splitlines()[:5]:
                print(f"              {line}")
        local_res = built.get("local")

    cpu = cpu_reference(out_dir, image=image, verbose=verbose)

    if verbose:
        kind = "int8" if quantized else "fp32"
        print(f"\ndone: {kind} C in {out_dir}"
              f"{'' if compiles else '  (WARNING: does not compile)'}")
        if elf:
            print(f"      board ELF: {elf}")
        if local_res and local_res.get("ok"):
            print(f"      local ELF: {local_res['elf']}")
        if network_md:
            print(f"      network  : {network_md}")
        if graph_part and graph_part.get("aie_count") is not None:
            print(f"      aiegraph : {graph_part['aie_count']} invocation(s) on "
                  f"AIE, {graph_part['cpu_count']} reusing TVM CPU C "
                  f"(layers/partition.json)")
        _report_local_vs_cpu(local_res, cpu)
    return {"ok": compiles, "quantized": quantized, "c_path": str(c_path),
            "out_dir": str(out_dir), "fused": fuse,
            "layer_count": layers["layer_count"] if layers else None,
            "elf": elf, "cpu_top1": (cpu[0][1] if cpu else None),
            "cpu_top": cpu, "aie": aie, "aiegraph": graph_part,
            "user_kernels": user_kernels,
            "local": local_res, "network_md": str(network_md) if network_md
            else None,
            "quantizer": quantizer, "accuracy": accuracy}


def _report_local_vs_cpu(local_res, cpu) -> None:
    """Say whether the local ELF agreed with stage 7's onnxruntime reference.

    This is the reason ``--local`` exists: the board ELF and the reference are
    normally only comparable by flashing a board and reading a console. The
    same generated C, run here, answers it in seconds -- and a *disagreement*
    is the signal, since both sides classify the identical preprocessed image.
    """
    if not (local_res and local_res.get("ran") and cpu):
        return
    top1 = local_res.get("top1")
    if not top1 or top1[0] is None:
        return
    cls, logit = top1
    ref_cls, ref_name, ref_logit = cpu[0][1], cpu[0][2], cpu[0][3]
    if cls == ref_cls:
        print(f"      local vs cpu: MATCH class={cls} ({ref_name}) "
              f"logit {logit:.4f} vs {ref_logit:.4f}")
    else:
        print(f"      local vs cpu: MISMATCH -- local says class={cls} "
              f"logit={logit:.4f}, reference says class={ref_cls} "
              f"({ref_name}) logit={ref_logit:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="emit fp32 C without attempting stage 3")
    ap.add_argument("--global-scale", type=float, default=8.0,
                    help="relay.quantize global_scale (default: 8.0); only "
                         "used with --relay-ptq")
    ap.add_argument("--no-split", action="store_true",
                    help="skip stage 5 (do not write layers/)")
    ap.add_argument("--flat", action="store_true",
                    help="stage 5 writes one flat layers/ directory of .c "
                         "files instead of a folder per layer")
    ap.add_argument("--no-fuse", action="store_true",
                    help="disable FuseOps, so stage 5 emits one file per Relay "
                         "op (a real relu.c) instead of per fused group")
    ap.add_argument("--no-arm", action="store_true",
                    help="skip stage 6 (do not link the aarch64 board ELF)")
    ap.add_argument("--local", action="store_true",
                    help="also build main_local.elf for THIS host (x86) from "
                         "the same generated C and run it, printing the same "
                         "top-5 the board would. Needs no Vitis toolchain, so "
                         "it combines with --no-arm; cannot link an "
                         "AIE-offloaded build")
    ap.add_argument("--no-local-run", action="store_true",
                    help="with --local, link main_local.elf but do not run it")
    ap.add_argument("--no-pe-int8", action="store_true",
                    help="store conv operands as int16 instead of int8. The "
                         "int8 PE legalization is ON by default; this falls "
                         "back to TVM's x86 rule, which widens data and "
                         "weights to int16 (weights.bin 11.9 -> 23.5 MB, ELF "
                         "12.8 -> 24.4 MB) for the same bit-identical output. "
                         "Useful for A/B-ing against the old artifacts")
    ap.add_argument("--pe-int8", action="store_true",
                    help="force the int8 PE legalization ON. It is already the "
                         "default on every path, AIE ones included -- this flag "
                         "only makes that explicit, e.g. to override an earlier "
                         "--no-pe-int8 in a wrapper script")
    ap.add_argument("--no-fold-qnn-zp", action="store_true",
                    help="keep qnn's dead zero-point correction chains. They "
                         "are emitted because TVM's four-term elision tests "
                         "whether the kernel zero point is a SCALAR, and "
                         "per-channel quantization makes it a [64] tensor -- "
                         "even when every element is 0. Folding it to a scalar "
                         "takes int8 ResNet-18 from 120 to 41 kernel calls for "
                         "bit-identical output; this flag turns that off to "
                         "A/B against the old graph")
    ap.add_argument("--image", default=None,
                    help="image path or URL to classify "
                         "(default: the pytorch/hub dog.jpg sample)")
    ap.add_argument("--aie-offload", action="store_true",
                    help="offload the --aie-layers to AIE through TVM BYOC: "
                         "the generated C calls the aiehlc AIE library "
                         "(aout/libconv2dstem.a) in place of TVM's kernel. "
                         "Only the ResNet-18 stem has an AIE kernel today")
    ap.add_argument("--aie-layers", default="1",
                    help="which layers to offload, as numbered in layers/ by "
                         "the CPU build: an index, a comma list, or 'all'. "
                         "Applies to BOTH --aie-offload and --aiegraph "
                         "(default: 1, the 7x7/s2 conv stem. It moved 1 -> 6 "
                         "when int8 legalization split each conv into a "
                         "zero-point chain, then back to 1 when --fold-qnn-zp "
                         "removed those chains again; with --no-fold-qnn-zp it "
                         "is 6. Check layers/ if unsure)")
    ap.add_argument("--aie-ops", default=",".join(AIE_OP_KINDS),
                    help=f"op kinds eligible for AIE "
                         f"(default: {','.join(AIE_OP_KINDS)})")
    ap.add_argument("--aie-mesh", default="2x2",
                    help="AIE mesh as ROWSxCOLS for --aiegraph (default: 2x2); "
                         "--aie-offload uses the mesh built into the AIE library")
    ap.add_argument("--aiegraph", action="store_true",
                    help="lift the WHOLE graph into one aiegraph.func, then "
                         "partition: layers in --aie-layers matching "
                         "--aiegraph-ops go to the aiehlc kernel backend, the "
                         "rest reuse the TVM CPU C (verdicts in "
                         "layers/partition.json)")
    ap.add_argument("--relay-ptq", action="store_true",
                    help="quantize with relay.quantize instead of the default "
                         "ONNX PTQ: symmetric-only, leaves an fp32 head and "
                         "tail (skips the first conv and the dense layer), and "
                         "needs an LLVM-enabled TVM. Smaller ELF today because "
                         "target=\"c\" has no int8 qnn legalization")
    ap.add_argument("--aiegraph-ops", default=",".join(AIE_OP_KINDS),
                    help=f"aiegraph op kinds to offload to AIE; a layer goes "
                         f"to AIE only if all of its ops are listed "
                         f"(default: {','.join(AIE_OP_KINDS)})")
    ap.add_argument("--aie-kernel", action="append", default=[], metavar="SRC[@L,..]",
                    help="offload layers with YOUR kernel (include/aie_user_kernel.h): "
                         "an aiehlc source with AIEK_ENTRY + AIEK_SPEC. @1,4 picks "
                         "layers, default every layer the spec matches. Gated by lint, "
                         "detect, x86 calibration and whole-network emulation before "
                         "the aiehlc build; repeatable. See doc/tutorial_userkernel.md")
    args = ap.parse_args(argv)

    from frontend.tvmrelay.aie_offload import parse_selection

    try:
        aie_layers = parse_selection(args.aie_layers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        rows, cols = (int(v) for v in args.aie_mesh.lower().split("x"))
    except ValueError:
        print(f"error: bad --aie-mesh {args.aie_mesh!r} (want ROWSxCOLS)",
              file=sys.stderr)
        return 2

    # Reject an unknown op kind up front. Silently ignoring it would read as
    # "that layer is not eligible" rather than "you misspelled the flag".
    from frontend.tvmrelay.aiegraph_partition import AIEGRAPH_OP_KINDS

    aiegraph_ops = tuple(s.strip() for s in args.aiegraph_ops.split(",") if s.strip())
    unknown = [op for op in aiegraph_ops if op not in AIEGRAPH_OP_KINDS]
    if unknown:
        print(f"error: unknown --aiegraph-ops {', '.join(unknown)} "
              f"(the dialect defines {', '.join(AIEGRAPH_OP_KINDS)})",
              file=sys.stderr)
        return 2

    result = run(args.out_dir, skip_quantize=args.skip_quantize,
                 global_scale=args.global_scale, fuse=not args.no_fuse,
                 split=not args.no_split, flat=args.flat,
                 arm=not args.no_arm, image=args.image,
                 aie_offload=args.aie_offload, aie_layers=aie_layers,
                 aie_ops=tuple(s.strip() for s in args.aie_ops.split(",")),
                 mesh=(rows, cols),
                 aiegraph=args.aiegraph, aiegraph_ops=aiegraph_ops,
                 relay_ptq=args.relay_ptq, local=args.local,
                 local_run=not args.no_local_run,
                 fold_qnn_zp=not args.no_fold_qnn_zp,
                 aie_kernels=tuple(args.aie_kernel),
                 # None = default: on, AIE stages included.
                 pe_int8=(False if args.no_pe_int8
                          else True if args.pe_int8 else None))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
