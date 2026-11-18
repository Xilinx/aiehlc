###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""int8 PTQ in **ONNX**, before Relay: symmetric weights, asymmetric activations.

The alternative to ``deploy_flow.quantize_int8`` (``relay.quantize``), and the
only one of the two that can quantize the *whole* network including layer 0 and
the input image.

Why not ``relay.quantize``
--------------------------
It is structurally symmetric-only. Its core annotation op is::

    relay.op.annotation.simulated_quantize(data, scale, clip_min, clip_max)

-- a scale and a clip range, and **no zero-point anywhere**: not in the op, not
in ``QConfig._node_defaults``, not in the C++ pass. So "asymmetric input" is not
a setting it has. It also skips the first conv by default
(``skip_conv_layers=[0]``) and the dense layer (``skip_dense_layer=True``),
leaving an fp32 head and tail.

What this does instead
----------------------
``onnxruntime.quantization.quantize_static`` in QDQ format:

* **weights — symmetric int8**, per-channel (one scale per output channel,
  zero-point 0). Verified: the first conv's weight ``DequantizeLinear`` carries
  ``zero_point=[0]`` and 64 scales.
* **activations — asymmetric uint8**, including the network input. Verified:
  the input ``QuantizeLinear`` gets ``scale=0.0186, zero_point=113``. A
  non-zero zero-point is the whole point -- post-ReLU activations are
  one-sided, so a symmetric scale wastes half the range.

TVM's Relay importer maps ``QuantizeLinear``/``DequantizeLinear`` onto
``qnn.op.quantize``/``qnn.op.dequantize`` **with the zero-point** (see
``relay/frontend/onnx.py``: ``_qnn.op.quantize(data, scale, cast(zp, "int32"),
axis, out_dtype)``), so the asymmetry survives the import rather than being
silently dropped.

The opset dance is load-bearing
-------------------------------
``quantize_static`` needs opset >= 11; a pretrained ONNX-zoo v1 CNN is opset 8.
Letting onnxruntime auto-upgrade the **already-BN-folded** model fails::

    IndexError: Input ConvBnFusion_W_<conv0>_weight is undefined!

because ONNX's version converter cannot resolve the synthetic initializer names
that ORT's fusion introduced. So the order here is **upgrade first, fold
second** -- convert the raw model to opset 13 while the names are still the
original ones, then fold BN. Same 20-conv graph, no BatchNorm, no failure.

Known limit: weights land as int16, not int8
--------------------------------------------
The emitted param blob is ~23 MB, not the ~12 MB int8 implies, because the
**generic** ``qnn_conv2d_legalize`` upcasts data and kernel to int16 so the
zero-points can be folded by subtraction
(``relay/qnn/op/legalizations.py:115``). Targets with a fast int8 path register
their own legalization (``"cpu"`` uses VNNI/uint8xint8, plus ``arm_cpu``,
``cuda``, ``hexagon``); ``target="c"`` registers none, so it takes the generic
int16 route.

The *values* are genuinely int8 -- all 22 weight tensors verify in
``[-127, 127]`` -- so this is a storage-width artifact, not a quantization
failure. Fixing it properly means registering a ``"c"`` legalization that keeps
int8 and handles the zero-point in the accumulator instead. Worth doing for the
AIE path, where int8 operands are the point; out of scope here.

Calibration
-----------
Activation ranges come from running real images through the graph. The default
is the demo image, which is enough to exercise the path end to end; a wider set
tightens the ranges. Accuracy is *measured*, not assumed -- see
``compare_topk``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["quantize_onnx_int8", "compare_topk", "to_integer_ops",
           "qdq_name", "opset_name"]

#: Opset the model is converted to before quantizing (>= 11 required).
TARGET_OPSET = 13


def opset_name(raw_onnx) -> str:
    """``<stem>.op13.folded.onnx`` -- the upgraded + BN-folded model's file name."""
    return f"{Path(raw_onnx).stem}.op{TARGET_OPSET}.folded.onnx"


def qdq_name(raw_onnx) -> str:
    """``<stem>.int8.qdq.onnx`` -- the quantized model's file name."""
    return f"{Path(raw_onnx).stem}.int8.qdq.onnx"


def _upgrade_and_fold(raw_onnx: Path, out_dir: Path, verbose: bool = True) -> Path:
    """Convert to opset 13 **then** fold BN into Conv. Returns the folded path.

    Order matters -- see the module docstring. Folding first produces
    ``ConvBnFusion_*`` initializer names that ONNX's version converter cannot
    resolve, and the upgrade dies with a bare ``IndexError``.
    """
    import onnx
    from onnx import version_converter
    import onnxruntime as ort

    folded = out_dir / opset_name(raw_onnx)
    if folded.is_file() and folded.stat().st_size > 0:
        if verbose:
            print(f"[3/7] ptq    : cached {folded.name}")
        return folded

    model = onnx.load(str(raw_onnx))
    if model.opset_import[0].version < 11:
        model = version_converter.convert_version(model, TARGET_OPSET)
    upgraded = out_dir / "_op13.tmp.onnx"
    onnx.save(model, str(upgraded))

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(upgraded), opts, providers=["CPUExecutionProvider"])
    upgraded.unlink(missing_ok=True)

    if verbose:
        from collections import Counter
        counts = Counter(n.op_type for n in onnx.load(str(folded)).graph.node)
        print(f"[3/7] ptq    : opset {TARGET_OPSET} + BN folded -> "
              f"{' '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")
    return folded


def _calibration_reader(images, input_name: str):
    """A ``CalibrationDataReader`` over the preprocessed demo image(s).

    Preprocessing is *called*, not reimplemented, so calibration sees exactly
    the tensors inference will.
    """
    from onnxruntime.quantization import CalibrationDataReader
    from frontend.tvmrelay import image_input

    tensors = []
    for img in (images or [None]):
        tensor, _path = image_input.preprocess_image(img, verbose=False)
        tensors.append({input_name: tensor.astype("float32")})

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(tensors)

        def get_next(self):
            return next(self._it, None)

    return _Reader(), len(tensors)


def quantize_onnx_int8(raw_onnx, out_dir, images=None, *,
                       per_channel: bool = True, input_name: str = "data",
                       verbose: bool = True) -> Path:
    """Quantize *raw_onnx* to an int8 QDQ model. Returns its path.

    Symmetric int8 weights (per-channel) + asymmetric uint8 activations,
    covering **every** conv including the first, and the input image itself.
    """
    from onnxruntime.quantization import (quantize_static, CalibrationMethod,
                                          QuantFormat, QuantType)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qdq = out_dir / qdq_name(raw_onnx)
    folded = _upgrade_and_fold(Path(raw_onnx), out_dir, verbose=verbose)
    if qdq.is_file() and qdq.stat().st_size > 0:
        if verbose:
            print(f"[3/7] ptq    : cached {qdq.name} ({qdq.stat().st_size:,} B)")
        return qdq

    reader, n_cal = _calibration_reader(images, input_name)
    quantize_static(
        str(folded), str(qdq), reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,       # symmetric, zero_point == 0
        activation_type=QuantType.QUInt8,  # asymmetric, non-zero zero_point
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    if verbose:
        print(f"[3/7] ptq    : int8 QDQ -> {qdq.name} "
              f"({qdq.stat().st_size:,} B, {n_cal} calibration image(s), "
              f"{'per-channel' if per_channel else 'per-tensor'} weights)")
        _report_scheme(qdq)
    return qdq


def _report_scheme(qdq: Path) -> None:
    """Print the evidence that weights are symmetric and activations are not.

    Reads the zero-points straight out of the model rather than restating the
    request: a silently symmetric "asymmetric" build is the failure worth
    catching, and it is invisible in the file size.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(qdq))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    quant = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    if quant and quant[0].input[2] in init:
        zp = init[quant[0].input[2]]
        scale = init[quant[0].input[1]]
        print(f"[3/7]          input: scale={float(scale):.6f} zero_point={int(zp)} "
              f"({zp.dtype}) -- asymmetric")

    wq = [n for n in model.graph.node
          if n.op_type == "DequantizeLinear" and n.input[0] in init
          and init[n.input[0]].dtype == np.int8]
    if wq and wq[0].input[2] in init:
        zp = np.unique(init[wq[0].input[2]])
        n_scales = init[wq[0].input[1]].size
        kind = "symmetric" if not zp.any() else "ASYMMETRIC (unexpected)"
        axis = "per-channel" if n_scales > 1 else "per-tensor"
        print(f"[3/7]          weights: zero_point={zp.tolist()} -- {kind}, "
              f"{n_scales} scale(s) ({axis})")


def to_integer_ops(mod, *, hard_fail: bool = False, verbose: bool = True):
    """Rewrite the imported QDQ graph into **real** integer ops. Returns the module.

    This step is not optional, and skipping it is a silent trap. A QDQ import
    is a *simulated* quantization: the graph is
    ``dequantize -> fp32 op -> quantize``, so the weights get dequantized back
    to fp32 constants, ``FoldConstant`` bakes them in, and codegen emits
    ``float*`` kernels over a 46 MB fp32 param blob. The model is numerically
    int8-accurate and not int8-anything-else -- accuracy checks pass, the
    artifact is full size, and nothing reports a problem.

    ``FakeQuantizationToInteger`` matches those ``dq -> op -> q`` regions and
    replaces them with ``qnn.conv2d``/``qnn.add``/``qnn.requantize``, which
    lower to genuine int8 arithmetic. On a 20-conv int8 CNN this turns 20 ``nn.conv2d``
    into 20 ``qnn.conv2d`` and leaves exactly **one** ``qnn.quantize`` -- the
    input image -- with its asymmetric zero-point intact.

    ``hard_fail=False`` leaves any region the pass cannot convert as fp32
    rather than raising; the op counts are reported so a partial conversion is
    visible instead of assumed.
    """
    from tvm import relay

    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FakeQuantizationToInteger(hard_fail=hard_fail)(mod)
    mod = relay.transform.InferType()(mod)
    if verbose:
        from collections import Counter

        counts = Counter()

        def visit(expr):
            if isinstance(expr, relay.expr.Call) and hasattr(expr.op, "name"):
                counts[expr.op.name] += 1

        relay.analysis.post_order_visit(mod["main"], visit)
        qnn = {k: v for k, v in counts.items() if k.startswith("qnn.")}
        left = counts.get("nn.conv2d", 0)
        print(f"[3/7] fq2i   : {counts.get('qnn.conv2d', 0)} qnn.conv2d, "
              f"{counts.get('qnn.requantize', 0)} requantize, "
              f"{counts.get('qnn.quantize', 0)} quantize (the input)"
              + (f" -- WARNING {left} conv(s) still fp32" if left else ""))
        print(f"[3/7]          qnn ops: {', '.join(sorted(qnn))}")
    return mod


def compare_topk(fp32_onnx, int8_onnx, image=None, topk: int = 5,
                 verbose: bool = True) -> dict:
    """Run both models on the same image and compare. Returns a verdict dict.

    Quantization that silently destroys accuracy still produces a valid,
    smaller model -- so this executes both and reports whether the top-1 and
    the top-k *ordering* survived, rather than trusting that it did.
    """
    import onnxruntime as ort
    from frontend.tvmrelay import image_input

    tensor, path = image_input.preprocess_image(image, verbose=False)
    labels = image_input.load_labels(verbose=False)

    def run(model_path):
        sess = ort.InferenceSession(str(model_path),
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name:
                              tensor.astype("float32")})[0].ravel()
        order = out.argsort()[-topk:][::-1]
        return [(int(i), labels[int(i)], float(out[int(i)])) for i in order]

    ref, got = run(fp32_onnx), run(int8_onnx)
    same_top1 = ref[0][0] == got[0][0]
    same_order = [r[0] for r in ref] == [g[0] for g in got]

    if verbose:
        print(f"[3/7] accuracy: int8 vs fp32 on {Path(path).name}")
        for rank, (r, g) in enumerate(zip(ref, got), 1):
            flag = "" if r[0] == g[0] else "   <-- differs"
            print(f"[3/7]          {rank}. fp32 {r[0]:<4d} {r[1]:<22s} {r[2]:8.4f} | "
                  f"int8 {g[0]:<4d} {g[1]:<22s} {g[2]:8.4f}{flag}")
        verdict = ("top-5 order identical" if same_order else
                   "top-1 preserved, lower ranks reordered" if same_top1 else
                   "TOP-1 CHANGED -- quantization lost the classification")
        print(f"[3/7]          {verdict}")
    return {"same_top1": same_top1, "same_order": same_order,
            "fp32": ref, "int8": got}
