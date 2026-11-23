###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Pattern table: which Relay subgraphs an AIE kernel can take.

The first of the four BYOC files. It declares that "a subgraph shaped like this
is called ``aie.qconv_fused``", and -- through ``extract_fused`` -- what every
constant in it means. Merging and partitioning are left to TVM.

The boundary is the FUSED op, not the bare conv
-----------------------------------------------
The AIE kernel (``conv2d_spatial`` in ``libconv2dstem.a``) implements conv +
bias + zero-point + per-channel requantize + ReLU + uint8 store in one pass,
and that is the only thing it can produce. So the partition must cover exactly
that chain; cutting at the bare ``nn.conv2d`` (the earlier design) asks the AIE
for an int32 accumulator it cannot emit, and the only way to honour that was a
scalar loop on the APU -- an "offload" that never touched the AIE.

Written against the real IR, after ``qnn.transform.CanonicalizeOps`` +
``FoldConstant`` (dumped from a PT2E/ONNX int8 CNN classifier graph)::

    %5  = nn.pad(%4, 113, pad_width=[..,[3,3],[3,3]])           # outside: zp-padded
    %11 = nn.conv2d(%5, W:int8[64,3,7,7], strides=2, out_dtype=int32)
    %10 = repeat(avg_pool2d(multiply(sum(cast(%5)), 49)), 64)   # sum(x) per window
    %13 = subtract(%11, multiply(C1:[64,1,1], %10))              # weight-zp term
    %14 = add(%13, C2:[1,64,1,1])                                # -in_zp*sum(w) etc.
    %15 = nn.bias_add(%14, C3:[64])
    %16 = cast(%15, int32)                                       # optional
    %17 = subtract(%16, C4:[64,1,1])                             # requantize in-zp
    %18 = fixed_point_multiply_per_axis(%17, C5, C6=lshift, C7=rshift, axes=[1])
    %20 = cast(clip(%18, 0, 255), uint8)

Canonicalize first, then match. Matching ``qnn.*`` directly fails: MergeComposite
lifts the scale/zero-point constants into composite parameters and qnn's own
canonicalization then dies on ``GetScalarFromConstant`` (verified).

Two gates, deliberately separate: shape, then capability
--------------------------------------------------------
``check_fused_qconv`` is the ``MergeComposite`` gate and tests SHAPE ONLY -- is
this a well-formed fused qconv. ``kernel_can_run`` tests CAPABILITY -- can
``libconv2dstem.a`` reproduce it bit-exactly -- and is called during selection
(``aie_annotate``), not during fusion.

They used to be one function, and the cost was real: the stem geometry gate
lived in the ``MergeComposite`` checker, so a layer the kernel could not take
never became a composite at all. "Malformed" and "not the stem" were the same
silence, the inline-back path in ``aie_annotate`` was unreachable, and there was
nowhere to stand to ask "what WOULD it take to offload layer 7".

So: fusion is geometry-agnostic and describes the graph; capability describes
this one kernel and belongs to the step that picks layers. ``kernel_can_run``
returns ``(ok, reason)`` rather than only ``_debug``-printing, because the
caller has to be able to report why a layer the user explicitly asked for
stayed on the APU.

Rejections from either gate print under ``AIE_BYOC_DEBUG=1``, because "did not
match" and "matched but rejected" look identical in the final IR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from tvm import relay
from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard

__all__ = ["pattern_table", "fused_qconv_pattern", "check_fused_qconv",
           "kernel_can_run", "extract_fused", "FusedConv", "AIE_PATTERN_NAME",
           "STEM_GEOMETRY", "ATTR_WEIGHT_FP", "ATTR_LAYER_ID", "composite_name",
           "find_conv", "conv_geometry"]

#: Name of the composite function. After PartitionGraph it shows up in the
#: ``Composite`` attribute, and codegen dispatches on this name.
AIE_PATTERN_NAME = "aie.qconv_fused"

#: Identity of a fused layer, stamped on the composite function in step 1
#: (``aie_fuse``). The sorted-weight hash, not a position: Relay's post-order and
#: the graph executor's order differ (see ``aie_annotate``), so an index is wrong
#: on one side or the other.
ATTR_WEIGHT_FP = "aie.weight_fp"

#: The ``--aie-layers`` number, attached in step 2 once the fingerprint resolves
#: to one. Only SELECTED composites carry it: it is a name for reporting and
#: error messages, where ATTR_WEIGHT_FP is the identity.
ATTR_LAYER_ID = "aie.layer_id"


def composite_name(fn):
    """The ``Composite`` attribute of *fn*, or ``None`` if it is not a composite."""
    if not isinstance(fn, relay.Function) or not fn.attrs:
        return None
    return str(fn.attrs["Composite"]) if "Composite" in fn.attrs else None


def find_conv(expr):
    """The first ``nn.conv2d`` under *expr*, or ``None``."""
    found = []
    relay.analysis.post_order_visit(
        expr, lambda e: found.append(e) if isinstance(e, relay.Call)
        and getattr(e.op, "name", "") == "nn.conv2d" else None)
    return found[0] if found else None

#: The ONE geometry `libconv2dstem.a` implements: a 7x7/s2 stem conv1,
#: ifm[224,224,3] (*) wts[64,7,7,3], stride 2, pad 3 -> ofm[112,112,64].
#:
#: Baked into conv2dstem.cc three times over (the #defines, the GemmSpace
#: descriptors, and static_asserts pinning the mesh tiling), so the gate is
#: exact-match on every field. NOTE the input is 230x230 with padding 0: TVM
#: hoists the spatial padding into a separate ``nn.pad`` before we match.
STEM_GEOMETRY = {
    "in_hw": (230, 230),
    "in_c": 3,
    "kernel": (7, 7),
    "strides": (2, 2),
    "padding": (0, 0, 0, 0),
    "out_c": 64,
}


def _debug(msg: str) -> None:
    if os.environ.get("AIE_BYOC_DEBUG"):
        print(f"  [aie-byoc] {msg}")


def fused_qconv_pattern():
    """conv -> [- wzp term] -> + C2 -> bias_add -> [cast] -> - C4 -> fpm -> clip -> cast."""
    x = wildcard()
    conv = is_op("nn.conv2d")(x, is_constant())
    win_sum = is_op("repeat")(is_op("nn.avg_pool2d")(
        is_op("multiply")(is_op("sum")(is_op("cast")(x)), is_constant())))
    acc = is_op("subtract")(conv, is_op("multiply")(is_constant(), win_sum)) | conv
    acc = is_op("add")(acc, is_constant())
    acc = is_op("nn.bias_add")(acc, is_constant())
    acc = is_op("cast")(acc) | acc
    acc = is_op("subtract")(acc, is_constant())
    acc = is_op("fixed_point_multiply_per_axis")(
        acc, is_constant(), is_constant(), is_constant())
    return is_op("cast")(is_op("clip")(acc))


@dataclass
class FusedConv:
    """Everything the AIE kernel needs from one matched fused op, as numpy."""

    conv: object               # the nn.conv2d call
    weight: np.ndarray         # int8 [OC, IC, KH, KW]
    wzp: np.ndarray            # int32 [OC] weight zero-point (C1), zeros if absent
    add_c: np.ndarray          # int32 [OC] (C2)
    bias: np.ndarray           # int32 [OC] (C3)
    sub_c: np.ndarray          # int32 [OC] (C4)
    mult: np.ndarray           # int32 [OC] Q31 multiplier (C5)
    lshift: np.ndarray         # int32 [OC] (C6)
    rshift: np.ndarray         # int32 [OC] (C7)
    lshift_required: bool
    clip: tuple                # (a_min, a_max)
    out_dtype: str


def _op_name(e) -> str:
    return e.op.name if isinstance(e, relay.Call) and hasattr(e.op, "name") else ""


def _const(e, oc: int, what: str) -> np.ndarray:
    """A per-output-channel int32 vector from a scalar / [OC] / [OC,1,1] / [1,OC,1,1] const."""
    if not isinstance(e, relay.Constant):
        raise ValueError(f"{what} is not a Constant (run FoldConstant first)")
    a = e.data.numpy()
    if a.size == 1:
        return np.full(oc, int(a.reshape(-1)[0]), np.int64)
    if a.size != oc:
        raise ValueError(f"{what} has {a.size} elements, want 1 or {oc}")
    return a.reshape(-1).astype(np.int64)


def _const_arg(call, what: str):
    """(other_arg, constant) of a binary call, whichever side the constant is on."""
    a, b = call.args
    if isinstance(b, relay.Constant):
        return a, b
    if isinstance(a, relay.Constant):
        return b, a
    raise ValueError(f"{what}: no constant operand")


def extract_fused(root) -> FusedConv:
    """Walk the matched chain from the final cast down to the conv. Raises ValueError.

    Shared by the pattern checker and the codegen, so "what C2 means" is decided
    in exactly one place.
    """
    e = root
    if _op_name(e) != "cast":
        raise ValueError(f"root is {_op_name(e)!r}, want cast")
    out_dtype = str(e.attrs.dtype)
    e = e.args[0]
    if _op_name(e) != "clip":
        raise ValueError("no clip under the final cast")
    clip = (float(e.attrs.a_min), float(e.attrs.a_max))
    fpm = e.args[0]
    if _op_name(fpm) != "fixed_point_multiply_per_axis":
        raise ValueError("no fixed_point_multiply_per_axis under clip")
    if [int(a) for a in fpm.attrs.axes] != [1]:
        raise ValueError(f"fpm axes={list(fpm.attrs.axes)}, want [1] (NCHW channel)")
    e = fpm.args[0]
    if _op_name(e) != "subtract":
        raise ValueError("no requantize input-zp subtract")
    e, c4 = _const_arg(e, "requantize subtract")
    if _op_name(e) == "cast":
        e = e.args[0]
    if _op_name(e) != "nn.bias_add":
        raise ValueError("no nn.bias_add")
    e, c3 = e.args[0], e.args[1]
    if _op_name(e) != "add":
        raise ValueError("no zero-point add")
    e, c2 = _const_arg(e, "zero-point add")
    c1 = None
    if _op_name(e) == "subtract":
        e, term = e.args
        if _op_name(term) != "multiply":
            raise ValueError("weight-zp term is not a multiply")
        _, c1 = _const_arg(term, "weight-zp multiply")
    if _op_name(e) != "nn.conv2d":
        raise ValueError(f"chain ends at {_op_name(e)!r}, want nn.conv2d")
    conv = e
    if not isinstance(conv.args[1], relay.Constant):
        raise ValueError("conv weight is not a Constant")
    weight = conv.args[1].data.numpy()
    oc = int(weight.shape[0])
    return FusedConv(
        conv=conv, weight=weight,
        wzp=_const(c1, oc, "C1 weight-zp") if c1 is not None else np.zeros(oc, np.int64),
        add_c=_const(c2, oc, "C2"), bias=_const(c3, oc, "C3 bias"),
        sub_c=_const(c4, oc, "C4"),
        mult=_const(fpm.args[1], oc, "C5 mult"),
        lshift=_const(fpm.args[2], oc, "C6 lshift"),
        rshift=_const(fpm.args[3], oc, "C7 rshift"),
        lshift_required=bool(fpm.attrs.is_lshift_required),
        clip=clip, out_dtype=out_dtype)


def conv_geometry(conv) -> dict:
    """The six fields ``STEM_GEOMETRY`` compares, or ``None`` if unreadable.

    Shapes come off ``checked_type`` for the activation and off the Constant for
    the weight, so this still works on a composite body whose input is a param
    Var that no ``InferType`` has reached yet.
    """
    try:
        in_shape = [int(v) for v in conv.args[0].checked_type.shape]
        w = conv.args[1]
        w_shape = [int(v) for v in (w.data.shape if isinstance(w, relay.Constant)
                                    else w.checked_type.shape)]
    except Exception:
        return None
    if len(in_shape) != 4 or len(w_shape) != 4:
        return None
    return {
        "in_hw": (in_shape[2], in_shape[3]),
        "in_c": in_shape[1],
        "kernel": (w_shape[2], w_shape[3]),
        "strides": tuple(int(v) for v in conv.attrs.strides),
        "padding": tuple(int(v) for v in conv.attrs.padding),
        "out_c": w_shape[0],
    }


def _stem_geometry_ok(conv) -> tuple:
    """Exact-match the one convolution `libconv2dstem.a` implements. ``(ok, reason)``."""
    got = conv_geometry(conv)
    if got is None:
        return False, "cannot read the conv's shapes"
    for field, expect in STEM_GEOMETRY.items():
        if got[field] != expect:
            return False, f"not the stem -- {field}={got[field]} != {expect}"
    return True, "stem geometry"


def check_fused_qconv(root) -> bool:
    """Is this a well-formed fused qconv? SHAPE ONLY -- capability is ``kernel_can_run``.

    This is the ``MergeComposite`` gate, so it decides what FUSES, not what the
    AIE can run. Keep it geometry-agnostic: a layer that fuses but that this
    kernel cannot take is inlined back by ``aie_annotate`` and costs nothing,
    whereas a layer that never fuses cannot be reported on at all.
    """
    try:
        f = extract_fused(root)
    except (ValueError, AttributeError, IndexError) as exc:
        _debug(f"no match: {exc}")
        return False
    # NCHW is structural to this pattern family, not a limit of the kernel:
    # extract_fused and conv_geometry both index positionally.
    layout = str(f.conv.attrs.data_layout)
    if layout != "NCHW":
        _debug(f"no match: layout={layout}, want NCHW")
        return False
    _debug(f"fused qconv: {conv_geometry(f.conv)}")
    return True


def kernel_can_run(root) -> tuple:
    """Can ``libconv2dstem.a`` reproduce this fused op bit-exactly? ``(ok, reason)``.

    Capability, not shape -- called during SELECTION, so a layer the user asked
    for by number can be turned down with a reason instead of silently staying
    on the APU. Every rejection here is something the core's
    ``(acc + bias - zp) * mult >> (shift+31)`` epilogue cannot express.

    Not fail-closed on its own, and does not need to be: ``aie_codegen`` re-checks
    the dtypes and the full signature on the PARTITIONED function, where types
    always exist, and raises. That is why an unreadable input dtype defers here
    rather than rejecting.
    """
    try:
        f = extract_fused(root)
    except (ValueError, AttributeError, IndexError) as exc:
        return False, str(exc)
    conv = f.conv
    if int(conv.attrs.groups) != 1:
        return False, f"groups={conv.attrs.groups}, the kernel does dense conv only"
    try:
        in_dtype = str(conv.args[0].checked_type.dtype)
    except Exception:
        in_dtype = None          # no InferType here yet; aie_codegen will check
    if in_dtype is not None and in_dtype != "uint8":
        return False, f"input dtype {in_dtype}, want uint8"
    if str(f.weight.dtype) != "int8":
        return False, f"weight dtype {f.weight.dtype}, want int8"
    if f.wzp.any():
        return False, "non-zero weight zero-point -- the result depends on sum(x)"
    if f.lshift_required or f.lshift.any():
        return False, "left shift in the requantize -- the kernel only right-shifts"
    if (f.rshift < 0).any():
        return False, "negative right shift"
    if f.clip != (0.0, 255.0) or f.out_dtype != "uint8":
        return False, f"clip={f.clip} out={f.out_dtype}, want [0,255] -> uint8"
    ok, why = _stem_geometry_ok(conv)
    if not ok:
        return False, why
    return True, "stem fused qconv"


def pattern_table():
    """The ``[(name, pattern, checker)]`` that ``MergeComposite`` expects.

    One entry today. A second kernel is a pure addition here plus an emitter in
    ``aie_codegen._EMITTERS`` keyed by the same name; nothing else dispatches on
    the composite name. Order matters if two patterns can match the same chain --
    ``MergeComposite`` tries them in order, so the most specific goes first.
    """
    return [(AIE_PATTERN_NAME, fused_qconv_pattern(), check_fused_qconv)]
