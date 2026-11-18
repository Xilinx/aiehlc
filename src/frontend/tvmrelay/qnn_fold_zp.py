#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Collapse an all-equal per-channel qnn zero point to a rank-0 scalar.

Why this exists
---------------
A quantized conv lowers to four terms (``src/relay/qnn/op/convolution.cc``)::

    out = term1 - term2 - term3 + term4
      term1 = conv2d(data, kernel)
      term2 = kernel_zero_point * sum(data over the kernel window)
      term3 = input_zero_point  * sum(kernel)
      term4 = input_zp * kernel_zp * k_h * k_w * in_channels

``Conv2DCombineTerms`` drops **term2 and term4 when the kernel zero point is
zero** -- which it is for every conv in this model, because the quantizer uses
*symmetric* per-channel weights. It does not drop them here, and the reason is
narrow: the integer it tests is only extracted when
``IsConstScalar(kernel_zero_point)`` (``convolution.cc:747``), and
``Constant::is_scalar()`` is strictly ``ndim == 0``
(``include/tvm/relay/expr.h:80``). Per-channel quantization makes the zero
point a ``[64]`` tensor, so the test fails, ``dynamic_zp`` is set, both zero
points fall back to ``-1``, and the final ``else`` emits term2 unconditionally.

The elision keys on the zero point being a **scalar**, not on its values being
zero. A per-*tensor* symmetric model takes the fast path; a per-*channel*
symmetric one misses it even though every element is 0.

What it costs when it misses
----------------------------
term2 is not a cheap constant -- it depends on the live input, so
``FoldConstant`` cannot touch it. It lowers to a real four-kernel chain per
conv::

    cast_sum -> multiply -> avg_pool2d -> repeat_multiply_layout_transform

Measured on the int8 reference CNN (zoo default): **21 such chains, ~70 of the 120 kernel calls and
10.6 MB of buffers**, every one of them computing and then subtracting zeros.
For layer 06 alone that is 802,816 int32 zeros.

What this does
--------------
Rewrites a constant zero point whose elements are **all equal** into a rank-0
scalar of the same value and dtype, *before* ``qnn.CanonicalizeOps`` runs. TVM's
own elision then fires and the chains never get built. Nothing else changes:
``kernel_scale`` stays per-channel, because the requantization genuinely is.

This is output-preserving by construction -- a tensor of N copies of *z* and a
scalar *z* describe the same quantization -- and it is checked rather than
assumed: ``all equal`` is required, not ``all zero``, so a uniform non-zero
zero point also folds and TVM still emits the terms it needs.
"""
from __future__ import annotations


#: ``op name -> index of the kernel zero point in its argument list``.
#: Both take ``(data, weight, input_zero_point, kernel_zero_point, ...)``; the
#: index is spelled out rather than searched for, so a signature change breaks
#: loudly here instead of silently folding the wrong argument.
_ZP_ARG = {"qnn.conv2d": 3, "qnn.dense": 3}


def _folded(arr):
    """The scalar an array collapses to, or ``None`` if it cannot.

    ``None`` for a rank-0 array (already scalar, nothing to do), an empty one,
    or one whose elements differ.
    """
    if arr.ndim == 0 or arr.size == 0:
        return None
    flat = arr.reshape(-1)
    if not bool((flat == flat[0]).all()):
        return None
    return flat[0]


def fold_scalar_zero_points(mod, *, verbose: bool = True):
    """Return *mod* with uniform per-channel qnn zero points made rank-0.

    Must run **before** ``qnn.CanonicalizeOps`` -- i.e. before ``relay.build``
    and before ``aie_byoc.partition_for_aie``, both of which canonicalize
    internally. Afterwards there is nothing left to rewrite.
    """
    from tvm import relay

    class _Fold(relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.folded = []

        def visit_call(self, call):
            call = super().visit_call(call)
            idx = _ZP_ARG.get(getattr(call.op, "name", None))
            if idx is None or idx >= len(call.args):
                return call
            zp = call.args[idx]
            if not isinstance(zp, relay.Constant):
                return call
            arr = zp.data.numpy()
            value = _folded(arr)
            if value is None:
                return call
            args = list(call.args)
            # relay.const of a Python scalar gives ndim 0, which is what
            # Constant::is_scalar() demands. A `[1]`-shaped constant would
            # still fail the test and change nothing.
            args[idx] = relay.const(value.item(), str(arr.dtype))
            self.folded.append((call.op.name, arr.shape, value.item()))
            return relay.Call(call.op, args, call.attrs, call.type_args,
                              call.span)

    fold = _Fold()
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.InferType()(
        relay.transform.function_pass(
            lambda f, m, c: fold.visit(f), opt_level=0)(mod))
    if verbose:
        n = len(fold.folded)
        if n:
            zeros = sum(1 for _, _, v in fold.folded if v == 0)
            print(f"[3/7] qnn zp : folded {n} per-channel kernel zero point(s) "
                  f"to scalars ({zeros} of them zero -- those drop qnn's "
                  f"term2/term4 chains entirely)")
        else:
            print("[3/7] qnn zp : nothing to fold (zero points already scalar "
                  "or non-uniform)")
    return mod
