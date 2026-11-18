#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Tell TVM the target PE takes int8 operands, so int8 values are *stored* int8.

The quantizer already produces genuine int8: every one of the reference CNN's
11,678,912 weight values measures inside ``[-127, 127]``, using 255 distinct
levels. What widens them to int16 is **legalization**, which happens after
quantization and is a statement about the *hardware*, not about the numbers.

``Target("c")`` carries ``keys=['cpu']``, so TVM applies its **Intel x86**
rule: ``_qnn_conv2d_legalize_intel_cpu`` asks ``is_fast_int8_on_intel()`` ->
``target_has_features("sse4.2")``, which is false for a C-source target, and
falls back to ``helper_no_fast_int8_hw_legalization``. That helper casts
**data and kernel to int16** and subtracts the zero-points eagerly -- a
reasonable choice for an LLVM backend with no VNNI, where int16 vectorizes
better than int8, and the wrong one for a PE whose MACs are int8.

This module says so properly, by giving the target its own key::

    Target("c -keys=pe_int8,cpu", host="c")

``qnn_conv2d_legalize`` is a ``tvm.target.generic_func``: it tries the target's
keys in order, so ``pe_int8`` wins and ``cpu`` stays available as the fallback
for everything else (schedules, layout alteration). That is the supported way
to describe a new device, and it is why this is **not** done by
monkeypatching the ``"cpu"`` registration -- that would silently retarget
every x86 build in the same process, including anything else importing TVM.

What the int8 rule does
-----------------------
``helper_change_dtypes_to_int8`` (TVM's own, written for Nvidia ``dp4a``) is
exactly the convention the AIE kernel already uses::

    x_i8  = x_u8 - 128
    zp_i8 = zp   - 128          (113 -> -15 for this model)

and then re-emits ``qnn.conv2d`` with int8 operands. The shifted zero-point is
folded into the bias by ``QnnConv2DCanonicalize``'s four-term expansion, which
is the same algebra as ``conv2dstem``'s ``+128*sum(w)`` bias fold (skill
**byocaieoffload**) -- reached from the Relay side instead of by hand. It
returns ``None`` when the operands are already int8, which is what stops the
legalize pass from looping.

What does **not** become int8, and must not
-------------------------------------------
Only the *operands* are 8-bit. Measured on this model:

===================  =======  ==============  ==============
role                 dtype    min             max
===================  =======  ==============  ==============
weight               int8     -127            127
bias                 int32    -3,191,545      498,553
requant multiplier   int32    1,073,764,180   2,147,291,785
requant shift        int32    6               27
===================  =======  ==============  ==============

The bias lives at the **accumulator** scale (``s_x * s_w``), so it is ~10^6
times the real value -- 23 bits here. The multiplier is a fixed-point scale in
``[2^30, 2^31)``, int32 by construction. Both are int32 in every int8 scheme
(TFLite, ONNX QDQ, PyTorch); narrowing them would be arithmetically wrong, not
an optimization. They are 116 KB of a 23 MB blob.

Measured effect of switching (reference CNN int8, dog.jpg)
---------------------------------------------------------
======================  ==============  ==============
                        int16 operands  int8 operands
======================  ==============  ==============
``weights.bin``         23,484,752 B    11,851,606 B
activation buffers      25.7 MB         18.6 MB
``main_local.elf``      24,253,696 B    12,650,344 B
top-1 / logit           258 / 12.017338 **identical**
======================  ==============  ==============

Output is bit-identical because this is algebra on the same int8 values, not a
different quantization. The cost is the four-term expansion made explicit: 30
-> 120 graph nodes, 28 -> 64 kernels, and int32 scratch for the zero-point
correction reductions.

Use::

    PYTHONPATH=src python src/frontend/tvmrelay/deploy_flow.py --pe-int8
"""

from __future__ import annotations

__all__ = ["PE_KEY", "pe_target", "register_int8_legalization", "enable"]

#: The target key this module registers against. Appears first in the target
#: string so it is tried before ``cpu``.
PE_KEY = "pe_int8"

#: Set once registration has happened; registering twice in one process is
#: harmless (we pass ``override=True``) but reporting it twice is noise.
_registered = False


def pe_target(host: str = "c"):
    """``Target("c -keys=pe_int8,cpu", host=...)`` -- a C target with an int8 PE.

    ``cpu`` is kept as the second key deliberately. Only the QNN legalization
    is overridden here; operator strategies, schedules and
    ``conv2d_alter_op`` are all registered on ``cpu`` and must still resolve,
    which they do because ``generic_func`` falls through to the next key when
    the first has no registration.
    """
    import tvm

    return tvm.target.Target(f"c -keys={PE_KEY},cpu", host=host)


def register_int8_legalization(verbose: bool = True) -> bool:
    """Register the int8 QNN legalization for :data:`PE_KEY`. Idempotent.

    Returns True the first time it actually registers. Both ``qnn.conv2d`` and
    ``qnn.dense`` are covered -- leaving ``dense`` out would quietly keep the
    final fully-connected layer's 1 MB of weights at int16.
    """
    global _registered
    if _registered:
        return False

    from tvm import relay
    from tvm.relay.qnn.op import legalizations as L

    def _conv2d(attrs, inputs, types):
        # Returns None once the operands are already int8, which terminates
        # the legalize pass instead of rewriting forever.
        return L.helper_change_dtypes_to_int8(attrs, inputs, types,
                                              relay.qnn.op.conv2d)

    def _dense(attrs, inputs, types):
        return L.helper_change_dtypes_to_int8(attrs, inputs, types,
                                              relay.qnn.op.dense)

    # override=True so a second import in the same process is not fatal.
    L.qnn_conv2d_legalize.register(PE_KEY, _conv2d, override=True)
    L.qnn_dense_legalize.register(PE_KEY, _dense, override=True)
    _registered = True
    if verbose:
        print(f"  [pe] target key '{PE_KEY}': qnn.conv2d/qnn.dense legalized "
              f"to int8 x int8 (x_i8 = x_u8 - 128, zp -= 128)")
    return True


def enable(host: str = "c", verbose: bool = True):
    """Register the rule and return the target to build against."""
    register_int8_legalization(verbose=verbose)
    return pe_target(host=host)
