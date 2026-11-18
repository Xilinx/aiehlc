###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Step 2 of the BYOC flow: SELECT which fused composites go to the AIE.

The second of the four BYOC files. ``AnnotateTarget`` offloads everything that
matches; ``--aie-offload --aie-layers`` asks for specific layers, so this pass
inserts ``compiler_begin``/``compiler_end`` itself, around the composites whose
conv was selected. ``MergeCompilerRegions`` + ``PartitionGraph`` then work as
usual.

Two questions, asked in this order
----------------------------------
1. *Did the user ask for this layer?* -- its ``aie.weight_fp`` tag (stamped by
   ``aie_fuse``) is in ``selected``.
2. *Can the kernel run it?* -- ``aie_patterns.kernel_can_run``, which returns a
   reason.

Keeping these apart is the whole point of the fuse/select split. Fusion is now
geometry-agnostic, so a layer that the AIE cannot take still becomes a composite
and still reaches this pass, where it is turned down BY NAME and inlined back.
Before the split the geometry gate sat in the ``MergeComposite`` checker, so
such a layer never fused, "malformed" and "not the stem" were the same silence,
and the inline-back path below was unreachable code.

A conv is identified by its WEIGHTS, not its position
-----------------------------------------------------
Relay's post-order and the graph executor's order are NOT the same: in every
residual downsampling block TVM fuses the 1x1 shortcut conv with the residual
add, so it runs after the main branch, while Relay's post-order visits it
first (convs 5-7, 10-12, 15-17 swap -- measured). Any "n-th conv" numbering is
therefore wrong on one side. ``weight_fingerprint`` -- a hash of the SORTED
weight values -- is layout-independent (the CPU build stores the same weights
as NCHWc-blocked int16) and unique for all 20 convs of the int8 CNN, so
``aie_offload.resolve_conv_targets`` selects by it.

The boundary goes on the composite function, not on individual operators
-----------------------------------------------------------------------
``compiler_begin`` wraps the composite's inputs and ``compiler_end`` its output.
Marking individual operators inside it would tear the fused kernel apart again.
"""

from __future__ import annotations

import hashlib

import numpy as np
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end

from frontend.tvmrelay.byoc.aie_patterns import (ATTR_LAYER_ID, ATTR_WEIGHT_FP,
                                                 composite_name, find_conv,
                                                 kernel_can_run)

__all__ = ["ConvSelectAnnotator", "AIE_COMPILER_NAME", "weight_fingerprint"]

#: The codegen name registered with TVM. It must match the registration name of
#: ``relay.ext.<name>``, otherwise after PartitionGraph TVM cannot find the
#: backend and reports "No compiler registered".
AIE_COMPILER_NAME = "aie"


def weight_fingerprint(weights) -> str:
    """Layout- and dtype-independent identity of a conv: hash of its sorted weights."""
    a = np.sort(np.asarray(weights).astype(np.int64).reshape(-1))
    return hashlib.sha1(a.tobytes()).hexdigest()


def _conv_geometry(conv) -> dict:
    """(Cin, Cout, K, stride) of an nn.conv2d -- read off the constant weight and
    attrs, so it works on freshly rebuilt nodes that carry no checked_type."""
    w = conv.args[1]
    shape = [int(d) for d in (w.data.shape if isinstance(w, relay.Constant)
                              else w.checked_type.shape)]
    return {"Cin": shape[1], "Cout": shape[0], "K": shape[2],
            "stride": int(conv.attrs.strides[0])}


class ConvSelectAnnotator(relay.ExprMutator):
    """Offload the composites that were both asked for AND are runnable.

    *selected* is a ``{weight_fingerprint: layer_index}`` mapping (a bare
    iterable of fingerprints also works, giving ``None`` layer numbers; ``None``
    means every composite the kernel can run).

    Composites that are not offloaded -- not selected, or selected but rejected
    by ``kernel_can_run`` -- are inlined back (see ``visit_call``). ``convs``
    records ``{fingerprint, geometry, composite, offloaded[, reason]}`` for every
    conv, matched or not, for the caller to verify; ``rejected`` maps the
    fingerprint of an asked-for-but-unrunnable layer to its reason.
    """

    def __init__(self, selected, pattern_name: str,
                 compiler: str = AIE_COMPILER_NAME):
        super().__init__()
        if selected is None:
            self.selected = None
        elif isinstance(selected, dict):
            self.selected = dict(selected)
        else:
            self.selected = {fp: None for fp in selected}
        self.compiler = compiler
        self.pattern_name = pattern_name
        self.convs = []
        self.rejected = {}

    @staticmethod
    def _fingerprint(conv):
        w = conv.args[1]
        return weight_fingerprint(w.data.numpy()) if isinstance(w, relay.Constant) else None

    def _identity(self, op, conv):
        """The composite's fingerprint: the step-1 tag, else recomputed.

        The fallback is not decoration. If the tag were ever lost -- a pass that
        rebuilds Function attrs, a composite formed somewhere other than
        ``aie_fuse`` -- trusting only the attribute would select NOTHING, which
        on the console is indistinguishable from "the pattern did not match".
        Recomputing costs one hash and keeps selection correct either way.
        """
        if op.attrs and ATTR_WEIGHT_FP in op.attrs:
            return str(op.attrs[ATTR_WEIGHT_FP])
        return self._fingerprint(conv) if conv is not None else None

    def _record(self, conv, composite: bool, offloaded: bool,
                fingerprint=None, reason=None) -> None:
        if conv is None:
            return
        rec = {"fingerprint": (fingerprint if fingerprint is not None
                               else self._fingerprint(conv)),
               **_conv_geometry(conv),
               "composite": composite, "offloaded": offloaded}
        if reason:
            rec["reason"] = reason
        self.convs.append(rec)

    def visit_call(self, call):
        new_args = [self.visit(a) for a in call.args]
        op = call.op

        if composite_name(op) != self.pattern_name:
            if getattr(op, "name", "") == "nn.conv2d":
                self._record(call, composite=False, offloaded=False)
            return relay.Call(self.visit(op) if isinstance(op, relay.Function) else op,
                              new_args, call.attrs, call.type_args, call.span)

        conv = find_conv(op.body)
        fp = self._identity(op, conv)
        want = self.selected is None or (fp is not None and fp in self.selected)
        # Only ask the kernel about layers somebody wants -- the answer for the
        # rest is "nobody asked", not "the AIE cannot do it".
        ok, reason = kernel_can_run(op.body) if want else (False, "not selected")
        take = want and ok
        self._record(conv, composite=True, offloaded=take, fingerprint=fp,
                     reason=None if take else reason)

        if not take:
            if want and fp is not None:
                self.rejected[fp] = reason
            # Inline the composite back. A Composite function left in main has
            # neither a Compiler nor a Primitive attribute, and TECompiler dies:
            #   Check failed: (prim_fns) is false: primitive functions not set
            # Inlining = bind the params to the args, then take the body.
            #
            # This runs for every fused-but-not-offloaded layer, so it is on the
            # hot path now that fusion is geometry-agnostic -- and the whole C is
            # rebuilt from this module, so it has to round-trip exactly. That is
            # what verify_aie_stem.py's whole-network logit check covers.
            return relay.bind(op.body, dict(zip(op.params, new_args)))

        layer = None if self.selected is None else self.selected.get(fp)
        if layer is not None:
            op = op.with_attr(ATTR_LAYER_ID, int(layer))
        begins = [compiler_begin(a, self.compiler) for a in new_args]
        out = relay.Call(op, begins, call.attrs, call.type_args, call.span)
        return compiler_end(out, self.compiler)

    @property
    def annotated(self) -> int:
        return sum(1 for c in self.convs if c["offloaded"])

    def report(self) -> str:
        """One-line summary the caller can print directly."""
        matched = sum(1 for c in self.convs if c["composite"])
        asked = "all" if self.selected is None else len(self.selected)
        extra = f", {len(self.rejected)} rejected" if self.rejected else ""
        return (f"annotated {self.annotated} {self.pattern_name} for AIE "
                f"({asked} conv(s) selected{extra}; "
                f"{matched}/{len(self.convs)} convs fused)")
