###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""User AIE kernels: bring your own kernel, offload one layer or a bunch of them.

The C side is ``include/aie_user_kernel.h``: an aiehlc source (library mode)
that defines ``AIEK_ENTRY(name)`` and describes it with ``AIEK_SPEC(name,
"...")``, plus an optional x86 ``AIEK_REFERENCE(name)``. This package is the
Python side; it never needs a board, and only ``build`` needs Vitis.

    from frontend.tvmrelay import aiekernel

    rep = aiekernel.lint_source("src/aietensorop/conv2dstem/conv2dstem_uk.cc")
    rep = aiekernel.check("worklocal/tvmrelay_deploy",
                          ["src/aietensorop/conv2dstem/conv2dstem_uk.cc"])
    rep = aiekernel.offload("worklocal/tvmrelay_deploy",
                            ["src/aietensorop/conv2dstem/conv2dstem_uk.cc@1"], make=True)

or ``python -m frontend.tvmrelay.aiekernel {lint,detect,check,offload} ...``, or
``deploy_flow.py --aie-kernel <src>[@layers]``. Tutorial:
``doc/tutorial_userkernel.md``.

==========  ==================================================================
module      gate
==========  ==================================================================
spec        read the source the way aiehlc does; parse ``AIEK_SPEC``
lint        silent-failure traps in the source (L0xx codes)
graph       the deployed network as typed kernel calls (TVM-free)
match       DETECT: which layers fit, incl. param VALUE requirements
calibrate   reference vs TVM's own kernel, per call, adversarial inputs
emulate     whole network on x86 through the generated adapter
adapter     the ``aie_<symbol>`` TVM packed-call glue (board and x86 alike)
build       aiehlc + hostcompile + symbol isolation (compile only)
flow        the gates in order (check, offload); ``partition.json``; ``arm_build/``
==========  ==================================================================
"""

from __future__ import annotations

# NOTE: export functions under names that differ from the submodules (`lint`,
# `spec`, ...): `from .lint import lint_source as lint` would replace the
# package attribute `lint` and make the submodule unreachable by attribute.
from .graph import load_network
from .lint import lint_source
from .match import detect_layers
from .flow import check, offload, parse_kernel_arg, plan
from .spec import load_source, parse_specs

__all__ = ["lint_source", "load_network", "detect_layers", "check", "offload",
           "plan", "parse_kernel_arg", "load_source", "parse_specs", "detect"]


def detect(out_dir, source, check_values: bool = True) -> list:
    """``[LayerVerdict]`` for every layer of the network at *out_dir* against the
    first ``AIEK_SPEC`` in *source*. Lint errors raise ``ValueError``."""
    rep = lint_source(source)
    if not rep.specs:
        raise ValueError(f"{source}: no usable AIEK_SPEC\n{rep.text()}")
    return detect_layers(rep.specs[0], load_network(out_dir), check_values=check_values)
