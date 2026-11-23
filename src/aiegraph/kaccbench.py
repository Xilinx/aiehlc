###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.kaccbench -- the names a stand-alone gemm_kacc bench needs (one instance, K in the core).

    from aiegraph import kaccbench as kb
    geo  = kb.kacc_geom(ch, k, px)        # walk of one launch; NoWalk (AiegraphError + AssertionError) if none
    name = kb.kacc_name(ch, k, px)        # launch callee gemm_kacc<...>
    src  = kb.CTRLPKT_PRAGMAS + prelude + kb.KACC_KERNEL + host

Generic over the model: everything is keyed on the GEMM shape (ch, k, px). The bench unit carries the AOT
control-packet pragmas; AIEGRAPH_CTRLPKT_KCO=1 in the environment adds KERNELCONFIGOFFLOAD (aiegraph.kincore
ctrlpkt "aot-kco"), the same switch the network build's --ctrlpkt aot-kco uses.
"""
from __future__ import annotations

import os

from . import cnn, kincore
from .errors import AiegraphError

__all__ = ["NoWalk", "kacc_geom", "kacc_name", "KACC_KERNEL", "CTRLPKT_KCO", "CTRLPKT_MODE", "CTRLPKT_PRAGMAS",
           "CTRLPKT_NPRAGMAS", "ctrlpkt_mode"]

KCO_ENV = "AIEGRAPH_CTRLPKT_KCO"


class NoWalk(AiegraphError, AssertionError):
    """(ch, k, px) has no gemm_kacc walk; an AssertionError too, so a bench's ``except AssertionError`` split
    search keeps working."""


def kacc_geom(ch: int, k: int, px: int) -> dict:
    """aiegraph.cnn.kacc_geom; a shape without a walk raises :class:`NoWalk`."""
    try:
        return cnn.kacc_geom(ch, k, px)
    except NoWalk:
        raise
    except AiegraphError as e:
        raise NoWalk(e.stage, e.reason, e.hint) from None


def kacc_name(ch: int, k: int, px: int) -> str:
    """the launch callee of instance (ch, k, px) (aiegraph.kincore.kacc_name); :class:`NoWalk` if no walk."""
    kacc_geom(ch, k, px)
    return kincore.kacc_name(ch, k, px)


def ctrlpkt_mode(env=None) -> str:
    """``"aot-kco"`` when AIEGRAPH_CTRLPKT_KCO=1 in *env* (default os.environ), else ``"aot"``."""
    env = os.environ if env is None else env
    return "aot-kco" if env.get(KCO_ENV, "0") == "1" else "aot"


KACC_KERNEL = kincore.KACC_KERNEL
CTRLPKT_MODE = ctrlpkt_mode()
CTRLPKT_KCO = CTRLPKT_MODE == "aot-kco"
CTRLPKT_PRAGMAS = kincore.ctrlpkt_head(CTRLPKT_MODE)
CTRLPKT_NPRAGMAS = kincore.ctrlpkt_npragmas(CTRLPKT_MODE)
