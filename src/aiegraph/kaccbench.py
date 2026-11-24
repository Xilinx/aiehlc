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
ctrlpkt "aot-kco"), the same switch the network build's --ctrlpkt aot-kco uses; AIEGRAPH_CTRLPKT=off drops the
pragmas (MMIO path, the cost model's p0_one_kernel.sh --ctrlpkt off).

Bench-only walk knobs (cost-model probes, script/perf/p0_one_kernel.sh; the network builds never read them, they use
aiegraph.cnn.kacc_geom directly):
  KACC_TR_TKW_CAP  cap on the transposed b window ch/4 x tkw in words (default 1024, as cnn.kacc_geom)
  KACC_KP_PAD      pad K with zero chunks up to this kp (stride probe; default 0 = off); applied only when > kp,
                   a multiple of 2*tkw, and kp/2 avoids ch, 32, 64 -- and px when transposed (square a port)
and, unlike cnn.kacc_geom, the default kp pad also covers a transposed launch with px == k/2 (square a port; the
bench host takes kp from the walk, the network hosts compute Kp themselves).
"""
from __future__ import annotations

import os

from aiednn import gemm_kacc_launch

from . import cnn, kincore  # noqa: F401  (cnn: the network walk this one extends)
from .errors import AiegraphError

__all__ = ["NoWalk", "kacc_geom", "kacc_name", "KACC_KERNEL", "CTRLPKT_KCO", "CTRLPKT_MODE", "CTRLPKT_PRAGMAS",
           "CTRLPKT_NPRAGMAS", "ctrlpkt_mode"]

KCO_ENV = "AIEGRAPH_CTRLPKT_KCO"
CTRLPKT_ENV = "AIEGRAPH_CTRLPKT"
TKW_CAP_ENV = "KACC_TR_TKW_CAP"
KP_PAD_ENV = "KACC_KP_PAD"
PORT_BYTES = 29696


class NoWalk(AiegraphError, AssertionError):
    """(ch, k, px) has no gemm_kacc walk; an AssertionError too, so a bench's ``except AssertionError`` split
    search keeps working."""


def kacc_geom(ch: int, k: int, px: int, env=None) -> dict:
    """Walk of one bench launch: the aiegraph.cnn.kacc_geom rule plus the bench knobs (module docstring) read from
    *env* (default os.environ). A shape without a walk raises :class:`NoWalk`."""
    env = os.environ if env is None else env
    cap, kpad = int(env.get(TKW_CAP_ENV, "1024")), int(env.get(KP_PAD_ENV, "0"))
    if ch % 64 or px % 64 or k % 64:
        raise NoWalk("kaccbench", f"gemm_kacc instance {ch}x{k}x{px}: every dim must be a multiple of 64")
    tkw = min(32, 4096 // px)
    tr = 2 * (16 * tkw * 4 + (px // 4) * tkw * 4 + 16 * (px // 4) * 4) > PORT_BYTES
    if tr:
        tkw = 32
        while tkw >= 4 and ((ch // 4) * tkw > cap or (k // 2) % tkw):
            tkw //= 2
    if tkw < 4 or (k // 2) % tkw:
        raise NoWalk("kaccbench", f"gemm_kacc instance {ch}x{k}x{px}: no K chunk (tkw={tkw})")
    kp = k + 2 * tkw if ch == k // 2 or (tr and px == k // 2) else k
    if kpad > kp and kpad % (2 * tkw) == 0 and kpad // 2 not in (ch, 32, 64) + ((px,) if tr else ()):
        kp = kpad
    rows, cols = (px, ch) if tr else (ch, px)
    l1 = 2 * (16 * tkw * 4 + (cols // 4) * tkw * 4 + 16 * (cols // 4) * 4)
    if l1 > PORT_BYTES:
        raise NoWalk("kaccbench", f"gemm_kacc instance {ch}x{k}x{px}: port buffers {l1} B > {PORT_BYTES}")
    return dict(ntile=rows // 64, S=cols // 64, tkw=tkw, kch=2 * tkw, nkc=(kp // 2) // tkw, kp=kp, tr=int(tr),
                arows=rows, brows=cols, l1=l1)


def kacc_name(ch: int, k: int, px: int) -> str:
    """the launch callee of instance (ch, k, px) with the bench walk (:func:`kacc_geom`); :class:`NoWalk` if none."""
    return gemm_kacc_launch(ch, k, px, kacc_geom(ch, k, px))


def ctrlpkt_mode(env=None) -> str:
    """``"off"`` when AIEGRAPH_CTRLPKT=off in *env* (default os.environ), ``"aot-kco"`` when AIEGRAPH_CTRLPKT_KCO=1,
    else ``"aot"``."""
    env = os.environ if env is None else env
    if env.get(CTRLPKT_ENV, "") == "off":
        return "off"
    return "aot-kco" if env.get(KCO_ENV, "0") == "1" else "aot"


KACC_KERNEL = kincore.KACC_KERNEL
CTRLPKT_MODE = ctrlpkt_mode()
CTRLPKT_KCO = CTRLPKT_MODE == "aot-kco"
CTRLPKT_PRAGMAS = kincore.ctrlpkt_head(CTRLPKT_MODE)
CTRLPKT_NPRAGMAS = kincore.ctrlpkt_npragmas(CTRLPKT_MODE)
