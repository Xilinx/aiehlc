###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.wlayout -- build-time weight pre-layout for gemm_kacc (the W port chunk order).

A gemm_kacc launch of instance (N, K, px) (``cnn.kacc_geom``: K chunk ``kch``) reads its weight as
16-row blocks; in every block row ``n`` the K axis is cut into chunks of ``kch`` bf16 and inside a
chunk the element order is ``f = mi*8*kch + kb*64 + mm*8 + kk`` (``n = 16*nb + 8*mi + mm``,
``k = j*kch + 8*kb + kk``). Stored order: ``nb, row, j, q``. The weight is laid out ONCE at build
time (``weights.bin``), so the board does only the activation im2col.

    from aiegraph import wlayout
    lw = wlayout.kacc_wlayout(w6, N, K, M)          # conv: TVM OIHW{i}i{o}o [O/ob][C/ib][KH][KW][ib][ob]
    lw = wlayout.kacc_dense_wlayout(wp, N, K, pxs)  # dense: TVM dense_pack [N/nb][K][nb]
    nb = wlayout.prelayout_weights(wbin, layers)    # rewrite bf16 regions of a weights.bin in place

Used as the ``layout=`` of a step weight (``g.w.as_bf16(layout=...)``). The Kp zero chunk
(``kacc_geom`` kp > k) is NOT stored: the host appends it per row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import AiegraphError

__all__ = ["kacc_wlayout", "kacc_dense_wlayout", "prelayout_weights"]

_STAGE = "group"


def _kch(N: int, K: int, px: int) -> int:
    from . import cnn
    kch = cnn.kacc_geom(N, K, px)["kch"]
    if N % 16 or K % kch or kch % 8:
        raise AiegraphError(_STAGE, f"kacc weight layout {N}x{K} (px {px}): N must be a multiple of 16 and K of "
                                    f"the K chunk {kch}")
    return kch


def _chunks(wn, N: int, K: int, kch: int):
    """natural [N][K] -> nb row j q"""
    t = wn.reshape(N // 16, 2, 8, K // kch, kch // 8, 8)                    # nb mi mm j kb kk
    t = t.transpose(0, 3, 1, 4, 2, 5).reshape(N // 16, K // kch, 16, kch)   # nb j (f//kch) (f%kch)
    return np.ascontiguousarray(t.transpose(0, 2, 1, 3)).reshape(-1)       # nb row j q


def kacc_wlayout(w6, N: int, K: int, M: int):
    """Conv weight [O/ob][C/ib][KH][KW][ib][ob] (TVM OIHW{i}i{o}o, any dtype) -> the kacc W chunk layout of
    instance (N, K, M): natural [N][K] with k = c*KH*KW + t, then :func:`_chunks`. Flat array."""
    w6 = np.asarray(w6)
    if w6.ndim != 6:
        raise AiegraphError(_STAGE, f"kacc_wlayout: weight {list(w6.shape)} is not OIHW{{i}}i{{o}}o (6-D)")
    on, cn, kh, kw, ib, ob = w6.shape
    if on * ob != N or cn * ib * kh * kw != K:
        raise AiegraphError(_STAGE, f"kacc_wlayout: weight {list(w6.shape)} is not N={N} x K={K}")
    kch = _kch(N, K, M)
    return _chunks(w6.transpose(0, 5, 1, 4, 2, 3).reshape(N, K), N, K, kch)


def kacc_dense_wlayout(wp, N: int, K: int, pxs: int):
    """Dense weight [N/nb][K][nb] (TVM dense_pack) -> the kacc W chunk layout of instance (N, K, pxs), same
    order as :func:`kacc_wlayout`. Flat array."""
    wp = np.asarray(wp)
    if wp.ndim != 3:
        raise AiegraphError(_STAGE, f"kacc_dense_wlayout: weight {list(wp.shape)} is not [N/nb][K][nb] (3-D)")
    on, k, ob = wp.shape
    if on * ob != N or k != K:
        raise AiegraphError(_STAGE, f"kacc_dense_wlayout: weight {list(wp.shape)} is not N={N} x K={K}")
    kch = _kch(N, K, pxs)
    return _chunks(wp.transpose(0, 2, 1).reshape(N, K), N, K, kch)


def prelayout_weights(wbin, layers: list) -> int:
    """Rewrite each layer's bf16 weight region of ``wbin`` in place (same size) into the kacc chunk layout.
    A layer is a dict ``woff`` (byte offset), ``wshape`` (stored shape), ``N``, ``K`` and ``M`` (conv) or
    ``pxs`` (``kind == "linear"``: dense_pack weight). Returns the bytes rewritten; a region past the end of
    the file or two layers sharing bytes raise AiegraphError before anything is written."""
    wbin = Path(wbin)
    size = wbin.stat().st_size
    spans = []
    for c in layers:
        nel = int(np.prod(c["wshape"]))
        lo, hi = int(c["woff"]), int(c["woff"]) + 2 * nel
        if lo < 0 or hi > size:
            raise AiegraphError(_STAGE, f"prelayout_weights: weight [{lo}, {hi}) outside {wbin.name} ({size} B)")
        spans.append((lo, hi))
    spans.sort()
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if b0 < a1:
            raise AiegraphError(_STAGE, f"prelayout_weights: weight regions [{a0}, {a1}) and [{b0}, ...) overlap")
    blob = np.memmap(wbin, dtype=np.uint8, mode="r+")
    tot = 0
    for c in layers:
        nel = int(np.prod(c["wshape"]))
        w = np.frombuffer(blob[c["woff"]:c["woff"] + 2 * nel].tobytes(), dtype="<u2").reshape(c["wshape"])
        if c.get("kind") == "linear":
            lw = kacc_dense_wlayout(w, c["N"], c["K"], c["pxs"])
        else:
            lw = kacc_wlayout(w, c["N"], c["K"], c["M"])
        blob[c["woff"]:c["woff"] + 2 * nel] = lw.view(np.uint8)
        tot += 2 * nel
    blob.flush()
    del blob
    return tot
