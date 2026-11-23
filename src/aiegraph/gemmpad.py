###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Zero-pad a GEMM so M/K/N are multiples of the AIE macro tile (64).

GAP (2026-11-15, Steve): the K-accumulation plan (example/.../kplan.plan) only accepts
M%64 == K%64 == N%64 == 0, so the im2col GEMMs of e.g. a 4-channel input conv (4096x36x320, K=36) and
conv_out (4096x2880x4, N=4) have no legal schedule.  Padding is exact:
  * extra K columns of A and K rows of B are zero -> they add 0 to every dot product;
  * extra M rows / N columns only produce output that unpad_output() drops.
Host side: pad_gemm_operands() before the AIE launch, unpad_output() after it.
"""
import numpy as np

TILE = 64


def _up(x, t=TILE):
    if x <= 0:
        raise ValueError(f"GEMM dim must be > 0, got {x}")
    return -(-x // t) * t


def pad_mkn(m, k, n, tile=TILE):
    """Return (M, K, N) rounded up to multiples of `tile`."""
    return _up(m, tile), _up(k, tile), _up(n, tile)


def pad_gemm_operands(a, b, tile=TILE):
    """Zero-pad A (M x K) and B (K x N) to tile multiples; returns (A_pad, B_pad)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"need A[M,K] @ B[K,N], got {a.shape} @ {b.shape}")
    m, k = a.shape
    n = b.shape[1]
    pm, pk, pn = pad_mkn(m, k, n, tile)
    ap = np.pad(a, ((0, pm - m), (0, pk - k))) if (pm, pk) != (m, k) else a
    bp = np.pad(b, ((0, pk - k), (0, pn - n))) if (pk, pn) != (k, n) else b
    return ap, bp


def unpad_output(y, m, n):
    """Drop padded rows/cols of the M_pad x N_pad result; returns a contiguous M x N array."""
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[0] < m or y.shape[1] < n:
        raise ValueError(f"output {y.shape} smaller than {(m, n)}")
    return np.ascontiguousarray(y[:m, :n])
