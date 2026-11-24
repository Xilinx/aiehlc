###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""x86 test of aie_kacc_pack.h (runtime kacc operand pack / output un-tile) against the build-time numpy
reference (unet_bk_sdm_tiny/kacc_gemm.layout + kacc_yidx), for the 4 attention bmm geometries of PBK.
Run: python3 src/mlir/runtime/test/test_aie_kacc_pack.py   (needs gcc + numpy)"""
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parents[1]
ROOT = RT.parents[2]
sys.path.insert(0, str(ROOT / "example/aiegraph/unet_bk_sdm_tiny"))
import kacc_gemm as kg  # noqa: E402
from aiegraph import kaccbench as ua  # noqa: E402

SRC = r"""
#include "aie_kacc_pack.h"
void pack(uint16_t *d, const float *s, int R, int rr, int kp, int kr, int kch, long rs, long ks, int w, int z)
{ akp_pack(d, s, R, rr, kp, kr, kch, rs, ks, w, z); }
void unpad(uint16_t *o, long ldo, const float *dev, int Ml, int Nl, int tr, int mr, int nr, int otr)
{ akp_unpad(o, ldo, dev, Ml, Nl, tr, mr, nr, otr); }
void unpad_steps(uint16_t *o, long ldo, const float *dev, int Ml, int Nl, int tr, int mr, int nr, int otr)
{ akp_unpad_job j; akp_unpad_job_init(&j, o, ldo, dev, Ml, Nl, tr, mr, nr, otr); while (akp_unpad_step(&j)) {} }
"""


def lib():
    d = Path(tempfile.mkdtemp())
    (d / "w.c").write_text(SRC)
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", f"-I{RT}", "-o", str(d / "w.so"), str(d / "w.c")])
    return ctypes.CDLL(str(d / "w.so"))


def yidx(m, n, Ml, Nl, tr):  # python transliteration of kacc_gemm.HOST kacc_yidx (one launch)
    r, c = n & 15, m & 15
    if not tr:
        q = Ml // 4; cc = m % q
        wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7)
        return ((n & ~15) + wi // q) * Ml + (m - cc) + wi % q
    q = Nl // 4; cc = n % q
    wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7)
    return ((m & ~15) + wi // q) * Nl + (n - cc) + wi % q


def P(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def check(L, M, K, N, swap=0):
    """A [M,K] @ B [K,N]. swap=0: kernel ch = N (W = B^T), px = M (X = A); swap=1: ch = M (W = A), px = N (X = B^T),
    the kernel computes Y^T and akp_unpad(otr=1) writes Y."""
    rng = np.random.default_rng(M * 7 + K * 3 + N + swap)
    Mp, Kp, Np = (-(-x // 64) * 64 for x in (M, K, N))
    CHR, PXR = (M, N) if swap else (N, M)
    CHP, PXP = (Mp, Np) if swap else (Np, Mp)
    mode, S, chs, pxs = kg.plan(CHP, Kp, PXP)
    geo = ua.kacc_geom(chs, Kp, pxs)
    kch, kp, tr = geo["kch"], geo["kp"], geo["tr"]
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    Wn = A if swap else B.T          # ch rows
    Xn = B.T if swap else A          # px rows
    W = np.zeros((CHP, kp), np.float32); W[:CHR, :K] = Wn
    X = np.zeros((PXP, kp), np.float32); X[:PXR, :K] = Xn
    refX = kg.layout(kg.bf16(X), kp, kch, False); refW = kg.layout(kg.bf16(W), kp, kch, True)
    # strides of the natural tensors: A row-major (rs=K, ks=1), B^T rows = B columns (rs=1, ks=N)
    wsr, wsk = (K, 1) if swap else (1, N)
    xsr, xsk = (1, N) if swap else (K, 1)
    Wsrc, Xsrc = (A, B) if swap else (B, A)
    ok = True
    for z in (0, 1):
        dX = np.zeros((PXP, kp), np.uint16) if z else np.full((PXP, kp), 0x5a5a, np.uint16)
        dW = np.zeros((CHP, kp), np.uint16) if z else np.full((CHP, kp), 0xbeef, np.uint16)
        L.pack(P(dX), P(Xsrc), PXP, PXR, kp, K, kch, ctypes.c_long(xsr), ctypes.c_long(xsk), 0, z)
        L.pack(P(dW), P(Wsrc), CHP, CHR, kp, K, kch, ctypes.c_long(wsr), ctypes.c_long(wsk), 1, z)
        okx, okw = np.array_equal(dX, refX), np.array_equal(dW, refW)
        ok = ok and okx and okw
    dev = rng.standard_normal(S * chs * pxs).astype(np.float32)
    out = np.zeros((M, N), np.uint16); out2 = np.zeros((M, N), np.uint16)
    exp = np.zeros((M, N), np.float32)
    for s in range(S):
        p0, c0 = (s * pxs, 0) if mode == "px" else (0, s * chs)
        pr, cr = min(pxs, PXR - p0), min(chs, CHR - c0)
        if pr <= 0 or cr <= 0:
            continue
        base = s * chs * pxs
        for p in range(pr):
            for c in range(cr):
                v = dev[base + yidx(p, c, pxs, chs, tr)]
                if swap:
                    exp[c0 + c, p0 + p] = v
                else:
                    exp[p0 + p, c0 + c] = v
        off = (c0 * N + p0) if swap else (p0 * N + c0)
        for fn, o in ((L.unpad, out), (L.unpad_steps, out2)):
            fn(ctypes.c_void_p(o.ctypes.data + off * 2), ctypes.c_long(N), ctypes.c_void_p(dev.ctypes.data + base * 4),
               pxs, chs, tr, pr, cr, swap)
    oku = np.array_equal(out, kg.bf16(exp)) and np.array_equal(out2, out)
    print(f"AKPTEST M={M} K={K} N={N} swap={swap} split={mode} S={S} kch={kch} kp={kp} tr={tr} pack={int(ok)} "
          f"unpad={int(oku)}")
    return ok and oku


def test_pack():
    L = lib()
    cases = ((1024, 40, 1024), (1024, 1024, 40), (256, 80, 256), (256, 256, 80), (64, 160, 77), (1024, 40, 77),
             (1024, 168, 1024), (256, 168, 320))
    assert all([check(L, *c) for c in cases] + [check(L, 1024, 168, 1024, 1), check(L, 320, 80, 256, 1),
                                                 check(L, 77, 160, 64, 1)])


if __name__ == "__main__":
    test_pack()
    print("AKPTEST ALL OK")
