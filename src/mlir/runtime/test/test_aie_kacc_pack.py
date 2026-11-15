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
import unet_aie as ua  # noqa: E402

SRC = r"""
#include "aie_kacc_pack.h"
void pack(uint16_t *d, const float *s, int R, int rr, int kp, int kr, int kch, long rs, long ks, int w)
{ akp_pack(d, s, R, rr, kp, kr, kch, rs, ks, w); }
void unpad(uint16_t *o, long ldo, const float *dev, int Ml, int Nl, int tr, int mr, int nr)
{ akp_unpad(o, ldo, dev, Ml, Nl, tr, mr, nr); }
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


def check(L, M, K, N):
    rng = np.random.default_rng(M * 7 + K * 3 + N)
    Mp, Kp, Np = (-(-x // 64) * 64 for x in (M, K, N))
    mode, S, chs, pxs = kg.plan(Np, Kp, Mp)
    geo = ua.kacc_geom(chs, Kp, pxs)
    kch, kp, tr = geo["kch"], geo["kp"], geo["tr"]
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    X = np.zeros((Mp, kp), np.float32); X[:M, :K] = A
    W = np.zeros((Np, kp), np.float32); W[:N, :K] = B.T
    refX = kg.layout(kg.bf16(X), kp, kch, False); refW = kg.layout(kg.bf16(W), kp, kch, True)
    dX = np.zeros((Mp, kp), np.uint16); dW = np.full((Np, kp), 0xbeef, np.uint16)
    L.pack(P(dX), P(A), Mp, M, kp, K, kch, ctypes.c_long(K), ctypes.c_long(1), 0)
    L.pack(P(dW), P(B), Np, N, kp, K, kch, ctypes.c_long(1), ctypes.c_long(N), 1)
    okx, okw = np.array_equal(dX, refX), np.array_equal(dW, refW)
    # un-tile: random device buffer per launch, expected via yidx
    dev = rng.standard_normal(S * chs * pxs).astype(np.float32)
    out = np.zeros((M, N), np.uint16)
    exp = np.zeros((M, N), np.float32)
    for s in range(S):
        m0, n0 = (s * pxs, 0) if mode == "px" else (0, s * chs)
        mr, nr = min(pxs, M - m0), min(chs, N - n0)
        if mr <= 0 or nr <= 0:
            continue
        base = s * chs * pxs
        for m in range(mr):
            for n in range(nr):
                exp[m0 + m, n0 + n] = dev[base + yidx(m, n, pxs, chs, tr)]
        L.unpad(ctypes.c_void_p(out.ctypes.data + (m0 * N + n0) * 2), ctypes.c_long(N),
                ctypes.c_void_p(dev.ctypes.data + base * 4), pxs, chs, tr, mr, nr)
    oku = np.array_equal(out, kg.bf16(exp))
    print(f"AKPTEST M={M} K={K} N={N} split={mode} S={S} kch={kch} kp={kp} tr={tr} packX={int(okx)} packW={int(okw)} "
          f"unpad={int(oku)}")
    return okx and okw and oku


def test_pack():
    L = lib()
    assert all([check(L, *s) for s in ((1024, 40, 1024), (1024, 1024, 40), (256, 80, 256), (256, 256, 80),
                                       (64, 160, 77), (1024, 40, 77))])


if __name__ == "__main__":
    test_pack()
    print("AKPTEST ALL OK")
