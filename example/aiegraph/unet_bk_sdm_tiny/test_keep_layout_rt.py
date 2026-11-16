"""P4a keep-layout: x86 check of the strided FA packs (fa_attn.py, scalar path) -- packing Q/K/V straight from the
[rows][H*D] linear outputs (scale in the pack) gives the same AIE port images as packing TVM's [H][R][D] / V^T
tensors, and O written by stride equals the [H][N][D] unpack followed by TVM's head merge."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

HARNESS = r'''
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#define UC_NEON 0
typedef struct { int H, N, M, D, split, dq, nkv, E, units, upr, launches, qkrow, vrow, orow; } fa_cls;
static uint16_t *g_fa_qk[1], *g_fa_v[1], *g_fa_o[1];
static float g_mu[256]; static float *g_fa_mu = g_mu;
static unsigned long long g_uc_macs; static unsigned g_fa_sites;
@SRC@
int main(int argc, char **argv) {
  int H = 8, N = atoi(argv[1]), M = atoi(argv[2]), D = atoi(argv[3]);
  int split = D > 80, dq = split ? D / 2 : D, nkv = (M + 31) / 32, E = 32 * dq;
  fa_cls c = {H, N, M, D, split, dq, nkv, E, split ? H : H * N / 128, split ? 1 : N / 128, 0, 8 * E * 2,
              nkv * 2 * (split ? 2 : 1) * E * 2, 8 * E * 2};
  size_t qkn = (size_t)(H * N / 32 + 64) * 4 * c.qkrow, vn = (size_t)H * c.vrow, on = (size_t)(H * N / 32 + 64) * 4 * c.orow;
  std::vector<uint16_t> qa(qkn), qb(qkn), va(vn), vb(vn), oo(on);
  int C = H * D; float s = powf((float)D, -0.25f);
  std::vector<float> Ql(N * C), Kl(M * C), Vl(M * C), Qt(H * N * D), Kt(H * M * D), VT(H * D * M);
  srand(1);
  for (auto *v : {&Ql, &Kl, &Vl}) for (auto &x : *v) x = (rand() % 2001 - 1000) / 1000.f;
  for (int h = 0; h < H; h++) for (int n = 0; n < N; n++) for (int d = 0; d < D; d++) Qt[(h * N + n) * D + d] = Ql[n * C + h * D + d] * s;
  for (int h = 0; h < H; h++) for (int j = 0; j < M; j++) for (int d = 0; d < D; d++) Kt[(h * M + j) * D + d] = Kl[j * C + h * D + d] * s;
  for (int h = 0; h < H; h++) for (int j = 0; j < M; j++) for (int d = 0; d < D; d++) VT[(h * D + d) * M + j] = Vl[j * C + h * D + d];
  fa_pack_q(c, Qt.data(), (size_t)N * D, D, 1.0f, qa.data()); fa_pack_q(c, Ql.data(), D, C, s, qb.data());
  fa_pack_k(c, Kt.data(), (size_t)M * D, D, 1.0f, va.data()); fa_pack_k(c, Kl.data(), D, C, s, vb.data());
  fa_pack_v(c, VT.data(), (size_t)D * M, 0, 1, va.data()); fa_pack_v(c, Vl.data(), D, C, 0, vb.data());
  long dq_ = 0, dk = 0, bigk = 0;
  for (size_t i = 0; i < qkn; i++) dq_ += qa[i] != qb[i];
  for (size_t i = 0; i < vn; i++) if (va[i] != vb[i]) { dk++; if (abs((int)va[i] - (int)vb[i]) > 2) bigk++; }
  for (size_t i = 0; i < on; i++) oo[i] = fa_f2bf((rand() % 2001 - 1000) / 1000.f);
  g_fa_o[0] = oo.data();
  std::vector<float> Yt(H * N * D, 7.f), Yl(N * C, 9.f);
  fa_run_o(c, 0, Yt.data(), (size_t)N * D, D); fa_run_o(c, 0, Yl.data(), D, C);
  long dO = 0;
  for (int h = 0; h < H; h++) for (int n = 0; n < N; n++) for (int d = 0; d < D; d++) dO += Yt[(h * N + n) * D + d] != Yl[n * C + h * D + d];
  printf("KLTEST qdiff=%ld kvdiff=%ld big=%ld odiff=%ld kvn=%zu\n", dq_, dk, bigk, dO, vn);
  return 0;
}
'''


def _src() -> str:
    src = (HERE / "fa_attn.py").read_text()
    a = src.index("static inline uint16_t fa_f2bf(float f)")
    helpers = src[a:src.index("#if UC_NEON", a)]
    packs = src[src.index("static void fa_pack_k("):src.index("static int fa_run_o(")]
    runo = src[src.index("static int fa_run_o("):src.index("static int uc_fa(void *const *argv")]
    runo = runo[:runo.index("#if UC_CTRLPKT")] + runo[runo.index("    /* v5 O: o row pm"):]
    runo = runo.replace("    uint16_t *QK = g_fa_qk[k], *V = g_fa_v[k], *O = g_fa_o[k];", "    uint16_t *O = g_fa_o[k];")
    return HARNESS.replace("@SRC@", helpers + packs + runo)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
@pytest.mark.parametrize("N,M,D", [(1024, 1024, 40), (1024, 77, 40), (256, 256, 80), (256, 77, 80), (64, 64, 160),
                                   (64, 77, 160)])
def test_fa_strided_packs(tmp_path, N, M, D):
    (tmp_path / "t.cc").write_text(_src())
    subprocess.run(["g++", "-O1", "-std=c++17", "-o", str(tmp_path / "t"), str(tmp_path / "t.cc")], check=True)
    out = subprocess.run([str(tmp_path / "t"), str(N), str(M), str(D)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"KLTEST qdiff=(\d+) kvdiff=(\d+) big=(\d+) odiff=(\d+) kvn=(\d+)", out)
    assert m, out
    qd, kd, big, od, kvn = (int(x) for x in m.groups())
    assert qd == 0 and od == 0, out          # Q pack and O unpack bit-identical
    assert kd * 1000 < kvn and big <= 2, out  # K: (k - mu) * s vs k*s - mean(k*s): fp32 rounding only
