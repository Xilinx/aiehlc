"""aiegraph.cnn (x86): the generated conv kernel source -- gemm_kacc instances AND the chunked host glue
(aiednn/conv2d/kacc_chunk_*.cc) -- compiled with g++ under the kemu kernel model (aiegraph.kemu._PRELUDE,
aie_api_x86emu.h, ASan) and run on NCHWc tensors; the output must match a numpy conv on the same
bf16-rounded activation / weight with the fused epilogue (bias, SiLU, residual).

Cases cover the geometry of a detector backbone / head: 3x3 stride-2 stem (K = 27 padded to 64), 3x3 s1,
1x1, depthwise 3x3 (s1, s2), a 4-channel head conv, a residual add after SiLU, a partial last pixel chunk,
a square-W instance (Np == Kk/2: one zero K chunk) -- and plan_conv's refusal of an unknown epilogue."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from aiegraph import cnn  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.kemu import _LAUNCH, _PRELUDE  # noqa: E402

SPEC = ('AIEK_SPEC(aiek_cnntest,\n    "kernel    aiek_cnntest\\n"\n    "op        conv2d\\n"\n'
        '    "out   float32 [1,1,1,1,4]         ofm\\n");\n')
STUBS = {
    "xil_cache.h": "#pragma once\n#include <stdint.h>\ntypedef uintptr_t UINTPTR;\ntypedef intptr_t INTPTR;\n"
                   "static inline void Xil_DCacheFlushRange(UINTPTR, UINTPTR) {}\n"
                   "static inline void Xil_DCacheInvalidateRange(UINTPTR, UINTPTR) {}\n",
}
GLUE = r"""
void __Runtime_phase_cycles(unsigned long long *c, unsigned int *n) { for (int i = 0; i < 4; i++) { c[i] = 0; n[i] = 0; } }
void __Runtime_wait_io_cycles(unsigned long long *c, unsigned int *n) { *c = 0; *n = 0; }
"""
MAIN = r"""
#include <vector>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r");
    int nargs, c[13];
    if (fscanf(f, "%d", &nargs) != 1) return 2;
    std::vector<uc_dlt> t(nargs);
    std::vector<void *> a(nargs);
    std::vector<std::vector<int64_t>> shp(nargs);
    char path[512];
    for (int i = 0; i < nargs; i++) {
        int nd;
        if (fscanf(f, "%d", &nd) != 1) return 2;
        shp[i].resize(nd);
        long n = 1;
        for (int d = 0; d < nd; d++) { long v; if (fscanf(f, "%ld", &v) != 1) return 2; shp[i][d] = v; n *= v; }
        if (fscanf(f, "%511s", path) != 1) return 2;
        float *p = (float *)calloc(n, 4);
        FILE *b = fopen(path, "rb");
        if (b) { if (fread(p, 4, n, b) != (size_t)n) return 3; fclose(b); }
        t[i] = uc_dlt{p, 1, 0, nd, 2, 32, 1, shp[i].data(), nullptr, 0};
        a[i] = &t[i];
    }
    for (int i = 0; i < 13; i++) if (fscanf(f, "%d", &c[i]) != 1) return 2;
    if (fscanf(f, "%511s", path) != 1) return 2;
    fclose(f);
    const int rc = cnn_aie_conv(7, a.data(), nargs, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10],
                                c[11], c[12]);
    long n = 1;
    for (auto v : shp[nargs - 1]) n *= v;
    FILE *o = fopen(path, "wb");
    fwrite(t[nargs - 1].data, 4, n, o);
    fclose(o);
    int launches = 0;
    cnn_aie_stats(&launches, 0, 0, 0);
    printf("RC %d LAUNCHES %d KEMU_ERR %d\n", rc, launches, kemu::g_err);
    return rc != 0 || kemu::g_err != 0;
}
"""


def bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


# (name, C, H, W, O, KH, s, p, dw, act, residual)
CASES = [
    ("stem3x3s2_k27", 3, 32, 32, 16, 3, 2, 1, 0, 1, 0),
    ("conv3x3s1_silu", 16, 20, 20, 32, 3, 1, 1, 0, 1, 0),
    ("conv1x1_res", 32, 20, 20, 32, 1, 1, 0, 0, 1, 1),
    ("conv1x1_head4", 16, 20, 20, 4, 1, 1, 0, 0, 0, 0),
    ("dw3x3s1_silu", 64, 20, 20, 64, 3, 1, 1, 1, 1, 0),
    ("dw3x3s2", 32, 16, 16, 32, 3, 2, 1, 1, 0, 0),
    ("conv3x3_128_res", 32, 8, 8, 128, 3, 1, 1, 0, 0, 1),  # K = 288 -> Kk 320, residual, no act
    ("square_w_128x256", 256, 8, 8, 128, 1, 1, 0, 0, 1, 0),  # Np=128, Kk=256: W square -> zero chunk
    ("ch256_mc512_res", 128, 24, 24, 256, 1, 1, 0, 0, 0, 1),  # Np == 256 == MC -> chunk 512, partial
]


def _info(C, H, W, O, KH, s, p, dw, act, res, nc):
    OH = (H + 2 * p - KH) // s + 1
    sym = ("tvmgen_default_fused_nn_contrib_" + ("depthwise_" if dw else "") + "conv2d_NCHWc_add" +
           ("_sigmoid_multiply" if act else "") + ("_add" if res else ""))
    return dict(symbol=sym, nargs=4 + res, N=O, M=OH * OH, K=(KH * KH if dw else C * KH * KH), groups=C if dw else 1,
                cid=nc, call=nc, onnx=f"/case{nc}/Conv", sh=s, sw=s, ph=p, pw=p, macs=0), OH


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.fail("no g++ on PATH (x86 test needs it)")
    t = tmp_path_factory.mktemp("cnn")
    plans = []
    for i, (_, C, H, W, O, KH, s, p, dw, act, res) in enumerate(CASES):
        plans.append(cnn.plan_conv(_info(C, H, W, O, KH, s, p, dw, act, res, i)[0], mc=64 if i == 1 else cnn.MC,
                                   split=3 if i % 2 else 1))
    kcc = cnn.write_kernel(t / "src", SPEC, plans, kprefix="cnn", entry="aiek_cnntest", warm=False,
                           selfcheck=True)
    text = _LAUNCH.sub(r"kemu::launch(\2, \1, ", kcc.read_text())
    (t / "kemu_prelude.h").write_text(_PRELUDE)
    for n, s in STUBS.items():
        (t / n).write_text(s)
    (t / "h.cc").write_text('#include "kemu_prelude.h"\n' + GLUE + text + MAIN)
    inc = [f"-I{t}", f"-I{ROOT / 'include'}", f"-I{ROOT / 'src/mlir/runtime'}"]
    r = subprocess.run(["g++", "-std=c++20", "-w", "-D__AIESIM__", "-fsanitize=address", "-g", "-O1", *inc,
                        str(t / "h.cc"), "-o", str(t / "h")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return t, plans


def _conv_ref(x, w, b, s, p, dw, act, r):
    C, H, W = x.shape
    O, _, KH, KW = w.shape
    OH, OW = (H + 2 * p - KH) // s + 1, (W + 2 * p - KW) // s + 1
    xp = np.zeros((C, H + 2 * p, W + 2 * p), np.float64)
    xp[:, p:p + H, p:p + W] = x
    y = np.zeros((O, OH, OW), np.float64)
    for kh in range(KH):
        for kw in range(KW):
            sl = xp[:, kh:kh + s * OH:s, kw:kw + s * OW:s]
            if dw:
                y += w[:, 0, kh, kw][:, None, None] * sl
            else:
                y += np.einsum("oc,chw->ohw", w[:, :, kh, kw].astype(np.float64), sl)
    y += b[:, None, None]
    if act:
        y = y / (1 + np.exp(-y))
    if r is not None:
        y += r
    return y


def _nchwc(a, cb):
    C, H, W = a.shape
    return np.ascontiguousarray(a.reshape(C // cb, cb, H, W).transpose(0, 2, 3, 1))


@pytest.mark.parametrize("i", range(len(CASES)), ids=[c[0] for c in CASES])
def test_chunk_conv_vs_numpy(harness, i):
    t, plans = harness
    name, C, H, W, O, KH, s, p, dw, act, res = CASES[i]
    pl = plans[i]
    rng = np.random.default_rng(i)
    x = rng.uniform(-2, 2, (C, H, W)).astype(np.float32)
    w = (rng.standard_normal((O, 1 if dw else C, KH, KH)) / np.sqrt(KH * KH * (1 if dw else C))).astype(np.float32)
    if pl.split == 1:
        w = bf16(w)  # weights.bin holds bf16 (AIEGRAPH_WDTYPE=bf16): the call's expanded fp32 weight is bf16-exact
    b = rng.standard_normal(O).astype(np.float32)
    OH = (H + 2 * p - KH) // s + 1
    r = rng.standard_normal((O, OH, OH)).astype(np.float32) if res else None
    cb = 4 if C % 4 == 0 else C
    ob = 4 if O % 4 == 0 else O
    d = t / name
    d.mkdir(exist_ok=True)
    _nchwc(x, cb).tofile(d / "x.bin")
    if dw:   # TVM depthwise NCHWc weight [C/cb][1][KH][KW][1][cb]
        w6 = w.reshape(O // ob, ob, 1, KH, KH).transpose(0, 2, 3, 4, 1).reshape(O // ob, 1, KH, KH, 1, ob)
        wshape = (O // ob, 1, KH, KH, 1, ob)
    else:    # TVM OIHW{i}{o} [O/ob][C/ib][KH][KW][ib][ob]
        w6 = w.reshape(O // ob, ob, C // cb, cb, KH, KH).transpose(0, 2, 4, 5, 3, 1)
        wshape = (O // ob, C // cb, KH, KH, cb, ob)
    np.ascontiguousarray(w6).astype(np.float32).tofile(d / "w.bin")
    b.reshape(O // ob, ob).astype(np.float32).tofile(d / "b.bin")
    args = [(5, (1, C // cb, H, W, cb), d / "x.bin"), (6, wshape, d / "w.bin"), (5, (1, O // ob, 1, 1, ob), d / "b.bin")]
    if res:
        _nchwc(r, ob).tofile(d / "r.bin")
        args.append((5, (1, O // ob, OH, OH, ob), d / "r.bin"))
    args.append((5, (1, O // ob, OH, OH, ob), d / "none"))
    consts = [pl.sh, pl.sw, pl.ph, pl.pw, pl.act, pl.dw, pl.Np, pl.Kk, pl.Kp, pl.mc, pl.kch, pl.split, pl.ks]
    lines = [str(len(args))] + [f"{nd} {' '.join(map(str, sh))} {f}" for nd, sh, f in args]
    lines += [" ".join(map(str, consts)), str(d / "y.bin")]
    (d / "case.txt").write_text("\n".join(lines) + "\n")
    rr = subprocess.run([str(t / "h"), str(d / "case.txt")], capture_output=True, text=True, timeout=600,
                        env=dict(os.environ, ASAN_OPTIONS="detect_leaks=0"))
    assert rr.returncode == 0, (rr.stdout + rr.stderr)[-3000:]
    assert f"LAUNCHES {pl.launches}" in rr.stdout, rr.stdout
    assert " BAD" not in rr.stdout, [l for l in rr.stdout.splitlines() if "[CHK]" in l]
    print("".join(l + "\n" for l in rr.stdout.splitlines() if "[CHK]" in l), end="")
    got = np.fromfile(d / "y.bin", np.float32).reshape(O // ob, OH, OH, ob).transpose(0, 3, 1, 2).reshape(O, OH, OH)
    xr = bf16(x) if pl.split == 1 else x     # split 3: hi/lo parts ~ the fp32 activation and weight
    ref = _conv_ref(xr.astype(np.float64), w.astype(np.float64), b.astype(np.float64), s, p, dw, act, r)
    cos = float((got * ref).sum() / np.sqrt((got.astype(np.float64) ** 2).sum() * (ref ** 2).sum()))
    err = float(np.abs(got - ref).max())
    print(f"{name}: plan split={pl.split} Np={pl.Np} Kk={pl.Kk} Kp={pl.Kp} mc={pl.mc} launches={pl.launches} "
          f"cos={cos:.9f} maxabs={err:.2e}")
    assert np.isfinite(got).all()
    tol = 1e-3 if pl.split == 1 else 2e-5
    assert cos >= 0.99999 and err < tol * max(1.0, float(np.abs(ref).max())), (cos, err)


def test_plan_refuses_unknown_epilogue():
    info, _ = _info(16, 8, 8, 16, 3, 1, 1, 0, 1, 0, 0)
    info["symbol"] = "tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_clip"
    with pytest.raises(AiegraphError):
        cnn.plan_conv(info)
    info, _ = _info(16, 8, 8, 16, 3, 1, 1, 0, 1, 0, 0)
    info["nargs"] = 5
    with pytest.raises(AiegraphError):
        cnn.plan_conv(info)


def test_ch_equal_px_takes_another_chunk():
    info, _ = _info(128, 20, 20, 256, 1, 1, 0, 0, 1, 0, 0)
    pl = cnn.plan_conv(info)
    assert pl.Np == 256 and pl.mc == 512 and pl.launches == 1
    with pytest.raises(AiegraphError):
        cnn.kacc_geom(256, 128, 256)


def test_split3_plan():
    pl = cnn.plan_conv(_info(64, 20, 20, 16, 3, 1, 1, 0, 1, 0, 0)[0], split=3)       # K = 576 -> 3 x 576
    assert (pl.split, pl.ks, pl.Kk, pl.Np) == (3, 576, 1728, cnn.NP_MIN_BIGK)
    with pytest.raises(AiegraphError):
        cnn.plan_conv(_info(64, 20, 20, 16, 3, 1, 1, 0, 1, 0, 0)[0], split=2)


def test_square_w_gets_zero_chunk():
    info, _ = _info(256, 16, 16, 128, 1, 1, 0, 0, 1, 0, 0)
    pl = cnn.plan_conv(info)
    assert (pl.Np, pl.Kk) == (128, 256) and pl.Kp == 256 + pl.kch


def test_small_k_padded_big_k_gets_rows():
    pl = cnn.plan_conv(_info(3, 32, 32, 16, 3, 2, 1, 0, 1, 0, 0)[0])        # K = 27
    assert (pl.Np, pl.Kk) == (64, cnn.KK_MIN)
    pl = cnn.plan_conv(_info(256, 20, 20, 16, 3, 1, 1, 0, 1, 0, 0)[0])      # K = 2304
    assert (pl.Np, pl.Kk) == (cnn.NP_MIN_BIGK, 2304)
