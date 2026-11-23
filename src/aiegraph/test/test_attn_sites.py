###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GATT x86 regression test of aiegraph.attnsite (moved from example/aiegraph/unet_bk_sdm_tiny/test_attn_sites.py,
unet-decl P3: the attention site geometry / host / driver patch left the example for src/aiegraph).

1. test_attn_host: the attention host entry (attnsite.ATTN_HOST, verbatim) compiled with g++ on x86 against stubs;
   uc_kacc_launch is emulated in Python from the kacc port layouts (kacc_gemm.layout inverse, kacc_yidx). Every
   distinct attention geometry of the diffusion example (attnsite.bmm_geom): the site's f32 output == the PKC
   recipe kc_bf16out_pad64 (bf16 operands, keys centred over the real keys for QK^T, one bf16 SRS), and
   softmax(QK^T) == softmax of the fp32 reference (centring is softmax-invariant).
2. test_help: the example's build_unet.py --aie accepts conv,linear,attn (old choices kept).
3. test_driver (needs the fixture = an unpatched graph_driver.c + ag_work of a full convert and the GPAT
   partition; GATT_REQUIRE_FIXTURE=1 makes a missing fixture a failure): part_attn.json -> 36 attn sites
   (18 groups) via attnsite.attn_sites, all patched by attnsite.patch_driver to aiednn_aie_attn with PROF
   op=aie_host_attn; the kincore source with attnsite.attn_kernel_parts has aiednn_aie_attn + aie_kacc_pack.h +
   the attn gemm_kacc instances. (The old (a) part -- the conv,linear patch byte-identical to the example's
   unet_aie.py at GATT_BASE -- went with the example's map_layers; d5e9f15 has it.)
4. test_<fn>_rejects_*: counter-examples of every attnsite planner.
Run: python3 -m pytest -q src/aiegraph/test/test_attn_sites.py   (gcc/g++, numpy, onnx)
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EX = ROOT / "example" / "aiegraph" / "unet_bk_sdm_tiny"   # build_unet.py, kacc_gemm.py, the x86 harness
sys.path.insert(0, str(EX))
sys.path.insert(0, str(ROOT / "src"))
import aiednn  # noqa: E402
import kacc_gemm as kg  # noqa: E402
from aiegraph import attnsite, kincore  # noqa: E402
from aiegraph.cnn import kacc_geom  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

#: the conv host glue whose uc_dlt typedef the harness reuses
HOST_CC = (aiednn.template("conv2d/tile64_conv_host.cc") + aiednn.template("elementwise/epilogue_bias_residual.cc")
           + aiednn.template("conv2d/tile64_conv_entry.cc"))

FIX = Path(os.environ.get("GATT_FIXTURE", "/scratch/staff/huaj/tmp/unet_m2/s8/gatt/fixture"))
PATT = Path(os.environ.get("GATT_PART_ATTN", "/scratch/staff/huaj/tmp/unet_m2/s7/gpat/part_attn.json"))
PALL = Path(os.environ.get("GATT_PART_ALL", "/scratch/staff/huaj/tmp/unet_m2/s6/gpa/part_all.json"))
ONNX = Path(os.environ.get("GATT_ONNX", "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx"))
BASE = os.environ.get("GATT_BASE", "85ec4ff")

HARNESS = (HERE / "data" / "kacc_bmm_host_x86_harness.cc").read_text(encoding="utf-8")  # x86 stubs around ATTN_HOST


class DLT(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32),
                ("ndim", ctypes.c_int32), ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.c_void_p), ("byte_offset", ctypes.c_uint64)]


def build_lib():
    host = HOST_CC
    i0 = host.index("typedef struct {\n    void *data;")
    dlt = host[i0:host.index("} uc_dlt;", i0) + len("} uc_dlt;")]
    src = HARNESS.replace("@DLT@", dlt).replace("@ATTN@", attnsite.ATTN_HOST)
    d = Path(tempfile.mkdtemp(prefix="gatt_"))
    (d / "h.cc").write_text(src)
    subprocess.check_call(["g++", "-O2", "-shared", "-fPIC", "-Wall", "-Werror", "-Wno-unused-function",
                           "-Wno-unused-variable",   # aie_kacc_pack.h x86 path: glen, akp_unpad_cfg (NEON-only users)
                           f"-I{ROOT / 'src/mlir/runtime'}", "-o", str(d / "h.so"), str(d / "h.cc")])
    L = ctypes.CDLL(str(d / "h.so"))
    L.bufs.restype = ctypes.c_void_p
    return L


def unlayout(p, R, kp, kch, wside):
    """inverse of kacc_gemm.layout: packed [R][kp] u16 -> natural [R][kp] u16"""
    t = p.reshape(R // 16, 16, kp // kch, kch).transpose(0, 2, 1, 3)          # blk j row q
    t = t.reshape(R // 16, kp // kch, 2, kch // 8, 8, 8)                         # blk j hi kb (lo kk | kk lo)
    t = t.transpose(0, 2, 4, 1, 3, 5) if wside else t.transpose(0, 2, 5, 1, 3, 4)  # blk hi lo j kb kk
    return np.ascontiguousarray(t).reshape(R, kp)


def yidx(m, n, Ml, Nl, tr):  # kacc device output index (test_aie_kacc_pack.yidx), numpy-vectorised
    r, c = n & 15, m & 15
    if not tr:
        q = Ml // 4
        cc = m % q
        wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7)
        return ((n & ~15) + wi // q) * Ml + (m - cc) + wi % q
    q = Nl // 4
    cc = n % q
    wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7)
    return ((m & ~15) + wi // q) * Nl + (n - cc) + wi % q


def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def run_site(L, H, M, K, N, qk, seed):
    g = attnsite.bmm_geom(H, M, K, N)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((H, M, K)).astype(np.float32)
    W = (rng.standard_normal((H, N, K)) + (1.5 if qk else 0.0)).astype(np.float32)   # QK: keys with a common offset
    Y = np.full((H, M, N), np.nan, np.float32)
    kw = np.ctypeslib.as_array(ctypes.cast(L.bufs(0), ctypes.POINTER(ctypes.c_uint16)), (2 << 22,))
    kx = np.ctypeslib.as_array(ctypes.cast(L.bufs(1), ctypes.POINTER(ctypes.c_uint16)), (2 << 22,))
    ky = np.ctypeslib.as_array(ctypes.cast(L.bufs(2), ctypes.POINTER(ctypes.c_float)), (1 << 21,))
    seen = []

    def launch(ch, k, px, ow, ox, oy):   # emulated gemm_kacc instance (ch, k, px): Y[ch][px] = W[ch,:] . X[px,:]
        geo = kacc_geom(ch, k, px)
        kp, kch, tr = geo["kp"], geo["kch"], geo["tr"]
        wn = f32(unlayout(kw[2 * ow:2 * ow + ch * kp].copy(), ch, kp, kch, True)).astype(np.float64)
        xn = f32(unlayout(kx[2 * ox:2 * ox + px * kp].copy(), px, kp, kch, False)).astype(np.float64)
        y = (wn @ xn.T).astype(np.float32)                                  # [ch][px]
        pm, cn = np.meshgrid(np.arange(px), np.arange(ch), indexing="xy")   # cn [ch][px] = ch index, pm = px index
        ky[oy + yidx(pm, cn, px, ch, tr)] = y
        seen.append((ch, k, px))

    cb = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long,
                          ctypes.c_long)(launch)
    L.set_cb(cb)
    shp = [(ctypes.c_int64 * 3)(*s) for s in (X.shape, W.shape, Y.shape)]
    dl = [DLT(a.ctypes.data, 1, 0, 3, 2, 32, 1, s, None, 0) for a, s in zip((X, W, Y), shp)]
    args = (ctypes.c_void_p * 3)(*[ctypes.addressof(d) for d in dl])
    n0 = L.launches()
    rc = L.site(0, args, int(qk), g["swap"], g["CHP"], g["PXP"], g["Kp"], g["kp"], g["kch"], g["splitpx"], g["S"],
                g["chs"], g["pxs"], g["tr"])
    nl = L.launches() - n0
    # reference: PKC kc_bf16out_pad64 recipe in numpy
    Wc = W - (W.astype(np.float64).mean(1, keepdims=True)).astype(np.float32) if qk else W
    bf = lambda a: f32(kg.bf16(a)).astype(np.float64)
    ref = f32(kg.bf16(np.einsum("hmk,hnk->hmn", bf(X), bf(Wc)).astype(np.float32)))
    c = cos(Y, ref)
    ulp = int(np.abs(kg.bf16(Y).astype(np.int64) - kg.bf16(ref).astype(np.int64)).max())
    ok = rc == 0 and nl == H * g["S"] and set(seen) == {(g["chs"], g["Kp"], g["pxs"])} and np.isfinite(Y).all() \
        and c >= 0.99999 and ulp <= 1
    extra = ""
    if qk:   # softmax over keys vs the fp32 reference (no centring, no bf16)
        cs = cos(softmax(Y), softmax(np.einsum("hmk,hnk->hmn", X.astype(np.float64), W.astype(np.float64))))
        ok = ok and cs >= 0.999
        extra = f" softmax_cos={cs:.6f}"
    print(f"GATTHOST H={H} M={M} K={K} N={N} {'qk' if qk else 'pv'} swap={g['swap']} ch={g['chs']} k={g['Kp']} "
          f"kp={g['kp']} px={g['pxs']} S={g['S']} tr={g['tr']} rc={rc} launches={nl} cos={c:.7f} max_ulp={ulp}"
          f"{extra} {'ok' if ok else 'FAIL'}")
    return ok


# every distinct attention bmm (M, K, N, role) of bk-sdm-tiny (part_attn.json, 36 bmm = 18 groups)
CASES = ((1024, 40, 1024, 1), (1024, 1024, 40, 0), (1024, 40, 77, 1), (1024, 77, 40, 0), (256, 80, 256, 1),
         (256, 256, 80, 0), (256, 80, 77, 1), (256, 77, 80, 0), (64, 160, 64, 1), (64, 64, 160, 0),
         (64, 160, 77, 1), (64, 77, 160, 0))


def test_attn_host():
    L = build_lib()
    res = [run_site(L, 2 if M * N >= 1 << 20 else 8, M, K, N, qk, i) for i, (M, K, N, qk) in enumerate(CASES)]
    assert all(res), res


def test_help():
    r = subprocess.run([sys.executable, str(EX / "build_unet.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "conv,linear,attn" in r.stdout and "conv,linear}" not in r.stdout, r.stdout
    assert "{none,conv,conv,linear,conv,linear,attn}" in r.stdout.replace("\n", "").replace(" ", ""), r.stdout


def _fixture():
    need = [FIX / "graph_driver.c", FIX / "ag_work", PATT, PALL, ONNX]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        if os.environ.get("GATT_REQUIRE_FIXTURE") == "1":
            raise AssertionError(f"fixture missing: {miss}")
        print(f"GATTDRV SKIP (fixture missing: {miss})")
        return None
    return FIX


KATTN = 'extern "C" int aiednn_aie_attn('
KPACK = next(ln for ln in aiednn.template("attention/kacc_bmm_host.cc").splitlines() if ln.endswith('"aie_kacc_pack.h"'))
SPEC = '"out x"'   # stand-in for the carrier's offload prototype spec


def test_driver():
    fx = _fixture()
    if fx is None:
        return
    drv = (fx / "graph_driver.c").read_text()
    work = fx / "ag_work"
    pa = json.loads(PATT.read_text())
    att = attnsite.attn_sites(pa, work, onnx=ONNX)
    natt = sum(1 for L in pa["layers"] if L.get("cls") == "attn")
    t, n = attnsite.patch_driver(drv, att)
    shapes = attnsite.attn_shapes(att)
    k = kincore.write_source(Path(tempfile.mkdtemp(prefix="gatt_attn_")), SPEC, shapes, entry="aiek_attn",
                             warm=True, ctrlpkt="aot", **attnsite.attn_kernel_parts(att)).read_text()
    nsite, ntag = t.count("    aiednn_aie_attn("), len(re.findall(r'GRAPH_PROF_END\(\d+, "aie aiednn_aie_attn \d+"\);', t))
    groups = len({c["group"] for c in att})
    insts = sum(1 for s in shapes if f"{kincore.kacc_name(*s)}<<<mesh>>>(" in k)
    print(f"GATTDRV attn partition_attn={natt} mapped={len(att)} groups={groups} sites={nsite} prof_tags={ntag} "
          f"prof_attn_branch={int('op=aie_host_attn%s' in t)} batch_matmul_left={t.count('_nn_batch_matmul(args')} "
          f"kernel_attn={int(KATTN in k)} pack_inc={int(KPACK in k)} "
          f"attn_instances={insts}/{len(shapes)} launches={sum(c['launches'] for c in att)}")
    assert natt == 36 and len(att) == 36 and groups == 18 and nsite == 36 and ntag == n == 36
    assert "op=aie_host_attn%s" in t and t.count("_nn_batch_matmul(args") == 0
    assert KATTN in k and KPACK in k and 'export    aiednn_aie_attn' in k and insts == len(shapes)


# ── counter-examples (static_check: test_<fn>_rejects*) ─────────────────────────────────────────────────────

def test_kacc_plan_splits_and_rejects_no_fit():
    assert attnsite.kacc_plan(64, 64, 64) == ("none", 1, 64, 64)
    mode, S, ch, px = attnsite.kacc_plan(1024, 1024, 4096)
    assert mode in ("px", "ch") and S > 1 and kacc_geom(ch, 1024, px)


def test_kacc_plan_rejects_unaligned():
    with pytest.raises(AiegraphError, match="no split"):
        attnsite.kacc_plan(40, 64, 64)          # ch not a multiple of 64: no instance, no split


def test_bmm_geom_rejects_empty_dim():
    with pytest.raises(AiegraphError, match="must all be >= 1"):
        attnsite.bmm_geom(8, 0, 64, 64)


def test_bmm_geom_pads_and_swaps():
    g = attnsite.bmm_geom(8, 1024, 40, 1024)    # square, big: swapped, K padded to 64 (and past 64 / 128)
    assert g["swap"] == 1 and g["Kp"] % 64 == 0 and g["kp"] // 2 not in (32, 64)
    g = attnsite.bmm_geom(8, 256, 80, 77)       # N = 77 keys padded to 128, not swapped
    assert g["swap"] == 0 and g["CHP"] >= 128 and g["PXP"] == 256


def test_attn_sites_rejects_partition_without_attn(tmp_path):
    with pytest.raises(AiegraphError, match="no attn rows"):
        attnsite.attn_sites({"layers": [{"node": "c0", "cls": "conv"}]}, tmp_path)


def test_attn_shapes_rejects_rows_without_geometry():
    assert attnsite.attn_shapes([{"geo": attnsite.bmm_geom(8, 256, 80, 256)}] * 2) == \
        attnsite.attn_shapes([{"geo": attnsite.bmm_geom(8, 256, 80, 256)}])
    with pytest.raises(AiegraphError, match="without geometry"):
        attnsite.attn_shapes([{"onnx": "/qk/MatMul"}])


def test_attn_kernel_parts_rejects_no_site():
    with pytest.raises(AiegraphError, match="no attention site"):
        attnsite.attn_kernel_parts([])
    kw = attnsite.attn_kernel_parts([{"geo": attnsite.bmm_geom(8, 256, 80, 256)}])
    assert kw["exports"] == ("aiednn_aie_attn",) and KATTN in kw["host_entries"] and len(kw["bufs"]) == 1


def test_prof_c_attn_rejects_missing_anchor(monkeypatch):
    from aiegraph import boardmain
    assert "op=aie_host_attn" in attnsite.prof_c_attn() and "op=aie_host_attn" not in boardmain.prof_c()
    monkeypatch.setattr(boardmain, "_rt", lambda name: "/* no branch anchor */\n")
    with pytest.raises(AiegraphError, match="anchor"):
        attnsite.prof_c_attn()


_DRV = """static uint8_t kernel_ws[64];
void graph_bf16_expand(float *d, const uint16_t *src, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) d[i] = (uint32_t)src[i] << 16;
}
float *graph_run(void) {
    args[0].v_handle = &tensors[3];
    args[1].v_handle = &tensors[4];
    args[2].v_handle = &tensors[5];
    GRAPH_PROF_BEGIN(7);
    tvmgen_default_fused_nn_batch_matmul(args, codes, 3, 0, 0, 0);
    GRAPH_PROF_END(7, "tvmgen_default_fused_nn_batch_matmul");
    return 0;
}
"""


def _site_row(**kw):
    r = dict(kind="attn", eids=[3, 4], yeid=5, symbol="tvmgen_default_fused_nn_batch_matmul", nargs=3, cid=9,
             onnx="/attn/MatMul", node=12, role="qk", group=0, geo=attnsite.bmm_geom(8, 256, 80, 256))
    r.update(kw)
    return r


def test_patch_driver_attn_site():
    t, n = attnsite.patch_driver(_DRV, [_site_row()])
    assert n == 1 and t.count("    aiednn_aie_attn(9, args, 3, 1, ") == 1 and "batch_matmul(args" not in t
    assert 'GRAPH_PROF_END(7, "aie aiednn_aie_attn 9");' in t and attnsite.DECL_ATTN in t
    assert "up_pending_exp = 0;" in t and "op=aie_host_attn" in t


def test_patch_driver_rejects_twice_patched_and_wrong_callee():
    t, _ = attnsite.patch_driver(_DRV, [_site_row()])
    with pytest.raises(AiegraphError, match="already patched"):
        attnsite.patch_driver(t, [_site_row()])
    with pytest.raises(AiegraphError, match="expected tvmgen_default_fused_nn_softmax"):
        attnsite.patch_driver(_DRV, [_site_row(symbol="tvmgen_default_fused_nn_softmax")])
    with pytest.raises(AiegraphError, match="group steps"):   # FA rows that aiegraph did not place
        attnsite.patch_driver(_DRV, [_site_row(fa=0)])
