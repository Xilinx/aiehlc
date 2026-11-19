"""GATT x86 regression test (GAP-aiegraph-attn-site + GAP-aiegraph-prof-attn, sprint unet-m2-s8).

1. test_attn_host: the generated host C of an attention site (unet_aie.ATTN_HOST, verbatim) compiled with g++ on x86
   against stubs; uc_kacc_launch is emulated in Python from the kacc port layouts (kacc_gemm.layout inverse,
   kacc_yidx). Every distinct attention geometry of the bk-sdm-tiny UNet (unet_aie.bmm_geom): the site's f32 output
   == the PKC recipe kc_bf16out_pad64 (bf16 operands, keys centred over the real keys for QK^T, one bf16 SRS),
   and softmax(QK^T) == softmax of the fp32 reference (centring is softmax-invariant).
2. test_help: build_unet.py --aie accepts conv,linear,attn (old choices kept).
3. test_driver (needs the fixture = an unpatched graph_driver.c + ag_work of a full UNet convert and the GPAT
   partition; GATT_REQUIRE_FIXTURE=1 makes a missing fixture a failure):
   - part_attn.json -> 36 attn sites (18 groups) mapped to TVM nn_batch_matmul, all patched to aiednn_aie_attn with
     PROF op=aie_host_attn; the kernel source has aiednn_aie_attn + aie_kacc_pack.h + the attn gemm_kacc instances;
   - --aie conv,linear (part_all.json) output (graph_driver.c patch + aiek_unetconv.cc) byte-identical to the
     unet_aie.py of the base commit (GATT_BASE, default 85ec4ff = hl + gpat + bmk merge).
Run: python3 example/aiegraph/unet_bk_sdm_tiny/test_attn_sites.py   (gcc/g++, numpy, onnx)
"""
import ctypes
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import aiednn  # noqa: E402
import kacc_gemm as kg  # noqa: E402
import unet_aie as ua  # noqa: E402

FIX = Path(os.environ.get("GATT_FIXTURE", "/scratch/staff/huaj/tmp/unet_m2/s8/gatt/fixture"))
PATT = Path(os.environ.get("GATT_PART_ATTN", "/scratch/staff/huaj/tmp/unet_m2/s7/gpat/part_attn.json"))
PALL = Path(os.environ.get("GATT_PART_ALL", "/scratch/staff/huaj/tmp/unet_m2/s6/gpa/part_all.json"))
ONNX = Path(os.environ.get("GATT_ONNX", "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx"))
BASE = os.environ.get("GATT_BASE", "85ec4ff")

HARNESS = aiednn.template("attention/kacc_bmm_host_x86_harness.cc")  # x86 stubs around ATTN_HOST


class DLT(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32),
                ("ndim", ctypes.c_int32), ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.c_void_p), ("byte_offset", ctypes.c_uint64)]


def build_lib():
    host = ua.HOST_CC
    i0 = host.index("typedef struct {\n    void *data;")
    dlt = host[i0:host.index("} uc_dlt;", i0) + len("} uc_dlt;")]
    src = HARNESS.replace("@DLT@", dlt).replace("@ATTN@", ua.ATTN_HOST)
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
    g = ua.bmm_geom(H, M, K, N)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((H, M, K)).astype(np.float32)
    W = (rng.standard_normal((H, N, K)) + (1.5 if qk else 0.0)).astype(np.float32)   # QK: keys with a common offset
    Y = np.full((H, M, N), np.nan, np.float32)
    kw = np.ctypeslib.as_array(ctypes.cast(L.bufs(0), ctypes.POINTER(ctypes.c_uint16)), (2 << 22,))
    kx = np.ctypeslib.as_array(ctypes.cast(L.bufs(1), ctypes.POINTER(ctypes.c_uint16)), (2 << 22,))
    ky = np.ctypeslib.as_array(ctypes.cast(L.bufs(2), ctypes.POINTER(ctypes.c_float)), (1 << 21,))
    seen = []

    def launch(ch, k, px, ow, ox, oy):   # emulated gemm_kacc instance (ch, k, px): Y[ch][px] = W[ch,:] . X[px,:]
        geo = ua.kacc_geom(ch, k, px)
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
    r = subprocess.run([sys.executable, str(HERE / "build_unet.py"), "--help"], capture_output=True, text=True)
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


def _base_module():
    src = subprocess.run(["git", "-C", str(HERE), "show", f"{BASE}:example/aiegraph/unet_bk_sdm_tiny/unet_aie.py"],
                         capture_output=True, text=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="gatt_base_"))
    (d / "aiednn_aie_base.py").write_text(src.replace("Path(__file__).resolve().parent", f"Path({str(HERE)!r})"))
    spec = importlib.util.spec_from_file_location("aiednn_aie_base", d / "aiednn_aie_base.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


KATTN = 'extern "C" int aiednn_aie_attn('
KPACK = next(ln for ln in aiednn.template("attention/kacc_bmm_host.cc").splitlines() if ln.endswith('"aie_kacc_pack.h"'))
SPEC = '"out x"'   # stand-in for the carrier's offload prototype spec (same for both modules)


def _kernel(mod, d, shapes, lin, **kw):
    kcc = mod.write_kernel_kacc(d, SPEC, shapes, warm=True, ctrlpkt="aot", lin=lin, **kw)
    return kcc.read_text()


def test_driver():
    fx = _fixture()
    if fx is None:
        return
    from frontend.tvmrelay.arm_build import param_offsets
    drv = (fx / "graph_driver.c").read_text()
    work = fx / "ag_work"
    # (a) --aie conv,linear: byte-identical to the base commit's unet_aie.py
    base = _base_module()
    part = json.loads(PALL.read_text())
    out = {}
    # one mapping (map_layers / map_linears behave the same on a partition without attn layers), fed to both
    sel0, _ = ua.map_layers(ONNX, part, work, param_offsets)
    lin0 = ua.map_linears(ONNX, part, work, param_offsets)
    for name, mod in (("base", base), ("new", ua)):
        sel, lin = [dict(c) for c in sel0], [dict(c) for c in lin0]
        allsel = sel + lin
        t, n = mod.patch_driver(drv, [dict(c) for c in allsel])
        shapes = sorted({(c["N"], c["K"], c["M"]) for c in sel} | {(c["N"], c["K"], c["pxs"]) for c in lin})
        ltup = [(c["N"], c["K"], -(-c["M"] // 64) * 64, c["pxs"]) for c in lin]
        k = _kernel(mod, Path(tempfile.mkdtemp(prefix=f"gatt_{name}_")), shapes, ltup)
        out[name] = (t, n, k)
    same_drv, same_k = out["base"][0] == out["new"][0], out["base"][2] == out["new"][2]
    print(f"GATTDRV conv,linear sites={out['new'][1]} driver_identical={int(same_drv)} kernel_identical={int(same_k)}")
    assert same_drv and same_k and out["new"][1] == out["base"][1] and "aie_host_attn" not in out["new"][0]
    # (b) --aie conv,linear,attn on the GPAT partition
    pa = json.loads(PATT.read_text())
    sel, _ = ua.map_layers(ONNX, pa, work, param_offsets)
    lin = ua.map_linears(ONNX, pa, work, param_offsets)
    att = ua.map_attn(pa, work)
    natt = sum(1 for L in pa["layers"] if L.get("cls") == "attn")
    t, n = ua.patch_driver(drv, [dict(c) for c in sel + lin] + att)
    shapes = sorted({(c["N"], c["K"], c["M"]) for c in sel} | {(c["N"], c["K"], c["pxs"]) for c in lin} |
                    set(ua.attn_shapes(att)))
    k = _kernel(ua, Path(tempfile.mkdtemp(prefix="gatt_attn_")), shapes,
                [(c["N"], c["K"], -(-c["M"] // 64) * 64, c["pxs"]) for c in lin], att=att)
    nsite, ntag = t.count("    aiednn_aie_attn("), len(re.findall(r'GRAPH_PROF_END\(\d+, "aie aiednn_aie_attn \d+"\);', t))
    groups = len({c["group"] for c in att})
    insts = sum(1 for s in ua.attn_shapes(att) if f"{ua.kacc_name(*s)}<<<mesh>>>(" in k)
    print(f"GATTDRV attn partition_attn={natt} mapped={len(att)} groups={groups} sites={nsite} prof_tags={ntag} "
          f"prof_attn_branch={int('op=aie_host_attn%s' in t)} batch_matmul_left={t.count('_nn_batch_matmul(args')} "
          f"kernel_attn={int(KATTN in k)} pack_inc={int(KPACK in k)} "
          f"attn_instances={insts}/{len(ua.attn_shapes(att))} launches={sum(c['launches'] for c in att)}")
    assert natt == 36 and len(att) == 36 and groups == 18 and nsite == 36 and ntag == 36
    assert "op=aie_host_attn%s" in t and t.count("_nn_batch_matmul(args") == 0
    assert KATTN in k and KPACK in k and \
        'export    aiednn_aie_attn' in k and insts == len(ua.attn_shapes(att))
    # the conv / linear sites of the attn build are the same calls as without attn
    assert t.count("    aiednn_aie_conv(") == len(sel) and t.count("    aiednn_aie_linear(") == len(lin)


if __name__ == "__main__":
    test_help()
    print("GATTHELP ok")
    test_attn_host()
    test_driver()
    print("GATT ALL OK")
