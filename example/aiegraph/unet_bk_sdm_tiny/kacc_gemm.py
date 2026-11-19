#!/usr/bin/env python3
"""PDK: kacc_bench.sh --gemm -- the K-in-core gemm_kacc kernel (aiednn gemm/gemm_kacc_kernel.cc template, unet_aie.KACC_KERNEL) as a dense GEMM.

  kacc_gemm.py <op_golden matmul dir> <out dir>   -> <out>/kacc_gemm.cc + kacc_{w,x,y}.bin + kacc_data.h

Golden dir: a.npy [1, M, K] f32, b.npy [K, N] f32, y.npy [1, M, N] f32, meta.json (op_golden p0 MatMul).
Mapping onto gemm_kacc(ch, k, px) (Y[ch][px] = W[ch, :] . X[px, :], see unet_aie.kacc_geom):
  ch = N (out features, W = b^T [N][K]), px = M padded to a multiple of 64 (zero rows, not compared), k = K.
If one instance does not fit the core (kacc_geom asserts: the fp32 accumulator window is too big), the GEMM is
split into S launches of one instance, along px first (X rows contiguous), then along ch (W rows contiguous).
Host side (main): one warm-up pass of all launches (kernel load + first-launch config), then the timed pass;
[BENCH] aie_wall_ms is the timed pass. Numerics: bf16 operands, fp32 accumulation over all of K in the core,
no bias; KCMP cos is against the fp32 golden y.npy.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_SRC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../src"))
if _SRC not in _sys.path:  # C text lives in src/aiednn/<cat>/ (kernels) and src/aiegraph/runtime/
    _sys.path.insert(0, _SRC)
from aiednn import render as _render, runtime_source as _rt, template as _tpl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unet_aie as ua  # noqa: E402


def bf16(a):  # round to nearest even
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def fits(ch, k, px):
    try:
        ua.kacc_geom(ch, k, px)
        return True
    except AssertionError:
        return False


def plan(N, K, Mp):
    if fits(N, K, Mp):
        return "none", 1, N, Mp
    for s in (2, 4, 8, 16):
        if Mp % (64 * s) == 0 and fits(N, K, Mp // s):
            return "px", s, N, Mp // s
    for s in (2, 4, 8, 16):
        if N % (64 * s) == 0 and fits(N // s, K, Mp):
            return "ch", s, N // s, Mp
    raise SystemExit(f"[kacc_gemm] no split of N={N} K={K} M={Mp} fits gemm_kacc")


def layout(rows_bf16, kp, kch, wside):
    """natural [R][kp] bf16 -> kacc chunk layout: 16-row block, chunk j at j*kch of each row, in-chunk flat
    f = hi*8*kch + kb*64 + (W: mm*8 + kk | X: kk*8 + nn)  (unet_aie.kacc_wlayout / uc_conv im2col table)"""
    R = rows_bf16.shape[0]
    t = rows_bf16.reshape(R // 16, 2, 8, kp // kch, kch // 8, 8)       # blk hi lo j kb kk
    t = t.transpose(0, 3, 1, 4, 2, 5) if wside else t.transpose(0, 3, 1, 4, 5, 2)
    t = t.reshape(R // 16, kp // kch, 16, kch).transpose(0, 2, 1, 3)  # blk row j q
    return np.ascontiguousarray(t).reshape(R, kp)


def main():
    g, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.load(open(g / "meta.json"))
    assert meta["op"] == "MatMul", meta["op"]
    a = np.load(g / "a.npy").astype(np.float32)
    b = np.load(g / "b.npy").astype(np.float32)
    y = np.load(g / "y.npy")
    assert y.dtype == np.float32, y.dtype
    ysha = hashlib.sha256(y.tobytes()).hexdigest()[:12]
    assert meta["y_sha256"].startswith(ysha), (ysha, meta["y_sha256"])
    assert a.ndim == 3 and a.shape[0] == 1 and b.ndim == 2 and a.shape[2] == b.shape[0], (a.shape, b.shape)
    M, K = a.shape[1:]
    N = b.shape[1]
    Mp = -(-M // 64) * 64
    mode, S, chs, pxs = plan(N, K, Mp)
    geo = ua.kacc_geom(chs, K, pxs)
    kch, kp, tr = geo["kch"], geo["kp"], geo["tr"]
    X = np.zeros((Mp, kp), np.float32)
    X[:M, :K] = a[0]
    W = np.zeros((N, kp), np.float32)
    W[:, :K] = b.T
    Xb, Wb = bf16(X), bf16(W)
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    ref = f(Xb[:M]) @ f(Wb).T
    yr = y.reshape(M, N).astype(np.float64)
    cos = (ref * yr).sum() / np.sqrt((ref * ref).sum() * (yr * yr).sum())
    gmac = M * N * K / 1e9
    print(f"[kacc_gemm] M={M} Mp={Mp} K={K} N={N} split={mode} S={S} instance=(ch {chs}, k {K}, px {pxs}) "
          f"kch={kch} kp={kp} tr={tr} ntile={geo['ntile']} sub={geo['S']} nkc={geo['nkc']} l1={geo['l1']} "
          f"gmac={gmac:.4f} bf16-gemm vs golden cos={cos:.6f} y_sha256={ysha}")
    layout(Wb, kp, kch, True).astype("<u2").tofile(out / "kacc_w.bin")
    layout(Xb, kp, kch, False).astype("<u2").tofile(out / "kacc_x.bin")
    y.astype("<f4").reshape(-1).tofile(out / "kacc_y.bin")
    data = "op_golden/" + "/".join(str(g).rstrip("/").split("/")[-2:])
    data = data[:data.rindex("/") + 5]  # op_golden/<set>/cNNN
    tag = data.split("/")[-1] + "-kacc-gemm"
    (out / "kacc_data.h").write_text(_render("gemm/kacc_bench_data_gemm.h", CHS=chs, DATA=data, GMAC=f"{gmac:.6f}", GOLDEN=g, K=K, KP=kp, M=M, MP=Mp, N=N, OUT=out, PXS=pxs, S=S, SPLIT_PX=int(mode == 'px'), TAG=tag, TR=tr, Y_SHA=ysha))
    name = ua.kacc_name(chs, K, pxs)
    args = "Xs, Ws, Ys" if tr else "Ws, Xs, Ys"
    src = (ua.CTRLPKT_PRAGMAS + _tpl("gemm/kacc_bench_prelude_gemm.cc") + ua.KACC_KERNEL + "\n" +
           HOST.replace("@NAME@", name).replace("@ARGS@", args))
    (out / "kacc_gemm.cc").write_text(src)
    print(f"[kacc_bench] data={data} tag={tag} kernel={name} launches_per_pass={S}")


# KACC_HOST_FILE=<path>: an alternative bench host (same @NAME@/@ARGS@ fields), e.g. script/perf/p0_kacc_host.cc
HOST = (Path(_os.environ["KACC_HOST_FILE"]).read_text() if _os.environ.get("KACC_HOST_FILE")
        else _tpl("gemm/kacc_bench_gemm_host.cc"))

def main_bmm(g, out):
    """PBK: attention bmm (a.npy [1,H,M,K], b.npy [1,H,K,N]) -- B and A packed by the HOST at run time
    (src/mlir/runtime/aie_kacc_pack.h), M/K/N zero-padded to 64 (aiegraph.gemmpad.pad_mkn), H heads x S launches
    of one gemm_kacc instance; numerics bf16 in, fp32 acc over all of K, one bf16 RNE SRS on the output."""
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from aiegraph.gemmpad import pad_mkn
    meta = json.load(open(g / "meta.json"))
    assert meta["op"] == "MatMul", meta["op"]
    a = np.load(g / "a.npy").astype(np.float32)
    b = np.load(g / "b.npy").astype(np.float32)
    y = np.load(g / "y.npy")
    assert y.dtype == np.float32, y.dtype
    ysha = hashlib.sha256(y.tobytes()).hexdigest()[:12]
    assert meta["y_sha256"].startswith(ysha), (ysha, meta["y_sha256"])
    assert a.ndim == 4 and b.ndim == 4 and a.shape[:2] == b.shape[:2] == (1, a.shape[1]) and a.shape[3] == b.shape[2]
    H, M, K = a.shape[1:]
    N = b.shape[3]
    Mp, Kp, Np = pad_mkn(M, K, N)
    # orientation: the kernel computes Y[ch][px] = W[ch,:] . X[px,:]. Default ch = N (W = B^T), px = M (X = A).
    # swap (big square outputs, c004 QK^T 1024x1024): ch = M (W = A), px = N (X = B^T) -- the device's 8x8 tiles
    # are then row-major in Y, so the host un-tiling needs no transpose (akp_unpad otr=1).
    swap = int(Mp == Np and H * M * N >= (4 << 20))
    CHR, PXR = (M, N) if swap else (N, M)
    CHP, PXP = (Mp, Np) if swap else (Np, Mp)
    mode, S, chs, pxs = plan(CHP, Kp, PXP)
    geo = ua.kacc_geom(chs, Kp, pxs)
    # GAP (PBK, c013 256x80x256): when the a and b ports have the same shape (ch == px, both <rows, kp/2>) aiehlc's
    # GemmSpace derivation sizes the b ping/pong window as the a block (1 KiB instead of {px/4, tkw} = 4 KiB) and
    # the kernel link fails ("cannot find free area ... buf_in_ping_1"). Workaround: one more zero 64-row ch block.
    while geo["arows"] == geo["brows"] and mode == "none":
        CHP += 64
        mode, S, chs, pxs = plan(CHP, Kp, PXP)
        geo = ua.kacc_geom(chs, Kp, pxs)
    # PBK board result: every build whose b-port BD aiehlc emitted as 2-dim + iteration (short K walk: kp/2 = 32 or
    # 64 words) came back wrong (c013 kp 128 cos 0.645, c004 kp 64 / 128 cos 0.315 / 0.540); the 3-dim BD builds
    # (kp/2 = 96, 144, 512) are right. AIE time is ~180 us per a-tile iteration (host BD programming), not bytes,
    # so pad K with zero 64-blocks until kp/2 is outside {32, 64} (pad tiles are never re-packed: the device
    # buffers are zeroed once); kacc_bench.sh --bmm rejects any iter BD left in host.cc.
    while geo["kp"] // 2 in (32, 64):
        Kp += 64
        geo = ua.kacc_geom(chs, Kp, pxs)
    kch, kp, tr = geo["kch"], geo["kp"], geo["tr"]
    # x86 self-check of the recipe (what the AIE should reach): bf16 operands, fp64 acc, bf16 out
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    ref = f(bf16((f(bf16(a[0])) @ f(bf16(b[0]))).astype(np.float32)))
    yr = y.reshape(ref.shape).astype(np.float64)
    cos = (ref * yr).sum() / np.sqrt((ref * ref).sum() * (yr * yr).sum())
    gmac = H * CHP * PXP * Kp / 1e9
    print(f"[kacc_gemm] bmm H={H} M={M} K={K} N={N} swap={swap} pad(ch,K,px)=({CHP},{Kp},{PXP}) split={mode} S={S} "
          f"instance=(ch {chs}, k {Kp}, px {pxs}) kch={kch} kp={kp} tkw={geo['tkw']} nkc={geo['nkc']} tr={tr} l1={geo['l1']} launches={H * S} "
          f"gmac_padded={gmac:.4f} x86 bf16-recipe vs golden cos={cos:.6f} y_sha256={ysha}")
    np.ascontiguousarray(a[0]).astype("<f4").tofile(out / "kacc_a.bin")   # natural [H][M][K] f32: packed at run time
    np.ascontiguousarray(b[0]).astype("<f4").tofile(out / "kacc_b.bin")   # natural [H][K][N] f32: packed at run time
    y.astype("<f4").reshape(-1).tofile(out / "kacc_y.bin")
    data = "op_golden/" + "/".join(str(g).rstrip("/").split("/")[-2:])
    tag = data.split("/")[-1][:4] + "-kacc-bmm"
    (out / "kacc_data.h").write_text(_render("gemm/kacc_bench_data_bmm.h", CHP=CHP, CHR=CHR, CHS=chs, DATA=data, GMAC=f"{gmac:.6f}", GOLDEN=g, H=H, K=K, KCH=kch, KP=kp, M=M, N=N, OUT=out, PXP=PXP, PXR=PXR, PXS=pxs, S=S, SPLIT_PX=int(mode == 'px'), SWAP=swap, TAG=tag, TR=tr, Y_SHA=ysha))
    name = ua.kacc_name(chs, Kp, pxs)
    args = "Xs, Ws, Ys" if tr else "Ws, Xs, Ys"
    src = (ua.CTRLPKT_PRAGMAS + _tpl("gemm/kacc_bench_prelude_bmm.cc") +
           ua.KACC_KERNEL + "\n" + HOST_BMM.replace("@NAME@", name).replace("@ARGS@", args))
    (out / "kacc_gemm.cc").write_text(src)
    print(f"[kacc_bench] data={data} tag={tag} kernel={name} launches_per_pass={H * S}")


HOST_BMM = _tpl("gemm/kacc_bench_bmm_host.cc")


if __name__ == "__main__":
    if "--bmm" in sys.argv:
        sys.argv.remove("--bmm")
        main_bmm(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()
