###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.kincore -- one gemm_kacc launch per conv / linear layer, K accumulated in the core.

    from aiegraph import kincore
    kp = kincore.kplan(M, K, N)                       # conv: legal one-launch walk?
    S, pxs = kincore.kacc_split(N, K, Mp)             # linear: px split of one instance
    shapes, lin = kincore.plan_kincore(convs, linears)
    kcc = kincore.write_source(kdst, spec, shapes, entry=..., lin=lin, warm=True, ctrlpkt="aot-kco")

Generic over the model: every rule is keyed on the GEMM shape (ch = out channels N, K, px = pixels / rows),
never on a model or layer name.

  kplan         the analytic per-layer K-in-core plan (4x4 mesh of 16x16 fp32 accumulators, K in KC chunks)
  kacc_split    a dense GEMM on ONE gemm_kacc instance: whole px range, else S equal px launches
  c_window      GAP-aiehlc-tilinglinalg-a45e5b: the C window of a core is its whole B range (see there)
  kacc_name     the launch callee ``gemm_kacc<...>`` of an instance (aiednn.gemm_kacc_launch)
  plan_kincore  conv / linear layer dicts -> launches, the distinct instances and the linear buffer rows
  kernel_source the aiek source: ctrlpkt pragmas, gemm_kacc template, kacc host glue with one dispatch per
                instance, the linear host entry and optional extra parts (prologue / kernels / host entries)
  write_source  kernel_source -> <kdst>/<entry>.cc + the carrier's x86 reference <entry>_ref.c
"""
from __future__ import annotations

from pathlib import Path

from aiednn import gemm_kacc_launch, path as _tplpath, template as _tpl

from .cnn import kacc_geom
from .errors import AiegraphError

_STAGE = "kincore"
PORT_BYTES = 29696                   # aiehlc port-buffer region (trace_routing kPortBufferBytes)
MESH_R, MESH_C = 4, 4
TM = TN = 16                         # per-core output tile (fp32 acc 16x16 = 1 KiB)
KC = 64                              # K chunk per core loop trip
ACC_MAX, PORT_MAX, L1_MAX = 1024, 8192, 57344
SHIM_BDS, TILE_BDS = 16, 16
WRAP_MAX, ITER_MAX, STRIDE_MAX = 1023, 64, 1 << 20
BF16, FP32 = 2, 4
SPLITS = (2, 4, 8, 16)
# exports of the kacc host glue, in AIEK_SPEC order; the linear entry goes after STATS when there are linears
EXPORTS_HEAD = ("aiednn_aie_conv", "aiednn_aie_stats")
EXPORTS_TAIL = ("aiednn_aie_launch_stats", "aiednn_aie_host_split_site", "aiednn_aie_host_split_line",
                "aiednn_aie_split_stats")
EXPORT_LINEAR = "aiednn_aie_linear"
_CONV_ENTRY = 'extern "C" int aiednn_aie_conv(int cid'
_DEAD = ("static float g_uc_acc[UC_MAX_ACC] __attribute__((aligned(64))); // .bss, never malloc\n",
         "static int32_t *g_uc_a = nullptr; // port a: bf16 [64][128]\n",
         "static int32_t *g_uc_b = nullptr; // port b: bf16 [UC_BR][128]\n",
         "static int32_t *g_uc_c = nullptr; // port c: f32  [64][UC_CW]\n")
__all__ = ["kplan", "kacc_split", "c_window", "kacc_name", "plan_kincore", "kernel_source", "write_source"]


def kplan(m: int, k: int, n: int) -> dict:
    """The K-in-core plan of one GEMM M x K x N (im2col conv: M pixels, K = C*KH*KW, N out channels).
    Per core a TM x TN fp32 accumulator; per launch the R x C mesh walks 64x64 output macro tiles, A panel i and
    B panel j streamed in KC chunks. One shim BD per macro-row (A), one per launch over the nt weight panels (B,
    iterated per macro-row): launches = ceil(mt / SHIM_BDS) * ceil(nt / ITER_MAX). ``legal`` is False (``why``
    lists the broken limits) when a dim is not tile aligned or a buffer / BD limit is exceeded."""
    why = []
    mt_rows, nt_cols = MESH_R * TM, MESH_C * TN
    if m % mt_rows:
        why.append(f"M%{mt_rows}")
    if n % nt_cols:
        why.append(f"N%{nt_cols}")
    if k % KC:
        why.append(f"K%{KC}")
    mt, nt, kch = -(-m // mt_rows), -(-n // nt_cols), -(-k // KC)
    acc = TM * TN * FP32
    a_core, b_core, c_core = TM * KC * BF16, KC * TN * BF16, TM * TN * BF16
    a_port, b_port = mt_rows * KC * BF16, KC * nt_cols * BF16
    l1 = 2 * (a_core + b_core + c_core) + acc
    if acc > ACC_MAX:
        why.append(f"acc{acc}>{ACC_MAX}")
    if max(a_port, b_port) > PORT_MAX:
        why.append(f"port{max(a_port, b_port)}>{PORT_MAX}")
    if l1 > L1_MAX:
        why.append(f"l1{l1}>{L1_MAX}")
    if 6 > TILE_BDS:
        why.append("tile_bds")
    for name, v in (("d0", KC * BF16 // 4), ("d1", mt_rows), ("d2", kch)):
        if v > WRAP_MAX:
            why.append(f"A.{name}{v}>{WRAP_MAX}")
    if k * BF16 // 4 > STRIDE_MAX:
        why.append("A.stride")
    if n * BF16 // 4 > STRIDE_MAX:
        why.append("B.stride")
    launches = -(-mt // SHIM_BDS) * -(-nt // ITER_MAX)
    return dict(m=m, k=k, n=n, mt=mt, nt=nt, k_chunks=kch, out_tiles_per_core=mt * nt,
                acc=acc, l1=l1, a_port=a_port, b_port=b_port, launches=launches,
                launches_k_only=mt * nt, legal=not why, why=why)


def _fits(ch: int, k: int, px: int) -> bool:
    try:
        kacc_geom(ch, k, px)
        return True
    except AiegraphError:
        return False


def kacc_split(N: int, K: int, Mp: int) -> tuple:
    """(S, px per launch) of a dense GEMM N x K x Mp on ONE gemm_kacc instance: the whole GEMM if (N, K, Mp) has
    a walk, else the smallest S in SPLITS with Mp % (64 S) == 0 whose (N, K, Mp / S) has one (X rows / Y blocks
    stay contiguous per launch). No such S -> AiegraphError."""
    if _fits(N, K, Mp):
        return 1, Mp
    for s in SPLITS:
        if Mp % (64 * s) == 0 and _fits(N, K, Mp // s):
            return s, Mp // s
    raise AiegraphError(_STAGE, f"kacc_split: no px split of N={N} K={K} M={Mp} fits gemm_kacc",
                        hint=f"px splits tried: {SPLITS}")


def c_window(ch: int, k: int, px: int) -> dict:
    """GAP-aiehlc-tilinglinalg-a45e5b shape rule. aiehlc takes tileN from the C port over the B block: a {16, 16}
    C block with a {px/4, tkw} B block makes it emit the N-sub-tiled walk (A re-sent px/64 times per K walk while
    the kernel reads it once: the launch hangs, no build error). So a core's C window (= its fp32 accumulator) is
    {16, cols/4}: the core's WHOLE B range, tileN == tileCols. When the ping-pong a / b / c port buffers of that
    window do not fit PORT_BYTES (px = 1024: c = 16 KiB) the launch runs transposed: b = W, cols = ch.
    Returns dict(tr, rows=16, cols, bytes, b_rows, tkw); AiegraphError when no orientation fits."""
    g = kacc_geom(ch, k, px)
    cols = g["brows"] // 4
    win = dict(tr=g["tr"], rows=16, cols=cols, bytes=16 * cols * FP32, b_rows=cols, tkw=g["tkw"])
    if 2 * (16 * g["tkw"] * 4 + cols * g["tkw"] * 4 + win["bytes"]) > PORT_BYTES:
        raise AiegraphError(_STAGE, f"gemm_kacc ({ch}, {k}, {px}): C window {16}x{cols} fp32 does not fit "
                            f"{PORT_BYTES} B in either orientation")
    return win


def kacc_name(ch: int, k: int, px: int) -> str:
    """the launch callee of instance (ch, k, px): gemm_kacc<template args> (aiednn.gemm_kacc_launch); the
    driver glue keys on this exact text"""
    return gemm_kacc_launch(ch, k, px, kacc_geom(ch, k, px))


def plan_kincore(convs: list, linears: list = ()) -> tuple:
    """Conv / linear layer dicts (``M`` pixels or rows, ``K``, ``N``) -> one launch per conv layer and S px launches
    per linear layer. Sets ``launches`` on every layer and ``S`` / ``pxs`` on the linears. A conv whose kplan is not
    one legal launch is refused (AiegraphError). Returns (instances [(N, K, M)] of the convs + [(N, K, pxs)] of the
    linears, sorted distinct; linear rows [(N, K, Mp, pxs)] for :func:`kernel_source`)."""
    for c in convs:
        kp = kplan(c["M"], c["K"], c["N"])
        if not kp["legal"] or kp["launches"] != 1:
            raise AiegraphError(_STAGE, f"kplan: layer {c.get('cid')} M={c['M']} K={c['K']} N={c['N']} "
                                f"legal={kp['legal']} launches={kp['launches']} why={kp['why']}",
                                hint="one launch per layer needs M, N multiples of 64 and K of 64")
        c["launches"] = kp["launches"]
    for c in linears:
        if "S" not in c:
            c["S"], c["pxs"] = kacc_split(c["N"], c["K"], -(-c["M"] // 64) * 64)
        c["launches"] = c["S"]
    shapes = sorted({(c["N"], c["K"], c["M"]) for c in convs} | {(c["N"], c["K"], c["pxs"]) for c in linears})
    lin = [(c["N"], c["K"], -(-c["M"] // 64) * 64, c["pxs"]) for c in linears]
    return shapes, lin


def _host_glue(shapes: list, lin: list, bufs: list) -> str:
    """uc_conv with ONE gemm_kacc launch per layer (no host K-sum): dispatch per instance; buffers hold the
    largest W / X / Y of the instances, the linears (all S px launches) and *bufs* [(w, x, y) elements]"""
    disp = "\n".join("    if (N == %d && K == %d && M == %d) {\n        %s<<<mesh>>>(%s, ky);\n        return 0;\n    }"
                     % (ch, k, px, kacc_name(ch, k, px), "kx, kw" if kacc_geom(ch, k, px)["tr"] else "kw, kx")
                     for ch, k, px in shapes)
    text = _tpl("conv2d/kacc_conv_host_head.cc") + _tpl("layout/neon_tr4_f32.cc") + _tpl("conv2d/kacc_conv_host.cc")
    mw = max([s[0] * kacc_geom(*s)["kp"] for s in shapes] + [n * kacc_geom(n, k, p)["kp"] for n, k, mp, p in lin] +
             [b[0] for b in bufs])
    mx = max([s[2] * kacc_geom(*s)["kp"] for s in shapes] + [mp * kacc_geom(n, k, p)["kp"] for n, k, mp, p in lin] +
             [b[1] for b in bufs])
    my = max([ch * px for ch, k, px in shapes] + [n * mp for n, k, mp, p in lin] + [b[2] for b in bufs])
    return text.replace("@DISPATCH@", disp).replace("@MAXW@", str(mw)).replace("@MAXX@", str(mx)) \
        .replace("@MAXY@", str(my))


def kernel_source(spec: str, shapes: list, *, entry: str, warm: bool = False, ctrlpkt: str = "off",
                  lin: list = (), bufs: list = (), exports: tuple = (), prologue: str = "", kernels: str = "",
                  host_entries: str = "") -> str:
    """The --kincore aiek source of the conv instances *shapes* [(ch, K, px)] and linear rows *lin*
    [(N, K, Mp, pxs)]; *spec* = the carrier prototype's AIEK_SPEC (gets the host exports before its out line).
    Extra parts of other layer kinds: *exports* (after the linear export), *prologue* (after the unit head),
    *kernels* (after the gemm_kacc template), *host_entries* (before the conv entry, after the linear one),
    *bufs* [(w, x, y)] buffer sizes of their gemm_kacc launches."""
    if ctrlpkt not in ("off", "aot", "aot-kco"):
        raise AiegraphError(_STAGE, f"ctrlpkt={ctrlpkt!r}", hint="off, aot or aot-kco")
    if not shapes and not lin and not bufs:
        raise AiegraphError(_STAGE, "kernel_source: no gemm_kacc instance (no conv, linear or extra buffer)")
    if spec.count('"out ') < 1:
        raise AiegraphError(_STAGE, "carrier AIEK_SPEC has no out line")
    names = list(EXPORTS_HEAD) + ([EXPORT_LINEAR] if lin else []) + list(exports) + list(EXPORTS_TAIL)
    spec = spec.replace('"out ', "".join(f'"export    {e}\\n"\n        ' for e in names) + '"out ', 1)
    head = (_tpl("gemm/kacc_unit_head.cc").replace("@E@", entry).replace("@SRC@", str(_tplpath("gemm/gemm_kacc_kernel.cc")))
            .replace("@NSHAPES@", str(len(shapes))))
    if ctrlpkt != "off":
        head = ctrlpkt_head(ctrlpkt) + head
    head += prologue
    ktext = (_tpl("gemm/gemm_kacc_kernel.cc") if shapes else "") + kernels
    from .boardmain import kt_select
    host = kt_select(_tpl("conv2d/tile64_conv_host.cc") + _tpl("elementwise/epilogue_bias_residual.cc")
                     + _tpl("conv2d/tile64_conv_entry.cc"), False)
    i0, i1 = host.index("static int uc_conv("), host.index('extern "C" int aiednn_aie_conv(')
    host = host[:i0] + _host_glue(shapes, lin, bufs) + host[i1:]
    if lin:
        j = host.index(_CONV_ENTRY)
        host = host[:j] + _tpl("gemm/kacc_linear_host.cc").lstrip("\n") + host[j:]
    if host_entries:
        j = host.index(_CONV_ENTRY)
        host = host[:j] + host_entries + host[j:]
    a0, a1 = host.index("static int uc_alloc(void) {"), host.index("static long uc_numel(")
    host = host[:a0] + host[a1:]
    for dead in _DEAD:
        if host.count(dead) != 1:
            raise AiegraphError(_STAGE, f"host template changed: {dead.strip()}")
        host = host.replace(dead, "")
    if "gemm_bf16<<<" in host or "g_uc_acc[(size_t)(mb" in host:
        raise AiegraphError(_STAGE, "host glue still has the 64x64-tile launch / host K-sum")
    on = ctrlpkt != "off"
    return head + ktext + "\n" + (host.replace("@MAXACC@", "1").replace("@WARM@", "1" if warm else "0")
                                  .replace("@CTRLPKT@", "1" if on else "0").replace("@PART_C0@", "2" if on else "3")
                                  .replace("@E@", entry).replace("@SPEC@", spec))


KACC_KERNEL = _tpl("gemm/gemm_kacc_kernel.cc")   # C++ template __global__: aiehlc instantiates it per launch


def ctrlpkt_head(ctrlpkt: str) -> str:
    """The aiehlc control-packet pragmas a kernel unit starts with: ``"off"`` none, ``"aot"`` the AOT plan
    (aiednn gemm/ctrlpkt_aot_pragmas.h), ``"aot-kco"`` the same + KERNELCONFIGOFFLOAD (the core programs its
    own DMA). Raises AiegraphError on any other mode."""
    if ctrlpkt not in ("off", "aot", "aot-kco"):
        raise AiegraphError(_STAGE, f"ctrlpkt={ctrlpkt!r}", hint="off, aot or aot-kco")
    if ctrlpkt == "off":
        return ""
    return _tpl("gemm/ctrlpkt_aot_pragmas.h") + (_tpl("gemm/ctrlpkt_kco_pragma.h") if ctrlpkt == "aot-kco" else "")


def ctrlpkt_npragmas(ctrlpkt: str) -> int:
    """Number of pragma lines :func:`ctrlpkt_head` puts in front of a unit (build log / plan check)."""
    return sum(1 for ln in ctrlpkt_head(ctrlpkt).splitlines() if ln.lstrip().startswith("#pragma"))


def write_source(kdst: Path, spec: str, shapes: list, *, entry: str, **kw) -> Path:
    """:func:`kernel_source` -> <kdst>/<entry>.cc, plus the carrier's x86 reference <entry>_ref.c
    (aiednn conv2d/conv_bf16_carrier_ref.c). Returns the .cc path."""
    if not entry.isidentifier():
        raise AiegraphError(_STAGE, f"entry {entry!r} is not a C identifier")
    text = kernel_source(spec, shapes, entry=entry, **kw)
    kdst = Path(kdst)
    kdst.mkdir(parents=True, exist_ok=True)
    kcc = kdst / f"{entry}.cc"
    kcc.write_text(text)
    (kdst / f"{entry}_ref.c").write_text(_tpl("conv2d/conv_bf16_carrier_ref.c").replace("@E@", entry))
    return kcc
