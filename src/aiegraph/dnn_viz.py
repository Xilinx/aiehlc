###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Model-flow visualization of a lowered aiednn graph: one self-contained HTML file."""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from .dnn_torch import DT_NP, _plan_memory

__all__ = ["write_html", "write_model_html", "build_data", "run_timed", "parse_board_log", "parse_routing",
           "find_routing", "gemm_geometry", "BLOCK"]

REPO = Path(__file__).resolve().parents[2]

#: the AIE GEMM block of src/aietensorop/gemm (gemm.h): C[M][N] s32 = A[M][K] s8 x B[N][K]^T s8
BLOCK = dict(M=128, N=128, K=512, a_stage=128 * 512, b_stage=512 * 512, c_stage=128 * 128 * 4,
             mesh_rows=4, mesh_cols=4, core_tile=(32, 32), partition="{0, 8, 0, 7}",
             source="src/aietensorop/gemm/gemm.h, gemm_uk.cc")

#: the GEMM megakernel's modes (src/aietensorop/gemm/gemm_uk.cc): opcode in byte 0 of B's row 127,
#: what the cores do, and the aie_modes_cfg.h macro (None: always compiled in). The AIE_MODE_OP_*
#: opcodes are read from src/aietensorop/ops/aie_modes_*.h (_mode_table).
MODES = (("GEMM", "0x00 / 0xA0", None, "int8 GEMM block, int32 C; 0xA0 = accumulate only (K > 512: the sum stays "
                                         "on the core for the next launch; d[2] bit 0 continues from it)"),
         ("GEMMQ", "0xF1", "GEMMQ", "GEMM + integer requantize on the core (int8 C), optional fused s8 residual"),
         ("DW", "0xD7", "DW", "depthwise KHxKW, one item per core, int32 out"),
         ("DWQ", "0xD8", "DWQ", "depthwise + integer requantize, up to 3 items per core, int8 out"),
         ("POOL", "AIE_MODE_OP_POOL", "POOL", "pooling and trailing-axis reductions (8-bit)"),
         ("ELTWISE", "AIE_MODE_OP_ELTWISE", "ELTWISE", "binary of two 8-bit tensors"),
         ("LUT", "AIE_MODE_OP_LUT", "LUT", "8-bit table lookup (eltwise, prelu, batch norm, reorder)"),
         ("STEPS", "AIE_MODE_OP_STEPS", "STEPS", "step tables for monotone ops -> s8 / u8"),
         ("CONV", "+region", "CONV", "direct conv: A is a raw input region, the core gathers the im2col tile "
                                     "(B row 127 byte 4; with 0x00 / 0xA0 / 0xF1)"),
         ("MMT", "AIE_MODE_OP_MMT", "MMT", "short-K (<= 128) int8 GEMM + requantize on mlir-aie's matmul "
                                           "micro-kernel: 128 rows x 32 channels per core (opt-in)"),
         ("SDPA", "AIE_MODE_OP_SDPA (0x5D)", "SDPA", "attention per (batch, head, query block): int32 scores, "
                                                    "table softmax, weighted V (aie_modes_sdpa.h, opt-in)"),
         ("BF16", "AIE_MODE_OP_BF16 (0xBF)", "BF16", "bf16 GEMM, f32 accumulation: bf16 x bf16 and f32 as bf16x3 "
                                                    "(aie_modes_bf16.h, opt-in)"),
         ("FVEC", "AIE_MODE_OP_FVEC (0xF5)", "FVEC", "f32 pointwise / binary / softmax / layer norm rows "
                                                    "(aie_modes_fvec.h, opt-in)"))

#: which modes (opcodes) each megakernel entry sends
ENTRY_MODES = {"gemm_blk_s8s32": ("GEMM",), "gemm_kc_s8s32": ("GEMM",), "gemm_q_s8s8": ("GEMMQ",),
               "dw_blk_s8s32": ("DW",), "dwq_blk_s8s8": ("DWQ",), "pool_blk_s8s32": ("POOL",),
               "elt_blk_s8s32": ("ELTWISE",), "lut_blk_s8": ("LUT", "STEPS"),
               "conv_kc_s8s32": ("GEMM", "CONV"), "conv_q_s8s8": ("GEMMQ", "CONV"), "mmt_blk_s8s8": ("MMT",),
               "sdpa_blk_s8s32": ("SDPA",), "bf16_blk_f32": ("BF16",),
               "fvec_blk_f32": ("FVEC",)}

_GEMM_FNS = ("aiednn_convolution_forward", "aiednn_deconvolution_forward", "aiednn_matmul_forward",
             "aiednn_inner_product_forward")


# ── graph facts ────────────────────────────────────────────────────────────

def _kind(fn: str) -> str:
    k = fn.replace("aiednn_", "")
    for s in ("_create_attr", "_create", "_forward_inference", "_forward"):
        k = k.replace(s, "")
    return k


def _nbytes(t) -> int:
    return int(np.prod(t.shape)) * np.dtype(DT_NP[t.dt]).itemsize


def _blob_offsets(g) -> tuple:
    """constant / input blob offsets, in emit_app's order (64-byte aligned)"""
    off, top = {}, 0
    for t in g.tensors:
        if t.base is None and t.kind in ("const", "input"):
            off[t.id] = top
            top += _nbytes(t)
            top += -top % 64
    return off, top


def _lifetimes(g) -> dict:
    first, last = {}, {}
    for i, op in enumerate(g.ops):
        for _, t in op.exec:
            r = g.root(t)
            first.setdefault(r, i)
            last[r] = i
    if g.output is not None:
        last[g.root(g.output)] = len(g.ops)
    return {r: (first[r], last[r]) for r in first}


def gemm_geometry(g, op, impl: str = ""):
    """GEMM(s) an integer/float conv / deconv / matmul primitive reduces to, with its AIE launches and staged bytes."""
    if not op.fn.startswith(_GEMM_FNS):
        return None
    ex = {a: t for a, t in op.exec if not isinstance(a, tuple)}
    if "SRC" not in ex or "WEIGHTS" not in ex or "DST" not in ex:
        return None
    s, w, d = (g.tensors[ex[k]].shape for k in ("SRC", "WEIGHTS", "DST"))
    cp = next((c[1] for c in op.create if c[0] == "conv"), None)
    G = int(cp["groups"]) if cp else 1
    if op.fn.startswith("aiednn_convolution"):
        imgs, M, N, K, form = s[0], d[2] * d[3], d[1] // G, w[1] * w[2] * w[3], "im2col"
    elif op.fn.startswith("aiednn_deconvolution"):
        imgs, M, N, K, form = s[0], s[2] * s[3], (d[1] // G) * w[2] * w[3], s[1] // G, "col2im"
    elif op.fn.startswith("aiednn_inner_product"):
        imgs, M, N, K, form = 1, s[0], w[0], int(np.prod(s[1:])), "ip"
    else:
        imgs, M, N, K, form = int(np.prod(d[:-2])) if len(d) > 2 else 1, d[-2], d[-1], s[-1], "matmul"
    count = int(imgs) * G
    on_aie = "aie:" in impl
    per = math.ceil(M / BLOCK["M"]) * math.ceil(K / BLOCK["K"]) * math.ceil(N / BLOCK["N"])
    launches = count * per if on_aie else 0
    macs = count * M * N * K
    r = dict(form=form, M=int(M), N=int(N), K=int(K), gemms=count, groups=G, images=int(imgs), macs=int(macs),
             launches=int(launches), per_gemm_launches=int(per), on_aie=on_aie,
             blocks=[math.ceil(M / BLOCK["M"]), math.ceil(N / BLOCK["N"]), math.ceil(K / BLOCK["K"])])
    if on_aie:
        r.update(util=macs / (launches * BLOCK["M"] * BLOCK["N"] * BLOCK["K"]),
                 dma_in=launches * (BLOCK["a_stage"] + BLOCK["b_stage"]), dma_out=launches * BLOCK["c_stage"],
                 useful_a=count * M * K, useful_b=count * N * K, useful_c=count * M * N * 4,
                 host_col=(count * M * K) if form in ("im2col", "col2im") else 0)
    return r


def _op_record(g, i, op, impl, loc_of, producer, t_ms, calls=None, opcodes=None):
    args, ins, outs, consts = [], 0, 0, 0
    for a, t in op.exec:
        tt = g.tensors[t]
        name = a if not isinstance(a, tuple) else (f"POST_SRC{a[1]}" if a[0] == "post" else f"SRC{a[1]}")
        role = "out" if a == "DST" else ("const" if tt.kind == "const" else "in")
        b = _nbytes(tt)
        if role == "out":
            outs += b
        elif role == "const":
            consts += b
        else:
            ins += b
        prod = producer.get((i, g.root(t)))
        args.append(dict(arg=name, t=t, role=role, bytes=b, src=prod))
    params, post, nscales, quant = {}, [], 0, None
    for c in op.create:
        if c[0] in ("conv", "pool"):
            params[c[0]] = {k: list(v) if isinstance(v, tuple) else v for k, v in c[1].items()}
        elif c[0] == "alg":
            params["alg"] = c[1]
        elif c[0] == "perm":
            params["perm"] = list(c[1])
        elif c[0] in ("float", "int", "i64", "uint"):
            params.setdefault("scalars", []).append(_num(c[1]))
        elif c[0] == "attr":
            sc = c[1]["scales"]
            nscales = 0 if sc is None else int(np.asarray(sc).size)
            for po in c[1]["post"]:
                if po[0] == "eltwise":
                    post.append(dict(kind="eltwise", alg=po[1], alpha=_num(po[2]), beta=_num(po[3])))
                else:   # a quantized src1 (s8 residual) carries its scale / zero point
                    post.append(dict(kind="binary", alg=po[1], t=int(po[2]),
                                     scale=_num(po[3]) if len(po) > 3 else None, zp=_num(po[4]) if len(po) > 4 else None))
            if c[1].get("dst_scale"):     # --aiednn-fold-requant / --aiednn-fold-norm-quant: integer dst
                quant = dict(scale=_num(c[1]["dst_scale"]), zp=int(c[1].get("dst_zp", 0)), dt=g.tensors[op.out].dt)
    gm = gemm_geometry(g, op, impl)
    fused = sorted(n for n, t in g.nodes.items() if g.root(t) == g.root(op.out) and n != op.name)
    flags = next((int(c[1]) for c in op.create if c[0] == "uint"), 0) if _kind(op.fn) == "batch_normalization" else 0
    calls = {k: int(v) for k, v in (calls or {}).items() if v}
    return dict(i=i, name=op.name, fn=op.fn, kind=_kind(op.fn), impl=impl,
                place="aie" if "aie:" in impl else "cpu", composite=impl.split("+")[0] if "+" in impl else "",
                args=args, params=params, post=post, nscales=nscales, gemm=gm, in_bytes=ins, out_bytes=outs,
                const_bytes=consts, t_ms=t_ms, out=op.out, out_loc=loc_of(op.out), fused=fused, quant=quant,
                norm_relu=bool(flags & 4), calls=calls, opcodes=_op_opcodes(calls, impl, opcodes or {}),
                epilogue=_epilogue_place(op, impl, calls, quant, post))


def _op_opcodes(calls: dict, impl: str, opc: dict) -> list:
    """[{entry, calls, modes, opcodes}] for the megakernel entries one op called (x86 counts)"""
    out = []
    for e, n in sorted(calls.items(), key=lambda kv: -kv[1]):
        modes = ENTRY_MODES.get(e, ())
        if e == "lut_blk_s8":
            modes = ("STEPS",) if impl.startswith("steps:") else ("LUT",)
        out.append(dict(entry=e, calls=n, modes=list(modes), opcodes=[opc.get(m, "?") for m in modes]))
    return out


def _epilogue_place(op, impl: str, calls: dict, quant, post) -> str:
    """where the op's epilogue (scales, bias, post-ops, requantize) runs"""
    if "aie:" not in impl:
        return "host"
    if calls.get("gemm_q_s8s8"):
        return "aie-core"            # 0xF1: fixed-point requant (+ residual) on the core
    if calls.get("dwq_blk_s8s8"):
        return "aie-core"            # 0xD8
    if op.fn.startswith(_GEMM_FNS) or impl.startswith("dwblk:"):
        return "host" if (quant or post or any(c[0] == "attr" for c in op.create)) else "none"
    return "aie-core"                # lut / steps / eltblk / poolblk compute the whole op


def _num(v):
    v = float(v)
    return v if math.isfinite(v) else str(v)


def _producers(g):
    """(op index, root) -> producing op index (-1 = model input) for every input read"""
    last_writer, out = {}, {}
    for i, op in enumerate(g.ops):
        for a, t in op.exec:
            if a == "DST":
                continue
            r = g.root(t)
            if g.tensors[r].kind == "input":
                out[(i, r)] = -1
            elif r in last_writer:
                out[(i, r)] = last_writer[r]
        last_writer[g.root(op.out)] = i
    return out


def _layout(g, producer, life) -> list:
    """git-graph lanes: an op takes a lane its input frees, a live tensor holds its lane"""
    held, lanes = {}, []          # lane -> root it holds (until its last use)
    inp = g.root(g.input) if g.input is not None else None
    if inp is not None:
        held[0] = inp
    for i, op in enumerate(g.ops):
        mine = None
        for a, t in op.exec:
            r = g.root(t)
            if a == "DST" or (i, r) not in producer:
                continue
            for ln, hr in held.items():
                if hr == r and life.get(r, (0, -1))[1] <= i:
                    mine = ln if mine is None else min(mine, ln)
        for ln in [ln for ln, hr in held.items() if life.get(hr, (0, -1))[1] <= i]:
            del held[ln]
        if mine is None or mine in held:
            mine = next(k for k in range(len(g.ops) + 2) if k not in held)
        lanes.append(mine)
        r = g.root(op.out)
        if life.get(r, (0, i))[1] > i:
            held[mine] = r
    return lanes


def _edges(g, producer) -> list:
    ed = {}
    for (i, r), p in producer.items():
        ed.setdefault((p, i), []).append(r)
    out = []
    for (p, i), roots in sorted(ed.items()):
        ts = [t for a, t in g.ops[i].exec if a != "DST" and g.root(t) in roots]
        out.append(dict(src=p, dst=i, tensors=sorted(set(ts)), bytes=sum(_nbytes(g.tensors[t]) for t in set(ts))))
    return out


# ── board log, routing, device ─────────────────────────────────────────────

def parse_board_log(path) -> dict:
    """PROF / aiednn_stats / aiednn_kernel_stats / MODEL / PASS lines of a board run.log"""
    p = Path(path)
    text = p.read_text(errors="replace").replace("\r", "")
    prof, stats = {}, {}
    for m in re.finditer(r"^PROF (\S+)\s+(\d+) ops\s+([\d.]+) ms", text, re.M):
        prof[m.group(1)] = dict(n=int(m.group(2)), ms=float(m.group(3)))
    for m in re.finditer(r"^aiednn_stats: (\S+)\s+(\d+)\s+(\d+)\s*$", text, re.M):
        stats[m.group(1)] = dict(execs=int(m.group(2)), launches=int(m.group(3)))
    out = dict(path=str(p), prof=prof, stats=stats, passed="RESULT: PASS" in text or "\nPASS:" in text)
    m = re.search(r"^MODEL (\S+) mode=(\S+) ops=(\d+) time_ms=(\S+) relerr=(\S+) top1=(-?\d+) ref_top1=(-?\d+)"
                  r"(?: float_top1=(-?\d+))?(?: gemm_checked=(\d+) gemm_bad=(\d+))?", text, re.M)
    if m:
        out["model"] = dict(name=m.group(1), mode=m.group(2), ops=int(m.group(3)), time_ms=float(m.group(4)),
                            relerr=float(m.group(5)), top1=int(m.group(6)), ref_top1=int(m.group(7)),
                            gemm_checked=int(m.group(9) or 0), gemm_bad=int(m.group(10) or 0))
    m = re.search(r"aiednn_kernel_stats: (\S+) calls=(\d+) copy_in_ms=([\d.]+) launch_ms=([\d.]+) copy_out_ms=([\d.]+)",
                  text)
    if m:
        out["kernel"] = dict(name=m.group(1), calls=int(m.group(2)), copy_in_ms=float(m.group(3)),
                             launch_ms=float(m.group(4)), copy_out_ms=float(m.group(5)))
    return out


_STEP = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
_TILE = r"XAie_TileLoc\((\d+),\s*(\d+)\)"


def parse_routing(path, board: str = "vek280") -> dict:
    """stream paths of an aiehlc ``routing.cc``."""
    p = Path(path)
    text = p.read_text()
    c0 = 0
    rm = p.parent / "routingresourcemap.json"
    if rm.is_file():
        try:
            c0 = int(json.loads(rm.read_text()).get("partition_start_col", 0))
        except (ValueError, OSError):
            c0 = 0
    rounds = []
    for m in re.finditer(r"//round is (\d+) hw split in : (\w+)(.*?)(?=//round is|\Z)", text, re.S):
        rounds.append(_parse_round(int(m.group(1)), m.group(2), m.group(3), c0, board))
    return dict(path=str(p), start_col=c0, streams=[s for r in rounds for s in r])


def _parse_round(idx, split, body, c0, board: str = "vek280") -> list:
    tl = lambda c, r: (int(c) + c0, int(r))  # noqa: E731
    src = re.search(r"XAie_EnableShimDmaToAieStrmPort\(\w+, " + _TILE + r", (\d+)\)", body)
    sink = re.search(r"XAie_EnableAieToShimDmaStrmPort\(\w+, " + _TILE + r", (\d+)\)", body)
    e_in, e_out, dests, cores_out = [], [], [], []
    for m in re.finditer(r"XAie_StrmConnCctEnable\(\w+, " + _TILE + r", (\w+), (\d+), (\w+), (\d+)\)", body):
        c, r = tl(m.group(1), m.group(2))
        mdir = m.group(5)
        if mdir == "DMA":
            dests.append([c, r])
        elif mdir in _STEP:
            dc, dr = _STEP[mdir]
            if r + dr >= 0:
                (e_out if mdir == "SOUTH" else e_in).append([c, r, c + dc, r + dr])
    for m in re.finditer(r"XAie_StrmPktSwMstrPortEnable\(\w+, " + _TILE + r", (\w+), (\d+)", body):
        c, r = tl(m.group(1), m.group(2))
        if m.group(3) in _STEP:
            dc, dr = _STEP[m.group(3)]
            if r + dr >= 0:
                e_out.append([c, r, c + dc, r + dr])
    for m in re.finditer(r"XAie_StrmPktSwSlaveSlotEnable\(\w+, " + _TILE + r", DMA, \d+, \d+, XAie_PacketInit\((\d+)",
                         body):
        cores_out.append([*tl(m.group(1), m.group(2)), int(m.group(3))])
    out = []
    if src:
        sc, _ = tl(src.group(1), src.group(2))
        port = int(src.group(3))
        # AIE-ML shim mux port -> MM2S channel; another array (VEK385 / AIE2PS): the port number
        ch = {3: 0, 7: 1}.get(port, port) if board == "vek280" else port
        rows = sorted({d[1] for d in dests})
        cols = sorted({d[0] for d in dests})
        if split == "col":
            data, what = "B", f"B / weights: column chunk {idx} broadcast to the cores of column {cols}"
        else:
            data, what = "A", f"A / src: row chunk {idx} broadcast to the cores of row {rows}"
        out.append(dict(name=f"{data}{idx}", data=data, round=idx, split=split, shim=[sc, ch], dir="MM2S",
                        port=port, edges=e_in, dests=dests, what=what))
    if sink:
        kc, _ = tl(sink.group(1), sink.group(2))
        port = int(sink.group(3))
        # AIE-ML shim demux port -> S2MM channel; another array (VEK385 / AIE2PS uses ports 1 / 3): the port
        # number, kept numeric so shim tuples sort
        ch = {2: 0, 3: 1}.get(port, port) if board == "vek280" else port
        out.append(dict(name=f"C{idx}", data="C", round=idx, split=split, shim=[kc, ch], dir="S2MM", port=port,
                        edges=e_out,
                        dests=[c[:2] for c in cores_out], packets=cores_out,
                        what=f"C / int32 result: packet-merged from {len(cores_out)} cores into one shim S2MM"))
    return out


def find_routing(board_log=None, board: str = None):
    """routing.cc of a built gemm_s8s32 app: next to *board_log*, else any built model's for the same board"""
    cands = []
    if board_log:
        cands.append(Path(board_log).parent / "aout" / "worklocal" / "routing.cc")
    b = board_of(board_log, board)
    cands += sorted((REPO / "worklocal" / "aiednn" / "models").glob(f"*/{b}/aout/worklocal/routing.cc"))
    for c in cands:
        if c.is_file() and (c.parent / "gemm_s8s32.cc").is_file():
            return c
    return None


# board -> labels the page prints and the array it draws (rows: shim 0, then mem tiles, then cores)
BOARDS = {
    "vek280": dict(name="VEK280", host="Arm Cortex-A72", host_short="A72", array="AIE-ML (xcve2802)",
                   gen="XAIE_DEV_GEN_AIEML", aie_gen=2, cols=38, rows=11, mem_rows=[1, 2],
                   core_rows=list(range(3, 11))),
    # two memtile rows, cores rows 3..6, every column has a DMA shim
    "vek385": dict(name="VEK385", host="Arm Cortex-A78", host_short="A78", array="AIE2PS",
                   gen="XAIE_DEV_GEN_AIE2PS", aie_gen=5, cols=36, rows=7, mem_rows=[1, 2],
                   core_rows=list(range(3, 7))),
}
# the 4x4 GEMM library's partition (gemm.h): string as written, drawn columns / rows, shim columns that reach DDR
# (None: every column of the partition)
PARTITIONS = {"vek280": dict(text="{0, 8, 0, 7}", start_col=0, cols=8, start_row=0, rows=7, ddr=[2, 3, 6, 7]),
              "vek385": dict(text="{0, 3, 0, 6}", start_col=0, cols=4, start_row=0, rows=7, ddr=None)}


def board_of(board_log=None, board=None) -> str:
    """*board* if given, else the board directory a ``.../<board>/run.log`` sits in, else vek280"""
    if board:
        return board
    if board_log and Path(board_log).parent.name in BOARDS:
        return Path(board_log).parent.name
    return "vek280"


def device_geometry(board: str = "vek280") -> dict:
    """the *board*'s AIE array"""
    b = BOARDS[board]
    d = dict(cols=b["cols"], rows=b["rows"], shim_row=0, mem_rows=list(b["mem_rows"]), core_rows=list(b["core_rows"]),
             source=f"documented constants ({b['name']} / {b['array']})")
    srcs = []
    dm = REPO / "include" / "aie_device_map.h"
    if dm.is_file():
        tail = dm.read_text().split(b["gen"])[-1]
        mc, mr = re.search(r"XAIE_NUM_COLS (\d+)", tail), re.search(r"XAIE_NUM_ROWS (\d+)", tail)
        if mc and mr:
            d["cols"], d["rows"] = int(mc.group(1)), int(mr.group(1))
            srcs.append(f"include/aie_device_map.h (AIE_GEN {b['aie_gen']})")
    hw = REPO / "thirdparty" / "alib" / "internal" / "xaiengine" / "xaie_lite_hwcfg.h"
    if board == "vek280" and hw.is_file():    # the AIE-ML lite header; VEK385 keeps its documented row split
        t = hw.read_text()
        v = {k: [int(x) for x in re.findall(rf"#define {k}\s+(\d+)", t)]
             for k in ("XAIE_MEM_TILE_ROW_START", "XAIE_MEM_TILE_NUM_ROWS", "XAIE_AIE_TILE_ROW_START",
                       "XAIE_AIE_TILE_NUM_ROWS")}
        if all(v.values()) and max(v["XAIE_MEM_TILE_NUM_ROWS"]) > 0:
            k = v["XAIE_MEM_TILE_NUM_ROWS"].index(max(v["XAIE_MEM_TILE_NUM_ROWS"]))
            ms, mn = v["XAIE_MEM_TILE_ROW_START"][k], v["XAIE_MEM_TILE_NUM_ROWS"][k]
            cs, cn = v["XAIE_AIE_TILE_ROW_START"][k], v["XAIE_AIE_TILE_NUM_ROWS"][k]
            d["mem_rows"], d["core_rows"] = list(range(ms, ms + mn)), list(range(cs, cs + cn))
            srcs.append("thirdparty/alib/internal/xaiengine/xaie_lite_hwcfg.h (AIE-ML)")
    if srcs:
        d["source"] = " + ".join(srcs)
    return d


def _aie_section(routing_cc, board: str = "vek280") -> dict:
    dev = device_geometry(board)
    part = PARTITIONS[board]
    aie = dict(device=dev, block=dict(BLOCK, partition=part["text"]), routing=None,
               shim_cols_ddr=part["ddr"] if part["ddr"] is not None else list(range(part["start_col"],
                                                                                     part["start_col"] + part["cols"])),
               partition={k: part[k] for k in ("start_col", "cols", "start_row", "rows")})
    if routing_cc and Path(routing_cc).is_file():
        rt = parse_routing(routing_cc, board)
        rt["path"] = _rel(rt["path"])
        aie["routing"] = rt
        cores = sorted({tuple(d) for s in rt["streams"] if s["dir"] == "MM2S" for d in s["dests"]})
        aie["cores"] = [list(c) for c in cores]
        aie["shims"] = sorted({tuple(s["shim"]) + (s["dir"],) for s in rt["streams"]})
        aie["placement"] = f"parsed from {rt['path']}"
    else:
        aie["cores"] = [[c, r] for c in range(4) for r in range(3, 7)]
        aie["shims"] = []
        aie["placement"] = ("documented mapping (4x4 mesh at partition cols 0-3, core rows 3-6; doc/design/aiednn.md "
                            "section 5); no built routing.cc found, so the shim columns and stream paths are not shown")
    return aie


def _mode_table() -> dict:
    """{mode: opcode text}: AIE_MODE_OP_* resolved from src/aietensorop/ops/aie_modes_*.h"""
    defs = {}
    for h in (REPO / "src" / "aietensorop" / "ops").glob("aie_modes_*.h"):
        defs.update(re.findall(r"#define (AIE_MODE_OP_\w+) (0x[0-9A-Fa-f]+)", h.read_text()))
    return {m: defs.get(op, op) for m, op, _, _ in MODES}


def _build_info(board_log) -> dict:
    """what the board build next to *board_log* compiled: modes, entry counts, core program size."""
    if not board_log:
        return {}
    d = Path(board_log).parent
    info = dict(dir=_rel(d))
    cfg = d / "aie_modes_cfg.h"
    if cfg.is_file():
        info["modes"] = {m: v == "1" for m, v in re.findall(r"#define AIEK_MODE_([A-Z0-9_]+) ([01])\b", cfg.read_text())}
        info["cfg"] = _rel(cfg)
    use = d / "usage_x86" / "run.log"
    if use.is_file():
        info["usage"] = {e: int(n) for e, n in re.findall(r"^AIEDNN_ENTRY_USED (\w+) (\d+)$", use.read_text(), re.M)}
    km = d / "aout" / "worklocal" / "build" / "kernel.map"
    if km.is_file():
        t = km.read_text(errors="replace")
        m = re.search(r"Section summary for memory 'PM'.*?(\d+)\s+Total", t, re.S)
        k = re.search(r"Section summary for memory 'PM'.*?(\d+)\s+\S*kernel\.o", t, re.S)
        if m:
            info["pm_bytes"] = int(m.group(1))
            info["pm_kernel_bytes"] = int(k.group(1)) if k else None
            info["pm_total"] = 16384
    return info


def _megakernel(ops, board_log) -> dict:
    """modes / opcodes / entries of the GEMM megakernel and how this model uses them"""
    opc = _mode_table()
    b = _build_info(board_log)
    modes = []
    for m, _, macro, what in MODES:
        users = [o["i"] for o in ops if any(m in c["modes"] for c in o["opcodes"])]
        launches = sum(c["calls"] for o in ops for c in o["opcodes"] if m in c["modes"])
        compiled = True if macro is None else (b["modes"].get(macro, True) if "modes" in b else None)
        modes.append(dict(mode=m, opcode=opc[m], macro=f"AIEK_MODE_{macro}" if macro else "(always)", what=what,
                          entries=[e for e, ms in ENTRY_MODES.items() if m in ms], ops=users, launches=launches,
                          compiled=compiled))
    entries = []
    for e, ms in ENTRY_MODES.items():
        users = [o for o in ops if e in o["calls"]]
        entries.append(dict(entry=e, modes=list(ms), opcodes=[opc[m] for m in ms], ops=[o["i"] for o in users],
                            calls=sum(o["calls"][e] for o in users), impls=sorted({o["impl"] for o in users}),
                            board_usage=(b.get("usage") or {}).get(e)))
    return dict(source="src/aietensorop/gemm/gemm_uk.cc", modes=modes, entries=entries, build=b,
                counted=any(o["calls"] for o in ops))


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# ── x86 timing ─────────────────────────────────────────────────────────────

class _TimedLib:
    """``AiednnLib`` proxy that times every ``execute`` (create / destroy excluded)"""

    def __init__(self, lib):
        self._lib, self.times, self.calls = lib, [], []

    def __getattr__(self, k):
        return getattr(self._lib, k)

    def execute(self, prim, bufs):
        counts = getattr(self._lib, "entry_calls", None)
        if counts:
            counts(reset=True)               # drop whatever create / earlier work counted
        t = time.perf_counter()
        st = self._lib.execute(prim, bufs)
        self.times.append(1e3 * (time.perf_counter() - t))
        self.calls.append(counts(reset=True) if counts else {})
        return st


def run_timed(g, lib, x=None, calls: bool = False):
    """``dnn_torch.run_ctypes`` with each primitive's execution timed on x86."""
    from .dnn_torch import run_ctypes
    tl = _TimedLib(lib)
    y, impls, bufs = run_ctypes(g, tl, None if x is None else np.asarray(x), keep=True)
    return (y, impls, bufs, tl.times, tl.calls) if calls else (y, impls, bufs, tl.times)


# ── assembling the page data ───────────────────────────────────────────────

def build_data(g, impls, *, title=None, board_log=None, op_times_ms=None, routing_cc=None, op_calls=None,
               board_name: str = None) -> dict:
    """everything the page renders, as one JSON-able dict"""
    if len(impls) != len(g.ops):
        raise ValueError(f"{len(impls)} impls for {len(g.ops)} ops")
    aoff, arena = _plan_memory(g)
    boff, blob = _blob_offsets(g)
    life = _lifetimes(g)
    names = {}
    for n, t in g.nodes.items():
        names.setdefault(t, []).append(n)

    def loc_of(t):
        r = g.root(t)
        rt = g.tensors[r]
        if r in boff:
            return dict(space="input" if rt.kind == "input" else "blob", off=boff[r], size=_nbytes(rt), root=r)
        if r in aoff:
            return dict(space="arena", off=aoff[r], size=(_nbytes(rt) + 63) // 64 * 64, root=r)
        return dict(space="unused", off=0, size=_nbytes(rt), root=r)

    producer = _producers(g)
    tensors = []
    for t in g.tensors:
        tensors.append(dict(id=t.id, shape=list(t.shape), dt=t.dt, kind=t.kind, base=t.base, root=g.root(t.id),
                            bytes=_nbytes(t), loc=loc_of(t.id), names=names.get(t.id, []),
                            life=list(life.get(g.root(t.id), (-1, -1)))))
    times = list(op_times_ms) if op_times_ms is not None else [None] * len(g.ops)
    calls = list(op_calls) if op_calls is not None else [None] * len(g.ops)
    opc = _mode_table()
    ops = [_op_record(g, i, op, impls[i], loc_of, producer, times[i], calls[i], opc) for i, op in enumerate(g.ops)]
    for o, ln in zip(ops, _layout(g, producer, life)):
        o["lane"] = ln
    bname = board_of(board_log, board_name)
    if routing_cc is None:
        routing_cc = find_routing(board_log, bname)
    board = parse_board_log(board_log) if board_log and Path(board_log).is_file() else None
    if board:
        board["path"] = _rel(board["path"])
        bops = (board.get("model") or {}).get("ops")
        board["stale"] = bops is not None and bops != len(g.ops)
    data = dict(title=title or f"{g.name} [{g.mode}] -- aiednn model flow", model=g.name, mode=g.mode,
                generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), tensors=tensors, ops=ops,
                edges=_edges(g, producer), input=g.input, output=g.output,
                memory=dict(arena=arena, blob=blob, consts=sum(_nbytes(t) for t in g.tensors
                                                                if t.kind == "const" and t.base is None),
                            input=_nbytes(g.tensors[g.input]) if g.input is not None else 0),
                aie=_aie_section(routing_cc, bname), board=board, megakernel=_megakernel(ops, board_log),
                target=dict(key=bname, **{k: v for k, v in BOARDS[bname].items() if k in ("name", "host", "host_short",
                                                                                         "array")}),
                time_source="board" if board and board.get("prof") and not (board["stale"] and op_times_ms is not None)
                else ("x86" if op_times_ms is not None else None))
    data["summary"] = _summary(data)
    return data


def _summary(d) -> dict:
    by = {}
    for o in d["ops"]:
        e = by.setdefault(o["impl"], dict(impl=o["impl"], place=o["place"], n=0, x86_ms=0.0, launches=0, macs=0,
                                          dma_in=0, dma_out=0))
        e["n"] += 1
        e["x86_ms"] += o["t_ms"] or 0.0
        gm = o["gemm"] or {}
        for k in ("launches", "macs", "dma_in", "dma_out"):
            e[k] += gm.get(k, 0)
    b = d["board"] or {}
    for imp, e in by.items():
        if imp in b.get("prof", {}):
            e["board_ms"] = b["prof"][imp]["ms"]
            e["board_n"] = b["prof"][imp]["n"]
    gms = [o["gemm"] for o in d["ops"] if o["gemm"]]
    n_aie = sum(1 for o in d["ops"] if o["place"] == "aie")
    st = b.get("stats", {})
    counted = sum(c["calls"] for o in d["ops"] for c in o["opcodes"])
    return dict(ops=len(d["ops"]), aie_ops=n_aie, cpu_ops=len(d["ops"]) - n_aie, counted_launches=counted or None,
                launches=sum(g_["launches"] for g_ in gms),
                board_launches=sum(v["launches"] for v in st.values()) if st else None,
                dma_in=sum(g_.get("dma_in", 0) for g_ in gms), dma_out=sum(g_.get("dma_out", 0) for g_ in gms),
                macs=sum(g_["macs"] for g_ in gms), aie_macs=sum(g_["macs"] for g_ in gms if g_["on_aie"]),
                x86_ms=sum(o["t_ms"] or 0.0 for o in d["ops"]) if d["time_source"] else None,
                board_ms=(b.get("model") or {}).get("time_ms"),
                act_bytes=sum(o["out_bytes"] for o in d["ops"]),
                impls=sorted(by.values(), key=lambda e: -e["n"]))


# ── HTML ───────────────────────────────────────────────────────────────────

def write_html(g, impls, out_path, *, title=None, board_log=None, op_times_ms=None, routing_cc=None,
               op_calls=None, board_name: str = None) -> Path:
    """Write the self-contained model-flow page for lowered graph *g* whose ops dispatched to *impls*"""
    data = build_data(g, impls, title=title, board_log=board_log, op_times_ms=op_times_ms, routing_cc=routing_cc,
                      op_calls=op_calls, board_name=board_name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"), default=_json_default).replace("</", "<\\/")
    tpl = (Path(__file__).parent / "dnn_viz_template.html").read_text()
    html = tpl.replace("__TITLE__", _esc(data["title"])).replace("__DATA__", blob)
    out.write_text(html)
    return out


def write_model_html(g, lib, x, out_path, **kw) -> Path:
    """run *g* on x86 with per-op timing, then ``write_html``"""
    _, impls, _, times, calls = run_timed(g, lib, x, calls=True)
    return write_html(g, impls, out_path, op_times_ms=times, op_calls=calls, **kw)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _num(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o).__name__)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
