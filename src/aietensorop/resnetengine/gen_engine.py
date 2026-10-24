#!/usr/bin/env python3
"""Generate the persistent conv engine's network description for int8 ResNet-18.

Reads stage 4's TVM artifacts (``resnet18_graph.json``, ``resnet18_params.bin``)
and the per-layer C (``layers/NN_*/NN_*.c``, for the epilogue constants), and
produces everything ``resnetengine/eng`` needs:

  eng_net.h       per-layer host table (geometry, DDR buffers, shim BD shapes)
  eng_blob.bin    per (layer, mesh column): header + weight/param chunks, the
                  exact byte stream the column's weight channel delivers
  eng_x1.bin      engine input (call 02 output = maxpool), L8 int8
  eng_gold.bin    TVM's own output of every engine layer, L8 int8 (from the
                  ``make TRACE=1`` layeriohex dumps), for the board self-check

Before writing anything it runs ``model_network``: a numpy model of the
engine's integer semantics (fused downsample tap, on-core residual, every
inter-layer tensor 8-bit) chained over all 16 engine layers from call 02's
dump, compared against TVM's dump of every call. A mismatch stops here.
See doc/design/resnet_aie_engine.md.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARAM_MAGIC, NDARRAY_MAGIC = 0xF7E58D4F05049CB7, 0xDD5E40F096B4A13F


def param_offsets(blob: Path) -> dict:
    """{name: (offset, nbytes)} of a relay.save_param_dict blob (as arm_build's,
    without importing TVM)."""
    data = blob.read_bytes()
    assert struct.unpack_from("<Q", data, 0)[0] == PARAM_MAGIC, blob
    n, off, names = struct.unpack_from("<Q", data, 16)[0], 24, []
    for _ in range(n):
        ln = struct.unpack_from("<Q", data, off)[0]
        names.append(data[off + 8:off + 8 + ln].decode())
        off += 8 + ln
    off += 8
    out = {}
    for name in names:
        off += 8 + 8 + 4 + 4
        ndim = struct.unpack_from("<i", data, off)[0]
        off += 4 + 4 + 8 * ndim
        nbytes = struct.unpack_from("<q", data, off)[0]
        off += 8
        out[name] = (off, nbytes)
        off += nbytes
    return out

MESH = 4                      # mesh rows == mesh cols
CHUNK_BLK = 8                 # ic blocks (of 8) per weight chunk = 64 ch
NB = 2                        # oc blocks of 8 per core per pass = 16 oc
HDR_BYTES = 256
PRM_BYTES = 512
DM_TOP = 0x80000              # core data memory end (core view)
DM_W = 0x71C00                # weight chunk ring base (see eng/aieml.bcf)
EPSET_BYTES = 1344            # sizeof(EpSet) in kernel.cc, 64-aligned
RING_SLACK = 128

# ── TVM artifacts ──────────────────────────────────────────────────────────

class Net:
    def __init__(self, deploy: Path):
        self.deploy = deploy
        g = json.loads((deploy / "resnet18_graph.json").read_text())
        self.nodes, self.rp = g["nodes"], g["node_row_ptr"]
        self.shape, self.dtype = g["attrs"]["shape"][1], g["attrs"]["dltype"][1]
        blob = deploy / "resnet18_params.bin"
        self.pdata, self.poff = blob.read_bytes(), param_offsets(blob)
        self.calls, call_of = [], {}
        for i, n in enumerate(self.nodes):
            if n["op"] != "null":
                call_of[i] = len(self.calls)
                self.calls.append(i)
        self.call_of = call_of
        self.src = {}
        for c in sorted((deploy / "layers").glob("*/*.c")):
            for sym in re.findall(r"TVM_DLL int32_t (tvmgen_\w+)\(", c.read_text()):
                self.src[sym] = c

    def fn(self, call):
        return self.nodes[self.calls[call]]["attrs"]["func_name"]

    def args(self, call):
        return [e[0] for e in self.nodes[self.calls[call]]["inputs"]]

    def param(self, nid):
        name = self.nodes[nid]["name"]
        off, n = self.poff[name]
        dt = {"int8": np.int8, "int32": np.int32, "uint8": np.uint8, "float32": np.float32}[self.dtype[self.rp[nid]]]
        return np.frombuffer(self.pdata[off:off + n], dtype=dt).reshape(self.shape[self.rp[nid]])

    def is_param(self, nid):
        return self.nodes[nid]["op"] == "null" and self.nodes[nid]["name"] in self.poff

    def body(self, call):
        """Cleaned epilogue text of the call's kernel: casts dropped, pN_1 -> pN."""
        sym = self.fn(call)
        text = self.src[sym].read_text()
        i = text.find("int32_t v_ = ")
        text = text[i:] if i >= 0 else text[text.index("conv2d_NCHWc_global[(cse"):]  # raw requant (F)
        text = re.sub(r"\(\(u?int\d+_t\*\)(p\d)_1\)", r"\1", text)
        return re.sub(r"\(u?int(8|32|64)_t\*?\)", "", text)


def dump(deploy: Path, call: int, side: str):
    """{slot: np.ndarray} from arm_build/layeriohex/l<call>_*_<side>.h."""
    hexdir = deploy / "arm_build" / "layeriohex"
    path = next(hexdir.glob(f"l{call:02d}_*_{side}.h"))
    text = path.read_text()
    out = {}
    for m in re.finditer(r"static const long long (\w+)_shape\[\d+\] = \{([^}]*)\};", text):
        var = m.group(1)
        shape = [int(v) for v in m.group(2).split(",")]
        dt = re.search(rf'#define {var}_DTYPE "(\w+)"', text).group(1)
        body = re.search(rf"{var}_data\[\d+LL\] = \{{([^}}]*)\}};", text, re.S).group(1)
        raw = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))
        slot = int(re.search(rf"_{side}(\d+)(?:_p\d+_\w+)?$", var).group(1))
        out[slot] = np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape)
    return out

# ── per-conv extraction ────────────────────────────────────────────────────

_RQ = re.compile(r"\+ p(\d)\[\w+\]\) - p(\d)\[\w+\]\) - p(\d)\[\w+\]\)\) \* \(p(\d)\[\w+\]\)\) "
                 r"\+ \(1 << \(\(\(p(\d)\[")
_QT = re.compile(r"\(\((\d) != 0\) \? (.*?) \* (\d+)\) \+ \(1 << \(\(\((\d) \+ 31\) - 1\)\)\)\) >>")


def conv_info(net: Net, call: int) -> dict:
    """Geometry, raw params and epilogue constants of one TVM conv call."""
    args, body = net.args(call), net.body(call)
    m = _RQ.search(body)
    if not m:
        raise SystemExit(f"call {call}: requant pattern not found")
    bias, zpa, zpb, mult, shift = (net.param(args[int(i)]) for i in m.groups())
    wnid = next(a for a in args if net.is_param(a) and len(net.shape[net.rp[a]]) == 6)
    w = net.param(wnid)
    co_o, ci_o, kh, kw, ci_i, co_i = w.shape  # OIHW{ci_i}i{co_i}o
    W = w.transpose(0, 5, 1, 4, 2, 3).reshape(co_o * co_i, ci_o * ci_i, kh, kw)  # OIHW
    acts = [a for a in args if not net.is_param(a)]
    out_shape = net.shape[net.rp[net.calls[call]]]
    ho = out_shape[2]
    # data = the activation at the conv's input resolution; residual = at its output's
    data = next((a for a in acts if net.shape[net.rp[a]][2] in (ho + 2, 2 * ho + 2, 2 * ho)), acts[0])
    stride = 1 if net.shape[net.rp[data]][2] == ho + 2 else 2
    acts = [data] + [a for a in acts if a != data]
    first = body.split(";")[0]
    c0 = int(re.search(r"\)\) \+ (\d+)$", first).group(1)) if re.search(r"\)\) \+ (\d+)$", first) else 0
    qterms = [dict(l=int(l), operand=op, M=int(M), s=31 + int(s)) for l, op, M, s in _QT.findall(body)]
    zps = {int(i): int(net.param(args[int(i)]).reshape(-1)[0]) for i in re.findall(r"p(\d)\[0\]", body)}
    return dict(call=call, W=W, K=kh, Cin=ci_o * ci_i, Cout=co_o * co_i, Ho=out_shape[2], Wo=out_shape[3],
                bz=(bias.reshape(-1).astype(np.int64) - zpa.reshape(-1) - zpb.reshape(-1)),
                mult=mult.reshape(-1).astype(np.int64), shift=shift.reshape(-1).astype(np.int64),
                c0=c0, q=qterms, zps=zps, acts=[net.call_of.get(a, -1) for a in acts], body=body,
                stride=stride, out_dtype=net.dtype[net.rp[net.calls[call]]])

# ── engine layer table ─────────────────────────────────────────────────────

def build_layers(net: Net) -> list:
    """The engine's 16 layers: every non-stem conv, 1x1 downsamples fused into
    the 3x3 s2 conv that shares their input, residual adds done on-core."""
    convs = [k for k in range(len(net.calls)) if "conv2d" in net.fn(k) and k != 1]
    info = {k: conv_info(net, k) for k in convs}
    down = {k for k in convs if info[k]["K"] == 1}
    layers = []
    for k in convs:
        if k in down:
            continue
        c = info[k]
        L = dict(call=k, K=3, Cin=c["Cin"], Cout=c["Cout"], Ho=c["Ho"], Wo=c["Wo"], W=c["W"],
                 bz=c["bz"], mult=c["mult"], shift=c["shift"], c0=c["c0"], S=c["stride"],
                 down=None, comb=None, grab=False, res_from_down=False, final_q=None,
                 golden_call=k, golden_dtype=c["out_dtype"])
        layers.append(L)
    by_call = {L["call"]: L for L in layers}
    for k in sorted(down):
        d = info[k]
        # the downsample reads the same tensor as a 3x3 s2 conv -> fuse there
        src = d["acts"][0]
        host = next(L for L in layers if L["S"] == 2 and _unpad(net, info[L["call"]]["acts"][0]) == src)
        host["down"] = dict(call=k, W=d["W"], bz=d["bz"], mult=d["mult"], shift=d["shift"], c0=d["c0"])
        # the conv whose partial the downsample adds is the block's conv2
        partner = by_call[[a for a in d["acts"] if a != src][0]]
        _pair_downsample(partner, d, info[partner["call"]])
    for L in layers:
        if L["comb"] is None and len(info[L["call"]]["acts"]) == 2:
            _residual_identity(L, info[L["call"]])
    _mark_grabs(layers)
    return layers


def _unpad(net: Net, call: int) -> int:
    """The tensor a (cast/add +) pad op pads: what the conv reads, unpadded."""
    while call >= 0 and re.search(r"nn_pad(_\d+)?$", net.fn(call)):
        call = net.call_of[net.args(call)[0]]
    return call


def _pair_downsample(conv2: dict, d: dict, c2: dict):
    """conv2 of a downsample block: final = clamp(Qself(v2) + Qres(v_d))."""
    q = d["q"]
    if len(q) == 1:      # pattern C: d = clamp(Q(v_d - zp) + conv2_raw), conv2 = Q(v2 - zp2)
        (qs,) = c2["q"]
        zs = _zp_of(qs, c2)
        conv2["comb"] = dict(szp=zs, sl=qs["l"], sM=qs["M"], ss=qs["s"],
                             rzr=_zp_of(q[0], d), rl=q[0]["l"], rM=q[0]["M"], rs=q[0]["s"])
        conv2["golden_call"] = d["call"]
    else:                # pattern F/G: conv2 raw R; d = clamp(Q(v_d-zp) + Q(clamp(conv2+c)-zp))
        own = next(t for t in q if "v__1" in t["operand"])
        oth = next(t for t in q if "v__3" in t["operand"])
        conv2["c0"] = int(re.search(r"v__2 = p9\[\w+\] \+ (\d+);", d["body"]).group(1))
        conv2["comb"] = dict(szp=_zp_of(oth, d), sl=oth["l"], sM=oth["M"], ss=oth["s"],
                             rzr=_zp_of(own, d), rl=own["l"], rM=own["M"], rs=own["s"])
        conv2["golden_call"] = d["call"]
    conv2["res_from_down"] = True
    conv2["golden_dtype"] = d["out_dtype"]


def _residual_identity(L: dict, c: dict):
    """conv2 of an identity block (TVM's E/D/H): residual is the block input."""
    own = next(t for t in c["q"] if "v__1" in t["operand"])
    res = next(t for t in c["q"] if "v__1" not in t["operand"] and "v__3" not in t["operand"])
    L["comb"] = dict(szp=_zp_of(own, c), sl=own["l"], sM=own["M"], ss=own["s"],
                     rzr=0, rl=res["l"], rM=res["M"], rs=res["s"])
    tail = [t for t in c["q"] if "v__3" in t["operand"]]
    L["final_q"] = tail[0] if tail else None    # H: avgpool input = Q(clamp(..) << 2)


def _zp_of(term: dict, c: dict) -> int:
    m = re.search(r"- p(\d)\[0\]", term["operand"])
    return c["zps"][int(m.group(1))] if m else 0


def _mark_grabs(layers: list):
    """conv1 of an identity block keeps its slice of x for the next layer."""
    for a, b in zip(layers, layers[1:]):
        if b["comb"] is not None and not b["res_from_down"] and a["down"] is None and a["S"] == 1:
            a["grab"] = True

def build_stem(net: Net) -> dict:
    """The stem conv (call 01) + maxpool (call 02) as one engine layer."""
    c = conv_info(net, 1)
    assert c["K"] == 7 and c["Cin"] in (3, 4) and c["Ho"] == 112 and "max_pool2d" in net.fn(2)
    return dict(call=1, stem=True, K=7, S=2, Cin=c["Cin"], Cout=c["Cout"], Ho=56, Wo=56, W=c["W"], bz=c["bz"],
                mult=c["mult"], shift=c["shift"], c0=c["c0"], down=None, comb=None, grab=False,
                res_from_down=False, final_q=None, golden_call=2, golden_dtype="uint8")


def model_stem(net: Net, L: dict):
    """conv 7x7 s2 on TVM's call-01 input, requant, then maxpool 3x3 s2 pad 1 (pad = 0)."""
    x = dump(net.deploy, 1, "in")[0][0, 0].transpose(2, 0, 1).astype(np.int64)  # [3, 230, 230]
    acc = conv(x, L["W"], 2, L["bz"], K=7)[None]
    v = requant(acc, L, L["c0"])[0]                                              # [64, 112, 112]
    p = np.pad(v, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    mp = np.max([p[:, i:i + 112:2, j:j + 112:2] for i in range(3) for j in range(3)], axis=0)
    want = from_nchwc(dump(net.deploy, 2, "out")[0]).astype(np.int64)
    bad = int((mp != want).sum())
    print(f"  stem call 01+02 7x7 s2 + maxpool vs call 02: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    if bad:
        raise SystemExit("stem model mismatch")
    return mp, x


# ── numpy model of the engine ──────────────────────────────────────────────

def requant(acc, L, c0):
    sh = L["shift"][None, :, None, None] + 31
    v = (acc * L["mult"][None, :, None, None] + (np.int64(1) << (sh - 1))) >> sh
    return np.clip(v + c0, 0, 255)


def qmul(x, l, M, s):
    return ((x.astype(np.int64) << l) * M + (np.int64(1) << (s - 1))) >> s


def conv(xpad, W, S, bz, K=3):
    """xpad: int [C, H+K-1, W+K-1] (int8 values), W: OIHW -> int64 [O, Ho, Wo]."""
    C, Hp, Wp = xpad.shape
    Ho, Wo = (Hp - K) // S + 1, (Wp - K) // S + 1
    acc = np.zeros((W.shape[0], Ho, Wo), np.int64) + bz[:, None, None]
    x = xpad.astype(np.int64)
    for kh in range(K):
        for kw in range(K):
            patch = x[:, kh: kh + S * Ho: S, kw: kw + S * Wo: S]
            acc += np.einsum("oc,chw->ohw", W[:, :, kh, kw].astype(np.int64), patch)
    return acc


def run_layer(L, xpad, res):
    """One engine layer on int8 [C, H+2, W+2]; returns (v8 out, res for next)."""
    acc = conv(xpad, L["W"], L["S"], L["bz"])[None]
    v = requant(acc, L, L["c0"])[0]
    nres = None
    if L["down"] is not None:
        d = L["down"]
        accd = conv(xpad[:, 1:-1, 1:-1], d["W"], 2, d["bz"], K=1)[None]
        nres = requant(accd, d, d["c0"])[0]
    if L["grab"]:
        nres = xpad[:, 1:-1, 1:-1].astype(np.int64) + 128
    if L["comb"] is not None:
        c = L["comb"]
        t = qmul(v - c["szp"], c["sl"], c["sM"], c["ss"]) + qmul(res - c["rzr"], c["rl"], c["rM"], c["rs"])
        v = np.clip(t, 0, 255)
    return v, nres


def from_nchwc(t):
    """[1, C/4, H, W, 4] -> [C, H, W]."""
    _, cb, h, w, f = t.shape
    return t[0].transpose(0, 3, 1, 2).reshape(cb * f, h, w)


def model_network(net: Net, layers: list) -> dict:
    x = from_nchwc(dump(net.deploy, 2, "out")[0]).astype(np.int64)  # maxpool, uint8
    x8 = np.pad(x - 128, ((0, 0), (1, 1), (1, 1)), constant_values=-128)
    res, outs = None, {}
    for L in layers:
        v, nres = run_layer(L, x8, res)
        want = from_nchwc(dump(net.deploy, L["golden_call"], "out")[0]).astype(np.int64)
        got = v
        if L.get("final_q") is not None:
            fq = L["final_q"]
            got = qmul(v, fq["l"], fq["M"], fq["s"])
        elif L["golden_dtype"] == "int8":
            got = v - 128
        bad = int((got != want).sum())
        print(f"  layer call {L['call']:2d} S{L['S']} {L['Cin']:3d}->{L['Cout']:3d} {L['Ho']:2d}px "
              f"down={'y' if L['down'] else '-'} grab={'y' if L['grab'] else '-'} "
              f"comb={'y' if L['comb'] else '-'} vs call {L['golden_call']:2d}: "
              f"{'OK' if bad == 0 else f'{bad} MISMATCH'}")
        if bad:
            raise SystemExit("model mismatch -- extraction or fusion is wrong")
        outs[L["call"]] = v
        x8 = np.pad(v - 128, ((0, 0), (1, 1), (1, 1)), constant_values=-128)
        res = nres
    return outs



# ── emission ───────────────────────────────────────────────────────────────

EH = ["MAGIC", "LAYER", "S", "Q", "P", "WST", "WO", "MB", "FLAGS", "C0", "C0D", "SZP", "SL", "SM", "SS",
      "RZR", "RL", "RM", "RS", "NTAP", "CHUNK", "RING_SLOTS", "A_RING", "A_ACC", "A_ACCD", "A_OUT", "A_RES",
      "OUT_SLOTS", "A_EPD", "A_SROW", "ROWS"]
ENG_MAGIC = 0x31474E45


def wst(w):
    return (w + 2 + 7) // 8 * 8


def mesh_rows(ho):
    per = -(-ho // MESH)
    rows = [max(0, min(per, ho - r * per)) for r in range(MESH)]
    assert all(rows), f"Ho={ho}: a mesh row would get no rows"
    return rows


STEM_IN_ROW, STEM_SLOT, STEM_TAPS = 928, 960, 28


def plan_stem(L):
    rows = mesh_rows(L["Ho"])
    g = dict(S=2, Q=1, P=1, rows=rows, MB=7, WR=56, WST=232, WSTO=wst(56), Hin=230, NTAP=STEM_TAPS,
             resident=False, A_EPD=0, A_ACC=0, A_ACCD=0, A_RES=DM_TOP)
    g["chunk"] = PRM_BYTES + STEM_TAPS * 128
    a = DM_W + 2 * g["chunk"]
    g["A_SROW"], a = a, a + 3 * 2 * 112 * 8
    g["A_OUT"], a = a, a + 2 * 2 * 56 * 8
    g["A_RING"] = a
    g["ring_slots"] = min((DM_TOP - a - RING_SLACK) // STEM_SLOT, 31)
    assert g["ring_slots"] >= 9
    g["rows_in"] = [61] + [63] * 3
    g["used"] = DM_TOP - DM_W
    return g


def chunk_stem(L, c):
    oc = np.array([(c * 2 + i // 8) * 8 + i % 8 for i in range(16)])
    prm = bytearray(PRM_BYTES)
    prm[0:192] = params(L, oc)
    W = np.zeros((16, 4, 7, 8), np.int64)                 # [oc][ci 4][kh][kw 8]
    W[:, :L["W"].shape[1], :, :7] = L["W"][oc]
    # tap (kh, kp): k = (kw & 1) * 4 + ci, kw = 2 kp + (k >> 2) -> [nb 2][k 8][n 8]
    taps = [W[:, :, kh, 2 * kp:2 * kp + 2].transpose(0, 2, 1).reshape(16, 8)
            for kh in range(7) for kp in range(4)]
    b = np.stack([t.reshape(2, 8, 8).transpose(0, 2, 1) for t in taps])
    return bytes(prm) + b.astype(np.int8).tobytes()


def plan(L, prev_res):
    """Geometry + core-memory map of one layer (addresses are core view)."""
    if L.get("stem"):
        return plan_stem(L)
    S, Q, P = L["S"], L["Cin"] // 64, L["Cout"] // 64
    rows = mesh_rows(L["Ho"])
    rmax, mb = max(rows), -(-L["Wo"] // 8)
    win = L["Ho"] * S if S == 2 else L["Ho"]
    g = dict(S=S, Q=Q, P=P, rows=rows, MB=mb, WR=mb * 8, WST=wst(win), WSTO=wst(L["Wo"]), Hin=win,
             NTAP=10 if L["down"] else 9, ring_slots=3 + S)
    g["chunk"] = PRM_BYTES + g["NTAP"] * CHUNK_BLK * 128
    g["rows_in"] = [(r - 1) * S + 3 for r in rows]
    assert P * Q * max(g["rows_in"]) <= 256, "input task repeat > 256"
    res_bytes = rmax * P * 2 * g["WR"] * 8
    uses_res = L["grab"] or L["down"] is not None or L["comb"] is not None
    a = DM_W + 2 * g["chunk"]
    g["A_EPD"] = a if L["down"] else 0
    a += EPSET_BYTES if L["down"] else 0
    acc = (rmax if Q > 1 else 1) * mb * 2 * 256  # Q == 1: one row (the epilogue reads it back)
    others = acc * (2 if L["down"] else 1) + (2 if P == 1 else rmax) * P * 2 * g["WR"] * 8
    top = DM_TOP - res_bytes if uses_res else DM_TOP
    slot = CHUNK_BLK * g["WST"] * 8
    room = (top - a - others - RING_SLACK) // slot
    resident = P > 1 and room >= Q * max(g["rows_in"])
    if resident:          # every input row of every chunk stays: streamed once, not P times
        g["ring_slots"] = Q * max(g["rows_in"])
    else:                 # spare memory -> prefetch: up to two chunks' rows; short of memory, the
        # ring may shrink to S + 2 slots (the 3-row window + the S rows the next row brings)
        g["ring_slots"] = max(min(g["ring_slots"], max(room, S + 2)), min(room, 2 * max(g["rows_in"]), 31))
    assert room >= g["ring_slots"], f"call {L['call']}: core memory overflow"
    g["resident"] = resident
    g["A_RING"] = a
    a += g["ring_slots"] * slot + RING_SLACK
    g["A_ACC"], a = a, a + acc
    g["A_ACCD"], a = a, a + (acc if L["down"] else 0)
    g["A_OUT"] = a
    a += (2 if P == 1 else rmax) * P * 2 * g["WR"] * 8
    g["A_RES"] = top
    assert a <= top
    if L["comb"] is not None:
        assert prev_res == g["A_RES"], "residual moved between producer and consumer"
    if L["down"]:  # layer-major schedule (ENG_F_DOWNOUT): OUT row slots hold [main | v_d]
        assert g["A_OUT"] + 2 * (a - g["A_OUT"]) <= DM_TOP, f"call {L['call']}: DOWNOUT slots overflow"
    if L["comb"] is not None and resident:  # ENG_F_RESX staging: 2 input slots inside OUT
        assert a - g["A_OUT"] >= 2 * slot, f"call {L['call']}: RESX staging does not fit in OUT"
    g["used"] = a - DM_W + (res_bytes if uses_res else 0)
    return g


RI_P = 4  # passes per row=image sub-layer (core memory: ACC grows with P' x all rows)


def ri_eligible(L):
    """7x7 stride-1 layers run row=image in the layer-major schedule: each mesh
    row takes a different image, so one column weight stream serves 4 images."""
    return not L.get("stem") and L["S"] == 1 and L["Ho"] == 7 and L["down"] is None


def plan_ri(L):
    """Core-memory map of a row=image sub-layer (RI_P passes, all 7 rows, q-outer)."""
    S, Q, P = 1, L["Cin"] // 64, RI_P
    rows, mb = [L["Ho"]] * MESH, -(-L["Wo"] // 8)
    g = dict(S=S, Q=Q, P=P, rows=rows, MB=mb, WR=mb * 8, WST=wst(L["Ho"]), WSTO=wst(L["Wo"]), Hin=L["Ho"],
             NTAP=9, resident=False, ri=True, A_EPD=0, A_ACCD=0)
    g["chunk"] = PRM_BYTES + 9 * CHUNK_BLK * 128
    g["rows_in"] = [(r - 1) * S + 3 for r in rows]
    rmax = max(rows)
    a = DM_W + 2 * g["chunk"]
    acc = P * rmax * mb * 2 * 256          # ACC row = p * rows + oh
    out = rmax * P * 2 * g["WR"] * 8       # every row stays until the last pass
    res = out if L["comb"] is not None else 0
    top = DM_TOP - res
    slot = CHUNK_BLK * g["WST"] * 8
    room = (top - a - acc - out - RING_SLACK) // slot
    g["ring_slots"] = min(room, 2 * max(g["rows_in"]), 31)
    assert g["ring_slots"] >= max(g["rows_in"]), f"call {L['call']}: RI core memory overflow"
    g["A_RING"] = a
    a += g["ring_slots"] * slot + RING_SLACK
    g["A_ACC"], a = a, a + acc
    g["A_OUT"], a = a, a + out
    g["A_RES"] = top
    assert a <= top
    g["used"] = a - DM_W + res
    return g


def header(L, g):
    h = dict(MAGIC=ENG_MAGIC, LAYER=L["call"], S=g["S"], Q=g["Q"], P=g["P"], WST=g["WST"], WO=L["Wo"],
             MB=g["MB"], FLAGS=(1 if L["down"] else 0) | (2 if L["grab"] else 0) | (4 if L["comb"] else 0)
             | (8 if g["resident"] else 0) | (16 if L.get("stem") else 0) | (256 if g.get("ri") else 0),
             A_EPD=g["A_EPD"],
             A_SROW=g.get("A_SROW", 0),
             C0=L["c0"], C0D=L["down"]["c0"] if L["down"] else 0, NTAP=g["NTAP"], CHUNK=g["chunk"],
             RING_SLOTS=g["ring_slots"], A_RING=g["A_RING"], A_ACC=g["A_ACC"], A_ACCD=g["A_ACCD"],
             A_OUT=g["A_OUT"], A_RES=g["A_RES"], OUT_SLOTS=2 if g["P"] == 1 else max(g["rows"]))
    c = L["comb"] or {}
    for k in ("szp", "sl", "sM", "ss", "rzr", "rl", "rM", "rs"):
        h[k.upper()] = c.get(k, 0)
    words = [int(h.get(k, 0)) for k in EH[:-1]] + list(g["rows"])
    raw = struct.pack(f"<{len(words)}i", *words)
    return raw + bytes(HDR_BYTES - len(raw))


def params(src, oc):
    a = np.stack([src["bz"][oc], src["mult"][oc], src["shift"][oc]], 1)
    return ((a + 2**31) % 2**32 - 2**31).astype("<i4").tobytes()


def chunk(L, g, c, p, q):
    oc = np.array([((4 * p + c) * 2 + i // 8) * 8 + i % 8 for i in range(16)])
    prm = bytearray(PRM_BYTES)
    prm[0:192] = params(L, oc)
    if L["down"]:
        prm[256:448] = params(L["down"], oc)
    ic = np.arange(q * 64, q * 64 + 64)
    taps = [L["W"][oc][:, ic, kh, kw] for kh in range(3) for kw in range(3)]
    if L["down"]:
        taps.append(L["down"]["W"][oc][:, ic, 0, 0])
    # [oc 16][ic 64] -> [kc 8][nb 2][k 8][n 8]
    b = np.stack([t.reshape(2, 8, 8, 8).transpose(2, 0, 3, 1) for t in taps])
    return bytes(prm) + b.astype(np.int8).tobytes()


def to_l8(v):
    """[C, H, W] values 0..255 -> L8 int8 bytes [H+2][C/8][WST][8], border -128."""
    C, H, W = v.shape
    t = np.full((H + 2, C // 8, wst(W), 8), -128, np.int8)
    t[1:H + 1, :, 1:W + 1, :] = (v - 128).reshape(C // 8, 8, H, W).transpose(2, 0, 3, 1)
    return t.tobytes()


def emit(net, layers, outs, x1, img_in, out_dir: Path, cpu=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    blob, gold, rows_c, res = bytearray(), bytearray(), [], DM_TOP
    for L in layers:
        g = plan(L, res)
        res = g["A_RES"]
        hdr, offs = header(L, g), []
        for c in range(MESH):
            offs.append(len(blob))
            blob += hdr
            if L.get("stem"):
                blob += chunk_stem(L, c)
                continue
            for p in range(g["P"]):
                for q in range(g["Q"]):
                    blob += chunk(L, g, c, p, q)
        glen = len(to_l8(outs[L["call"]]))
        rows_c.append((L, g, offs, len(blob) - offs[-1], len(gold), glen))
        gold += to_l8(outs[L["call"]])
        print(f"  call {L['call']:2d}: P{g['P']} Q{g['Q']} chunk {g['chunk']} rows {g['rows']} "
              f"ring {g['ring_slots']}{' resident' if g['resident'] else ''} "
              f"core mem {g['used'] / 1024:.1f} KB  weights/col {(len(blob) - offs[-1]) / 1024:.0f} KB")
    lm = layer_major_table(rows_c, blob)
    (out_dir / "eng_blob.bin").write_bytes(blob)
    (out_dir / "eng_gold.bin").write_bytes(gold)
    (out_dir / "eng_x1.bin").write_bytes(to_l8(x1))
    img = np.zeros((230, 232, 4), np.int8)
    img[:, :230, :3] = (img_in.transpose(1, 2, 0)).astype(np.int8)
    (out_dir / "eng_img.bin").write_bytes(img.tobytes())
    lines = ["/* Generated by src/aietensorop/resnetengine/gen_engine.py -- do not edit. */",
             "#ifndef ENG_NET_H", "#define ENG_NET_H",
             f"#define ENG_NLAYERS {len(layers)}",
             f'#define ENG_BLOB_PATH "{(out_dir / "eng_blob.bin").resolve()}"',
             f'#define ENG_GOLD_PATH "{(out_dir / "eng_gold.bin").resolve()}"',
             f'#define ENG_X1_PATH "{(out_dir / "eng_x1.bin").resolve()}"',
             f'#define ENG_IMG_PATH "{(out_dir / "eng_img.bin").resolve()}"',
             "#define ENG_IMG_BYTES (230 * 232 * 4)",
             f"#define ENG_BLOB_BYTES {len(blob)}", f"#define ENG_GOLD_BYTES {len(gold)}",
             f"#define ENG_X1_BYTES {len(to_l8(x1))}",
             "/* in_blk/out_blk: 8-channel blocks per row of the input/output tensor (a DOWN",
             " * layer's output carries v_d after y: 2*Cout/8). res_act/res_blk0: where a COMB",
             " * layer's residual lives in DDR for the layer-major schedule (activation index,",
             " * first block), -1 if none. flags: the header's ENG_F_* (image-major schedule). */",
             "typedef struct {",
             "    int call, stem, S, P, Q, Cin, Cout, Hin, Ho, Wo, wst_in, wst_out, resident, rows[4];",
             "    long blob_off[4], blob_len, gold_off, gold_len;",
             "    int flags, in_blk, out_blk, res_act, res_blk0;",
             "    int p0, ri, in_act, out_act; /* first pass (sub-layer); row=image; act buffers read/written */",
             "} eng_layer_t;",
             "static const eng_layer_t eng_layers[ENG_NLAYERS] = {"]
    for i, (L, g, offs, blen, goff, glen) in enumerate(rows_c):
        lines.append(table_line(rows_c, i, L, g, offs, blen, goff, glen, 0, i))
    lines += ["};", f"#define ENG_LM_NLAYERS {len(lm)}",
              "/* layer-major schedule: the 7x7 stride-1 layers split into row=image sub-layers */",
              "static const eng_layer_t eng_lm_layers[ENG_LM_NLAYERS] = {"]
    lines += [table_line(rows_c, i, L, g, offs, blen, goff, glen, p0, i) for (i, L, g, offs, blen, goff, glen, p0) in lm]
    lines += ["};"] + (emit_cpu(cpu, out_dir) if cpu else []) + ["#endif", ""]
    (out_dir / "eng_net.h").write_text("\n".join(lines))
    print(f"wrote {out_dir}: blob {len(blob) / 1e6:.2f} MB, gold {len(gold) / 1e6:.2f} MB")


# ── CPU ends: input quantize (call 00), avgpool + flatten + dense (38..40) ──

def cpu_tail(net: Net, last: dict) -> dict:
    """Constants and blobs of the ops the A78 keeps, parsed from TVM's C so the
    float literals are the same tokens TVM compiled (bit-exact logits)."""
    q0 = (net.src[net.fn(0)]).read_text()
    m = re.search(r"roundf\(\(\(\(float\*\)p0_1\)\[[^\]]*\] \* ([-0-9.e+]+f)\)\) \+ ([-0-9.e+]+f);", q0)
    pad = re.search(r"condval = \(int8_t\)(-?\d+);", q0).group(1)
    k_dense = next(k for k in range(len(net.calls)) if "dense" in net.fn(k))
    body = net.body(k_dense)
    rq = _RQ.search(body)
    args = net.args(k_dense)
    bias, zpa, zpb, mult, shift = (net.param(args[int(i)]) for i in rq.groups())
    w = next(net.param(a) for a in args if net.is_param(a) and len(net.shape[net.rp[a]]) == 3)
    dm = re.search(r"\)\) \+ (\d+);\s*int32_t v__1.*?- (\d+)\)\) \* ([-0-9.e+]+f)\);", body, re.S)
    assert m and dm and "global_avg_pool2d" in net.fn(k_dense - 2), "CPU tail pattern changed"
    fq = last["final_q"]
    return dict(q_scale=m.group(1), q_zp=m.group(2), q_pad=int(pad), dense_c0=int(dm.group(1)),
                dense_zp=int(dm.group(2)), dense_scale=dm.group(3), dense_w=w.astype(np.int8),
                dense_bz=(bias.reshape(-1).astype(np.int64) - zpa.reshape(-1) - zpb.reshape(-1)),
                dense_mult=mult.reshape(-1), dense_shift=shift.reshape(-1), fq=fq,
                image=dump(net.deploy, 0, "in")[0].astype(np.float32),
                logits=dump(net.deploy, k_dense, "out")[0].astype(np.float32))


def table_line(rows_c, i, L, g, offs, blen, goff, glen, p0, act):
    """One eng_layer_t initializer. act: index of the layer's input activation."""
    prev = rows_c[i - 1][0] if i else None
    in_blk = L["Cin"] // 8 * (2 if prev is not None and prev["down"] else 1)
    out_blk = L["Cout"] // 8 * (2 if L["down"] else 1)
    res_act, res_blk0 = -1, 0
    if L["comb"] is not None:  # identity: block input (conv1's input); downsample: v_d half of conv1's output
        res_act, res_blk0 = (i, L["Cout"] // 8) if L["res_from_down"] else (i - 1, 0)
    flags = struct.unpack_from("<i", header(L, g), 4 * EH.index("FLAGS"))[0]
    return (f"    {{{L['call']}, {int(bool(L.get('stem')))}, {g['S']}, {g['P']}, {g['Q']}, {L['Cin']}, {L['Cout']}, {g['Hin']}, "
            f"{L['Ho']}, {L['Wo']}, {g['WST']}, {g['WSTO']}, {int(g['resident'])}, {{{', '.join(map(str, g['rows']))}}}, "
            f"{{{', '.join(map(str, offs))}}}, {blen}, {goff}, {glen}, {flags}, {in_blk}, {out_blk}, {res_act}, {res_blk0}, "
            f"{p0}, {int(bool(g.get('ri')))}, {act}, {act + 1}}},")


def layer_major_table(rows_c, blob):
    """Entries of the layer-major schedule: the spatial ones, except that a
    ri_eligible layer becomes P / RI_P row=image sub-layers (blob appended:
    per column header + chunks q-outer, passes p0 .. p0 + RI_P). Only the last
    sub-layer carries the golden (the output is complete after it)."""
    lm = []
    for i, (L, g, offs, blen, goff, glen) in enumerate(rows_c):
        if not ri_eligible(L):
            lm.append((i, L, g, offs, blen, goff, glen, 0))
            continue
        gr = plan_ri(L)
        for p0 in range(0, g["P"], RI_P):
            hdr, roffs = header(L, gr), []
            for c in range(MESH):
                roffs.append(len(blob))
                blob += hdr
                for q in range(gr["Q"]):
                    for p in range(p0, p0 + RI_P):
                        blob += chunk(L, gr, c, p, q)
            last = p0 + RI_P >= g["P"]
            lm.append((i, L, gr, roffs, len(blob) - roffs[-1], goff if last else 0, glen if last else 0, p0))
            print(f"  call {L['call']:2d} RI p0={p0}: P{gr['P']} Q{gr['Q']} ring {gr['ring_slots']} "
                  f"core mem {gr['used'] / 1024:.1f} KB  weights/col {(len(blob) - roffs[-1]) / 1024:.0f} KB")
    return lm


def emit_cpu(t: dict, out_dir: Path) -> list:
    (out_dir / "eng_image.bin").write_bytes(t["image"].tobytes())
    (out_dir / "eng_logits.bin").write_bytes(t["logits"].tobytes())
    (out_dir / "eng_dense.bin").write_bytes(t["dense_w"].tobytes())
    i32 = lambda a: ", ".join(str((int(v) + 2**31) % 2**32 - 2**31) for v in a)
    fq = t["fq"]
    lut = [int(((v << fq["l"]) * fq["M"] + (1 << (fq["s"] - 1))) >> fq["s"]) for v in range(256)]
    return [f'#define ENG_IMAGE_PATH "{(out_dir / "eng_image.bin").resolve()}"',
            f'#define ENG_LOGITS_PATH "{(out_dir / "eng_logits.bin").resolve()}"',
            f'#define ENG_DENSE_PATH "{(out_dir / "eng_dense.bin").resolve()}"',
            f"#define ENG_Q_SCALE {t['q_scale']}", f"#define ENG_Q_ZP {t['q_zp']}",
            f"#define ENG_Q_PAD {t['q_pad']}", f"#define ENG_DENSE_C0 {t['dense_c0']}",
            f"#define ENG_DENSE_ZP {t['dense_zp']}", f"#define ENG_DENSE_SCALE {t['dense_scale']}",
            f"#define ENG_NCLASS {t['dense_w'].shape[0] * t['dense_w'].shape[2]}",
            f"#define ENG_FEAT {t['dense_w'].shape[1]}",
            f"static const int32_t eng_dense_bz[] = {{{i32(t['dense_bz'])}}};",
            f"static const int32_t eng_dense_mult[] = {{{i32(t['dense_mult'])}}};",
            f"static const int32_t eng_dense_shift[] = {{{i32(t['dense_shift'])}}};",
            "/* avgpool input = Q(v << l, M) of the last engine layer's 8-bit output */",
            f"static const int32_t eng_final_lut[256] = {{{', '.join(map(str, lut))}}};"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deploy", default=str(REPO / "worklocal" / "tvmrelay_deploy"))
    ap.add_argument("--out", default=None, help="output dir (default: model check only)")
    a = ap.parse_args(argv)
    net = Net(Path(a.deploy))
    layers = build_layers(net)
    print(f"engine: {len(layers)} layers")
    stem = build_stem(net)
    x1, img_in = model_stem(net, stem)
    outs = model_network(net, layers)
    outs[1] = x1
    if a.out:
        emit(net, [stem] + layers, outs, x1, img_in, Path(a.out), cpu_tail(net, layers[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
