"""Per-site host layout features of the UNet AIE sites, from a build directory only (costmodel M2, N segment).

layout_sites(build) -> [dict] in graph_run order (one per aiednn_aie_* call = one board 'PROF part=<i> op=aie_host_*'):
  kind, cid, part           conv | linear | attn (attn_kl / attn steps), op cid, graph_run call index (= PROF part)
  in_pack_bytes             host bytes read by the pre-launch pack:
                              conv    x fp32 (C*H*W*4) + W copy N*Kp*2 when the W port is not zero-copy
                                      (Kp != K or the bf16 weight is not 64-B aligned in g_params)
                              linear  x fp32 (M*K*4)
                              attn    role 2 Q (N*H*D*4), role 3 K / role 5 V (M*H*D*4), role 6 0;
                                      attn role 0 Q+K, role 1 V
  out_scatter_bytes         host bytes the post-launch scatter writes + epilogue tensors it reads:
                              conv/linear  M*N*4 + sum(epilogue numel*4)   (bias / temb / residual)
                              attn         role 6 / role 1: O (N*H*D*4), pack roles 0
  relayout                  host transposes beyond the kernel's natural [px][ch] order:
                              conv    (y block yb == 1: planar NCHW output, keep_layout label)
                                      + epilogue residuals in NCHWc into that planar output
                              linear  0
                              attn    1 if the pack / O row stride (lda / ldb) != H*D (strided head view), else 0
  up, down                  'arm' | 'aie': the live call that produces the site's input / reads its output,
                            following TVM calls the build removed ('/* [i] removed: group ...') through to the
                            live ones. Pack steps write the FA class buffer (down = 'aie'); role 6 reads it
                            (up = 'aie'). down = 'arm' if any live reader is a cpu call.
Dataflow comes from graph_driver.c: actual args (args[j].v_handle / g_lbl_* labels / tensors[T].data pointers) of
the live calls and the 'in[k] = tensors[T]' / 'out[k] = tensors[T]' tensor notes the driver emits per call.
The (kind, cid) order is cross-checked against unet_seq.parse_driver.
"""
import os
import re

from costmodel.unet_seq import func_body, parse_driver, read

_ARG = re.compile(r"args\[(\d+)\]\.v_handle = &(tensors\[(\d+)\]|\w+);")
_LBL = re.compile(r"(\w+) = tensors\[(\d+)\]; \1\.ndim = (\d+); \1\.shape = (\w+);")
_HEX = re.compile(r'graph_hex_tensor\(hf, "l(\d+)_[^"]*", (\d+), "(in|out)\[\d+\] = tensors\[(\d+)\] \(([^)]*)\)')
_END = re.compile(r'GRAPH_PROF_END\((\d+), "(aie|cpu) ')
_RM = re.compile(r"/\* \[(\d+)\] removed: ")
_CALL = re.compile(r"aiednn_aie_(conv|linear|attn_kl|attn)\(([^;]*)\);")
_PTR = re.compile(r"\(\s*(?:const )?float \*\)\s*tensors\[(\d+)\]\.data")


def _fa_cls(src):
    hdr = re.search(r"typedef struct \{\s*int ([^;]*);\s*\} fa_cls;", src)
    fields = [f.strip() for f in hdr.group(1).split(",")]
    m = re.search(r"static const fa_cls g_fa_cls\[\] = \{(.*?)\n\};", src, re.S)
    return [dict(zip(fields, (int(v) for v in row.split(",")))) for row in re.findall(r"\{([^{}]*)\}", m.group(1))]


def _numel(s):
    n = 1
    for v in s:
        n *= v
    return n


def _conv_kp(M, N, K):
    """uc_conv / uc_linear kacc geometry: Kp (zero chunk when the W port would be square)"""
    tkw = 4096 // M if 4096 // M < 32 else 32
    if 2 * (16 * tkw * 4 + (M // 4) * tkw * 4 + 16 * (M // 4) * 4) > 29696:
        tkw = 32
        while (N // 4) * tkw > 1024 or (K // 2) % tkw:
            tkw //= 2
    kch = 2 * tkw
    return K + kch if N == K // 2 else K


def dataflow(src):
    """status[i] in aie|cpu|removed, producer[T] -> i, readers[T] -> {i}, ins[i] / outs[i] tensor lists"""
    status = {int(i): k for i, k in _END.findall(src)}
    for i in _RM.findall(src):
        status.setdefault(int(i), "removed")
    producer, readers, ins, outs = {}, {}, {}, {}
    for i, t, io, t2, note in _HEX.findall(src):
        i, t = int(i), int(t)
        if "param " in note:
            continue
        if io == "out":
            producer.setdefault(t, i)
            outs.setdefault(i, []).append(t)
        else:
            readers.setdefault(t, set()).add(i)
            ins.setdefault(i, []).append(t)
    return status, producer, readers, ins, outs


def _up(t, df, seen=None):
    status, producer, _r, ins, _o = df
    seen = seen if seen is not None else set()
    p = producer.get(t)
    if p is None or p in seen:
        return "arm"  # graph input
    seen.add(p)
    st = status.get(p, "cpu")
    if st == "aie":
        return "aie"
    if st == "cpu":
        return "arm"
    ups = [_up(u, df, seen) for u in ins.get(p, [])]
    return "aie" if ups and all(u == "aie" for u in ups) else "arm"


def _down(t, df, self_i, seen=None):
    status, _p, readers, _i, outs = df
    seen = seen if seen is not None else set()
    res = []
    for r in sorted(readers.get(t, ())):
        if r == self_i or r in seen:
            continue
        seen.add(r)
        st = status.get(r, "cpu")
        if st == "removed":
            for o in outs.get(r, []):
                res += _down(o, df, self_i, seen)
        else:
            res.append("aie" if st == "aie" else "arm")
    return res


def layout_sites(build):
    gsrc = read(os.path.join(build, "ag_work", "arm_build", "graph_driver.c"))
    fa = _fa_cls(read(os.path.join(build, "aiek_src", "aiek_unetconv.cc")))
    seq = parse_driver(gsrc)
    shape_store = {}
    for i, j, v in re.findall(r"shape_store\[(\d+)\]\[(\d+)\] = (-?\d+);", gsrc):
        shape_store.setdefault(int(i), {})[int(j)] = int(v)
    static_arr = {n: [int(x) for x in v.split(",")] for n, v in
                  re.findall(r"static int64_t (\w+)\[\d+\] = \{([^}]*)\};", gsrc)}
    tshape = {}
    for t, k in re.findall(r"tensors\[(\d+)\]\.shape = shape_store\[(\d+)\];", gsrc):
        st = shape_store.get(int(k), {})
        tshape[int(t)] = [st[j] for j in sorted(st)]
    gi = gsrc.index("float *graph_run(void) {")
    run = func_body(gsrc[gi:], r"float \*graph_run\(void\) \{")
    df = dataflow(run)
    pbeg = {int(m.group(1)): m.start() for m in re.finditer(r"GRAPH_PROF_BEGIN\((\d+)\);", run)}

    # walk the body once: args / labels state at each aie call
    events = []
    for m in _ARG.finditer(run):
        events.append((m.start(), "arg", m))
    for m in _LBL.finditer(run):
        events.append((m.start(), "lbl", m))
    for m in _CALL.finditer(run):
        events.append((m.start(), "call", m))
    events.sort(key=lambda e: e[0])
    args, lbl = {}, {}
    out = []
    for pos, ev, m in events:
        if ev == "arg":
            args[int(m.group(1))] = m.group(2)
            continue
        if ev == "lbl":
            lbl[m.group(1)] = (int(m.group(2)), static_arr[m.group(4)])
            continue
        part = max((i for i, p in pbeg.items() if p < pos), key=lambda i: pbeg[i])
        if df[0].get(part) != "aie":
            raise ValueError("aiednn call at part %d not in an aie PROF block" % part)
        kind, a = m.group(1), [s.strip() for s in m.group(2).split(",")]

        def h(j):
            v = args[j]
            mm = re.match(r"tensors\[(\d+)\]", v)
            if mm:
                return int(mm.group(1)), tshape[int(mm.group(1))]
            return lbl[v]

        cid = int(a[0])
        s = dict(part=part, cid=cid)
        if kind in ("conv", "linear"):
            nargs = int(a[2])
            (tx, xs), (tw, ws), (ty, ys) = h(0), h(1), h(nargs - 1)
            eps = [h(j) for j in range(2, nargs - 1)]
            woff = int(re.search(r"g_params \+ (\d+)", a[3]).group(1))
            if kind == "conv":
                C, H, W = xs[1] * xs[4], xs[2], xs[3]
                N, KK = ws[0] * ws[5], ws[2] * ws[3]
                M, K, yb = ys[2] * ys[3], C * KK, ys[4]
                Kp = _conv_kp(M, N, K)
                wcopy = 0 if (Kp == K and woff % 64 == 0) else N * Kp * 2
                in_b = C * H * W * 4 + wcopy
                rel = int(yb == 1) + sum(1 for _t, es in eps if yb == 1 and _numel(es) == M * N and es[4] != 1)
            else:
                M, K, N = xs[0], xs[1], ys[1]
                in_b, rel = M * K * 4, 0
            out_b = M * N * 4 + sum(_numel(es) * 4 for _t, es in eps)
            tin = [tx]
            tout = ty
            s.update(kind=kind, M=M, N=N, K=K)
        elif kind == "attn_kl":
            role, k = int(a[1]), int(a[2])
            c = fa[k]
            hd = c["H"] * c["D"]
            if role in (2, 3, 5):
                tin = [int(_PTR.search(a[3]).group(1))]
                lda = int(a[4])
                rows = c["N"] if role == 2 else c["M"]
                in_b, out_b, rel, tout = rows * hd * 4, 0, int(lda != hd), None
            elif role == 6:
                tin = []
                tout = int(_PTR.search(a[5]).group(1))
                ldb = int(a[6])
                in_b, out_b, rel = 0, c["N"] * hd * 4, int(ldb != hd)
            else:
                raise ValueError("attn_kl role %d" % role)
            s.update(kind="attn", role=role, cls=k)
        else:  # aiednn_aie_attn(cid, args, nargs, role, cls)
            role, k = int(a[3]), int(a[4])
            c = fa[k]
            hd = c["H"] * c["D"]
            if role == 0:
                tin = [h(0)[0], h(1)[0]]
                in_b, out_b, tout = (c["N"] + c["M"]) * hd * 4, 0, None
            else:
                tin = [h(1)[0]]
                tout = h(2)[0]
                in_b, out_b = c["M"] * hd * 4, c["N"] * hd * 4
            rel = 0
            s.update(kind="attn", role=role, cls=k)
        s["up"] = ("aie" if all(_up(t, df) == "aie" for t in tin) else "arm") if tin else "aie"
        if tout is None:
            s["down"] = "aie"
        else:
            d = _down(tout, df, part)
            s["down"] = "arm" if (not d or "arm" in d) else "aie"
        s.update(in_pack_bytes=in_b, out_scatter_bytes=out_b, relayout=rel)
        out.append(s)

    # cross-check with unet_seq.parse_driver (kind, cid) order
    ref = [("attn" if c[0] in ("attn", "attn_kl") else c[0], c[1]) for c in seq]
    mine = [(s["kind"], s["cid"]) for s in out]
    if ref != mine:
        raise ValueError("n_layout site order differs from parse_driver: %d vs %d sites" % (len(mine), len(ref)))
    return out
