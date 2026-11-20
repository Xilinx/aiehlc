#!/usr/bin/env python3
"""kswitch.py -- M1.2.1 P0b: static per-tile kernel-switch count for the UNet aiegraph build.

(Moved into the repo in M1.2.2 P2 from maxwell tasks/2026-11-20_M1-2/kswitch.py; unet_launches() is the
library entry kload_static.py uses.)

Reads ONLY the build directory (--unet-build):
  ag_work/arm_build/graph_driver.c                  graph_run call order + tensor shapes (graph_init)
  aiek_src/aiek_unetconv.cc                         uc_conv / uc_linear launch geometry, uc_kacc_launch and
                                                    uc_fa_launch dispatch, g_fa_cls launches, which sites
                                                    sample __Runtime_pm_warm_stats (the board kloads window)
  ag_work/aiek/aiek_unetconv/worklocal/host.cc      kernel name -> ELF blob (__Runtime_set_kernel_elf) and
                                                    host_canonicalized_<k> -> tiles (__Runtime_load_kernel_group_*)
and replays the runtime's PM_WARM_SKIP rule (aie_runtime.c rt_pm_warm_resident, :7371-7376): a tile's
program memory is skipped only if this loader last wrote the SAME ELF pointer there, in the same partition
generation, with no foreign PM write since. In this build the partition is set up once and no foreign PM
writer runs (ctrlpkt=off), so the rule reduces to "same ELF pointer on that tile as last time".

The board's kloads / cold_tile_loads (aiek_unetconv.cc g_uc_kloads / g_uc_cold_tiles) are sampled only around
the launches of the sites that call __Runtime_pm_warm_stats (uc_conv, uc_linear); a cold load in an
unsampled launch (FA, fa_run_o) is not in the board counter. switches / tile_loads on the KSWITCH line are
therefore over the sampled launches (like for like); KSWITCH_ALL prints every launch.

--check-log: only the LAUNCH warm line (kloads, cold_tile_loads, launches) and the CMP board= are read, for
comparison; nothing from the log feeds the simulation.
"""
import argparse
import os
import re
import sys


def die(msg):
    print("KSWITCH ERROR " + msg)
    sys.exit(2)


def read(p):
    if not os.path.isfile(p):
        die("missing " + p)
    with open(p, errors="replace") as f:
        return f.read()


def func_body(src, name_re):
    """text of the first function whose definition line matches name_re (brace matched)"""
    m = re.search(name_re, src)
    if not m:
        return None
    i = src.index("{", m.end() - 1 if src[m.end() - 1] == "{" else m.end())
    d = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return src[i:j + 1]
    return None


# ---------------------------------------------------------------- aiek source: geometry + dispatch
def parse_aiek(src, require_warm=True):
    kacc = func_body(src, r"static int uc_kacc_launch\([^)]*\)\s*\{")
    if not kacc:
        die("uc_kacc_launch not found")
    gemm = {}
    for m in re.finditer(r"if \(N == (\d+) && K == (\d+) && M == (\d+)\) \{\s*gemm_kacc<([^>]*)><<<mesh>>>", kacc):
        targs = [a.strip() for a in m.group(4).split(",")]
        gemm[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = "gemm_kacc__" + "_".join(targs)
    fal = func_body(src, r"static int uc_fa_launch\([^)]*\)\s*\{")
    if not fal:
        die("uc_fa_launch not found")
    fa = {}
    for m in re.finditer(r"if \(k == (\d+)\) \{\s*fa_kernel<([^>]*)><<<mesh>>>", fal):
        fa[int(m.group(1))] = "fa_kernel__" + "_".join(a.strip() for a in m.group(2).split(","))
    m = re.search(r"static const fa_cls g_fa_cls\[\] = \{(.*?)\n\};", src, re.S)
    if not m:
        die("g_fa_cls not found")
    hdr = re.search(r"typedef struct \{\s*int ([^;]*);\s*\} fa_cls;", src)
    fields = [f.strip() for f in hdr.group(1).split(",")]
    fa_launches = []
    for row in re.findall(r"\{([^{}]*)\}", m.group(1)):
        vals = [int(v) for v in row.split(",")]
        fa_launches.append(dict(zip(fields, vals))["launches"])
    # which launch sites sample the board's kload counter
    sampled = {}
    for fn, pat in (("uc_conv", r"static int uc_conv\([^)]*\)\s*\{"),
                    ("uc_linear", r"static int uc_linear\([^)]*\)\s*\{"),
                    ("fa_run_o", r"static int fa_run_o\([^)]*\)\s*\{")):
        b = func_body(src, pat)
        if b is None:
            die(fn + " not found")
        sampled[fn] = "__Runtime_pm_warm_stats" in b
    warm = re.search(r"^#define UC_WARM (\d+)", src, re.M)
    if require_warm and (not warm or warm.group(1) != "1"):
        die("UC_WARM is not 1 in this build: the board counter is not compiled in")
    return gemm, fa, fa_launches, sampled


def geom_tkw(P, N, K):
    """uc_conv / uc_linear tkw (= kacc_geom); only used to sanity check the kch constraint"""
    tkw = 4096 // P if 4096 // P < 32 else 32
    tr = 2 * (16 * tkw * 4 + (P // 4) * tkw * 4 + 16 * (P // 4) * 4) > 29696
    if tr:
        tkw = 32
        while (N // 4) * tkw > 1024 or (K // 2) % tkw:
            tkw //= 2
    return tkw


# ---------------------------------------------------------------- host.cc: kernel -> ELF, tiles
def parse_host(src):
    elf = {}
    for m in re.finditer(r'strcmp\(kernel, "([^"]+)"\) == 0\) \{\s*__Runtime_set_kernel_elf\((\w+)\)', src):
        elf[m.group(1)] = m.group(2)
    tiles = {}
    for m in re.finditer(r"^void host_canonicalized_(\w+)\(", src, re.M):
        name = m.group(1)
        body = func_body(src[m.start():], r"^void host_canonicalized_\w+\([^)]*\)\s*\{")
        loc = {v: (int(c), int(r)) for v, c, r in re.findall(r"XAie_LocType (v\d+) = XAie_TileLoc\((\d+), (\d+)\);", body)}
        groups = re.findall(r"__Runtime_load_kernel_group_(\d+)t(_ctrl)?\(([^;]*)\);", body)
        if len(groups) != 1:
            die("host_canonicalized_%s: %d load_kernel_group calls" % (name, len(groups)))
        width, ctrl = int(groups[0][0]), bool(groups[0][1])
        args = [a.strip() for a in groups[0][2].split(",")]
        first = 2 if ctrl else 1  # (dev, [&fabric,] t0..t{w-1}, n[, bd_id, mm2s_ch])
        n = int(args[first + width])
        tl = [loc[a] for a in args[first:first + n] if a in loc]
        if len(tl) != n:
            die("host_canonicalized_%s: %d tiles, group says %d" % (name, len(tl), n))
        tiles[name] = tl
    return elf, tiles


# ---------------------------------------------------------------- graph_driver.c: launch order
def strip_trace(src):
    """drop #ifdef GRAPH_TRACE* blocks (not defined in the board build)"""
    out, stack = [], []
    for ln in src.split("\n"):
        s = ln.strip()
        if s.startswith("#if"):
            stack.append(bool(re.match(r"#ifn?def GRAPH_TRACE", s)) and s.startswith("#ifdef"))
            continue
        if s.startswith("#endif"):
            stack.pop()
            continue
        if s.startswith("#else") or s.startswith("#elif"):
            if stack:
                stack[-1] = False if stack[-1] else stack[-1]
            continue
        if any(stack):
            continue
        out.append(ln)
    return "\n".join(out)


def parse_driver(src):
    shape_store = {}
    for i, j, v in re.findall(r"shape_store\[(\d+)\]\[(\d+)\] = (-?\d+);", src):
        shape_store.setdefault(int(i), {})[int(j)] = int(v)
    static_arr = {n: [int(x) for x in v.split(",")] for n, v in
                  re.findall(r"static int64_t (\w+)\[\d+\] = \{([^}]*)\};", src)}
    tens = {}
    for t, k in re.findall(r"tensors\[(\d+)\]\.shape = shape_store\[(\d+)\];", src):
        st = shape_store.get(int(k), {})
        tens[int(t)] = [st[j] for j in sorted(st)]
    gi = src.index("float *graph_run(void) {")
    run = strip_trace(func_body(src[gi:], r"float \*graph_run\(void\) \{"))
    handles = {}  # name -> shape for non-tensors[] handles
    args = {}
    seq = []
    stmt_re = re.compile(
        r"args\[(\d+)\]\.v_handle = &(tensors\[(\d+)\]|\w+);"
        r"|(\w+) = tensors\[(\d+)\]; \4\.ndim = (\d+); \4\.shape = (\w+);"
        r"|aiednn_silu_scale_x\(&(tensors\[(\d+)\]|\w+),"
        r"|aiednn_aie_conv\((\d+), args, (\d+), [^;]*?, (\d+), (\d+), (\d+), (\d+)\);"
        r"|aiednn_aie_linear\((\d+), args, (\d+), [^;]*?, (\d+)\);"
        r"|aiednn_aie_attn\((\d+), args, (\d+), (\d+), (\d+)\);"
        r"|aiednn_aie_attn_kl\((\d+), (\d+), (\d+),")

    def shp(h):
        m = re.match(r"tensors\[(\d+)\]", h)
        if m:
            return tens[int(m.group(1))]
        if h not in handles:
            die("unknown handle " + h)
        return handles[h]

    for m in stmt_re.finditer(run):
        g = m.groups()
        if g[0] is not None:
            args[int(g[0])] = g[1]
        elif g[3] is not None:
            arr = static_arr.get(g[6])
            if arr is None or len(arr) != int(g[5]):
                die("label %s shape %s" % (g[3], g[6]))
            handles[g[3]] = arr
        elif g[7] is not None:
            handles["g_silu_x_t"] = shp(g[7])
        elif g[9] is not None:
            nargs = int(g[10])
            seq.append(("conv", int(g[9]), [shp(args[0]), shp(args[1]), shp(args[nargs - 1])],
                        tuple(int(x) for x in g[11:15])))
        elif g[15] is not None:
            nargs = int(g[16])
            seq.append(("linear", int(g[15]), [shp(args[0]), shp(args[1]), shp(args[nargs - 1])], int(g[17])))
        elif g[18] is not None:
            seq.append(("attn", int(g[18]), int(g[20]), int(g[21])))
        elif g[22] is not None:
            seq.append(("attn_kl", int(g[22]), int(g[23]), int(g[24])))
    return seq


def launches_of(call, gemm, fa, fa_launches):
    """[(kernel, site_fn)] in launch order for one driver call (uc_conv / uc_linear / fa_run_o logic)"""
    kind = call[0]
    if kind == "conv":
        x, w, y = call[2]
        sh, sw, ph, pw = call[3]
        C, H, W = x[1] * x[4], x[2], x[3]
        O, KH, KW = w[0] * w[5], w[2], w[3]
        OH, OW = y[2], y[3]
        if OH != (H + 2 * ph - KH) // sh + 1 or OW != (W + 2 * pw - KW) // sw + 1:
            die("conv cid %d geometry" % call[1])
        M, N, K = OH * OW, O, C * KH * KW
        key = (N, K, M)
        if key not in gemm:
            die("conv cid %d (N,K,M)=%s has no gemm_kacc instance" % (call[1], key))
        return [(gemm[key], "uc_conv", "conv cid=%d M=%d N=%d K=%d" % (call[1], M, N, K))]
    if kind == "linear":
        x, w, y = call[2]
        S = call[3]
        M, K, N = x[0], x[1], y[1]
        Mp = (M + 63) & ~63
        if S < 1 or Mp % (64 * S):
            die("linear cid %d S=%d" % (call[1], S))
        P = Mp // S
        key = (N, K, P)
        if key not in gemm:
            die("linear cid %d (N,K,P)=%s has no gemm_kacc instance" % (call[1], key))
        return [(gemm[key], "uc_linear", "lin cid=%d M=%d N=%d K=%d px=%d s=%d/%d" % (call[1], M, N, K, P, s, S))
                for s in range(S)]
    if kind == "attn":
        role, k = call[2], call[3]
    else:
        role, k = call[2], call[3]
        role = 1 if role == 6 else (0 if role in (2, 3, 5) else role)
    if role != 1:
        return []
    return [(fa[k], "fa_run_o", "fa cid=%d cls=%d l=%d/%d" % (call[1], k, l, fa_launches[k]))
            for l in range(fa_launches[k])]


def unet_launches(b, require_warm=True):
    """[{kernel, site, desc, blob, tiles}] for every AIE launch of the UNet build *b*, in order, and the
    {site: sampled} map.  Reads only the build directory."""
    gemm, fa, fa_launches, sampled = parse_aiek(read(os.path.join(b, "aiek_src", "aiek_unetconv.cc")), require_warm)
    elf, tiles = parse_host(read(os.path.join(b, "ag_work", "aiek", "aiek_unetconv", "worklocal", "host.cc")))
    seq = parse_driver(read(os.path.join(b, "ag_work", "arm_build", "graph_driver.c")))
    out = []
    for c in seq:
        for kern, site, desc in launches_of(c, gemm, fa, fa_launches):
            if kern not in elf or kern not in tiles:
                die("kernel %s missing in host.cc" % kern)
            out.append({"kernel": kern, "site": site, "desc": desc, "blob": elf[kern], "tiles": tiles[kern]})
    return out, sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-build", required=True)
    ap.add_argument("--check-log", required=True)
    ap.add_argument("--cfg", default="off")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every launch")
    a = ap.parse_args()
    b = a.unet_build
    gemm, fa, fa_launches, sampled = parse_aiek(read(os.path.join(b, "aiek_src", "aiek_unetconv.cc")))
    elf, tiles = parse_host(read(os.path.join(b, "ag_work", "aiek", "aiek_unetconv", "worklocal", "host.cc")))
    seq = parse_driver(read(os.path.join(b, "ag_work", "arm_build", "graph_driver.c")))

    launches = []
    for c in seq:
        launches += launches_of(c, gemm, fa, fa_launches)

    resident = {}  # tile -> ELF blob symbol last written by the loader
    sw = tl = sw_all = tl_all = 0
    per = []
    for i, (kern, site, desc) in enumerate(launches):
        if kern not in elf or kern not in tiles:
            die("kernel %s missing in host.cc" % kern)
        blob = elf[kern]
        cold = [t for t in tiles[kern] if resident.get(t) != blob]
        for t in tiles[kern]:
            resident[t] = blob
        if cold:
            sw_all += 1
            tl_all += len(cold)
            if sampled[site]:
                sw += 1
                tl += len(cold)
        per.append((i, kern, site, desc, len(cold)))
        if a.verbose:
            print("[KSW] l=%d %s %s cold_tiles=%d%s" % (i, desc, kern, len(cold), "" if sampled[site] else " (unsampled)"))

    log = read(a.check_log)
    lw = re.search(r"^LAUNCH warm .*?launches=(\d+) .*?kloads=(\d+) .*?cold_tile_loads=(\d+)", log, re.M)
    bd = re.search(r"^CMP \S+ board=(\S+)", log, re.M)
    if not lw:
        die("no LAUNCH warm line in " + a.check_log)
    meas_launches, meas_k, meas_c = int(lw.group(1)), int(lw.group(2)), int(lw.group(3))
    board = bd.group(1) if bd else "unknown"

    n_samp = sum(1 for p in per if sampled[p[2]])
    n_un = len(per) - n_samp
    fa_cold = [p for p in per if not sampled[p[2]] and p[4]]
    print("KSWITCH_INFO kernels_used=%d gemm_instances=%d fa_instances=%d tiles_per_launch=%s sampled_sites=%s"
          % (len({p[1] for p in per}), len(gemm), len(fa), sorted({len(tiles[p[1]]) for p in per}),
             ",".join(k for k, v in sampled.items() if v)))
    print("KSWITCH_ALL launches=%d sampled=%d unsampled=%d switches_all=%d tile_loads_all=%d "
          "unsampled_switches=%d unsampled_tile_loads=%d" % (len(per), n_samp, n_un, sw_all, tl_all,
                                                            len(fa_cold), sum(p[4] for p in fa_cold)))
    ok = (len(per) == meas_launches and sw == meas_k and tl == meas_c)
    if not ok:
        # name the launches where the counts can diverge: sampled launches the model calls cold / warm
        print("KSWITCH_DIFF launches model=%d board=%d switches model=%d board=%d tile_loads model=%d board=%d"
              % (len(per), meas_launches, sw, meas_k, tl, meas_c))
        for p in per:
            if sampled[p[2]] and p[4]:
                print("[KSW-COLD] l=%d %s %s cold_tiles=%d" % (p[0], p[3], p[1], p[4]))
    print("KSWITCH cfg=%s board=%s launches=%d switches=%d tile_loads=%d meas_kloads=%d meas_cold_tile_loads=%d %s"
          % (a.cfg, board, len(per), sw, tl, meas_k, meas_c, "MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
