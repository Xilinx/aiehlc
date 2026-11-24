###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-RT1 x86 tests of the aie_host split (src/mlir/runtime/aie_host_split.h) and its UNet wiring
(aiednn templates + aiegraph.kincore / attnsite / boardmain; the example unet_aie.py is gone since unet-decl P4d):
1. the header alone: site selection, phase accumulation, window split dma + wait == window, PROF line format;
2. the generated attention host code (unet_aie.ATTN_HOST, verbatim) with a counting XTime / wait counter:
   pack/scatter/dma/wait land on the site, dma + wait == g_uc_aie_ticks (the aie_ms window is split, not moved);
3. the generated graph_driver PROF code (unet_aie.PROF_C and prof_c_attn()) compiled with gcc: every AIE site
   prints `PROF aie_host_split part=<n> kind=<attn|conv|linear> pack_ms= scatter_ms= dma_ms= wait_ms=` and a
   `PROF aie_host_split_total host_split_n=` line; CPU sites print none;
4. the kernel source writers export the two host-split entry points and the kacc conv/linear hosts are wired.
Run: python3 -m pytest -q src/mlir/runtime/test/test_host_split.py   (needs gcc/g++)"""
import ctypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parents[1]
ROOT = RT.parents[2]
UNET = ROOT / "example/aiegraph/unet_bk_sdm_tiny"
sys.path.insert(0, str(ROOT / "src"))
import aiednn  # noqa: E402
from aiegraph import attnsite, boardmain as _bm  # noqa: E402


class ua:  # noqa: N801
    """the texts the deleted example unet_aie.py exposed (unet-decl P4d), from the aiednn templates / aiegraph"""
    HOST_CC = (aiednn.template("conv2d/tile64_conv_host.cc") + aiednn.template("elementwise/epilogue_bias_residual.cc")
               + aiednn.template("conv2d/tile64_conv_entry.cc"))
    KACC_HOST = (aiednn.template("conv2d/kacc_conv_host_head.cc") + aiednn.template("layout/neon_tr4_f32.cc")
                 + aiednn.template("conv2d/kacc_conv_host.cc"))
    LINEAR_HOST = aiednn.template("gemm/kacc_linear_host.cc")
    ATTN_HOST = attnsite.ATTN_HOST
    PROF_C = _bm.prof_c()
    bmm_geom = staticmethod(attnsite.bmm_geom)
    prof_c_attn = staticmethod(attnsite.prof_c_attn)

GLUE = r" glue_ms=[\d.]+ flush_ms=[\d.]+ inval_ms=[\d.]+ launches=\d+"   # M2: appended, old fields unchanged
LINE = re.compile(r"^PROF aie_host_split part=(\d+) kind=(attn|conv|linear|fa) pack_ms=([\d.]+) scatter_ms=([\d.]+) "
                  r"dma_ms=([\d.]+) wait_ms=([\d.]+)(?=" + GLUE + "$)", re.M)
TOTAL = re.compile(r"^PROF aie_host_split_total host_split_n=(\d+) pack_ms=([\d.]+) scatter_ms=([\d.]+) "
                   r"dma_ms=([\d.]+) wait_ms=([\d.]+)(?=" + GLUE + "$)", re.M)
NOGLUE = " glue_ms=0.000 flush_ms=0.000 inval_ms=0.000 launches=0"


def _cc(src: str, name: str, cxx: bool = False, shared: bool = True) -> Path:
    d = Path(tempfile.mkdtemp(prefix="hs_"))
    f = d / (name + (".cc" if cxx else ".c"))
    f.write_text(src)
    out = d / (name + (".so" if shared else ""))
    cmd = ["g++" if cxx else "gcc", "-O2", "-Wall", "-Werror", "-Wno-unused-function", "-Wno-unused-variable",
           f"-I{RT}", "-o", str(out), str(f)]
    if shared:
        cmd[1:1] = ["-shared", "-fPIC"]
    subprocess.check_call(cmd)
    return out


# ---------------------------------------------------------------- 1. header
HDR = r"""
#include "aie_host_split.h"
static ahs_table T;
void set_site(int s) { ahs_set_site(&T, s); }
void begin(int k) { ahs_begin(&T, k); }
void add(int ph, unsigned long long t) { ahs_add(&T, ph, t); }
void span(int ph, unsigned long long a, unsigned long long b) { ahs_add_span(&T, ph, a, b); }
void window(unsigned long long w, unsigned long long wt) { ahs_add_window(&T, w, wt); }
int count(void) { return ahs_count(&T); }
int fmt(int s, double hz, char *b, int n) { return ahs_format(&T, s, hz, b, (size_t)n); }
unsigned long long ticks(int s, int ph) { return T.s[s].t[ph]; }
unsigned calls(int s) { return T.s[s].calls; }
"""


def _hdr():
    L = ctypes.CDLL(str(_cc(HDR, "hdr")))
    for f in ("add", "span", "window"):
        getattr(L, f).argtypes = None
    L.ticks.restype = ctypes.c_ulonglong
    L.fmt.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_int]
    return L


def _fmt(L, s, hz=1e6):
    b = ctypes.create_string_buffer(256)
    n = L.fmt(s, hz, b, 256)
    return n, b.value.decode()


U = ctypes.c_ulonglong


def test_header_accumulate_and_format():
    L = _hdr()
    PACK, SCAT, DMA, WAIT = 0, 1, 2, 3
    L.add(PACK, U(5))                       # no site selected: dropped
    assert L.count() == 0 and _fmt(L, 3) == (0, "") and _fmt(L, -1)[0] == 0
    L.set_site(3)
    L.begin(1)                              # conv
    L.span(PACK, U(1000), U(3500))          # 2500
    L.span(SCAT, U(9000), U(8000))          # backwards -> 0
    L.window(U(4000), U(1500))              # dma 2500, wait 1500
    L.window(U(1000), U(5000))              # wait clamped to the window: dma 0, wait 1000
    L.span(SCAT, U(0), U(750))
    assert [L.ticks(3, p) for p in range(4)] == [2500, 750, 2500, 2500]
    assert L.ticks(3, DMA) + L.ticks(3, WAIT) == 4000 + 1000
    L.set_site(-1)
    L.add(DMA, U(10**6))                    # site cleared: dropped
    L.set_site(5000)                        # out of range: dropped
    L.add(DMA, U(10**6))
    L.set_site(7)
    L.begin(3)                              # attn
    L.add(WAIT, U(2000))
    L.set_site(8)
    L.begin(2)                              # linear, twice (two graph_run calls)
    L.begin(2)
    L.add(PACK, U(1))
    assert L.count() == 3 and L.calls(8) == 2
    n, s = _fmt(L, 3)
    assert n == len(s) and s == ("PROF aie_host_split part=3 kind=conv pack_ms=2.500 scatter_ms=0.750 "
                                 "dma_ms=2.500 wait_ms=2.500" + NOGLUE), s
    assert _fmt(L, 7)[1] == ("PROF aie_host_split part=7 kind=attn pack_ms=0.000 scatter_ms=0.000 dma_ms=0.000 "
                             "wait_ms=2.000" + NOGLUE)
    assert LINE.match(_fmt(L, 8)[1]).group(2) == "linear"
    assert _fmt(L, 4) == (0, "")            # never entered
    t = TOTAL.match(_fmt(L, -1)[1])
    assert t and t.groups() == ("3", "2.501", "0.750", "2.500", "4.500"), _fmt(L, -1)
    assert _fmt(L, 3, hz=2e6)[1].endswith("dma_ms=1.250 wait_ms=1.250" + NOGLUE)


# ---------------------------------------------------------------- 2. generated attention host
ATTN_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@DLT@
typedef unsigned long long XTime;
static XTime g_now = 0;
static void XTime_GetTime(XTime *t) { *t = (g_now += 10); }   /* every read advances 10 ticks */
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
static ahs_table g_ahs;
static unsigned long long g_wait = 0;
static inline unsigned long long uc_hs_wait(void) { return g_wait += 3; } /* 3 wait ticks per window */
#define UC_CTRLPKT 0
#define UC_KW_ELEMS (1 << 22)
#define UC_KX_ELEMS (1 << 22)
#define UC_KY_ELEMS (1 << 21)
static int32_t *g_kw = nullptr, *g_kx = nullptr, *g_ky = nullptr;
static int uc_kalloc(void) {
    if (g_ky)
        return 0;
    g_kw = (int32_t *)calloc(UC_KW_ELEMS / 2, 4);
    g_kx = (int32_t *)calloc(UC_KX_ELEMS / 2, 4);
    g_ky = (int32_t *)calloc(UC_KY_ELEMS, 4);
    return (g_kw && g_kx && g_ky) ? 0 : -1;
}
static unsigned long long g_uc_aie_ticks = 0, g_uc_macs = 0;
static int g_aie_launches = 0, g_uc_calls = 0, g_uc_rc = 0;
static void uc_sync_pre(const void *, size_t, const void *, size_t, const void *, size_t) {}
static void uc_sync_post(const void *, size_t) {}
static int uc_kacc_launch(int, int, int, int32_t *, int32_t *, int32_t *) { return 0; }
@ATTN@
extern "C" int site(int part, int cid, void *args, int qk, int swap, int CHP, int PXP, int Kp, int kp, int kch,
                    int splitpx, int S, int chs, int pxs, int tr) {
    ahs_set_site(&g_ahs, part);   /* graph_driver UP_AIE_BEGIN */
    const int rc = aiednn_aie_attn(cid, args, 3, qk, swap, CHP, PXP, Kp, kp, kch, splitpx, S, chs, pxs, tr);
    ahs_set_site(&g_ahs, -1);     /* UP_AIE_END */
    return rc;
}
extern "C" unsigned long long aie_ticks(void) { return g_uc_aie_ticks; }
extern "C" int launches(void) { return g_aie_launches; }
extern "C" unsigned long long ticks(int s, int ph) { return g_ahs.s[s].t[ph]; }
extern "C" int kind(int s) { return g_ahs.s[s].kind; }
extern "C" int line(int s, char *b, int n) { return ahs_format(&g_ahs, s, 1e6, b, (size_t)n); }
"""


class DLT(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32),
                ("ndim", ctypes.c_int32), ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.c_void_p), ("byte_offset", ctypes.c_uint64)]


def _dlt(a, keep):
    shp = (ctypes.c_int64 * a.ndim)(*a.shape)
    keep.append(shp)
    return DLT(a.ctypes.data, 1, 0, a.ndim, 2, 32, 1, shp, None, 0)


def test_attn_host_split():
    host = ua.HOST_CC
    i0 = host.index("typedef struct {\n    void *data;")
    dlt = host[i0:host.index("} uc_dlt;", i0) + len("} uc_dlt;")]
    L = ctypes.CDLL(str(_cc(ATTN_HARNESS.replace("@DLT@", dlt).replace("@ATTN@", ua.ATTN_HOST), "attn", cxx=True)))
    L.aie_ticks.restype = ctypes.c_ulonglong
    L.ticks.restype = ctypes.c_ulonglong
    rng = np.random.default_rng(1)
    part, tot_launch, tot_heads = 11, 0, 0
    for (H, M, K, N, qk) in ((2, 64, 64, 64, 1), (2, 256, 40, 77, 0)):
        g = ua.bmm_geom(H, M, K, N)
        X = rng.standard_normal((H, M, K)).astype(np.float32)
        W = rng.standard_normal((H, N, K)).astype(np.float32)
        Y = np.zeros((H, M, N), np.float32)
        keep = []
        dl = [_dlt(X, keep), _dlt(W, keep), _dlt(Y, keep)]
        args = (ctypes.c_void_p * 3)(*[ctypes.addressof(d) for d in dl])
        l0, a0 = L.launches(), L.aie_ticks()
        rc = L.site(part, 5, args, qk, g["swap"], g["CHP"], g["PXP"], g["Kp"], g["kp"], g["kch"],
                    g["splitpx"], g["S"], g["chs"], g["pxs"], g["tr"])
        assert rc == 0
        nl = L.launches() - l0
        assert nl == H * g["S"]
        tot_launch += nl
        tot_heads += H
        assert L.aie_ticks() - a0 == 10 * nl   # one XTime step per window: the aie_ms window is unchanged
    t = [L.ticks(part, p) for p in range(4)]
    assert L.kind(part) == 3
    assert t[0] == 10 * tot_heads and t[1] == 10 * tot_heads     # pack [ht0, ht1), scatter [ht2, ht3) per head
    assert t[3] == 3 * tot_launch and t[2] == 7 * tot_launch
    assert t[2] + t[3] == L.aie_ticks()                          # dma + wait == aie window
    b = ctypes.create_string_buffer(256)
    assert L.line(part, b, 256) > 0
    m = LINE.match(b.value.decode())
    assert m and m.group(1) == str(part) and m.group(2) == "attn", b.value
    assert all(L.ticks(p, q) == 0 for p in (0, part - 1, part + 1) for q in range(4))  # nothing leaked to others


# ---------------------------------------------------------------- 3. generated graph_driver PROF
DRV = r"""
#include <stdio.h>
#include <string.h>
typedef unsigned long long XTime;
static XTime g_now = 0;
static void XTime_GetTime(XTime *t) { *t = (g_now += 1000); }
#define COUNTS_PER_SECOND 1000000ULL
#define GRAPH_PROF_BEGIN(i)
#define GRAPH_PROF_END(i, tag)
#include "aie_host_split.h"
static ahs_table g_ahs;
static double g_aie = 0.0;
int aiednn_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles) {
    if (aie_ms) *aie_ms = g_aie;
    (void)kloads; (void)cold_tiles;
    return 0;
}
void aiednn_aie_host_split_site(int site) { ahs_set_site(&g_ahs, site); }
int aiednn_aie_host_split_line(int site, char *buf, int n) { return ahs_format(&g_ahs, site, 1e6, buf, (size_t)n); }
@PROF@
static void aie_site(int kind, unsigned long long pk, unsigned long long sc, unsigned long long win,
                     unsigned long long wt) {
    ahs_begin(&g_ahs, kind);
    ahs_add(&g_ahs, AHS_PACK, pk);
    ahs_add_window(&g_ahs, win, wt);
    ahs_add(&g_ahs, AHS_SCATTER, sc);
    g_aie += 1000.0 * (double)win / 1e6;
}
int main(void) {
    { GRAPH_PROF_BEGIN(0); GRAPH_PROF_END(0, "cpu tvmgen_add"); }
    { GRAPH_PROF_BEGIN(2); UP_AIE_BEGIN(2); aie_site(1, 2000, 500, 4000, 1000); UP_AIE_END(2);
      GRAPH_PROF_END(2, "aie aiednn_aie_conv 4"); }
    { GRAPH_PROF_BEGIN(3); UP_AIE_BEGIN(3); aie_site(2, 1000, 250, 3000, 3000); UP_AIE_END(3);
      GRAPH_PROF_END(3, "aie aiednn_aie_lin 9"); }
@ATTNSITE@
    ahs_add(&g_ahs, AHS_PACK, 123456);   /* outside any site: dropped */
    aiednn_prof_print(100.0, g_aie);
    return 0;
}
"""
ATTNSITE = ('    { GRAPH_PROF_BEGIN(5); UP_AIE_BEGIN(5); aie_site(3, 4000, 4000, 2000, 500); UP_AIE_END(5);\n'
            '      GRAPH_PROF_END(5, "aie aiednn_aie_attn 1"); }\n')


def _drv(prof: str, attn: bool) -> str:
    assert prof.count('#include "aie_timer.h"\n') == 1
    src = DRV.replace("@PROF@", prof.replace('#include "aie_timer.h"\n', "")).replace("@ATTNSITE@",
                                                                                      ATTNSITE if attn else "")
    # GRAPH_PROF_* are redefined by PROF_C itself (#undef + #define)
    return subprocess.run([str(_cc(src, "drv", shared=False))], check=True, capture_output=True,
                          text=True).stdout


def test_graph_driver_prof_lines():
    out = _drv(ua.PROF_C, False)
    rows = {int(m.group(1)): m.groups()[1:] for m in LINE.finditer(out)}
    assert rows == {2: ("conv", "2.000", "0.500", "3.000", "1.000"),
                    3: ("linear", "1.000", "0.250", "0.000", "3.000")}, out
    assert TOTAL.search(out).groups() == ("2", "3.000", "0.750", "3.000", "4.000"), out
    assert "PROF part=0 op=tvmgen_add" in out and "PROF part=2 op=aie_host_conv4" in out
    assert "PROF total_nonaie_ms=" in out
    # each split line follows its own site's PROF line
    assert out.index("PROF part=2 op=aie_host_conv4") < out.index("PROF aie_host_split part=2 ")
    out = _drv(ua.prof_c_attn(), True)
    rows = {int(m.group(1)): m.group(2) for m in LINE.finditer(out)}
    assert rows == {2: "conv", 3: "linear", 5: "attn"}, out
    assert TOTAL.search(out).group(1) == "3"
    assert "PROF part=5 op=aie_host_attn1" in out


def test_no_aie_sites_total_zero():
    src = DRV.replace("@PROF@", ua.PROF_C.replace('#include "aie_timer.h"\n', "")).replace("@ATTNSITE@", "")
    src = src.replace("aie_site(1, 2000, 500, 4000, 1000);", "").replace("aie_site(2, 1000, 250, 3000, 3000);", "")
    out = subprocess.run([str(_cc(src, "drv0", shared=False))], check=True, capture_output=True, text=True).stdout
    assert not LINE.search(out) and "PROF aie_host_split_total host_split_n=0" in out


# ---------------------------------------------------------------- 4. wiring of the generated sources
def test_kernel_sources_wired():
    assert ua.HOST_CC.count('extern "C" void aiednn_aie_host_split_site(int site)') == 1
    assert ua.HOST_CC.count('extern "C" int aiednn_aie_host_split_line(int site') == 1
    kacc_host = "".join(aiednn.template(t) for t in ("conv2d/kacc_conv_host_head.cc", "layout/neon_tr4_f32.cc",
                                                     "conv2d/kacc_conv_host.cc"))
    for host, kind in ((kacc_host, "AHS_KIND_CONV"), (aiednn.template("gemm/kacc_linear_host.cc"), "AHS_KIND_LINEAR"),
                       (ua.ATTN_HOST, "AHS_KIND_ATTN")):
        assert f"ahs_begin(&g_ahs, {kind});" in host
        for ph in ("AHS_PACK", "AHS_SCATTER"):
            assert f"ahs_add_span(&g_ahs, {ph}," in host, (kind, ph)
        # every aie_ms window (g_uc_aie_ticks += lt1 - lt0) is split into dma/wait
        assert host.count("g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);") == \
            host.count("ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);") == 1, kind
    # the wait counter comes from the runtime (always on) and both kernel writers export the entry points
    assert "__Runtime_host_split_wait_ticks" in ua.HOST_CC
    rt = (RT / "aie_runtime.c").read_text()
    assert "void __Runtime_host_split_wait_ticks(" in rt and "g_hs_wait_ticks += ahs_now() - __hs_t0;" in rt
    # the kacc kernel writer (aiegraph.kincore) exports both entry points (unet-decl P4d removed the 64x64 writer)
    from aiegraph import cnn, cnn_chunk, kincore
    # the 64x64 host glue's entries are found by cnn._EXPORT_RE, the chunked writer adds them in
    # cnn_chunk._spec_exports, the kacc one lists them in kincore.EXPORTS_TAIL
    assert {"aiednn_aie_host_split_site", "aiednn_aie_host_split_line"} <= set(cnn._EXPORT_RE.findall(ua.HOST_CC))
    spec = cnn_chunk._spec_exports('"out y\\n"', "aiednn")
    assert spec.count('"export    aiednn_aie_host_split_site\\n"') == 1
    assert spec.count('"export    aiednn_aie_host_split_line\\n"') == 1
    assert {"aiednn_aie_host_split_site", "aiednn_aie_host_split_line"} <= set(kincore.EXPORTS_TAIL)


# ---------------------------------------------------------------- 5. M2 N segment: in-launch glue per site
# AHS_GLUE / AHS_FLUSH / AHS_INVAL (PMU cycles, same values as the global g_sp_glue / g_sp_pre / g_sp_post) and the
# per-site launch count, appended to the PROF line (old fields and order unchanged), for conv / linear / fa sites.
GLUE_HDR = HDR + r"""
void glue(int g, int f, int i) { ahs_add(&T, AHS_GLUE, g); ahs_add(&T, AHS_FLUSH, f); ahs_add(&T, AHS_INVAL, i);
                                ahs_launch(&T); }
void cal(unsigned long long c, unsigned long long t) { ahs_cal(&T, c, t); }
"""

FA_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@DLT@
#define UC_NEON 0
#define UC_CTRLPKT 0
typedef unsigned long long XTime;
static XTime g_now = 0;
static void XTime_GetTime(XTime *t) { *t = (g_now += 10); }
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
static ahs_table g_ahs;
static unsigned long long g_wait = 0, g_cyc = 0, g_rt = 0;
static inline unsigned long long uc_hs_wait(void) { return g_wait += 3; }
static inline unsigned long long uc_pmc(void) { return g_cyc += 100; }    /* 100 cycles per read */
static inline unsigned long long uc_rt_cyc(void) { return g_rt += 30; }   /* runtime phases: 30 per read pair */
static unsigned long long g_sp_pre = 0, g_sp_call = 0, g_sp_glue = 0, g_sp_post = 0;
static unsigned long long g_uc_aie_ticks = 0, g_uc_macs = 0;
static int g_aie_launches = 0, g_uc_calls = 0, g_uc_rc = 0;
static void *__Runtime_Alloc(size_t n) { return calloc(1, n); }
static void uc_sync_pre(const void *, size_t, const void *, size_t, const void *, size_t) {}
static void uc_sync_post(const void *, size_t) {}
@FA@
extern "C" int site(int part, int role, void *args) {
    ahs_set_site(&g_ahs, part);
    const int rc = aiednn_aie_attn(7, args, 3, role, 0);
    ahs_set_site(&g_ahs, -1);
    return rc;
}
extern "C" unsigned long long ticks(int s, int ph) { return g_ahs.s[s].t[ph]; }
extern "C" unsigned nlaunch(int s) { return g_ahs.s[s].launches; }
extern "C" unsigned long long sp(int i) { return i == 0 ? g_sp_pre : i == 1 ? g_sp_glue : g_sp_post; }
extern "C" int launches(void) { return g_aie_launches; }
extern "C" int line(int s, char *b, int n) { g_ahs.cyc_hz = 1e6; return ahs_format(&g_ahs, s, 1e6, b, (size_t)n); }
"""


def test_glue_header_fields_kinds_and_length():
    """AHS_GLUE/FLUSH/INVAL + launches appended for conv / linear / fa; cycles -> ms by ahs_cal; line <= 191"""
    L = ctypes.CDLL(str(_cc(GLUE_HDR, "ghdr")))
    for f in ("add", "span", "window", "cal"):
        getattr(L, f).argtypes = None
    L.ticks.restype = ctypes.c_ulonglong
    L.fmt.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_int]
    L.cal(U(0), U(0))                   # 1000 PMU cycles per tick: cyc_hz = 1e9 at tick_hz 1e6
    L.cal(U(10**9), U(10**6))
    for part, kind in ((2, 1), (3, 2), (5, 4)):
        L.set_site(part)
        L.begin(kind)
        L.window(U(4000), U(1000))
        for _ in range(part):           # `part` launches of glue 1.5 ms, flush 0.25 ms, inval 0.125 ms each
            L.glue(1500000, 250000, 125000)
    rows = {}
    for part in (2, 3, 5):
        n, s = _fmt(L, part)
        m = LINE.match(s)
        assert m and n == len(s) <= 191, s
        rows[m.group(2)] = s
        assert s.endswith(" wait_ms=1.000 glue_ms=%.3f flush_ms=%.3f inval_ms=%.3f launches=%d"
                          % (1.5 * part, 0.25 * part, 0.125 * part, part)), s
    assert set(rows) == {"conv", "linear", "fa"}, rows
    t = _fmt(L, -1)[1]
    assert TOTAL.match(t) and t.endswith("glue_ms=15.000 flush_ms=2.500 inval_ms=1.250 launches=10"), t
    # worst case under site_prof.c's char hs_[192]: last site, 5-digit ms everywhere, 6-digit launch count
    L.set_site(1023)
    L.begin(2)
    for ph in range(4):
        L.add(ph, U(99999 * 10**3))
    for ph in (4, 5, 6):                # AHS_GLUE, AHS_FLUSH, AHS_INVAL
        L.add(ph, U(99999 * 10**6))
    for _ in range(3):
        L.glue(0, 0, 0)
    for s in (1023, -1):
        n, txt = _fmt(L, s)
        assert n == len(txt) and n <= 191 and re.search(GLUE + "$", txt), (n, txt)


def test_glue_fa_host_split():
    """the generated FA host (aiegraph.flashattn.fa_host: kind=fa, from zero) books glue/flush/inval/launches on its site and
    the same values into the global g_sp_* (split_detail); the QK (pack) site gets kind=fa with 0 launches"""
    from aiegraph import flashattn as fa_attn
    att = [dict(group=0, role="qk", xshape=[8, 128, 40], wshape=[8, 77, 40], yshape=[8, 128, 77]),
           dict(group=0, role="pv", xshape=[8, 128, 77], wshape=[8, 40, 77], yshape=[8, 128, 40])]
    cls = fa_attn.fa_classes(att)
    host = fa_attn.fa_host(cls)
    for a in ("ahs_add(&g_ahs, AHS_GLUE,", "ahs_add(&g_ahs, AHS_FLUSH,", "ahs_add(&g_ahs, AHS_INVAL,",
              "ahs_launch(&g_ahs);", "ahs_begin(&g_ahs, AHS_KIND_FA);", "g_sp_glue +="):
        assert a in host, a
    i0, i1 = host.index("static int uc_fa_launch("), host.index("/* block layouts")
    host = host[:i0] + "static int uc_fa_launch(int, uint16_t *, uint16_t *, uint16_t *) { g_cyc += 1000; return 0; }\n" \
        + host[i1:]
    src = ua.HOST_CC
    j = src.index("typedef struct {\n    void *data;")
    dlt = src[j:src.index("} uc_dlt;", j) + len("} uc_dlt;")]
    L = ctypes.CDLL(str(_cc(FA_HARNESS.replace("@DLT@", dlt).replace("@FA@", host), "fa", cxx=True)))
    for f in ("ticks", "sp"):
        getattr(L, f).restype = ctypes.c_ulonglong
    c = cls[0]
    H, N, M, D = c["H"], c["N"], c["M"], c["D"]
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((H, N, D)).astype(np.float32)
    K = rng.standard_normal((H, M, D)).astype(np.float32)
    P = np.zeros((H, N, M), np.float32)
    VT = rng.standard_normal((H, D, M)).astype(np.float32)
    O = np.zeros((H, N, D), np.float32)
    keep = []
    a0 = [_dlt(Q, keep), _dlt(K, keep), _dlt(P, keep)]
    a1 = [_dlt(P, keep), _dlt(VT, keep), _dlt(O, keep)]
    args0 = (ctypes.c_void_p * 3)(*[ctypes.addressof(d) for d in a0])
    args1 = (ctypes.c_void_p * 3)(*[ctypes.addressof(d) for d in a1])
    assert L.site(8, 0, args0) == 0 and L.site(9, 1, args1) == 0
    nl = c["launches"]
    assert L.launches() == nl == L.nlaunch(9) and L.nlaunch(8) == 0
    # per launch: flush = sp1-sp0 = 100, inval = sp3-sp2 = 100, glue = (sp2-sp1) - (rt1-rt0) = 1100 - 30
    assert [L.ticks(9, p) for p in (4, 5, 6)] == [1070 * nl, 100 * nl, 100 * nl]
    assert (L.sp(1), L.sp(0), L.sp(2)) == (L.ticks(9, 4), L.ticks(9, 5), L.ticks(9, 6))   # == global g_sp_*
    assert L.ticks(9, 2) + L.ticks(9, 3) == 10 * nl and L.ticks(9, 3) == 3 * nl       # dma + wait = windows
    assert L.ticks(8, 0) > 0 and L.ticks(9, 0) > 0 and L.ticks(9, 1) > 0                # pack (QK, V) / scatter (O)
    b = ctypes.create_string_buffer(256)
    for part, launches in ((8, 0), (9, nl)):
        n = L.line(part, b, 256)
        s = b.value.decode()
        m = LINE.match(s)
        assert m and m.group(2) == "fa" and n <= 191 and s.endswith(" launches=%d" % launches), s
    L.line(9, b, 256)                   # cyc_hz = 1e6: 1070 cycles = 1.070 ms
    assert b.value.decode().endswith("glue_ms=%.3f flush_ms=%.3f inval_ms=%.3f launches=%d"
                                     % (1.070 * nl, 0.1 * nl, 0.1 * nl, nl)), b.value


def test_glue_kacc_sources_wired():
    """conv (kacc_conv_host.cc) and linear (kacc_linear_host.cc) book the same glue/flush/inval as g_sp_* per site"""
    for host in (ua.KACC_HOST, ua.LINEAR_HOST):
        assert host.count("ahs_add(&g_ahs, AHS_GLUE, (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL);") \
            == host.count("g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;") == 1
        for a in ("ahs_add(&g_ahs, AHS_FLUSH, sp1 - sp0);", "ahs_add(&g_ahs, AHS_INVAL, sp3 - sp2);",
                  "ahs_launch(&g_ahs);", "ahs_cal(&g_ahs, sp0, (unsigned long long)lt0);"):
            assert host.count(a) == 1, a


# ---------------------------------------------------------------- 5. M3.1 site-perf counter line
PERF = re.compile(r"^PROF aie_site_perf part=(\d+) kind=(\w+) launches=(\d+) perf_n=(\d+) io_act_cyc=(\d+) "
                  r"lock_stall_cyc=(\d+) stream_stall_cyc=(\d+) pre_io_act_cyc=(\d+) snap_gap_cyc=(\d+)"
                  r"(?= compute_cyc=\d+ ovl_cyc=\d+ dact_cyc=\d+$)", re.M)  # M3.1.2: ovl fields appended
PERF_OVL = re.compile(r"^PROF aie_site_perf part=(\d+) .* snap_gap_cyc=\d+ compute_cyc=(\d+) ovl_cyc=(\d+) "
                      r"dact_cyc=(\d+)$", re.M)
OLD_SPLIT_END = re.compile(r"^PROF aie_host_split part=\d+ .*launches=(\d+)$", re.M)  # n_fit.py:71 / m3_site.py:61
STUB = r"""
#include <stdio.h>
#include <string.h>
#include "aie_host_split.h"
/* stub of the runtime getter: launch j reads (1000*(j+1), 10*(j+1), j+1, pre 7*(j+1), gap 2*(j+1));
 * window 3 has no reading */
static unsigned g_nl;
int __Runtime_core_perf_last_launch_win(unsigned long long *ia, unsigned long long *ls, unsigned long long *ss,
                                        unsigned long long *pre, unsigned long long *gap) {
    const unsigned j = g_nl++;
    if (j == 3) { *ia = *ls = *ss = *pre = *gap = 0; return 0; }
    *ia = 1000ULL * (j + 1); *ls = 10ULL * (j + 1); *ss = j + 1; *pre = 7ULL * (j + 1); *gap = 2ULL * (j + 1);
    return 1;
}
/* the same tile's overlap fields of the launch _win just handed out: (900, 100, 1000) * (j+1) */
int __Runtime_core_perf_last_launch_ovl(unsigned long long *cc, unsigned long long *ov, unsigned long long *da) {
    const unsigned j = g_nl - 1;
    *cc = 900ULL * (j + 1); *ov = 100ULL * (j + 1); *da = 1000ULL * (j + 1);
    return 1;
}
"""


def _perf_drv(site_perf: bool) -> str:
    src = DRV.replace("@PROF@", ua.PROF_C.replace('#include "aie_timer.h"\n', "")).replace("@ATTNSITE@", "")
    # sites 2 (conv): 2 launches, 3 (linear): 0 launches, 4 (linear): 2 launches
    src = src.replace("aie_site(1, 2000, 500, 4000, 1000);",
                      "aie_site(1, 2000, 500, 4000, 1000); ahs_launch(&g_ahs); aie_site(1, 0, 0, 10, 5); "
                      "ahs_launch(&g_ahs);")
    src = src.replace("int main(void) {", "int main(void) {\n    printf(\"DBG %d\\n\", g_runtime_debug_level);")
    src = src.replace('GRAPH_PROF_END(3, "aie aiednn_aie_lin 9"); }',
                      'GRAPH_PROF_END(3, "aie aiednn_aie_lin 9"); }\n'
                      '    { GRAPH_PROF_BEGIN(4); UP_AIE_BEGIN(4); aie_site(2, 1, 1, 5, 1); ahs_launch(&g_ahs);'
                      ' aie_site(2, 1, 1, 5, 1); ahs_launch(&g_ahs); UP_AIE_END(4);\n'
                      '      GRAPH_PROF_END(4, "aie aiednn_aie_lin 7"); }')
    if site_perf:
        src = "#define AIEHLC_SITE_PERF 1\n" + STUB + src
    else:
        src = "int g_runtime_debug_level = 0;\n" + src
    return subprocess.run([str(_cc(src, "drvp" if site_perf else "drvn", shared=False))], check=True,
                          capture_output=True, text=True).stdout


def test_site_perf_line():
    out = _perf_drv(True)
    assert "DBG 256" in out, out  # site_prof.c: strong g_runtime_debug_level = 0x100 under AIEHLC_SITE_PERF
    rows = {int(m.group(1)): m.groups()[1:] for m in PERF.finditer(out)}
    # windows 0,1 -> site 2 (both read); window 2 -> site 3 (launches=0: no line); 3 (no reading), 4 -> site 4
    assert rows == {2: ("conv", "2", "2", "3000", "30", "3", "21", "6"),
                    4: ("linear", "2", "1", "5000", "50", "5", "35", "10")}, out
    # one perf line per launches>0 part, right after that part's aie_host_split line, same part=
    split = {int(m.group(0).split("part=")[1].split()[0]): int(m.group(1)) for m in OLD_SPLIT_END.finditer(out)}
    assert split == {2: 2, 3: 0, 4: 2}, out
    assert {p for p, n in split.items() if n > 0} == set(rows)
    for p in rows:
        a = out.index(f"PROF aie_host_split part={p} ")
        b = out.index(f"PROF aie_site_perf part={p} ")
        assert a < b and "PROF " not in out[a + 1:b].replace(f"PROF aie_host_split part={p} ", "", 1)
    # the old parsers' view is unchanged: the split lines still end in launches=<n> and match LINE
    assert {int(m.group(1)) for m in LINE.finditer(out)} == {2, 3, 4}
    # without AIEHLC_SITE_PERF: no perf lines, debug level untouched
    out0 = _perf_drv(False)
    assert not PERF.search(out0) and "aie_site_perf" not in out0 and "DBG 0" in out0, out0
    assert {int(m.group(1)) for m in LINE.finditer(out0)} == {2, 3, 4}


def test_site_perf_ovl_fields():
    """M3.1.2: the site line carries compute_cyc/ovl_cyc/dact_cyc summed over the perf_n launches, from the tile
    __Runtime_core_perf_last_launch_win chose (same tile/window as io_act_cyc and [PERFIO]); old fields first."""
    hdr = (RT / "aie_host_split.h").read_text()
    assert hdr.count('" compute_cyc=%llu ovl_cyc=%llu dact_cyc=%llu"') == 1
    w = hdr[hdr.index("static inline void ahs_add_window("):]
    w = w[:w.index("\n}\n")]
    i = w.index("__Runtime_core_perf_last_launch_win(&ia, &ls, &ss, &pre, &gap)")
    assert i < w.index("__Runtime_core_perf_last_launch_ovl(&cc, &ov, &da)")
    out = _perf_drv(True)
    ovl = {int(m.group(1)): m.groups()[1:] for m in PERF_OVL.finditer(out)}
    # site 2: launches 0,1 -> (900+1800, 100+200, 1000+2000); site 4: launch 3 no reading, launch 4 only
    assert ovl == {2: ("2700", "300", "3000"), 4: ("4500", "500", "5000")}, out
    assert {int(m.group(1)) for m in PERF.finditer(out)} == set(ovl)
    # runtime: _ovl returns the tile _win handed out (tile 0 here), 0 before / after re-arm
    L = _lw_lib()
    L.setup(1000000, 1)
    L.arm(2)
    o3 = (U * 3)()
    assert L.get_ovl(o3) == 0 and list(o3) == [0, 0, 0]
    L.set_ctr(0, 0, 0, 0)
    L.set_ctr(1, 0, 0, 0)
    L.snap(0)
    t0 = L.now()
    L.advance(50)
    L.wait_done(t0, L.now())
    L.set_ctr(0, 30000, 100, 200)
    L.set_ctr(1, 90000, 0, 0)       # tile 1 has the most io -> chosen
    o = (U * 5)()
    assert L.get(o) == 1
    b = 0 if o[0] == L.io(0) else 1
    assert o[0] == L.io(b) and b == 1, (list(o), L.io(0), L.io(1))
    assert L.get_ovl(o3) == 1
    assert list(o3) == [L.compute(b), L.tovl(b), 90000], list(o3)
    assert L.get_ovl(o3) == 1       # reads no counters, repeatable within the launch
    L.arm(2)
    assert L.get_ovl(o3) == 0 and list(o3) == [0, 0, 0]


def test_site_perf_line_maxlen():
    """the longest possible aie_site_perf line (every counter ULLONG_MAX, launches/perf_n UINT_MAX, longest kind)
    fits site_prof.c's hs_ buffer: snprintf's return value < sizeof hs_"""
    sp = (ROOT / "src/aiegraph/runtime/site_prof.c").read_text()
    blk = sp[sp.index("if (is_aie) {"):]
    m = re.search(r"char hs_\[(\d+)\];", blk)
    assert m and "aiednn_aie_host_split_line((int)i + 0x10000, hs_, (int)sizeof hs_)" in blk[m.end():], blk[:400]
    size = int(m.group(1))
    src = r"""
#define AIEHLC_SITE_PERF 1
#include <limits.h>
#include "aie_host_split.h"
int __Runtime_core_perf_last_launch_win(unsigned long long *a, unsigned long long *b, unsigned long long *c,
                                        unsigned long long *d, unsigned long long *e) { (void)a; (void)b; (void)c;
    (void)d; (void)e; return 0; }
int __Runtime_core_perf_last_launch_ovl(unsigned long long *a, unsigned long long *b, unsigned long long *c) {
    (void)a; (void)b; (void)c; return 0; }
static ahs_table T;
int maxlen(int kind, char *buf, int n) {
    const int site = AHS_MAX_SITES - 1;
    ahs_site *s = &T.s[site];
    s->calls = s->launches = s->perf_n = UINT_MAX;
    s->kind = kind;
    s->io_act = s->lock_stall = s->stream_stall = s->pre_io_act = s->snap_gap = ULLONG_MAX;
    s->compute = s->ovl = s->dact = ULLONG_MAX;
    return ahs_format(&T, AHS_PERF_LINE + site, 1e6, buf, (size_t)n);
}
"""
    L = ctypes.CDLL(str(_cc(src, "maxlen")))
    L.maxlen.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    for kind in range(5):
        b = ctypes.create_string_buffer(size)
        n = L.maxlen(kind, b, size)
        line = b.value.decode()
        assert 0 < n < size, (kind, n, size)
        assert len(line) == n and line.endswith(f"dact_cyc={2**64 - 1}"), line
        assert line.count(str(2**64 - 1)) == 8 and line.count(str(2**32 - 1)) == 2, line


def test_site_perf_runtime_wired():
    rt = (RT / "aie_runtime.c").read_text()
    assert rt.count("int __Runtime_core_perf_last_launch(") == 1
    w = rt[rt.index("void __Runtime_wait_io(XAie_DevInst *dev, struct_ioevent io_event) {"):]
    w = w[:w.index("\n}\n")]
    # the counter read sits after the timed wait window (WINLINT), never inside it
    assert w.index("g_hs_wait_ticks += ahs_now() - __hs_t0;") < w.index("rt_core_perf_site_read();")
    assert w.index("RT_PROF_ADD(g_wait_io_cycles") < w.index("rt_core_perf_site_read();")
    hdr = (RT / "aie_host_split.h").read_text()
    assert "__Runtime_core_perf_last_launch_win(&ia, &ls, &ss, &pre, &gap)" in hdr


def _wait_io_body(rt):
    w = rt[rt.index("void __Runtime_wait_io(XAie_DevInst *dev, struct_ioevent io_event) {"):]
    return w[:w.index("\n}\n")]


def _fn_body(rt, sig):
    f = rt[rt.index(sig):]
    return f[:f.index("\n}\n")]


def test_site_perf_kco_snapshot():
    # M3.1.1 launch window: A once per launch -- off before the first startio (__Runtime_startio), aot-kco before
    # the first wait_io's rt_sb_commit (both calls no-op unless A is pending); no per-wait snapshot by default
    rt = (RT / "aie_runtime.c").read_text()
    assert "rt_core_perf_snap_at_startio" not in rt and "s_core_perf_snap_pending" not in rt
    w = _wait_io_body(rt)
    i = w.index("rt_core_perf_launch_snap(0);")
    assert i < w.index("rt_sb_commit();") < w.index("(void)rt_sb_ack_wait();")
    assert i < w.index("RT_PROF_TIC()") and i < w.index("ahs_now()")
    st = _fn_body(rt, "struct_ioevent __Runtime_startio(XAie_DevInst *dev")
    assert st.count("rt_core_perf_") == 1 and "rt_core_perf_launch_snap(0);" in st
    assert st.index("rt_core_perf_launch_snap(0);") < st.index("RT_PROF_TIC()")
    # launch start (kernel group enabled, counters zeroed) clears the per-launch sums and arms A
    kp = _fn_body(rt, "static void rt_kernel_profile_setup(XAie_DevInst *dev, struct_kernel_group kg) {")
    assert kp.index("rt_core_perf_launch_reset();") < kp.index("__Runtime_core_perf_setup(dev, kg.tiles[i]);")
    rs = _fn_body(rt, "static void rt_core_perf_launch_reset(void) {")
    assert "s_klw_state = 1;" in rs


def _guard_of(rt, needle):
    """innermost #if line guarding the first occurrence of needle"""
    depth = 0
    for ln in reversed(rt[:rt.index(needle)].split("\n")):
        t = ln.strip()
        if t.startswith("#endif"):
            depth += 1
        elif t.startswith("#if"):
            if depth == 0:
                return t
            depth -= 1
    return ""


def test_core_perf_wait_window():
    """M3.1.1: the per-wait snapshot (old wait window, ~27.8 us MMIO per wait, ISS-1120090911-PD) is compiled
    only with AIEHLC_PROFILING && AIEHLC_CORE_PERF_WIN (default 0); still outside __wio_t0..RT_PROF_ADD."""
    rt = (RT / "aie_runtime.c").read_text()
    assert "#ifndef AIEHLC_CORE_PERF_WIN\n#define AIEHLC_CORE_PERF_WIN 0\n#endif" in rt
    assert rt.count("static void rt_core_perf_wait_snap(void)") == 1
    assert rt.count("rt_core_perf_wait_snap();") == 1
    w = _wait_io_body(rt)
    for f in ("rt_core_perf_wait_snap();", "rt_core_perf_site_read();"):
        g = _guard_of(w, f)
        assert "AIEHLC_PROFILING" in g and "AIEHLC_CORE_PERF_WIN" in g, (f, g)
    assert _guard_of(rt, "static void rt_core_perf_wait_snap(void)") == "#if AIEHLC_CORE_PERF_WIN"
    order = ["(void)rt_sb_ack_wait();", "rt_core_perf_wait_snap();", "unsigned long long __wio_t0 = RT_PROF_TIC();",
             "RT_PROF_ADD(g_wait_io_cycles, g_wait_io_calls, __wio_t0);", "rt_core_perf_site_read();"]
    pos = [w.index(x) for x in order]
    assert pos == sorted(pos), list(zip(order, pos))
    # WINLINT: nothing of the counter path inside __wio_t0 .. RT_PROF_ADD
    win = w[pos[2]:pos[3]]
    assert "rt_core_perf_" not in win and "XAie_PerfCounterGet" not in win and "s_kwin_" not in win
    assert "s_klw_" not in win
    # default build: the only per-wait counter-path code is the plain-store wait accounting after RT_PROF_ADD
    assert pos[3] < w.index("rt_core_perf_wait_done(__hs_t0, ahs_now());") < pos[4]
    wd = _fn_body(rt, "static inline void rt_core_perf_wait_done(")
    assert "__Runtime_core_perf_read" not in wd and "XAie_" not in wd and "printf" not in wd
    assert "#define RT_CORE_PERF_AIE_MHZ 1000ULL" in rt
    # the aie_host_split line is untouched (n_fit.py:71 / m3_site.py:61 still end in launches=<n>)
    hdr = (RT / "aie_host_split.h").read_text()
    assert '"stream_stall_cyc=%llu pre_io_act_cyc=%llu snap_gap_cyc=%llu"' in hdr
    assert "pre_io_act_cyc" not in hdr[hdr.index("static inline int ahs_format(const ahs_table *a"):]


LW_STUB = r"""
#include <stdio.h>
#include <stdint.h>
typedef int AieRC;
#define XAIE_OK 0
typedef struct { uint8_t Row, Col; } XAie_LocType;
typedef struct { int x; } XAie_DevInst;
static unsigned long long g_now, g_tic;
static unsigned long long ahs_now(void) { return g_now; }
#define RT_PROF_TIC() (++g_tic)
static unsigned long long g_frq;          /* host ticks/s; stands in for cntfrq_el0 */
static uint32_t g_a[4], g_ss[4], g_ls[4]; /* counters 0/2/3 per tile */
static int g_reads;                       /* counter reads (one per tile per call) */
static unsigned long long g_read_cost;    /* host ticks one tile read takes */
AieRC __Runtime_core_perf_read(XAie_DevInst *dev, XAie_LocType t, uint32_t *a, uint32_t *v, uint32_t *ss,
                               uint32_t *ls) {
    (void)dev; (void)v; g_reads++; g_now += g_read_cost;
    if (a) *a = g_a[t.Col];
    if (ss) *ss = g_ss[t.Col];
    if (ls) *ls = g_ls[t.Col];
    return XAIE_OK;
}
int __Runtime_core_perf_last_launch_win(unsigned long long *, unsigned long long *, unsigned long long *,
                                        unsigned long long *, unsigned long long *);
"""
LW_API = r"""
static XAie_DevInst g_dev;
void arm(int n) {
    s_core_perf_ntiles = n; s_core_perf_probe_dev = &g_dev;
    for (int i = 0; i < n; i++) { s_core_perf_tiles[i].Col = (uint8_t)i; s_core_perf_tiles[i].Row = 2; }
    rt_core_perf_launch_reset();
}
void set_ctr(int t, unsigned a, unsigned ss, unsigned ls) { g_a[t] = a; g_ss[t] = ss; g_ls[t] = ls; }
void setup(unsigned long long frq, unsigned long long cost) { g_frq = frq; g_read_cost = cost; }
void advance(unsigned long long t) { g_now += t; }
void snap(int end) { rt_core_perf_launch_snap(end); }
void wait_done(unsigned long long t0, unsigned long long t1) { rt_core_perf_wait_done(t0, t1); }
unsigned long long now(void) { return g_now; }
int reads(void) { return g_reads; }
int get(unsigned long long *o) { return __Runtime_core_perf_last_launch_win(&o[0], &o[1], &o[2], &o[3], &o[4]); }
int get_ovl(unsigned long long *o) { return __Runtime_core_perf_last_launch_ovl(&o[0], &o[1], &o[2]); }
unsigned long long tovl(int i) { return s_klw_tovl[i]; }
void report(int tag) { __Runtime_core_perf_report(tag); fflush(stdout); }
unsigned long long compute(int i) { return s_klw_compute[i]; }
unsigned long long io(int i) { return s_klw_io[i]; }
unsigned long long ovl(void) { return s_klw_ovl_cyc; }
unsigned long long tic(int e) { return s_klw_tic[e]; }
"""


def _lw_lib(win=False):
    """the runtime's counter-window block + __Runtime_core_perf_report, verbatim, over a counter/clock stub"""
    rt = (RT / "aie_runtime.c").read_text()
    blk = rt[rt.index("static int s_core_perf_probe_valid = 0;"):rt.index("/* Core perf counter 0 is not free")]
    assert "unsigned long long f = 0ULL;" in blk
    blk = blk.replace("unsigned long long f = 0ULL;", "unsigned long long f = g_frq;", 1)
    rep = _fn_body(rt, "void __Runtime_core_perf_report(int tag) {") + "\n}\n"
    src = f"#define AIEHLC_CORE_PERF_WIN {int(win)}\n" + LW_STUB + blk + rep + LW_API
    L = ctypes.CDLL(str(_cc(src, "lw_win" if win else "lw")))
    for f in ("compute", "io", "ovl", "now", "tic", "tovl"):
        getattr(L, f).restype = U
    L.setup.argtypes = [U, U]
    L.wait_done.argtypes = [U, U]
    L.advance.argtypes = [U]
    return L


def _capture(fn):
    import os
    r, wfd = os.pipe()
    saved = os.dup(1)
    os.dup2(wfd, 1)
    try:
        fn()
    finally:
        os.dup2(saved, 1)
        os.close(wfd)
        os.close(saved)
    out = os.read(r, 1 << 16).decode()
    os.close(r)
    return out


def test_core_perf_launch_window():
    """M3.1.1 launch window: counters 0/2/3 read once at A and once at B per launch (never per wait);
    compute = dACTIVE - dLOCK - dSTREAM, per-tile ovl = (this tile's B read - its A read) - sum(wait),
    io_active = max(0, dACTIVE - ovl) (counter 0 inside the waits, lower bound),
    snap_gap = last wait end -> B; [PERFIO] old fields unchanged + compute_cyc/ovl_cyc/dact_cyc appended."""
    rt = (RT / "aie_runtime.c").read_text()
    assert rt.count("static void rt_core_perf_launch_snap(int end)") == 1
    calls = re.findall(r"^\s*rt_core_perf_launch_snap\(([01])\);", rt, re.M)
    assert calls.count("0") == 2 and calls.count("1") >= 1, calls
    sn = _fn_body(rt, "static void rt_core_perf_launch_snap(int end)")
    assert "RT_PROF_TIC()" in sn and "__Runtime_core_perf_read(" in sn
    # B: before [PERFIO] and in the site-perf getter, never in wait_io
    pr = _fn_body(rt, "void __Runtime_core_perf_report(int tag) {")
    assert pr.index("rt_core_perf_launch_snap(1);") < pr.index('"[PERFIO]')
    assert "rt_core_perf_launch_snap(1);" in _fn_body(rt, "int __Runtime_core_perf_last_launch_win(")
    w = _wait_io_body(rt)
    assert "rt_core_perf_launch_snap(1);" not in w
    assert w.index("RT_PROF_ADD(g_wait_io_cycles") < w.index("rt_core_perf_wait_done(")
    assert re.search(r'"\[PERFIO\] tag=%d tile=\(%u,%u\) pre_io_active=%llu pre_io_lock_stall=%llu io_active=%llu "'
                     r'\s*"io_lock_stall=%llu snap_gap_cyc=%llu compute_cyc=%llu ovl_cyc=%llu dact_cyc=%llu\\n"', pr), pr
    hdr = (RT / "aie_host_split.h").read_text()
    assert "__Runtime_core_perf_last_launch_win(&ia, &ls, &ss, &pre, &gap)" in hdr

    # ---- semantics, the runtime source compiled on x86: 1 tick = 1 us, AIE 1000 MHz -> 1000 cyc/tick
    L = _lw_lib()
    L.setup(1000000, 1)
    L.arm(2)
    L.set_ctr(0, 100, 5, 7)
    L.set_ctr(1, 50, 0, 0)
    L.snap(1)                       # B before A: no-op
    assert L.reads() == 0
    L.advance(10)
    L.snap(0)                       # A: 2 tile reads of 1 tick -> A ends at 12
    assert L.reads() == 2 and L.now() == 12 and L.tic(0) > 0
    L.snap(0)                       # A once per launch
    assert L.reads() == 2
    L.advance(3)
    for _ in range(3):              # 3 waits of 20 ticks, 2 ticks of host glue after each
        t0 = L.now()
        L.advance(20)
        L.wait_done(t0, L.now())
        L.advance(2)
    assert L.reads() == 2           # nothing read per wait
    L.advance(4)                    # last wait ended at 79; B starts at 85
    L.set_ctr(0, 100 + 70000, 5 + 2000, 7 + 1000)
    L.set_ctr(1, 50 + 1000, 0, 0)
    o = (U * 5)()
    assert L.get(o) == 1            # the getter takes B
    assert L.reads() == 4 and L.tic(1) > L.tic(0)
    assert L.compute(0) == 67000 and L.compute(1) == 1000          # dA - dLOCK - dSTREAM
    assert L.ovl() == 13000                                         # launch-wide host: (85 - 12) - 60 ticks
    # per tile: tile 0 read at 10 (A) and 85 (B), tile 1 at 11 and 86 -> ovl = 75 - 60 ticks = 15000 each
    # io = dACTIVE 70000 - 15000; lock, stream; pre_io = ACTIVE total - io; snap_gap = 85 - 79 ticks
    assert list(o) == [55000, 1000, 2000, 70100 - 55000, 6000], list(o)
    assert L.io(1) == 0             # max(0, 1000 - 15000)
    assert L.get(o) == 0            # handed out once per launch
    L.snap(1)
    assert L.reads() == 4           # B once per launch
    out = _capture(lambda: L.report(3))
    rows = re.findall(r"^\[PERFIO\] tag=3 tile=\((\d+),2\) pre_io_active=(\d+) pre_io_lock_stall=(\d+) "
                      r"io_active=(\d+) io_lock_stall=(\d+) snap_gap_cyc=(\d+) compute_cyc=(\d+) ovl_cyc=(\d+) dact_cyc=(\d+)$",
                      out, re.M)
    assert rows == [("0", str(70100 - 55000), "7", "55000", "1000", "6000", "67000", "15000", "70000"),
                    ("1", "1050", "0", "0", "0", "6000", "1000", "15000", "1000")], out
    assert L.reads() == 6           # [PERF] totals only; B not re-taken
    # next launch: re-armed, report also takes B; io bounded by the waits (io <= compute - host time)
    L.arm(2)
    L.set_ctr(0, 0, 0, 0)
    L.set_ctr(1, 0, 0, 0)
    L.snap(0)
    L.advance(100)
    t0 = L.now()
    L.advance(50)
    L.wait_done(t0, L.now())
    L.set_ctr(0, 152000, 0, 0)      # busy all 152 ticks of tile 0's read-to-read window; only the 50-tick wait is io
    out = _capture(lambda: L.report(4))
    assert re.search(r"^\[PERFIO\] tag=4 tile=\(0,2\) .* io_active=50000 .* snap_gap_cyc=0 compute_cyc=152000 "
                     r"ovl_cyc=102000 dact_cyc=152000$", out, re.M), out
    # AIEHLC_CORE_PERF_WIN=1 still compiles (diagnostic per-wait path)
    _lw_lib(win=True)


# ---------------------------------------------------------------- 6. cnn_chunk (yolo) board glue
CHUNK_STUBS = {
    "aie_user_kernel.h": "#pragma once\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
                         "#include <stddef.h>\n"
                         "struct aieMesh {};\nstruct aieArray { aieMesh partition(const int (&)[4], int, int) "
                         "{ return aieMesh(); } };\nstatic inline void aieSetDevice(int) {}\n"
                         "void *__Runtime_Alloc(size_t n);\ntypedef struct { void *data; } aiek_tensor;\n"
                         "#define AIEK_ERR_KERNEL (-4)\n"
                         "#define AIEK_ENTRY(name) extern \"C\" int name(const aiek_tensor *args, int num_args)\n",
    "xil_cache.h": "#pragma once\n#include <stdint.h>\ntypedef uintptr_t UINTPTR;\n"
                   "void Xil_DCacheFlushRange(UINTPTR a, unsigned long n);\n"
                   "void Xil_DCacheInvalidateRange(UINTPTR a, unsigned long n);\n",
    "aie_timer.h": "#pragma once\ntypedef unsigned long long XTime;\n#define COUNTS_PER_SECOND 100000000ULL\n"
                   "void XTime_GetTime(XTime *t);\n",
}

CHUNK_DISPATCH = "    (void)mesh; (void)kw; (void)kx; (void)ky;\n    if (N == 64 && K == 64 && M == 64)\n        return 0;"


def test_chunk_board_host_split():
    """aiegraph.cnn_chunk board glue (yolo): <K>_aie_host_split_site/_line are wired to aie_host_split.h (no empty
    bodies), every chunk launch books dma/wait + glue/flush/inval + launches, and the archive text
    (board + kacc_chunk_conv + entry, as cnn_chunk.write_kernel concatenates it) compiles under
    -DAIEHLC_SITE_PERF=1 with x86 g++ (no skip: g++ is required)."""
    import shutil
    sys.path.insert(0, str(ROOT / "src"))
    import aiednn
    board = aiednn.render("conv2d/kacc_chunk_board.cc", K="cnn", WARM=1, CTRLPKT=1, PART_C0=2, MAXW=4096,
                          MAXX=4096, MAXY=4096, DISPATCH=CHUNK_DISPATCH, SELFCHECK=0)
    assert "(void)buf" not in board and "(void)site" not in board, "empty host_split body"
    assert "ahs_format(&g_ahs, site," in board and "ahs_set_site(&g_ahs, site);" in board
    assert board.count('extern "C" void cnn_aie_host_split_site(int site)') == 1
    assert board.count('extern "C" int cnn_aie_host_split_line(int site') == 1
    # the symbols are the ones site_prof.c (boardmain.prof_c, retargeted to the archive prefix) calls
    from aiegraph import boardmain
    sp = boardmain.prof_c(kprefix="cnn")
    assert "cnn_aie_host_split_site(int site)" in sp and "cnn_aie_host_split_line(int site" in sp, sp[:800]
    assert "#ifndef UC_HS_DEFINED" in board and "static ahs_table g_ahs;" in board
    assert board.count("g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);") == \
        board.count("ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);") == 1
    for a in ("ahs_add(&g_ahs, AHS_GLUE, (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL);",
              "ahs_add(&g_ahs, AHS_FLUSH, sp1 - sp0);", "ahs_add(&g_ahs, AHS_INVAL, sp3 - sp2);",
              "ahs_launch(&g_ahs);", "ahs_cal(&g_ahs, sp0, (unsigned long long)lt0);",
              "ahs_begin(&g_ahs, AHS_KIND_CONV);", "ahs_add_span(&g_ahs, AHS_PACK,"):
        assert board.count(a) == 1, a
    conv = aiednn.template("conv2d/kacc_chunk_conv.cc")
    entry = aiednn.render("conv2d/kacc_chunk_entry.cc", K="cnn", E="aiek_cnntest", SPEC="")
    assert conv.count("uc_hs_mark(AHS_SCATTER);") == 1 and entry.count("uc_hs_enter();") == 1
    # sp1..sp2 (the launch call) holds no console I/O
    w = board[board.index("sp1 = uc_pmc();"):board.index("sp2 = uc_pmc()")]
    assert "printf" not in w and "fflush" not in w, w
    gxx = shutil.which("g++")
    assert gxx, "g++ not found: test_chunk_board_host_split needs x86 g++"
    d = Path(tempfile.mkdtemp(prefix="hs_chunk_"))
    for n, s in CHUNK_STUBS.items():
        (d / n).write_text(s)
    f = d / "aiek_cnntest.cc"
    f.write_text(board + conv + entry)
    r = subprocess.run([gxx, "-fsyntax-only", "-std=c++17", "-Wall", "-Werror", "-Wno-unused-function",
                        "-Wno-unused-variable", "-DAIEHLC_SITE_PERF=1", f"-I{d}", f"-I{RT}", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
