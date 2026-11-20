"""GAP-RT1 x86 tests of the aie_host split (src/mlir/runtime/aie_host_split.h) and its UNet wiring
(example/aiegraph/unet_bk_sdm_tiny/unet_aie.py):
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
sys.path.insert(0, str(UNET))
import unet_aie as ua  # noqa: E402

LINE = re.compile(r"^PROF aie_host_split part=(\d+) kind=(attn|conv|linear) pack_ms=([\d.]+) scatter_ms=([\d.]+) "
                  r"dma_ms=([\d.]+) wait_ms=([\d.]+)$", re.M)
TOTAL = re.compile(r"^PROF aie_host_split_total host_split_n=(\d+) pack_ms=([\d.]+) scatter_ms=([\d.]+) "
                   r"dma_ms=([\d.]+) wait_ms=([\d.]+)$", re.M)


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
                                 "dma_ms=2.500 wait_ms=2.500"), s
    assert _fmt(L, 7)[1] == ("PROF aie_host_split part=7 kind=attn pack_ms=0.000 scatter_ms=0.000 dma_ms=0.000 "
                             "wait_ms=2.000")
    assert LINE.match(_fmt(L, 8)[1]).group(2) == "linear"
    assert _fmt(L, 4) == (0, "")            # never entered
    t = TOTAL.match(_fmt(L, -1)[1])
    assert t and t.groups() == ("3", "2.501", "0.750", "2.500", "4.500"), _fmt(L, -1)
    assert _fmt(L, 3, hz=2e6)[1].endswith("dma_ms=1.250 wait_ms=1.250")


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
    import aiednn   # the kacc conv / linear host glue aiegraph.kincore assembles
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
    src = Path(ua.__file__).read_text()     # the 64x64 writer (unet_aie.write_kernel) + the kacc one (aiegraph.kincore)
    assert src.count('"export    aiednn_aie_host_split_site\\\\n"') == 1
    assert src.count('"export    aiednn_aie_host_split_line\\\\n"') == 1
    from aiegraph import kincore
    assert {"aiednn_aie_host_split_site", "aiednn_aie_host_split_line"} <= set(kincore.EXPORTS_TAIL)
