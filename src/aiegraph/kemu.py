###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``kemu``: run a prototype-form user kernel -- AIE kernel AND host code -- on x86,
and byte-compare what its entry returns with the truth.

``offload_check`` cannot judge the kernel itself: calibrate and emulate run the
``_ref.c`` reference, so an unfilled backbone is READY there. ``kemu`` compiles
the user's actual ``<entry>.cc`` with g++ under ASan:

  * ``aie::port<window, aie::IFM|WEIGHTS|OFM, aie::shape<...>>`` becomes a view
    of the host buffer the launch passed; ``aie::load_buffer`` GATHERS the
    requested block into a fresh exactly-sized allocation (ASan catches any read
    past it), ``aie::store_buffer`` hands out a poisoned block that
    ``aie::commit`` SCATTERS back; holding a block past the kernel's end, or a
    block outside the tensor, is an error;
  * ``kernel<<<mesh>>>(...)`` runs the kernel once per core of the mesh with
    ``aie::program_id`` / ``num_programs`` set, vector code through
    ``src/mlir/runtime/aie_api_x86emu.h``;
  * the entry gets the layer's tensors in TVM call order: TVM's own tensors from
    the deploy's ``make TRACE=1`` dumps when they exist (golden = TVM's output),
    else seeded synthetic tensors with ``<entry>_ref`` as the golden.

It models the DATA (which block a core reads and writes), not the DMA timing:
``aiehlc-convtile trace`` judges DMA legality, an aiehlc build the AIE compiler.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _env

__all__ = ["KemuResult", "kemu"]

_PRELUDE = r"""
// kemu prelude: the aiehlc kernel model on x86 (aiegraph/kemu.py)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <type_traits>
#include <utility>
#define __global__
#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_flatten_loop
#define chess_unroll_loop(...)
#define chess_storage(...)
namespace kemu {
inline int g_p0, g_p1, g_rows = 1, g_cols = 1, g_err;
#define KEMU_E(...) do { if (kemu::g_err++ < 20) { fprintf(stdout, "KEMU ERROR core(%d,%d): ", kemu::g_p0, kemu::g_p1); \
    fprintf(stdout, __VA_ARGS__); fputc('\n', stdout); } } while (0)
struct Blk { unsigned char *m; int o[4], s[4]; };
struct AnyPtr { void *p; template <class T> operator T *() const { return (T *)p; } };
} // namespace kemu
namespace aie {
enum role_t { IFM, WEIGHTS, OFM };
enum class pad_value { none, zero };
// Block options, as the kernel analysis takes them (kernel_trace.cpp); .tile re-lays out the block.
struct load_opts { int reuse = 0; pad_value pad = pad_value::none; int depth = 0; bool resident = false;
                   int tile[2] = {0, 0}; };
using store_opts = load_opts;
template <int... D> struct shape { static constexpr int rank = sizeof...(D); };
template <class W, role_t R, class S = shape<>> struct port;
template <class W, role_t R, int... D> struct port<W, R, shape<D...>> {
    using win = std::remove_pointer_t<W>;
    static constexpr int rank = sizeof...(D), out = win::out, eb = win::eb;
    static constexpr int dim[4] = {D..., };
    unsigned char *base;
    std::deque<kemu::Blk> live;
    explicit port(unsigned char *b) : base(b) {}
    port(const port &) = delete;
    ~port() {
        if (!live.empty())
            KEMU_E("%zu block(s) still held at kernel end: never %s", live.size(), out ? "committed" : "released");
        for (auto &b : live) free(b.m);
    }
};
} // namespace aie
#define KEMU_WIN(name, bytes, o) struct name { static constexpr int eb = bytes, out = o; };
KEMU_WIN(input_window_int8, 1, 0) KEMU_WIN(input_window_uint8, 1, 0) KEMU_WIN(input_window_int16, 2, 0)
KEMU_WIN(input_window_uint16, 2, 0) KEMU_WIN(input_window_int32, 4, 0) KEMU_WIN(input_window_float, 4, 0)
KEMU_WIN(output_window_int8, 1, 1) KEMU_WIN(output_window_uint8, 1, 1) KEMU_WIN(output_window_int16, 2, 1)
KEMU_WIN(output_window_uint16, 2, 1) KEMU_WIN(output_window_int32, 4, 1) KEMU_WIN(output_window_float, 4, 1)
KEMU_WIN(input_window_bfloat16, 2, 0) KEMU_WIN(output_window_bfloat16, 2, 1)
namespace kemu {
// Walk every element of block (o, s) of a [dim] tensor: f(tensor offset or -1 if outside, block offset).
template <class P, class F> void walk(const int *o, const int *s, F f) {
    long n = 1;
    for (int d = 0; d < P::rank; d++) n *= s[d];
    for (long i = 0; i < n; i++) {
        long r = i, t = 0, stride = 1;
        bool in = true;
        for (int d = P::rank - 1; d >= 0; d--) {
            const int c = o[d] + (int)(r % s[d]);
            r /= s[d];
            in = in && c >= 0 && c < P::dim[d];
            t += (long)c * stride;
            stride *= P::dim[d];
        }
        f(in ? t * P::eb : -1L, i * P::eb);
    }
}
template <class P> AnyPtr block(P &p, std::initializer_list<int> off, std::initializer_list<int> shp, int st,
                                aie::load_opts opt) {
    Blk b{};
    int ro = 0, rs = 0;
    for (int v : off) { if (ro < 4) b.o[ro] = v; ro++; }
    for (int v : shp) { if (rs < 4) b.s[rs] = v; rs++; }
    if (st != P::out) KEMU_E("%s on an %s port", st ? "aie::store_buffer" : "aie::load_buffer", P::out ? "output" : "input");
    if (ro != P::rank || rs != P::rank) { KEMU_E("block has %d/%d dims, the tensor %d", ro, rs, P::rank); abort(); }
    long bytes = P::eb;
    bool outside = false;
    for (int d = 0; d < P::rank; d++) {
        if (b.s[d] <= 0) { KEMU_E("block dim %d has size %d", d, b.s[d]); abort(); }
        outside = outside || b.o[d] < 0 || b.o[d] + b.s[d] > P::dim[d];
        bytes *= b.s[d];
    }
    if (outside && (st || opt.pad != aie::pad_value::zero))
        KEMU_E("block [%d,%d,%d,%d]+[%d,%d,%d,%d] leaves the tensor", b.o[0], b.o[1], b.o[2], b.o[3], b.s[0], b.s[1], b.s[2], b.s[3]);
    b.m = (unsigned char *)malloc(bytes);  // exactly the block: ASan flags any access past it
    if (st)
        memset(b.m, 0xA5, bytes);
    else
        walk<P>(b.o, b.s, [&](long t, long i) { if (t >= 0) memcpy(b.m + i, p.base + t, P::eb); else memset(b.m + i, 0, P::eb); });
    if (!st && opt.tile[1] > 0 && P::rank == 2) { // .tile: panel-major (tc-wide panels of all rows), as the DMA delivers it
        const int R = b.s[0], Cc = b.s[1], tc = opt.tile[1];
        unsigned char *t = (unsigned char *)malloc(bytes);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < Cc; c++)
                memcpy(t + ((long)(c / tc) * R * tc + (long)r * tc + c % tc) * P::eb, b.m + ((long)r * Cc + c) * P::eb, P::eb);
        memcpy(b.m, t, bytes);
        free(t);
    }
    p.live.push_back(b);
    return {b.m};
}
template <class P> void unblock(P &p, int st) {
    if (st != P::out) KEMU_E("%s on an %s port", st ? "aie::commit" : "aie::release", P::out ? "output" : "input");
    if (p.live.empty()) { KEMU_E("%s without a block held", st ? "aie::commit" : "aie::release"); return; }
    Blk b = p.live.front();
    p.live.pop_front();
    if (st)
        walk<P>(b.o, b.s, [&](long t, long i) { if (t >= 0) memcpy(p.base + t, b.m + i, P::eb); });
    free(b.m);
}
} // namespace kemu
namespace aie {
template <class P> kemu::AnyPtr load_buffer(P &p, std::initializer_list<int> o, std::initializer_list<int> s, load_opts opt = {}) {
    return kemu::block(p, o, s, 0, opt);
}
template <class P> kemu::AnyPtr store_buffer(P &p, std::initializer_list<int> o, std::initializer_list<int> s, store_opts = {}) {
    return kemu::block(p, o, s, 1, {});
}
template <class P> void release(P &p) { kemu::unblock(p, 0); }
template <class P> void commit(P &p) { kemu::unblock(p, 1); }
inline int program_id(int a) { return a == 0 ? kemu::g_p0 : kemu::g_p1; }
inline int num_programs(int a) { return a == 0 ? kemu::g_rows : kemu::g_cols; }
} // namespace aie
static inline unsigned get_coreid() { return ((unsigned)kemu::g_p1 << 16) | (unsigned)kemu::g_p0; }
static inline void klog(const char *, int) {}
#define AIEHLC_EMU_REPORT(msg) KEMU_E("vector: %s", msg)
#include "aie_api_x86emu.h"
// host side of the launch
struct aieMesh { int rows, cols; };
struct aieArray { aieMesh partition(std::initializer_list<int>, int r, int c) { return {r, c}; } };
static inline void aieSetDevice(int) {}
void *__Runtime_Alloc(size_t bytes) { void *p = nullptr; return posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63) ? nullptr : p; }
namespace kemu {
template <class T> unsigned char *as_ptr(T *p) { return (unsigned char *)p; }
template <class T> std::enable_if_t<std::is_arithmetic_v<T>, unsigned char *> as_ptr(T) { return nullptr; }
template <class... P, size_t... I> void call(void (*k)(P...), unsigned char **a, std::index_sequence<I...>) {
    k(P(a[I])...);
}
template <class... P, class... A> void launch(const aieMesh &m, void (*k)(P...), A... args) {
    static_assert(sizeof...(A) >= sizeof...(P), "launch passes fewer arguments than the kernel has ports");
    unsigned char *a[] = {as_ptr(args)...};
    for (int i = 0; i < (int)sizeof...(P); i++)
        if (!a[i]) { KEMU_E("launch argument %d is not a buffer", i); return; }
    g_rows = m.rows; g_cols = m.cols;
    for (g_p0 = 0; g_p0 < m.rows; g_p0++)
        for (g_p1 = 0; g_p1 < m.cols; g_p1++)
            call(k, a, std::index_sequence_for<P...>{});
    g_p0 = g_p1 = -1;
}
} // namespace kemu
"""

# `name<<<mesh>>>(` or, for a template kernel, `name<args><<<mesh>>>(` -- the
# instance `name<args>` is then the function passed to kemu::launch.
_LAUNCH = re.compile(r"\b(\w+(?:\s*<[^<>;(){}]*>)?)\s*<<<\s*([^<>]+?)\s*>>>\s*\(")


@dataclass
class KemuResult:
    ok: bool
    golden: str          # "tvm" (make TRACE=1 dumps) | "reference" | "none"
    detail: str          # the verdict line(s)
    log: str = ""        # compiler / run output, trimmed

    def __repr__(self) -> str:
        return f"kemu {'PASS' if self.ok else 'FAIL'} (golden: {self.golden}): {self.detail}"


def _main(p) -> str:
    """The harness main: the entry vs the golden, every output byte."""
    e = p.entry
    return f"""
extern "C" int {e}_ref(const aiek_tensor *args, int num_args);

int main() {{
    static aiek_tensor gold[AIEKT_NARGS];
    for (int i = 0; i < AIEKT_NARGS; i++) {{
        gold[i] = aiekt_args[i];
        aiekt_args[i].data = malloc(aiekt_args[i].nbytes);
        gold[i].data = malloc(gold[i].nbytes);
        if (!aiekt_args[i].data || !gold[i].data) {{ printf("KEMU FAIL alloc\\n"); return 2; }}
    }}
#if AIEKT_HAVE_DATA
    for (int i = 0; i < AIEKT_NARGS; i++) {{
        memcpy(aiekt_args[i].data, aiekt_data[i], aiekt_args[i].nbytes);
        memcpy(gold[i].data, aiekt_data[i], aiekt_args[i].nbytes);
    }}
    const char *golden = "TVM's output (make TRACE=1 dump)";
#else
    aiekt_fill_synthetic();
    for (int i = 0; i < AIEKT_NARGS; i++) memcpy(gold[i].data, aiekt_args[i].data, aiekt_args[i].nbytes);
    const int rrc = {e}_ref(gold, AIEKT_NARGS);
    if (rrc != AIEK_OK) {{
        printf("KEMU FAIL no golden: no make TRACE=1 dump and {e}_ref returned %d (fill the reference)\\n", rrc);
        return 1;
    }}
    const char *golden = "{e}_ref on synthetic tensors";
#endif
    for (int k = 0; k < AIEKT_NOUTS; k++) memset(aiekt_args[aiekt_outs[k]].data, 0xA5, aiekt_args[aiekt_outs[k]].nbytes);
    const int rc = {e}(aiekt_args, AIEKT_NARGS);
    if (rc != AIEK_OK) {{ printf("KEMU FAIL the entry returned %d (AIEK_ERR_*; -4 also = _run still returns -1)\\n", rc); return 1; }}
    int fail = kemu::g_err != 0;
    for (int k = 0; k < AIEKT_NOUTS; k++) {{
        const aiek_tensor *t = &aiekt_args[aiekt_outs[k]];
        const unsigned char *got = (const unsigned char *)t->data, *want = (const unsigned char *)gold[aiekt_outs[k]].data;
        uint64_t bad = 0, first = 0;
        for (uint64_t i = 0; i < t->nbytes; i++)
            if (got[i] != want[i]) {{
                if (bad < 8) {{
                    uint64_t r = i / (t->bits / 8);
                    char idx[96] = "";
                    for (int d = t->ndim - 1, n = 0; d >= 0; d--) {{ n = (int)(r % t->shape[d]); r /= t->shape[d];
                        char s[96]; snprintf(s, sizeof s, "%d%s%s", n, d == t->ndim - 1 ? "" : ",", idx); strcpy(idx, s); }}
                    printf("KEMU   %s byte %llu [%s]: got 0x%02x want 0x%02x\\n", aiekt_desc[aiekt_outs[k]],
                           (unsigned long long)i, idx, got[i], want[i]);
                }}
                if (!bad++) first = i;
            }}
        if (bad) {{
            printf("KEMU FAIL %llu / %llu bytes of %s differ from %s (first at byte %llu)\\n", (unsigned long long)bad,
                   (unsigned long long)t->nbytes, aiekt_desc[aiekt_outs[k]], golden, (unsigned long long)first);
            fail = 1;
        }}
    }}
    if (kemu::g_err) printf("KEMU FAIL %d kernel-model error(s) above\\n", kemu::g_err);
    if (!fail) printf("KEMU PASS every output byte equals %s\\n", golden);
    return fail;
}}
"""


def _tail(text: str, n: int = 40) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def kemu(proto, src_dir, timeout_s: float = 600) -> KemuResult:
    """Emulate ``<src_dir>/<entry>.cc`` (+ ``<entry>_ref.c``) for *proto*'s layer."""
    from .prototype import _test_tables
    src_dir = Path(src_dir).resolve()
    e = proto.entry
    cc, ref = src_dir / f"{e}.cc", src_dir / f"{e}_ref.c"
    if not cc.is_file():
        return KemuResult(False, "none", f"{cc} does not exist")
    gxx, gcc = shutil.which("g++"), shutil.which("gcc")
    if not gxx or not gcc:
        return KemuResult(False, "none", "no host g++/gcc on PATH")
    repo = _env.repo_root()
    dumps, data_h = proto.test_data()
    with tempfile.TemporaryDirectory(prefix="kemu_") as tmp:
        t = Path(tmp)
        text = cc.read_text()
        n_launch = len(_LAUNCH.findall(text))
        if n_launch == 0:
            return KemuResult(False, "none", f"{cc.name} has no kernel launch (name<<<mesh>>>(...))")
        (t / cc.name).write_text(_LAUNCH.sub(r"kemu::launch(\2, \1, ", text))
        (t / "kemu_prelude.h").write_text(_PRELUDE)
        golden = "tvm" if dumps else "reference"
        if dumps:
            (t / f"{e}_test_data.h").write_text(data_h)
        harness = ['#include "kemu_prelude.h"', f'#include "{cc.name}"', "#include <stdint.h>",
                   f"#define AIEKT_HAVE_DATA {1 if dumps else 0}"]
        if dumps:
            harness.append(f'#include "{e}_test_data.h"')
        (t / "harness.cc").write_text("\n".join(harness) + "\n" + _test_tables(proto) + _main(proto))
        inc = [f"-I{t}", f"-I{src_dir}", f"-I{repo / 'include'}", f"-I{repo / 'src/mlir/runtime'}"]
        if dumps:
            inc.append(f"-I{dumps[0].parent}")
        san = ["-fsanitize=address", "-fno-omit-frame-pointer", "-g", "-O1"]
        steps = [[gxx, "-std=c++20", "-w", *san, *inc, "-c", str(t / "harness.cc"), "-o", str(t / "h.o")]]
        objs = [str(t / "h.o")]
        if ref.is_file():
            steps.append([gcc, "-std=gnu11", "-w", *san, *inc, "-c", str(ref), "-o", str(t / "r.o")])
            objs.append(str(t / "r.o"))
        else:
            (t / "r.cc").write_text(f'#include "aie_user_kernel.h"\nAIEK_REFERENCE({e}) {{ (void)args; (void)num_args; '
                                    "return AIEK_ERR_KERNEL; }\n")
            steps.append([gxx, "-w", *inc, "-c", str(t / "r.cc"), "-o", str(t / "r.o")])
            objs.append(str(t / "r.o"))
        steps.append([gxx, *san, *objs, "-o", str(t / "kemu")])
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if r.returncode:
                what = "x86 compile of the kernel + host code" if "harness" in " ".join(cmd) else "x86 build"
                return KemuResult(False, golden, f"{what} failed", _tail(r.stdout + r.stderr))
        try:
            r = subprocess.run([str(t / "kemu")], capture_output=True, text=True, timeout=timeout_s, cwd=t)
        except subprocess.TimeoutExpired:
            return KemuResult(False, golden, f"x86 run exceeded {timeout_s:.0f} s")
        out = r.stdout + r.stderr
        verdict = [l for l in out.splitlines() if l.startswith("KEMU")]
        if r.returncode == 0 and any(l.startswith("KEMU PASS") for l in verdict):
            return KemuResult(True, golden, verdict[-1][5:], _tail(out, 10))
        if not verdict:  # ASan / crash
            return KemuResult(False, golden, "the x86 run crashed (ASan report below)", _tail(out, 40))
        return KemuResult(False, golden, "\n".join(v[5:] for v in verdict[:12]), _tail(out, 40))
