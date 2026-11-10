/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel block analysis -- see kernel_trace.h.
 ******************************************************************************/
#include "kernel_trace.h"
#include "conv_desugar.h"
#include "kernel_static.h"

#include <algorithm>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>

#ifndef AIEHLC_SOURCE_DIR // the repo root (CMake), for the AIE API x86 emulation header
#define AIEHLC_SOURCE_DIR "."
#endif

namespace convautotile {

namespace {

// ---------------------------------------------------------------------------
// Kernel / port parsing
// ---------------------------------------------------------------------------
std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

bool isIdentChar(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

// Split at top-level commas (outside (), <>, {}, []).
std::vector<std::string> splitTop(const std::string &s) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '(' || c == '<' || c == '{' || c == '[')
            depth++;
        else if (c == ')' || c == '>' || c == '}' || c == ']')
            depth--;
        if (c == ',' && depth == 0) {
            out.push_back(trim(cur));
            cur.clear();
        } else
            cur += c;
    }
    if (!trim(cur).empty())
        out.push_back(trim(cur));
    return out;
}

int elemBytesOf(const std::string &windowType) {
    if (windowType.find("int16") != std::string::npos)
        return 2;
    if (windowType.find("int32") != std::string::npos || windowType.find("float") != std::string::npos ||
        windowType.find("cint16") != std::string::npos)
        return 4;
    return 1;
}

// One `aie::port<WindowT, aie::ROLE[, aie::shape<d...>]> var` parameter.
bool parsePort(const std::string &param, const std::map<std::string, std::string> &macros, TracePort &p,
               std::string &err) {
    size_t a = param.find("aie::port<");
    if (a == std::string::npos)
        return false;
    size_t open = a + 9;
    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < param.size(); i++) {
        if (param[i] == '<')
            depth++;
        else if (param[i] == '>' && --depth == 0) {
            close = i;
            break;
        }
    }
    if (close == std::string::npos)
        return false;
    std::vector<std::string> targs = splitTop(param.substr(open + 1, close - open - 1));
    if (targs.size() < 2)
        return false;
    for (const char *role : {"IFM", "WEIGHTS", "OFM"})
        if (targs[1] == std::string("aie::") + role)
            p.role = role;
    if (p.role.empty())
        return false;
    p.elem_bytes = elemBytesOf(targs[0]);
    p.output = targs[0].find("output") != std::string::npos;
    if (targs.size() >= 3) {
        const std::string &sh = targs[2];
        size_t lt = sh.find('<'), gt = sh.rfind('>');
        if (sh.compare(0, 11, "aie::shape<") != 0 || gt == std::string::npos) {
            err = "port " + param + ": third template argument must be aie::shape<d0, d1, ...>";
            return false;
        }
        for (auto &d : splitTop(sh.substr(lt + 1, gt - lt - 1))) {
            long v;
            if (!sourceEvalInt(d, macros, v) || v <= 0) {
                err = "port " + param + ": aie::shape dimension '" + d + "' is not a positive constant";
                return false;
            }
            p.shape.push_back((int)v);
        }
        if (p.shape.empty() || p.shape.size() > 4) {
            err = "port " + param + ": aie::shape must have 1..4 dimensions";
            return false;
        }
    }
    size_t v = param.find_first_not_of(" \t\r\n*&", close + 1);
    size_t ve = v;
    while (ve != std::string::npos && ve < param.size() && isIdentChar(param[ve]))
        ve++;
    if (v == std::string::npos || ve == v)
        return false;
    p.var = param.substr(v, ve - v);
    return true;
}

// Every __global__ kernel in the source. `wantOpForm`: only __global__(a, b)
// kernels; otherwise only plain `__global__ <ret> name(...)` kernels.
std::vector<TraceKernel> findKernels(const std::string &src, const std::map<std::string, std::string> &macros,
                                     bool wantOpForm, std::string &err) {
    std::vector<TraceKernel> out;
    std::string blank = sourceBlankComments(src);
    size_t p = 0;
    while ((p = sourceFindToken(blank, "__global__", p)) != std::string::npos) {
        TraceKernel k;
        k.kwBegin = p;
        size_t q = blank.find_first_not_of(" \t\r\n", p + 10);
        bool opForm = false;
        k.kwEnd = p + 10;
        if (q != std::string::npos && blank[q] == '(') {
            size_t cp = sourceMatchClose(blank, q, '(', ')');
            if (cp == std::string::npos)
                break;
            opForm = blank.substr(q, cp - q).find(',') != std::string::npos;
            k.kwEnd = cp + 1;
        }
        p = k.kwEnd;
        if (opForm != wantOpForm)
            continue;
        k.paramOpen = blank.find('(', k.kwEnd);
        if (k.paramOpen == std::string::npos)
            break;
        size_t ne = k.paramOpen;
        while (ne > 0 && std::isspace((unsigned char)blank[ne - 1]))
            ne--;
        size_t nb = ne;
        while (nb > 0 && isIdentChar(blank[nb - 1]))
            nb--;
        k.name = blank.substr(nb, ne - nb);
        k.paramClose = sourceMatchClose(blank, k.paramOpen, '(', ')');
        if (k.paramClose == std::string::npos)
            break;
        k.bodyOpen = blank.find('{', k.paramClose);
        if (k.bodyOpen == std::string::npos)
            break;
        k.bodyClose = sourceMatchClose(blank, k.bodyOpen, '{', '}');
        if (k.bodyClose == std::string::npos)
            break;
        k.bodyLine = 1 + (int)std::count(src.begin(), src.begin() + k.bodyOpen, '\n');
        std::string params = blank.substr(k.paramOpen + 1, k.paramClose - k.paramOpen - 1);
        for (auto &prm : splitTop(params)) {
            TracePort port;
            if (parsePort(prm, macros, port, err))
                k.ports.push_back(port);
            else if (!err.empty())
                return {};
        }
        p = k.bodyClose;
        out.push_back(k);
    }
    return out;
}

// The object-like #define lines of the .cc itself (what the kernel TU sees).
std::string sourceDefines(const std::string &src) {
    std::istringstream is(src);
    std::string line, out;
    bool cont = false;
    while (std::getline(is, line)) {
        std::string t = trim(line);
        if (cont || t.compare(0, 7, "#define") == 0) {
            out += line + "\n";
            cont = !line.empty() && line.back() == '\\';
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Harness: the kernel body on x86 with recording stubs
// ---------------------------------------------------------------------------
const char *kHarnessRuntime = R"HARNESS(
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <sanitizer/asan_interface.h>
namespace aiehlc_trace {
struct Port { const char *name; int rank; int dim[4]; int eb; int out; std::deque<void *> live; int n; };
static FILE *g_out;
static int g_p0, g_p1, g_rows, g_cols, g_seed, g_pp;
struct AnyPtr { void *p; template <class T> operator T *() const { return (T *)p; } };
} // namespace aiehlc_trace
namespace aie {
enum class pad_value { none, zero };
struct load_opts { int reuse = 0; pad_value pad = pad_value::none; };
} // namespace aie
namespace aiehlc_trace {
#define AIEHLC_E(port, ...) do { fprintf(g_out, "E %d %d %s %d ", g_p0, g_p1, (port).name, (port).n); \
    fprintf(g_out, __VA_ARGS__); fputc('\n', g_out); } while (0)
static AnyPtr block(Port &p, std::initializer_list<int> off, std::initializer_list<int> shape, int st,
                    aie::load_opts opt = {}) {
    const int pad = opt.pad == aie::pad_value::zero;
    int o[4] = {0, 0, 0, 0}, s[4] = {0, 0, 0, 0}, ro = 0, rs = 0;
    for (int v : off) { if (ro < 4) o[ro] = v; ro++; }
    for (int v : shape) { if (rs < 4) s[rs] = v; rs++; }
    fprintf(g_out, "%c %d %d %s %d %d %d %d %d %d %d %d %d %d %d %d %d\n", st ? 'S' : 'L', g_p0, g_p1, p.name, p.n,
            ro, rs, o[0], o[1], o[2], o[3], s[0], s[1], s[2], s[3], opt.reuse, pad);
    if (st && !p.out) AIEHLC_E(p, "aie::store_buffer on an input port");
    if (!st && p.out) AIEHLC_E(p, "aie::load_buffer on an output port");
    if (ro != p.rank || rs != p.rank)
        AIEHLC_E(p, "offset has %d and shape %d dims, but the tensor has %d", ro, rs, p.rank);
    long bytes = p.eb;
    for (int d = 0; d < p.rank && d < 4; d++) {
        if (s[d] <= 0) AIEHLC_E(p, "shape dim %d is %d (must be > 0)", d, s[d]);
        else if ((o[d] < 0 || o[d] + s[d] > p.dim[d]) && !pad)
            AIEHLC_E(p, "block dim %d covers [%d, %d) outside the tensor extent %d%s", d, o[d], o[d] + s[d], p.dim[d],
                     p.rank == 3 && d < 2 && !p.out && o[d] + s[d] > 0 && o[d] < p.dim[d]
                         ? " (a conv border? load it with {.pad = aie::pad_value::zero})" : "");
        bytes *= s[d] > 0 ? s[d] : 1;
    }
    if (bytes > 49152) AIEHLC_E(p, "block is %ld B, larger than the core data memory (49152 B)", bytes);
    if ((int)p.live.size() >= g_pp)
        AIEHLC_E(p, "takes a block while %d of this port are still held (pp_depth %d): release first", (int)p.live.size(), g_pp);
    // Like a DMA window on the core: 32-byte aligned and NOT 64-byte aligned,
    // with ASan guarding both ends of exactly `bytes`.
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, (bytes > 0 ? bytes : 1) + 32)) abort();
    ASAN_POISON_MEMORY_REGION(raw, 32);
    unsigned char *m = (unsigned char *)raw + 32;
    for (long i = 0; i < bytes; i++) m[i] = (unsigned char)(i * 31 + g_seed * 97 + p.n * 7);
    p.live.push_back(m);
    p.n++;
    return {m};
}
static void unblock(void *m) {
    void *raw = (unsigned char *)m - 32;
    ASAN_UNPOISON_MEMORY_REGION(raw, 32);
    free(raw);
}
static void release(Port &p, int st) {
    if (st && !p.out) AIEHLC_E(p, "aie::commit on an input port (use aie::release)");
    if (!st && p.out) AIEHLC_E(p, "aie::release on an output port (use aie::commit)");
    if (p.live.empty()) { AIEHLC_E(p, "%s without a block held", st ? "aie::commit" : "aie::release"); return; }
    unblock(p.live.front());
    p.live.pop_front();
    fprintf(g_out, "R %d %d %s %d\n", g_p0, g_p1, p.name, p.n);
}
static void finish(Port &p) {
    if (!p.live.empty())
        AIEHLC_E(p, "%d block(s) still held at kernel end: never %s (the next DMA round would deadlock)",
                 (int)p.live.size(), p.out ? "committed" : "released");
    while (!p.live.empty()) { unblock(p.live.front()); p.live.pop_front(); }
}
} // namespace aiehlc_trace
namespace aie {
using aiehlc_trace::Port;
inline aiehlc_trace::AnyPtr load_buffer(Port &p, std::initializer_list<int> o, std::initializer_list<int> s,
                                        load_opts opt = {}) {
    return aiehlc_trace::block(p, o, s, 0, opt);
}
inline aiehlc_trace::AnyPtr store_buffer(Port &p, std::initializer_list<int> o, std::initializer_list<int> s) {
    return aiehlc_trace::block(p, o, s, 1);
}
inline void release(Port &p) { aiehlc_trace::release(p, 0); }
inline void commit(Port &p) { aiehlc_trace::release(p, 1); }
inline int program_id(int a) { return a == 0 ? aiehlc_trace::g_p0 : aiehlc_trace::g_p1; }
inline int num_programs(int a) { return a == 0 ? aiehlc_trace::g_rows : aiehlc_trace::g_cols; }
} // namespace aie
static inline unsigned get_coreid() { return ((unsigned)aiehlc_trace::g_p1 << 16) | (unsigned)aiehlc_trace::g_p0; }
// Vector kernels: the AIE API subset, emulated per element (ASan sees every
// byte a vector load touches); misaligned vector accesses become diagnostics.
#define AIEHLC_EMU_REPORT(msg) fprintf(aiehlc_trace::g_out, "E %d %d vector 0 %s\n", aiehlc_trace::g_p0, aiehlc_trace::g_p1, msg)
#include AIEHLC_EMU_HEADER
static inline void klog(const char *, int) {}
)HARNESS";

std::string buildHarness(const std::string &src, const TraceKernel &k, const std::string &prelude,
                         const std::string &srcName) {
    std::ostringstream os;
    os << kHarnessRuntime << "\n" << prelude << "\n";
    os << "static void aiehlc_kernel(";
    for (size_t i = 0; i < k.ports.size(); i++)
        os << (i ? ", " : "") << "aiehlc_trace::Port &" << k.ports[i].var;
    os << ")\n#line " << k.bodyLine << " \"" << srcName << "\"\n";
    // Keep the body's column layout so ASan line numbers are the source's.
    size_t lineStart = src.rfind('\n', k.bodyOpen);
    lineStart = lineStart == std::string::npos ? 0 : lineStart + 1;
    os << std::string(k.bodyOpen - lineStart, ' ') << src.substr(k.bodyOpen, k.bodyClose - k.bodyOpen + 1) << "\n";
    os << "int main(int argc, char **argv) {\n"
          "    using namespace aiehlc_trace;\n"
          "    g_out = fopen(argv[1], \"w\");\n"
          "    g_rows = atoi(argv[2]); g_cols = atoi(argv[3]); g_seed = atoi(argv[4]); g_pp = atoi(argv[5]);\n";
    for (auto &p : k.ports) {
        os << "    Port P_" << p.var << "{\"" << p.var << "\", " << p.shape.size() << ", {";
        for (int d = 0; d < 4; d++)
            os << (d ? ", " : "") << (d < (int)p.shape.size() ? p.shape[d] : 0);
        os << "}, " << p.elem_bytes << ", " << (p.output ? 1 : 0) << ", {}, 0};\n";
    }
    os << "    for (g_p0 = 0; g_p0 < g_rows; g_p0++)\n"
          "        for (g_p1 = 0; g_p1 < g_cols; g_p1++) {\n"
          "            fprintf(stderr, \"CORE %d %d\\n\", g_p0, g_p1);\n";
    for (auto &p : k.ports)
        os << "            P_" << p.var << ".n = 0;\n";
    os << "            aiehlc_kernel(";
    for (size_t i = 0; i < k.ports.size(); i++)
        os << (i ? ", " : "") << "P_" << k.ports[i].var;
    os << ");\n";
    for (auto &p : k.ports)
        os << "            finish(P_" << p.var << ");\n";
    os << "        }\n    fclose(g_out);\n    return 0;\n}\n";
    return os.str();
}

std::string readAll(const std::string &path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// An AddressSanitizer report, as a kernel-level diagnostic.
std::string asanDiagnostic(const std::string &log, const std::string &srcName) {
    size_t e = log.find("ERROR: AddressSanitizer:");
    if (e == std::string::npos)
        e = log.find("AddressSanitizer:DEADLYSIGNAL");
    if (e == std::string::npos)
        return "";
    std::string kind = log.substr(e, log.find('\n', e) - e);
    std::string where;
    size_t f = log.find(srcName + ":", e);
    if (f != std::string::npos) {
        size_t le = log.find_first_not_of("0123456789", f + srcName.size() + 1);
        where = srcName + ":" + log.substr(f + srcName.size() + 1, le - f - srcName.size() - 1);
    }
    std::string core;
    size_t c = log.rfind("CORE ", e);
    if (c != std::string::npos) {
        int p0 = 0, p1 = 0;
        if (std::sscanf(log.c_str() + c, "CORE %d %d", &p0, &p1) == 2)
            core = " on core (" + std::to_string(p0) + "," + std::to_string(p1) + ")";
    }
    std::string at = where.empty() ? "" : " at " + where;
    if (kind.find("heap-buffer-overflow") != std::string::npos)
        return "the kernel accesses memory outside the block load_buffer/store_buffer returned" + at + core +
               ": data used without being loaded (index past the block shape)";
    if (kind.find("heap-use-after-free") != std::string::npos)
        return "the kernel uses a block after aie::release / aie::commit" + at + core +
               ": data used after its window was handed back to the DMA";
    if (kind.find("SEGV") != std::string::npos || kind.find("DEADLYSIGNAL") != std::string::npos)
        return "invalid memory access" + at + core + ": a pointer that was never set by load_buffer/store_buffer";
    return "AddressSanitizer: " + kind + at + core;
}

struct Event {
    char kind = 0; // L, S, E
    int p0 = 0, p1 = 0, n = 0, ro = 0, rs = 0;
    std::string port, msg;
    int off[4] = {0, 0, 0, 0}, shape[4] = {0, 0, 0, 0};
    int reuse = 0, pad = 0; // load_opts of an L event
};

std::vector<Event> parseTrace(const std::string &text) {
    std::vector<Event> out;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        std::istringstream ls(line);
        Event ev;
        ls >> ev.kind >> ev.p0 >> ev.p1 >> ev.port >> ev.n;
        if (ev.kind == 'E') {
            std::getline(ls, ev.msg);
            ev.msg = trim(ev.msg);
        } else if (ev.kind != 'R') {
            ls >> ev.ro >> ev.rs;
            for (int &v : ev.off)
                ls >> v;
            for (int &v : ev.shape)
                ls >> v;
            ls >> ev.reuse >> ev.pad;
        }
        out.push_back(ev);
    }
    return out;
}

// Compile + run the harness twice; fills events (run 1) and records errors.
bool runHarness(const std::string &harness, const std::string &srcName, int rows, int cols, int pp,
                std::vector<Event> &events, KernelTraceResult &r) {
    const char *tmpEnv = std::getenv("TMPDIR");
    std::string tmplStr = std::string(tmpEnv && *tmpEnv ? tmpEnv : "/tmp") + "/aiehlc_trace_XXXXXX";
    std::vector<char> tmpl(tmplStr.begin(), tmplStr.end());
    tmpl.push_back('\0');
    if (!mkdtemp(tmpl.data())) {
        r.errors.push_back("kernel analysis: cannot create a temp directory");
        return false;
    }
    std::string dir = tmpl.data();
    {
        std::ofstream f(dir + "/harness.cc");
        f << harness;
    }
    const char *cxx = std::getenv("CXX");
    std::string cc = std::string(cxx && *cxx ? cxx : "g++") +
                     " -std=c++20 -O1 -g -w -fsanitize=address -fno-omit-frame-pointer " + "-DAIEHLC_EMU_HEADER='\"" +
                     std::string(AIEHLC_SOURCE_DIR) + "/src/mlir/runtime/aie_api_x86emu.h\"' -o " + dir + "/h " + dir +
                     "/harness.cc > " + dir + "/cc.log 2>&1";
    if (std::system(cc.c_str()) != 0) {
        std::string log = readAll(dir + "/cc.log");
        r.errors.push_back("kernel analysis: the kernel body does not compile for tracing (x86, " + srcName + "):\n" +
                           log.substr(0, 1500));
        return false;
    }
    std::vector<std::vector<Event>> runs;
    for (int seed : {1, 2}) {
        std::string trace = dir + "/trace" + std::to_string(seed) + ".txt";
        std::string run = "ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 " + dir + "/h " + trace + " " +
                          std::to_string(rows) + " " + std::to_string(cols) + " " + std::to_string(seed) + " " +
                          std::to_string(pp) + " > " + dir + "/run.log 2>&1";
        int rc = std::system(run.c_str());
        std::string asan = asanDiagnostic(readAll(dir + "/run.log"), srcName);
        if (!asan.empty() || rc != 0) {
            r.errors.push_back(
                asan.empty() ? "kernel analysis: the traced kernel crashed (exit " + std::to_string(rc) + ")" : asan);
            if (!std::getenv("AIEHLC_TRACE_KEEP"))
                std::system(("rm -rf " + dir).c_str());
            return false;
        }
        runs.push_back(parseTrace(readAll(trace)));
    }
    if (std::getenv("AIEHLC_TRACE_KEEP"))
        r.notes.push_back("kernel analysis harness kept in " + dir);
    else
        std::system(("rm -rf " + dir).c_str());
    events = runs[0];
    // Same schedule for any data (R1).
    size_t n = std::min(runs[0].size(), runs[1].size());
    for (size_t i = 0; i <= n; i++) {
        if (i == n) {
            if (runs[0].size() != runs[1].size())
                r.errors.push_back("the load/store schedule depends on the data: the kernel issues " +
                                   std::to_string(runs[0].size()) + " vs " + std::to_string(runs[1].size()) +
                                   " block operations for two different inputs");
            break;
        }
        const Event &a = runs[0][i], &b = runs[1][i];
        bool same = a.kind == b.kind && a.port == b.port && a.p0 == b.p0 && a.p1 == b.p1 &&
                    std::equal(a.off, a.off + 4, b.off) && std::equal(a.shape, a.shape + 4, b.shape);
        if (!same) {
            r.errors.push_back("the load/store schedule depends on the data: " + a.port + " block " +
                               std::to_string(a.n) + " on core (" + std::to_string(a.p0) + "," + std::to_string(a.p1) +
                               ") differs between two inputs (DMA order is fixed at " + "compile time)");
            break;
        }
    }
    return true;
}

std::string vecStr(const int *v, int n) {
    std::string s = "(";
    for (int i = 0; i < n; i++)
        s += (i ? "," : "") + std::to_string(v[i]);
    return s + ")";
}
std::string vecStr(const std::vector<int> &v) { return vecStr(v.data(), (int)v.size()); }

// Harness-reported problems, one line per (port, message), with where/how often.
void collectHarnessErrors(const std::vector<Event> &events, KernelTraceResult &r) {
    std::map<std::string, std::pair<const Event *, int>> seen;
    std::vector<std::string> order;
    for (auto &e : events) {
        if (e.kind != 'E')
            continue;
        std::string key = e.port + ": " + e.msg;
        if (!seen.count(key)) {
            seen[key] = {&e, 0};
            order.push_back(key);
        }
        seen[key].second++;
    }
    for (auto &key : order) {
        const Event *e = seen[key].first;
        int cnt = seen[key].second;
        r.errors.push_back(
            (e->port == "vector" ? std::string("vector access") : e->port + " block " + std::to_string(e->n)) +
            " on core (" + std::to_string(e->p0) + "," + std::to_string(e->p1) + "): " + e->msg +
            (cnt > 1 ? " [" + std::to_string(cnt) + " occurrences]" : ""));
    }
}

// ---------------------------------------------------------------------------
// Fit + coverage
// ---------------------------------------------------------------------------
bool fitPort(const TracePort &port, const std::vector<Event> &events, int rows, int cols, PortFit &f,
             std::vector<int> &origin, std::string &err) {
    std::map<std::pair<int, int>, std::vector<const Event *>> seq;
    for (auto &e : events)
        if ((e.kind == 'L' || e.kind == 'S') && e.port == port.var)
            seq[{e.p0, e.p1}].push_back(&e);
    if (seq.empty()) {
        err = port.var + " (" + port.role + ") is never " +
              (port.output ? "stored (aie::store_buffer)" : "loaded (aie::load_buffer)");
        return false;
    }
    if ((int)seq.size() != rows * cols) {
        err = port.var + ": only " + std::to_string(seq.size()) + " of " + std::to_string(rows * cols) +
              " cores use this port (every core runs the same DMA program)";
        return false;
    }
    const auto &s0 = seq[{0, 0}];
    const int n = (int)s0.size(), rank = s0[0]->rs;
    f.rank = rank;
    f.shape.assign(s0[0]->shape, s0[0]->shape + rank);
    for (auto &kv : seq) {
        if ((int)kv.second.size() != n) {
            err = port.var + ": core (" + std::to_string(kv.first.first) + "," + std::to_string(kv.first.second) +
                  ") takes " + std::to_string(kv.second.size()) + " blocks, core (0,0) takes " + std::to_string(n) +
                  " (the per-core block count must be the same)";
            return false;
        }
        for (auto *e : kv.second)
            if (!std::equal(e->shape, e->shape + 4, s0[0]->shape)) {
                err = port.var + ": block " + std::to_string(e->n) + " has shape " + vecStr(e->shape, rank) +
                      ", block 0 has " + vecStr(f.shape) + " (a DMA window has one fixed size)";
                return false;
            }
    }
    // Rounds: i outer, j inner; try the most inner-heavy factorization first.
    for (int rj = n; rj >= 1; rj--) {
        if (n % rj)
            continue;
        int ri = n / rj;
        std::vector<int> sj(rank, 0), si(rank, 0);
        for (int d = 0; d < rank; d++) {
            sj[d] = rj > 1 ? s0[1]->off[d] - s0[0]->off[d] : 0;
            si[d] = ri > 1 ? s0[rj]->off[d] - s0[0]->off[d] : 0;
        }
        bool ok = true;
        for (auto &kv : seq)
            for (int r = 0; r < n && ok; r++)
                for (int d = 0; d < rank && ok; d++)
                    ok = kv.second[r]->off[d] == kv.second[0]->off[d] + (r / rj) * si[d] + (r % rj) * sj[d];
        if (!ok)
            continue;
        f.rounds_i = ri;
        f.rounds_j = rj;
        f.step_i = si;
        f.step_j = sj;
        break;
    }
    if (f.step_i.empty()) {
        err = port.var + ": the block offsets of a core are not an affine sequence (base + i*step_i + j*step_j); " +
              "core (0,0) visits " + vecStr(s0[0]->off, rank) + ", " +
              (n > 1 ? vecStr(s0[1]->off, rank) : std::string("")) + " ...";
        return false;
    }
    origin.assign(s0[0]->off, s0[0]->off + rank);
    f.base_p0.assign(rank, 0);
    f.base_p1.assign(rank, 0);
    for (int d = 0; d < rank; d++) {
        if (rows > 1)
            f.base_p0[d] = seq[{1, 0}][0]->off[d] - origin[d];
        if (cols > 1)
            f.base_p1[d] = seq[{0, 1}][0]->off[d] - origin[d];
    }
    for (auto &kv : seq)
        for (int d = 0; d < rank; d++)
            if (kv.second[0]->off[d] != origin[d] + kv.first.first * f.base_p0[d] + kv.first.second * f.base_p1[d]) {
                err = port.var + ": the first block of core (" + std::to_string(kv.first.first) + "," +
                      std::to_string(kv.first.second) + ") is " + vecStr(kv.second[0]->off, rank) +
                      ", not affine in program_id like the other cores";
                return false;
            }
    return true;
}

// Every output element written exactly once over all cores.
void checkCoverage(const TracePort &port, const std::vector<Event> &events, std::vector<std::string> &errors) {
    long total = 1;
    for (int d : port.shape)
        total *= d;
    if (total > (64L << 20))
        return;
    std::vector<unsigned char> cnt(total, 0);
    const int rank = (int)port.shape.size();
    for (auto &e : events) {
        if (e.kind != 'S' || e.port != port.var || e.rs != rank)
            continue;
        int lo[4] = {0, 0, 0, 0}, hi[4] = {1, 1, 1, 1};
        for (int d = 0; d < rank; d++) {
            lo[d] = std::max(0, e.off[d]);
            hi[d] = std::min(port.shape[d], e.off[d] + e.shape[d]);
        }
        int dims[4] = {1, 1, 1, 1};
        for (int d = 0; d < rank; d++)
            dims[d] = port.shape[d];
        for (int a = lo[0]; a < hi[0]; a++)
            for (int b = lo[1]; b < hi[1]; b++)
                for (int c = lo[2]; c < hi[2]; c++)
                    for (int x = lo[3]; x < hi[3]; x++) {
                        long idx = ((long)(a * dims[1] + b) * dims[2] + c) * dims[3] + x;
                        if (cnt[idx] < 255)
                            cnt[idx]++;
                    }
    }
    long gaps = 0, overlaps = 0, firstGap = -1, firstOver = -1;
    for (long i = 0; i < total; i++) {
        if (cnt[i] == 0 && gaps++ == 0)
            firstGap = i;
        if (cnt[i] > 1 && overlaps++ == 0)
            firstOver = i;
    }
    auto coord = [&](long i) {
        std::vector<int> c(rank);
        for (int d = rank - 1; d >= 0; d--) {
            c[d] = (int)(i % port.shape[d]);
            i /= port.shape[d];
        }
        return vecStr(c);
    };
    if (gaps)
        errors.push_back(port.var + ": " + std::to_string(gaps) + " output element(s) are never stored, first " +
                         coord(firstGap));
    if (overlaps)
        errors.push_back(port.var + ": " + std::to_string(overlaps) +
                         " output element(s) are stored by more than one block, first " + coord(firstOver));
}

// ---------------------------------------------------------------------------
// Spatial-halo conv: what the derived DMA delivers, and the comparison
// ---------------------------------------------------------------------------
// The block each port delivers under the derivation (same numbers as
// kernelTUDecls / the ports' GemmSpace).
PortFit expectedFit(const std::string &role, const Conv2dSpec &c, const Derived &d) {
    PortFit f;
    if (role == "WEIGHTS") {
        f.rank = 4;
        f.shape = {d.oc_mesh, c.kh, c.kw, d.c_align};
        f.base_p0 = {0, 0, 0, 0};
        f.base_p1 = {d.oc_mesh, 0, 0, 0};
        f.step_i = f.step_j = {0, 0, 0, 0};
        return f;
    }
    f.rank = 3;
    f.rounds_i = d.h_rounds;
    f.rounds_j = d.w_rounds;
    if (role == "IFM") {
        f.shape = {d.tile_h, d.tile_w, d.c_align};
        f.base_p0 = {d.mesh_step_h, 0, 0};
        f.base_p1 = {0, 0, 0};
        f.step_i = {d.tile_step_h, 0, 0};
        f.step_j = {0, d.tile_step_w, 0};
    } else {
        f.shape = {d.oh_t, d.ow_t, d.oc_mesh};
        f.base_p0 = {d.oh_mesh, 0, 0};
        f.base_p1 = {0, 0, d.oc_mesh};
        f.step_i = {d.oh_t, 0, 0};
        f.step_j = {0, d.ow_t, 0};
    }
    return f;
}

} // namespace

const char *traceDimName(const std::string &role, int d) {
    static const char *ifm[] = {"rows", "cols", "channels", "?"};
    static const char *ofm[] = {"rows", "cols", "channels", "?"};
    static const char *wgt[] = {"output channels", "kernel rows", "kernel cols", "channels"};
    return role == "WEIGHTS" ? wgt[d] : role == "IFM" ? ifm[d] : ofm[d];
}

namespace {

// Kernel blocks vs the derived DMA; one message per mismatching aspect.
void compareFit(const TracePort &port, const PortFit &got, const PortFit &want, const std::string &why,
                std::vector<std::string> &errors) {
    const std::string who = "load_buffer/store_buffer on " + port.var + " (" + port.role + ")";
    if (got.rank != want.rank) {
        errors.push_back(who + ": blocks have " + std::to_string(got.rank) + " dims, expected " +
                         std::to_string(want.rank));
        return;
    }
    for (int d = 0; d < got.rank; d++)
        if (got.shape[d] != want.shape[d])
            errors.push_back(who + ": block " + vecStr(got.shape) + " is too " +
                             (got.shape[d] > want.shape[d] ? "big" : "small") + " in dim " + std::to_string(d) + " (" +
                             traceDimName(port.role, d) + "): it takes " + std::to_string(got.shape[d]) +
                             ", the DMA block is " + std::to_string(want.shape[d]) + why);
    auto cmp = [&](const std::vector<int> &g, const std::vector<int> &w, const char *what) {
        if (g != w)
            errors.push_back(who + ": " + what + " " + vecStr(g) + ", the DMA uses " + vecStr(w));
    };
    cmp(got.base_p0, want.base_p0, "per-mesh-row offset (x program_id(0))");
    cmp(got.base_p1, want.base_p1, "per-mesh-col offset (x program_id(1))");
    if (got.rounds_i * got.rounds_j != want.rounds_i * want.rounds_j)
        errors.push_back(who + ": " + std::to_string(got.rounds_i * got.rounds_j) + " blocks per core, the DMA sends " +
                         std::to_string(want.rounds_i * want.rounds_j));
    else {
        if (got.rounds_j != want.rounds_j || got.rounds_i != want.rounds_i)
            errors.push_back(who + ": rounds " + std::to_string(got.rounds_i) + "x" + std::to_string(got.rounds_j) +
                             " (outer x inner), the DMA sends " + std::to_string(want.rounds_i) + "x" +
                             std::to_string(want.rounds_j) + " in that order");
        else {
            if (want.rounds_i > 1)
                cmp(got.step_i, want.step_i, "outer-loop step");
            if (want.rounds_j > 1)
                cmp(got.step_j, want.step_j, "inner-loop step");
        }
    }
}

// The static bounds analysis (kernel_static.h) says WHY a block access leaves
// its block. A trace that hit "data used without being loaded" at line L gets
// the static finding for L appended, plus the constant that causes it; static
// findings the trace did not hit (it stops at the first) are errors then too.
// Without a trace failure they are notes: a guard the analysis does not model
// may keep the access from running.
void addStaticBounds(const std::string &src, const TraceKernel &k, int rows, int cols, const std::string &prelude,
                     const std::string &srcName, KernelTraceResult &r) {
    StaticBoundsResult sb = staticBlockBounds(src, k, rows, cols, prelude, srcName);
    if (sb.analysed + sb.skipped)
        r.notes.push_back("static bounds analysis: " + std::to_string(sb.analysed) + " block access(es) checked" +
                          (sb.skipped ? ", " + std::to_string(sb.skipped) + " not evaluable (skipped)" : ""));
    if (sb.findings.empty())
        return;
    bool confirmed = false;
    std::vector<bool> used(sb.findings.size(), false);
    for (auto &e : r.errors) {
        if (e.find("data used without being loaded") == std::string::npos)
            continue;
        confirmed = true;
        for (size_t i = 0; i < sb.findings.size(); i++)
            if (!used[i] && e.find(srcName + ":" + std::to_string(sb.findings[i].line) + " ") != std::string::npos) {
                e += "\n    static analysis: " + sb.findings[i].message;
                used[i] = true;
            }
    }
    auto &out = confirmed ? r.errors : r.notes;
    for (size_t i = 0; i < sb.findings.size(); i++)
        if (!used[i])
            out.push_back(std::string(confirmed ? "" : "possible ") + "out-of-block access (static analysis) at " +
                          srcName + ":" + std::to_string(sb.findings[i].line) + ": " + sb.findings[i].message);
    if (!sb.cause.empty())
        out.push_back(sb.cause);
}

std::string fitText(const TracePort &p, const PortFit &f) {
    std::ostringstream os;
    os << "  " << p.var << " (" << p.role << ", tensor " << vecStr(p.shape) << "): block " << vecStr(f.shape)
       << ", core offset " << vecStr(f.base_p0) << "*p0 + " << vecStr(f.base_p1) << "*p1, rounds " << f.rounds_i << "x"
       << f.rounds_j << " steps " << vecStr(f.step_i) << " / " << vecStr(f.step_j);
    return os.str();
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
KernelTraceResult traceKernel(const std::string &src, const TraceKernel &k, int rows, int cols,
                              const std::string &prelude, const std::string &srcName, int pp_depth) {
    KernelTraceResult r;
    std::vector<Event> events;
    const bool ran = runHarness(buildHarness(src, k, prelude, srcName), srcName, rows, cols, pp_depth, events, r);
    addStaticBounds(src, k, rows, cols, prelude, srcName, r);
    if (!ran) {
        r.ok = false;
        return r;
    }
    collectHarnessErrors(events, r);
    for (auto &p : k.ports) {
        PortFit f;
        std::vector<int> origin;
        std::string err;
        if (!fitPort(p, events, rows, cols, f, origin, err)) {
            r.errors.push_back(err);
            continue;
        }
        if (std::any_of(origin.begin(), origin.end(), [](int v) { return v != 0; }))
            r.errors.push_back(p.var + ": core (0,0) starts at " + vecStr(origin) +
                               ", not at the tensor origin (the DMA tiles a tensor from its origin)");
        r.fits[p.var] = f;
        if (p.output)
            checkCoverage(p, events, r.errors);
    }
    r.ok = r.errors.empty();
    return r;
}

KernelTraceResult traceLogical(const std::string &src, const TraceKernel &k, int rows, int cols,
                               const std::string &prelude, const std::string &srcName, int pp_depth,
                               std::map<std::string, PortUse> &use) {
    KernelTraceResult r;
    std::vector<Event> events;
    const bool ran = runHarness(buildHarness(src, k, prelude, srcName), srcName, rows, cols, pp_depth, events, r);
    addStaticBounds(src, k, rows, cols, prelude, srcName, r);
    if (!ran) {
        r.ok = false;
        return r;
    }
    collectHarnessErrors(events, r);
    // Blocks held per (core, port), oldest first, with the store count when taken.
    std::map<std::string, std::deque<std::pair<const Event *, long>>> held;
    std::map<std::pair<int, int>, long> stores;
    std::set<std::string> reuseReported;
    for (auto &e : events) {
        const std::string key = std::to_string(e.p0) + "," + std::to_string(e.p1) + "," + e.port;
        if (e.kind == 'S')
            stores[{e.p0, e.p1}]++;
        if (e.kind == 'L' || e.kind == 'S') {
            held[key].push_back({&e, stores[{e.p0, e.p1}]});
            const TracePort *tp = nullptr;
            for (auto &p : k.ports)
                if (p.var == e.port)
                    tp = &p;
            if (!tp || e.rs != (int)tp->shape.size())
                continue;
            PortUse &u = use[e.port];
            if (u.lo.empty()) {
                u.lo.assign(e.off, e.off + e.rs);
                u.hi.assign(e.rs, 0);
                for (int d = 0; d < e.rs; d++)
                    u.hi[d] = e.off[d] + e.shape[d];
            }
            for (int d = 0; d < e.rs; d++) {
                u.lo[d] = std::min(u.lo[d], e.off[d]);
                u.hi[d] = std::max(u.hi[d], e.off[d] + e.shape[d]);
            }
            u.padded |= e.pad != 0;
            const int last = e.rs - 1;
            u.whole_last &= e.off[last] == 0 && e.shape[last] == tp->shape[last];
        } else if (e.kind == 'R' && !held[key].empty()) {
            auto [l, at] = held[key].front();
            held[key].pop_front();
            const long used = stores[{e.p0, e.p1}] - at;
            if (l->kind == 'L' && l->reuse > 0 && used != l->reuse && reuseReported.insert(e.port).second)
                r.errors.push_back(e.port + " block " + std::to_string(l->n) + " on core (" + std::to_string(e.p0) +
                                   "," + std::to_string(e.p1) +
                                   "): load_buffer says {.reuse = " + std::to_string(l->reuse) + "}, but " +
                                   std::to_string(used) + " output block(s) are stored while it is held");
        }
    }
    r.ok = r.errors.empty();
    return r;
}

std::vector<TraceKernel> findPlainKernels(const std::string &src, const std::string &srcDir, std::string &err) {
    return findKernels(src, sourceMacros(src, srcDir), false, err);
}

std::string traceDefines(const std::string &src) { return sourceDefines(src); }

namespace {

// Recognize the fits as a spatial-halo conv (OH over mesh rows, OC over mesh
// cols, OH x OW time tiles). Fills conv/mapping or explains why not.
bool recognizeConv(const TracePort &ifm, const TracePort &wgt, const TracePort &ofm, const PortFit &fi,
                   const PortFit &fw, const PortFit &fo, int rows, int cols, Conv2dSpec &c, MappingSpec &m,
                   std::string &err) {
    if (ifm.shape.size() != 3 || ofm.shape.size() != 3 || wgt.shape.size() != 4) {
        err = "a conv2d kernel needs IFM [H, W, C], WEIGHTS [K, KH, KW, C] and OFM [OH, OW, K] tensors";
        return false;
    }
    const int OH = ofm.shape[0], OW = ofm.shape[1], K = ofm.shape[2];
    const int oh_t = fo.shape[0], ow_t = fo.shape[1], oc_t = fo.shape[2];
    if (OH % rows || fo.base_p0 != std::vector<int>{OH / rows, 0, 0}) {
        err = ofm.var + ": each mesh row must own a contiguous band of OH/" + std::to_string(rows) + " = " +
              std::to_string(OH / rows) + " output rows (program_id(0) * " + std::to_string(OH / rows) +
              "), the kernel uses " + vecStr(fo.base_p0);
        return false;
    }
    if (oc_t * cols != K || fo.base_p1 != std::vector<int>{0, 0, oc_t}) {
        err = ofm.var + ": each mesh col must own " + std::to_string(K / cols) + " output channels (program_id(1) * " +
              std::to_string(K / cols) + "), the kernel uses blocks of " + std::to_string(oc_t) + " at " +
              vecStr(fo.base_p1);
        return false;
    }
    // The DMA streams a core's output rows outer and columns inner; a kernel
    // that walks columns in the outer loop asks for the blocks in another order.
    if (fo.rounds_i > 1 && fo.rounds_j > 1 && fo.step_i[0] == 0 && fo.step_j[0] != 0) {
        err = ofm.var + ": blocks are visited column-outer (outer step " + vecStr(fo.step_i) + ", inner step " +
              vecStr(fo.step_j) + "); the DMA streams output rows in the outer loop and columns in the inner " +
              "loop -- swap the two loops";
        return false;
    }
    const int kh = wgt.shape[1], kw = wgt.shape[2];
    int S = 0;
    if (fo.rounds_j > 1 && fi.rounds_j == fo.rounds_j && ow_t > 0 && fi.step_j.size() == 3 && fi.step_j[1] % ow_t == 0)
        S = fi.step_j[1] / ow_t;
    else if (fo.rounds_i > 1 && fi.rounds_i == fo.rounds_i && oh_t > 0 && fi.step_i[0] % oh_t == 0)
        S = fi.step_i[0] / oh_t;
    else if (fi.base_p0.size() == 3 && fi.base_p0[0] % (OH / rows) == 0)
        S = fi.base_p0[0] / (OH / rows);
    if (S <= 0) {
        err = ifm.var + ": cannot infer the conv stride from the input block steps " + vecStr(fi.step_i) + " / " +
              vecStr(fi.step_j) + " against the output tile " + std::to_string(oh_t) + "x" + std::to_string(ow_t);
        return false;
    }
    c = Conv2dSpec();
    c.h = ifm.shape[0];
    c.w = ifm.shape[1];
    c.c = ifm.shape[2];
    c.k = K;
    c.kh = kh;
    c.kw = kw;
    c.stride = S;
    c.pad = 0; // the IFM tensor is the DDR buffer as the kernel reads it (already padded)
    c.c_align = ifm.shape[2];
    m = MappingSpec();
    m.space_rows = {Axis::OH, rows};
    m.space_cols = {Axis::OC, cols};
    m.time = {{Axis::OH, oh_t}, {Axis::OW, ow_t}};
    // Non-halo port buffers are sized by hints.l1_budget (else chunked
    // unevenly): give them room for the blocks the kernel itself loads. The
    // core-memory / stack checks of derive() still bound the total; kernels
    // whose blocks fit the 4 KB default keep it (and their exact DMA).
    const long wgtBytes = (long)fw.shape[0] * fw.shape[1] * fw.shape[2] * fw.shape[3] * wgt.elem_bytes;
    const long ofmBytes = (long)oh_t * ow_t * oc_t * ofm.elem_bytes;
    const long need = std::max(wgtBytes, ofmBytes);
    if (need > m.hints.l1_budget)
        m.hints.l1_budget = (int)((need + 255) / 256 * 256);
    // The pipeline sizes all 3 tensors' buffers at l1_budget, pp_depth deep:
    // single-buffer when double buffering cannot fit (DMA and compute then
    // alternate instead of overlapping -- correct, slower).
    if ((long)m.hints.pp_depth * m.hints.l1_budget * 3 > 49152 && (long)m.hints.l1_budget * 3 <= 49152)
        m.hints.pp_depth = 1;
    return true;
}

std::string declsText(const std::string &kernel, const Conv2dSpec &c, const MappingSpec &m) {
    std::ostringstream os;
    os << "// ==== aiehlc kernel analysis: inferred from " << kernel << "'s load_buffer / store_buffer ====\n";
    os << "constexpr aie::Conv2d __aie_conv_" << kernel << " = {.h = " << c.h << ", .w = " << c.w << ", .c = " << c.c
       << ", .k = " << c.k << ", .kh = " << c.kh << ", .kw = " << c.kw << ", .stride = " << c.stride
       << ", .pad = " << c.pad << ", .c_align = " << c.c_align << "};\n";
    os << "constexpr aie::Mapping __aie_map_" << kernel << " = {.space = {aie::split(aie::OH, " << m.space_rows.factor
       << "), aie::split(aie::OC, " << m.space_cols.factor << ")}, .time = {aie::tile(aie::OH, " << m.time[0].factor
       << "), aie::tile(aie::OW, " << m.time[1].factor << ")}, .hints = {.pp_depth = " << m.hints.pp_depth
       << ", .l1_budget = aie::Bytes{" << m.hints.l1_budget << "}}};\n";
    return os.str();
}

// Rewrite one analyzed kernel into the __global__(conv, map) form.
void rewriteKernelOnly(std::string &src, const TraceKernel &k, const Conv2dSpec &c, const MappingSpec &m, int rows,
                       int cols) {
    std::string body = src.substr(k.bodyOpen, k.bodyClose - k.bodyOpen + 1);
    auto repl = [&body](const std::string &from, const std::string &to) {
        for (size_t p = 0; (p = body.find(from, p)) != std::string::npos; p += to.size())
            body.replace(p, from.size(), to);
    };
    repl("aie::num_programs(0)", std::to_string(rows));
    repl("aie::num_programs(1)", std::to_string(cols));
    std::string params = src.substr(k.paramOpen, k.paramClose - k.paramOpen + 1);
    for (size_t p = 0; (p = params.find(", aie::shape<", p)) != std::string::npos;) {
        size_t e = params.find('>', p);
        params.erase(p, e - p + 1);
    }
    std::string head = src.substr(k.kwEnd, k.paramOpen - k.kwEnd);
    std::string kw = "__global__(__aie_conv_" + k.name + ", __aie_map_" + k.name + ")";
    src = src.substr(0, k.kwBegin) + declsText(k.name, c, m) + kw + head + params +
          src.substr(k.paramClose + 1, k.bodyOpen - k.paramClose - 1) + body + src.substr(k.bodyClose + 1);
}

} // namespace

KernelOnlyResult inferKernelOnly(std::string &src, const std::string &srcDir, const std::string &srcName) {
    KernelOnlyResult res;
    auto macros = sourceMacros(src, srcDir);
    std::string err;
    std::vector<TraceKernel> kernels = findKernels(src, macros, false, err);
    if (!err.empty()) {
        res.found = true;
        res.ok = false;
        res.error = err;
        return res;
    }
    // Kernel-only = a plain __global__ whose ports declare aie::shape.
    kernels.erase(std::remove_if(kernels.begin(), kernels.end(),
                                 [](const TraceKernel &k) {
                                     return k.ports.empty() ||
                                            std::any_of(k.ports.begin(), k.ports.end(),
                                                        [](const TracePort &p) { return p.shape.empty(); });
                                 }),
                  kernels.end());
    if (kernels.empty())
        return res;
    res.found = true;
    auto mesh = sourceLaunchMesh(src, srcDir);
    if (mesh.first <= 0 || mesh.second <= 0) {
        res.ok = false;
        res.error = "kernel analysis needs the launch mesh: no device.partition({...}, ROWS, COLS) found";
        return res;
    }
    // Back to front so earlier offsets stay valid.
    std::sort(kernels.begin(), kernels.end(),
              [](const TraceKernel &a, const TraceKernel &b) { return a.kwBegin > b.kwBegin; });
    std::ostringstream rep;
    for (auto &k : kernels) {
        rep << "[KernelAnalysis] " << k.name << ": traced on the " << mesh.first << "x" << mesh.second << " mesh\n";
        KernelTraceResult tr = traceKernel(src, k, mesh.first, mesh.second, sourceDefines(src), srcName);
        for (auto &n : tr.notes)
            res.notes.push_back(n);
        for (auto &p : k.ports)
            if (tr.fits.count(p.var))
                rep << fitText(p, tr.fits[p.var]) << "\n";
        std::vector<std::string> errs = tr.errors;
        const TracePort *ifm = nullptr, *wgt = nullptr, *ofm = nullptr;
        for (auto &p : k.ports)
            (p.role == "IFM" ? ifm : p.role == "WEIGHTS" ? wgt : ofm) = &p;
        Conv2dSpec c;
        MappingSpec m;
        if (errs.empty()) {
            if (!ifm || !wgt || !ofm || k.ports.size() != 3)
                errs.push_back(
                    "kernel-only DMA inference supports one aie::IFM, one aie::WEIGHTS and one aie::OFM port");
            else if (!recognizeConv(*ifm, *wgt, *ofm, tr.fits[ifm->var], tr.fits[wgt->var], tr.fits[ofm->var],
                                    mesh.first, mesh.second, c, m, err))
                errs.push_back(err);
        }
        if (errs.empty()) {
            Derived d = derive(c, m);
            if (!d.ok)
                errs.insert(errs.end(), d.errors.begin(), d.errors.end());
            else {
                std::string why = " (a " + std::to_string(d.oh_t) + "x" + std::to_string(d.ow_t) +
                                  " output tile of a " + std::to_string(c.kh) + "x" + std::to_string(c.kw) +
                                  " stride-" + std::to_string(c.stride) + " conv)";
                for (auto *p : {ifm, wgt, ofm})
                    compareFit(*p, tr.fits[p->var], expectedFit(p->role, c, d), why, errs);
                res.derived[k.name] = d;
                rep << "  => conv " << c.h << "x" << c.w << "x" << c.c << " -> " << d.oh << "x" << d.ow << "x" << c.k
                    << ", kernel " << c.kh << "x" << c.kw << " stride " << c.stride << "; mapping split(OH,"
                    << mesh.first << ") x split(OC," << mesh.second << "), tile(OH," << d.oh_t << ") x tile(OW,"
                    << d.ow_t << "); DMA derived and checked against every load/store\n";
            }
        }
        if (!errs.empty()) {
            res.ok = false;
            for (auto &e : errs)
                res.errors.push_back(k.name + ": " + e);
            if (res.error.empty())
                res.error = res.errors[0];
            continue;
        }
        rewriteKernelOnly(src, k, c, m, mesh.first, mesh.second);
    }
    res.report = rep.str();
    return res;
}

std::vector<std::string> verifyKernelBlocks(const std::string &src, const std::string &srcDir,
                                            const std::string &srcName, const KernelDesugar &kd,
                                            std::vector<std::string> &notes) {
    std::vector<std::string> errs;
    std::string blank = sourceBlankComments(src);
    if (blank.find("aie::load_buffer") == std::string::npos && blank.find("aie::store_buffer") == std::string::npos)
        return errs;
    std::string err;
    std::vector<TraceKernel> kernels = findKernels(src, sourceMacros(src, srcDir), true, err);
    const TraceKernel *k = nullptr;
    for (auto &kk : kernels)
        if (kk.name == kd.kernel)
            k = &kk;
    if (!k)
        return errs;
    const Derived &d = kd.derived;
    const Conv2dSpec &c = kd.conv;
    TraceKernel tk = *k;
    for (auto &p : tk.ports) {
        if (p.role == "IFM")
            p.shape = {d.h_pad, d.w_pad, d.c_align};
        else if (p.role == "WEIGHTS")
            p.shape = {c.k, c.kh, c.kw, d.c_align};
        else
            p.shape = {d.oh, d.ow, c.k};
    }
    std::string prelude = sourceDefines(src) +
                          "struct aiehlc_conv2d_t { int h, w, c, k, kh, kw, stride, pad; };\n"
                          "struct aiehlc_kernel_view_t { int oh, ow, oh_per_core, oc_per_core, oh_t, ow_t, ih_t, iw_t, "
                          "c_align; };\n";
    for (auto &line : kernelTUDecls(kd))
        if (line.find("aiehlc_conv2d_t") != std::string::npos || line.find("aiehlc_kernel_view_t") != std::string::npos)
            prelude += line + "\n";
    KernelTraceResult tr = traceKernel(src, tk, d.mesh_rows, d.mesh_cols, prelude, srcName, d.ifm.pp_depth);
    notes.insert(notes.end(), tr.notes.begin(), tr.notes.end());
    errs = tr.errors;
    std::string why = " (derived from " + kd.convVar + " + " + kd.mapVar + ")";
    for (auto &p : tk.ports)
        if (tr.fits.count(p.var))
            compareFit(p, tr.fits[p.var], expectedFit(p.role, c, d), why, errs);
    return errs;
}

} // namespace convautotile
