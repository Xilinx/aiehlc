/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tensor-form kernels -- see kernel_front.h.
 ******************************************************************************/
#include "kernel_front.h"
#include "conv_desugar.h"
#include "kernel_trace.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <set>
#include <sstream>

namespace convautotile {

namespace {

std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

bool isIdent(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

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

// `repl` padded with the newlines `orig` had, so later source lines keep their numbers.
std::string keepLines(const std::string &orig, std::string repl) {
    long n = (long)std::count(orig.begin(), orig.end(), '\n') - (long)std::count(repl.begin(), repl.end(), '\n');
    if (n > 0)
        repl += std::string(n, '\n');
    return repl;
}

// Whole-token replacement of identifier `from`.
void replaceToken(std::string &s, const std::string &from, const std::string &to) {
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos;) {
        bool l = p == 0 || !isIdent(s[p - 1]), r = p + from.size() >= s.size() || !isIdent(s[p + from.size()]);
        if (l && r && (p < 2 || s.compare(p - 2, 2, "::") != 0)) {
            s.replace(p, from.size(), to);
            p += to.size();
        } else
            p += from.size();
    }
}

size_t skipWs(const std::string &s, size_t p) {
    while (p < s.size() && std::isspace((unsigned char)s[p]))
        p++;
    return p;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
struct Tensor {
    std::string elem; // int8_t, ...
    std::vector<int> dims;
};

struct TPort {
    std::string var, alias;
    bool out = false;
};

struct TKernel {
    std::string name;
    size_t tmplBegin = std::string::npos, tmplEnd = 0; // `template <...>`
    size_t kwBegin = 0, kwEnd = 0;                     // `__global__ [aie::autotune<...>]`
    size_t paramOpen = 0, paramClose = 0, bodyOpen = 0, bodyClose = 0;
    std::vector<std::pair<std::string, int>> tparams; // name, default
    std::map<std::string, std::vector<int>> tune;
    std::vector<TPort> ports;
};

int elemBytes(const std::string &t) {
    if (t.find("16") != std::string::npos)
        return 2;
    if (t.find("32") != std::string::npos || t == "float")
        return 4;
    return 1;
}

// `using Name = aie::tensor<T, d0, d1, ...>;`
std::map<std::string, Tensor> parseAliases(const std::string &blank, const std::map<std::string, std::string> &macros,
                                           std::string &err) {
    std::map<std::string, Tensor> out;
    for (size_t p = 0; (p = sourceFindToken(blank, "using", p)) != std::string::npos; p += 5) {
        size_t eq = blank.find('=', p), semi = blank.find(';', p);
        if (eq == std::string::npos || semi == std::string::npos || eq > semi)
            continue;
        std::string name = trim(blank.substr(p + 5, eq - p - 5)), rhs = trim(blank.substr(eq + 1, semi - eq - 1));
        if (rhs.compare(0, 12, "aie::tensor<") != 0 || rhs.back() != '>')
            continue;
        auto args = splitTop(rhs.substr(12, rhs.size() - 13));
        Tensor t;
        t.elem = args.empty() ? "" : args[0];
        for (size_t i = 1; i < args.size(); i++) {
            long v;
            if (!sourceEvalInt(args[i], macros, v) || v <= 0) {
                err = "aie::tensor " + name + ": dimension '" + args[i] + "' is not a positive constant";
                return {};
            }
            t.dims.push_back((int)v);
        }
        if (t.dims.empty() || t.dims.size() > 4) {
            err = "aie::tensor " + name + ": needs 1..4 dimensions";
            return {};
        }
        out[name] = t;
    }
    return out;
}

// `template <int A = 1, int B = 2>` ending right before `kw`.
bool parseTemplateHead(const std::string &blank, size_t kw, TKernel &k, std::string &err) {
    size_t e = kw;
    while (e > 0 && std::isspace((unsigned char)blank[e - 1]))
        e--;
    if (e == 0 || blank[e - 1] != '>')
        return true;
    size_t t = blank.rfind("template", e);
    if (t == std::string::npos)
        return true;
    size_t lt = blank.find('<', t);
    if (lt == std::string::npos || sourceMatchClose(blank, lt, '<', '>') != e - 1)
        return true;
    k.tmplBegin = t;
    k.tmplEnd = e;
    for (auto &prm : splitTop(blank.substr(lt + 1, e - 1 - lt - 1))) {
        size_t eq = prm.find('=');
        std::string decl = trim(prm.substr(0, eq)), name = decl.substr(decl.find_last_of(" \t") + 1);
        long v = 0;
        if (decl.compare(0, 3, "int") != 0 || eq == std::string::npos ||
            !sourceEvalInt(trim(prm.substr(eq + 1)), {}, v)) {
            err = k.name + ": template parameter '" + prm + "' must be `int NAME = default`";
            return false;
        }
        k.tparams.push_back({name, (int)v});
    }
    return true;
}

// `aie::autotune<A, {v, ...}, B, {...}>` at `p`; returns the position after it.
size_t parseAutotune(const std::string &blank, size_t p, TKernel &k, std::string &err) {
    p = skipWs(blank, p);
    if (blank.compare(p, 14, "aie::autotune<") != 0)
        return p;
    size_t close = sourceMatchClose(blank, p + 13, '<', '>');
    auto args = splitTop(blank.substr(p + 14, close - p - 14));
    for (size_t i = 0; i + 1 < args.size(); i += 2) {
        std::string list = args[i + 1];
        if (list.size() < 2 || list.front() != '{' || list.back() != '}') {
            err = k.name + ": aie::autotune expects NAME, {values...} pairs";
            return std::string::npos;
        }
        for (auto &v : splitTop(list.substr(1, list.size() - 2))) {
            long x;
            if (!sourceEvalInt(v, {}, x)) {
                err = k.name + ": aie::autotune value '" + v + "' is not a constant";
                return std::string::npos;
            }
            k.tune[args[i]].push_back((int)x);
        }
    }
    return close + 1;
}

std::vector<TKernel> findTensorKernels(const std::string &src, const std::map<std::string, Tensor> &aliases,
                                       std::string &err) {
    std::vector<TKernel> out;
    std::string blank = sourceBlankComments(src);
    for (size_t p = 0; (p = sourceFindToken(blank, "__global__", p)) != std::string::npos;) {
        TKernel k;
        k.kwBegin = p;
        k.paramOpen = blank.find('(', p);
        p += 10;
        if (k.paramOpen == std::string::npos)
            break;
        k.paramClose = sourceMatchClose(blank, k.paramOpen, '(', ')');
        std::string params = blank.substr(k.paramOpen + 1, k.paramClose - k.paramOpen - 1);
        if (params.find("aie::in<") == std::string::npos && params.find("aie::out<") == std::string::npos)
            continue;
        size_t ne = k.paramOpen;
        while (ne > 0 && std::isspace((unsigned char)blank[ne - 1]))
            ne--;
        size_t nb = ne;
        while (nb > 0 && isIdent(blank[nb - 1]))
            nb--;
        k.name = blank.substr(nb, ne - nb);
        k.kwEnd = parseAutotune(blank, k.kwBegin + 10, k, err);
        if (k.kwEnd == std::string::npos || !parseTemplateHead(blank, k.kwBegin, k, err))
            return {};
        for (auto &prm : splitTop(params)) {
            TPort tp;
            tp.out = prm.compare(0, 8, "aie::out") == 0;
            size_t lt = prm.find('<'), gt = prm.rfind('>');
            if ((prm.compare(0, 8, "aie::in<") != 0 && !tp.out) || gt == std::string::npos) {
                err = k.name + ": parameter '" + prm + "' must be aie::in<Tensor> or aie::out<Tensor>";
                return {};
            }
            tp.alias = trim(prm.substr(lt + 1, gt - lt - 1));
            tp.var = trim(prm.substr(gt + 1));
            if (!aliases.count(tp.alias)) {
                err = k.name + ": " + tp.var + ": '" + tp.alias + "' is not a `using X = aie::tensor<...>` type";
                return {};
            }
            k.ports.push_back(tp);
        }
        k.bodyOpen = blank.find('{', k.paramClose);
        k.bodyClose = sourceMatchClose(blank, k.bodyOpen, '{', '}');
        if (k.bodyClose == std::string::npos) {
            err = k.name + ": unbalanced braces in the kernel body";
            return {};
        }
        p = k.bodyClose;
        out.push_back(k);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------
// Per port: the staged DDR layout the DMA reads (= the tensor in logical mode).
struct PortLayout {
    std::vector<int> lo, phys; // zero border before each dim, staged extents
};

std::string role(const TPort &p, const Tensor &t) { return p.out ? "OFM" : t.dims.size() == 4 ? "WEIGHTS" : "IFM"; }

std::string windowType(const TPort &p, const Tensor &t) {
    std::string e = t.elem;
    if (e.size() > 2 && e.compare(e.size() - 2, 2, "_t") == 0)
        e = e.substr(0, e.size() - 2);
    return std::string(p.out ? "output_window_" : "input_window_") + e + " *";
}

std::string listText(const std::vector<int> &v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

// Names the conv desugar #defines into the kernel TU (conv_autotile.cpp
// kernel_defines, plus the GEMM M / N / K): kernel locals named like them are
// renamed, or the preprocessor would rewrite their declarations.
const char *kDesugarMacros[] = {
    "SP_KH",         "SP_KW",         "SP_C",          "SP_S",   "SP_K",   "SP_OW",  "PAD_H",         "PAD_W",
    "OH_T",          "OW_T",          "SP_OHR",        "SP_OWR", "TILE_H", "TILE_W", "TILE_STRIDE_H", "TILE_STRIDE_W",
    "OUTPUT_FULL_H", "OUTPUT_FULL_W", "OUTPUT_FULL_C", "M",      "N",      "K"};

struct Lowering {
    const std::map<std::string, Tensor> *aliases;
    const TKernel *k;
    std::map<std::string, int> values; // template parameter values
    std::map<std::string, PortLayout> layout;
    bool logical = true; // keep the kernel's coordinates and the load options
    int rows = 1, cols = 1;
};

// One load_buffer / store_buffer call `aie::fn(port, off, shape[, opts])`.
std::string lowerCall(const Lowering &L, const std::string &fn, const std::vector<std::string> &args,
                      std::string &shapeOut, const TPort *&port, std::string &err) {
    port = nullptr;
    for (auto &p : L.k->ports)
        if (p.var == args[0])
            port = &p;
    if (!port || args.size() < 3 || args.size() > 4) {
        err = "aie::" + fn + "(" + (args.empty() ? "" : args[0]) +
              ", ...): expects a port of the kernel, an offset and a shape";
        return "";
    }
    std::string off = args[1], shape = args[2];
    if (!L.logical) {
        const PortLayout &pl = L.layout.at(port->var);
        const Tensor &t = L.aliases->at(port->alias);
        auto o = splitTop(off.substr(1, off.size() - 2)), s = splitTop(shape.substr(1, shape.size() - 2));
        if (o.size() != t.dims.size() || s.size() != t.dims.size()) {
            err = "aie::" + fn + "(" + port->var + ", ...): offset and shape need " + std::to_string(t.dims.size()) +
                  " dims";
            return "";
        }
        off = "{";
        for (size_t d = 0; d < o.size(); d++)
            off += (d ? ", " : "") + (pl.lo[d] ? "(" + o[d] + ") + " + std::to_string(pl.lo[d]) : o[d]);
        off += "}";
        const size_t last = t.dims.size() - 1;
        if (pl.phys[last] != t.dims[last]) // aligned channels: the block carries them all
            s[last] = std::to_string(pl.phys[last]);
        shape = "{";
        for (size_t d = 0; d < s.size(); d++)
            shape += (d ? ", " : "") + s[d];
        shape += "}";
    }
    shapeOut = shape;
    std::string call = "aie::" + fn + "(" + port->var + ", " + off + ", " + shape;
    if (L.logical && args.size() == 4)
        call += ", aie::load_opts" + args[3];
    return call + ")";
}

// Rewrite the load/store calls; `auto v = aie::load_buffer(...)` becomes a
// block view, released / committed at the end of its scope.
bool lowerBlocks(const Lowering &L, std::string &body, std::set<std::pair<std::string, int>> &views, std::string &err) {
    struct Decl {
        std::string name, port;
        bool out;
    };
    std::vector<Decl> decls;
    for (const char *fn : {"load_buffer", "store_buffer"}) {
        const std::string pat = std::string("aie::") + fn + "(";
        for (size_t q = 0; (q = sourceBlankComments(body).find(pat, q)) != std::string::npos;) {
            std::string blank = sourceBlankComments(body);
            size_t open = q + pat.size() - 1, close = sourceMatchClose(blank, open, '(', ')');
            std::string shape;
            const TPort *port = nullptr;
            std::string call = lowerCall(L, fn, splitTop(blank.substr(open + 1, close - open - 1)), shape, port, err);
            if (call.empty())
                return false;
            // `auto NAME = <call>;`
            size_t e = q;
            while (e > 0 && std::isspace((unsigned char)blank[e - 1]))
                e--;
            size_t semi = skipWs(blank, close + 1), st = e, ne = e;
            std::string name;
            if (e > 0 && blank[e - 1] == '=' && semi < blank.size() && blank[semi] == ';') {
                ne = e - 1;
                while (ne > 0 && std::isspace((unsigned char)blank[ne - 1]))
                    ne--;
                size_t nb = ne;
                while (nb > 0 && isIdent(blank[nb - 1]))
                    nb--;
                size_t ab = nb;
                while (ab > 0 && std::isspace((unsigned char)blank[ab - 1]))
                    ab--;
                if (ab >= 4 && blank.compare(ab - 4, 4, "auto") == 0 && (ab == 4 || !isIdent(blank[ab - 5]))) {
                    name = blank.substr(nb, ne - nb);
                    st = ab - 4;
                }
            }
            if (name.empty()) {
                body.replace(q, close + 1 - q, keepLines(body.substr(q, close + 1 - q), call));
                q += call.size();
                continue;
            }
            const Tensor &t = L.aliases->at(port->alias);
            const int rank = (int)t.dims.size();
            views.insert({t.elem, rank});
            std::string view =
                "aiehlc_view_" + t.elem + "_" + std::to_string(rank) + " " + name + "{" + call + ", " + shape + "};";
            body.replace(st, semi + 1 - st, keepLines(body.substr(st, semi + 1 - st), view));
            decls.push_back({name, port->var, port->out});
            q = st + view.size();
        }
    }
    // RAII: hand each view's block back at the end of its scope, last declared
    // first, unless the kernel releases / commits that port itself.
    std::map<size_t, std::string> at;
    std::string blank = sourceBlankComments(body);
    for (auto &d : decls) {
        std::string op = d.out ? "aie::commit(" : "aie::release(";
        if (blank.find(op + d.port + ")") != std::string::npos)
            continue;
        size_t pos = blank.find(" " + d.name + "{aie::");
        int depth = 0;
        size_t open = pos, close = body.size(); // function scope: the end of the body
        while (open > 0) {
            char c = blank[--open];
            if (c == '}')
                depth++;
            else if (c == '{' && depth-- == 0) {
                close = sourceMatchClose(blank, open, '{', '}');
                break;
            }
        }
        at[close] = op + d.port + "); " + at[close];
    }
    for (auto it = at.rbegin(); it != at.rend(); ++it)
        body.insert(it->first, it->second);
    return true;
}

std::string viewStruct(const std::string &elem, int rank) {
    static const char *idx[] = {"a", "b", "c", "d"};
    std::string args, expr = "(long)a";
    for (int d = 0; d < rank; d++) {
        args += std::string(d ? ", " : "") + "int " + idx[d];
        if (d)
            expr = "(" + expr + " * s[" + std::to_string(d) + "] + " + idx[d] + ")";
    }
    return "struct aiehlc_view_" + elem + "_" + std::to_string(rank) + " { " + elem + " *p; int s[" +
           std::to_string(rank) + "]; " + elem + " &operator()(" + args + ") const { return p[" + expr + "]; } }; ";
}

std::string saturateLambda(const std::string &elem) {
    static const std::map<std::string, std::pair<long long, long long>> lim = {
        {"int8_t", {-128, 127}},
        {"uint8_t", {0, 255}},
        {"int16_t", {-32768, 32767}},
        {"uint16_t", {0, 65535}},
        {"int32_t", {-2147483648LL, 2147483647LL}}};
    auto it = lim.find(elem);
    if (it == lim.end())
        return "";
    std::string lo = std::to_string(it->second.first), hi = std::to_string(it->second.second);
    return "auto aiehlc_saturate_" + elem + " = [](long long v) -> " + elem + " { return (" + elem + ")(v < " + lo +
           "LL ? " + lo + "LL : v > " + hi + "LL ? " + hi + "LL : v); }; ";
}

// The kernel (template head .. body) in the kernel-only form.
bool lowerKernel(const Lowering &L, const std::string &src, std::string &out, std::string &err) {
    const TKernel &k = *L.k;
    std::string body = src.substr(k.bodyOpen + 1, k.bodyClose - k.bodyOpen - 1);
    std::set<std::string> clash(std::begin(kDesugarMacros), std::end(kDesugarMacros));
    for (auto &c : clash)
        replaceToken(body, c, c + "_");
    for (auto &a : *L.aliases)
        for (size_t d = 0; d < a.second.dims.size(); d++)
            for (const std::string sp : {"", " "})
                for (size_t p = 0;
                     (p = body.find(a.first + "::dim(" + sp + std::to_string(d) + sp + ")", p)) != std::string::npos;)
                    body.replace(p, a.first.size() + 8 + 2 * sp.size(), std::to_string(a.second.dims[d]));
    for (int a : {0, 1})
        for (size_t p = 0; (p = body.find("aie::num_programs(" + std::to_string(a) + ")", p)) != std::string::npos;)
            body.replace(p, 20, std::to_string(a ? L.cols : L.rows));
    std::set<std::string> sats;
    for (size_t p = 0; (p = body.find("aie::saturate<", p)) != std::string::npos;) {
        size_t gt = body.find('>', p);
        std::string t = trim(body.substr(p + 14, gt - p - 14));
        if (saturateLambda(t).empty()) {
            err = k.name + ": aie::saturate<" + t + "> is not supported (int8/16/32, uint8/16)";
            return false;
        }
        sats.insert(t);
        body.replace(p, gt + 1 - p, "aiehlc_saturate_" + t);
    }
    std::set<std::pair<std::string, int>> views;
    if (!lowerBlocks(L, body, views, err)) {
        err = k.name + ": " + err;
        return false;
    }
    std::string pre;
    if (!k.tparams.empty()) {
        pre = "constexpr int ";
        for (size_t i = 0; i < k.tparams.size(); i++) {
            std::string n = k.tparams[i].first;
            pre += (i ? ", " : "") + (clash.count(n) ? n + "_" : n) + " = " + std::to_string(L.values.at(n));
        }
        pre += "; ";
    }
    for (auto &v : views)
        pre += viewStruct(v.first, v.second);
    for (auto &s : sats)
        pre += saturateLambda(s);
    std::string params;
    for (auto &p : k.ports) {
        const Tensor &t = L.aliases->at(p.alias);
        const PortLayout &pl = L.layout.at(p.var);
        params += (params.empty() ? "" : ", ") + std::string("aie::port<") + windowType(p, t) + ", aie::" + role(p, t) +
                  ", aie::shape<" + listText(pl.phys) + ">> " + p.var;
    }
    std::string head = src.substr(k.kwEnd, k.paramOpen - k.kwEnd); // ` void conv2d`
    size_t from = k.tmplBegin != std::string::npos ? k.tmplBegin : k.kwBegin;
    // The template head and signature collapse onto one line; the newlines they
    // had follow it, so the body keeps every line number.
    out = src.substr(0, from) +
          keepLines(src.substr(from, k.paramClose + 1 - from), "__global__" + head + "(" + params + ")") +
          src.substr(k.paramClose + 1, k.bodyOpen - k.paramClose - 1) + "{ " + pre + body + "}" +
          src.substr(k.bodyClose + 1);
    return true;
}

// The staged DDR layout from the logical trace: a zero border as far as the
// .pad loads reach outside the tensor (symmetric when the far side needs no
// more), and the channel dim aligned to 4 bytes.
bool layoutFrom(const TKernel &k, const std::map<std::string, Tensor> &aliases,
                const std::map<std::string, PortUse> &use, std::map<std::string, PortLayout> &lay,
                std::vector<std::string> &errs) {
    for (auto &p : k.ports) {
        const Tensor &t = aliases.at(p.alias);
        const int rank = (int)t.dims.size(), last = rank - 1;
        PortLayout pl;
        pl.lo.assign(rank, 0);
        pl.phys = t.dims;
        auto u = use.find(p.var);
        if (u != use.end() && (int)u->second.lo.size() == rank)
            for (int d = 0; d < rank; d++) {
                int lo = std::max(0, -u->second.lo[d]), hi = std::max(0, u->second.hi[d] - t.dims[d]);
                if (!lo && !hi)
                    continue;
                if (p.out || d == last) {
                    errs.push_back(k.name + ": " + p.var + ": blocks reach outside the tensor in dim " +
                                   std::to_string(d) + (p.out ? " of an output" : " (the channel dim)") +
                                   "; only an input's spatial dims can be zero-padded");
                    return false;
                }
                pl.lo[d] = lo;
                pl.phys[d] = t.dims[d] + lo + std::max(lo, hi);
            }
        const int align = std::max(1, 4 / elemBytes(t.elem));
        if (t.dims[last] % align) {
            if (p.out) {
                errs.push_back(k.name + ": " + p.var + ": an output's last dim (" + std::to_string(t.dims[last]) +
                               ") must be a multiple of " + std::to_string(align));
                return false;
            }
            if (u != use.end() && !u->second.whole_last) {
                errs.push_back(k.name + ": " + p.var + ": blocks take part of the channel dim (C = " +
                               std::to_string(t.dims[last]) + "); the staged layout aligns channels to " +
                               std::to_string(align) + ", so a block must take all of them");
                return false;
            }
            pl.phys[last] = (t.dims[last] + align - 1) / align * align;
        }
        lay[p.var] = pl;
    }
    return true;
}

// Host side: aie::tensor / aie::alloc and the staging copy, on ONE line so
// the source keeps its line numbers.
const char *kHostPrelude =
    "namespace aie { enum class pad_value { none, zero }; template <class T, int... D> struct tensor { using elem = T; "
    "static constexpr int rank = sizeof...(D); static constexpr int dim(int i) { constexpr int d[] = {D...}; return "
    "d[i]; } static constexpr long size() { return (1L * ... * D); } T *data = nullptr; template <class... I> T "
    "&operator()(I... i) const { long k = 0; int n = 0; ((k = k * dim(n++) + i), ...); return data[k]; } }; template "
    "<class Tn> Tn alloc() { Tn t; t.data = (typename Tn::elem *)__Runtime_Alloc(Tn::size() * sizeof(typename "
    "Tn::elem)); __builtin_memset(t.data, 0, Tn::size() * sizeof(typename Tn::elem)); return t; } } template <class "
    "Tn> static typename Tn::elem *aiehlc_stage(const Tn &t, int p0, int p1, int p2, int p3, int l0, int l1, int l2, "
    "int l3) { const int pd[4] = {p0, p1, p2, p3}, lo[4] = {l0, l1, l2, l3}; long n = 1; for (int d = 0; d < "
    "Tn::rank; d++) n *= pd[d]; typename Tn::elem *p = (typename Tn::elem *)__Runtime_Alloc(n * sizeof(typename "
    "Tn::elem)); __builtin_memset(p, 0, n * sizeof(typename Tn::elem)); for (long i = 0; i < Tn::size(); i++) { long "
    "r = i, q = 0, m = 1; for (int d = Tn::rank - 1; d >= 0; d--) { q += (r % Tn::dim(d) + lo[d]) * m; r /= "
    "Tn::dim(d); m *= pd[d]; } p[q] = t.data[i]; } return p; } ";

struct Launch {
    size_t begin = 0, end = 0; // `name[<...>]<<<mesh>>>(args)`
    std::string mesh;
    std::vector<std::string> args;
    std::vector<int> targs; // explicit template arguments
};

std::vector<Launch> findLaunches(const std::string &src, const TKernel &k, std::string &err) {
    std::vector<Launch> out;
    std::string blank = sourceBlankComments(src);
    for (size_t p = 0; (p = sourceFindToken(blank, k.name, p)) != std::string::npos; p += k.name.size()) {
        Launch l;
        l.begin = p;
        size_t q = skipWs(blank, p + k.name.size());
        if (blank.compare(q, 3, "<<<") != 0 && blank[q] == '<') {
            size_t gt = sourceMatchClose(blank, q, '<', '>');
            for (auto &a : splitTop(blank.substr(q + 1, gt - q - 1))) {
                long v;
                if (!sourceEvalInt(a, {}, v)) {
                    err = "launch of " + k.name + ": template argument '" + a + "' is not a constant";
                    return {};
                }
                l.targs.push_back((int)v);
            }
            q = skipWs(blank, gt + 1);
        }
        if (blank.compare(q, 3, "<<<") != 0)
            continue;
        size_t me = blank.find(">>>", q);
        l.mesh = trim(blank.substr(q + 3, me - q - 3));
        size_t open = blank.find('(', me), close = sourceMatchClose(blank, open, '(', ')');
        l.args = splitTop(blank.substr(open + 1, close - open - 1));
        l.end = close + 1;
        out.push_back(l);
    }
    return out;
}

// `auto a = aie::alloc<X>(), b = aie::alloc<Y>();` deduces two types in one
// declaration, which C++ rejects: split it into one declaration per tensor.
void splitAllocDecls(std::string &src, std::vector<std::string> &notes) {
    for (size_t p = 0; (p = src.find("auto ", p)) != std::string::npos; p += 5) {
        std::string blank = sourceBlankComments(src);
        size_t semi = blank.find(';', p);
        std::string decl = blank.substr(p + 5, semi - p - 5);
        auto parts = splitTop(decl);
        if (parts.size() < 2 || (p > 0 && isIdent(blank[p - 1])) ||
            !std::all_of(parts.begin(), parts.end(),
                         [](const std::string &d) { return d.find("aie::alloc<") != std::string::npos; }))
            continue;
        std::string out;
        for (auto &d : parts)
            out += (out.empty() ? "auto " : " auto ") + d + ";";
        notes.push_back("`auto " + trim(decl) + ";` declares tensors of different types; lowered as `" + out + "`");
        src.replace(p, semi + 1 - p, keepLines(src.substr(p, semi + 1 - p), out));
    }
}

// Each launch stages its inputs into the DMA layout and passes raw buffers.
bool lowerHost(std::string &src, const TKernel &k, const std::map<std::string, Tensor> &aliases,
               const std::map<std::string, PortLayout> &lay, std::vector<std::string> &notes, std::string &err) {
    std::vector<Launch> launches = findLaunches(src, k, err);
    if (!err.empty())
        return false;
    for (size_t li = launches.size(); li-- > 0;) {
        const Launch &l = launches[li];
        if (l.args.size() != k.ports.size()) {
            err = k.name + "<<<" + l.mesh + ">>>: " + std::to_string(l.args.size()) + " arguments for " +
                  std::to_string(k.ports.size()) + " ports";
            return false;
        }
        std::string decl, args;
        const Tensor *ofm = nullptr, *wgt = nullptr;
        const PortLayout *wl = nullptr;
        for (size_t i = 0; i < k.ports.size(); i++) {
            const TPort &p = k.ports[i];
            const Tensor &t = aliases.at(p.alias);
            const PortLayout &pl = lay.at(p.var);
            std::string v = "aiehlc_" + std::to_string(li) + "_" + p.var;
            if (pl.phys == t.dims)
                decl += t.elem + " *" + v + " = (" + l.args[i] + ").data; ";
            else {
                std::vector<int> pd = pl.phys, lo = pl.lo;
                pd.resize(4, 1);
                lo.resize(4, 0);
                decl += t.elem + " *" + v + " = aiehlc_stage(" + l.args[i] + ", " + listText(pd) + ", " + listText(lo) +
                        "); ";
            }
            args += (i ? ", " : "") + v;
            std::string r = role(p, t);
            if (r == "OFM")
                ofm = &t;
            else if (r == "WEIGHTS") {
                wgt = &t;
                wl = &pl;
            }
        }
        if (ofm && wgt && ofm->dims.size() == 3) // the conv's GEMM view: M, N, K
            args += ", " + std::to_string(ofm->dims[0] * ofm->dims[1]) + ", " + std::to_string(ofm->dims[2]) + ", " +
                    std::to_string(wl->phys[1] * wl->phys[2] * wl->phys[3]);
        std::string call = k.name + "<<<" + l.mesh + ">>>(" + args + ")";
        src.replace(l.begin, l.end - l.begin, keepLines(src.substr(l.begin, l.end - l.begin), decl + call));
    }
    splitAllocDecls(src, notes);
    std::string blank = sourceBlankComments(src);
    size_t t = blank.find("aie::tensor<");
    size_t ls = t == std::string::npos ? 0 : blank.rfind('\n', t);
    ls = (t == std::string::npos || ls == std::string::npos) ? 0 : ls + 1;
    src.insert(ls, kHostPrelude);
    return true;
}

std::string valuesText(const std::map<std::string, int> &v) {
    std::string s;
    for (auto &kv : v)
        s += (s.empty() ? "" : ", ") + kv.first + "=" + std::to_string(kv.second);
    return s;
}

struct Candidate {
    std::map<std::string, int> values;
    bool ok = false;
    std::vector<std::string> errors, notes;
    double est_ms = 0;
    std::string physical; // the source with the kernel lowered to the staged layout
    std::map<std::string, PortLayout> layout;
};

// Pass A (kernel coordinates) + layout + pass B (staged layout) for one choice
// of the template parameters.
Candidate evalCandidate(const std::string &src, const std::string &srcDir, const std::string &srcName,
                        const std::map<std::string, Tensor> &aliases, const TKernel &k,
                        const std::map<std::string, int> &values, int rows, int cols) {
    Candidate c;
    c.values = values;
    Lowering L{&aliases, &k, values, {}, true, rows, cols};
    for (auto &p : k.ports)
        L.layout[p.var] = {std::vector<int>(aliases.at(p.alias).dims.size(), 0), aliases.at(p.alias).dims};
    std::string logical, err;
    if (!lowerKernel(L, src, logical, err)) {
        c.errors.push_back(err);
        return c;
    }
    if (const char *dump = std::getenv("AIEHLC_FRONT_DUMP")) // debug: the lowered kernel, per pass
        std::ofstream(std::string(dump) + "." + valuesText(values) + ".logical.cc") << logical;
    std::vector<TraceKernel> tks = findPlainKernels(logical, srcDir, err);
    auto tk = std::find_if(tks.begin(), tks.end(), [&](const TraceKernel &t) { return t.name == k.name; });
    if (tk == tks.end()) {
        c.errors.push_back(k.name + ": lowering failed" + (err.empty() ? "" : ": " + err));
        return c;
    }
    std::map<std::string, PortUse> use;
    KernelTraceResult a = traceLogical(logical, *tk, rows, cols, traceDefines(logical), srcName, 2, use);
    c.notes = a.notes;
    for (auto &e : a.errors)
        c.errors.push_back(k.name + ": " + e);
    if (!c.errors.empty() || !layoutFrom(k, aliases, use, c.layout, c.errors))
        return c;
    L.logical = false;
    L.layout = c.layout;
    if (!lowerKernel(L, src, c.physical, err)) {
        c.errors.push_back(err);
        return c;
    }
    if (const char *dump = std::getenv("AIEHLC_FRONT_DUMP"))
        std::ofstream(std::string(dump) + "." + valuesText(values) + ".staged.cc") << c.physical;
    std::string copy = c.physical;
    KernelOnlyResult b = inferKernelOnly(copy, srcDir, srcName);
    c.notes.insert(c.notes.end(), b.notes.begin(), b.notes.end());
    c.errors = b.errors;
    if (b.ok && b.est_ms.count(k.name)) {
        c.ok = true;
        c.est_ms = b.est_ms[k.name];
    } else if (c.errors.empty())
        c.errors.push_back(k.name + ": " + (b.error.empty() ? "the DMA could not be derived" : b.error));
    return c;
}

// The template-parameter choices: explicit launch arguments, else the
// defaults first and then every aie::autotune combination.
std::vector<std::map<std::string, int>> candidateValues(const TKernel &k, const std::vector<int> &fixed) {
    std::map<std::string, int> def;
    for (size_t i = 0; i < k.tparams.size(); i++)
        def[k.tparams[i].first] = i < fixed.size() ? fixed[i] : k.tparams[i].second;
    std::vector<std::map<std::string, int>> out{def};
    if (!fixed.empty())
        return out;
    for (auto &t : k.tune) {
        std::vector<std::map<std::string, int>> next;
        for (auto &base : out)
            for (int v : t.second) {
                auto m = base;
                m[t.first] = v;
                next.push_back(m);
            }
        out = next;
    }
    std::vector<std::map<std::string, int>> uniq{def};
    for (auto &m : out)
        if (std::find(uniq.begin(), uniq.end(), m) == uniq.end())
            uniq.push_back(m);
    return uniq;
}

} // namespace

TensorFormResult lowerTensorForm(std::string &src, const std::string &srcDir, const std::string &srcName) {
    TensorFormResult res;
    std::string err, blank = sourceBlankComments(src);
    if (blank.find("aie::in<") == std::string::npos && blank.find("aie::out<") == std::string::npos)
        return res;
    auto aliases = parseAliases(blank, sourceMacros(src, srcDir), err);
    std::vector<TKernel> kernels = err.empty() ? findTensorKernels(src, aliases, err) : std::vector<TKernel>{};
    if (kernels.empty() && err.empty())
        return res;
    res.found = true;
    auto fail = [&](const std::string &e) {
        res.ok = false;
        res.errors.push_back(e);
        return res;
    };
    if (!err.empty())
        return fail(err);
    if (kernels.size() != 1)
        return fail("tensor form: one aie::in / aie::out kernel per source is supported");
    const TKernel &k = kernels[0];
    std::string meshWhy;
    auto mesh = sourceLaunchMesh(src, srcDir, &meshWhy);
    if (mesh.first <= 0 || mesh.second <= 0)
        return fail(k.name + ": no launch mesh: " + meshWhy);
    std::vector<Launch> launches = findLaunches(src, k, err);
    if (!err.empty())
        return fail(err);
    std::vector<int> fixed = launches.empty() ? std::vector<int>{} : launches[0].targs;
    std::vector<std::future<Candidate>> jobs;
    for (auto &v : candidateValues(k, fixed))
        jobs.push_back(std::async(std::launch::async, evalCandidate, std::cref(src), std::cref(srcDir),
                                  std::cref(srcName), std::cref(aliases), std::cref(k), v, mesh.first, mesh.second));
    std::vector<Candidate> cands;
    for (auto &j : jobs)
        cands.push_back(j.get());
    std::ostringstream rep;
    rep << "[TensorForm] " << k.name << ": " << cands.size() << " candidate(s) traced on the " << mesh.first << "x"
        << mesh.second << " mesh\n";
    const Candidate *best = nullptr;
    for (auto &c : cands) {
        rep << "  " << valuesText(c.values) << ": ";
        if (c.ok)
            rep << "legal, est " << c.est_ms << " ms\n";
        else
            rep << "rejected: " << (c.errors.empty() ? "?" : c.errors[0]) << "\n";
        if (c.ok && (!best || c.est_ms < best->est_ms))
            best = &c;
    }
    if (!best) { // report the default's (first) problems in full
        res.report = rep.str();
        res.ok = false;
        res.errors = cands[0].errors;
        res.notes = cands[0].notes;
        return res;
    }
    rep << "  => " << valuesText(best->values) << (cands.size() > 1 ? " (fastest legal under the cost model)" : "")
        << "\n";
    for (auto &p : k.ports) {
        const PortLayout &pl = best->layout.at(p.var);
        const Tensor &t = aliases.at(p.alias);
        if (pl.phys != t.dims)
            rep << "  " << p.var << ": " << p.alias << " [" << listText(t.dims) << "] staged as [" << listText(pl.phys)
                << "] (zero border [" << listText(pl.lo) << "], channels aligned) before the launch\n";
    }
    res.notes = best->notes;
    std::string out = best->physical;
    if (!lowerHost(out, k, aliases, best->layout, res.notes, err))
        return fail(err);
    src = out;
    res.report = rep.str();
    return res;
}

} // namespace convautotile
