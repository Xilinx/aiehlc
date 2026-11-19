/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C++ templates on __global__ kernels, instantiated on the source text.
 * See kernel_template.h.
 ******************************************************************************/
#include "kernel_template.h"
#include "conv_desugar.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <tuple>

namespace convautotile {

namespace {

bool identChar(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

size_t skipWs(const std::string &s, size_t i) {
    while (i < s.size() && std::isspace((unsigned char)s[i]))
        i++;
    return i;
}

int lineOf(const std::string &s, size_t pos) { return 1 + (int)std::count(s.begin(), s.begin() + pos, '\n'); }

// Comments AND string / char literals blanked (offsets and newlines kept): the
// text the token scans look at.
std::string mask(const std::string &s) {
    std::string r = sourceBlankComments(s);
    for (size_t i = 0; i < r.size(); i++) {
        if (r[i] != '"' && r[i] != '\'')
            continue;
        const char q = r[i];
        size_t j = i + 1;
        while (j < r.size() && r[j] != q && r[j] != '\n') {
            if (r[j] == '\\')
                r[j++] = ' ';
            if (j < r.size())
                r[j++] = ' ';
        }
        i = j;
    }
    return r;
}

// Top-level comma split (depth over () [] {} <>).
std::vector<std::string> splitTop(const std::string &s) {
    std::vector<std::string> out;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (c == '(' || c == '[' || c == '{' || c == '<')
            depth++;
        else if (c == ')' || c == ']' || c == '}' || c == '>')
            depth--;
        else if (c == ',' && depth == 0) {
            out.push_back(trim(s.substr(start, i - start)));
            start = i + 1;
        }
    }
    if (!trim(s.substr(start)).empty() || !out.empty())
        out.push_back(trim(s.substr(start)));
    return out;
}

struct TParam {
    std::string name, dflt;
    bool type = false; // typename / class
};

struct TKernel {
    size_t tmplBegin = 0, tmplEnd = 0; // `template <...>`
    size_t nameBegin = 0, nameEnd = 0; // the kernel name in the declaration
    size_t bodyEnd = 0;                // one past the closing brace
    std::string name;
    std::vector<TParam> params;
};

// The next `template <...> __global__ [(...)] void name(...) {...}` at or after `from`.
bool findTemplateKernel(const std::string &src, const std::string &m, size_t from, TKernel &k, std::string &err) {
    for (size_t t = sourceFindToken(m, "template", from); t != std::string::npos;
         t = sourceFindToken(m, "template", t + 8)) {
        size_t lt = skipWs(m, t + 8);
        if (lt >= m.size() || m[lt] != '<')
            continue;
        const size_t gt = sourceMatchClose(m, lt, '<', '>');
        if (gt == std::string::npos)
            continue;
        size_t g = skipWs(m, gt + 1);
        if (m.compare(g, 10, "__global__") != 0 || (g + 10 < m.size() && identChar(m[g + 10])))
            continue; // a template, but not of a kernel
        size_t i = skipWs(m, g + 10);
        if (i < m.size() && m[i] == '(') { // __global__(policy)
            i = sourceMatchClose(m, i, '(', ')');
            if (i == std::string::npos)
                continue;
            i = skipWs(m, i + 1);
        }
        if (m.compare(i, 4, "void") != 0)
            continue;
        i = skipWs(m, i + 4);
        const size_t nb = i;
        while (i < m.size() && identChar(m[i]))
            i++;
        if (i == nb)
            continue;
        k.name = m.substr(nb, i - nb);
        k.nameBegin = nb;
        k.nameEnd = i;
        i = skipWs(m, i);
        if (i >= m.size() || m[i] != '(')
            continue;
        const size_t pc = sourceMatchClose(m, i, '(', ')');
        const size_t bo = pc == std::string::npos ? pc : skipWs(m, pc + 1);
        if (bo == std::string::npos || bo >= m.size() || m[bo] != '{') {
            err = "template kernel " + k.name + " (line " + std::to_string(lineOf(src, t)) + "): no body";
            return false;
        }
        const size_t bc = sourceMatchClose(m, bo, '{', '}');
        if (bc == std::string::npos) {
            err = "template kernel " + k.name + ": unbalanced braces";
            return false;
        }
        k.tmplBegin = t;
        k.tmplEnd = gt + 1;
        k.bodyEnd = bc + 1;
        k.params.clear();
        for (const std::string &p : splitTop(src.substr(lt + 1, gt - lt - 1))) {
            TParam tp;
            std::string decl = p;
            int depth = 0;
            for (size_t q = 0; q < p.size(); q++) { // the first top-level '='
                if (p[q] == '(' || p[q] == '<' || p[q] == '[')
                    depth++;
                else if (p[q] == ')' || p[q] == '>' || p[q] == ']')
                    depth--;
                else if (p[q] == '=' && depth == 0) {
                    decl = trim(p.substr(0, q));
                    tp.dflt = trim(p.substr(q + 1));
                    break;
                }
            }
            if (decl.find("...") != std::string::npos || decl.find("template") != std::string::npos) {
                err = "template kernel " + k.name + ": parameter `" + p +
                      "` is not supported (integral or typename parameters only)";
                return false;
            }
            size_t e = decl.size();
            while (e > 0 && identChar(decl[e - 1]))
                e--;
            tp.name = decl.substr(e);
            const std::string kind = trim(decl.substr(0, e));
            if (tp.name.empty() || kind.empty()) {
                err = "template kernel " + k.name + ": cannot read parameter `" + p + "`";
                return false;
            }
            tp.type = kind == "typename" || kind == "class";
            k.params.push_back(tp);
        }
        return true;
    }
    return false;
}

// Replace every template-parameter token of src[b, e) (not a member: no `.`,
// `->` or `::` in front) by its argument text.
std::string substitute(const std::string &src, size_t b, size_t e, const std::map<std::string, std::string> &args) {
    const std::string m = mask(src);
    std::string out;
    size_t i = b;
    while (i < e) {
        if (!identChar(m[i]) || (i > b && identChar(m[i - 1]))) {
            out += src[i++];
            continue;
        }
        size_t j = i;
        while (j < e && identChar(m[j]))
            j++;
        const std::string tok = m.substr(i, j - i);
        auto it = args.find(tok);
        size_t p = i;
        while (p > b && (m[p - 1] == ' ' || m[p - 1] == '\t'))
            p--;
        const bool member = (p > b && m[p - 1] == '.') || (p > b + 1 && m.compare(p - 2, 2, "->") == 0) ||
                            (p > b + 1 && m.compare(p - 2, 2, "::") == 0);
        out +=
            (it != args.end() && !member && !std::isdigit((unsigned char)tok[0])) ? it->second : src.substr(i, j - i);
        i = j;
    }
    return out;
}

std::string mangle(const std::string &name, const std::vector<std::string> &args) {
    std::string s = name + "_";
    for (const std::string &a : args) {
        s += "_";
        for (char c : a)
            s += identChar(c) ? c : (c == '-' ? 'm' : '_');
    }
    return s;
}

struct Edit {
    size_t b, e;
    std::string text;
};

} // namespace

KernelTemplateResult instantiateKernelTemplates(std::string &src, const std::string &srcDir) {
    KernelTemplateResult r;
    const std::map<std::string, std::string> macros = sourceMacros(src, srcDir);
    size_t from = 0;
    for (int guard = 0; guard < 64; guard++) {
        const std::string m = mask(src);
        TKernel k;
        std::string err;
        if (!findTemplateKernel(src, m, from, k, err)) {
            if (!err.empty()) {
                r.found = true;
                r.ok = false;
                r.errors.push_back(err);
            }
            return r;
        }
        r.found = true;
        // Every launch `name<args><<<` outside the definition selects an instance.
        std::vector<std::vector<std::string>> insts;
        std::vector<std::tuple<size_t, size_t, size_t>> launches; // name begin, `<<<` begin, instance
        for (size_t p = sourceFindToken(m, k.name, 0); p != std::string::npos;
             p = sourceFindToken(m, k.name, p + k.name.size())) {
            if (p >= k.tmplBegin && p < k.bodyEnd)
                continue;
            const size_t lt = skipWs(m, p + k.name.size());
            if (lt >= m.size() || m[lt] != '<' || m.compare(lt, 3, "<<<") == 0) {
                if (m.compare(lt, 3, "<<<") == 0) {
                    r.ok = false;
                    r.errors.push_back("template kernel " + k.name + " launched without template arguments (line " +
                                       std::to_string(lineOf(src, p)) + "): write " + k.name + "<...><<<mesh>>>(...)");
                }
                continue;
            }
            const size_t gt = sourceMatchClose(m, lt, '<', '>');
            if (gt == std::string::npos)
                continue;
            const size_t ll = skipWs(m, gt + 1);
            if (m.compare(ll, 3, "<<<") != 0)
                continue; // not a launch (e.g. a pointer to an instance)
            std::vector<std::string> args = splitTop(src.substr(lt + 1, gt - lt - 1));
            if (args.size() > k.params.size()) {
                r.ok = false;
                r.errors.push_back(k.name + "<" + src.substr(lt + 1, gt - lt - 1) + "> at line " +
                                   std::to_string(lineOf(src, p)) + ": " + std::to_string(args.size()) +
                                   " template arguments, the kernel has " + std::to_string(k.params.size()));
                continue;
            }
            // Defaults (which may use earlier parameters), then fold integers.
            std::map<std::string, std::string> bound;
            bool complete = true;
            for (size_t a = 0; a < k.params.size(); a++) {
                const TParam &tp = k.params[a];
                std::string v = a < args.size() ? args[a] : tp.dflt;
                if (v.empty()) {
                    r.ok = false;
                    r.errors.push_back(k.name + " launched at line " + std::to_string(lineOf(src, p)) +
                                       ": no argument and no default for template parameter " + tp.name);
                    complete = false;
                    break;
                }
                if (a >= args.size())
                    v = trim(substitute(v, 0, v.size(), bound));
                if (!tp.type) {
                    long n = 0;
                    const std::string e = v == "true" ? "1" : v == "false" ? "0" : v;
                    v = sourceEvalInt(e, macros, n) ? std::to_string(n) : "(" + v + ")";
                }
                bound[tp.name] = v;
                if (a >= args.size())
                    args.push_back(v);
                else
                    args[a] = v;
            }
            if (!complete)
                continue;
            size_t idx = std::find(insts.begin(), insts.end(), args) - insts.begin();
            if (idx == insts.size())
                insts.push_back(args);
            launches.push_back({p, ll, idx});
        }
        if (insts.empty()) {
            r.ok = false;
            r.errors.push_back("template kernel " + k.name + " (line " + std::to_string(lineOf(src, k.tmplBegin)) +
                               ") is never launched as " + k.name + "<...><<<mesh>>>(...): nothing to instantiate");
            return r;
        }
        if (!r.ok)
            return r;
        std::vector<std::string> names;
        for (auto &a : insts)
            names.push_back(insts.size() == 1 ? k.name : mangle(k.name, a));
        std::vector<Edit> edits;
        for (auto &[p, ll, idx] : launches)
            edits.push_back({p, ll, names[idx]});
        // The definition: the first instance in place (the header blanked, so
        // line numbers stay), the others right after it.
        std::string defs;
        for (size_t i = 0; i < insts.size(); i++) {
            std::map<std::string, std::string> args;
            std::string list;
            for (size_t a = 0; a < k.params.size(); a++) {
                args[k.params[a].name] = insts[i][a];
                list += (a ? ", " : "") + insts[i][a];
            }
            std::string text =
                src.substr(k.tmplEnd, k.nameBegin - k.tmplEnd) + names[i] + substitute(src, k.nameEnd, k.bodyEnd, args);
            if (i == 0) {
                std::string header = src.substr(k.tmplBegin, k.tmplEnd - k.tmplBegin);
                for (char &c : header)
                    if (c != '\n')
                        c = ' ';
                defs += header + text;
            } else {
                defs += "\n\n" + text;
            }
            r.notes.push_back(k.name + "<" + list + "> -> " + names[i]);
        }
        edits.push_back({k.tmplBegin, k.bodyEnd, defs});
        std::sort(edits.begin(), edits.end(), [](const Edit &a, const Edit &b) { return a.b > b.b; });
        for (const Edit &e : edits)
            src.replace(e.b, e.e - e.b, e.text);
        from = k.tmplBegin; // the instances are no longer templates: find the next one
    }
    r.ok = false;
    r.errors.push_back("too many template kernels (more than 64)");
    return r;
}

} // namespace convautotile
