/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source-level desugaring of aie::Conv2d / aie::Mapping — see conv_desugar.h.
 ******************************************************************************/
#include "conv_desugar.h"
#include "kernel_trace.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

namespace convautotile {

namespace {

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------
bool isIdent(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

// Replace comments with spaces (offsets preserved) so scanning ignores them.
std::string blankComments(const std::string &s) {
    std::string r = s;
    size_t i = 0, n = s.size();
    while (i < n) {
        if (s[i] == '"' || s[i] == '\'') {
            char q = s[i++];
            while (i < n && s[i] != q) {
                if (s[i] == '\\')
                    i++;
                i++;
            }
            i++;
        } else if (s.compare(i, 2, "//") == 0) {
            while (i < n && s[i] != '\n')
                r[i++] = ' ';
        } else if (s.compare(i, 2, "/*") == 0) {
            size_t e = s.find("*/", i + 2);
            e = (e == std::string::npos) ? n : e + 2;
            for (; i < e; i++)
                if (r[i] != '\n')
                    r[i] = ' ';
        } else {
            i++;
        }
    }
    return r;
}

size_t matchClose(const std::string &s, size_t open, char o, char c) {
    int depth = 0;
    for (size_t i = open; i < s.size(); i++) {
        if (s[i] == o)
            depth++;
        else if (s[i] == c && --depth == 0)
            return i;
    }
    return std::string::npos;
}

// Find `word` as a whole token at or after `from` in comment-blanked text.
size_t findToken(const std::string &s, const std::string &word, size_t from) {
    while ((from = s.find(word, from)) != std::string::npos) {
        bool l = from == 0 || !isIdent(s[from - 1]);
        bool r = from + word.size() >= s.size() || !isIdent(s[from + word.size()]);
        if (l && r)
            return from;
        from += word.size();
    }
    return std::string::npos;
}

// ---------------------------------------------------------------------------
// Object-like integer macros (#define NAME expr) from the source and its
// quoted headers.
// ---------------------------------------------------------------------------
void collectMacros(const std::string &text, const std::string &dir, std::map<std::string, std::string> &out,
                   int depth) {
    std::string t = blankComments(text);
    std::istringstream is(t);
    std::string line;
    while (std::getline(is, line)) {
        size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] != '#')
            continue;
        std::string rest = line.substr(p + 1);
        size_t q = rest.find_first_not_of(" \t");
        if (q == std::string::npos)
            continue;
        rest = rest.substr(q);
        if (rest.compare(0, 6, "define") == 0) {
            size_t a = rest.find_first_not_of(" \t", 6);
            if (a == std::string::npos)
                continue;
            size_t b = a;
            while (b < rest.size() && isIdent(rest[b]))
                b++;
            if (b < rest.size() && rest[b] == '(') // function-like: skip
                continue;
            std::string name = rest.substr(a, b - a);
            std::string body = rest.substr(b);
            if (!body.empty() && body.back() == '\\')
                continue; // multi-line macros are never tile constants
            if (!out.count(name))
                out[name] = body;
        } else if (rest.compare(0, 7, "include") == 0 && depth < 4 && !dir.empty()) {
            size_t a = rest.find('"');
            size_t b = a == std::string::npos ? a : rest.find('"', a + 1);
            if (b == std::string::npos)
                continue;
            std::ifstream f(dir + "/" + rest.substr(a + 1, b - a - 1));
            if (!f)
                continue;
            std::stringstream ss;
            ss << f.rdbuf();
            collectMacros(ss.str(), dir, out, depth + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Designated-initializer value tree
// ---------------------------------------------------------------------------
struct Value {
    enum Kind { Int, Sym, Call, List } kind = Int;
    long i = 0;
    std::string sym;                                  // Sym / Call name
    std::vector<std::pair<std::string, Value>> items; // List (field, value) / Call args
};

class Parser {
  public:
    Parser(const std::string &s, const std::map<std::string, std::string> &macros, int depth = 0)
        : s_(s), macros_(macros), depth_(depth) {}

    bool parseInit(Value &v) {
        ws();
        if (!peek('{'))
            return fail("expected '{'");
        return parseList(v) && (ws(), true);
    }
    bool parseExprOnly(long &out) {
        Value v;
        if (!parseExpr(v))
            return false;
        ws();
        if (pos_ != s_.size())
            return fail("trailing text");
        if (v.kind != Value::Int)
            return fail("not an integer expression");
        out = v.i;
        return true;
    }
    const std::string &error() const { return err_; }

  private:
    const std::string &s_;
    const std::map<std::string, std::string> &macros_;
    int depth_;
    size_t pos_ = 0;
    std::string err_;

    bool fail(const std::string &m) {
        if (err_.empty())
            err_ = m + " near '" + s_.substr(pos_, 24) + "'";
        return false;
    }
    void ws() {
        while (pos_ < s_.size() && std::isspace((unsigned char)s_[pos_]))
            pos_++;
    }
    bool peek(char c) {
        ws();
        return pos_ < s_.size() && s_[pos_] == c;
    }
    bool eat(char c) {
        if (!peek(c))
            return false;
        pos_++;
        return true;
    }
    std::string ident() {
        ws();
        size_t b = pos_;
        while (pos_ < s_.size()) {
            if (isIdent(s_[pos_]))
                pos_++;
            else if (s_.compare(pos_, 2, "::") == 0)
                pos_ += 2;
            else
                break;
        }
        return s_.substr(b, pos_ - b);
    }
    bool parseList(Value &v) {
        v = Value();
        v.kind = Value::List;
        eat('{');
        while (!peek('}')) {
            std::string field;
            if (eat('.')) {
                field = ident();
                if (!eat('='))
                    return fail("expected '=' after ." + field);
            }
            Value item;
            if (!parseValue(item))
                return false;
            v.items.push_back({field, item});
            if (!eat(','))
                break;
        }
        if (!eat('}'))
            return fail("expected '}'");
        return true;
    }
    bool parseValue(Value &v) {
        if (peek('{'))
            return parseList(v);
        return parseExpr(v);
    }
    // expr := term (('+'|'-') term)* ; term := unary (('*'|'/'|'%') unary)*
    bool parseExpr(Value &v) {
        if (!parseTerm(v))
            return false;
        while (peek('+') || peek('-')) {
            char op = s_[pos_++];
            Value r;
            if (!parseTerm(r) || !bothInt(v, r))
                return false;
            v.i = op == '+' ? v.i + r.i : v.i - r.i;
        }
        return true;
    }
    bool parseTerm(Value &v) {
        if (!parseUnary(v))
            return false;
        while (peek('*') || peek('/') || peek('%')) {
            char op = s_[pos_++];
            Value r;
            if (!parseUnary(r) || !bothInt(v, r))
                return false;
            if ((op == '/' || op == '%') && r.i == 0)
                return fail("division by zero");
            v.i = op == '*' ? v.i * r.i : op == '/' ? v.i / r.i : v.i % r.i;
        }
        return true;
    }
    bool bothInt(const Value &a, const Value &b) {
        if (a.kind == Value::Int && b.kind == Value::Int)
            return true;
        return fail("arithmetic on a non-integer");
    }
    bool parseUnary(Value &v) {
        if (eat('-')) {
            if (!parseUnary(v))
                return false;
            if (v.kind != Value::Int)
                return fail("negation of a non-integer");
            v.i = -v.i;
            return true;
        }
        if (eat('+'))
            return parseUnary(v);
        if (eat('(')) {
            if (!parseExpr(v) || !eat(')'))
                return fail("expected ')'");
            return true;
        }
        ws();
        if (pos_ < s_.size() && std::isdigit((unsigned char)s_[pos_])) {
            size_t b = pos_;
            while (pos_ < s_.size() && (std::isalnum((unsigned char)s_[pos_])))
                pos_++;
            std::string num = s_.substr(b, pos_ - b);
            while (!num.empty() && (num.back() == 'u' || num.back() == 'U' || num.back() == 'l' || num.back() == 'L'))
                num.pop_back();
            v = Value();
            v.kind = Value::Int;
            char *end = nullptr;
            v.i = std::strtol(num.c_str(), &end, 0);
            if (!end || *end)
                return fail("bad number '" + num + "'");
            return true;
        }
        std::string id = ident();
        if (id.empty())
            return fail("expected a value");
        if (peek('(') || peek('{')) { // aie::split(aie::OH, 4) / aie::Bytes{4096}
            char close = peek('(') ? ')' : '}';
            pos_++;
            v = Value();
            v.kind = Value::Call;
            v.sym = id;
            while (!peek(close)) {
                Value a;
                if (eat('.')) { // Bytes{.value = N}
                    ident();
                    if (!eat('='))
                        return fail("expected '='");
                }
                if (!parseValue(a))
                    return false;
                v.items.push_back({"", a});
                if (!eat(','))
                    break;
            }
            if (!eat(close))
                return fail(std::string("expected '") + close + "'");
            return true;
        }
        auto m = macros_.find(id);
        if (m != macros_.end() && depth_ < 16) {
            Parser sub(m->second, macros_, depth_ + 1);
            long val;
            if (sub.parseExprOnly(val)) {
                v = Value();
                v.kind = Value::Int;
                v.i = val;
                return true;
            }
        }
        v = Value();
        v.kind = Value::Sym;
        v.sym = id;
        return true;
    }
};

const Value *field(const Value &list, const std::string &name) {
    for (auto &kv : list.items)
        if (kv.first == name)
            return &kv.second;
    return nullptr;
}

bool asInt(const Value *v, int &out, std::string &err, const std::string &what) {
    if (!v)
        return true; // keep default
    if (v->kind == Value::Int) {
        out = (int)v->i;
        return true;
    }
    if (v->kind == Value::Call && v->items.size() == 1 && v->items[0].second.kind == Value::Int) {
        out = (int)v->items[0].second.i; // aie::Bytes{4096}
        return true;
    }
    err = what + " must be an integer constant expression";
    return false;
}

std::string lastComponent(const std::string &s) {
    size_t p = s.rfind("::");
    return p == std::string::npos ? s : s.substr(p + 2);
}

bool toConv(const Value &v, Conv2dSpec &c, std::string &err) {
    static const char *known[] = {"h", "w", "c", "k", "kh", "kw", "stride", "pad", "dilation", "c_align"};
    for (auto &kv : v.items) {
        bool ok = false;
        for (auto *k : known)
            ok |= kv.first == k;
        if (!ok) {
            err = "aie::Conv2d: unknown field '." + kv.first + "'";
            return false;
        }
    }
    return asInt(field(v, "h"), c.h, err, ".h") && asInt(field(v, "w"), c.w, err, ".w") &&
           asInt(field(v, "c"), c.c, err, ".c") && asInt(field(v, "k"), c.k, err, ".k") &&
           asInt(field(v, "kh"), c.kh, err, ".kh") && asInt(field(v, "kw"), c.kw, err, ".kw") &&
           asInt(field(v, "stride"), c.stride, err, ".stride") && asInt(field(v, "pad"), c.pad, err, ".pad") &&
           asInt(field(v, "dilation"), c.dilation, err, ".dilation") &&
           asInt(field(v, "c_align"), c.c_align, err, ".c_align");
}

// Launch mesh shape, from the host's `device.partition({...}, ROWS, COLS)`.
struct MeshShape {
    int rows = 0, cols = 0;
};

bool toSplit(const Value &v, const char *fn, const MeshShape &mesh, AxisSplit &out, std::string &err) {
    if (v.kind != Value::Call || lastComponent(v.sym) != fn || v.items.size() != 2 ||
        v.items[0].second.kind != Value::Sym) {
        err = std::string("expected aie::") + fn + "(aie::<AXIS>, <int>)";
        return false;
    }
    out.axis = axisFromName(v.items[0].second.sym);
    const Value &f = v.items[1].second;
    if (f.kind == Value::Int) {
        out.factor = (int)f.i;
    } else if (f.kind == Value::Sym && (lastComponent(f.sym) == "MESH_ROW" || lastComponent(f.sym) == "MESH_ROWS" ||
                                        lastComponent(f.sym) == "MESH_COL" || lastComponent(f.sym) == "MESH_COLS")) {
        // Symbolic mesh extent: resolved from the launch partition.
        bool row = lastComponent(f.sym).compare(0, 8, "MESH_ROW") == 0;
        out.factor = row ? mesh.rows : mesh.cols;
        if (out.factor <= 0) {
            err = f.sym + " needs the launch mesh shape, but no device.partition({...}, ROWS, COLS) was found";
            return false;
        }
    } else {
        err = std::string("aie::") + fn + ": the factor must be an integer constant expression, MESH_ROW or MESH_COL";
        return false;
    }
    if (out.axis == Axis::None) {
        err = "unknown axis '" + v.items[0].second.sym + "' (use aie::OH, aie::OW, aie::OC, aie::IC)";
        return false;
    }
    return true;
}

bool toMapping(const Value &v, const MeshShape &mesh, MappingSpec &m, std::string &err) {
    const Value *sp = field(v, "space");
    if (!sp || sp->kind != Value::List || sp->items.size() != 2) {
        err = "aie::Mapping: .space must list exactly two splits {mesh rows, mesh cols}";
        return false;
    }
    if (!toSplit(sp->items[0].second, "split", mesh, m.space_rows, err) ||
        !toSplit(sp->items[1].second, "split", mesh, m.space_cols, err))
        return false;
    if ((mesh.rows > 0 && m.space_rows.factor != mesh.rows) || (mesh.cols > 0 && m.space_cols.factor != mesh.cols)) {
        err = "aie::Mapping: .space splits " + std::to_string(m.space_rows.factor) + "x" +
              std::to_string(m.space_cols.factor) + " do not match the launch mesh " + std::to_string(mesh.rows) + "x" +
              std::to_string(mesh.cols) + " (device.partition)";
        return false;
    }
    if (const Value *t = field(v, "time")) {
        if (t->kind != Value::List) {
            err = "aie::Mapping: .time must be a list of aie::tile(...)";
            return false;
        }
        for (auto &kv : t->items) {
            AxisSplit s;
            if (!toSplit(kv.second, "tile", MeshShape(), s, err))
                return false;
            m.time.push_back(s);
        }
    }
    if (const Value *h = field(v, "hints")) {
        for (auto &kv : h->items) {
            const Value &x = kv.second;
            std::string sym = x.kind == Value::Sym ? lastComponent(x.sym) : "";
            if (kv.first == "pad")
                m.hints.pad = sym == "Memtile" ? 1 : 0;
            else if (kv.first == "im2col")
                m.hints.im2col = sym == "None" ? 0 : 1;
            else if (kv.first == "pp_depth") {
                if (!asInt(&x, m.hints.pp_depth, err, ".pp_depth"))
                    return false;
            } else if (kv.first == "l1_budget") {
                if (!asInt(&x, m.hints.l1_budget, err, ".l1_budget"))
                    return false;
            } else {
                err = "aie::Mapping: unknown hint '." + kv.first + "'";
                return false;
            }
        }
    }
    return true;
}

struct Decl {
    std::string type, name;
    size_t begin = 0, end = 0; // [begin, end) covers `constexpr ... ;`
    Value init;
};

// Find `[static] constexpr aie::<type> <name> = { ... };` declarations.
bool findDecls(const std::string &src, const std::string &blank, const std::string &type,
               const std::map<std::string, std::string> &macros, std::vector<Decl> &out, std::string &err) {
    const std::string tok = "aie::" + type;
    size_t p = 0;
    while ((p = findToken(blank, tok, p)) != std::string::npos) {
        Decl d;
        d.type = type;
        size_t q = p + tok.size();
        while (q < blank.size() && std::isspace((unsigned char)blank[q]))
            q++;
        size_t nb = q;
        while (q < blank.size() && isIdent(blank[q]))
            q++;
        d.name = blank.substr(nb, q - nb);
        size_t eq = blank.find_first_not_of(" \t\r\n", q);
        if (d.name.empty() || eq == std::string::npos || blank[eq] != '=') {
            p = q; // a use, not a definition (e.g. a parameter type)
            continue;
        }
        size_t ob = blank.find('{', eq);
        size_t cb = ob == std::string::npos ? ob : matchClose(blank, ob, '{', '}');
        size_t semi = cb == std::string::npos ? cb : blank.find(';', cb);
        if (semi == std::string::npos) {
            err = "unterminated aie::" + type + " " + d.name;
            return false;
        }
        size_t lineStart = blank.rfind('\n', p);
        d.begin = lineStart == std::string::npos ? 0 : lineStart + 1;
        d.end = semi + 1;
        std::string initText = blank.substr(ob, cb - ob + 1);
        Parser ps(initText, macros);
        if (!ps.parseInit(d.init)) {
            err = "aie::" + type + " " + d.name + ": " + ps.error();
            return false;
        }
        (void)src;
        out.push_back(d);
        p = semi;
    }
    return true;
}

struct GlobalUse {
    size_t kwBegin = 0, argsEnd = 0; // `__global__(a, b)` span
    std::string convVar, mapVar, kernel;
    size_t paramOpen = 0, paramClose = 0;
};

bool findGlobalUses(const std::string &blank, std::vector<GlobalUse> &out, std::string &err) {
    size_t p = 0;
    while ((p = findToken(blank, "__global__", p)) != std::string::npos) {
        size_t q = blank.find_first_not_of(" \t", p + 10);
        if (q == std::string::npos || blank[q] != '(') {
            p += 10;
            continue;
        }
        size_t cp = matchClose(blank, q, '(', ')');
        if (cp == std::string::npos) {
            err = "unterminated __global__(...)";
            return false;
        }
        std::string args = blank.substr(q + 1, cp - q - 1);
        size_t comma = args.find(',');
        if (comma == std::string::npos) { // __global__(policy): legacy form
            p = cp;
            continue;
        }
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        GlobalUse g;
        g.kwBegin = p;
        g.argsEnd = cp + 1;
        g.convVar = trim(args.substr(0, comma));
        g.mapVar = trim(args.substr(comma + 1));
        g.paramOpen = blank.find('(', cp + 1);
        if (g.paramOpen == std::string::npos) {
            err = "__global__(" + args + "): kernel parameter list not found";
            return false;
        }
        size_t nameEnd = g.paramOpen;
        while (nameEnd > 0 && std::isspace((unsigned char)blank[nameEnd - 1]))
            nameEnd--;
        size_t nameBegin = nameEnd;
        while (nameBegin > 0 && isIdent(blank[nameBegin - 1]))
            nameBegin--;
        g.kernel = blank.substr(nameBegin, nameEnd - nameBegin);
        g.paramClose = matchClose(blank, g.paramOpen, '(', ')');
        if (g.paramClose == std::string::npos) {
            err = "kernel " + g.kernel + ": unterminated parameter list";
            return false;
        }
        out.push_back(g);
        p = g.paramClose;
    }
    return true;
}

std::string spaceVar(const std::string &kernel, const std::string &role) { return "__aie_" + kernel + "_" + role; }

// Names already #defined by the user are not re-emitted (the user's value wins;
// a different value is reported).
std::vector<std::string> userDefinedKernelNames(const Derived &d, const std::map<std::string, std::string> &macros,
                                                std::vector<std::string> &warnings) {
    std::vector<std::string> skip;
    for (auto &kv : d.kernel_defines) {
        auto m = macros.find(kv.first);
        if (m == macros.end())
            continue;
        skip.push_back(kv.first);
        Parser ps(m->second, macros);
        long v;
        if (ps.parseExprOnly(v) && v != kv.second)
            warnings.push_back("user #define " + kv.first + " = " + std::to_string(v) +
                               " differs from the derived value " + std::to_string(kv.second));
    }
    return skip;
}

std::string generatedBlock(const KernelDesugar &k, const std::vector<std::string> &skip) {
    const Derived &d = k.derived;
    std::ostringstream os;
    os << "// ==== aiehlc conv auto-tiling: generated from " << k.convVar << " + " << k.mapVar << " for " << k.kernel
       << " ====\n";
    os << "// time tile(OH," << d.oh_t << ") x tile(OW," << d.ow_t << ")" << (d.time_autotuned ? " [autotuned]" : "")
       << ", rounds/core=" << d.rounds << ", halo slab " << d.tile_h << "x" << d.tile_w << "x" << d.c_align << ", L1 "
       << d.l1_bytes << " B\n";
    os << emitKernelDefines(d, skip);
    os << emitGemmSpace(d.ifm, spaceVar(k.kernel, "IFM"));
    os << emitGemmSpace(d.weights, spaceVar(k.kernel, "WEIGHTS"));
    os << emitGemmSpace(d.ofm, spaceVar(k.kernel, "OFM"));
    os << "constexpr aie::GlobalPolicy " << spaceVar(k.kernel, "policy") << " = {.fullconnect_auto = 0};\n";
    os << "// ==== end aiehlc conv auto-tiling ====\n";
    return os.str();
}

// Comment a declaration out line by line (keeps it visible in aout/newfile.cpp).
std::string commentOut(const std::string &text) {
    std::string r = "// [aiehlc conv auto-tiling] lowered:\n";
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line))
        r += "// " + line + "\n";
    return r;
}

// `<obj>.partition({...}, ROWS, COLS)` -> mesh shape (0x0 if absent/unevaluable).
MeshShape findMeshShape(const std::string &blank, const std::map<std::string, std::string> &macros) {
    MeshShape ms;
    size_t p = blank.find(".partition(");
    if (p == std::string::npos)
        return ms;
    size_t open = p + 10, close = matchClose(blank, open, '(', ')');
    if (close == std::string::npos)
        return ms;
    std::string args = blank.substr(open + 1, close - open - 1);
    size_t brace = args.find('{');
    size_t afterBrace = brace == std::string::npos ? 0 : matchClose(args, brace, '{', '}') + 1;
    std::vector<std::string> parts;
    std::string cur;
    for (size_t i = afterBrace; i < args.size(); i++) {
        if (args[i] == ',') {
            parts.push_back(cur);
            cur.clear();
        } else
            cur += args[i];
    }
    parts.push_back(cur);
    std::vector<long> vals;
    for (auto &t : parts) {
        if (t.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;
        Parser ps(t, macros);
        long v;
        if (ps.parseExprOnly(v))
            vals.push_back(v);
    }
    if (vals.size() >= 2) {
        ms.rows = (int)vals[vals.size() - 2];
        ms.cols = (int)vals[vals.size() - 1];
    }
    return ms;
}

struct Parsed {
    MeshShape mesh;
    std::map<std::string, std::string> macros;
    std::vector<Decl> convs, maps;
    std::vector<GlobalUse> uses;
};

bool parseAll(const std::string &src, const std::string &srcDir, Parsed &P, DesugarResult &r) {
    std::string blank = blankComments(src);
    if (findToken(blank, "aie::Mapping", 0) == std::string::npos)
        return false;
    r.found = true;
    collectMacros(src, srcDir, P.macros, 0);
    P.mesh = findMeshShape(blank, P.macros);
    std::string err;
    if (!findDecls(src, blank, "Conv2d", P.macros, P.convs, err) ||
        !findDecls(src, blank, "Mapping", P.macros, P.maps, err) || !findGlobalUses(blank, P.uses, err)) {
        r.ok = false;
        r.error = err;
        return false;
    }
    for (auto &g : P.uses) {
        KernelDesugar k;
        k.kernel = g.kernel;
        k.convVar = g.convVar;
        k.mapVar = g.mapVar;
        const Decl *cd = nullptr, *md = nullptr;
        for (auto &d : P.convs)
            if (d.name == g.convVar)
                cd = &d;
        for (auto &d : P.maps)
            if (d.name == g.mapVar)
                md = &d;
        if (!cd || !md) {
            r.ok = false;
            r.error = "__global__(" + g.convVar + ", " + g.mapVar + ") on " + g.kernel + ": " +
                      (!cd ? "no constexpr aie::Conv2d named '" + g.convVar + "'"
                           : "no constexpr aie::Mapping named '" + g.mapVar + "'");
            return false;
        }
        if (!toConv(cd->init, k.conv, err) || !toMapping(md->init, P.mesh, k.mapping, err)) {
            r.ok = false;
            r.error = g.kernel + ": " + err;
            return false;
        }
        r.kernels.push_back(k);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Hand-written GemmSpace -> PortSpace (kernel_trace.cpp checks it against the
// kernel's blocks)
// ---------------------------------------------------------------------------
namespace {

// `constexpr int NAME = expr;` anywhere in the text, as extra macros: GemmSpace
// fields often use them (simpleconv2d.cc: OC_PER_G * HW_COLS).
void addConstexprInts(const std::string &blank, std::map<std::string, std::string> &m) {
    size_t p = 0;
    while ((p = findToken(blank, "constexpr", p)) != std::string::npos) {
        size_t semi = blank.find(';', p);
        if (semi == std::string::npos)
            break;
        std::istringstream is(blank.substr(p + 9, semi - p - 9));
        std::string type, name, eq;
        is >> type >> name >> eq;
        if ((type == "int" || type == "long" || type == "unsigned" || type == "int32_t") && eq == "=" &&
            !name.empty() && std::all_of(name.begin(), name.end(), isIdent)) {
            std::string expr;
            std::getline(is, expr, '\0');
            if (!m.count(name))
                m[name] = expr;
        }
        p = semi;
    }
}

bool spaceInt(const Value &v, int &out, const std::string &what, std::string &err) {
    if (v.kind == Value::Int) {
        out = (int)v.i;
        return true;
    }
    if (v.kind == Value::Call && v.items.size() == 1 && v.items[0].second.kind == Value::Int) {
        out = (int)v.items[0].second.i; // aie::Bytes{4096}
        return true;
    }
    err = what + " is not an integer constant (only #define and constexpr int names are evaluated)";
    return false;
}

bool spaceTileDim(const Value &v, TileDimOut &t, const std::string &who, std::string &err) {
    t = TileDimOut();
    t.used = true;
    t.has_stride = false;
    for (auto &kv : v.items) {
        const std::string what = who + "." + kv.first;
        if (kv.first == "slice_tiling") {
            for (auto &lv : kv.second.items) {
                int *dst = lv.first == "tile_size" ? &t.l2_size
                           : lv.first == "stride"  ? &t.l2_stride
                           : lv.first == "rounds"  ? &t.l2_rounds
                                                   : nullptr;
                if (!dst) {
                    err = "unknown field " + what + "." + (lv.first.empty() ? "<positional>" : lv.first);
                    return false;
                }
                if (!spaceInt(lv.second, *dst, what + "." + lv.first, err))
                    return false;
            }
            continue;
        }
        int *dst = kv.first == "fullsize"     ? &t.fullsize
                   : kv.first == "tile_round" ? &t.tile_round
                   : kv.first == "tile_size"  ? &t.tile_size
                   : kv.first == "stride"     ? &t.stride
                   : kv.first == "padsize"    ? &t.padsize
                                              : nullptr;
        if (!dst) {
            err = "unknown field " + (kv.first.empty() ? who + " <positional>" : what);
            return false;
        }
        if (!spaceInt(kv.second, *dst, what, err))
            return false;
        t.has_stride = t.has_stride || kv.first == "stride";
    }
    return true;
}

bool spacePolicy(const Value &v, PortSpace &s, std::string &err) {
    for (auto &grp : v.items) {
        for (auto &kv : grp.second.items) {
            const std::string what = ".policy." + grp.first + "." + kv.first;
            const std::string sym = kv.second.kind == Value::Sym ? lastComponent(kv.second.sym) : "";
            bool ok = true;
            if (grp.first == "map" && kv.first == "act")
                s.act = sym, ok = !sym.empty();
            else if (grp.first == "map" && kv.first == "wgt")
                s.wgt = sym, ok = !sym.empty();
            else if (grp.first == "map" && kv.first == "layout")
                s.layout = sym, ok = !sym.empty();
            else if (grp.first == "map" && kv.first == "merge_order")
                s.merge_order = sym, ok = !sym.empty();
            else if (grp.first == "map" && kv.first == "mesh_tiling_group1_dim")
                ok = spaceInt(kv.second, s.group1_dim, what, err);
            else if (grp.first == "map" && kv.first == "mesh_tiling_group2_dim")
                ok = spaceInt(kv.second, s.group2_dim, what, err);
            else if (grp.first == "mat" && kv.first == "pad")
                s.pad = sym == "Memtile", ok = sym == "Memtile" || sym == "DDR";
            else if (grp.first == "mat" && kv.first == "im2col")
                s.im2col = sym == "Dma", ok = sym == "Dma" || sym == "None";
            else if (grp.first == "sched" && kv.first == "pp_depth")
                ok = spaceInt(kv.second, s.pp_depth, what, err);
            else if (grp.first == "sched" && kv.first == "l1_budget")
                ok = spaceInt(kv.second, s.l1_budget, what, err);
            else {
                err = "unknown field " + what;
                return false;
            }
            if (!ok) {
                if (err.empty())
                    err = what + ": unexpected value";
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool sourceGemmSpace(const std::string &src, const std::string &srcDir, const std::string &name, PortSpace &out,
                     std::string &err) {
    std::string blank = blankComments(src);
    std::map<std::string, std::string> macros;
    collectMacros(src, srcDir, macros, 0);
    addConstexprInts(blank, macros);
    std::vector<Decl> decls;
    std::string perr;
    // Parse only `name`: another GemmSpace of the file may use what this
    // checker does not evaluate, and is no concern of this kernel.
    size_t p = 0;
    const Decl *d = nullptr;
    while ((p = findToken(blank, "aie::GemmSpace", p)) != std::string::npos) {
        size_t q = p + 14;
        while (q < blank.size() && std::isspace((unsigned char)blank[q]))
            q++;
        size_t nb = q;
        while (q < blank.size() && isIdent(blank[q]))
            q++;
        if (blank.substr(nb, q - nb) == name) {
            size_t lineStart = blank.rfind('\n', p);
            lineStart = lineStart == std::string::npos ? 0 : lineStart;
            size_t ob = blank.find('{', q);
            size_t cb = ob == std::string::npos ? ob : matchClose(blank, ob, '{', '}');
            size_t semi = cb == std::string::npos ? cb : blank.find(';', cb);
            if (semi == std::string::npos) {
                err = "unterminated aie::GemmSpace " + name;
                return false;
            }
            std::string one = blank.substr(lineStart, semi + 1 - lineStart);
            if (!findDecls(one, one, "GemmSpace", macros, decls, perr)) {
                err = perr;
                return false;
            }
            for (auto &x : decls)
                if (x.name == name)
                    d = &x;
            break;
        }
        p = q;
    }
    if (!d) {
        err = "no `constexpr aie::GemmSpace " + name + " = {...};` in the source (headers are not searched)";
        return false;
    }
    PortSpace s;
    s.act = s.wgt = "Broadcast";
    s.layout = "Row";
    s.merge_order = "Default";
    s.im2col = 0;
    s.l1_budget = 32 * 1024;
    for (auto &kv : d->init.items) {
        const std::string what = "aie::GemmSpace " + name + ": ";
        if (kv.first == "policy") {
            if (!spacePolicy(kv.second, s, err)) {
                err = what + err;
                return false;
            }
        } else if (kv.first.size() == 2 && kv.first[0] == 'd' && kv.first[1] >= '1' && kv.first[1] <= '4') {
            if (!spaceTileDim(kv.second, s.d[kv.first[1] - '1'], "." + kv.first, err)) {
                err = what + err;
                return false;
            }
        } else if (kv.first == "m" || kv.first == "n" || kv.first == "k") {
            err = what + "uses ." + kv.first +
                  " (the GEMM form); a block-load kernel's GemmSpace describes its "
                  "tensor with .d1 ... .d4";
            return false;
        } else {
            err = what + "unknown field ." + (kv.first.empty() ? "<positional>" : kv.first);
            return false;
        }
    }
    out = s;
    return true;
}

DesugarResult parseConvMapping(const std::string &src, const std::string &srcDir) {
    DesugarResult r;
    Parsed P;
    parseAll(src, srcDir, P, r);
    return r;
}

DesugarResult desugarConvMapping(std::string &src, const std::string &srcDir, const Overrides &ov, const Target &tgt,
                                 const std::string &srcName) {
    DesugarResult r;
    Parsed P;
    if (!parseAll(src, srcDir, P, r) || !r.ok)
        return r;
    if (r.kernels.empty()) {
        r.ok = false;
        r.error = "aie::Mapping declared but no kernel uses __global__(<conv>, <mapping>)";
        return r;
    }
    // Derive per kernel.
    for (auto &k : r.kernels) {
        k.derived = derive(k.conv, k.mapping, tgt, ov);
        if (!k.derived.ok) {
            r.ok = false;
            r.error = k.kernel + ": " + (k.derived.errors.empty() ? "derivation failed" : k.derived.errors[0]);
            return r;
        }
    }
    // Block-load kernels: their load_buffer / store_buffer must be exactly
    // what the derived DMA delivers. Kernels inferred by inferKernelOnly were
    // already traced against this derivation.
    for (auto &k : r.kernels) {
        if (k.convVar.compare(0, 11, "__aie_conv_") == 0)
            continue;
        std::vector<std::string> errs = verifyKernelBlocks(src, srcDir, srcName, k, r.warnings);
        if (!errs.empty()) {
            r.ok = false;
            r.error = k.kernel + ": the kernel's blocks do not match the DMA derived from " + k.convVar + " + " +
                      k.mapVar + ":";
            for (auto &e : errs)
                r.error += "\n  " + e;
            return r;
        }
    }
    // Rewrite back-to-front so earlier offsets stay valid.
    struct Edit {
        size_t begin, end;
        std::string text;
    };
    std::vector<Edit> edits;
    for (size_t i = 0; i < P.uses.size(); i++) {
        const GlobalUse &g = P.uses[i];
        const KernelDesugar &k = r.kernels[i];
        edits.push_back({g.kwBegin, g.argsEnd, "__global__(" + spaceVar(k.kernel, "policy") + ")"});
        // Role tags inside this kernel's parameter list.
        std::string params = src.substr(g.paramOpen, g.paramClose - g.paramOpen + 1);
        std::string blankParams = blankComments(params);
        for (const char *role : {"IFM", "WEIGHTS", "OFM"}) {
            std::string tag = std::string("aie::") + role;
            size_t p = 0;
            while ((p = findToken(blankParams, tag, p)) != std::string::npos) {
                edits.push_back({g.paramOpen + p, g.paramOpen + p + tag.size(), spaceVar(k.kernel, role)});
                // The port variable follows the closing '>' of aie::port<...>.
                size_t close = blankParams.find('>', p);
                size_t v = close == std::string::npos ? close : blankParams.find_first_not_of(" \t\r\n", close + 1);
                size_t ve = v;
                while (ve != std::string::npos && ve < blankParams.size() && isIdent(blankParams[ve]))
                    ve++;
                if (v != std::string::npos && ve > v)
                    r.kernels[i].ports.emplace_back(blankParams.substr(v, ve - v), role);
                p += tag.size();
            }
        }
    }
    // Generated block goes where the first used Mapping declaration was; the
    // Conv2d / Mapping declarations themselves are commented out.
    std::string block;
    for (auto &k : r.kernels) {
        auto skip = userDefinedKernelNames(k.derived, P.macros, r.warnings);
        block += generatedBlock(k, skip);
    }
    bool placed = false;
    std::vector<const Decl *> decls;
    for (auto &d : P.convs)
        decls.push_back(&d);
    for (auto &d : P.maps)
        decls.push_back(&d);
    for (auto *d : decls) {
        bool used = false;
        for (auto &k : r.kernels)
            used |= (d->type == "Conv2d" && d->name == k.convVar) || (d->type == "Mapping" && d->name == k.mapVar);
        if (!used)
            continue;
        std::string text = commentOut(src.substr(d->begin, d->end - d->begin));
        if (d->type == "Mapping" && !placed) {
            text += block;
            placed = true;
        }
        edits.push_back({d->begin, d->end, text});
    }
    std::sort(edits.begin(), edits.end(), [](const Edit &a, const Edit &b) { return a.begin > b.begin; });
    for (auto &e : edits)
        src.replace(e.begin, e.end - e.begin, e.text);
    return r;
}

// ---------------------------------------------------------------------------
// Text helpers shared with kernel_trace.cpp
// ---------------------------------------------------------------------------
std::string sourceBlankComments(const std::string &s) { return blankComments(s); }
size_t sourceMatchClose(const std::string &s, size_t open, char o, char c) { return matchClose(s, open, o, c); }
size_t sourceFindToken(const std::string &s, const std::string &word, size_t from) { return findToken(s, word, from); }
std::map<std::string, std::string> sourceMacros(const std::string &src, const std::string &srcDir) {
    std::map<std::string, std::string> m;
    collectMacros(src, srcDir, m, 0);
    return m;
}
bool sourceEvalInt(const std::string &expr, const std::map<std::string, std::string> &macros, long &v) {
    Parser ps(expr, macros);
    return ps.parseExprOnly(v);
}
std::pair<int, int> sourceLaunchMesh(const std::string &src, const std::string &srcDir) {
    MeshShape ms = findMeshShape(blankComments(src), sourceMacros(src, srcDir));
    return {ms.rows, ms.cols};
}

std::vector<std::string> kernelTUDecls(const KernelDesugar &k) {
    const Derived &d = k.derived;
    const Conv2dSpec &c = k.conv;
    auto list = [](std::initializer_list<long> v) {
        std::string s = "{";
        for (long x : v)
            s += (s.size() > 1 ? ", " : "") + std::to_string(x);
        return s + "}";
    };
    std::vector<std::string> out;
    out.push_back("#include \"aie_block_io.h\"");
    out.push_back("static constexpr aiehlc_conv2d_t " + k.convVar + " = " +
                  list({c.h, c.w, c.c, c.k, c.kh, c.kw, c.stride, c.pad}) + "; // __global__ op");
    out.push_back("static constexpr aiehlc_kernel_view_t " + k.mapVar + " = " +
                  list({d.oh, d.ow, d.oh_mesh, d.oc_mesh, d.oh_t, d.ow_t, d.tile_h, d.tile_w, d.c_align}) +
                  "; // __global__ mapping block sizes");
    // Per port: the block it delivers in round (i, j) on core (p0, p1), in
    // element coordinates of its whole tensor -- the same Derived numbers the
    // port's GemmSpace (and so its DMA) is built from.
    for (const auto &p : k.ports) {
        std::string desc;
        if (p.second == "IFM") // [h+2p, w+2p, c_align]: overlapping halo slabs
            desc = "{3, " + list({d.tile_h, d.tile_w, d.c_align, 0}) + ", " + list({d.mesh_step_h, 0, 0, 0}) + ", " +
                   list({0, 0, 0, 0}) + ", " + list({d.tile_step_h, 0, 0, 0}) + ", " + list({0, d.tile_step_w, 0, 0}) +
                   ", " + std::to_string(d.h_rounds) + ", " + std::to_string(d.w_rounds) + "}";
        else if (p.second == "OFM") // [oh, ow, k]
            desc = "{3, " + list({d.oh_t, d.ow_t, d.oc_mesh, 0}) + ", " + list({d.oh_mesh, 0, 0, 0}) + ", " +
                   list({0, 0, d.oc_mesh, 0}) + ", " + list({d.oh_t, 0, 0, 0}) + ", " + list({0, d.ow_t, 0, 0}) + ", " +
                   std::to_string(d.h_rounds) + ", " + std::to_string(d.w_rounds) + "}";
        else // WEIGHTS: B^T [k, kh, kw, c_align], this core's OC slice, sent once
            desc = "{4, " + list({d.oc_mesh, c.kh, c.kw, d.c_align}) + ", " + list({0, 0, 0, 0}) + ", " +
                   list({d.oc_mesh, 0, 0, 0}) + ", " + list({0, 0, 0, 0}) + ", " + list({0, 0, 0, 0}) + ", 1, 1}";
        out.push_back("static const aiehlc_blk_desc aiehlc_desc_" + p.first + " = " + desc + ";");
        out.push_back("static aiehlc_blk_cursor aiehlc_cur_" + p.first + ";");
    }
    return out;
}

std::string reportJson(const DesugarResult &r) {
    std::ostringstream os;
    auto str = [](const std::string &s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '\n')
                o += "\\n"; // multi-line messages (kernel analysis) stay valid JSON
            else if (c == '\t')
                o += "\\t";
            else if ((unsigned char)c < 0x20)
                o += ' ';
            else {
                if (c == '"' || c == '\\')
                    o += '\\';
                o += c;
            }
        }
        return o + "\"";
    };
    os << "{\n  \"ok\": " << (r.ok ? "true" : "false") << ",\n  \"error\": " << str(r.error) << ",\n  \"warnings\": [";
    for (size_t i = 0; i < r.warnings.size(); i++)
        os << (i ? ", " : "") << str(r.warnings[i]);
    os << "],\n  \"kernels\": [";
    for (size_t i = 0; i < r.kernels.size(); i++) {
        const auto &k = r.kernels[i];
        os << (i ? "," : "") << "\n    {\"kernel\": " << str(k.kernel) << ", \"conv\": " << toJson(k.conv)
           << ",\n     \"mapping\": {\"mesh_rows\": " << k.mapping.space_rows.factor
           << ", \"mesh_cols\": " << k.mapping.space_cols.factor << ", \"pp_depth\": " << k.mapping.hints.pp_depth
           << ", \"l1_budget\": " << k.mapping.hints.l1_budget << "},\n     \"derived\": " << toJson(k.derived, "     ")
           << "}";
    }
    os << "]\n}\n";
    return os.str();
}

} // namespace convautotile
