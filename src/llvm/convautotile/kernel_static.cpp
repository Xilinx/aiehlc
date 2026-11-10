/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Static block-bounds analysis -- see kernel_static.h.
 ******************************************************************************/
#include "kernel_static.h"
#include "conv_desugar.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace convautotile {

namespace {

// ---------------------------------------------------------------------------
// Tokens and expressions
// ---------------------------------------------------------------------------
struct Tok {
    std::string s;
    size_t off = 0;
    char kind = 'p'; // i ident (a::b joined), n integer, f other number, s string/char, p punctuation
    long num = 0;
};

std::vector<Tok> tokenize(const std::string &text, size_t b, size_t e) {
    static const char *ops3[] = {"<<=", ">>="};
    static const char *ops2[] = {"::", "->", "++", "--", "<=", ">=", "==", "!=", "&&", "||",
                                 "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="};
    auto identAt = [&](size_t p) { return p < e && (std::isalpha((unsigned char)text[p]) || text[p] == '_'); };
    std::vector<Tok> out;
    for (size_t i = b; i < e;) {
        const char c = text[i];
        if (std::isspace((unsigned char)c)) {
            i++;
            continue;
        }
        Tok t;
        t.off = i;
        if (identAt(i)) {
            size_t j = i;
            for (;;) {
                while (j < e && (std::isalnum((unsigned char)text[j]) || text[j] == '_'))
                    j++;
                if (j + 2 < e && text[j] == ':' && text[j + 1] == ':' && identAt(j + 2)) {
                    j += 2;
                    continue;
                }
                break;
            }
            t.kind = 'i';
            t.s = text.substr(i, j - i);
            i = j;
        } else if (std::isdigit((unsigned char)c)) {
            size_t j = i;
            while (j < e && (std::isalnum((unsigned char)text[j]) || text[j] == '.' || text[j] == '\''))
                j++;
            t.s = text.substr(i, j - i);
            i = j;
            std::string d;
            for (char ch : t.s)
                if (ch != '\'')
                    d += ch;
            while (!d.empty() && std::strchr("uUlL", d.back()))
                d.pop_back();
            char *end = nullptr;
            t.num = std::strtol(d.c_str(), &end, 0);
            t.kind = !d.empty() && end && *end == 0 ? 'n' : 'f';
        } else if (c == '"' || c == '\'') {
            size_t j = i + 1;
            while (j < e && text[j] != c)
                j += text[j] == '\\' ? 2 : 1;
            j = std::min(j + 1, e);
            t.kind = 's';
            t.s = text.substr(i, j - i);
            i = j;
        } else {
            t.s = std::string(1, c);
            for (auto *o : ops3)
                if (text.compare(i, 3, o) == 0)
                    t.s = o;
            if (t.s.size() == 1)
                for (auto *o : ops2)
                    if (text.compare(i, 2, o) == 0)
                        t.s = o;
            i += t.s.size();
        }
        out.push_back(t);
    }
    return out;
}

bool isScalarType(const std::string &s) {
    static const std::set<std::string> w = {"int",     "unsigned", "signed",   "long",     "short",      "char",
                                            "bool",    "size_t",   "int8_t",   "int16_t",  "int32_t",    "int64_t",
                                            "uint8_t", "uint16_t", "uint32_t", "uint64_t", "std::size_t"};
    return w.count(s) != 0;
}
bool isTypeName(const std::string &s) {
    return isScalarType(s) || (s.size() > 2 && s.compare(s.size() - 2, 2, "_t") == 0);
}
bool isAssignOp(const std::string &s) {
    static const std::set<std::string> w = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="};
    return w.count(s) != 0;
}
int precOf(const std::string &s) {
    static const std::map<std::string, int> p = {{"||", 1}, {"&&", 2}, {"|", 3},  {"^", 4},  {"&", 5},  {"==", 6},
                                                 {"!=", 6}, {"<", 7},  {"<=", 7}, {">", 7},  {">=", 7}, {"<<", 8},
                                                 {">>", 8}, {"+", 9},  {"-", 9},  {"*", 10}, {"/", 10}, {"%", 10}};
    auto it = p.find(s);
    return it == p.end() ? 0 : it->second;
}
bool cmpOp(const std::string &op, long a, long b) {
    return op == "<" ? a < b : op == "<=" ? a <= b : op == ">" ? a > b : op == ">=" ? a >= b : a != b;
}

// Expression tree. `x` = something the analysis does not model (a call, a
// float, a member access): evaluating it fails and the access is skipped.
struct Node {
    char op =
        'x'; // n number, v name, i name[index], p program_id(val), q num_programs(val), u unary, b binary, ? ternary
    long val = 0;
    std::string name; // variable / operator
    std::vector<Node> kids;
};

class ExprParser {
  public:
    // Parse t[b, e); the token at `zero` (a base pointer) reads as 0.
    ExprParser(const std::vector<Tok> &t, long b, long e, long zero = -1) : t_(t), i_(b), e_(e), zero_(zero) {}
    Node parse() {
        Node n = ternary();
        return bad_ || i_ != e_ ? Node() : n;
    }

  private:
    const std::vector<Tok> &t_;
    long i_, e_, zero_;
    bool bad_ = false;

    bool at(const char *s) const { return i_ < e_ && t_[i_].kind != 's' && t_[i_].s == s; }
    void expect(const char *s) {
        if (at(s))
            i_++;
        else
            bad_ = true;
    }
    Node ternary() {
        Node c = binary(1);
        if (!at("?"))
            return c;
        i_++;
        Node n;
        n.op = '?';
        n.kids.push_back(c);
        n.kids.push_back(ternary());
        expect(":");
        n.kids.push_back(ternary());
        return n;
    }
    Node binary(int minPrec) {
        Node l = unary();
        while (i_ < e_ && t_[i_].kind == 'p' && precOf(t_[i_].s) >= minPrec) {
            Node n;
            n.op = 'b';
            n.name = t_[i_].s;
            int p = precOf(t_[i_++].s);
            n.kids.push_back(l);
            n.kids.push_back(binary(p + 1));
            l = n;
        }
        return l;
    }
    long skipGroup(long open, const char *o, const char *c) const {
        int d = 0;
        for (long j = open; j < e_; j++) {
            if (t_[j].s == o)
                d++;
            else if (t_[j].s == c && --d == 0)
                return j;
        }
        return e_;
    }
    Node unary();
};

Node ExprParser::unary() {
    Node n;
    if (i_ >= e_) {
        bad_ = true;
        return n;
    }
    const Tok &t = t_[i_];
    if (i_ == zero_) {
        i_++;
        n.op = 'n';
        return n;
    }
    if (t.kind == 'p' && (t.s == "-" || t.s == "+" || t.s == "!" || t.s == "~")) {
        i_++;
        n.op = 'u';
        n.name = t.s;
        n.kids.push_back(unary());
        return n;
    }
    if (t.s == "(") {
        if (i_ + 2 < e_ && t_[i_ + 1].kind == 'i' && isTypeName(t_[i_ + 1].s) && t_[i_ + 2].s == ")") {
            i_ += 3; // (int)x
            return unary();
        }
        i_++;
        n = ternary();
        expect(")");
        return n;
    }
    if (t.kind == 'n') {
        i_++;
        n.op = 'n';
        n.val = t.num;
        return n;
    }
    if (t.kind != 'i') {
        bad_ = true;
        return n;
    }
    i_++;
    if (t.s == "true" || t.s == "false") {
        n.op = 'n';
        n.val = t.s == "true";
    } else if (t.s == "static_cast" && at("<")) {
        i_ = skipGroup(i_, "<", ">") + 1;
        expect("(");
        n = ternary();
        expect(")");
    } else if ((t.s == "aie::program_id" || t.s == "aie::num_programs") && at("(") && i_ + 2 < e_ &&
               t_[i_ + 1].kind == 'n' && t_[i_ + 2].s == ")") {
        n.op = t.s == "aie::program_id" ? 'p' : 'q';
        n.val = t_[i_ + 1].num;
        i_ += 3;
    } else if ((t.s == "std::min" || t.s == "std::max") && at("(")) {
        i_++;
        n.op = 'b';
        n.name = t.s == "std::min" ? "min" : "max";
        n.kids.push_back(ternary());
        expect(",");
        n.kids.push_back(ternary());
        expect(")");
    } else if (at("(")) {
        i_ = skipGroup(i_, "(", ")") + 1; // an unmodelled call
        n.op = 'x';
    } else if (at("[")) {
        i_++;
        n.op = 'i';
        n.name = t.s;
        n.kids.push_back(ternary());
        expect("]");
    } else {
        n.op = 'v';
        n.name = t.s;
    }
    return n;
}

bool binop(const std::string &op, long a, long b, long &out) {
    if ((op == "/" || op == "%") && b == 0)
        return false;
    if ((op == "<<" || op == ">>") && (b < 0 || b > 62))
        return false;
    if (op == "+")
        out = a + b;
    else if (op == "-")
        out = a - b;
    else if (op == "*")
        out = a * b;
    else if (op == "/")
        out = a / b;
    else if (op == "%")
        out = a % b;
    else if (op == "<<")
        out = a << b;
    else if (op == ">>")
        out = a >> b;
    else if (op == "&")
        out = a & b;
    else if (op == "|")
        out = a | b;
    else if (op == "^")
        out = a ^ b;
    else if (op == "min")
        out = std::min(a, b);
    else if (op == "max")
        out = std::max(a, b);
    else if (op == "==" || op == "!=" || op == "<" || op == "<=" || op == ">" || op == ">=")
        out = op == "==" ? a == b : cmpOp(op, a, b);
    else
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// The kernel model
// ---------------------------------------------------------------------------
struct Def { // a named constant / local, or a small const array
    std::string name, text;
    long pos = 0, end = 0; // name token; last token of its scope (macros: -1 / LONG_MAX)
    int line = 0;
    Node expr;
    std::vector<Node> elems;
    bool array = false, mut = false, root = false;
    long value = 0; // root: the literal
};
struct Block { // what a load_buffer / store_buffer returns, or a local array
    std::string var, port, role, text;
    std::vector<Node> shape;
    long pos = 0;
    int line = 0;
    bool local = false;
};
struct Ptr { // p = block, or p = base +/- off
    std::string name;
    long pos = 0, end = 0;
    int block = -1, base = -1;
    Node off;
    bool hasOff = false, mut = false;
};
struct Loop {
    std::string var, cmp = "<", cond;
    long hdr = 0, bodyB = 0, bodyE = 0, hi = 0;
    int line = 0;
    Node init, bound, step;
    bool ok = false, pseudo = false; // pseudo: program_id(a) over [0, hi)
};
struct Guard {
    Node cond;
    long b = 0, e = 0, pos = 0;
    bool neg = false;
};
struct Access {
    int ptr = -1;
    long pos = 0, span = 1;
    int line = 0;
    bool write = false;
    Node idx;
    std::string text;
    std::vector<Guard> guards; // ?: guards of the statement
};
struct Env {
    std::map<std::string, long> v;            // enumerated loop variables ($p0 / $p1 = program ids)
    const std::map<int, long> *ovr = nullptr; // def -> forced value (the cause search)
};
struct AccessEval {
    bool ok = false, out = false;
    long lo = LONG_MAX, worst = LONG_MIN; // lowest index; largest (end - size)
    long hiEnd = 0, size = 0;             // at the worst leaf
    Env wLo, wHi;
};

class Analyzer {
  public:
    Analyzer(const std::string &src, const TraceKernel &k, int rows, int cols, const std::string &srcName)
        : src_(src), k_(k), rows_(rows), cols_(cols), srcName_(srcName) {
        for (int d = 0; d < 2; d++) {
            pid_[d].var = d ? "$p1" : "$p0";
            pid_[d].pseudo = pid_[d].ok = true;
        }
    }
    void build(const std::string &prelude);
    StaticBoundsResult run();

  private:
    const std::string &src_;
    const TraceKernel &k_;
    int rows_, cols_;
    std::string srcName_;
    std::vector<Tok> t_;
    std::vector<long> match_;
    std::set<long> declNames_;
    std::vector<Def> defs_;
    std::multimap<std::string, int> defByName_;
    std::vector<Block> blocks_;
    std::vector<Ptr> ptrs_;
    std::vector<Loop> loops_;
    std::vector<Guard> ifs_;
    std::vector<Access> acc_;
    std::map<std::string, long> vecLen_;
    Loop pid_[2]; // program_id(0 / 1) as loops over the mesh

    long n() const { return (long)t_.size(); }
    const std::string &s(long i) const {
        static const std::string none;
        return i >= 0 && i < n() ? t_[i].s : none;
    }
    int lineOf(long tok) const {
        size_t off = tok >= 0 && tok < n() ? t_[tok].off : 0;
        return 1 + (int)std::count(src_.begin(), src_.begin() + off, '\n');
    }
    std::string textOf(long b, long e) const;
    Node parse(long b, long e, long zero = -1) const { return ExprParser(t_, b, e, zero).parse(); }
    long scanTo(long i, const char *stop) const;
    long stmtEnd(long i) const;
    long skipChess(long b) const;
    long scopeEnd(long tok) const;
    std::vector<std::pair<long, long>> splitArgs(long open) const;

    void parseMacros(const std::string &prelude);
    void scanDecls();
    void addDeclarators(long name, long end);
    void addPtrDecl(long name);
    void addArrayDecl(long typeTok, long name);
    void scanControl();
    void scanAccesses();
    void addAccess(long a, long b, long e, long span, bool write, long textEnd);
    std::vector<Guard> ternaryGuards(long a) const;
    void markMutations();
    void markRoots();

    int findDef(const std::string &name, long pos) const;
    int findPtr(const std::string &name, long pos) const;
    bool lookup(const std::string &name, long pos, const Env &env, long &out, int depth) const;
    bool eval(const Node &nd, long pos, const Env &env, long &out, int depth = 0) const;
    void collect(const Node &nd, long pos, const std::set<std::string> &loopVars, std::set<std::string> &vars,
                 std::set<int> &seen, int depth = 0) const;

    std::vector<std::pair<const Node *, long>> terms(const Access &a, int &block) const;
    bool plan(const Access &a, std::vector<const Loop *> &its, std::set<int> &defsUsed) const;
    int walk(const std::vector<const Loop *> &its, size_t i, Env &env, const std::function<int(Env &)> &leaf,
             long &budget) const;
    bool flatAt(const Access &a, const Env &env, long &idx, long &size, std::vector<long> *shape) const;
    bool check(const Access &a, const std::map<int, long> *ovr, bool stopEarly, AccessEval &r) const;
    std::set<int> decoupledRoots(const Access &a) const;
    std::string findCause(const std::vector<int> &bad, const std::vector<int> &good, int &causeDef) const;
    std::string describe(const Access &a, const AccessEval &r, int causeDef) const;
    std::string witnessText(const Access &a, const Env &w) const;
};

std::string Analyzer::textOf(long b, long e) const {
    if (b >= e || e > n())
        return "";
    std::string raw = src_.substr(t_[b].off, t_[e - 1].off + t_[e - 1].s.size() - t_[b].off), out;
    for (char c : raw)
        if (!std::isspace((unsigned char)c))
            out += c;
        else if (!out.empty() && out.back() != ' ')
            out += ' ';
    return out;
}

// First token from i at nesting depth 0 that is `stop`, or the closer of the
// enclosing group.
long Analyzer::scanTo(long i, const char *stop) const {
    for (long j = i; j < n(); j++) {
        const std::string &c = t_[j].s;
        if (t_[j].kind != 's' && c == stop)
            return j;
        if (t_[j].kind == 'p' && (c == "(" || c == "[" || c == "{") && match_[j] > j)
            j = match_[j];
        else if (t_[j].kind == 'p' && (c == ")" || c == "]" || c == "}"))
            return j;
    }
    return n() - 1;
}

long Analyzer::skipChess(long b) const {
    while (b < n() && t_[b].kind == 'i' && t_[b].s.compare(0, 6, "chess_") == 0)
        b = s(b + 1) == "(" && match_[b + 1] > b ? match_[b + 1] + 1 : b + 1;
    return b;
}

// Last token of the statement starting at i.
long Analyzer::stmtEnd(long i) const {
    if (i >= n())
        return n() - 1;
    const std::string &c = t_[i].s;
    if (c == "{")
        return match_[i] > i ? match_[i] : n() - 1;
    if ((c == "for" || c == "while" || c == "if" || c == "switch") && s(i + 1) == "(" && match_[i + 1] > i) {
        long e = stmtEnd(skipChess(match_[i + 1] + 1));
        if (c == "if" && s(e + 1) == "else")
            e = stmtEnd(e + 2);
        return e;
    }
    if (c == "else")
        return stmtEnd(i + 1);
    if (c == "do")
        return scanTo(stmtEnd(i + 1) + 1, ";");
    return scanTo(i, ";");
}

// The closing brace of the innermost block enclosing `tok`.
long Analyzer::scopeEnd(long tok) const {
    for (long j = tok; j >= 0; j--)
        if (t_[j].s == "{" && t_[j].kind == 'p' && match_[j] > tok)
            return match_[j];
    return n() - 1;
}

// Top-level comma-separated ranges inside the group opened at `open`.
std::vector<std::pair<long, long>> Analyzer::splitArgs(long open) const {
    std::vector<std::pair<long, long>> out;
    if (open < 0 || open >= n() || match_[open] <= open)
        return out;
    const long close = match_[open];
    for (long b = open + 1; b < close;) {
        long e = std::min(scanTo(b, ","), close);
        out.push_back({b, e});
        b = e + 1;
    }
    return out;
}

void Analyzer::build(const std::string &prelude) {
    std::string blank = sourceBlankComments(src_);
    t_ = tokenize(blank, k_.bodyOpen, k_.bodyClose + 1);
    match_.assign(n(), -1);
    std::vector<long> st;
    for (long i = 0; i < n(); i++) {
        if (t_[i].kind != 'p')
            continue;
        const std::string &c = t_[i].s;
        if (c == "(" || c == "[" || c == "{")
            st.push_back(i);
        else if ((c == ")" || c == "]" || c == "}") && !st.empty()) {
            match_[i] = st.back();
            match_[st.back()] = i;
            st.pop_back();
        }
    }
    parseMacros(prelude);
    scanDecls();
    scanControl();
    scanAccesses();
    markMutations();
    markRoots();
}

void Analyzer::parseMacros(const std::string &prelude) {
    std::istringstream is(prelude);
    std::string line;
    while (std::getline(is, line)) {
        std::vector<Tok> mt = tokenize(line, 0, line.size());
        // # define NAME value...   (object-like only)
        if (mt.size() < 4 || mt[0].s != "#" || mt[1].s != "define" || mt[2].kind != 'i' ||
            (mt[3].s == "(" && mt[3].off == mt[2].off + mt[2].s.size()))
            continue;
        Def d;
        d.name = mt[2].s;
        d.pos = -1;
        d.end = LONG_MAX;
        d.expr = ExprParser(mt, 3, (long)mt.size()).parse();
        d.text = "#define " + line.substr(mt[2].off);
        defByName_.insert({d.name, (int)defs_.size()});
        defs_.push_back(d);
    }
}

// Declarations at statement starts: scalars (`const int a = 1, b = 2;`),
// pointers (`T *p = ...`), arrays (`T a[N] (= {...})`), aie::vector lengths.
void Analyzer::scanDecls() {
    for (long i = 1; i + 1 < n(); i++) {
        const std::string &pv = t_[i - 1].s;
        if (!(pv == ";" || pv == "{" || pv == "}" || pv == ")"))
            continue;
        long j = i;
        while (j < n() && (s(j) == "const" || s(j) == "constexpr" || s(j) == "static" || s(j) == "volatile" ||
                           s(j) == "alignas")) {
            if (s(j) == "alignas" && s(j + 1) == "(" && match_[j + 1] > j)
                j = match_[j + 1] + 1;
            else
                j++;
        }
        if (j >= n() || t_[j].kind != 'i')
            continue;
        const long typeTok = j;
        const std::string type = t_[j++].s;
        while (isScalarType(type) && j < n() && t_[j].kind == 'i' && isScalarType(t_[j].s))
            j++; // unsigned int, long long
        long vlen = 0;
        if (s(j) == "<") { // template arguments
            long d = 0, q = j;
            for (; q < n() && s(q) != ";" && s(q) != "{"; q++) {
                if (s(q) == "<")
                    d++;
                else if (s(q) == ">" && --d == 0)
                    break;
            }
            if (s(q) != ">")
                continue;
            if (type == "aie::vector" && t_[q - 1].kind == 'n')
                vlen = t_[q - 1].num;
            j = q + 1;
        }
        bool ptr = false;
        while (s(j) == "const" || s(j) == "*") {
            ptr |= s(j) == "*";
            j++;
        }
        if (j + 1 >= n() || t_[j].kind != 'i')
            continue;
        if (vlen)
            vecLen_[t_[j].s] = vlen;
        if ((ptr || type == "auto") && s(j + 1) == "=")
            addPtrDecl(j);
        else if (!ptr && s(j + 1) == "[")
            addArrayDecl(typeTok, j);
        else if (!ptr && isScalarType(type) && (s(j + 1) == "=" || s(j + 1) == "," || s(j + 1) == ";"))
            addDeclarators(j, scanTo(j, ";"));
    }
}

// `a = e1, b, c = e2` up to `end`.
void Analyzer::addDeclarators(long name, long end) {
    for (long j = name; j < end && t_[j].kind == 'i';) {
        Def d;
        d.name = t_[j].s;
        d.pos = j;
        d.end = scopeEnd(j);
        d.line = lineOf(j);
        declNames_.insert(j);
        long e = std::min(scanTo(j, ","), end);
        if (s(j + 1) == "=") {
            d.expr = parse(j + 2, e);
            d.text = textOf(j, e);
        } else
            d.mut = true; // uninitialized
        defByName_.insert({d.name, (int)defs_.size()});
        defs_.push_back(d);
        j = e + 1;
    }
}

void Analyzer::addPtrDecl(long name) {
    const long rhs = name + 2, end = scanTo(rhs, ";");
    Ptr p;
    p.name = t_[name].s;
    p.pos = name;
    p.end = scopeEnd(name);
    declNames_.insert(name);
    const std::string &f = s(rhs);
    if ((f == "aie::load_buffer" || f == "aie::store_buffer") && s(rhs + 1) == "(") {
        auto args = splitArgs(rhs + 1);
        if (args.size() < 3 || s(args[2].first) != "{")
            return;
        Block b;
        b.var = p.name;
        b.port = s(args[0].first);
        for (auto &port : k_.ports)
            if (port.var == b.port)
                b.role = port.role;
        for (auto &a : splitArgs(args[2].first))
            b.shape.push_back(parse(a.first, a.second));
        b.pos = name;
        b.line = lineOf(name);
        b.text = textOf(rhs, match_[rhs + 1] + 1);
        p.block = (int)blocks_.size();
        blocks_.push_back(b);
    } else if (t_[rhs].kind == 'i' && (p.base = findPtr(f, name)) >= 0) {
        if (end > rhs + 1 && s(rhs + 1) != "+" && s(rhs + 1) != "-")
            return;
        p.hasOff = end > rhs + 1;
        if (p.hasOff)
            p.off = parse(rhs, end, rhs);
    } else
        return;
    ptrs_.push_back(p);
}

// `T a[N];` -- a local array, bounds-checked like a block -- and
// `T a[N] = {e0, e1, ...};` -- also a const array the index math may read.
void Analyzer::addArrayDecl(long typeTok, long name) {
    const long close = match_[name + 1];
    if (close <= name || s(close + 1) == "[")
        return; // multi-dimensional: not modelled
    declNames_.insert(name);
    Block b;
    b.var = t_[name].s;
    b.local = true;
    b.shape.push_back(parse(name + 2, close));
    b.pos = name;
    b.line = lineOf(name);
    b.text = textOf(typeTok, close + 1);
    Ptr p;
    p.name = b.var;
    p.pos = name;
    p.end = scopeEnd(name);
    p.block = (int)blocks_.size();
    blocks_.push_back(b);
    ptrs_.push_back(p);
    if (s(close + 1) == "=" && s(close + 2) == "{") {
        Def d;
        d.name = b.var;
        d.pos = name;
        d.end = p.end;
        d.line = b.line;
        d.array = true;
        for (auto &a : splitArgs(close + 2))
            d.elems.push_back(parse(a.first, a.second));
        defByName_.insert({d.name, (int)defs_.size()});
        defs_.push_back(d);
    }
}

// `for (int v = a; v < b; v++)` loops and `if (c) ... else ...` guards.
void Analyzer::scanControl() {
    for (long i = 0; i + 1 < n(); i++) {
        if (t_[i].kind != 'i' || s(i + 1) != "(" || match_[i + 1] <= i)
            continue;
        const long close = match_[i + 1];
        if (t_[i].s == "if") {
            Guard g;
            g.cond = parse(i + 2, close);
            g.pos = i;
            g.b = close + 1;
            g.e = stmtEnd(g.b);
            ifs_.push_back(g);
            if (s(g.e + 1) == "else") {
                g.neg = true;
                g.b = g.e + 2;
                g.e = stmtEnd(g.b);
                ifs_.push_back(g);
            }
            continue;
        }
        if (t_[i].s != "for")
            continue;
        Loop L;
        L.hdr = i;
        L.line = lineOf(i);
        L.bodyB = skipChess(close + 1);
        L.bodyE = stmtEnd(L.bodyB);
        const long s1 = scanTo(i + 2, ";"), s2 = s1 < close ? scanTo(s1 + 1, ";") : close;
        long eq = i + 2;
        while (eq < s1 && s(eq) != "=")
            eq++;
        if (eq < s1 && eq > i + 2 && s2 < close) {
            L.var = s(eq - 1);
            L.init = parse(eq + 1, s1);
            L.cond = textOf(s1 + 1, s2);
            const std::string &op = s(s1 + 2);
            if (s(s1 + 1) == L.var && (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "!=")) {
                L.cmp = op;
                L.bound = parse(s1 + 3, s2);
                Node one;
                one.op = 'n';
                one.val = 1;
                const long c = s2 + 1;
                if ((s(c) == L.var && s(c + 1) == "++" && c + 2 == close) || (s(c) == "++" && s(c + 1) == L.var))
                    L.step = one;
                else if ((s(c) == L.var && s(c + 1) == "--" && c + 2 == close) || (s(c) == "--" && s(c + 1) == L.var)) {
                    L.step.op = 'u';
                    L.step.name = "-";
                    L.step.kids.push_back(one);
                } else if (s(c) == L.var && (s(c + 1) == "+=" || s(c + 1) == "-=")) {
                    L.step = parse(c + 2, close);
                    if (s(c + 1) == "-=") {
                        Node neg;
                        neg.op = 'u';
                        neg.name = "-";
                        neg.kids.push_back(L.step);
                        L.step = neg;
                    }
                }
                L.ok = L.step.op != 'x';
            }
        }
        loops_.push_back(L);
    }
}

// The `cond ? <a> : <b>` an access at `a` sits in (same statement, same group).
std::vector<Guard> Analyzer::ternaryGuards(long a) const {
    std::vector<Guard> out;
    bool inElse = false;
    auto boundary = [&](long j) {
        const std::string &c = t_[j].s;
        return t_[j].kind == 'p' &&
               (c == "(" || c == "[" || c == "{" || c == ";" || c == "}" || c == "," || isAssignOp(c));
    };
    for (long j = a - 1; j > 0; j--) {
        if (t_[j].kind == 'p' && (t_[j].s == ")" || t_[j].s == "]") && match_[j] >= 0) {
            j = match_[j];
            continue;
        }
        if (boundary(j) || t_[j].s == "return")
            break;
        if (t_[j].s == ":")
            inElse = true;
        if (t_[j].s != "?")
            continue;
        long st = j - 1;
        while (st > 0 && !boundary(st) && t_[st].s != "return") {
            if (t_[st].kind == 'p' && (t_[st].s == ")" || t_[st].s == "]") && match_[st] >= 0)
                st = match_[st];
            st--;
        }
        Guard g;
        g.cond = parse(st + 1, j);
        g.pos = a;
        g.neg = inElse;
        out.push_back(g);
        break;
    }
    return out;
}

void Analyzer::addAccess(long a, long b, long e, long span, bool write, long textEnd) {
    if (b >= e || t_[b].kind != 'i')
        return;
    const int p = findPtr(t_[b].s, a);
    if (p < 0)
        return;
    Access x;
    x.ptr = p;
    x.pos = a;
    x.line = lineOf(a);
    x.span = span;
    x.write = write;
    x.text = textOf(a, textEnd);
    if (e == b + 1) {
        x.idx.op = 'n';
    } else if (s(b + 1) == "+" || s(b + 1) == "-")
        x.idx = parse(b, e, b);
    else
        return;
    x.guards = ternaryGuards(a);
    acc_.push_back(x);
}

// p[e], aie::load_v<N>(p + e), aie::load_unaligned_v<N>(p + e, ..), aie::store_v(p + e, v).
void Analyzer::scanAccesses() {
    for (long a = 1; a + 1 < n(); a++) {
        if (t_[a].kind != 'i' || s(a - 1) == "." || s(a - 1) == "->")
            continue;
        const std::string &f = t_[a].s;
        if ((f == "aie::load_v" || f == "aie::load_unaligned_v") && s(a + 1) == "<" && a + 4 < n() &&
            t_[a + 2].kind == 'n' && s(a + 3) == ">" && s(a + 4) == "(" && match_[a + 4] > a) {
            const long c = match_[a + 4];
            addAccess(a, a + 5, std::min(scanTo(a + 5, ","), c), t_[a + 2].num, false, c + 1);
        } else if (f == "aie::store_v" && s(a + 1) == "(" && match_[a + 1] > a) {
            const long c = match_[a + 1], arg = std::min(scanTo(a + 2, ","), c);
            long span = 1;
            for (long j = arg + 1; j + 2 < c; j++)
                if (s(j) == "extract" && s(j + 1) == "<" && t_[j + 2].kind == 'n')
                    span = t_[j + 2].num;
            if (c == arg + 2 && vecLen_.count(s(arg + 1)))
                span = vecLen_.at(s(arg + 1));
            addAccess(a, a + 2, arg, span, true, c + 1);
        } else if (s(a + 1) == "[" && !declNames_.count(a) && match_[a + 1] > a) {
            const long c = match_[a + 1];
            if (findPtr(f, a) < 0)
                continue;
            Access x;
            x.ptr = findPtr(f, a);
            x.pos = a;
            x.line = lineOf(a);
            x.write = isAssignOp(s(c + 1)) || s(c + 1) == "++" || s(c + 1) == "--";
            x.idx = parse(a + 2, c);
            x.text = textOf(a, c + 1);
            x.guards = ternaryGuards(a);
            acc_.push_back(x);
        }
    }
}

// A name assigned after its declaration has no single value: drop it.
void Analyzer::markMutations() {
    for (long i = 1; i + 1 < n(); i++) {
        if (t_[i].kind != 'i' || declNames_.count(i))
            continue;
        const std::string &nx = s(i + 1), &pv = s(i - 1);
        if (nx == "[" && match_[i + 1] > i && isAssignOp(s(match_[i + 1] + 1))) {
            int d = findDef(t_[i].s, i);
            if (d >= 0 && defs_[d].array)
                defs_[d].mut = true;
            continue;
        }
        if (!(isAssignOp(nx) || nx == "++" || nx == "--" || pv == "++" || pv == "--"))
            continue;
        int d = findDef(t_[i].s, i), p = findPtr(t_[i].s, i);
        if (d >= 0)
            defs_[d].mut = true;
        if (p >= 0)
            ptrs_[p].mut = true;
    }
}

void Analyzer::markRoots() {
    std::function<bool(const Node &)> literal = [&](const Node &x) {
        if (x.op == 'v' || x.op == 'i' || x.op == 'p' || x.op == 'x')
            return false;
        for (auto &c : x.kids)
            if (!literal(c))
                return false;
        return true;
    };
    Env none;
    for (auto &d : defs_)
        d.root = !d.array && !d.mut && literal(d.expr) && eval(d.expr, d.pos, none, d.value);
}

int Analyzer::findDef(const std::string &name, long pos) const {
    int best = -1;
    auto r = defByName_.equal_range(name);
    for (auto it = r.first; it != r.second; ++it) {
        const Def &d = defs_[it->second];
        if ((d.pos == -1 || (d.pos < pos && pos <= d.end)) && (best < 0 || d.pos > defs_[best].pos))
            best = it->second;
    }
    return best;
}

int Analyzer::findPtr(const std::string &name, long pos) const {
    int best = -1;
    for (int i = 0; i < (int)ptrs_.size(); i++)
        if (ptrs_[i].name == name && ptrs_[i].pos < pos && pos <= ptrs_[i].end &&
            (best < 0 || ptrs_[i].pos > ptrs_[best].pos))
            best = i;
    return best;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
bool Analyzer::lookup(const std::string &name, long pos, const Env &env, long &out, int depth) const {
    auto v = env.v.find(name);
    if (v != env.v.end()) {
        out = v->second;
        return true;
    }
    const int d = findDef(name, pos);
    if (d < 0)
        return false;
    if (env.ovr && env.ovr->count(d)) {
        out = env.ovr->at(d);
        return true;
    }
    if (defs_[d].mut || defs_[d].array)
        return false;
    return eval(defs_[d].expr, defs_[d].pos, env, out, depth + 1);
}

bool Analyzer::eval(const Node &nd, long pos, const Env &env, long &out, int depth) const {
    if (depth > 64)
        return false;
    long a = 0, b = 0;
    switch (nd.op) {
    case 'n':
        out = nd.val;
        return true;
    case 'p': {
        auto it = env.v.find(nd.val ? "$p1" : "$p0");
        if (it == env.v.end())
            return false;
        out = it->second;
        return true;
    }
    case 'q':
        out = nd.val ? cols_ : rows_;
        return true;
    case 'v':
        return lookup(nd.name, pos, env, out, depth);
    case 'i': {
        const int d = findDef(nd.name, pos);
        if (d < 0 || !defs_[d].array || defs_[d].mut || !eval(nd.kids[0], pos, env, a, depth + 1) || a < 0 ||
            a >= (long)defs_[d].elems.size())
            return false;
        return eval(defs_[d].elems[a], defs_[d].pos, env, out, depth + 1);
    }
    case 'u':
        if (!eval(nd.kids[0], pos, env, a, depth + 1))
            return false;
        out = nd.name == "-" ? -a : nd.name == "!" ? !a : nd.name == "~" ? ~a : a;
        return true;
    case '?':
        return eval(nd.kids[0], pos, env, a, depth + 1) && eval(nd.kids[a ? 1 : 2], pos, env, out, depth + 1);
    case 'b':
        if (!eval(nd.kids[0], pos, env, a, depth + 1))
            return false;
        if (nd.name == "&&" || nd.name == "||") {
            if ((nd.name == "&&") != (a != 0)) {
                out = a != 0;
                return true;
            }
            if (!eval(nd.kids[1], pos, env, b, depth + 1))
                return false;
            out = b != 0;
            return true;
        }
        return eval(nd.kids[1], pos, env, b, depth + 1) && binop(nd.name, a, b, out);
    }
    return false;
}

// Loop variables (of `loopVars`) and defs an expression depends on, transitively.
void Analyzer::collect(const Node &nd, long pos, const std::set<std::string> &loopVars, std::set<std::string> &vars,
                       std::set<int> &seen, int depth) const {
    if (depth > 64)
        return;
    if (nd.op == 'p')
        vars.insert(nd.val ? "$p1" : "$p0");
    if (nd.op == 'v' && loopVars.count(nd.name))
        vars.insert(nd.name);
    else if (nd.op == 'v' || nd.op == 'i') {
        const int d = findDef(nd.name, pos);
        if (d >= 0 && seen.insert(d).second) {
            collect(defs_[d].expr, defs_[d].pos, loopVars, vars, seen, depth + 1);
            for (auto &e : defs_[d].elems)
                collect(e, defs_[d].pos, loopVars, vars, seen, depth + 1);
        }
    }
    for (auto &c : nd.kids)
        collect(c, pos, loopVars, vars, seen, depth + 1);
}

// Offset terms of an access: its index plus every derived pointer's offset.
std::vector<std::pair<const Node *, long>> Analyzer::terms(const Access &a, int &block) const {
    std::vector<std::pair<const Node *, long>> out{{&a.idx, a.pos}};
    block = -1;
    for (int p = a.ptr; p >= 0; p = ptrs_[p].base) {
        if (ptrs_[p].mut)
            return {};
        if (ptrs_[p].hasOff)
            out.push_back({&ptrs_[p].off, ptrs_[p].pos});
        if (ptrs_[p].block >= 0) {
            block = ptrs_[p].block;
            break;
        }
    }
    return out;
}

// The loops to enumerate for an access, outer to inner (program ids first).
bool Analyzer::plan(const Access &a, std::vector<const Loop *> &its, std::set<int> &defsUsed) const {
    int block = -1;
    auto ts = terms(a, block);
    if (block < 0)
        return false;
    std::vector<const Loop *> encl;
    std::set<std::string> loopVars, vars;
    for (auto &L : loops_)
        if (L.bodyB <= a.pos && a.pos <= L.bodyE) {
            encl.push_back(&L);
            loopVars.insert(L.var);
        }
    for (auto &t : ts)
        collect(*t.first, t.second, loopVars, vars, defsUsed);
    for (auto &sh : blocks_[block].shape)
        collect(sh, blocks_[block].pos, loopVars, vars, defsUsed);
    for (auto &g : ifs_)
        if (g.b <= a.pos && a.pos <= g.e)
            collect(g.cond, g.pos, loopVars, vars, defsUsed);
    for (auto &g : a.guards)
        collect(g.cond, g.pos, loopVars, vars, defsUsed);
    std::vector<const Loop *> need;
    for (auto it = encl.rbegin(); it != encl.rend(); ++it) {
        const Loop &L = **it;
        if (!vars.count(L.var))
            continue;
        if (!L.ok)
            return false;
        collect(L.init, L.hdr, loopVars, vars, defsUsed);
        collect(L.bound, L.hdr, loopVars, vars, defsUsed);
        collect(L.step, L.hdr, loopVars, vars, defsUsed);
        need.insert(need.begin(), &L);
    }
    its.clear();
    for (int d = 0; d < 2; d++)
        if (vars.count(pid_[d].var))
            its.push_back(&pid_[d]);
    its.insert(its.end(), need.begin(), need.end());
    return true;
}

// Run `leaf` for every value of the loop nest. -1: not evaluable / over
// budget, 1: the leaf stopped it, 0: done.
int Analyzer::walk(const std::vector<const Loop *> &its, size_t i, Env &env, const std::function<int(Env &)> &leaf,
                   long &budget) const {
    if (i == its.size())
        return --budget < 0 ? -1 : leaf(env);
    const Loop &L = *its[i];
    long v = 0, bound = L.var == "$p0" ? rows_ : cols_, step = 1;
    if (!L.pseudo && (!eval(L.init, L.hdr, env, v) || !eval(L.bound, L.hdr, env, bound) ||
                      !eval(L.step, L.hdr, env, step) || step == 0))
        return -1;
    int r = 0;
    for (long iter = 0; r == 0 && cmpOp(L.cmp, v, bound); v += step) {
        if (++iter > 1000000)
            return -1;
        env.v[L.var] = v;
        r = walk(its, i + 1, env, leaf, budget);
    }
    env.v.erase(L.var);
    return r;
}

bool Analyzer::flatAt(const Access &a, const Env &env, long &idx, long &size, std::vector<long> *shape) const {
    int block = -1;
    auto ts = terms(a, block);
    if (block < 0)
        return false;
    idx = 0;
    for (auto &t : ts) {
        long v;
        if (!eval(*t.first, t.second, env, v))
            return false;
        idx += v;
    }
    size = 1;
    for (auto &sh : blocks_[block].shape) {
        long v;
        if (!eval(sh, blocks_[block].pos, env, v) || v <= 0)
            return false;
        size *= v;
        if (shape)
            shape->push_back(v);
    }
    return true;
}

bool Analyzer::check(const Access &a, const std::map<int, long> *ovr, bool stopEarly, AccessEval &r) const {
    std::vector<const Loop *> its;
    std::set<int> used;
    r = AccessEval();
    if (!plan(a, its, used))
        return false;
    std::vector<const Guard *> guards;
    for (auto &g : ifs_)
        if (g.b <= a.pos && a.pos <= g.e)
            guards.push_back(&g);
    for (auto &g : a.guards)
        guards.push_back(&g);
    Env env;
    env.ovr = ovr;
    long budget = 4000000;
    auto leaf = [&](Env &e) -> int {
        for (auto *g : guards) {
            long c;
            if (eval(g->cond, g->pos, e, c) && (c != 0) == g->neg)
                return 0; // not executed here (an unknown guard counts as taken)
        }
        long idx, size;
        if (!flatAt(a, e, idx, size, nullptr))
            return -1;
        if (idx < r.lo) {
            r.lo = idx;
            r.wLo = e;
        }
        if (idx + a.span - size > r.worst) {
            r.worst = idx + a.span - size;
            r.hiEnd = idx + a.span;
            r.size = size;
            r.wHi = e;
        }
        r.out |= idx < 0 || idx + a.span > size;
        return stopEarly && r.out ? 1 : 0;
    };
    r.ok = walk(its, 0, env, leaf, budget) >= 0 && r.worst != LONG_MIN;
    return r.ok;
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------
std::string Analyzer::witnessText(const Access &a, const Env &w) const {
    std::string out, core;
    for (auto &L : loops_) // outer to inner
        if (L.bodyB <= a.pos && a.pos <= L.bodyE && w.v.count(L.var))
            out += (out.empty() ? "" : ", ") + L.var + "=" + std::to_string(w.v.at(L.var));
    auto p0 = w.v.find("$p0"), p1 = w.v.find("$p1");
    if (p0 != w.v.end() || p1 != w.v.end())
        core = " on core (" + std::to_string(p0 != w.v.end() ? p0->second : 0) + "," +
               std::to_string(p1 != w.v.end() ? p1->second : 0) + ")";
    return (out.empty() ? "" : " at " + out) + core;
}

// The smallest change of ONE literal constant that makes every analysed access fit.
// Constants that size the block of an out-of-block access but that its index
// (and the loops around it) never read: the block and the access disagree on
// them. A constant read by both only adds slack when changed -- not a cause.
std::set<int> Analyzer::decoupledRoots(const Access &a) const {
    int block = -1;
    auto ts = terms(a, block);
    std::set<std::string> vars, none;
    std::set<int> shapeDefs, indexDefs, out;
    if (block < 0)
        return out;
    for (auto &sh : blocks_[block].shape)
        collect(sh, blocks_[block].pos, none, vars, shapeDefs);
    for (auto &t : ts)
        collect(*t.first, t.second, none, vars, indexDefs);
    for (auto &L : loops_)
        if (L.bodyB <= a.pos && a.pos <= L.bodyE) {
            collect(L.init, L.hdr, none, vars, indexDefs);
            collect(L.bound, L.hdr, none, vars, indexDefs);
            collect(L.step, L.hdr, none, vars, indexDefs);
        }
    for (int d : shapeDefs)
        if (defs_[d].root && !indexDefs.count(d))
            out.insert(d);
    return out;
}

std::string Analyzer::findCause(const std::vector<int> &bad, const std::vector<int> &good, int &causeDef) const {
    std::set<int> roots;
    for (int b : bad)
        for (int d : decoupledRoots(acc_[b]))
            roots.insert(d);
    struct Fix {
        int def;
        long v;
        double rel;
    };
    std::vector<Fix> fixes;
    for (int d : roots) {
        const long v0 = defs_[d].value, lim = std::min(64L, std::max(8L, 2 * std::labs(v0)));
        bool found = false;
        for (long delta = 1; delta <= lim && !found; delta++)
            for (long v : {v0 + delta, v0 - delta}) {
                if (found || (v0 > 0 && v <= 0))
                    continue;
                std::map<int, long> ovr{{d, v}};
                bool fits = true;
                for (auto *set : {&bad, &good})
                    for (int x : *set) {
                        AccessEval r;
                        if (fits && (!check(acc_[x], &ovr, true, r) || r.out))
                            fits = false;
                    }
                if (fits) {
                    fixes.push_back({d, v, (double)delta / std::max(1L, std::labs(v0))});
                    found = true;
                }
            }
    }
    causeDef = -1;
    if (fixes.empty())
        return ""; // the index arithmetic itself (or several constants) is wrong
    std::stable_sort(fixes.begin(), fixes.end(), [](const Fix &a, const Fix &b) { return a.rel < b.rel; });
    const Def &d = defs_[fixes[0].def];
    causeDef = fixes[0].def;
    std::set<int> lines, sized;
    for (int b : bad) {
        lines.insert(acc_[b].line);
        if (decoupledRoots(acc_[b]).count(causeDef))
            sized.insert(acc_[b].line);
    }
    auto list = [](const std::set<int> &ls) {
        std::string t;
        for (int l : ls)
            t += (t.empty() ? "" : ", ") + std::to_string(l);
        return (ls.size() > 1 ? "lines " : "line ") + t;
    };
    std::string out = "cause (static analysis): `" + d.text + "` (" +
                      (d.line ? srcName_ + ":" + std::to_string(d.line) : std::string("#define")) +
                      ") sizes the block of the access at " + list(sized) + ", whose index never reads " + d.name +
                      " -- with " + d.name + " = " + std::to_string(fixes[0].v) +
                      " every analysed block access fits, including the out-of-block one" +
                      (lines.size() > 1 ? "s at " : " at ") + list(lines);
    for (size_t i = 1; i < fixes.size() && i < 3; i++)
        out += i == 1 ? "; a single change of " + defs_[fixes[i].def].name + " = " + std::to_string(fixes[i].v)
                      : ", or " + defs_[fixes[i].def].name + " = " + std::to_string(fixes[i].v);
    if (fixes.size() > 1)
        out += " would also fit";
    return out;
}

std::string Analyzer::describe(const Access &a, const AccessEval &r, int causeDef) const {
    int block = -1;
    terms(a, block);
    const Block &B = blocks_[block];
    const bool under = r.lo < 0 && !(r.worst > 0);
    const Env &w = under ? r.wLo : r.wHi;
    long idx = 0, size = 0;
    std::vector<long> shape;
    flatAt(a, w, idx, size, &shape);
    std::ostringstream os;
    const long first = under ? r.lo : r.hiEnd - 1;
    os << "`" << a.text << "` " << (a.write ? "writes" : "reads") << " element " << first;
    if (a.span > 1)
        os << " (a " << a.span << "-lane access from element " << (under ? r.lo : r.hiEnd - a.span) << ")";
    os << " of " << (B.local ? "local array" : "block") << " `" << B.var << "` (" << size << " elements";
    if (!B.local) {
        os << ", shape (";
        for (size_t d = 0; d < shape.size(); d++)
            os << (d ? "," : "") << shape[d];
        os << ") = " << B.text << (B.role.empty() ? "" : " [" + B.role + "]");
    }
    os << ", line " << B.line << ")" << witnessText(a, w);
    if (under) {
        os << ": before the start of the block";
        return os.str();
    }
    // The overflow seen in block coordinates, and the loops that move it.
    long inner = shape.empty() ? 1 : size / shape[0];
    const long c0 = first / std::max(1L, inner);
    std::vector<long> coord(shape.size());
    long rest = first;
    for (long d = (long)shape.size() - 1; d > 0; d--) {
        coord[d] = rest % shape[d];
        rest /= shape[d];
    }
    if (!coord.empty())
        coord[0] = rest;
    if (shape.size() > 1) {
        os << ": element " << first << " is (";
        for (size_t d = 0; d < coord.size(); d++)
            os << (d ? "," : "") << coord[d];
        os << ")";
    }
    const std::string dim = B.local ? std::string("the array") : std::string("dim 0 (") + traceDimName(B.role, 0) + ")";
    os << ", " << dim << " index " << c0 << " is past its extent " << (shape.empty() ? 0 : shape[0]) << " (" << r.worst
       << " element" << (r.worst == 1 ? "" : "s") << " past the end)";
    std::vector<const Loop *> drivers;
    for (auto &L : loops_) {
        if (!w.v.count(L.var) || !(L.bodyB <= a.pos && a.pos <= L.bodyE))
            continue;
        Env w2 = w;
        long v0, i2, s2;
        if (!eval(L.init, L.hdr, w2, v0))
            continue;
        w2.v[L.var] = v0;
        if (flatAt(a, w2, i2, s2, nullptr) && (i2 + a.span - 1) / std::max(1L, inner) < c0)
            drivers.push_back(&L);
    }
    if (!drivers.empty()) {
        os << "; it is driven by";
        for (auto *L : drivers)
            os << (L == drivers[0] ? " " : ", ") << "`" << L->cond << "` (line " << L->line << ")";
    }
    // An index that never reads the cause walks a fixed extent of dim 0.
    if (causeDef >= 0) {
        std::set<std::string> vars, none;
        std::set<int> seen;
        for (auto &t : terms(a, block))
            collect(*t.first, t.second, none, vars, seen);
        for (auto *L : drivers) {
            collect(L->init, L->hdr, none, vars, seen);
            collect(L->bound, L->hdr, none, vars, seen);
        }
        if (!seen.count(causeDef))
            os << "; neither the index nor these loops depend on " << defs_[causeDef].name << ": they walk " << c0 + 1
               << " entries of " << dim << " whatever " << defs_[causeDef].name << " is";
    }
    return os.str();
}

StaticBoundsResult Analyzer::run() {
    StaticBoundsResult res;
    std::vector<AccessEval> evals(acc_.size());
    std::vector<int> bad, good;
    for (size_t i = 0; i < acc_.size(); i++) {
        std::vector<const Loop *> its;
        std::set<int> used;
        if (!plan(acc_[i], its, used) || !check(acc_[i], nullptr, false, evals[i])) {
            res.skipped++;
            continue;
        }
        res.analysed++;
        (evals[i].out ? bad : good).push_back((int)i);
    }
    if (bad.empty())
        return res;
    int causeDef = -1;
    res.cause = findCause(bad, good, causeDef);
    for (int b : bad)
        res.findings.push_back({acc_[b].line, describe(acc_[b], evals[b], causeDef)});
    return res;
}

} // namespace

StaticBoundsResult staticBlockBounds(const std::string &src, const TraceKernel &k, int rows, int cols,
                                     const std::string &prelude, const std::string &srcName) {
    Analyzer a(src, k, rows, cols, srcName);
    a.build(prelude);
    return a.run();
}

} // namespace convautotile
