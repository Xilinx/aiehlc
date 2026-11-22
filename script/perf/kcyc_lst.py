#!/usr/bin/env python3
"""kcyc_lst -- static per-launch AIE core cycles of a gemm_kacc instance from its chess kernel.lst (no simulator).

  kcyc_lst.py LST [--func gemm_kacc] [-v]

Counts issued bundles (one bundle = one cycle; NOPs and branch delay slots are explicit bundles in the .lst) along
the static control flow of the kernel function, each multiplied by the trip counts of its enclosing loops:
  .loop_nesting N        the region level of the following bundles
  ZLS_.. / ZLE_.. labels zero-overhead loop body (ZLS label .. the bundle after the ZLE label), trip = lc
                         ('MOV lc, #c' / 'MOVX lc, rX' resolved from the last 'rX, #c' immediate)
  other loops            a level-(L+1) span ends with a back branch 'JNZD rX, rX, ..' (taken while rX != 0, then
                         decrement): trip = c + 1, c = the last immediate move into rX before the span
  calls 'JL #addr'       callee cost added once per execution of the call bundle; memset's byte loop
                         (ZLS 'MOVX lc, r1') takes r1 from the caller's last 'r1, #c' before the JL
Reported per function: total cycles, cycles before the first input-window lock acquire inside the outermost loop
('pre' = prologue + output-window acquire + accumulator zeroing of the first tile), vmac count, stack frame, SPILL
(VLD/VST on [sp] inside a loop holding VMACs).
Library use: analyse(path) -> dict(total, pre, post, vmac, frame, spill, loops=[...]).
Lock stalls, DMA waits and memory-bank stalls are not modelled (static issue count only).
"""
import argparse
import json
import re
import sys

_ADDR = re.compile(r"^\s+(\d+)\s+0x")
_IMM = re.compile(r"\b(?:MOV|MOVX|MOVA|MOVXM|MOVS)\s+(r\d+|lc|m\d+),\s*#(-?\d+)")
_LC_REG = re.compile(r"\bMOVX?\s+lc,\s*(r\d+)")
_JNZD = re.compile(r"\bJNZD\s+(r\d+),")
_JL = re.compile(r"\bJL\s+#(\d+)")
_LABEL = re.compile(r"^\.label\s+(\S+)")
_NEST = re.compile(r"^\.loop_nesting\s+(\d+)")
_SP = re.compile(r"\b(?:VLD[AB]?|VST)[^;]*\[sp")


def load(path):
    """-> list of items (kind, value, text) in file order and {func label: start index}"""
    items, funcs, cur = [], {}, None
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        m = _LABEL.match(ln)
        if m:
            items.append(("label", m.group(1), ln))
            continue
        m = _NEST.match(ln)
        if m:
            items.append(("nest", int(m.group(1)), ln))
            continue
        if ln.startswith(".function_start"):
            items.append(("fstart", None, ln))
            continue
        m = _ADDR.match(ln)
        if m:
            items.append(("b", int(m.group(1)), ln))
    return items


def functions(items):
    """{name: (i0, i1, addr0)} for each .function_start (name = last plain label before it)"""
    out, i = {}, 0
    starts = [k for k, it in enumerate(items) if it[0] == "fstart"]
    for n, k in enumerate(starts):
        name = None
        for j in range(k - 1, -1, -1):
            if items[j][0] == "label":
                name = items[j][1]
                if not name.startswith("__") and not name.endswith("func_begin0"):
                    break
            elif items[j][0] == "b":
                break
        end = starts[n + 1] if n + 1 < len(starts) else len(items)
        addr0 = next((items[j][1] for j in range(k, end) if items[j][0] == "b"), None)
        out[name] = (k, end, addr0)
    return out


def last_imm(items, i, reg, lo=0):
    for j in range(i - 1, lo - 1, -1):
        if items[j][0] != "b":
            continue
        for r, c in _IMM.findall(items[j][2]):
            if r == reg:
                return int(c)
    return None


class Counter:
    def __init__(self, items):
        self.items = items
        self.funcs = functions(items)
        self.by_addr = {v[2]: k for k, v in self.funcs.items()}
        self.memo = {}

    def bundles(self, i0, i1):
        """[(index, level, text)] of the bundles in [i0, i1) with their region level, zls body flag"""
        out, lvl, in_zls, zle_pending = [], 0, False, False
        for k in range(i0, i1):
            kind, v, txt = self.items[k]
            if kind == "nest":
                lvl = v
            elif kind == "label" and v.startswith("ZLS_"):
                in_zls = True
            elif kind == "label" and v.startswith("ZLE_"):
                zle_pending = True
            elif kind == "b":
                out.append(dict(k=k, lvl=lvl, zls=in_zls, txt=txt))
                if zle_pending:
                    in_zls, zle_pending = False, False
        return out

    def cost(self, name, arg_r1=None):
        key = (name, arg_r1)
        if key in self.memo:
            return self.memo[key]
        i0, i1, _ = self.funcs[name]
        bs = self.bundles(i0, i1)
        res = self.walk(bs, 0, len(bs), 0, name, arg_r1)
        self.memo[key] = res
        return res

    def trip_zls(self, bs, a, arg_r1):
        # lc set in a bundle before the ZLS body
        for j in range(a - 1, -1, -1):
            t = bs[j]["txt"]
            m = re.search(r"\bMOVX?\s+lc,\s*#(\d+)", t)
            if m:
                return int(m.group(1))
            m = _LC_REG.search(t)
            if m:
                reg = m.group(1)
                if reg == "r1" and arg_r1 is not None:
                    return arg_r1
                v = last_imm(self.items, bs[j]["k"], reg, bs[0]["k"])
                if v is None:
                    raise ValueError("lc from %s: no immediate" % reg)
                return v
        raise ValueError("ZLS without lc")

    def walk(self, bs, a, b, lvl, fname, arg_r1):
        """cost of bundles bs[a:b] all at level >= lvl, once"""
        tot = dict(cyc=0, vmac=0, loops=[], spill=0, calls=[])
        i = a
        while i < b:
            x = bs[i]
            if x["lvl"] > lvl:
                # loop span at level lvl+1 .. until level drops back to <= lvl
                j = i
                while j < b and bs[j]["lvl"] > lvl:
                    j += 1
                if all(bs[q]["zls"] for q in range(i, j)):
                    trip = self.trip_zls(bs, i, arg_r1)
                    kind = "zls"
                else:
                    br = None
                    for q in range(j - 1, i - 1, -1):
                        if bs[q]["lvl"] == lvl + 1:
                            m = _JNZD.search(bs[q]["txt"])
                            if m:
                                br = m.group(1)
                                break
                    if br is None:
                        raise ValueError("%s: no JNZD back branch for level %d span at bundle %s" %
                                         (fname, lvl + 1, x["txt"][:20].strip()))
                    c = last_imm(self.items, x["k"], br, bs[0]["k"])
                    if c is None:
                        raise ValueError("%s: no immediate init for %s" % (fname, br))
                    trip = c + 1
                    kind = "jnzd"
                sub = self.walk(bs, i, j, lvl + 1, fname, arg_r1)
                tot["cyc"] += trip * sub["cyc"]
                tot["vmac"] += trip * sub["vmac"]
                tot["spill"] += sub["spill"]
                tot["loops"].append(dict(level=lvl + 1, kind=kind, trip=trip, body_cyc=sub["cyc"],
                                         vmac=sub["vmac"], nb=j - i, inner=sub["loops"]))
                tot["calls"] += sub["calls"]
                i = j
                continue
            t = x["txt"]
            tot["cyc"] += 1
            nv = len(re.findall(r"\bVMAC\b|\bVMAC\.f\b|\bVMUL\b|\bVMUL\.f\b", t))
            tot["vmac"] += nv
            if lvl > 0 and _SP.search(t):
                tot["spill"] += 1
            m = _JL.search(t)
            if m:
                callee = self.by_addr.get(int(m.group(1)))
                if callee is None:
                    raise ValueError("%s: call to unknown #%s" % (fname, m.group(1)))
                r1 = call_r1(self.items, x["k"], bs[0]["k"]) if callee == "memset" else None
                c = self.cost(callee, r1)
                tot["cyc"] += c["cyc"]
                tot["calls"].append(dict(callee=callee, r1=r1, cyc=c["cyc"]))
            i += 1
        return tot


def analyse(path, func="gemm_kacc", entry="_main"):
    """static cycles of one launch: entry (core main incl. any kco register-write prelude) -> kernel func.
    pre = cycles before the kernel's first input-window ACQ (the first ACQ after the first-tile accumulator
    memset; without a memset call the first ACQ at loop level >= 1)"""
    items = load(path)
    C = Counter(items)
    names = [n for n in C.funcs if n and func in n and not n.endswith("_end")]
    if len(names) != 1:
        raise ValueError("%s: %d functions match %r: %s" % (path, len(names), func, names))
    name = names[0]
    kern = C.cost(name)
    i0, i1, _ = C.funcs[name]
    bs = C.bundles(i0, i1)
    has_ms = any(_JL.search(x["txt"]) and C.by_addr.get(int(_JL.search(x["txt"]).group(1))) == "memset" for x in bs)
    pre, seen_ms = 0, not has_ms
    for x in bs:
        if re.search(r"\bACQ\b", x["txt"]) and seen_ms and (has_ms or x["lvl"] >= 1):
            break
        pre += 1
        m = _JL.search(x["txt"])
        if m:
            callee = C.by_addr.get(int(m.group(1)))
            r1 = call_r1(items, x["k"], bs[0]["k"]) if callee == "memset" else None
            pre += C.cost(callee, r1)["cyc"]
            seen_ms = seen_ms or callee == "memset"
    main_cyc = main_pre = 0
    if entry in C.funcs:
        mres = C.cost(entry)
        main_cyc = mres["cyc"] - kern["cyc"] * sum(1 for c in mres["calls"] if c["callee"] == name)
        mi0, mi1, _ = C.funcs[entry]
        for x in C.bundles(mi0, mi1):
            m = _JL.search(x["txt"])
            if m and C.by_addr.get(int(m.group(1))) == name:
                break
            main_pre += 1
            if m:
                main_pre += C.cost(C.by_addr.get(int(m.group(1))))["cyc"]
    frame = max([int(v) for v in re.findall(r"PADDXM \[sp\], #(\d+)", "\n".join(x["txt"] for x in bs))] or [0])
    return dict(func=name, kernel=kern["cyc"], main=main_cyc, total=kern["cyc"] + main_cyc, pre=main_pre + pre,
                post=kern["cyc"] + main_cyc - main_pre - pre, vmac=kern["vmac"], frame=frame, spill=kern["spill"],
                loops=kern["loops"], calls=kern["calls"])


def call_r1(items, k, lo):
    """memset size: the last 'r1, #c' in the JL's delay slots (the 5 bundles after it), else before the JL"""
    v, n, j = None, 0, k + 1
    while j < len(items) and n < 5:
        if items[j][0] == "b":
            n += 1
            for r, c in _IMM.findall(items[j][2]):
                if r == "r1":
                    v = int(c)
        j += 1
    return v if v is not None else last_imm(items, k, "r1", lo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lst")
    ap.add_argument("--func", default="gemm_kacc")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    r = analyse(a.lst, a.func)
    if not a.v:
        r.pop("loops")
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
