#!/usr/bin/env python3
"""Static provenance-map generator for raw-XAie single-kernel aiehlc apps.

Extracts coarse tile / kernel-placement / flow structure from the XAie driver
calls in a generated aout/host.cc and emits the dfscheduleprovenancemap.json +
dmaphopprovenacemap.json schema consumed by schedule_view.py. Runtime-decided
BD/lock/port values are NOT statically visible and are emitted as placeholders.
"""
import argparse, json, os, re, shutil, sys


_SIZEOF = {"u32": 4, "uint32_t": 4, "int32_t": 4, "int": 4, "float": 4,
           "u16": 2, "int16_t": 2, "u8": 1, "char": 1, "uint8_t": 1}


def eval_int(expr, defs):
    e = expr
    e = re.sub(r"sizeof\s*\(\s*(\w+)\s*\)",
               lambda m: str(_SIZEOF.get(m.group(1), 4)), e)
    for _ in range(10):
        prev = e
        for name, val in defs.items():
            e = re.sub(r"\b%s\b" % re.escape(name), "(%s)" % str(val), e)
        if e == prev:
            break
    if re.search(r"[A-Za-z_]", e):
        return None
    try:
        v = eval(e, {"__builtins__": {}}, {})
        return int(v)
    except Exception:
        return None


class MacroResolver:
    def __init__(self, aie_gen, aiesim):
        self.defs = {"AIE_GEN": aie_gen}
        if aiesim:
            self.defs["__AIESIM__"] = 1

    def _defined(self, name):
        return name in self.defs

    def _eval_cond(self, expr):
        e = expr.strip()
        for name, val in self.defs.items():
            # Function replacement so macro bodies with backslashes or \g refs
            # (e.g. multi-line BENCH defines) are substituted literally.
            e = re.sub(r"\b%s\b" % re.escape(name), lambda _m, v=str(val): v, e)
        e = re.sub(r"\b[A-Za-z_]\w*\b", "0", e)
        try:
            return bool(eval(e, {"__builtins__": {}}, {}))
        except Exception:
            return False

    def active_source(self, src):
        out, stack = [], []
        def active():
            return all(s[0] and s[1] for s in stack) if stack else True
        for line in src.splitlines():
            s = line.strip()
            m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
            if not m:
                # Honor inline #define/#undef in active regions so a later
                # #ifdef of an in-file-defined macro (e.g. the controlperf
                # `#define _CONTROL_WRITE_TEST_` guard) resolves correctly.
                dm = re.match(r"#\s*(define|undef)\s+(\w+)(.*)", s)
                if dm and active():
                    if dm.group(1) == "define":
                        body = dm.group(3).strip()
                        self.defs[dm.group(2)] = body if body else 1
                    else:
                        self.defs.pop(dm.group(2), None)
                if active():
                    out.append(line)
                continue
            kw, rest = m.group(1), m.group(2).strip()
            parent = all(x[0] and x[1] for x in stack) if stack else True
            if kw == "ifdef":
                taken = self._defined(rest)
                stack.append((parent, taken, taken))
            elif kw == "ifndef":
                taken = not self._defined(rest)
                stack.append((parent, taken, taken))
            elif kw == "if":
                taken = self._eval_cond(rest)
                stack.append((parent, taken, taken))
            elif kw == "elif":
                p, _, done = stack[-1]
                taken = (not done) and self._eval_cond(rest)
                stack[-1] = (p, taken, done or taken)
            elif kw == "else":
                p, _, done = stack[-1]
                stack[-1] = (p, not done, True)
            elif kw == "endif":
                if stack:
                    stack.pop()
        return "\n".join(out)


RE_LOADELF = re.compile(
    r"XAie_LoadElfMem\s*\([^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)\s*,"
    r"\s*\(unsigned char\s*\*\)\s*(\w+)")
RE_MOVE_IN = re.compile(
    r"XAie_MoveDataExternal2Aie\s*\([^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)\s*,"
    r"[^,]+,\s*([^,]+),[^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)", re.DOTALL)
RE_MOVE_OUT = re.compile(
    r"XAie_MoveDataAie2External\s*\([^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)\s*,"
    r"[^,]+,\s*([^,]+),[^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)", re.DOTALL)
RE_ROUTE = re.compile(
    r"XAie_Route\s*\([^,]+,\s*[^,]+,\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)\s*,"
    r"\s*XAie_TileLoc\s*\(([^,]+),([^)]+)\)", re.DOTALL)

# Control-packet send: a __Runtime_CtrlInstance designated-initializer block, or a
# composite __Runtime_ctrl_read_target / __Runtime_ctrl_push_target call. Both are
# same-column (dest_col == shim_col); a send reduces to (shim_col, dest_row,
# resp_words). The struct body may span many lines (DOTALL); the call form gives
# shim_col/dest_col/dest_row as positional args 2/3/4.
RE_CTRL_STRUCT = re.compile(
    r"__Runtime_CtrlInstance\s+\w+\s*=\s*\{(.*?)\}", re.DOTALL)
RE_CTRL_FIELD = re.compile(r"\.(\w+)\s*=\s*([^,}]+)")
RE_CTRL_CALL = re.compile(
    r"__Runtime_ctrl_(?:read|push)_target\s*\(\s*[^,]+,\s*"
    r"([^,]+),\s*([^,]+),\s*([^,]+),")


def collect_defines(src):
    """Object-like #define NAME body -> {name: body}."""
    defs = {}
    for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$",
                         src, re.MULTILINE):
        defs[m.group(1)] = m.group(2).strip()
    return defs


def resolve_tileloc(col_expr, row_expr, defs):
    c = eval_int(col_expr, defs)
    r = eval_int(row_expr, defs)
    if c is None or r is None:
        return None
    return (c, r)


def _ctrl_int(expr, defs, default=None):
    """eval_int with C integer-suffix stripping (0u/3u/0x1000u) and a fallback,
    for control-packet field exprs. eval_int rejects any [A-Za-z_], so the u/l
    suffix on unsigned literals must be removed first (scoped here, not in the
    shared eval_int). Only numeric literals are stripped, so identifiers like
    _rspcap / RAW_BD_SLOT still fail to fold and take the default."""
    cleaned = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", expr.strip())
    v = eval_int(cleaned, defs)
    return default if v is None else v


def extract_ctrl_sends(active, defs):
    """Reduce every control-packet send in @active to a same-column send dict
    {shim_col, dest_row, resp_words}, deduped by (shim_col, dest_row). Values fold
    through @defs; resp_words defaults to 1 when unresolvable. Sends whose shim_col
    or dest_row cannot be resolved are skipped (mirrors tile-loc resolution)."""
    sends, seen = [], set()

    def add(shim_col, dest_row, resp_words):
        if shim_col is None or dest_row is None:
            return
        key = (shim_col, dest_row)
        if key in seen:
            return
        seen.add(key)
        sends.append({"shim_col": shim_col, "dest_row": dest_row,
                      "resp_words": resp_words if resp_words else 1})

    for m in RE_CTRL_STRUCT.finditer(active):
        fields = {k: v for k, v in RE_CTRL_FIELD.findall(m.group(1))}
        add(_ctrl_int(fields.get("shim_col", ""), defs),
            _ctrl_int(fields.get("dest_row", ""), defs),
            _ctrl_int(fields.get("resp_words", "1"), defs, default=1))
    for m in RE_CTRL_CALL.finditer(active):
        add(_ctrl_int(m.group(1), defs), _ctrl_int(m.group(3), defs), 1)
    return sends


def strip_comments(src):
    """Drop C block/line comments so inline /*src=*/ notes inside XAie call
    argument lists do not break the tile-loc regexes."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


RE_XAIE_CALL = re.compile(
    r"(XAie_(LoadElfMem|MoveData\w+|Route)|__Runtime_ctrl_setup_routing"
    r"|__Runtime_ctrl_(read|push)_target)\b")
RE_FUNC_HDR = re.compile(r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$")


def find_entry_fn(active):
    """Name of the function that encloses the XAie routing/DMA calls.

    The view's find_function_range defaults to the tiling flow's
    'host_canonicalized'; the raw-XAie host.cc uses a differently named function
    (e.g. test_routing), so we name it here for build_dfschedule to surface.
    Brace-depth aware: a function header sits at depth 0, its body at depth >=1.
    """
    cur_fn, depth = None, 0
    for ln in active.split("\n"):
        s = ln.strip()
        if depth == 0 and s and not s.startswith(("#", "//")):
            m = RE_FUNC_HDR.search(s)
            if m:
                cur_fn = m.group(1)
        if depth >= 1 and cur_fn and RE_XAIE_CALL.search(ln):
            return cur_fn
        depth += ln.count("{") - ln.count("}")
        if depth < 0:
            depth = 0
    return None


def extract_model(raw_src, aie_gen, aiesim):
    active = strip_comments(MacroResolver(aie_gen, aiesim).active_source(raw_src))
    defs = collect_defines(active)
    m = re.search(r"\bint\s+shimcol\s*=\s*(\d+)", active)
    if m:
        defs["shimcol"] = int(m.group(1))
    for lm in re.finditer(r"\bu32\s+(\w+)\s*=\s*([^;]+);", active):
        v = eval_int(lm.group(2), defs)
        if v is not None:
            defs[lm.group(1)] = v

    tiles, seen = [], set()

    def add_tile(loc, ttype=None):
        if loc is None or loc in seen:
            return
        seen.add(loc)
        if ttype is None:
            ttype = "shim" if loc[1] == 0 else "core"
        tiles.append({"col": loc[0], "row": loc[1], "type": ttype})

    kernel_placements = {}
    for km in RE_LOADELF.finditer(active):
        loc = resolve_tileloc(km.group(1), km.group(2), defs)
        if loc is not None:
            add_tile(loc)
            kernel_placements[loc] = km.group(3)

    flows, covered = [], set()
    for mm in RE_MOVE_IN.finditer(active):
        shim = resolve_tileloc(mm.group(1), mm.group(2), defs)
        core = resolve_tileloc(mm.group(4), mm.group(5), defs)
        length = eval_int(mm.group(3), defs)
        if shim is None or core is None:
            continue
        add_tile(shim)
        add_tile(core)
        flows.append({"src": shim, "dst": core, "direction": "S2MM",
                      "len": length or 0})
        covered.add((shim, core))
    for mm in RE_MOVE_OUT.finditer(active):
        core = resolve_tileloc(mm.group(1), mm.group(2), defs)
        shim = resolve_tileloc(mm.group(4), mm.group(5), defs)
        length = eval_int(mm.group(3), defs)
        if core is None or shim is None:
            continue
        add_tile(core)
        add_tile(shim)
        flows.append({"src": core, "dst": shim, "direction": "MM2S",
                      "len": length or 0})
        covered.add((core, shim))

    for rm in RE_ROUTE.finditer(active):
        src = resolve_tileloc(rm.group(1), rm.group(2), defs)
        dst = resolve_tileloc(rm.group(3), rm.group(4), defs)
        if src is None or dst is None:
            continue
        add_tile(src)
        add_tile(dst)
        # XAie_Route only adds connectivity if no MoveData covers that pair.
        if (src, dst) in covered:
            continue

    for s in extract_ctrl_sends(active, defs):
        col, drow = s["shim_col"], s["dest_row"]
        length = s["resp_words"] * 4
        for r in range(0, drow + 1):
            add_tile((col, r), ctrl_tile_type(r, aie_gen))
        shim, dst = (col, 0), (col, drow)
        flows.append({"src": shim, "dst": dst, "direction": "S2MM", "len": length})
        flows.append({"src": dst, "dst": shim, "direction": "MM2S", "len": length})

    return {"tiles": tiles, "kernel_placements": kernel_placements,
            "flows": flows, "entry_fn": find_entry_fn(active)}


# AIE-core row start per generation (row 0 shim; 0<row<start memtile; row>=start
# core). Mirrors the C rt_port_evt_base geometry and aiediag.AIE_TILE_ROW_START:
# gen5/AIE2PS have 2 memtile rows (cores from 3), gen2 one (from 2), gen1 none.
_CORE_ROW_START = {1: 1, 2: 2, 5: 3}


def ctrl_tile_type(row, aie_gen):
    if row == 0:
        return "shim"
    start = _CORE_ROW_START.get(int(aie_gen), 3)
    return "memtile" if row < start else "core"


def hw_gen_str(g):
    return {1: "Gen1", 2: "Gen2", 5: "Gen5"}.get(int(g), "Gen%s" % g)


def placeholder_bd(length):
    """Runtime-decided BD/lock values are not statically visible."""
    return {"bd_id": "runtime", "buffer_offset": 0, "len": int(length),
            "enable_packet": False, "packet_id": 0, "next_bd": -1,
            "acquire_lock": [{"id": -1, "val": 0}],
            "release_lock": [{"id": -1, "val": 0}]}


def _kernel_name(symbol):
    match = re.fullmatch(r"_binary_kernel_(.+)_start", symbol or "")
    return match.group(1) if match else symbol


def collect_kernel_artifacts(model, artifacts_dir, out_dir):
    found = {}
    if not artifacts_dir:
        return found
    for loc, symbol in model["kernel_placements"].items():
        name = _kernel_name(symbol)
        if not name:
            continue
        sources = {
            "source": os.path.join(artifacts_dir, "%s.cc" % name),
            "kernel_cc": os.path.join(
                artifacts_dir, "kernelcfg", name, "wrapper.cc"),
            "bcf": os.path.join(
                artifacts_dir, "kernelcfg", name, "aieml.bcf"),
            "dm_offsets": os.path.join(
                artifacts_dir, "kernelcfg", name, "dm_offsets.h"),
        }
        rels = {
            "source": "%s.cc" % name,
            "kernel_cc": os.path.join("kernelcfg", name, "wrapper.cc"),
            "bcf": os.path.join("kernelcfg", name, "aieml.bcf"),
            "dm_offsets": os.path.join("kernelcfg", name, "dm_offsets.h"),
        }
        copied = {}
        for kind, source in sources.items():
            if not os.path.isfile(source):
                continue
            dest = os.path.join(out_dir, rels[kind])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(source, dest)
            copied[kind] = rels[kind]
        if copied:
            found[loc] = copied
    return found


def build_dfschedule(model, aie_gen, tile_artifacts=None):
    tiles_out, ch_counter = [], {}
    for t in model["tiles"]:
        key = (t["col"], t["row"])
        chans = []
        for fi, f in enumerate(model["flows"]):
            touches = (f["src"] == key and f["direction"] == "MM2S") or \
                      (f["dst"] == key and f["direction"] == "S2MM")
            if not touches:
                continue
            idx = ch_counter.get(key, 0)
            ch_counter[key] = idx + 1
            chans.append({"channel": idx, "direction": f["direction"],
                          "enable_out_of_order": False, "flow_index": fi,
                          "bd_chain": [placeholder_bd(f["len"])],
                          "start_io": [{"repeat_count": 1}]})
        tile = {"col": t["col"], "row": t["row"], "type": t["type"],
                "dma_channels": chans}
        artifact = (tile_artifacts or {}).get(key, {})
        if artifact.get("kernel_cc"):
            tile["kernel_cc"] = artifact["kernel_cc"]
        if artifact.get("bcf"):
            tile["bcf"] = artifact["bcf"]
        tiles_out.append(tile)
    kernel_cfgs, kg_tiles = [], []
    for (c, r), name in model["kernel_placements"].items():
        kernel_cfgs.append({"sym_name": name, "flow_index": 0, "packet_id": 0,
                            "dma_channel": 0, "tile_index": 0})
        kg_tiles.append({"col": c, "row": r})
    flow_summary = []
    for fi, f in enumerate(model["flows"]):
        flow_summary.append({"flow_index": fi, "direction": f["direction"]})
    doc = {"version": 1, "startcol": 0, "aie_gen": hw_gen_str(aie_gen),
           "provenance_source": "static-xaie", "module_attrs": [],
           "tiles": tiles_out, "kernel_configs": kernel_cfgs,
           "load_kernel_group": [{"callee": "__Runtime_load_kernel_group",
                                  "tiles": kg_tiles}],
           "flow_summary": flow_summary, "invariant_checks": []}
    # Name the host.cc function the view should attribute source lines to; the
    # raw-XAie flow uses test_routing, not the tiling flow's host_canonicalized.
    if model.get("entry_fn"):
        doc["host_entry_fn"] = model["entry_fn"]
    return doc


def build_dmaphop(model):
    """Emit communication_paths in the schedule_view._load_comm_paths shape.

    Stages carry roles ('producer'/'channel'/'consumer'), tiles are {col,row}
    dicts, and the channel stage holds hops as '(c,r)' from/to strings. Static
    XAie parsing cannot see the intermediate routing path (XAie_Route decides it
    at runtime), so the channel carries only the direct producer->consumer hop.
    """
    paths = []
    for fi, f in enumerate(model["flows"]):
        src, dst = f["src"], f["dst"]
        direction = "push" if f["direction"] == "S2MM" else "pull"
        producer = {"role": "producer",
                    "tile": {"col": src[0], "row": src[1]},
                    "port_sym": "f%d_prod" % fi}
        channel = {"role": "channel",
                   "hops": [{"from": "(%d,%d)" % src, "to": "(%d,%d)" % dst,
                             "hop_type": None, "shmem_kind": None}]}
        consumer = {"role": "consumer",
                    "tile": {"col": dst[0], "row": dst[1]},
                    "port_sym": "f%d_cons" % fi}
        paths.append({"id": "path%d" % fi, "direction": direction,
                      "stages": [producer, channel, consumer]})
    return {"communication_paths": paths}


def model_is_empty(model):
    return not model["tiles"] and not model["flows"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("host_cc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aie-version", type=int, required=True)
    ap.add_argument("--platform", default="baremetal")
    ap.add_argument("--artifacts-dir")
    a = ap.parse_args(argv)
    with open(a.host_cc) as f:
        raw = f.read()
    model = extract_model(raw, a.aie_version, a.platform == "sim")
    if model_is_empty(model):
        print("[xaiehost2provenance] no XAie routing found - skipping provenance")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    artifacts = collect_kernel_artifacts(model, a.artifacts_dir, a.out_dir)
    with open(os.path.join(a.out_dir, "dfscheduleprovenancemap.json"), "w") as f:
        json.dump(build_dfschedule(model, a.aie_version, artifacts), f, indent=2)
    with open(os.path.join(a.out_dir, "dmaphopprovenacemap.json"), "w") as f:
        json.dump(build_dmaphop(model), f, indent=2)
    print("[xaiehost2provenance] wrote provenance to %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
