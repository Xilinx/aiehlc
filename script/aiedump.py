#!/usr/bin/env python3
"""Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0

aiedump: dump the AIE array registers of a lab board over JTAG (hw_server/xsdb),
optionally with every core tile's program memory (PM) and data memory (DM).

    script/aiedump.py palmyra_local                  # registers of every live column
    script/aiedump.py vek385-12 --pm --dm            # + core-tile PM/DM binaries
    script/aiedump.py palmyra_local --cols 3-6 --rows 3-6 --core-mem -o /tmp/d

Board names: ``palmyra_local`` (the palmyra on 10.23.224.213) or ``vek385-N`` (a
VEK385 farm board, resolved to its farm host with ``cluster-ping``). Any other
name is taken as a host/IP; pass ``--type palmyra|vek385``.

hw_server: a plain TCP probe of <host>:3121 first (instant). If it is closed, or
xsdb cannot connect, ``hws.ensure()`` (script/test/hws.py, the library behind
apppaltest.py's preflight) starts/recovers it under the board lock.
``--no-start`` only reports.

Why it is fast (aiedbg's xsdb backend spawns one xsdb per operation and reads
one word per ``mrd``):
  * one xsdb process, one ``connect``, fed through a pipe for the whole dump;
  * the ``Versal*`` device target (DAP memory access): no APU/RPU halt, works
    while an app is running;
  * named registers merged into a few block ``mrd``s per tile (holes in a live
    column read 0); one ``mrd`` costs ~2.5 ms round trip, a 64 KB block ~150 ms;
  * columns outside the live partition (every word reads 0x00000002) are found
    with one read per column and skipped.

Never read (read side effects): ``lock_request`` (a read is a lock acquire),
``*_fifo_pop``, ``mdm_dbg_data``, and the memory windows (dumped only with
--pm / --dm / --memtile-dm). No block read crosses one of these windows.

Register names come from aiedbg's aie2ps register JSON (``pip install`` aiedbg,
``script/debug/bootstrap_aiedbg.sh``, or ``AIEDUMP_REGDB=<dir>``).

Output (``-o``, default ``./aiedump/<board>_<YYYYmmdd_HHMMSS>``):
  summary.txt    live columns, core status/PC, DMA channel status per tile
  regs.txt       ``col row module register offset value`` for every dumped register
  regs.json      {"c<col>_r<row>": {"<module>.<register>": value}}
  mem/           c<col>_r<row>_pm.bin / _dm.bin / _memtile_dm.bin (raw little-endian)
  meta.json      board, host, columns, timings
"""
import argparse
import dataclasses
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "test"))

HWS_PORT = 3121
XSDB = (os.environ.get("AIEDUMP_XSDB") or os.environ.get("HWS_XSDB")
        or "/proj/custbuilds/2026.2_daily_latest/installs/lin64/2026.2/Vitis/bin/xsdb")
CLUSTER_PING = "/proj/systest/bin/cluster-ping"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "aiedump_boards.json")
BOARD_ALIASES = {"palmyra_local": ("10.23.224.213", "palmyra")}

# AIE2PS (Gen5) array: include/aie_device_map.h -- 36 x 7, shim row 0, memtiles 1-2.
BASE, COL_SHIFT, ROW_SHIFT = 0x20000000000, 25, 20
NUM_COLS, NUM_ROWS, MEM_ROW0, CORE_ROW0 = 36, 7, 1, 3
GATED_WORD = 0x00000002            # every word of a column outside the live partition
PROBE_OFF, PROBE_WORDS = 0x38000, 8  # core module control/status block (core row)

# (lo, hi) byte windows per tile kind that the register dump must never touch.
MEMS = {
    "core": {"dm": (0x00000, 0x10000), "pm": (0x20000, 0x24000)},
    "mem": {"memtile_dm": (0x00000, 0x80000)},
    "shim": {},
}
FORBIDDEN = {
    "core": [(0x00000, 0x10000), (0x20000, 0x24000), (0x1DF20, 0x1DF28), (0x40000, 0x50000)],
    "mem": [(0x00000, 0x80000), (0xA06C8, 0xA06E0), (0xD0000, 0xE0000)],
    "shim": [(0x09338, 0x09340), (0x0C000, 0x10000), (0x80000, 0xB0000),
             (0xB0014, 0xB0018), (0xD0000, 0x100000)],
}
MERGE_GAP = 0x400   # bytes: read a hole this small rather than pay another round trip

# XAIE_CORE_STATUS_*_BIT (aielib xaie_util_status.h)
CORE_STATUS_BITS = {0: "Enable", 1: "Reset", 2: "MemStall_S", 3: "MemStall_W", 4: "MemStall_N",
                    5: "MemStall_E", 6: "LockStall_S", 7: "LockStall_W", 8: "LockStall_N",
                    9: "LockStall_E", 10: "StreamStall_SS0", 12: "StreamStall_MS0",
                    14: "CascadeStall_SCD", 15: "CascadeStall_MCD", 16: "DebugHalt",
                    17: "ECC_ErrStall", 18: "ECC_ScrubStall", 19: "ErrorHalt", 20: "Done",
                    21: "ProcBusStall"}


class DumpError(Exception):
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} aiedump] {msg}", flush=True)


def tile_kind(row):
    return "shim" if row == 0 else ("mem" if row < CORE_ROW0 else "core")


def tile_addr(col, row, off):
    return BASE + (col << COL_SHIFT) + (row << ROW_SHIFT) + off


# ---------------------------------------------------------------------------
# Board -> host
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Board:
    name: str
    host: str
    type: str


def _farm_hosts(refresh=False):
    """{"vek385-12": "crimini2", ...} from cluster-ping, cached for a day."""
    if not refresh:
        try:
            with open(CACHE) as f:
                c = json.load(f)
            if time.time() - c.get("time", 0) < 86400:
                return c["boards"]
        except (OSError, ValueError, KeyError):
            pass
    import hws_vek385
    out = subprocess.run([CLUSTER_PING], capture_output=True, text=True, timeout=120).stdout
    boards = {vb: host for host, (vb, _holder, _job) in hws_vek385.parse_cluster_ping(out).items()}
    if boards:
        try:
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump({"time": time.time(), "boards": boards}, f)
        except OSError:
            pass
    return boards


def resolve_board(name, host=None, btype=None):
    key = name.lower()
    if key in BOARD_ALIASES:
        h, t = BOARD_ALIASES[key]
        return Board(name, host or h, btype or t)
    if re.fullmatch(r"vek385-\d+", key):
        if host:
            return Board(name, host, "vek385")
        h = _farm_hosts().get(key) or _farm_hosts(refresh=True).get(key)
        if not h:
            raise DumpError(f"{name}: not in cluster-ping (pass --host <farm host>)")
        return Board(name, h, "vek385")
    import hws
    try:
        return Board(name, host or name, btype or hws.board_type(host or name))
    except hws.HwsError as e:
        raise DumpError(f"{name}: unknown board ({e.msg}); use palmyra_local, vek385-N "
                        "or a host with --type") from None


# ---------------------------------------------------------------------------
# hw_server
# ---------------------------------------------------------------------------
def port_open(host, port=HWS_PORT, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_hw_server(board, start=True, wait_lock=600):
    """Fast path: 3121 answers. Otherwise hws.ensure (apppaltest's preflight library)."""
    if port_open(board.host):
        log(f"hw_server {board.host}:{HWS_PORT} is up")
        return
    if not start:
        raise DumpError(f"hw_server {board.host}:{HWS_PORT} is not running (--no-start)")
    log(f"hw_server {board.host}:{HWS_PORT} is not running; starting it (hws.ensure, {board.type})")
    import hws
    try:
        h = hws.ensure(board.host, board.type, wait_lock=wait_lock)
    except hws.HwsError as e:
        raise DumpError(f"cannot start hw_server on {board.host}: rc={e.rc} {e.msg}") from None
    log(f"hw_server {h.state}: {h.msg}")


# ---------------------------------------------------------------------------
# Register database (aiedbg's aie2ps JSON)
# ---------------------------------------------------------------------------
def _regdb_dirs():
    if os.environ.get("AIEDUMP_REGDB"):
        yield os.environ["AIEDUMP_REGDB"]
    try:
        from aiedbg.device import regdata
        yield str(regdata.data_dir("aie2ps", require=False))
    except Exception:  # noqa: BLE001 -- any import failure means "not installed"
        pass
    for rel in ("thirdparty/aiedbg/src/aiedbg/device/data/aie2ps",
                "aiedbg/src/aiedbg/device/data/aie2ps", "aiedbg/device/aie2ps"):
        yield os.path.join(REPO, rel)


def find_regdb():
    for d in _regdb_dirs():
        if os.path.isfile(os.path.join(d, "aie2pstile.json")):
            return d
    raise DumpError("aie2ps register JSON not found: install aiedbg "
                    "(script/debug/bootstrap_aiedbg.sh) or set AIEDUMP_REGDB=<dir>")


def _forbidden(kind, off):
    return any(lo <= off < hi for lo, hi in FORBIDDEN[kind])


def _merge(regs, kind):
    """Sorted register offsets -> [(lo, nwords)] block reads; never spans a forbidden window."""
    ranges = []
    for off in sorted({o for _m, _n, o in regs}):
        if ranges:
            lo, hi = ranges[-1]
            gap = range(hi, off, 4)
            if off - hi <= MERGE_GAP and not any(_forbidden(kind, a) for a in gap):
                ranges[-1][1] = off + 4
                continue
        ranges.append([off, off + 4])
    return [(lo, (hi - lo) // 4) for lo, hi in ranges]


def load_regdb(path):
    """{kind: {"regs": [(module, name, off)], "ranges": [...], "fields": {(module,name): [...]}}}"""
    db = {}
    for kind, fname in (("core", "tile"), ("mem", "mem"), ("shim", "shim")):
        with open(os.path.join(path, f"aie2ps{fname}.json")) as f:
            data = json.load(f)
        regs = []
        for m in data["modules"]:
            mod = m["module_name"].replace("aie2ps_", "")
            for r in m["registers"]:
                off = int(r["register_address"], 16)
                if not _forbidden(kind, off):
                    regs.append((mod, r["register_name"].lower(), off))
        db[kind] = {"regs": regs, "ranges": _merge(regs, kind), "fields": {}}
    for kind, fname in (("core", "tiledmastatus"), ("shim", "shimdmastatus")):
        for p in glob.glob(os.path.join(path, f"aie2ps{fname}.json")):
            with open(p) as f:
                data = json.load(f)
            for r in data.get("registers", []):
                db[kind]["fields"][r["register_name"].lower()] = r.get("fields", [])
    return db


# ---------------------------------------------------------------------------
# One persistent xsdb session
# ---------------------------------------------------------------------------
class Xsdb:
    """xsdb fed through a pipe; every request ends with a unique marker line."""

    def __init__(self, path=XSDB):
        if not os.access(path, os.X_OK):
            raise DumpError(f"xsdb not found: {path} (set AIEDUMP_XSDB)")
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.seq = 0

    def run(self, tcl):
        self.seq += 1
        mark = f"@@AIEDUMP_DONE {self.seq}"
        self.p.stdin.write(f"{tcl}\nputs \"{mark}\"\n")
        self.p.stdin.flush()
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise DumpError("xsdb exited:\n" + "".join(out[-20:]))
            if line.strip() == mark:
                return out
            out.append(line.rstrip("\n"))

    def connect(self, host, port=HWS_PORT):
        out = self.run(f"if {{[catch {{connect -url TCP:{host}:{port}}} e]}} "
                       "{puts \"@@ERR [string map {\\n { }} $e]\"}")
        err = [l for l in out if l.startswith("@@ERR")]
        if err:
            raise DumpError(f"xsdb connect {host}:{port}: {err[0][6:]}")
        for flt in ("Versal*", "DPC", "PMC"):
            out = self.run(f"if {{[catch {{targets -set -nocase -filter {{name =~ \"{flt}\"}}}} e]}} "
                           "{puts \"@@ERR $e\"} else {puts \"@@OK\"}")
            if "@@OK" in out:
                return flt
        raise DumpError(f"no Versal/DPC/PMC target on {host}:{port}")

    def read_blocks(self, reqs):
        """[(addr, nwords)] -> {addr: [words] | error string}; one pipe round trip."""
        tcl = "\n".join(
            f"if {{[catch {{mrd -force -value 0x{a:x} {n}}} v]}} "
            f"{{puts \"E 0x{a:x} [string map {{\\n {{ }}}} $v]\"}} else {{puts \"R 0x{a:x} $v\"}}"
            for a, n in reqs)
        res = {}
        for line in self.run(tcl):
            m = re.match(r"([RE]) 0x([0-9a-f]+) ?(.*)", line)
            if m:
                a = int(m.group(2), 16)
                res[a] = [int(v) & 0xFFFFFFFF for v in m.group(3).split()] if m.group(1) == "R" \
                    else m.group(3)
        return res

    def read_to_file(self, addr, nwords, path):
        out = self.run(f"if {{[catch {{mrd -force -bin -file {{{path}}} 0x{addr:x} {nwords}}} e]}} "
                       "{puts \"@@ERR [string map {\\n { }} $e]\"}")
        err = [l for l in out if l.startswith("@@ERR")]
        return err[0][6:] if err else None

    def close(self):
        try:
            self.p.stdin.write("catch {disconnect}\nexit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
def parse_span(spec, limit):
    """"3-6,8" / "all" -> sorted list within [0, limit)."""
    if spec in (None, "all", "auto", "open"):
        return list(range(limit))
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    bad = [v for v in out if not 0 <= v < limit]
    if bad:
        raise DumpError(f"{spec}: out of range 0..{limit - 1}")
    return sorted(out)


def probe_points(db):
    """[(row, offset, nwords)] read per column: the core control/status block, plus the
    memtile and shim DMA channel status blocks (so a partition whose cores are not yet
    enabled still shows up)."""
    pts = [(CORE_ROW0, PROBE_OFF, PROBE_WORDS)]
    for kind, row in (("mem", MEM_ROW0), ("shim", 0)):
        offs = [o for _m, n, o in (db or {}).get(kind, {}).get("regs", [])
                if re.fullmatch(r"dma_(s2mm|mm2s)_status_\d+", n)]
        if offs:
            pts.append((row, min(offs), (max(offs) - min(offs)) // 4 + 1))
    return pts


def probe_columns(x, cols, db):
    """{col: "live" | "idle" | "gated" | "error: .."}. gated = every probed memtile/core
    word reads 0x00000002 (outside the live partition; its shim row still answers);
    idle = accessible but all zero."""
    pts = probe_points(db)
    res = x.read_blocks([(tile_addr(c, r, o), n) for c in cols for r, o, n in pts])
    state = {}
    for c in cols:
        words, array, err = [], [], None
        for r, o, n in pts:
            v = res.get(tile_addr(c, r, o), "no reply")
            if isinstance(v, str):
                err = err or v
                continue
            words += v
            if r > 0:  # the shim row of a gated column stays accessible
                array += v
        if err and not words:
            state[c] = f"error: {err}"
        elif array and all(w == GATED_WORD for w in array):
            state[c] = "gated"
        else:
            state[c] = "live" if any(words) else "idle"
    return state


def dump_tile_regs(x, db, col, row):
    """{(module, name): value | None} for one tile; None = the block read failed."""
    kind = tile_kind(row)
    ranges = db[kind]["ranges"]
    res = x.read_blocks([(tile_addr(col, row, lo), n) for lo, n in ranges])
    words, errors = {}, []
    for lo, n in ranges:
        v = res.get(tile_addr(col, row, lo))
        if isinstance(v, list) and len(v) == n:
            for i, w in enumerate(v):
                words[lo + 4 * i] = w
        else:
            errors.append(f"0x{lo:x}+{n}: {v}")
    vals = {(m, name): words.get(off) for m, name, off in db[kind]["regs"]}
    return vals, errors


def dump_tile_mems(x, col, row, which, memdir):
    kind = tile_kind(row)
    done = []
    for mem, (lo, hi) in MEMS[kind].items():
        if mem not in which:
            continue
        path = os.path.join(memdir, f"c{col}_r{row}_{mem}.bin")
        err = x.read_to_file(tile_addr(col, row, lo), (hi - lo) // 4, path)
        done.append((mem, path, err))
    return done


def decode_fields(value, fields):
    out = []
    for f in fields:
        hi, _, lo = f["bits"].partition(":")
        hi, lo = int(hi), int(lo or hi)
        v = (value >> lo) & ((1 << (hi - lo + 1)) - 1)
        if v:
            out.append(f"{f['field_name']}={v}")
    return out


def tile_summary(db, col, row, vals):
    kind = tile_kind(row)
    by_name = {n: v for (_m, n), v in vals.items() if v is not None}
    parts = []
    if kind == "core":
        cs = by_name.get("core_status")
        if cs is not None:
            flags = [n for b, n in CORE_STATUS_BITS.items() if cs >> b & 1]
            parts.append(f"core_status=0x{cs:08x} [{' '.join(flags) or '-'}]")
        if "core_pc" in by_name:
            parts.append(f"pc=0x{by_name['core_pc']:x}")
    for name in sorted(n for n in by_name if re.fullmatch(r"dma_(s2mm|mm2s)_status_\d+", n)):
        v = by_name[name]
        if v:
            dec = decode_fields(v, db[kind]["fields"].get(name, []))
            parts.append(f"{name}=0x{v:08x}" + (f" [{' '.join(dec)}]" if dec else ""))
    return f"c{col:<2} r{row} {kind:<4} " + ("  ".join(parts) or "idle")


def write_outputs(out, board, regs, summaries, meta):
    with open(os.path.join(out, "regs.txt"), "w") as f:
        f.write("# col row module register offset value\n")
        for (c, r), vals in sorted(regs.items()):
            for (m, n), v in sorted(vals.items(), key=lambda kv: kv[0]):
                f.write(f"{c} {r} {m} {n} 0x{_off(meta, c, r, m, n):05x} "
                        f"{'ERR' if v is None else f'0x{v:08x}'}\n")
    with open(os.path.join(out, "regs.json"), "w") as f:
        json.dump({f"c{c}_r{r}": {f"{m}.{n}": v for (m, n), v in sorted(vals.items())}
                   for (c, r), vals in sorted(regs.items())}, f, indent=1)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(f"# aiedump {board.name} ({board.host}) {meta['time']}\n")
        f.write("\n".join(summaries) + "\n")
    meta.pop("_offsets", None)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def _off(meta, c, r, m, n):
    return meta["_offsets"][tile_kind(r)][(m, n)]


def run_dump(x, db, board, args, out):
    t0 = time.time()
    cols = parse_span(args.cols, NUM_COLS)
    rows = parse_span(args.rows, NUM_ROWS)
    state = probe_columns(x, cols, db)
    live = [c for c in cols if state[c] == "live"]
    idle = [c for c in cols if state[c] == "idle"]
    gated = [c for c in cols if state[c] == "gated"]
    log(f"columns: live {live or '-'}; idle (all zero) {idle or '-'}; "
        f"gated (0x00000002) {gated or '-'}"
        + "".join(f"; col {c} {s}" for c, s in state.items() if s.startswith("error")))
    targets = {"auto": live, "open": sorted(live + idle)}.get(args.cols, cols)
    if args.cols in ("auto", "open") and not targets:
        raise DumpError("no live AIE column (partition not initialized? run an app, "
                        "or force columns with --cols)")
    which = set(args.mem)
    memdir = os.path.join(out, "mem")
    if which:
        os.makedirs(memdir, exist_ok=True)
    regs, summaries, mems, errs = {}, [], [], []
    t_reg = t_mem = 0.0
    for c in targets:
        for r in rows:
            t = time.time()
            if not args.no_regs:
                vals, e = dump_tile_regs(x, db, c, r)
                regs[(c, r)] = vals
                errs += [f"c{c} r{r} {s}" for s in e]
                summaries.append(tile_summary(db, c, r, vals))
            t_reg += time.time() - t
            t = time.time()
            if which:
                for mem, path, err in dump_tile_mems(x, c, r, which, memdir):
                    mems.append({"tile": f"c{c}_r{r}", "mem": mem, "file": os.path.relpath(path, out),
                                 "error": err})
                    if err:
                        errs.append(f"c{c} r{r} {mem}: {err}")
            t_mem += time.time() - t
        log(f"col {c} done ({time.time() - t0:.1f}s)")
    return {"state": state, "dumped_cols": targets, "rows": rows, "regs": regs,
            "summaries": summaries, "mems": mems, "errors": errs,
            "t_regs_s": round(t_reg, 2), "t_mem_s": round(t_mem, 2)}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[2])
    ap.add_argument("board", help="palmyra_local | vek385-N | host/IP (with --type)")
    ap.add_argument("--host", help="override the host serving hw_server (skips cluster-ping)")
    ap.add_argument("--type", choices=("palmyra", "vek385"), help="board type for a raw host")
    ap.add_argument("--cols", default="auto",
                    help="columns: auto (live ones, default), open (live + idle), all, or e.g. 3-6,8")
    ap.add_argument("--rows", default="all", help="rows: all (default) or e.g. 3-6")
    ap.add_argument("--pm", dest="mem", action="append_const", const="pm",
                    help="also dump each core tile's program memory (16 KB)")
    ap.add_argument("--dm", dest="mem", action="append_const", const="dm",
                    help="also dump each core tile's data memory (64 KB)")
    ap.add_argument("--core-mem", action="store_true", help="= --pm --dm")
    ap.add_argument("--memtile-dm", dest="mem", action="append_const", const="memtile_dm",
                    help="also dump each memtile's data memory (512 KB, ~1.3 s per tile)")
    ap.add_argument("--no-regs", action="store_true", help="memories only")
    ap.add_argument("--no-start", action="store_true", help="do not start hw_server if it is down")
    ap.add_argument("--wait-lock", type=int, default=600,
                    help="seconds to wait for the board lock when hw_server must be started")
    ap.add_argument("-o", "--out", help="output directory")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.mem = list(dict.fromkeys((args.mem or []) + (["pm", "dm"] if args.core_mem else [])))
    t0 = time.time()
    try:
        board = resolve_board(args.board, args.host, args.type)
        log(f"board {board.name} -> {board.host} ({board.type})")
        db = None if args.no_regs else load_regdb(find_regdb())
        ensure_hw_server(board, start=not args.no_start, wait_lock=args.wait_lock)
        out = os.path.abspath(args.out or os.path.join(
            "aiedump", f"{board.name}_{time.strftime('%Y%m%d_%H%M%S')}"))
        os.makedirs(out, exist_ok=True)
        x = Xsdb()
        try:
            t_c = time.time()
            target = x.connect(board.host)
            t_conn = time.time() - t_c
            log(f"connected ({t_conn:.1f}s, target {target})")
            res = run_dump(x, db, board, args, out)
        finally:
            x.close()
    except DumpError as e:
        log(f"ERROR: {e}")
        return 1
    meta = {"board": board.name, "host": board.host, "type": board.type,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "columns": res["state"],
            "dumped_cols": res["dumped_cols"], "rows": res["rows"], "mems": res["mems"],
            "errors": res["errors"], "connect_s": round(t_conn, 2), "regs_s": res["t_regs_s"],
            "mem_s": res["t_mem_s"], "total_s": round(time.time() - t0, 2),
            "_offsets": {k: {(m, n): o for m, n, o in v["regs"]} for k, v in (db or {}).items()}}
    write_outputs(out, board, res["regs"], res["summaries"], meta)
    for s in res["summaries"]:
        print("  " + s)
    nreg = sum(len(v) for v in res["regs"].values())
    log(f"{len(res['regs'])} tiles, {nreg} registers, {len(res['mems'])} memories, "
        f"{len(res['errors'])} errors; regs {res['t_regs_s']}s, mem {res['t_mem_s']}s, "
        f"total {time.time() - t0:.1f}s -> {out}")
    for e in res["errors"][:10]:
        log(f"  {e}")
    return 2 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
