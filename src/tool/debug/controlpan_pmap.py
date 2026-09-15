"""Parse CONTROLPAN-PMAP applog lines into device-map ports and edges.

A CONTROLPAN-PMAP line (emitted by the aie_ctrl* runtime when
__Runtime_ctrl_pmap_enable is on, or AIE_CTRL_PMAP is set) describes one
programmed stream port:

    CONTROLPAN-PMAP col=<c> row=<r> port=<SOUTH|NORTH|CTRL|EAST|WEST> idx=<n> \
        dir=<fwd|ret> ms=<master|slave> id=<id> sw=<pkt|circuit> slot=<n> \
        arb=<n> msel=<n> mask=<n>

sw/slot/arb/msel/mask are optional (legacy lines default sw=circuit, slot=-1 and
arb/msel/mask=-1). arb/msel/mask are the AIE stream-switch packet-routing params:
a packet slave slot carries (mask, msel, arbiter); a packet master port carries
(arbiter, mselen, emitted as msel). Circuit ports emit -1 for all three.
parse_ports() returns the port dicts; parse_edges() pairs master ports with the
slave port on the adjacent tile to synthesize device-map routing edges carrying
the master hop's sw/slot and both port indices.
"""
import re

_TAG = "CONTROLPAN-PMAP"
_KV = re.compile(r"(\w+)=(\S+)")
_INT_KEYS = ("col", "row", "idx", "id", "slot", "arb", "msel", "mask")
_REQUIRED = ("col", "row", "port", "idx", "dir", "ms", "id")
_OPTIONAL = ("sw", "slot", "arb", "msel", "mask")

# Master-port direction -> neighbor tile delta (col,row). AIE grid: row 0 = shim
# at the bottom; NORTH points up (+row), SOUTH down (-row), EAST +col, WEST -col.
_DELTA = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
_OPP = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


def parse_ports(text):
    ports = []
    for line in text.splitlines():
        i = line.find(_TAG)
        if i < 0:
            continue
        kv = dict(_KV.findall(line[i + len(_TAG):]))
        if not set(_REQUIRED) <= set(kv):
            continue
        # sw/slot/arb/msel/mask are optional (legacy lines omit them).
        kv.setdefault("sw", "circuit")
        kv.setdefault("slot", "-1")
        kv.setdefault("arb", "-1")
        kv.setdefault("msel", "-1")
        kv.setdefault("mask", "-1")
        try:
            rec = {k: (int(kv[k]) if k in _INT_KEYS else kv[k])
                   for k in _REQUIRED + _OPTIONAL}
        except ValueError:
            continue
        ports.append(rec)
    return ports


def parse_edges(text):
    """Pair each master port with the opposite-direction slave port on the
    neighbor tile. CTRL master/slave are tile-local consume/emit markers (no
    edge). Returns a list of {from:[c,r], to:[c,r], dir, id, port} edges."""
    ports = parse_ports(text)
    # Key slaves by (col,row,port,dir,id): one neighbor port can serve several
    # flows (e.g. EAST<->WEST for both a fwd and a ret route), so keying on the
    # port alone would keep only the last slave and bind the wrong to_idx.
    slaves = {(p["col"], p["row"], p["port"], p["dir"], p["id"]): p
              for p in ports if p["ms"] == "slave"}
    edges = []
    for p in ports:
        if p["ms"] != "master" or p["port"] not in _DELTA:
            continue
        dc, dr = _DELTA[p["port"]]
        nb = (p["col"] + dc, p["row"] + dr, _OPP[p["port"]], p["dir"], p["id"])
        if nb in slaves:
            edges.append({"from": [p["col"], p["row"]],
                          "to": [nb[0], nb[1]], "dir": p["dir"],
                          "id": p["id"], "port": p["port"],
                          "sw": p["sw"], "slot": p["slot"],
                          "from_idx": p["idx"], "to_idx": slaves[nb]["idx"]})
    return edges


def parse(text):
    """Convenience: {ports, edges, count}."""
    ports = parse_ports(text)
    return {"ports": ports, "edges": parse_edges(text), "count": len(ports)}


def tile_switch_view(text, col, row):
    """Merge one tile's CONTROLPAN-PMAP ports into per-direction switch views.

    Ports are grouped by DIRECTION only (fwd/ret). Within a direction each
    physical port (port,idx) is MERGED to a single node -- a slave input arms
    one or more packet slots on that one physical port, so the runtime's
    one-line-per-slot emission collapses back to a single slave box carrying a
    list of slots. Masters likewise merge by (port,idx).

    Per AIE stream-switch packet routing the routing params live on the SLOT
    (configured on the slave port): a slot carries (pkt_id, mask, msel,
    arb=arbiter) and matches a header when (incoming_id & mask) == (pkt_id &
    mask), injecting into arbiter `arb` with select `msel`. Each MASTER carries
    (arb=arbiter, mselen), a BITMASK of accepted msel values; a master pulls a
    slot iff master.arb == slot.arb and ((master.mselen >> slot.msel) & 1). The
    caller draws a slot->master link for every such match (this is request #3:
    a master connects to ANY slot on ANY physical slave whose msel bit is set in
    its mselen, not just same-id slots). Circuit ports carry no slot; a circuit
    slave links straight to the circuit master with the matching id.

    A master destination is the neighbor tile it feeds -- "(c,r) OPP" when that
    neighbor tile has a slave on the opposite port in the same direction, else a
    dash; CTRL is the tile-local endpoint. Pairing is by (port,idx,dir) and
    ignores id, because a packet master (id=pkt_id=0) feeds a whole physical
    neighbor slave port that arms several pkt_ids.

    Returns {col, row, dirs:[{dir,
        slaves:[{port,idx,sw,id,slots:[{slot,pkt_id,mask,msel,arb}]}],
        masters:[{port,idx,dest,sw,id,arb,mselen}]}]}.
    """
    ports = [p for p in parse_ports(text)
             if p["col"] == col and p["row"] == row]
    # Every slave port anywhere -> for neighbor-dest pairing (id-agnostic).
    all_slaves = {(p["col"], p["row"], p["port"], p["dir"])
                  for p in parse_ports(text) if p["ms"] == "slave"}

    def dest_for(port, idx, dir_):
        if port == "CTRL":
            return "CTRL (local endpoint)"
        if port not in _DELTA:
            return "\u2014"
        dc, dr = _DELTA[port]
        nb = (col + dc, row + dr, _OPP[port], dir_)
        return "({},{}) {}".format(col + dc, row + dr, _OPP[port]) \
            if nb in all_slaves else "\u2014"

    dirs = {}
    for p in ports:
        d = dirs.setdefault(p["dir"], {"dir": p["dir"], "_sl": {}, "_ms": {}})
        pk = (p["port"], p["idx"])
        if p["ms"] == "slave":
            s = d["_sl"].setdefault(pk, {"port": p["port"], "idx": p["idx"],
                                         "sw": "circuit", "id": p["id"],
                                         "_slots": {}})
            if p["sw"] == "pkt":
                s["sw"] = "pkt"
            # A real slot line carries params (slot>=0, arb>=0); bare enable
            # lines (slot<0 / arb<0) only mark the port, they add no slot node.
            if p["slot"] >= 0 and p["arb"] >= 0:
                s["_slots"][p["slot"]] = {"slot": p["slot"], "pkt_id": p["id"],
                                          "mask": p["mask"], "msel": p["msel"],
                                          "arb": p["arb"]}
        else:
            m = d["_ms"].setdefault(pk, {"port": p["port"], "idx": p["idx"],
                                         "dest": dest_for(p["port"], p["idx"],
                                                          p["dir"]),
                                         "sw": "circuit", "id": p["id"],
                                         "arb": -1, "mselen": -1})
            if p["sw"] == "pkt":
                m["sw"] = "pkt"
            # MASTER_EN line carries (arb, mselen-as-msel); bare enable arb<0.
            if p["arb"] >= 0 and m["arb"] < 0:
                m["arb"], m["mselen"] = p["arb"], p["msel"]

    out = []
    for d in sorted(dirs.values(), key=lambda x: x["dir"]):
        slaves = []
        for s in sorted(d["_sl"].values(), key=lambda x: (x["port"], x["idx"])):
            slots = [s["_slots"][k] for k in sorted(s["_slots"])]
            slaves.append({"port": s["port"], "idx": s["idx"], "sw": s["sw"],
                           "id": s["id"], "slots": slots})
        masters = sorted(d["_ms"].values(), key=lambda x: (x["port"], x["idx"]))
        out.append({"dir": d["dir"], "slaves": slaves, "masters": masters})
    return {"col": col, "row": row, "dirs": out}
