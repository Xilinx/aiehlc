import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import controlpan_pmap as c

SAMPLE = """
some unrelated log line
CONTROLPAN-PMAP col=0 row=0 port=SOUTH idx=3 dir=fwd ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1 sw=circuit slot=-1
[aie_runtime] ctrl_route ok: ...
CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=ret ms=slave id=1 sw=pkt slot=0
"""

def test_parse_ports_basic():
    ports = c.parse_ports(SAMPLE)
    assert len(ports) == 3
    assert ports[0] == {"col": 0, "row": 0, "port": "SOUTH", "idx": 3,
                        "dir": "fwd", "ms": "master", "id": 1,
                        "sw": "circuit", "slot": -1,
                        "arb": -1, "msel": -1, "mask": -1}
    assert ports[1]["port"] == "CTRL" and ports[1]["ms"] == "master"
    assert ports[2]["dir"] == "ret" and ports[2]["sw"] == "pkt" and ports[2]["slot"] == 0

def test_parse_ports_legacy_defaults_sw_slot():
    """Old-format line (no sw/slot/arb/msel/mask) still parses, defaulting
    sw=circuit slot=-1 and arb/msel/mask=-1."""
    legacy = "CONTROLPAN-PMAP col=1 row=2 port=EAST idx=0 dir=fwd ms=master id=7\n"
    assert c.parse_ports(legacy) == [{"col": 1, "row": 2, "port": "EAST", "idx": 0,
                                      "dir": "fwd", "ms": "master", "id": 7,
                                      "sw": "circuit", "slot": -1,
                                      "arb": -1, "msel": -1, "mask": -1}]

def test_parse_ports_packet_params():
    """A packet line carries arb/msel/mask (slave slot params)."""
    txt = ("CONTROLPAN-PMAP col=0 row=4 port=SOUTH idx=4 dir=fwd ms=slave id=0 "
           "sw=pkt slot=0 arb=2 msel=1 mask=31\n")
    assert c.parse_ports(txt) == [{"col": 0, "row": 4, "port": "SOUTH", "idx": 4,
                                   "dir": "fwd", "ms": "slave", "id": 0,
                                   "sw": "pkt", "slot": 0,
                                   "arb": 2, "msel": 1, "mask": 31}]

def test_parse_ports_ignores_malformed():
    bad = "CONTROLPAN-PMAP col=x row= port=SOUTH\nCONTROLPAN-PMAP col=1 row=2 port=EAST idx=0 dir=fwd ms=master id=7 sw=pkt slot=2\n"
    ports = c.parse_ports(bad)
    assert ports == [{"col": 1, "row": 2, "port": "EAST", "idx": 0,
                      "dir": "fwd", "ms": "master", "id": 7,
                      "sw": "pkt", "slot": 2,
                      "arb": -1, "msel": -1, "mask": -1}]

SPINE = """
CONTROLPAN-PMAP col=0 row=0 port=NORTH idx=0 dir=fwd ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=1 port=SOUTH idx=0 dir=fwd ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=1 port=NORTH idx=0 dir=fwd ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=2 port=SOUTH idx=0 dir=fwd ms=slave id=1 sw=circuit slot=-1
"""

def test_edges_spine_pairs_north_master_to_south_slave():
    edges = c.parse_edges(SPINE)
    assert {"from": [0, 0], "to": [0, 1], "dir": "fwd", "id": 1, "port": "NORTH",
            "sw": "circuit", "slot": -1, "from_idx": 0, "to_idx": 0} in edges
    assert {"from": [0, 1], "to": [0, 2], "dir": "fwd", "id": 1, "port": "NORTH",
            "sw": "circuit", "slot": -1, "from_idx": 0, "to_idx": 0} in edges
    assert len(edges) == 2

EAST = """
CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=fwd ms=master id=2 sw=pkt slot=1
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=4 dir=fwd ms=slave id=2 sw=pkt slot=1
"""

def test_edges_east_chain_carries_sw_slot_and_idxs():
    edges = c.parse_edges(EAST)
    assert edges == [{"from": [0, 3], "to": [1, 3], "dir": "fwd", "id": 2,
                      "port": "EAST", "sw": "pkt", "slot": 1,
                      "from_idx": 0, "to_idx": 4}]

def test_edges_ctrl_is_not_an_edge():
    ctrl = "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1 sw=circuit slot=-1\n"
    assert c.parse_edges(ctrl) == []

def test_parse_summary_counts():
    r = c.parse(SPINE)
    assert r["count"] == 4 and len(r["edges"]) == 2

# Return route emitted by rt_ctrl_row_return_setup for an off-spine target
# (shim_col=0, row=3, target_col=2, ret_sid=1, vret=3, rport=1): target CTRL slave
# -> WEST master -> WEST pass-through -> head EAST->SOUTH VRET -> vertical VRET
# drain -> shim NORTH->SOUTH S2MM. The shim SOUTH master drains to the S2MM (no
# neighbor slave below row 0), so it forms no edge.
RET_ROUTE = """
CONTROLPAN-PMAP col=2 row=3 port=CTRL idx=0 dir=ret ms=slave id=1 sw=pkt slot=0
CONTROLPAN-PMAP col=2 row=3 port=WEST idx=0 dir=ret ms=master id=1 sw=pkt slot=0
CONTROLPAN-PMAP col=1 row=3 port=EAST idx=0 dir=ret ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=0 dir=ret ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=ret ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=3 dir=ret ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=2 port=NORTH idx=3 dir=ret ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=2 port=SOUTH idx=3 dir=ret ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=1 port=NORTH idx=3 dir=ret ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=1 port=SOUTH idx=3 dir=ret ms=master id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=0 port=NORTH idx=3 dir=ret ms=slave id=1 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=0 port=SOUTH idx=1 dir=ret ms=master id=1 sw=circuit slot=-1
"""

def test_edges_return_route_forms_full_ret_chain():
    edges = c.parse_edges(RET_ROUTE)
    assert all(e["dir"] == "ret" for e in edges)
    hops = {(tuple(e["from"]), tuple(e["to"])) for e in edges}
    # WEST horizontal hops back to the spine column, then the VRET drain down.
    assert ((2, 3), (1, 3)) in hops   # target CTRL turn -> west
    assert ((1, 3), (0, 3)) in hops   # west pass-through to head
    assert ((0, 3), (0, 2)) in hops   # head turns down onto the spine
    assert ((0, 2), (0, 1)) in hops   # vret pass-through
    assert ((0, 1), (0, 0)) in hops   # vret drain into the shim
    # The shim SOUTH master drains to the S2MM; there is no tile below row 0.
    assert not any(e["to"] == [0, -1] for e in edges)
    assert len(edges) == 5


# tile_switch_view: MERGE one tile's ports per direction. A physical slave
# (port,idx) appears ONCE and carries its packet slots; masters cross-match
# every slot by (arb, mselen>>msel). The runtime emits one MASTER_EN line with
# id=pkt_id(=0) per master and one SLOT line per slot on a slave port.
SWITCH = """
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=4 dir=fwd ms=slave  id=0  sw=pkt slot=0 arb=0 msel=0 mask=31
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=4 dir=fwd ms=slave  id=8  sw=pkt slot=1 arb=0 msel=1 mask=24
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=4 dir=fwd ms=slave  id=16 sw=pkt slot=2 arb=0 msel=2 mask=16
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=4 dir=fwd ms=slave  id=0  sw=pkt slot=3 arb=0 msel=3 mask=24
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=4 dir=fwd ms=slave  id=0  sw=pkt slot=-1 arb=-1 msel=-1 mask=-1
CONTROLPAN-PMAP col=0 row=3 port=CTRL  idx=0 dir=fwd ms=master id=0  sw=pkt slot=-1 arb=0 msel=7 mask=-1
CONTROLPAN-PMAP col=0 row=3 port=EAST  idx=0 dir=fwd ms=master id=0  sw=pkt slot=-1 arb=0 msel=10 mask=-1
CONTROLPAN-PMAP col=0 row=3 port=NORTH idx=4 dir=fwd ms=master id=0  sw=pkt slot=-1 arb=0 msel=4 mask=-1
CONTROLPAN-PMAP col=0 row=4 port=SOUTH idx=4 dir=fwd ms=slave  id=0  sw=circuit slot=-1
CONTROLPAN-PMAP col=1 row=3 port=WEST  idx=4 dir=fwd ms=slave  id=0  sw=pkt slot=0 arb=0 msel=0 mask=31
"""

def test_tile_switch_view_merges_physical_slave_port():
    # SOUTH-4 arms four slots but must appear as ONE merged slave port.
    v = c.tile_switch_view(SWITCH, 0, 3)
    assert v["col"] == 0 and v["row"] == 3
    assert len(v["dirs"]) == 1
    d = v["dirs"][0]
    assert d["dir"] == "fwd"
    assert len(d["slaves"]) == 1
    s = d["slaves"][0]
    assert s["port"] == "SOUTH" and s["idx"] == 4 and s["sw"] == "pkt"
    # Slots carry pkt_id and mask (request #1: mask alongside pkt_id).
    assert s["slots"] == [
        {"slot": 0, "pkt_id": 0,  "mask": 31, "msel": 0, "arb": 0},
        {"slot": 1, "pkt_id": 8,  "mask": 24, "msel": 1, "arb": 0},
        {"slot": 2, "pkt_id": 16, "mask": 16, "msel": 2, "arb": 0},
        {"slot": 3, "pkt_id": 0,  "mask": 24, "msel": 3, "arb": 0},
    ]

def test_tile_switch_view_master_dests_and_params():
    v = c.tile_switch_view(SWITCH, 0, 3)
    m = {x["port"]: x for x in v["dirs"][0]["masters"]}
    assert m["CTRL"]["dest"] == "CTRL (local endpoint)"
    assert m["CTRL"]["arb"] == 0 and m["CTRL"]["mselen"] == 7
    assert m["EAST"]["dest"] == "(1,3) WEST"
    assert m["EAST"]["mselen"] == 10
    # NORTH climbs to (0,4), whose SOUTH slave exists -> paired.
    assert m["NORTH"]["dest"] == "(0,4) SOUTH"
    assert m["NORTH"]["mselen"] == 4

def test_tile_switch_view_empty_tile():
    assert c.tile_switch_view(SWITCH, 9, 9) == {"col": 9, "row": 9, "dirs": []}

def test_tile_switch_view_unpaired_master_dest_dash():
    # EAST master with no neighbor slave -> dest "\u2014"
    txt = "CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=fwd ms=master id=0 sw=pkt slot=-1 arb=0 msel=2\n"
    v = c.tile_switch_view(txt, 0, 3)
    assert v["dirs"][0]["masters"][0]["dest"] == "\u2014"

def test_tile_switch_view_splits_fwd_ret():
    # Same tile, same port, different dir -> two direction sections.
    txt = (
        "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=fwd ms=master id=1 sw=pkt slot=0\n"
        "CONTROLPAN-PMAP col=0 row=3 port=CTRL idx=0 dir=ret ms=slave  id=1 sw=pkt slot=0\n"
    )
    v = c.tile_switch_view(txt, 0, 3)
    assert {d["dir"] for d in v["dirs"]} == {"fwd", "ret"}

# A circuit direction (no slots): slaves carry an id and link straight to the
# master with the matching id; the physical port still merges once.
CIRCUIT = """
CONTROLPAN-PMAP col=0 row=3 port=EAST  idx=0 dir=ret ms=slave  id=0 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=NORTH idx=3 dir=ret ms=slave  id=0 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=3 dir=ret ms=master id=0 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=3 port=SOUTH idx=3 dir=ret ms=master id=0 sw=circuit slot=-1
CONTROLPAN-PMAP col=0 row=2 port=NORTH idx=3 dir=ret ms=slave  id=0 sw=circuit slot=-1
"""

def test_tile_switch_view_circuit_merges_and_has_no_slots():
    v = c.tile_switch_view(CIRCUIT, 0, 3)
    d = v["dirs"][0]
    assert d["dir"] == "ret"
    # Two distinct circuit slave ports, each merged once; no packet slots.
    assert sorted((s["port"], s["idx"]) for s in d["slaves"]) == [("EAST", 0), ("NORTH", 3)]
    for s in d["slaves"]:
        assert s["slots"] == [] and s["sw"] == "circuit" and s["id"] == 0
    # The duplicated SOUTH-3 master lines merge into one master.
    assert len(d["masters"]) == 1
    m = d["masters"][0]
    assert m["port"] == "SOUTH" and m["idx"] == 3 and m["sw"] == "circuit" and m["id"] == 0
    assert m["dest"] == "(0,2) NORTH"

# One neighbor port pair (EAST<->WEST) serves two flows with different (dir,id)
# and different slave idx; each edge must carry its own flow's to_idx.
SHARED = """
CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=fwd ms=master id=1 sw=pkt slot=1
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=0 dir=fwd ms=slave  id=1 sw=pkt slot=1
CONTROLPAN-PMAP col=0 row=3 port=EAST idx=0 dir=ret ms=master id=2 sw=pkt slot=0
CONTROLPAN-PMAP col=1 row=3 port=WEST idx=5 dir=ret ms=slave  id=2 sw=pkt slot=0
"""

def test_edges_shared_port_disambiguated_by_dir_id():
    edges = c.parse_edges(SHARED)
    fwd = [e for e in edges if e["dir"] == "fwd" and e["id"] == 1]
    ret = [e for e in edges if e["dir"] == "ret" and e["id"] == 2]
    assert fwd and fwd[0]["to_idx"] == 0
    assert ret and ret[0]["to_idx"] == 5
