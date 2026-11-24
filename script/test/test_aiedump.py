###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""script/aiedump.py without a board: board names, hw_server start, block merging, probe.

hws, cluster-ping and xsdb are replaced by fakes.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import aiedump as ad  # noqa: E402
import hws  # noqa: E402


def test_palmyra_local_alias():
    b = ad.resolve_board("palmyra_local")
    assert (b.host, b.type) == ("10.23.224.213", "palmyra")


def test_vek385_name_via_cluster_ping(monkeypatch):
    monkeypatch.setattr(ad, "_farm_hosts", lambda refresh=False: {"vek385-12": "crimini2"})
    b = ad.resolve_board("vek385-12")
    assert (b.host, b.type) == ("crimini2", "vek385")
    with pytest.raises(ad.DumpError, match="not in cluster-ping"):
        ad.resolve_board("vek385-99")


def test_vek385_host_override_skips_cluster_ping(monkeypatch):
    monkeypatch.setattr(ad, "_farm_hosts", lambda refresh=False: pytest.fail("cluster-ping called"))
    assert ad.resolve_board("vek385-3", host="gomphus3").host == "gomphus3"


def test_raw_host_needs_type():
    assert ad.resolve_board("10.23.224.213").type == "palmyra"
    with pytest.raises(ad.DumpError, match="unknown board"):
        ad.resolve_board("somehost")
    assert ad.resolve_board("somehost", btype="vek385").type == "vek385"


def test_hw_server_up_does_not_call_hws(monkeypatch):
    monkeypatch.setattr(ad, "port_open", lambda host, *a, **k: True)
    monkeypatch.setattr(hws, "ensure", lambda *a, **k: pytest.fail("hws.ensure called"))
    ad.ensure_hw_server(ad.Board("palmyra_local", "10.23.224.213", "palmyra"))


def test_hw_server_down_is_started_through_hws(monkeypatch):
    calls = []
    monkeypatch.setattr(ad, "port_open", lambda host, *a, **k: False)
    monkeypatch.setattr(hws, "ensure", lambda host, btype, **kw: calls.append((host, btype, kw))
                        or hws.Health(host, btype, hws.UP, {}, "up"))
    ad.ensure_hw_server(ad.Board("vek385-12", "crimini2", "vek385"), wait_lock=5)
    assert calls == [("crimini2", "vek385", {"wait_lock": 5})]


def test_hw_server_down_no_start_and_start_failure(monkeypatch):
    monkeypatch.setattr(ad, "port_open", lambda host, *a, **k: False)
    b = ad.Board("palmyra_local", "10.23.224.213", "palmyra")
    with pytest.raises(ad.DumpError, match="--no-start"):
        ad.ensure_hw_server(b, start=False)

    def boom(*a, **k):
        raise hws.HwsError(hws.RC_FOREIGN, "held by someone")
    monkeypatch.setattr(hws, "ensure", boom)
    with pytest.raises(ad.DumpError, match="held by someone"):
        ad.ensure_hw_server(b)


def test_merge_never_crosses_a_side_effect_window():
    regs = [("m", "a", 0x1DF00), ("m", "b", 0x1DF1C), ("m", "c", 0x1DF28), ("m", "d", 0x1DF2C)]
    ranges = ad._merge(regs, "core")
    covered = [lo + 4 * i for lo, n in ranges for i in range(n)]
    assert 0x1DF20 not in covered and 0x1DF24 not in covered   # fifo_pop
    assert set(o for _m, _n, o in regs) <= set(covered)
    assert ad._merge([("m", "x", 0x100), ("m", "y", 0x100 + ad.MERGE_GAP + 8)], "core") == \
        [(0x100, 1), (0x100 + ad.MERGE_GAP + 8, 1)]


def test_regdb_excludes_reads_with_side_effects():
    try:
        db = ad.load_regdb(ad.find_regdb())
    except ad.DumpError:
        pytest.skip("aie2ps register JSON not installed")
    names = {n for k in db for _m, n, _o in db[k]["regs"]}
    assert not names & {"lock_request", "datamemory", "program_memory", "mdm_dbg_data"}
    assert not any("fifo_pop" in n for n in names)
    for kind in db:
        for lo, n in db[kind]["ranges"]:
            assert not any(ad._forbidden(kind, lo + 4 * i) for i in range(n))


class FakeXsdb:
    """read_blocks over a {(col,row): word} rule."""

    def __init__(self, word):
        self.word = word

    def read_blocks(self, reqs):
        out = {}
        for a, n in reqs:
            col, row = (a - ad.BASE) >> ad.COL_SHIFT, ((a - ad.BASE) >> ad.ROW_SHIFT) & 0x1F
            out[a] = [self.word(col, row)] * n
        return out


def test_probe_live_idle_gated():
    def word(col, row):
        if col == 0:
            return 0x3F if row == 0 else ad.GATED_WORD   # gated: shim row still answers
        return 0x00100000 if col == 1 else 0
    db = {"mem": {"regs": [("m", "dma_s2mm_status_0", 0xA0660)]},
          "shim": {"regs": [("m", "dma_s2mm_status_0", 0x1D220)]}}
    assert ad.probe_columns(FakeXsdb(word), [0, 1, 2], db) == {0: "gated", 1: "live", 2: "idle"}


def test_parse_span():
    assert ad.parse_span("3-6,8", 36) == [3, 4, 5, 6, 8]
    assert ad.parse_span("auto", 7) == list(range(7))
    with pytest.raises(ad.DumpError):
        ad.parse_span("40", 36)
