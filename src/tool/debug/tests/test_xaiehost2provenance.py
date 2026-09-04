import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import xaiehost2provenance as x

SHIMCOL_SRC = """
#ifdef __AIESIM__
    int shimcol = 3;
#elif AIE_GEN == 5
    int shimcol = 10;
#else
    int shimcol = 6;
#endif
"""

def test_resolve_shimcol_gen5_baremetal():
    r = x.MacroResolver(aie_gen=5, aiesim=False)
    active = r.active_source(SHIMCOL_SRC)
    assert "int shimcol = 10;" in active
    assert "int shimcol = 3;" not in active
    assert "int shimcol = 6;" not in active

def test_resolve_shimcol_gen2_baremetal():
    r = x.MacroResolver(aie_gen=2, aiesim=False)
    active = r.active_source(SHIMCOL_SRC)
    assert "int shimcol = 6;" in active

def test_resolve_shimcol_aiesim():
    r = x.MacroResolver(aie_gen=5, aiesim=True)
    active = r.active_source(SHIMCOL_SRC)
    assert "int shimcol = 3;" in active


def test_eval_int_with_defines():
    r = x.MacroResolver(aie_gen=5, aiesim=False)
    defs = {"N": 4, "MAT_SIZE": "(N * N)"}
    assert x.eval_int("MAT_SIZE * 2", defs) == 32
    assert x.eval_int("mlen * sizeof(u32)", {"mlen": 32}) == 128
    assert x.eval_int("unknown_thing", {}) is None


PERF_SNIPPET = """
#define N 4
#define MAT_SIZE (N * N)
    int shimcol = 10;
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol, 0), XAie_TileLoc(4, 4));
    XAie_Route(routingInstance, NULL, XAie_TileLoc(4, 4), XAie_TileLoc(shimcol, 0));
    u32 mlen = MAT_SIZE * 2;
    XAie_MoveDataExternal2Aie(routingInstance, XAie_TileLoc(shimcol, 0), in,
                              mlen * sizeof(u32), CORE_IP_MEM, XAie_TileLoc(4, 4));
    XAie_MoveDataAie2External(routingInstance, XAie_TileLoc(4, 4), CORE_OP_MEM,
                              mlen * sizeof(u32), out, XAie_TileLoc(shimcol, 0));
"""


def test_extract_model_gen5():
    model = x.extract_model(PERF_SNIPPET, aie_gen=5, aiesim=False)
    # tiles
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    assert tiles[(4, 4)] == "core"
    assert tiles[(10, 0)] == "shim"
    # kernel placement
    assert model["kernel_placements"][(4, 4)] == "perf"
    # flows: push shim->core S2MM 128B, pull core->shim MM2S 128B
    dirs = {(f["src"], f["dst"], f["direction"]): f["len"] for f in model["flows"]}
    assert dirs[((10, 0), (4, 4), "S2MM")] == 128
    assert dirs[((4, 4), (10, 0), "MM2S")] == 128


def test_build_dfschedule_json():
    model = x.extract_model(PERF_SNIPPET, aie_gen=5, aiesim=False)
    doc = x.build_dfschedule(model, aie_gen=5)
    assert doc["version"] == 1
    assert doc["startcol"] == 0
    assert doc["aie_gen"] == "Gen5"
    assert doc["provenance_source"] == "static-xaie"
    tiles = {(t["col"], t["row"]): t for t in doc["tiles"]}
    core = tiles[(4, 4)]
    # core tile has both an S2MM (input) and MM2S (output) channel
    dirs = {c["direction"] for c in core["dma_channels"]}
    assert {"S2MM", "MM2S"} <= dirs
    # placeholder BD signals runtime-decided
    bd = core["dma_channels"][0]["bd_chain"][0]
    assert bd["bd_id"] == "runtime"
    assert bd["len"] == 128
    assert bd["acquire_lock"][0]["id"] == -1
    # load_kernel_group is a LIST of groups (schedule_view.kernel_for_tile
    # contract); each group has a callee + {col,row} tiles. (Deviation from
    # plan's dict shape, which would crash build_view.)
    grp = doc["load_kernel_group"][0]
    assert grp["callee"] == "__Runtime_load_kernel_group"
    assert {"col": 4, "row": 4} in grp["tiles"]


def test_collect_kernel_artifacts_populates_code_view(tmp_path):
    artifacts = tmp_path / "aout"
    kernel_cfg = artifacts / "kernelcfg" / "perf"
    kernel_cfg.mkdir(parents=True)
    (artifacts / "perf.cc").write_text("void perf() {}\n")
    (kernel_cfg / "wrapper.cc").write_text('#include "../../perf.cc"\n')
    (kernel_cfg / "aieml.bcf").write_text("_symbol 0x1000 0x20 win_ping\n")
    (kernel_cfg / "dm_offsets.h").write_text("#define CORE_IP_MEM 0x1000\n")

    model = x.extract_model(PERF_SNIPPET, aie_gen=5, aiesim=False)
    bundle = tmp_path / "worklocal"
    bundle.mkdir()
    copied = x.collect_kernel_artifacts(model, str(artifacts), str(bundle))
    doc = x.build_dfschedule(model, aie_gen=5, tile_artifacts=copied)
    core = next(t for t in doc["tiles"] if (t["col"], t["row"]) == (4, 4))

    assert core["kernel_cc"] == "kernelcfg/perf/wrapper.cc"
    assert core["bcf"] == "kernelcfg/perf/aieml.bcf"
    assert (bundle / core["kernel_cc"]).is_file()
    assert (bundle / core["bcf"]).is_file()
    assert (bundle / "perf.cc").is_file()
    assert (bundle / "kernelcfg/perf/dm_offsets.h").is_file()


COMMENTED_SNIPPET = """
    int shimcol = 10;
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol, 0) /* Source*/, XAie_TileLoc(4, 4) /* destination*/);
    u32 mlen = 16 * 2;
    XAie_MoveDataExternal2Aie(routingInstance, /*src=*/XAie_TileLoc(shimcol, 0), in, mlen * sizeof(u32),
                              CORE_IP_MEM, /*dest=*/XAie_TileLoc(4, 4));
    XAie_MoveDataAie2External(routingInstance, XAie_TileLoc(4, 4), CORE_OP_MEM, mlen * sizeof(u32), out,
                              XAie_TileLoc(shimcol, 0));
"""


def test_extract_model_tolerates_inline_comments():
    # The real generated aout/host.cc embeds /*src=*/ and /* Source*/ comments
    # inside XAie call argument lists; the parser must still find both flows.
    model = x.extract_model(COMMENTED_SNIPPET, aie_gen=5, aiesim=False)
    dirs = {(f["src"], f["dst"], f["direction"]): f["len"] for f in model["flows"]}
    assert dirs[((10, 0), (4, 4), "S2MM")] == 128
    assert dirs[((4, 4), (10, 0), "MM2S")] == 128


def test_build_dmaphop_and_degrade():
    model = x.extract_model(PERF_SNIPPET, aie_gen=5, aiesim=False)
    hop = x.build_dmaphop(model)
    paths = hop["communication_paths"]
    assert len(paths) == 2
    p = paths[0]
    # Consumer contract (schedule_view._load_comm_paths): stages carry roles,
    # tiles are {col,row} dicts. (Deviation from plan's top-level producer/
    # consumer keys, which _load_comm_paths cannot read.)
    roles = {s["role"] for s in p["stages"]}
    assert {"producer", "consumer"} <= roles
    prod = next(s for s in p["stages"] if s["role"] == "producer")
    assert "col" in prod["tile"] and "row" in prod["tile"]
    # graceful: pure-compute source yields empty model -> no flows/tiles
    empty = x.extract_model("int main(){return 0;}", aie_gen=5, aiesim=False)
    assert empty["flows"] == [] and empty["tiles"] == []
    assert x.model_is_empty(empty) is True


ENTRY_FN_SRC = """
int test_routing(XAie_DevInst *DevInst)
{
    int shimcol = 10;
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
}

int main(int argc, char* argv[]) {
    return 0;
}
"""


def test_extract_model_reports_entry_fn():
    # schedule_view.build_view's find_function_range defaults to the tiling
    # flow's 'host_canonicalized'; the raw-XAie host.cc wraps its XAie calls in
    # test_routing instead, so the generator names the enclosing function and
    # build_dfschedule surfaces it as host_entry_fn for the view to target.
    model = x.extract_model(ENTRY_FN_SRC, aie_gen=5, aiesim=False)
    assert model["entry_fn"] == "test_routing"
    doc = x.build_dfschedule(model, aie_gen=5)
    assert doc["host_entry_fn"] == "test_routing"


def test_ctrl_tile_type_geometry():
    # gen5/AIE2PS: rows 1,2 memtile, cores from row 3
    assert x.ctrl_tile_type(0, 5) == "shim"
    assert x.ctrl_tile_type(2, 5) == "memtile"
    assert x.ctrl_tile_type(3, 5) == "core"
    # gen2: cores from row 2
    assert x.ctrl_tile_type(1, 2) == "memtile"
    assert x.ctrl_tile_type(2, 2) == "core"
    # gen1: cores from row 1 (no memtile rows)
    assert x.ctrl_tile_type(1, 1) == "core"


CTRL_STRUCT_SRC = """
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u, .stream_id = 0u,
                                  .bd_id = RAW_BD_SLOT, .mm2s_ch = 0, .s2mm_ch = 1, .token = NULL,
                                  .resp_words = 2u};
    AieRC _rrc = __Runtime_ctrl_setup_routing(&_ri, 1);
"""

CTRL_CALL_SRC = """
    __Runtime_ctrl_read_target(dev, 0u, 0u, 2u, 5u, 6u, 0x1000u, 1u, out, RAW_BD_SLOT, 0, 1);
"""


def test_extract_ctrl_sends_struct():
    active = x.strip_comments(x.MacroResolver(5, False).active_source(CTRL_STRUCT_SRC))
    defs = x.collect_defines(active)
    sends = x.extract_ctrl_sends(active, defs)
    assert sends == [{"shim_col": 0, "dest_row": 3, "resp_words": 2}]


def test_extract_ctrl_sends_call():
    active = x.strip_comments(x.MacroResolver(5, False).active_source(CTRL_CALL_SRC))
    defs = x.collect_defines(active)
    sends = x.extract_ctrl_sends(active, defs)
    # read_target positional: (dev, shim_col, dest_col, dest_row, ...)
    assert sends == [{"shim_col": 0, "dest_row": 2, "resp_words": 1}]


def test_extract_ctrl_sends_dedup_and_unresolved_respwords():
    # resp_words references an unfoldable local -> defaults to 1; duplicate
    # (shim_col,dest_row) collapses to one send.
    src = CTRL_STRUCT_SRC.replace(".resp_words = 2u", ".resp_words = _rspcap") + CTRL_STRUCT_SRC
    active = x.strip_comments(x.MacroResolver(5, False).active_source(src))
    sends = x.extract_ctrl_sends(active, x.collect_defines(active))
    assert sends == [{"shim_col": 0, "dest_row": 3, "resp_words": 1}]


CTRL_MODEL_SRC = """
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u, .stream_id = 0u,
                                  .bd_id = RAW_BD_SLOT, .mm2s_ch = 0, .s2mm_ch = 1, .token = NULL,
                                  .resp_words = 2u};
    AieRC _rrc = __Runtime_ctrl_setup_routing(&_ri, 1);
"""


def test_extract_model_ctrl_packet():
    model = x.extract_model(CTRL_MODEL_SRC, aie_gen=5, aiesim=False)
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    # shim, pass-through memtile rows 1,2, and core dest row 3
    assert tiles[(0, 0)] == "shim"
    assert tiles[(0, 1)] == "memtile"
    assert tiles[(0, 2)] == "memtile"
    assert tiles[(0, 3)] == "core"
    # two flows: forward shim->dest S2MM (up), return dest->shim MM2S (down)
    dirs = {(f["src"], f["dst"], f["direction"]) for f in model["flows"]}
    assert ((0, 0), (0, 3), "S2MM") in dirs
    assert ((0, 3), (0, 0), "MM2S") in dirs
    # len tracks resp_words*4
    fwd = next(f for f in model["flows"] if f["direction"] == "S2MM")
    assert fwd["len"] == 8
    # no kernel loaded on the control path
    assert model["kernel_placements"] == {}


def test_extract_model_ctrl_memtile_dest():
    src = CTRL_MODEL_SRC.replace(".dest_row = 3u", ".dest_row = 2u")
    model = x.extract_model(src, aie_gen=5, aiesim=False)
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    assert tiles[(0, 2)] == "memtile"


CTRL_ENTRY_SRC = """
int controlperf_main(XAie_DevInst *dev)
{
    __Runtime_CtrlInstance _ri = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u,
                                  .resp_words = 1u};
    __Runtime_ctrl_setup_routing(&_ri, 1);
}
"""


def test_ctrl_entry_fn():
    model = x.extract_model(CTRL_ENTRY_SRC, aie_gen=5, aiesim=False)
    assert model["entry_fn"] == "controlperf_main"
    doc = x.build_dfschedule(model, aie_gen=5)
    assert doc["host_entry_fn"] == "controlperf_main"


# The real controlperf host.cc gates its sends behind a bare in-file
# `#define _CONTROL_WRITE_TEST_` immediately followed by `#ifdef`. A real
# preprocessor keeps the block; MacroResolver must honor inline define/undef.
INLINE_DEFINE_SRC = """
#define _CONTROL_WRITE_TEST_
#ifdef _CONTROL_WRITE_TEST_
    __Runtime_CtrlInstance _wi = {.dev = dev, .shim_col = 0u, .dest_col = 0u, .dest_row = 3u,
                                  .resp_words = 1u};
    __Runtime_ctrl_setup_routing(&_wi, 1);
#endif
#undef _CONTROL_WRITE_TEST_
#ifdef _CONTROL_WRITE_TEST_
    __Runtime_CtrlInstance _late = {.dev = dev, .shim_col = 1u, .dest_col = 1u, .dest_row = 3u,
                                    .resp_words = 1u};
#endif
"""


def test_macro_resolver_honors_inline_define_and_undef():
    active = x.MacroResolver(5, False).active_source(INLINE_DEFINE_SRC)
    assert "_wi" in active
    assert "_late" not in active


def test_extract_model_ctrl_behind_inline_define():
    model = x.extract_model(INLINE_DEFINE_SRC, aie_gen=5, aiesim=False)
    tiles = {(t["col"], t["row"]): t["type"] for t in model["tiles"]}
    assert tiles[(0, 0)] == "shim"
    assert tiles[(0, 3)] == "core"
    # the #undef'd second send must not appear
    assert (1, 3) not in tiles


# A macro body captured from an inline #define may contain backslashes or \g
# group-ref-like text (multi-line BENCH macros). _eval_cond must substitute it
# literally; a bare re.sub replacement would raise "bad escape".
MACRO_BODY_SRC = """
#define BENCH(a, b) do { raw \\
    stmt; } while (0)
#if AIE_GEN == 5
    int marker = 5;
#else
    int marker = 0;
#endif
"""


def test_eval_cond_tolerates_macro_body_with_backslash():
    active = x.MacroResolver(5, False).active_source(MACRO_BODY_SRC)
    assert "int marker = 5;" in active
    assert "int marker = 0;" not in active
