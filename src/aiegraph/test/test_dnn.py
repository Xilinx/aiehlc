###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.dnn_build``: the aiednn implementation list and the x86 target.

    python3 -m pytest -q src/aiegraph/test/test_dnn.py

Needs gcc/g++ for the x86 build; no TVM, onnx, Vitis or board.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import aiegraph as ag  # noqa: E402
from aiegraph.dnn import impl_list_source, needed_families  # noqa: E402

APP = REPO / "example" / "aiednn" / "matmul_app.cc"
MATMUL = REPO / "src" / "aietensorop" / "matmul"
has_cc = pytest.mark.skipif(not shutil.which("g++") or not shutil.which("gcc"), reason="needs gcc/g++")


def test_needed_families():
    assert needed_families(APP.read_text()) == {"aiednn_matmul": "dense"}
    assert needed_families("int main() { return 0; }") == {}
    # fused-epilogue apps call only the _attr variants (model apps): they still need the GEMM kernel
    assert needed_families("st = aiednn_convolution_forward_create_attr(&p, &m0);") == {"aiednn_convolution": "conv2d"}
    assert needed_families("aiednn_matmul_forward_create_attr(&p") == {"aiednn_matmul": "dense"}


def test_matmul_kernel_is_in_the_library():
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    e = lib["matmul_s8"]
    assert e.contract.op == "dense" and e.runnable and e.weight == (256, 256)


def test_impl_list_from_spec():
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    src = impl_list_source([lib["matmul_s8"]], x86=False)
    assert 'extern "C" int matmul_s8(const aiek_tensor *args, int num_args);' in src
    assert '{"aie:matmul_s8", aiednn_matmul, matmul_s8, 3,' in src
    assert "{aiednn_s8, 2, {256, 256}}" in src and "6, \"matmul_uk.cc\"" in src
    assert "aiednn_impl_count = 1;" in src
    assert "matmul_s8_ref" in impl_list_source([lib["matmul_s8"]], x86=True)


def test_symbolic_dims_become_symbols(tmp_path):
    k = tmp_path / "k"
    k.mkdir()
    (k / "mm_uk.cc").write_text(
        '#include "aie_user_kernel.h"\n'
        'AIEK_SPEC(mm, "op dense\\n" "target vek280\\n" "in int8 [M,KD] a\\n" "param int8 [*,KD] w\\n" "out int8 [M,8] c\\n");\n'
        "AIEK_REFERENCE(mm) { (void)args; (void)num_args; return 0; }\n")
    lib = ag.kernel_lib(k, quiet=True)
    src = impl_list_source([lib["mm"]], x86=True)
    assert "{AIEDNN_DIM_SYM(0), AIEDNN_DIM_SYM(1)}" in src          # M, KD
    assert "{AIEDNN_DIM_ANY, AIEDNN_DIM_SYM(1)}" in src             # *, KD again
    assert "{AIEDNN_DIM_SYM(0), 8}" in src and "     1, " in src     # one fixed dim



def _mesh_kernel(tmp_path, mesh_line):
    k = tmp_path / "k"
    k.mkdir(exist_ok=True)
    (k / "blk_uk.cc").write_text(
        '#include "aie_user_kernel.h"\n'
        'AIEK_SPEC(blk, "op dense\\n" "target vek280\\n" "' + mesh_line + '\\n" '
        '"in int8 [128,512] src\\n" "param int8 [128,512] w\\n" "out int32 [128,128] dst\\n");\n'
        "AIEK_REFERENCE(blk) { (void)args; (void)num_args; return 0; }\n")
    return ag.kernel_lib(k, quiet=True)


def test_mesh_dims_scale_with_the_array(tmp_path):
    """`mesh src[0]=32*rows ...`: the literal is the 4x4 value, dnn_build(mesh=(6, 4)) scales it"""
    lib = _mesh_kernel(tmp_path, "mesh src[0]=32*rows dst[0]=32*rows")
    assert "{128, 512}" in impl_list_source([lib["blk"]], x86=True)
    src = impl_list_source([lib["blk"]], x86=True, mesh=(6, 4))
    assert "{192, 512}" in src and "{128, 512}" in src and "{192, 128}" in src   # w (N rows) stays


def test_mesh_literal_must_match_the_default_mesh(tmp_path):
    """src[0] = 128 is 32*rows on 4x4, not cores (16): the spec is refused, with the reason"""
    lib = _mesh_kernel(tmp_path, "mesh src[0]=cores")
    assert "blk" not in lib
    assert any("default 4x4 mesh it is 16" in why for _, why in lib.skipped), lib.skipped


def test_gemm_block_spec_scales_rows_and_items():
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    src = impl_list_source([lib["gemm_s8s32"], lib["dw_s8s32"], lib["pool_s8s32"], lib["gemm_q_s8s8"]], x86=True,
                           mesh=(6, 4))
    assert src.count("{192, 512}") == 2 and src.count("{24, 4096}") == 2 and src.count("{24, 1024}") == 2


@pytest.mark.skipif(not shutil.which("clang++"), reason="needs clang++")
def test_runtime_has_no_narrowing_in_braces():
    """aiehlc's clang frontend parses the runtime as C++20 and makes narrowing in a brace
    initializer an ERROR (g++ only warns): an int64 size in an aiek_tensor initializer broke
    every board build once"""
    import subprocess
    res = subprocess.run(["clang++", "-std=c++20", "-Werror=c++11-narrowing", "-fsyntax-only",
                          f"-I{REPO / 'include'}", f"-I{REPO / 'src' / 'aiednn'}",
                          f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", str(REPO / "src" / "aiednn" / "aiednn_rt.cc")],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]


_A72_GXX = Path("/proj/custbuilds/SWIP/2026.2_1027_1626/installs/lin64/2026.2/Vitis/gnu/aarch64/lin/aarch64-none/bin/"
                "aarch64-none-elf-g++")


@pytest.mark.skipif(not _A72_GXX.is_file(), reason="needs the Vitis aarch64-none-elf-g++")
def test_runtime_compiles_for_the_board_cpu():
    """the runtime has an __aarch64__ branch (cycle counters) the x86 tests never compile: an include
    placed inside the x86 #else left rt_phi undeclared on the board and broke every model build"""
    import subprocess
    res = subprocess.run([str(_A72_GXX), "-std=c++20", "-O2", "-fsyntax-only", f"-I{REPO / 'include'}",
                          f"-I{REPO / 'src' / 'aiednn'}", f"-I{REPO / 'src' / 'aietensorop' / 'ops'}",
                          str(REPO / "src" / "aiednn" / "aiednn_rt.cc")], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]

@has_cc
def test_activation_int8_table_matches_the_formula(tmp_path):
    """GELU / swish -> s8 / u8 epilogues look codes up in a step table (aiednn_rt_geluq.h); it must
    return exactly what f(y) / ds + zp rounds to, including the doubles right at every step, where
    the formula itself is not monotone (1 - p loses bits for negative y)"""
    import subprocess
    src = tmp_path / "actq.cc"
    src.write_text("""#include "aiednn_rt_geluq.h"
#include <stdio.h>
int main() {
    const double ds[] = {0.08346197, 0.01, 0.003, 0.2, 0.5, 0.0417};
    const double zp[] = {0, -128, 5, -20, 100, -3};
    const struct { int kind; double a; } act[] = {{RT_ACTQ_GELU, 0}, {RT_ACTQ_GELU_TANH, 0}, {RT_ACTQ_SWISH, 1.0}, {RT_ACTQ_SWISH, 1.702}};
    long long bad = 0, none = 0, built = 0;
    for (auto f : act)
        for (int u8 = 0; u8 < 2; u8++)
            for (double d : ds)
                for (double z : zp) {
                    const double zz = u8 ? z + 128 : z;
                    const long long b = rt_actq_check(f.kind, f.a, d, zz, u8 ? 0 : -128, u8 ? 255 : 127, 20000, 12345);
                    if (b < 0) none++; else { bad += b; built++; }
                }
    printf("bad=%lld built=%lld none=%lld\\n", bad, built, none);
    return bad != 0 || built < 150;
}
""")
    exe = tmp_path / "actq"
    subprocess.run(["g++", "-std=c++17", "-O2", f"-I{REPO / 'src' / 'aiednn'}", str(src), "-o", str(exe)], check=True)
    res = subprocess.run([str(exe)], capture_output=True, text=True)
    print(res.stdout)
    assert res.returncode == 0, res.stdout + res.stderr


@has_cc
def test_quantize_without_divide_matches_the_divide(tmp_path):
    """ref:'s f32 -> s8 / u8 quantize (rt_quant8 in aiednn_rt_ops.h: norm int8 tails, reorder_contig) multiplies
    by 1 / scale and divides only near a rounding tie; it must equal sat(nearbyint(f / scale) + zp) for every
    float: every 509th bit pattern (NaN, inf, -0.0, denormals) and +-50 ulps around 601 exact ties per scale"""
    import re
    import subprocess
    ops = (REPO / "src" / "aiednn" / "aiednn_rt_ops.h").read_text()
    fn = re.search(r"static inline double rt_quant8\(.*?\n}\n", ops, re.S).group(0)
    src = tmp_path / "q8.cc"
    src.write_text("#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n" + fn + """
static int direct(double s, double zp, double lo, float f) {
    double r = nearbyint((double)f / s) + zp;
    r = r != r ? 0.0 : r < lo ? lo : r > lo + 255.0 ? lo + 255.0 : r;
    return (int)r;
}
int main() {
    const double scales[] = {0.0371784829, 0.00114474213, 1.0, 3.0, 7.66e-9, 1e-30, 1e30, 0.1, 1.0 / 3};
    const double zps[] = {0, 5, 113, -3};
    long long bad = 0;
    for (double s : scales)
        for (double z : zps)
            for (int s8 = 0; s8 < 2; s8++) {
                const double lo = s8 ? -128.0 : 0.0, inv = 1.0 / s;
                for (uint64_t b = 0; b < (1ull << 32); b += 509) {
                    float f; uint32_t bb = (uint32_t)b; memcpy(&f, &bb, 4);
                    bad += (int)rt_quant8(f, s, inv, z, lo) != direct(s, z, lo, f);
                }
                for (int k = -300; k <= 300; k++)
                    for (int d = -50; d <= 50; d++) {
                        float g = (float)((k + 0.5) * s); int32_t gb; memcpy(&gb, &g, 4); gb += d; memcpy(&g, &gb, 4);
                        bad += (int)rt_quant8(g, s, inv, z, lo) != direct(s, z, lo, g);
                    }
            }
    printf("bad=%lld\\n", bad);
    return bad != 0;
}
""")
    exe = tmp_path / "q8"
    subprocess.run(["g++", "-std=c++17", "-O2", str(src), "-o", str(exe)], check=True)
    res = subprocess.run([str(exe)], capture_output=True, text=True)
    print(res.stdout)
    assert res.returncode == 0, res.stdout + res.stderr


def test_empty_list_still_compiles_as_cxx():
    src = impl_list_source([], x86=True)
    assert "aiednn_impl_count = 0;" in src and "placeholder" in src


def test_two_sources_are_refused(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(MATMUL, root / "matmul")
    (root / "matmul2").mkdir()
    text = (MATMUL / "matmul_uk.cc").read_text().replace("matmul_s8", "matmul_s8b")
    (root / "matmul2" / "matmul2_uk.cc").write_text(text)
    shutil.copy2(MATMUL / "matmul.h", root / "matmul2" / "matmul.h")
    (root / "matmul2" / "matmul_uk_ref.c").write_text(
        (MATMUL / "matmul_uk_ref.c").read_text().replace("matmul_s8", "matmul_s8b"))
    assert len(ag.kernel_lib(root, quiet=True)) == 2
    with pytest.raises(ag.AiegraphError, match="several aiehlc sources serve the app equally"):
        ag.dnn_build(APP, target="x86", root=root, out=tmp_path / "o")


@has_cc
def test_x86_app_dispatches_to_the_aie_entry_and_passes(tmp_path):
    rep = ag.dnn_build(APP, target="x86", out=tmp_path / "o", run=True)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    assert "aiednn picked aie:matmul_s8" in text
    assert "aiednn_verbose,create,matmul,ref:matmul" in text         # the filtered reference
    assert "PASS: aiednn matmul 256x256x256 on aie:matmul_s8" in text


@has_cc
def test_x86_without_aie_kernels_falls_back_to_ref(tmp_path):
    rep = ag.dnn_build(APP, target="x86", root=tmp_path, out=tmp_path / "o", run=True)
    assert rep.kernels == [] and rep.ok and rep.run_ok, rep
    assert "aiednn picked ref:matmul" in rep.run_log.read_text()


# ── automatic kernel selection (families x board) ─────────────────────────

def _lib():
    return ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)


def test_gemm_block_serves_conv_and_dense_as_composite():
    from aiegraph.dnn import covers
    lib = _lib()
    assert covers(lib["gemm_s8s32"]) == {"dense": "composite", "conv2d": "composite"}
    assert covers(lib["matmul_s8"]) == {"dense": "direct"}


def test_selection_by_family_and_board():
    from aiegraph.dnn import _pick_kernels
    lib = _lib()
    picked, why = _pick_kernels(lib, {"dense"}, None, "vek280")
    assert [e.name for e in picked] == ["matmul_s8"]                 # direct beats composite
    picked, why = _pick_kernels(lib, {"dense", "conv2d"}, None, "vek280")
    assert sorted(e.name for e in picked) == sorted(["bf16_f32", "conv_kc_s8s32", "conv_q_s8s8", "dw_s8s32", "dwq_s8s8",
                                                     "elt_s8s32", "gemm_kc_s8s32", "gemm_q_s8s8", "gemm_s8s32", "lut8",
                                                     "mmt_s8s8", "pool_s8s32", "fpool_f32", "sdpa_s8s32"] + (["fvec_f32"] if HAS_FVEC else []))
    assert any("conv2dstem: AIEK_SPEC target vek385, not vek280" in w for w in why)
    # VEK385 (design section 33): conv2dstem_uk.cc and gemm_uk.cc both serve conv2d directly; the source with
    # more entries serving the app's families wins (the general GEMM library over a one-shape kernel), so a
    # conv-only (or sdpa-only) app builds; naming conv2dstem still picks it
    picked, _ = _pick_kernels(lib, {"dense"}, None, "vek385")
    assert [e.name for e in picked] == ["matmul_s8"]                 # matmul_uk.cc targets vek385 too
    picked, why = _pick_kernels(lib, {"conv2d"}, None, "vek385")
    assert {e.contract.source.name for e in picked} == {"gemm_uk.cc"}, why
    picked, _ = _pick_kernels(lib, {"conv2d"}, ["conv2dstem"], "vek385")
    assert [e.name for e in picked] == ["conv2dstem"]


def test_target_line_is_parsed():
    assert _lib()["conv2dstem"].contract.spec.targets == ("vek385",)


GEMM_CONV_APP = REPO / "example" / "aiednn" / "gemm_conv_app.cc"
R18_APP = REPO / "example" / "aiednn" / "resnet18" / "resnet18_app.cc"
DEPLOY = REPO / "worklocal" / "tvmrelay_deploy"


@has_cc
def test_x86_int32_paths_match_ref(tmp_path):
    rep = ag.dnn_build(GEMM_CONV_APP, target="x86", out=tmp_path / "o", run=True)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    assert "conv 3x3 s1 64->96 28px on im2col+blocked:aie:gemm_s8s32: 0 of" in text


@has_cc
@pytest.mark.skipif(not (DEPLOY / "arm_build" / "layeriohex").is_dir(), reason="needs the TVM deploy tree + dumps")
def test_x86_resnet18_logits_bit_identical_to_tvm(tmp_path):
    import subprocess
    data = tmp_path / "data"
    gen = subprocess.run([sys.executable, str(R18_APP.parent / "gen_resnet18.py"), "--out", str(data)],
                         capture_output=True, text=True)
    assert gen.returncode == 0, gen.stdout + gen.stderr
    rep = ag.dnn_build(R18_APP, target="x86", out=tmp_path / "o", run=True, headers=[data], timeout=600)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    assert "PASS: aiednn ResNet-18, all 1000 logits bit-identical to TVM" in text
    assert "im2col+blocked:aie:gemm_s8s32" in text and "ref:" not in text.split("aiednn_stats")[0]


# the FVEC mode is registered when gemm_uk.cc carries its entry (aie_modes_fvec.h / aiednn_rt_fvec.h)
HAS_FVEC = "AIEK_SPEC(fvec_blk_f32" in (REPO / "src" / "aietensorop" / "gemm" / "gemm_uk.cc").read_text()

OPS_APP = REPO / "example" / "aiednn" / "ops_app.cc"


@has_cc
def test_x86_ops_app_every_primitive(tmp_path):
    from aiegraph.dnn import compare_ops_logs
    rep = ag.dnn_build(OPS_APP, target="x86", out=tmp_path / "o", run=True)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    assert text.count("\nOP ") >= 80 and "OPCHK conv_s8 mismatches=0 of 1728" in text
    for kind in ("conv_s8 impl=im2col+", "deconv_s8 impl=col2im+", "matmul_s8 impl=blocked:", "inner_product_s8 impl=ip+"):
        assert kind in text
    assert compare_ops_logs(rep.run_log, rep.run_log) == []
    other = tmp_path / "other.log"
    other.write_text(text.replace("impl=ref:lrn", "impl=ref:other"))   # (f32 sdpa is gemm+sdpa on the AIE now)
    assert compare_ops_logs(rep.run_log, other) == [f"lrn_{a} on ref:lrn vs lrn_{a} on ref:other" for a in (176, 177)]


LUT_APP = REPO / "example" / "aiednn" / "lut_app.cc"


@has_cc
def test_x86_lut_app_8bit_ops_on_the_aie_modes(tmp_path):
    """lut: / eltblk: / redblk: / steps: (the GEMM ELF's table, eltwise, pool and step-table
    modes) take every op of lut_app and match ref: byte for byte"""
    rep = ag.dnn_build(LUT_APP, target="x86", out=tmp_path / "o", run=True, timeout=600)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    checks = [l for l in text.splitlines() if l.startswith("OPCHK ")]
    assert len(checks) == 21 and all(" mismatches=0 of " in l for l in checks), checks
    for impl in ("impl=lut:", "impl=eltblk:", "impl=redblk:", "impl=steps:"):
        assert impl in text
    assert "OPCHK bnorm_quant_f32_s8_c128_56 impl=ref:batch_normalization" in text  # ref's loop wins
    assert "OPCHK bnorm_relu_f32_s8_aie impl=steps:" in text   # ...but the default preference is the AIE


def test_gemm_megakernel_entries_declare_their_modes():
    """each optional mode of gemm_uk.cc belongs to the entries that send its opcode (AIEK_SPEC
    `modes`): a per-app build drops a mode only with every entry that needs it"""
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    want = {"gemm_s8s32": (), "gemm_kc_s8s32": (), "gemm_q_s8s8": ("GEMMQ",), "dw_s8s32": ("DW",),
            "dwq_s8s8": ("DW", "DWQ"), "pool_s8s32": ("POOL",), "elt_s8s32": ("ELTWISE",), "lut8": ("LUT", "STEPS"),
            "conv_kc_s8s32": ("CONV",), "conv_q_s8s8": ("GEMMQ", "CONV"), "mmt_s8s8": ("MMT",), "sdpa_s8s32": ("SDPA",),
            "bf16_f32": ("BF16",), "wq_s8f32": ("WQ",), "fpool_f32": ("FPOOL",), **({"fvec_f32": ("FVEC",)} if HAS_FVEC else {})}
    assert {k: lib[k].contract.spec.modes for k in want} == want
    cfg = (REPO / "src" / "aietensorop" / "ops" / "aie_modes_cfg.h").read_text()
    declared = {m for v in want.values() for m in v}
    assert declared == set(__import__("re").findall(r"#define AIEK_MODE_([A-Z0-9]+) [01]", cfg))
    # MMT is opt-in: next to every other mode it overflows the core's 16 KB program memory
    assert "#define AIEK_MODE_MMT 0" in cfg and "#define AIEK_MODE_SDPA 0" in cfg and "#define AIEK_MODE_BF16 0" in cfg
    assert not HAS_FVEC or "#define AIEK_MODE_FVEC 0" in cfg


def test_modes_header_turns_off_only_the_unused_modes():
    from aiegraph.dnn import modes_header
    cfg = (REPO / "src" / "aietensorop" / "ops" / "aie_modes_cfg.h").read_text()
    out = modes_header(cfg, {"GEMMQ", "DW"})
    assert "#define AIEK_MODE_GEMMQ 1" in out and "#define AIEK_MODE_DW 1" in out
    for m in ("DWQ", "POOL", "ELTWISE", "LUT", "STEPS", "MMT"):
        assert f"#define AIEK_MODE_{m} 0" in out
    on = modes_header(cfg, {"MMT"})               # an opt-in mode (default 0) turns on when used
    assert "#define AIEK_MODE_MMT 1" in on and "#define AIEK_MODE_GEMMQ 0" in on
    assert re.search(r"#define AIEK_MODE_ITEMS\s*\\?\s*\(AIEK_MODE_DW", out)  # derived macros untouched (any line wrap)


@has_cc
def test_entry_usage_counts_the_entries_an_app_calls(tmp_path):
    """dnn_build(modes="auto") builds the megakernel from the entries the app calls on x86;
    lut_app uses the table / eltwise / pool modes and no GEMM"""
    from aiegraph.dnn import _pick_kernels, _restrict_to_usage, entry_usage, needed_families
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    picked, _ = _pick_kernels(lib, set(needed_families(LUT_APP.read_text()).values()), None, "vek280")
    used = entry_usage(LUT_APP, picked, tmp_path / "u", timeout=600)
    assert used["lut_blk_s8"] > 0 and used["elt_blk_s8s32"] > 0 and used["pool_blk_s8s32"] > 0
    assert used["gemm_q_s8s8"] == 0 and used["dwq_blk_s8s8"] == 0
    keep, on = _restrict_to_usage(picked, used)
    assert {"LUT", "STEPS", "ELTWISE", "POOL"} <= set(on) and "GEMMQ" not in on and "DWQ" not in on
    assert "gemm_q_s8s8" not in [e.contract.spec.entry for e in keep]


def test_megakernel_segments_fit_the_core_program():
    """several megakernels per app when the modes do not fit one core program: contiguous op
    segments, greedy (fewest), each under the program-memory and stack limits; ops needing no mode
    stay in the current segment"""
    from aiegraph.dnn import MK_PM_LIMIT, MK_STACK_LIMIT, megakernel_cost, plan_segments
    assert megakernel_cost(set()) == (4464, 10848)
    one = [{"GEMMQ"}, set(), {"GEMMQ", "DW"}, {"DWQ"}]
    assert plan_segments(one) == [(0, 3, {"GEMMQ", "DW", "DWQ"})]
    many = [{"GEMMQ", "CONV"}, {"POOL"}, set(), {"DW", "DWQ"}, {"LUT", "STEPS", "ELTWISE"}, {"GEMMQ"}]
    segs = plan_segments(many)
    assert len(segs) > 1 and segs[0][0] == 0 and segs[-1][1] == len(many) - 1
    assert all(b + 1 == a for (_, b, _), (a, _, _) in zip(segs, segs[1:]))           # contiguous
    for a, b, ms in segs:
        pm, st = megakernel_cost(ms)
        assert pm <= MK_PM_LIMIT and st <= MK_STACK_LIMIT and set().union(*many[a:b + 1]) <= ms


@has_cc
def test_usage_is_counted_per_op(tmp_path):
    """the usage run logs which entries each op of an emitted model app called (aiednn_op_begin):
    the per-op counts add up to the per-entry totals, and the op modes plan into segments"""
    sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))
    import zoo
    from aiegraph import dnn_torch as dt
    from aiegraph.dnn import _pick_kernels, entry_usage, needed_families, plan_segments, usage_by_op
    m, x = zoo.build("tiny_unet", pretrained=False)
    g, ref, yf = dt.references(m, x, "int8", "tiny_unet")
    app = dt.emit_app(g, tmp_path / "app", ref, dt.TOL["int8"], int(yf.argmax()), verify_gemm=False)
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    picked, _ = _pick_kernels(lib, set(needed_families(app.read_text()).values()), None, "vek280")
    used = entry_usage(app, picked, tmp_path / "u", timeout=900)
    by_op = usage_by_op((tmp_path / "u" / "run.log").read_text())
    assert by_op and max(by_op) < len(g.ops)
    for e, n in used.items():
        assert n == sum(c.get(e, 0) for c in by_op.values()), e
    modes_of = {e.contract.spec.entry: set(e.contract.spec.modes) for e in picked}
    op_modes = [set().union(*[modes_of[e] for e in by_op.get(i, {})]) for i in range(len(g.ops))]
    assert len(plan_segments(op_modes)) == 1          # a small model fits one core program


def test_library_megakernels_fit_and_cover_every_entry():
    """modes="library": the fixed ELF set fits the core program, every entry with optional modes runs
    on at least one ELF, and the impl list routes those entries through aiednn_megakernel_need"""
    from aiegraph.dnn import (LIBRARY_MEGAKERNELS, MK_PM_LIMIT, MK_STACK_LIMIT, _pick_kernels,
                              library_need, megakernel_cost)
    from aiegraph.dnn import CORE_PM_BYTES
    for name, ms in LIBRARY_MEGAKERNELS:  # the fixed set is checked against kernel.map at build: the hard limit
        pm, st = megakernel_cost(ms)
        assert pm <= CORE_PM_BYTES and st <= MK_STACK_LIMIT, (name, pm)
    lib = ag.kernel_lib(REPO / "src" / "aietensorop", quiet=True)
    fams = set(needed_families((REPO / "example" / "aiednn" / "ops_app.cc").read_text()).values())
    picked, _ = _pick_kernels(lib, fams, None, "vek280")
    entries = {e.contract.spec.entry: e.contract.spec.modes for e in picked}
    held = set().union(*(set(ms) for _, ms in LIBRARY_MEGAKERNELS))
    entries = {e: ms for e, ms in entries.items() if set(ms) <= held}   # _plan_library leaves the rest out
    need = library_need(entries)
    assert set(need) == {e for e, ms in entries.items() if ms}
    # sets: cnn (bit 0), dwq (1), act (2), attn (3), float (4), fvec (5), wq (6)
    assert need["dwq_blk_s8s8"] == 0b000010 and need["mmt_blk_s8s8"] == 0b000010 and need["conv_q_s8s8"] == 0b000001
    assert need["lut_blk_s8"] == 0b000100 and need["elt_blk_s8s32"] == 0b1011100 and need["gemm_q_s8s8"] == 0b1011111
    assert need["sdpa_blk_s8s32"] == 0b001000 and need["bf16_blk_f32"] == 0b010000
    assert need["fvec_blk_f32"] == 0b100000     # "fvec": FVEC alone (it fills a core program)
    assert need.get("wq_blk_s8f32", 0b1000000) == 0b1000000   # "wq": GEMMQ + WQ + ELTWISE
    assert need["dw_blk_s8s32"] == 0b00011 and need["pool_blk_s8s32"] == 0b10101
    src = impl_list_source(picked, x86=False, mk_need=need)
    assert src.count("aiednn_megakernel_need(") == len(need) + 1          # + the weak declaration
    with pytest.raises(ag.AiegraphError):
        library_need({"x": ("NOSUCHMODE",)})


def test_core_program_memory_overflow_is_read_from_the_maps(tmp_path):
    """a megakernel whose modes outgrow the 16 KB program memory still links; dnn_build reads the
    kernel*.map PM totals and refuses the ELF (core_program_memory / CORE_PM_BYTES)"""
    from aiegraph.dnn import CORE_PM_BYTES, core_program_memory
    pm = "Section summary for memory 'PM':\n\n      .text  File\n ----------\n      {n}  kernel.o\n ----------\n      {n}  Total\n"
    (tmp_path / "kernel.map").write_text("Section summary for memory 'DM':\n 28684 Total\n" + pm.format(n=29968))
    (tmp_path / "kernel_mk1.map").write_text(pm.format(n=15968))
    got = core_program_memory(tmp_path)
    assert got == {"kernel.map": 29968, "kernel_mk1.map": 15968}
    assert [m for m, n in got.items() if n > CORE_PM_BYTES] == ["kernel.map"]


def test_core_stack_frame_is_read_from_the_maps(tmp_path):
    """the kernel's stack frame against DM_stack: a frame that runs into the statics below the stack
    gives wrong pixels and no error, so dnn_build refuses it (core_stack / CORE_STACK_MARGIN)"""
    from aiegraph.dnn import CORE_STACK_MARGIN, core_stack
    t = ("Memory map for memory 'DM_stack':\n\n    0x00070000..0x00077fff (     32768 items) : Stack\n"
         "  gemm_s8s32 (Function, Global, .text) (stack frame size = {f}\n  main (stack frame size = 160\n")
    (tmp_path / "kernel.map").write_text(t.format(f=32480))
    (tmp_path / "kernel_mk1.map").write_text(t.format(f=21856))
    got = core_stack(tmp_path)
    assert got == {"kernel.map": (32480, 32768), "kernel_mk1.map": (21856, 32768)}
    assert [m for m, (f, n) in got.items() if f > n - CORE_STACK_MARGIN] == ["kernel.map"]


def test_emit_autotune_lines():
    """emit_app(autotune=): "tune" turns aiednn_set_autotune on for the warm-up run and dumps the table;
    a saved table is preloaded line by line (non-AIEDNN_TUNE lines dropped)"""
    from aiegraph import dnn_torch as dt
    assert dt._autotune_lines(None) == []
    assert "aiednn_set_autotune(1)" in "".join(dt._autotune_lines("tune"))
    txt = "".join(dt._autotune_lines("noise\nAIEDNN_TUNE 0a6 2 ref:convolution\nAIEDNN_TUNE 6f8 8 poolblk:aie:pool_s8s32\n"))
    assert "aiednn_autotune_load(tuned)" in txt and "noise" not in txt
    assert txt.count("AIEDNN_TUNE") == 2


def test_light_islands_stay_on_the_host_where_switches_are_slow():
    """an op needing only a light mode the heavy megakernel cannot hold (gpt2's 24 int8 FVEC norms between bf16
    attention and int8 GEMMs) goes to ref: on a board with slow ELF loads: one segment instead of 25"""
    from aiegraph.dnn import _light_islands, plan_segments
    ops = []
    for _ in range(24):
        ops += [{"FVEC"}, set(), {"BF16"}, set()]       # norm, int8 GEMM (no mode), bf16x3 attention, GEMM
    assert len(plan_segments(ops)) > 20
    forced = _light_islands(ops)
    assert len(forced) == 24 and all(ops[i] == {"FVEC"} for i in forced)
    assert len(plan_segments([set() if i in forced else ms for i, ms in enumerate(ops)])) == 1
    assert _light_islands([{"FVEC"}, {"LUT"}]) == []              # light-only app: nothing to keep resident


def test_islands_on_ref_leave_the_build(tmp_path):
    """a mode only the islands used is not compiled: resnet18 on VEK385 kept FVEC for ops sent to ref: and its one
    ELF (GEMMQ POOL CONV FVEC, 22,884 B) was refused for core program memory"""
    from types import SimpleNamespace as NS
    from aiegraph.dnn import DnnBuild, _plan_megakernels
    ent = lambda name, *modes: NS(contract=NS(spec=NS(entry=name, modes=list(modes))))
    gemm, conv, fvec = ent("gemm_s8s32"), ent("conv_q_s8s8", "GEMMQ", "CONV"), ent("fvec_f32", "FVEC")
    app = tmp_path / "m_app.cc"
    app.write_text("#define NOPS 3\n")
    (tmp_path / "run.log").write_text("AIEDNN_ENTRY_OP conv_q_s8s8 0 1\nAIEDNN_ENTRY_OP fvec_f32 1 1\n"
                                      "AIEDNN_ENTRY_OP gemm_s8s32 2 1\n")
    rep = DnnBuild(app, "vek385", tmp_path, [gemm, conv, fvec], modes=("CONV", "FVEC", "GEMMQ"))
    _plan_megakernels(rep, app, rep.kernels, tmp_path)
    assert rep.force_ref == [1] and rep.segments is None
    assert rep.modes == ("CONV", "GEMMQ") and rep.kernels == [gemm, conv]
