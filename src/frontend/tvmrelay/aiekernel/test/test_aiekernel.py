###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Tests for the user-kernel interface (``frontend.tvmrelay.aiekernel``).

    PYTHONPATH=src python3 -m pytest -q src/frontend/tvmrelay/aiekernel/test

No board, ever. Unit tests need only gcc/g++. The ``deployed`` tests use the
network in ``worklocal/tvmrelay_deploy`` (``deploy_flow.py`` output) and skip
when it is absent; ``isolate`` needs the aarch64 binutils and skips without.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from frontend.tvmrelay.aiekernel import adapter, lint, spec  # noqa: E402
from frontend.tvmrelay.aiekernel.emulate import swap_driver  # noqa: E402
from frontend.tvmrelay.aiekernel.graph import LayerCall, TensorArg  # noqa: E402
from frontend.tvmrelay.aiekernel.lint import KERNEL_KEYWORD as KW  # noqa: E402
from frontend.tvmrelay.aiekernel.lint import LAUNCH_TOKEN as LT  # noqa: E402
from frontend.tvmrelay.aiekernel.match import match_call  # noqa: E402

DEPLOY = REPO / "worklocal" / "tvmrelay_deploy"
STEM = REPO / "src" / "aietensorop" / "conv2dstem" / "conv2dstem_uk.cc"
CONV3 = REPO / "example" / "userkernel" / "conv3x3_requant_ref.c"
INC = REPO / "include"
deployed = pytest.mark.skipif(not (DEPLOY / "layers" / "manifest.json").is_file(),
                              reason="needs deploy_flow.py output in worklocal/tvmrelay_deploy")

SPEC_OK = ('AIEK_SPEC(my_k,\n'
           '    "kernel k\\n"   // comment between literals is fine\n'
           '    "op conv2d\\n"\n'
           '    "match conv2d # trailing comment\\n"\n'
           '    "in    int8  [1,C,H,W]  x\\n"\n'
           '    "param int32 [C]        zp  require=zero require=range(-1,1)\\n"\n'
           '    "out   uint8 [1,C,H,W]  y   tol=1\\n");\n')


def _kernel_src(body_extra="", top="", after="", spec_text=SPEC_OK):
    """A minimal well-formed user kernel source, with injection points."""
    return f"""{top}
#define TILE 4
#define HELPER(x) ((x) + 1)
{KW}(pol) void kern(aie::port<input_window_int8 *, A> a) {{
    int s = HELPER(TILE);
    {body_extra}
}}
#include "aie_user_kernel.h"
static int run_it(const int8_t *x) {{
    aieArray device;
    aieMesh mesh = device.partition({{0, 4, 0, 6}}, 4, 4);
    kern{LT}mesh>>>(x);
    return 0;
}}
AIEK_ENTRY(my_k) {{ return run_it((const int8_t *)args[0].data); }}
{after}
{spec_text}
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _codes(rep):
    return {f.code for f in rep.findings}


# ═══════════════════════════════════════════════════════════════════════════
#  spec
# ═══════════════════════════════════════════════════════════════════════════

def test_spec_parses(tmp_path):
    src = spec.load_source(_write(tmp_path, "k.cc", SPEC_OK))
    (s,) = spec.parse_specs(src)
    assert (s.entry, s.name, s.op, s.match) == ("my_k", "k", "conv2d", ("conv2d",))
    assert [a.kind for a in s.args] == ["in", "param", "out"]
    assert s.args[0].dims == (1, "C", "H", "W") and s.symbolic
    assert s.args[1].requires == (("zero",), ("range", -1, 1))
    assert s.args[2].tol == 1 and s.args[2].name == "y"


@pytest.mark.parametrize("line,needle", [
    ("in int7 [1] x", "dtype"),
    ("out int8 [1] y\\nin int8 [1] x", "outputs must come last"),
    ("in int8 [1] x require=zero\\nout int8 [1] y", "only applies to a param"),
    ("param int8 [1] x tol=2\\nout int8 [1] y", "only applies to an out"),
    ("in int8 [1] x bogus=1\\nout int8 [1] y", "unknown key"),
    ("in int8 [1] x", "no `out`"),
    ("in int8 [1,-2] x\\nout int8 [1] y", "not an integer"),
    ("param int8 [1] x require=range(3,1)\\nout int8 [1] y", "lo > hi"),
])
def test_spec_errors(tmp_path, line, needle):
    src = spec.load_source(_write(tmp_path, "k.cc", f'AIEK_SPEC(e, "{line}\\n");\n'))
    with pytest.raises(spec.SpecError, match=needle):
        spec.parse_specs(src)


def test_spec_rejects_code_between_literals(tmp_path):
    src = spec.load_source(_write(tmp_path, "k.cc", 'AIEK_SPEC(e, "in int8 [1] x\\n" X);\n'))
    with pytest.raises(spec.SpecError, match="only adjacent string literals"):
        spec.parse_specs(src)


def test_splice_like_aiehlc(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(tmp_path, "b.cc", "int b;\n")
    # Nested include resolves against the ENTRY dir, as aiehlc.cc does.
    _write(tmp_path, "sub/a.cc", 'int a;\n#include "b.cc"\n')
    entry = _write(tmp_path, "e.cc", '#include "sub/a.cc"\n#include "b.cc"\n'
                                     '#include "nope.cc"\n#include "h.h"\nint e;\n')
    src = spec.load_source(entry)
    assert [p.name for p in src.included] == ["a.cc", "b.cc"]
    assert src.text.count("int b;") == 1                   # repeat dropped
    assert [m[0] for m in src.missing] == ["nope.cc"]
    assert '#include "h.h"' in src.text                    # headers left to clang
    assert src.where(src.text.index("int a;")) == "a.cc:1"


# ═══════════════════════════════════════════════════════════════════════════
#  lint
# ═══════════════════════════════════════════════════════════════════════════

def test_lint_clean(tmp_path):
    rep = lint.lint_source(_write(tmp_path, "k.cc", _kernel_src()))
    assert rep.ok, rep.text()
    assert rep.kernels == ["kern"] and rep.launches == [("run_it", "kern")]
    assert _codes(rep) == {"L027", "L028"}   # symbolic dims + no reference: warnings


@pytest.mark.parametrize("kwargs,code", [
    ({"top": f"// the {KW} keyword in prose"}, "L011"),
    ({"top": f'const char *s = "{LT}";'}, "L012"),
    ({"top": "static int run_it(const int8_t *x);"}, "L014"),
    ({"after": "int main() { return 0; }"}, "L016"),
    ({"top": "static inline int helper(int x) { return x; }",
      "body_extra": "int h = helper(s);"}, "L017"),
    ({"body_extra": "int z = FROM_A_HEADER;"}, "L018"),
    ({"top": '#include "data/fixture.h"'}, "L019"),
    ({"top": '#include "missing.cc"'}, "L020"),
    ({"top": "#pragma control_plan_shim_bd_ctrl"}, "L021"),
    ({"top": "#pragma control_plan_op_control_packet\n#pragma control_plan_shim_bd_ctrl"}, "L023"),
    ({"after": "static int rd(const unsigned char *p_data) "
               "{ return *(const int32_t *)p_data; }"}, "L025"),
    ({"body_extra": "int16_t sum = 0;"}, "L031"),
    ({"spec_text": ""}, "L001"),
    ({"spec_text": SPEC_OK.replace("my_k", "other")}, "L003"),
    ({"spec_text": SPEC_OK.replace("my_k", "aie_k")}, "L005"),
    ({"spec_text": SPEC_OK.replace('"kernel k', '"reference gone.c\\n" "kernel k')}, "L028"),
    ({"spec_text": SPEC_OK.replace('in    int8  [1,C,H,W]', 'in    int8  [1,C,H,Q,')}, "L002"),
])
def test_lint_rules(tmp_path, kwargs, code):
    rep = lint.lint_source(_write(tmp_path, "k.cc", _kernel_src(**kwargs)))
    assert code in _codes(rep), rep.text()


def test_lint_l015_launch_without_partition(tmp_path):
    text = _kernel_src().replace("aieMesh mesh = device.partition({0, 4, 0, 6}, 4, 4);",
                                 "aieMesh mesh;")
    rep = lint.lint_source(_write(tmp_path, "k.cc", text))
    assert "L015" in _codes(rep)


def test_lint_l022_and_l029(tmp_path):
    _write(tmp_path, "ref.c", "int unrelated(void) { return 0; }\n")
    text = _kernel_src(
        top="#pragma control_plan_op_control_packet\n#pragma control_plan_shim_bd_ctrl",
        after="void f(void) { __Runtime_ctrl_high_throughput_enable(1); }",
        spec_text=SPEC_OK.replace('"kernel k', '"reference ref.c\\n" "kernel k'))
    rep = lint.lint_source(_write(tmp_path, "k.cc", text))
    assert {"L022", "L029"} <= _codes(rep) and "L023" not in _codes(rep)


def test_lint_reference_only_file_is_its_own_reference(tmp_path):
    p = _write(tmp_path, "r.c", '#include "aie_user_kernel.h"\n'
               "AIEK_REFERENCE(my_k) { (void)args; (void)num_args; return 0; }\n" + SPEC_OK)
    rep = lint.lint_source(p)
    assert rep.ok and not rep.has_aie_kernel and rep.specs[0].reference == p.resolve()
    assert _codes(rep) == {"L010"}


def test_shipped_examples_lint_clean():
    stem = lint.lint_source(STEM)
    assert stem.ok and not [f for f in stem.findings if f.level != "info"], stem.text()
    assert stem.kernels == ["conv2d_spatial"]
    conv3 = lint.lint_source(CONV3)
    assert conv3.ok and conv3.specs[0].symbolic


# ═══════════════════════════════════════════════════════════════════════════
#  match (synthetic network)
# ═══════════════════════════════════════════════════════════════════════════

class _Net:
    def __init__(self, params):
        self.params = params

    def param_array(self, name, dtype, shape):
        import numpy as np
        return np.asarray(self.params[name], dtype=np.int64).reshape(shape)


def _call(args, symbol="tvmgen_conv2d_x"):
    c = LayerCall(call=3, node=9, symbol=symbol, layer=2, dir="02_conv2d_x")
    for i, (kind, dtype, shape, pname) in enumerate(args):
        c.args.append(TensorArg(i, kind, dtype, tuple(shape), eid=i, param=pname))
    return c


def _spec(tmp_path):
    return spec.parse_specs(spec.load_source(_write(tmp_path, "k.cc", SPEC_OK)))[0]


def test_match_ok_and_bindings(tmp_path):
    v = match_call(_spec(tmp_path), _call([("in", "int8", (1, 4, 8, 8), None),
                                           ("param", "int32", (4,), "p0"),
                                           ("out", "uint8", (1, 4, 8, 8), None)]),
                   _Net({"p0": [0, 0, 0, 0]}))
    assert v.ok and v.bindings == {"C": 4, "H": 8, "W": 8}


@pytest.mark.parametrize("args,params,needle", [
    ([("in", "int8", (1, 4, 8, 8), None), ("out", "uint8", (1, 4, 8, 8), None)],
     {}, "2 args"),
    ([("param", "int8", (1, 4, 8, 8), "w"), ("param", "int32", (4,), "p0"),
      ("out", "uint8", (1, 4, 8, 8), None)], {"p0": [0] * 4}, "is a constant"),
    ([("in", "int16", (1, 4, 8, 8), None), ("param", "int32", (4,), "p0"),
      ("out", "uint8", (1, 4, 8, 8), None)], {"p0": [0] * 4}, "dtype int16"),
    ([("in", "int8", (1, 4, 8, 8), None), ("param", "int32", (5,), "p0"),
      ("out", "uint8", (1, 4, 8, 8), None)], {"p0": [0] * 5}, "C = 4"),
    ([("in", "int8", (1, 4, 8, 8), None), ("param", "int32", (4,), "p0"),
      ("out", "uint8", (1, 4, 8, 8), None)], {"p0": [0, 3, 0, 0]}, "require=zero"),
])
def test_match_reasons(tmp_path, args, params, needle):
    v = match_call(_spec(tmp_path), _call(args), _Net(params))
    assert not v.ok and any(needle in r for r in v.reasons), v.reasons


def test_match_symbol_filter(tmp_path):
    v = match_call(_spec(tmp_path), _call([("in", "int8", (1, 4, 8, 8), None),
                                           ("param", "int32", (4,), "p0"),
                                           ("out", "uint8", (1, 4, 8, 8), None)],
                                          symbol="tvmgen_dense"), _Net({"p0": [0] * 4}))
    assert not v.ok and "symbol lacks 'conv2d'" in v.reasons[0]


def test_parse_kernel_arg():
    from frontend.tvmrelay.aiekernel.flow import parse_kernel_arg

    assert parse_kernel_arg("k.cc") == (Path("k.cc"), None)
    assert parse_kernel_arg("k.cc@1, 4") == (Path("k.cc"), [1, 4])
    assert parse_kernel_arg("k.cc@auto") == (Path("k.cc"), None)
    assert parse_kernel_arg(("k.cc", [9])) == (Path("k.cc"), [9])


def test_gate_6a_keeps_several_libraries():
    """Gate 6a: with per-library routing several kernel sources share the ELF;
    only a reference-only source (nothing to build) is rejected."""
    from frontend.tvmrelay.aiekernel.flow import KernelPlan, _check_one_aie_library

    class _L:
        def __init__(self, k):
            self.has_aie_kernel = k

    class _S:
        def __init__(self, n):
            self.name = n

    a = KernelPlan(Path("a.cc"), _L(True), _S("a"), selected=[1])
    r = KernelPlan(Path("r.c"), _L(False), _S("r"), selected=[4])
    b = KernelPlan(Path("b.cc"), _L(True), _S("b"), selected=[9, 14])
    _check_one_aie_library([a, r, b], verbose=False)
    assert a.selected == [1]
    assert r.selected == [] and r.rejected[4][0] == "build"
    assert b.selected == [9, 14]


# ═══════════════════════════════════════════════════════════════════════════
#  C side: header, adapter, driver swap -- compiled and run
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cc,std", [("gcc", "-std=c11"), ("g++", "-std=c++17")])
def test_header_compiles_warning_free(tmp_path, cc, std):
    src = _write(tmp_path, "t.c" if cc == "gcc" else "t.cc",
                 '#include "aie_user_kernel.h"\nAIEK_ENTRY(k) { (void)args; (void)num_args; '
                 'return AIEK_OK; }\nAIEK_SPEC(k, "in int8 [1] x\\nout int8 [1] y\\n");\n')
    proc = subprocess.run([cc, std, "-Wall", "-Wextra", "-Werror", f"-I{INC}", "-c",
                           str(src), "-o", str(tmp_path / "t.o")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


_DRIVER = r"""
#include <stdio.h>
#include <string.h>
#include "aie_user_kernel.h"
typedef union { int64_t i; double d; void *v_handle; } TVMValue;
int32_t aie_tvmgen_x(void *, int32_t *, int32_t, void *, int32_t *, void *);
int main(void) {
    static int8_t x[2 * 3]; static int32_t zp[3]; static uint8_t y[2 * 3];
    int64_t sx[] = {2, 3}, sz[] = {3}, sy[] = {2, 3};
    aiek_dltensor d[3] = {
        {x, 1, 0, 2, 0, 8, 1, sx, 0, 0}, {zp, 1, 0, 1, 0, 32, 1, sz, 0, 0},
        {y, 1, 0, 2, 1, 8, 1, sy, 0, 0}};
    TVMValue v[3]; int32_t codes[3] = {7, 7, 7};
    for (int i = 0; i < 6; i++) x[i] = (int8_t)(i - 2);
    for (int i = 0; i < 3; i++) v[i].v_handle = &d[i];
    int rc = aie_tvmgen_x(v, codes, 3, 0, 0, 0);
    printf("RC %d Y %d %d %d %d %d %d\n", rc, y[0], y[1], y[2], y[3], y[4], y[5]);
    sx[1] = 4;                                            /* wrong shape */
    printf("BAD %d\n", aie_tvmgen_x(v, codes, 3, 0, 0, 0));
    printf("ARITY %d\n", aie_tvmgen_x(v, codes, 2, 0, 0, 0));
    return 0;
}
"""

_REF = r"""
#include "aie_user_kernel.h"
AIEK_REFERENCE(my_k) {
    const int8_t *x = AIEK_PTR(&args[0], const int8_t);
    uint8_t *y = AIEK_PTR(&args[2], uint8_t);
    for (int64_t i = 0; i < aiek_numel(&args[0]); i++) y[i] = (uint8_t)(x[i] < 0 ? 0 : x[i]);
    return num_args == 3 ? AIEK_OK : AIEK_ERR_ARGS;
}
"""


def test_adapter_runs_and_checks(tmp_path):
    s = _spec(tmp_path)
    call = _call([("in", "int8", (2, 3), None), ("param", "int32", (3,), "p0"),
                  ("out", "uint8", (2, 3), None)], symbol="tvmgen_x")
    s.args[0].dims, s.args[2].dims = ("*", "*"), ("*", "*")
    # Same text both ways: valid C++ (host.cc) and C (emulate).
    _write(tmp_path, "aie.cc", adapter.adapter_source(s, call, target="aie"))
    proc = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", f"-I{INC}",
                           str(tmp_path / "aie.cc")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    _write(tmp_path, "ad.c", adapter.adapter_source(s, call, target="ref"))
    _write(tmp_path, "ref.c", _REF)
    _write(tmp_path, "drv.c", _DRIVER)
    exe = tmp_path / "t"
    proc = subprocess.run(["gcc", "-std=gnu11", "-Wall", f"-I{INC}", "-o", str(exe),
                           *[str(tmp_path / f) for f in ("ad.c", "ref.c", "drv.c")]],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    assert "RC 0 Y 0 0 0 1 2 3" in out, out
    assert "ENTER k (my_k_ref)" in out and "EXIT rc=0" in out
    assert "BAD -1" in out and "ARITY -1" in out


def test_swap_driver():
    text = ('#include "tvm_graph_types.h"\n'
            "    tvmgen_a(args, codes, 9, 0, 0, 0);\n"
            "    tvmgen_ab(args, codes, 3, 0, 0, 0);\n"
            "    tvmgen_a(args, codes, 9, 0, 0, 0);\n")
    out, counts = swap_driver(text, ["tvmgen_a"])
    assert counts == {"tvmgen_a": 2}
    assert out.count("    aie_tvmgen_a(args, codes, 9, 0, 0, 0);") == 2
    assert "    tvmgen_ab(args" in out and "int32_t aie_tvmgen_a(void* args" in out


# ═══════════════════════════════════════════════════════════════════════════
#  Deployed network (worklocal/tvmrelay_deploy)
# ═══════════════════════════════════════════════════════════════════════════

@deployed
def test_detect_deployed():
    from frontend.tvmrelay import aiekernel

    assert [v.layer for v in aiekernel.detect(DEPLOY, STEM) if v.ok] == [1]
    hits = [v for v in aiekernel.detect(DEPLOY, CONV3) if v.ok]
    assert [v.layer for v in hits] == [4, 9, 14, 16, 21, 23, 28]
    assert len(hits[0].calls) == 2                       # layer 04 is called twice


@deployed
def test_calibrate_catches_rounding(tmp_path):
    from frontend.tvmrelay.aiekernel import calibrate, graph

    net = graph.load_network(DEPLOY)
    s = lint.lint_source(STEM).specs[0]
    (good,) = calibrate.calibrate_layer(net, s, net.layers()[1])
    assert good.ok, good.summary()
    bad = _write(tmp_path, "bad.c", s.reference.read_text().replace("(sh + 30)", "(sh + 29)"))
    s.reference, s.name = bad, "pytest_badref"
    (res,) = calibrate.calibrate_layer(net, s, net.layers()[1])
    assert not res.ok and "MISMATCH" in res.summary()


@deployed
def test_emulate_bisects_the_bad_layer(tmp_path):
    from frontend.tvmrelay.aiekernel import flow

    # Wrong only on layer 09's stride-2 geometry: emulation must pin it there.
    bad = _write(tmp_path, "conv3x3_bad.c", CONV3.read_text().replace(
        "od[((occ * ho + oh) * wo + ow) * 4 + ocb] = (int8_t)(v - 128);",
        "od[((occ * ho + oh) * wo + ow) * 4 + ocb] = (int8_t)(v - 128 + (s == 2));"))
    rep = flow.check(DEPLOY, [f"{bad}@4,9"], calibrate=False, emulate=True, verbose=False)
    (k,) = rep["kernels"]
    assert k["selected"] == [4] and k["rejected"]["9"]["gate"] == "emulate"


@pytest.mark.skipif(shutil.which("aarch64-none-elf-nm") is None, reason="needs aarch64 binutils")
@deployed
def test_isolate_keeps_entry_and_routing_collides(tmp_path):
    from frontend.tvmrelay.aiekernel.build import ROUTING_SYMBOL, isolate_archive, link_collisions

    src = next((DEPLOY / "layers").glob("01_*/aie/build/*.a"), None)
    if src is None:
        pytest.skip("no layer-01 archive in the deployed tree")
    libs = []
    for name in ("a", "b"):
        (tmp_path / name / "build").mkdir(parents=True)
        libs.append(tmp_path / name / "build" / f"lib{name}.a")
        shutil.copy2(src, libs[-1])
        assert isolate_archive(libs[-1], ["conv2d_stem"], verbose=False)["ok"]
    hits = dict(link_collisions(libs))
    # Everything private now -- except the runtime's routing() singleton, which
    # must stay strong so two libraries in one ELF are caught, not mis-routed.
    assert ROUTING_SYMBOL in hits and "conv2d_stem_nchwc" not in hits
