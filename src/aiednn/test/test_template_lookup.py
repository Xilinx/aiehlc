"""aiednn template lookup (sprint m4 P1, 2026-11-18): every category has templates, names resolve to the files, text is
verbatim, parameters are listed / filled, bad names fail, and the aiegraph examples take their generic C from here
(a bench/ sub-directory of an example may hold a stand-alone board bench source; example-only C text, e.g. a
model-specific kernel or an x86 test harness, sits next to the example .py that names it and never copies a template)."""
import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import aiednn  # noqa: E402

REPO = SRC.parent
EXAMPLES = sorted(d for d in (REPO / "example" / "aiegraph").iterdir() if d.is_dir() and not d.name.startswith("_"))


@pytest.mark.parametrize("cat", aiednn.CATEGORIES)
def test_every_category_has_cc(cat):
    names = aiednn.templates(cat)
    assert any(n.endswith(".cc") for n in names), (cat, names)
    for n in names:
        assert n.startswith(cat + "/") and aiednn.path(n).parent == aiednn.ROOT / cat


def test_template_is_verbatim():
    for n in aiednn.templates():
        assert aiednn.template(n).encode("utf-8") == aiednn.path(n).read_bytes(), n


def test_aiednn_prefix_accepted():
    assert aiednn.path("aiednn/conv2d/tile64_conv_host.cc") == aiednn.path("conv2d/tile64_conv_host.cc")


@pytest.mark.parametrize("bad", ["conv2d", "nosuchcat/x.cc", "conv2d/nosuch.cc", "conv2d/../gemm/gemm_kacc_kernel.cc",
                                 "conv2d/__init__.py", "/conv2d/bf16_gemm_convin.cc"])
def test_bad_names_fail(bad):
    with pytest.raises(aiednn.TemplateError):
        aiednn.template(bad)


def test_params_and_render():
    name = "gemm/kacc_unit_head.cc"
    assert aiednn.params(name) == ["E", "NSHAPES", "SRC"]
    t = aiednn.render(name, E="aiek_x", SRC="gemm/gemm_kacc_kernel.cc", NSHAPES=3)
    assert t.startswith("// aiek_x.cc --") and "(3)" in t and "@" not in t
    with pytest.raises(aiednn.TemplateError):
        aiednn.render(name, E="aiek_x")                      # missing
    with pytest.raises(aiednn.TemplateError):
        aiednn.render(name, E="a", SRC="b", NSHAPES=1, X=2)  # unknown


def test_fa_class_params():
    assert {"NS", "KVL", "NKV", "DQ", "NQH"} <= set(aiednn.params("attention/fa_ns_head.cc"))
    assert aiednn.params("norm/fa_online_softmax.cc") == []


def test_runtime_source():
    t = aiednn.runtime_source("cmp_cos.c")
    assert "int p1_cmp(const char *tag" in t and 'printf("CMP %s board=' in t
    with pytest.raises(aiednn.TemplateError):
        aiednn.runtime_source("nosuch.c")


def _template_refs(py: Path):
    """("_tpl"|"_tplpath"|"_render"|"aiednn.template"|"_rt", name) of every template call in an example module"""
    out = []
    for n in ast.walk(ast.parse(py.read_text())):
        if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) \
                and isinstance(n.args[0].value, str):
            f = ast.unparse(n.func)
            if f in ("_tpl", "_tplpath", "_render", "_rt", "aiednn.template", "aiednn.render", "_bench", "_bench_render",
                     "_local"):
                out.append((f, n.args[0].value))
    return out


def test_example_templates_resolve():
    refs = [(py, *r) for d in EXAMPLES for py in sorted(d.glob("*.py")) for r in _template_refs(py)]
    assert len(refs) >= 30, refs
    cats = set()
    for py, f, name in refs:
        if f in ("_bench", "_bench_render"):   # example-only bench text, loaded by path from <example>/bench/
            assert (py.parent / "bench" / name).is_file(), (py.name, name)
        elif f == "_local":                    # example-only C text next to the example .py
            assert (py.parent / name).is_file(), (py.name, name)
        elif f == "_rt":
            aiednn.runtime_path(name)
        else:
            cats.add(aiednn.path(name).parent.name)
    assert cats == set(aiednn.CATEGORIES), cats


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda d: d.name)
def test_example_has_no_c(ex):
    local = [p for p in ex.rglob("*") if p.suffix in (".c", ".cc", ".h") and "bench" not in p.relative_to(ex).parts]
    named = "\n".join(py.read_text() for py in ex.glob("*.py"))
    lib = {aiednn.path(n).name for n in aiednn.templates()}
    for p in local:   # example-only C text: named by an example .py, not a copy of an aiednn template
        assert p.parent == ex and f'"{p.name}"' in named and p.name not in lib, p
    for py in ex.glob("*.py"):
        t = py.read_text()
        assert "#include" not in t and "#define" not in t, py.name
