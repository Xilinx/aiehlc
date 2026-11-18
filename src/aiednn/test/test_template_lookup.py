"""aiednn template lookup (sprint unet-m4 P1): every category has templates, names resolve to the files, text is
verbatim, parameters are listed / filled, bad names fail, and the UNet example takes its C from here."""
import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import aiednn  # noqa: E402

REPO = SRC.parent
UNET = REPO / "example" / "aiegraph" / "unet_bk_sdm_tiny"


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
    assert aiednn.path("aiednn/conv2d/bf16_gemm_convin.cc") == aiednn.path("conv2d/bf16_gemm_convin.cc")


@pytest.mark.parametrize("bad", ["conv2d", "nosuchcat/x.cc", "conv2d/nosuch.cc", "conv2d/../gemm/bf16_gemm_kacc.cc",
                                 "conv2d/__init__.py", "/conv2d/bf16_gemm_convin.cc"])
def test_bad_names_fail(bad):
    with pytest.raises(aiednn.TemplateError):
        aiednn.template(bad)


def test_params_and_render():
    name = "gemm/kacc_unit_head.cc"
    assert aiednn.params(name) == ["E", "NSHAPES", "SRC"]
    t = aiednn.render(name, E="aiek_x", SRC="gemm/bf16_gemm_kacc.cc", NSHAPES=3)
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
            if f in ("_tpl", "_tplpath", "_render", "_rt", "aiednn.template", "aiednn.render"):
                out.append((f, n.args[0].value))
    return out


def test_unet_example_templates_resolve():
    refs = [r for py in sorted(UNET.glob("*.py")) for r in _template_refs(py)]
    assert len(refs) >= 30, refs
    cats = set()
    for f, name in refs:
        if f == "_rt":
            aiednn.runtime_path(name)
        else:
            cats.add(aiednn.path(name).parent.name)
    assert cats == set(aiednn.CATEGORIES), cats


def test_unet_example_has_no_c():
    assert not [p for p in UNET.rglob("*") if p.suffix in (".c", ".cc", ".h")]
    for py in UNET.glob("*.py"):
        if py.name == "build_unet.py":      # MAIN_C stays until P4
            continue
        t = py.read_text()
        assert "#include" not in t and "#define" not in t, py.name
