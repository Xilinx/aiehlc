"""aiegraph.boardmain (unet-m4 P4): the generated board main.c / inputs header / driver glue are
byte-identical (up to the aiednn-tpl P4 kernel-ABI rename, see _old_module) to what the UNet example produced with its own MAIN_C, runtime_io_main(), unet_aie.main_c()
and unet_aie.driver_template() before P4 (commit 5649d6a, read with `git show` from THIS worktree's repo).

The generated main differs only in its leading comment (it starts with the aiegraph MARKER)."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from aiegraph import boardmain as bm  # noqa: E402

EX = ROOT / "example" / "aiegraph" / "unet_bk_sdm_tiny"
OLD = "5649d6a"
OLD_K = "unet"
FA_LAST = "d5e9f15"   # last commit with example/.../fa_attn.py (unet-decl P3 moved it to aiegraph.flashattn)   # the kernel archive prefix at OLD
IO = ("sample", "timestep_i32", "ehs")


LEGACY_TEXT = ("bf16_gemm_convin.cc", "bf16_gemm_convin_ktimer.cc", "softfloat_rm_reset.cc", "tile64_conv_head.cc")
_LEGACY_DIR = []


def _legacy_text_dir() -> Path:
    """the example's 64x64-path C text as of FA_LAST, written once to a temp directory"""
    if not _LEGACY_DIR:
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="old_unet_text_"))
        for f in LEGACY_TEXT:
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{FA_LAST}:example/aiegraph/unet_bk_sdm_tiny/{f}"],
                               capture_output=True)
            if r.returncode:
                pytest.skip(f"git show {FA_LAST}:{f} unavailable")
            (d / f).write_bytes(r.stdout)
        _LEGACY_DIR.append(d)
    return _LEGACY_DIR[0]


def _old_module(rel: str, name: str):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD}:{rel}"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git show {OLD}:{rel} unavailable: {e}")
    # aiednn-tpl P4: the kernel archive prefix is aiednn (was the old model name OLD_K: <k>_aie_*, <k>_prof_print,
    # site tag '"aie <k>_aie_lin ", <len>') and the stand-alone bench kernel bf16_gemm_kacc.cc left src/aiednn;
    # apply the same rename to the old module (as boardmain._retarget does for a kernel prefix)
    src = (src.replace(f'"aie {OLD_K}_aie_lin ", 17', '"aie aiednn_aie_lin ", 19')
           .replace(f"{OLD_K}_aie_", "aiednn_aie_").replace(f"{OLD_K}_prof_print", "aiednn_prof_print")
           .replace('"gemm/bf16_gemm_kacc.cc"', '"gemm/gemm_kacc_kernel.cc"'))
    # aiednn-reorg P3 moved the UNet-only C text to the example directory; unet-decl P4d dropped it there too:
    # the old module reads the last copies (FA_LAST = d5e9f15, git show) from a temp directory
    d = _legacy_text_dir()
    for f in LEGACY_TEXT[:2]:
        src = src.replace(f'_tplpath("conv2d/{f}")', f'(Path({str(d)!r}) / "{f}")')
    for f in LEGACY_TEXT[2:]:
        src = src.replace(f'_tpl("conv2d/{f}")', f'open(Path({str(d)!r}) / "{f}", encoding="utf-8", newline="").read()')
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / rel)   # the old module resolves its templates relative to its own path
    if str(EX) not in sys.path:
        sys.path.insert(0, str(EX))
    exec(compile(src, str(ROOT / rel), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def old():
    bu = _old_module("example/aiegraph/unet_bk_sdm_tiny/build_unet.py", "old_build_unet")
    ua = _old_module("example/aiegraph/unet_bk_sdm_tiny/unet_aie.py", "old_unet_aie")
    # unet-decl P3: fa_attn.py left the example (aiegraph.flashattn); the old driver_template imports it lazily
    # (its OLD text names templates that are gone, so the last example copy, at FA_LAST, stands in for it)
    if "fa_attn" not in sys.modules:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{FA_LAST}:example/aiegraph/unet_bk_sdm_tiny/fa_attn.py"],
                             capture_output=True, text=True).stdout
        if not src:
            pytest.skip(f"git show {FA_LAST}:fa_attn.py unavailable")
        fa = types.ModuleType("fa_attn")
        fa.__file__ = str(EX / "fa_attn.py")
        exec(compile(src, fa.__file__, "exec"), fa.__dict__)
        sys.modules["fa_attn"] = fa
    return bu, ua


def _body(c: str) -> str:
    """the text after the leading comment block"""
    if c.startswith(bm.MARKER + "\n"):
        c = c[len(bm.MARKER) + 1:]
    assert c.startswith("/*")
    return c[c.index("*/\n") + 3:]


@pytest.mark.parametrize("aie,ktimer,rtio", [(False, False, False), (False, False, True), (True, False, False),
                                             (True, True, False), (True, False, True), (True, True, True)])
def test_main_byte_identical(old, aie, ktimer, rtio):
    bu, ua = old
    ref = ua.main_c(bu.MAIN_C, ktimer=ktimer) if aie else bu.MAIN_C
    # P5: the AIE templates the old module reads carry @M@ / @MU@ (model prefix) where they had unet / UNET
    ref = ref.replace("@MU@", "UNET").replace("@M@", "unet")
    if rtio:
        ref = bu.runtime_io_main(ref)
    got = bm.main_c("unet", IO, aie=aie, ktimer=ktimer, runtime_io=rtio, note=(1, "timestep"))
    assert got.splitlines()[0] == bm.MARKER
    assert _body(got) == _body(ref)


def test_main_one_line_each():
    """the gate build (aie + runtime_io): one LAUNCH warm / graph_run / in_crc32_board / CMP printf each"""
    c = bm.main_c("net", ("a", "b"), aie=True, runtime_io=True)
    assert c.count('printf("LAUNCH warm ') == 1
    assert c.count('printf("net: graph_run %.3f ms') == 1
    assert c.count("in_crc32_board=%s expect=%s %s") == 1
    assert c.count('int rc = p1_cmp("unet"') == 0 and c.count("rc = p1_cmp(") == 1
    assert "@" not in c.replace("@param", "")
    with pytest.raises(ValueError):
        bm.main_c("Net-1", ("a",))


def test_inputs_header_identical(old, tmp_path):
    bu, _ = old
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4 * 32 * 32).astype("<f4")
    t = np.array([981], dtype="<i4")
    e = rng.standard_normal(77 * 768).astype("<f4")
    for n, a in (("sample", s), ("timestep_i32", t), ("ehs", e)):
        a.tofile(tmp_path / f"{n}.bin")
    t.astype("<i8").tofile(tmp_path / "timestep_i64.bin")
    import zlib
    (tmp_path / "unet_in.crc32").write_text("%08x\n" % zlib.crc32(s.tobytes() + t.tobytes() + e.tobytes()))
    crc_old = bu.write_inputs_header(tmp_path, tmp_path / "old.h", "conv,linear,attn", "bf16")
    crc = bm.inputs_header(tmp_path / "new.h", "unet",
                           [("sample", "sample", s), ("timestep", "timestep", t), ("encoder_hidden_states", "ehs", e)],
                           dict(AIE_MODE="conv,linear,attn", WDTYPE="bf16"), origin=tmp_path)
    assert crc == crc_old
    old_l, new_l = (tmp_path / "old.h").read_text().split("\n"), (tmp_path / "new.h").read_text().split("\n")
    assert new_l[1:] == old_l[1:]


def test_driver_glue_identical(old):
    _, ua = old
    from aiegraph import flashattn
    assert bm.driver_glue() == ua.driver_template([]) == flashattn.fa_driver_glue([])
    att = [dict(fa=0, fa_steps=True)]
    assert bm.driver_glue(flashattn.DECL_FA + flashattn.DECL_KL, attn=True) == ua.driver_template(att)
    assert flashattn.fa_driver_glue(att) == ua.driver_template(att)
