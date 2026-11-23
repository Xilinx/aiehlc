###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-aiehlc-arm_build-cflags: the ``layer_cflags`` knob of arm_build.write_makefile.

* empty knob  -> the generated Makefile is byte-identical to the one the
  pre-knob generator (base commit 3d5404c) writes;
* non-empty   -> ``LAYER_CFLAGS ?= ...`` + a target-specific
  ``$(LAYER_OBJS): CFLAGS += $(LAYER_CFLAGS)``; ``make -n`` shows the flags on
  the layer objects only, never on graph_driver.o / main.o;
* build_unet.py --arm-layer-cflags-file: lines joined with spaces, the 6 words
  of the P5n FLAGS file reach the LAYER_CFLAGS line in order; missing/empty
  file is an error.

Run from the repo root:  python -m pytest -q src/frontend/tvmrelay/test/test_arm_build_layer_cflags.py
"""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay import arm_build as ab  # noqa: E402

BASE = "3d5404c"
# unet-m4 P4 (08f880b) reworded the generated Makefile's comments / error text without model names;
# the base output with exactly these rewordings applied is the expected empty-knob Makefile.
P4_REWORDS = (("make KERNELS=/path/to/resnet18.c", "make KERNELS=/path/to/<stem>.c"),
              ("on int8 ResNet-18 with the stem on the AIE", "on an int8 18-layer CNN with the stem on the AIE"),
              ("arrive as resnet18.o inside OBJS", "arrive as <stem>.o inside OBJS"),
              ("he single resnet18.c, used when", "he single <stem>.c, used when"),
              ("# resnet18.o here, and a `clean`", "# <stem>.o here, and a `clean`"))
P5N_FLAGS = ("-mcpu=cortex-a78 -ffp-contract=fast -fassociative-math "
             "-fno-signed-zeros -fno-trapping-math -ftree-vectorize")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp: Path):
    """A minimal per-layer tree: two layer sources, a kernel file with no AIE entry."""
    layers = tmp / "layers"
    for n in ("00_conv", "01_add"):
        (layers / n).mkdir(parents=True)
        (layers / n / f"{n}.c").write_text("int f_%s(void){return 0;}\n" % n[:2])
    (layers / "layers_common.h").write_text("\n")
    kernel = tmp / "net.c"
    kernel.write_text("/* no AIE entry */\n")
    build = tmp / "arm_build"
    build.mkdir()
    for c in ("graph_driver.c", "main.c", "tvm_runtime_shim.c", "input_image.h", "imagenet_labels.h"):
        (build / c).write_text("\n")
    return build, kernel, layers


def _mk(build, kernel, layers, **kw):
    return ab.write_makefile(build, ROOT, kernel, layers_dir=layers,
                             verbose=False, **kw).read_text()


def test_empty_knob_identical_to_base(tmp_path):
    old_src = subprocess.run(["git", "-C", str(ROOT), "show",
                              f"{BASE}:src/frontend/tvmrelay/arm_build.py"],
                             capture_output=True, text=True, check=True).stdout
    assert "LAYER_CFLAGS" not in old_src
    (tmp_path / "old_arm_build.py").write_text(old_src)
    old = _load(tmp_path / "old_arm_build.py", "old_arm_build")
    build, kernel, layers = _tree(tmp_path)
    want = old.write_makefile(build, ROOT, kernel, layers_dir=layers, verbose=False).read_text()
    for a, b in P4_REWORDS:
        assert a in want, a
        want = want.replace(a, b)
    assert _mk(build, kernel, layers) == want
    assert _mk(build, kernel, layers, layer_cflags="") == want
    assert _mk(build, kernel, layers, layer_cflags="  \n ") == want
    assert "LAYER_CFLAGS" not in want


def test_flags_only_on_layer_objs_rule(tmp_path):
    build, kernel, layers = _tree(tmp_path)
    base = _mk(build, kernel, layers)
    text = _mk(build, kernel, layers, layer_cflags="-ffp-contract=fast -ftree-vectorize")
    lines = [l for l in text.splitlines() if "LAYER_CFLAGS" in l and not l.startswith("#")]
    assert lines == ["LAYER_CFLAGS ?= -ffp-contract=fast -ftree-vectorize",
                     "$(LAYER_OBJS): CFLAGS += $(LAYER_CFLAGS)"]
    # nothing else changed: removing the added block gives the base text back
    assert text.count("-ftree-vectorize") == 1
    added = [l for l in text.splitlines() if l not in base.splitlines()]
    assert all("LAYER_CFLAGS" in l or l.startswith("#") or not l for l in added)
    # global CFLAGS line untouched
    assert "CFLAGS  ?= -O3 -funroll-loops -mcpu=$(CPU) -std=c11" in text


@pytest.mark.skipif(shutil.which("make") is None, reason="needs GNU make")
def test_make_applies_flags_to_layers_only(tmp_path):
    build, kernel, layers = _tree(tmp_path)
    _mk(build, kernel, layers, layer_cflags="-DLAYERFLAG_X -ftree-vectorize")
    objs = [str(p.with_suffix(".o")) for p in sorted(layers.glob("*/*.c"))]
    out = subprocess.run(["make", "-n", "CC=cc", "CPU=cortex-a78", *objs,
                          "graph_driver.o", "main.o"],
                         cwd=build, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    cmds = [l for l in out.stdout.splitlines() if " -c " in l]
    assert len(cmds) == 4, out.stdout
    for l in cmds:
        layer = "/layers/" in l
        assert ("-DLAYERFLAG_X -ftree-vectorize" in l) == layer, l
    # `make LAYER_CFLAGS=...` overrides without regenerating
    out = subprocess.run(["make", "-n", "CC=cc", "LAYER_CFLAGS=-DOTHER", objs[0]],
                         cwd=build, capture_output=True, text=True)
    assert "-DOTHER" in out.stdout and "LAYERFLAG_X" not in out.stdout


def _build_unet():
    return _load(ROOT / "example/aiegraph/unet_bk_sdm_tiny/build_unet.py", "build_unet_t")


def _read_flags(path):
    # unet-m4 P4: build_unet.read_layer_cflags_file moved to aiegraph.boardmain.read_cflags_file
    from aiegraph.boardmain import read_cflags_file
    return read_cflags_file(path)


def test_flags_file_six_words_in_order(tmp_path):
    f = tmp_path / "FLAGS"
    f.write_text(" ".join(P5N_FLAGS.split()[:3]) + "\n" + " ".join(P5N_FLAGS.split()[3:]) + "\n")
    flags = _read_flags(f)
    assert flags == P5N_FLAGS
    build, kernel, layers = _tree(tmp_path)
    text = _mk(build, kernel, layers, layer_cflags=flags)
    line = [l for l in text.splitlines() if l.startswith("LAYER_CFLAGS ?=")]
    assert len(line) == 1
    words = line[0].split("?=", 1)[1].split()
    assert words == P5N_FLAGS.split() and len(words) == 6


def test_flags_file_missing_or_empty_is_error(tmp_path):
    bu = _build_unet()
    with pytest.raises(ValueError):
        _read_flags(tmp_path / "nope")
    (tmp_path / "empty").write_text(" \n\n")
    with pytest.raises(ValueError):
        _read_flags(tmp_path / "empty")
    # the CLI refuses (rc 2) before doing any work
    assert bu.main(["--onnx", "x", "--real-io", str(tmp_path), "--out", str(tmp_path / "o"),
                    "--arm-layer-cflags-file", str(tmp_path / "empty")]) == 2


def test_flags_file_comment_lines_skipped(tmp_path):
    # a license header (`#` lines) is not flags; a comment-only file is still empty
    f = tmp_path / "FLAGS"
    f.write_text("#####\n# Copyright (C) 2026 AMD\n  # SPDX-License-Identifier: Apache-2.0\n" + P5N_FLAGS + "\n")
    assert _read_flags(f) == P5N_FLAGS
    (tmp_path / "hdr_only").write_text("# Copyright\n# SPDX-License-Identifier: Apache-2.0\n")
    with pytest.raises(ValueError):
        _read_flags(tmp_path / "hdr_only")


def test_flags_file_committed_example_reads():
    # example/aiegraph/unet_bk_sdm_tiny/a78_layer_cflags.txt carries a license header
    flags = _read_flags(ROOT / "example/aiegraph/unet_bk_sdm_tiny/a78_layer_cflags.txt")
    assert flags.startswith("-mcpu=cortex-a78") and "#" not in flags


def test_build_arm_elf_and_embedded_take_layer_cflags():
    import inspect
    assert "layer_cflags" in inspect.signature(ab.build_arm_elf).parameters
    emb = (ROOT / "src/aiegraph/embedded.py").read_text()
    assert "layer_cflags=layer_cflags" in emb
