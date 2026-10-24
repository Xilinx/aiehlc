###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph -- ONNX / PyTorch model -> int8 C on the A78, with layers on the AIE.

    import aiegraph as ag

    model  = ag.load("resnet18")                       # zoo name, .onnx path or torch.nn.Module
    module = ag.convert(model, target="a78")           # int8, one C file per layer
    print(module.regions)                              # conv1, layer1.0.conv1, ..., fc
    module.register_kernel(name="conv2dstem", op="conv2d",
                           source="src/aietensorop/conv2dstem/conv2dstem_uk.cc",
                           constraints={"dtype": "int8", "quantization": "per_channel"})
    print(module.check_offload(region="conv1", kernel="conv2dstem").summary())
    module.assign(region="conv1", backend="aie:conv2dstem")
    print(module.partition(fallback="cpu").report())
    module.build(target="a78", output="resnet18_a78")  # main.elf (never runs it)

    module = ag.compile("resnet18", "aie", "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc")

The layer-index API (``offload_check("layer1", k.cc)``, ``offload``) still works;
there ``layer1`` is TVM kernel 1, not torchvision's ``layer1``.

Install: ``pip install -e <repo>`` (then ``aiegraph doctor``). Design:
``doc/design/aiegraph_python_api.md``. Kernels: ``doc/tutorial_userkernel.md``.
Nothing in this package ever runs on a board.
"""

from __future__ import annotations

from pathlib import Path

from ._env import doctor, setup
from .contract import KernelContract
from .embedded import EmbeddedModel, Layer, convert, current, embedded_convert
from .embedded import open_workdir as open  # noqa: A001  (ag.open(workdir))
from .errors import AiegraphError
from .model import ZOO, Model, download_model, from_onnx, from_torch, load
from .region import Region, Regions
from .report import BuildReport, CheckReport, DetectReport, PartitionPlan, RegionCheck

__version__ = "0.1.0"

__all__ = ["load", "from_onnx", "from_torch", "download_model", "ZOO", "Model",
           "convert", "embedded_convert", "EmbeddedModel", "Layer", "open", "current",
           "Region", "Regions", "KernelContract", "RegionCheck", "PartitionPlan",
           "offload_check", "offload", "detect", "lint", "build", "run_local",
           "compile", "guide", "doctor", "setup", "AiegraphError", "CheckReport",
           "DetectReport", "BuildReport"]


# ── module-level forms: act on ag.current() (pyplot-style convenience) ─────

def offload_check(spec, source=None, **kw) -> CheckReport:
    """``ag.offload_check("layer1", "k.cc")`` on the current model."""
    return current().offload_check(spec, source, **kw)


def offload(spec, source=None, **kw) -> CheckReport:
    """``ag.offload(("layer1", "k.cc"))`` on the current model."""
    return current().offload(spec, source, **kw)


def detect(source) -> DetectReport:
    """Which layers of the current model the kernel in *source* fits."""
    return current().detect(source)


def build(**kw) -> BuildReport:
    return current().build(**kw)


def run_local(**kw) -> BuildReport:
    return current().run_local(**kw)


def lint(source):
    """Static checks on a kernel source; needs no model."""
    from ._env import frontend
    frontend("")
    from frontend.tvmrelay.aiekernel.lint import lint_source
    src = Path(source)
    if not src.is_file():
        raise AiegraphError("check", f"{source} does not exist")
    return lint_source(src)


def compile(model, target: str = "cpu", offload=None, *, calib=None, image=None,  # noqa: A001
            workdir=None, build: bool = True, strict: bool = True, verbose: bool = False,
            example_inputs=None) -> EmbeddedModel:
    """One call: convert, optionally offload, optionally build.

    ``target``: ``"cpu"``/``"a78"`` (all layers on the A78), ``"x86"`` (build and
    run ``main_local.elf`` here), ``"aie"`` (needs *offload*). *offload*: any spec
    ``EmbeddedModel.offload`` takes -- ``"layer1=k.cc"``, a tuple, a dict, a list,
    or a bare kernel path meaning "every layer it fits". With ``strict`` any
    failed gate raises; otherwise failing layers stay on the CPU and the
    report is kept in ``emod.offload_report``.
    """
    if target == "aie" and not offload:
        raise AiegraphError("compile", 'target "aie" needs offload=',
                            hint='ag.compile(m, "aie", "layer1=path/to/kernel.cc")')
    emod = embedded_convert(model, target=target, calib=calib, image=image,
                            workdir=workdir, verbose=verbose, example_inputs=example_inputs)
    emod.offload_report = None
    if offload:
        rep = emod.offload(offload, verbose=verbose)
        emod.offload_report = rep
        print(rep)
        if strict:
            rep.raise_for_status()
    if build:
        rep = emod.run_local(verbose=verbose) if target == "x86" else emod.build(verbose=verbose)
        print(rep)
        if strict and not rep.ok and target != "x86" and "toolchain" not in (rep.reason or ""):
            raise AiegraphError("build", rep.reason, hint="emod.build(verbose=True)")
    return emod


def guide() -> str:
    """Print where you are and the next call."""
    from . import embedded
    emod = embedded._CURRENT
    if emod is None:
        msg = ("step 1/4: get a model and convert it\n"
               '  m = ag.load("resnet18")            # or ag.from_torch(net), ag.load("x.onnx")\n'
               '  emod = ag.embedded_convert(m, target="a78")\n'
               "  (environment problems? ag.doctor())")
    else:
        aie = [l.name for l in emod.layers if l.placement != "cpu"]
        arm = emod.workdir / "arm_build"
        elf, drv = arm / "main.elf", arm / "graph_driver.c"
        if aie and emod.last_build is None and elf.is_file() and drv.is_file() \
                and elf.stat().st_mtime >= drv.stat().st_mtime:
            emod.last_build = BuildReport(True, str(elf))   # built in an earlier session
        if not aie:
            msg = ("step 2/4: pick layers and prove your kernel on x86\n"
                   "  print(emod)                                   # layers\n"
                   '  emod.detect("my_kernel.cc")                   # which layers it fits\n'
                   '  emod.offload_check("layer1", "my_kernel.cc")  # lint+calibrate+emulate\n'
                   "  (write a kernel: doc/tutorial_userkernel.md)")
        elif not (emod.last_build and emod.last_build.ok):
            msg = (f"step 4/4: on the AIE: {', '.join(aie)} -- link the ELF\n"
                   "  emod.build()        # main.elf; run it on a board yourself")
        else:
            msg = f"done: {emod.last_build.elf}"
    print(msg)
    return msg
