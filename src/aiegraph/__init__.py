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
    module.build(target="a78", output="resnet18_a78")  # main.elf
    print(module.run(profile=True))                    # on palmyra; module.run(ag.VEK385, "host")
    print(module.profile(ag.ALL))                      # each layer's time on the board
    report = module.AI_perf_analysis("layer1")         # why it is slow (claude + skill kernelperfcheck)
    result = module.AI_kernel_accelerate("layer1", report)  # claude optimizes a copy; gated

    print(ag.offload_prototype())                      # the kernel-file format
    proto = module.offload_prototype("layer1")         # one layer: args, AIEK_SPEC, kernel backbone
    proto.write("my_kernels/")                         # <entry>.cc (kernel+host+entry) + <entry>_ref.c

    module = ag.compile("resnet18", "aie", "layer1=src/aietensorop/conv2dstem/conv2dstem_uk.cc")

The layer-index API (``offload_check("layer1", k.cc)``, ``offload``) still works;
there ``layer1`` is TVM kernel 1, not torchvision's ``layer1``.

Install: ``pip install -e <repo>`` (then ``aiegraph doctor``). Design:
``doc/design/aiegraph_python_api.md``. Kernels: ``doc/tutorial_userkernel.md``.
Only ``run()`` touches a board (through ``script/test/apppaltest.py`` /
``appvek385.py``); everything else stays on this machine.
"""

from __future__ import annotations

from pathlib import Path

from ._env import doctor, setup
from .contract import KernelContract
from .board import PALMYRA, VEK385
from .embedded import EmbeddedModel, Layer, convert, current, embedded_convert
from .embedded import open_workdir as open  # noqa: A001  (ag.open(workdir))
from .errors import AiegraphError
from .prototype import Prototypes, KernelFormat, OffloadPrototype
from .model import ZOO, Model, download_model, from_onnx, from_torch, load
from .region import Region, Regions
from .report import (AIProtoResult, BuildReport, CheckReport, DetectReport, PartitionPlan,
                     KernelOptResult, PerfReport, ProfileReport, RegionCheck, RunReport)

__version__ = "0.1.0"

#: ``emod.profile(ag.ALL)``: every call / layer.
ALL = "all"

__all__ = ["load", "from_onnx", "from_torch", "download_model", "ZOO", "Model",
           "convert", "embedded_convert", "EmbeddedModel", "Layer", "open", "current",
           "Region", "Regions", "KernelContract", "RegionCheck", "PartitionPlan",
           "offload_check", "offload", "offload_reset", "detect", "lint", "build", "run_local",
           "run", "profile", "ALL", "PALMYRA", "VEK385", "RunReport", "ProfileReport",
           "PerfReport", "KernelOptResult",
           "compile", "guide", "doctor", "setup", "AiegraphError", "CheckReport",
           "DetectReport", "BuildReport", "AIProtoResult", "offload_prototype", "ai_offload_prototype",
           "OffloadPrototype", "Prototypes", "KernelFormat"]


# ── module-level forms: act on ag.current() (pyplot-style convenience) ─────

def offload_check(spec, source=None, **kw) -> CheckReport:
    """``ag.offload_check("layer1", "k.cc")`` on the current model."""
    return current().offload_check(spec, source, **kw)


def offload(spec, source=None, **kw) -> CheckReport:
    """``ag.offload(("layer1", "k.cc"))`` on the current model."""
    return current().offload(spec, source, **kw)


def offload_prototype(layer=None, *, region=None, entry: str = None):
    """What a kernel file must look like to be offloaded.

    ``ag.offload_prototype()`` -- the kernel-file format and rules (no model).
    ``ag.offload_prototype("layer1")`` / ``(region="conv1")`` -- that layer of the
    current model: arguments in TVM call order with kind/dtype/shape/role, an
    ``AIEK_SPEC`` proven against every call, and ``.write(dir)`` a kernel-file
    backbone (aiehlc kernel frame + host run function + entry; compute TODO)."""
    if layer is None and region is None:
        from .prototype import kernel_format
        return kernel_format()
    return current().offload_prototype(layer, region=region, entry=entry)


def ai_offload_prototype(layer=None, **kw):
    """``ag.ai_offload_prototype("layer4")`` on the current model: the backbone, then
    ``claude -p`` fills its TODOs until ``prototype_check`` says it is correct."""
    return current().ai_offload_prototype(layer, **kw)


def offload_reset(layers=None):
    """Put offloaded layers of the current model back on the CPU (all, or *layers*)."""
    return current().offload_reset(layers)


def detect(source) -> DetectReport:
    """Which layers of the current model the kernel in *source* fits."""
    return current().detect(source)


def build(**kw) -> BuildReport:
    return current().build(**kw)


def run_local(**kw) -> BuildReport:
    return current().run_local(**kw)


def run(board: str = PALMYRA, host: str = None, **kw) -> RunReport:
    """``ag.run()`` / ``ag.run(ag.VEK385, "hostname")``: the current model's
    ``main.elf`` on a lab board (see ``EmbeddedModel.run``)."""
    return current().run(board, host, **kw)


def profile(which=ALL, **kw) -> ProfileReport:
    """``ag.profile(ag.ALL)``: per-layer board time of the current model's last run."""
    return current().profile(which, **kw)


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
