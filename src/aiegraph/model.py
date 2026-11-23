###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.Model``: an fp32 network, from the ONNX zoo, a file, or a torch.nn.Module.

Every model is normalized to ONE ONNX file whose input is named ``data``, the
name the TVM flow (``deploy_flow.INPUT_NAME``) imports by. PyTorch models are
exported to ONNX (opset 13) and join the same path -- the only path that
reaches per-layer C today (``frontend/pytorchmlir`` stops at linalg MLIR).

Scope of the downstream flow today: image classifiers with a
``float32[1,3,224,224]`` input (ImageNet preprocessing, the ELF's input image
and the ONNX-PTQ calibration reader are all that shape). Anything else is
refused here, by name, rather than failing three stages later.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
import json
from pathlib import Path

from .errors import AiegraphError

__all__ = ["Model", "load", "from_onnx", "from_torch", "download_model", "ZOO", "DEFAULT_MODEL"]

#: Model zoo: name -> (url, file name), read from the data file ``zoo.json`` beside this
#: module (see its ``_doc`` for why the default is a v1 export: BN folds into the conv).
_ZOO_DATA = json.loads((Path(__file__).resolve().parent / "zoo.json").read_text())
ZOO = {k: (v["url"], v["file"]) for k, v in _ZOO_DATA["models"].items()}
#: Zoo name used when the caller names no model.
DEFAULT_MODEL = _ZOO_DATA["default"]
#: Stem of the default model's generated ``<stem>.c`` / ``<stem>_graph.json``.
DEFAULT_MODULE_STEM = _ZOO_DATA["models"][DEFAULT_MODEL]["module"]

INPUT_NAME = "data"
INPUT_SHAPE = [1, 3, 224, 224]


def cache_dir() -> Path:
    d = Path(os.environ.get("AIEGRAPH_CACHE", Path.home() / ".cache" / "aiegraph"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Model:
    """An fp32 model, ready for :func:`aiegraph.embedded_convert`."""

    name: str
    onnx_path: Path
    source: str                     # "onnx-zoo" | "onnx" | "torch->onnx"
    input_shape: list = field(default_factory=list)
    output_shape: list = field(default_factory=list)
    ops: dict = field(default_factory=dict)
    input_shapes: dict = field(default_factory=dict)   # name -> shape, every graph input

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.onnx_path.read_bytes()).hexdigest()[:16]

    def __fspath__(self) -> str:
        return str(self.onnx_path)

    def convert(self, target: str = "a78", **kw):
        """``model.convert(target="a78")`` == ``ag.convert(model, target="a78")``."""
        from .embedded import embedded_convert
        return embedded_convert(self, target=target, **kw)

    def __repr__(self) -> str:
        n = sum(self.ops.values())
        top = ", ".join(f"{k} x{v}" if v > 1 else k
                        for k, v in sorted(self.ops.items(), key=lambda kv: -kv[1])[:8])
        return (f"Model {self.name}  ({self.source}, fp32)  input {INPUT_NAME} "
                f"float32{self.input_shape} -> float32{self.output_shape}\n"
                f"  {n} ops: {top}{' ...' if len(self.ops) > 8 else ''}\n"
                f"  file: {self.onnx_path}\n"
                f"  next: ag.convert(model, target=\"a78\")")


def _dims(value_info) -> list:
    return [d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in value_info.type.tensor_type.shape.dim]


def _normalize(path: Path, name: str, source: str) -> Model:
    """Validate the input contract and rename the input to ``data`` if needed."""
    import onnx

    try:
        proto = onnx.load(str(path))
    except Exception as exc:                      # noqa: BLE001
        raise AiegraphError("load", f"{path} is not a readable ONNX model ({exc})",
                            hint="pass an .onnx file, a zoo name (ag.ZOO), or use "
                                 "ag.from_torch(module, example_inputs=...)")
    inits = {i.name for i in proto.graph.initializer}
    inputs = [i for i in proto.graph.input if i.name not in inits]
    if not inputs:
        raise AiegraphError("load", f"{name} has no graph inputs")
    # aiegraph-multiinput: N inputs, any static shape. A dynamic leading
    # (batch) dim becomes 1; any other dynamic dim is refused by name, since the
    # flow compiles one static graph.
    input_shapes = {}
    for i in inputs:
        s = _dims(i)
        if s and (s[0] in ("?", 0) or isinstance(s[0], str)):
            s = [1] + s[1:]                        # dynamic batch -> 1
        bad = [d for d in s[1:] if not isinstance(d, int) or d <= 0]
        if bad:
            raise AiegraphError("load", f"{name}: input {i.name} has dynamic dims {s}",
                                hint="export the model with static shapes")
        input_shapes[i.name] = s
    inp = inputs[0]
    shape = input_shapes[inp.name]
    ops = {}
    for node in proto.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    out_shape = _dims(proto.graph.output[0]) if proto.graph.output else []
    if len(inputs) == 1 and inp.name != INPUT_NAME:
        old = inp.name
        inp.name = INPUT_NAME
        for node in proto.graph.node:
            node.input[:] = [INPUT_NAME if x == old else x for x in node.input]
        dims = inp.type.tensor_type.shape.dim
        if dims and not dims[0].dim_value:
            dims[0].dim_value = 1
        path = cache_dir() / "models" / f"{name}.data.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(proto, str(path))
    if len(inputs) == 1:
        input_shapes = {INPUT_NAME: shape}
    return Model(name=name, onnx_path=Path(path).resolve(), source=source,
                 input_shape=shape, output_shape=out_shape, ops=ops,
                 input_shapes=input_shapes)


def _fetch(name: str) -> Path:
    """Fetch a zoo model into ``~/.cache/aiegraph/models`` (once). Returns its path."""
    if name not in ZOO:
        raise AiegraphError("load", f"unknown model {name!r}",
                            hint=f"zoo has: {', '.join(ZOO)}; or pass a path to an .onnx file")
    url, fname = ZOO[name]
    path = cache_dir() / "models" / fname
    if not path.is_file() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[aiegraph] downloading {name} from {url}")
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as exc:                  # noqa: BLE001
            raise AiegraphError("load", f"download failed: {exc}",
                                hint=f"fetch {url} by hand and pass the file to ag.load()")
        tmp.replace(path)
    return path


def download_model(name: str = DEFAULT_MODEL) -> Model:
    """Fetch a zoo model (once) and load it: ``ag.load(name)`` for zoo names.

    Returns a :class:`Model`, so ``m = ag.download_model(<zoo name>);
    m.convert("a78")`` works. A ``Model`` is path-like (``os.fspath(m)`` is the
    ONNX file), for callers that used the file path this returned before. Named
    by the file stem (``<zoo file>`` without ``.onnx``), as ``ag.convert(<that path>)`` named
    it, so the default workdir ``worklocal/aiegraph/<name>`` stays the same.
    """
    path = _fetch(name)
    return _normalize(path, path.name.split(".")[0], "onnx-zoo")


def from_onnx(path, name: str = None) -> Model:
    path = Path(path)
    if not path.is_file():
        raise AiegraphError("load", f"{path} does not exist")
    return _normalize(path, name or path.name.split(".")[0], "onnx")


def from_torch(module, example_inputs=None, name: str = None) -> Model:
    """Export a ``torch.nn.Module`` to ONNX (opset 13, TorchScript exporter) and load it."""
    import torch

    if example_inputs is None:
        example_inputs = (torch.randn(*INPUT_SHAPE),)
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)
    name = name or type(module).__name__.lower()
    out = cache_dir() / "models" / f"{name}.torch.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    kwargs = dict(input_names=[INPUT_NAME], output_names=["output"], opset_version=13,
                  do_constant_folding=True)
    try:
        try:
            torch.onnx.export(module, example_inputs, str(out), dynamo=False, **kwargs)
        except TypeError:                         # torch without the dynamo= switch
            torch.onnx.export(module, example_inputs, str(out), **kwargs)
    except Exception as exc:                      # noqa: BLE001
        raise AiegraphError("load", f"torch.onnx.export failed: {exc}",
                            hint="the module must be exportable at batch 1, 3x224x224")
    return _normalize(out, name, "torch->onnx")


def load(model=DEFAULT_MODEL, source: str = None, example_inputs=None, name: str = None) -> Model:
    """Anything model-like -> :class:`Model`.

    ``"<zoo name>"`` (``ag.ZOO``) | ``"<torchvision builder>", source="torchvision"`` |
    ``"path/to/model.onnx"`` | a ``torch.nn.Module`` | a :class:`Model`.
    """
    if isinstance(model, Model):
        return model
    if not isinstance(model, (str, Path)):
        return from_torch(model, example_inputs, name)
    text = str(model)
    if source == "torchvision":
        try:
            import torchvision
        except ImportError:
            raise AiegraphError("load", "torchvision is not installed",
                                hint="pip install -e '<repo>[torch]'")
        builder = getattr(torchvision.models, text, None)
        if builder is None:
            raise AiegraphError("load", f"torchvision.models has no {text!r}")
        return from_torch(builder(weights="DEFAULT"), example_inputs, name or text)
    if text in ZOO and not Path(text).exists():
        return _normalize(_fetch(text), name or text, "onnx-zoo")
    if text.endswith(".onnx") or Path(text).is_file():
        return from_onnx(text, name)
    raise AiegraphError("load", f"don't know how to load {text!r}",
                        hint=f"use a zoo name ({', '.join(ZOO)}), an .onnx path, or a "
                             "torch.nn.Module")


def copy_into(model: Model, dest: Path) -> None:
    """Place the model where ``deploy_flow.fetch_model`` looks for it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.read_bytes() != model.onnx_path.read_bytes():
        shutil.copyfile(model.onnx_path, dest)
