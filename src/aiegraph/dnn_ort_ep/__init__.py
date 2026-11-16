"""aiednn as an ONNX Runtime execution provider (plugin EP library, onnxruntime >= 1.23)."""
from __future__ import annotations

import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .. import _env
from ..errors import AiegraphError

HERE = Path(__file__).resolve().parent
_REGISTERED = {}


def ort_include() -> Path:
    """ONNX Runtime's C/C++ API headers for the installed onnxruntime version"""
    import onnxruntime as ort
    env = os.environ.get("AIEDNN_ORT_INCLUDE")
    if env:
        return Path(env)
    ver = ort.__version__.split("+")[0]
    root = _env.repo_root() / "worklocal" / "aiednn" / "ort_headers" / ver
    inc = root / f"onnxruntime-linux-x64-{ver}" / "include"
    if (inc / "onnxruntime_ep_c_api.h").is_file():
        return inc
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "ort.tgz"
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-linux-x64-{ver}.tgz"
    try:
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root, members=[m for m in t.getmembers() if "/include/" in m.name])
    except Exception as exc:  # noqa: BLE001
        raise AiegraphError("dnn_ort_ep", f"no ONNX Runtime {ver} headers ({exc})",
                            hint="set AIEDNN_ORT_INCLUDE to the include/ of the matching onnxruntime release")
    tgz.unlink(missing_ok=True)
    return inc


def build_ep(out=None, mesh=None) -> Path:
    """libaiednn_ort_ep.so: the EP + the aiednn runtime with the graph API"""
    from ..dnn import impl_list_source
    from ..kernel_lib import kernel_lib
    repo = _env.repo_root()
    if not (repo / "include" / "aiednn_graph.h").is_file():
        raise AiegraphError("dnn_ort_ep", "the partition graph API (include/aiednn_graph.h) is not in this tree")
    out = Path(out or repo / "worklocal" / "aiednn" / "ort_ep")
    out.mkdir(parents=True, exist_ok=True)
    so = out / "libaiednn_ort_ep.so"
    srcs = [HERE / "aiednn_ort_ep.cc", HERE / "aiednn_ort_map.h"]
    lib = kernel_lib(repo / "src" / "aietensorop", quiet=True)
    picked = [e for e in lib if e.contract.source.name == "gemm_uk.cc" and "vek280" in e.contract.spec.targets]
    impl = out / "impl_list.cc"
    text = impl_list_source(picked, x86=True, mesh=mesh, count=bool(picked))
    if not impl.is_file() or impl.read_text() != text:
        impl.write_text(text)
    (out / "graph_tu.cc").write_text('#include "aiednn_rt.cc"\n#include "aiednn_rt_graph.h"\n')
    deps = srcs + [impl] + list((repo / "src" / "aiednn").glob("*")) + [repo / "include" / "aiednn.h"]
    if so.is_file() and all(so.stat().st_mtime >= p.stat().st_mtime for p in deps if p.is_file()):
        return so
    mdefs = [f"-DHW_ROWS={mesh[0]}", f"-DHW_COLS={mesh[1]}"] if mesh else []
    cmd = ["g++", "-O2", "-shared", "-fPIC", "-std=gnu++17", "-fvisibility=hidden", *mdefs,
           f"-I{repo / 'include'}", f"-I{repo / 'src' / 'aietensorop' / 'ops'}", f"-I{repo / 'src' / 'aiednn'}",
           f"-I{ort_include()}", str(HERE / "aiednn_ort_ep.cc"), str(out / "graph_tu.cc"), str(impl),
           "-x", "c", str(repo / "src" / "aietensorop" / "gemm" / "gemm_uk_ref.c"), "-x", "none", "-o", str(so), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise AiegraphError("dnn_ort_ep", "build failed:\n" + res.stderr[-4000:])
    return so


def register(so=None, name: str = "aiednn") -> str:
    """register the EP library with onnxruntime (once per process)"""
    import onnxruntime as ort
    if name not in _REGISTERED:
        so = Path(so or build_ep())
        ort.register_execution_provider_library(name, str(so))
        _REGISTERED[name] = so
    return name


def _pin_batch(model, batch: int):
    """the model with every dynamic leading input dim fixed to *batch*"""
    import onnx
    m = onnx.load(str(model)) if not isinstance(model, onnx.ModelProto) else model
    changed = False
    for i in m.graph.input:
        d = i.type.tensor_type.shape.dim
        if d and not d[0].dim_value:
            d[0].Clear()
            d[0].dim_value = batch
            changed = True
    return m.SerializeToString() if changed or not isinstance(model, (str, Path)) else str(model)


def session(model, benchmark: bool = False, options=None, cpu_fallback: bool = True, batch: int = 1,
            name: str = "aiednn"):
    """an InferenceSession with aiednn first and ORT's CPU EP for what aiednn does not claim"""
    import onnxruntime as ort
    register(name=name)
    model = _pin_batch(model, batch)
    so = options or ort.SessionOptions()
    if benchmark:
        so.add_session_config_entry("ep.aiednn.benchmark", "1")
    devs = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not devs:
        raise AiegraphError("dnn_ort_ep", f"onnxruntime lists no {name} device after registration")
    so.add_provider_for_devices(devs, {})
    if not cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model, sess_options=so)


def assigned(sess) -> dict:
    """{"aiednn": n, "CPUExecutionProvider": m}"""
    getter = getattr(sess, "get_provider_graph_assignment_info", None)
    if getter is None:
        return {}
    out = {}
    for sub in getter():
        out[sub.ep_name] = out.get(sub.ep_name, 0) + len(getattr(sub, "nodes", []))
    return out
