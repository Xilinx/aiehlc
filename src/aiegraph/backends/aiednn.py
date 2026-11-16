"""Flags of ``torch.compile(model, backend="aiednn")``"""
benchmark = False
mode = "f32"
mlir = False
allow_ref = True
graph = None
fuse_residual_requant = False

try:  # register backend="aiednn" when torch is importable
    from ..dnn_torch_backend import register as _register
    _register()
except ImportError:
    pass
