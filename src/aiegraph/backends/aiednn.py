"""Flags of ``torch.compile(model, backend="aiednn")`` (aiegraph.dnn_torch_backend), in the spirit of
``torch.backends.cudnn``. Importing this module registers the backend with torch.compile.

benchmark  -- time every implementation that accepts each problem on the first run and keep the
              fastest (aiednn_set_autotune; aiednn_primitive_find(MEASURE) once it exists)
mode       -- "f32" (default) or "int8" (calibrated on the example inputs)
mlir       -- optimize regions with the aiednn MLIR pipeline (dnn_mlir.PIPELINE)
allow_ref  -- False: a region whose primitives would all run on the CPU reference stays torch code
              (interim lowering path)
fuse_residual_requant -- graph API fusion level 2 (aiednn_graph_set_fusion): also absorb a residual add into
              the conv anchor, dropping the conv output's requantization before the add -- faster PT2E
              graphs, NOT exact vs eager (off by default; the graph API's level 1 fusions are exact)
graph      -- None (default): partition with the graph API (aiednn_graph.h) when the tree has it and
              mode is "f32", else the interim dnn_torch.lower regions; True: require it; False: never
dump       -- a directory: write the op graph at every stage of the graph-API path there
              (dnn_torch_viz, index.md + a .dot / .svg per stage); dump_consts also draws constants
"""
benchmark = False
mode = "f32"
mlir = False
allow_ref = True
graph = None
fuse_residual_requant = False
dump = None
dump_consts = False

try:  # register backend="aiednn" when torch is importable
    from ..dnn_torch_backend import register as _register
    _register()
except ImportError:
    pass
