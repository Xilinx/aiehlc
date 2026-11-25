# aiednn runtime -- the LIBRARY layer

This directory is the aiednn primitive library's runtime: validation, the implementation list,
dispatch, the `ref:` implementations and the composites (`blocked:`, `im2col+`, `lut:`, ...)
that drive the AIE kernels in `src/aietensorop/`. Public API: `include/aiednn.h`.

**Library, not model compiler** (`doc/design/aiednn.md` §0). Nothing here may know about a
model: no model names, op indices or graph positions, no MLIR, no Python. A faster
implementation for some shape is a new `AIEK_SPEC` / impl entry that wins dispatch by
specificity, with a bit-exact `ref:` fallback. Fusion across primitives, memory planning and
megakernel planning belong to the model compiler (`src/aiegraph/dnn_torch.py`, `dnn_mlir.py`,
`src/mlir/mlirfront/frontend/aiednn/`), which talks to this layer only through `aiednn.h`.

Owner: the `aie-kernel-library-expansion` session. Tests: `src/aiegraph/test/test_aiednn_ops.py`
(every primitive vs PyTorch), `example/aiednn/ops_app.cc` (board).
