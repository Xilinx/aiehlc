# Whole models through aiednn -- the MODEL COMPILER layer

`run_models.py` lowers PyTorch models (`src/aiegraph/dnn_torch.py`), optimizes them with the
`aiednn` MLIR dialect (`dnn_mlir.PIPELINE`), emits a C app that calls only `include/aiednn.h`,
builds it with `ag.dnn_build` and runs it (workstation emulation or VEK280).

This is the layer **above** the aiednn library (`doc/design/aiednn.md` §0): it decides which
primitives to call and how to fuse them; per-model megakernel mode selection (`modes="auto"`)
and co-resident placement are optimizations here. Operator coverage and per-shape AIE
implementations belong to the library (`src/aiednn/`, `src/aietensorop/`).
