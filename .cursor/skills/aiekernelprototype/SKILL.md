---
name: aiekernelprototype
description: Start a user AIE kernel for a network layer from its generated backbone -- `ag.offload_prototype()` (the format), `emod.offload_prototype("layer4" | region=...)` / `aiegraph prototype layer4 --write dir/` (args in TVM call order with roles and value facts, a proven AIEK_SPEC, and `<entry>.cc` in the conv2dstem_uk.cc architecture: aiehlc kernel frame + host `_run` + entry, compute/staging as TODOs, plus `<entry>_ref.c`). Use when starting a kernel for a layer, when detect says the args do not match, or when a generated backbone does not build.
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Kernel backbone of a layer (`offload_prototype`)

Code: `src/aiegraph/prototype.py`; contract `include/aie_user_kernel.h`;
tutorial `doc/tutorial_aiegraph.md` §4b.

- Layer numbering is TVM's (as `offload_check`); `region=` for ONNX names.
- The backbone is the LAYER's part only -- ports/shapes, mesh split, block
  loop, host buffers + launch, entry, spec. Compute, staging (NCHWc -> HWC),
  parameter delivery and output unpacking are `TODO(...)` for the user; until
  then `_run` returns -1 and the entry `AIEK_ERR_KERNEL`. Do not grow it into
  a full implementation -- the user asked for a backbone.
- The tile is chosen by `_tile_candidates` (pipeline + 32 KB buffer region +
  stack rules) and then `build/aiehlc-convtile trace` on the generated file; a
  tile that passes both builds (`libaiek_layer01.a`, `libaiek_layer04.a`).
  A link failure means the rules drifted from the pipeline: skill
  **dmbufferregion**.
- Only the 4x4 mesh (`device.partition({3, 6, 0, 6}, 4, 4)`): 4x8 segfaults in
  `DmapToDmaphop`.
- The spec is parsed with `aiekernel.spec` and matched with `aiekernel.match`
  against every call (`verified: ... N/N call(s)`); a failure there is a
  generator bug. `require=zero` only for constants zero in EVERY call.
- Kernel frames: TVM NCHWc conv2d. Other ops: entry + spec skeleton.
- `_ref.c` is generated COMPLETE for the fused conv + requantize signature
  (`_is_requant`: roles zp/bias/zp/mult/shift, arg 6 all zero), so
  `write("./")` + `offload_check("./aiek_layerNN.cc")` is READY as written
  (verified layers 1 and 4). `ref=-4` in calibrate = a reference still at its
  `return AIEK_ERR_KERNEL;` placeholder (or files regenerated over an edit).
- Shapes are concrete; for one kernel over several geometries turn dims into
  NAMEs by hand and re-run `detect`.
