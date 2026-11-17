---
name: aiekernelprototype
description: Start a user AIE kernel for a network layer from its generated backbone -- `ag.offload_prototype()` (the format), `emod.offload_prototype("layer4" | region=...)` / `aiegraph prototype layer4 --write dir/` (args in TVM call order with roles and value facts, a proven AIEK_SPEC, and `<entry>.cc` in the conv2dstem_uk.cc architecture: aiehlc kernel frame + host `_run` + entry, compute/staging as TODOs, plus `<entry>_ref.c` and `test_<entry>.cc`, a main() entry file that runs the entry on TVM's dumped tensors (golden) or synthetic ones (smoke)). Use when starting a kernel for a layer, when detect says the args do not match, or when a generated backbone does not build.
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
- `test_<entry>.cc` (`_test_cc` / `_test_data`): an aiehlc entry file --
  `#include "<entry>.cc"` is spliced by `inlineSourceIncludes`, `main()` makes
  hostcompile link an ELF. Golden data comes from
  `<workdir>/arm_build/layeriohex/l<call0>_*_in.h` / `_out.h` (prototype field
  `iohex_dir`, set by `prototype_for`): the header's `_DTYPE` defines give the
  variables in slot order and `_shape[]` must equal the args' shapes, else the
  dump is treated as stale and the test falls back to smoke. The dumps are
  copied FLAT beside the test (aiehlc.sh's flat `*.h` glob). Never put the
  kernel keyword or the launch token in the test text (frontend traps); the
  unit test asserts it. Build with an ABSOLUTE `--runtime-source-file` --
  a relative path became `/test_...cc` and failed. Verified: layer 1 golden and
  layer 4 smoke both build to `aout/worklocal/build/host`; the backbone FAILs at
  run time (`entry returned -4`) until its TODOs are filled -- by design.
- The test TIMES every call (`_test_perf`): clock `XTime` via
  `__has_include("xiltimer.h")` (not `aie_timer.h` -- AIE_GEN is undefined in
  the frontend parse), else `CLOCK_MONOTONIC`; poison/hash stay outside the
  clock. First call reported apart; the repeats are per-call latency INCLUDING
  config (aiehlc re-programs per launch), never "AIE compute". The PMU split
  snapshots `__Runtime_phase_cycles` after launch 0 and reads 0 without
  `aiehlc.sh --profiling` (it says so). `AIEKT_MACS` = useful MACs of the conv
  frame (0 otherwise -> no GOPS line).
