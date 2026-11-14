---
name: multiaiehlclibrary
description: Two user-kernel offloads (`offload("layer1", a.cc)` then `offload("layer4", b.cc)`) and only the last stays on the AIE / partition.json says `superseded by user kernel ...: one aiehlc library per ELF`; link gate reports `_Z7routingP12XAie_DevInst` defined twice; how several aiehlc libraries share one ELF (per-library routing).
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Several aiehlc libraries in one ELF

Mechanism (runtime + codegen + aiekernel):
- `aie_runtime.c`: the global `routing()` reference is WEAK.
  `__Runtime_init_mesh_partition_with(meshId, startCol, numCols, route)` keeps
  `route` in the partition registry; another library's `route` on the same
  meshId -> `__Runtime_explicit_teardown` + re-init with it. The old
  `__Runtime_init_mesh_partition` = `_with(..., NULL)` (global routing).
- Generated `aieArray::partition` (emitted by `tilinglinalg_pipeline.cpp` --
  the copy that reaches host.cc -- and by aiehlc.cc's prelude) declares
  `void routing(XAie_DevInst*) __attribute__((weak));` and passes `routing`.
- `aiekernel/build.py`: `isolate_archive` makes `routing` LOCAL when the
  merged own object references `__Runtime_init_mesh_partition_with`
  (`multi_library`); else keeps it a strong hook. `_stamp` includes the
  `build/aiehlc` binary, so a cached tree from an older aiehlc rebuilds.
- `aiekernel/flow.py`: `_write_partition` supersedes another AIE row only
  when either library is old-style; gate 6a no longer rejects a second kernel.

Diagnose: `aarch64-none-elf-nm -A layers/*/aie/build/libaiek_*.a | grep -E
'init_mesh_partition|routingP12'` -- new-style: `U ..._with...` and
`t _Z7routingP12XAie_DevInst` (lower-case = local). `U _Z29__Runtime_init_mesh_partitioniii`
+ `T routing` = old-style: rebuild it with the current aiehlc (re-run offload).
Single-library apps: routing.cc / kernel.cc / bcf / prx unchanged; host.cc only
switches to the `_with` call. Board behavior of a library switch (partition
re-init per launch) is not yet measured.
