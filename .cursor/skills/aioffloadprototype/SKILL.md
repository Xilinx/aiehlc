---
name: aioffloadprototype
description: Let Claude complete a layer's kernel backbone -- `emod.ai_offload_prototype("layer4")` / `aiegraph aiprototype layer4` (offload_prototype writes it, `claude -p` fills the TODOs, `prototype_check` judges) -- and the gates themselves (`emod.prototype_check`, `aiegraph protocheck`, `kemu` x86 emulation of kernel + host vs TVM). Use when a prototype's verdict FAILS, kemu reports a mismatch / ASan crash / KEMU ERROR, or when offload_check says READY for a kernel that cannot be right (an unfilled backbone).
---
<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# AI-completed kernel prototypes (`ai_offload_prototype`)

Code: `src/aiegraph/aiprototype.py` (loop, gates), `src/aiegraph/kemu.py`
(emulator); tutorial `doc/tutorial_aiegraph.md` §4c; backbone: skill
**aiekernelprototype**.

- **`offload_check` READY does not mean the kernel is right.** Its calibrate
  and emulate run `<entry>_ref.c`; the prototype's reference is COMPLETE for
  the conv + requantize layers, so a backbone with every TODO unfilled is
  READY. Only `kemu` executes `<entry>.cc`. Never use `offload_check` alone as
  the verdict for a kernel.
- Gates, in order: `todo` (regex `\bTODO\s*[(:]` -- prose like "TODOs filled"
  must not count), `kemu`, `trace` (only if the file has a kernel),
  `check`, and `build` only once all pass (aiehlc in `<dir>/aiehlc_build/`,
  absolute `--runtime-source-file`, ELF must be newer than the run).
- kemu model: `aie::port<W, aie::IFM|WEIGHTS|OFM, aie::shape<..>>` is a
  template over the window type (eb/out from `input_window_int8` etc.);
  `load_buffer` = gather into `malloc(exact bytes)` (ASan = out-of-block read),
  `store_buffer` = 0xA5-poisoned block, `commit` = scatter, blocks held at
  kernel end = `KEMU ERROR`. `name<<<mesh>>>(` is rewritten to
  `kemu::launch(mesh, name, ...)` (regex), which runs every core; trailing
  int launch args are ignored. Golden: TVM's `make TRACE=1` dump of the
  layer's first call (shapes must match, else stale -> reference), else
  `<entry>_ref` on synthetic tensors (fails if the reference is still a TODO).
  It models data, not DMA timing; host code that needs more than
  `aieSetDevice` / `aieArray::partition` / `__Runtime_Alloc` will not compile
  there -- extend `_PRELUDE`, do not bend the kernel.
- Verified mutants: wrong rounding, wrong store offset (both mismatch), missing
  commit (KEMU ERROR), read one pixel past the block (ASan) -- all FAIL; the
  solved layer-1 example passes bit-exact against TVM in ~2 s.
- Claude's tools: Read/Grep/Glob, Edit/Write inside the prototype dir only,
  Bash only `python -m aiegraph protocheck`. Prompt on stdin (variadic
  `--allowedTools`). A rerun moves the old dir to `.prev-<time>` (skill
  **aistagenodelete**); never rmtree.
- `prototype_check` re-derives the prototype (same `entry=`) each call, so its
  verdict is against the current model, not the files' own spec.
