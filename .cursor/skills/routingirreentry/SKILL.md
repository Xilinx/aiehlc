---
name: routingirreentry
description: Editing aiehlc's routing IR (ir/dfschedule/0_initial.mlir) and feeding it back -- `AIEHLC_ROUTING_IR_IN`, the `aiebackend.routing_ir` Python API (RoutingIR view / set_pp_depth / set_level / set_attr / lint / check / build), the round-trip test, or an IR edit that "does nothing", builds but hangs, or is accepted although illegal.
---

# Routing IR re-entry (agents / tuners)

Design + measurements: `doc/design/routing_ir_reentry.md`.

- Feed back: `AIEHLC_ROUTING_IR_IN=<file.mlir> source script/aiehlc.sh ...`
  (`{kernel}` = kernel name). The log line `routing IR read from ... N line(s)
  differ` proves the edit was seen; `identical to the generated IR` = no-op file.
- Prove a source round-trips before trusting edits on it:
  `script/test/routing_ir_roundtrip.sh <src.cc>` (byte-identical outputs + 28 IR stages).
- Python: `PYTHONPATH=src/mlir/mlirfront/frontend`, `from aiebackend.routing_ir import
  RoutingIR, build` (needs `make _aiebackend`). Gate every candidate on BOTH
  `ir.lint()` and `ir.check().ok`: the pipeline silently builds illegal tilings
  (splitnum 3 on a 4-row mesh). `check()` has no kernel body; confirm with
  `build()` + board.
- "My edit did nothing": `pp_depth` on an input tensor is never read by codegen (only
  output Pull flows); `lint()` flags it. Diff the generated files, not just the IR.
- "Builds but hangs / BLKE": the kernel source still loads its old blocks. IR edits
  that change a tensor's per-core blocks (tiling levels) must be mirrored in the
  kernel (or done in the source); `aie::get_*` values come from the source too.
- Edit the routing IR, never dmaphop/blueprint/schedule dumps: host and kernel clones
  must agree on every tile's BDs and locks.
