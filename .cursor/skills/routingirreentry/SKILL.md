---
name: routingirreentry
description: flowcheck FAILED / `FLOWCHECK ERROR: ... does not reach / also reaches` after a build; editing aiehlc's routing IR (ir/dfschedule/0_initial.mlir) and feeding it back -- `AIEHLC_ROUTING_IR_IN`, the `aiebackend.routing_ir` Python API (RoutingIR view / set_pp_depth / set_level / set_attr / lint / check / build), the round-trip test, or an IR edit that "does nothing", builds but hangs, or is accepted although illegal.
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
- dmaphop: `AIEHLC_DMAPHOP_IR_IN` (`3_DmapToDmaphopPass.mlir`) round-trips byte-identical
  (`routing_ir_roundtrip.sh --stage dmaphop`), but EDITS are not safe yet: swapping a
  flow's shim column or channel builds, verifies (0 errors) and computes wrong data on
  the board -- other flows' stream routes get re-paired. Read-only for tuning; see
  doc/design/routing_ir_reentry.md "dmaphop re-entry".
- `script/test/flowcheck.py <worklocal>` (runs at the end of every aiehlc.sh tiling build
  and in `RoutingIR.check()`): routing.cc simulated against the dmaphop provenance map.
  `does not reach` / `also reaches` / `several flows` = a flow re-paired onto another's
  consumers -- RoutingHWVerify passes such builds; the board computes wrong data. Shim
  port map Gen5: MM2S ch0/1 -> SOUTH 3/7, S2MM ch0/1 <- SOUTH 1/3.
