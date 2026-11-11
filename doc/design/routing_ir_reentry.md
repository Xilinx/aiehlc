# Routing IR re-entry and the routing IR API (agents and tuners)

Status: implemented 2026-11-10 (step 1: re-entry + round trip; step 2: Python API).
Motivation: give aiehlc the "edit the IR, feed it back" tuning loop of TVM's TIRx
(https://tvm.apache.org/2026/06/22/tirx): compiler IR that an agent can inspect,
mutate and check, with dense compiler signals instead of only a board result.

## Why the routing IR

The routing IR is what the frontend hands to `TilingLinalgPipeline::runPipeline`:
mesh (`routingcreatehwmesh`), per tensor `pp_depth` / `#routing.halo` / `shim_dma` /
`layout_transform` (`routingcreatescheduletensor`), per partition `#routing.partition`
and `#routing.tiling` (one `#routing.dim` per tensor dim, nested `#routing.level`
base/total/slice/step/rounds), `routing.kernel_tiling` on the func and the control
switches on the module. Every later pass reads its tiling from these carriers at pass
entry (routing_ir_tiling_carriers.md), so downstream IR (dmap, dmaphop, blueprint,
schedule) and the generated host / kernel / routing C are **re-derived consistently**
from an edit. Editing later IR is not recommended: after the blueprint the host and
kernel paths each own a clone that must agree on every tile's BDs and locks
(passblueprintlowering/helper/), and a one-sided edit hangs the board.

## Step 1: re-entry (`AIEHLC_ROUTING_IR_IN`)

`runPipeline` dumps its input as `ir/dfschedule/0_initial.mlir`. With
`AIEHLC_ROUTING_IR_IN=<file.mlir>` (`{kernel}` expands to the kernel name for
multi-kernel sources) it parses and verifies that file and replaces the frontend's
module with it before anything reads the module; the kernel body, port names and
tensor params still come from the frontend. The log says whether the file differs:

    [TilingLinalg] routing IR read from edit.mlir (AIEHLC_ROUTING_IR_IN): 1 line(s) differ from the generated IR

Round trip (`script/test/routing_ir_roundtrip.sh <src.cc>`: build, feed
`0_initial.mlir` back unchanged, compare): **byte-identical** kernel.cc / routing.cc /
host.cc / aieml.bcf / aieml.prx and all 28 IR stage dumps for
`simplegemm_kernel_port.cc` and `simpleconv2dkernel_port.cc` -- the routing IR holds
the whole tiling decision for both; no frontend state outside it changed the result.

Edits that take effect, board-verified (palmyra 213): output C `pp_depth` 2 -> 1
(host.cc + aieml.bcf change, `PASS: all 4096 output elements match`).

## Step 2: Python API (`aiebackend.routing_ir`)

C++ (`_aiebackend`, `frontend/aiebackend/routing_ir_api.cpp`) does the IR work through
the dialect; `routing_ir.py` wraps it:

| call | what |
|---|---|
| `RoutingIR.load(path)` / `.view` | `routing_ir_parse`: module + func attrs, mesh, tensors (`arg`, `direction` input/output from the gather, `shape`, attrs), partitions (`partition`, typed `tiling` dims with `axis` and nested `levels`) |
| `.set_pp_depth(arg, n)`, `.set_level(tensor_arg, dim, depth, **fields)`, `.set_attr(target, attr, mlir_text, **select)`, `.edit(*dicts)` | `routing_ir_edit`: typed level rebuild or any attribute by MLIR attribute text; the result always parses and verifies; `history` records every edit for replay |
| `.diff(other)` | unified diff of the IR text |
| `.lint()` | static findings in milliseconds (below) |
| `.check(work_dir)` | `routing_ir_check`: verify, then the whole pipeline in-process in a scratch dir (well under a second), log captured; `CheckResult(ok, verified, diagnostics, errors, buffers, core_free_bytes, lint, artifacts, log)` |
| `build(src, ir_path)` | full aiehlc build with `AIEHLC_ROUTING_IR_IN`; returns rc, the re-entry line, the ELF |

`lint()` rules -- each holds on every IR known to build correctly (GEMM, conv halo,
simplematmul2): every level `total == slice * rounds`, `step <= slice`, positive
fields; a nested level's `base` equals its parent's `slice`; `splitnum` equals the
mesh size on its axis and (without halo) divides the split dim; plus effect notes.

Tests: `frontend/aiebackend/test/test_routing_ir.py` (9; fixtures are the GEMM and
conv `0_initial.mlir`), `script/test/routing_ir_roundtrip.sh`.

## What the signals showed (worth knowing before tuning)

- **Not every attribute is live.** `pp_depth` on an *input* tensor reaches the
  blueprint but no code generator reads it (only output Pull flows,
  `passdmaphoptodfscheblueprint.cpp`); `lint()` says so instead of letting a tuner
  spend a build on a no-op.
- **The pipeline accepts some illegal tilings silently.** `splitnum = 3` on a 4-row
  mesh builds with no error; `lint()` reports it. A tuner must gate on `lint()` as well
  as `check()`.
- **Real edits propagate.** A's K chunk 64 -> 128 (`set_level(0, dim=1, depth=1, ...)`)
  doubles the input windows to 2048 B and shrinks free core memory 28416 -> 24064 B in
  `check()`.

## Limits

- The IR does not carry the kernel body. A block-load kernel's own
  `aie::load_buffer` blocks follow the source, not an IR edit: changing the DMA in the
  IR (e.g. the K chunk above) without changing the kernel is caught only at run time
  (klog `BLKE`). Frontend-substituted `aie::get_*` values also come from the source.
  Tile-size changes therefore belong in the kernel source; the IR is the place for
  scheduling and data movement the kernel does not see (pp_depth, partition and
  routing choices, control switches).
- `check()` runs the pipeline with no kernel body: it proves the data movement
  compiles and fits, not that the kernel computes the right thing (`build()` + board).
- Expressing new dataflows (cascade, neighbour forwarding) needs new routing
  constructs, not attribute edits.

## Next

Typed primitives over this API that keep the kernel and the IR in step (e.g. "K chunk
= n" edits both the kernel constant and the levels), then a search loop (convtune
extended to GEMM) gated on `lint()` + `check()` and confirmed by `build()` + board.
