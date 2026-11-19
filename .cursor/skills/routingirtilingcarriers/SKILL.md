---
name: routingirtilingcarriers
description: Where tiling / per-tensor DMA information lives in the routing IR (typed attrs on ops, not module attrs) and how to add or read a field. Use when adding halo/shim_dma/pp_depth/kernel-round data, when code looks for `tensor_N.halo`, `tensor_N.shim_dma`, `routing.pp_depth_map`, `routing.spatial_halo_buf_size`, `routing.spatial_out_rounds` or flat `routing.tile_*`/`*_k`/`*_rounds` module attrs (all removed), or when a value reads 0 inside a conversion pattern.
---

# Routing IR tiling carriers

## Rule
The routing **module** carries only control switches (`routing.fullconnect_auto`,
`kernel_config_offload`, `output_tile_major`, `control_*`, `control_packet_*`,
`iter_policy`). Everything else is a typed attribute on an op:

| Info | Carrier |
|---|---|
| halo slab geometry + slab size | `#routing.halo<...>` on `routing.routingcreatescheduletensor` (`halo`) |
| user DmaTransform | `#routing.shim_dma<...>` (`shim_dma`) on the same op |
| `dma_shuffle` / `core_shuffle` | `layout_transform` (StrAttr), same op |
| ping-pong depth | `pp_depth` (I64Attr), same op |
| fca=0 / halo kernel rounds, `out_rounds` | `#routing.kernel_tiling<...>` on the routing `func.func` |
| GEMM fca=1 tiling | `#routing.tiling` on `routing.partitiontensor` |
| kernel-only port's traced blocks + `.resident`/`.tile` | `#routing.kernel_block` (`kernel_block`) on the tensor op; `resolveKernelBlock` (skill gemmdataflowoptions) |

`DmaphopTodfscheblueprintPass` copies the per-tensor attrs onto
`dfscheblueprint.declare_data` as `routing.halo` / `routing.shim_dma` /
`routing.layout_transform` / `routing.pp_depth` / `routing.kernel_block`
(`copyTensorRoutingInfo`).
Design: `doc/design/routing_ir_tiling_carriers.md`.

## Reading
`routing::collectTensorRoutingInfo(module)` → map keyed by **func-arg index**;
`lookupHalo`, `spatialHaloBufSize`, `sortedTensorIndices`, `readKernelTiling`.
Read them in `runOnOperation` **before `applyPartialConversion`** and pass them
down (`BlueprintPassState::tensorInfo` / `kernelTiling`, pattern ctor args):
the declaring op is itself converted, so a lookup from inside a pattern can
find nothing and silently read 0.

## Traps
- Optional `HaloAttr`/`KernelTilingAttr` fields are **0 = absent**. Old code that
  defaulted `l2_rounds`/`k_rounds` to 1 must keep a `> 0 ? x : 1` guard.
- A new per-tensor attr must also be added to `copyTensorRoutingInfo` and
  `collectTensorRoutingInfo`, or it vanishes at the blueprint stage.
- `DenseMap` order is unstable — iterate `sortedTensorIndices` when "first halo
  tensor" matters (`detectConvHalo`, `spatialHaloBufSize`).
- The generated `createscheduletensor` builder takes the five optional attrs
  explicitly: pass `nullptr` for each.
- `aiehlc.cc` includes `routingmanager.h`; the top-level `CMakeLists.txt` adds
  `tilinglinalg/routing/` and `routing/inc/` to its include path.
- Verifying a change: compare `aout/` sources and ELFs byte-for-byte, but the
  provenance JSON maps (`dfscheduleprovenancemap.json`, `schedule_view.json`'s
  `invariant_checks`) reorder **between two runs of the same compiler** — compare
  them order-insensitively.
