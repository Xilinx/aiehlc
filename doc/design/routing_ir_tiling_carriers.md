# Routing IR: tiling lives on ops, the module carries only control

The routing module's attribute dictionary holds **only control switches**: the
pragma/policy flags `routing.fullconnect_auto`, `routing.kernel_config_offload`,
`routing.output_tile_major`, `routing.control_*`, `routing.control_packet_*`,
`routing.iter_policy`. Everything that describes *data movement or tiling* is a
typed attribute on the IR op it belongs to.

```mlir
module attributes {routing.control_packet_mode = 0 : i64, ..., routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, ...)
      attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    %2 = routing.routingcreatescheduletensor %1 : tensor<230x230x4xi8> shape = [230, 230, 4], dim = 3
           {halo = #routing.halo<slice = 61, step = 56, split_dim = 0, w_slice = 61, w_step = 56,
                                 w_rounds = 4, ow_t = 28, row_pitch = 920, l2_slice = 19, l2_step = 14,
                                 l2_rounds = 4, k_slice = 244, k_step = 224, k_rounds = 4, buf_size = 4636>,
            layout_transform = "dma_shuffle", pp_depth = 2 : i64} -> tensor<230x230x4xi8>
```

(`simpleconv2d.cc`, `ir/dfschedule/0_initial.mlir`.)

## Where each piece lives

| Was (module attribute) | Now | Carrier op |
|---|---|---|
| `tensor_N.halo` (dictionary) | `#routing.halo<...>` | `routing.routingcreatescheduletensor` of func arg N (`halo`) |
| `routing.spatial_halo_buf_size` | `#routing.halo` `buf_size` | same (every halo tensor) |
| `tensor_N.shim_dma` (dictionary) | `#routing.shim_dma<...>` | same (`shim_dma`) |
| `tensor_N.layout_transform` | `StrAttr` | same (`layout_transform`) |
| `routing.pp_depth_map` | `I64Attr` per tensor | same (`pp_depth`) |
| `tensor_N.core_shuffle` | removed (was written, never read) | — |
| `routing.{tile_m,tile_rows,m_rounds,tile_n,tile_cols,n_rounds,effective_k,full_k,k_rounds}` on the `fullconnect_auto=0` / spatial-halo paths | `#routing.kernel_tiling<...>` | routing `func.func` (`routing.kernel_tiling`) |
| `routing.spatial_out_rounds` | `#routing.kernel_tiling` `out_rounds` | same |
| GEMM `fullconnect_auto=1` tiling | unchanged: `#routing.tiling` on `routing.partitiontensor` (`readGemmTilingScalars`) | `routing.partitiontensor` |
| (new) the blocks a kernel-only kernel takes on one port | `#routing.kernel_block<tensor_shape, shape, rounds_i, rounds_j, step_i, step_j, resident, tile_r, tile_c>` | `routing.routingcreatescheduletensor` (`kernel_block`) |

`DmaphopTodfscheblueprintPass` replaces each `routingcreatescheduletensor` by a
`dfscheblueprint.declare_data` and copies the five per-tensor attributes onto it
as `routing.halo` / `routing.shim_dma` / `routing.layout_transform` /
`routing.pp_depth` / `routing.kernel_block` (`routing::copyTensorRoutingInfo`). The dfscheblueprint
dialect does not depend on the routing dialect, so on `declare_data` they are
discardable, dialect-prefixed attributes; on the routing op they are declared in
ODS and type-checked. The routing `func.func` survives every stage of both the
host and the kernel clone, so `kernel_tiling` needs no copying.

Field semantics: every optional `#routing.halo` / `#routing.kernel_tiling`
parameter defaults to **0 = absent** and the struct format omits it on print.
That is exactly how the old readers treated a missing dictionary key (`if (auto a
= getAs(...))` with a 0 default; the `l2_rounds`/`k_rounds` reads that defaulted
to 1 are all guarded `> 0 ? x : 1`). `pp_depth` is the exception: it is
`std::optional` in `TensorRoutingInfo` because 0 is a value, not absence.

`kernel_block` is set only for kernel-only sources (convautotile's kernel
analysis traced the `aie::load_buffer` / `store_buffer` calls). Its
`tensor_shape` is the tensor **as the kernel declares it** -- not the
`TensorParam` shape, which the GEMM path relabels (B as `[K, N]` while its bytes
are `[N, K]`). The blueprint lowering resolves it through one shared helper,
`blueprint_sched::resolveKernelBlock` (`helper/flowtransfer_common.cpp`), so host
and kernel always agree: on the GEMM path (`fullconnect_auto = 1`, no halo) every
such port is **checked** (final core window and round count = the kernel's block
and block count, else a build error), and an input whose blocks are
row-contiguous is **followed** (window, rounds, per-round DDR offset, `.tile`
shim dims and its own `pp_depth` come from the kernel). Design:
[trace_to_routing.md](trace_to_routing.md), "Dataflow options".

## Reading it (`routing/routingmanager.h`)

| Call | Returns |
|---|---|
| `collectTensorRoutingInfo(module)` | `DenseMap<funcArgIdx, TensorRoutingInfo{halo, shimDma, layoutTransform, ppDepth}>` from every `createscheduletensor` / `declare_data` |
| `lookupTensorRoutingInfo(map, idx)` / `lookupHalo(map, idx)` | one tensor's descriptors / halo |
| `spatialHaloBufSize(map)` | the kernel's halo slab size (`buf_size` of the lowest-index halo tensor) |
| `sortedTensorIndices(map)` | func-arg indices ascending — iterate this, `DenseMap` order is unstable |
| `readKernelTiling(module)` | the routing func's `#routing.kernel_tiling` (all-zero when absent) |
| `mergeKernelTiling(module, KernelTilingFields)` / `setHaloBufSize(module, n)` | writers used by `aiehlc.cc` |

**Read at pass entry, never inside a conversion pattern.** The declaring op is
itself converted (`createscheduletensor` → `declare_data` in
DmaphopTodfscheblueprint, `declare_data` erased in BlueprintToSchedule), so every
pass collects the map in `runOnOperation` before `applyPartialConversion` and
hands it down: `PushOpConversion`/`PullOpConversion` (constructor),
`BlueprintPassState::tensorInfo` / `::kernelTiling` (host and kernel blueprint
passes, `flowtransfer_*` helpers, `detectConvHalo`, `buildOutputTileDescriptor`,
`analyzeKernelParams`). This is the same rule `readGemmTilingScalars` follows
(`docs/plans/2026-07-24-deprecate-module-tiling-attrs.md`).

## Adding a new per-tensor or per-kernel field

1. Add the parameter to `HaloAttr` / `ShimDmaAttr` / `KernelTilingAttr` in
   `routing/td/routingattr.td` (default 0 if it is optional), or a new
   `OptionalAttr` on `routing_createscheduletensor` in `routingop.td`. The
   `.inc` files are regenerated by the build (`routing/gen.sh`).
2. Set it where the value is known: `attachTensorRoutingInfo` in
   `tilinglinalg_pipeline.cpp` (per tensor) or `mergeKernelTiling` in `aiehlc.cc`.
3. A new per-tensor attribute also needs a line in `copyTensorRoutingInfo` and
   `collectTensorRoutingInfo`, or it stops at the blueprint stage.
4. Read it from `passState->tensorInfo` / `passState->kernelTiling` (or the map
   your pass collected at entry). Do **not** add a module attribute.

## Verification

Built `example/tileprogram/ccode/simpleconv2d.cc`,
`src/aietensorop/conv2dstem/test_conv2d.cc` (control-plane pragmas,
`output_tile_major`, `KERNELCONFIGOFFLOAD`) and
`example/tileprogram/ccode/simplematmul2.cc` (GEMM, `fullconnect_auto=1`) before
and after: `main.elf`, the kernel ELF, `host.cc`, `kernel.cc`, `routing.cc`, bcf,
prx are byte-identical; the provenance/resource JSON maps are identical up to
element order (which already varies run to run — `DfscheduleProvenanceMapPass`
keys a `std::map` by `Operation*`); every `ir/dfschedule` and `ir/simplerouting`
dump differs only by the moved attributes.
