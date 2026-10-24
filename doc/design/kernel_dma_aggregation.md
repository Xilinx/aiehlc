# DfscheduleKernelAggregationPass & the per-tile DMA operands

`pass/passdfschedulekernelaggregation/`, gated on `routing.kernel_config_offload`.
Collapses the **per-core-tile** core DMA config into one BD group per window.

Related: [kernel_config_offload.md](kernel_config_offload.md).

---

## What it collapses

`BlueprintToScheduleKernelPass` emits a `declaretile` / `dma_bd` / `create_io` /
`start_io` group **per tile** — 48 groups / 96 `dma_bd` on a 4×4 mesh with 3 windows.
But one `kernel.cc` is broadcast to every core tile, so those 96 describe at most **two**
distinct configurations:

- **S2MM** — identical on every tile (circuit-switched: no packet id, no ooo bd).
- **MM2S** — differs only in `packet_id` and `ooo_bd_id`.

The pass rewrites each cluster onto **`dfschedule.declaretile.self`** — the tile
executing this kernel, resolved on-core via `get_coreid()` — carrying the fan-out as
discardable attrs `aggregated` / `tile_coords` / `tile_packet_ids` / `tile_ooo_bd_ids`.

Result: 96 → 6 `dma_bd`; IR 2002 → 247 lines.

One self-tile is created at the top of the kernel module body so it dominates every
group; the now-dead per-tile `declaretile`s are swept along with the redundant BDs.

## Why `declaretile.self` is its own op

It is a distinct **op**, not a `declaretile {col = -1, row = -1}` sentinel, so that "has
no coordinates" is a *type-level* fact: all ~18 `getDefiningOp<DeclareTileOp>()` sites
return null for it and **fail closed**, rather than silently reading `-1`.

> The sentinel was also easy to test wrongly — `DeclareTileOp::getCol()` returns
> `uint32_t`, so `getCol() < 0` is always false while `getCol() == -1` works.

It produces the same `!dfschedule.tile`, so the four kernel-path consumers —
`config.dma_bd`, `config.create_io`, `schedule.getbdid`, `bind_core_buffer` — accept it
unchanged. `load_kernel_group` is host-only.

**It is kernel-path only.** `DfscheduleToApiPass` walks for it up front and fails the
build with an explicit message: the host lowers a tile handle to `XAie_TileLoc(col,row)`,
and a self-tile would otherwise become a bogus address.

## `packet_id` / `out_of_order_bd_id` are SSA operands, not attributes

Declared in `td/dfscheduleop.td`, sitting alongside `bd_id` and `offset`:

```
dfschedule.config.dma_bd(%buf, %tile, %bd_id, %offset, %packet_id, %ooo_bd_id [, %linked_bd])
```

`linked_bd` stays **last** in the operand list because it is the only optional one —
that is what lets the hand-written parser in `dfschedulemanager.cpp` disambiguate it
positionally. The generated `getODSOperandIndexAndLength` derives its variadic offset
from the non-optional count, so no `AttrSizedOperandSegments` is needed.

They must remain `arith.constant`-foldable: the kernel path bakes them into `kernel.cc`
and has no runtime path for them. So `DfscheduleKernelAggregationPass` **hard-fails** on
a non-constant (same contract as `bd_id`), while on the host side the provenance comment
degrades to `-1`, the provenance JSON to `"runtime"`, and `ScheduleCanonicalizePass`
simply never folds.

> **Watch out:** that pass's merge-legality check compares the *attribute dictionary*.
> These two fields leaving it would make a per-tile difference **invisible** rather than
> merely unchecked — so the check also compares operands, skipping exactly indices 4 and
> 5 (`isPerTileVaryingOperand`).

## `ooo_bd_id` on the kernel clone

`out_of_order_bd_id` derives from `FlowLoweringCtx::shimPerTileBdIds`, which only the
*host* clone populates (`helper/flowtransfer_host.cpp`) — on the kernel clone that vector
is empty.

`emitCoreBufferDma` therefore falls back to the value the host published per
`(col, row, MM2S, flow)` into the `ResourceMgr` **singleton** (`coreOffloadPlan()`), the
one channel that survives the clone.

> Without that fallback, every MM2S core `dma_bd` in the stage-15 dump read `-1` while
> the emitted `kernel.cc` carried real ids — the IR actively misdescribed the hardware,
> and anyone reading it would conclude the core targeted no particular shim BD.

Both `packet_id` and `ooo_bd_id` now exist in the op *and* the plan as independent
derivations. The aggregation pass cross-checks them and **fails the build** on
disagreement: a wrong ooo id routes a tile's output into another tile's shim BD and
surfaces only as a silent data mismatch on hardware.

## Merge and erasure rules

**Mergeability is verified strictly:** every attribute except those two must match across
a cluster, and every tile must have a plan entry. The merge key is an allowlist of
*varying* fields, so a newly added `dma_bd` attribute defaults to being **compared** — a
silent wrong merge would program the wrong DMA registers and still compile.

**Erasure is a fixpoint sweep, not per-cluster.** The BDs of one tile form a chain
(pong `dma_bd` → ping's `linked_bd` → `create_io`), and ping/pong land in *different*
clusters — so cluster-by-cluster erasure would orphan a pong, which `ConfigDmaBdOp`'s
verifier rejects.

## Downstream

`DfscheduleToKernelApiPass::emitCoreDmaConfigBlocks` reads the aggregated ops
(`collectAggregatedOutTiles`) and falls back to `core_offload_plan` when the pass did not
run. `emitWindowBdAndLocks` is unchanged, so `kernel.cc` is **byte-identical either way**.

Tested by `unitest/test.cpp` → `./test kernelagg`.
