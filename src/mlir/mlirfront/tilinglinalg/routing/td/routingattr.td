/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td" // This line is crucial
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"

def routingdialect : Dialect {
	let name = "routing";
	let cppNamespace = "routing";
	dag discardableAttrs = (ins);
	let hasOperationAttrVerify = 0;
	let useDefaultTypePrinterParser = 1;
	let useDefaultAttributePrinterParser = 1;
	let usePropertiesForAttributes = 0;
}

class routingop<string mnemonic, list<Trait> traits = []> : Op<routingdialect, mnemonic, traits>;
//class routingop<string name, list<Trait>  traits=[]> : Op<routingdialect, name, traits> {}
class routingtype<string name, list<Trait> traits=[]> : TypeDef<routingdialect, name, traits> {}
class routingattr<string name> : AttrDef<routingdialect, name> {}

def TileRangeAttr : routingattr<"TileRange"> {
  let mnemonic = "tilerangeattr";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$rowNum,
    "int64_t":$colNum
  );

  let assemblyFormat = "`{` $rowNum `,` $colNum `}`";
}

def TileRangeArrayAttr : routingattr<"TileRangeArray"> {
  let mnemonic = "TileRangeArrayAttr";
  let summary = "a tile array";
  let cppNamespace = "routing";
  let parameters = (ins
    OptionalArrayRefParameter<"TileRangeAttr">:$items
  );
  let assemblyFormat = "`{` $items `}`";
}

// Spatial-halo (overlapping slab) descriptor of ONE tensor, carried on the op
// that declares it (routing.routingcreatescheduletensor, then
// dfscheblueprint.declare_data). Replaces the old `tensor_N.halo` module
// dictionary plus `routing.spatial_halo_buf_size`. Every optional field
// defaults to 0 = "level absent" (the struct format omits it on print), which
// is exactly how readers treated a missing dictionary key.
def HaloAttr : routingattr<"Halo"> {
  let mnemonic = "halo";
  let summary = "spatial overlapping (halo) tile distribution descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$slice,       // L1 rows each mesh tile owns (overlapping slice)
    "int64_t":$step,        // L1 stride between consecutive tile slices
    "int64_t":$split_dim,   // tensor dim the slices walk
    // 2D width split: per-chunk width geometry (0 => height-only)
    DefaultValuedParameter<"int64_t", "0">:$w_slice,
    DefaultValuedParameter<"int64_t", "0">:$w_step,
    DefaultValuedParameter<"int64_t", "0">:$w_rounds,
    DefaultValuedParameter<"int64_t", "0">:$ow_t,
    // padded DDR row pitch in elements (set with a W or L2 split)
    DefaultValuedParameter<"int64_t", "0">:$row_pitch,
    // L2 (nested) on-core temporal rounds along split_dim
    DefaultValuedParameter<"int64_t", "0">:$l2_slice,
    DefaultValuedParameter<"int64_t", "0">:$l2_step,
    DefaultValuedParameter<"int64_t", "0">:$l2_rounds,
    // K-contraction accumulate split (independent of the L2 split)
    DefaultValuedParameter<"int64_t", "0">:$k_slice,
    DefaultValuedParameter<"int64_t", "0">:$k_step,
    DefaultValuedParameter<"int64_t", "0">:$k_rounds,
    // contiguous per-round IFM slab the kernel window holds, in elements
    DefaultValuedParameter<"int64_t", "0">:$buf_size
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// User DmaTransform for ONE tensor's shim BD (element strides/wraps plus the
// BD iteration dim). Replaces the old `tensor_N.shim_dma` module dictionary.
def ShimDmaAttr : routingattr<"ShimDma"> {
  let mnemonic = "shim_dma";
  let summary = "user-specified multi-dim shim DMA addressing of one tensor";
  let cppNamespace = "routing";
  let parameters = (ins
    "::mlir::ArrayAttr":$strides,    // i32 element strides, innermost first
    "::mlir::ArrayAttr":$wraps,      // i32 wraps, same order
    "int64_t":$iter_step,
    "int64_t":$iter_wrap,
    OptionalParameter<"::mlir::ArrayAttr">:$ddr_shape,  // i64 DDR shape (tile_m base math)
    DefaultValuedParameter<"int64_t", "0">:$tile_m_alignment
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// The blocks a kernel-only kernel takes on ONE port, as its traced
// aie::load_buffer / store_buffer calls give them (convautotile kernel
// analysis): per core `rounds_i * rounds_j` blocks of `shape` elements, block
// (i, j) at the core's base + step_i * i + step_j * j (i outer). `pp_depth` and
// `tile` are the kernel's {.depth} / {.resident} / {.tile} options (0 = not
// given). The lowering checks its core windows against this and follows it
// where the GEMM schedule cannot express the kernel (an input taken once while
// other ports loop, full-K strips, a panel-major .tile).
def KernelBlockAttr : routingattr<"KernelBlock"> {
  let mnemonic = "kernel_block";
  let summary = "per-core block sequence a kernel-only kernel takes on one port";
  let cppNamespace = "routing";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tensor_shape, // the whole tensor, elements (row-major in DDR)
    ArrayRefParameter<"int64_t">:$shape,   // block extent per tensor dim, elements
    "int64_t":$rounds_i,                    // outer loop rounds
    "int64_t":$rounds_j,                    // inner loop rounds
    ArrayRefParameter<"int64_t">:$step_i,  // offset per outer round, per dim
    ArrayRefParameter<"int64_t">:$step_j,  // offset per inner round, per dim
    DefaultValuedParameter<"bool", "false">:$resident,
    DefaultValuedParameter<"int64_t", "0">:$tile_r,
    DefaultValuedParameter<"int64_t", "0">:$tile_c
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// Kernel-level on-core round schedule that is not a property of any single
// tensor. Carried on the routing func (`routing.kernel_tiling`), which lives
// through every stage. Replaces the flat `routing.{tile_m,tile_rows,m_rounds,
// tile_n,tile_cols,n_rounds,effective_k,full_k,k_rounds,spatial_out_rounds}`
// module attrs on the paths that have no #routing.tiling op
// (fullconnect_auto=0, spatial halo). 0 = not set, matching the old
// "attr absent" reads.
def KernelTilingAttr : routingattr<"KernelTiling"> {
  let mnemonic = "kernel_tiling";
  let summary = "kernel on-core round schedule";
  let cppNamespace = "routing";
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "0">:$tile_m,
    DefaultValuedParameter<"int64_t", "0">:$tile_rows,
    DefaultValuedParameter<"int64_t", "0">:$m_rounds,
    DefaultValuedParameter<"int64_t", "0">:$tile_n,
    DefaultValuedParameter<"int64_t", "0">:$tile_cols,
    DefaultValuedParameter<"int64_t", "0">:$n_rounds,
    DefaultValuedParameter<"int64_t", "0">:$effective_k,
    DefaultValuedParameter<"int64_t", "0">:$full_k,
    DefaultValuedParameter<"int64_t", "0">:$k_rounds,
    // output slabs per core per launch (spatialMRounds * spatialNRounds)
    DefaultValuedParameter<"int64_t", "0">:$out_rounds
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// Recursive per-dimension tiling level. Each level describes one tiling
// granularity over a tensor dimension; `slice_tiling` (optional) nests the next
// (finer/on-core) level. A null `slice_tiling` marks a leaf level. Custom
// assembly format prints the nested level with its full `#routing.level<...>`
// mnemonic (the auto struct(params) form would strip the prefix).
def LevelAttr : routingattr<"Level"> {
  let mnemonic = "level";
  let summary = "recursive per-dimension tiling level descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$base,    // full extent this level partitions
    "int64_t":$total,   // slice * rounds (covered extent, incl. overlap)
    "int64_t":$slice,   // elements each step owns
    "int64_t":$step,    // stride between consecutive steps
    "int64_t":$rounds,  // number of steps at this level
    OptionalParameter<"routing::LevelAttr">:$slice_tiling // nested finer level (null => leaf)
  );
  let hasCustomAssemblyFormat = 1;
}

// One tensor dimension's tiling: a single outer level (which may nest a finer
// slice_tiling level). Custom assembly so `outer` prints the full
// `#routing.level<...>` mnemonic rather than the stripped struct form.
def DimAttr : routingattr<"Dim"> {
  let mnemonic = "dim";
  let summary = "per-dimension tiling descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "routing::LevelAttr":$outer,
    DefaultValuedParameter<"int64_t", "0">:$axis   // 0=on-core, 1=mesh_row(group1), 2=mesh_col(group2)
  );
  let hasCustomAssemblyFormat = 1;
}

// Whole-tensor tiling: one DimAttr per tensor dimension, printed as d0/d1/...
def TilingAttr : routingattr<"Tiling"> {
  let mnemonic = "tiling";
  let summary = "per-tensor tiling descriptor (one DimAttr per dim)";
  let cppNamespace = "routing";
  let parameters = (ins
    OptionalArrayRefParameter<"routing::DimAttr">:$dims
  );
  let hasCustomAssemblyFormat = 1;
}

// Mesh-partition descriptor (the former scalar partitiontensor args).
def PartitionAttr : routingattr<"Partition"> {
  let mnemonic = "partition";
  let summary = "tensor mesh-partition descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$splitnum,
    "int64_t":$splitdim,
    StringRefParameter<>:$hwAxisOwner,
    StringRefParameter<>:$replicateOn,
    StringRefParameter<>:$singleTileOwner
  );
  let assemblyFormat = "`<` struct(params) `>`";
}
#endif
