"""
/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
"""

'''
This a tiling example to simulate following things
1. design a structure to represent a piece data after tiling
2. connect data into tile
'''
class logictile:
    def __init__(self, id):
        print("{} is id".format(id))
        self.id = id

class tiling:
    class buffer_dimention:
        def __init__(self, x, y ,z):
            print("buffer_dimetnion")

    class tiling_dimention:
        def __init__(self, x, y, z):
            print("tiling_dimention")

    class offset_dimention:
        def __init__(self, x, y, z):
            print("offset_dimention")

    class tile_travesal:
        def __init__(self, x, y, z):
            print("tile_travesal")
        
    def __init__(self):
        print("tiling")

class data:
    def __init__(self, tilem):
        print("data")

class connect:
    def __init__(self, data, ltile):
        print(ltile.id)

    print("tiling")

lt = logictile(1)
connect(lt)


'''
// XilinxDialect.td  
def Xilinx_WrapSubviewOp : Xilinx_Op<"wrap_subview",   
    [NoSideEffect, DeclareOpInterfaceMethods<MemoryEffectOpInterface>]> {  
  let summary = "Represents a subview on a buffer with wrap-around in one or more dimensions";  
  let description = [{  
    This op allows specifying wrap-around access in certain dimensions.  
    The offset, size, and wrap arrays must match the rank of the source memref.  
  }];  
  
  let arguments = (ins  
    AnyMemRef:$source,  
    I64ArrayAttr:$offsets,  
    I64ArrayAttr:$sizes,  
    I64ArrayAttr:$wrapDimensions // array of 0/1 flags for each dimension  
  );  
  
  let results = (outs  
    AnyMemRef:$result  
  );  
}  

#include "mlir/IR/PatternMatch.h"  
#include "mlir/Dialect/SCF/IR/SCF.h"  
// ... other includes for your Xilinx dialect, etc.  
  
using namespace mlir;  
  
// Forward declare our TilingOp and WrapSubviewOp classes  
// (generated from ODS .inc files), or include the .h file that has them  
class XilinxTilingOp;  
class Xilinx_WrapSubviewOp;  
  
namespace {  
struct XilinxTilingLowering : public OpRewritePattern<XilinxTilingOp> {  
  using OpRewritePattern<XilinxTilingOp>::OpRewritePattern;  
  
  LogicalResult matchAndRewrite(XilinxTilingOp tilingOp,  
                                PatternRewriter &rewriter) const override {  
    // Gathers array attributes for offsets, sizes, wraps, etc.  
    SmallVector<int64_t, 4> offsetVals = tilingOp.getOffsets();  
    SmallVector<int64_t, 4> sizeVals   = tilingOp.getTilingDimensions();  
    SmallVector<int64_t, 4> wrapVals   = tilingOp.getWrapFlags();  
    Value inputBuf = tilingOp.getInputBuffer();  
  
    // For simplicity, assume rank=3  
    // E.g: offset = [o0, o1, o2], sizes = [s0, s1, s2], wrap = [w0, w1, w2]  
    // If wrap=1 in any dimension, we will create a Xilinx_WrapSubviewOp  
    // else we build a standard memref.subview.  
  
    bool anyWrap = false;  
    for (auto w : wrapVals) {  
      if (w == 1) {    
        anyWrap = true;  
        break;  
      }  
    }  
  
    if (!anyWrap) {  
      // No wrap, use standard subview  
      // 1) Build static/dynamic indices if needed  
      // 2) Create MemRefType for subview result  
      auto baseType = inputBuf.getType().cast<MemRefType>();  
      auto resultType = // ... derived from baseType, sizeVals, maybe dynamic dims  
  
      // Create subview op  
      Value subView = rewriter.create<memref::SubViewOp>(  
          tilingOp.getLoc(),   
          resultType,   
          inputBuf,   
          getOffsetOperands(offsetVals, rewriter, tilingOp.getLoc()),  
          getSizeOperands(sizeVals, rewriter, tilingOp.getLoc()),  
          getStrideOperands(/* default stride = 1 */));  
  
      // Replace tilingOp with subView or continue transformations  
      // ...  
    } else {  
      // We have at least one dimension with wrap=1 -> create Xilinx_WrapSubviewOp  
      auto loc = tilingOp.getLoc();  
  
      // Build arrays for offsets, sizes, wrapFlags as attributes  
      auto wrapSubviewOffsets = rewriter.getI64ArrayAttr(offsetVals);  
      auto wrapSubviewSizes   = rewriter.getI64ArrayAttr(sizeVals);  
      auto wrapSubviewWraps   = rewriter.getI64ArrayAttr(wrapVals);  
  
      // We'll produce a new result memref to represent the "wrapped" subview  
      // Possibly dynamic shape or known shape  
      auto baseType = inputBuf.getType().cast<MemRefType>();  
      // Example type derivation:  
      auto resultType = mlir::MemRefType::get(  
          llvm::SmallVector<int64_t, 4>(sizeVals.begin(), sizeVals.end()),  
          baseType.getElementType());  
  
      Value wrapSubviewOp = rewriter.create<Xilinx_WrapSubviewOp>(  
          loc, resultType, inputBuf,   
          wrapSubviewOffsets,   
          wrapSubviewSizes,   
          wrapSubviewWraps);  
  
      // Now we have a single op representing the wrap-logic subview  
      // We'll replace the usage of tilingOp with this op  
      // or store it in a new data flow path  
      tilingOp.replaceAllUsesWith(wrapSubviewOp);  
    }  
  
    // Remove or erase original tilingOp  
    rewriter.eraseOp(tilingOp);  
    return success();  
  }  
  
  // Helper: builds Value range for offsets, sizes, etc.  
  SmallVector<Value, 4> getOffsetOperands(const SmallVector<int64_t, 4> &vals,  
                                          PatternRewriter &rewriter, Location loc) const {  
    SmallVector<Value, 4> results;  
    for (auto v : vals) {  
      results.push_back(  
          rewriter.create<arith::ConstantIndexOp>(loc, v));  
    }  
    return results;  
  }  
  // Similarly for getSizeOperands(), getStrideOperands() ...  
};  
} // end anonymous namespace

'''

'''
module {
  func.func @main(%A : memref<256x64xi8>,
                  %B : memref<64x128xi8>,
                  %C : memref<256x128xi8>) {
    %c0 = arith.constant 0 : index

    // ───────────────────────────────────────
    // tile-parallel  (m,k,n)   (4,2,2)
    // ───────────────────────────────────────
    scf.forall (%m, %k, %n) in (4, 2, 2) {

      // ---------------------------------
      // (1) compute the per-dimension offsets
      // ---------------------------------
      %m64 = affine.apply affine_map<(d0) -> (d0 * 64)>(%m)  // row
      %n32 = affine.apply affine_map<(d0) -> (d0 * 32)>(%n)  // col-A
      %n64 = affine.apply affine_map<(d0) -> (d0 * 64)>(%n)  // col-B/C
      %k32 = affine.apply affine_map<(d0) -> (d0 * 32)>(%k)  // depth slice

      // ---------------------------------

      // ---------------------------------
      %A_sv = memref.subview %A[%m64, %n32] [64, 32] [1, 1]
                : memref<256x64xi8> to memref<64x32xi8>
      %B_sv = memref.subview %B[%k32, %n64] [32, 64] [1, 1]
                : memref<64x128xi8> to memref<32x64xi8>
      %C_sv = memref.subview %C[%m64, %n64] [64, 64] [1, 1]
                : memref<256x128xi8> to memref<64x64xi8>

      // -----------------------------------------------------------------

      // -----------------------------------------------------------------

  
      %isFirstK = arith.cmpi eq, %k, %c0 : i1
      %A_alias = scf.if %isFirstK -> memref<64x32xi8> {
        %bcA = routing.broadcast %A_sv
                 { src_pe = 0, dst_pe = [1, 2, 3] }
                 : memref<64x32xi8>      // 
        scf.yield %bcA : memref<64x32xi8>
      } else {

        %bcA2 = routing.lookup %A_sv : memref<64x32xi8>
        scf.yield %bcA2 : memref<64x32xi8>
      }

  
      %isFirstM = arith.cmpi eq, %m, %c0 : i1
      %B_alias = scf.if %isFirstM -> memref<32x64xi8> {
        %bcB = routing.broadcast %B_sv
                 { src_pe = 0, dst_pe = [4, 5, 6, 7] }
                 : memref<32x64xi8>
        scf.yield %bcB : memref<32x64xi8>
      } else {
        %bcB2 = routing.lookup %B_sv : memref<32x64xi8>
        scf.yield %bcB2 : memref<32x64xi8>
      }

      // -----------------------------------------------------------------

      // -----------------------------------------------------------------
      linalg.matmul
        ins (%A_alias, %B_alias
             : memref<64x32xi8>, memref<32x64xi8>)
        outs(%C_sv : memref<64x64xi8>)
    }

    return
  }
}

'''

#change into

'''
module attributes {transform.with_named_sequence = @do_tiling} {  
  func.func @main(%arg0: memref<256x64xi8>, %arg1: memref<64x128xi8>, %arg2: memref<256x128xi8>) {  
    %c0 = arith.constant 0 : index  
    %c1 = arith.constant 1 : index  
    %c2 = arith.constant 2 : index  
    %c4 = arith.constant 4 : index  
      
    // 
    scf.for %arg3 = %c0 to %c4 step %c1 {  
      // compute the offset of the current row  
      %row_offset = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)  
        
      // 
      %row_broadcast = memref.subview %arg0[%row_offset, 0] [64, 64] [1, 1]  
        : memref<256x64xi8> to memref<64x64xi8, strided<[64, 1], offset: ?>>  
        
      // 
      "routing.broadcast"(%row_broadcast) {  
        dimension = "row",  
        source_tile = [0, #arg3],  
        targets = [{row = #arg3, col_range = [0, 1]}]  
      } : (memref<64x64xi8, strided<[64, 1], offset: ?>>) -> ()  
        
      // 
      scf.for %arg4 = %c0 to %c2 step %c1 {  
        //  
        %col_offset = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)  
          
        //  
        %col_broadcast = memref.subview %arg1[0, %col_offset] [64, 64] [1, 1]  
          : memref<64x128xi8> to memref<32x64xi8, strided<[128, 1], offset: ?>>  
          
        //
        "routing.broadcast"(%col_broadcast) {  
          dimension = "column",  
          source_tile = [#arg4, 0],  
          targets = [{col = #arg4, row_range = [0, 1]}]  
        } : (memref<32x64xi8, strided<[128, 1], offset: ?>>) -> ()  
          
        // 
        %out_row = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)  
        %out_col = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)  
          
        //  
        %merge = "routing.pktmerge"() {dimension = "k", size = 2} : () -> !routing.merge  
          
        //   
        scf.for %arg5 = %c0 to %c2 step %c1 {  
          //  
          %k_offset = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg5)  
            
          //   
          %subview = memref.subview %arg0[%out_row, %k_offset] [64, 32] [1, 1]  
            : memref<256x64xi8> to memref<64x32xi8, strided<[64, 1], offset: ?>>  
            attributes {broadcast_from = "row_broadcast"}  
            
          // 
          %subview_1 = memref.subview %arg1[%k_offset, %out_col] [32, 64] [1, 1]  
            : memref<64x128xi8> to memref<32x64xi8, strided<[128, 1], offset: ?>>  
            attributes {broadcast_from = "column_broadcast"}  
            
          //
          %subview_2 = memref.subview %arg2[%out_row, %out_col] [64, 64] [1, 1]  
            : memref<256x128xi8> to memref<64x64xi8, strided<[128, 1], offset: ?>>  
            
          //  
          linalg.matmul   
            ins(%subview, %subview_1 :   
              memref<64x32xi8, strided<[64, 1], offset: ?>>,  
              memref<32x64xi8, strided<[128, 1], offset: ?>>)  
            outs(%subview_2 : memref<64x64xi8, strided<[128, 1], offset: ?>>)  
            
          //  
          "routing.merge"(%merge, %subview_2) {index = #arg5} :   
            (!routing.merge, memref<64x64xi8, strided<[128, 1], offset: ?>>) -> ()  
        }  
          
        //  
        "routing.flush"(%merge) : (!routing.merge) -> ()  
      }  
    }  
    return  
  }  
    
  transform.named_sequence public @do_tiling(%arg0: !transform.any_op) {  
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op  
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [64, 64, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)  
    transform.yield   
  }  
}  
'''