/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "testattr.td"
def Test_createtilearrayOp : testop<"testcreatetile"> {
  let summary = "Elementwise add with optional scale";
  let description = [{
    **MyAddOp** computes `result = (lhs + rhs) * scale`  
    - `lhs` / `rhs` are `tensor<f32>`  
    - `scale` is an `f32` attribute, default 1.0  
    - `mode` is the enum attribute `RoundingMode`
  }];

  //---------------------------------------------------------------------//
  // 4.1 interfaces, traits
  //---------------------------------------------------------------------//
  let arguments = (ins
    TensorOf<[F32]>:$lhs,
    TensorOf<[F32]>:$rhs,
    F32Attr:$scale
  );
  let results = (outs
    TensorOf<[F32]>:$result
  );
}
