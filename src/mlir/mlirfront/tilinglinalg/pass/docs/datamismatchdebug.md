<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
#Test steps
## laungch test script
- cd <repo>/src/mlir/mlirfront/tilinglinalg/pass/unitest
- source ./piplinerun.sh rebuild
## check the output
- check [aie_runtime]     shim buf@0x***** [0..15]: xx  , the xx should be consistent at least have  64 item,  each 64 should either 30 or 40 and another 64 should be 31 or 41
- the value shim received is comming from aie core tile
## if the the test failed
- analsysis to check what is the reason
- run test again and checkt output, util the the test past
## do not do
- use other script to test host binary or use it as verify test
- rebuild the aiehlc or unitest/test without use "source ./piplinerun.sh rebuild"
