<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
#Test steps
## laungch test script
- cd <repo>/src/mlir/mlirfront/tilinglinalg/pass/unitest
- source ./piplinerun.sh rebuild
## check the output
- if no aieruntime or AieRt_Debug not in the log, the test run failed , redo the "launch test"
- check the "[AieRt_Debug] tile(1,4) MEM module events " and check whether DMA_MM2S_0_FINISHED_BD is under the list
- if tile(1,4) have DMA_MM2S_0_FINISHED_BD, or MM2S FINISHED_TASK FIRED, the test passt
## if the the test failed
- analsysis to check what is the reason
- run test again and checkt output, util the the test past
## do not do
- use other script to test host binary or use it as verify test
- rebuild the aiehlc or unitest/test without use "source ./piplinerun.sh rebuild"
