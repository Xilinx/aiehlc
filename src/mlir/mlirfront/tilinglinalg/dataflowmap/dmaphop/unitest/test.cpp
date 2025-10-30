/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include <iostream>
#include "dmaphopmanager.h"

int main(int argc, char* argv[]) {
    MLIRContext ctx;
    dmaphopmanager hoptest;
    
    // Load the dmaphop dialect
    hoptest.loaddialect(&ctx);
    
    // Run the cache-and-forward broadcast test
    hoptest.ops_test(&ctx);
    
    std::cout << "Completed dmaphop dialect test" << std::endl;
    return 0;
}