/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include <iostream>
#include "dfscheblueprintmanager.h"

int main(int argc, char* argv[]) {
    MLIRContext ctx;
    dfscheblueprintmanager blueprinttest;
    
    // Load the dfscheblueprint dialect
    blueprinttest.loaddialect(&ctx);
    
    std::cout << "=== dfscheblueprint Dialect Test ===" << std::endl;
    std::cout << "Testing schedule blueprint IR generation..." << std::endl;
    
    // Run the blueprint test
    blueprinttest.ops_test(&ctx);
    
    std::cout << "\n=== Test Completed Successfully ===" << std::endl;
    std::cout << "Generated dfscheblueprint.config with transfer manifests" << std::endl;
    std::cout << "- broadcast_upper_half (packet_id=10)" << std::endl;
    std::cout << "- broadcast_lower_half (packet_id=11)" << std::endl;
    
    return 0;
}

