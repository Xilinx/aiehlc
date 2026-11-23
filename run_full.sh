#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
BUILD_DIR=/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/build
LOG_FILE=/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/loghw

touch /scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/dataflowmap/dfscheblueprint/dfscheblueprintmanager.cpp
touch /scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp

make -j4 -C "$BUILD_DIR" > /scratch/staff/huaj/amdaiehlc/aiehlc/build_out3.txt 2>&1
echo "BUILD_EXIT:$?" >> /scratch/staff/huaj/amdaiehlc/aiehlc/build_out3.txt

if grep -q "BUILD_EXIT:0" /scratch/staff/huaj/amdaiehlc/aiehlc/build_out3.txt; then
    "$BUILD_DIR/test" dfschedule > "$LOG_FILE" 2>&1
    echo "TEST_EXIT:$?" >> "$LOG_FILE"
    grep "data_id" "$LOG_FILE" | head -20
fi
