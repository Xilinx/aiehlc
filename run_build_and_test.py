#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import subprocess, os

build_dir = '/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/build'
out_file = '/scratch/staff/huaj/amdaiehlc/aiehlc/build_out3.txt'
log_file = '/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/loghw'

# Build
r = subprocess.run(['make', '-j4', '-C', build_dir], capture_output=True, text=True)
build_out = r.stdout + r.stderr + f'\nEXIT:{r.returncode}\n'
open(out_file, 'w').write(build_out)

if r.returncode != 0:
    print("BUILD FAILED")
    print(build_out[-2000:])
else:
    print("BUILD OK")
    # Run test
    r2 = subprocess.run(['./test', 'dfschedule'], capture_output=True, text=True, cwd=build_dir)
    test_out = r2.stdout + r2.stderr
    open(log_file, 'w').write(test_out)
    print(f"TEST EXIT:{r2.returncode}")
    # Show config.dma_bd sections
    lines = test_out.splitlines()
    for i, line in enumerate(lines):
        if 'config.dma_bd' in line and 'data_id' not in line:
            chunk = lines[max(0,i):min(len(lines),i+15)]
            print('\n'.join(chunk))
            print('---')
            break
