#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import subprocess

build_dir = '/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/build'
log_file = '/scratch/staff/huaj/amdaiehlc/aiehlc/src/mlir/mlirfront/tilinglinalg/pass/unitest/loghw'

r = subprocess.run(['./test', 'dfschedule'], capture_output=True, text=True, cwd=build_dir)
test_out = r.stdout + r.stderr
open(log_file, 'w').write(test_out)
print(f"TEST EXIT:{r.returncode}")

# Show first config.dma_bd with data_id
lines = test_out.splitlines()
found = False
for i, line in enumerate(lines):
    if 'config.dma_bd' in line:
        chunk = lines[max(0,i):min(len(lines),i+16)]
        print('\n'.join(chunk))
        print('---')
        found = True
        break
if not found:
    print("No config.dma_bd found in output!")
    print(test_out[-1000:])
