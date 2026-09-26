/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*****************************************************************************/
/**
 *
 * @file xpseudo_asm_armclang.h
 *
 * BSP compatibility shim for the aiehlc Clang front-end.
 *
 * The standalone BSP's xpseudo_asm.h dispatches on __clang__:
 *
 *     #ifdef __clang__
 *     #include "xpseudo_asm_armclang.h"
 *     #else
 *     #include "xpseudo_asm_gcc.h"
 *     #endif
 *
 * That test really means "armclang toolchain", but aiehlc parses the user
 * source with Clang (a parser only - the host object is produced later by
 * aarch64-none-elf-g++, which takes the _gcc.h branch). So any source that
 * reaches xil_io.h / sleep.h drags in xpseudo_asm_armclang.h during the
 * front-end stage.
 *
 * The VEK385 / cortexa78_0 (Gen5) BSP ships that header; the VEK280 /
 * psv_cortexa72_0 (Gen2) BSP does not, so --aie-version 2 failed with
 * "'xpseudo_asm_armclang.h' file not found".
 *
 * This directory is appended LAST on the aiehlc include path
 * (script/aiehlc.sh), so where the BSP provides the real header it still wins;
 * only when it is absent do we land here and fall back to the GCC inline-asm
 * definitions - which is exactly what the real host compile uses anyway.
 *
 * Do not add this directory to the host (g++) or kernel (xchesscc) include
 * paths: neither defines __clang__, so neither needs the shim.
 *
 ******************************************************************************/

#ifndef XPSEUDO_ASM_ARMCLANG_H /* prevent circular inclusions */
#define XPSEUDO_ASM_ARMCLANG_H /* by using protection macros */

/* Resolved against the BSP include dir already on the search path. */
#include "xpseudo_asm_gcc.h"

#endif /* XPSEUDO_ASM_ARMCLANG_H */
