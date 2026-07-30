/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef AIE_DEVICE_MAP_H
#define AIE_DEVICE_MAP_H

// AIE tile memory map fallbacks. The correct values are injected by aiehlc (dm_offsets.h).
#ifndef CORE_IP_MEM
#define CORE_IP_MEM 0x1000
#endif
#ifndef CORE_OP_MEM
#define CORE_OP_MEM 0x6000
#endif
#define DEBUG_LOG_BASE 0x73000
#define SYNC_BUFFER_BASE 0x74000

// DMA buffer sizes
#define DEFAULT_BUFFER_SIZE 256
#define ALLOC_LIST_MAX_SIZE 256

// Device layout (reference: aieml_perf.cc lines 26-50)
#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2

#ifndef AIE_GEN
#  if defined(__AIE_ARCH__) && __AIE_ARCH__ >= 22
#    define AIE_GEN 5
#  elif defined(__AIE_ARCH__) && __AIE_ARCH__ >= 20
#    define AIE_GEN 2
#  else
#    define AIE_GEN 1
#  endif
#endif

#if AIE_GEN == 1
#define HW_GEN XAIE_DEV_GEN_AIE
#  define XAIE_NUM_ROWS            9
#  define XAIE_NUM_COLS           50
#  define XAIE_AIE_TILE_ROW_START  1
#  define XAIE_AIE_TILE_NUM_ROWS   8
#  undef  XAIE_RES_TILE_ROW_START
#  undef  XAIE_RES_TILE_NUM_ROWS
#  define XAIE_RES_TILE_ROW_START  0
#  define XAIE_RES_TILE_NUM_ROWS   0
#elif AIE_GEN == 5
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#  define XAIE_AIE_TILE_ROW_START  3
#define XAIE_AIE_TILE_NUM_ROWS 4
#else
#define HW_GEN XAIE_DEV_GEN_AIEML
#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8
#endif

#endif // AIE_DEVICE_MAP_H
