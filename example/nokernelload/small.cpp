/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "xaiengine.h"

#include "xil_cache.h"
#include "xil_printf.h"

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

#ifdef __AIELINUX__
#define OUT_COL 35
#else
#define OUT_COL 2
#endif

#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x5000

__kernel__ void k(int *in, int *out) {
    for (int i = 0; i < 20; ++i)
        *(out++) = 100 + i;
}

int test_kernel(XAie_DevInst *DevInst) {
    printf("\nLoading kernel 1...\n");
    // XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
    // XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
    // XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)k);
    printf("Finished. Continuing...\n");

    printf("\nRouting...\n");
    XAie_RoutingInstance *routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, /*src=*/XAie_TileLoc(2, 0), /*dest=*/XAie_TileLoc(4, 4));
    XAie_Route(routingInstance, NULL, /*src=*/XAie_TileLoc(4, 4), /*dest=*/XAie_TileLoc(OUT_COL, 0));
    printf("Finished. Continuing...\n");

    printf("\nAllocating DDR memory...\n");
    int *in_ptr = 0, *out_ptr = 0;
    const int len = 20;
    XAie_MemInst *in = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    in_ptr = (int *)XAie_MemGetVAddr(in);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    out_ptr = (int *)XAie_MemGetVAddr(out);
    printf("Finished. Continuing...\n");

    printf("\nInitializing DDR memory...\n");
    for (int i = 0; i < len; i++) {
        in_ptr[i] = 9999;
        out_ptr[i] = 9999;
    }

    XAie_MemSyncForDev(in);
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nMoving input data to AIE...\n");
    XAie_MoveDataExternal2Aie(routingInstance, /*src=*/XAie_TileLoc(2, 0), in, len * sizeof(u32), CORE_IP_MEM,
                              /*dest=*/XAie_TileLoc(4, 4));
    printf("Finished. Continuing...\n");

    printf("\nRunning kernel 1...\n");
    // XAie_Run(routingInstance, 1);
    // // wait until core is done
    // while(XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0) != XAIE_OK) {

    // }

    // routing needs to be handled for args to work
    // kernel<4, 4>(in);

    k<4, 4>(DevInst);

    // go through adjacent kernel calls
    // add to kernel list (name, tile loc)
    // start all of them: reset, unreset, load elf, core enable
    // wait for all of them to finish

    printf("Finished. Continuing...\n");

    printf("\nMoving output data to CPU...\n");
    XAie_MoveDataAie2External(routingInstance, /*src=*/XAie_TileLoc(4, 4), CORE_OP_MEM, len * sizeof(u32), out,
                              /*dest=*/XAie_TileLoc(OUT_COL, 0));
    printf("Finished. Continuing...\n");

    printf("\nVerifying output data...\n");
    int mismatches = 0;
    for (int i = 0; i < len; i++) {
        if (100 + i != out_ptr[i]) {
            printf("Mismatch at index %d: CPU=%d, AIE=%d\n", i, 100 + i, out_ptr[i]);
            mismatches++;
        } else {
            printf("%d: CPU=%d, AIE=%d\n", i, 100 + i, out_ptr[i]);
        }
    }

    if (mismatches == 0) {
        printf("Sucess: CPU result matches AIE.\n");
        printf("\n\nDone with kernel 1!\n\n");
        return 0;
    } else {
        printf("Failure: There were %d mismatches.\n", mismatches);
        printf("\n\nDone with kernel 1!\n\n");
        return -1;
    }
}

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8

int main(int argc, char *argv[]) {
    Xil_DCacheDisable();
    Xil_ICacheDisable();

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if (RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
    if (DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
        if (RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }
    RC = XAie_PartitionInitialize(&DevInst, NULL);
    if (RC != XAIE_OK) {
        printf("Failed to initialize partition\n");
        return -1;
    }
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif

    if (test_kernel(&DevInst) == 0) {
        printf("\nKernel test passed!\n");
        return 0;
    } else {
        printf("\nKernel test failed!\n");
        return -1;
    }
}
