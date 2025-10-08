/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

// #include <iostream>
// #include <sstream>
#include "xil_printf.h"
#include "xaiengine.h"
//#include "xil_io.h"
#include "xil_cache.h"
#include <stdio.h>
#include <math.h>
#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
//#include "xtime_l.h"
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
//#include "xiltimer.h"
#endif
// #include "unistd.h"
#define uint_TYPE uint32_t

#if AIE_GEN <= 2

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
#else

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4

#endif

#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x6000

#define MAT_SIZE 256 // Size of each matrix (A and B)
#define N 16         // Dimension of the square matrices (16x16)

#define DISABLE_CACHE

__global__ void perf(input_window_int32 * win __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
        output_window_int32 *out __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512")))) {
#define DATA_SIZE 512 
#define MAT_SIZE 256
#define N 16          // Dimension of the square matrices
#define VECTOR_LENGTH 16
	//aie::vector<int32_t, VECTOR_LENGTH> temp_a = window_readincr_v<VECTOR_LENGTH>(win);
	//aie::store_unaligned_v<VECTOR_LENGTH>(A_mat + (w*VECTOR_LENGTH), temp_a);
	uint32_t * ptr_out = (uint32_t *)(0x70000 + 0x6000);
	uint32_t * ptr_in = (uint32_t *)(0x70000 + 0x1000);

	uint32_t * vec1 = ((uint32_t*)ptr_in), * vec2 = ((uint32_t*)ptr_in + MAT_SIZE);

	for (int i = 0; i < N; i++) {
		for (uint32_t j = 0; j < N; j++) {
			uint32_t ret = 0;
			for (int k = 0; k < N; k++) {
				ret += vec1[i * N + k] * vec2[j * N + k];
			}
			ptr_out[i * N + j] = ret;
		}
	}
}
void blockread(XAie_DevInst *DevInst, uint64_t addr)
{
#define DSIZE  512
  uint32_t odata[DSIZE];
	XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4,4),  addr,
			 (void*)odata, DSIZE * sizeof(uint32_t));

	// step 5 validate data
	printf("addr = 0x%lx\n", addr);
	for(uint32_t i = 0U; i < DSIZE; i++) {
		printf("odata[%d] = %x\n", i, odata[i]);
	}

}
void breakprint(char *info) {
	return;
	char input;
	printf("input key to run %s\n", info);
	scanf("%c", input);
	printf("log--- %s done\n", info);
}
int routing_run_kernel(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	XAie_RoutingInstance* routingInstance;
  	//XTime tStart, tEnd;
	breakprint("core reset");
	int shimcol = 2;
	XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
	XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
	XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)perf);

	breakprint("  XAie_InitRoutingHandler---\n");
	routingInstance = XAie_InitRoutingHandler(DevInst);
	breakprint("  XAie_Route-4--\n");
	XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol,0) /* Source*/,
				 XAie_TileLoc(4,4) /* destination*/	);
	XAie_Route(routingInstance, NULL, XAie_TileLoc(4,4) /* Source*/,
				 XAie_TileLoc(shimcol,0) /* destination*/	);

	breakprint("  XAie_MemAllocate---\n");

	//printf("Routing successful\n");
	u64 phy = 0, phy_out = 0;
	u32 mlen = MAT_SIZE*2;
	const u32 recv_len = MAT_SIZE;
	
	//Prepare DDR data
	XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
	phy = (u32)XAie_MemGetDevAddr(in);
	XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
	phy_out = (u32)XAie_MemGetDevAddr(out);

	breakprint(" XAie_MemSyncForDev---\n");

	u64 vmem =    phy;
	u64 vmem_out = phy_out;
	for(int i = 0; i < mlen; i++) {
		((int32_t*)vmem)[i] = i + 1;
		((int32_t*)vmem_out)[i] = 0;
	}

	//((u32*)vmem)[0] = 1024*1024;

	XAie_MemSyncForDev(in);
	XAie_MemSyncForCPU(out);

	breakprint("Starting to Move data\n");
	// step 3: move data to destination tile
	//XTime_GetTime(&tStart);
	printf("vmem = 0x%p\n",vmem);

	XAie_MoveDataExternal2Aie(routingInstance, /*src=*/ XAie_TileLoc(shimcol,0),  in, mlen*sizeof(u32), CORE_IP_MEM, /*dest=*/ XAie_TileLoc(4,4));
	
	breakprint("XAie_RouteDmaWait\n");
		
#if AIE_GEN == XAIE_DEV_GEN_AIE2PS
	//Wait until the data transfer completes.
	breakprint("XAie_RouteDmaWait--bbb--\n");
	XAie_RouteDmaWait(routingInstance, XAie_TileLoc(shimcol,0), XAie_TileLoc(4,4), true);
#endif
	breakprint("blockread\n");
  	//blockread(DevInst, CORE_IP_MEM);
	XAie_Run(routingInstance, 1);

	breakprint("XAie_CoreWaitForDone\n");
	//wait until core done
	u8 allDone  = 0;
	uint32_t CoreStatus = 0;
	//usleep(1000*1000);
	do {
		allDone = 1; // Assume all cores are done initially
		uint32_t coreStatCharWritten = 0;
		for (int i = 0; i < 1; i++) { // Iterate over the specified tiles
			RC = XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0);

			if (RC != XAIE_OK) {
				allDone = 0;
			}
		}
	} while (!allDone);
	breakprint("fflush\n");
	fflush(stdout);

	XAie_MoveDataAie2External(routingInstance,  XAie_TileLoc(4,4),CORE_OP_MEM, mlen*sizeof(u32),out, XAie_TileLoc(shimcol,0));					

	//XTime_GetTime(&tEnd);
	//printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));

	//printf("\nFinished moving data back to DDR\n");
	// step 5 validate data
  int32_t vmem_out_cpu[recv_len];

	//vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms respectively) and vmem_out contains the output samples (64 of result) 
	//compute CPU Result for softmax
    int32_t A_mat[N][N];  // Matrix A
    int32_t B_mat[N][N];  // Matrix B
    int32_t result[N][N] = {0};  // Result matrix

    // Extract matrix A (row major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_mat[i][j] = ((int32_t*)vmem)[i * N + j];
        }
    }

    // Extract matrix B (column major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_mat[i][j] = ((int32_t*)vmem)[MAT_SIZE + i * N + j];  // Adjust index for column major
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                result[i][j] += A_mat[i][k] * B_mat[j][k];
            }
        }
    }

    // Store the result in vmem_out_cpu
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            vmem_out_cpu[i * N + j] = result[i][j];
        }
    }

 	int mismatches = 0;
    for (int i = 0; i < MAT_SIZE; i++) {
        if (vmem_out_cpu[i] != ((int32_t*)vmem_out)[i]) {

            printf("Mismatch at index %d: CPU=%d, vmem_out=%d\n", i, vmem_out_cpu[i], ((int32_t*)vmem_out)[i]);
            mismatches++;
        }
    }

	for (int i = 0; i < ((16 > MAT_SIZE) ? MAT_SIZE : 16); i++) {
            printf("match example at index %d: CPU=%d, vmem_out=%d\n", i, vmem_out_cpu[i], ((int32_t*)vmem_out)[i]);
    }

    if (mismatches == 0) {
        printf("CPU result matches vmem_out.\n");
    } else {
        printf("There were %d mismatches.\n", mismatches);
    }

	printf("\nDone\n");
	return 0;
}

int main(int argc, char* argv[]) {
#ifdef DISABLE_CACHE
	Xil_DCacheDisable();
	Xil_ICacheDisable();
	printf("1Cache Disabled performance will have big drop (this test should >350us(8*8 and 16*16)\n ");
#else
	printf("1cache enabled, this test should be 15us(8*8) 215 us(16*16)");
#endif

	for (int i = 0; i < 100; i++) {
		printf("round %d\n", i);

		XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
				XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
				XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
				XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
				XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

		///*

		XAie_InstDeclare(DevInst, &ConfigPtr);

		int partitonnum = 34;
		int startcol = 2;
		int colnum = (startcol + partitonnum <= XAIE_NUM_COLS) ? partitonnum : (XAIE_NUM_COLS- startcol);

		AieRC RC;
		///*
		RC = XAie_SetupPartitionConfig(&DevInst, XAIE_BASE_ADDR + (startcol<<XAIE_COL_SHIFT), 
										startcol, colnum);
		if(RC != XAIE_OK) {
			printf("Driver XAie_SetupPartitionConfig failed.\n");
			return -1;
		}
			//*/

		RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
		if(RC != XAIE_OK) {
			printf("Driver initialization failed.\n");
			return -1;
		}

		XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

	#if AIE_GEN >= 2
		if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
			printf("XAie_UpdateNpiAddr()\n");
			#if AIE_GEN == 5 //aie2ps
				printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
				RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
			#else
				RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
			#endif
			if(RC != XAIE_OK) {
				printf("Failed to update NPI address\n");
				return -1;
			}
		}
		printf("before XAie_PartitionInitialize-2--\n");
		//fix in aie2 the shim dma not work issue
		RC = XAie_PartitionInitialize(&DevInst, NULL);
	#else
		//fix in aie1 shimd dma not work issue
		XAie_PmRequestTiles(&DevInst, NULL, 0); 
	#endif

		routing_run_kernel(&DevInst);

		RC = XAie_PartitionTeardown(&DevInst);
		if(RC != XAIE_OK) {
			printf("Failed to Teardown partition\n");
			return -1;
		}
	}

	return 1;
}
