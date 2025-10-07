/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/* 0.10v Test DBIST TDF Chain - Pattern 3 */

#include "xaiengine.h"
#include "sleep.h"
#include "xil_printf.h"
#include "xil_cache.h"
#define PARTITION_APIS

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#include "xtime_l.h"
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#include "xiltimer.h"
#endif

#include <stdio.h>
#define PATTERN_LENGTH 1862
#define GOLDEN_MISR 0x54AB14E2F2BD
#define PATTERN_TYPE 3

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_ROW_END 8
#define XAIE_AIE_TILE_NUM_ROWS 4

#define N 3
#define MATRIX_SIZE 3*3
// default memory addresses for input and output
#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x5000

#define DISABLE_CACHE

/* Select the columns / rows to test; valid columns 0 to 35; valid rows 3 to 6 */
//int cols_to_test[] = {2,3};
//int cols_to_test[] = {13, 14, 15};
int cols_to_test[] = {0,1,2};
//int cols_to_test[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35};
//int rows_to_test[] = {3,4,5,6};
int rows_to_test[] = {3};
//int dbist_columns[] = {12,13,14,15};
int dbist_columns[] = {0,1,2};
//int rows_to_test[] = {3,6};


// multiply "in" matrix by itself
__global__ void mm(int *in, int *out) {
    #define N 3
    uint64_t count = 0;
    //while(true) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += in[i * N + k] * in[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
    out[N * N] = ++count;
    //}
}

void check_col_status(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    //printf("\nChecking Core Status...\n");
    for (int row = 0; row < no_of_rows; ++row) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, rows_to_test[row]);
        u8 enabled = 0;
        XAie_CoreReadDoneBit(DevInst, tileLoc, &enabled);
        printf("Tile (%d, %d): %s\n", test_col, rows_to_test[row], enabled ? "DONE" : "EXECUTING");
    }
    //printf("Finished. Continuing...\n");
}

bool check_tile_result(XAie_DevInst *DevInst, XAie_LocType Loc, u32 Value) {
    u32 output[N*N+1];
    u32 output_value = 0;
    XAie_DataMemBlockRead(DevInst, Loc, CORE_OP_MEM, (void*)&output, sizeof(output));
    bool matches = true;
    for(int i = 0; i < N * N; i++) {
        if(output[i] != Value) {
            matches = false;  
            output_value = output[i];
            printf("Error: Tile (%d, %d) output mismatch: expected %d, got %d in element %d\n", Loc.Col, Loc.Row, Value, output[i],i);
        }
    }
    if(matches) {
        printf("Tile (%d, %d) output matches expected value: %d\n", Loc.Col, Loc.Row, Value);
        printf("Iterations: %llu\n", output[N*N]);
        return true;
    } else {
        //printf("Error: Tile (%d, %d) output mismatch: expected %d, got %d\n", Loc.Col, Loc.Row, Value, output_value);
        printf("Iterations: %llu\n", output[N*N]);
        return false;
    }

}

bool check_col_output(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nChecking tile results...\n");
    int expected_results[XAIE_AIE_TILE_NUM_ROWS] = {27, 48, 75, 108};
    int mismatches = 0;
    for(int i = 0; i < no_of_rows; i++) {
        //printf("\n\nReading Tile (%d, %d) output...\n", test_col, i);
        if(!check_tile_result(DevInst, XAie_TileLoc(test_col, rows_to_test[i]), expected_results[rows_to_test[i] - XAIE_AIE_TILE_ROW_START])) {
            mismatches++;
        }
    }
    //printf("Finished. Continuing...\n");

    if (mismatches == 0) {
        //printf("\nSucess: CPU result matches AIE.\n");
        return true;
    } else {
        printf("\nFailure: There were %d mismatches.\n", mismatches);
        return false;
    }
}

void read_ip_op_mem(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nReading IP OP Mem of %d no of rows %d ...\n", test_col, no_of_rows );


    //printf("\nSetting up input data...\n");
    for(int i = 0; i < no_of_rows; i++) {
        int input[MATRIX_SIZE];
        int output[10];
        int routput[10];
        int rinput[10];
        XAie_LocType Loc = XAie_TileLoc(test_col, rows_to_test[i]);
        printf("\nReading data for tile (%d, %d)\n", test_col, rows_to_test[i]);

        for(int j=0; j < MATRIX_SIZE; j++) {
            rinput[j] = 0; //rows_to_test[i];
        }

        AieRC RC = XAie_DataMemBlockRead(DevInst, Loc, CORE_IP_MEM, (void*)&rinput, sizeof(rinput));
        if(RC != XAIE_OK) {
            printf("XAie_DataMemBlockRead IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }
        printf("IP Memory : [");

        for(int i = 0; i < N * N; i++) {   
            printf("%d",rinput[i]);
            if(i < 8){ 
                        printf(", ");
            }
        }  
        printf("]\n");

        for(int j=0; j < 10; j++) {
            routput[j] = 0;
        }

        RC = XAie_DataMemBlockRead(DevInst, Loc, CORE_OP_MEM, (void*)&routput, sizeof(routput));
        if(RC != XAIE_OK) {
            printf("XAie_DataMemBlockRead OP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }

        printf("OP Memory : [");

        for(int i = 0; i < N * N; i++) {   
            printf("%d",routput[i]);
            if(i < 8){ 
                        printf(", ");
            }
        }  
        printf("]\n");
    }

 }

void start_col(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nLoading column of kernels %d no of rows %d ...\n", test_col, no_of_rows );
    for(int i = 0; i < no_of_rows; i++) {
        int test_row = rows_to_test[i];
        printf("\nLoading column %d row %d ...\n", test_col, test_row );
        XAie_LocType tileLoc = XAie_TileLoc(test_col, rows_to_test[i]);
        //printf("\n Core Reset started...\n");
        AieRC RC = XAie_CoreReset(DevInst, tileLoc);
        if(RC != XAIE_OK) {
            printf("XAie_CoreReset IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }
        //printf("\n Core Reset Done...\n");
        //printf("\n Core Un-Reset started...\n");
        RC = XAie_CoreUnreset(DevInst, tileLoc);
        if(RC != XAIE_OK) {
            printf("XAie_CoreUnreset IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }

        //printf("\n Core Un-Reset Done...\n");
        printf("Tile (%d, %d) reset and unreset.\n", test_col, rows_to_test[i]);
      
        RC = XAie_LoadElfMem(DevInst, XAie_TileLoc(test_col, test_row), (unsigned char *)mm);
        if(RC != XAIE_OK) {
            printf("XAie_LoadElfMem IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }

        printf("Tile (%d, %d) loaded elf\n", test_col, rows_to_test[i]);
    }
    //printf("Finished. Continuing...\n");

    //printf("\nSetting up input data...\n");
    for(int i = 0; i < no_of_rows; i++) {
        int input[MATRIX_SIZE];
        int output[10];
        int routput[10];
        int rinput[10];
        XAie_LocType Loc = XAie_TileLoc(test_col, rows_to_test[i]);
        printf("Setting up input data for tile (%d, %d)\n", test_col, rows_to_test[i]);

        for(int j=0; j < MATRIX_SIZE; j++) {
            input[j] = rows_to_test[i];
        }

        AieRC RC = XAie_DataMemBlockWrite(DevInst, Loc, CORE_IP_MEM, (void*)&input, sizeof(input));
        if(RC != XAIE_OK) {
            printf("XAie_DataMemBlockWrite IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }
    }
    
    printf("After XAie_DataMemBlockWrite\n");
    read_ip_op_mem(DevInst, test_col, no_of_rows);

    //printf("\nStarting kernels...\n");
    for(int i = 0; i < no_of_rows; i++) {
        printf("\nStarting kernel (%d, %d)...\t", test_col, rows_to_test[i]);
        XAie_CoreEnable(DevInst, XAie_TileLoc(test_col, rows_to_test[i]));
        XAie_CoreWaitForDone(DevInst, XAie_TileLoc(test_col, rows_to_test[i]), 0);
       // printf("Finished. Continuing...\n");
    }

    //printf("\n\nStarted all kernels in col %d!\n\n", test_col);
}

void stop_col(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nStopping column of kernels %d...\n", test_col);
    for(int i = 0; i < no_of_rows; i++) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, rows_to_test[i]);
        XAie_CoreDisable(DevInst, tileLoc);
        printf("Tile (%d, %d) disabled.\n", test_col, rows_to_test[i]);
    }
    //printf("Finished. Continuing...\n");
}

void restart_col(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nRestart column of kernels %d...\n", test_col);
    for(int i = 0; i < no_of_rows; i++) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, rows_to_test[i]);
        XAie_CoreEnable(DevInst, tileLoc);
        printf("Tile (%d, %d) enabled.\n", test_col, rows_to_test[i]);
    }
    //printf("Finished. Continuing...\n");
    //printf("\n");
}





 void start_col_no_elf(XAie_DevInst *DevInst, int test_col, int no_of_rows) {
    printf("\nLoading column of kernels %d no of rows %d ...\n", test_col, no_of_rows );
    for(int i = 0; i < no_of_rows; i++) {
        int test_row = rows_to_test[i];
        printf("\nLoading column %d row %d ...\n", test_col, test_row );

        XAie_LocType tileLoc = XAie_TileLoc(test_col, rows_to_test[i]);
        //printf("\n Core Reset started...\n");

#if 0
        AieRC RC = XAie_CoreReset(DevInst, tileLoc);
        if(RC != XAIE_OK) {
            printf("XAie_CoreReset IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }
        //printf("\n Core Reset Done...\n");
#endif
        //printf("\n Core Un-Reset started...\n");
        AieRC RC = XAie_CoreUnreset(DevInst, tileLoc);
        if(RC != XAIE_OK) {
            printf("XAie_CoreUnreset IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }

        //printf("\n Core Un-Reset Done...\n");
        printf("Tile (%d, %d) reset and unreset.\n", test_col, rows_to_test[i]);
      
       /* RC = XAie_LoadElfMem(DevInst, XAie_TileLoc(test_col, test_row), (unsigned char *)mm);
        if(RC != XAIE_OK) {
            printf("XAie_LoadElfMem IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }

        printf("Tile (%d, %d) loaded elf\n", test_col, rows_to_test[i]); */
    }
    //printf("Finished. Continuing...\n");

    //printf("\nSetting up input data...\n");
    for(int i = 0; i < no_of_rows; i++) {
        int input[MATRIX_SIZE];
        int output[10];
        int routput[10];
        int rinput[10];
        XAie_LocType Loc = XAie_TileLoc(test_col, rows_to_test[i]);
        printf("Setting up input data for tile (%d, %d)\n", test_col, rows_to_test[i]);

        for(int j=0; j < MATRIX_SIZE; j++) {
            input[j] = rows_to_test[i];
        }

        AieRC RC = XAie_DataMemBlockWrite(DevInst, Loc, CORE_IP_MEM, (void*)&input, sizeof(input));
        if(RC != XAIE_OK) {
            printf("XAie_DataMemBlockWrite IP MEM failed for Tile [%d : %d]\n",test_col,rows_to_test[i]);
            break;
        }
    }

    printf("Before Starting kernels\n");
    read_ip_op_mem(DevInst, 0, no_of_rows);
    
    //printf("\nStarting kernels...\n");
    for(int i = 0; i < no_of_rows; i++) {
        printf("\nStarting kernel (%d, %d)...\t", test_col, rows_to_test[i]);
        XAie_CoreEnable(DevInst, XAie_TileLoc(test_col, rows_to_test[i]));
       // printf("Finished. Continuing...\n");
    }

    //printf("\n\nStarted all kernels in col %d!\n\n", test_col);
}



int main(int argc, char* argv[]) {
#ifdef DISABLE_CACHE
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    printf("Cache disabled\n");
#else
    printf("Cache enabled\n");
#endif

    int dbist_num_cols = sizeof(dbist_columns) / sizeof(dbist_columns[0]);
    int num_cols = sizeof(cols_to_test) / sizeof(cols_to_test[0]);
    int num_rows = sizeof(rows_to_test) / sizeof(rows_to_test[0]);
    int dbist_flag = 0;
    int loop_counter = 0;
    int dbist_ret_val = 0;
    int error = 0;

    for(int i=0; i <num_cols ; i++) {
        if(cols_to_test[i] < 0 || cols_to_test[i] >= XAIE_NUM_COLS) {
            printf("Error: Incorrect AIE Column %d, Valid Columns 0 to 35\n", cols_to_test[i]);
            exit(0);
        }
    }

    for(int i=0; i <num_rows ; i++) {
        if(rows_to_test[i] < XAIE_AIE_TILE_ROW_START || rows_to_test[i] >= XAIE_AIE_TILE_ROW_END) {
            printf("Error: Incorrect AIE Row %d, Valid Rows 3 to 6\n", rows_to_test[i]);
            exit(0);
        }
    }

    printf("\nVer Partition APIs enabled, No of columns to test %d; No of Rows to test %d\n", num_cols, num_rows);

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
                     XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
                     XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
                     XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
                     XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_ROW_END);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    XAie_SetupPartitionConfig(&DevInst, XAIE_BASE_ADDR + (dbist_columns[0]<<XAIE_COL_SHIFT), 
                                        dbist_columns[0], dbist_num_cols);


    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if(RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }  
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
   /* DBIST partition Configuration end */

#if AIE_GEN >= 2
    if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
      #if AIE_GEN == 5 //aie2ps
			printf("XAie_UpdateNpiAddr(0xf6d50000) - 3\n");
			RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
		#else
			RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
		#endif
        if(RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }

    RC = XAie_PartitionInitialize(&DevInst, NULL);
    if(RC != XAIE_OK) {
        printf("Failed to initialize partition\n");
        return -1;
    }

#else

    XAie_PmRequestTiles(&DevInst, NULL, 0); 
#endif
 

    for(int i = 0; i < num_cols; i++) {
        //start_col(&DevInst, cols_to_test[i], num_rows);
        start_col(&DevInst, i, num_rows);
    }
    printf("a-after start and before sleep\n");
    sleep(1);
    printf("after sleep\n");
    for(int i = 0; i < num_cols; i++) {
        //if(check_col_output(&DevInst, cols_to_test[i], num_rows)) {
        if(check_col_output(&DevInst, i, num_rows)) {
            printf("Col %d test passed!\n\n", cols_to_test[i]);
        } else {
            printf("\nCol %d test failed!\n", cols_to_test[i]);
            error = 1;
        }
    }

    if(error) 
    {
        printf("Core Module Status R3 Value : 0x%x\n", Xil_In32(0x2001C338004));
        printf("Core Module Status R4 Value : 0x%x\n", Xil_In32(0x2001C438004));
        printf("Core Module Status R5 Value : 0x%x\n", Xil_In32(0x2001C538004));
        printf("Core Module Status R6 Value : 0x%x\n", Xil_In32(0x2001C638004));
        printf("C14 R3 Tile Event Status\n");
        printf("Core Module Event Status 1 0x2001C334200 : 0x%x\n", Xil_In32(0x2001C334200));  
        printf("Core Module Event Status 2 0x2001C334204 : 0x%x\n", Xil_In32(0x2001C334204));      
        printf("Core Module Event Status 3 0x2001C334208 : 0x%x\n", Xil_In32(0x2001C334208));      
        printf("Core Module Event Status 4 0x2001C33420C : 0x%x\n\n", Xil_In32(0x2001C33420C));  
        printf("Mem Module Event Status 1 0x2001C314200 : 0x%x\n", Xil_In32(0x2001C314200));  
        printf("Mem Module Event Status 2 0x2001C314204 : 0x%x\n", Xil_In32(0x2001C314204));      
        printf("Mem Module Event Status 3 0x2001C314208 : 0x%x\n", Xil_In32(0x2001C314208));      
        printf("Mem Module Event Status 4 0x2001C31420C : 0x%x\n\n", Xil_In32(0x2001C31420C));      
        //break;
    }
    sleep(2);

    for(int i = 0; i < num_cols; i++) {
        //stop_col(&DevInst, cols_to_test[i], num_rows);
        stop_col(&DevInst, i, num_rows);
    }

#ifdef PARTITION_APIS
    RC = XAie_PartitionTeardown(&DevInst);
    if(RC != XAIE_OK) {
        printf("Failed to Teardown partition\n");
        return -1;
    }
#endif
    printf("After Teardown partition 2 \n");
    //read_ip_op_mem(&DbistDevInst, 0, num_rows);
    XAie_Finish(&DevInst);
    sleep(1);
#ifdef PARTITION_APIS

    while(1) {
        printf("DBIST loop counter = %d\n", ++loop_counter);
        /*  DBIST partition Configuration */
        XAie_SetupConfig(DbistConfigPtr, HW_GEN, XAIE_BASE_ADDR,
                        XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
                        XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
                        XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
                        XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_ROW_END);

        XAie_InstDeclare(DbistDevInst, &DbistConfigPtr);
        RC = XAie_SetupPartitionConfig(&DbistDevInst, XAIE_BASE_ADDR + (dbist_columns[0]<<XAIE_COL_SHIFT), 
                                        dbist_columns[0], dbist_num_cols);

        printf("DbistDevInst.BaseAddr=0x %p, DbistDevInst.StartCol = %d ,DbistDevInst.NumCols = %d\n", DbistDevInst.BaseAddr , DbistDevInst.StartCol,DbistDevInst.NumCols);

        if(RC != XAIE_OK) {
            printf("DBIST Partition initialization failed.\n");
            return -1;
        }

        RC = XAie_CfgInitialize(&DbistDevInst, &DbistConfigPtr);
        if(RC != XAIE_OK) {
            printf("DBIST Partition initialization failed.\n");
            return -1;
        }

        XAie_SetIOBackend(&DbistDevInst, XAIE_IO_BACKEND_BAREMETAL);

    #if AIE_GEN >= 2
        if(DbistDevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        #if AIE_GEN == 5 //aie2ps
                printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
                RC = XAie_UpdateNpiAddr(&DbistDevInst, 0xf6d50000);
            #else
                RC = XAie_UpdateNpiAddr(&DbistDevInst, 0xF6D10000);
            #endif
            if(RC != XAIE_OK) {
                printf("Failed to update NPI address\n");
                return -1;
            }
        }
        
        RC = XAie_PartitionInitialize(&DbistDevInst, NULL);
        if(RC != XAIE_OK) {
            printf("Failed to initialize partition\n");
            return -1;
        }
    #else
        XAie_SetIOBackend(&DbistDevInst, XAIE_IO_BACKEND_BAREMETAL);
        XAie_PmRequestTiles(&DbistDevInst, NULL, 0);
 
    #endif


        printf("After partition init\n");
        if(RC != XAIE_OK) {
        printf("Failed to initialize partition\n");
        return -1;
        }

        printf("After XAie_PartitionInitialize\n");
    #endif
        sleep(5);

        for(int j = 0; j < dbist_num_cols; j++) {
            //start_col(&DevInst, dbist_columns[j], num_rows);
            printf("before start_col 2\n");
            start_col(&DbistDevInst, j, num_rows);
            //start_col_no_elf(&DbistDevInst, j, num_rows);
        }

        printf("a-2 after start and before sleep\n");
        sleep(1);
        printf("after 2 sleep\n");

        for(int i = 0; i < dbist_num_cols; i++) {
            if(check_col_output(&DbistDevInst, i, num_rows)) {
                printf("Col %d test passed!\n\n", i);
            } else {
                printf("\nCol %d test failed!\n", i);
                error = 1;
            }
        }

        for(int i = 0; i < dbist_num_cols; i++) {
            stop_col(&DbistDevInst, i, num_rows);
        }

    #ifdef PARTITION_APIS
        RC = XAie_PartitionTeardown(&DbistDevInst);
        if(RC != XAIE_OK) {
            printf("Failed to Teardown partition\n");
            return -1;
        }
    #endif
        XAie_Finish(&DbistDevInst);
        printf("After Teardown partition 2 \n");
        //read_ip_op_mem(&DbistDevInst, 0, num_rows);
        sleep(1);

        printf("After loop\n");
        printf("\n");

    }    
}
