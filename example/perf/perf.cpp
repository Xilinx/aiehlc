#include "xaiengine.h"

#include "xil_cache.h"
#include "xil_printf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __HOST_CODE_ONLY__
#ifdef __chess__
#include <aie_api/aie.hpp>
#else
namespace aie {
template <int Size, typename T> vector<T, Size> load_v(T *ptr) { return {}; }

template <typename T, typename Ty> void store_v(T ptr, Ty vec) {}

} // namespace aie
#endif
#endif

// #include "sleep.h"

// #define PROFILE 0
#define PROFILE 1

// Only enable one of these
// #define PROFILE_FULL
// #define PROFILE_MEM_ONLY
#define PROFILE_DATA_MOVEMENT

// enable to use passthrough kernel instead of matrix multiplication
#define PASSTHROUGH

#define debug_printf(...)                                                                                              \
    do {                                                                                                               \
        if (PROFILE == 0) {                                                                                            \
            printf(__VA_ARGS__);                                                                                       \
            fflush(stdout);                                                                                            \
        }                                                                                                              \
    } while (0);

// #define ITERATIONS 6000
int ITERATIONS = 1000;

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

#ifdef __AIELINUX__
#ifdef PROFILE
#include <time.h>
#endif
#define OUT_COL 35
#else
#ifdef PROFILE
#include "xtime_l.h"
#endif
#define OUT_COL 2
#endif

#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x5000

#ifdef PROFILE
#ifdef __AIELINUX__
struct timespec linux_start, linux_end;
#else
XTime tStart, tEnd;
#endif
#endif

#ifdef PROFILE_DATA_MOVEMENT
#ifdef __AIELINUX__
struct timespec data_move_start, data_move_end;
double total_input_time = 0.0;
double total_output_time = 0.0;
double total_compute_time = 0.0;
#else
XTime data_move_start, data_move_end;
double total_input_time = 0.0;
double total_output_time = 0.0;
double total_compute_time = 0.0;
#endif
#endif

void profile_start() {
#ifdef PROFILE
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &linux_start);
#else
    XTime_GetTime(&tStart);
#endif
#endif
}

void profile_end() {
#ifdef PROFILE
// sleep(1);
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &linux_end);
#else
    XTime_GetTime(&tEnd);
#endif
#endif
}

void data_movement_profile_start() {
#ifdef PROFILE_DATA_MOVEMENT
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &data_move_start);
#else
    XTime_GetTime(&data_move_start);
#endif
#endif
}

void data_movement_profile_end_input() {
#ifdef PROFILE_DATA_MOVEMENT
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &data_move_end);
    double elapsed =
        (data_move_end.tv_sec - data_move_start.tv_sec) + (data_move_end.tv_nsec - data_move_start.tv_nsec) / 1e9;
    total_input_time += elapsed;
#else
    XTime_GetTime(&data_move_end);
    double elapsed = 1.0 * (data_move_end - data_move_start) / (COUNTS_PER_SECOND);
    total_input_time += elapsed;
#endif
#endif
}

void data_movement_profile_end_output() {
#ifdef PROFILE_DATA_MOVEMENT
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &data_move_end);
    double elapsed =
        (data_move_end.tv_sec - data_move_start.tv_sec) + (data_move_end.tv_nsec - data_move_start.tv_nsec) / 1e9;
    total_output_time += elapsed;
#else
    XTime_GetTime(&data_move_end);
    double elapsed = 1.0 * (data_move_end - data_move_start) / (COUNTS_PER_SECOND);
    total_output_time += elapsed;
#endif
#endif
}

void data_movement_profile_end_compute() {
#ifdef PROFILE_DATA_MOVEMENT
#ifdef __AIELINUX__
    clock_gettime(CLOCK_MONOTONIC, &data_move_end);
    double elapsed =
        (data_move_end.tv_sec - data_move_start.tv_sec) + (data_move_end.tv_nsec - data_move_start.tv_nsec) / 1e9;
    total_compute_time += elapsed;
#else
    XTime_GetTime(&data_move_end);
    double elapsed = 1.0 * (data_move_end - data_move_start) / (COUNTS_PER_SECOND);
    total_compute_time += elapsed;
#endif
#endif
}

void print_profile_result() {
#ifndef PROFILE_DATA_MOVEMENT
#ifdef PROFILE
#ifdef __AIELINUX__
    double elapsed_time_sec = (linux_end.tv_sec - linux_start.tv_sec) + (linux_end.tv_nsec - linux_start.tv_nsec) / 1e9;
    printf("Elapsed time: %f ms\n", elapsed_time_sec * 1000.0);
    printf("Elapsed time: %f s\n", elapsed_time_sec);
#else
    double elapsed_time_sec = 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND);
    printf("Elapsed time: %f ms\n", elapsed_time_sec * 1000.0);
    printf("Elapsed time: %f s\n", elapsed_time_sec);
#endif
#endif
#endif

#ifdef PROFILE_DATA_MOVEMENT
    printf("\n=== Data Movement Profiling Results (Total over %d iterations) ===\n", ITERATIONS);
    printf("Total input data movement time: %f ms\n", total_input_time * 1000.0);
    printf("Total output data movement time: %f ms\n", total_output_time * 1000.0);
    printf("Total compute time: %f ms\n", total_compute_time * 1000.0);
    printf("Total data movement time: %f ms\n", (total_input_time + total_output_time) * 1000.0);
    printf("Total execution time: %f ms\n", (total_input_time + total_output_time + total_compute_time) * 1000.0);
    printf("\nAverage per iteration:\n");
    printf("  Input data movement: %f ms\n", (total_input_time * 1000.0) / ITERATIONS);
    printf("  Output data movement: %f ms\n", (total_output_time * 1000.0) / ITERATIONS);
    printf("  Compute: %f ms\n", (total_compute_time * 1000.0) / ITERATIONS);
    printf("  Total per iteration: %f ms\n",
           ((total_input_time + total_output_time + total_compute_time) * 1000.0) / ITERATIONS);
#endif
}

// __global__ void passthrough(int * __restrict in, int * __restrict out) {
#ifdef PASSTHROUGH
__global__ void passthrough(int *in, int *out) {
// 1kb
// #define N 16
// 4kb
#define N 32
// 16kb
// #define N 64
#define VECTOR_SIZE 16

    unsigned int *__restrict inp = (unsigned int *)in;
    unsigned int *__restrict outp = (unsigned int *)out;

    [[chess::prepare_for_pipelining, chess::min_loop_count(N * N / VECTOR_SIZE),
      chess::max_loop_count(N * N / VECTOR_SIZE)]]
    for (int i = 0; i < N * N / VECTOR_SIZE; i++) {
        // outp[i] = inp[i];
        auto vec_in = aie::load_v<VECTOR_SIZE>(&inp[i * VECTOR_SIZE]);
        aie::store_v(&outp[i * VECTOR_SIZE], vec_in);
    }
}
#else
__global__ void mm(int *in, int *out) {
// #define N 32
#define N 64
    // #define N 128
    uint64_t count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                chess_prepare_for_pipelining { sum += in[i * N + k] * in[k * N + j]; }
            out[i * N + j] = sum;
        }
    }
    // out[N * N] = ++count;
}
#endif

int test_kernel(XAie_DevInst *DevInst) {
    // 1k
    // int n = 16;
    // 4k;
    int n = 32;
    // int n = 64;
    // n = int(n / 4.0); // keep compatible with existing code

    ITERATIONS = 100 * 1024 * 1024 / (n * n * 4);
    // ITERATIONS =  (n * n * 4);

#ifdef PROFILE_MEM_ONLY
    profile_start();
#endif

    debug_printf("\nLoading kernel 1...\n");
    XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));
#ifdef PASSTHROUGH
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)passthrough);
#else
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)mm);
#endif
    debug_printf("Finished. Continuing...\n");

    debug_printf("\nRouting...\n");
    XAie_RoutingInstance *routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, /*src=*/XAie_TileLoc(2, 0), /*dest=*/XAie_TileLoc(4, 4));
    XAie_Route(routingInstance, NULL, /*src=*/XAie_TileLoc(4, 4), /*dest=*/XAie_TileLoc(OUT_COL, 0));
    debug_printf("Finished. Continuing...\n");

    debug_printf("\nAllocating DDR memory...\n");
    int *in_ptr = 0, *out_ptr = 0;
    const int len = n * n;
    XAie_MemInst *in = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    in_ptr = (int *)XAie_MemGetVAddr(in);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    out_ptr = (int *)XAie_MemGetVAddr(out);
    debug_printf("Finished. Continuing...\n");

    debug_printf("\nInitializing DDR memory...\n");

    for (int i = 0; i < len; i++) {
        in_ptr[i] = i;
        out_ptr[i] = 0;
    }
    for (int IT = 0; IT < ITERATIONS; IT++) {
        debug_printf("Generating %dx%d matrix with random values...\n", n, n);
        // srand(100);
        // for(int i = 0; i < len; i++) {
        //     // in_ptr[i] = (rand() % 10) + 1;
        //     in_ptr[i] = i;
        //     out_ptr[i] = 0;
        // }

        in_ptr[0] = IT;

        debug_printf("Finished. Continuing...\n");

        debug_printf("\nMoving input data to AIE...\n");
        data_movement_profile_start();
        XAie_MemSyncForDev(in);
        // XAie_MemSyncForCPU(out);
        XAie_MoveDataExternal2Aie(routingInstance, /*src=*/XAie_TileLoc(2, 0), in, len * sizeof(u32), CORE_IP_MEM,
                                  /*dest=*/XAie_TileLoc(4, 4));
        XAie_MemSyncForDev(in);
        // XAie_MemSyncForCPU(out);
        data_movement_profile_end_input();
        debug_printf("Finished. Continuing...\n");

        debug_printf("\nRunning kernel 1...\n");
        data_movement_profile_start();
        XAie_Run(routingInstance, 1);
        // wait until core is done
        while (XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 0) != XAIE_OK) {
        }
        data_movement_profile_end_compute();
        debug_printf("Finished. Continuing...\n");

        debug_printf("\nMoving output data to tile memory...\n");
        data_movement_profile_start();
        // XAie_MemSyncForDev(in);
        XAie_MemSyncForCPU(out);
        XAie_MoveDataAie2External(routingInstance, /*src=*/XAie_TileLoc(4, 4), CORE_OP_MEM, len * sizeof(u32), out,
                                  /*dest=*/XAie_TileLoc(OUT_COL, 0));
        // XAie_MemSyncForDev(in);
        XAie_MemSyncForCPU(out);
        data_movement_profile_end_output();
        debug_printf("Finished. Continuing...\n");
    }
    profile_end();

    debug_printf("\nVerifying output data...\n");

    int *expected = (int *)malloc(n * n * sizeof(int));
    if (!expected) {
        debug_printf("Failed to allocate memory for expected results\\n");
        return -1;
    }

    debug_printf("Computing expected result on CPU...\\n");
#ifdef PASSTHROUGH
    // For passthrough kernel, output should match input
    for (int i = 0; i < n * n; i++) {
        expected[i] = in_ptr[i];
    }
#else
    // For matrix multiplication kernel, compute A^2
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += in_ptr[i * n + k] * in_ptr[k * n + j];
            }
            expected[i * n + j] = sum;
        }
    }
#endif

    debug_printf("\nInput matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            debug_printf("%4d ", in_ptr[i * n + j]);
        }
        debug_printf("\n");
    }

#ifdef PASSTHROUGH
    debug_printf("\nExpected output matrix (passthrough):\n");
#else
    debug_printf("\nExpected output matrix (A^2):\n");
#endif
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            debug_printf("%6d ", expected[i * n + j]);
        }
        debug_printf("\n");
    }

    printf("\nActual output matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6d ", out_ptr[i * n + j]);
        }
        printf("\n");
    }

    int mismatches = 0;
    int max_mismatches_to_show = 10;

    debug_printf("\nVerifying all %d matrix elements...\n", n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            if (expected[idx] != out_ptr[idx]) {
                if (mismatches < max_mismatches_to_show) {
                    debug_printf("Mismatch at [%d][%d]: Expected=%d, AIE=%d\n", i, j, expected[idx], out_ptr[idx]);
                }
                mismatches++;
            }
        }
    }

    if (mismatches > max_mismatches_to_show) {
        debug_printf("... and %d more mismatches\n", mismatches - max_mismatches_to_show);
    }

    debug_printf("\nCount value: %d\n", out_ptr[n * n]);
    debug_printf("Matrix size: %dx%d (%d elements)\n", n, n, n * n);

    // free(expected);

    if (mismatches == 0) {
        debug_printf("Success: CPU result matches AIE for all %d elements.\n", n * n);
        debug_printf("\n\nDone with kernel 1!\n\n");
        return 0;
    } else {
        printf("Failure: There were %d mismatches out of %d elements (%.2f%% accuracy).\n", mismatches, n * n,
               100.0 * (n * n - mismatches) / (n * n));
        debug_printf("\n\nDone with kernel 1!\n\n");
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
    printf("Perf Test\n");
    // Xil_DCacheDisable();
    // Xil_ICacheDisable();

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if (RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

#ifdef __AIELINUX__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_LINUX);
#else
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
#endif

#ifdef PROFILE_FULL
    profile_start();
#endif
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
        print_profile_result();
        return 0;
    } else {
        printf("\nKernel test failed!\n");
        print_profile_result();
        return -1;
    }
}