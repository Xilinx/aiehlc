/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <stdint.h>
#define FOR_READ 1
#define FOR_WRITE 0
#define BUF_SZ 16
inline int8_t *acquire_output_window(output_window_int8 *win) {
    window_internal *w = (window_internal *)win;
    w->buffer = (window_datatype *)select(w->current_bufid, w->buffers[1], w->buffers[0]);
    w->head = w->ptr = (window_datatype *)select(w->current_bufid, w->heads[1], w->heads[0]);
    acquire_greater_equal(w->lockids[0], 1);
    return (int8_t *)w->ptr;
}
inline void release_output_window(output_window_int8 *win) {
    chess_memory_fence();
    window_internal *w = (window_internal *)win;
    release(w->lockids[1], 1);
    w->heads[w->current_bufid] = w->head;
    w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);
}
inline int8_t *acquire_input_window(input_window_int8 *win) {
    window_internal *w = (window_internal *)win;
    w->buffer = (window_datatype *)select(w->current_bufid, w->buffers[1], w->buffers[0]);
    w->head = w->ptr = (window_datatype *)select(w->current_bufid, w->heads[1], w->heads[0]);
    acquire_greater_equal(w->lockids[1], 1);
    return (int8_t *)w->ptr;
}
inline void release_input_window(input_window_int8 *win) {
    chess_memory_fence();
    window_internal *w = (window_internal *)win;
    release(w->lockids[0], 1);
    w->heads[w->current_bufid] = w->head;
    w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);
}
#define LOCK_window_in_0_ACQ 48
#define LOCK_window_in_0_REL 49
v4int8 buf_in_ping_0[BUF_SZ];
v4int8 buf_in_pong_0[BUF_SZ];
// window_def window_in_0
#define LOCK_window_in_1_ACQ 50
#define LOCK_window_in_1_REL 51
v4int8 buf_in_ping_1[BUF_SZ];
v4int8 buf_in_pong_1[BUF_SZ];
// window_def window_in_1
#define LOCK_window_out_0_ACQ 52
#define LOCK_window_out_0_REL 53
v4int8 buf_out_ping_0[BUF_SZ];
v4int8 buf_out_pong_0[BUF_SZ];
// window_def window_out_0
#include "kernel_log.h"
#include "matmul.cc"
// kernel_decl matmul
int32_t main() {
    volatile static int sync_buffer[8] = {0, -1};
    sync_buffer[0] = 0;
    klog_init();
    // alloc_sync_buffer
    // sync_buffer_write
    // log(...)
    window_internal window_window_in_0[1];
    window_init(window_window_in_0, 1, buf_in_ping_0, LOCK_window_in_0_ACQ, buf_in_pong_0, LOCK_window_in_0_REL, BUF_SZ,
                BUF_SZ);
    window_internal window_window_in_1[1];
    window_init(window_window_in_1, 1, buf_in_ping_1, LOCK_window_in_1_ACQ, buf_in_pong_1, LOCK_window_in_1_REL, BUF_SZ,
                BUF_SZ);
    window_internal window_window_out_0[1];
    window_init(window_window_out_0, 1, buf_out_ping_0, LOCK_window_out_0_ACQ, buf_out_pong_0, LOCK_window_out_0_REL,
                BUF_SZ, BUF_SZ);
    // kernel_invoke matmul
    matmul(get_input_async_window_int8(window_window_in_0), get_input_async_window_int8(window_window_in_1),
           get_output_async_window_int8(window_window_out_0));
    done();
    int32_t v1 = 0;
    return v1;
}
