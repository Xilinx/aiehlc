/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
__global__ void stem_add(aie::port<input_window_int8 *, aie::shape<256, 1024>> a,
                         aie::port<input_window_int8 *, aie::shape<256, 1024>> b,
                         aie::port<output_window_int8 *, aie::shape<256, 1024>> c) {
    const
}