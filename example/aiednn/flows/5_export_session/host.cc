/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flow 5 -- hand-written host against an exported model's session API.
 * run.py emits <name>_model.h / .cc next to a rewritten copy of this file, then
 * dnn_build. Placeholders @@NAME@@ / @@CNAME@@ / @@CNAME_UP@@ / @@TOL@@ are filled in.
 *
 *   python example/aiednn/flows/5_export_session/run.py
 *   ... --target vek280 --board galerina13 --vitis-settings <.../settings64.sh>
 *
 * dnn_build picks kernels from create calls inside the included model .cc:
 * aiednn_convolution_forward_create( aiednn_matmul_forward_create(
 ******************************************************************************/
#include "@@NAME@@_model.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "@@NAME@@_model.cc"

#define INCBIN(sym, path) \
    __asm__(".section .rodata\n.balign 64\n.global " #sym "\n" #sym ":\n.incbin \"" path "\"\n.previous\n")
INCBIN(host_in_data, "@@NAME@@_host_in.bin");
INCBIN(host_golden_data, "@@NAME@@_golden.bin"); /* run.py: the PyTorch reference output (float) */
extern "C" const unsigned char host_in_data[], host_golden_data[];
#define TOL @@TOL@@

static unsigned char g_out[@@CNAME_UP@@_OUT_BYTES] __attribute__((aligned(64)));

int main() {
    printf("=== flow 5 export session (@@NAME@@) ===\n");
    @@CNAME@@_t *m = @@CNAME@@_create();
    if (!m) {
        printf("FAIL: @@CNAME@@_create\n");
        return 1;
    }
    const aiednn_memory_desc_t *in = @@CNAME@@_input_desc(m);
    const aiednn_memory_desc_t *out = @@CNAME@@_output_desc(m);
    printf("input ndims=%d dt=%d  output ndims=%d dt=%d\n",
           in->ndims, (int)in->data_type, out->ndims, (int)out->data_type);

    aiednn_status_t st = @@CNAME@@_run(m, host_in_data, g_out);
    if (st != aiednn_success) {
        printf("FAIL: @@CNAME@@_run: %s\n", aiednn_status_str(st));
        @@CNAME@@_destroy(m);
        return 1;
    }

    /* check against the reference: relative L2 error under TOL and the same top-1 */
    const float *y = (const float *)g_out, *ref = (const float *)host_golden_data;
    const int n = @@CNAME_UP@@_OUT_BYTES / (int)sizeof(float);
    double num = 0, den = 0;
    int top = 0, rtop = 0;
    for (int i = 0; i < n; i++) {
        num += ((double)y[i] - ref[i]) * ((double)y[i] - ref[i]);
        den += (double)ref[i] * ref[i];
        if (y[i] > y[top])
            top = i;
        if (ref[i] > ref[rtop])
            rtop = i;
    }
    const double rel = sqrt(num / (den > 0 ? den : 1));
    printf("top-1 %d (reference %d)  rel err %.3e  logits[0]=%.4f\n", top, rtop, rel, y[0]);
    @@CNAME@@_destroy(m);
    if (!(rel < TOL) || top != rtop) {
        printf("FAIL: flow 5 export session differs from the reference (rel %.3e, tol %g)\n", rel, TOL);
        return 1;
    }
    printf("PASS: flow 5 export session\n");
    return 0;
}
