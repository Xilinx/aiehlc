/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* cmp_cos.c — aiegraph board-side compare tool (was example/aiegraph/unet_bk_sdm_tiny/p1_cos.c, Shuhua zoo/p1_apu).
 * Prints exactly one "CMP <model> board=... cos=... PASS|FAIL" line: <model> is the tag the caller passes (the
 * model's name, e.g. "unet-kincore"); the board checkers (check_smoke.py) parse that line.
 * Reference arrays are EMBEDDED from p1_ref.h, generated on x86 by the model's export script (text from
 * src/aiegraph/runtime/cmp_ref_head.h + cmp_ref_array.h).  Build with -I<real_io dir>.
 *
 * Library use (link into the board elf, compile with -DP1_COS_NO_MAIN):
 *   int p1_cmp(const char *tag, const float *got, const char *dtype,
 *              int aie_launches, int coreen_calls, int run_rc, double aie_flop_frac);
 *   tag "part01-apu" / "part01-*" -> ref_part01_out (in_crc32 = sample.crc32)
 *   any other tag                 -> ref_unet_out   (in_crc32 = unet_in.crc32)
 *   returns 0 on PASS, 1 on FAIL; prints exactly one CMP line.
 * Standalone: cmp_cos <tag> <got.bin> <dtype> <aie_launches> <coreen_calls> <run_rc> <aie_flop_frac>
 * PASS: cos >= 0.999; and unless tag == "part01-apu" or tag starts with "apu-": N >= 1, N == M, run_rc == 0.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "p1_ref.h"

/* a generic p1_ref.h (aiegraph.boardmain.ref_header) names its one reference P1_REF_OUT; the UNet
 * export's p1_ref.h has the part01 + unet pair */
#ifndef P1_REF_OUT
#define P1_REF_OUT ref_unet_out
#define P1_REF_OUT_N REF_UNET_OUT_N
#define P1_REF_IN_CRC32 P1_UNET_IN_CRC32
#define P1_HAVE_PART01 1
#else
#define P1_HAVE_PART01 0
#define ref_part01_out P1_REF_OUT
#define REF_PART01_OUT_N P1_REF_OUT_N
#define P1_SAMPLE_CRC32 P1_REF_IN_CRC32
#endif

#ifndef P1_BOARD
#define P1_BOARD "palmyra213"
#endif

int p1_cmp(const char *tag, const float *got, const char *dtype, int n_launch, int n_coreen, int run_rc,
           double aie_flop_frac) {
    int is_part01 = P1_HAVE_PART01 && strncmp(tag, "part01", 6) == 0;
    const float *ref = is_part01 ? ref_part01_out : P1_REF_OUT;
    long n = is_part01 ? REF_PART01_OUT_N : P1_REF_OUT_N;
    const char *crc = is_part01 ? P1_SAMPLE_CRC32 : P1_REF_IN_CRC32;
    double dot = 0, ng = 0, nr = 0, mx = 0;
    int finite = 1;
    for (long i = 0; i < n; i++) {
        double g = got[i], r = ref[i], d = fabs(g - r);
        if (!isfinite(g))
            finite = 0;
        dot += g * r;
        ng += g * g;
        nr += r * r;
        if (d > mx)
            mx = d;
    }
    double cs = (finite && ng > 0 && nr > 0) ? dot / (sqrt(ng) * sqrt(nr)) : 0.0;
    int pass = cs >= 0.999;
    /* part01-apu and apu-* (x86/APU-only runs, no AIE) are exempt from the launch-count checks;
     * every other tag (incl. P4 "unet") still needs N >= 1, N == M, run_rc == 0. */
    if (strcmp(tag, "part01-apu") != 0 && strncmp(tag, "apu-", 4) != 0)
        pass = pass && n_launch >= 1 && n_launch == n_coreen && run_rc == 0;
    printf("CMP %s board=%s data=%s in_crc32=%s cos=%.6f maxabs=%g dtype=%s aie_launches=%d coreen_calls=%d "
           "run_rc=%d aie_flop_frac=%.3f %s\n",
           tag, P1_BOARD, P1_DATA, crc, cs, finite ? mx : INFINITY, dtype, n_launch, n_coreen, run_rc, aie_flop_frac,
           pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass ? 0 : 1;
}

#ifndef P1_COS_NO_MAIN
int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s <tag> <got.bin> <dtype apu/aie> <aie_launches> <coreen_calls> <run_rc> <aie_flop_frac>\n",
                argv[0]);
        return 2;
    }
    long n = P1_HAVE_PART01 && strncmp(argv[1], "part01", 6) == 0 ? REF_PART01_OUT_N : P1_REF_OUT_N;
    float *got = malloc(n * sizeof(float));
    FILE *f = fopen(argv[2], "rb");
    if (!got || !f || fread(got, sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "cmp_cos: cannot read %ld floats from %s\n", n, argv[2]);
        return 2;
    }
    fclose(f);
    int rc = p1_cmp(argv[1], got, argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atof(argv[7]));
    free(got);
    return rc;
}
#endif
