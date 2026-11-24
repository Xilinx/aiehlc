/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
char expect[16];
int aio_rc = aiehlc_io_check();
if (aio_rc != 0 || aiehlc_io_n() != @MU @_N_INPUTS) {
    printf("@M@: .aiehlc_io section invalid rc=%d n_in=%d\n", aio_rc, aiehlc_io_n());
    return 1;
    }
    uint32_t aio_crc = aiehlc_io_crc();   /* the expect: computed here from the (patched) section */
    snprintf(expect, sizeof expect, "%08x", (unsigned)aio_crc);
    if (aio_crc != aiehlc_io_hdr_crc()) {
        printf("@M@: .aiehlc_io crc32=%s != header crc32=%08x\n", expect, (unsigned)aiehlc_io_hdr_crc());
        return 1;
    }
    uint32_t c = 0xFFFFFFFFu;
@NOTE_DECL@    for (int k = 0; k < @MU@_N_INPUTS; ++k) {
        if (strcmp(graph_input_name(k), @M@_in_name[k]) != 0) {
            printf("@M@: graph input %d is %s, expected %s\n", k, graph_input_name(k), @M@_in_name[k]);
            return 1;
        }
        if (aiehlc_io_bytes(k) != @M@_in_bytes[k]) {
            printf("@M@: .aiehlc_io input %d (%s) has %lu bytes, graph wants %lu\n", k, aiehlc_io_name(k),
                   aiehlc_io_bytes(k), @M@_in_bytes[k]);
            return 1;
        }
        uint8_t *dst = (uint8_t *)graph_input_k(k);
        memcpy(dst, aiehlc_io_ptr(k), @M@_in_bytes[k]);
        c = crc32_upd(c, dst, @M@_in_bytes[k]);
@NOTE_GRAB@        printf("INPUT @M@ %s bytes=%lu (runtime-io %s)\n", @M@_in_name[k], @M@_in_bytes[k], aiehlc_io_name(k));
    }
    snprintf(crc, sizeof crc, "%08x", (unsigned)(c ^ 0xFFFFFFFFu));
    printf("INPUT @M@ @IN_DESC@ in_crc32_board=%s expect=%s %s@NOTE_FMT@ runtime_io=1 "
           "build_crc32=%s\n", crc, expect, strcmp(crc, expect) == 0 ? "MATCH" : "MISMATCH",@NOTE_RARG@
           P1_@MU@_IN_CRC32);
    if (strcmp(crc, expect) != 0) {
        printf("@M@: input crc mismatch -- not comparing\n");
        return 1;
    }
    const int @M@_rtio_own = strcmp(crc, P1_@MU@_IN_CRC32) == 0;   /* the build's real_io: p1_ref applies */
