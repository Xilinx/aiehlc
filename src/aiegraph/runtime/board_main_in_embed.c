/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
uint32_t c = 0xFFFFFFFFu;
for (int k = 0; k < @MU @_N_INPUTS; ++k) {
    if (strcmp(graph_input_name(k), @M @_in_name[k]) != 0) {
        printf("@M@: graph input %d is %s, expected %s\n", k, graph_input_name(k), @M @_in_name[k]);
        return 1;
    }
    uint8_t *dst = (uint8_t *)graph_input_k(k);
    memcpy(dst, @M @_in_ptr[k], @M @_in_bytes[k]);
    c = crc32_upd(c, dst, @M @_in_bytes[k]);
    printf("INPUT @M@ %s bytes=%lu\n", @M @_in_name[k], @M @_in_bytes[k]);
    }
    snprintf(crc, sizeof crc, "%08x", (unsigned)(c ^ 0xFFFFFFFFu));
    printf("INPUT @M@ @IN_DESC@ in_crc32_board=%s expect=%s %s@NOTE_FMT@\n",
           crc, P1_@MU@_IN_CRC32, strcmp(crc, P1_@MU@_IN_CRC32) == 0 ? "MATCH" : "MISMATCH"@NOTE_EARG@);
    if (strcmp(crc, P1_@MU@_IN_CRC32) != 0) {
        printf("@M@: input crc mismatch -- not comparing\n");
        return 1;
    }
