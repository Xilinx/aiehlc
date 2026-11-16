/* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Direct-conv mode of the GEMM megakernel (gemm_uk.cc, AIEK_MODE_CONV). A conv launch's A is not an
 * im2col block: each mesh row's 16 KB A slice is the raw HWC input region its 32 output pixels need
 * for the launch's K block (input rows ih_lo .. ih_lo + rows - 1, every padded column, Cg channels,
 * zero where the conv pads), and the core gathers the 32 x 512 im2col tile from it. The K order is
 * the host's (kh, kw, ci) im2col order, so B is the same weights block as for an im2col launch.
 * Mesh row r's geometry (16 int32, CONV_GEOM_*) is the last 64 bytes of its own A slice, and word 0
 * of B row CONV_GEOM_ROW0 is CONV_MAGIC for the whole layer, so one B block can serve many tiles.
 * Not a flag in row 127: for 0xF1 its bytes 4.. are the requant zero point. */
#ifndef AIE_MODES_CONV_H
#define AIE_MODES_CONV_H

#define CONV_GEOM_ROW0 112 /* B row 112 word 0: CONV_MAGIC (per-launch tail, padding for a GEMM) */
#define CONV_GEOM_CG 0     /* input channels of the group (multiple of 4) */
#define CONV_GEOM_KH 1
#define CONV_GEOM_KW 2
#define CONV_GEOM_S0 3
#define CONV_GEOM_S1 4
#define CONV_GEOM_D0 5
#define CONV_GEOM_D1 6
#define CONV_GEOM_PL0 7 /* top padding */
#define CONV_GEOM_WP 8  /* region row pitch in pixels: W + left + right padding */
#define CONV_GEOM_WO 9
#define CONV_GEOM_M 10        /* output pixels of the image */
#define CONV_GEOM_M0 11       /* this mesh row's first output pixel */
#define CONV_GEOM_K0 12       /* the launch's first K column */
#define CONV_GEOM_IHLO 13     /* the region's first input row (may be < 0: top padding) */
#define CONV_GEOM_KTOT 14     /* K of the conv: units past it are zero */
#define CONV_GEOM_MAGIC 15    /* CONV_MAGIC (a check: the geometry was written) */
#define CONV_MAGIC 0x564E4F43 /* "CONV" */
#define CONV_REGION_BYTES 16384
#define CONV_REGION_DATA (CONV_REGION_BYTES - 64) /* region bytes; the geometry follows */
/* a pixel's base may be negative (a K block starting at kernel row kh > 0: the tap offset brings it
 * back into the region), so "no pixel" (past M) is its own value */
#define CONV_NOPX (-2147483647 - 1)

/* the gather tables from a mesh row's geometry: per 4-byte K unit of the block, the region
 * offset of its (tap, channel) (-1 past K); per A row, the offset of its pixel (CONV_NOPX past M) */
static inline void conv_tables(const int32_t *g, int32_t *base, int32_t *toff, int nunits, int tile_rows) {
    const int Cg = g[CONV_GEOM_CG], KW = g[CONV_GEOM_KW], Wp = g[CONV_GEOM_WP], Wo = g[CONV_GEOM_WO];
    const int s0 = g[CONV_GEOM_S0], s1 = g[CONV_GEOM_S1], d0 = g[CONV_GEOM_D0], d1 = g[CONV_GEOM_D1];
    const int k0 = g[CONV_GEOM_K0], kt = g[CONV_GEOM_KTOT], m0 = g[CONV_GEOM_M0], mt = g[CONV_GEOM_M];
    int ci = k0 % Cg, t = k0 / Cg, kh = t / KW, kw = t - kh * KW; // the block's first K unit
    for (int u = 0; u < nunits; u++) {
        toff[u] = k0 + 4 * u < kt ? ((kh * d0) * Wp + kw * d1) * Cg + ci : -1;
        ci += 4;
        if (ci >= Cg) {
            ci = 0;
            if (++kw == KW)
                kw = 0, kh++;
        }
    }
    int oh = m0 / Wo, ow = m0 - oh * Wo; // the core row's first pixel
    const int top = g[CONV_GEOM_PL0] + g[CONV_GEOM_IHLO];
    for (int p = 0; p < tile_rows; p++) {
        base[p] = m0 + p < mt ? ((oh * s0 - top) * Wp + ow * s1) * Cg : CONV_NOPX;
        if (++ow == Wo)
            ow = 0, oh++;
    }
}

#if defined(__chess__)
/* round 0 on the core: all of this core row's A windows (its 16 KB region, row-major as sent) into
 * reg, then the tables */
static inline void conv_region_in(input_window_int8 *win_a, int8_t *reg, int32_t *base, int32_t *toff, int k_rounds,
                                  int num_a_rounds, int rows_per_round, int eff_k, int tile_rows) {

    for (int kr = 0; kr < k_rounds; kr++)
        for (int ra = 0; ra < num_a_rounds; ra++) {
            const uint32_t *A_w = (const uint32_t *)acquire_input_window(win_a);
            for (int r = 0; r < rows_per_round; r++) {
                uint32_t *dst = (uint32_t *)(reg + (ra * rows_per_round + r) * (k_rounds * eff_k) + kr * eff_k);
                const uint32_t *src = A_w + r * (eff_k / 4);
                for (int w = 0; w < eff_k / 4; w++)
                    dst[w] = src[w];
            }
            release_input_window(win_a);
        }
    conv_tables((const int32_t *)(reg + CONV_REGION_DATA), base, toff, k_rounds * eff_k / 4, tile_rows);
}
#endif

/* Direct conv, every round: the tile_rows x eff_k im2col slice of K round kr, in a_pack's 8x8 order.
 * run64 (Cg % 64 == 0, eff_k == 64): the round's 64 bytes of a row are one contiguous run of one
 * tap -- one table lookup per row, copied like the plain A pack (two words per 8 bytes). */
static inline void conv_pack_round(const int8_t *reg, const int32_t *base, const int32_t *toff, int8_t *a_pack, int kr,
                                   int eff_k, int k_tiles, int tile_rows, int run64) {
    const int nu = eff_k / 4;
    if (run64) {
        const int o = toff[kr * nu];
        for (int p = 0; p < tile_rows; p++) {
            uint32_t *row = (uint32_t *)(a_pack + (p >> 3) * k_tiles * 64 + (p & 7) * 8);
            const int b = base[p];
            if (b == CONV_NOPX || o < 0) {
                for (int kt = 0; kt < k_tiles; kt++)
                    row[kt * 16] = 0u, row[kt * 16 + 1] = 0u;
                continue;
            }
            const uint32_t *src = (const uint32_t *)(reg + b + o);
            for (int kt = 0; kt < k_tiles; kt++) {
                row[kt * 16] = src[kt * 2];
                row[kt * 16 + 1] = src[kt * 2 + 1];
            }
        }
        return;
    }
    for (int p = 0; p < tile_rows; p++) {
        uint32_t *row = (uint32_t *)(a_pack + (p >> 3) * k_tiles * 64 + (p & 7) * 8);
        const int b = base[p];
        for (int u = 0; u < nu; u++) {
            const int o = toff[kr * nu + u];
            row[(u >> 1) * 16 + (u & 1)] = b == CONV_NOPX || o < 0 ? 0u : *(const uint32_t *)(reg + b + o);
        }
    }
}

#endif /* AIE_MODES_CONV_H */
