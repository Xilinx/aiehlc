
/* GFA (--attn-fa): fused flash-attention sites. unet_aie_attn(cid, args, nargs, role, cls): role 0 = the QK^T
 * bmm site (packs Q x log2e and the centred K into the class qk buffer, no launch), role 1 = the PV bmm site
 * (packs V, launches the class kernel units/4 times, writes O f32). The TVM softmax between them is gone. */
typedef struct {
    int H, N, M, D, split, dq, nkv, E, units, upr, launches, qkrow, vrow, orow;
} fa_cls;
static const fa_cls g_fa_cls[] = {
@CLASSES@
};
#define FA_NCLS ((int)(sizeof(g_fa_cls) / sizeof(g_fa_cls[0])))
static uint16_t *g_fa_qk[FA_NCLS], *g_fa_v[FA_NCLS], *g_fa_o[FA_NCLS];
static float *g_fa_mu = nullptr;
static unsigned g_fa_sites = 0;
static inline uint16_t fa_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float fa_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#if UC_NEON
/* 4 x fp32 -> bf16 round-to-nearest-even (bit-identical to fa_f2bf for finite values) */
static inline uint16x4_t fa_f2bf4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u));
    u = vaddq_u32(u, vaddq_u32(lsb, vdupq_n_u32(0x7fffu)));
    return vshrn_n_u32(u, 16);
}
static inline uint16x8_t fa_f2bf8(float32x4_t a, float32x4_t b) { return vcombine_u16(fa_f2bf4(a), fa_f2bf4(b)); }
/* in-place 8x8 transpose of 16-bit lanes: r[i][j] -> r[j][i] */
static inline void fa_tr8(uint16x8_t r[8]) {
    const uint16x8x2_t t0 = vtrnq_u16(r[0], r[1]), t1 = vtrnq_u16(r[2], r[3]), t2 = vtrnq_u16(r[4], r[5]),
                       t3 = vtrnq_u16(r[6], r[7]);
    const uint32x4x2_t u0 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[0]), vreinterpretq_u32_u16(t1.val[0]));
    const uint32x4x2_t u1 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[1]), vreinterpretq_u32_u16(t1.val[1]));
    const uint32x4x2_t u2 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[0]), vreinterpretq_u32_u16(t3.val[0]));
    const uint32x4x2_t u3 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[1]), vreinterpretq_u32_u16(t3.val[1]));
#define FA_T1(a, b) vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(a), vreinterpretq_u64_u32(b)))
#define FA_T2(a, b) vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(a), vreinterpretq_u64_u32(b)))
    r[0] = FA_T1(u0.val[0], u2.val[0]);
    r[4] = FA_T2(u0.val[0], u2.val[0]);
    r[1] = FA_T1(u1.val[0], u3.val[0]);
    r[5] = FA_T2(u1.val[0], u3.val[0]);
    r[2] = FA_T1(u0.val[1], u2.val[1]);
    r[6] = FA_T2(u0.val[1], u2.val[1]);
    r[3] = FA_T1(u1.val[1], u3.val[1]);
    r[7] = FA_T2(u1.val[1], u3.val[1]);
#undef FA_T1
#undef FA_T2
}
#endif
static int fa_alloc(int k) {
    if (g_fa_qk[k])
        return 0;
    const fa_cls &c = g_fa_cls[k];
    g_fa_qk[k] = (uint16_t *)__Runtime_Alloc((size_t)c.units * c.qkrow * 2);
    g_fa_v[k] = (uint16_t *)__Runtime_Alloc((size_t)c.H * c.vrow * 2); /* kv stream per head */
    g_fa_o[k] = (uint16_t *)__Runtime_Alloc((size_t)c.units * c.orow * 2);
    if (!g_fa_mu)
        g_fa_mu = (float *)aligned_alloc(64, 256 * 4);
    return (g_fa_qk[k] && g_fa_v[k] && g_fa_o[k] && g_fa_mu) ? 0 : -1;
}
static int uc_fa_launch(int k, uint16_t *qk, uint16_t *v, uint16_t *o) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // aiegraph lint L015: the mesh VarDecl in the launching function
    int16_t *q16 = (int16_t *)qk, *v16 = (int16_t *)v, *o16 = (int16_t *)o;
@DISPATCH@
    return -6;
}
/* block layouts (Maxwell fa_load.h); dq = block width (80 for the halves of a split D=160 block) */
static inline unsigned fa_qidx(unsigned d, unsigned r) { return (((d >> 3) * 4 + (r >> 3)) * 8 + (d & 7)) * 8 + (r & 7); }
static inline unsigned fa_kidx(unsigned j, unsigned d, unsigned dq) { return (((j >> 2) * (dq >> 3) + (d >> 3)) * 4 + (j & 3)) * 8 + (d & 7); }
static inline unsigned fa_vidx(unsigned d, unsigned j) { return (((d >> 2) * 4 + (j >> 3)) * 4 + (d & 3)) * 8 + (j & 7); }
/* P4a keep-layout: FA operand packs with explicit element strides, so the TVM head split / merge ops can go.
 * Q/K (h, r, d) at base + h*hs + r*rs + d (TVM [H][R][D]: hs = R*D, rs = D; linear out [R][H*D]: hs = D, rs = H*D),
 * scaled by s (the D^-1/4 of the removed reshape_transpose_multiply ops; 1 = already scaled);
 * V: vt = 1 -> V^T [H][D][M] (h, d, j) at base + h*hs + d*M + j; vt = 0 -> (h, j, d) at base + h*hs + j*rs + d;
 * O (h, n, d) written at base + h*hs + n*rs + d. */
static void fa_pack_k(const fa_cls &c, const float *K, size_t hs, size_t rs, float s, uint16_t *V) {
    const int H = c.H, M = c.M, D = c.D, dq = c.dq, E = c.E, nkv = c.nkv, nh = c.split ? 2 : 1;
    for (int h = 0; h < H; h++) {
        const float *kh = K + (size_t)h * hs;
        /* PKC: keys centred over the M real keys (per column mean) */
        float *mu = g_fa_mu;
        for (int d = 0; d < D; d++)
            mu[d] = 0.0f;
        for (int j = 0; j < M; j++) {
            const float *kr = kh + (size_t)j * rs;
            for (int d = 0; d < D; d++)
                mu[d] += kr[d];
        }
        const float im = 1.0f / (float)M;
        for (int d = 0; d < D; d++)
            mu[d] *= im;
        /* K block (b, half hh) at kv stream slot b*2nh + hh of head h: kidx order = j4, d8, j&3, d&7 */
        for (int b = 0; b < nkv; b++)
            for (int hh = 0; hh < nh; hh++) {
                uint16_t *kp = V + (size_t)h * c.vrow + (size_t)(b * 2 * nh + hh) * E;
                for (int j4 = 0; j4 < 8; j4++)
                    for (int d8 = 0; d8 < dq / 8; d8++)
                        for (int jj = 0; jj < 4; jj++) {
                            const int j = b * 32 + j4 * 4 + jj, c0 = hh * dq + d8 * 8;
                            if (j < M) {
                                const float *kr = kh + (size_t)j * rs + c0, *mr = mu + c0;
#if UC_NEON
                                float32x4_t a = vsubq_f32(vld1q_f32(kr), vld1q_f32(mr)),
                                            b2 = vsubq_f32(vld1q_f32(kr + 4), vld1q_f32(mr + 4));
                                if (s != 1.0f) {
                                    a = vmulq_n_f32(a, s);
                                    b2 = vmulq_n_f32(b2, s);
                                }
                                vst1q_u16(kp, fa_f2bf8(a, b2));
#else
                                for (int dd = 0; dd < 8; dd++)
                                    kp[dd] = fa_f2bf(s != 1.0f ? (kr[dd] - mr[dd]) * s : kr[dd] - mr[dd]);
#endif
                            } else
                                for (int dd = 0; dd < 8; dd++)
                                    kp[dd] = 0;
                            kp += 8;
                        }
            }
    }
}
static void fa_pack_q(const fa_cls &c, const float *Q, size_t hs, size_t rs, float s, uint16_t *QK) {
    const int H = c.H, N = c.N, dq = c.dq, E = c.E, nh = c.split ? 2 : 1;
    const float lg2e = 1.4426950408889634f * s;
    for (int h = 0; h < H; h++) {
        const float *qh = Q + (size_t)h * hs;
        /* v5 Q: query block qb (32 rows) of head h -> launch l = (qb/4)*(H/4) + h/4, q row pm = qb%4, slot h%4;
         * split (N=64): q rows pm = 2qb, 2qb+1 both hold sub-blocks (h%4)*2 + hq of query block qb */
        for (int qb = 0; qb < N / 32; qb++)
            for (int rep = 0; rep < (c.split ? 2 : 1); rep++) {
                const int l = c.split ? h / 4 : (qb / 4) * (H / 4) + h / 4, pm = c.split ? qb * 2 + rep : qb % 4;
                for (int hq = 0; hq < nh; hq++) {
                    uint16_t *qp = QK + (size_t)(l * 4 + pm) * c.qkrow + (size_t)((h % 4) * nh + hq) * E;
                    const float *q0 = qh + (size_t)(qb * 32) * rs + hq * dq;
                    for (int d8 = 0; d8 < dq / 8; d8++)
                        for (int r8 = 0; r8 < 4; r8++) {
#if UC_NEON
                            const float32x4_t sc = vdupq_n_f32(lg2e);
                            uint16x8_t t[8];
                            for (int rr = 0; rr < 8; rr++) {
                                const float *qs = q0 + (size_t)(r8 * 8 + rr) * rs + d8 * 8;
                                t[rr] = fa_f2bf8(vmulq_f32(vld1q_f32(qs), sc), vmulq_f32(vld1q_f32(qs + 4), sc));
                            }
                            fa_tr8(t);
                            for (int dd = 0; dd < 8; dd++)
                                vst1q_u16(qp + dd * 8, t[dd]);
                            qp += 64;
#else
                            for (int dd = 0; dd < 8; dd++) {
                                const float *qc = q0 + (size_t)(r8 * 8) * rs + d8 * 8 + dd;
                                for (int rr = 0; rr < 8; rr++)
                                    qp[rr] = fa_f2bf(qc[(size_t)rr * rs] * lg2e);
                                qp += 8;
                            }
#endif
                        }
                }
            }
    }
}
#if UC_NEON
static inline void fa_tr4f(float32x4_t &a, float32x4_t &b, float32x4_t &c, float32x4_t &d) {
    const float32x4x2_t t0 = vtrnq_f32(a, b), t1 = vtrnq_f32(c, d);
    a = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
    b = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
    c = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
    d = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
}
#endif
static void fa_pack_v(const fa_cls &c, const float *VS, size_t hs, size_t rs, int vt, uint16_t *V) {
    const int H = c.H, M = c.M, dq = c.dq, E = c.E, nkv = c.nkv, nh = c.split ? 2 : 1;
    for (int h = 0; h < H; h++) {
        const float *vh = VS + (size_t)h * hs;
        /* V block (b, half hh) at kv stream slot b*2nh + nh + hh of head h: vidx order = d4, j8, d&3, j&7 */
        for (int b = 0; b < nkv; b++)
            for (int hh = 0; hh < nh; hh++) {
                uint16_t *vp = V + (size_t)h * c.vrow + (size_t)(b * 2 * nh + nh + hh) * E;
                for (int d4 = 0; d4 < dq / 4; d4++)
                    for (int j8 = 0; j8 < 4; j8++) {
                        const int j0 = b * 32 + j8 * 8, dc = hh * dq + d4 * 4;
                        if (vt) {
                            for (int dd = 0; dd < 4; dd++) {
                                const float *vr = vh + (size_t)(dc + dd) * M + j0;
                                if (j0 + 8 <= M)
#if UC_NEON
                                    vst1q_u16(vp, fa_f2bf8(vld1q_f32(vr), vld1q_f32(vr + 4)));
#else
                                    for (int jj = 0; jj < 8; jj++)
                                        vp[jj] = fa_f2bf(vr[jj]);
#endif
                                else
                                    for (int jj = 0; jj < 8; jj++)
                                        vp[jj] = j0 + jj < M ? fa_f2bf(vr[jj]) : 0;
                                vp += 8;
                            }
                            continue;
                        }
                        /* vt = 0: rows j of the linear output, 4 contiguous d per row -> 4x8 block by transposes */
                        if (j0 + 8 <= M) {
#if UC_NEON
                            float32x4_t r[8];
                            for (int jj = 0; jj < 8; jj++)
                                r[jj] = vld1q_f32(vh + (size_t)(j0 + jj) * rs + dc);
                            fa_tr4f(r[0], r[1], r[2], r[3]);
                            fa_tr4f(r[4], r[5], r[6], r[7]);
                            for (int dd = 0; dd < 4; dd++)
                                vst1q_u16(vp + dd * 8, fa_f2bf8(r[dd], r[4 + dd]));
#else
                            for (int dd = 0; dd < 4; dd++)
                                for (int jj = 0; jj < 8; jj++)
                                    vp[dd * 8 + jj] = fa_f2bf(vh[(size_t)(j0 + jj) * rs + dc + dd]);
#endif
                        } else
                            for (int dd = 0; dd < 4; dd++)
                                for (int jj = 0; jj < 8; jj++)
                                    vp[dd * 8 + jj] = j0 + jj < M ? fa_f2bf(vh[(size_t)(j0 + jj) * rs + dc + dd]) : 0;
                        vp += 32;
                    }
            }
    }
}
static int fa_run_o(const fa_cls &c, int k, float *Y, size_t hs, size_t rs) {
    const int H = c.H, N = c.N, M = c.M, D = c.D, dq = c.dq, E = c.E;
    uint16_t *QK = g_fa_qk[k], *V = g_fa_v[k], *O = g_fa_o[k];
#if UC_CTRLPKT
    if (!g_uc_ht) {
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    const size_t qkb = (size_t)c.units * c.qkrow * 2, vb = (size_t)H * c.vrow * 2, ob = (size_t)c.units * c.orow * 2;
    for (int l = 0; l < c.launches; l++) {
        XTime lt0, lt1;
        XTime_GetTime(&lt0);
        if (l == 0)
            uc_sync_pre(QK, qkb, V, vb, O, ob);
        /* launch l = chunk l / (H/4) x head group l % (H/4): q/o rows 4l.., kv rows of the head group */
        const int lrc = uc_fa_launch(k, QK + (size_t)l * 4 * c.qkrow, V + (size_t)(l % (H / 4)) * 4 * c.vrow,
                                     O + (size_t)l * 4 * c.orow);
        if (l == c.launches - 1)
            uc_sync_post(O, ob);
        XTime_GetTime(&lt1);
        g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
        g_aie_launches++;
        if (lrc != 0)
            return lrc;
    }
    /* v5 O: o row pm, block h%4 of launch l; non-split = query block qb (pm = qb%4, l = (qb/4)*(H/4) + h/4),
     * split = (query block pm>>1, half pm&1) of launch h/4 */
    for (int h = 0; h < H; h++) {
        float *yh = Y + (size_t)h * hs;
        for (int ob = 0; ob < (c.split ? 4 : N / 32); ob++) {
            const int qb = c.split ? ob >> 1 : ob, hh = c.split ? ob & 1 : 0;
            const int l = c.split ? h / 4 : (qb / 4) * (H / 4) + h / 4, pm = c.split ? ob : qb % 4;
            const uint16_t *op = O + (size_t)(l * 4 + pm) * c.orow + (size_t)(h % 4) * E;
            float *y0 = yh + (size_t)(qb * 32) * rs + hh * dq;
            for (int d8 = 0; d8 < dq / 8; d8++)
                for (int r8 = 0; r8 < 4; r8++) {
#if UC_NEON
                    uint16x8_t t[8];
                    for (int dd = 0; dd < 8; dd++)
                        t[dd] = vld1q_u16(op + dd * 8);
                    fa_tr8(t);
                    for (int rr = 0; rr < 8; rr++) {
                        float *yd = y0 + (size_t)(r8 * 8 + rr) * rs + d8 * 8;
                        vst1q_f32(yd, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(t[rr]), 16)));
                        vst1q_f32(yd + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(t[rr]), 16)));
                    }
                    op += 64;
#else
                    for (int dd = 0; dd < 8; dd++) {
                        float *yc = y0 + (size_t)(r8 * 8) * rs + d8 * 8 + dd;
                        for (int rr = 0; rr < 8; rr++)
                            yc[(size_t)rr * rs] = fa_bf2f(op[rr]);
                        op += 8;
                    }
#endif
                }
        }
    }
    g_uc_macs += 2ULL * H * N * M * D;
    g_fa_sites++;
    return 0;
}
static int uc_fa(void *const *argv, int nargs, int role, int k) {
    if (nargs != 3 || k < 0 || k >= FA_NCLS)
        return -2;
    const fa_cls &c = g_fa_cls[k];
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[2];
    if (x->ndim != 3 || w->ndim != 3 || y->ndim != 3)
        return -2;
    const int H = c.H, N = c.N, M = c.M, D = c.D;
    if (fa_alloc(k) != 0)
        return -1;
    if (role == 0) { /* x = Q [H][N][D], w = K [H][M][D] */
        if ((int)x->shape[0] != H || (int)x->shape[1] != N || (int)x->shape[2] != D || (int)w->shape[0] != H ||
            (int)w->shape[1] != M || (int)w->shape[2] != D)
            return -3;
        fa_pack_k(c, (const float *)w->data, (size_t)M * D, D, 1.0f, g_fa_v[k]);
        fa_pack_q(c, (const float *)x->data, (size_t)N * D, D, 1.0f, g_fa_qk[k]);
        return 0;
    }
    /* role 1: x = P (unused), w = V^T [H][D][M], y = O [H][N][D] */
    if ((int)w->shape[0] != H || (int)w->shape[1] != D || (int)w->shape[2] != M || (int)y->shape[0] != H ||
        (int)y->shape[1] != N || (int)y->shape[2] != D)
        return -3;
    fa_pack_v(c, (const float *)w->data, (size_t)D * M, 0, 1, g_fa_v[k]);
    return fa_run_o(c, k, (float *)y->data, (size_t)N * D, D);
}
/* P4a keep-layout sites (keep_layout_rt.py): the packs run where the removed TVM head split ops ran, reading the
 * [rows][H*D] linear outputs (ld = row stride) -- role 2 = pack Q (x log2e x s), 3 = pack the centred K (x s),
 * 5 = pack V, 6 = launch + write O into the [N][H*D] buffer the next linear reads */
static int uc_fa_kl(int role, int k, const float *a, int lda, float *b, int ldb, float s) {
    if (k < 0 || k >= FA_NCLS)
        return -2;
    const fa_cls &c = g_fa_cls[k];
    if (fa_alloc(k) != 0)
        return -1;
    if (role == 2)
        fa_pack_q(c, a, c.D, lda, s, g_fa_qk[k]);
    else if (role == 3)
        fa_pack_k(c, a, c.D, lda, s, g_fa_v[k]);
    else if (role == 5)
        fa_pack_v(c, a, c.D, lda, 0, g_fa_v[k]);
    else if (role == 6)
        return fa_run_o(c, k, b, c.D, ldb);
    else
        return -2;
    return 0;
}

extern "C" int unet_aie_attn(int cid, void *args, int nargs, int role, int cls) {
    g_uc_calls++;
    const int rc = uc_fa((void *const *)args, nargs, role, cls);
    if (rc != 0) {
        printf("[unet_aie_attn] fa layer %d role %d cls %d rc=%d\n", cid, role, cls, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}
extern "C" int unet_aie_attn_kl(int cid, int role, int cls, const float *a, int lda, float *b, int ldb, float s) {
    if (role == 3 || role == 6)
        g_uc_calls++; /* one count per original QK / PV site */
    const int rc = uc_fa_kl(role, cls, a, lda, b, ldb, s);
    if (rc != 0) {
        printf("[unet_aie_attn_kl] fa layer %d role %d cls %d rc=%d\n", cid, role, cls, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

