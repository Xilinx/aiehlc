
#include "aie_user_kernel.h"
extern "C" int unet_aie_attn_kl(int cid, int role, int cls, const float *a, int lda, float *b, int ldb, float s);

static int fa_step_buf_ok(const aiek_buf *b) {
    return b && b->data && b->code == AIEK_FLOAT && b->bits == 32 && b->lanes == 1 && b->ndim >= 2 &&
           b->ld >= b->shape[b->ndim - 1];
}

AIEK_STEP(fa_step_pack_q, reads(q:float32), writes(0), consts(cid, cls, s:float));
AIEK_STEP_ENTRY(fa_step_pack_q) {
    (void)out;
    if (!fa_step_buf_ok(&in[0]) || c->n != 3)
        return AIEK_ERR_ARGS;
    return unet_aie_attn_kl((int)AIEK_CI(c, 0), 2, (int)AIEK_CI(c, 1), (const float *)in[0].data, (int)in[0].ld, 0,
                            0, AIEK_CF(c, 2));
}

AIEK_STEP(fa_step_pack_k, reads(k:float32), writes(0), consts(cid, cls, s:float));
AIEK_STEP_ENTRY(fa_step_pack_k) {
    (void)out;
    if (!fa_step_buf_ok(&in[0]) || c->n != 3)
        return AIEK_ERR_ARGS;
    return unet_aie_attn_kl((int)AIEK_CI(c, 0), 3, (int)AIEK_CI(c, 1), (const float *)in[0].data, (int)in[0].ld, 0,
                            0, AIEK_CF(c, 2));
}

AIEK_STEP(fa_step_pack_v, reads(v:float32), writes(0), consts(cid, cls));
AIEK_STEP_ENTRY(fa_step_pack_v) {
    (void)out;
    if (!fa_step_buf_ok(&in[0]) || c->n != 2)
        return AIEK_ERR_ARGS;
    return unet_aie_attn_kl((int)AIEK_CI(c, 0), 5, (int)AIEK_CI(c, 1), (const float *)in[0].data, (int)in[0].ld, 0,
                            0, 1.0f);
}

AIEK_STEP(fa_step_run_o, reads(0), writes(o:float32), consts(cid, cls));
AIEK_STEP_ENTRY(fa_step_run_o) {
    (void)in;
    if (!fa_step_buf_ok(&out[0]) || c->n != 2)
        return AIEK_ERR_ARGS;
    return unet_aie_attn_kl((int)AIEK_CI(c, 0), 6, (int)AIEK_CI(c, 1), 0, 0, (float *)out[0].data, (int)out[0].ld,
                            1.0f);
}
