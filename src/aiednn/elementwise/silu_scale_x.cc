/* aiednn elementwise/silu_scale_x.cc -- driver step of a "dense+silu" group (aiegraph.fold.dense_silu).
 *
 * The dense's packed weight was layout(W * s) every call, s a per-K scale (sigmoid(x) of the dense input
 * x, or a K-vector). dense(x, W*s) == dense(x*s, W): the weight is pre-packed once in weights.bin as bf16
 * [N/b][K][b] (bit-exact permutation of the bf16 W), this step only widens it into the dense's packed
 * weight buffer and computes the dense input x*s, which the dense step passes as its argument 0.
 *
 * params: WPACK packed weight tensor (the dense's weight argument), NK = N*K elements,
 *         X the dense input tensor, S the scale: "0" = sigmoid(x), else "&tensors[<s>]", KLEN = K.
 * generator token: WBF = the pre-packed bf16 weight in weights.bin.
 */
#include <math.h>
static float g_silu_x[4096] __attribute__((aligned(64)));
static DLTensor g_silu_x_t;
static void aiednn_silu_scale_x(const DLTensor *t, const DLTensor *s, int k) {
    const float *tv = (const float *)t->data;
    g_silu_x_t = *t;
    g_silu_x_t.data = g_silu_x;
    if (s) {
        const float *sv = (const float *)s->data;
        for (int i = 0; i < k; i++) g_silu_x[i] = tv[i] * sv[i];
    } else {
        for (int i = 0; i < k; i++) g_silu_x[i] = tv[i] * (1.0f / (1.0f + expf(0.0f - tv[i])));
    }
}
//@@ step
graph_bf16_expand((float *)tensors[@WPACK@].data, @WBF@, @NK@ULL);   /* pre-packed [N/b][K][b] */
aiednn_silu_scale_x(&tensors[@X@], @S@, @KLEN@);
