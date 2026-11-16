/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn -- a primitive-based deep-learning operator API over the aiehlc kernel library.
 *
 *   aiednn_memory_desc_t a, b, c;                    // shape + data type
 *   aiednn_memory_desc_init(&a, 2, (int64_t[]){M, K}, aiednn_s8);
 *   ...
 *   aiednn_primitive_t mm;
 *   aiednn_matmul_primitive_create(&mm, &a, &b, &c); // dispatch happens HERE
 *   printf("%s\n", aiednn_primitive_impl_info(mm));  // "aie:matmul_s8"
 *   aiednn_primitive_execute(mm, A, B, C);           // or _execute_args(...)
 *   aiednn_primitive_destroy(mm);
 *
 * Tensors are dense row-major (NCHW for the spatial ops), batch included.
 * Dispatch: the most specific accepting implementation wins (most fixed dims in its
 * AIEK_SPEC), composites over AIE kernels next, CPU `ref:` last. See doc/design/aiednn.md.
 ******************************************************************************/
#ifndef AIEDNN_H
#define AIEDNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AIEDNN_MAX_NDIMS 6
#define AIEDNN_MAX_SRCS 64 /* concat / sum inputs */

typedef enum {
    aiednn_success = 0,
    aiednn_unimplemented = 1,     /* no implementation accepts the problem */
    aiednn_invalid_arguments = 2, /* malformed descriptor / null pointer / missing argument */
    aiednn_runtime_error = 3,     /* the implementation failed while running */
    aiednn_out_of_memory = 4,
} aiednn_status_t;

typedef enum {
    aiednn_data_type_undef = 0,
    aiednn_s8 = 1,
    aiednn_u8 = 2,
    aiednn_s32 = 3,
    aiednn_f32 = 4,
    aiednn_bf16 = 5,    /* bfloat16 */
    aiednn_f16 = 6,     /* IEEE binary16 */
    aiednn_f8_e5m2 = 7, /* OCP FP8 E5M2 (inf, NaN) */
    aiednn_f8_e4m3 = 8, /* OCP FP8 E4M3 "fn" (no inf, max 448, NaN 0x7f) */
    aiednn_s4 = 9,      /* two per byte, element 2i in the low nibble */
    aiednn_u4 = 10,
    aiednn_e8m0 = 11,    /* OCP MX scale: 2^(b - 127), 0xff NaN; group weight scales (attr.wei_scales) */
    aiednn_f4_e2m1 = 12, /* OCP MX FP4: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (+ sign), two per byte like s4 */
} aiednn_data_type_t;
/* Floating types compute in double and round once on store (to nearest even, via f32); integer
 * destinations round to nearest even and saturate. */

typedef enum {
    aiednn_matmul = 1,
    aiednn_convolution = 2,
    aiednn_deconvolution = 3,
    aiednn_inner_product = 4,
    aiednn_eltwise = 5,
    aiednn_binary = 6,
    aiednn_reduction = 7,
    aiednn_pooling = 8,
    aiednn_resampling = 9,
    aiednn_softmax = 10,
    aiednn_batch_normalization = 11,
    aiednn_layer_normalization = 12,
    aiednn_group_normalization = 13,
    aiednn_lrn = 14,
    aiednn_prelu = 15,
    aiednn_concat = 16,
    aiednn_sum = 17,
    aiednn_reorder = 18,
    aiednn_shuffle = 19,
    aiednn_sdpa = 20,
    aiednn_rnn = 21,
    aiednn_grouped_matmul = 22,
    aiednn_dropout = 23,
    aiednn_grid_sample = 24,
    aiednn_ctc_loss = 25,
    aiednn_block_quantize = 26,   /* block-scale (MX) quantize / dequantize */
    aiednn_sparse_matmul = 27,    /* CSR src x dense weights */
    aiednn_slice = 28,            /* strided sub-tensor copy */
    aiednn_gated_mlp = 29,        /* (act(x Wg) * (x Wu)) Wd */
    aiednn_rope = 30,             /* rotary position embedding */
    aiednn_gen_index = 31,        /* the index along an axis */
    aiednn_dynamic_quantize = 32, /* quantize / dequantize with scales / zero points given at execute */
    aiednn_linear_attention = 33, /* gated delta rule: Gated DeltaNet, KDA, gated linear attention */
} aiednn_primitive_kind_t;

typedef enum {
    aiednn_alg_undef = 0,
    /* eltwise: dst = f(src; alpha, beta) -- pointwise unary */
    aiednn_eltwise_relu = 1,  /* x > 0 ? x : alpha * x (alpha 0: ReLU, else leaky) */
    aiednn_eltwise_elu,       /* x > 0 ? x : alpha * (e^x - 1) */
    aiednn_eltwise_gelu_erf,  /* 0.5 x (1 + erf(x / sqrt 2)) */
    aiednn_eltwise_gelu_tanh, /* 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) */
    aiednn_eltwise_logistic,  /* sigmoid */
    aiednn_eltwise_tanh,
    aiednn_eltwise_swish,       /* x * sigmoid(alpha x) (alpha 1: SiLU) */
    aiednn_eltwise_soft_relu,   /* softplus: log(1 + e^(alpha x)) / alpha */
    aiednn_eltwise_mish,        /* x tanh(softplus(x)) */
    aiednn_eltwise_hardswish,   /* x * clip(alpha x + beta, 0, 1) */
    aiednn_eltwise_hardsigmoid, /* clip(alpha x + beta, 0, 1) */
    aiednn_eltwise_clip,        /* clip(x, alpha, beta) */
    aiednn_eltwise_linear,      /* alpha x + beta */
    aiednn_eltwise_abs,
    aiednn_eltwise_exp,
    aiednn_eltwise_log,
    aiednn_eltwise_sqrt,
    aiednn_eltwise_square,
    aiednn_eltwise_pow,   /* alpha * x^beta */
    aiednn_eltwise_round, /* nearest, ties to even */
    aiednn_eltwise_neg,
    aiednn_eltwise_reciprocal,
    aiednn_eltwise_rsqrt,
    aiednn_eltwise_erf,
    aiednn_eltwise_floor,
    aiednn_eltwise_ceil,
    aiednn_eltwise_sin,
    aiednn_eltwise_cos,
    aiednn_eltwise_tan,
    aiednn_eltwise_identity,
    aiednn_eltwise_logical_not, /* x == 0 */
    aiednn_eltwise_sign,
    aiednn_eltwise_softsign,    /* x / (1 + |x|) */
    aiednn_eltwise_log_sigmoid, /* log(sigmoid(x)) */
    aiednn_eltwise_selu,        /* 1.0507 * (x > 0 ? x : 1.6733 (e^x - 1)) */
    /* binary: dst = src0 op src1 (numpy broadcasting, same rank); select: src2 ? src0 : src1 */
    aiednn_binary_add = 64,
    aiednn_binary_sub,
    aiednn_binary_mul,
    aiednn_binary_div,
    aiednn_binary_max,
    aiednn_binary_min,
    aiednn_binary_pow,
    aiednn_binary_eq, /* comparisons produce 1 / 0 in dst's data type */
    aiednn_binary_ne,
    aiednn_binary_gt,
    aiednn_binary_ge,
    aiednn_binary_lt,
    aiednn_binary_le,
    aiednn_binary_select,
    aiednn_binary_fmod,        /* C fmod: the remainder with the sign of src0 */
    aiednn_binary_logical_and, /* nonzero = true; 1 / 0 out */
    aiednn_binary_logical_or,
    aiednn_binary_logical_xor,
    aiednn_binary_add_square, /* src0 + src1^2 */
    /* reduction over every axis where dst's dim is 1 and src's is not */
    aiednn_reduction_sum = 96,
    aiednn_reduction_mean,
    aiednn_reduction_max,
    aiednn_reduction_min,
    aiednn_reduction_mul,
    aiednn_reduction_amax,                /* max |x| */
    aiednn_reduction_norm_lp_sum,         /* (sum |x|^p + eps)^(1/p) */
    aiednn_reduction_norm_lp_power_p_sum, /* sum |x|^p + eps */
    aiednn_reduction_norm_lp_max,         /* max(sum |x|^p, eps)^(1/p) */
    aiednn_reduction_norm_lp_power_p_max, /* max(sum |x|^p, eps) */
    aiednn_reduction_mul_no_zeros,        /* product of the nonzero elements */
    /* pooling (2D, NCHW) */
    aiednn_pooling_max = 128,
    aiednn_pooling_avg_include_padding,
    aiednn_pooling_avg_exclude_padding,
    /* resampling (spatial dims after N, C; half-pixel centers) */
    aiednn_resampling_nearest = 144,
    aiednn_resampling_linear,
    /* softmax */
    aiednn_softmax_accurate = 160,
    aiednn_softmax_log,
    aiednn_softmax_accurate_inf_as_zero, /* accurate; a row of only -inf gives 0, not NaN */
    /* lrn */
    aiednn_lrn_across_channels = 176,
    aiednn_lrn_within_channel,
    /* rnn cells */
    aiednn_rnn_vanilla_relu = 192, /* h = relu(W x + U h + b), leaky with alpha */
    aiednn_rnn_vanilla_tanh,
    aiednn_rnn_vanilla_logistic,
    aiednn_rnn_lstm,      /* gates i, f, c~, o */
    aiednn_rnn_gru,       /* gates u, r, o: o = tanh(W_o x + U_o (r * h) + b_o) */
    aiednn_rnn_lbr_gru,   /* linear-before-reset GRU: o = tanh(W_o x + b_o + r * (U_o h + b_u)) (PyTorch's GRU) */
    aiednn_rnn_augru,     /* attention-update GRU: the GRU with u' = (1 - a_t) u (aiednn_augru_forward_create) */
    aiednn_rnn_lbr_augru, /* the linear-before-reset GRU with the same attention update */
} aiednn_alg_kind_t;

/* normalization flags */
#define AIEDNN_USE_SCALE 0x1u
#define AIEDNN_USE_SHIFT 0x2u
#define AIEDNN_FUSE_NORM_RELU 0x4u
#define AIEDNN_RMS_NORM 0x8u /* layer normalization without mean subtraction / shift */

/* execute argument tags */
#define AIEDNN_ARG_SRC 1
#define AIEDNN_ARG_SRC_1 2
#define AIEDNN_ARG_SRC_2 3
#define AIEDNN_ARG_WEIGHTS 4
#define AIEDNN_ARG_BIAS 5
#define AIEDNN_ARG_DST 6
#define AIEDNN_ARG_MEAN 7
#define AIEDNN_ARG_VARIANCE 8
#define AIEDNN_ARG_SCALE 9
#define AIEDNN_ARG_SHIFT 10
#define AIEDNN_ARG_ATTN_MASK 11
#define AIEDNN_ARG_QUERY AIEDNN_ARG_SRC
#define AIEDNN_ARG_KEY AIEDNN_ARG_SRC_1
#define AIEDNN_ARG_VALUE AIEDNN_ARG_SRC_2
#define AIEDNN_ARG_SRC_ITER 12     /* rnn: initial hidden state (src = src_layer) */
#define AIEDNN_ARG_SRC_ITER_C 13   /* rnn: initial LSTM cell state */
#define AIEDNN_ARG_WEIGHTS_ITER 14 /* rnn: recurrent weights (weights = weights_layer) */
#define AIEDNN_ARG_DST_ITER 15     /* rnn: final hidden state (dst = dst_layer) */
#define AIEDNN_ARG_DST_ITER_C 16   /* rnn: final LSTM cell state */
/* training: gradients (backward primitives) and the statistics a training forward writes */
#define AIEDNN_ARG_DIFF_SRC 20
#define AIEDNN_ARG_DIFF_WEIGHTS 21
#define AIEDNN_ARG_DIFF_BIAS 22
#define AIEDNN_ARG_DIFF_DST 23
#define AIEDNN_ARG_DIFF_SCALE 24
#define AIEDNN_ARG_DIFF_SHIFT 25
#define AIEDNN_ARG_GROUP_SIZES 26   /* grouped matmul: s32 [E], rows of src per group, in order */
#define AIEDNN_ARG_MASK 27          /* dropout: u8, 1 = kept */
#define AIEDNN_ARG_GRID 28          /* grid sample: [N, Ho, Wo, 2] (x, y) in [-1, 1]; affine grid: dst */
#define AIEDNN_ARG_LABELS 29        /* ctc: s32 [N, Lmax] */
#define AIEDNN_ARG_LABEL_LENGTHS 30 /* ctc: s32 [N] */
#define AIEDNN_ARG_INPUT_LENGTHS 31 /* ctc: s32 [N] */
#define AIEDNN_ARG_DIFF_SRC_ITER 32 /* rnn backward */
#define AIEDNN_ARG_DIFF_SRC_ITER_C 33
#define AIEDNN_ARG_DIFF_WEIGHTS_ITER 34
#define AIEDNN_ARG_DIFF_DST_ITER 35
#define AIEDNN_ARG_DIFF_DST_ITER_C 36             /* grouped matmul: s32 [E], rows of src per group, in order */
#define AIEDNN_ARG_DIFF_QUERY AIEDNN_ARG_DIFF_SRC /* sdpa backward: dQ, dK, dV */
#define AIEDNN_ARG_DIFF_KEY 37
#define AIEDNN_ARG_DIFF_VALUE 38
#define AIEDNN_ARG_AUGRU_ATTENTION 39 /* augru: the per-step, per-sequence attention a [T, N] */
#define AIEDNN_ARG_SEQ_LENS_Q 40      /* varlen sdpa: s32 [B] valid query rows per sequence */
#define AIEDNN_ARG_SEQ_LENS_KV 41     /* varlen sdpa: s32 [B] valid keys per sequence */
#define AIEDNN_ARG_PAGE_TABLE 42      /* paged sdpa: s32 [B, max pages], page indices into K / V */
#define AIEDNN_ARG_SRC_INDICES 43     /* sparse matmul: CSR column index of each src value, s32 [nnz] */
#define AIEDNN_ARG_SRC_POINTERS 44    /* sparse matmul: CSR row pointers, s32 [M + 1] */
#define AIEDNN_ARG_ATTR_SCALE 45   /* sdpa (fwd / varlen / bwd): f32 [1] score scale at execute; overrides create's */
#define AIEDNN_ARG_WEIGHTS_UP 46   /* gated MLP: W_up [D, F] (WEIGHTS = W_gate) */
#define AIEDNN_ARG_WEIGHTS_DOWN 47 /* gated MLP: W_down [F, D] */
#define AIEDNN_ARG_ROPE_COS 48     /* rope: cos, broadcast to src */
#define AIEDNN_ARG_ROPE_SIN 49     /* rope: sin, broadcast to src */
#define AIEDNN_ARG_SCALES 50       /* dynamic quantize: f32 scales [1] or [dims[axis]] at execute */
#define AIEDNN_ARG_ZERO_POINTS 51  /* dynamic quantize: s32 / s8 / u8 zero points, the scales' shape (optional) */
#define AIEDNN_ARG_GATE 52         /* linear attention: log gates */
#define AIEDNN_ARG_BETA 53         /* linear attention: delta-rule step sizes */
#define AIEDNN_ARG_STATE 54        /* linear attention: initial state [B, H, Dk, Dv] */
#define AIEDNN_ARG_STATE_OUT 55    /* linear attention: final state */
#define AIEDNN_ARG_MULTIPLE_SRC 64 /* concat / sum input i is AIEDNN_ARG_MULTIPLE_SRC + i */

/* Element layout; dims are always logical ([N, C, H, W]). dense = row-major over dims;
 * nhwc = 4-D only, the bytes of a dense [N, H, W, C]. */
typedef enum {
    aiednn_format_dense = 0,
    aiednn_format_nhwc = 1,
} aiednn_format_t;

typedef struct {
    int ndims;
    int64_t dims[AIEDNN_MAX_NDIMS];
    aiednn_data_type_t data_type;
    aiednn_format_t format; /* aiednn_memory_desc_init sets dense */
} aiednn_memory_desc_t;

typedef struct {
    int arg;    /* AIEDNN_ARG_* */
    void *data; /* dense buffer matching that argument's descriptor */
} aiednn_exec_arg_t;

/* 2D convolution / deconvolution. dilates are 0-based: 0 = dense. */
typedef struct {
    int64_t strides[2];
    int64_t padding_l[2];
    int64_t padding_r[2];
    int64_t dilates[2];
    int64_t groups; /* 1: dense; = channels: depthwise */
} aiednn_conv_params_t;

typedef struct {
    int64_t kernel[2];
    int64_t strides[2];
    int64_t padding_l[2];
    int64_t padding_r[2];
    int64_t dilates[2];
} aiednn_pool_params_t;

/* ── fused epilogue (primitive attributes: scales, post-ops, dst scale) ──
 * For convolution, deconvolution, matmul and inner product, the accumulator acc of
 * output element (.., channel, ..) becomes, in this order:
 *   v = acc * scales[channel or 0]          (n_scales 0: none)
 *   v = v + bias[channel]                   (bias s32 or f32)
 *   v = post_op_k(v)                        eltwise(alg, alpha, beta), or
 *                                           binary(alg, v, src1[broadcast to dst])
 *   dst = v                                 f32 dst, or integer dst:
 *         saturate(round_even(v / dst_scale) + dst_zero_point)
 * The channel is dim 1 for (de)conv and
 * the last dim for matmul / inner product. A binary post-op's src1 is passed at
 * execute as AIEDNN_ARG_ATTR_POST_OP_SRC(k). */
#define AIEDNN_MAX_POST_OPS 4
#define AIEDNN_ARG_ATTR_POST_OP_SRC(k) (128 + (k))
#define AIEDNN_ARG_ATTR_WEI_SCALES 136      /* attr.wei_scales */
#define AIEDNN_ARG_ATTR_WEI_ZERO_POINTS 137 /* attr.wei_zero_points */

typedef enum {
    aiednn_post_op_eltwise = 1,
    aiednn_post_op_binary = 2,
    /* requantize to q_data_type (q_scale / q_zero_point), dequantize, then binary with the dequantized src1
     * (add / sub / mul / max / min; src1 scale alpha, zero point beta). Later post-ops are eltwise. Bit-identical
     * to the int8-dst primitive then aiednn_binary_create_attr. GEMM family; at most one per primitive. */
    aiednn_post_op_requant_binary = 3,
    /* requantize to q_data_type, dequantize, then eltwise alg (alpha, beta); later post-ops are eltwise.
     * Bit-identical to the int8-dst primitive then aiednn_eltwise_forward_create_attr. GEMM family; at most one
     * requant post-op. */
    aiednn_post_op_requant_eltwise = 4,
} aiednn_post_op_kind_t;

typedef struct {
    aiednn_post_op_kind_t kind;
    aiednn_alg_kind_t alg;     /* eltwise_* or binary_* (not select) */
    float alpha, beta;         /* eltwise parameters; binary with an s8 / u8 src1: its scale (alpha,
                                  0 = the stored codes) and zero point (beta), real = (src1 - beta) * alpha */
    aiednn_memory_desc_t src1; /* binary: broadcastable to dst */
    float q_scale;             /* requant_binary: the intermediate quantization */
    int32_t q_zero_point;
    aiednn_data_type_t q_data_type; /* s8 / u8 */
} aiednn_post_op_t;

typedef struct {
    int64_t n_scales;    /* 0: none, 1: per tensor, output channels: per channel */
    const float *scales; /* copied at create */
    float dst_scale;     /* integer dst requantize; 0 means 1 */
    int32_t dst_zero_point;
    int n_post_ops;
    aiednn_post_op_t post_ops[AIEDNN_MAX_POST_OPS];
    /* GEMM family (matmul, inner product, (de)convolution, grouped matmul), per tensor: the integer
     * src / weights are asymmetric, acc = sum (src - src_zero_point) * (weights - wei_zero_point);
     * a padded src element contributes 0. Nonzero zero points run on ref:. */
    int32_t src_zero_point;
    int32_t wei_zero_point;
    /* matmul / inner product weight-only quantization (grouped decompression, MX block scaling): the
     * weights dequantize per (K group, output column n), w = (code - zp[k / g][n]) * s[k / g][n] with
     * g = K / wei_scales.dims[0]. ndims 0: none. wei_scales: f32 / bf16 / f16 / e8m0, [K / g, N]
     * (shared by every batch), passed at execute as AIEDNN_ARG_ATTR_WEI_SCALES; wei_zero_points
     * (optional, needs wei_scales): s8 / u8 / s4 / u4 / s32, the same shape, as
     * AIEDNN_ARG_ATTR_WEI_ZERO_POINTS. Floating src, or s8 / u8 src with integer weights (floating dst);
     * weights any type but s32. */
    aiednn_memory_desc_t wei_scales;
    aiednn_memory_desc_t wei_zero_points;
    /* binary (aiednn_binary_create_attr), per tensor: an s8 / u8 source i is dequantized first,
     * real_i = (src_i - src_zero_points[i]) * src_scales[i] (src_scales[i] 0: the codes as is) */
    float src_scales[2];
    int32_t src_zero_points[2];
    /* the same for sources 2 and 3 (sdpa: src_scales[0] Q, [1] K, src_scales_23[0] V). Layer norm, softmax and
     * sdpa dequantize an s8 / u8 source with a scale > 0 as it is read: f32((code - zero point) * scale). */
    float src_scales_23[2];
    int32_t src_zero_points_23[2];
    /* AIEDNN_ATTR_* opt-ins (0: every implementation dispatch may pick is exact w.r.t. ref:) */
    unsigned flags;
} aiednn_attr_t;

/* attr.flags: an AIE implementation within one quantization code of ref: may serve the primitive. */
#define AIEDNN_ATTR_TOLERANCE_1CODE 1u

typedef struct aiednn_primitive *aiednn_primitive_t;

aiednn_status_t aiednn_memory_desc_init(aiednn_memory_desc_t *md, int ndims, const int64_t *dims,
                                        aiednn_data_type_t data_type);
/* nhwc: 4-D only; every primitive accepts it on any 4-D operand. A reorder between a dense and an nhwc
 * descriptor converts. */
aiednn_status_t aiednn_memory_desc_set_format(aiednn_memory_desc_t *md, aiednn_format_t format);
/* bytes of a buffer for md (4-bit types packed two per byte) */
size_t aiednn_memory_desc_get_size(const aiednn_memory_desc_t *md);

/* ── compute primitives ────────────────────────────────────────────────── */

/* src [N,C,H,W], weights [O,C/groups,KH,KW], bias [O] or NULL, dst [N,O,Ho,Wo].
 * s8|u8 x s8 -> s32 (bias s32) or f32 -> f32. Zero padding. */
aiednn_status_t aiednn_convolution_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                  const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *bias,
                                                  const aiednn_memory_desc_t *dst, const aiednn_conv_params_t *cp);

/* Transposed convolution (the data-gradient of convolution, as a forward op): src [N,C,H,W],
 * weights [O,C/groups,KH,KW], dst [N,O,Ho,Wo], Ho = (H-1)*s - pl - pr + (KH-1)*(d+1) + 1. */
aiednn_status_t aiednn_deconvolution_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                    const aiednn_memory_desc_t *weights,
                                                    const aiednn_memory_desc_t *bias, const aiednn_memory_desc_t *dst,
                                                    const aiednn_conv_params_t *cp);

/* src [..., M, K], weights [..., K, N] (or [..., N, K] when weights_transposed), bias
 * broadcastable to dst or NULL, dst [..., M, N]; batch dims broadcast. s8|u8 x s8 -> s32 /
 * s8 (saturated), f32 -> f32. */
aiednn_status_t aiednn_matmul_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                             const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *bias,
                                             const aiednn_memory_desc_t *dst, int weights_transposed);

/* src [N, IC...] (flattened), weights [OC, IC...], bias [OC] or NULL, dst [N, OC]. */
aiednn_status_t aiednn_inner_product_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                    const aiednn_memory_desc_t *weights,
                                                    const aiednn_memory_desc_t *bias, const aiednn_memory_desc_t *dst);

/* Q [B,H,Sq,D], K [B,H,Skv,D], V [B,H,Skv,Dv], mask broadcastable to [B,H,Sq,Skv] (added
 * before softmax) or NULL, dst [B,H,Sq,Dv]; f32. causal: key j > query i + (Skv - Sq)
 * (bottom-right aligned) is masked out. */
/* sdpa's causal argument, bits above 0: Q, K, V (resp. dst) are [B, S, H, D] in memory -- the head
 * split / merge permutes around attention folded in (--aiednn-fold-head-permute) */
#define AIEDNN_SDPA_QKV_BSHD 2
#define AIEDNN_SDPA_DST_BSHD 4
/* Q, K and V are slices of one packed projection output [B, S, 3, H, D]: pass that buffer as each of QUERY,
 * KEY and VALUE; the descriptors stay the logical [B, H, S, D] (same S, H, D for all three). */
#define AIEDNN_SDPA_QKV_PACKED 8
/* the same with the quantizing reorder after it folded in (--aiednn-fold-norm-quant): dst s8 / u8,
 * attr with dst_scale / dst_zero_point only; bit-identical to the f32 sdpa + reorder */
aiednn_status_t aiednn_sdpa_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                                const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                                const aiednn_memory_desc_t *mask, const aiednn_memory_desc_t *dst,
                                                float scale, int causal, const aiednn_attr_t *attr);
aiednn_status_t aiednn_sdpa_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                           const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                           const aiednn_memory_desc_t *mask, const aiednn_memory_desc_t *dst,
                                           float scale, int causal);
/* Block-scale (OCP microscaling) quantize along the last axis: src [..., K] f32 / bf16 / f16 -> dst [..., K]
 * f8_e4m3 / f8_e5m2 / f4_e2m1 and scales [..., K / block] e8m0 (block 32 = MXFP8 / MXFP4): shared exponent
 * floor(log2 max|x|) - emax, elements rounded to nearest even and saturated. Arguments SRC, DST, SCALE. */
aiednn_status_t aiednn_block_quantize_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                             const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *scales,
                                             int64_t block);
/* its inverse: src (MX elements) + SCALE -> dst f32 / bf16 / f16 */
aiednn_status_t aiednn_block_dequantize_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                               const aiednn_memory_desc_t *scales, const aiednn_memory_desc_t *dst,
                                               int64_t block);
/* sdpa backward (training), f32: QUERY, KEY, VALUE, [ATTN_MASK], DIFF_DST -> DIFF_QUERY, DIFF_KEY, DIFF_VALUE.
 * The forward is recomputed (softmax of scale Q K^T + mask, bottom-right causal); K / V may carry Hk heads
 * (GQA / MQA, Hk dividing H: dK / dV sum over each group). causal bit AIEDNN_SDPA_QKV_BSHD: every tensor
 * [B, S, H, D] in memory. */
aiednn_status_t aiednn_sdpa_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                            const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                            const aiednn_memory_desc_t *mask, const aiednn_memory_desc_t *diff_dst,
                                            const aiednn_memory_desc_t *diff_q, const aiednn_memory_desc_t *diff_k,
                                            const aiednn_memory_desc_t *diff_v, float scale, int causal);

/* The same four with a fused epilogue (attr NULL = none). With an attr, the int8
 * family may produce s8 / u8 / f32 directly (types: s8|u8 x s8, bias s32|f32,
 * dst s32|s8|u8|f32). */
aiednn_status_t aiednn_convolution_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                       const aiednn_memory_desc_t *weights,
                                                       const aiednn_memory_desc_t *bias,
                                                       const aiednn_memory_desc_t *dst, const aiednn_conv_params_t *cp,
                                                       const aiednn_attr_t *attr);
aiednn_status_t aiednn_deconvolution_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                         const aiednn_memory_desc_t *weights,
                                                         const aiednn_memory_desc_t *bias,
                                                         const aiednn_memory_desc_t *dst,
                                                         const aiednn_conv_params_t *cp, const aiednn_attr_t *attr);
aiednn_status_t aiednn_matmul_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                  const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *bias,
                                                  const aiednn_memory_desc_t *dst, int weights_transposed,
                                                  const aiednn_attr_t *attr);
aiednn_status_t aiednn_inner_product_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                         const aiednn_memory_desc_t *weights,
                                                         const aiednn_memory_desc_t *bias,
                                                         const aiednn_memory_desc_t *dst, const aiednn_attr_t *attr);

/* ── pointwise / reduction ─────────────────────────────────────────────── */

aiednn_status_t aiednn_eltwise_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, float alpha, float beta,
                                              const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst);
/* Quantized eltwise: dst = q(post(alg(real))), real = f32((src - attr.src_zero_points[0]) * attr.src_scales[0])
 * for an s8 / u8 src (scale 0 or f32 src: as is), alg and attr.post_ops (eltwise only) in f32, and
 * q(f) = saturate(round_even(f / attr.dst_scale) + attr.dst_zero_point) for an s8 / u8 dst (f32 dst: f).
 * Bit-identical to dequantize -> eltwise -> [eltwise ...] -> quantize (aiednn_reorder_create). */
aiednn_status_t aiednn_eltwise_forward_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg, float alpha,
                                                   float beta, const aiednn_memory_desc_t *src,
                                                   const aiednn_memory_desc_t *dst, const aiednn_attr_t *attr);

/* src1 (and src2, the condition, for select; NULL otherwise) broadcast to dst. */
/* Quantized binary: dst = q(post(real0 op real1)), real_i = f32((src_i - attr.src_zero_points[i]) *
 * attr.src_scales[i]) for an s8 / u8 source (f32 as is), op in f32, attr.post_ops eltwise in f32, and
 * q(f) = saturate(round_even(f / attr.dst_scale) + attr.dst_zero_point) for an s8 / u8 dst (f32 dst: f).
 * Bit-identical to the unfused chain dequantize -> binary -> eltwise -> quantize (aiednn_reorder_create).
 * alg add / sub / mul / max / min; sources broadcast as aiednn_binary_create. */
aiednn_status_t aiednn_binary_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                          const aiednn_memory_desc_t *src0, const aiednn_memory_desc_t *src1,
                                          const aiednn_memory_desc_t *dst, const aiednn_attr_t *attr);
aiednn_status_t aiednn_binary_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, const aiednn_memory_desc_t *src0,
                                     const aiednn_memory_desc_t *src1, const aiednn_memory_desc_t *src2,
                                     const aiednn_memory_desc_t *dst);

/* src [N, C, ...], weights broadcastable to src: dst = x > 0 ? x : w * x */
aiednn_status_t aiednn_prelu_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                            const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *dst);

aiednn_status_t aiednn_reduction_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, float pnorm, float eps,
                                        const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst);

/* ── spatial ───────────────────────────────────────────────────────────── */

aiednn_status_t aiednn_pooling_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                              const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                              const aiednn_pool_params_t *pp);

/* ── grouped matmul (mixture-of-experts) ──────────────────────────────────
 * src [T, K] holds the tokens grouped by expert (group e's rows follow group e - 1's); weights
 * [E, K, N] ([E, N, K] when transposed), bias [E, N] or NULL, dst [T, N]. The rows per group come at
 * execute, AIEDNN_ARG_GROUP_SIZES (s32 [E], sum <= T; rows past the sum are not written). Each
 * group is a matmul dispatched on its own shape. attr: output scales and eltwise post-ops (no binary
 * post-op), as matmul. */
aiednn_status_t aiednn_grouped_matmul_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                             const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *bias,
                                             const aiednn_memory_desc_t *dst, int weights_transposed,
                                             const aiednn_attr_t *attr);

/* ── dropout (training) ────────────────────────────────────────────────────
 * dst = mask ? src / (1 - p) : 0, mask (u8, written) from a counter-based hash of (seed, element):
 * reproducible for a seed, independent of threads; backward: diff_src = mask ? diff_dst / (1 - p) : 0. */
aiednn_status_t aiednn_dropout_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                              const aiednn_memory_desc_t *dst, float probability, uint64_t seed);
aiednn_status_t aiednn_dropout_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                               const aiednn_memory_desc_t *diff_dst, float probability);

/* ── spatial transformer ───────────────────────────────────────────────────
 * affine grid: theta [N, 2, 3] (SRC) -> grid [N, Ho, Wo, 2] (DST); grid sample: src [N, C, H, W] and
 * GRID [N, Ho, Wo, 2] -> dst [N, C, Ho, Wo], bilinear or nearest, zeros outside (PyTorch's
 * affine_grid / grid_sample with padding_mode="zeros"); align_corners as in PyTorch. Backward:
 * GRID, SRC, DIFF_DST -> DIFF_SRC and the grid's gradient (DIFF_WEIGHTS, [N, Ho, Wo, 2]; bilinear). */
#define AIEDNN_GRID_SAMPLE_NEAREST 1
#define AIEDNN_GRID_ALIGN_CORNERS 2
aiednn_status_t aiednn_affine_grid_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *theta,
                                          const aiednn_memory_desc_t *grid, unsigned flags);
aiednn_status_t aiednn_grid_sample_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                  const aiednn_memory_desc_t *grid, const aiednn_memory_desc_t *dst,
                                                  unsigned flags);
aiednn_status_t aiednn_grid_sample_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                   const aiednn_memory_desc_t *diff_grid,
                                                   const aiednn_memory_desc_t *diff_dst,
                                                   const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *grid,
                                                   unsigned flags);

/* ── linear attention (gated delta rule) ────────────────────────────────────
 * Per (b, h) a state h [Dk][Dv] (STATE, or 0); per token t, with a_t = exp(gate_t) (1 without a gate):
 *   h <- Diag(a_t) h;  u_t = beta_t (v_t - h^T k_t);  h <- h + k_t u_t^T;  dst_t = scale h^T q_t
 * Gated DeltaNet: gate [B, H, T]; KDA: gate [B, H, T, Dk] with AIEDNN_LA_GATE_PER_CHANNEL; AIEDNN_LA_NO_DELTA:
 * u_t = v_t, no beta (gated linear attention, RetNet with a constant gate). q, k [B, H, T, Dk]; v, dst
 * [B, H, T, Dv]; beta [B, H, T]; STATE / STATE_OUT optional [B, H, Dk, Dv]. Keys are used as given. */
#define AIEDNN_LA_GATE_PER_CHANNEL 1
#define AIEDNN_LA_NO_DELTA 2
aiednn_status_t aiednn_linear_attention_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                               const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                               const aiednn_memory_desc_t *gate, const aiednn_memory_desc_t *beta,
                                               const aiednn_memory_desc_t *state_in, const aiednn_memory_desc_t *dst,
                                               const aiednn_memory_desc_t *state_out, float scale, unsigned flags);

/* ── CTC loss ──────────────────────────────────────────────────────────────
 * SRC: log-probabilities [T, N, C] (log-softmax over C, blank = class 0), LABELS s32 [N, Lmax],
 * INPUT_LENGTHS / LABEL_LENGTHS s32 [N] -> DST: the negative log-likelihood per sequence [N] and,
 * if diff_src is given, its gradient w.r.t. SRC (DIFF_SRC [T, N, C]) -- PyTorch's ctc_loss with
 * reduction="none", zero_infinity=False. */
aiednn_status_t aiednn_ctc_loss_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *log_probs,
                                       int64_t max_label_length, const aiednn_memory_desc_t *loss,
                                       const aiednn_memory_desc_t *diff_src /* or NULL */);

/* ── recurrent networks (forward / inference) ──────────────────────────────
 * Multi-layer, uni- or bidirectional RNN / LSTM / GRU over a sequence, in the oneDNN layouts:
 *   src_layer     [T, N, SLC]            dst_layer [T, N, DLC] (DLC = DHC, 2 DHC for bi-concat)
 *   src_iter      [L, D, N, DHC]  (NULL: zeros)       dst_iter   [L, D, N, DHC]  (NULL: not written)
 *   src_iter_c    [L, D, N, DHC]  (LSTM; NULL: zeros) dst_iter_c [L, D, N, DHC]  (LSTM; NULL: not written)
 *   weights_layer [L, D, SLC, G, DHC]    weights_iter [L, D, DHC, G, DHC]
 *   bias          [L, D, G, DHC]  (lbr_gru: G + 1, the last the U_o h bias; NULL: zeros)
 * L layers, D = 1 (uni) or 2 (bi); G gates: 1 vanilla, 4 LSTM, 3 GRU. Layer l > 0 reads layer
 * l - 1's dst_layer, so SLC must equal DLC when L > 1. Floating types (f32 / bf16 / f16), computed
 * in double. alpha: the vanilla ReLU's negative slope. */
typedef enum {
    aiednn_rnn_left2right = 0,
    aiednn_rnn_right2left = 1,
    aiednn_rnn_bidirectional_concat = 2,
    aiednn_rnn_bidirectional_sum = 3,
} aiednn_rnn_direction_t;

aiednn_status_t aiednn_rnn_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir,
                                          float alpha, const aiednn_memory_desc_t *src_layer,
                                          const aiednn_memory_desc_t *src_iter, const aiednn_memory_desc_t *src_iter_c,
                                          const aiednn_memory_desc_t *weights_layer,
                                          const aiednn_memory_desc_t *weights_iter, const aiednn_memory_desc_t *bias,
                                          const aiednn_memory_desc_t *dst_layer, const aiednn_memory_desc_t *dst_iter,
                                          const aiednn_memory_desc_t *dst_iter_c);

/* Gated MLP (SwiGLU / GeGLU): dst = (act(src W_gate) * (src W_up)) W_down. src / dst [..., D], W_gate / W_up
 * [D, F], W_down [F, D]; act an eltwise algorithm with alpha / beta (aiednn_eltwise_swish with alpha 1 = SiLU).
 * Floating types. Bound: SRC, WEIGHTS (gate), WEIGHTS_UP, WEIGHTS_DOWN, DST. */
#define AIEDNN_GMLP_WEIGHTS_TRANSPOSED 1 /* nn.Linear's layout: W_gate / W_up [F, D], W_down [D, F] */
aiednn_status_t aiednn_gated_mlp_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                        const aiednn_memory_desc_t *w_gate, const aiednn_memory_desc_t *w_up,
                                        const aiednn_memory_desc_t *w_down, const aiednn_memory_desc_t *dst,
                                        aiednn_alg_kind_t act, float alpha, float beta, unsigned flags);
/* RoPE: dst = src * cos + rotate(src) * sin over the last dim; rotate = cat(-x2, x1) of the halves (interleaved = 0,
 * LLaMA / HF) or (-x[2k+1], x[2k]) pairs (interleaved = 1, GPT-J). cos / sin broadcast to src (same rank). f32
 * arithmetic in the framework's order: bit-identical to mul, mul, add. Bound: SRC, ROPE_COS, ROPE_SIN, DST. */
/* gen_index: dst (s32, any shape)[i0, .., in] = i_axis -- the position along one axis (attention masks, ALiBi).
 * dynamic quantize / dequantize: the scales (and optional zero points) are tensors given at execute
 * (AIEDNN_ARG_SCALES / AIEDNN_ARG_ZERO_POINTS), per tensor ([1], axis -1) or per channel along axis
 * ([dims[axis]]). f32 / bf16 / f16 src -> s8 / u8 dst: dst = sat(nearbyint(src / scale) + zp); s8 / u8 / s32 src ->
 * floating dst: dst = (src - zp) * scale -- the reorder's arithmetic. */
aiednn_status_t aiednn_gen_index_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *dst, int axis);
aiednn_status_t aiednn_dynamic_quantize_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                               const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *scales,
                                               const aiednn_memory_desc_t *zero_points, int axis);

aiednn_status_t aiednn_rope_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                   const aiednn_memory_desc_t *cos, const aiednn_memory_desc_t *sin,
                                   const aiednn_memory_desc_t *dst, int interleaved);
/* rope with quantized operands: x / cos / sin given as s8 / u8 codes
 * with attr.src_scales[0] / src_scales[1] / src_scales_23[0] (+ the matching zero points) are dequantized as read; an
 * s8 / u8 dst is quantized by attr.dst_scale / dst_zero_point. Bit-identical to the dequantizing reorders, the f32
 * rope and the quantizing reorder. No post-ops, no flags. */
aiednn_status_t aiednn_rope_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                        const aiednn_memory_desc_t *cos, const aiednn_memory_desc_t *sin,
                                        const aiednn_memory_desc_t *dst, int interleaved, const aiednn_attr_t *attr);

/* Slice: dst[i...] = src[offsets + i * steps] (dst dims are the sizes; offsets / steps NULL = 0 / 1). Same type,
 * same rank; nhwc operands take the canon path (offsets / steps follow the physical axes). */
aiednn_status_t aiednn_slice_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                    const aiednn_memory_desc_t *dst, const int64_t *offsets, const int64_t *steps);
/* Slice with an attr (f32 src): dst quantization alone -- dst s8 / u8 = saturate(round_even(x / dst_scale) +
 * dst_zero_point), bit-identical to the slice then a quantizing reorder -- or post-ops (eltwise / binary, then
 * that quantization) as the "+pt" tail of the other non-GEMM ops. Per-tensor output scale only. */
aiednn_status_t aiednn_slice_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                         const aiednn_memory_desc_t *dst, const int64_t *offsets, const int64_t *steps,
                                         const aiednn_attr_t *attr);

/* Post-ops (output scale, eltwise / binary post-ops, dst quantization) on non-GEMM primitives: the op runs as
 * without the attr into f32, then the same epilogue as the GEMM family (impl_info ends in "+pt"). Per-tensor
 * output scale only. Batch / layer norm take theirs through their existing _create_attr. */
aiednn_status_t aiednn_pooling_forward_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                                   const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                                   const aiednn_pool_params_t *pp, const aiednn_attr_t *attr);
aiednn_status_t aiednn_softmax_forward_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg, int axis,
                                                   const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                                   const aiednn_attr_t *attr);
aiednn_status_t aiednn_reduction_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg, float pnorm, float eps,
                                             const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                             const aiednn_attr_t *attr);
aiednn_status_t aiednn_resampling_forward_create_attr(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                                      const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                                      const aiednn_attr_t *attr);
aiednn_status_t aiednn_group_normalization_forward_inference_create_attr(aiednn_primitive_t *p, int64_t groups,
                                                                         const aiednn_memory_desc_t *src,
                                                                         const aiednn_memory_desc_t *dst, float eps,
                                                                         unsigned flags, const aiednn_attr_t *attr);

/* Matmul with a CSR-encoded sparse src: dst [M, N] = A [M, K] x weights [K, N] (+ bias broadcast to dst), A as
 * values [nnz] (SRC), column indices [nnz] s32 (SRC_INDICES) and row pointers [M + 1] s32 (SRC_POINTERS; 0 ..
 * nnz, non-decreasing; duplicates add). s8|u8 x s8 -> s32 (bias s32) or f32. nnz is fixed at create; a malformed
 * CSR is a runtime error at execute. */
aiednn_status_t aiednn_sparse_matmul_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *values, int64_t M,
                                            int64_t K, const aiednn_memory_desc_t *weights,
                                            const aiednn_memory_desc_t *bias, const aiednn_memory_desc_t *dst);

/* Attention over variable-length sequences, optionally against a paged K / V cache (LLM prefill / decode).
 * Q [B, H, Sq, D] padded, dst [B, H, Sq, Dv] (query rows >= seq_q[b] written 0). K / V: dense [B, Hk, Skv, D|Dv],
 * or with page_table [B, maxp] s32 the pages [P, Hk, page, D|Dv] -- key j of sequence b is row j % page of page
 * page_table[b, j / page]. seq_kv [B] s32 (required), seq_q [B] s32 or NULL (= Sq). Hk divides H (GQA / MQA).
 * causal: bottom-right aligned per sequence (key j visible to query i when j <= i + seq_kv - seq_q). Floating
 * types. Bound at execute: QUERY, KEY, VALUE, DST, SEQ_LENS_KV [, SEQ_LENS_Q] [, PAGE_TABLE]. */
aiednn_status_t aiednn_sdpa_varlen_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                          const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                          const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q,
                                          const aiednn_memory_desc_t *seq_kv, const aiednn_memory_desc_t *page_table,
                                          float scale, int causal);

/* the same over a compressed (8-bit) K / V cache: K / V s8 or u8, dequantized as gathered, K = (k - kv_zps[0]) *
 * kv_scales[0], V with [1] (kv_zps NULL = 0) */
aiednn_status_t aiednn_sdpa_varlen_create_kvq(aiednn_primitive_t *p, const aiednn_memory_desc_t *q,
                                              const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                              const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q,
                                              const aiednn_memory_desc_t *seq_kv,
                                              const aiednn_memory_desc_t *page_table, float scale, int causal,
                                              const float *kv_scales, const int32_t *kv_zps);

/* AUGRU / linear-before-reset AUGRU: the GRU cell with the
 * update gate scaled by an attention score per step and sequence, u' = (1 - a[t, n]) u, then
 * h = u' h_prev + (1 - u') o. attention [T, N] or [T, N, 1], bound as AIEDNN_ARG_AUGRU_ATTENTION. The other
 * tensors as aiednn_rnn_forward_create (no iter_c). Forward / inference. */
aiednn_status_t aiednn_augru_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir,
                                            const aiednn_memory_desc_t *src_layer, const aiednn_memory_desc_t *src_iter,
                                            const aiednn_memory_desc_t *attention,
                                            const aiednn_memory_desc_t *weights_layer,
                                            const aiednn_memory_desc_t *weights_iter, const aiednn_memory_desc_t *bias,
                                            const aiednn_memory_desc_t *dst_layer,
                                            const aiednn_memory_desc_t *dst_iter);

/* RNN backward (BPTT): the forward's tensors (as aiednn_rnn_forward_create; dst_layer is recomputed,
 * its descriptor fixes DLC) and their gradients in the same order. Bound at execute with the forward
 * tags and DIFF_SRC (src_layer) / DIFF_SRC_ITER(_C) / DIFF_WEIGHTS (weights_layer) /
 * DIFF_WEIGHTS_ITER / DIFF_BIAS / DIFF_DST (dst_layer) / DIFF_DST_ITER(_C). Optional members NULL. */
typedef struct {
    const aiednn_memory_desc_t *src_layer, *src_iter, *src_iter_c, *weights_layer, *weights_iter, *bias, *dst_layer,
        *dst_iter, *dst_iter_c;
} aiednn_rnn_descs_t;
aiednn_status_t aiednn_rnn_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir,
                                           float alpha, const aiednn_rnn_descs_t *fwd, const aiednn_rnn_descs_t *diff);

/* ── N-D convolution / deconvolution / pooling (1..3 spatial dims) ──────────
 * src [N, C, (D,) (H,) W], weights [O, C/groups, (KD,) (KH,) KW], dst [N, O, ...]; entry i of every
 * array is spatial dim i in src order (1-D uses [0] only, 2-D [0..1]). */
#define AIEDNN_MAX_SPATIAL 3
typedef struct aiednn_conv_nd_params {
    int64_t strides[AIEDNN_MAX_SPATIAL];
    int64_t padding_l[AIEDNN_MAX_SPATIAL];
    int64_t padding_r[AIEDNN_MAX_SPATIAL];
    int64_t dilates[AIEDNN_MAX_SPATIAL]; /* 0 = dense */
    int64_t groups;
} aiednn_conv_nd_params_t;

typedef struct aiednn_pool_nd_params {
    int64_t kernel[AIEDNN_MAX_SPATIAL];
    int64_t strides[AIEDNN_MAX_SPATIAL];
    int64_t padding_l[AIEDNN_MAX_SPATIAL];
    int64_t padding_r[AIEDNN_MAX_SPATIAL];
    int64_t dilates[AIEDNN_MAX_SPATIAL];
} aiednn_pool_nd_params_t;

aiednn_status_t aiednn_convolution_forward_create_nd(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                     const aiednn_memory_desc_t *weights,
                                                     const aiednn_memory_desc_t *bias, const aiednn_memory_desc_t *dst,
                                                     const aiednn_conv_nd_params_t *cp, const aiednn_attr_t *attr);
aiednn_status_t aiednn_deconvolution_forward_create_nd(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                       const aiednn_memory_desc_t *weights,
                                                       const aiednn_memory_desc_t *bias,
                                                       const aiednn_memory_desc_t *dst,
                                                       const aiednn_conv_nd_params_t *cp, const aiednn_attr_t *attr);
aiednn_status_t aiednn_pooling_forward_create_nd(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                                 const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst,
                                                 const aiednn_pool_nd_params_t *pp);

/* ── training: backward passes and the training forward of the normalizations ──
 * Floating types. Gradients use the AIEDNN_ARG_DIFF_* tags; the forward tensors a gradient needs
 * keep their own tags (SRC, WEIGHTS, DST, MEAN, VARIANCE, SCALE). Shapes as the forward's; the
 * convolution family takes the N-D parameters (1..3 spatial dims). */
aiednn_status_t aiednn_convolution_backward_data_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                        const aiednn_memory_desc_t *weights,
                                                        const aiednn_memory_desc_t *diff_dst,
                                                        const struct aiednn_conv_nd_params *cp);
aiednn_status_t aiednn_convolution_backward_weights_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                           const aiednn_memory_desc_t *diff_weights,
                                                           const aiednn_memory_desc_t *diff_bias /* or NULL */,
                                                           const aiednn_memory_desc_t *diff_dst,
                                                           const struct aiednn_conv_nd_params *cp);
aiednn_status_t aiednn_deconvolution_backward_data_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                          const aiednn_memory_desc_t *weights,
                                                          const aiednn_memory_desc_t *diff_dst,
                                                          const struct aiednn_conv_nd_params *cp);
aiednn_status_t aiednn_deconvolution_backward_weights_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                             const aiednn_memory_desc_t *diff_weights,
                                                             const aiednn_memory_desc_t *diff_bias,
                                                             const aiednn_memory_desc_t *diff_dst,
                                                             const struct aiednn_conv_nd_params *cp);
aiednn_status_t aiednn_inner_product_backward_data_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                          const aiednn_memory_desc_t *weights,
                                                          const aiednn_memory_desc_t *diff_dst);
aiednn_status_t aiednn_inner_product_backward_weights_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                             const aiednn_memory_desc_t *diff_weights,
                                                             const aiednn_memory_desc_t *diff_bias,
                                                             const aiednn_memory_desc_t *diff_dst);
/* d/dx of the forward eltwise at SRC: DIFF_SRC = DIFF_DST * f'(SRC) */
aiednn_status_t aiednn_eltwise_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, float alpha, float beta,
                                               const aiednn_memory_desc_t *diff_src,
                                               const aiednn_memory_desc_t *diff_dst, const aiednn_memory_desc_t *src);
/* from the forward's DST */
aiednn_status_t aiednn_softmax_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, int axis,
                                               const aiednn_memory_desc_t *diff_src,
                                               const aiednn_memory_desc_t *diff_dst, const aiednn_memory_desc_t *dst);
/* max pooling routes each gradient to its window's first maximum of SRC */
aiednn_status_t aiednn_pooling_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                               const aiednn_memory_desc_t *diff_src,
                                               const aiednn_memory_desc_t *diff_dst, const aiednn_memory_desc_t *src,
                                               const struct aiednn_pool_nd_params *pp);
/* training forward: the statistics are computed (biased variance) and written to MEAN / VARIANCE
 * (batch norm: [C]; layer norm: one per row, src without its last dim; group norm: [N, groups]) */
aiednn_status_t aiednn_batch_normalization_forward_training_create(aiednn_primitive_t *p,
                                                                   const aiednn_memory_desc_t *src,
                                                                   const aiednn_memory_desc_t *dst, float eps,
                                                                   unsigned flags);
aiednn_status_t aiednn_layer_normalization_forward_training_create(aiednn_primitive_t *p,
                                                                   const aiednn_memory_desc_t *src,
                                                                   const aiednn_memory_desc_t *dst, float eps,
                                                                   unsigned flags);
aiednn_status_t aiednn_group_normalization_forward_training_create(aiednn_primitive_t *p, int64_t groups,
                                                                   const aiednn_memory_desc_t *src,
                                                                   const aiednn_memory_desc_t *dst, float eps,
                                                                   unsigned flags);
/* SRC, MEAN, VARIANCE, DIFF_DST [, SCALE] -> DIFF_SRC [, DIFF_SCALE, DIFF_SHIFT] (per flags) */
aiednn_status_t aiednn_batch_normalization_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                           const aiednn_memory_desc_t *diff_dst,
                                                           const aiednn_memory_desc_t *src, float eps, unsigned flags);
aiednn_status_t aiednn_layer_normalization_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                           const aiednn_memory_desc_t *diff_dst,
                                                           const aiednn_memory_desc_t *src, float eps, unsigned flags);
aiednn_status_t aiednn_group_normalization_backward_create(aiednn_primitive_t *p, int64_t groups,
                                                           const aiednn_memory_desc_t *diff_src,
                                                           const aiednn_memory_desc_t *diff_dst,
                                                           const aiednn_memory_desc_t *src, float eps, unsigned flags);
/* SRC, WEIGHTS, DIFF_DST -> DIFF_SRC, DIFF_WEIGHTS (weights broadcast to src as in the forward) */
aiednn_status_t aiednn_prelu_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                             const aiednn_memory_desc_t *diff_weights,
                                             const aiednn_memory_desc_t *diff_dst, const aiednn_memory_desc_t *src,
                                             const aiednn_memory_desc_t *weights);
aiednn_status_t aiednn_lrn_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, int64_t local_size,
                                           float alpha, float beta, float k, const aiednn_memory_desc_t *diff_src,
                                           const aiednn_memory_desc_t *diff_dst, const aiednn_memory_desc_t *src);
aiednn_status_t aiednn_resampling_backward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                                  const aiednn_memory_desc_t *diff_src,
                                                  const aiednn_memory_desc_t *diff_dst);
aiednn_status_t aiednn_shuffle_backward_create(aiednn_primitive_t *p, int axis, int64_t group_size,
                                               const aiednn_memory_desc_t *diff_src,
                                               const aiednn_memory_desc_t *diff_dst);

aiednn_status_t aiednn_resampling_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg,
                                                 const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst);

aiednn_status_t aiednn_lrn_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, int64_t local_size, float alpha,
                                          float beta, float k, const aiednn_memory_desc_t *src,
                                          const aiednn_memory_desc_t *dst);

/* ── normalization (inference: statistics are inputs) ──────────────────── */

aiednn_status_t aiednn_softmax_forward_create(aiednn_primitive_t *p, aiednn_alg_kind_t alg, int axis,
                                              const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst);

/* src [N, C, ...]; MEAN, VARIANCE [C] (and SCALE, SHIFT [C] per flags) at execute. */
aiednn_status_t aiednn_batch_normalization_forward_inference_create(aiednn_primitive_t *p,
                                                                    const aiednn_memory_desc_t *src,
                                                                    const aiednn_memory_desc_t *dst, float eps,
                                                                    unsigned flags);

/* The same with the quantizing reorder after it folded in: src f32, dst s8 / u8, attr carries only
 * dst_scale / dst_zero_point. dst = saturate(round_even((float)bn(x) / dst_scale) + dst_zero_point). */
aiednn_status_t aiednn_batch_normalization_forward_inference_create_attr(aiednn_primitive_t *p,
                                                                         const aiednn_memory_desc_t *src,
                                                                         const aiednn_memory_desc_t *dst, float eps,
                                                                         unsigned flags, const aiednn_attr_t *attr);

/* normalizes over the last dim; SCALE / SHIFT [last dim]; AIEDNN_RMS_NORM: RMSNorm. The _attr
 * form folds the quantizing reorder in, as for batch norm. */
aiednn_status_t aiednn_layer_normalization_forward_inference_create_attr(aiednn_primitive_t *p,
                                                                         const aiednn_memory_desc_t *src,
                                                                         const aiednn_memory_desc_t *dst, float eps,
                                                                         unsigned flags, const aiednn_attr_t *attr);

aiednn_status_t aiednn_layer_normalization_forward_inference_create(aiednn_primitive_t *p,
                                                                    const aiednn_memory_desc_t *src,
                                                                    const aiednn_memory_desc_t *dst, float eps,
                                                                    unsigned flags);

/* src [N, C, ...] normalized per (n, group); SCALE / SHIFT [C]. groups = C: instance norm. */
aiednn_status_t aiednn_group_normalization_forward_inference_create(aiednn_primitive_t *p, int64_t groups,
                                                                    const aiednn_memory_desc_t *src,
                                                                    const aiednn_memory_desc_t *dst, float eps,
                                                                    unsigned flags);

/* ── data movement ─────────────────────────────────────────────────────── */

aiednn_status_t aiednn_concat_create(aiednn_primitive_t *p, int n, const aiednn_memory_desc_t *srcs, int axis,
                                     const aiednn_memory_desc_t *dst);

/* int8 concat: s8 / u8 sources each with (scales[i], zero_points[i] or NULL = 0) into an s8 / u8 dst with
 * (dst_scale, dst_zero_point) -- dequantize -> concat -> quantize as one primitive. */
/* reorder + post-ops: the reorder (an integer src dequantized by scale / zero_point) into f32, then attr's
 * post-ops and dst quantization. */
aiednn_status_t aiednn_reorder_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                           const aiednn_memory_desc_t *dst, const int *perm, float scale,
                                           int32_t zero_point, const aiednn_attr_t *attr);
aiednn_status_t aiednn_concat_create_q(aiednn_primitive_t *p, int n, const aiednn_memory_desc_t *srcs,
                                       const float *scales, const int32_t *zero_points, int axis,
                                       const aiednn_memory_desc_t *dst, float dst_scale, int32_t dst_zero_point);

/* dst = sum_i scales[i] * src_i */
aiednn_status_t aiednn_sum_create(aiednn_primitive_t *p, int n, const float *scales, const aiednn_memory_desc_t *srcs,
                                  const aiednn_memory_desc_t *dst);

/* dst = permute(src) by perm (dst dim i = src dim perm[i]; NULL: identity) with a type
 * conversion: to an integer type round-to-nearest-even(x / scale) + zero_point, saturated;
 * from an integer type to f32 (x - zero_point) * scale; integer to integer saturates. */
aiednn_status_t aiednn_reorder_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                      const aiednn_memory_desc_t *dst, const int *perm, float scale,
                                      int32_t zero_point);

/* channel shuffle along axis: view as [group_size][C/group_size], transpose */
aiednn_status_t aiednn_shuffle_forward_create(aiednn_primitive_t *p, int axis, int64_t group_size,
                                              const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst);

/* ── shorthands kept from the first API ────────────────────────────────── */

/* src [M,K] s8, weights [N,K] s8, dst [M,N] s8 (saturated) or s32: matmul_forward with
 * weights_transposed = 1 and no bias. */
aiednn_status_t aiednn_matmul_primitive_create(aiednn_primitive_t *primitive, const aiednn_memory_desc_t *src,
                                               const aiednn_memory_desc_t *weights, const aiednn_memory_desc_t *dst);

/* valid (unpadded) convolution, src [1,C,H,W] s8, weights [O,C,KH,KW] s8 -> dst s32 */
aiednn_status_t aiednn_convolution_primitive_create(aiednn_primitive_t *primitive, const aiednn_memory_desc_t *src,
                                                    const aiednn_memory_desc_t *weights,
                                                    const aiednn_memory_desc_t *dst, int stride);

/* ── execution and introspection ───────────────────────────────────────── */

aiednn_status_t aiednn_primitive_execute_args(aiednn_primitive_t primitive, int nargs, const aiednn_exec_arg_t *args);

/* SRC, WEIGHTS, DST -- for primitives with exactly those three arguments */
aiednn_status_t aiednn_primitive_execute(aiednn_primitive_t primitive, const void *src, const void *weights, void *dst);

/* The WEIGHTS argument holds the same values on every execute (a model constant, not an
 * activation): an implementation may keep a transformed copy (packed / padded GEMM blocks),
 * keyed by its pointer, instead of rebuilding it per execute. Off by default. */
aiednn_status_t aiednn_primitive_set_constant_weights(aiednn_primitive_t primitive, int constant);

/* zero the counters aiednn_print_stats reports (e.g. between a warm-up run and the timed run) */
void aiednn_stats_reset(void);

/* Model-compiler hook, not part of the primitive contract: called before op i; per-op counting and,
 * with an op -> megakernel table, switches the resident megakernel. Optional. */
void aiednn_op_begin(int i);

/* The implementation dispatch picked: "aie:<kernel>", a composite such as
 * "im2col+blocked:aie:gemm_s8s32", or "ref:<name>". Valid until the
 * primitive is destroyed (copy it to keep it). */
const char *aiednn_primitive_impl_info(aiednn_primitive_t primitive);

void aiednn_primitive_destroy(aiednn_primitive_t primitive);

/* 0 = quiet (default), 1 = one line per dispatch decision and execute,
 * 2 = also every implementation's verdict. */
void aiednn_set_verbose(int level);

/* Only consider implementations whose name starts with `prefix` ("ref:",
 * "aie:"); NULL clears it. For checking an AIE result against the reference. */
void aiednn_set_impl_filter(const char *prefix);

/* Autotune: the first execute of each problem signature times every acceptor and keeps the fastest
 * for that signature. In-place primitives are not timed. aiednn_autotune_dump prints the choices
 * ("AIEDNN_TUNE <sig> <kind> <impl>" lines); aiednn_autotune_load preloads them (returns how many). */
void aiednn_set_autotune(int on);

/* aiednn_prefer_aie (default): every problem an AIE implementation can compute runs on the AIE, and
 * autotuning chooses only among AIE implementations. aiednn_prefer_fastest: acceptors may decline where
 * ref: is faster and autotuning may pick ref:. Read at create. */
typedef enum { aiednn_prefer_aie = 0, aiednn_prefer_fastest = 1 } aiednn_preference_t;
void aiednn_set_preference(aiednn_preference_t preference);
aiednn_preference_t aiednn_get_preference(void);

/* ── streams, caller-owned workspace, async execute ─────────────────────────
 * Executes on one stream complete in order. A caller workspace is 64-byte aligned host memory of
 * >= aiednn_primitive_get_workspace_size bytes, untouched until the stream is waited; the execute then
 * allocates nothing (persistent caches are filled by the first execute). workspace NULL: the library
 * allocates. */
typedef struct aiednn_stream *aiednn_stream_t;
aiednn_status_t aiednn_stream_create(aiednn_stream_t *stream, unsigned flags); /* flags 0 */
aiednn_status_t aiednn_stream_wait(aiednn_stream_t stream);                    /* first error since the last wait */
void aiednn_stream_destroy(aiednn_stream_t stream);                            /* waits first */
size_t aiednn_primitive_get_workspace_size(aiednn_primitive_t primitive);
aiednn_status_t aiednn_primitive_execute_async(aiednn_primitive_t primitive, aiednn_stream_t stream, int nargs,
                                               const aiednn_exec_arg_t *args, void *workspace, size_t workspace_size);

/* ── find / query ────────────────────────────────────────────────────────── */
typedef struct {
    char impl[64];          /* an implementation name (aiednn_primitive_set_impl takes it) */
    int on_aie;             /* 1: an AIE implementation; 0: ref: */
    float time_us;          /* FIND_MEASURE; < 0 unknown */
    size_t workspace_size;  /* measured with the time; 0 unknown (HEURISTIC) */
    aiednn_status_t status; /* success, or why a measured run failed */
} aiednn_impl_perf_t;
#define AIEDNN_FIND_HEURISTIC 0 /* dispatch rank, no execute */
#define AIEDNN_FIND_MEASURE 1   /* every acceptor timed on library-owned buffers, fastest first */
/* Every implementation that accepted the problem (returns the count, may exceed cap). Never reads or
 * writes caller buffers. MEASURE puts the fastest the preference allows into the autotune table. */
int aiednn_primitive_find(aiednn_primitive_t primitive, int mode, aiednn_impl_perf_t *out, int cap);
/* Pin an accepted implementation by name (find / impl_candidates); unimplemented if it did not accept */
aiednn_status_t aiednn_primitive_set_impl(aiednn_primitive_t primitive, const char *impl);

/* Names from aiednn_primitive_find / _impl_candidates are base names ("im2col", "gemm+conv"), which
 * aiednn_primitive_set_impl takes; aiednn_primitive_impl_info composes the inner one ("im2col+blocked:aie:..."). */
/* Every implementation that accepted the primitive's problem, in dispatch rank (AIE first, ref:
 * last), comma-separated into buf (truncated to cap); returns how many. */
int aiednn_primitive_impl_candidates(aiednn_primitive_t primitive, char *buf, size_t cap);
void aiednn_autotune_dump(void);
int aiednn_autotune_load(const char *text);

const char *aiednn_status_str(aiednn_status_t status);

/* Per-implementation counters since start: executes and AIE block launches. */
void aiednn_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* AIEDNN_H */
