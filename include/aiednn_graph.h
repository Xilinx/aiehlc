/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn graph API: hand over a whole graph of framework-level ops, get back
 * partitions -- each either supported (fused into one aiednn primitive) or not
 * (the framework runs those ops itself) -- then compile and execute the
 * supported ones.
 *
 *   aiednn_graph_create(&g);
 *   aiednn_graph_add_op(g, id, aiednn_gop_convolution, &attr, 2, in, 1, out);   // every op
 *   aiednn_graph_finalize(g);
 *   n = aiednn_graph_get_partitions(g, parts, cap);
 *   for each part: if (aiednn_partition_is_supported(part))
 *       aiednn_partition_compile(part, &cp), aiednn_compiled_partition_execute(cp, ...);
 *
 * A partition compiles to one aiednn primitive.
 ******************************************************************************/
#ifndef AIEDNN_GRAPH_H
#define AIEDNN_GRAPH_H

#include "aiednn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { aiednn_lt_variable = 0, aiednn_lt_constant = 1 } aiednn_lt_property_t;

/* A tensor between ops: id unique in the graph, dense row-major md. constant: same values on
 * every execute (weights, bias, quantization parameters) -- lets an implementation cache a
 * packed copy (aiednn_primitive_set_constant_weights). */
typedef struct {
    uint64_t id;
    aiednn_memory_desc_t md;
    aiednn_lt_property_t property;
} aiednn_logical_tensor_t;

/* Graph op kinds and their inputs -> outputs (one output each unless noted below):
 *   convolution        src [N,C,H,W], weights [O,C/g,KH,KW] [, bias [O]]       attr.conv
 *   matmul             src [..,M,K], weights [..,K,N] [, bias [N]]            attr.weights_transposed ([..,N,K])
 *                      (weights of lower rank broadcast over src's leading dims: nn.Linear on tokens)
 *   inner_product      src [N,..], weights [OC,..] [, bias [OC]]
 *   bias_add           x, b [C] along attr.axis; -1: fused, the anchor's channel (conv / deconv dim 1,
 *                      matmul / inner product the last); alone, the dim matching C (a tie: dim 1 at
 *                      rank >= 4, else the last) -- set it explicitly when that can be wrong
 *   add, multiply, binary (attr.alg)   a, b  (numpy broadcasting)
 *   relu (attr.alpha: leaky slope), clamp (alpha = min, beta = max), gelu (attr.alg
 *     aiednn_eltwise_gelu_tanh for the tanh form), sigmoid, tanh, hardswish, eltwise (attr.alg,
 *     alpha, beta)                                                             x
 *   max_pool, avg_pool attr.pool; avg counts padding unless attr.alg = aiednn_pooling_avg_exclude_padding
 *   softmax            x, attr.axis
 *   layer_norm         x [, scale [, shift]] over the last dim, attr.eps, attr.flags (AIEDNN_RMS_NORM)
 *   batch_norm_inference  x, mean, variance [, scale [, shift]], attr.eps
 *   concat             n inputs, attr.axis
 *   reshape            x (output md gives the shape; element count kept)
 *   transpose          x, attr.perm (dst dim i = src dim perm[i])
 *   quantize           x f32 -> s8 / u8: round_even(x / scale) + zero_point, per tensor
 *   dequantize         q s8 / u8 / s32 -> f32: (q - zero_point) * scale; per tensor, or per
 *                      channel (attr.n_scales, attr.scales, attr.zero_points, attr.axis) for weights
 *   sdpa               q, k, v [, mask], attr.sdpa_scale, attr.causal
 *   wildcard           anything else: never in a supported partition (the framework keeps it)
 *   deconvolution      src, weights [O,C/g,KH,KW] [, bias [O]], attr.conv; fuses like convolution
 *   resampling         x -> dst spatial size; attr.alg aiednn_resampling_nearest (default) / _linear
 *   reduction          x, attr.alg (aiednn_reduction_*, default sum), attr.pnorm, attr.eps over the
 *                      dims in attr.reduce_mask (bit i = src dim i); dst may drop them (keepdim off);
 *                      mask 0: the dims where a same-rank dst is 1
 *   group_norm         x [, scale [, shift]] [N,C,...], attr.groups (= C: instance norm), attr.eps
 *   prelu              x, slope (broadcastable to x after leading 1s)
 *   slice              x -> the sub-tensor starting at attr.starts[axis], of the output md's dims (step 1):
 *                      Dense only.
 *   type_cast          x -> y of another data type, same dims (round to nearest even, saturating)
 *   reorder            x -> y, same dims; data type and / or memory format may differ
 *   select             cond, then, else -> y = cond ? then : else (numpy broadcasting, cond nonzero = true)
 *   squared_difference a, b -> (a - b)^2 (one binary primitive with a square post-op)
 *   dropout            x -> y [, mask u8]: attr.alpha = drop probability, attr.seed; probability 0: identity
 *   rms_norm           x [, gamma] -> y over the last dim, attr.eps (= layer_norm with AIEDNN_RMS_NORM)
 *   layer_norm / group_norm with 3 outputs: the training forward -- y, mean, variance (biased)
 *   softmax            attr.alg aiednn_softmax_log: log-softmax
 * Training (backward) ops -- gradients in the order listed, inputs in the order listed:
 *   eltwise_backward          src, diff_dst -> diff_src; attr.alg / alpha / beta as the forward eltwise
 *   softmax_backward          dst, diff_dst -> diff_src; attr.axis, attr.alg (accurate / log)
 *   pool_backward             diff_dst, src -> diff_src; attr.alg aiednn_pooling_max / avg_*, attr.pool;
 *                             avg: src may be omitted (diff_src's md gives the shape)
 *   resampling_backward       diff_dst -> diff_src; attr.alg
 *   bias_add_backward         diff_dst -> diff_bias [C]; attr.axis = the channel (default as bias_add)
 *   convolution_backward_data / deconvolution_backward_data       diff_dst, weights -> diff_src; attr.conv
 *   convolution_backward_weights / deconvolution_backward_weights src, diff_dst -> diff_weights [, diff_bias]
 *   batch_norm_training       x [, gamma [, beta]] -> y, mean, variance (batch statistics); attr.eps
 *   batch_norm_backward / layer_norm_backward / group_norm_backward
 *                             x, diff_y, mean, variance [, gamma] -> diff_x [, diff_gamma [, diff_beta]];
 *                             gamma with 2 / 3 outputs; layer norm with AIEDNN_RMS_NORM has no diff_beta;
 *                             group norm: attr.groups
 *   prelu_backward            x, slope, diff_y -> diff_x, diff_slope
 *   gen_index                 x -> y s32 of x's dims: y[...] = the index along attr.axis (x is not read)
 *   dynamic_quantize          x f32, scales [, zero_points] -> y s8 / u8; scales / zero points are tensors read
 *   dynamic_dequantize        x s8 / u8 / s32, scales [, zero_points] -> y f32   at execute; [1] per tensor, or
 *                             [dims[attr.axis]] per channel (attr.axis -1: per tensor)
 *   linear_attention          q, k, v, gate [, beta] [, state_in] -> dst [, state_out]: the gated delta rule
 *                             (aiednn_linear_attention_create). attr.flags
 *                             AIEDNN_LA_* (NO_DELTA: no beta input), attr.sdpa_scale (0: 1 / sqrt(Dk)) */
typedef enum {
    aiednn_gop_convolution = 1,
    aiednn_gop_matmul,
    aiednn_gop_inner_product,
    aiednn_gop_bias_add,
    aiednn_gop_add,
    aiednn_gop_multiply,
    aiednn_gop_binary,
    aiednn_gop_relu,
    aiednn_gop_clamp,
    aiednn_gop_gelu,
    aiednn_gop_sigmoid,
    aiednn_gop_tanh,
    aiednn_gop_hardswish,
    aiednn_gop_eltwise,
    aiednn_gop_max_pool,
    aiednn_gop_avg_pool,
    aiednn_gop_softmax,
    aiednn_gop_layer_norm,
    aiednn_gop_batch_norm_inference,
    aiednn_gop_concat,
    aiednn_gop_reshape,
    aiednn_gop_transpose,
    aiednn_gop_quantize,
    aiednn_gop_dequantize,
    aiednn_gop_sdpa,
    aiednn_gop_wildcard,
    aiednn_gop_deconvolution, /* appended: values above stay stable */
    aiednn_gop_resampling,
    aiednn_gop_reduction,
    aiednn_gop_group_norm,
    aiednn_gop_prelu,
    aiednn_gop_slice, /* appended */
    aiednn_gop_type_cast,
    aiednn_gop_reorder,
    aiednn_gop_select,
    aiednn_gop_squared_difference,
    aiednn_gop_dropout,
    aiednn_gop_rms_norm,
    aiednn_gop_eltwise_backward,
    aiednn_gop_softmax_backward,
    aiednn_gop_pool_backward,
    aiednn_gop_resampling_backward,
    aiednn_gop_bias_add_backward,
    aiednn_gop_convolution_backward_data,
    aiednn_gop_convolution_backward_weights,
    aiednn_gop_deconvolution_backward_data,
    aiednn_gop_deconvolution_backward_weights,
    aiednn_gop_batch_norm_training,
    aiednn_gop_batch_norm_backward,
    aiednn_gop_layer_norm_backward,
    aiednn_gop_group_norm_backward,
    aiednn_gop_prelu_backward,
    aiednn_gop_gen_index,
    aiednn_gop_dynamic_quantize,
    aiednn_gop_dynamic_dequantize,
    aiednn_gop_linear_attention,
    aiednn_gop_last_ /* not an op: one past the last kind */
} aiednn_graph_op_kind_t;

typedef struct aiednn_graph_op_attr {
    aiednn_alg_kind_t alg; /* eltwise / binary algorithm; gelu / avg_pool variant; 0 = the kind's default */
    float alpha, beta;     /* eltwise parameters (aiednn_alg_kind_t), clamp min / max */
    aiednn_conv_params_t conv;
    aiednn_pool_params_t pool;
    int weights_transposed; /* matmul */
    int axis;               /* softmax, concat, bias_add channel, per-channel dequantize; -1 default */
    int perm[AIEDNN_MAX_NDIMS];
    float eps;
    unsigned flags;
    float scale; /* quantize / dequantize, per tensor */
    int32_t zero_point;
    int64_t n_scales;           /* dequantize per channel along axis: 0 = per tensor (scale, zero_point) */
    const float *scales;        /* copied at add_op */
    const int32_t *zero_points; /* NULL = all 0; copied */
    float sdpa_scale;
    int causal;
    /* appended */
    int64_t groups;                   /* group_norm */
    float pnorm;                      /* reduction norm_lp_* */
    unsigned reduce_mask;             /* reduction: bit i = src dim i is reduced */
    int64_t starts[AIEDNN_MAX_NDIMS]; /* slice: per-axis start (appended) */
    uint64_t seed;                    /* dropout (appended) */
} aiednn_graph_op_attr_t;

/* zeroes *attr, axis -1, scale 1 */
void aiednn_graph_op_attr_init(aiednn_graph_op_attr_t *attr);

typedef struct aiednn_graph *aiednn_graph_t;
typedef struct aiednn_partition *aiednn_partition_t;
typedef struct aiednn_compiled_partition *aiednn_compiled_partition_t;

aiednn_status_t aiednn_graph_create(aiednn_graph_t *graph);

/* attr may be NULL (all defaults). Ops may be added in any order; a tensor id must be produced by
 * at most one op. n_out is 1, or up to 4 for the multi-output ops above (training). Copies everything
 * it is given. */
aiednn_status_t aiednn_graph_add_op(aiednn_graph_t graph, uint64_t op_id, aiednn_graph_op_kind_t kind,
                                    const aiednn_graph_op_attr_t *attr, int n_in, const aiednn_logical_tensor_t *in,
                                    int n_out, const aiednn_logical_tensor_t *out);

/* A tensor the caller reads after the graph runs, even though ops also consume it: never fused
 * away as an intermediate. Tensors no op consumes are outputs already. */
aiednn_status_t aiednn_graph_mark_output(aiednn_graph_t graph, uint64_t tensor_id);

/* Topological sort + partitioning; invalid_arguments on a cycle or a tensor produced twice. */
aiednn_status_t aiednn_graph_finalize(aiednn_graph_t graph);

/* Partitions in an executable order (every input of partition i is a graph input or an output of a
 * partition before i); fills up to cap, returns the total. Owned by the graph. */
int aiednn_graph_get_partitions(aiednn_graph_t graph, aiednn_partition_t *out, int cap);

int aiednn_partition_is_supported(aiednn_partition_t partition);
int aiednn_partition_get_ops(aiednn_partition_t partition, uint64_t *op_ids, int cap);
int aiednn_partition_get_inputs(aiednn_partition_t partition, aiednn_logical_tensor_t *lts, int cap);
int aiednn_partition_get_outputs(aiednn_partition_t partition, aiednn_logical_tensor_t *lts, int cap);
/* "conv+bias_add+relu+quantize -> im2col+blocked:aie:gemm_s8s32", or why it is unsupported */
const char *aiednn_partition_info(aiednn_partition_t partition);

aiednn_status_t aiednn_partition_compile(aiednn_partition_t partition, aiednn_compiled_partition_t *compiled);
/* the implementation dispatch picked for it (aiednn_primitive_impl_info) */
const char *aiednn_compiled_partition_impl_info(aiednn_compiled_partition_t compiled);
size_t aiednn_compiled_partition_get_workspace_size(aiednn_compiled_partition_t compiled);

typedef struct {
    uint64_t id;
    void *data;
} aiednn_tensor_t;

/* aiednn_primitive_find / _set_impl on the partition's primitive. */
int aiednn_compiled_partition_find(aiednn_compiled_partition_t compiled, int mode, aiednn_impl_perf_t *out, int cap);
aiednn_status_t aiednn_compiled_partition_set_impl(aiednn_compiled_partition_t compiled, const char *impl);

/* inputs / outputs by logical tensor id (any order; extra entries ignored); forwards to
 * aiednn_primitive_execute_async. stream NULL: synchronous. workspace: NULL = the library allocates,
 * else >= get_workspace_size bytes of 64-byte aligned host memory, untouched until the stream is
 * waited. */
aiednn_status_t aiednn_compiled_partition_execute(aiednn_compiled_partition_t compiled, aiednn_stream_t stream,
                                                  int n_in, const aiednn_tensor_t *in, int n_out,
                                                  const aiednn_tensor_t *out, void *workspace, size_t workspace_size);
void aiednn_compiled_partition_destroy(aiednn_compiled_partition_t compiled);

/* Fusion level (AIEDNN_GRAPH_NO_FUSE=1 sets 0): 0 = every op its own partition; 1 (default) = exact
 * fusions only (anchor epilogues, dequantize read-through, norm + quantize, quantized binary / eltwise /
 * max_pool / views, requant post-ops), each bit-identical to the int8 ops it replaces; 2 = also drops a
 * quantize -> dequantize pair with equal per-tensor parameters before a relu / clamp in an anchor's chain. */
void aiednn_graph_set_fusion(aiednn_graph_t graph, int level);

/* Layout propagation (default on; AIEDNN_GRAPH_DENSE=1 turns it off): a 4-D activation that only connects
 * supported partitions taking nhwc natively is kept aiednn_format_nhwc. Graph inputs / outputs stay dense.
 * Partition inputs / outputs report the chosen md: allocate aiednn_memory_desc_get_size(&lt.md) bytes. */
void aiednn_graph_set_layout_propagation(aiednn_graph_t graph, int on);

void aiednn_graph_destroy(aiednn_graph_t graph); /* and its partitions; compiled partitions stay valid */

#ifdef __cplusplus
}
#endif

#endif /* AIEDNN_GRAPH_H */
