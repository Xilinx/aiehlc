/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_ort_map.h -- ONNX node -> aiednn graph op (include/aiednn_graph.h), for the ONNX Runtime
 * execution provider (aiednn_ort_ep.cc). The same mapping as aiegraph/dnn_onnx_graph.py: one graph op
 * per ONNX node, initializers as constant logical tensors, anything without a mapping (or with a
 * dynamic shape / an unknown element type) as a wildcard op, so the partitioner sees the whole graph
 * and never fuses across what the framework keeps.
 *
 * GraphMap::add_nodes(nodes) builds an aiednn_graph_t; GetCapability and Compile both use it, the first
 * over the model, the second over one fused node's subgraph.
 ******************************************************************************/
#ifndef AIEDNN_ORT_MAP_H
#define AIEDNN_ORT_MAP_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "aiednn_graph.h"

namespace aiednn_ort {

struct ConstData { /* an initializer's bytes: bound to every execute (constants dropped by ORT) */
    std::vector<uint8_t> bytes;
};

static inline aiednn_data_type_t onnx_dt(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return aiednn_f32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return aiednn_s8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return aiednn_u8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return aiednn_s32;
    default:
        return (aiednn_data_type_t)0;
    }
}

class GraphMap {
  public:
    explicit GraphMap(bool fuse = true) {
        aiednn_graph_create(&g_);
        aiednn_graph_set_fusion(g_, fuse ? 1 : 0);
    }
    ~GraphMap() { aiednn_graph_destroy(g_); }
    GraphMap(const GraphMap &) = delete;
    GraphMap &operator=(const GraphMap &) = delete;

    aiednn_graph_t graph() const { return g_; }
    std::vector<Ort::ConstNode> nodes;             /* op id - 1 -> ONNX node */
    std::unordered_map<std::string, uint64_t> ids; /* value name -> logical tensor id */
    std::unordered_map<uint64_t, aiednn_logical_tensor_t> lts;
    std::unordered_map<uint64_t, ConstData> consts; /* constant initializers by id */
    int wildcards = 0;

    /* every node of a graph (or of a fused node's subgraph); marks graph outputs */
    void add_nodes(const std::vector<Ort::ConstNode> &ns, const std::vector<Ort::ConstValueInfo> &outputs) {
        for (const auto &n : ns)
            add_node(n);
        for (const auto &o : outputs) {
            auto it = ids.find(o.GetName());
            if (it != ids.end())
                aiednn_graph_mark_output(g_, it->second);
        }
        aiednn_graph_finalize(g_);
    }

  private:
    aiednn_graph_t g_ = nullptr;

    /* the logical tensor of a value, or false (no static shape / unsupported type) */
    bool lt_of(const Ort::ConstValueInfo &v, aiednn_logical_tensor_t &lt) {
        const std::string name = v.GetName();
        auto it = ids.find(name);
        if (it != ids.end()) {
            lt = lts[it->second];
            return true;
        }
        auto ti = v.TypeInfo();
        if (ti.GetONNXType() != ONNX_TYPE_TENSOR)
            return false;
        auto ts = ti.GetTensorTypeAndShapeInfo();
        const aiednn_data_type_t dt = onnx_dt(ts.GetElementType());
        std::vector<int64_t> dims = ts.GetShape();
        bool ok = dt && dims.size() <= AIEDNN_MAX_NDIMS;
        for (int64_t d : dims)
            ok = ok && d >= 0;
        if (!ok) {
            if (getenv("AIEDNN_ORT_EP_LOG")) {
                fprintf(stderr, "[aiednn-ep]     tensor %s: type %d dims", name.c_str(), (int)ts.GetElementType());
                for (int64_t d : dims)
                    fprintf(stderr, " %lld", (long long)d);
                fprintf(stderr, "\n");
            }
            return false;
        }
        memset(&lt, 0, sizeof lt);
        lt.id = ids.size() + 1;
        if (aiednn_memory_desc_init(&lt.md, (int)dims.size(), dims.data(), dt) != aiednn_success)
            return false;
        lt.property = aiednn_lt_variable;
        if (v.IsConstantInitializer()) {
            Ort::ConstValue val;
            if (v.GetInitializer(val).IsOK() && val) {
                const size_t n = val.GetTensorSizeInBytes();
                ConstData c;
                c.bytes.assign((const uint8_t *)val.GetTensorRawData(), (const uint8_t *)val.GetTensorRawData() + n);
                consts.emplace(lt.id, std::move(c));
                lt.property = aiednn_lt_constant;
            }
        }
        ids[name] = lt.id;
        lts[lt.id] = lt;
        return true;
    }

    /* a constant's values as floats (scales, clip bounds), or empty */
    std::vector<float> const_floats(const Ort::ConstValueInfo &v) {
        aiednn_logical_tensor_t lt;
        if (!v || !lt_of(v, lt) || !consts.count(lt.id))
            return {};
        const auto &b = consts[lt.id].bytes;
        std::vector<float> out;
        if (lt.md.data_type == aiednn_f32) {
            out.resize(b.size() / 4);
            memcpy(out.data(), b.data(), out.size() * 4);
        } else if (lt.md.data_type == aiednn_s32) {
            for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                int32_t x;
                memcpy(&x, b.data() + i, 4);
                out.push_back((float)x);
            }
        } else {
            for (uint8_t x : b)
                out.push_back(lt.md.data_type == aiednn_s8 ? (float)(int8_t)x : (float)x);
        }
        return out;
    }

    static bool attr_ints(const Ort::ConstNode &n, const char *name, std::vector<int64_t> &out) {
        Ort::ConstOpAttr a;
        return n.GetAttributeByName(name, a).IsOK() && a && a.GetValueArray(out).IsOK();
    }
    static int64_t attr_int(const Ort::ConstNode &n, const char *name, int64_t dflt) {
        Ort::ConstOpAttr a;
        int64_t v = dflt;
        if (n.GetAttributeByName(name, a).IsOK() && a)
            (void)a.GetValue(v);
        return v;
    }
    static float attr_float(const Ort::ConstNode &n, const char *name, float dflt) {
        Ort::ConstOpAttr a;
        float v = dflt;
        if (n.GetAttributeByName(name, a).IsOK() && a)
            (void)a.GetValue(v);
        return v;
    }

    bool map(const Ort::ConstNode &n, const std::vector<Ort::ConstValueInfo> &in, aiednn_graph_op_kind_t &kind,
             aiednn_graph_op_attr_t &a, int &n_in);
    bool map_pool(const Ort::ConstNode &n, const std::string &t, const std::vector<Ort::ConstValueInfo> &in,
                  aiednn_graph_op_kind_t &kind, aiednn_graph_op_attr_t &a);
    bool map_quant(const Ort::ConstNode &n, const std::string &t, const std::vector<Ort::ConstValueInfo> &in,
                   aiednn_graph_op_kind_t &kind, aiednn_graph_op_attr_t &a);
    std::vector<float> scales_, zps_f_;
    std::vector<int32_t> zps_;
    std::vector<int> reorder_; /* graph-op input i = ONNX input reorder_[i] (empty: identity) */

    void add_node(const Ort::ConstNode &n) {
        std::vector<Ort::ConstValueInfo> in = n.GetInputs(), out = n.GetOutputs();
        aiednn_graph_op_attr_t a;
        aiednn_graph_op_attr_init(&a);
        aiednn_graph_op_kind_t kind = aiednn_gop_wildcard;
        int n_in = (int)in.size();
        reorder_.clear();
        bool ok = out.size() == 1 && map(n, in, kind, a, n_in);
        std::vector<aiednn_logical_tensor_t> lin;
        aiednn_logical_tensor_t lout;
        for (int i = 0; ok && i < n_in; i++) {
            const Ort::ConstValueInfo &v = in[reorder_.empty() ? i : reorder_[i]];
            aiednn_logical_tensor_t lt;
            ok = v && lt_of(v, lt);
            lin.push_back(lt);
        }
        const bool mapped = ok;
        ok = ok && lt_of(out[0], lout);
        if (!ok && getenv("AIEDNN_ORT_EP_LOG"))
            fprintf(stderr, "[aiednn-ep]   wildcard %s (%s): %s\n", n.GetName().c_str(), n.GetOperatorType().c_str(),
                    out.size() != 1 ? "outputs != 1"
                    : !mapped       ? "no mapping / an input without a static shape"
                                    : "output without a static shape");
        if (!ok) { /* the framework keeps it: wildcard over the tensor inputs */
            kind = aiednn_gop_wildcard;
            aiednn_graph_op_attr_init(&a);
            lin.clear();
            for (const auto &v : in) {
                aiednn_logical_tensor_t lt;
                if (v && lt_of(v, lt))
                    lin.push_back(lt);
            }
            if (out.empty() || !lt_of(out[0], lout)) { /* no static output: an opaque op with no output */
                memset(&lout, 0, sizeof lout);
                lout.id = 1000000000ull + nodes.size();
                const int64_t one = 1;
                aiednn_memory_desc_init(&lout.md, 1, &one, aiednn_f32);
            }
            wildcards++;
        }
        nodes.push_back(n);
        aiednn_graph_add_op(g_, nodes.size(), kind, &a, (int)lin.size(), lin.data(), 1, &lout);
    }
};

inline bool GraphMap::map_quant(const Ort::ConstNode &n, const std::string &t,
                                const std::vector<Ort::ConstValueInfo> &in, aiednn_graph_op_kind_t &kind,
                                aiednn_graph_op_attr_t &a) {
    scales_ = const_floats(in.size() > 1 ? in[1] : Ort::ConstValueInfo{});
    zps_f_ = const_floats(in.size() > 2 ? in[2] : Ort::ConstValueInfo{});
    if (scales_.empty())
        return false;
    kind = t == "QuantizeLinear" ? aiednn_gop_quantize : aiednn_gop_dequantize;
    if (scales_.size() == 1) {
        a.scale = scales_[0];
        a.zero_point = zps_f_.empty() ? 0 : (int32_t)zps_f_[0];
        return true;
    }
    if (kind == aiednn_gop_quantize)
        return false;
    zps_.assign(scales_.size(), 0);
    for (size_t i = 0; i < zps_.size() && i < zps_f_.size(); i++)
        zps_[i] = (int32_t)zps_f_[i];
    a.n_scales = (int64_t)scales_.size();
    a.scales = scales_.data();
    a.zero_points = zps_.data();
    a.axis = (int)attr_int(n, "axis", 1);
    return true;
}

inline bool GraphMap::map_pool(const Ort::ConstNode &n, const std::string &t,
                               const std::vector<Ort::ConstValueInfo> &in, aiednn_graph_op_kind_t &kind,
                               aiednn_graph_op_attr_t &a) {
    if (t == "GlobalAveragePool") {
        auto s = in[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
        if (s.size() != 4)
            return false;
        a.pool.kernel[0] = s[2], a.pool.kernel[1] = s[3];
        a.pool.strides[0] = a.pool.strides[1] = 1;
        kind = aiednn_gop_avg_pool;
        return true;
    }
    std::vector<int64_t> k, st, p;
    if (!attr_ints(n, "kernel_shape", k) || k.size() != 2 || attr_int(n, "ceil_mode", 0))
        return false;
    if (!attr_ints(n, "strides", st))
        st = {1, 1};
    if (!attr_ints(n, "pads", p))
        p = {0, 0, 0, 0};
    for (int i = 0; i < 2; i++) {
        a.pool.kernel[i] = k[i], a.pool.strides[i] = st[i];
        a.pool.padding_l[i] = p[i], a.pool.padding_r[i] = p[i + 2];
    }
    kind = t == "MaxPool" ? aiednn_gop_max_pool : aiednn_gop_avg_pool;
    if (t == "AveragePool" && !attr_int(n, "count_include_pad", 0))
        a.alg = aiednn_pooling_avg_exclude_padding;
    return true;
}

/* ONNX op -> graph op kind + attrs; n_in: how many leading inputs the graph op takes */
inline bool GraphMap::map(const Ort::ConstNode &n, const std::vector<Ort::ConstValueInfo> &in,
                          aiednn_graph_op_kind_t &kind, aiednn_graph_op_attr_t &a, int &n_in) {
    const std::string t = n.GetOperatorType();
    if (!n.GetDomain().empty() && n.GetDomain() != "ai.onnx")
        return false;
    for (const auto &v : in)
        if (!v)
            return false; /* an omitted optional input: keep it simple */
    if (t == "QuantizeLinear" || t == "DequantizeLinear") {
        n_in = 1;
        return map_quant(n, t, in, kind, a);
    }
    if (t == "Conv") {
        std::vector<int64_t> st, p, d;
        if (!attr_ints(n, "strides", st))
            st = {1, 1};
        if (!attr_ints(n, "pads", p))
            p = {0, 0, 0, 0};
        if (!attr_ints(n, "dilations", d))
            d = {1, 1};
        if (st.size() != 2 || p.size() != 4 || d.size() != 2)
            return false;
        for (int i = 0; i < 2; i++) {
            a.conv.strides[i] = st[i], a.conv.padding_l[i] = p[i], a.conv.padding_r[i] = p[i + 2];
            a.conv.dilates[i] = d[i] - 1;
        }
        a.conv.groups = attr_int(n, "group", 1);
        kind = aiednn_gop_convolution;
        return true;
    }
    if (t == "Gemm") {
        if (attr_float(n, "alpha", 1.f) != 1.f || attr_float(n, "beta", 1.f) != 1.f || attr_int(n, "transA", 0))
            return false;
        a.weights_transposed = (int)attr_int(n, "transB", 0);
        kind = aiednn_gop_matmul;
        return true;
    }
    static const std::unordered_map<std::string, aiednn_graph_op_kind_t> simple = {
        {"MatMul", aiednn_gop_matmul}, {"Relu", aiednn_gop_relu},           {"Sigmoid", aiednn_gop_sigmoid},
        {"Tanh", aiednn_gop_tanh},     {"HardSwish", aiednn_gop_hardswish}, {"Add", aiednn_gop_add},
        {"Mul", aiednn_gop_multiply}};
    if (auto it = simple.find(t); it != simple.end()) {
        kind = it->second;
        return true;
    }
    if (t == "Clip") {
        if (in.size() != 3)
            return false;
        auto lo = const_floats(in[1]), hi = const_floats(in[2]);
        if (lo.size() != 1 || hi.size() != 1)
            return false;
        a.alpha = lo[0], a.beta = hi[0], n_in = 1;
        kind = aiednn_gop_clamp;
        return true;
    }
    if (t == "MaxPool" || t == "AveragePool" || t == "GlobalAveragePool")
        return map_pool(n, t, in, kind, a);
    if (t == "Flatten" || t == "Reshape" || t == "Squeeze" || t == "Unsqueeze") {
        n_in = 1;
        kind = aiednn_gop_reshape;
        return true;
    }
    if (t == "Transpose") {
        std::vector<int64_t> perm;
        if (!attr_ints(n, "perm", perm) || perm.size() > AIEDNN_MAX_NDIMS)
            return false;
        for (size_t i = 0; i < perm.size(); i++)
            a.perm[i] = (int)perm[i];
        kind = aiednn_gop_transpose;
        return true;
    }
    if (t == "Softmax" || t == "Concat") {
        auto r = in[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape().size();
        int64_t ax = attr_int(n, "axis", t == "Softmax" ? -1 : 0);
        a.axis = (int)(ax < 0 ? ax + (int64_t)r : ax);
        kind = t == "Softmax" ? aiednn_gop_softmax : aiednn_gop_concat;
        return true;
    }
    if (t == "BatchNormalization") { /* ONNX: x, scale, B, mean, var -> graph: x, mean, var, scale, shift */
        if (in.size() != 5 || n.GetOutputs().size() != 1)
            return false;
        a.eps = attr_float(n, "epsilon", 1e-5f);
        kind = aiednn_gop_batch_norm_inference;
        reorder_ = {0, 3, 4, 1, 2};
        return true;
    }
    if (t == "LayerNormalization") {
        auto r = (int64_t)in[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape().size();
        if (attr_int(n, "axis", -1) != -1 && attr_int(n, "axis", -1) != r - 1)
            return false;
        a.eps = attr_float(n, "epsilon", 1e-5f);
        kind = aiednn_gop_layer_norm;
        return true;
    }
    return false;
}

} // namespace aiednn_ort

#endif /* AIEDNN_ORT_MAP_H */
