/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_ort_ep.cc -- aiednn as an ONNX Runtime execution provider (plugin EP library, ORT >= 1.23).
 *
 *   GetCapability = the graph API's partitions: every ONNX node becomes a graph op (aiednn_ort_map.h),
 *                   aiednn_graph_get_partitions decides, each supported partition is one fused node;
 *   Compile       = aiednn_partition_compile of that fused node's subgraph (one primitive, dispatch
 *                   picks AIE or ref: as for any primitive);
 *   Compute       = aiednn_compiled_partition_execute on ORT's CPU buffers.
 *
 * Constant initializers are copied at Compile (drop_constant_initializers), so ORT feeds only the
 * activations. Session option "ep.aiednn.benchmark" = "1": aiednn_compiled_partition_find(MEASURE)
 * once per partition at Compile and pin the fastest. Python: aiegraph.dnn_ort_ep.
 ******************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aiednn_ort_map.h"

#define RETURN_IF_ERROR(fn)                                                                                            \
    do {                                                                                                               \
        OrtStatus *_s = (fn);                                                                                          \
        if (_s)                                                                                                        \
            return _s;                                                                                                 \
    } while (0)

namespace aiednn_ort {

struct Apis {
    const OrtApi &ort;
    const OrtEpApi &ep;
};

/* one compiled fused node */
struct Kernel {
    aiednn_compiled_partition_t cp = nullptr;
    std::vector<uint64_t> in_ids;  /* kernel-context input i -> logical tensor id */
    std::vector<uint64_t> out_ids; /* kernel-context output i -> logical tensor id */
    std::vector<std::vector<int64_t>> out_dims;
    std::unordered_map<uint64_t, ConstData> consts;
    ~Kernel() {
        if (cp)
            aiednn_compiled_partition_destroy(cp);
    }
};

class Factory;

class Ep : public OrtEp {
  public:
    Ep(Factory &f, const Apis &apis, bool benchmark, const OrtLogger &logger);
    Apis apis;
    bool benchmark;
    const OrtLogger &logger;
    std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;

  private:
    static const char *ORT_API_CALL GetNameImpl(const OrtEp *) noexcept { return "aiednn"; }
    static OrtStatus *ORT_API_CALL GetCapabilityImpl(OrtEp *self, const OrtGraph *graph,
                                                     OrtEpGraphSupportInfo *info) noexcept;
    static OrtStatus *ORT_API_CALL CompileImpl(OrtEp *self, const OrtGraph **graphs, const OrtNode **fused,
                                               size_t count, OrtNodeComputeInfo **infos,
                                               OrtNode **ep_context_nodes) noexcept;
    static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp *, OrtNodeComputeInfo **infos, size_t n) noexcept;
    OrtStatus *compile_one(const OrtGraph *graph, const OrtNode *fused);
};

struct ComputeInfo : OrtNodeComputeInfo {
    explicit ComputeInfo(Ep &e) : ep(e) {
        ort_version_supported = ORT_API_VERSION;
        CreateState = CreateStateImpl;
        Compute = ComputeImpl;
        ReleaseState = ReleaseStateImpl;
    }
    Ep &ep;
    static OrtStatus *ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo *self, OrtNodeComputeContext *ctx, void **state) {
        Ep &ep = static_cast<ComputeInfo *>(self)->ep;
        auto it = ep.kernels.find(ep.apis.ep.NodeComputeContext_NodeName(ctx));
        if (it == ep.kernels.end())
            return ep.apis.ort.CreateStatus(ORT_EP_FAIL, "aiednn: no compiled partition for this fused node");
        *state = it->second.get();
        return nullptr;
    }
    static OrtStatus *ORT_API_CALL ComputeImpl(OrtNodeComputeInfo *self, void *state, OrtKernelContext *kctx);
    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo *, void *) {}
};

OrtStatus *ComputeInfo::ComputeImpl(OrtNodeComputeInfo *self, void *state, OrtKernelContext *kctx) {
    Ep &ep = static_cast<ComputeInfo *>(self)->ep;
    Kernel &k = *static_cast<Kernel *>(state);
    try {
        Ort::KernelContext ctx(kctx);
        std::vector<aiednn_tensor_t> ins, outs;
        for (size_t i = 0; i < k.in_ids.size(); i++)
            ins.push_back({k.in_ids[i], const_cast<void *>(ctx.GetInput(i).GetTensorRawData())});
        for (auto &c : k.consts)
            ins.push_back({c.first, c.second.bytes.data()});
        for (size_t i = 0; i < k.out_ids.size(); i++)
            outs.push_back({k.out_ids[i], ctx.GetOutput(i, k.out_dims[i]).GetTensorMutableRawData()});
        aiednn_status_t st = aiednn_compiled_partition_execute(k.cp, nullptr, (int)ins.size(), ins.data(),
                                                               (int)outs.size(), outs.data(), nullptr, 0);
        if (st != aiednn_success)
            return ep.apis.ort.CreateStatus(ORT_EP_FAIL,
                                            (std::string("aiednn execute: ") + aiednn_status_str(st)).c_str());
    } catch (const std::exception &e) {
        return ep.apis.ort.CreateStatus(ORT_EP_FAIL, e.what());
    }
    return nullptr;
}

Ep::Ep(Factory &, const Apis &a, bool bench, const OrtLogger &lg) : OrtEp{}, apis(a), benchmark(bench), logger(lg) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

OrtStatus *ORT_API_CALL Ep::GetCapabilityImpl(OrtEp *self, const OrtGraph *ort_graph,
                                              OrtEpGraphSupportInfo *info) noexcept {
    Ep *ep = static_cast<Ep *>(self);
    try {
        Ort::ConstGraph graph{ort_graph};
        GraphMap gm;
        gm.add_nodes(graph.GetNodes(), graph.GetOutputs());
        const int np = aiednn_graph_get_partitions(gm.graph(), nullptr, 0);
        std::vector<aiednn_partition_t> parts(np > 0 ? np : 1);
        aiednn_graph_get_partitions(gm.graph(), parts.data(), np);
        OrtNodeFusionOptions opts = {};
        opts.ort_version_supported = ORT_API_VERSION;
        opts.drop_constant_initializers = true; /* copied at Compile */
        const bool log = getenv("AIEDNN_ORT_EP_LOG") != nullptr;
        if (log)
            fprintf(stderr, "[aiednn-ep] GetCapability: %zu nodes, %d wildcards, %d partitions\n", gm.nodes.size(),
                    gm.wildcards, np);
        for (int p = 0; p < np; p++) {
            if (log)
                fprintf(stderr, "[aiednn-ep]   %c %s\n", aiednn_partition_is_supported(parts[p]) ? '+' : '-',
                        aiednn_partition_info(parts[p]));
            if (!aiednn_partition_is_supported(parts[p]))
                continue;
            const int n = aiednn_partition_get_ops(parts[p], nullptr, 0);
            std::vector<uint64_t> ops(n);
            aiednn_partition_get_ops(parts[p], ops.data(), n);
            std::vector<const OrtNode *> nodes;
            for (uint64_t id : ops)
                nodes.push_back(static_cast<const OrtNode *>(gm.nodes[id - 1]));
            RETURN_IF_ERROR(ep->apis.ep.EpGraphSupportInfo_AddNodesToFuse(info, nodes.data(), nodes.size(), &opts));
        }
    } catch (const std::exception &e) {
        return ep->apis.ort.CreateStatus(ORT_EP_FAIL, e.what());
    }
    return nullptr;
}

/* the fused node's subgraph -> one compiled partition, its ids in kernel-context order */
OrtStatus *Ep::compile_one(const OrtGraph *ort_graph, const OrtNode *ort_fused) {
    Ort::ConstGraph graph{ort_graph};
    Ort::ConstNode fused{ort_fused};
    GraphMap gm;
    gm.add_nodes(graph.GetNodes(), graph.GetOutputs());
    const int np = aiednn_graph_get_partitions(gm.graph(), nullptr, 0);
    std::vector<aiednn_partition_t> parts(np > 0 ? np : 1);
    aiednn_graph_get_partitions(gm.graph(), parts.data(), np);
    if (np != 1 || !aiednn_partition_is_supported(parts[0]))
        return apis.ort.CreateStatus(ORT_EP_FAIL, "aiednn: a fused node's subgraph is not one supported partition");
    auto k = std::make_unique<Kernel>();
    aiednn_status_t st = aiednn_partition_compile(parts[0], &k->cp);
    if (st != aiednn_success)
        return apis.ort.CreateStatus(ORT_EP_FAIL, (std::string("aiednn compile: ") + aiednn_status_str(st)).c_str());
    if (benchmark) { /* time every implementation that accepts the fused problem, pin the fastest */
        aiednn_impl_perf_t perf[32];
        const int n = aiednn_compiled_partition_find(k->cp, AIEDNN_FIND_MEASURE, perf, 32);
        if (n > 1 && perf[0].status == aiednn_success)
            (void)aiednn_compiled_partition_set_impl(k->cp, perf[0].impl);
    }
    for (const auto &v : fused.GetInputs()) { /* activations only: ORT dropped the constant initializers */
        auto it = gm.ids.find(v.GetName());
        if (it != gm.ids.end() && !gm.consts.count(it->second))
            k->in_ids.push_back(it->second);
    }
    for (const auto &v : fused.GetOutputs()) {
        auto it = gm.ids.find(v.GetName());
        if (it == gm.ids.end())
            return apis.ort.CreateStatus(ORT_EP_FAIL, "aiednn: fused output without a logical tensor");
        const aiednn_logical_tensor_t &lt = gm.lts[it->second];
        k->out_ids.push_back(it->second);
        k->out_dims.emplace_back(lt.md.dims, lt.md.dims + lt.md.ndims);
    }
    k->consts = std::move(gm.consts);
    kernels[fused.GetName()] = std::move(k);
    return nullptr;
}

OrtStatus *ORT_API_CALL Ep::CompileImpl(OrtEp *self, const OrtGraph **graphs, const OrtNode **fused, size_t count,
                                        OrtNodeComputeInfo **infos, OrtNode ** /*ep_context_nodes*/) noexcept {
    Ep *ep = static_cast<Ep *>(self);
    try {
        for (size_t i = 0; i < count; i++) {
            RETURN_IF_ERROR(ep->compile_one(graphs[i], fused[i]));
            infos[i] = new ComputeInfo(*ep);
        }
    } catch (const std::exception &e) {
        return ep->apis.ort.CreateStatus(ORT_EP_FAIL, e.what());
    }
    return nullptr;
}

void ORT_API_CALL Ep::ReleaseNodeComputeInfosImpl(OrtEp *, OrtNodeComputeInfo **infos, size_t n) noexcept {
    for (size_t i = 0; i < n; i++)
        delete static_cast<ComputeInfo *>(infos[i]);
}

/* ── factory: the CPU device (AIE work goes through the library's own buffers) ── */
class Factory : public OrtEpFactory {
  public:
    Factory(const char *name, const Apis &a) : OrtEpFactory{}, apis(a), name_(name) {
        ort_version_supported = ORT_API_VERSION;
        GetName = [](const OrtEpFactory *f) noexcept { return static_cast<const Factory *>(f)->name_.c_str(); };
        GetVendor = [](const OrtEpFactory *) noexcept { return "AMD"; };
        GetVendorId = [](const OrtEpFactory *) noexcept { return (uint32_t)0x1022; };
        GetVersion = [](const OrtEpFactory *) noexcept { return "0.1.0"; };
        GetSupportedDevices = GetSupportedDevicesImpl;
        CreateEp = CreateEpImpl;
        ReleaseEp = [](OrtEpFactory *, OrtEp *ep) noexcept { delete static_cast<Ep *>(ep); };
        ValidateCompiledModelCompatibilityInfo = nullptr;
        CreateAllocator = [](OrtEpFactory *, const OrtMemoryInfo *, const OrtKeyValuePairs *,
                             OrtAllocator **alloc) noexcept -> OrtStatus * {
            *alloc = nullptr; /* the default CPU allocator */
            return nullptr;
        };
        ReleaseAllocator = [](OrtEpFactory *, OrtAllocator *) noexcept {};
        CreateDataTransfer = [](OrtEpFactory *, OrtDataTransferImpl **dt) noexcept -> OrtStatus * {
            *dt = nullptr;
            return nullptr;
        };
        IsStreamAware = [](const OrtEpFactory *) noexcept { return false; };
        CreateSyncStreamForDevice = [](OrtEpFactory *, const OrtMemoryDevice *, const OrtKeyValuePairs *,
                                       OrtSyncStreamImpl **s) noexcept -> OrtStatus * {
            *s = nullptr;
            return nullptr;
        };
    }
    Apis apis;

  private:
    std::string name_;
    static OrtStatus *ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory *self, const OrtHardwareDevice *const *devs,
                                                           size_t ndev, OrtEpDevice **out, size_t max_out,
                                                           size_t *n_out) noexcept {
        auto *f = static_cast<Factory *>(self);
        *n_out = 0;
        for (size_t i = 0; i < ndev && *n_out < max_out; i++)
            if (f->apis.ort.HardwareDevice_Type(devs[i]) == OrtHardwareDeviceType_CPU) {
                OrtEpDevice *d = nullptr;
                RETURN_IF_ERROR(f->apis.ep.CreateEpDevice(f, devs[i], nullptr, nullptr, &d));
                out[(*n_out)++] = d;
                break;
            }
        return nullptr;
    }
    static OrtStatus *ORT_API_CALL CreateEpImpl(OrtEpFactory *self, const OrtHardwareDevice *const *,
                                                const OrtKeyValuePairs *const *, size_t, const OrtSessionOptions *so,
                                                const OrtLogger *logger, OrtEp **ep) noexcept {
        auto *f = static_cast<Factory *>(self);
        int has = 0;
        RETURN_IF_ERROR(f->apis.ort.HasSessionConfigEntry(so, "ep.aiednn.benchmark", &has));
        bool bench = false;
        if (has) {
            char buf[8] = {0};
            size_t n = sizeof buf;
            RETURN_IF_ERROR(f->apis.ort.GetSessionConfigEntry(so, "ep.aiednn.benchmark", buf, &n));
            bench = buf[0] == '1';
        }
        *ep = new Ep(*f, f->apis, bench, *logger);
        return nullptr;
    }
};

} // namespace aiednn_ort

extern "C" {
__attribute__((visibility("default"))) OrtStatus *CreateEpFactories(const char *registration_name,
                                                                    const OrtApiBase *base, const OrtLogger *,
                                                                    OrtEpFactory **factories, size_t max_factories,
                                                                    size_t *num_factories) {
    const OrtApi *ort = base->GetApi(ORT_API_VERSION);
    if (!ort)
        return nullptr;
    Ort::InitApi(ort);
    if (max_factories < 1)
        return ort->CreateStatus(ORT_INVALID_ARGUMENT, "aiednn: no room for the EP factory");
    factories[0] = new aiednn_ort::Factory(registration_name, aiednn_ort::Apis{*ort, *ort->GetEpApi()});
    *num_factories = 1;
    return nullptr;
}

__attribute__((visibility("default"))) OrtStatus *ReleaseEpFactory(OrtEpFactory *factory) {
    delete static_cast<aiednn_ort::Factory *>(factory);
    return nullptr;
}
}
