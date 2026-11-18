###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Synthetic graph.json nets for the aiegraph x86 tests (moved from the retired test_keep_layout_codegen.py):
``Net`` builds a graph executor JSON (one output per node: tensor id == node id) and a driver in the
arm_build format; ``opsem`` writes the matching <stem>_opsem.json (op names only, element-conserving
shapes); ``_fa_net`` is the attention head split / merge net with its counter-example switches."""
import numpy as np


class Net:
    """One output per node: tensor id == node id == ``tensors[i]`` of the driver."""

    def __init__(self):
        self.nodes, self.shape, self.sid, self.calls = [], [], [], []

    def const(self, name, shape):
        self.nodes.append({"op": "null", "name": name, "inputs": []})
        self.shape.append(list(shape))
        self.sid.append(len(self.sid) + 100)
        return len(self.nodes) - 1

    def call(self, fn, ins, shape, sid=None, site=None, expand=()):
        """site: driver text of an AIE call site instead of the TVM call; expand: [(tensor, g_params off, n)]"""
        t = len(self.nodes)
        self.nodes.append({"op": "tvm_op", "name": fn, "inputs": [[i, 0, 0] for i in ins],
                           "attrs": {"func_name": "tvmgen_default_fused_" + fn, "num_outputs": "1",
                                     "flatten_data": "0", "num_inputs": str(len(ins))}})
        self.shape.append(list(shape))
        self.sid.append(t if sid is None else sid)
        self.calls.append((fn, list(ins), t, site, list(expand)))
        return t

    def graph(self):
        n = len(self.nodes)
        return {"nodes": self.nodes, "arg_nodes": [i for i, x in enumerate(self.nodes) if x["op"] == "null"],
                "heads": [[n - 1, 0, 0]], "node_row_ptr": list(range(n + 1)),
                "attrs": {"storage_id": ["list_int", self.sid], "shape": ["list_shape", self.shape],
                          "dltype": ["list_str", ["float32"] * n]}}

    def driver(self):
        out = ["/* synthetic arm_build driver */\n"]
        for t, s in enumerate(self.shape):
            for k, d in enumerate(s):
                out.append("    shape_store[%d][%d] = %d;\n" % (t, k, d))
        out.append("float *graph_run(void) {\n")
        ws = 0
        for k, (fn, ins, o, site, expand) in enumerate(self.calls):
            for t, off, cnt in expand:
                out.append("    graph_bf16_expand((float *)(g_wscratch + %d), (const uint16_t *)(g_params + %d), "
                           "%dULL);\n    tensors[%d].data = g_wscratch + %d;\n" % (ws, off, cnt, t, ws))
                ws += 4 * cnt
            for i, t in enumerate(ins + [o]):
                out.append("    args[%d].v_handle = &tensors[%d];\n" % (i, t))
            if site:
                out.append("    GRAPH_PROF_BEGIN(%d);\n    UP_AIE_BEGIN(%d);\n    %s\n    UP_AIE_END(%d);\n"
                           "    GRAPH_PROF_END(%d, \"aie %s\");\n" % (k, k, site, k, k, site.split("(")[0]))
            else:
                out.append("    GRAPH_PROF_BEGIN(%d);\n    tvmgen_default_fused_%s(args, codes, %d, 0, 0, 0);\n"
                           "    GRAPH_PROF_END(%d, \"cpu %s\");\n" % (k, fn, len(ins) + 1, k, fn))
        out.append("    return 0;\n}\n")
        return "".join(out)


# Relay op sequence of each synthetic fused function (what deploy_flow.build_c records in <stem>_opsem.json)
FN_OPS = {
    "reshape_transpose_multiply_reshape": ["reshape", "transpose", "multiply", "reshape"],
    "reshape_transpose_multiply_reshape_transpose": ["reshape", "transpose", "multiply", "reshape", "transpose"],
    "reshape_transpose_reshape_transpose": ["reshape", "transpose", "reshape", "transpose"],
    "reshape_transpose_reshape": ["reshape", "transpose", "reshape"],
    "reshape_exp_transpose": ["reshape", "exp", "transpose"],
    "sigmoid_squeeze_multiply_layout_transform": ["sigmoid", "squeeze", "multiply", "layout_transform"],
    "layout_transform_transpose_reshape": ["layout_transform", "transpose", "reshape"],
    "layout_transform_reshape": ["layout_transform", "reshape"],
    "reshape_add_reshape_transpose_layout_transform": ["reshape", "add", "reshape", "transpose", "layout_transform"],
}


def opsem(net) -> dict:
    """<stem>_opsem.json for a synthetic Net: element-conserving shapes (n = output elements), no axes attr."""
    nodes = {}
    for t, nd in enumerate(net.nodes):
        if nd["op"] != "tvm_op":
            continue
        fn = nd["attrs"]["func_name"].replace("tvmgen_default_fused_", "")
        n = int(np.prod(net.shape[t])) if net.shape[t] else 1
        ops = []
        for o in FN_OPS.get(fn, ["nn." + fn]):
            two = o in ("multiply", "add") or o.startswith("nn.")
            ops.append({"op": o, "ins": [[n], [1]] if two else [[n]], "out": [n], "attrs": {}})
        nodes[str(t)] = {"func_name": nd["attrs"]["func_name"], "name_ok": True, "ops": ops,
                         "params": [net.shape[i] for i, _, _ in nd["inputs"]], "outs": [net.shape[t]]}
    return {"version": 1, "nodes": nodes}


def _bf16(x):
    return (np.array([x], np.float32).view(np.uint32)[0] >> 16).astype(np.uint16)


def _fa_net(H=2, N=4, M=4, D=2, clobber=False, outside_reader=False, impure=False):
    """linear outs Q/K/V [1,R,H*D] -> head split Q, K (x scale), V^T -> QK site -> softmax -> PV site -> head merge"""
    C = H * D
    n = Net()
    x = n.const("x", [1, N, C])
    sq = n.const("p_scale_q", [])
    sk = n.const("p_scale_k", [])
    ql = n.call("nn_dense", [x], [1, N, C])
    kl = n.call("nn_dense_1", [x], [1, M, C])
    vl = n.call("nn_dense_2", [x], [1, M, C])
    tq = n.call("reshape_exp_transpose" if impure else "reshape_transpose_multiply_reshape", [ql, sq], [H, N, D],
                expand=[(sq, 0, 1)])
    tk = n.call("reshape_transpose_multiply_reshape_transpose", [kl, sk], [H, M, D], expand=[(sk, 2, 1)])
    ts = n.call("nn_batch_matmul", [tq, tk], [H, N, M], site="unet_aie_attn(7, args, 3, 0, 0);")
    if outside_reader:
        n.call("nn_relu", [tq], [H, N, D])
    tp = n.call("nn_softmax", [ts], [H, N, M])
    tv = n.call("reshape_transpose_reshape_transpose", [vl], [H, D, M])
    to = n.call("nn_batch_matmul_1", [tp, tv], [H, N, D], site="unet_aie_attn(8, args, 3, 1, 0);")
    o2_sid = 900
    if clobber:   # a kept call between PV and the head merge writes the merged O buffer
        n.call("nn_relu_1", [vl], [1, N, C], sid=o2_sid)
    o2 = n.call("reshape_transpose_reshape", [to], [1, N, C], sid=o2_sid)
    n.call("nn_dense_3", [o2], [1, N, C])
    w = np.zeros(64, np.uint16)
    w[0] = _bf16(0.5)
    w[1] = _bf16(0.5)
    return n, w, dict(ql=ql, kl=kl, vl=vl, o2=o2)
