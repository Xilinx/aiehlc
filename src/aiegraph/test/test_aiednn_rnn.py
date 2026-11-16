###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiednn_rnn_forward_create (RNN / LSTM / GRU) against PyTorch and a numpy GRU recurrence."""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, f, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
ARG.update({"src_iter": 12, "src_iter_c": 13, "weights_iter": 14, "dst_iter": 15, "dst_iter_c": 16})
DIRS = {"left2right": 0, "right2left": 1, "bi_concat": 2, "bi_sum": 3}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_rnn"))


def torch_layout(mod, L, D, G, H, gru):
    """nn.RNN/LSTM/GRU parameters -> aiednn [L, D, I, G, H] weights, [L, D, Gb, H] bias"""
    order = [1, 0, 2] if gru else list(range(G))      # torch GRU gates r, z, n -> aiednn u(z), r, o(n)
    Ws, Us, Bs = [], [], []
    for l in range(L):
        for d in range(D):
            sfx = f"_l{l}" + ("_reverse" if d else "")
            wih = getattr(mod, "weight_ih" + sfx).detach().double().reshape(G, H, -1)[order]   # [G, H, I]
            whh = getattr(mod, "weight_hh" + sfx).detach().double().reshape(G, H, H)[order]
            bih = getattr(mod, "bias_ih" + sfx).detach().double().reshape(G, H)[order]
            bhh = getattr(mod, "bias_hh" + sfx).detach().double().reshape(G, H)[order]
            Ws.append(wih.permute(2, 0, 1))
            Us.append(whh.permute(2, 0, 1))
            if gru:   # u, r: summed; o: b_in; extra: b_hn (multiplied by r)
                Bs.append(torch.cat([bih[:2] + bhh[:2], bih[2:3], bhh[2:3]]))
            else:
                Bs.append(bih + bhh)
    I = Ws[0].shape[0]
    W = torch.stack(Ws).reshape(L, D, I, G, H)
    U = torch.stack(Us).reshape(L, D, H, G, H)
    B = torch.stack(Bs).reshape(L, D, -1, H)
    return W.float().numpy(), U.float().numpy(), B.float().numpy()


def run(lib, cell, dirn, alpha, x, W, U, B, h0, c0, H, L, D):
    T, N, _ = x.shape
    DLC = 2 * H if dirn == "bi_concat" else H
    y = np.zeros((T, N, DLC), np.float32)
    hT = np.zeros((L, D, N, H), np.float32)
    cT = np.zeros((L, D, N, H), np.float32) if c0 is not None else None
    bufs = {ARG["src"]: x, ARG["weights"]: W, ARG["weights_iter"]: U, ARG["bias"]: B, ARG["dst"]: y,
            ARG["src_iter"]: h0, ARG["dst_iter"]: hT}
    if c0 is not None:
        bufs[ARG["src_iter_c"]], bufs[ARG["dst_iter_c"]] = c0, cT
    m = lambda a: C.byref(md(a.shape, "f32"))
    got, impl = lib.call("aiednn_rnn_forward_create",
                         [C.c_int(lib.ALG[cell]), C.c_int(DIRS[dirn]), f(alpha), m(x), m(h0),
                          m(c0) if c0 is not None else None, m(W), m(U), m(B), m(y), m(hT),
                          m(cT) if cT is not None else None], bufs, ARG["dst"])
    assert impl.startswith("gemm+rnn+") and "aie:" in impl, impl   # projection / recurrent GEMMs on the AIE
    return got, hT, cT


@pytest.mark.parametrize("kind", ["lstm", "gru", "tanh", "relu"])
@pytest.mark.parametrize("L,bi", [(1, False), (2, False), (2, True)])
def test_rnn_vs_torch(lib, kind, L, bi):
    torch.manual_seed(3)
    T, N, I, H = 5, 3, 6, 6
    D = 2 if bi else 1
    I = 2 * H if (bi and L > 1) else H   # layer > 0 reads DLC: keep SLC = DLC
    mod = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}.get(kind, torch.nn.RNN)
    kw = {} if kind in ("lstm", "gru") else {"nonlinearity": kind}
    net = mod(I, H, num_layers=L, bidirectional=bi, **kw).double()
    x = torch.randn(T, N, I, dtype=torch.float64).float().double()
    h0 = torch.randn(L * D, N, H, dtype=torch.float64).float().double()
    c0 = torch.randn(L * D, N, H, dtype=torch.float64).float().double()
    G = {"lstm": 4, "gru": 3}.get(kind, 1)
    for prm in net.parameters():   # f32-representable parameters, so both sides see the same numbers
        prm.data = prm.data.float().double()
    W, U, B = torch_layout(net, L, D, G, H, kind == "gru")
    with torch.no_grad():
        want, out = net(x, (h0, c0) if kind == "lstm" else h0)
    cell = {"lstm": "rnn_lstm", "gru": "rnn_lbr_gru", "tanh": "rnn_vanilla_tanh", "relu": "rnn_vanilla_relu"}[kind]
    got, hT, cT = run(lib, cell, "bi_concat" if bi else "left2right", 0.0, x.float().numpy(), W, U, B,
                      h0.float().numpy().reshape(L, D, N, H), c0.float().numpy().reshape(L, D, N, H) if kind == "lstm" else None,
                      H, L, D)
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)
    hn = out[0] if kind == "lstm" else out
    np.testing.assert_allclose(hT.reshape(L * D, N, H), hn.numpy(), rtol=1e-5, atol=1e-5)
    if kind == "lstm":
        np.testing.assert_allclose(cT.reshape(L * D, N, H), out[1].numpy(), rtol=1e-5, atol=1e-5)


def test_rnn_right2left_and_bi_sum(lib):
    torch.manual_seed(4)
    T, N, H = 4, 2, 5
    net = torch.nn.LSTM(H, H, bidirectional=True).double()
    for prm in net.parameters():
        prm.data = prm.data.float().double()
    x = torch.randn(T, N, H, dtype=torch.float64).float().double()
    W, U, B = torch_layout(net, 1, 2, 4, H, False)
    with torch.no_grad():
        want = net(x)[0]
    z = np.zeros((1, 2, N, H), np.float32)
    got, _, _ = run(lib, "rnn_lstm", "bi_sum", 0.0, x.float().numpy(), W, U, B, z, z.copy(), H, 1, 2)
    np.testing.assert_allclose(got, (want[..., :H] + want[..., H:]).numpy(), rtol=1e-5, atol=1e-5)
    got, _, _ = run(lib, "rnn_lstm", "right2left", 0.0, x.float().numpy(), W[:, 1:], U[:, 1:], B[:, 1:], z[:, :1],
                    z[:, :1].copy(), H, 1, 1)
    np.testing.assert_allclose(got, want[..., H:].numpy(), rtol=1e-5, atol=1e-5)


def test_gru_reset_before_matmul(lib):
    """aiednn_rnn_gru: o = tanh(W_o x + U_o (r * h) + b_o), checked against a numpy recurrence"""
    rng = np.random.default_rng(9)
    T, N, I, H = 4, 2, 3, 4
    W = rng.normal(0, 0.5, (1, 1, I, 3, H)).astype(np.float32)
    U = rng.normal(0, 0.5, (1, 1, H, 3, H)).astype(np.float32)
    B = rng.normal(0, 0.5, (1, 1, 3, H)).astype(np.float32)
    x = rng.normal(0, 1, (T, N, I)).astype(np.float32)
    h0 = rng.normal(0, 1, (1, 1, N, H)).astype(np.float32)
    sig = lambda v: 1 / (1 + np.exp(-v))
    h, ys = h0[0, 0].astype(np.float64), []
    for t in range(T):
        xw = np.einsum("ni,igh->ngh", x[t], W[0, 0]) + B[0, 0]
        u = sig(xw[:, 0] + h @ U[0, 0, :, 0])
        r = sig(xw[:, 1] + h @ U[0, 0, :, 1])
        o = np.tanh(xw[:, 2] + (r * h) @ U[0, 0, :, 2])
        h = u * h + (1 - u) * o
        ys.append(h)
    got, hT, _ = run(lib, "rnn_gru", "left2right", 0.0, x, W, U, B, h0, None, H, 1, 1)
    np.testing.assert_allclose(got, np.stack(ys), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hT[0, 0], h, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cell", ["rnn_augru", "rnn_lbr_augru"])
@pytest.mark.parametrize("dirn", ["left2right", "bi_concat"])
@pytest.mark.parametrize("path", ["gemm+rnn", "ref:"])
def test_augru(lib, cell, dirn, path):
    """AUGRU / linear-before-reset AUGRU (u' = (1 - a[t, n]) u) against a numpy recurrence"""
    rng = np.random.default_rng(12)
    lbr = cell == "rnn_lbr_augru"
    D = 2 if dirn == "bi_concat" else 1
    T, N, I, H = 5, 3, 4, 6
    W = rng.normal(0, 0.5, (1, D, I, 3, H)).astype(np.float32)
    U = rng.normal(0, 0.5, (1, D, H, 3, H)).astype(np.float32)
    B = rng.normal(0, 0.5, (1, D, 3 + lbr, H)).astype(np.float32)
    x = rng.normal(0, 1, (T, N, I)).astype(np.float32)
    h0 = rng.normal(0, 1, (1, D, N, H)).astype(np.float32)
    att = rng.uniform(0, 1, (T, N)).astype(np.float32)
    sig = lambda v: 1 / (1 + np.exp(-v))
    want = np.zeros((T, N, D * H))
    hT = np.zeros((1, D, N, H))
    for d in range(D):
        h = h0[0, d].astype(np.float64)
        for t in (range(T - 1, -1, -1) if d else range(T)):
            xw = np.einsum("ni,igh->ngh", x[t], W[0, d]) + B[0, d, :3]
            u = sig(xw[:, 0] + h @ U[0, d, :, 0]) * (1 - att[t])[:, None]
            r = sig(xw[:, 1] + h @ U[0, d, :, 1])
            o = np.tanh(xw[:, 2] + (r * (h @ U[0, d, :, 2] + B[0, d, 3]) if lbr else (r * h) @ U[0, d, :, 2]))
            h = u * h + (1 - u) * o
            want[t, :, d * H:(d + 1) * H] = h
        hT[0, d] = h
    y, hTo = np.zeros((T, N, D * H), np.float32), np.zeros((1, D, N, H), np.float32)
    m = lambda a: C.byref(md(a.shape, "f32"))
    lib.set_filter(None if path == "gemm+rnn" else path)
    try:
        got, impl = lib.call("aiednn_augru_forward_create",
                             [C.c_int(lib.ALG[cell]), C.c_int(DIRS[dirn]), m(x), m(h0), m(att), m(W), m(U), m(B), m(y),
                              m(hTo)],
                             {ARG["src"]: x, ARG["src_iter"]: h0, 39: att, ARG["weights"]: W, ARG["weights_iter"]: U,
                              ARG["bias"]: B, ARG["dst"]: y, ARG["dst_iter"]: hTo}, ARG["dst"])
    finally:
        lib.set_filter(None)
    assert impl.startswith(path), impl
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hTo, hT, rtol=1e-5, atol=1e-5)


def test_augru_refuses_plain_create_and_backward(lib):
    T, N, I, H = 2, 1, 3, 4
    a = lambda *s: C.byref(md(s, "f32"))
    prim, st = lib.create("aiednn_rnn_forward_create", C.c_int(lib.ALG["rnn_augru"]), C.c_int(0), f(0.0), a(T, N, I),
                          None, None, a(1, 1, I, 3, H), a(1, 1, H, 3, H), None, a(T, N, H), None, None)
    assert st != 0                                     # attention is not optional: aiednn_augru_forward_create
    prim, st = lib.create("aiednn_augru_forward_create", C.c_int(lib.ALG["rnn_gru"]), C.c_int(0), a(T, N, I), None,
                          a(T, N), a(1, 1, I, 3, H), a(1, 1, H, 3, H), None, a(T, N, H), None)
    assert st != 0                                     # not an augru cell


# ---------------------------------------------------------------- backward (BPTT) vs torch autograd
ARG.update({"diff_src": 20, "diff_weights": 21, "diff_bias": 22, "diff_dst": 23, "diff_src_iter": 32,
            "diff_src_iter_c": 33, "diff_weights_iter": 34, "diff_dst_iter": 35, "diff_dst_iter_c": 36})


def grad_layout(net, L, D, G, H, gru):
    """parameter gradients in aiednn's layout; a summed bias's gradient is either half's (they are equal)"""
    order = [1, 0, 2] if gru else list(range(G))
    Ws, Us, Bs = [], [], []
    for l in range(L):
        for d in range(D):
            sfx = f"_l{l}" + ("_reverse" if d else "")
            g = lambda n: getattr(net, n + sfx).grad.double()
            Ws.append(g("weight_ih").reshape(G, H, -1)[order].permute(2, 0, 1))
            Us.append(g("weight_hh").reshape(G, H, H)[order].permute(2, 0, 1))
            bih, bhh = g("bias_ih").reshape(G, H)[order], g("bias_hh").reshape(G, H)[order]
            Bs.append(torch.cat([bih, bhh[2:3]]) if gru else bih)
    I = Ws[0].shape[0]
    return (torch.stack(Ws).reshape(L, D, I, G, H).numpy(), torch.stack(Us).reshape(L, D, H, G, H).numpy(),
            torch.stack(Bs).reshape(L, D, -1, H).numpy())


class RnnDescs(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("sl", "si", "sc", "wl", "wi", "b", "dl", "di", "dc")]


def descs(arrs):
    """aiednn_rnn_descs_t over 9 arrays (None = absent); keeps the md structs alive on the object"""
    mds = [md(a.shape, "f32") if a is not None else None for a in arrs]
    s = RnnDescs(*[C.cast(C.pointer(m), C.c_void_p) if m is not None else None for m in mds])
    s._keep = mds
    return s


@pytest.mark.parametrize("kind", ["lstm", "gru", "tanh", "relu"])
@pytest.mark.parametrize("L,bi", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("path", ["gemm+rnnbwd", "ref:"])
def test_rnn_backward_vs_autograd(lib, kind, L, bi, path):
    torch.manual_seed(5)
    T, N, H = 4, 2, 5
    D = 2 if bi else 1
    I = 2 * H if (bi and L > 1) else H
    G = {"lstm": 4, "gru": 3}.get(kind, 1)
    mod = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}.get(kind, torch.nn.RNN)
    kw = {} if kind in ("lstm", "gru") else {"nonlinearity": kind}
    net = mod(I, H, num_layers=L, bidirectional=bi, **kw).double()
    for prm in net.parameters():
        prm.data = prm.data.float().double()
    r = lambda *s: torch.randn(*s, dtype=torch.float64).float().double()
    x, h0, c0 = r(T, N, I).requires_grad_(), r(L * D, N, H).requires_grad_(), r(L * D, N, H).requires_grad_()
    lstm = kind == "lstm"
    y, out = net(x, (h0, c0) if lstm else h0)
    hn, cn = (out[0], out[1]) if lstm else (out, None)
    gy, ghn, gcn = r(*y.shape), r(*hn.shape), r(L * D, N, H)
    loss = (y * gy).sum() + (hn * ghn).sum() + ((cn * gcn).sum() if lstm else 0)
    loss.backward()
    W, U, B = torch_layout(net, L, D, G, H, kind == "gru")
    dW, dU, dB = grad_layout(net, L, D, G, H, kind == "gru")
    a32 = lambda t: t.detach().float().numpy().reshape(L, D, N, H) if t.dim() == 3 and t.shape[0] == L * D else t.detach().float().numpy()
    xs, h0s, c0s = x.detach().float().numpy(), a32(h0), a32(c0) if lstm else None
    ys = y.detach().float().numpy()
    hT = np.zeros((L, D, N, H), np.float32)
    gx, gh0 = np.zeros_like(xs), np.zeros_like(h0s)
    gc0 = np.zeros_like(c0s) if lstm else None
    gW, gU, gB = np.zeros_like(W), np.zeros_like(U), np.zeros_like(B)
    fwd = descs([xs, h0s, c0s, W, U, B, ys, hT, hT if lstm else None])
    dif = descs([gx, gh0, gc0, gW, gU, gB, ys, hT, hT if lstm else None])
    bufs = {ARG["src"]: xs, ARG["src_iter"]: h0s, ARG["weights"]: W, ARG["weights_iter"]: U, ARG["bias"]: B,
            ARG["diff_src"]: gx, ARG["diff_src_iter"]: gh0, ARG["diff_weights"]: gW, ARG["diff_weights_iter"]: gU,
            ARG["diff_bias"]: gB, ARG["diff_dst"]: gy.float().numpy(),
            ARG["diff_dst_iter"]: ghn.float().numpy().reshape(L, D, N, H)}
    if lstm:
        bufs.update({ARG["src_iter_c"]: c0s, ARG["diff_src_iter_c"]: gc0,
                     ARG["diff_dst_iter_c"]: gcn.float().numpy().reshape(L, D, N, H)})
    cell = {"lstm": "rnn_lstm", "gru": "rnn_lbr_gru", "tanh": "rnn_vanilla_tanh", "relu": "rnn_vanilla_relu"}[kind]
    lib.set_filter("ref:" if path == "ref:" else None)
    try:
        _, impl = lib.call("aiednn_rnn_backward_create",
                           [C.c_int(lib.ALG[cell]), C.c_int(DIRS["bi_concat" if bi else "left2right"]), f(0.0),
                            C.byref(fwd), C.byref(dif)], bufs, ARG["diff_src"])
    finally:
        lib.set_filter(None)
    if path == "ref:":
        assert impl == "ref:rnn", impl
    else:   # BPTT's GEMMs on the AIE float GEMM
        assert impl.startswith("gemm+rnnbwd+") and "aie:" in impl, impl
    tol = dict(rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gx, x.grad.numpy(), **tol)
    np.testing.assert_allclose(gh0.reshape(L * D, N, H), h0.grad.numpy(), **tol)
    if lstm:
        np.testing.assert_allclose(gc0.reshape(L * D, N, H), c0.grad.numpy(), **tol)
    np.testing.assert_allclose(gW, dW, **tol)
    np.testing.assert_allclose(gU, dU, **tol)
    np.testing.assert_allclose(gB, dB, **tol)


def test_gru_backward_finite_difference(lib):
    """aiednn_rnn_gru (reset before U_o, no torch counterpart): gradients by central differences"""
    rng = np.random.default_rng(11)
    T, N, I, H = 3, 2, 3, 4
    W = rng.normal(0, 0.5, (1, 1, I, 3, H)).astype(np.float32)
    U = rng.normal(0, 0.5, (1, 1, H, 3, H)).astype(np.float32)
    B = rng.normal(0, 0.5, (1, 1, 3, H)).astype(np.float32)
    x = rng.normal(0, 1, (T, N, I)).astype(np.float32)
    h0 = rng.normal(0, 1, (1, 1, N, H)).astype(np.float32)
    gy = rng.normal(0, 1, (T, N, H)).astype(np.float32)

    def loss(W, U, B, x, h0):
        sig = lambda v: 1 / (1 + np.exp(-v))
        h, tot = h0[0, 0].astype(np.float64), 0.0
        for t in range(T):
            a = np.einsum("ni,igh->ngh", x[t], W[0, 0]) + B[0, 0]
            u = sig(a[:, 0] + h @ U[0, 0, :, 0]); rr = sig(a[:, 1] + h @ U[0, 0, :, 1])
            o = np.tanh(a[:, 2] + (rr * h) @ U[0, 0, :, 2])
            h = u * h + (1 - u) * o
            tot += (h * gy[t]).sum()
        return tot

    gx, gh0, gW, gU, gB = (np.zeros_like(a) for a in (x, h0, W, U, B))
    ys = np.zeros((T, N, H), np.float32)
    fwd = descs([x, h0, None, W, U, B, ys, None, None])
    dif = descs([gx, gh0, None, gW, gU, gB, ys, None, None])
    bufs = {ARG["src"]: x, ARG["src_iter"]: h0, ARG["weights"]: W, ARG["weights_iter"]: U, ARG["bias"]: B,
            ARG["diff_src"]: gx, ARG["diff_src_iter"]: gh0, ARG["diff_weights"]: gW, ARG["diff_weights_iter"]: gU,
            ARG["diff_bias"]: gB, ARG["diff_dst"]: gy}
    lib.call("aiednn_rnn_backward_create", [C.c_int(lib.ALG["rnn_gru"]), C.c_int(0), f(0.0), C.byref(fwd), C.byref(dif)],
             bufs, ARG["diff_src"])
    args = [W, U, B, x, h0]
    for k, g in enumerate([gW, gU, gB, gx, gh0]):
        num = np.zeros(g.size)
        for i in range(g.size):
            p, m = [a.astype(np.float64) for a in args], [a.astype(np.float64) for a in args]
            p[k].reshape(-1)[i] += 1e-6
            m[k].reshape(-1)[i] -= 1e-6
            num[i] = (loss(*p) - loss(*m)) / 2e-6
        np.testing.assert_allclose(g.reshape(-1), num, rtol=1e-3, atol=1e-4)
