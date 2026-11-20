###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Group-level x86 check (design 2026-11-16-aiegraph-group-offload §6 "x86 check", step S4).

A group's steps only agree with TVM *together*, so the check runs the group, not the
network:

reference   the group's TVM C functions (``layers/NN_op/NN_op.c``, the code the board
            ELF runs) chained in their original order on x86, fed the group's boundary
            inputs and its real constants from ``params.bin`` (``tvm_chain``);
candidate   the group's steps in their placed order on x86 (a callable: boundary
            inputs -> group outputs; for a layer offload, the kernel's ``<entry>_ref``);
verdict     per group output: ``nonfinite == 0`` (candidate AND reference) and
            ``cos >= 0.999`` (``COS_MIN``); ``maxabs`` is printed, never judged. With
            ``golden=`` the TVM reference is also held to the golden output (real data:
            the reference itself is right on it).

A failure raises ``AiegraphError("group-check", ...)`` naming the group, tensor and the
failing number -- there is no flag that turns a failed check into a pass.

``layer_checker()`` is the same check as ``flow.offload(checker=...)``'s gates 3-4 for a
layer offload (every call of the layer = a one-call group): it replaces calibrate
(bit-exact, adversarial inputs) and whole-network emulation, both of which a large
fp32 network cannot afford and the bf16 AIE numerics cannot meet bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .errors import AiegraphError

__all__ = ["COS_MIN", "TensorCheck", "GroupCheckReport", "compare", "tvm_chain",
           "group_check", "layer_checker"]

_STAGE = "group-check"
COS_MIN = 0.999

_NP = {"int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2", "int32": "<i4",
       "uint32": "<u4", "int64": "<i8", "uint64": "<u8", "float16": "<f2",
       "float32": "<f4", "float64": "<f8"}
_DL = {"int": 0, "uint": 1, "float": 2}


def _np_dtype(dtype: str):
    import numpy as np
    if dtype not in _NP:
        raise AiegraphError(_STAGE, f"tensor dtype {dtype!r} not supported")
    return np.dtype(_NP[dtype])


def _dl(dtype: str) -> tuple:
    kind = dtype.rstrip("0123456789")
    return _DL[kind], int(dtype[len(kind):])


# ═══════════════════════════════════════════════════════════════════════════
#  the verdict
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TensorCheck:
    name: str
    against: str                # "tvm" (candidate vs TVM chain) | "golden" (TVM chain vs golden)
    elems: int
    cos: float
    nonfinite: int              # candidate side
    ref_nonfinite: int          # reference side
    maxabs: float
    cos_min: float = COS_MIN
    error: str = ""

    @property
    def ok(self) -> bool:
        return (not self.error and self.nonfinite == 0 and self.ref_nonfinite == 0
                and math.isfinite(self.cos) and self.cos >= self.cos_min)

    def summary(self) -> str:
        if self.error:
            return f"{self.name} vs {self.against}: ERROR {self.error}"
        return (f"{self.name} vs {self.against}: cos={self.cos:.6f} (min {self.cos_min}) "
                f"nonfinite={self.nonfinite} ref_nonfinite={self.ref_nonfinite} "
                f"maxabs={self.maxabs:.4g} elems={self.elems:,} -> {'PASS' if self.ok else 'FAIL'}")


def compare(name: str, ref, got, cos_min: float = COS_MIN, against: str = "tvm") -> TensorCheck:
    """``got`` (candidate) against ``ref``: nonfinite counts, cosine, max |d| (printed only)."""
    import numpy as np

    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.size != got.size:
        return TensorCheck(name, against, int(ref.size), float("nan"), 0, 0, float("nan"), cos_min,
                           error=f"{got.size:,} elements, want {ref.size:,} (shape {got.shape} "
                                 f"vs {ref.shape})")
    r = ref.astype(np.float64).ravel()
    g = got.astype(np.float64).ravel()
    bad_g, bad_r = ~np.isfinite(g), ~np.isfinite(r)
    nf, rnf = int(bad_g.sum()), int(bad_r.sum())
    fin = ~(bad_g | bad_r)
    d = np.abs(r[fin] - g[fin])
    maxabs = float(d.max()) if d.size else 0.0
    if nf or rnf:
        cos = float("nan")
    else:
        nr, ng = float(np.linalg.norm(r)), float(np.linalg.norm(g))
        if nr == 0.0 and ng == 0.0:
            cos = 1.0
        elif nr == 0.0 or ng == 0.0:
            cos = 0.0
        else:
            cos = float(np.dot(r, g) / (nr * ng))
    return TensorCheck(name, against, int(r.size), cos, nf, rnf, maxabs, cos_min)


@dataclass
class GroupCheckReport:
    group: str
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failed(self) -> list:
        return [c for c in self.checks if not c.ok]

    def __str__(self) -> str:
        head = f"GROUPCHECK {self.group}: {'PASS' if self.ok else 'FAIL'} ({len(self.checks)} check(s))"
        return "\n".join([head] + [f"  {c.summary()}" for c in self.checks])

    def raise_if_failed(self) -> "GroupCheckReport":
        if not self.ok:
            bad = self.failed[0] if self.failed else None
            raise AiegraphError(_STAGE, f"group {self.group}: "
                                + (bad.summary() if bad else "nothing was checked"),
                                hint="the group's steps disagree with its TVM calls on x86; "
                                     "fix the step (layout, scale, overwritten buffer), do not "
                                     "skip the check")
        return self


# ═══════════════════════════════════════════════════════════════════════════
#  reference: the group's TVM C chained on x86
# ═══════════════════════════════════════════════════════════════════════════

_CHAIN_C = r"""
/* Generated by aiegraph/groupcheck.py (tvm_chain) -- do not edit.
 * argv: plan.txt inputs.bin outputs.bin
 * plan.txt: T, per tensor "init code bits ndim d0..d{ndim-1} nbytes";
 *           C, per call "fn nargs t0..t{nargs-1}"; O, then O tensor indices to dump. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tvm_graph_types.h"

typedef int32_t (*tvm_fn_t)(void *, int32_t *, int32_t, void *, int32_t *, void *);
@DECLS@
static tvm_fn_t g_fns[] = {@FNS@};
void tvm_shim_workspace_init(void *buf, uint64_t bytes);

#define MAXT 256
#define MAXA 64
#define MAXD 8
static int64_t shape[MAXT][MAXD];
static int code[MAXT], bits[MAXT], ndim[MAXT], init[MAXT];
static uint64_t nbytes[MAXT];
static void *buf[MAXT];

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: plan inputs outputs\n"); return 2; }
    FILE *p = fopen(argv[1], "r"), *in = fopen(argv[2], "rb");
    int nt = 0;
    if (!p || !in || fscanf(p, "%d", &nt) != 1 || nt > MAXT) return 3;
    for (int i = 0; i < nt; i++) {
        if (fscanf(p, "%d %d %d %d", &init[i], &code[i], &bits[i], &ndim[i]) != 4 || ndim[i] > MAXD) return 4;
        for (int k = 0; k < ndim[i]; k++)
            if (fscanf(p, "%lld", (long long *)&shape[i][k]) != 1) return 5;
        if (fscanf(p, "%llu", (unsigned long long *)&nbytes[i]) != 1) return 6;
        buf[i] = aligned_alloc(64, (nbytes[i] + 63) / 64 * 64);
        if (!buf[i]) return 7;
        if (init[i]) {
            if (fread(buf[i], 1, nbytes[i], in) != nbytes[i]) return 8;
        } else {
            memset(buf[i], 0xA5, nbytes[i]); /* poison: an element no call writes cannot pass */
        }
    }
    static uint8_t ws[256u << 20];
    tvm_shim_workspace_init(ws, sizeof ws);
    int nc = 0;
    if (fscanf(p, "%d", &nc) != 1) return 9;
    for (int c = 0; c < nc; c++) {
        int fn = 0, na = 0, t[MAXA];
        if (fscanf(p, "%d %d", &fn, &na) != 2 || na > MAXA) return 10;
        DLTensor dl[MAXA]; TVMValue v[MAXA]; int32_t codes[MAXA];
        for (int a = 0; a < na; a++) {
            if (fscanf(p, "%d", &t[a]) != 1 || t[a] < 0 || t[a] >= nt) return 11;
            memset(&dl[a], 0, sizeof dl[a]);
            dl[a].data = buf[t[a]]; dl[a].device.device_type = 1; dl[a].ndim = ndim[t[a]];
            dl[a].dtype.code = (uint8_t)code[t[a]]; dl[a].dtype.bits = (uint8_t)bits[t[a]];
            dl[a].dtype.lanes = 1; dl[a].shape = shape[t[a]];
            v[a].v_handle = &dl[a]; codes[a] = 7;
        }
        int rc = g_fns[fn](v, codes, na, 0, 0, 0);
        printf("TVMCHAIN call %d fn %d rc=%d\n", c, fn, rc);
        if (rc) return 12;
    }
    int no = 0;
    FILE *o = fopen(argv[3], "wb");
    if (!o || fscanf(p, "%d", &no) != 1) return 13;
    for (int k = 0; k < no; k++) {
        int t = 0;
        if (fscanf(p, "%d", &t) != 1) return 14;
        fwrite(buf[t], 1, nbytes[t], o);
    }
    fclose(o);
    printf("TVMCHAIN OK calls=%d\n", nc);
    return 0;
}
"""


class _Params:
    """Graph constants straight from ``<stem>_params.bin`` (memory-mapped: can be > 1 GB)."""

    def __init__(self, params_path: Path):
        import numpy as np
        ab = _env.frontend("arm_build")
        self.path = Path(params_path)
        self.offsets = ab.param_offsets(self.path) if self.path.is_file() else {}
        self._mm = np.memmap(self.path, np.uint8, "r") if self.path.is_file() else None

    def __contains__(self, name) -> bool:
        return name in self.offsets

    def array(self, name: str, dtype: str, shape):
        import numpy as np
        off, nb = self.offsets[name]
        dt = _np_dtype(dtype)
        return np.frombuffer(self._mm, dt, nb // dt.itemsize, off).reshape(tuple(int(d) for d in shape))


def _layer_sources(workdir: Path, symbols) -> list:
    """The stage-5 layer C files that define *symbols*."""
    import json
    man = workdir / "layers" / "manifest.json"
    by_sym = {}
    if man.is_file():
        for e in json.loads(man.read_text()).get("layers", []):
            by_sym[e.get("symbol")] = workdir / "layers" / e["path"]
    out = []
    for s in symbols:
        p = by_sym.get(s)
        if p is None or not p.is_file():
            hits = [q for q in sorted((workdir / "layers").glob("*/*.c"))
                    if f" {s}(void* args" in q.read_text(errors="replace")]
            p = hits[0] if len(hits) == 1 else None
        if p is None:
            raise AiegraphError(_STAGE, f"no stage-5 layer C defines {s}",
                                hint="convert the model first; layers/manifest.json names every layer")
        if p not in out:
            out.append(p)
    return out


def tvm_chain(workdir, calls, feeds: dict, outputs, work=None, cg=None) -> dict:
    """Run TVM calls *calls* (graph.json execution order) on x86, in that order.

    *feeds*: ``{tensor id: array}`` for every tensor the calls read that no call of
    the chain produces and that is not a parameter in ``params.bin`` (boundary
    activations, graph inputs). Parameters come from ``params.bin``. Returns
    ``{tensor id: array}`` for *outputs* (tensor ids). Raises ``AiegraphError`` on a
    missing feed, a build error or a call returning non-zero."""
    import numpy as np
    from .group import CallGraph

    workdir = Path(workdir).resolve()
    cg = cg or CallGraph.load(workdir)
    calls = [int(c) for c in calls]
    hb = _env.frontend("aiekernel.hostbuild")
    stem = cg.path.name[:-len("_graph.json")] if cg.path and cg.path.name.endswith("_graph.json") \
        else next(workdir.glob("*_graph.json")).name[:-len("_graph.json")]
    params = _Params(workdir / f"{stem}_params.bin")

    # tensors the chain touches (raw graph.json entry ids -> canonical; one buffer each)
    tids, idx = [], {}

    def slot(t):
        t = cg.canon(t)
        if t not in idx:
            idx[t] = len(tids)
            tids.append(t)
        return idx[t]

    produced = set()
    plan_calls = []
    syms = []
    for c in calls:
        gc = cg.calls[c]
        if gc.symbol not in syms:
            syms.append(gc.symbol)
        # the node's real argument list (graph.json inputs, __nop NOT resolved: TVM's
        # driver passes the aliased buffer, which canon() gives)
        args = [slot(t) for t in gc.ins] + [slot(t) for t in gc.outs]
        plan_calls.append((syms.index(gc.symbol), args))
        produced.update(cg.canon(t) for t in gc.outs)
    outs = [cg.canon(int(getattr(t, "tensor", t))) for t in outputs]
    for t in outs:
        if t not in produced:
            raise AiegraphError(_STAGE, f"tensors[{t}] is not produced by calls {calls}")

    feeds = {cg.canon(int(getattr(k, "tensor", k))): v for k, v in (feeds or {}).items()}
    init_data, plan_t = [], []
    for t in tids:
        dt, shp = cg.dtype[t], tuple(int(d) for d in cg.shape[t])
        code, bits = _dl(dt)
        n = int(np.prod(shp)) if shp else 1
        nb = n * _np_dtype(dt).itemsize
        init = 0
        if t not in produced:
            name = cg.names.get(t)
            if t in feeds:
                a = np.ascontiguousarray(np.asarray(feeds[t]), _np_dtype(dt))
                if a.size != n:
                    raise AiegraphError(_STAGE, f"feed tensors[{t}]: {a.size:,} elements, "
                                        f"want {n:,} {list(shp)}")
            elif name is not None and name in params:
                a = params.array(name, dt, shp)
            else:
                raise AiegraphError(_STAGE, f"tensors[{t}] {dt}{list(shp)} "
                                    f"({name or 'activation'}) has no feed",
                                    hint="pass the group's boundary inputs: feeds={tensor: array}")
            init_data.append(a)
            init = 1
        plan_t.append(f"{init} {code} {bits} {len(shp)} {' '.join(map(str, shp))} {nb}")

    work = Path(work) if work else workdir / "aiek" / "groupcheck" / f"chain_{'_'.join(map(str, calls))}"
    work.mkdir(parents=True, exist_ok=True)
    hdir = hb.host_dir(workdir)
    inc = [hdir, workdir / "layers", hb.REPO / "include"]
    src = _CHAIN_C.replace("@DECLS@", "\n".join(
        f"int32_t {s}(void *, int32_t *, int32_t, void *, int32_t *, void *);" for s in syms)) \
        .replace("@FNS@", ", ".join(syms))
    harness = work / "chain.c"
    if not harness.is_file() or harness.read_text() != src:
        harness.write_text(src)
    try:
        objs = [hb.compile_c(harness, work / "chain.o", inc, force=True)]
        for p in _layer_sources(workdir, syms):
            objs.append(hb.compile_c(p, workdir / "aiek" / "_host" / "obj" / f"{p.stem}.o", inc))
        objs.append(hb.compile_c(hdir / "tvm_runtime_shim.c", hdir / "obj" / "tvm_runtime_shim.o", inc))
        exe = hb.link(objs, work / "chain")
    except hb.HostBuildError as exc:
        raise AiegraphError(_STAGE, f"x86 build of TVM calls {calls}: {exc}") from None

    lines = [str(len(tids)), *plan_t, str(len(plan_calls))]
    lines += [f"{fn} {len(a)} {' '.join(map(str, a))}" for fn, a in plan_calls]
    lines += [str(len(outs)), " ".join(str(idx[t]) for t in outs)]
    (work / "plan.txt").write_text("\n".join(lines) + "\n")
    with open(work / "inputs.bin", "wb") as fh:
        for a in init_data:
            fh.write(memoryview(np.ascontiguousarray(a)).cast("B"))
    proc = hb.run([exe, work / "plan.txt", work / "inputs.bin", work / "outputs.bin"], cwd=work)
    if proc.returncode != 0 or "TVMCHAIN OK" not in proc.stdout:
        raise AiegraphError(_STAGE, f"TVM calls {calls} on x86: exit {proc.returncode}: "
                            f"{(proc.stdout + proc.stderr)[-400:]}")
    blob = (work / "outputs.bin").read_bytes()
    res, off = {}, 0
    for t in outs:
        dt = _np_dtype(cg.dtype[t])
        shp = tuple(int(d) for d in cg.shape[t])
        n = int(np.prod(shp)) if shp else 1
        res[t] = np.frombuffer(blob, dt, n, off).reshape(shp).copy()
        off += n * dt.itemsize
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  the group check
# ═══════════════════════════════════════════════════════════════════════════

def group_check(workdir, group, run_steps, inputs: dict, golden: dict = None,
                cos_min: float = COS_MIN, raise_on_fail: bool = True,
                verbose: bool = True) -> GroupCheckReport:
    """x86 check of one group (a ``Group`` or a ``GroupPlan``).

    *inputs*: ``{tensor: array}`` for the group's boundary inputs (activations and graph
    inputs; parameters come from ``params.bin``). *run_steps(inputs)* runs the steps in
    their placed order and returns ``{tensor: array}`` for every group output.
    *golden*: optional ``{tensor: array}`` the TVM chain itself must match (real data).
    Raises ``AiegraphError`` unless every output passes (``raise_on_fail=False``: return
    the failing report instead)."""
    from .group import CallGraph, GroupPlan

    workdir = Path(workdir).resolve()
    cg = CallGraph.load(workdir)
    g = group.group if isinstance(group, GroupPlan) else group
    rep = GroupCheckReport(g.name)
    outs = [t.tensor for t in g.outputs]
    if not outs:
        raise AiegraphError(_STAGE, f"group {g.name}: no outputs to check")
    feeds = {cg.canon(int(getattr(k, "tensor", k))): v for k, v in (inputs or {}).items()}
    ref = tvm_chain(workdir, g.calls, feeds, outs, cg=cg)
    got = run_steps(dict(feeds)) or {}
    got = {cg.canon(int(getattr(k, "tensor", k))): v for k, v in got.items()}
    gold = {cg.canon(int(getattr(k, "tensor", k))): v for k, v in (golden or {}).items()}
    for t in outs:
        name = f"tensors[{t}]"
        if t not in got:
            rep.checks.append(TensorCheck(name, "tvm", int(ref[t].size), float("nan"), 0, 0,
                                          float("nan"), cos_min, error="no step wrote it"))
        else:
            rep.checks.append(compare(name, ref[t], got[t], cos_min, "tvm"))
        if t in gold:
            rep.checks.append(compare(name, gold[t], ref[t], cos_min, "golden"))
    if verbose:
        print(rep)
    return rep.raise_if_failed() if raise_on_fail else rep


# ═══════════════════════════════════════════════════════════════════════════
#  layer offload: gates 3-4 of flow.offload as a per-call group check
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _LayerCheck:
    layer: int
    call: int
    report: GroupCheckReport

    def summary(self) -> str:
        return f"layer {self.layer:02d} call [{self.call:02d}]: " + \
            "; ".join(c.summary() for c in self.report.checks)


def _layer_activation(arg, inputs, rng):
    import numpy as np
    if inputs and arg.eid in inputs:
        return np.ascontiguousarray(np.asarray(inputs[arg.eid]), _np_dtype(arg.dtype))
    dt = _np_dtype(arg.dtype)
    shape = tuple(int(d) for d in arg.shape)
    if dt.kind == "f":
        return rng.standard_normal(shape).astype(dt)
    info = np.iinfo(dt)
    lo, hi = max(info.min, -(1 << 15)), min(info.max, 1 << 15)
    return rng.integers(lo, hi, size=shape, endpoint=True, dtype=np.int64).astype(dt)


def layer_checker(cos_min: float = COS_MIN, inputs: dict = None, seed: int = 1000):
    """A ``checker(net, plans, verbose)`` for ``flow.offload`` / ``flow.check``: every call
    of every selected layer is a one-call group; reference = the TVM C function, candidate
    = the kernel's ``<entry>_ref`` (the step on x86), both on the call's real parameters
    and the same activations (*inputs* ``{graph entry id: array}``, else seeded normal).
    A layer failing any output is rejected at gate ``group-check``."""

    def check(net, plans, verbose: bool = True) -> None:
        import numpy as np
        cal = _env.frontend("aiekernel.calibrate")
        spec_mod = _env.frontend("aiekernel.spec")
        for kp in plans:
            for layer in list(kp.selected):
                calls = [v.call for v in kp.verdicts[layer].calls]
                if kp.spec.reference is None or not Path(kp.spec.reference).is_file():
                    kp.reject(layer, _STAGE, "the spec has no x86 reference (<entry>_ref) to check")
                    continue
                work = net.out_dir / "aiek" / kp.spec.name / "groupcheck"
                work.mkdir(parents=True, exist_ok=True)
                try:
                    exe = cal._build(net, kp.spec, calls[0], work)
                except Exception as exc:  # HostBuildError
                    kp.reject(layer, _STAGE, f"x86 build: {exc}")
                    continue
                results = []
                for k, call in enumerate(calls):
                    rng = np.random.default_rng(seed + k)
                    sig = [str(len(call.args))]
                    for a in call.args:
                        code, bits = spec_mod.DTYPES[a.dtype]
                        sig.append(f"{int(a.kind == 'out')} {code} {bits} {len(a.shape)} "
                                   f"{' '.join(str(int(d)) for d in a.shape)} {a.nbytes}")
                    (work / "sig.txt").write_text("\n".join(sig) + "\n")
                    with open(work / "inputs.bin", "wb") as fh:
                        for a in call.args:
                            if a.kind == "param":
                                fh.write(net.param_bytes(a.param))
                            elif a.kind == "in":
                                fh.write(_layer_activation(a, inputs, rng).tobytes())
                    proc = cal.run([exe, work / "sig.txt", work / "inputs.bin",
                                    work / "out_tvm.bin", work / "out_ref.bin"], cwd=work)
                    rep = GroupCheckReport(f"layer{layer:02d}[{call.call}]")
                    line = next((ln for ln in proc.stdout.splitlines()
                                 if ln.startswith("AIEK_CALIB")), "")
                    rcs = dict(tok.split("=") for tok in line.split()[1:]) if line else {}
                    if proc.returncode != 0 or not line or rcs.get("rc_tvm") != "0" \
                            or rcs.get("rc_ref") != "0":
                        rep.checks.append(TensorCheck(
                            f"call [{call.call}]", "tvm", 0, float("nan"), 0, 0, float("nan"), cos_min,
                            error=f"harness exit {proc.returncode} {line or (proc.stderr or '')[-300:]}"))
                    else:
                        a_b, b_b = (work / "out_tvm.bin").read_bytes(), (work / "out_ref.bin").read_bytes()
                        off = 0
                        for a in call.args:
                            if a.kind != "out":
                                continue
                            dt = _np_dtype(a.dtype)
                            n = a.nbytes // dt.itemsize
                            rep.checks.append(compare(f"tensors[{a.eid}]", np.frombuffer(a_b, dt, n, off),
                                                      np.frombuffer(b_b, dt, n, off), cos_min, "tvm"))
                            off += a.nbytes
                    results.append(_LayerCheck(layer, call.call, rep))
                    if verbose:
                        print(f"  [group-check] {rep}")
                kp.calib[layer] = results
                bad = [r for r in results if not r.report.ok]
                if bad:
                    kp.reject(layer, _STAGE, bad[0].summary())
    return check
