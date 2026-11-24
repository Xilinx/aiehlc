###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""TVM AutoTVM integration for the aiehlc conv2d auto-tiling model.

The tunable parameters of a declarative conv2d (``aie::Conv2d`` +
``aie::Mapping``) are exposed as an AutoTVM ``ConfigSpace``:

    time_oh   -> tile(OH, n): IFM d1.slice_tiling (halo slice/step/rounds), OFM d1.slice_tiling
    time_ow   -> tile(OW, n): IFM d2 (width halo slice/step/rounds),       OFM d2
    pp_depth  -> hints.pp_depth (ping-pong depth of every port)
    l1_budget -> hints.l1_budget (cap on the non-halo OFM/WEIGHTS port buffers)

restricted (``multi_filter``) to the configurations the C++ model accepts
(divisibility, halo bounds, core data budget, kernel stack). Every d1/d2/d3
value is *derived* from these knobs by the model -- the tuner cannot produce an
inconsistent halo. Any AutoTVM tuner works; ``XGBTuner(feature_type="knob")``
learns directly on the tile sizes.

A measurement is a custom Builder/Runner pair:

    mode="analytic"  no compile; cost = the model's calibrated estimate (seconds)
    mode="build"     run aiehlc.sh with AIEHLC_CONV_TUNE=<knobs>; cost = estimate.
                     Proves the pipeline accepts every tried config.
    mode="board"     build, then run ``run_cmd`` (default: apppaltest --netload)
                     and parse ``[BENCH] aie_wall_ms=<ms>``; a config whose
                     output check fails is recorded as an error, never as fast.

aiehlc.sh is not re-entrant (thirdparty/alib/lib is wiped per run), so builds
and runs are strictly sequential.
"""

import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from . import convtile

try:
    import tvm
    from tvm import autotvm
    from tvm.autotvm.measure.measure import Builder, MeasureErrorNo, MeasureResult, Runner
    from tvm.autotvm.measure.measure_methods import BuildResult
    from tvm.autotvm.task.space import ConfigSpace
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "frontend.convtune needs TVM 0.16 (thirdparty/tvm-0.16/python on PYTHONPATH) and xgboost "
        "(`pip install xgboost`)"
    ) from exc

TASK_NAME = "aiehlc_conv2d_tile"
KNOBS = ("time_oh", "time_ow", "pp_depth", "l1_budget")
DEFAULT_RUN_CMD = (
    "python3 {repo}/script/test/apppaltest.py {elf} -y --netload --netload-no-force --netload-wait 300"
)
BENCH_RE = r"\[BENCH\]\s*aie_wall_ms=([0-9.]+)"
PASS_RE = r"PASS: all \d+ output elements match"
FAIL_RE = r"FAIL:|MISMATCH|AIE ERROR|TIMEOUT"


def knobs_of(config):
    """ConfigEntity -> {"time_oh": .., "time_ow": .., "pp_depth": .., "l1_budget": ..}."""
    return {k: int(config[k].val) for k in KNOBS}


def create_task(src, pp_depths=None, l1_budgets=None, model_args=()):
    """Build an AutoTVM Task whose ConfigSpace is the legal tiling space of ``src``.

    ``pp_depths`` / ``l1_budgets``: values to explore (default: the mapping's hints).
    Returns (task, legal) where legal maps (time_oh, time_ow, pp_depth, l1_budget) -> derived.
    """
    src = str(Path(src).resolve())
    base = convtile.space(src, model_args=model_args)
    pps = list(pp_depths) if pp_depths else [base["pp_depth"]]
    l1s = list(l1_budgets) if l1_budgets else [base["l1_budget"]]
    legal = {}
    for pp in pps:
        for l1 in l1s:
            try:
                sp = convtile.space(src, pp, model_args, l1_budget=l1)
            except convtile.ConvTileError:
                continue
            for cand in sp["candidates"]:
                kb = cand["knobs"]
                legal[(kb["time_oh"], kb["time_ow"], pp, l1)] = cand
    if not legal:
        raise convtile.ConvTileError(f"{src}: no legal tiling for pp_depth in {pps}, l1_budget in {l1s}")

    conv = base["conv"]
    geo = next(iter(legal.values()))["geometry"]
    args = (src, conv["h"], conv["w"], conv["c"], conv["k"], conv["kh"], conv["kw"], conv["stride"], conv["pad"],
            base["mesh_rows"], base["mesh_cols"])
    task = autotvm.task.Task(TASK_NAME, args)
    cfg = ConfigSpace()
    cfg.define_knob("time_oh", sorted({k[0] for k in legal}))
    cfg.define_knob("time_ow", sorted({k[1] for k in legal}))
    cfg.define_knob("pp_depth", sorted({k[2] for k in legal}))
    cfg.define_knob("l1_budget", sorted({k[3] for k in legal}))
    cfg.multi_filter(
        filter=lambda e: (e["time_oh"].val, e["time_ow"].val, e["pp_depth"].val, e["l1_budget"].val) in legal)
    task.config_space = cfg
    task.target = tvm.target.Target("c")
    # Whole-layer FLOPs (2 per MAC) so tuners report GFLOPS.
    macs = geo["oh"] * geo["ow"] * conv["k"] * geo["k_red"]
    task.flop = 2 * macs
    task.aie_kernel = base["kernel"]
    task.aie_legal = legal
    return task, legal


def check_pipeline_log(text, derived):
    """Cross-check the tilinglinalg log of one build against the model.

    The model's legality rules mirror pipeline behaviour; this catches drift:
    every port must be one round of exactly the modelled size, the pipeline's
    buffer total must equal the modelled L1, and no pipeline ERROR may appear.
    Returns a list of problems (empty == consistent).
    """
    f = derived["features"]
    want = {0: f["ifm_round_bytes"], 1: f["wgt_bytes"], 2: f["ofm_round_bytes"]}
    problems = [ln.strip() for ln in text.splitlines() if ln.startswith("[TilingLinalg] ERROR")]
    ports = {int(m.group(1)): (int(m.group(2)), int(m.group(3)))
             for m in re.finditer(r"\[TilingLinalg\]\s+port\[(\d+)\]: numRounds=(\d+) bufferSize=(\d+)", text)}
    for i, size in want.items():
        got = ports.get(i)
        if got != (1, size):
            problems.append(f"port[{i}] pipeline numRounds/bufferSize={got}, model expects (1, {size})")
    m = re.search(r"Buffer memory check passed: (\d+) /", text)
    if not m or int(m.group(1)) != f["l1_bytes"]:
        problems.append(f"pipeline buffer total {m.group(1) if m else None} != model l1_bytes {f['l1_bytes']}")
    return problems


class AieConvBuilder(Builder):
    """Compiles one trial per config with aiehlc.sh (or not, in analytic mode)."""

    def __init__(self, mode="analytic", workdir="convtune_out", aie_version=5, aiehlc_args=(), timeout=900,
                 model_args=()):
        super().__init__(timeout=timeout, n_parallel=1)
        self.mode = mode
        self.workdir = Path(workdir).resolve()
        self.aie_version = aie_version
        self.aiehlc_args = list(aiehlc_args)
        self.model_args = tuple(model_args)
        self.trial = 0

    def _compile(self, src, knobs, tdir):
        tdir.mkdir(parents=True, exist_ok=True)
        tune_file = tdir / "conv_tune.json"
        tune_file.write_text(json.dumps(knobs))
        env = dict(os.environ, AIEHLC_CONV_TUNE="@" + str(tune_file))
        script = convtile.REPO / "script" / "aiehlc.sh"
        cmd = (f"source {script} --aie-version {self.aie_version} --runtime-source-file {src} "
               + " ".join(self.aiehlc_args))
        start = time.time()
        with open(tdir / "build.log", "w") as log:
            proc = subprocess.run(["bash", "-c", cmd], cwd=tdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=self.timeout)
        text = (tdir / "build.log").read_text(errors="replace")
        elf = tdir / "aout" / "main.elf"
        if proc.returncode != 0 or not elf.is_file() or elf.stat().st_mtime < start:
            raise RuntimeError(f"aiehlc.sh failed (rc={proc.returncode}); see {tdir / 'build.log'}")
        # Guard against a stale aiehlc that ignores AIEHLC_CONV_TUNE.
        want = f"time tile(OH,{knobs['time_oh']}) x tile(OW,{knobs['time_ow']})"
        if want not in text:
            raise RuntimeError(f"aiehlc did not apply {knobs} (stale build/aiehlc?); see {tdir / 'build.log'}")
        return str(elf)


    def build(self, measure_inputs):
        results = []
        for inp in measure_inputs:
            tic = time.time()
            knobs = knobs_of(inp.config)
            src = inp.task.args[0]
            self.trial += 1
            tdir = self.workdir / "trials" / (f"{self.trial:04d}_oh{knobs['time_oh']}_ow{knobs['time_ow']}"
                                              f"_pp{knobs['pp_depth']}_l1{knobs['l1_budget']}")
            try:
                derived = convtile.derive(src, knobs, self.model_args)
                elf = None
                if self.mode in ("build", "board"):
                    elf = self._compile(src, knobs, tdir)
                    problems = check_pipeline_log((tdir / "build.log").read_text(errors="replace"), derived)
                    if problems:
                        raise RuntimeError("pipeline disagrees with the tiling model: " + "; ".join(problems))
                results.append(BuildResult(elf, {"knobs": knobs, "derived": derived, "dir": str(tdir)}, None,
                                           time.time() - tic))
            except Exception as exc:  # recorded as a failed trial, tuning continues
                results.append(BuildResult(None, {"knobs": knobs, "dir": str(tdir)},
                                           (traceback.format_exc(), exc), time.time() - tic))
        return results


class AieConvRunner(Runner):
    """Turns a build into a cost: model estimate, or a measured board run."""

    def __init__(self, mode="analytic", run_cmd=DEFAULT_RUN_CMD, bench_re=BENCH_RE, pass_re=PASS_RE,
                 fail_re=FAIL_RE, timeout=900, repeat=1):
        super().__init__(timeout=timeout, n_parallel=1)
        self.mode = mode
        self.run_cmd = run_cmd
        self.bench_re = re.compile(bench_re)
        self.pass_re = re.compile(pass_re) if pass_re else None
        self.fail_re = re.compile(fail_re) if fail_re else None
        self.repeat = repeat

    def get_build_kwargs(self):
        return {}

    def _board(self, elf, tdir):
        costs = []
        for i in range(self.repeat):
            cmd = self.run_cmd.format(repo=convtile.REPO, elf=elf, dir=tdir)
            out = subprocess.run(["bash", "-c", cmd], cwd=tdir, capture_output=True, text=True,
                                 timeout=self.timeout)
            text = out.stdout + out.stderr
            Path(tdir, f"run{i}.log").write_text(text)
            m = self.bench_re.findall(text)
            if not m:
                raise RuntimeError(f"no benchmark line ({self.bench_re.pattern}) in run output; see {tdir}/run{i}.log")
            if self.fail_re and self.fail_re.search(text):
                raise RuntimeError(f"output check failed; see {tdir}/run{i}.log")
            if self.pass_re and not self.pass_re.search(text):
                raise RuntimeError(f"no pass line ({self.pass_re.pattern}); see {tdir}/run{i}.log")
            costs.append(float(m[-1]) / 1000.0)
        return costs

    def run(self, measure_inputs, build_results):
        results = []
        for inp, br in zip(measure_inputs, build_results):
            tic = time.time()
            if br.error is not None:
                results.append(MeasureResult((br.error[1],), MeasureErrorNo.COMPILE_HOST, br.time_cost, time.time()))
                continue
            try:
                if self.mode == "board":
                    costs = self._board(br.filename, br.arg_info["dir"])
                else:
                    costs = [br.arg_info["derived"]["features"]["est_ms"] / 1000.0]
                results.append(MeasureResult(tuple(costs), MeasureErrorNo.NO_ERROR,
                                             br.time_cost + time.time() - tic, time.time()))
            except Exception as exc:
                results.append(MeasureResult((exc,), MeasureErrorNo.RUNTIME_DEVICE,
                                             br.time_cost + time.time() - tic, time.time()))
        return results


def make_tuner(task, name="xgb", plan_size=None):
    """AutoTVM tuner over the knob space. ``xgb`` learns on the knob values."""
    n = len(task.config_space)
    if name == "xgb":
        return autotvm.tuner.XGBTuner(task, plan_size=plan_size or max(4, min(16, n // 2)), feature_type="knob",
                                      loss_type="reg")
    if name == "xgb_rank":
        return autotvm.tuner.XGBTuner(task, plan_size=plan_size or max(4, min(16, n // 2)), feature_type="knob",
                                      loss_type="rank")
    if name == "random":
        return autotvm.tuner.RandomTuner(task)
    if name == "grid":
        return autotvm.tuner.GridSearchTuner(task)
    if name == "ga":
        return autotvm.tuner.GATuner(task, pop_size=min(16, n))
    raise ValueError(f"unknown tuner {name!r} (xgb, xgb_rank, random, grid, ga)")


def mapping_snippet(knobs):
    return (f".time  = {{ aie::tile(aie::OH, {knobs['time_oh']}), aie::tile(aie::OW, {knobs['time_ow']}) }},\n"
            f"  // and .hints = {{... .pp_depth = {knobs['pp_depth']}, .l1_budget = aie::Bytes{{{knobs['l1_budget']}}}}}")


def tune(src, n_trial=32, mode="analytic", tuner="xgb", workdir="convtune_out", pp_depths=None, l1_budgets=None,
         run_cmd=None, aie_version=5, aiehlc_args=(), repeat=1, early_stopping=None, model_args=(), seed=None):
    """Tune the tiling knobs of ``src``. Returns a summary dict (also written to workdir/best.json)."""
    if seed is not None:
        import numpy as np
        import random

        np.random.seed(seed)
        random.seed(seed)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    task, legal = create_task(src, pp_depths, l1_budgets, model_args)
    n_trial = min(n_trial, len(task.config_space))
    builder = AieConvBuilder(mode, workdir, aie_version, aiehlc_args, model_args=model_args)
    runner = AieConvRunner(mode, run_cmd or DEFAULT_RUN_CMD, repeat=repeat)
    measure_option = autotvm.measure_option(builder=builder, runner=runner)
    log_file = workdir / "tune.log"
    if log_file.exists():
        log_file.unlink()
    history = []

    def record(_tuner, inputs, results):
        for inp, res in zip(inputs, results):
            ok = res.error_no == 0
            history.append({"knobs": knobs_of(inp.config), "ok": ok,
                            "cost_ms": (sum(res.costs) / len(res.costs) * 1000.0) if ok else None,
                            "error": None if ok else str(res.costs[0])[:300]})

    t = make_tuner(task, tuner)
    t.tune(n_trial=n_trial, measure_option=measure_option, early_stopping=early_stopping,
           callbacks=[autotvm.callback.log_to_file(str(log_file)), record])
    good = [h for h in history if h["ok"]]
    if not good:
        raise convtile.ConvTileError(f"all {len(history)} trials failed; see {workdir}/trials/*/build.log")
    best = min(good, key=lambda h: h["cost_ms"])
    derived = convtile.derive(task.args[0], best["knobs"], model_args)
    summary = {
        "source": task.args[0], "kernel": task.aie_kernel, "mode": mode, "tuner": tuner,
        "space_size": len(task.config_space), "trials": len(history), "failed": len(history) - len(good),
        "best": {"knobs": best["knobs"], "cost_ms": best["cost_ms"], "derived": derived},
        "history": history,
    }
    (workdir / "best.json").write_text(json.dumps(summary, indent=2))
    (workdir / "best_knobs.json").write_text(json.dumps(best["knobs"]))
    return summary
