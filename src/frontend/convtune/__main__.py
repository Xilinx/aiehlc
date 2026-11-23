###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""CLI:  PYTHONPATH=src python -m frontend.convtune {space,derive,tune} <src.cc> ...

  space   list the legal tilings (ranked by the analytical model)
  derive  show the derived d1/d2/d3 + halo for one knob setting
  tune    AutoTVM search (default XGBTuner) -> <workdir>/best.json, best_knobs.json

Apply a result:  AIEHLC_CONV_TUNE=@<workdir>/best_knobs.json source script/aiehlc.sh ...
or copy the printed `.time = {...}` into the aie::Mapping.
"""

import argparse
import json
import sys

from . import convtile


def _model_args(a):
    out = []
    if a.data_mem_budget:
        out += ["--data-mem-budget", str(a.data_mem_budget)]
    if a.stack_budget is not None:
        out += ["--stack-budget", str(a.stack_budget)]
    if a.copy_cycles_per_byte is not None:
        out += ["--copy-cycles-per-byte", str(a.copy_cycles_per_byte)]
    return out


def _cmd_space(a):
    sp = convtile.space(a.src, a.pp_depth, _model_args(a), a.l1_budget)
    print(f"{sp['kernel']}: {len(sp['candidates'])} legal tilings (mesh {sp['mesh_rows']}x{sp['mesh_cols']}, "
          f"pp_depth {sp['pp_depth']}, l1_budget {sp['l1_budget']}), best first")
    print(f"{'time_oh':>7} {'time_ow':>7} {'rounds':>6} {'slab':>12} {'L1 B':>7} {'stack B':>7} {'halo':>5} {'est ms':>8}")
    for c in sp["candidates"][: a.top]:
        g, f, k = c["geometry"], c["features"], c["knobs"]
        print(f"{k['time_oh']:>7} {k['time_ow']:>7} {g['rounds']:>6} {g['tile_h']:>4}x{g['tile_w']:<4}x{k['c_align']:<2} "
              f"{f['l1_bytes']:>7} {f['stack_bytes']:>7} {f['halo_factor']:>5.2f} {f['est_ms']:>8.3f}")


def _cmd_derive(a):
    knobs = {k: v for k, v in (("time_oh", a.time_oh), ("time_ow", a.time_ow), ("pp_depth", a.pp_depth),
                                ("l1_budget", a.l1_budget)) if v}
    print(json.dumps(convtile.derive(a.src, knobs, _model_args(a)), indent=2))


def _cmd_tune(a):
    from . import autotvm_bridge as br

    pps = [int(x) for x in a.pp_depths.split(",")] if a.pp_depths else None
    l1s = [int(x) for x in a.l1_budgets.split(",")] if a.l1_budgets else None
    s = br.tune(a.src, n_trial=a.n_trial, mode=a.mode, tuner=a.tuner, workdir=a.workdir, pp_depths=pps,
                l1_budgets=l1s,
                run_cmd=a.run_cmd, aie_version=a.aie_version, repeat=a.repeat, early_stopping=a.early_stopping,
                model_args=_model_args(a), seed=a.seed)
    b = s["best"]
    g = b["derived"]["geometry"]
    print(f"\n[convtune] {s['trials']} trials ({s['failed']} failed) of {s['space_size']} legal tilings, "
          f"tuner={s['tuner']} mode={s['mode']}")
    print(f"[convtune] best {b['knobs']}  cost {b['cost_ms']:.3f} ms")
    print(f"[convtune] IFM d1 {g['mesh_slice_h']}/{g['mesh_step_h']} x{g['mesh_rows']}, slice_tiling "
          f"{g['tile_h']}/{g['tile_step_h']} x{g['h_rounds']};  d2 {g['tile_w']}/{g['tile_step_w']} x{g['w_rounds']}")
    print("[convtune] aie::Mapping:\n  " + br.mapping_snippet(b["knobs"]))
    print(f"[convtune] apply: AIEHLC_CONV_TUNE=@{a.workdir}/best_knobs.json source script/aiehlc.sh ...")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m frontend.convtune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("src", help="conv2d source using aie::Conv2d + aie::Mapping")
        sp.add_argument("--data-mem-budget", type=int, default=0, help="core buffer budget (default 49152)")
        sp.add_argument("--stack-budget", type=int, default=None, help="core stack bytes (default 32768, 0=off)")
        sp.add_argument("--copy-cycles-per-byte", type=float, default=None,
                        help="analytical model: kernel window-copy cost (default 7, 0 = zero-copy kernel)")

    s = sub.add_parser("space", help="list legal tilings")
    common(s)
    s.add_argument("--pp-depth", type=int, default=None)
    s.add_argument("--l1-budget", type=int, default=None)
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(fn=_cmd_space)

    d = sub.add_parser("derive", help="derived d1/d2/d3 for one setting")
    common(d)
    d.add_argument("--time-oh", type=int, default=0)
    d.add_argument("--time-ow", type=int, default=0)
    d.add_argument("--pp-depth", type=int, default=0)
    d.add_argument("--l1-budget", type=int, default=0)
    d.set_defaults(fn=_cmd_derive)

    t = sub.add_parser("tune", help="AutoTVM search")
    common(t)
    t.add_argument("--mode", choices=("analytic", "build", "board"), default="analytic")
    t.add_argument("--tuner", choices=("xgb", "xgb_rank", "random", "grid", "ga"), default="xgb")
    t.add_argument("--n-trial", type=int, default=32)
    t.add_argument("--early-stopping", type=int, default=None)
    t.add_argument("--pp-depths", default=None, help="comma list, e.g. 1,2 (default: the mapping's hint)")
    t.add_argument("--l1-budgets", default=None,
                   help="comma list, e.g. 4096,8192 (default: the mapping's hint); caps the OFM/WEIGHTS buffers")
    t.add_argument("--workdir", default="convtune_out")
    t.add_argument("--run-cmd", default=None, help="board command; {elf} {repo} {dir} are substituted")
    t.add_argument("--repeat", type=int, default=1, help="board runs per trial")
    t.add_argument("--aie-version", type=int, default=5)
    t.add_argument("--seed", type=int, default=None)
    t.set_defaults(fn=_cmd_tune)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except convtile.ConvTileError as exc:
        print(f"[convtune] error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
