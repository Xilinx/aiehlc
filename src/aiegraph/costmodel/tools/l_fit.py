#!/usr/bin/env python3
"""Fit / check the L (launch overhead) model.

  l_fit.py --kernels DIR [DIR ...] [--board palmyra213] --out FIT.json
      DIR/<src>/*.log, union over all DIRs; per-cfg least squares of
      L_meas = const + a*bdcfg_calls + b*startio_calls + c*bdcfg_w + d*cp_w + e*kload_w + f*rst (per launch).
      Sources: off <- {off, off-reload} (reload = UNet switching kernels on the same tiles), other cfgs <- the
      same-named dir.  off-cold fits only const+kload_w (other coefs from off); aot/aot-kco fix kload_w=rst=0
      (kload_w is 11 on every aot point -> collinear with const).  Non-const features that are constant over
      a cfg's points are unidentifiable and get coef 0.  off-cold is skipped when no DIR has it (one-DIR runs
      on M1-L/kernels behave as before).
      A DIR may also be a source leaf itself (basename off|off-reload|off-alt|aot|...); with only leaf DIRs,
      cfgs without logs are skipped instead of failing.  off-alt (two kernels alternating on the same tiles,
      core_reset every launch) is an off source that makes rst identifiable (M1.2.2 P2c).
      Prints  LFIT cfg=<cfg> board=<b> n=<n> r2=%.3f maxrel=%.3f loo_maxrel=%.3f kload_w=%.4e rst=%.4e
      and writes a reproducible JSON (sort_keys, fixed decimals, no paths/timestamps).
  l_fit.py --check-unet CFG=LOG [...] [--params JSON]
      apppaltest UNet logs (no BOARD line): board/launches from 'CMP unet board=.. aie_launches=.. run_rc=0',
      ctrlpkt from 'LAUNCH warm' or 'LAUNCH cold' (aot-kco prints aot), active [CTRLPKT_AOT] (table!=0) iff
      cfg is off/off-cold, split_words required, aie_launches == --unet-launches (258).
      Prediction uses only per-launch call counts and word counts.  Prints
      LGATE cfg=.. board=.. launches=.. pred_ms= meas_ms= abs_us= rel=%.3f  and
      LGATE ALL n=<n> maxrel=%.3f PASS|FAIL (<=0.200).
  l_fit.py --check-unet CFG=LOG [...] --static-root R      (M1.2.2 P4)
      pred uses only the static features of the build R/unet_<cfg> (l_static.static_calls_words: static_feat,
      kload_static, rst_static; no log count enters it); meas and the unet_check (CMP board, 258 launches,
      ctrlpkt, [CTRLPKT_AOT]) still come from the log.  Prints
      LGATE cfg=.. src=static board=.. launches=.. pred_ms= meas_ms= abs_us= rel=%.3f,
      LDIFF cfg=.. pred_static=%.4f pred_logfeat=%.4f dus=%.1f   (|static - log-feature pred|, us/launch) and
      LGATE ALL n=<n> src=static maxrel=%.3f PASS|FAIL.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aiegraph.costmodel.launch_split import lines  # noqa: E402
from aiegraph.costmodel.launch_l import CFGS, FEATURES, feature_row, point, predict_l, load_params  # noqa: E402


def _l_static():
    """src/aiegraph/costmodel/tools/l_static.py (python -I does not put the script dir on sys.path)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("l_static", Path(__file__).resolve().with_name("l_static.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_static(build, cfg, prm):
    """L pred (ms/launch) from the build directory's static features only."""
    calls, words = _l_static().static_calls_words(build)
    return predict_l(calls, words, cfg, prm)

ND = 6


def _fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.round(coef, ND)


SOURCES = {"off": ("off", "off-reload", "off-alt")}  # off-reload/off-alt are dir aliases only, no LFIT line of their own
# off-alt (M1.2.2 P2b): two kernels alternating on the same tiles -> core_reset every launch; makes rst identifiable


def src_logs(d, src):
    """DIR/<src>/*.log, or DIR/*.log when DIR itself is the <src> leaf (basename == src)."""
    d = Path(d)
    if d.name == src and any(d.glob("*.log")):
        return sorted(d.glob("*.log"))
    return sorted((d / src).glob("*.log"))


def is_leaf(d):
    d = Path(d)
    return d.name in {s for v in SOURCES.values() for s in v} | set(CFGS) and any(d.glob("*.log"))
FIXED0 = {"aot": ("kload_w", "rst"), "aot-kco": ("kload_w", "rst")}
COLD_FREE = ("const", "kload_w")
BASE_CFGS = ("off", "aot", "aot-kco")  # always reported (fa539124); others only when some DIR has them


def free_idx(X, cfg):
    fixed = set(FIXED0.get(cfg, ()))
    if cfg == "off-cold":
        fixed = set(FEATURES) - set(COLD_FREE)
    idx = []
    for i, k in enumerate(FEATURES):
        if k in fixed:
            continue
        if k != "const" and np.ptp(X[:, i]) == 0:
            continue  # constant over the points: carried by const, unidentifiable
        idx.append(i)
    return idx


def _fit_free(X, y, idx, base):
    coef = base.copy()
    coef[idx] = 0.0
    r = y - X @ coef
    coef[idx] = _fit(X[:, idx], r)
    return coef


def fit_cfg(pts, cfg="off", base=None):
    X = np.array([feature_row(p["calls"], p["words"]) for p in pts])
    y = np.array([p["l_meas"] for p in pts])
    base = np.zeros(len(FEATURES)) if base is None else np.asarray(base, float)
    idx = free_idx(X, cfg)
    coef = _fit_free(X, y, idx, base)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    maxrel = float(np.max(np.abs(pred - y) / y))
    loo = 0.0
    for i in range(len(pts)):
        m = np.ones(len(pts), bool)
        m[i] = False
        c = _fit_free(X[m], y[m], idx, base)
        loo = max(loo, abs(float(X[i] @ c) - y[i]) / y[i])
    return coef, r2, maxrel, loo

_CMP = re.compile(r"CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)")
_WARM = re.compile(r"LAUNCH (?:warm|cold) .*launches=(\d+).* ctrlpkt=(\S+)")


def unet_meta(log):
    """apppaltest UNet log: board/launches from the CMP line, ctrlpkt from LAUNCH warm (or cold)."""
    u = {"board": None, "launches": None, "run_rc": None, "ctrlpkt": None, "warm_launches": None,
         "cp_lines": 0, "split_words": False}
    for line in lines(log):
        m = _CMP.search(line)
        if m:
            u["board"], u["launches"], u["run_rc"] = m.group(1), int(m.group(2)), int(m.group(3))
        m = _WARM.search(line)
        if m:
            u["warm_launches"], u["ctrlpkt"] = int(m.group(1)), m.group(2)
        if "[CTRLPKT_AOT]" in line and not re.search(r"\btable=0\b", line):
            u["cp_lines"] += 1  # off prints an all-zero 'table=0' summary; count active tables only
        if "LAUNCH split_words" in line:
            u["split_words"] = True
    return u


def unet_check(u, cfg, launches=258):
    if u["board"] is None:
        return "no 'CMP unet board=... aie_launches=... run_rc=' line"
    if u["run_rc"] != 0:
        return f"run_rc={u['run_rc']}"
    if launches and u["launches"] != launches:
        return f"aie_launches={u['launches']} != {launches}"
    if u["ctrlpkt"] is None:
        return "no 'LAUNCH warm|cold ... ctrlpkt=' line"
    off = cfg in ("off", "off-cold")
    want = "off" if off else "aot"  # aot-kco prints ctrlpkt=aot (build_unet.py)
    if u["ctrlpkt"] != want:
        return f"ctrlpkt={u['ctrlpkt']} but cfg={cfg}"
    if off != (u["cp_lines"] == 0):
        return f"[CTRLPKT_AOT] lines={u['cp_lines']} inconsistent with cfg={cfg}"
    if not u["split_words"]:
        return "no 'LAUNCH split_words' line"
    if u["warm_launches"] is not None and u["warm_launches"] != u["launches"]:
        return f"LAUNCH warm|cold launches={u['warm_launches']} != aie_launches={u['launches']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", metavar="DIR")
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--out")
    ap.add_argument("--check-unet", nargs="+", metavar="CFG=LOG")
    ap.add_argument("--params")
    ap.add_argument("--unet-launches", type=int, default=258, help="required aie_launches (0 = any)")
    ap.add_argument("--static-root", help="--check-unet pred from the static features of <root>/unet_<cfg>")
    a = ap.parse_args()
    rc = 0
    if a.kernels:
        out = {"board": a.board, "features": list(FEATURES),
               "l_def": "kload+core_reset+coreen+rt_bdcfg+rt_startio (ms/launch)", "cfg": {}}
        coefs = {}
        for cfg in CFGS:
            logs = []  # (src, path), union over all DIRs
            for d in a.kernels:
                for src in SOURCES.get(cfg, (cfg,)):
                    logs += [(src, lg) for lg in src_logs(d, src)]
            leaf_only = all(is_leaf(d) for d in a.kernels)  # leaf DIRs: report only the cfgs they hold
            if not logs and (cfg not in BASE_CFGS or leaf_only):
                continue
            names = [f"{src}/{lg.stem}" for src, lg in logs]
            if len(set(names)) != len(names):
                print(f"LFIT cfg={cfg} board={a.board} duplicate kernel logs across DIRs: {sorted(names)}")
                rc = 1
                continue
            pts = [point(str(lg)) for _, lg in logs]
            base = None
            if cfg == "off-cold":
                if "off" not in coefs:
                    print(f"LFIT cfg={cfg} board={a.board} n={len(pts)} needs the off fit")
                    rc = 1
                    continue
                base = coefs["off"]
            if len(pts) < 2:
                print(f"LFIT cfg={cfg} board={a.board} n={len(pts)} too few points")
                rc = 1
                continue
            coef, r2, maxrel, loo = fit_cfg(pts, cfg, base)
            coefs[cfg] = coef
            for p, nm in zip(pts, names):
                pr = float(np.dot(feature_row(p["calls"], p["words"]), coef))
                print(f"  {cfg:8s} {nm:35s} kload_w={p['words']['kload_w']:7.1f} rst={p['calls']['core_reset']:5.2f} "
                      f"meas={p['l_meas']:.4f} pred={pr:.4f} rel={abs(pr - p['l_meas']) / p['l_meas']:.3f}")
            print(f"LFIT cfg={cfg} board={a.board} n={len(pts)} r2={r2:.3f} maxrel={maxrel:.3f} loo_maxrel={loo:.3f} "
                  f"kload_w={coef[FEATURES.index('kload_w')]:.4e} rst={coef[FEATURES.index('rst')]:.4e}")
            out["cfg"][cfg] = {"coef": {k: float(f"{v:.{ND}f}") for k, v in zip(FEATURES, coef)},
                               "n": len(pts), "r2": float(f"{r2:.3f}"), "maxrel": float(f"{maxrel:.3f}"),
                               "loo_maxrel": float(f"{loo:.3f}"),
                               "kernels": names}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(out, f, sort_keys=True, indent=1)
                f.write("\n")
    if a.check_unet:
        prm = load_params(a.params) if a.params else load_params()
        worst, n_ok = 0.0, 0
        for item in a.check_unet:
            if "=" in item and item.split("=", 1)[0] in CFGS:
                cfg, log = item.split("=", 1)
            else:  # legacy LOG:CFG
                log, cfg = item.rsplit(":", 1)
            err = None
            src = " src=static" if a.static_root else ""
            try:
                u = unet_meta(log)
                p = point(log)
                if a.static_root:
                    pr_s = pred_static(Path(a.static_root) / f"unet_{cfg}", cfg, prm)
            except Exception as e:  # noqa: BLE001
                err, u = f"parse error {e!r}", {}
            if err is None:
                err = unet_check(u, cfg, a.unet_launches)
            if err:
                print(f"LGATE cfg={cfg}{src} FAIL {err}")
                worst = max(worst, 9.999)
                continue
            pr = predict_l(p["calls"], p["words"], cfg, prm)
            if a.static_root:
                print(f"LDIFF cfg={cfg} pred_static={pr_s:.4f} pred_logfeat={pr:.4f} dus={abs(pr_s - pr) * 1000:.1f}")
                pr = pr_s
            meas = p["l_meas"]
            rel = abs(pr - meas) / meas
            worst = max(worst, rel)
            n_ok += 1
            print(f"  LDET cfg={cfg} calls/launch bdcfg={p['calls']['bdcfg']:.3f} startio={p['calls']['startio']:.3f} "
                  f"kload={p['calls']['kload']:.3f} rst={p['calls']['core_reset']:.3f} words bdcfg_w={p['words']['bdcfg_w']:.1f} "
                  f"cp_w={p['words']['cp_w']:.1f} kload_w={p['words']['kload_w']:.1f} "
                  f"ctrlpkt={u['ctrlpkt']} cp_aot_lines={u['cp_lines']}")
            print(f"LGATE cfg={cfg}{src} board={u['board']} launches={u['launches']} pred_ms={pr:.4f} meas_ms={meas:.4f} "
                  f"abs_us={abs(pr - meas) * 1000:.1f} rel={rel:.3f}")
        ok = worst <= 0.200 and n_ok == len(a.check_unet)
        print(f"LGATE ALL n={len(a.check_unet)}{' src=static' if a.static_root else ''} maxrel={worst:.3f} {'PASS' if ok else 'FAIL'}")
        rc = rc or (0 if ok else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
