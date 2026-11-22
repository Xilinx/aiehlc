#!/usr/bin/env python3
"""report_scatter -- predicted-vs-measured table + scatter for the channel report (costmodel M7 P4).

  report_scatter.py --out DIR [--board palmyra213] [--r2-min 0.90]

Prediction: aiegraph.costmodel.estimate({"net", "build", "cfg"}) per site (site_ms = T_hat x launches), the same
call and build dirs as M6 costmodel_cfg_decision.py (unet: /scratch/staff/huaj/tmp/costmodel_m31223, resnet18 and
yolo26n: /scratch/staff/huaj/tmp/costmodel_m40).  Nothing is refit.
Measurement: the palmyra213 board logs of costmodel_cfg_decision.NETS (the 'main' run: unet shuhua M34 *.a.log,
resnet18 M41 *.d.log, yolo26n M42 *.d.log), read with its read_log() (launches>0 sites, site_ms).  The board is
taken from each log's CMP line.
Per net: R^2 = 1 - SS_res/SS_tot of measured site_ms against the identity line (prediction as is, both cfgs).

Every run first runs the selftest on the loaded data (a clean copy must PASS; site count mismatch,
board != palmyra213, constant prediction, shuffled measurement, a missing net must each FAIL), prints
  M7SCAT selftest cases=<n> pass=<n> PASS|FAIL
then the real check:
  M7SCAT net=<n> cfg=<c> board=<b> sites=<n> pred_ms=X meas_ms=X
  M7SCAT net=<n> pts=<n> r2=<.3f> maxrel=<.3f>
  M7SCAT nets=3 board=<b> pts=<n> png=<n> r2_min=<.3f, capped at 0.999> PASS|FAIL
Writes DIR/scatter.png (checked with `file`), DIR/pred_vs_meas.tsv (every point), DIR/summary.md.
"""
import argparse
import copy
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))                    # -I drops PYTHONPATH (same as params_table.py)
sys.path.insert(0, str(ROOT / "example" / "aiegraph"))

import costmodel_cfg_decision as DEC  # noqa: E402
from aiegraph.costmodel import estimate  # noqa: E402

NETS = ("unet", "resnet18", "yolo26n")


def load(board):
    """{net: {cfg: dict(log, boards, pred{part: ms}, meas{part: ms})}} -- estimate() and board logs, no refit."""
    data = {}
    for net in NETS:
        ca, cb, L, logs = DEC.NETS[net]
        build = DEC.UNET_BUILD if net == "unet" else DEC.NET_BUILD
        data[net] = {}
        for cfg, lg in zip((ca, cb), logs["main"]):
            r = estimate({"net": net, "build": build, "cfg": cfg}, board=board)
            m = DEC.read_log(lg, net, L)
            data[net][cfg] = dict(log=str(lg), boards=list(m["boards"]),
                                  pred={int(s["part"]): s["T_hat"] * s["launches"] for s in r["sites"]},
                                  meas=dict(m["sites"]))
    return data


def r2(x, y):
    my = sum(y) / len(y)
    tot = sum((v - my) ** 2 for v in y)
    res = sum((b - a) ** 2 for a, b in zip(x, y))
    return 1.0 - res / tot if tot > 0 else float("-inf")


def check(data, board, r2_min, quiet=True):
    """-> (ok, summary, rows, errors). rows = (net, cfg, part, pred_ms, meas_ms)."""
    err, rows, per = [], [], {}
    if sorted(data) != sorted(NETS):
        err.append("nets %s != %s" % (sorted(data), sorted(NETS)))
    boards = set()
    for net in sorted(data):
        xs, ys = [], []
        for cfg, d in data[net].items():
            if d["boards"] != [board]:
                err.append("%s %s: board %s != %s" % (net, cfg, d["boards"], board))
            boards |= set(d["boards"])
            if len(d["pred"]) != len(d["meas"]) or set(d["pred"]) != set(d["meas"]):
                err.append("%s %s: sites pred=%d meas=%d" % (net, cfg, len(d["pred"]), len(d["meas"])))
            parts = sorted(set(d["pred"]) & set(d["meas"]))
            for p in parts:
                rows.append((net, cfg, p, d["pred"][p], d["meas"][p]))
                xs.append(d["pred"][p])
                ys.append(d["meas"][p])
            if not quiet:
                print("M7SCAT net=%s cfg=%s board=%s sites=%d pred_ms=%.1f meas_ms=%.1f"
                      % (net, cfg, ",".join(d["boards"]) or "-", len(parts),
                         sum(d["pred"][p] for p in parts), sum(d["meas"][p] for p in parts)))
        if len(xs) < 3:
            err.append("%s: %d points" % (net, len(xs)))
            per[net] = (len(xs), float("-inf"), float("inf"))
            continue
        if max(xs) - min(xs) <= 1e-9 * max(1.0, abs(max(xs))):
            err.append("%s: prediction is constant (%.4f)" % (net, xs[0]))
        q = r2(xs, ys)
        mr = max(abs(a - b) / b for a, b in zip(xs, ys) if b > 0)
        per[net] = (len(xs), q, mr)
        if q < r2_min:
            err.append("%s: r2=%.3f < %.3f" % (net, q, r2_min))
        if not quiet:
            print("M7SCAT net=%s pts=%d r2=%.3f maxrel=%.3f" % (net, len(xs), q, mr))
    rmin = min((v[1] for v in per.values()), default=float("-inf"))
    summ = dict(nets=len(data), board=",".join(sorted(boards)) or "-", pts=len(rows), r2_min=rmin, per=per)
    return not err, summ, rows, err


def selftest(data, board, r2_min):
    cases = []

    def mut(name, fn, want_ok):
        d = copy.deepcopy(data)
        fn(d)
        ok, _s, _r, e = check(d, board, r2_min)
        cases.append((name, ok == want_ok, e))

    def drop_site(d):
        m = d["resnet18"]["off"]["meas"]
        m.pop(sorted(m)[0])

    def bad_board(d):
        d["yolo26n"]["aot"]["boards"] = ["palmyra214"]

    def const_pred(d):
        for c in d["unet"].values():
            c["pred"] = {p: 10.0 for p in c["pred"]}

    def shuffle_meas(d):
        rnd = random.Random(7)
        for c in d["unet"].values():
            k, v = list(c["meas"]), list(c["meas"].values())
            rnd.shuffle(v)
            c["meas"] = dict(zip(k, v))

    def drop_net(d):
        d.pop("yolo26n")

    mut("clean_copy", lambda d: None, True)
    mut("site_count_mismatch", drop_site, False)
    mut("board_not_palmyra213", bad_board, False)
    mut("constant_prediction", const_pred, False)
    mut("shuffled_measurement", shuffle_meas, False)
    mut("missing_net", drop_net, False)
    good = sum(1 for _n, g, _e in cases if g)
    for n, g, e in cases:
        print("M7SCAT selftest case=%s %s%s" % (n, "ok" if g else "WRONG", (" (" + e[0] + ")") if e else ""))
    print("M7SCAT selftest cases=%d pass=%d %s" % (len(cases), good, "PASS" if good == len(cases) else "FAIL"))
    return good == len(cases)


def plot(rows, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"off": "#2a78d6"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, net in zip(axs, NETS):
        rs = [r for r in rows if r[0] == net]
        cfgs = list(dict.fromkeys(r[1] for r in rs))
        for cfg in cfgs:
            x = [r[3] for r in rs if r[1] == cfg]
            y = [r[4] for r in rs if r[1] == cfg]
            ax.scatter(x, y, s=18, alpha=0.75, color=col.get(cfg, "#eb6834"), edgecolors="white", linewidths=0.5,
                       label="%s (%d sites)" % (cfg, len(x)))
        lo = min(min(r[3], r[4]) for r in rs if min(r[3], r[4]) > 0) * 0.8
        hi = max(max(r[3], r[4]) for r in rs) * 1.2
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        n, q, mr = summ["per"][net]
        ax.set_title("%s  R²=%.3f  n=%d" % (net, min(q, 0.999), n))
        ax.set_xlabel("predicted site ms (estimate())")
        ax.set_ylabel("measured site ms (%s)" % summ["board"])
        ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("costmodel M7: predicted vs measured per site, board=%s" % summ["board"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--r2-min", type=float, default=0.90)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load(a.board)
    st_ok = selftest(data, a.board, a.r2_min)

    ok, summ, rows, err = check(data, a.board, a.r2_min, quiet=False)
    for e in err:
        print("ERROR %s" % e)
    with open(out / "pred_vs_meas.tsv", "w") as f:
        f.write("net\tcfg\tpart\tpred_ms\tmeas_ms\trel\n")
        for net, cfg, p, x, y in rows:
            f.write("%s\t%s\t%d\t%.4f\t%.4f\t%+.4f\n" % (net, cfg, p, x, y, (x - y) / y if y else 0.0))
    png = out / "scatter.png"
    if png.exists():
        png.unlink()
    plot(rows, summ, png)
    ft = subprocess.run(["file", "-b", str(png)], capture_output=True, text=True).stdout.strip()
    npng = 1 if ft.startswith("PNG image data") else 0
    print("M7SCAT png=%s file=%s" % (png, ft or "-"))

    with open(out / "summary.md", "w") as f:
        f.write("## Predicted vs measured (board %s, prediction = estimate(), no refit)\n\n" % summ["board"])
        f.write("| net | cfg | sites | pred ms | meas ms | rel |\n|---|---|---|---|---|---|\n")
        for net in NETS:
            for cfg in data[net]:
                rs = [r for r in rows if r[0] == net and r[1] == cfg]
                x, y = sum(r[3] for r in rs), sum(r[4] for r in rs)
                f.write("| %s | %s | %d | %.1f | %.1f | %+.3f |\n" % (net, cfg, len(rs), x, y, (x - y) / y))
        f.write("\n| net | points | R² | max site rel |\n|---|---|---|---|\n")
        for net in NETS:
            n, q, mr = summ["per"][net]
            f.write("| %s | %d | %.3f | %.3f |\n" % (net, n, q, mr))
        f.write("\nLogs:\n")
        for net in NETS:
            for cfg, d in data[net].items():
                f.write("- %s %s: `%s`\n" % (net, cfg, d["log"]))
        f.write("\nScatter: `scatter.png`; every point: `pred_vs_meas.tsv`.\n")

    allok = st_ok and ok and npng == 1
    print("M7SCAT nets=%d board=%s pts=%d png=%d r2_min=%.3f %s"
          % (summ["nets"], summ["board"], summ["pts"], npng, min(summ["r2_min"], 0.999), "PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
