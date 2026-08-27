#!/usr/bin/env python3
"""One-shot: applog --> timeline.json/csv --> timeline.png.

A single driver that chains the two existing tools so you don't run them by
hand:

  #1  parse the [TIMESYNC] block in an applog/stdout capture and correlate the
      host clock with each AIE tile's core timer onto one microsecond axis
      (host_aie_timeline.parse_timesync -> correlate -> emit), writing
      timeline.json + timeline.csv;
  #2  render that model to a Gantt-style PNG showing the host running / host
      idle-waiting / AIE running horizontal lanes (timeline_gui.render).

Usage:
    timeline.py <applog>                       # -> <out-dir>/timeline.{json,csv,png}
    timeline.py <applog> --out-dir /tmp/tl     # choose output directory
    timeline.py <applog> --png my.png          # choose the PNG path
    timeline.py <applog> --show                # open a window instead of a PNG
    timeline.py --self-test                    # synthetic end-to-end, no board
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import host_aie_timeline as hat   # noqa: E402  (#1 parse/correlate/emit)
import timeline_gui as tg         # noqa: E402  (#2 render PNG)


def run(applog_path, out_dir, png_path=None, show=False, host_cc=None):
    """applog -> (model, csv_path, json_path, png_or_None). Writes json+csv into
    out_dir, then a PNG (default <out-dir>/timeline.png) unless show=True.

    `host_cc` is the generated host.cc used to resolve phase->line numbers for
    the colour-by-API host lane; auto-detected next to the applog when omitted."""
    with open(applog_path) as f:
        text = f.read()

    # #1: parse + correlate + emit json/csv. Resolve host.cc phase lines so the
    # host lane is coloured by API and annotated with its emit-call line.
    host_cc = host_cc or hat.autodetect_host_cc(applog_path)
    hostcc_lines = hat.resolve_hostcc_lines(host_cc)
    model = hat.correlate(hat.parse_timesync(text), hostcc_lines)
    csv_path, json_path, nrows = hat.emit(model, out_dir)

    # #2: render the PNG (or open a window).
    if show:
        tg.render(model, save=None, want_window=True)
        png_out = None
    else:
        png_out = png_path or os.path.join(out_dir, "timeline.png")
        tg.render(model, save=png_out, want_window=False)

    return model, csv_path, json_path, png_out, nrows


def _self_test():
    """End-to-end on a synthetic block: applog text -> json/csv -> png."""
    import tempfile
    block = "\n".join([
        "[TIMESYNC] cps=1000000",
        "[TIMESYNC] anchor0 host=1000",
        "[TIMESYNC] anchor0 tile=4,4 aie=0",
        "[TIMESYNC] anchor1 host=2000",
        "[TIMESYNC] anchor1 tile=4,4 aie=1000",
        "[TIMESYNC] hostevt iter=0 phase=iter_start host=1000",
        "[TIMESYNC] hostevt iter=0 phase=run host=1130",
        "[TIMESYNC] hostevt iter=0 phase=wait_done host=1700",
        "[TIMESYNC] hostevt iter=0 phase=dma_out_done host=1800",
        "[TIMESYNC] trace tile=4,4 words=16",
        "[TIMESYNC] trace tile=4,4 135 -- 690  ACTIVE  (556 cyc)",
    ])
    with tempfile.TemporaryDirectory() as d:
        applog = os.path.join(d, "applog")
        with open(applog, "w") as f:
            f.write(block + "\n")
        model, csv_path, json_path, png, nrows = run(applog, d)
        assert os.path.getsize(json_path) > 0, "json not written"
        assert os.path.getsize(csv_path) > 0, "csv not written"
        assert png and os.path.getsize(png) > 0, "png not written"
    print("self-test OK: applog -> json/csv -> png (%d rows)" % nrows)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(
        description="applog -> timeline.json/csv -> timeline.png (one shot).")
    ap.add_argument("applog", nargs="?",
                    help="applog / stdout capture containing the [TIMESYNC] block")
    ap.add_argument("--out-dir", default="/tmp/claude/tl",
                    help="output directory for timeline.json/csv/png (default %(default)s)")
    ap.add_argument("--png", metavar="PATH",
                    help="PNG path (default <out-dir>/timeline.png)")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window instead of writing a PNG")
    ap.add_argument("--host-cc", metavar="PATH",
                    help="generated host.cc to resolve host phase->line numbers "
                         "(auto-detected next to the applog / aout/worklocal if omitted)")
    ap.add_argument("--self-test", action="store_true",
                    help="run a synthetic end-to-end test and exit")
    args = ap.parse_args(argv[1:])

    if args.self_test:
        return _self_test()
    if not args.applog:
        ap.error("applog is required (or use --self-test)")

    model, csv_path, json_path, png, nrows = run(
        args.applog, args.out_dir, png_path=args.png, show=args.show,
        host_cc=args.host_cc)

    m = model["meta"]
    print("[timeline] cps=%d  tiles=%d  host_events=%d  rows=%d  host_span=%.2f us" % (
        m["cps"], m["num_tiles"], m["num_host_events"], nrows, m["host_span_us"]))
    print("[timeline] wrote %s" % json_path)
    print("[timeline] wrote %s" % csv_path)
    if png:
        print("[timeline] wrote %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
