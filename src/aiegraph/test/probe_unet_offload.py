#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Probe (x86 only): aiegraph's own launch planner on a real UNet partition.

    python3 src/aiegraph/test/probe_unet_offload.py --onnx unet_sim.onnx \\
        --partition example/aiegraph/unet_bk_sdm_tiny/partition.json [--mode kincore]

Prints one ``OF layer`` line per AIE conv and ``OF unet layers=<n> launches_planned=<m>;``
(GAP-aiegraph-offload-7c4415: kincore = one launch per layer, K accumulated in the core)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiegraph.partition import plan_partition_launches  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--mode", default="kincore", choices=("kincore", "tile"))
    a = ap.parse_args()
    r = plan_partition_launches(a.onnx, a.partition, a.mode)
    for i, p in enumerate(r["layers"]):
        g = p.get("geom") or {}
        print(f"OF layer {i} node={p['node']} mkn={p.get('m')}x{p.get('k')}x{p.get('n')} mode={p['mode']} "
              f"launches={p['launches']} tile_launches={p.get('tile_launches')} "
              + (f"tr={g['tr']} tkw={g['tkw']} nkc={g['nkc']} ntile={g['ntile']} S={g['S']} kp={g['kp']} l1={g['l1']}"
                 if g else "why=" + ",".join(p.get("why") or [])))
    print(f"OF info mode={r['mode']} tile_launches={r['tile_launches']} fallbacks={len(r['fallbacks'])}")
    print(f"OF unet layers={len(r['layers'])} launches_planned={r['launches_planned']};")
    return 0 if not r["fallbacks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
