#!/usr/bin/env python3
"""P4t K-accumulation plan for the 37 AIE conv layers of partition.json (analytic, x86 only, no kernel change).

Ported from steve/work/r18_tilecap/ic_split.py (R18 int8, int32 acc) and redone for bf16 GEMM:
each layer is a GEMM M x K x N (partition.json "mkn", im2col conv). Today one launch = one
64x64x64 tile on a 4x4 mesh (86,600 launches). P4b target: one launch per layer, K accumulated on the core.

Per core (contract = $U/p4/K_CONTRACT.md, Maxwell): fp32 accumulator tm x tn = 16 x 16 = 1 KiB,
  init -> accumulate K/kc chunks -> one SRS to bf16 -> write back once per output tile.
Per launch the mesh (R x C cores) walks a list of 64x64 output macro tiles; for each it streams
  A panel i (64 x K bf16) and B panel j (K x 64 bf16) in kc-wide chunks.
Launch splits come from the shim DMA program needed to express that walk without host BD re-config:
  A channel: one BD per macro-row i (3-D: kc words x 64 rows x K/kc chunks), iterated nt times;
  B channel: one BD per launch covering the nt weight panels, iterated per macro-row;
  C channel: one BD for the whole M x N output.
  -> launches = ceil(mt / BDS_PER_CH) * ceil(nt / ITER_MAX)

Limits (source):
  SHIM_BDS=16 per shim tile, 2 ch/dir           amit/knowledge/2026-11-06-aie2ps-hw-limits-cheatsheet.md (aie-rt DevMod)
  TILE_BDS=16 per core tile (needs <= 6: A,B ping-pong + C ping-pong)   same source
  ACC_MAX=1024 B fp32 acc per core              Maxwell's K contract (16x16 fp32), per P4t brief
  PORT_MAX=8192 B per chunk buffer (mesh port)  ASSUMED (ic_split budget; today's 64x64 ports)
  L1_MAX=57344 B per core (64 KiB - 8 KiB stack) ASSUMED (ic_split default)
  WRAP_MAX=1023 per BD dim, ITER_MAX=64, STRIDE_MAX=2^20 words   ASSUMED (not in Amit's notes)
usage: kplan.py --partition partition.json [--json out.json]"""
import argparse, json, os, sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src")))
from aiegraph.kincore import MESH_C, MESH_R, KC, TM, TN, kplan as plan  # noqa: E402,F401  (moved: aiegraph.kincore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = json.load(open(a.partition))
    layers = d["layers"]
    rows, legal, tot, tot_k, tot_old = [], 0, 0, 0, 0
    for l in layers:
        ps = [plan(*mkn) for mkn in l["mkn"]]
        ok = all(p["legal"] for p in ps)
        n = sum(p["launches"] for p in ps)
        legal += ok; tot += n; tot_k += sum(p["launches_k_only"] for p in ps); tot_old += l["launches"]
        p = ps[0]
        print(f"KPLAN idx={l['idx']} legal={int(ok)} launches={n} mkn={p['m']}x{p['k']}x{p['n']} "
              f"mesh={MESH_R}x{MESH_C} core_tile={TM}x{TN} kc={KC} k_chunks={p['k_chunks']} "
              f"out_tiles/core={p['out_tiles_per_core']} acc={p['acc']}B l1={p['l1']}B "
              f"port_a={p['a_port']}B port_b={p['b_port']}B old_launches={l['launches']}"
              + ("" if ok else " why=" + ",".join(w for q in ps for w in q["why"])))
        rows.append(dict(idx=l["idx"], node=l["node"], legal=ok, launches=n, plans=ps))
    print(f"KPLAN_INFO old_launches={tot_old} (partition N={d.get('N')}) k_in_core_only={tot_k} "
          f"per_layer_walk={tot}")
    print(f"KPLAN_SUM layers={len(layers)} legal={legal} launches_planned={tot};")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0 if legal == len(layers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
