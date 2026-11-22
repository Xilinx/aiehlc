#!/usr/bin/env bash
# kln_frozen.sh [--net NET] BUILD_ROOT LOGDIR OUTDIR -- M3.1.2.2.3.2 PFZ: frozen K+L+N per-launch prediction, nothing refit.
#   --net NET  unet (default, output unchanged) | resnet18 | yolo26n (M4.1): BUILD_ROOT/NET_<cfg>, LOGDIR/NET_<cfg>.<run>.log,
#              cfgs off + C2 (unet/resnet18: aot-kco; yolo26n: aot); passed as --net to every step;
#              N params default script/perf/params/n_NET_palmyra213.m3121.json (KLN_N_JSON overrides).
#   BUILD_ROOT/unet_{off,aot-kco}  build dirs (read only; BUILD_ROOT/kln is never written)
#   LOGDIR/unet_{off,aot-kco}.{a,b,c}.log  board logs
#   OUTDIR  all outputs: lpred.tsv npred.tsv kcore.tsv kdelta_core.tsv dwhat.tsv khat.tsv site_pairs.tsv + *.out
# Env: KLN_K_JSON = K params json (default script/perf/params/k_palmyra213.m3112c.json; relative = repo root;
#      a missing/empty file -> rc=2).  KLN_N_JSON = N params json (default script/perf/params/n_unet_palmyra213.m3121.json;
#      same rules).  KLN_PY = python.
# Steps (frozen params from script/perf/params, md5 printed as KLNFROZEN PARAMS):
#   1 L  m3_site --lpred --kfirst site --params l_palmyra213.m3121.json
#   2 N  n_fit --predict --ovl --refit-a no --params $KLN_N_JSON (default n_unet_palmyra213.m3121.json) (NCHK passed through)
#   3 core k_diag --kdelta core (kcyc_unet, default --iss-unet kcyc_unet_iss.json)
#   4 dw k_dmafeat --perch --ddrt --dwpredict --dw-params kdw_palmyra213.m3112c.json
#   5 K  k_diag --khat-static --params $KLN_K_JSON --check a,b,c
#   6 kln_site --k-tsv khat.tsv --k-label frozen --check all 6 logs (no --fit, nothing is fitted: ALL = 6 logs /
#     828 sites; no --loo -> LOO loo=n/a) --site-tsv site_pairs.tsv
# Prints: KLNFROZEN PARAMS file= md5= (one per json), step output, KLNFROZEN rc_l= rc_n= rc_core= rc_dw= rc_k= rc_kln=
# rc: 0 ok; 1 kln_site missed its metric targets; 2 any step rc>1 (or bad args / missing inputs).
# This chain is the inside of M6 estimate().
set -u
NET=unet
if [ "${1:-}" = "--net" ]; then NET=${2:-}; shift 2 || { echo "usage: kln_frozen.sh [--net NET] BUILD_ROOT LOGDIR OUTDIR" >&2; exit 2; }; fi
case "$NET" in unet|resnet18) C2=aot-kco;; yolo26n) C2=aot;; *) echo "KLNFROZEN ERR unknown net=$NET" >&2; exit 2;; esac
if [ $# -ne 3 ]; then echo "usage: kln_frozen.sh [--net NET] BUILD_ROOT LOGDIR OUTDIR" >&2; exit 2; fi
B=$(readlink -m "$1"); D=$(readlink -m "$2"); O=$(readlink -m "$3")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
PY=${KLN_PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}
PR=script/perf/params
J_L=$PR/l_palmyra213.m3121.json; J_N=${KLN_N_JSON:-$PR/n_${NET}_palmyra213.m3121.json}; J_C=$PR/kcyc_unet_iss.json
J_DW=$PR/kdw_palmyra213.m3112c.json; J_K=${KLN_K_JSON:-$PR/k_palmyra213.m3112c.json}
case "$O/" in "$B/kln/"*|"$B/kln/") echo "KLNFROZEN ERR outdir=$O is under BUILD_ROOT/kln"; exit 2;; esac
cd "$ROOT" || exit 2
for c in off $C2; do for r in a b c; do
  [ -s "$D/${NET}_$c.$r.log" ] || { echo "KLNFROZEN ERR missing log $D/${NET}_$c.$r.log"; exit 2; }
done; done
mkdir -p "$O" || exit 2
for j in $J_K $J_L $J_N $J_C $J_DW; do
  m=$(md5sum "$j" 2>&1 | awk '{print $1}'); echo "KLNFROZEN PARAMS file=$j md5=$m"
  [ -s "$j" ] || { echo "KLNFROZEN ERR missing params $j"; exit 2; }
done
L() { echo "$D/${NET}_$1.$2.log"; }
RUNS="off=$(L off a),$(L off b),$(L off c) $C2=$(L $C2 a),$(L $C2 b),$(L $C2 c)"
FIT="--fit off=$(L off a) $C2=$(L $C2 a)"
CHK="--check off=$(L off b),$(L off c) $C2=$(L $C2 b),$(L $C2 c)"
CHK3="--check $RUNS"
# 1 L
$PY -I script/perf/m3_site.py --net $NET --lpred --kfirst site --params $J_L --tsv "$O/lpred.tsv" --build-root "$B" $RUNS > "$O/l.out" 2>&1; rl=$?
grep -E '^(LCHK|LCHKRUN) ' "$O/l.out"
# 2 N
$PY -I script/perf/n_fit.py --net $NET --predict --ovl --refit-a no --params $J_N --tsv "$O/npred.tsv" --build-root "$B" $RUNS > "$O/n.out" 2>&1; rn=$?
grep -E '^NCHK ' "$O/n.out"
# 3 core
$PY -I script/perf/k_diag.py --net $NET --kdelta core --core-tsv "$O/kcore.tsv" --delta-tsv "$O/kdelta_core.tsv" --build-root "$B" $FIT $CHK > "$O/core.out" 2>&1; rc_=$?
grep -E '^(KCORE ALL|KDELTA) ' "$O/core.out"
# 4 dw
$PY -I script/perf/k_dmafeat.py --net $NET --perch --ddrt --dwpredict --dw-params $J_DW --dw-tsv "$O/dwhat.tsv" --build-root "$B" off=$(L off a) $C2=$(L $C2 a) > "$O/dw.out" 2>&1; rd=$?
grep -E '^KDWP ' "$O/dw.out"
# 5 K
$PY -I script/perf/k_diag.py --net $NET --khat-static --params $J_K --core-tsv "$O/kcore.tsv" --dwhat "$O/dwhat.tsv" --npred "$O/npred.tsv" --lpred "$O/lpred.tsv" --khat-tsv "$O/khat.tsv" --build-root "$B" $CHK3 > "$O/k.out" 2>&1; rk=$?
grep -E '^KHATS ' "$O/k.out"
# 6 kln_site
$PY -I script/perf/kln_site.py --net $NET --k-tsv "$O/khat.tsv" --k-label frozen --site-src log --r2 0.90 --max-rel 0.20 --lpred "$O/lpred.tsv" --npred "$O/npred.tsv" --site-tsv "$O/site_pairs.tsv" --build-root "$B" $CHK3 > "$O/kln.out" 2>&1; rq=$?
grep -E '^(KLNSRC|KLNHAT|KLNSITES|KLN AUDIT) ' "$O/kln.out"
echo "KLNFROZEN rc_l=$rl rc_n=$rn rc_core=$rc_ rc_dw=$rd rc_k=$rk rc_kln=$rq"
for x in $rl $rn $rc_ $rd $rk $rq; do [ "$x" -gt 1 ] && exit 2; done
[ "$rq" = 1 ] && exit 1
exit 0
