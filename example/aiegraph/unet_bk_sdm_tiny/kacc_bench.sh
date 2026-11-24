#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# P4bk: build the one-launch K-in-core bench (bf16_gemm_kacc.cc) for one conv layer from an op_golden dir.
#   kacc_bench.sh --golden <op_golden/.../cNNN_conv_...> --out <dir>   ->  <dir>/main.elf
# The golden dir holds a.npy (1xCxHxW f32), b.npy (OxCx3x3 f32), c.npy (O f32), y.npy (1xOxHxW f32), meta.json.
# Writes into <dir>: kacc_{w,x,bias,y}.bin (W/X im2col as pre-tiled bf16, see bf16_gemm_kacc.cc), kacc_data.h
# (.incbin of those files, host side only), then runs script/aiehlc.sh --aie-version 5 in <dir>.
#   kacc_bench.sh --gemm --golden <op_golden/p0_matmul_gemm_s0_cond/cNNN_matmul_...> --out <dir>   (PDK)
# --gemm: a MatMul golden (a.npy [1,M,K], b.npy [K,N], y.npy) through kacc_gemm.py: the generic gemm_kacc instance
# (aiegraph.kaccbench: kincore gemm_kacc template, M padded to 64) + a bench main (warm-up pass, timed pass, KCMP vs y.npy), built
# like the UNet --kincore path: --launch-opt warm defs + --skip-bss + --ctrlpkt aot pragmas.
#   kacc_bench.sh --bmm --golden <op_golden/p0_matmul_gemm_s0_cond/cNNN_matmul_1x8x...> --out <dir>   (PBK)
# --bmm: attention bmm golden (a.npy [1,H,M,K], b.npy [1,H,K,N]); A and B are embedded in their NATURAL f32 layout
# and packed by the host at run time (src/mlir/runtime/aie_kacc_pack.h), M/K/N padded to 64 (aiegraph.gemmpad),
# H heads x S launches; [BENCH] prints aie_wall_ms, host_ms (A/B pack + flush + invalidate + unpad) and launches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
GOLDEN="" OUT="" GEMM=0 BMM=0
while [ $# -gt 0 ]; do
    case "$1" in
    --golden) GOLDEN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --gemm) GEMM=1; shift ;;
    --bmm) GEMM=1; BMM=1; shift ;;
    *) echo "usage: $0 [--gemm|--bmm] --golden <dir> --out <dir>" >&2; exit 2 ;;
    esac
done
[ -n "$GOLDEN" ] && [ -n "$OUT" ] || { echo "usage: $0 --golden <dir> --out <dir>" >&2; exit 2; }
PY="${KACC_PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
echo "[kacc_bench] root=$ROOT golden=$GOLDEN out=$OUT commit=$(git -C "$HERE" rev-parse --short HEAD) gemm=$GEMM bmm=$BMM"

if [ "$GEMM" = 1 ]; then
    if [ "$BMM" = 1 ]; then "$PY" -I "$HERE/kacc_gemm.py" --bmm "$GOLDEN" "$OUT"; else "$PY" -I "$HERE/kacc_gemm.py" "$GOLDEN" "$OUT"; fi
    cd "$OUT"
    rm -rf aout
    # same launch options as build_unet.py --launch-opt warm (WARM_DEFS, aiehlc.sh --skip-bss): kernel loaded once
    export EXTRA_DEFS="${EXTRA_DEFS:-} -DAIEHLC_ZFILL_WIDE=1 -DAIEHLC_PMLOAD_WIDE=1 -DAIEHLC_PM_WARM_SKIP=1 -DAIEHLC_SKIP_BSS_DEFAULT=1"
    set +eu
    # shellcheck disable=SC1091
    source "$ROOT/script/aiehlc.sh" --aie-version 5 --skip-bss --runtime-source-file "$OUT/kacc_gemm.cc"
    rc=$?
    set -u
    [ $rc -eq 0 ] && [ -f "$OUT/aout/main.elf" ] || { echo "[kacc_bench] aiehlc.sh failed rc=$rc"; exit 1; }
    if [ "$BMM" = 1 ] && grep -q 'DMA BD Config:.*iter_wrap' "$OUT/aout/worklocal/host.cc"; then
        # GAP-aiehlc-bmm-lowering item 2: aiehlc's 2-dim + iteration b-port BD gave wrong results on the board
        echo "[kacc_bench] REJECT: host.cc has a 2-dim+iter DMA BD (known-wrong on palmyra213, see kacc_gemm.main_bmm)"
        exit 1
    fi
    cp -f "$OUT/aout/main.elf" "$OUT/main.elf"
    ls -l "$OUT/main.elf"
    echo "[kacc_bench] BUILD_OK $OUT/main.elf"
    exit 0
fi

"$PY" -I - "$GOLDEN" "$OUT" "$ROOT" <<'EOF'
import sys, json, hashlib, numpy as np
g, out = sys.argv[1], sys.argv[2]
import re  # kacc_data.h text: bench/kacc_bench_data_conv.h (loaded by path)
meta = json.load(open(f"{g}/meta.json"))
x = np.load(f"{g}/a.npy").astype(np.float32); w = np.load(f"{g}/b.npy").astype(np.float32)
bias = np.load(f"{g}/c.npy").astype(np.float32); y = np.load(f"{g}/y.npy")
assert y.dtype == np.float32, y.dtype
ysha = hashlib.sha256(y.tobytes()).hexdigest()[:12]
assert meta["y_sha256"].startswith(ysha), (ysha, meta["y_sha256"])
_, C, H, W_ = x.shape; O = w.shape[0]; kh, kw = w.shape[2:]
assert meta["attrs"]["strides"] == [1, 1] and meta["attrs"]["pads"] == [1, 1, 1, 1] and (kh, kw) == (3, 3)
M, K, N = H * W_, C * 9, O
assert (M, K, N) == (64, 23040, 1280), (M, K, N)   # shapes compiled into bf16_gemm_kacc.cc
xp = np.pad(x[0], ((0, 0), (1, 1), (1, 1)))
A = np.stack([xp[:, i:i + H, j:j + W_] for i in range(3) for j in range(3)], 1).reshape(C * 9, M).T  # [M][c*9+kh*3+kw]
B = w.reshape(O, K)
def render(p, **kv):  # fill every @NAME@; missing or unknown parameters are errors
    t = open(p, encoding="utf-8", newline="").read()
    want = set(re.findall(r"@([A-Z][A-Z0-9_]*)@", t))
    assert set(kv) == want, (p, sorted(want - set(kv)), sorted(set(kv) - want))
    for k, v in kv.items():
        t = t.replace(f"@{k}@", str(v))
    return t
def bf16(a):  # round to nearest even
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)
Ab, Bb = bf16(A), bf16(B)
# self-check: bf16 GEMM vs golden (what the AIE should reach)
ref = (Ab.astype(np.uint32) << 16).view(np.float32).astype(np.float64) @ \
      (Bb.astype(np.uint32) << 16).view(np.float32).astype(np.float64).T + bias
yr = y.reshape(N, M).T.astype(np.float64)
cos = (ref * yr).sum() / np.sqrt((ref * ref).sum() * (yr * yr).sum())
print(f"[kacc_bench] M={M} K={K} N={N} bf16-gemm vs golden cos={cos:.6f} y_sha256={ysha}")
kc = K // 64
# the kernel runs Y^T = W . X^T: W is the M side (a), X the N side (b)
# a chunk (16 channels, 64 k) -> [mi][kb][mm][kk]   (8x8 M x K tiles)
Wt = Bb.reshape(N // 16, 2, 8, kc, 8, 8).transpose(0, 3, 1, 4, 2, 5).reshape(N // 16, kc, 16, 64)
# b chunk (16 pixels, 64 k)   -> [ni][kb][kk][nn]   (8x8 K x N tiles)
Xt = Ab.reshape(M // 16, 2, 8, kc, 8, 8).transpose(0, 3, 1, 4, 5, 2).reshape(M // 16, kc, 16, 64)
# laid back as rows of K so the DMA block {16 rows, 64 bf16} at chunk j carries the pre-tiled chunk
Wp = Wt.transpose(0, 2, 1, 3).reshape(N, K)
Xp = Xt.transpose(0, 2, 1, 3).reshape(M, K)
Wp.astype("<u2").tofile(f"{out}/kacc_w.bin"); Xp.astype("<u2").tofile(f"{out}/kacc_x.bin")
bias.astype("<f4").tofile(f"{out}/kacc_bias.bin"); y.astype("<f4").reshape(-1).tofile(f"{out}/kacc_y.bin")
data = "op_golden/" + "/".join(g.rstrip("/").split("/")[-2:])
data = data[:data.rindex("/") + 5]  # op_golden/<set>/cNNN
tag = data.split("/")[-1] + "-kacc"
with open(f"{out}/kacc_data.h", "w") as f:
    f.write(render(sys.argv[3] + "/example/aiegraph/unet_bk_sdm_tiny/bench/kacc_bench_data_conv.h", GOLDEN=g, M=M, K=K, N=N, DATA=data, TAG=tag, Y_SHA=ysha,
                          OUT=out))
print(f"[kacc_bench] data={data} tag={tag}")
EOF

cp -f "$ROOT/example/aiegraph/unet_bk_sdm_tiny/bench/bf16_gemm_kacc.cc" "$OUT/bf16_gemm_kacc.cc"
cd "$OUT"
rm -rf aout
set +eu
# shellcheck disable=SC1091
source "$ROOT/script/aiehlc.sh" --aie-version 5 --runtime-source-file "$OUT/bf16_gemm_kacc.cc"
rc=$?
set -u
[ $rc -eq 0 ] && [ -f "$OUT/aout/main.elf" ] || { echo "[kacc_bench] aiehlc.sh failed rc=$rc"; exit 1; }
cp -f "$OUT/aout/main.elf" "$OUT/main.elf"
ls -l "$OUT/main.elf"
echo "[kacc_bench] BUILD_OK $OUT/main.elf"
