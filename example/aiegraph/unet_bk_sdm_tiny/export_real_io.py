#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Sprint unet-board-apu-aie P1a — export the REAL UNet step input + x86 ORT reference outputs for the board CMP.

Data: modelzoo/artifacts/bk-sdm-tiny/pipeline_ref_256/p00_steps.npz (FP32 torch diffusers run, prompt 0, seed 1000).
Batch order of unet_sample / encoder_hidden_states is [uncond, cond]: see zoo/sd_unet/pipeline_reference.py
line 10 ("UNet batch 2 = [uncond, cond]") and line 99 (torch.cat([emb[1], emb[0]]) # uncond,cond)  -> cond = 1.
The script also proves it: ORT(cond input) must match the recorded torch unet_noise_pred[step, 1].

Writes to --out:
  sample.bin (f32 [1,4,32,32]) timestep_i32.bin timestep_i64.bin (int [1]) ehs.bin (f32 [1,77,768])
  part01_out.bin (ORT part_01_stem_in output, f32 [1,320,32,32])  unet_out.bin (ORT unet_sim noise_pred, f32 [1,4,32,32])
  sample.crc32  unet_in.crc32 (crc32 of sample.bin+timestep_i32.bin+ehs.bin)   — 8-hex lowercase, no newline issue
  p1_ref.h  (embedded reference arrays + crcs for src/aiegraph/runtime/cmp_cos.c)   meta.json
Last line: REAL_IO OK parts=25 ... sim_vs_full cos=...   (or REAL_IO FAIL ..., rc 1)
"""
import argparse, json, os, sys, zlib
import numpy as np, onnxruntime as ort

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src")))
from aiednn import runtime_source  # noqa: E402  -- p1_ref.h text: src/aiegraph/runtime/cmp_ref_*.h


def fill(t, **kv):
    for k, v in kv.items():
        t = t.replace(f"@{k}@", str(v))
    assert "@" not in t, t
    return t


A = "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny"
p = argparse.ArgumentParser()
p.add_argument("--step", type=int, default=0)
p.add_argument("--branch", choices=["cond", "uncond"], default="cond")
p.add_argument("--model", default=f"{A}/unet_sim.onnx")
p.add_argument("--full", default=f"{A}/unet.onnx")
p.add_argument("--steps_npz", default=f"{A}/pipeline_ref_256/p00_steps.npz")
p.add_argument("--parts", default=f"{A}/p1/parts")
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
BIDX = {"uncond": 0, "cond": 1}[a.branch]  # [uncond, cond] — pipeline_reference.py:99

d = np.load(a.steps_npz)
sample = np.ascontiguousarray(d["unet_sample"][a.step, BIDX:BIDX + 1], dtype="<f4")          # [1,4,32,32]
t = int(d["timesteps"][a.step])
ehs = np.ascontiguousarray(d["encoder_hidden_states"][BIDX:BIDX + 1], dtype="<f4")           # [1,77,768]
torch_ref = d["unet_noise_pred"][a.step, BIDX:BIDX + 1].astype(np.float32)
ts32, ts64 = np.array([t], dtype="<i4"), np.array([t], dtype="<i8")
assert sample.shape == (1, 4, 32, 32) and ehs.shape == (1, 77, 768)

cos = lambda u, v: float((u.astype(np.float64) * v).sum() / (np.linalg.norm(u.astype(np.float64)) * np.linalg.norm(v.astype(np.float64)) + 1e-30))
mabs = lambda u, v: float(np.abs(u.astype(np.float64) - v).max())
so = ort.SessionOptions()
S = lambda f: ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
feed = {"sample": sample, "timestep": ts64, "encoder_hidden_states": ehs}

sim = S(a.model).run(None, feed)[0].astype("<f4")
full = S(a.full).run(None, feed)[0]
c_sf = cos(sim, full)

# Steve's 25 partitions chained (same as p1_run.py) — the board graph; part_01 output is the per-part reference
parts = json.load(open(f"{a.parts}/partitions.json"))["partitions"]
env = dict(feed); part01 = None
for r in parts:
    s = S(f"{a.parts}/{r['file']}")
    ins = [x[0] for x in r["inputs"]]; outs = [x[0] for x in r["outputs"]]
    env.update(zip(outs, s.run(outs, {k: env[k] for k in ins})))
    if r["name"] == "stem_in":
        part01 = np.ascontiguousarray(env[outs[0]], dtype="<f4")
chain = env[parts[-1]["outputs"][0][0]]
c_chain, c_torch = cos(chain, sim), cos(sim, torch_ref)
c_other = cos(sim, d["unet_noise_pred"][a.step, 1 - BIDX:2 - BIDX])

wr = lambda n, x: open(f"{a.out}/{n}", "wb").write(x.tobytes())
wr("sample.bin", sample); wr("timestep_i32.bin", ts32); wr("timestep_i64.bin", ts64); wr("ehs.bin", ehs)
wr("part01_out.bin", part01); wr("unet_out.bin", sim)
crc_s = "%08x" % (zlib.crc32(sample.tobytes()) & 0xFFFFFFFF)
crc_in = "%08x" % (zlib.crc32(sample.tobytes() + ts32.tobytes() + ehs.tobytes()) & 0xFFFFFFFF)
open(f"{a.out}/sample.crc32", "w").write(crc_s + "\n")
open(f"{a.out}/unet_in.crc32", "w").write(crc_in + "\n")

def carr(name, x):
    v = x.reshape(-1).astype(np.float32)
    body = ",\n".join(",".join("%.9gf" % e for e in v[i:i + 8]) for i in range(0, v.size, 8))
    return fill(runtime_source("cmp_ref_array.h"), NAME_U=name.upper(), N=v.size, NAME=name, BODY=body)
with open(f"{a.out}/p1_ref.h", "w") as f:
    f.write(fill(runtime_source("cmp_ref_head.h"), STEP=a.step, BRANCH=a.branch, T=t, ORT=ort.__version__,
                 CRC_S=crc_s, CRC_IN=crc_in))
    f.write(carr("ref_part01_out", part01)); f.write(carr("ref_unet_out", sim))

meta = dict(steps_npz=a.steps_npz, step=a.step, branch=a.branch, batch_idx=BIDX, timestep=t, model=a.model,
            sample_crc32=crc_s, unet_in_crc32=crc_in, sim_vs_full_cos=c_sf, sim_vs_full_maxabs=mabs(sim, full),
            chain_vs_sim_cos=c_chain, chain_vs_sim_maxabs=mabs(chain, sim), sim_vs_torch_cos=c_torch,
            sim_vs_torch_maxabs=mabs(sim, torch_ref), sim_vs_other_branch_cos=c_other, ort=ort.__version__, parts=len(parts))
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
ok = len(parts) == 25 and c_sf >= 0.99999 and c_chain >= 0.99999 and c_torch >= 0.999 and part01.shape == (1, 320, 32, 32)
print(f"REAL_IO {'OK' if ok else 'FAIL'} parts={len(parts)} data=p00/step{a.step}/{a.branch} batch_idx={BIDX} t={t} "
      f"sample_crc32={crc_s} unet_in_crc32={crc_in} chain_vs_sim cos={c_chain:.7f} sim_vs_torch cos={c_torch:.7f} "
      f"(other_branch cos={c_other:.5f}) sim_vs_full cos={c_sf:.7f}")
sys.exit(0 if ok else 1)
