#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GIN: write one UNet call's inputs + reference from a pipeline steps npz, in the real_io layout
that aiehlc_elfio.py patch takes (--in sample=... --in timestep_i32=... --in ehs=...).

  $AGPY rtio_prep.py --npz p00_steps.npz --step 12 --branch 1 --out <dir>
    -> <dir>/sample.bin (f32 [1,4,32,32]), timestep_i32.bin (i32 [1]), timestep_i64.bin,
       ehs.bin (f32 [1,77,768]), ref.npy (noise_pred f32 [1,4,32,32]), unet_in.crc32, meta.json

branch 1 = cond (the real_io of P1: p00/step0, branch 1 -> in_crc32 a5b911e2).
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--branch", required=True, type=int, choices=(0, 1))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    z = np.load(a.npz)
    s = np.ascontiguousarray(z["unet_sample"][a.step, a.branch], "<f4").reshape(1, 4, 32, 32)
    e = np.ascontiguousarray(z["encoder_hidden_states"][a.branch], "<f4").reshape(1, 77, 768)
    t = int(z["timesteps"][a.step])
    ref = np.ascontiguousarray(z["unet_noise_pred"][a.step, a.branch], "<f4").reshape(1, 4, 32, 32)
    a.out.mkdir(parents=True, exist_ok=True)
    s.tofile(a.out / "sample.bin")
    np.array([t], "<i4").tofile(a.out / "timestep_i32.bin")
    np.array([t], "<i8").tofile(a.out / "timestep_i64.bin")
    e.tofile(a.out / "ehs.bin")
    np.save(a.out / "ref.npy", ref)
    crc = "%08x" % zlib.crc32(s.tobytes() + np.array([t], "<i4").tobytes() + e.tobytes())
    (a.out / "unet_in.crc32").write_text(crc)
    (a.out / "meta.json").write_text(json.dumps(dict(steps_npz=str(a.npz.resolve()), step=a.step,
                                                     branch="cond" if a.branch else "uncond",
                                                     batch_idx=a.branch, timestep=t, unet_in_crc32=crc), indent=1))
    print(f"RTIO_PREP step={a.step} branch={a.branch} timestep={t} in_crc32={crc} out={a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
