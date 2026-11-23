"""Core vector paths of the GEMM ELF's modes vs their scalar code, through the x86 AIE-API emulation."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_pool_vector_path_matches_scalar(tmp_path):
    """aie_modes_poolv.h (MAX / SUM, stride 1 / 2, int8 / uint8) == AIE_MODE_POOL bit for bit"""
    exe = tmp_path / "poolv"
    subprocess.run(["g++", "-std=c++20", "-O1", "-DAIE_POOLV_EMU", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "poolv_test.cc")], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 mismatching" in r.stdout


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_bf16_mode_layout_and_rounds(tmp_path):
    """aie_modes_bf16.h: the core's bf16 mmul rounds over k-round windows == the f32 reference."""
    exe = tmp_path / "bf16"
    subprocess.run(["g++", "-std=c++20", "-O1", "-DAIE_BF16_EMU", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "bf16_test.cc")], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "0 mismatching" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_sdpa_vector_item_matches_reference(tmp_path):
    """aie_modes_sdpa.h: the core's vector item == the scalar reference, integer for integer."""
    exe = tmp_path / "sdpa"
    subprocess.run(["g++", "-std=c++20", "-O1", "-DAIE_SDPA_EMU", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "sdpa_test.cc")], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "0 mismatching" in r.stdout, r.stdout + r.stderr


def test_wq_flush_matches_reference(tmp_path):
    """aie_modes_wq.h: the core's group flush vs the C reference and a float64 recomputation."""
    exe = tmp_path / "wq"
    subprocess.run(["g++", "-O1", "-std=c++20", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "wq_test.cc")], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_fpool_vector_item_matches_reference(tmp_path):
    """aie_modes_fpool.h: the core's f32 pooling item == the plain C reference."""
    exe = tmp_path / "fpool"
    subprocess.run(["g++", "-O1", "-std=c++20", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "fpool_test.cc")], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0 and "0 mismatching" in out.stdout, out.stdout
