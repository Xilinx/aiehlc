"""GAP-B x86 test: partition.json ``keep_layout`` on the bk-sdm-tiny UNet (unet_aie.keep_layout_plan,
build_unet.keep_layout_step, aiegraph.partition.plan_keep_layout).

1. test_no_field_unchanged: the shipped partition.json (no keep_layout) -> keep_layout_step returns None, writes
   nothing, the mapped layers are not touched (the --aie build stays byte-identical; test_attn_sites checks
   the generated code itself).
2. test_unet_proj_in_kept: real UNet ONNX, attention block 0: proj_in Conv kept in "nhwc"; the ONNX
   Transpose(0,2,3,1) + Reshape [1,1024,320] after it are labels only, the consumers (norm1 LayerNorm + the
   residual Add, put on the AIE as P4a will) read the AIE buffer as [1,1024,320] row-major. The tags land on
   the mapped layer dicts, keep_layout.json + the KEEP_LAYOUT line are written.
3. test_unet_proj_in_label_proof: numpy on a real-size tensor: the nhwc AIE image of the proj_in output IS the
   host Transpose+Reshape result (bit-identical, no data movement); nchw is not -> plan rejects it.
4. test_unet_host_consumer_rejected: same row with the consumers left on the host -> AiegraphError naming the
   LayerNormalization (today's partition cannot keep the layout there; never silently ignored).
Run: python3 -m pytest -q example/aiegraph/unet_bk_sdm_tiny/test_layout_keep.py   (numpy, onnx)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import build_unet as bu  # noqa: E402
import unet_aie as ua  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

ONNX = Path(os.environ.get("GATT_ONNX", "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx"))
A0 = "/u/down_blocks.0/attentions.0"
PROJ_IN = f"{A0}/proj_in/Conv"
NORM1 = f"{A0}/transformer_blocks.0/norm1/LayerNormalization"
RES = f"{A0}/transformer_blocks.0/Add"
need_onnx = pytest.mark.skipif(not ONNX.is_file() and not os.environ.get("GATT_REQUIRE_FIXTURE"),
                               reason=f"{ONNX} missing")


@pytest.fixture(scope="module")
def model():
    import onnx
    return onnx.load(str(ONNX))


def test_no_field_unchanged(tmp_path):
    part = json.loads((HERE / "partition.json").read_text())
    assert not any("keep_layout" in L for L in part["layers"])
    layers = [dict(onnx=L["node"], cid=L["idx"]) for L in part["layers"]]
    before = json.dumps(layers)
    assert bu.keep_layout_step(ua, ONNX, part, layers, tmp_path / "o") is None
    assert json.dumps(layers) == before and not (tmp_path / "o").exists()


def _part():
    return {"layers": [{"node": PROJ_IN, "cls": "conv", "keep_layout": {"layout": "nhwc"}},
                       {"node": NORM1, "cls": "norm"}, {"node": RES, "cls": "add"}]}


@need_onnx
def test_unet_proj_in_kept(model, tmp_path, capsys):
    layers = [dict(onnx=PROJ_IN, cid=2), dict(onnx=NORM1, cid=100), dict(onnx=RES, cid=101)]
    keep = bu.keep_layout_step(ua, model, _part(), layers, tmp_path)
    assert keep["kept"] == 1 and keep["label_ops"] == 2
    e = keep["edges"][0]
    assert e["node"] == PROJ_IN and e["layout"] == "nhwc"
    assert e["label_ops"] == [f"{A0}/Transpose", f"{A0}/Reshape"]
    assert sorted(e["consumers"]) == sorted([NORM1, RES])
    assert e["views"] == [[1, 1024, 320], [1, 1024, 320]]
    assert layers[0]["keep_layout"] == "nhwc"
    assert layers[1]["keep_in"] == [PROJ_IN] and layers[2]["keep_in"] == [PROJ_IN]
    assert json.loads((tmp_path / "keep_layout.json").read_text()) == keep
    assert "KEEP_LAYOUT edges=1 label_ops=2 consumers=2" in capsys.readouterr().out


@need_onnx
def test_unet_proj_in_label_proof(model):
    import onnx
    tr = next(n for n in model.graph.node if n.name == f"{A0}/Transpose")
    perm = list(onnx.helper.get_attribute_value(tr.attribute[0]))
    assert perm == [0, 2, 3, 1]
    rng = np.random.default_rng(0)
    y = rng.standard_normal((1, 320, 32, 32)).astype(np.float32)           # proj_in output, logical NCHW
    host = y.transpose(perm).reshape(1, 1024, 320)                           # what the host Transpose+Reshape make
    aie_nhwc = np.ascontiguousarray(y.transpose(0, 2, 3, 1)).tobytes()       # kacc transposed walk Y^T [M][N]
    assert np.frombuffer(aie_nhwc, np.float32).reshape(1, 1024, 320).tobytes() == host.tobytes()
    aie_nchw = np.ascontiguousarray(y).tobytes()                             # kacc Y [N][M]
    assert np.frombuffer(aie_nchw, np.float32).reshape(1, 1024, 320).tobytes() != host.tobytes()
    p = _part()
    p["layers"][0]["keep_layout"] = True                                     # nchw -> needs a real transpose
    with pytest.raises(AiegraphError, match="moves data|real transpose"):
        ua.keep_layout_plan(model, p)


@need_onnx
def test_unet_host_consumer_rejected(model):
    p = {"layers": [_part()["layers"][0]]}
    with pytest.raises(AiegraphError, match="LayerNormalization.*host"):
        ua.keep_layout_plan(model, p)
