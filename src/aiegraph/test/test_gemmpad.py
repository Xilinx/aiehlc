import numpy as np
import pytest

from aiegraph.gemmpad import pad_mkn, pad_gemm_operands, unpad_output


@pytest.mark.parametrize("mkn,want", [((4096, 36, 320), (4096, 64, 320)),   # UNet conv_in
                                      ((4096, 2880, 4), (4096, 2880, 64)),  # UNet conv_out
                                      ((64, 64, 64), (64, 64, 64)),
                                      ((1, 65, 127), (64, 128, 128))])
def test_pad_mkn(mkn, want):
    assert pad_mkn(*mkn) == want


@pytest.mark.parametrize("m,k,n", [(4096, 36, 320), (4096, 2880, 4), (100, 70, 3)])
def test_padded_gemm_matches(m, k, n):
    r = np.random.default_rng(0)
    a = r.standard_normal((m, k)).astype(np.float32)
    b = r.standard_normal((k, n)).astype(np.float32)
    ap, bp = pad_gemm_operands(a, b)
    assert ap.shape == pad_mkn(m, k, n)[:2] and bp.shape == pad_mkn(m, k, n)[1:]
    # padding is zeros, original block untouched
    assert np.array_equal(ap[:m, :k], a) and np.array_equal(bp[:k, :n], b)
    assert not ap[m:].any() and not ap[:, k:].any() and not bp[k:].any() and not bp[:, n:].any()
    yp = ap @ bp
    # padded rows/cols of the output are exact zeros
    assert not yp[m:].any() and not yp[:, n:].any()
    y = unpad_output(yp, m, n)
    assert y.shape == (m, n) and y.flags.c_contiguous
    np.testing.assert_allclose(y, a @ b, rtol=1e-5, atol=1e-4)


def test_aligned_is_noop():
    a = np.ones((64, 128), np.float32); b = np.ones((128, 64), np.float32)
    ap, bp = pad_gemm_operands(a, b)
    assert ap is a and bp is b


def test_bad_shapes():
    with pytest.raises(ValueError):
        pad_gemm_operands(np.ones((4, 5)), np.ones((6, 2)))
    with pytest.raises(ValueError):
        unpad_output(np.ones((4, 4)), 8, 4)
    with pytest.raises(ValueError):
        pad_mkn(0, 1, 1)
