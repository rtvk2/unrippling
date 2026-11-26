import numpy as np
import pytest
from stage_3.rpca import rpca, svt


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svt(dtype):
    a = np.random.default_rng(4).normal(size=(48, 9)).astype(dtype)
    u, s, v = np.linalg.svd(a, full_matrices=False)
    got, rank = svt(a, 5)
    np.testing.assert_allclose(got, (u * (s - 5).clip(0)) @ v, atol=2e-6)
    assert rank == (s > 5).sum()


def test_sparse_corruption():
    rng = np.random.default_rng(2)
    low = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 40))
    sparse = (rng.random(low.shape) < 0.015) * rng.choice([-4.0, 4.0], low.shape)
    l, s, h = rpca(low + sparse)
    assert h[-1]["converged"]
    assert np.linalg.norm(l - low) / np.linalg.norm(low) < 1e-3
    assert np.linalg.norm(s - sparse) / np.linalg.norm(sparse) < 1e-3
    assert not rpca(low + sparse, max_iter=1)[2][-1]["converged"]


def test_zero():
    low, s, h = rpca(np.zeros((6, 2)))
    assert not low.any() and not s.any() and h[-1]["residual"] == 0


def test_tiny_nonzero_matrix():
    reference = np.full((64, 8), 1e-27, np.float32)
    low, sparse, h = rpca(reference, tol=1e-5)
    assert h[-1]["converged"]
    np.testing.assert_allclose((low + sparse) / reference, 1.0, rtol=1e-5)
