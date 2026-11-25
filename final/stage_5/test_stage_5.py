import numpy as np
import pytest
from .fista import adjoint, forward, recover, shrink
from .fourier_motion import register
from .feature_tracking import observations


def test_adjoint_sparse_recovery():
    rng = np.random.default_rng(3)
    shape = (8, 9, 10)
    theta = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    mask = rng.random(shape) < 0.4
    v = rng.normal(size=mask.sum()) + 1j * rng.normal(size=mask.sum())
    np.testing.assert_allclose(
        np.vdot(forward(theta, mask), v), np.vdot(theta, adjoint(v, mask)), rtol=1e-11
    )
    t, y, x = np.indices(shape)
    truth = np.cos(2 * np.pi * (t / 8 + x / 10)) + 0.4 * np.sin(2 * np.pi * y / 9)
    field, h = recover(truth[mask], mask, lam=0.001, tol=1e-7, max_iter=600)
    assert h[-1]["converged"]
    assert np.sqrt(np.mean((field - truth) ** 2)) < 0.001
    np.testing.assert_allclose(shrink(np.array([3 + 4j]), 2), [1.8 + 2.4j])


def test_registration_sign():
    im = np.random.default_rng(2).random((80, 80), dtype=np.float32)
    shifts = [0, 1, 0, -1] * 2
    frames = np.stack([np.roll(im, i, axis=1) for i in shifts])
    stack, _, field, _, valid, _ = register(frames, step=10, lam=0.01)
    assert field.shape == (*frames.shape, 2) and valid.shape == stack.shape
    assert (
        np.abs(field[:, 10:-10, 10:-10, 0] - np.array(shifts)[:, None, None]).mean()
        < 0.15
    )
    assert np.abs(stack[:, 10:-10, 10:-10] - im[10:-10, 10:-10]).mean() < 0.06


def test_missing_and_zero_motion():
    tracks = np.float32([[[3, 6]], [[5, 6]], [[np.nan, np.nan]]])
    v, m = observations(tracks, 16, 16, 4)
    np.testing.assert_equal(v[m], [[-1, 0], [1, 0]])
    f, h = recover(np.zeros(m.sum()), m)
    assert not f.any() and h[-1]["converged"]
    with pytest.raises(ValueError, match="corners"):
        register(np.zeros((4, 32, 32), np.float32))


def test_split_confidence_and_refinement():
    im = np.random.default_rng(3).random((64, 64), dtype=np.float32)
    frames = np.repeat(im[None], 4, axis=0)
    stack, _, _, _, mask, confidence = register(frames, diagnostics=True)
    np.testing.assert_allclose(confidence[mask], 1, atol=1e-5)
    stack, _, _, _, _, _ = register(frames, refine=True)
    np.testing.assert_allclose(stack[:, 8:-8, 8:-8], frames[:, 8:-8, 8:-8], atol=0.001)
