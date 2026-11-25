import cv2
import numpy as np
import pytest
from .fusion import fuse


@pytest.mark.parametrize("shape", [(2, 7, 9), (4, 53, 41, 3)])
def test_constant_and_fallback(shape):
    f = np.full(shape, 0.4, np.float32)
    mask = np.ones(shape[:3], bool)
    mask[:, 2, 3] = False
    ref = f[0].copy()
    ref[2, 3] = 0.7
    got, _ = fuse(f, ref, mask, variant=1)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_complementary():
    sharp = np.random.default_rng(1).random((32, 64), dtype=np.float32)
    blur = cv2.GaussianBlur(sharp, (9, 9), 2)
    f = np.stack(
        [
            np.column_stack((sharp[:, :32], blur[:, 32:])),
            np.column_stack((blur[:, :32], sharp[:, 32:])),
        ]
    )
    got, idx = fuse(f, f.mean(0), patch=32, stride=32, variant=1)
    np.testing.assert_equal(idx[:, :, 0], [[0, 1]])
    np.testing.assert_allclose(got, sharp, atol=1e-6)


def test_confidence_selects():
    f = np.stack([np.zeros((8, 8)), np.ones((8, 8))]).astype(np.float32)
    got, idx = fuse(f, f.mean(0), confidence=f, variant=3)
    assert idx.item() == 1 and np.allclose(got, 1)
    got, idx = fuse(f, f.mean(0), variant=2, fraction=1)
    assert idx.size == 2 and np.allclose(got, 0.5)
