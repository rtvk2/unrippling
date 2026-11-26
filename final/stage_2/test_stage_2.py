import numpy as np
from .registration import farneback, register
from .warping import warp, validity
from utils.synthetic import smooth_flows, inverse_flow


def test_translation_and_border():
    im = np.random.default_rng(8).random((72, 80), dtype=np.float32)
    frame = np.roll(im, 2, axis=1)
    flow = farneback(im, frame, 21)
    assert np.median(np.linalg.norm(flow[10:-10, 10:-10] - [2, 0], axis=-1)) < 0.08
    f = np.full_like(flow, [2, 0])
    np.testing.assert_equal(warp(frame, f)[:, :-2], im[:, :-2])
    assert validity(f)[:, :-2].all() and not validity(f)[:, -2:].any()


def test_nonrigid():
    rng = np.random.default_rng(12)
    im = rng.random((80, 80), dtype=np.float32)
    fields = smooth_flows(8, 80, 80, 1.2, 8, rng)
    frames = np.stack([warp(im, f).clip(0, 1) for f in fields])
    reg, _, est, _, mask, conf = register(
        frames, rounds=2, workers=1, diagnostics=True, flow_sigma=2
    )
    target = np.stack([inverse_flow(f) for f in fields])
    crop = np.s_[:, 8:-8, 8:-8]
    assert np.linalg.norm((est - target)[crop], axis=-1).mean() < 0.65
    assert reg.shape == mask.shape == conf.shape == frames.shape
    assert reg.min() >= 0 and reg.max() <= 1


def test_static():
    f = np.full((4, 30, 40), 0.3, np.float32)
    stack, template, field, _ = register(f, rounds=2)
    np.testing.assert_allclose(stack, f)
    np.testing.assert_allclose(template, f[0])
    np.testing.assert_allclose(field, 0)


def test_crop_preserves_time_and_components():
    from utils.metrics import spatial_crop

    f = np.arange(3 * 10 * 16 * 2).reshape(3, 10, 16, 2)
    np.testing.assert_equal(spatial_crop(f, 2), f[:, 2:8, 2:14])
    np.testing.assert_equal(spatial_crop(f, 0), f)


def test_invalid_input_and_overshoot():
    import pytest

    with pytest.raises(ValueError, match="finite"):
        register(np.full((3, 16, 24), np.nan))
    im = np.zeros((16, 32), np.float32)
    im[:, 16:] = 1
    f = np.full((*im.shape, 2), [0.5, 0], np.float32)
    raw = warp(im, f)
    assert raw.min() < 0 and raw.max() > 1


def test_rounds_resample_originals():
    rng = np.random.default_rng(88)
    im = rng.random((48, 64), dtype=np.float32)
    f = np.stack([np.roll(im, dx, axis=1) for dx in [-1, 0, 1]])
    aligned, _, field, _ = register(f, rounds=3)
    np.testing.assert_array_equal(
        aligned, np.stack([warp(frame, d).clip(0, 1) for frame, d in zip(f, field)])
    )


def test_motion_summary_uses_first_round_interior():
    rng = np.random.default_rng(21)
    im = rng.random((42, 48), dtype=np.float32)
    frames = np.stack([np.roll(im, k, axis=1) for k in (-1, 0, 1, 0, 2)])
    _, _, flow, history = register(frames, rounds=1)
    expected = np.linalg.norm(flow[::4, 16:-16, 16:-16], axis=-1).mean()
    np.testing.assert_allclose(history[0]["sampled_first_round_px"], expected)
