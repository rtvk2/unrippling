import numpy as np
import pytest
from .baselines import temporal_mean, temporal_median


@pytest.mark.parametrize("fn,target", [(temporal_mean, 4), (temporal_median, 3)])
def test_time_axis(fn, target):
    f = np.broadcast_to(np.array([2, 3, 7])[:, None, None, None], (3, 5, 9, 3))
    np.testing.assert_equal(fn(f), np.full((5, 9, 3), target))


@pytest.mark.parametrize("bad", [[], np.ones((2, 4)), np.full((3, 4, 5), np.nan)])
def test_invalid(bad):
    with pytest.raises(ValueError):
        temporal_mean(bad)


def test_local_folder_and_no_reference(tmp_path):
    import cv2
    from utils.data_loader import load

    for index, value in [(2, 85), (1, 34)]:
        cv2.imwrite(
            str(tmp_path / f"{index:03d}.png"), np.full((12, 18), value, np.uint8)
        )
    frames, gt = load("custom", 2, 8, tmp_path)
    assert gt is None
    np.testing.assert_array_equal(np.rint(frames[:, 0, 0] * 255), [34, 85])


def test_interrupted_dataset_download(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    from data import download_data as downloader

    class Response(io.BytesIO):
        headers = {}

    digest = hashlib.sha1(b"blob 4\0abcd").hexdigest()
    monkeypatch.setattr(downloader, "ROOT", tmp_path)
    monkeypatch.setattr(downloader, "FILES", {"Checker.avi": (4, digest)})
    responses = iter([Response(b"ab"), Response(b"abcd")])
    monkeypatch.setattr(
        downloader.urllib.request, "urlopen", lambda *args, **kwargs: next(responses)
    )
    with pytest.raises(IOError, match="Incomplete"):
        downloader.download(["checker"], 6)
    dest = tmp_path / "data/real"
    assert not (dest / "Checker.avi").exists()
    with pytest.raises(SystemExit, match="Budget exhausted"):
        downloader.download(["checker"], 5)
    downloader.download(["checker"], 6)
    downloader.download(["checker"], 6)
    assert (dest / "Checker.avi").read_bytes() == b"abcd"
    assert json.loads((dest / "downloads.json").read_text())["charged_bytes"] == 6


def test_metrics_identical_protocol():
    from utils.metrics import measure

    ref = np.full((24, 32), 0.5, np.float32)
    im = ref.copy()
    im[2:-2, 2:-2] = 0.4
    result = measure(im, ref, 2, False)
    assert abs(result["psnr"] - 20) < 1e-4
    assert (
        abs(result["ssim"] - (2 * 0.4 * 0.5 + 0.0001) / (0.4**2 + 0.5**2 + 0.0001))
        < 1e-5
    )
    assert not measure(im, None, 2, False)
