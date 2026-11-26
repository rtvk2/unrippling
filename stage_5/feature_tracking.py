"""Sparse displacements at track centroids, in native pixel units."""

import cv2
import numpy as np
from stage_2.registration import to_gray, _u8


def trajectories(frames, corners=700, fb_limit=1.0):
    video = _u8(to_gray(frames))
    first = cv2.goodFeaturesToTrack(video[0], corners, 0.01, 4)
    if first is None:
        raise ValueError("insufficient textured corners for Fourier registration")
    # Track directly to frame zero to avoid accumulating sequential drift.
    tracks = np.full((len(video), len(first), 2), np.nan, np.float32)
    tracks[0] = first[:, 0]
    h, w = video.shape[1:]
    for t, frame in enumerate(video[1:], 1):
        point, status, _ = cv2.calcOpticalFlowPyrLK(video[0], frame, first, None)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(frame, video[0], point, None)
        xy = point[:, 0]
        ok = (status[:, 0] & reverse[:, 0]).astype(bool)
        ok &= np.linalg.norm(back[:, 0] - first[:, 0], axis=1) <= fb_limit
        ok &= ((xy >= 0) & (xy <= [w - 1, h - 1])).all(axis=1)
        tracks[t, ok] = xy[ok]
    keep = np.isfinite(tracks[..., 0]).sum(axis=0) >= max(1, np.ceil(0.7 * len(video)))
    if keep.sum() < 4:
        raise ValueError("fewer than four reliable KLT trajectories")
    return tracks[:, keep]


def observations(tracks, h, w, step):
    center = np.nanmean(tracks, axis=0)
    gh, gw = max(2, int(np.ceil(h / step))), max(2, int(np.ceil(w / step)))
    x, y = np.rint(center * [(gw - 1) / (w - 1), (gh - 1) / (h - 1)]).astype(int).T
    shape = (len(tracks), gh, gw)
    t, j = np.where(np.isfinite(tracks[..., 0]))
    cells = (t, y[j], x[j])
    count = np.zeros(shape, np.int32)
    np.add.at(count, cells, 1)
    values = np.zeros(shape + (2,), np.float32)
    np.add.at(values, cells, tracks[t, j] - center[j])
    values /= count.clip(1)[..., None]
    return values, count > 0
