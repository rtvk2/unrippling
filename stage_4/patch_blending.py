import numpy as np


def starts(length, patch, stride):
    last = max(length - patch, 0)
    return sorted(set(range(0, last + 1, stride)) | {last})


def window(height, width):
    # Interior Hann samples keep outside boundaries strictly positive.
    return np.outer(np.hanning(height + 2)[1:-1], np.hanning(width + 2)[1:-1])
