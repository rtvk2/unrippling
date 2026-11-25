"""Unitary partial Fourier synthesis. A selects distinct grid cells after the inverse FFT."""

import numpy as np
from scipy.fft import fftn, ifftn


def forward(theta, mask):
    return ifftn(theta, norm="ortho")[mask]


def adjoint(values, mask):
    grid = np.zeros(mask.shape, np.result_type(values.dtype, np.complex64))
    grid[mask] = values
    return fftn(grid, norm="ortho")


def shrink(z, threshold):
    return z * np.maximum(0.0, 1 - threshold / np.maximum(np.abs(z), 1e-30))


def recover(values, mask, lam=0.06, max_iter=250, tol=1e-4):
    if mask.dtype != bool or mask.sum() != values.size or not mask.any():
        raise ValueError("provide one observation per distinct sampled grid cell")
    if lam <= 0 or max_iter < 1 or tol <= 0 or not np.isfinite(values).all():
        raise ValueError("invalid solver input")
    theta = np.zeros(mask.shape, np.complex64)
    momentum, y, history = 1.0, theta, []
    # AA* = I for distinct rows of a unitary transform, so L = 1 exactly.
    for k in range(max_iter):
        gradient = adjoint(forward(y, mask) - values, mask)
        new = shrink(y - gradient, lam)
        residual = forward(new, mask) - values
        objective = 0.5 * np.vdot(residual, residual).real + lam * np.abs(new).sum()
        # Proximal gradient mapping at the candidate gives a stationarity test.
        stationarity = np.linalg.norm(new - shrink(new - adjoint(residual, mask), lam))
        relative = float(stationarity / max(1.0, np.linalg.norm(values)))
        history.append(
            dict(
                iter=k + 1,
                objective=float(objective),
                stationarity=relative,
                converged=relative < tol,
            )
        )
        prev, theta = theta, new
        if relative < tol:
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = theta + np.float32((momentum - 1) / next_momentum) * (theta - prev)
        momentum = next_momentum
    return ifftn(theta, norm="ortho").real, history
