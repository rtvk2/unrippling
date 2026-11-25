"""Stage 3 of the proposal: robust PCA of the registered stack (Oreifej et al. baseline).

Registered frames are approximately one image (low rank) plus sparse errors (glints, local misalignment).
Solve  min ||L||_* + lambda ||S||_1  s.t.  D = L + S  by the inexact augmented Lagrange multiplier method
(Lin, Chen and Ma 2010), with one frame per column of D, and return the temporal mean of L.
"""

import numpy as np


def _gram(M, chunk=65536):
    """M^T M in float64, accumulated over row chunks so a float32 M is never copied whole."""
    G = np.zeros((M.shape[1], M.shape[1]))
    for i in range(0, len(M), chunk):
        c = M[i:i + chunk].astype(np.float64)
        G += c.T @ c
    return G


def svt(M, tau, out=None):
    """Singular value thresholding of a tall (n, T) matrix: U max(s - tau, 0) V^T. Returns (result, rank);
    the result is written into `out` when given.

    Uses the (T, T) Gram matrix: with M = U s V^T, the result is M V diag(max(s - tau, 0) / s) V^T, which
    costs two (n, T) x (T, T) products instead of a full SVD. Only singular values above tau survive, so the
    loss of precision of s^2 near zero does not matter.
    """
    w, V = np.linalg.eigh(_gram(M))
    s = np.sqrt(np.clip(w, 0, None))
    keep = s > tau
    out = np.empty_like(M) if out is None else out
    if not keep.any():
        out[:] = 0
        return out, 0
    Vk = V[:, keep]
    W = (Vk * ((s[keep] - tau) / s[keep])) @ Vk.T
    return np.matmul(M, W.astype(M.dtype), out=out), int(keep.sum())


def shrink(X, tau, out=None, buf=None):
    """Soft thresholding, sign(X) max(|X| - tau, 0); into `out` when given, `buf` is scratch space like X."""
    out = np.empty_like(X) if out is None else out
    np.abs(X, out=out)
    out -= tau
    np.maximum(out, 0, out=out)
    out *= np.sign(X, out=buf)
    return out


def rpca(D, lam=None, tol=1e-7, max_iter=500, rho=1.5, verbose=False):
    """Inexact ALM robust PCA of D (n, T). Returns (L, S, history).

    lam defaults to 1 / sqrt(max(n, T)). Stops when ||D - L - S||_F / ||D||_F < tol. history has one dict per
    iteration with the rank of L, the fraction of nonzeros in S and the relative residual.
    """
    D = np.asarray(D)
    n, T = D.shape
    lam = 1 / np.sqrt(max(n, T)) if lam is None else lam
    norm_fro = np.linalg.norm(D)
    norm_two = np.sqrt(np.linalg.eigvalsh(_gram(D))[-1])  # largest singular value
    Y = D / max(norm_two, np.abs(D).max() / lam)  # dual initialisation from the paper
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    L, S = np.zeros_like(D), np.zeros_like(D)
    # Two preallocated buffers instead of ~6 temporaries per iteration; the steps are the plain expressions in
    # the comments. Writing into `out=` also keeps everything in D's dtype: written plainly, the np.float64 mu
    # silently promotes float32 data to float64 (NumPy 2 rules), which doubles memory and time for a change of
    # at most 3e-7 in the output image.
    buf, buf2 = np.empty_like(D), np.empty_like(D)
    history = []
    for k in range(max_iter):
        # L = svt(D - S + Y / mu, 1 / mu)
        np.subtract(D, S, out=buf)
        buf += np.divide(Y, mu, out=buf2)
        L, rank = svt(buf, 1 / mu, out=L)
        # S = shrink(D - L + Y / mu, lam / mu)
        np.subtract(D, L, out=buf)
        buf += np.divide(Y, mu, out=buf2)
        S = shrink(buf, lam / mu, out=S, buf=buf2)
        # Z = D - L - S;  Y += mu * Z
        np.subtract(D, L, out=buf)
        buf -= S
        residual = float(np.linalg.norm(buf) / norm_fro)
        Y += np.multiply(buf, mu, out=buf)
        mu = min(mu * rho, mu_bar)
        row = {"iter": k + 1, "rank": rank, "sparsity": float(np.count_nonzero(S) / S.size), "residual": residual}
        history.append(row)
        if verbose:
            print("  " + "  ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))
        if row["residual"] < tol:
            break
    return L, S, history


def rpca_torch(D, lam=None, tol=1e-7, max_iter=500, rho=1.5, device="cuda", verbose=False):
    """Same inexact ALM as rpca(), step for step, in torch on the GPU (float32 data, Gram products in float32 per
    4096-row chunk summed in float64, float64 eigh, TF32 off). D: numpy (n, T) float32. Returns (L, S, history) as
    numpy, like rpca(). Needs ~7 x D.nbytes of GPU memory (4.4 GB for a 199-frame 512x512 RGB MFIR clip).
    Checked against rpca() on 2 MFIR test clips (2026-11-23, RTX 4050): same iterations and PSNR (4 decimals),
    output max |diff| 1.4e-6 (1/2800 of an 8-bit level), rank equal or +1; 17.5 s vs 86 s on the CPU.
    (A float64 Gram matched to 2.4e-7 but took ~35 s: consumer GPUs are slow in float64.)"""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 would round the products to 10 bits; keep exact float32

    def gram(M, chunk=4096):  # float32 products per chunk (fast on consumer GPUs), summed in float64
        G = torch.zeros(M.shape[1], M.shape[1], dtype=torch.float64, device=M.device)
        for i in range(0, M.shape[0], chunk):
            c = M[i:i + chunk]
            G += (c.T @ c).double()
        return G

    Dt = torch.from_numpy(np.ascontiguousarray(D, dtype=np.float32)).to(device)
    n, T = Dt.shape
    lam = 1 / np.sqrt(max(n, T)) if lam is None else lam
    norm_fro = torch.linalg.norm(Dt).item()
    norm_two = float(np.sqrt(torch.linalg.eigvalsh(gram(Dt))[-1].item()))
    Y = Dt / max(norm_two, Dt.abs().max().item() / lam)
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    L, S = torch.zeros_like(Dt), torch.zeros_like(Dt)
    buf = torch.empty_like(Dt)
    history = []
    for k in range(max_iter):
        torch.sub(Dt, S, out=buf).add_(Y, alpha=1 / mu)  # D - S + Y / mu
        w, V = torch.linalg.eigh(gram(buf))  # svt via the Gram matrix, as svt()
        s = w.clamp_min(0).sqrt()
        keep = s > 1 / mu
        rank = int(keep.sum())
        if rank:
            Vk = V[:, keep]
            torch.matmul(buf, ((Vk * ((s[keep] - 1 / mu) / s[keep])) @ Vk.T).float(), out=L)
        else:
            L.zero_()
        torch.sub(Dt, L, out=buf).add_(Y, alpha=1 / mu)  # D - L + Y / mu
        torch.mul(buf.sign(), (buf.abs() - lam / mu).clamp_min_(0), out=S)  # shrink
        torch.sub(Dt, L, out=buf).sub_(S)  # Z
        residual = (torch.linalg.norm(buf) / norm_fro).item()
        Y.add_(buf, alpha=mu)
        mu = min(mu * rho, mu_bar)
        row = {"iter": k + 1, "rank": rank, "sparsity": (torch.count_nonzero(S) / S.numel()).item(),
               "residual": residual}
        history.append(row)
        if verbose:
            print("  " + "  ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))
        if residual < tol:
            break
    out = L.cpu().numpy(), S.cpu().numpy(), history
    del Dt, L, S, Y, buf
    torch.cuda.empty_cache()
    return out


def rpca_image(registered, lam=None, tol=1e-4, max_iter=200, verbose=False, device=None):
    """Stage 3 output for a registered stack (T, H, W[, 3]) in [0, 1]: the temporal mean of the low-rank part.

    Every frame (all channels) is one column of D, so the colour channels share one low-rank model.
    Returns (image (H, W[, 3]) float32, L as a (T, H, W[, 3]) stack, S as a stack, history).
    tol 1e-4 instead of the paper's 1e-7: on James BlueTiles (30 frames) it stops after 20 instead of 30 iterations
    and moves the output by at most 3e-4 (under a tenth of an 8-bit grey level, PSNR +0.0001 dB) vs tol 1e-6.
    """
    R = np.asarray(registered, dtype=np.float32)
    D = np.ascontiguousarray(R.reshape(len(R), -1).T)  # a transposed view is Fortran-ordered and ~20x slower
    if device is None:  # numpy (the version every reported RPCA number so far was made with)
        L, S, history = rpca(D, lam=lam, tol=tol, max_iter=max_iter, verbose=verbose)
    else:
        L, S, history = rpca_torch(D, lam=lam, tol=tol, max_iter=max_iter, device=device, verbose=verbose)
    L, S = L.T.reshape(R.shape), S.T.reshape(R.shape)
    return L.mean(axis=0), L, S, history
