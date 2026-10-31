"""
Our own truncated SVD by power iteration (Phase 2B).

Goal: the top-k singular triplets of A (n x d):  A ~= U_k diag(S_k) Vt_k

Idea: the right singular vectors of A are the eigenvectors of A^T A, and the largest
one dominates when you multiply by A^T A again and again (power iteration).
We never build A^T A (d x d): we compute A^T (A v), two matrix-vector products.

Two versions:
  power_iteration_svd   textbook: one vector at a time; after a vector converges we
                        remove its direction from the next ones (deflation by
                        orthogonalisation). Easy to explain, slow for large k.
  block_power_svd       all k vectors at once (subspace iteration). Matrix-matrix
                        products (BLAS-3) + a QR per step to keep the block orthonormal,
                        then a small SVD to split the block into singular vectors
                        (Rayleigh-Ritz). Same idea as randomized SVD.
"""

import numpy as np


def power_iteration_svd(A, k, max_iter=300, tol=1e-6, seed=0):
    """Top-k SVD, one singular vector at a time. Returns U, S, Vt, iterations per vector."""
    rng = np.random.default_rng(seed)
    n, d = A.shape
    V = np.zeros((d, k), dtype=A.dtype)
    S = np.zeros(k, dtype=A.dtype)
    U = np.zeros((n, k), dtype=A.dtype)
    iters = []
    for j in range(k):
        v = rng.standard_normal(d).astype(A.dtype)
        v /= np.linalg.norm(v)
        Vj = V[:, :j]                                   # vectors found so far
        for it in range(1, max_iter + 1):
            w = A.T @ (A @ v)                           # (A^T A) v without forming A^T A
            w -= Vj @ (Vj.T @ w)                        # deflation: remove known directions
            w -= Vj @ (Vj.T @ w)                        # twice, for numerical safety (float32)
            w /= np.linalg.norm(w)
            converged = 1.0 - abs(float(w @ v)) < tol   # angle between two iterates ~ 0
            v = w
            if converged:
                break
        iters.append(it)
        Av = A @ v
        S[j] = np.linalg.norm(Av)                       # sigma_j = ||A v_j||
        U[:, j] = Av / S[j]                             # u_j = A v_j / sigma_j
        V[:, j] = v
    return U, S, V.T, iters


def block_power_svd(A, k, n_iter=10, oversample=10, seed=0):
    """Top-k SVD with block power (subspace) iteration. Returns U, S, Vt."""
    rng = np.random.default_rng(seed)
    n, d = A.shape
    l = min(k + oversample, d)                          # a few extra vectors speed up convergence
    V, _ = np.linalg.qr(rng.standard_normal((d, l)).astype(A.dtype))
    for _ in range(n_iter):
        V, _ = np.linalg.qr(A.T @ (A @ V))              # multiply the whole block, re-orthonormalise
    B = A @ V                                           # n x l: A restricted to the found subspace
    Ub, s, Wt = np.linalg.svd(B, full_matrices=False)   # small SVD (l columns) splits the block
    U = Ub[:, :k]
    S = s[:k]
    Vt = (V @ Wt.T)[:, :k].T
    return U, S, Vt


def reconstruction_error(A, U, S, Vt, mean=None):
    """Relative Frobenius error ||A - A_k||_F / ||A||_F (float64)."""
    A64 = A.astype(np.float64)
    A_k = (U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64)
    if mean is not None:
        A_k += mean.astype(np.float64)
    return float(np.linalg.norm(A64 - A_k) / np.linalg.norm(A64))
