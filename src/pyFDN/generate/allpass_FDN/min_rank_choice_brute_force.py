"""
Brute-force choice of element-wise x1 or x2 so that the resulting matrix Y has rank 1.
Used for allpass FDN completion (Y = bX * cX).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def min_rank_choice_brute_force(
    x1: ArrayLike,
    x2: ArrayLike,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Find b (N,1) and c (1,N) such that Y = b @ c has Y_ij in {x1_ij, x2_ij} and Y has rank 1.

    Tries all 2^(N*N) choices; feasible only for small N (e.g. N <= 4).

    Parameters
    ----------
    x1, x2 : (N, N) array-like
        Two candidate values per element.
    tol : float
        Tolerance for rank and reconstruction checks.

    Returns
    -------
    bX : (N, 1) ndarray
        Column vector such that Y = bX @ cX.
    cX : (1, N) ndarray
        Row vector such that Y = bX @ cX.
    choice : (N, N) ndarray, bool
        True where x1 was chosen, False where x2 was chosen.
    is_valid : bool
        True if a rank-1 completion was found.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    N = x1.shape[0]
    if x1.shape != (N, N) or x2.shape != (N, N):
        raise ValueError("x1 and x2 must be NxN with same N")

    n_choices = 1 << (N * N)  # 2^(N*N)
    for k in range(n_choices):
        choice = np.zeros((N, N), dtype=bool)
        for idx in range(N * N):
            if (k >> idx) & 1:
                choice.flat[idx] = True
        Y = np.where(choice, x1, x2)
        r = np.linalg.matrix_rank(Y, tol=tol)
        if r != 1:
            continue
        # Decompose Y = u v^T via SVD
        U, S, Vh = np.linalg.svd(Y)
        u = U[:, 0] * np.sqrt(S[0])
        v = Vh[0, :] * np.sqrt(S[0])
        bX = u.reshape(-1, 1)
        cX = v.reshape(1, -1)
        # Optional: check reconstruction
        Y_recon = bX @ cX
        if np.allclose(Y, Y_recon, atol=tol):
            return bX, cX, choice, True
        # If SVD gave a valid rank-1 matrix, accept even if round-off differs slightly
        if np.max(np.abs(Y - Y_recon)) < tol * (1 + np.max(np.abs(Y))):
            return bX, cX, choice, True

    return np.zeros((N, 1)), np.zeros((1, N)), np.zeros((N, N), dtype=bool), False
