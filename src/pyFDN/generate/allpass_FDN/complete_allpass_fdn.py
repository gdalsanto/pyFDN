"""
Complete a feedback matrix A to a uniallpass FDN [A,b;c,d] with diagonal Lyapunov X.

Solves [A,b;c,d] [X 0; 0 1] [A,b;c,d]' = [X 0; 0 1]; V is the balanced (orthogonal) system matrix.

See "Allpass Feedback Delay Networks", Sebastian J. Schlecht (IEEE Trans. Signal Processing).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov

from .mroots import mroots
from .min_rank_choice_brute_force import min_rank_choice_brute_force


def complete_allpass_fdn(
    A: ArrayLike,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the general completion problem for SISO uniallpass FDN.

    Parameters
    ----------
    A : (N, N) array-like
        Feedback matrix.
    tol : float
        Error tolerance for inverse and low-rank choice.

    Returns
    -------
    b : (N, 1) ndarray
        Input gain.
    c : (1, N) ndarray
        Output gain.
    d : (1, 1) ndarray
        Direct gain.
    X : (N, N) ndarray
        Diagonal similarity (Lyapunov) matrix.
    V : (N+1, N+1) ndarray
        Balanced FDN system matrix (orthogonal).
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError("A must be square")

    iA = np.linalg.inv(A)
    iA[np.abs(iA) < tol] = 0

    d = np.array([[((-1) ** N) * np.linalg.det(A)]])
    dA = np.diag(A) - np.diag(iA)
    dA = dA.reshape(-1, 1)
    # F = d * (A.*A' - iA.*iA' - dA*dA')
    F = float(d) * (A * A.T - iA * iA.T - dA @ dA.T)

    # Quadratic: iA * Y^2 - F * Y + (iA' * d^2 * (dA*dA')) = 0
    c_term = iA.T * (float(d) ** 2) * (dA @ dA.T)
    x1, x2 = mroots(iA, -F, c_term)
    x1 = np.real(x1)
    x2 = np.real(x2)

    bX, cX, _choice, is_valid = min_rank_choice_brute_force(x1, x2, tol=tol)
    if not is_valid:
        raise RuntimeError("min_rank_choice_brute_force did not find a rank-1 completion")

    # X = diag( -(A*cX') ./ bX / d ), replace bad with 1
    ac = A @ cX.T
    denom = bX.ravel() * float(d)
    denom = np.where(np.abs(denom) > 1e-20, denom, 1.0)
    diag_X = -(ac.ravel() / denom)
    diag_X = np.where(np.isfinite(diag_X) & (np.abs(diag_X) <= 100000), diag_X, 1.0)
    X = np.diag(diag_X)

    b = X @ bX
    c = cX @ np.linalg.inv(X)

    # Recover full diagonal from Lyapunov
    P = solve_discrete_lyapunov(A, b @ b.T)
    X = np.diag(np.diag(P))

    PVP = np.block([[A, b], [c, d]])
    sqrt_X = np.diag(np.sqrt(np.maximum(np.diag(X), 1e-20)))
    P1 = np.block([[sqrt_X, np.zeros((N, 1))], [np.zeros((1, N)), np.array([[1.0]])]])
    V = np.linalg.solve(P1, PVP @ P1)

    return b, c, d, X, V
