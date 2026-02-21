"""
Homogeneous allpass FDN: construct [A,b;c,d] uniallpass with A = U*G (U unitary, G gain).

See "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def homogeneous_allpass_fdn(
    G: ArrayLike,
    X: ArrayLike,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate allpass FDN with homogeneous decay: V = [A,b;c,d] uniallpass, A = U*G.

    Parameters
    ----------
    G : (N, N) array-like
        Diagonal gain matrix.
    X : (N, N) array-like
        Diagonal design matrix.
    verbose : bool
        If True, print verification.

    Returns
    -------
    A : (N, N) ndarray
        Feedback matrix A = U @ G.
    b : (N, 1) ndarray
        Input gains.
    c : (1, N) ndarray
        Output gains.
    d : (1, 1) ndarray
        Direct gain.
    U : (N, N) ndarray
        Unitary matrix.
    """
    G = np.asarray(G, dtype=float)
    X = np.asarray(X, dtype=float)
    N = G.shape[0]
    R = G @ G @ X
    p = np.diag(X).copy()
    r = np.diag(R).copy()

    # Cauchy-like orthogonal U: K = 1/(p - r')
    K = 1.0 / (p.reshape(-1, 1) - r.reshape(1, -1))
    beta_alpha = np.linalg.inv(K) / K.T
    u_svd, s_svd, v_svd = np.linalg.svd(beta_alpha)
    beta = -np.sqrt(s_svd[0]) * v_svd[0, :]
    U = np.sqrt(beta_alpha.T) * K

    A = U @ G
    d = np.array([[((-1) ** N) * np.linalg.det(A)]])
    b = np.sqrt(beta).reshape(-1, 1)
    c = (-np.linalg.inv(X) @ np.linalg.inv(A) @ b * float(d)).T

    if verbose:
        from pyFDN.auxiliary.allpass import is_uniallpass, is_allpass
        print("U @ U' =\n", U @ U.T)
        print("X*U - U*R =\n", X @ U - U @ R)
        print("b*b'*U =\n", b @ b.T @ U)
        is_a1, XX = is_uniallpass(A, b, c, d)
        print("isUniallpass:", is_a1, "XX =\n", XX)
        delays = 2 ** np.arange(N)
        is_a2, den, num = is_allpass(A, b, c, d, delays)
        print("isAllpass (delays 2^0..2^(N-1)):", is_a2)

    return A, b, c, d, U
