"""
Orthogonal completion for FDN: given A, find b, c, d so that V = [A,b;c,d] is orthogonal.

See "Allpass Feedback Delay Networks" by Sebastian J. Schlecht.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def complete_orthogonal(
    A: ArrayLike,
    num_io: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve orthogonal completion: find b, c, d such that V = [A,b;c,d] is orthogonal,
    with d of size num_io x num_io. Singular values of A are 1 except for num_io which are < 1.

    Parameters
    ----------
    A : (N, N) array-like
        Feedback matrix.
    num_io : int
        Number of input/output channels.

    Returns
    -------
    b : (N, num_io) ndarray
        Input gains.
    c : (num_io, N) ndarray
        Output gains.
    d : (num_io, num_io) ndarray
        Direct gains.
    V : (N+num_io, N+num_io) ndarray
        System matrix V = [A,b;c,d].
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError("A must be square")
    if num_io < 1 or N + num_io > N + N:
        raise ValueError("num_io must be positive")

    # Low-rank approximation: I - A A' has rank num_io (approx)
    M1 = np.eye(N) - A @ A.T
    U1, S1, _ = np.linalg.svd(M1)
    U = U1[:, :num_io]
    S = S1[:num_io]
    S = np.maximum(S, 0.0)
    b = U * np.sqrt(S)

    M2 = np.eye(N) - A.T @ A
    _, S2, Vt2 = np.linalg.svd(M2)
    V2 = Vt2.T
    V = V2[:, :num_io]
    S2 = S2[:num_io]
    S2 = np.maximum(S2, 0.0)
    c = (V * np.sqrt(S2)).T

    # d = - b \ A c'  =>  b d = -A c'  =>  d = -pinv(b) @ A @ c.T
    d = -np.linalg.lstsq(b, A @ c.T, rcond=None)[0]

    V_block = np.block([[A, b], [c, d]])

    return b, c, d, V_block
