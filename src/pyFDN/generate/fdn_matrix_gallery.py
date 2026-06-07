"""Gallery of feedback matrices for FDNs.

Translation of fdnMatrixGallery.m from fdnToolbox.
"""

from __future__ import annotations

import numpy as np

from .householder_matrix import householder_matrix
from .random_orthogonal import random_orthogonal


def _circulant(v: np.ndarray, direction: int = 1) -> np.ndarray:
    """Build a circulant matrix from first-row vector v.

    ``direction=1``: each row shifts right; ``direction=-1``: shifts left.
    """
    v = np.asarray(v, dtype=float).ravel()
    N = len(v)
    rows = [np.roll(v, i * direction) for i in range(N)]
    return np.array(rows)


# Types that require additional functions not yet translated (allpassInFDN,
# SchroederReverberator) raise NotImplementedError.
_SUPPORTED_TYPES = [
    "orthogonal",
    "Hadamard",
    "circulant",
    "Householder",
    "parallel",
    "series",
    "diagonalConjugated",
    "tinyRotation",
    "nestedAllpass",
]

_ALL_TYPES = _SUPPORTED_TYPES + [
    "allpassInFDN",
    "SchroederReverberator",
    "AndersonMatrix",
]


def fdn_matrix_gallery(
    N: int | None = None,
    matrix_type: str | None = None,
) -> np.ndarray | list[str]:
    """Return a feedback matrix of the requested type, or list all type names.

    Args:
        N: Matrix size.  Ignored when ``matrix_type`` is ``None``.
        matrix_type: One of the supported type strings.  Pass ``None`` (or call
                     with no arguments) to get the list of all type names.

    Returns:
        Feedback matrix of shape ``(N, N)``, or a list of type-name strings.

    Example::

        fdn_matrix_gallery()             # → list of type strings
        fdn_matrix_gallery(4, "orthogonal")
        fdn_matrix_gallery(8, "Hadamard")
    """
    if matrix_type is None:
        return list(_ALL_TYPES)

    if N is None:
        raise ValueError("N must be provided when matrix_type is specified")

    if matrix_type == "orthogonal":
        return random_orthogonal(N)

    if matrix_type == "Hadamard":
        from scipy.linalg import hadamard

        return hadamard(N) / np.sqrt(N)

    if matrix_type == "circulant":
        r_fft = np.fft.fft(np.random.randn(N))
        r_fft /= np.abs(r_fft)
        r = np.real(np.fft.ifft(r_fft))
        direction = np.random.choice([-1, 1])
        return _circulant(r, direction)

    if matrix_type == "Householder":
        return householder_matrix(np.random.rand(N))

    if matrix_type == "parallel":
        return np.eye(N)

    if matrix_type == "series":
        return np.tril(np.random.randn(N, N), -1) / N + np.eye(N)

    if matrix_type == "diagonalConjugated":
        D = np.diag(np.random.randn(N))
        return np.linalg.solve(D, random_orthogonal(N)) @ D

    if matrix_type == "tinyRotation":
        from ..auxiliary.tiny_rotation_matrix import tiny_rotation_matrix

        return tiny_rotation_matrix(N, 0.01).numpy()

    if matrix_type == "nestedAllpass":
        from ..auxiliary.allpass import nested_allpass

        g = np.random.rand(N) * 0.2 + 0.6
        matrix, _, _, _ = nested_allpass(g)
        return matrix

    if matrix_type in ("allpassInFDN", "SchroederReverberator"):
        raise NotImplementedError(
            f"'{matrix_type}' is not yet implemented in pyFDN. "
            "Use the allpass_FDN submodule for allpass structures."
        )

    if matrix_type == "AndersonMatrix":
        from .anderson_matrix import anderson_matrix

        return anderson_matrix(N)

    raise ValueError(f"Unknown matrix_type {matrix_type!r}. Supported: {_ALL_TYPES}")
