"""
Element-wise quadratic equation solver: a*x^2 + b*x + c = 0.
Returns two real roots per element (same shape as a, b, c).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mroots(
    a: ArrayLike,
    b_coef: ArrayLike,
    c: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve element-wise a*x^2 + b*x + c = 0.

    Parameters
    ----------
    a, b_coef, c : array-like, same shape
        Coefficients (can be 2D matrices).

    Returns
    -------
    x1 : ndarray, same shape as a
        First root (or -c/b when a=0).
    x2 : ndarray, same shape as a
        Second root (or 0 when a=0).
    """
    a = np.asarray(a, dtype=float)
    b_coef = np.asarray(b_coef, dtype=float)
    c = np.asarray(c, dtype=float)
    disc = b_coef**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        # a == 0: linear, single root -c/b
        denom = np.where(np.abs(a) > 1e-20, 2 * a, 1.0)
        x1 = np.where(np.abs(a) > 1e-20, (-b_coef + sqrt_disc) / denom, -c / np.where(np.abs(b_coef) > 1e-20, b_coef, 1.0))
        x2 = np.where(np.abs(a) > 1e-20, (-b_coef - sqrt_disc) / denom, 0.0)
    x1 = np.real(x1)
    x2 = np.real(x2)
    return x1, x2
