"""Matrix polynomial evaluation."""

import numpy as np


def matrix_polyval(poly_matrix, z):
    """
    Evaluate matrix polynomial at z.
    
    Args:
        poly_matrix: shape (m, n, degree)
        z: evaluation point
    Returns:
        result: shape (m, n)
    """
    degree = poly_matrix.shape[2]
    exponents = np.arange(degree)
    zz = z ** exponents
    zz = zz.reshape((1, 1, degree))
    result = np.sum(poly_matrix * zz, axis=2)
    return result
