"""Tests for random orthogonal matrix generation."""

import numpy as np
from pyFDN.generate.random_orthogonal import random_orthogonal


def test_random_orthogonal():
    """Test random orthogonal matrix generation."""
    n = 4
    matrix = random_orthogonal(n)
    assert matrix.shape == (n, n)
    # matrix should be orthogonal: matrix.T @ matrix = identity
    identity_matrix = matrix.T @ matrix
    np.testing.assert_allclose(identity_matrix, np.eye(n), atol=1e-7) 