"""Tests for matrix polynomial evaluation."""

import numpy as np
from pyFDN.auxiliary.matrix_polyval import matrix_polyval


def test_matrix_polyval_basic():
    """Test basic matrix polynomial evaluation."""
    poly = np.ones((2, 2, 3))
    z = 2
    result = matrix_polyval(poly, z)
    assert result.shape == (2, 2) 