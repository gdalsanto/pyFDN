"""Tests for matrix convolution operations."""

import numpy as np
from pyFDN.auxiliary.matrix_convolution import matrix_convolution


def test_matrix_convolution_basic():
    """Test basic matrix convolution functionality."""
    matrix_a = np.ones((2, 2, 2))
    matrix_b = np.ones((2, 2, 2))
    result = matrix_convolution(matrix_a, matrix_b)
    assert result.shape == (2, 2, 3) 