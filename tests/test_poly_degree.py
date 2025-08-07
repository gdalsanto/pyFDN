"""Tests for polynomial degree calculation."""

import numpy as np
from pyFDN.auxiliary.poly_degree import poly_degree, mag2db


def test_mag2db():
    """Test magnitude to decibel conversion."""
    x = np.array([1, 10, 100])
    result = mag2db(x)
    expected = np.array([0, 20, 40])
    np.testing.assert_allclose(result, expected)


def test_poly_degree():
    """Test polynomial degree calculation."""
    poly = np.array([1, 0, 0])
    deg = poly_degree(poly, 'z^-1')
    assert deg == 0 