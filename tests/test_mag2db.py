"""Tests for magnitude to decibel conversion."""

import numpy as np
from pyFDN.auxiliary.mag2db import mag2db


def test_mag2db_basic():
    """Test basic magnitude to decibel conversion."""
    arr = np.array([1.0])
    result = mag2db(arr)
    assert result == 0.0


def test_mag2db_zero():
    """Test handling of zero values."""
    arr = np.array([0])
    result = mag2db(arr)
    assert np.isfinite(result) 