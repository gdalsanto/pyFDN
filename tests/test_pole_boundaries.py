"""Tests for pole boundaries calculation."""

import numpy as np
from types import SimpleNamespace
from pyFDN.auxiliary.pole_boundaries import pole_boundaries


def test_pole_boundaries_basic():
    """Test basic pole boundaries calculation."""
    delays = np.array([1, 2])
    absorption = np.ones((2, 2, 4))
    feedback_matrix = np.ones((2, 2, 4))
    fs = 48000
    min_curve, max_curve, f = pole_boundaries(delays, absorption, feedback_matrix, fs, nfft=8)
    assert min_curve.shape == (8,)
    assert max_curve.shape == (8,)
    assert f.shape == (8,)