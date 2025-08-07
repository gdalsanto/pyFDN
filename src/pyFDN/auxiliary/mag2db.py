"""Magnitude to decibel conversion utilities."""

import numpy as np


def mag2db(x):
    """Convert magnitude to decibels."""
    return 20 * np.log10(np.maximum(x, 1e-20))  # avoid log(0)