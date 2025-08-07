"""One-pole absorption filter design utilities."""

import numpy as np


def db2mag(db):
    """Convert decibels to magnitude."""
    return 10 ** (db / 20)


def rt60_to_slope(rt60, fs):
    """Convert RT60 to decay slope per sample."""
    # -60 dB decay over RT60 seconds, per sample
    return -60.0 / (rt60 * fs)


def clip(x, minmax=(-np.inf, -np.finfo(float).eps)):
    """Clip values to specified range."""
    return np.minimum(np.maximum(x, minmax[0]), minmax[1])


def slope_to_rt60(slope, fs):
    """Convert energy decay slope to RT60 in seconds."""
    # MATLAB: RT60 = (-60./ clip(slope, [-Inf, -eps]) )./fs;
    return (-60.0 / slope) / fs


def design_one_pole_filter(h_dc, h_nyq):
    """Design one-pole filter with specified DC and Nyquist gains."""
    r = h_dc / h_nyq
    a1 = (1 - r) / (1 + r)
    b = np.array([1 + a1]) / 2
    a = np.array([1, a1])
    return b, a


def one_pole_absorption(rt60_dc, rt60_nyq, delays, fs):
    """Design one-pole absorption filter based on RT60 values."""
    h_dc = db2mag(delays * rt60_to_slope(rt60_dc, fs))
    h_nyq = db2mag(delays * rt60_to_slope(rt60_nyq, fs))
    b, a = design_one_pole_filter(h_dc, h_nyq)
    return b, a
