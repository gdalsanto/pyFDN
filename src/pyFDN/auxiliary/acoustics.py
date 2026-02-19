"""Acoustics and RT60 related functions."""
from __future__ import annotations
import warnings
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from numpy.linalg import svd
from scipy.signal import firwin2, freqz, group_delay
from scipy.interpolate import interp1d

from pyFDN.auxiliary.utils import db_to_lin, lin_to_db, hertz_to_unit


def rt_to_slope(rt: ArrayLike, fs: float) -> np.ndarray:
    """Convert reverb time (RT, seconds) to energy decay slope (dB per sample)."""

    rt_arr = np.asarray(rt, dtype=float)
    return -60.0 / (rt_arr * fs)


def slope_to_rt(slope: ArrayLike, fs: float) -> np.ndarray:
    """Convert slope (dB/sample) to reverb time in seconds."""
    return -60.0 / (slope * fs)


def rt_to_gain_per_sample(rt: float, fs: float) -> float:
    """Convert reverb time (seconds) to gain coefficient per sample.

    The gain g satisfies g^(rt*fs) = 10^(-3), i.e. about -30 dB after rt seconds.
    """
    return 10 ** (-3 / (rt * fs))


def edc(ir: ArrayLike, axis: int = 0) -> np.ndarray:
    """Energy decay curve: backward cumulative sum of squared signal along an axis.

    EDC(t) = sum(ir[t:]^2), so the curve decreases from total energy to zero.
    Typically used with impulse responses with shape (n_samples, n_channels).

    Parameters
    ----------
    ir : array-like
        Signal(s). If 1D, EDC of that signal. If 2D (e.g. samples x channels),
        EDC is computed along the time axis for each channel.
    axis : int, optional
        Axis along which time runs (default 0). EDC is computed along this axis.

    Returns
    -------
    np.ndarray
        Same shape as ir. Values are non-negative and non-increasing along axis.
    """
    ir = np.asarray(ir, dtype=float)
    rev = np.flip(ir, axis=axis)
    cum = np.cumsum(rev**2, axis=axis)
    return np.flip(cum, axis=axis)


def absorption_filters(frequency, target_rt, filterOrder, delays, fs):
    """
    Generate FIR absorption filters for each channel.
    frequency: [freq_points]
    target_rt: shape (freq_points, channels)
    delays: array of length channels
    """
    num_channels = len(delays)
    unit_freq = hertz_to_unit(frequency, fs)
    FIR = np.zeros((num_channels, filterOrder + 1))

    if filterOrder == 0:
        rt = target_rt[0, :]
        db = delays * rt_to_slope(rt, fs)
        FIR[:, 0] = db_to_lin(db)
    else:
        for ch in range(num_channels):
            rt = target_rt[:, ch]
            delay = delays[ch] + int(np.ceil(filterOrder / 2))
            db = delay * rt_to_slope(rt, fs)
            target_amp = db_to_lin(db)
            # firwin2 expects normalized [0..1] freqs and gain values
            FIR[ch, :] = firwin2(filterOrder + 1, unit_freq, target_amp)
    return FIR


def absorption_to_rt(filterCoeffs, delays, nfft, fs):
    """Compute reverb time from recursive absorption filter with delay."""
    filterLen = filterCoeffs.shape[1]
    response = np.fft.fft(filterCoeffs, nfft, axis=1)
    freq = np.linspace(0, fs/2, nfft // 2, endpoint=False)

    response = response[:, :nfft // 2]
    freq = freq[:nfft // 2]

    totalDelay = delays[:, None] + filterLen / 2
    decayPerSample = lin_to_db(np.abs(response)) / totalDelay
    rt = slope_to_rt(decayPerSample, fs)
    return rt.T, freq  # shape: (freq_points, channels)





def one_pole_absorption(rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float) -> np.ndarray:
    """Design one-pole absorption filters according to specified reverb time.

    Returns SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel.
    """
    delays_arr = np.asarray(delays, dtype=float)
    
    # Calculate target gains
    slope_dc = rt_to_slope(rt_dc, fs)
    slope_ny = rt_to_slope(rt_ny, fs)
    
    # Convert to linear magnitude
    h_dc = db_to_lin(delays_arr * slope_dc)
    h_ny = db_to_lin(delays_arr * slope_ny)
    
    # Design filters
    r = h_dc / h_ny
    a1 = (1.0 - r) / (1.0 + r)
    b0 = (1.0 - a1) * h_ny

    num_filters = h_dc.size
    sos = np.zeros((6, num_filters))
    sos[0, :] = b0  # b0
    sos[3, :] = 1.0 # a0
    sos[4, :] = a1  # a1
    
    return sos



