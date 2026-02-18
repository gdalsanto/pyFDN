"""Acoustics and RT60 related functions."""
from __future__ import annotations
import warnings
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from numpy.linalg import svd
from scipy.signal import firwin2, freqz, group_delay
from scipy.interpolate import interp1d
from scipy.special import erfc

from pyFDN.auxiliary.utils import db2mag, mag2db, hertz2unit


def rt60_to_slope(rt60: ArrayLike, fs: float) -> np.ndarray:
    """Convert a 60 dB decay time to an energy decay slope (dB per sample)."""

    rt_arr = np.asarray(rt60, dtype=float)
    return -60.0 / (rt_arr * fs)


def slope_to_rt60(slope: ArrayLike, fs: float) -> np.ndarray:
    """Convert slope (dB/sample) to T60 in seconds."""
    return -60.0 / (slope * fs)


def absorption_filters(frequency, targetRT60, filterOrder, delays, fs):
    """
    Generate FIR absorption filters for each channel.
    frequency: [freq_points]
    targetRT60: shape (freq_points, channels)
    delays: array of length channels
    """
    num_channels = len(delays)
    unit_freq = hertz2unit(frequency, fs)
    FIR = np.zeros((num_channels, filterOrder + 1))

    if filterOrder == 0:
        rt60 = targetRT60[0, :]
        db = delays * rt60_to_slope(rt60, fs)
        FIR[:, 0] = db2mag(db)
    else:
        for ch in range(num_channels):
            rt60 = targetRT60[:, ch]
            delay = delays[ch] + int(np.ceil(filterOrder / 2))
            db = delay * rt60_to_slope(rt60, fs)
            target_amp = db2mag(db)
            # firwin2 expects normalized [0..1] freqs and gain values
            FIR[ch, :] = firwin2(filterOrder + 1, unit_freq, target_amp)
    return FIR


def absorption_to_t60(filterCoeffs, delays, nfft, fs):
    """Compute T60 from recursive absorption filter with delay."""
    filterLen = filterCoeffs.shape[1]
    response = np.fft.fft(filterCoeffs, nfft, axis=1)
    freq = np.linspace(0, fs/2, nfft // 2, endpoint=False)

    response = response[:, :nfft // 2]
    freq = freq[:nfft // 2]

    totalDelay = delays[:, None] + filterLen / 2
    decayPerSample = mag2db(np.abs(response)) / totalDelay
    T60 = slope_to_rt60(decayPerSample, fs)
    return T60.T, freq  # shape: (freq_points, channels)


def echo_density(
    ir: ArrayLike,
    n: int = 1024,
    fs: float = 48000.0,
    pre_delay: int = 0,
    mixing_thresh: float = 1.0,
    hop: int = 500,
) -> Tuple[float, np.ndarray]:
    """Echo density and mixing time (Abel & Huang 2006).

    Computes the transition time between early reflections and stochastic
    reverberation assuming sound pressure in a reverberant field is
    Gaussian distributed.

    Reference: Abel & Huang (2006), "A simple, robust measure of
    reverberation echo density", Proc. 121st AES Convention, San Francisco.

    Parameters
    ----------
    ir : array-like
        Impulse response (1 channel only). Converted to 1D.
    n : int, optional
        Window length (must be even). Default 1024.
    fs : float, optional
        Sampling rate in Hz. Default 48000.
    pre_delay : int, optional
        Onset delay in samples for mixing time. Default 0.
    mixing_thresh : float, optional
        Normalized echo density threshold for mixing time (Abel & Huang use 1).
        Default 1.0.
    hop : int, optional
        Hop size in samples for sparse analysis. Default 500.

    Returns
    -------
    t_abel : float
        Mixing time in milliseconds (time at which echo density first
        exceeds mixing_thresh, relative to pre_delay). 0 if not found.
    echo_dens : np.ndarray
        Echo density vector (length = len(ir)), normalized; interpolated
        from sparse analysis.
    """
    ir_arr = np.asarray(ir, dtype=float).ravel()
    len_ir = len(ir_arr)
    if n % 2 != 0:
        raise ValueError("Window length n must be even.")
    if len_ir < n:
        raise ValueError(
            f"IR length {len_ir} is shorter than analysis window {n}. "
            "Provide at least an IR of some 100 ms."
        )
    half_win = n // 2
    w_tau = np.hanning(n)
    w_tau = w_tau / np.sum(w_tau)

    sparse_ind = np.arange(0, len_ir, hop, dtype=int)
    if sparse_ind[-1] != len_ir - 1 and len_ir - 1 not in sparse_ind:
        sparse_ind = np.append(sparse_ind, len_ir - 1)
    echo_dens_sparse = np.zeros(len(sparse_ind))

    for ii, n_center in enumerate(sparse_ind):
        if n_center <= half_win:
            h_tau = ir_arr[0 : n_center + half_win]
            w_t = w_tau[-(n_center + half_win) :]
        elif n_center <= len_ir - half_win - 1:
            h_tau = ir_arr[n_center - half_win : n_center + half_win]
            w_t = w_tau.copy()
        else:
            h_tau = ir_arr[n_center - half_win : len_ir]
            w_t = w_tau[: len(h_tau)].copy()

        s = np.sqrt(np.sum(w_t * (h_tau ** 2)))
        tip_ct = (np.abs(h_tau) > s).astype(float)
        echo_dens_sparse[ii] = np.sum(w_t * tip_ct)

    echo_dens_sparse = echo_dens_sparse / erfc(1.0 / np.sqrt(2))
    echo_dens = np.interp(
        np.arange(len_ir, dtype=float),
        sparse_ind.astype(float),
        echo_dens_sparse,
    )

    d = np.where(echo_dens > mixing_thresh)[0]
    if d.size > 0:
        first_idx = int(d[0])
        t_abel = (first_idx - pre_delay) / fs * 1000.0
        if t_abel < 0:
            t_abel = 0.0
    else:
        t_abel = 0.0
        warnings.warn("Mixing time not found within given limits.", UserWarning)

    return float(t_abel), echo_dens



def one_pole_absorption(rt_dc: float, rt_ny: float, delays: ArrayLike, fs: float) -> np.ndarray:
    """Design one-pole absorption filters according to specified T60.

    Returns SOS format: shape (6, N) with [b0, b1, b2, a0, a1, a2] per channel.
    """
    delays_arr = np.asarray(delays, dtype=float)
    
    # Calculate target gains
    slope_dc = rt60_to_slope(rt_dc, fs)
    slope_ny = rt60_to_slope(rt_ny, fs)
    
    # Convert to linear magnitude
    h_dc = db2mag(delays_arr * slope_dc)
    h_ny = db2mag(delays_arr * slope_ny)
    
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



