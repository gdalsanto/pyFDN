"""Pole boundary calculation utilities."""

import numpy as np
from numpy.linalg import svd
from scipy.signal import freqz, group_delay


def pole_boundaries(delays, absorption, feedback_matrix, fs, nfft=2**12):
    """
    Calculate pole boundaries for a feedback delay network.
    
    Args:
        delays: array of delay lengths
        absorption: absorption filters
        feedback_matrix: feedback matrix
        fs: sampling frequency
        nfft: FFT size
    Returns:
        min_curve: minimum pole boundary
        max_curve: maximum pole boundary
        f: frequency points (Hz, shape: nfft)
    """
    num_delays = len(delays)
    # Compute frequency points
    w = np.linspace(0, np.pi, nfft)
    # FFT along the third axis
    feedback_fft = np.fft.fft(feedback_matrix, n=nfft*2, axis=2)
    feedback_fft = feedback_fft[:, :, :nfft]

    min_vals = np.zeros(nfft)
    max_vals = np.zeros(nfft)
    for it in range(nfft):
        s = svd(feedback_fft[:, :, it], compute_uv=False)
        min_vals[it] = np.min(np.abs(s))
        max_vals[it] = np.max(np.abs(s))

    # Absorption filter frequency response
    b = absorption.b
    a = absorption.a
    if len(b.shape) == 2:
        b = b.reshape(b.shape + (1,))
    if len(a.shape) == 2:
        a = a.reshape(a.shape + (1,))

    h = np.zeros((nfft, num_delays), dtype=complex)
    g = np.zeros((nfft, num_delays))
    for it in range(num_delays):
        # freqz expects (b, a) as 1D arrays
        h[:, it], _ = freqz(b[it, 0, :], a[it, 0, :], w)
        g[:, it], _ = group_delay((b[it, 0, :], a[it, 0, :]), w)

    # d: shape (nfft, num_delays)
    d = np.abs(h) ** (1.0 / (delays + g))
    d_min = np.min(d, axis=1)
    d_max = np.max(d, axis=1)

    min_curve = d_min * min_vals
    max_curve = d_max * max_vals
    f = w / np.pi * fs / 2
    return min_curve, max_curve, f
