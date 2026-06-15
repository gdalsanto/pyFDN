"""Torch-free scoring metrics for trained FDNs.

These are the NumPy counterparts of the training losses, used for reporting and
test assertions (they import without torch). Decay/RT and echo-density metrics
are not re-implemented here -- use :func:`pyFDN.estimate_rt_bands`,
:func:`pyFDN.edc`, and :func:`pyFDN.echo_density` from ``pyFDN.auxiliary.acoustics``.
"""

from __future__ import annotations

import numpy as np

from pyFDN.auxiliary.acoustics import edc


def magnitude_response(ir: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """One-sided magnitude spectrum ``|rfft(ir)|`` of a 1-D impulse response."""
    sig = np.asarray(ir, dtype=float).ravel()
    n = len(sig) if nfft is None else int(nfft)
    return np.abs(np.fft.rfft(sig, n))


def flatness_from_magnitude(magnitude: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy) of a magnitude spectrum, in ``[0, 1]``.

    Geometric mean over arithmetic mean of the power (DC excluded). ``1.0`` is
    perfectly flat. Use this when you already have a magnitude response (e.g. a
    trained FDN's ``|H|`` sampled at DFT bins), which is the right colorless
    measure for a lossless FDN -- its time-domain IR does not decay, so a
    finite-length render is unusable.
    """
    power = np.asarray(magnitude, dtype=float).ravel()[1:] ** 2  # drop DC
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def spectral_flatness(ir: np.ndarray, nfft: int | None = None) -> float:
    """Spectral flatness (Wiener entropy) of an impulse response, in ``[0, 1]``.

    ``1.0`` is perfectly flat (e.g. an impulse); a pure tone tends to ``0``. The
    primary colorless-quality metric for a decaying IR; for a *lossless* FDN use
    :func:`flatness_from_magnitude` on its sampled ``|H|`` instead.
    """
    return flatness_from_magnitude(magnitude_response(ir, nfft))


def octave_colouration(
    ir: np.ndarray, fs: float, fc: float = 1000.0, n: int = 8
) -> np.ndarray:
    """Per-octave-band level deviation (dB) from the across-band mean.

    Splits the magnitude spectrum into ``n`` octave bands geometrically centred
    around ``fc`` and returns each band's mean level minus the overall mean, in
    dB. A colorless (flat) response gives small deviations; the spread measures
    colouration. Bands with no FFT bins in range are ``nan``.
    """
    sig = np.asarray(ir, dtype=float).ravel()
    power = magnitude_response(sig) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    centers = fc * 2.0 ** (np.arange(n) - n // 2)
    levels = np.full(n, np.nan)
    for i, center in enumerate(centers):
        sel = (freqs >= center / np.sqrt(2)) & (freqs < center * np.sqrt(2))
        if np.any(sel):
            levels[i] = 10.0 * np.log10(np.mean(power[sel]) + 1e-20)
    return levels - np.nanmean(levels)


def edc_l1(
    ir_a: np.ndarray,
    ir_b: np.ndarray,
    *,
    normalize: bool = True,
    db: bool = True,
) -> float:
    """Mean L1 distance between the energy-decay curves of two IRs.

    By default the curves are normalized to start at ``0 dB`` and compared in dB,
    so this measures decay-shape mismatch independent of overall level.
    """
    ea = edc(np.asarray(ir_a, dtype=float).ravel())
    eb = edc(np.asarray(ir_b, dtype=float).ravel())
    length = min(len(ea), len(eb))
    ea, eb = ea[:length], eb[:length]
    if normalize:
        ea = ea / (ea[0] or 1.0)
        eb = eb / (eb[0] or 1.0)
    if db:
        ea = 10.0 * np.log10(ea + 1e-20)
        eb = 10.0 * np.log10(eb + 1e-20)
    return float(np.mean(np.abs(ea - eb)))


def _stft_magnitude(sig: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    """Magnitude STFT of a 1-D signal (Hann window), shape (frames, bins)."""
    if len(sig) < nfft:
        sig = np.pad(sig, (0, nfft - len(sig)))
    window = np.hanning(nfft)
    starts = range(0, len(sig) - nfft + 1, hop)
    return np.stack([np.abs(np.fft.rfft(sig[s : s + nfft] * window)) for s in starts])


def mr_stft_distance(
    ir_a: np.ndarray,
    ir_b: np.ndarray,
    *,
    fs: float = 48000.0,
    nfft: tuple[int, ...] = (256, 512, 1024),
) -> float:
    """Multi-resolution STFT magnitude distance between two IRs.

    A torch-free mirror of flamo's ``mss_loss``: the mean (over resolutions) of
    the mean absolute magnitude-STFT difference. ``0`` for identical signals.
    """
    a = np.asarray(ir_a, dtype=float).ravel()
    b = np.asarray(ir_b, dtype=float).ravel()
    dists = []
    for size in nfft:
        hop = max(1, size // 4)
        ma = _stft_magnitude(a, size, hop)
        mb = _stft_magnitude(b, size, hop)
        length = min(len(ma), len(mb))
        dists.append(float(np.mean(np.abs(ma[:length] - mb[:length]))))
    return float(np.mean(dists))
