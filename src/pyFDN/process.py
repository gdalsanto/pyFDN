"""FDN processing functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import sosfilt, sosfilt_zi

from pyFDN.dsp.dfilt_matrix import FIRMatrixFilter
from pyFDN.dsp.feedback_delay import FeedbackDelay


def _normalize_absorption_sos(absorption_filters: ArrayLike, n: int) -> np.ndarray:
    """Bring per-delay-line SOS filters to shape (N, n_sections, 6).

    Accepted inputs:
      - (6, N): one section per delay line (``one_pole_absorption`` convention);
      - (n_sections, 6, N): section cascade per delay line;
      - (N, n_sections, 6): scipy ``sosfilt`` convention per line.
    """
    sos = np.asarray(absorption_filters, dtype=float)
    if sos.ndim == 2 and sos.shape[0] == 6 and sos.shape[1] == n:
        return np.ascontiguousarray(sos.T[:, None, :])
    if sos.ndim == 3 and sos.shape[1] == 6 and sos.shape[2] == n:
        return np.ascontiguousarray(sos.transpose(2, 0, 1))
    if sos.ndim == 3 and sos.shape[0] == n and sos.shape[2] == 6:
        return np.ascontiguousarray(sos)
    raise ValueError(
        "absorption_filters must have shape (6, N), (n_sections, 6, N) "
        f"or (N, n_sections, 6); got {sos.shape} for N={n}"
    )


def process_fdn(
    input_signal: ArrayLike,
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    absorption_filters: ArrayLike | None = None,
    extra_matrix: Any | None = None,
) -> np.ndarray:
    """Simulate the feedback delay network using block processing.

    Recursion per block (same ordering as the MATLAB ``processFDN``):
    delay output -> absorption filters -> output gains C, and in the feedback
    path: absorbed delay output -> feedback matrix A -> extra matrix -> + B input.

    Parameters
    ----------
    input_signal : array
        Input of shape (num_samples,) or (num_samples, num_inputs).
    delays : array
        Delay lengths in samples, shape (N,).
    A : array
        Feedback matrix: static (N, N) or FIR polynomial (N, N, order) in
        z^{-1} convention.
    B, C, D : array
        Static input, output, and direct gains.
    absorption_filters : array, optional
        Per-delay-line SOS filters; see ``_normalize_absorption_sos`` for
        accepted shapes. Applied to the delay outputs inside the loop.
    extra_matrix : object, optional
        Object with a ``filter(block) -> block`` method applied after the
        feedback matrix (e.g. ``TimeVaryingMatrix``).

    Returns
    -------
    output : ndarray
        Shape (num_samples, num_outputs), squeezed.
    """
    x = np.asarray(input_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("Input signal must be a 1-D or 2-D array")

    A_mat = np.asarray(A, dtype=float)
    B_mat = np.asarray(B, dtype=float)
    C_mat = np.asarray(C, dtype=float)
    D_mat = np.asarray(D, dtype=float)

    delays_arr = np.asarray(delays, dtype=int).reshape(-1)
    if np.any(delays_arr <= 0):
        raise ValueError("Delays must be positive integers")
    n = delays_arr.size

    if A_mat.ndim == 3:
        feedback_filter: FIRMatrixFilter | None = FIRMatrixFilter(A_mat)
    elif A_mat.ndim == 2:
        feedback_filter = None
    else:
        raise ValueError("A must be a 2-D (static) or 3-D (FIR) matrix")

    if absorption_filters is not None:
        sos_per_line = _normalize_absorption_sos(absorption_filters, n)
        sos_state = [
            sosfilt_zi(sos_per_line[i]) * 0.0 for i in range(n)
        ]  # zero initial state
    else:
        sos_per_line = None
        sos_state = []

    max_block_size = min(int(2**12), int(np.min(delays_arr)))
    delay_bank = FeedbackDelay(delays_arr, max_block_size)

    num_samples = x.shape[0]
    num_outputs = C_mat.shape[0]
    output = np.zeros((num_samples, num_outputs), dtype=float)

    start = 0
    while start < num_samples:
        block_size = min(max_block_size, num_samples - start)
        block_in = x[start : start + block_size, :]

        delay_out = delay_bank.get_values(block_size)  # (block, N)
        if sos_per_line is not None:
            for i in range(n):
                filtered, sos_state[i] = sosfilt(
                    sos_per_line[i],
                    np.ascontiguousarray(delay_out[:, i]),
                    zi=sos_state[i],
                )
                delay_out[:, i] = filtered

        if feedback_filter is not None:
            feedback = feedback_filter.filter(delay_out)
        else:
            feedback = delay_out @ A_mat.T
        if extra_matrix is not None:
            feedback = extra_matrix.filter(feedback)

        delay_bank.set_values(block_in @ B_mat.T + feedback)

        output[start : start + block_size] = delay_out @ C_mat.T + block_in @ D_mat.T
        delay_bank.advance(block_size)
        start += block_size

    return output.squeeze()
