"""FDN processing functions."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from pyFDN.dsp.feedback_delay import FeedbackDelay


def process_fdn(
    input_signal: ArrayLike,
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
) -> np.ndarray:
    """Simulate the feedback delay network using block processing.

    All matrices must be static (numeric). For FDNs with absorption filters
    use the FLAMO path (dss_to_flamo / vanilla_FDN).

    Input is (num_samples, num_inputs); output is (num_samples, num_outputs).
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

    max_block_size = min(int(2 ** 12), int(np.min(delays_arr)))
    delay_bank = FeedbackDelay(delays_arr, max_block_size)

    num_samples = x.shape[0]
    num_outputs = C_mat.shape[0]
    output = np.zeros((num_samples, num_outputs), dtype=float)

    start = 0
    while start < num_samples:
        block_size = min(max_block_size, num_samples - start)
        block_in = x[start:start + block_size, :]

        delay_out = delay_bank.get_values(block_size)           # (block, N)
        delay_bank.set_values(block_in @ B_mat.T + delay_out @ A_mat.T)

        output[start:start + block_size] = delay_out @ C_mat.T + block_in @ D_mat.T
        delay_bank.advance(block_size)
        start += block_size

    return output.squeeze()
