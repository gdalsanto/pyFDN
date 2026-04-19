# dss_to_impz.py
import numpy as np

from pyFDN.process import process_fdn


def dss_to_impz(ir_len, delays, A, B, C, D):
    """
    Compute MIMO impulse response from delay state-space (DSS) representation.

    Runs one simulation per input channel (Dirac at t=0 on that channel only)
    and stacks the results into a single array.

    Parameters
    ----------
    ir_len : int
        Length of impulse response in samples
    delays : list or array
        Delay lengths in samples
    A, B, C, D : ndarray
        Delay state-space matrices (static, numeric only).
        For FDNs with absorption filters use dss_to_flamo.

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    num_inputs = np.asarray(B).shape[1]
    out_list = []

    for j in range(num_inputs):
        input_signal = np.zeros((ir_len, num_inputs))
        input_signal[0, j] = 1.0
        out_j = process_fdn(input_signal, delays, A, B, C, D)
        if out_j.ndim == 1:
            out_j = out_j[:, np.newaxis]
        out_list.append(out_j)

    return np.stack(out_list, axis=-1)
