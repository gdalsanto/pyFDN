"""Matrix polynomial derivative operations."""

import numpy as np


def matrix_polyder(numerator, denominator, var='z^-1'):
    """
    Wrapper function for polynomial derivative of filter matrices.
    
    Args:
        numerator: shape (order, num_rows, num_cols)
        denominator: shape (order, num_rows, num_cols)
        var: variable type ('z^-1' or 'z^1')
    Returns:
        num_der, den_der: shape (order, num_rows, num_cols)
    """
    order, num_rows, num_cols = numerator.shape
    num_der = np.zeros((order, num_rows, num_cols), dtype=numerator.dtype)
    den_der = np.zeros((order, num_rows, num_cols), dtype=numerator.dtype)
    for it1 in range(num_rows):
        for it2 in range(num_cols):
            if var == 'z^1':
                # Derivative of numerator
                for it3 in range(1, order):
                    num_der[it3-1, it1, it2] = it3 * numerator[it3, it1, it2]
                # Derivative of denominator
                for it3 in range(1, order):
                    den_der[it3-1, it1, it2] = it3 * denominator[it3, it1, it2]
            elif var == 'z^-1':
                # Derivative of numerator
                for it3 in range(order-1):
                    num_der[it3+1, it1, it2] = -(it3+1) * numerator[it3, it1, it2]
                # Derivative of denominator
                for it3 in range(order-1):
                    den_der[it3+1, it1, it2] = -(it3+1) * denominator[it3, it1, it2]
            else:
                raise ValueError("Unknown variable type")

    return num_der, den_der
