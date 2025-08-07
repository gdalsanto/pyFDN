"""Matrix convolution operations."""

import numpy as np


def matrix_convolution(matrix_a, matrix_b):
    """
    Matrix polynomial multiplication (convolution along the last axis).
    
    Args:
        matrix_a: shape (m, n, order_a)
        matrix_b: shape (n, k, order_b)
    Returns:
        result: shape (m, k, order_a + order_b - 1)
    """
    shape_a = matrix_a.shape
    shape_b = matrix_b.shape

    if shape_a[1] != shape_b[0]:
        raise ValueError("Invalid matrix dimension.")

    m, n, order_a = shape_a
    n2, k, order_b = shape_b
    order_c = order_a + order_b - 1

    result = np.zeros((m, k, order_c), dtype=matrix_a.dtype)

    # Permute to (order, m, n) for easier indexing
    a_perm = np.transpose(matrix_a, (2, 0, 1))
    b_perm = np.transpose(matrix_b, (2, 0, 1))

    for row in range(m):
        for col in range(k):
            for it in range(n):
                # Convolve the polynomials for (row, it) and (it, col)
                conv_result = np.convolve(a_perm[:, row, it], b_perm[:, it, col])
                result[row, col, :len(conv_result)] += conv_result

    return result
