import numpy as np
import pytest
from pyFDN.auxiliary.det_polynomial import det_polynomial

def test_det_polynomial_basic():
    # 2x2 identity polynomial matrix
    poly_mat = np.zeros((2, 2, 3))
    poly_mat[0, 0, 0] = 1
    poly_mat[1, 1, 0] = 1
    result = det_polynomial(poly_mat, 'z^-1')
    assert np.allclose(result[0], 1) 