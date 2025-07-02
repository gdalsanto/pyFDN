import numpy as np
from pyFDN.auxiliary.matrix_polyder import matrix_polyder

def test_matrix_polyder_basic():
    B = np.ones((3, 2, 2))
    A = np.ones((3, 2, 2))
    Q, P = matrix_polyder(B, A, var='z^-1')
    assert Q.shape == (3, 2, 2)
    assert P.shape == (3, 2, 2) 