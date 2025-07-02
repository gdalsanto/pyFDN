import numpy as np
from pyFDN.generate.random_orthogonal import random_orthogonal

def test_random_orthogonal():
    n = 4
    Q = random_orthogonal(n)
    assert Q.shape == (n, n)
    # Q should be orthogonal: Q.T @ Q = I
    I = Q.T @ Q
    np.testing.assert_allclose(I, np.eye(n), atol=1e-7) 