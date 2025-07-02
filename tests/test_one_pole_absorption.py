import numpy as np
from pyFDN.auxiliary.one_pole_absorption import db2mag, RT602slope, clip, slope2RT60, design_one_pole_filter, one_pole_absorption

def test_db2mag():
    assert np.isclose(db2mag(0), 1.0)
    assert db2mag(-20) < 1.0

def test_RT602slope():
    assert RT602slope(1, 1) == -60.0

def test_clip():
    arr = np.array([-2, 0, 2])
    clipped = clip(arr, [0, 1])
    np.testing.assert_array_equal(clipped, [0, 0, 1])

def test_slope2RT60():
    slope = -1.0
    fs = 1
    rt60 = slope2RT60(slope, fs)
    assert rt60 > 0

def test_design_one_pole_filter():
    HDc = np.array([0.5, 0.8])
    HNyq = np.array([0.2, 0.4])
    b, a = design_one_pole_filter(HDc, HNyq)
    assert b.shape[0] == 2
    assert a.shape[0] == 2

def test_one_pole_absorption():
    RT_DC = np.array([1, 1])
    RT_NY = np.array([1, 1])
    delays = np.array([1, 2])
    fs = 48000
    b, a = one_pole_absorption(RT_DC, RT_NY, delays, fs)
    assert b.shape[0] == 2
    assert a.shape[0] == 2 