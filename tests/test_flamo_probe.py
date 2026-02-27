"""Tests for arbitrary z-plane FLAMO probing utilities."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from pyFDN.auxiliary.flamo_probe import (
    FlamoGraphZFilter,
    flamo_graph_to_zfilter,
    probe_flamo_z,
)


class Gain:
    def __init__(self, value: np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Gain expects a 2-D matrix.")
        self.param = arr
        self.map = lambda x: x
        self.input_channels = arr.shape[1]
        self.output_channels = arr.shape[0]


class Matrix(Gain):
    pass


class parallelGain:
    def __init__(self, value: np.ndarray):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        self.param = arr
        self.map = lambda x: x
        self.input_channels = arr.size
        self.output_channels = arr.size


class parallelDelay:
    def __init__(
        self,
        delays_samples: np.ndarray,
        *,
        fs: float = 1.0,
        unit: float = 1.0,
        gamma: float = 1.0,
        isint: bool = True,
    ):
        arr = np.asarray(delays_samples, dtype=np.float64).reshape(-1)
        self.param = arr
        self.map = lambda x: x
        self.fs = fs
        self.unit = unit
        self.gamma = gamma
        self.isint = isint
        self.input_channels = arr.size
        self.output_channels = arr.size

    def s2sample(self, delay: np.ndarray):
        return np.asarray(delay, dtype=np.float64) * self.fs / self.unit


class parallelSOSFilter:
    def __init__(self, sos: np.ndarray, *, gamma: float = 1.0):
        coeff = np.asarray(sos, dtype=np.float64)
        if coeff.ndim != 3 or coeff.shape[1] != 6:
            raise ValueError("parallelSOSFilter expects shape (K, 6, N).")
        self.param = coeff
        self.map = lambda x: x
        self.gamma = gamma
        self.input_channels = coeff.shape[2]
        self.output_channels = coeff.shape[2]


class Series:
    def __init__(self, modules: OrderedDict[str, object]):
        self._modules = modules
        values = list(modules.values())
        self.input_channels = values[0].input_channels
        self.output_channels = values[-1].output_channels


class Parallel:
    def __init__(self, *, brA: object, brB: object, sum_output: bool = True):
        self.branchA = brA
        self.branchB = brB
        self.sum_output = sum_output
        self.input_channels = brA.input_channels
        if sum_output:
            self.output_channels = brA.output_channels
        else:
            self.output_channels = brA.output_channels + brB.output_channels


class Recursion:
    def __init__(self, *, fF: object, fB: object):
        self.feedforward = fF
        self.feedback = fB
        self.input_channels = fF.input_channels
        self.output_channels = fF.output_channels


class Shell:
    def __init__(self, *, core: object, input_layer: object | None = None, output_layer: object | None = None):
        self._Shell__core = core
        self._Shell__input_layer = input_layer
        self._Shell__output_layer = output_layer
        self.input_channels = core.input_channels
        self.output_channels = core.output_channels

    def get_core(self):
        return self._Shell__core

    def get_inputLayer(self):
        return self._Shell__input_layer

    def get_outputLayer(self):
        return self._Shell__output_layer


def test_probe_flamo_z_matches_closed_loop_reference():
    # Build a small FDN-like graph:
    # x -> B -> Recursion(Delay, A) -> C, in parallel with D.
    mat_a = np.array([[0.0, 0.4], [-0.2, 0.1]], dtype=np.float64)
    mat_b = np.array([[1.0], [0.5]], dtype=np.float64)
    mat_c = np.array([[0.7, -0.3]], dtype=np.float64)
    mat_d = np.array([[0.15]], dtype=np.float64)
    delays = np.array([2.0, 4.0], dtype=np.float64)

    gain_b = Gain(mat_b)
    gain_c = Gain(mat_c)
    gain_d = Gain(mat_d)
    gain_a = Matrix(mat_a)
    delay = parallelDelay(delays)

    feedback_loop = Recursion(fF=delay, fB=gain_a)
    fdn = Series(OrderedDict({"input_gain": gain_b, "feedback_loop": feedback_loop, "output_gain": gain_c}))
    core = Parallel(brA=fdn, brB=gain_d, sum_output=True)
    model = Shell(core=core)

    z = 0.83 + 0.37j
    h, dh = probe_flamo_z(model, z)

    dmat = np.diag(np.power(z, -delays))
    ddmat = np.diag(-delays * np.power(z, -delays - 1.0))

    eye = np.eye(dmat.shape[0], dtype=np.complex128)
    loop = np.linalg.solve(eye - dmat @ mat_a, dmat)
    dloop = np.linalg.solve(eye - dmat @ mat_a, ddmat + (ddmat @ mat_a) @ loop)

    h_ref = mat_c @ loop @ mat_b + mat_d
    dh_ref = mat_c @ dloop @ mat_b

    assert h.shape == (1, 1)
    assert dh.shape == (1, 1)
    assert np.allclose(h, h_ref, rtol=1e-11, atol=1e-12)
    assert np.allclose(dh, dh_ref, rtol=1e-10, atol=1e-11)

    zf = FlamoGraphZFilter(model)
    assert np.allclose(zf.at(z), h_ref, rtol=1e-11, atol=1e-12)
    assert np.allclose(zf.der(z), dh_ref, rtol=1e-10, atol=1e-11)
    assert isinstance(flamo_graph_to_zfilter(model), FlamoGraphZFilter)


def test_probe_flamo_z_parallel_concat_and_vectorized_points():
    br_a = Gain(np.array([[2.0]], dtype=np.float64))
    br_b = Gain(np.array([[-1.0]], dtype=np.float64))
    model = Parallel(brA=br_a, brB=br_b, sum_output=False)

    z = np.array([0.9 + 0.1j, 1.1 - 0.2j, -0.5 + 0.4j], dtype=np.complex128)
    h, dh = probe_flamo_z(model, z)

    expected = np.array([[[2.0], [-1.0]]] * z.size, dtype=np.complex128)
    assert h.shape == (z.size, 2, 1)
    assert dh.shape == (z.size, 2, 1)
    assert np.allclose(h, expected, rtol=0, atol=0)
    assert np.allclose(dh, 0.0, rtol=0, atol=0)


def test_probe_flamo_z_parallel_sos_value_and_derivative():
    # Single-channel, single-section SOS.
    # H(z) = (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)
    b0, b1, b2 = 1.0, 0.2, 0.1
    a0, a1, a2 = 1.0, -0.4, 0.12
    sos = np.array([[[b0], [b1], [b2], [a0], [a1], [a2]]], dtype=np.float64)  # (K=1,6,N=1)
    model = parallelSOSFilter(sos)

    z = 0.73 + 0.22j
    h, dh = probe_flamo_z(model, z)

    num = b0 + b1 * z**-1 + b2 * z**-2
    den = a0 + a1 * z**-1 + a2 * z**-2
    dnum = -b1 * z**-2 - 2.0 * b2 * z**-3
    dden = -a1 * z**-2 - 2.0 * a2 * z**-3

    h_ref = np.array([[num / den]], dtype=np.complex128)
    dh_ref = np.array([[(dnum * den - num * dden) / (den**2)]], dtype=np.complex128)

    assert h.shape == (1, 1)
    assert dh.shape == (1, 1)
    assert np.allclose(h, h_ref, rtol=1e-12, atol=1e-12)
    assert np.allclose(dh, dh_ref, rtol=1e-11, atol=1e-11)

