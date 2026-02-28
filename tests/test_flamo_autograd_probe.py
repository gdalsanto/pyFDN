"""Tests for autograd FLAMO probing backend."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from pyFDN.auxiliary.flamo_autograd_probe import (
    FlamoAutogradGraphZFilter,
    attach_autograd_probe,
    flamo_graph_to_autograd_zfilter,
    probe_flamo_z_autograd,
)


class Gain:
    def __init__(self, value: np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
        self.param = torch.as_tensor(arr, dtype=torch.float64)
        self.map = lambda x: x
        self.input_channels = arr.shape[1]
        self.output_channels = arr.shape[0]


class Matrix(Gain):
    pass


class parallelDelay:
    def __init__(self, delays_samples: np.ndarray):
        arr = np.asarray(delays_samples, dtype=np.float64).reshape(-1)
        self.param = torch.as_tensor(arr, dtype=torch.float64)
        self.map = lambda x: x
        self.input_channels = arr.size
        self.output_channels = arr.size
        self.gamma = torch.tensor(1.0, dtype=torch.float64)
        self.isint = True

    def s2sample(self, delay):
        return delay


class Series:
    def __init__(self, modules: OrderedDict[str, object]):
        self._modules = modules
        vals = list(modules.values())
        self.input_channels = vals[0].input_channels
        self.output_channels = vals[-1].output_channels


class Parallel:
    def __init__(self, *, brA: object, brB: object, sum_output: bool = True):
        self.branchA = brA
        self.branchB = brB
        self.sum_output = sum_output
        self.input_channels = brA.input_channels
        self.output_channels = brA.output_channels if sum_output else (brA.output_channels + brB.output_channels)


class Recursion:
    def __init__(self, *, fF: object, fB: object):
        self.feedforward = fF
        self.feedback = fB
        self.input_channels = fF.input_channels
        self.output_channels = fF.output_channels


class Shell:
    def __init__(self, *, core: object):
        self._Shell__core = core
        self.input_channels = core.input_channels
        self.output_channels = core.output_channels

    def get_core(self):
        return self._Shell__core


def _reference_closed_loop(
    z: complex,
    *,
    delays: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dmat = np.diag(np.power(z, -delays))
    ddmat = np.diag(-delays * np.power(z, -delays - 1.0))

    eye = np.eye(dmat.shape[0], dtype=np.complex128)
    loop = np.linalg.solve(eye - dmat @ a, dmat)
    dloop = np.linalg.solve(eye - dmat @ a, ddmat + (ddmat @ a) @ loop)

    h = c @ loop @ b + d
    dh = c @ dloop @ b
    return h, dh


def test_probe_flamo_z_autograd_matches_reference():
    a = np.array([[0.0, 0.4], [-0.2, 0.1]], dtype=np.float64)
    b = np.array([[1.0], [0.5]], dtype=np.float64)
    c = np.array([[0.7, -0.3]], dtype=np.float64)
    d = np.array([[0.15]], dtype=np.float64)
    delays = np.array([2.0, 4.0], dtype=np.float64)

    gain_b = Gain(b)
    gain_c = Gain(c)
    gain_d = Gain(d)
    gain_a = Matrix(a)
    delay = parallelDelay(delays)

    rec = Recursion(fF=delay, fB=gain_a)
    fdn = Series(OrderedDict({"input_gain": gain_b, "feedback_loop": rec, "output_gain": gain_c}))
    core = Parallel(brA=fdn, brB=gain_d, sum_output=True)
    model = Shell(core=core)

    z = 0.83 + 0.37j
    h, dh = probe_flamo_z_autograd(model, z)
    h_ref, dh_ref = _reference_closed_loop(z, delays=delays, a=a, b=b, c=c, d=d)

    np.testing.assert_allclose(h, h_ref, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(dh, dh_ref, rtol=1e-9, atol=1e-10)


def test_flamo_autograd_adapter_and_attach_methods():
    a = np.array([[0.2]])
    b = np.array([[1.0]])
    c = np.array([[1.0]])
    d = np.array([[0.0]])
    delays = np.array([3.0])

    model = Shell(
        core=Parallel(
            brA=Series(
                OrderedDict(
                    {
                        "input_gain": Gain(b),
                        "feedback_loop": Recursion(fF=parallelDelay(delays), fB=Matrix(a)),
                        "output_gain": Gain(c),
                    }
                )
            ),
            brB=Gain(d),
            sum_output=True,
        )
    )

    attached = attach_autograd_probe(model)
    assert hasattr(attached, "probe")
    assert hasattr(attached, "probe_with_derivative")

    zf = flamo_graph_to_autograd_zfilter(model)
    assert isinstance(zf, FlamoAutogradGraphZFilter)
    z = 0.9 + 0.1j
    h1 = attached.probe(z)
    h2 = zf.at(z)
    np.testing.assert_allclose(h1, h2, rtol=1e-11, atol=1e-12)
    dh1 = attached.probe_with_derivative(z)[1]
    dh2 = zf.der(z)
    np.testing.assert_allclose(dh1, dh2, rtol=1e-9, atol=1e-10)

