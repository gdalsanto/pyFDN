"""Tests for FLAMO runtime probing bridge."""

from __future__ import annotations

import numpy as np
import torch

from pyFDN.auxiliary.flamo_runtime_probe import probe_flamo_runtime


class NativeProbeModel:
    def __init__(self):
        self.input_channels = 1
        self.output_channels = 1
        self.probe_calls = 0
        self.probe_der_calls = 0

    def probe(self, z, derivative: bool = False, include_shell_io: bool = False):
        self.probe_calls += 1
        h = np.array([[2.0 + 1.0j]], dtype=np.complex128)
        if derivative:
            dh = np.array([[0.5 - 0.25j]], dtype=np.complex128)
            return h, dh
        return h

    def probe_with_derivative(self, z, include_shell_io: bool = False):
        self.probe_der_calls += 1
        h = np.array([[2.0 + 1.0j]], dtype=np.complex128)
        dh = np.array([[0.5 - 0.25j]], dtype=np.complex128)
        return h, dh


class Gain:
    """Stage-A compatible Gain-like leaf (for fallback path)."""

    def __init__(self, value: np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
        self.param = torch.as_tensor(arr, dtype=torch.float64)
        self.map = lambda x: x
        self.input_channels = arr.shape[1]
        self.output_channels = arr.shape[0]


def test_probe_flamo_runtime_prefers_native_model_methods():
    model = NativeProbeModel()
    z = 0.9 + 0.1j

    h = probe_flamo_runtime(model, z, derivative=False)
    np.testing.assert_allclose(h, np.array([[2.0 + 1.0j]]), rtol=0, atol=0)
    assert model.probe_calls == 1
    assert model.probe_der_calls == 0

    h2, dh2 = probe_flamo_runtime(model, z, derivative=True)
    np.testing.assert_allclose(h2, np.array([[2.0 + 1.0j]]), rtol=0, atol=0)
    np.testing.assert_allclose(dh2, np.array([[0.5 - 0.25j]]), rtol=0, atol=0)
    assert model.probe_der_calls == 1


def test_probe_flamo_runtime_falls_back_to_pyfdn_autograd():
    gain = Gain(np.array([[1.25]], dtype=np.float64))
    z = np.array([0.7 + 0.2j, 1.1 - 0.3j], dtype=np.complex128)
    h, dh = probe_flamo_runtime(gain, z, derivative=True)

    expected_h = np.array([[[1.25]], [[1.25]]], dtype=np.complex128)
    expected_dh = np.zeros_like(expected_h)
    np.testing.assert_allclose(h, expected_h, rtol=0, atol=0)
    np.testing.assert_allclose(dh, expected_dh, rtol=0, atol=0)

