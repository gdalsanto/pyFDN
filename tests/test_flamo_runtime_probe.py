"""Tests for FLAMO runtime probing bridge."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_probe_flamo_runtime_raises_without_native_api():
    class NoProbeModel:
        pass

    with pytest.raises(RuntimeError):
        probe_flamo_runtime(NoProbeModel(), 0.8 + 0.1j, derivative=True)

