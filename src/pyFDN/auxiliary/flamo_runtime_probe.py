"""
Runtime FLAMO probing bridge.

This module prepares pyFDN for native FLAMO probe APIs (Stage B).
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    out = value
    if hasattr(out, "detach"):
        out = out.detach()
    if hasattr(out, "cpu"):
        out = out.cpu()
    if hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out)


def _call_with_supported_kwargs(fn: Any, *args, **kwargs):
    """Call a callable while passing only kwargs it accepts."""
    sig = inspect.signature(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)


@lru_cache(maxsize=1)
def _flamo_probe_module():
    try:
        from flamo.processor import probe as probe_mod  # type: ignore
    except Exception:
        return None
    return probe_mod


def has_flamo_native_probe(model: Any) -> bool:
    """
    Return True if model appears to expose native FLAMO probe entrypoints.
    """
    if callable(getattr(model, "probe", None)) or callable(
        getattr(model, "probe_with_derivative", None)
    ):
        return True
    probe_mod = _flamo_probe_module()
    if probe_mod is None:
        return False
    return callable(getattr(probe_mod, "probe_with_derivative", None)) or callable(
        getattr(probe_mod, "probe_points", None)
    )


def probe_flamo_runtime(
    model: Any,
    z: complex | np.ndarray,
    *,
    derivative: bool = True,
    include_shell_io: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Probe FLAMO graph via native API.

    Priority order:
      1) model.probe_with_derivative / model.probe
      2) flamo.processor.probe helpers
    """
    # 1) model-bound native methods
    if derivative and callable(getattr(model, "probe_with_derivative", None)):
        out = _call_with_supported_kwargs(
            model.probe_with_derivative,
            z,
            include_shell_io=include_shell_io,
        )
        if isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])

    if callable(getattr(model, "probe", None)):
        out = _call_with_supported_kwargs(
            model.probe,
            z,
            derivative=derivative,
            include_shell_io=include_shell_io,
        )
        if derivative and isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])
        if not derivative and not isinstance(out, tuple):
            return _to_numpy(out)
        if derivative and not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError(
                "model.probe was found but did not return (H, dH) for derivative=True. "
                "Please use a FLAMO version exposing probe_with_derivative support."
            )

    # 2) module-level FLAMO helpers
    probe_mod = _flamo_probe_module()
    if probe_mod is not None:
        if derivative and callable(getattr(probe_mod, "probe_with_derivative", None)):
            out = _call_with_supported_kwargs(
                probe_mod.probe_with_derivative,
                model,
                z,
                include_shell_io=include_shell_io,
            )
            if isinstance(out, tuple) and len(out) == 2:
                return _to_numpy(out[0]), _to_numpy(out[1])

        if callable(getattr(probe_mod, "probe_points", None)):
            z_arr = np.asarray(z, dtype=np.complex128)
            scalar = z_arr.ndim == 0
            z_points = z_arr.reshape(-1) if scalar else z_arr
            out = _call_with_supported_kwargs(
                probe_mod.probe_points,
                model,
                z_points,
                include_shell_io=include_shell_io,
            )
            out_np = _to_numpy(out)
            if scalar:
                out_np = out_np.reshape(-1, *out_np.shape[-2:])[0]
            if not derivative:
                return out_np

    raise RuntimeError(
        "No native FLAMO probing API detected. "
        "Install/use the FLAMO branch that implements probe()/probe_with_derivative "
        "or flamo.processor.probe helpers."
    )

