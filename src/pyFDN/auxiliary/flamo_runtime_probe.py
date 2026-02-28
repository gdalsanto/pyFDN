"""
Runtime FLAMO probing bridge.

This module prepares pyFDN for native FLAMO probe APIs (Stage B).
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import numpy as np
import torch


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


def _is_flamo_object(obj: Any) -> bool:
    module_name = getattr(obj.__class__, "__module__", "")
    return module_name.startswith("flamo.")


def _infer_model_device(model: Any) -> torch.device:
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            first = next(params())
            return first.device
        except Exception:
            pass
    alias_decay = getattr(model, "alias_decay_db", None)
    if isinstance(alias_decay, torch.Tensor):
        return alias_decay.device
    return torch.device("cpu")


def _infer_model_complex_dtype(model: Any) -> torch.dtype:
    dt = getattr(model, "dtype", None)
    if dt in (torch.float16, torch.float32):
        return torch.complex64
    return torch.complex128


def _to_torch_complex_scalar(z: complex | np.ndarray, *, model: Any) -> torch.Tensor:
    z_arr = np.asarray(z, dtype=np.complex128)
    if z_arr.ndim != 0:
        raise ValueError("Expected scalar z for scalar probe call")
    device = _infer_model_device(model)
    dtype = _infer_model_complex_dtype(model)
    return torch.tensor(complex(z_arr.item()), device=device, dtype=dtype)


def _to_torch_complex_points(z: complex | np.ndarray, *, model: Any) -> torch.Tensor:
    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    device = _infer_model_device(model)
    dtype = _infer_model_complex_dtype(model)
    return torch.as_tensor(z_arr, device=device, dtype=dtype)


def _real_dtype_for_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    return torch.float64


def _autograd_probe_callable(
    model: Any,
    method_name: str,
    z: complex | np.ndarray,
    *,
    include_shell_io: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (H, dH/dz) for a model-bound probe-like callable via autograd.
    """
    method = getattr(model, method_name, None)
    if not callable(method):
        raise RuntimeError(f"Model does not expose callable {method_name}().")

    z_base = _to_torch_complex_scalar(z, model=model)
    real_dtype = _real_dtype_for_complex_dtype(z_base.dtype)
    x = z_base.real.detach().clone().to(real_dtype).requires_grad_(True)
    y = z_base.imag.detach().clone().to(real_dtype).requires_grad_(True)
    z_reconst = torch.complex(x, y)

    h = _call_with_supported_kwargs(method, z_reconst, include_shell_io=include_shell_io)
    if isinstance(h, tuple):
        if len(h) == 0:
            raise RuntimeError(f"{method_name} returned an empty tuple.")
        h = h[0]
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h, dtype=torch.complex128, device=z_base.device)
    h = torch.atleast_2d(h)

    u = h.real
    v = h.imag
    n_out, n_in = h.shape
    dh_dz = torch.zeros(n_out, n_in, dtype=torch.complex128, device=h.device)

    for i in range(n_out):
        for j in range(n_in):
            if u[i, j].requires_grad or u[i, j].grad_fn is not None:
                du_dx, du_dy = torch.autograd.grad(
                    u[i, j],
                    [x, y],
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                du_dx = None
                du_dy = None
            if v[i, j].requires_grad or v[i, j].grad_fn is not None:
                dv_dx, dv_dy = torch.autograd.grad(
                    v[i, j],
                    [x, y],
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                dv_dx = None
                dv_dy = None

            if du_dx is None:
                du_dx = torch.zeros_like(x)
            if du_dy is None:
                du_dy = torch.zeros_like(y)
            if dv_dx is None:
                dv_dx = torch.zeros_like(x)
            if dv_dy is None:
                dv_dy = torch.zeros_like(y)

            real_part = 0.5 * (du_dx + dv_dy)
            imag_part = 0.5 * (dv_dx - du_dy)
            dh_dz[i, j] = torch.complex(real_part, imag_part)

    return _to_numpy(h), _to_numpy(dh_dz)


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
    if not _is_flamo_object(model):
        return False
    if not callable(getattr(model, "probe", None)):
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
    z_for_model = _to_torch_complex_scalar(z, model=model) if _is_flamo_object(model) else z
    h_from_probe_only = None

    if derivative and callable(getattr(model, "probe_with_derivative", None)):
        out = _call_with_supported_kwargs(
            model.probe_with_derivative,
            z_for_model,
            include_shell_io=include_shell_io,
        )
        if isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])

    if callable(getattr(model, "probe", None)):
        out = _call_with_supported_kwargs(
            model.probe,
            z_for_model,
            derivative=derivative,
            include_shell_io=include_shell_io,
        )
        if derivative and isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])
        if not derivative:
            if isinstance(out, tuple):
                if len(out) == 0:
                    raise RuntimeError("model.probe returned empty tuple")
                return _to_numpy(out[0])
            return _to_numpy(out)
        if not (isinstance(out, tuple) and len(out) == 2):
            h_from_probe_only = out

    # 2) module-level FLAMO helpers
    probe_mod = _flamo_probe_module()
    if probe_mod is not None and _is_flamo_object(model):
        if derivative and callable(getattr(probe_mod, "probe_with_derivative", None)):
            out = _call_with_supported_kwargs(
                probe_mod.probe_with_derivative,
                model,
                _to_torch_complex_scalar(z, model=model),
                include_shell_io=include_shell_io,
            )
            if isinstance(out, tuple) and len(out) == 2:
                return _to_numpy(out[0]), _to_numpy(out[1])

        if callable(getattr(probe_mod, "probe_points", None)):
            z_arr = np.asarray(z, dtype=np.complex128)
            scalar = z_arr.ndim == 0
            z_points = _to_torch_complex_points(z, model=model)
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
            if derivative and h_from_probe_only is None:
                h_from_probe_only = out_np

    if derivative and h_from_probe_only is not None:
        raise RuntimeError(
            "Native FLAMO probe found only H(z) but not derivative support. "
            "Please use a FLAMO version exposing probe_with_derivative."
        )
    raise RuntimeError(
        "No native FLAMO probing API detected. "
        "Install/use the FLAMO branch that implements probe()/probe_with_derivative "
        "or flamo.processor.probe helpers."
    )


def probe_flamo_recursion_runtime(
    model: Any,
    z: complex | np.ndarray,
    *,
    derivative: bool = True,
    include_shell_io: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Probe recursion characteristic matrix via native FLAMO recursion API.

    Expected API on the recursion module:
      - probe_recursion(z) -> P(z)
      - probe_recursion_with_derivative(z) -> (P(z), dP/dz)  [optional]
    where P(z) = I - F(z)B(z).
    """
    z_for_model = _to_torch_complex_scalar(z, model=model) if _is_flamo_object(model) else z

    if derivative and callable(getattr(model, "probe_recursion_with_derivative", None)):
        out = _call_with_supported_kwargs(
            model.probe_recursion_with_derivative,
            z_for_model,
            include_shell_io=include_shell_io,
        )
        if isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])

    if callable(getattr(model, "probe_recursion", None)):
        out = _call_with_supported_kwargs(
            model.probe_recursion,
            z_for_model,
            derivative=derivative,
            include_shell_io=include_shell_io,
        )
        if derivative and isinstance(out, tuple) and len(out) == 2:
            return _to_numpy(out[0]), _to_numpy(out[1])
        if not derivative:
            if isinstance(out, tuple):
                if len(out) == 0:
                    raise RuntimeError("model.probe_recursion returned empty tuple")
                return _to_numpy(out[0])
            return _to_numpy(out)
        if _is_flamo_object(model):
            return _autograd_probe_callable(
                model,
                "probe_recursion",
                z,
                include_shell_io=include_shell_io,
            )
        raise RuntimeError(
            "model.probe_recursion was found but did not return (P, dP) for derivative=True."
        )

    raise RuntimeError(
        "No native FLAMO recursion probing API detected. "
        "Expected probe_recursion() or probe_recursion_with_derivative()."
    )

